import time
import tensorflow as tf
import pickle
import numpy as np
import os


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.gru1 = tf.keras.layers.GRU(self.dec_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        self.fc0 = tf.keras.layers.Dense(256)
        self.fc1 = tf.keras.layers.Dense(256)
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        output = self.fc0(output)
        output = self.fc1(output)
        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


class Seq2Seq:
    def __init__(self, vocab_size, embedding_size, units, BATCH_SIZE):
        self.encoder = None
        self.decoder = None
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.units = units
        self.BUFFER_SIZE = 256
        self.BATCH_SIZE = BATCH_SIZE
        self.dataset = None
        self.create_dataset()
        self.checkpoints()

    def checkpoints(self):
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                              encoder=self.encoder,
                                              decoder=self.decoder)

    def create_dataset(self):
        directory = 'drive/My Drive/DataProject/extracted/'
        f = open(directory + "diffload128.pkl", "rb")
        print("Loading pickle")
        forwrite = pickle.load(f)
        f.close()
        inp, out, melody = forwrite

        new_inp = np.array(inp[:26000])
        new_out = np.array(inp[:26000])
        self.steps_per_epoch = len(new_inp) // self.BATCH_SIZE
        self.dataset = tf.data.Dataset.from_tensor_slices((new_inp, new_out)).shuffle(self.BUFFER_SIZE)
        self.dataset = self.dataset.batch(self.BATCH_SIZE, drop_remainder=True)

    def build_model(self):
        self.encoder = Encoder(self.vocab_size, self.embedding_size, self.units, self.BATCH_SIZE)
        self.decoder = Decoder(self.vocab_size, self.embedding_size, self.units, self.BATCH_SIZE)

    def train(self):
        EPOCHS = 10

        for epoch in range(EPOCHS):
            start = time.time()

            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(self.dataset.take(self.steps_per_epoch)):
                batch_loss = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / self.steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    @tf.function
    def train_step(self, inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([0] * self.BATCH_SIZE, 1)
            predicted = []
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                predicted.append(predictions)
                dec_input = predictions
                # using teacher forcing
                # dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = loss_function(targ, tf.convert_to_tensor(predicted, dtype=tf.float32))
        # batch_loss = (loss / int(targ.shape[1]))

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    def evaluate(self, sequence):
        attention_plot = np.zeros((128, 40))
        inputs = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=40, padding='post')
        print(inputs)
        inputs = tf.convert_to_tensor(inputs)
        print(inputs)
        result = ''

        hidden = [tf.zeros((1, self.units))]
        print(hidden)
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([0], 0)
        result = []
        for t in range(128):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                      dec_hidden,
                                                                      enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()
            print(predictions)
            predicted_id = tf.argmax(predictions[0]).numpy()

            # result += targ_lang.index_word[predicted_id] + ' '
            print(predicted_id)
            result.append(predicted_id)
            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result


if __name__ == '__main__':
    s2s = Seq2Seq(86, embedding_size=20, units=512, BATCH_SIZE=800)
