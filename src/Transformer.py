import pickle
import numpy as np
import tensorflow as tf
import time


# import pickle
# directory = 'drive/My Drive/DataProject/extracted/'
# f = open(directory + "diffload128.pkl", "rb")
# print("Loading pickle")
# forwrite = pickle.load(f)
# f.close()
#
# inp,out,melody = forwrite


# print(len(inp),len(inp[0]))
# new_inp = np.array(inp[:26000])[:,:40]
# new_out = np.array(inp[:26000])
# print(new_inp.shape,new_out.shape)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the d_model that is last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


class MusicTransformer:

    def __init__(self):
        self.train_dataset = None
        self.test_dataset = None
        self.num_layers = 4
        self.d_model = 128
        self.dff = 512
        self.input_length = 0
        self.num_heads = 8
        self.EPOCHS = 20
        self.transformer = None
        self.input_vocab_size = 88
        self.target_vocab_size = 88
        self.dropout_rate = 0.1
        self.BATCH_SIZE = 64
        self.melody = []
        self.BUFFER_SIZE = 256
        self.checkpoint_path = 'models/transformer1/'

    def build_dataset(self, new_inp, new_out):
        self.steps_per_epoch = len(new_inp) // self.BATCH_SIZE
        self.train_dataset = tf.data.Dataset.from_tensor_slices((new_inp[:24000], new_out[:24000])).shuffle(
            self.BUFFER_SIZE)
        self.train_dataset = self.train_dataset.batch(self.BATCH_SIZE, drop_remainder=True)
        self.val_dataset = tf.data.Dataset.from_tensor_slices((new_inp[24000:], new_out[24000:])).shuffle(
            self.BUFFER_SIZE)
        self.val_dataset = self.val_dataset.batch(self.BATCH_SIZE, drop_remainder=True)

    def build_transformer(self):
        self.transformer = Transformer(
            num_layers=2, d_model=512, num_heads=8, dff=2048,
            input_vocab_size=88, target_vocab_size=88,
            pe_input=40, pe_target=128)

        learning_rate = CustomSchedule(self.d_model)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                                  epsilon=1e-9)

        self.ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                        optimizer=self.optimizer)

        ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            self.ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def create_look_ahead_mask(self, size):
        # The look-ahead mask is used to mask the future tokens in a sequence. In other words,
        # the mask indicates which entries should not be used.
        # This means that to predict the third word, only the first and second word will be used.
        # Similarly to predict the fourth word, only the first, second and the third word will be used and so on.
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def create_padding_mask(self, seq):
        # Mask all the pad tokens in the batch of sequence. It ensures that the model does not treat
        # padding as the input. The mask indicates where pad value 0 is present:
        # it outputs a 1 at those locations, and a 0 otherwise.
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = self.create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = self.create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    def __prepare_dataset(self):

        directory = 'drive/My Drive/DataProject/extracted/'
        f = open(directory + "diffload128.pkl", "rb")
        print("Loading pickle")
        forwrite = pickle.load(f)
        inp, out, g = forwrite
        f.close()

        new_inp = np.array(inp[:26000])[:, :40]
        new_out = np.array(inp[:26000])[:, 40:]
        print(new_inp.shape, new_out.shape)

        dataset = tf.data.Dataset.from_tensor_slices((new_inp, new_out)).shuffle(self.BUFFER_SIZE)
        dataset = dataset.batch(self.BATCH_SIZE, drop_remainder=True)

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(inp, tar_inp,
                                              True,
                                              enc_padding_mask,
                                              combined_mask,
                                              dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    def train(self):
        for epoch in range(self.EPOCHS):
            start = time.time()

            train_loss.reset_states()
            train_accuracy.reset_states()

            # inp -> portuguese, tar -> english
            for (batch, (inp, tar)) in enumerate(self.train_dataset):
                self.train_step(inp + 1, tar + 1)

                if batch % 50 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, train_loss.result(), train_accuracy.result()))

            if (epoch + 1) % 2 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                train_loss.result(),
                                                                train_accuracy.result()))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    def new_generator(self):
        self.build_transformer()

        sample = np.load("temp.npy")
        print(sample.shape)
        print("Input_data_to_model")
        self.input_data_to_model(sample)
        self.melody = self.melody[0]

        while len(self.melody) < self.input_length:
            self.melody.extend(self.melody)

        self.melody = self.melody[:40]
        print(len(self.melody))
        retuu = list(self.evaluate(self.melody)[0] - 1)
        self.decode_track(retuu)

    def evaluate(self, inp_sentence):
        # start_token = [tokenizer_pt.vocab_size]
        # end_token = [tokenizer_pt.vocab_size + 1]
        #
        # inp sentence is portuguese, hence adding the start and end token
        # inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
        inp_sentence = np.array(inp_sentence) + 1
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [1]
        output = tf.expand_dims(decoder_input, 0)
        self.MAX_LENGTH = 128
        for i in range(self.MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(encoder_input,
                                                              output,
                                                              False,
                                                              enc_padding_mask,
                                                              combined_mask,
                                                              dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def input_data_to_model(self, nparray):
        # 0 for starting music piece
        # 1-6 for time of note
        # 7-54 for notes
        # 55-86 for rest notes
        temp = self.encode_track(nparray)
        self.melody = []
        self.melody.append(temp)

    def encode_track(self, nparray):
        """Receives a array of shape (48,)"""

        def get_top_index(quant_notes):
            for i in range(len(quantum_notes)):
                if quantum_notes[i] >= 0:
                    return i
            return -1

        starting = True
        startnote = None
        encoded = [0]
        rest = 0
        for quantum in range(nparray.shape[1]):
            quantum_notes = nparray[:, quantum]
            if len(np.nonzero(quantum_notes + 1)[0]) > 0:
                if starting:
                    i = get_top_index(quantum_notes)
                    if i > -1:
                        starting = False
                        print(starting, (not starting))
                        startnote = i
                        time_quantum = quantum_notes[i]
                        curnote = 24
                        encoded.extend([curnote + 7,
                                        time_quantum + 1])

                elif not starting:
                    i = get_top_index(quantum_notes)
                    if i > -1:
                        if rest > 0:
                            if rest > 32:
                                rest = 32
                            encoded.append(54 + rest)
                            rest = 0
                        time_quantum = quantum_notes[i]
                        curnote = 24 + startnote - i
                        if 0 <= curnote <= 47:
                            encoded.extend([curnote + 7,
                                            time_quantum + 1])
            elif not starting:
                rest += 1

        return encoded

    def decode_track(self, encoded):

        decoded = []
        orray = np.ones((48, 1)) * -1
        # print(orray.shape)
        # print(encoded)
        for i in range(len(encoded)):
            x = encoded[i]
            if x == 0:
                pass
            elif 7 <= x <= 54:
                x = x - 24 - 7
                if i + 1 < len(encoded):
                    t = encoded[i + 1]
                    if 1 <= t <= 6:
                        # print("Here")
                        array = np.ones((48, 1)) * -1
                        if 26 - x < 48:
                            array[26 - x] = t - 1
                            orray = np.append(orray,
                                              array, axis=1)
            elif 55 <= x <= 86:
                x = x - 54
                array = np.ones((48, x)) * -1
                orray = np.append(orray,
                                  array, axis=1)
        np.save("generated.npy", orray)
        # print(orray)
        return orray


if __name__ == "__main__":
    music = MusicTransformer()
    music.build_dataset()
    music.build_transformer()
    music.train()
