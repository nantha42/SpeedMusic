from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization as BatchNorm
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow

from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import tensorflow.keras.metrics as metrics
import numpy as np
import os
import time
import pickle
import threading


class Generator2(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, x_set, y_set, batch_size=4):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(len(self.x))
        self.musicdataset = []
        self.melody = []

    def __len__(self):
        # return int(np.ceil(len(self.x) / self.batch_size))
        # print("__len__",)
        # print("__len__",int(np.ceil(len(self.musicdataset)/self.batch_size)))
        return int(np.ceil(len(self.musicdataset) / self.batch_size))

    def __getitem__(self, idx):
        # print()
        # print("Entering __getitem__")
        inds = []
        for song in self.musicdataset[idx * self.batch_size:(idx + 1) * self.batch_size]:
            start = song[0]
            end = song[1]
            # print("Extending ",start,":",end)
            inds.extend(self.indices[start:end])

        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        # print("Inds:",inds)
        batch_x = []
        batch_y = []
        for ind in inds:
            ip = []
            for q in self.x[ind]:
                o = np.zeros(323 + 1 + 1)
                o[int(q)] = 1
                ip.append(o)

            batch_x.append(ip)
            hot_encoded = []
            for val in self.y[ind]:
                t = np.zeros(323 + 1 + 1)
                t[int(val)] = 1
                hot_encoded.append(t)
            batch_y.append(hot_encoded)
        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        # np.random.shuffle(self.indices)
        np.random.shuffle(self.musicdataset)
        pass


histories = {'categorical_accuracy': [], 'val_categorical_accuracy': [], 'loss': [], 'val_loss': []}


class attention(Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self, x):
        print("Shape of X", x.shape)
        res = K.tanh(K.dot(x, self.W) + self.b)

        print("res shape", res.shape)
        print("Trying to squeeze")
        et = K.squeeze(res, axis=-1)
        print("Squeezed")
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at

        print("Output shape ", output.shape)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(attention, self).get_config()


class Melody:
    def __init__(self, name, input_length, hidden_size=256, epochs=10):
        self.hidden_size = hidden_size
        self.input_length = input_length
        self.notes_classes = 323 + 1 + 1  #
        self.model = Sequential()
        self.epochs = epochs
        self.loaded_model = ""

        self.directory = ""
        self.input_data = None
        self.model_name = name
        self.inp = None
        self.out = None
        self.melody = []
        self.history = None
        self.musicdataset = []
        self.epochs_ran = 0
        self.batch_size = 7
        # self.model.add(Embedding(self.notes_classes, self.hidden_size, input_length=self.input_length))

    def build_model(self):
        if self.loaded_model != self.model_name:
            print(self.notes_classes)
            dropout = 0.4
            self.model = Sequential()

            self.model.add(
                LSTM(self.hidden_size,
                     input_shape=(self.input_length, self.notes_classes),
                     return_sequences=True,
                     recurrent_dropout=dropout), )
            self.model.add(LSTM(self.hidden_size,
                                recurrent_dropout=dropout,
                                return_sequences=True))
            self.model.add(BatchNorm())
            self.model.add(Dropout(dropout))
            self.model.add(Dense(256))
            self.model.add(Activation('relu'))
            self.model.add(BatchNorm())
            self.model.add(Dropout(dropout))
            self.model.add(Dense(256))
            self.model.add(Activation('relu'))
            self.model.add(BatchNorm())
            self.model.add(Dense(self.notes_classes))
            self.model.add(Activation('softmax'))
            # self.directory = "drive/My Drive/DataProject/1"
            self.model.compile(loss='categorical_crossentropy',
                               optimizer='rmsprop',
                               metrics=['categorical_accuracy'])
            self.loaded_model = self.model_name

    def input_data_to_model(self, nparray):

        # 0 for starting music piece
        # 1 for ending sequence
        # 2 for start of a chord
        # 3 for filling chord that are empty with keys
        # every single note is chord
        temp = [0]
        rest = 0
        for i in range(np.min([32 * 20, nparray.shape[1]])):
            row = nparray[:, i]
            notefound = False
            args = np.nonzero(row + 1)[0]

            if len(args) > 0:
                if rest > 0:
                    n32rests = int(rest / 32)
                    remrest = rest % 32
                    temp.extend([2, 2 + 1 + 1 + 287 + 32] * n32rests)
                    temp.extend([2, 2 + 1 + 1 + 287 + remrest])

                rest = 0
                temp.append(2)
                if len(args) >= 5:

                    for v in args[:5]:
                        temp.append(2 + 1 + 1 + v * 6 + row[v])
                else:
                    for v in args:
                        temp.append(2 + 1 + 1 + v * 6 + row[v])
            else:
                rest += 1
        temp.append(1)
        self.melody.append(temp)

    def melody_to_testdata_1(self):
        input = []
        output = []
        mel = []
        for t in range(300):
            mel.append(98)

        for i in range(0, len(mel) - self.input_length, self.input_length):
            # input.append(melody[i:i+32])
            ip = []
            for q in mel[i:i + self.input_length]:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                ip.append(o)
            input.append(ip)
            jp = []
            for q in mel[i + 1:i + 1 + self.input_length]:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                jp.append(o)
            output.append(jp)
        input = np.array(input)
        print("prediction input shape", input.shape)
        return [input, output]

    def melody_to_testdata(self):
        input = []
        output = []

        # for t in range(300):
        # self.mel.append(t%288)

        for i in range(0, len(self.melody) - self.input_length, self.input_length):
            # input.append(melody[i:i+32])
            ip = []
            for q in self.melody[i:i + self.input_length]:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                ip.append(o)
            input.append(ip)
            jp = []
            for q in self.melody[i + 1:i + 1 + self.input_length]:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                jp.append(o)
            output.append(jp)
        input = np.array(input)
        print("prediction input shape", input.shape)
        return [input, output]

    def melody_to_dataset(self, option=0):
        step = 2
        if option == 1:
            step = self.input_length

        input = []
        output = []
        for i in range(0, len(self.melody) - self.input_length, step):
            ip = []
            # print(i, len(self.melody) - self.input_length, step)
            for q in self.melody[i:i + self.input_length]:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                ip.append(o)
            input.append(ip)
            jp = []
            for q in self.melody[i + 1:i + 1 + self.input_length]:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                jp.append(o)
            output.append(jp)
        print("Returning input and output from func melody_to_dataset")
        return [np.array(input), np.array(output)]

    def melody_to_datasetver2(self, option=0):
        step = 2
        if option == 1:
            step = self.input_length
        input = []
        output = []
        set_size = 30
        self.musicdataset = []
        print("melody length", len(self.melody), len(self.musicdataset))
        for i in range(0, len(self.melody) - self.input_length, step):
            ip = []
            for q in self.melody[i:i + self.input_length]:
                ip.append(int(q))
            input.append(ip)
            jp = []
            for q in self.melody[i + 1:i + 1 + self.input_length]:
                jp.append(int(q))
            output.append(jp)
        for i in range(0, len(input), set_size):
            self.musicdataset.append([i, i + set_size])
        print("Returning input and output from func melody_to_dataset")
        print(len(input), len(output))
        return [input, output]

    def minimise(self):
        minlen = 10 ** 10
        temps = []
        for i in range(len(self.melody)):
            if len(self.melody[i]) < minlen and len(self.melody[i]) > 430:
                minlen = len(self.melody[i])
            elif len(self.melody[i]) < 430:
                temps.append(self.melody[i])

        for temp in temps:
            self.melody.remove(temp)

        for i in range(len(self.melody)):
            self.melody[i] = self.melody[i][:minlen]
            # y = len(self.melody[i])-1
            # rem = 6 - y%6
            # if rem == 5:
            # self.melody[i][-1] = 1
            # elif rem == 0:
            # self.melody[i].append(1)
            # else:
            # self.melody[i].extend([3]*rem)
            self.melody[i].append(1)
            # print("Checking final length",len(self.melody[i]))
        print("minised", minlen)
        print("Minimised Applied")

    def add_padding(self):

        maxlen = 0
        for i in range(len(self.melody)):
            if len(self.melody[i]) > maxlen:
                maxlen = len(self.melody[i])

        for i in range(len(self.melody)):
            self.melody[i].extend([1] * (maxlen - len(self.melody[i])))

    def melody_to_datasetver3(self, option=0):

        self.minimise()
        step = 2
        if option == 1:
            step = self.input_length
        input = []
        output = []
        set_size = 0
        first = True
        for t in range(len(self.melody)):

            for i in range(0, len(self.melody[t]) - self.input_length, step):
                ip = []
                for q in self.melody[t][i:i + self.input_length]:
                    ip.append(int(q))
                input.append(ip)
                jp = []
                for q in self.melody[i + 1:i + 1 + self.input_length]:
                    jp.append(int(q))
                output.append(jp)

            if first:
                first = False
                set_size = len(input)
                print(len(input))
                print(len(output))

        for i in range(0, len(input), set_size):
            self.musicdataset.append([i, i + set_size])
        return input, output

    def isprime(self, x):
        for i in range(2, int(np.sqrt(x))):
            if x % i == 0:
                return False
        return True

    def train2(self):
        x = self.inp
        y = self.out
        try:
            self.model = load_model(self.model_name)
            print("Model loaded from disk")
            pass
        except Exception:
            self.build_model()
            print("Error")
        if self.isprime(len(x)):
            x = x[:-1]
            y = y[:-1]
        print("melody length", len(self.melody), len(self.musicdataset), len(x))
        print_callback = LambdaCallback(self.on_epoch_end)
        generator = Generator2(x, y, batch_size=7)
        print(self.musicdataset[:int(len(self.musicdataset) - len(self.musicdataset) * 0.3)])
        print(self.musicdataset[int(len(self.musicdataset) - len(self.musicdataset) * 0.3):])
        validation_generator = Generator2(x, y, batch_size=self.batch_size)
        generator.musicdataset = self.musicdataset[:int(len(self.musicdataset) - len(self.musicdataset) * 0.35)]
        validation_generator.musicdataset = self.musicdataset[
                                            int(len(self.musicdataset) - len(self.musicdataset) * 0.35):]

        for i in range(self.epochs):
            print(i, "/", self.epochs)
            self.history = self.model.fit_generator(generator, validation_data=validation_generator, epochs=10,
                                                    verbose=1, callbacks=[print_callback])

            histories['categorical_accuracy'].extend(self.history.history['categorical_accuracy'])
            histories['val_categorical_accuracy'].extend(self.history.history['val_categorical_accuracy'])
            histories['loss'].extend(self.history.history['loss'])
            histories['val_loss'].extend(self.history.history['val_loss'])

            plt.plot(histories['categorical_accuracy'])
            plt.plot(histories['val_categorical_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()

            # Plot training & validation loss values
            plt.plot(histories['loss'])
            plt.plot(histories['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()
            self.model.save(self.model_name)
        f = open(self.directory + "history.pkl", "wb")
        pickle.dump(self.history, f, -1)
        f.close()

    def advance_converter(self, j):
        print("@advanceConverter")
        orray = np.ones([48, 1]) * -1
        i = 0
        print("Len of Set ", len(set(j)))
        print("Set ", set(j))
        while (i < len(j)):
            krray = np.ones([48, 1]) * -1

            if j[i] == 0 or j[i] == 1:
                pass
            elif j[i] == 2:
                if i + 1 < len(j) and j[i + 1] - 2 - 1 - 1 < 288:
                    maxv = 0
                    v = 0
                    q = 1
                    while i + q < len(j) and j[i + q] > 3:
                        x = int(j[i + q] - (2 + 1 + 1))
                        p = int(x / 6)
                        v = int(x) % 6
                        if (p < 48):
                            # print(q,p,v,x,j[i+q]-(2+1+1))
                            krray[p, 0] = v
                            if maxv < v:
                                maxv = v
                        q += 1

                    orray = np.append(orray, krray, axis=1)
                    for ta in range(0, (2 ** maxv) - 1):
                        krray = np.ones([48, 1]) * -1
                        orray = np.append(orray, krray, axis=1)

                elif i + 1 < len(j) and j[i + 1] >= 288 + 2 + 1 + 1:
                    x = j[i + 1] - 287 - 2 - 1 - 1
                    for z in range(x):
                        orray = np.append(orray, krray, axis=1)
            i += 1

        print("@Converter Entered data into orray")
        print("@Converter Now Saving orray")
        np.save("generated.npy", orray)
        print("stored successfully")

    def converter(self, j):
        print("@Converter")
        orray = np.ones([48, 1]) * -1
        # print(j)
        for i in range(len(j)):
            krray = np.ones([48, 1]) * -1
            if i % 1000 == 0:
                print(i, len(j))
            if j[i] < 288 + 2 and j[i] > 1:
                x = int(j[i]) - 2
                p = int(x / 6)
                v = int(x) % 6
                krray[p, 0] = v

                orray = np.append(orray, krray, axis=1)
                for ta in range(0, (2 ** v) - 1):
                    krray = np.ones([48, 1]) * -1
                    orray = np.append(orray, krray, axis=1)
            else:  # for adding rest rows to the melody
                x = j[i] - 287 - 2
                for z in range(x):
                    orray = np.append(orray, krray, axis=1)
                    # print("orray shape", orray.shape)
        print("@Converter Entered data into orray")
        print("@Converter Now Saving orray")
        np.save("drive/My Drive/DataProject/generated.npy", orray)
        print("stored successfully")

    def on_epoch_end(self, epoch_end, x):
        self.epochs_ran += 1
        print("@on_epoch_end saving model to disk")
        self.model.save(self.model_name)

    def load_saved(self, savefile="chordload.pkl", option=0, google_drive=False):
        """
        This loads the npy data in the folder saved and use it
        train the
        :return:
        """

        if not google_drive:
            high_npy_path = "high_npy/"

        else:
            high_npy_path = "drive/My Drive/DataProject/high_npy/"

        savefilename = savefile

        defaultoption = False
        if option == 1:
            list_of_files = os.listdir(".")
            if savefilename in list_of_files:
                defaultoption = False
            else:
                defaultoption = True
        else:
            defaultoption = True

        if defaultoption:
            list_of_files = os.listdir(high_npy_path)
            total_files = len(list_of_files)
            index = -1
            for saved in list_of_files:
                index += 1
                if saved[-4:] == ".npy" or saved[-4:] == ".npz":
                    print(index, "/", total_files)
                    sample = np.load(high_npy_path + saved)
                    # print(sample["arr_0"])
                    self.input_data_to_model(sample["arr_0"])

            self.inp, self.out = self.melody_to_datasetver3()
            forwrite = [self.inp, self.out, self.musicdataset]
            f = open(self.directory + savefilename, "wb")
            pickle.dump(forwrite, f, -1)
            f.close()
            # self.train2()
        else:
            f = open(self.directory + savefilename, "rb")
            forwrite = pickle.load(f)
            f.close()
            print(len(forwrite))
            self.inp, self.out, self.musicdataset = forwrite

            print("Inp,Out loaded from pickle on disk")
            print("Inp size: ", len(self.inp))
            print("Out size:", len(self.out))
            # self.train2()

    def generate_tune(self):
        try:
            if self.loaded_model != self.model_name:
                self.model = load_model(self.model_name)
                self.loaded_model = self.model_name
                print(self.model.summary())
                print("Model loaded from disk")
            else:
                self.build_model()
                print("Already loaded")
        except Exception:
            print("Error in loading model")
        sample = np.load("temp.npy")
        print(sample.shape)
        print("Input_data_to_model")
        self.input_data_to_model(sample)

        self.melody = self.melody[0]

        self.melody.pop(-1)
        while len(self.melody) < self.input_length:
            self.melody.extend(self.melody[1:])

        print("Length:", len(self.melody))
        original_length = len(self.melody)
        print(original_length, self.input_length)

        input = []
        output = []
        inm = []
        ino = []
        for i in range(0, len(self.melody) - self.input_length, 1):
            ip = []
            ipm = []
            # print(i, len(self.melody) - self.input_length, step)
            for q in self.melody[i:i + self.input_length]:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                ip.append(o)
                ipm.append(q)
            input.append(ip)
            inm.append(ipm)
            jp = []
            jpm = []
            for q in self.melody[i + 1:i + 1 + self.input_length]:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                jp.append(o)
                jpm.append(q)
            output.append(jp)
            ino.append(jpm)
        print("inm", "ino")
        print(inm)
        print(ino)
        print("Going to Predict")
        x = list(input[0])
        y = list(output[0])
        # print(x)
        # print(y)
        # input = [x] * 7
        # output = [y] * 7
        ui = np.array(input).shape
        print("Ui shape", ui)
        print(self.model.summary())
        generated = inm[0]
        initial = generated[-self.input_length:]
        e = 0
        while e < self.input_length:
            e += 1
            # j = np.array([generated[-self.input_length:]])
            ip = []
            j = generated[-self.input_length:]
            # for i in range(int(self.input_length / 2)):
            #     r = np.random.randint(self.input_length)
            #     j[r] = np.random.randint(4, 288)
            #     j[r] = initial[r]
            input = []

            for q in j:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                ip.append(o)
                # ipm.append(q)
            input.append(ip)
            prediction = self.model.predict(np.array(input))
            p = []
            for piece in prediction:
                for r in piece:
                    p.append(int(np.argmax(r)))
            generated.append(p[-1])

        if len(inm[0]) - self.input_length > 0:
            t = []
            t.extend(inm[0][:len(inm[0]) - self.input_length])
            t.extend(generated)
            generated = t
        print("generated")
        print(self.melody)
        print(generated)
        self.advance_converter(generated)

    def generate_tune1(self):
        try:
            self.model = load_model(self.model_name)
            print("Model loaded from disk")
        except Exception:
            self.build_model()
            print("Error in loading model")
        sample = np.load("temp.npy")
        self.input_data_to_model(sample)
        self.melody.pop(-1)
        while len(self.melody) < self.input_length:
            self.melody.extend(self.melody[1:])
        print("Length:", len(self.melody))
        original_length = len(self.melody)
        print(original_length, self.input_length)

        input = []
        output = []
        for i in range(0, len(self.melody) - self.input_length, self.input_length):
            ip = []
            # print(i, len(self.melody) - self.input_length, step)
            for q in self.melody[i:i + self.input_length]:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                ip.append(o)
            input.append(ip)
            jp = []
            for q in self.melody[i + 1:i + 1 + self.input_length]:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                jp.append(o)
            output.append(jp)

        print("Going to Predict")
        prediction = self.model.predict(np.array(input))
        generated = []
        for measure in prediction:
            for row in measure:
                generated.append(int(np.argmax(row)))
        print(generated)

        self.converter(self.melody)

    def use_model(self):
        # try:
        # self.model = load_model(self.model_name)
        # print("Model loaded from disk")
        # pass
        # except Exception:
        # print("Error")

        # sample = np.load("temp.npy")
        # self.input_data_to_model(sample)
        # x, y = self.melody_to_dataset(option=1)
        # print(x.shape)
        # print(x)
        selected = self.inp[np.random.randint(len(self.inp))]
        ip = []
        for q in selected:
            o = np.zeros(323 + 1 + 1)
            o[int(q)] = 1
            ip.append(o)
        # print("IpLen",len(ip))
        ip = np.array([ip])
        print("Ip Shape", ip.shape)
        prediction = self.model.predict(ip)
        generated = []
        for measure in prediction:
            for row in measure:
                generated.append(np.argmax(row))
        print("Set of Generated", set(generated))
        print("Length of Generated", generated)
        self.advance_converter(generated)

    def test_model(self, npy):
        try:
            self.model = load_model(self.model_name)
            print("Model loaded from disk")
        except Exception:
            print("Error")

        x, y = self.melody_to_testdata()
        prediction = self.model.predict(x)
        generated = []
        for measure in prediction:
            for row in measure:
                generated.append(np.argmax(row))
        print(generated)
        self.converter(generated)


class DifferenceMelody(Melody):

    def __init__(self, name, input_length, hidden_size=256, epochs=10):
        Melody.__init__(self, name, input_length, hidden_size, epochs)
        self.notes_classes = 87

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

    def input_data_to_model(self, nparray):
        # 0 for starting music piece
        # 1-6 for time of note
        # 7-54 for notes
        # 55-86 for rest notes
        temp = self.encode_track(nparray)
        self.melody = []
        self.melody.append(temp)

    def thread_generator(self, multiple_notes):
        if self.loaded_model != self.model_name:
            print(self.model_name)
            self.model = load_model(self.model_name, custom_objects={"attention": attention})
            self.loaded_model = self.model_name
            print(self.model.summary())
            print("Model loaded from disk")
        else:
            self.build_model()
            print("Already loaded")

        generated_outputs = []
        for notes in multiple_notes:
            self.input_data_to_model(notes)
            self.melody = self.melody[0]
            req_length = len(self.melody) + 5
            print("Thread_generatror:", self.melody)
            input = []
            output = []
            inm = []
            ino = []

            for i in range(0, len(self.melody) - 1, 1):
                ip = []
                ipm = []
                # print(i, len(self.melody) - self.input_length, step)
                for q in self.melody[i:i + 1]:
                    o = np.zeros(self.notes_classes)
                    o[int(q)] = 1
                    ip.append(o)
                    ipm.append(q)
                input.append(ip)
                inm.append(ipm)

            # a,b,c,d = np.zeros((512,1)),np.zeros((512,1)),np.zeros((512,1)),np.zeros((512,1))

            a, b, c, d = np.random.normal(0, 1, (1, 512)), np.random.normal(0, 1, (1, 512)), np.random.normal(0, 1, (
                1, 512)), np.random.normal(0, 1, (1, 512))
            print(a.shape,len(input))
            p = []

            for i in range(req_length):
                print(i)
                if i < len(input):
                    inp = np.array([input[i]])
                    # a, b, c, d = a + np.random.normal(0, 0.1, (1, 512)), b + np.random.normal(0, 0.1, (
                    # 1, 512)), c + np.random.normal(0, 0.1, (1, 512)), d + np.random.normal(0, 0.1, (1, 512))
                    out, a, b, c, d = self.model.predict([inp] + [a, b, c, d])
                    p.append(int(np.argmax(out[0][0])))

                else:
                    o = np.zeros(self.notes_classes)
                    print("P",p)
                    o[p[-1]] = 1
                    inp = np.array([[o]])
                    out, a, b, c, d = self.model.predict([inp] + [a, b, c, d])
                    a, b = a + np.random.normal(0, 0.1, (1, 512)), b + np.random.normal(0, 0.1, (1, 512))
                    p.append(int(np.argmax(out[0][0])))
            p = self.melody+p[len(self.melody):]
            generated = self.decode_track(p)
            print("Thread:",generated)
            user_created = self.decode_track(self.melody)
            generated[:, :user_created.shape[1]] = -1
            print("NonZeros in Generated",np.nonzero(generated+1))
            generated_outputs.append(generated)
        return generated_outputs

    def inf_generator(self):
        if self.loaded_model != self.model_name:
            print(self.model_name)
            self.model = load_model(self.model_name, custom_objects={"attention": attention})
            self.loaded_model = self.model_name
            print(self.model.summary())
            print("Model loaded from disk")
        else:
            self.build_model()
            print("Already loaded")

        sample = np.load("temp.npy")
        self.input_data_to_model(sample)
        self.melody = self.melody[0]
        if len(self.melody) == 0:
            return
        left_tokens = []
        if len(self.melody) > 50:
            left_tokens = self.melody[:len(self.melody)-50]
            self.melody = self.melody[len(self.melody)-50:]
            req_length = len(self.melody)+20
        else:
            req_length = len(self.melody)+20

        input = []
        output = []
        inm = []
        ino = []

        for i in range(0, len(self.melody) - self.input_length, self.input_length):
            ip = []
            ipm = []
            # print(i, len(self.melody) - self.input_length, step)
            for q in self.melody[i:i + self.input_length]:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                ip.append(o)
                ipm.append(q)
            input.append(ip)
            inm.append(ipm)
            jp = []
            jpm = []
            for q in self.melody[i + 1:i + 1 + self.input_length]:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                jp.append(o)
                jpm.append(q)
            output.append(jp)
            ino.append(jpm)

        # a,b,c,d = np.zeros((512,1)),np.zeros((512,1)),np.zeros((512,1)),np.zeros((512,1))

        a, b, c, d = np.random.normal(0, 1, (1, 512)), np.random.normal(0, 1, (1, 512)), np.random.normal(0, 1, (
            1, 512)), np.random.normal(0, 1, (1, 512))
        print(a.shape)

        p = []
        for i in range(req_length):
            print(i)
            if i < len(input):
                inp = np.array([input[i]])
                out, a, b, c, d = self.model.predict(
                    [inp] + [a, b, c, d])
                p.append(int(np.argmax(out[0][0])))

            else:
                o = np.zeros(self.notes_classes)
                o[p[-1]] = 1
                inp = np.array([[o]])
                out, a, b, c, d = self.model.predict(
                    [inp] + [a, b, c, d])
                a = a + np.random.normal(0, 0.1, (1, 512))
                b = b + np.random.normal(0, 0.1, (1, 512))
                p.append(int(np.argmax(out[0][0])))
        p = left_tokens + self.melody + p[len(self.melody):]
        self.decode_track(p)

    def new_generator(self):

        if self.loaded_model != self.model_name:
            print(self.model_name)
            self.model = load_model(self.model_name, custom_objects={"attention": attention})
            self.loaded_model = self.model_name
            print(self.model.summary())
            print("Model loaded from disk")
        else:
            self.build_model()
            print("Already loaded")

        sample = np.load("temp.npy")
        print(sample.shape)
        print("Input_data_to_model")
        self.input_data_to_model(sample)
        self.melody = self.melody[0]

        while len(self.melody) < self.input_length:
            self.melody.extend(self.melody)

        input = []
        output = []
        inm = []
        ino = []

        for i in range(0, len(self.melody) - self.input_length, self.input_length):
            ip = []
            ipm = []
            # print(i, len(self.melody) - self.input_length, step)
            for q in self.melody[i:i + self.input_length]:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                ip.append(o)
                ipm.append(q)
            input.append(ip)
            inm.append(ipm)
            jp = []
            jpm = []
            for q in self.melody[i + 1:i + 1 + self.input_length]:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                jp.append(o)
                jpm.append(q)
            output.append(jp)
            ino.append(jpm)

            p = []
            count = 0
            for x in input:
                print(count, "/", len(input))
                prediction = self.model.predict(np.array([x]))
                for piece in prediction:
                    for r in piece:
                        p.append(int(np.argmax(r)))
                count += 1
            self.decode_track(p)

    def generate_tune(self):
        try:
            if self.loaded_model != self.model_name:
                self.model = load_model(self.model_name)
                self.loaded_model = self.model_name
                print(self.model.summary())
                print("Model loaded from disk")
            else:
                self.build_model()
                print("Already loaded")

        except Exception:
            print("Error in loading model")

        sample = np.load("temp.npy")
        print(sample.shape)
        print("Input_data_to_model")
        self.input_data_to_model(sample)

        self.melody = self.melody[0]

        # self.melody.pop(-1)
        while len(self.melody) < self.input_length:
            self.melody.extend(self.melody[1:])

        print("Length:", len(self.melody))
        original_length = len(self.melody)
        print(original_length, self.input_length)

        input = []
        output = []
        inm = []
        ino = []
        for i in range(0, len(self.melody) - self.input_length, 1):
            ip = []
            ipm = []
            # print(i, len(self.melody) - self.input_length, step)
            for q in self.melody[i:i + self.input_length]:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                ip.append(o)
                ipm.append(q)
            input.append(ip)
            inm.append(ipm)
            jp = []
            jpm = []
            for q in self.melody[i + 1:i + 1 + self.input_length]:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                jp.append(o)
                jpm.append(q)
            output.append(jp)
            ino.append(jpm)
        # print("inm", "ino")
        # print(inm)
        # print(ino)
        # print("Going to Predict")
        x = list(input[0])
        y = list(output[0])
        # print(x)
        # print(y)
        # input = [x] * 7
        # output = [y] * 7
        ui = np.array(input).shape
        print("Ui shape", ui)
        print(self.model.summary())
        generated = inm[0]
        initial = generated[-self.input_length:]
        e = 0
        while e < self.input_length:
            e += 1
            # j = np.array([generated[-self.input_length:]])
            ip = []
            j = generated[-self.input_length:]
            # for i in range(int(self.input_length / 2)):
            #     r = np.random.randint(self.input_length)
            #     j[r] = np.random.randint(4, 288)
            #     j[r] = initial[r]
            input = []

            for q in j:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                ip.append(o)
                # ipm.append(q)
            input.append(ip)
            prediction = self.model.predict(np.array(input))
            p = []
            for piece in prediction:
                for r in piece:
                    p.append(int(np.argmax(r)))
            generated.append(p[-1])

        if len(inm[0]) - self.input_length > 0:
            t = []
            t.extend(inm[0][:len(inm[0]) - self.input_length])
            t.extend(generated)
            generated = t
        generated = self.decode_track(generated)
        print("generated")
        # print(self.melody)
        # print(generated)
        # self.advance_converter(generated)

class Helper(threading.Thread):
    def __init__(self, notes, model_name, inp_length, h_size):
        threading.Thread.__init__(self)
        self.changed = False
        self.notes = notes  # nparray
        self.model_name = model_name
        self.stop = False
        self.running = True
        self.generated = None
        self.finished = False

        self.jarv = DifferenceMelody(self.model_name, input_length=inp_length, hidden_size=h_size)

    def run(self):
        print("Started")

        while self.running:
            if self.changed:
                self.generated = self.jarv.thread_generator(self.notes)
                self.changed = False
                self.finished = True




if __name__ == '__main__':
    JI = DifferenceMelody("models/" + "master2.h5", hidden_size=256, input_length=25, epochs=6)
    JI.build_model()
    print(JI.model.summary())
