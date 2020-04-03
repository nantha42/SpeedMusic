from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.layers import LSTM, BatchNormalization as BatchNorm
from keras.callbacks import LambdaCallback
from keras.models import load_model
from keras.utils import Sequence

import numpy as np
import pickle
import os

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
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                ip.append(o)

            batch_x.append(ip)
            hot_encoded = []
            for val in self.y[ind]:
                t = np.zeros(self.notes_classes)
                t[int(val)] = 1
                hot_encoded.append(t)
            batch_y.append(hot_encoded)
        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        # np.random.shuffle(self.indices)
        np.random.shuffle(self.musicdataset)
        pass

class EncoderModel:

    def __init__(self):
        self.inp_length = 128
        self.notes_classes = 87
        self.inp_dim = (self.inp_length, self.notes_classes)
        self.dropout = 0.3
        self.hidden_size = 512

        self.epochs = 1
        self.loaded_model = ""

        self.directory = ""
        self.input_data = None
        self.model_name = None
        self.inp = None
        self.out = None
        self.melody = []
        self.history = None
        self.musicdataset = []
        self.epochs_ran = 0
        self.batch_size = 157

        optimizer = RMSprop()
        self.encoder = None
        self.decoder = None
        self.model = None

        self.build_model()
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()
        self.melody = []

    def build_model(self):
        self.encoder = Sequential()

        self.encoder.add(
            LSTM(self.hidden_size, input_shape=(self.inp_length, self.notes_classes), return_sequences=True,
                 recurrent_dropout=self.dropout))
        self.encoder.add(LSTM(self.hidden_size, recurrent_dropout=self.dropout, return_sequences=True))
        self.encoder.add(BatchNorm())
        self.encoder.add(Dropout(self.dropout))
        self.encoder.add(Dense(256, activation="relu"))
        self.encoder.add(BatchNorm())
        self.encoder.add(Dense(128, activation="relu"))
        self.encoder.add(BatchNorm())
        self.encoder.add(Dense(64, activation="relu", name="encoder_out"))

        self.decoder = Sequential()
        # self.decoder.add(Input(shape=(128, 87)))
        self.decoder.add(Dense(64, name="decoder_in", activation="relu"))
        self.decoder.add(BatchNorm())
        self.decoder.add(Dense(128, activation="relu"))
        self.decoder.add(BatchNorm())
        self.decoder.add(Dense(256, activation="relu"))
        self.decoder.add(LSTM(self.hidden_size, recurrent_dropout=self.dropout, return_sequences=True))
        self.decoder.add(LSTM(self.hidden_size, recurrent_dropout=self.dropout, return_sequences=True))
        self.decoder.add(Dropout(self.dropout))
        self.decoder.add(Dense(256, activation="relu"))
        self.decoder.add(Dense(128, activation="relu"))
        self.decoder.add(BatchNorm())
        self.decoder.add(Dense(self.notes_classes, activation="softmax"))

        decoded_inp = Input(shape=(128, 64))
        decoded_out = self.decoder(decoded_inp)
        self.decoder = Model(decoded_inp, decoded_out)

        self.model = Sequential()
        self.model.add(self.encoder)
        self.model.add(self.decoder)
        self.model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=['categorical_accuracy'])

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
                        encoded.extend([curnote + 7, time_quantum + 1])  # 24 is the starting sequence for all songs

                if not starting:
                    i = get_top_index(quantum_notes)
                    if i > -1:
                        if rest > 0:
                            if rest > 32:
                                rest = 32
                            encoded.append(54 + rest)
                            rest = 0
                        # note is lower than startnote then curnote is high
                        # note is higher than startnote then curnote is low
                        time_quantum = quantum_notes[i]
                        curnote = 24 + startnote - i
                        if 0 <= curnote <= 47:
                            encoded.extend([curnote + 7, time_quantum + 1])


            elif not starting:
                rest += 1

        # print(encoded)
        return encoded

    def decode_track(self, encoded):

        orray = np.ones((48, 1)) * -1

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
                            orray = np.append(orray, array, axis=1)
            elif 55 <= x <= 86:
                x = x - 54
                array = np.ones((48, x)) * -1
                orray = np.append(orray, array, axis=1)
        np.save("generated.npy", orray)

        return orray

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
            self.melody[i].append(1)
        print("minised", minlen)
        print("Minimised Applied")

    def melody_to_datasetver3(self, option=0):

        self.minimise()
        step = 2
        if option == 1:
            step = self.inp_length
        input = []
        output = []
        set_size = 0
        first = True
        for t in range(len(self.melody)):

            for i in range(0, len(self.melody[t]) - self.inp_length, step):
                ip = []
                for q in self.melody[t][i:i + self.inp_length]:
                    ip.append(int(q))
                input.append(ip)
                jp = []
                for q in self.melody[t][i + 1:i + 1 + self.inp_length]:
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

    def load_saved(self, savefile="chordload.pkl", option=0, google_drive=False):
        """
        This loads the npy data in the folder saved and use it
        train the
        :return:
        """

        if not google_drive:
            high_npy_path = "high_npy/"

        else:
            high_npy_path = "drive/My Drive/DataProject/extracted/high_npy/"

        savefilename = savefile

        if option == 1:
            list_of_files = os.listdir(".")
            if savefilename in list_of_files:
                default_option = False
            else:
                default_option = True
        else:
            default_option = True

        if default_option:
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

    def input_data_to_model(self, nparray):
        # 0 for starting music piece
        # 1-6 for time of note
        # 7-54 for notes
        # 55-86 for rest notes
        temp = self.encode_track(nparray)
        self.melody.append(temp)

    def format(self, unformated):
        batch_x = []
        for mupiece in unformated:
            ip = []
            for q in mupiece:
                o = np.zeros(self.notes_classes)
                o[int(q)] = 1
                ip.append(o)

            batch_x.append(ip)
        # print("SHape", np.array(batch_x).shape)
        return np.array(batch_x)

    def deformat(self, formated):
        deformated = []
        for piece in formated:
            out = []
            for _87 in piece:
                j = np.argmax(_87)
                out.append(j)
            deformated.append(out)
        return deformated

    def train(self, epochs):
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
        self.history = self.model.fit_generator(generator, validation_data=validation_generator, epochs=10,
                                                verbose=1, callbacks=[print_callback])



    def on_epoch_end(self, epoch_end, x):
        self.epochs_ran += 1
        print("@on_epoch_end saving model to disk")
        # path  = "models/"
        path="drive/My Drive/DataProject/high_npy/"
        self.encoder.save(path+"Encoder")
        self.decoder.save(path+"saved_model")
        self.model.save(self.model_name)


if __name__ == '__main__':
    encoder = EncoderModel()
    encoder.load_saved("diffload128.pkl",google_drive=True,option=1)

    # print(gan.combined.summary())
    # gan.load_saved("sample.pkl", option=1)
    # print(gan.musicdataset)
    # gan.train(1)
