"""
from keras.callbacks import LambdaCallback
print("impoted LambdaCallback")

print("impoted Sequential")

print("impoted Dense")

print("impoted LSTM")
from keras.optimizers import RMSprop
print("imported RMSprop")
"""
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Activation
from keras.models import load_model
from keras.callbacks import LambdaCallback
from keras.utils import Sequence
import keras.metrics as metrics
import numpy as np
import os


class Advance_Melody:
    def __init__(self, epochs=10):
        self.hidden_size = 289
        self.input_length = 32
        self.notes_classes = 289
        self.model = Sequential()
        self.epochs = epochs
        self.model.add(LSTM(self.hidden_size, input_shape=(32, 289), return_sequences=True), )
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(self.hidden_size, return_sequences=True), )
        self.model.add(Dropout(0.3))
        self.model.add(Dense(300))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.notes_classes))
        self.model.add(Activation('relu'))
        self.input_data = None
        self.model.compile(loss="mean_squared_error", optimizer="adam", metrics=[metrics.accuracy])
        self.melody = []

    def input_data_to_model(self, nparray):
        maxlen = 32
        for j in range(nparray.shape[1]):
            row = nparray[:, j]
            temp = []
            for j in range(48):
                if row[j] != -1:
                    temp.append(j * 6 + row[j])
            if len(temp) > 0:
                self.melody.append(temp)
        # print(self.melody)
        xx = np.array(self.melody)
        print("xx", xx.shape)

    def melody_to_datas(self):
        input = []
        output = []
        print(len(self.melody))
        for i in range(0, len(self.melody) - 32, 2):
            all = []
            for q in self.melody[i:i + 32]:
                a = np.zeros(self.notes_classes)
                for sec in q:
                    # print(int(sec))
                    a[int(sec)] = 1
                print(a.shape)
                all.append(a)
            allnpy = np.array(all)
            print(allnpy.shape)
        pass

    def melody_to_dataset(self):
        input = []
        output = []
        print("In to Dataset", len(self.melody))
        for i in range(0, len(self.melody) - 32, 2):
            # input.append(melody[i:i+32])
            ip = []
            # print(self.melody[i:i + 32])
            for q in self.melody[i:i + 32]:
                o = np.zeros(self.notes_classes)
                # print("q:",q)
                for secondnotes in q:
                    o[int(secondnotes)] = 1

                # print(o)
                ip.append(np.array(o))
                print("IP Length", len(ip))
            input.append(np.array(ip))
            print("Input Length", len(input))
            # input = np.array(input)
            # print(len(input))
            # print(input.shape)

            jp = []
            for q in self.melody[i + 1:i + 1 + 32]:
                o = np.zeros(self.notes_classes)
                for secondnotes in q:
                    o[int(secondnotes)] = 1
                jp.append(np.array(o))
            output.append(np.array(jp))

        input = np.array(input)
        output = np.array(output)
        print("Input", input.shape, "Output", output.shape)
        print("Printing input", input)
        return [input, output]

    def isprime(self, x):
        for i in range(2, int(np.sqrt(x))):
            if x % i == 0:
                return False
        return True

    def melody_to_testdata(self):
        input = []
        output = []

        for i in range(0, len(self.melody) - 32, 32):
            # input.append(melody[i:i+32])
            ip = []
            for q in self.melody[i:i + 32]:
                o = np.zeros(self.notes_classes)
                for secondnotes in q:
                    o[int(secondnotes)] = 1
                ip.append(o)
            input.append(ip)
            jp = []
            for q in self.melody[i + 1:i + 1 + 32]:
                o = np.zeros(self.notes_classes)
                for secondnotes in q:
                    o[int(secondnotes)] = 1
                jp.append(o)
            output.append(jp)

        return [np.array(input), np.array(output)]

    def train(self, x, y):
        try:
            self.model = load_model("advance_flight_model.h5")
            print("Model loaded from disk")
        except Exception:
            print("Error")

        print(x.shape, y.shape)
        if self.isprime(x.shape[0]):
            x = x[:-1]
            y = y[:-1]
        print("Newshape", x.shape, y.shape)
        self.input_data = x

        # self.model.fit(z,[y[0]],epochs=5,verbose=1)
        self.a, self.b = self.melody_to_testdata()
        print_callback = LambdaCallback(self.on_epoch_end)
        print("XY", x.shape, y.shape)
        self.model.fit(x, y, epochs=self.epochs, verbose=1, callbacks=[print_callback])
        self.model.save("advance_flight_model.h5")
        a, b = self.melody_to_testdata()
        print("Predicting", self.a.shape)
        prediction = self.model.predict(self.a)
        print(prediction.shape)
        generated = []
        for measure in prediction:
            for row in measure:
                high_key_probability_index = np.argwhere(row > 0.89)
                if len(high_key_probability_index) > 0:
                    generated.append(high_key_probability_index)

        # print(generated)
        # print("Set of Generated", set(generated))
        self.converter(generated)

    def load_saved(self):
        """
        This loads the npy data in the folder saved and use it
        train the
        :return:
        """
        for saved in os.listdir("saved_advance/"):
            sample = np.load("saved_advance/" + saved)
            self.input_data_to_model(sample)
        # print(self.melody)
        inp, out = self.melody_to_dataset()
        # self.melody_to_datas()

        # print("InpShape", inp.shape)
        self.train(inp, out)

    def converter(self, j):
        orray = np.ones([48, 1]) * -1
        # print(j)
        for i in range(len(j)):
            krray = np.ones([48, 1]) * -1
            values = []
            for value in j[i]:
                x = value
                p = int(x / 6)
                v = x % 6
                krray[p, 0] = v
                values.append(int(v))
            # print("Error@",j[i])
            orray = np.append(orray, krray, axis=1)
            # print("Appending Spaces to Orray",0, (2 ** np.max(values)) - 1)
            for ta in range(0, (2 ** np.max(values)) - 1):
                krray = np.ones([48, 1]) * -1
                orray = np.append(orray, krray, axis=1)

        # print("orray shape", orray.shape)
        np.save("generated.npy", orray)
        print("stored successfully")

    def on_epoch_end(self, epoch_end, x):

        print("a shape", self.a.shape)
        prediction = self.model.predict(self.a)
        print("Prediction shape", prediction.shape)
        generated = []
        for measure in prediction:
            for row in measure:
                high_key_probability_index = np.argwhere(row > 0.89)
                if len(high_key_probability_index) > 0:
                    generated.append(high_key_probability_index)

        # print(generated)
        # print("Set of Generated", set(generated))
        print("Length of Generated", len(generated))
        self.converter(generated)
        print("saving model.....")
        self.model.save("advance_flight_model.h5")
        print("Model saved")


class Generator(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, x_set, y_set, batch_size=4):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])
        self.musicdataset = []
        self.melody = []

    def __len__(self):
        print("Infinitywar", self.x.shape[0], self.batch_size, np.ceil(self.x.shape[0] / self.batch_size))
        print("x shape", self.x.shape)
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        return batch_x, batch_y

    def on_epoch_end(self):
        # np.random.shuffle(self.indices)
        """
        idxs = list(range(len(self.musicdataset)))
        self.melody = []
        np.random.shuffle(idxs)
        #print("Idxs", idxs)
        for id in idxs:
            self.input_data_to_model(self.musicdataset[id])
        self.x, self.y = self.melody_to_dataset()
        rr = self.x
        #print("Shape of melody", rr.shape)
        """
        self.melody = []
        list_offiles = os.listdir("saved/")
        choosen_files = []
        for r in range(7):
            while True:
                ch = np.random.randint(0, len(list_offiles))
                if list_offiles[ch][-4:] == ".npy":
                    choosen_files.append(list_offiles[ch])
                    break

        for saved in choosen_files:
            if saved[-4:] == ".npy":
                print(saved)
                sample = np.load("saved/" + saved)
                self.input_data_to_model(sample)
                # self.musicdataset.append(sample)
        inp, out = self.melody_to_dataset()
        print("InpShape", inp.shape)

    def input_data_to_model(self, nparray):
        rest = 0
        rest_index = []

        # Rest array preparation
        for i in range(1, 32):
            rest_index.append(i)
        rest_index = np.array(rest_index)
        # print("Length of Melody",len(self.melody))
        for i in range(nparray.shape[1]):
            row = nparray[:, i]
            notefound = False
            for j in range(48):
                if row[j] != -1:
                    self.melody.append(j * 6 + row[j])
                    notefound = True
                    break
            # Below code is for inserting a symbol for rest duration in music
            if notefound:
                if rest > 0:
                    # e = np.abs(rest_index-rest).argmin()
                    n32rests = rest / 32
                    remrest = rest % 32
                    for o in range(int(n32rests)):
                        self.melody.append(287 + 32)
                    # index = 288+e
                    self.melody.append(287 + remrest)
                rest = 0
            else:
                rest += 1
        # print("Melody is", self.melody)
        # print(set(self.melody))
        ay = np.bincount(self.melody)
        ai = np.nonzero(ay)[0]
        # print("Frequency",np.vstack((ai,ay[ai])).T)

    def melody_to_dataset(self):
        input = []
        output = []
        for i in range(0, len(self.melody) - 128, 3):
            # input.append(melody[i:i+32])
            ip = []
            for q in self.melody[i:i + 128]:
                o = np.zeros(321)
                o[int(q)] = 1
                ip.append(o)
            input.append(ip)
            jp = []
            for q in self.melody[i + 1:i + 1 + 128]:
                o = np.zeros(321)
                o[int(q)] = 1
                jp.append(o)
            output.append(jp)

        return [np.array(input), np.array(output)]


class Melody:
    def __init__(self, model_name, epochs=10):
        self.hidden_size = 321
        self.input_length = 128
        self.notes_classes = 321
        self.model = Sequential()
        self.epochs = epochs
        # self.model.add(Embedding(self.notes_classes, self.hidden_size, input_length=self.input_length))
        self.model.add(LSTM(self.hidden_size, input_shape=(128, 321), return_sequences=True), )
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(self.hidden_size, return_sequences=True), )
        self.model.add(Dropout(0.3))
        self.model.add(Dense(300))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.notes_classes))
        self.model.add(Activation('softmax'))
        self.input_data = None

        """
        #self.model.add(LSTM(self.hidden_size, return_sequences=True))
        #self.model.add(Dropout(0.3))"""
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        self.melody = []
        self.musicdataset = []
        self.model_name = model_name
        pass

    def input_data_to_model(self, nparray):
        rest = 0
        rest_index = []

        # Rest array preparation
        for i in range(1, 32):
            rest_index.append(i)
        rest_index = np.array(rest_index)

        for i in range(nparray.shape[1]):
            row = nparray[:, i]
            notefound = False
            for j in range(48):
                if row[j] != -1:
                    self.melody.append(j * 6 + row[j])
                    notefound = True
                    break
            # Below code is for inserting a symbol for rest duration in music
            if notefound:
                if rest > 0:
                    # e = np.abs(rest_index-rest).argmin()
                    n32rests = rest / 32
                    remrest = rest % 32
                    for o in range(int(n32rests)):
                        self.melody.append(287 + 32)
                    # index = 288+e
                    self.melody.append(287 + remrest)

                rest = 0
            else:
                rest += 1

        # print("Melody is", self.melody)
        # print(set(self.melody))
        ay = np.bincount(self.melody)
        ai = np.nonzero(ay)[0]
        # print("Frequency",np.vstack((ai,ay[ai])).T)

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

    def isprime(self, x):
        for i in range(2, int(np.sqrt(x))):
            if x % i == 0:
                return False
        return True

    def train(self, x, y):
        try:
            self.model = load_model(self.model_name)
            print("Model loaded from disk")
        except Exception:
            print("Error")

        print(x.shape, y.shape)
        if self.isprime(x.shape[0]):
            x = x[:-1]
            y = y[:-1]
        print("Newshape", x.shape, y.shape)
        self.input_data = x

        # self.model.fit(z,[y[0]],epochs=5,verbose=1)
        self.a, self.b = self.melody_to_testdata()
        print_callback = LambdaCallback(self.on_epoch_end)
        # generator = Generator(x,y)
        # generator.musicdataset =self.musicdataset
        # self.model.fit_generator(generator,epochs=self.epochs,verbose=1,callbacks=[print_callback])
        print("Calling model.fit()")
        self.model.fit(x, y, epochs=self.epochs, validation_split=0.2, verbose=1, callbacks=[print_callback])

        self.model.save(self.model_name)
        a, b = self.melody_to_testdata()
        print("Predicting", self.a.shape)
        prediction = self.model.predict(self.a)
        print(prediction.shape)
        generated = []
        for measure in prediction:
            for row in measure:
                generated.append(np.argmax(row))
        # print(generated)
        print("Set of Generated", set(generated))
        self.converter(generated)

    def converter(self, j):
        print("@Converter")
        orray = np.ones([48, 1]) * -1
        # print(j)
        for i in range(len(j)):
            krray = np.ones([48, 1]) * -1
            if i%1000 == 0:
                print(i,len(j))
            if j[i] < 288:
                x = j[i]
                p = int(x / 6)
                v = x % 6
                krray[p, 0] = v

                orray = np.append(orray, krray, axis=1)
                for ta in range(0, (2 ** v) - 1):
                    krray = np.ones([48, 1]) * -1
                    orray = np.append(orray, krray, axis=1)
            else:  # for adding rest rows to the melody
                x = j[i] - 287
                for z in range(x):
                    orray = np.append(orray, krray, axis=1)
                    # print("orray shape", orray.shape)
        print("@Converter Entered data into orray")
        print("@Converter Now Saving orray")
        np.save("generated.npy", orray)
        print("stored successfully")

    def on_epoch_end(self, epoch_end, x):

        print("a shape", self.a.shape)
        prediction = self.model.predict(self.a)

        print("Prediction shape", prediction.shape)
        generated = []
        for measure in prediction:
            for row in measure:
                generated.append(np.argmax(row))
        # print(generated)
        print("Set of Generated", set(generated))
        print("Length of Generated", len(generated))
        print("Calling Converter")
        self.converter(generated)
        print("Bact to on_epoch_end")
        self.model.save(self.model_name)

    def melody_to_dataset(self, option=0):
        step = 5
        if option == 1:
            step = self.input_length
        print(len(self.melody))
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
        # print("Out of loop")
        # print(input)
        # print(n)
        print("Returning input and output from func melody_to_dataset")
        return [np.array(input), np.array(output)]

    def load_saved(self, option=0):
        """
        This loads the npy data in the folder saved and use it
        train the
        :return:
        """
        list_offiles = os.listdir("saved/")
        choosen_files = []
        """
        for r in range(10):
            while True:
                ch = np.random.randint(0, len(list_offiles))
                if list_offiles[ch][-4:] == ".npy":
                    choosen_files.append(list_offiles[ch])
                    break
        """
        #print("Random Done")
        #for saved in choosen_files:
        for saved in os.listdir("saved/"):
            if saved[-4:] == ".npy":
                print("Saved", saved)
                sample = np.load("saved/" + saved)
                self.input_data_to_model(sample)
                # self.musicdataset.append(sample)

        inp, out = self.melody_to_dataset()
        print("Got input output from melody_to_dataset")
        print("Printing Input Shape", inp.shape)
        print("Starting Training")
        self.train(inp, out)

    def use_model(self):

        try:
            self.model = load_model(self.model_name)
            print("Model loaded from disk")
        except Exception:
            print("Error")

        sample = np.load("temp.npy")
        print("sample.shape", sample.shape)
        self.input_data_to_model(sample)
        x, y = self.melody_to_dataset(option=1)
        print(x.shape)
        try:
            prediction = self.model.predict(x)
            generated = []
            for measure in prediction:
                for row in measure:
                    generated.append(np.argmax(row))
            # print(generated)
            print("Length of Generated", generated)
            self.converter(generated)
        except(Exception):
            print("Error", x.shape)

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


if __name__ == '__main__':
    # data = np.load('train_data.npy')
    AI = Melody("newmodel.h5", 3)
    AI.load_saved()
    # AI.test_model()
    # print(dir(AI))
    # AI.input_data_to_model(data)
