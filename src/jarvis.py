import traceback
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Activation
from keras.models import load_model
from keras.callbacks import LambdaCallback
from keras.utils import Sequence
import keras.metrics as metrics
import numpy as np
import os



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


class Generator2(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, x_set, y_set, batch_size=4):
        self.x, self.y = x_set, y_set

        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])
        self.musicdataset = []
        self.melody = []

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        for ind in inds:
            batch_x.append(self.x[ind])
            hot_encoded = []
            for val in self.y[ind]:
                t = np.zeros(323)
                t[int(val)] = 1
                hot_encoded.append(t)
            batch_y.append(hot_encoded)
            # batch_x = np.array(self.x[idx * self.batch_size:(idx + 1) * self.batch_size])
            # batch_y = np.array(self.y[idx * self.batch_size:(idx + 1) * self.batch_size])

        return np.array(batch_x), np.array(batch_y)

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
        """
        np.random.shuffle(self.indices)


    def input_data_to_model(self, nparray):
        rest = 0
        rest_index = []

        # Rest array preparation
        for i in range(1, 32):
            rest_index.append(i)
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
        self.notes_classes = 323
        self.model = Sequential()
        self.epochs = epochs
        # self.model.add(Embedding(self.notes_classes, self.hidden_size, input_length=self.input_length))
        self.model.add(LSTM(self.hidden_size, input_shape=(self.input_length,self.notes_classes), return_sequences=True), )
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(self.hidden_size, return_sequences=True), )
        self.model.add(Dropout(0.3))
        self.model.add(Dense(300))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.notes_classes))
        self.model.add(Activation('softmax'))
        self.input_data = None
        self.rest_index = []
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
        """
        if  len(self.rest_index) == 32:
            rest_index = self.rest_index
        else:
            for i in range(1, 32):
                self.rest_index.append(i)
            rest_index = self.rest_index
        """
        self.melody.append(0)
        for i in range(nparray.shape[1]):
            row = nparray[:, i]
            notefound = False
            for j in range(48):
                if row[j] != -1:
                    self.melody.append(2 + j * 6 + row[j])
                    notefound = True
                    break

            # Below code is for inserting a symbol for rest duration in music
            if notefound:
                if rest > 0:
                    # e = np.abs(rest_index-rest).argmin()
                    n32rests = rest / 32
                    remrest = rest % 32
                    for o in range(int(n32rests)):
                        self.melody.append(2+287 + 32)
                    self.melody.append(2+287 + remrest)
                rest = 0
            else:
                rest += 1
        self.melody.append(1)
        print("Self.melody", self.melody)

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

    def train2(self,x,y):
        try:
            self.model = load_model(self.model_name)
            print("Model loaded from disk")
        except Exception:
            print("Error")
        if self.isprime(x.shape[0]):
            x = x[:-1]
            y = y[:-1]

        generator = Generator2(x,y,batch_size=30)


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
        # self.model.fit(x, y, epochs=self.epochs, validation_split=0.2, verbose=1, callbacks=[print_callback])

        generator = Generator()
        self.model.fit_generator()
        self.model.save(self.model_name)
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
            if i % 1000 == 0:
                print(i, len(j))
            if j[i] < 288+2 and j[i]>1:
                x = int(j[i])-2
                p = int(x / 6)
                v = int(x) % 6
                krray[p, 0] = v

                orray = np.append(orray, krray, axis=1)
                for ta in range(0, (2 ** v) - 1):
                    krray = np.ones([48, 1]) * -1
                    orray = np.append(orray, krray, axis=1)
            else:  # for adding rest rows to the melody
                x = j[i] - 287-2
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
        print("Returning input and output from func melody_to_dataset")
        return [np.array(input), np.array(output)]

    def melody_to_datasetver2(self, option=0):
        step = 2
        if option == 1:
            step = self.input_length

        input = []
        output = []
        for i in range(0, len(self.melody) - self.input_length, step):
            ip = []
            # print(i, len(self.melody) - self.input_length, step)
            for q in self.melody[i:i + self.input_length]:
                ip.append(int(q))
            input.append(ip)
            jp = []
            for q in self.melody[i + 1:i + 1 + self.input_length]:
                #o = np.zeros(self.notes_classes)
                #o[int(q)] = 1
                jp.append(int(q))
            output.append(jp)
        print("Returning input and output from func melody_to_dataset")
        return [input,output]

    def random_noise(self):
        # self.melody = []
        try:
            print("Loading model", self.model_name)
            self.model = load_model(self.model_name)
            print("Model Loaded")
        except Exception:
            print("Error in loading model", self.model_name)
        # print(dir(self.model))
        # print("Self.melody",self.melody.layers)
        k = self.melody[1:-1]
        notes_used = list(set(k))

        print("Notesused", notes_used)
        for i in range(290):
            self.melody.append(np.random.choice(notes_used))
        x, y = self.melody_to_dataset(option=1)
        print(x.shape)

        print("Predicting")
        print(x.shape)
        print(self.model.summary())
        prediction = self.model.predict(x)
        generated = []
        print("Length of prediction", len(prediction))
        for measure in prediction:
            for row in measure:
                generated.append(np.argmax(row))
        # print(generated)
        print("Length of Generated", generated)
        self.converter(generated)
        #except Exception:
        print("Error in Random generation")
        pass

    def load_saved(self):
        """
        This loads the npy data in the folder saved and use it
        train the
        :return:
        """
        """
        for r in range(10):
            while True:
                ch = np.random.randint(0, len(list_offiles))
                if list_offiles[ch][-4:] == ".npy":
                    choosen_files.append(list_offiles[ch])
                    break
        """
        directory = "large_save/"
        # print("Random Done")
        # for saved in choosen_files:
        for saved in os.listdir(directory):
            if saved[-4:] == ".npy" or saved[-4:] == ".npz":
                print("Saved", saved)
                sample = np.load(directory + saved)
                self.input_data_to_model(sample["arr_0"])
                # self.musicdataset.append(sample)
        inp, out = self.melody_to_datasetver2()
        #print(inp.shape)
        #print("Got input output from melody_to_dataset")
        #print("Printing Input Shape", inp.shape)
        #print("Starting Training")
        #self.train(inp, out)
        self.train2(inp,out)

    def generate_tune(self):
        try:
            self.model = load_model(self.model_name)
            print("Model loaded from disk")
        except Exception:
            print("Error in loading model")
        sample = np.load("temp.npy")
        self.input_data_to_model(sample)
        self.melody.pop(-1)
        print("After Pop",self.melody)
        # x, y = self.melody_to_dataset(option=1)
        # step = 5
        print("Length:", len(self.melody))
        original_length = len(self.melody)
        print(original_length, self.input_length)
        if original_length < self.input_length:
            rem = self.input_length - original_length
            self.melody.extend([68] * (rem + 10))
            exception_occured = False
            while original_length < self.input_length:
                print("Inside Loop")
                for r in range(rem):
                    if exception_occured:
                        print("Exception")
                        break
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
                    print("Reinserting generated into melody", generated[original_length])
                    self.melody[original_length] = generated[original_length]
                    original_length += 1

                    print("Exception Occured")
                    exception_occured = True
                if exception_occured:
                    print("Exception occured")
                    break
                print(self.melody)
            self.converter(self.melody)

    def use_model(self):
        try:
            self.model = load_model(self.model_name)
            print("Input_Sequence_length")
            print(dir(self.model))
            print(self.model._feed_input_shapes)
            # print(self.model.input_sequence_length)
            print("Model loaded from disk")
        except Exception:
            print(traceback.print_exc())

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
    AI = Melody("modelA.h5", 3)
    AI.load_saved()
    # AI.test_model()
    # print(dir(AI))
    # AI.input_data_to_model(data)
