from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Activation
from keras.models import load_model
from keras.models import Model
from keras.layers import BatchNormalization as BatchNorm
import matplotlib.pyplot as plt

from keras.callbacks import LambdaCallback
from keras.utils import Sequence
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import keras.metrics as metrics
import numpy as np
import os
import time
import pickle



def recon():
    model = tf.keras.models.load_model("drive/My Drive/DataProject/models/diffretstate.h5")
    model.summary()

    state_h1 = Input(shape=(512,))
    state_c1 = Input(shape=(512,))

    state_h2 = Input(shape=(512,))
    state_c2 = Input(shape=(512,))
    lstm1_initial_states = [state_h1, state_c1]
    lstm2_initial_states = [state_h2, state_c2]

    inp_data = model.inputs[0]

    print(inp_data)
    print(model.layers)
    lstm1_out, h1, c1 = model.layers[1](inp_data, initial_state=lstm1_initial_states)
    lstm_1_states = [h1, c1]
    lstm2_out, h2, c2 = model.layers[2](lstm1_out, initial_state=lstm2_initial_states)
    lstm_2_states = [h2, c2]

    recent_out = model.layers[3](lstm2_out)
    for i in range(4, len(model.layers)):
        recent_out = model.layers[i](recent_out)
    final_model = Model([inp_data] + [lstm1_initial_states, lstm2_initial_states],
                        [recent_out] + [lstm_1_states, lstm_2_states])
    final_model.summary()
    # lstm2_out, h2,c2 = model.layers[2](lstm1_out,initial_state=lstm2_initial_states)
    final_model.save("drive/My Drive/DataProject/models/recon.h5")
    print(h1)
    print(c1)


dict_model = recon()
# inputs = model.input()
# decoder = Decoderr()
# decoder.generate()

