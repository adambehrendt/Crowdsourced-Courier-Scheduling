import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
import utils
from typing import List, Tuple
import sys
import pickle
import time
import keras.callbacks


class MyThresholdCallback(keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        loss = logs["loss"]
        if loss <= self.threshold:
            self.model.stop_training = True


class DiscretePredictor:
    """
    This class is essentially just a wrapper for a keras model
    We do this to seamlessly implement either a traditional neural net (ANN) 
    or a recurrent version (RANN), as described in our paper.
    """
    def __init__(self, in_dim, out_dim, filename, model_type) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.model = filename#pickle.load(open(filename, 'rb'))
        self.model_type = model_type

    def make_rolling_prediction(self, X, num_inputs, num_outputs, history_length):
        """
        Predicts each entry  of z individually.
        """

        y = [0 for _ in range(num_outputs)]
        for j in range(num_outputs):
            X_rolling = np.zeros(num_inputs+history_length+1+1)
            X_rolling[0:num_inputs] = X
            rolling_sum = sum(y[0:j])
            if j < history_length:
                num_zeros = history_length - j
                num_vals = history_length - num_zeros
                history = [0 for _ in range(num_zeros)]
                history.extend(y[0:num_vals])
            else:
                history = y[j-history_length:j]
            X_rolling[num_inputs] = rolling_sum
            X_rolling[num_inputs+1:num_inputs+1+history_length] = history
            X_rolling[num_inputs+1+history_length] = j
            #X_rolling_formatted = np.reshape(X, (X_rolling.shape[0], 1, X_rolling.shape[1]))

            y[j] = self.model.predict(np.array([X_rolling]))[0]

        return y

    def predict(self, X, history_length):
        """
        Predicts the vector z given data X.
        """

        if self.model_type == 'one_shot':
            #X_formatted = np.reshape(X, (X.shape[0], 1, X.shape[1]))
            #X = X.reshape(1, -1)
            y = self.model.predict(X)

        elif self.model_type == 'rolling':
            y = self.make_rolling_prediction(X, self.in_dim, self.out_dim, history_length)

        else:
            print('Invalid Model Type, Quitting.')
            quit()

        return y

def make_one_shot_nn(X_train, X_test, y_train, y_test, in_dim, out_dim, n_epochs, hidden_nodes, es=None):
    """
    Generates an ANN.
    """

    #X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    model = Sequential()
    model.add(Dense(hidden_nodes[0], input_dim=in_dim, activation='relu'))
    for i in range(len(hidden_nodes)-1):
        model.add(Dense(hidden_nodes[i+1], activation='relu'))
    model.add(Dense(out_dim, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    if es is not None:
        model.fit(X_train, y_train, epochs=n_epochs, callbacks=[es])
    else:
        model.fit(X_train, y_train, epochs=n_epochs)
    if X_test is not None:
        #X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        y_predict = model.predict(X_test)
        val = metrics.mean_absolute_error(y_test, y_predict)
        print ("One-Shot Split NN Keras Error: {}".format(val))

    return model


def make_rolling_nn(X_train, X_test, y_train, y_test, in_dim, n_epochs, hidden_nodes, es=None):
    """
    Generates a RANN.
    """

    #X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    model = Sequential()
    model.add(Dense(hidden_nodes[0], input_dim=in_dim, activation='relu'))
    for i in range(len(hidden_nodes)-1):
        model.add(Dense(hidden_nodes[i+1], activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    if es is not None:
        model.fit(X_train, y_train, epochs=n_epochs, callbacks=[es])
    else:
        model.fit(X_train, y_train, epochs=n_epochs)
    if X_test is not None:
        #X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        print(X_test)
        print(X_test.shape)
        y_predict = model.predict(X_test)
        val = metrics.mean_absolute_error(y_test, y_predict)
        print ("Rolling Split NN Keras Error: {}".format(val))

    return model    


def make_rolling_data(X, y, num_inputs, history_length):
    """
    Converts input (X) and output (required courier vector z) into form needed for the RANN.
    """

    mult = len(y[0])
    count = 0
    X_rolling = np.zeros((len(X)*mult, num_inputs+history_length+1+1))
    y_rolling = np.zeros(len(X)*mult)
    for i in range(len(X)):
        copy_index = int(np.floor(mult/(i+.01)))
        for j in range(len(y[i])):
            X_rolling[count, 0:num_inputs] = X[i]
            rolling_sum = sum(y[i, 0:j])
            if j < history_length:
                num_zeros = history_length - j
                num_vals = history_length - num_zeros
                history = [0 for _ in range(num_zeros)]
                history.extend(y[i, 0:num_vals])
            else:
                history = y[i, (j-history_length):j]
            X_rolling[count, num_inputs] = rolling_sum
            X_rolling[count, num_inputs+1:num_inputs+1+history_length] = history
            X_rolling[count, num_inputs+1+history_length] = j
            y_rolling[count] = y[i, j]
            count += 1

    return X_rolling, y_rolling



