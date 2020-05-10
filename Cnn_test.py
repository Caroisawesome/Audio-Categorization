# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:22:08 2020

@author: alysh
"""
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages

#from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.regularizers import l2
#from keras.callbacks import Callback
# import the necessary packages
#from scikitplot.metrics import plot_confusion_matrix
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import numpy as np
import pickle
import os
import csv
import feed_forward

DIM_ROWS = 234 # max 277
DIM_COLS = 200 # max 372
DIM_CHANNELS = 3 # max 3
TOTAL_INPUT_SIZE = DIM_ROWS*DIM_COLS*DIM_CHANNELS

EPOCHS = 30
BATCH_SIZE = 50
LEARNING_RATE = 0.001
MOMENTUM = 0.9
DECAY = 1e-6
NESTEROV = True
LOSS_FUNCTION = 'categorical_crossentropy'

# Number of training data (remainder of data will go to testing)
NUM_TRAINING = 1840
LEAKY_RELU_ALPHA = 0.01




def initialize_network(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    return model

def train_network(train_labels, train_data, test_labels, test_data, network):
    print('Training con NN network')
    print('====================================================================')
    sgd = SGD(lr=LEARNING_RATE, decay=DECAY, momentum=MOMENTUM, nesterov=NESTEROV)
    network.compile(loss=LOSS_FUNCTION, optimizer=sgd, metrics=['categorical_accuracy'])
    #opt = Adam(lr=LEAKY_RELU_ALPHA, decay=DECAY / EPOCHS)
    #model.compile(loss=LOSS_FUNCTION, optimizer=opt,metrics=['categorical_accuracy'])
    network.fit(train_data, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    print('Con NN trained')
    print('====================================================================')
    return network

def evaluate_network(test_labels, test_data, network):
    (loss, accuracy) = network.evaluate(test_data, test_labels, batch_size=BATCH_SIZE, verbose=1)
    print("loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
    return loss, accuracy

def run_CNN(train_data, train_labels, test_data, test_labels):
    input_shape    = (train_data.shape[1], train_data.shape[2], train_data.shape[3])
    network        = initialize_network(input_shape)
    t_network      = train_network(train_labels, train_data, test_labels, test_data, network)
    loss, accuracy = evaluate_network(test_labels, test_data, t_network)
    feed_forward.save_model(t_network,0)
    acc = accuracy * .01
    interval = 1.96 * math.sqrt( (acc * (1 - acc)) / NUM_TRAINING)
    print("loss", loss)
    print("accuracy", accuracy)
    print("confidence interval:", interval)


if __name__ == '__main__':

    testing_mfccs = True
    testing_spect = False
    testing_timeseries = False

    dir_mfcc = "project_mfccs/train"
    dir_spec = "project_spect/train"
    dir_tmsr = "project_timeseries/train"
    dir_tmsr_1 = "project_time_series/train"

    labels = feed_forward.gen_labels()
    data,labels_list = feed_forward.read_data(dir_tmsr,labels)
    run_CNN(data[:NUM_TRAINING], labels_list[:NUM_TRAINING], data[NUM_TRAINING:], labels_list[NUM_TRAINING:])
