import matplotlib
matplotlib.use('Agg')
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback
from keras import models
from keras import layers
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LeakyReLU
import tensorflow as tf
from PIL import Image
#from imutils import paths
#import imutils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scikitplot.metrics import plot_confusion_matrix, plot_roc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import argparse
import pickle
import os
import sys
import csv
import neptune
import math

# Dimensions of data being passed into network
DIM_ROWS = 234 # max 277
DIM_COLS = 200 # max 372
DIM_CHANNELS = 3 # max 3
TOTAL_INPUT_SIZE = DIM_ROWS*DIM_COLS*DIM_CHANNELS

# NN Parameters

EPOCHS = 20
BATCH_SIZE = 100
LEARNING_RATE = 0.00001
MOMENTUM = 0.5
DECAY = 1e-7
NESTEROV = True
LOSS_FUNCTION = 'categorical_crossentropy'

# Number of training data (remainder of data will go to testing)
NUM_TRAINING = 1840
LEAKY_RELU_ALPHA = 0.01

USE_NEPTUNE = False

if (USE_NEPTUNE):
    neptune.init('carolyna/Audio-Categorization')
    neptune.create_experiment(name='test-experiment')

#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
"""

    Various functions and implementations borrowed or inspired from:

    https://medium.com/@navdeepsingh_2336/identifying-the-genre-of-a-song-with-neural-networks-851db89c42f0
    
    and

    https://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras/

"""


'''
===============================================================================

class Neptune Logger - tracks loss and accuracies throughout training process
                       results are sent to our user account at Neptune.ai

===============================================================================
'''
class NeptuneLoggerCallback(Callback):
    def __init__(self, model, validation_data):
        super().__init__()
        self.model = model
        self.validation_data = validation_data

    def on_batch_end(self, batch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(f'batch_{log_name}', log_value)

    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(f'epoch_{log_name}', log_value)

        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
        y_true = self.validation_data[1]

        y_pred_class = np.argmax(y_pred, axis=1)
        y_true_class = np.argmax(y_true, axis=1)

        fig, ax = plt.subplots(figsize=(16, 12))

        plot_confusion_matrix(y_true_class, y_pred_class, ax=ax)
        neptune.log_image('confusion_matrix', fig)
        plt.close()


'''
===============================================================================

get_labels - generates a list of labels from the train.csv file

@params: void

@returns: labels - a list of labels
===============================================================================
'''
def gen_labels():
    labels   = {}
    #genres   = [0, 1, 2, 3, 4, 5]

    with open('data/project3/train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        tmp = list(list(rec) for rec in csv.reader(csvfile, delimiter=','))
        for row in tmp:
            if row[0] == 'new_id':
                continue
            labels[row[0]] = row[1]
    return labels

'''
===============================================================================

read_data :: Read image data and store as numpy array

@params: directory, :: directory to read input files from.
         labels     :: labels corresponding to both training and test data

@returns: data,     :: array of flattened data arrays.
          labels_l  :: list of labels in order of actual files.
===============================================================================
'''
def read_data(directory, labels):
    t_files  = []
    names    = []
    data     = []
    labels_l = []
    print('Walking data directory')
    print('====================================================================')
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            t_files.append(os.path.join(root, name))
            base = os.path.splitext(name)[0]
            names.append(base)

    print('Reading images')
    print('====================================================================')
    data     = np.zeros((len(t_files),DIM_ROWS,DIM_COLS,DIM_CHANNELS))
    for i in range(0, len(t_files)):
        image = mpimg.imread(t_files[i])
        np_image = np.array(image)[:DIM_ROWS, :DIM_COLS, :DIM_CHANNELS]
        data[i] = np_image
        labels_l.append(labels[names[i]])
        #print(data[i]) 

    labels_l = to_categorical(labels_l)
    return data, labels_l

'''
===============================================================================

initialize_network :: Creates feed forward neural network

@params: void

@returns: network
===============================================================================
'''
def initialize_network():
    print('Initializing feed forward network')
    print('====================================================================')
    network = models.Sequential()
    network.add(layers.Flatten(input_shape=[ DIM_ROWS, DIM_COLS, DIM_CHANNELS ]))
    network.add(Dense(500, input_dim=TOTAL_INPUT_SIZE, init='uniform', activation='relu'))
    #network.add(Dense(256, activation='relu', kernel_initializer='uniform'))
    network.add(Dense(200, activation='relu', kernel_initializer='uniform'))
    network.add(Dense(150, activation='tanh', kernel_initializer='uniform'))
    network.add(Dense(100, activation='tanh', kernel_initializer='uniform'))
    #network.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))
    #network.add(Dense(50, activation='tanh', kernel_initializer='uniform'))
    network.add(Dense(50, activation='relu', kernel_initializer='uniform'))
    #network.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))
    #network.add(Dense(20, activation='tanh', kernel_initializer='uniform'))
    network.add(Dense(10, activation='relu', kernel_initializer='uniform'))
    #network.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))
    network.add(Dense(6, activation='softmax'))
    print('Feed forward network initialized')
    print('====================================================================')
    return network

'''
===============================================================================

train_network :: Using stochastic gradient descent.

@params: train_labels, train_data, test_labels, test_data, network

@returns: network - feed forward neural network model
===============================================================================
'''
def train_network(train_labels, train_data, test_labels, test_data, network):
    print('Training feed forward network')
    print('====================================================================')
    sgd = SGD(lr=LEARNING_RATE, decay=DECAY, momentum=MOMENTUM, nesterov=NESTEROV)
    network.compile(loss=LOSS_FUNCTION, optimizer=sgd, metrics=['categorical_accuracy'])
    if USE_NEPTUNE:
        neptune_logger=NeptuneLoggerCallback(model=network,
                                         validation_data=(test_data, test_labels))
        network.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=EPOCHS,
                batch_size=BATCH_SIZE, verbose=1, callbacks=[neptune_logger])
    else:
        network.fit(train_data, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    print('Feed forward network trained')
    print('====================================================================')
    return network

'''
===============================================================================

evaluate_network :: Evaluates network to check for accuracy

@params: test_labels :: tensor of one-hot encoded labels for testing accuracy
         test_data   :: tensor of test data to classify
         network     :: trained feed forward model

@returns: loss, accuracy
===============================================================================
'''
def evaluate_network(test_labels, test_data, network):
    (loss, accuracy) = network.evaluate(test_data, test_labels, batch_size=BATCH_SIZE, verbose=1)
    print("loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
    return loss, accuracy

'''
===============================================================================

save_model :: Saves a trained keras model in a pickle file

@params: network - Keras Model :: trained keras model to serialize and save
         model_type            :: int - representing model type
                                  0: CNN, 
                                  1: FFNN

@returns: void
===============================================================================
'''
def save_model(network, model_type):
    if (model_type == 0):
        pickle.dump(network, open('CNN_model.p', 'wb'))
    elif (model_type == 1):
        pickle.dump(network, open('feed_forward_model.p', 'wb'))
    else:
        print("unrecongized model_type")
        print("0: CNN, 1: FFNN")

'''
===============================================================================

run_NN :: Initializes, trains, and evaluates the feed forward network

@params: train_data   :: tensor of data to use for training
         train_labels :: tensor of one-hot encoded lables for training
         test_data    :: tensor of data for testing
         test_labels  :: tensor of labels to check accuracy of test data

@returns: void
===============================================================================
'''
def run_NN(train_data, train_labels, test_data, test_labels):
    network        = initialize_network()
    t_network      = train_network(train_labels, train_data, test_labels, test_data, network)
    loss, accuracy = evaluate_network(test_labels, test_data, t_network)
    save_model(t_network,1)
    acc = accuracy * .01
    interval = 1.96 * math.sqrt( (acc * (1 - acc)) / NUM_TRAINING)
    print("loss", loss)
    print("accuracy", accuracy)
    print("confidence interval:", interval)


if (__name__ == '__main__'):
    paths       = ['project_spect/train/', 'project_timeseries/train/', 'project_mfccs/train/']
    data        = []
    labels_list = []

    if len(sys.argv) < 2:
        print('Argument required:')
        print('1: use spectograms')
        print('2: use timeseries')
        print('3: use MFCCs')
    else:
        input_num = int(sys.argv[1])
        labels_dict        = gen_labels()
        data, labels_list  = read_data(paths[input_num-1], labels_dict)
        run_NN(data[:NUM_TRAINING], labels_list[:NUM_TRAINING], data[NUM_TRAINING:], labels_list[NUM_TRAINING:])
