from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers import Dense
from keras.layers import Activation
from PIL import Image
#from imutils import paths
#import imutils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import librosa
import librosa.feature
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import argparse
import os
import csv

"""

    Various functions and implementations borrowed or inspired from:

    https://medium.com/@navdeepsingh_2336/identifying-the-genre-of-a-song-with-neural-networks-851db89c42f0
    
    and

    https://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras/

"""


''' 
===============================================================================

@params:

@returns: 
===============================================================================
'''
#def convert_to_mfcc(audio):
#    f, _ = librosa.load(audio)
#    mfcc = librosa.feature.mfcc(f)
#    mfcc /= np.amax(np.absolute(mfcc))
#    return np.ndarray.flatten(mfcc)[:25000]

''' 
===============================================================================

@params:

@returns: 
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
    #print(labels)
    return labels

    #for g in genres:

''' 
===============================================================================

read_data :: Read image data and store as numpy array

TODO :: finish code to read as image data and convert to numpy array.

@params:

@returns: 
===============================================================================
'''
def read_data(directory, labels):
    t_files  = []
    names    = []
    data     = []
    labels_l = []
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            t_files.append(os.path.join(root, name))
            base = os.path.splitext(name)[0]
            names.append(base)
            #print(os.path.join(root, name))
            print(name)

    for i in range(0, len(t_files)):
        #image = mpimg.imread(t_files[i])
        features = np.asarray(Image.open(t_files[i]))
        #features = image_to_feature_vector(image)
        data.append(features)
        labels_l.append(labels[names[i]])
        print(data[i])

    return data

''' 
===============================================================================

@params:

@returns: 
===============================================================================
'''
def initialize_network():
    network = models.Sequential()
    network.add(Dense(128, input_dim=1024, init='uniform', activation='relu'))
    network.add(Dense(64, input_dim=128, activation='relu', kernel_initializer='uniform'))
    network.add(Dense(2))
    network.add(Activation('softmax'))
    return network

''' 
===============================================================================

train_network :: Using stochastic gradient descent.

@params:

@returns: 
===============================================================================
'''
def train_network(labels, data, network):
    sgd = SGD(lr=0.01)
    network.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    network.fit(data, labels, epochs=50, batch_size=128, verbose=1)
    return network

if (__name__ == '__main__'):
    path    = 'project_timeseries/train/'
    labels  = gen_labels()
    data    = read_data(path, labels)
    #network = initialize_network()
    #gen_data_and_labels()

