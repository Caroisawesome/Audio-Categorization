from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers import Dense
from keras.layers import Activation
import tensorflow as tf
from PIL import Image
#from imutils import paths
#import imutils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import argparse
import pickle
import os
import sys
import csv

#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
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

Borrows from:
https://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras/

@params:

@returns: 
===============================================================================
'''
def image_to_feature_vector(image, size=(496, 369)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return np.resize(image, size).flatten()

''' 
===============================================================================

@params:

@returns: 
===============================================================================
'''
def write_pickle_file(data, directory, type_r):
    filename = directory.split('/')[0] + '_' + type_r + '_pickle'
    pickle.dump(data, open(filename, 'wb'))

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
    print('Walking data directory')
    print('====================================================================')
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            t_files.append(os.path.join(root, name))
            base = os.path.splitext(name)[0]
            names.append(base)
            #print(os.path.join(root, name))
            #print(base)

    print('Reading images')
    print('====================================================================')
    for i in range(0, len(t_files)):
        image = mpimg.imread(t_files[i])
        #features = np.asarray(Image.open(t_files[i]))
        features = np.array(image).flatten() #image_to_feature_vector(image)
        data.append(features)
        labels_l.append(labels[names[i]])
        #print(data[i])

    print('Writing pickle data')
    write_pickle_file(data, directory, 'data')
    print('Writing pickle labels')
    write_pickle_file(labels_l, directory, 'labels')
    return data, labels_l

''' 
===============================================================================

initialize_network :: 

* input_dim = 549072 = 496 X 369 X 3 RGB channels

@params:

@returns: 
===============================================================================
'''
def initialize_network():
    print('Initializing feed forward network')
    print('====================================================================')
    network = models.Sequential()
    network.add(Dense(10000, input_dim=549072, init='uniform', activation='relu'))
    network.add(Dense(1000, init='uniform', activation='relu'))
    network.add(Dense(100,  activation='relu', kernel_initializer='uniform'))
    network.add(Dense(10,  activation='relu', kernel_initializer='uniform'))
    network.add(Dense(6, activation='softmax'))
    print('Feed forward network initialized')
    print('====================================================================')
    #network.add(Activation('softmax'))
    return network

''' 
===============================================================================

train_network :: Using stochastic gradient descent.

@params:

@returns: 
===============================================================================
'''
def train_network(train_labels, train_data, network):
    print('Training feed forward network')
    print('====================================================================')
    sgd = SGD(lr=0.01)
    network.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    network.fit(train_data, train_labels, epochs=50, batch_size=100, verbose=1)
    print('Feed forward network trained')
    print('====================================================================')
    return network

''' 
===============================================================================

evaluate_network ::

@params:

@returns: 
===============================================================================
'''
def evaluate_network(test_labels, test_data, network):
    (loss, accuracy) = network.evaluate(test_data, test_labels, batch_size=100, verbose=1)
    print("loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
    return loss, accuracy

def get_picklefiles():
    pickles = []
    for root, dirs, files in os.walk('.', topdown=False):
        for name in files:
            base = os.path.splitext(name)[0]
            if 'pickle' in base:
                pickles.append(name)
                print(base)
    return pickles

if (__name__ == '__main__'):
    paths       = ['project_spect/train/', 'project_timeseries/train/']
    pickles     = get_picklefiles()
    labels_dict = []
    data        = []
    labels_list = []

    if len(sys.argv) < 2:
        print('Argument required:')
        print('1: use spectograms')
        print('2: use timeseries')
    else:
        input = int(sys.argv[1])
        if input == 1:
            if 'project_spect_data_pickle' not in pickles and 'project_spect_labels_pickle' not in pickles:
                print('Running read_data')
                labels_dict        = gen_labels()
                data, labels_list  = read_data(paths[0], labels_dict)
            else:
                print('Reading pickle files')
                data        = pickle.load(open('project_spect_data_pickle', 'rb'))
                labels_list = pickle.load(open('project_spect_labels_pickle', 'rb'))
            #test_labels, test_data = test_labels_and_data()
            #network                = initialize_network()
            #t_network              = train_network(labels_l, data, network)
            #loss, accuracy         = evaluate_network(test_labels, test_data, t_network)
        if input == 2:
            if 'project_timeseries_data_pickle' not in pickles and 'project_timeseries_labels_pickle' not in pickles:
                print('Running read_data')
                labels_dict        = gen_labels()
                data, labels_list  = read_data(paths[1], labels_dict)
            else:
                print('Reading pickle files')
                data        = pickle.load(open('project_timeseries_data_pickle', 'rb'))
                labels_list = pickle.load(open('project_timeseries_labels_pickle', 'rb'))
            #test_labels, test_data = test_labels_and_data()
            #network                = initialize_network()
            #t_network              = train_network(labels_l, data, network)
            #loss, accuracy         = evaluate_network(test_labels, test_data, t_network)




