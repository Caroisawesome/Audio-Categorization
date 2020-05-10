# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:22:08 2020

@author: alysh
"""
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages

from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
#from keras.optimizers import Adam
from keras.regularizers import l2
#from keras.callbacks import Callback
# import the necessary packages
#from scikitplot.metrics import plot_confusion_matrix
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import numpy as np
import pickle
import os
import csv

#DIM_ROWS = 234 # max 277
DIM_ROWS = 50 # max 277
#DIM_COLS = 200 # max 372
DIM_COLS = 50 # max 372
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


gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
session     = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))        
        
class StridedNet:
    @staticmethod
    def build(width, height, depth, classes, reg, init="glorot_uniform"):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        print('====================================================================')
        print('Starting CNN')
        print('====================================================================')
		# if we are using "channels first", update the input shape
		# and channels dimension
       
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
           
        # our first CONV layer will learn a total of 16 filters, each
		# Of which are 7x7 -- we'll then apply 2x2 strides to reduce
		# the spatial dimensions of the volume
        model.add(Conv2D(4, (2, 2), strides=(1, 1), padding="same",
        kernel_initializer=init, kernel_regularizer=reg,
        input_shape=inputShape))
        
        print('====================================================================')
        print('Starting CNN_32')
        print('====================================================================')
		# here we stack two CONV layers on top of each other where
		# each layerswill learn a total of 32 (3x3) filters
        model.add(Conv2D(8, (2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(8, (2, 2), strides=(1, 1), padding="same",	
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu")) 
        model.add(BatchNormalization(axis=chanDim)) 
        model.add(Dropout(0.25))
        model.add(Flatten())
        '''
        print('====================================================================')
        print('Starting CNN_64')
        print('====================================================================')
        # stack two more CONV layers, keeping the size of each filter
        # as 3x3 but increasing to 64 total learned filters
        model.add(Conv2D(64, (3, 3), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same",
			kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))
        print('====================================================================')
        print('Starting CNN_128')
        print('====================================================================')
        # increase the number of filters again, this time to 128
        model.add(Conv2D(128, (3, 3), padding="same",
			kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), strides=(1, 1), padding="same",
			kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))
        print('====================================================================')
        print('Starting CNN_512')
        print('====================================================================')
        # fully-connected layer
        model.add(Flatten())
        model.add(Dense(512, kernel_initializer=init))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        '''
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        model.summary()
        return model

     
def gen_labels():
    labels   = {}
    #genres   = [0, 1, 2, 3, 4, 5]

    with open('data/project3/train.csv', 'r') as csvfile:
        #reader = csv.reader(csvfile)
        tmp = list(list(rec) for rec in csv.reader(csvfile, delimiter=','))
        for row in tmp:
            if row[0] == 'new_id':
                continue
            labels[row[0]] = row[1]
    #print(labels)
    return labels
    #for g in genres:
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
    data     = np.zeros((len(t_files),DIM_ROWS,DIM_COLS,DIM_CHANNELS))
    for i in range(0, len(t_files)):
        image = mpimg.imread(t_files[i])
        np_image = np.array(image)[:DIM_ROWS, :DIM_COLS, :DIM_CHANNELS]
        data[i] = np_image#.flatten() #image_to_feature_vector(image)
        labels_l.append(labels[names[i]])
        #print(data[i]) 

    labels_l = to_categorical(labels_l)
    #print('Writing pickle data')
    #write_pickle_file(data, directory, 'data')
    #print('Writing pickle labels')
    #write_pickle_file(labels_l, directory, 'labels')
    return data, labels_l
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

train_network :: Using stochastic gradient descent.

@params:

@returns: 
===============================================================================
'''
def train_network(train_labels, train_data, test_labels, test_data, network):
    print('Training con NN network')
    print('====================================================================')
    print('train labels: ' + str(train_labels.shape))
    print('train data: ' + str(train_data.shape))
    print('test labels: ' + str(test_labels.shape))
    print('test data: ' + str(test_data.shape))
    
    sgd = SGD(lr=LEARNING_RATE, decay=DECAY, momentum=MOMENTUM, nesterov=NESTEROV)
    network.compile(loss=LOSS_FUNCTION, optimizer=sgd, metrics=['categorical_accuracy'])
    
    network.fit(train_data, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    print('Con NN trained')
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
    (loss, accuracy) = network.evaluate(test_data, test_labels, batch_size=BATCH_SIZE, verbose=1)
    print("loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
    return loss, accuracy
'''
===============================================================================
Start of our program
===============================================================================
'''
if __name__ == '__main__':
    
    testing_mfccs = True
    testing_spect = False
    testing_timeseries = False
    
    dir_mfcc = "project_mfccs/train"
    dir_spec = "project_spect/train"
    dir_tmsr = "project_timeseries/train"
    dir_tmsr_1 = "project_time_series/train"
    
    '''
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
    	help="path to input dataset")
    ap.add_argument("-e", "--epochs", type=int, default=50,
    	help="# of epochs to train our network for")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
    	help="path to output loss/accuracy plot")
    args = vars(ap.parse_args())
    '''
    
    # initialize the set of labels from the CALTECH-101 dataset we are
    # going to train our network on
    LABELS = range(41)
    print(LABELS)
    
    # grab the list of images in our dataset directory, then initialize
    # the list of data (i.e., images) and class images
    print("[INFO] loading images...")
    
    if testing_mfccs:
        imagePaths = dir_mfcc
    elif testing_spect:
        imagePaths = dir_spec
    elif testing_timeseries:
        imagePaths = dir_tmsr
        


    labels = gen_labels()
    data,labels_l = read_data(imagePaths,labels)
    print(labels)
    
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (train_data, test_data, train_labels, test_labels) = train_test_split(data, labels_l,
    	test_size=0.20, stratify=labels_l, random_state=42)
    
    
    # construct the training image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
    	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    	horizontal_flip=True, fill_mode="nearest")

    # initialize the optimizer and model
    print("[INFO] compiling model...")
    
    model = StridedNet.build(width=DIM_COLS, height=DIM_ROWS, depth=DIM_CHANNELS,
    	classes=len(labels_l), reg=l2(0.0005))
    
    '''
    opt = Adam(lr=LEAKY_RELU_ALPHA, decay=DECAY / EPOCHS)
    model.compile(loss=LOSS_FUNCTION, optimizer=opt,metrics=['categorical_accuracy'])
    
    # train the network
    print("[INFO] training network for {} epochs...".format(EPOCHS))
    t_network = model.fit_generator(aug.flow(train_data, train_labels, batch_size=BATCH_SIZE),
    	validation_data=(test_data, test_labels), steps_per_epoch=len(train_data),
    	epochs=EPOCHS)
    
    
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(test_data, batch_size=BATCH_SIZE)
    print(classification_report(test_labels.argmax(axis=1),
    predictions.argmax(axis=1), target_names=labels_l))
    '''
    
    t_network      = train_network(train_labels, train_data, test_labels, test_data, model)
    loss, accuracy = evaluate_network(test_labels, test_data, t_network)
    
    save_model(t_network, 0)
    
    # plot the training loss and accuracy
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), t_network.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), t_network.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), t_network.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), t_network.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plots.png")

    
