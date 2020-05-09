# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:22:08 2020

@author: alysh
"""
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages



from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.regularizers import l2
# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import numpy as np
import pickle
import os
import csv


DIM_ROWS = 234 # max 277
DIM_COLS = 200 # max 372
DIM_CHANNELS = 3 # max 3

EPOCHS = 30
BATCH_SIZE = 150
LEARNING_RATE = 0.001
MOMENTUM = 0.9
DECAY = 1e-6
NESTEROV = True
LOSS_FUNCTION = 'categorical_crossentropy'

# Number of training data (remainder of data will go to testing)
NUM_TRAINING = 1840
LEAKY_RELU_ALPHA = 0.01

class StridedNet:
	@staticmethod
	def build(width, height, depth, classes, reg, init="he_normal"):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
        # our first CONV layer will learn a total of 16 filters, each
		# Of which are 7x7 -- we'll then apply 2x2 strides to reduce
		# the spatial dimensions of the volume
		model.add(Conv2D(16, (7, 7), strides=(2, 2), padding="valid",
			kernel_initializer=init, kernel_regularizer=reg,
			input_shape=inputShape))

		# here we stack two CONV layers on top of each other where
		# each layerswill learn a total of 32 (3x3) filters
		model.add(Conv2D(32, (3, 3), padding="same",
			kernel_initializer=init, kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
			kernel_initializer=init, kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.25))
        # stack two more CONV layers, keeping the size of each filter
		# as 3x3 but increasing to 64 total learned filters
		model.add(Conv2D(64, (3, 3), padding="same",
			kernel_initializer=init, kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same",
			kernel_initializer=init, kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.25))

		# increase the number of filters again, this time to 128
		model.add(Conv2D(128, (3, 3), padding="same",
			kernel_initializer=init, kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same",
			kernel_initializer=init, kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.25))
        
        # fully-connected layer
		model.add(Flatten())
		model.add(Dense(512, kernel_initializer=init))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
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
    (trainX, testX, trainY, testY) = train_test_split(data, labels_l,
    	test_size=0.20, stratify=labels_l, random_state=42)
    
    
    # construct the training image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
    	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    	horizontal_flip=True, fill_mode="nearest")

    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = Adam(lr=LEAKY_RELU_ALPHA, decay=DECAY / EPOCHS)
    model = StridedNet.build(width=DIM_COLS, height=DIM_ROWS, depth=DIM_CHANNELS,
    	classes=len(labels_l), reg=l2(0.0005))
    
    
    model.compile(loss=LOSS_FUNCTION, optimizer=opt,
    	metrics=["accuracy"])
    
    # train the network
    print("[INFO] training network for {} epochs...".format(EPOCHS))
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BATCH_SIZE,
    	epochs=EPOCHS)
    
    
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=BATCH_SIZE)
    print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=labels_l))
    
    save_model(H, 0)
    
    # plot the training loss and accuracy
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plots.png")

    