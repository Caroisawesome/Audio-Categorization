# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:22:08 2020

@author: alysh
"""
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from p3_audiocategorization.stridednet import StridedNet
from p3_audiocategorization.feed_forward import FF_NN

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.regularizers import l2

import matplotlib.pyplot as plt
import numpy as np




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
        


    labels = FF_NN.gen_labels()
    data,labels_l = FF_NN.read_data(imagePaths,labels)
    
    
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
    model = StridedNet.build(width=96, height=96, depth=3,
    	classes=len(labels_l), reg=l2(0.0005))
    
    
    model.compile(loss=LOSS_FUNCTION, optimizer=opt,
    	metrics=["accuracy"])
    
    # train the network
    print("[INFO] training network for {} epochs...".format(EPOCHS))
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
    	validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32,
    	epochs=EPOCHS)
    
    
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=labels_l))
    
    FF_NN.save_model(H, 0)
    
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

    