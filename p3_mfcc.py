# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:03:26 2020

@author: aisli
"""
import librosa
import csv
import matplotlib.pyplot as plt
import librosa.display
import os
import os.path
import numpy as np


'''
===============================================================================

@params:

@returns:
===============================================================================
'''
def getMFCCs(in_path, out, type_of_data):
    if type_of_data == 0:
        csv_path = 'data/project3/train.csv'
    if type_of_data == 1:
        csv_path = 'data/project3/test_idx.csv'
    with open(csv_path) as file:
        data = csv.reader(file)
        files = list(data)
    files.remove(files[0])
    for i in files:
        #path = './data/project_waves_norm/train/'+str(i[0])+'.wav'
        path = in_path +str(i[0])+'.wav'
        print(path)
        try:
            plt.ylim(0, 5000)
            k,sr = librosa.load(path)
            mfccs = librosa.feature.mfcc(k, sr=sr)
            print(mfccs.shape)
            plt.figure(figsize=(10,4))
            xx = np.arange(0,mfccs.data.shape[1]+1,1)
            yy = np.arange(0,mfccs.data.shape[0]+1,1)
            librosa.display.specshow(mfccs,x_coords=xx, y_coords=yy)
            out_path = out+str(i[0])+'.png'
            plt.savefig(out_path,bbox_inches='tight',pad_inches=0, dpi=75 )
            plt.close()
        except:
            print(i)
            pass


'''
===============================================================================

@params:

@returns:
===============================================================================
'''
def wav_to_mfcc_overlay(path, IDList, out):
    plt.figure(figsize=(14, 5))
    plt.ylim(0, 8000)

    for i in IDList:
        #.\data\project_waves_norm\train
        #audio_path_new = "./data/project_waves_norm/train/"+str(i)+".wav"
        audio_path_new = path+str(i)+'.wav'
        #print(audio_path_new)
        try:
            k , sr = librosa.load(audio_path_new)
            #Plot the signal as a spectrogram:
            mfccs = librosa.feature.mfcc(k, sr=sr)
            librosa.display.specshow(mfccs,x_axis='time',y_axis='mel')
        except:
            print(i)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    #plt.show()
    plt.savefig(out)
'''
===============================================================================

@params:

@returns:
===============================================================================
'''
def make_graphs():
    #data/project_waves_norm/train
    audio_path = 'data/project_waves/train/'
    files = []
    with open('data/project3/train.csv') as file:
        data = csv.reader(file)
        files = list(data)
    files.remove(files[0])
    train_list = [[]for i in range(6)]

    for x in files:
        #print(x)
        train_list[int(x[1])].append(x[0])
    '''
    Rock:0
    Pop:1
    Folk:2
    Instrumental:3
    Electronic:4
    Hip-Hop:5
    '''

    print('Genre: Rock')
    wav_to_mfcc_overlay(audio_path, train_list[0],'Rock_MFCC.png')
    print('Genre: Pop')
    wav_to_mfcc_overlay(audio_path, train_list[1],'Pop_MFCC.png')
    print('Genre: Folk')
    wav_to_mfcc_overlay(audio_path, train_list[2],'Folk_MFCC.png')
    print('Genre: Instrumental')
    wav_to_mfcc_overlay(audio_path, train_list[3],'Instrumental_MFCC.png')
    print('Genre: Electronic')
    wav_to_mfcc_overlay(audio_path, train_list[4],'Electronic_MFCC.png')
    print('Genre: Hip-Hop')
    wav_to_mfcc_overlay(audio_path, train_list[5],'HipHop_MFCC.png')

'''
===============================================================================

@params:

@returns:
===============================================================================
'''
if __name__ == '__main__':

    initial_test = 0
    gen_all_graphs = 1
    train = 0
    classify = 0


    if initial_test:
        make_graphs()

    if gen_all_graphs:
        try:
            os.makedirs("data/project_mfccs/")
        except FileExistsError:
            # directory already exists
            pass
        out = "data/project_mfccs/"
        in_path = './data/project_waves_norm/train/'
        getMFCCs(in_path,out, 0)
