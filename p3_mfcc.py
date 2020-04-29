# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:03:26 2020

@author: aisli
"""
import librosa
import csv
import matplotlib.pyplot as plt
import librosa.display


''' 
===============================================================================

@params:

@returns: 
===============================================================================
'''
def getMFCC():
    mfccs = librosa.feature.mfcc(k, sr=sr)
    print(mfccs.shape)#Displaying  the MFCCs:
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
  

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
        audio_path_new = path+str(i)+'.wav'
        print(audio_path_new)
        try:
            k , sr = librosa.load(audio_path_new)
        
            #Plot the signal as a spectrogram:
            #spec is a short term Fourier Transform 
            spec = librosa.stft(k)
            spec_db = librosa.amplitude_to_db(abs(spec))
            #plt.subplot(10,1,int(x)+1)
            #plt.figure(figsize=(14, 5))
            librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='hz') 
            #If to pring log of frequencies  
            #librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='log')
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
    audio_path = './data/project_waves_norm/train/'
    files = []
    with open('./data/project3/train.csv') as file:
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
    wav_to_mfcc_overlay(audio_path, train_list[0],'Rock_MFCC_Norm.png')
    '''
    print('Genre: Pop')
    wav_to_mfcc_overlay(audio_path, train_list[1],'Pop_MFCC_Norm.png')
    print('Genre: Folk')
    wav_to_mfcc_overlay(audio_path, train_list[2],'Folk_MFCC_Norm.png')
    print('Genre: Instrumental')
    wav_to_mfcc_overlay(audio_path, train_list[3],'Instrumental_MFCC_Norm.png')
    print('Genre: Electronic')
    wav_to_mfcc_overlay(audio_path, train_list[4],'Electronic_MFCC_Norm.png')
    print('Genre: Hip-Hop')
    wav_to_mfcc_overlay(audio_path, train_list[5],'HipHop_MFCC_Norm.png')
    '''
''' 
===============================================================================

@params:

@returns: 
===============================================================================
'''
if __name__ == '__main__':
    
    initial_test = 1
    train = 0
    classify = 0
    
    
    if initial_test:
        make_graphs()
        
    '''
    for x in range(10):
        audio_path_new = audio_path + str(files[x][0])+'.wav'
        #print(audio_path_new)
        k , sr = librosa.load(audio_path_new)
        #print(type(k), type(sr))
        #Plot the signal:
        plt.figure(figsize=(14, 5))
        plt.subplot(10,1,int(x)+1)
        librosa.display.waveplot(k, sr=sr)
        
        #Plot the signal as a spectrogram:
        #spec is a short term Fourier Transform 
        spec = librosa.stft(k)
        spec_db = librosa.amplitude_to_db(abs(spec))
        specs.append(spec_db)
        #plt.subplot(10,1,int(x)+1)
        #plt.figure(figsize=(14, 5))
        librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='hz') 
        #If to pring log of frequencies  
        #librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()
    '''
    print('======================================')

