import numpy as np
import scipy.io.wavfile
from scipy import signal
from os import path
from pydub import AudioSegment
import matplotlib.pyplot as plt
import os
import csv
import sys
import normalization

def make_dirs():
    try:
        os.makedirs("data/project_waves")
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs("data/project_waves/train")
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs("data/project_spect/train")
    except FileExistsError:
        # directory already exists
        pass


def convert_mp3_to_wav(src, dst):
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")


def wav_to_spect(path, out):

    samplerate, data = scipy.io.wavfile.read(path)
    freq, times, spectrogram = signal.spectrogram(data, samplerate)

    #plt.figure()
    plt.ylim(0, 5000)
    plt.pcolormesh(times, freq, spectrogram, vmin=0,vmax=0.06)
    #plt.colorbar()
    plt.axis('off')
    plt.savefig(out)
    plt.clf()

def wav_to_spect_overlay(path, IDList, out):
    plt.figure(figsize=(14, 5))
    plt.ylim(0, 8000)

    for i in IDList:
        samplerate, data = scipy.io.wavfile.read(path+str(i)+'.wav')
        #print(data.shape)
        if len(data.shape) > 1:
            freq, times, spectrogram = signal.spectrogram(data[:,1], samplerate)
        else:
            print('missing data dimensions at file #',i)
        plt.pcolormesh(times, freq, spectrogram)
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    plt.colorbar()
    plt.savefig(out)
    #plt.show()


def generate_wav_from_mp3(in_path,out_path):

   for root, dirs, files in os.walk(in_path, topdown=False):
       for name in files:
           path = os.path.join(root, name)
           out  = os.path.join(out_path, name.split(".")[0]+".wav")

           convert_mp3_to_wav(path, out)

def generate_spects_from_wav(in_path2,out_path2):

    for root, dirs, files in os.walk(in_path2, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            out  = os.path.join(out_path2, name.split(".")[0]+".png")

            wav_to_spect(path, out)

def generate_wav_to_spect_overlay_genres():

    audio_path = './data/project_waves/train/'
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
    '''
    print('Genre: Rock')
    wav_to_spect_overlay(audio_path, train_list[0],'Rock.png')
    print('Genre: Pop')
    wav_to_spect_overlay(audio_path, train_list[1],'Pop.png')
    print('Genre: Folk')
    wav_to_spect_overlay(audio_path, train_list[2],'Folk.png')
    '''
    print('Genre: Instrumental')
    wav_to_spect_overlay(audio_path, train_list[3],'Instrumental.png')
    print('Genre: Electronic')
    wav_to_spect_overlay(audio_path, train_list[4],'Electronic.png')
    print('Genre: Hip-Hop')
    wav_to_spect_overlay(audio_path, train_list[5],'HipHop.png')
        # for name in dirs:
        #   print(os.path.join(root, name))


def generate_time_series_png():
    print("NOT IMPLEMENTED")
    return 0

def generate_mfcc_png():
    print("NOT IMPLEMENTED")
    return 0

if (__name__ == '__main__'):

    in_path = 'data/project3/train/'
    in_path2 = 'data/project_waves_norm/train/'
    out_path ='data/project_waves/train/'
    out_path2 ='data/project_spect/train/'

    if len(sys.argv) < 2:
        print("Argument required:")
        print("1: generate wav from mp3")
        print("2: normalize data")
        print("3: generate spectogram charts")
        print("4: generate time series charts")
        print("5: generate mfcc charts")
        print("6: build genre spectogram overlays")
    else:
        input = int(sys.argv[1])
        make_dirs()

        if input == 1:
            generate_wav_from_mp3(in_path,out_path)
        elif input == 2:
            normalization.normalize_data()
        elif input == 3:
            generate_spects_from_wav(in_path2,out_path2)
        elif input == 4:
            generate_time_series_png()
        elif input == 5:
            generate_mfcc_png()
        elif input == 6:
            generate_wav_to_spect_overlay_genres()
        else:
            print("could not understand input. try again.")








