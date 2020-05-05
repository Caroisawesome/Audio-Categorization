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
import librosa.display

IMG_DPI=75

'''
===============================================================================
make_dirs: Creates directories based on a given path
@params: path - string of the path to build directories for

@returns: void
===============================================================================
'''
def make_dirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        # directory already exists
        pass

'''
===============================================================================
convert_mp3_to_wav: Converts an mp3 file to a wave file
@params: src - string representing the path and name of the mp3 file as input
         dst - string representing the path and name of the wave file to output

@returns: void
===============================================================================
'''
def convert_mp3_to_wav(src, dst):
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")

'''
===============================================================================
wav_to_spect: Convert a wave file to a spectrogram
@params: path - string representing path to wav file
         out - path and filename of resulting spectrogram image

@returns: void
===============================================================================
'''
def wav_to_spect(path, out):

    samplerate, data = scipy.io.wavfile.read(path)
    freq, times, spectrogram = signal.spectrogram(data, samplerate)

    plt.ylim(0, 5000)
    plt.pcolormesh(times, freq, spectrogram, vmin=0,vmax=0.06)
    #plt.colorbar()
    plt.axis('off')
    plt.savefig(out,bbox_inches='tight',pad_inches=0, dpi=IMG_DPI)
    plt.clf()

'''
===============================================================================
wav_to_spect_overlay: Generate spectrogram of all songs in a genre
@params: path - string representing path to the wav files
         IDList -
         out - string representing path and name of saved image

@returns: void
===============================================================================
'''
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

'''
===============================================================================
generate_wav_from_mp3 - Converts mp3 files to wav files
@params: in_path - string - path to the directory containing mp3s
         out_path - string - path to the directory that will contain the wav files

@returns: void
===============================================================================
'''
def generate_wav_from_mp3(in_path,out_path):
   for root, dirs, files in os.walk(in_path, topdown=False):
       for name in files:
           path = os.path.join(root, name)
           out  = os.path.join(out_path, name.split(".")[0]+".wav")
           convert_mp3_to_wav(path, out)

'''
===============================================================================
generate_spects_from_wav - Generate spectograms from wav files
@params: in_path2 - string - path to directory containing wav files
         out_path2 - string - path to directory that will contain the spectograms

@returns: void
===============================================================================
'''
def generate_spects_from_wav(in_path2,out_path2):
    for root, dirs, files in os.walk(in_path2, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            out  = os.path.join(out_path2, name.split(".")[0]+".png")
            wav_to_spect(path, out)

'''
===============================================================================
generate_wav_to_spect_overlay_genres
@params: void

@returns: void
===============================================================================
'''
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

'''
===============================================================================
generate_time_series_png - Generates timeseries image from wav file
@params: in_path - string - path and name of wav file
         out_path - string - path and name of resulting timeseries image

@returns: void
===============================================================================
'''
def generate_time_series_png(in_path, out_path):
    for root, dirs, files in os.walk(in_path, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            out  = os.path.join(out_path, name.split(".")[0]+".png")
            samplerate, data = scipy.io.wavfile.read(path)
            librosa.display.waveplot(data, sr=samplerate)
            plt.axis('off')
            plt.savefig(out,bbox_inches='tight',pad_inches = 0, dpi=IMG_DPI)
            plt.clf()

'''
===============================================================================
generate_mfcc_png
@params:

@returns:
===============================================================================
'''
def generate_mfcc_png():
    print("NOT IMPLEMENTED")
    return 0

if (__name__ == '__main__'):

    path_train_mp3 = 'data/project3/train/'
    path_train_wav_norm = 'data/project_waves_norm/train/'
    path_train_wav ='data/project_waves/train/'
    path_train_spect ='data/project_spect/train/'
    out_path_timeseries = 'data/project_timeseries/train/'

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

        if input == 1:
            make_dirs(path_train_wav)
            generate_wav_from_mp3(path_train_mp3,path_train_wav)
        elif input == 2:
            normalization.normalize_wav_data()
        elif input == 3:
            make_dirs(path_train_spect)
            generate_spects_from_wav(path_train_wav_norm, path_train_spect)
        elif input == 4:
            make_dirs(out_path_timeseries)
            generate_time_series_png(path_train_wav_norm, out_path_timeseries)
        elif input == 5:
            generate_mfcc_png()
        elif input == 6:
            generate_wav_to_spect_overlay_genres()
        else:
            print("could not understand input. try again.")








