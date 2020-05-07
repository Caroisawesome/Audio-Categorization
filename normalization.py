import os
import math
import csv
import scipy.io.wavfile
import numpy as np

'''
===============================================================================
compute_mean - computes the mean value of all the timeseries data in directory
@params: path - string - path/directory containing all the wav files

@returns: float - the mean timeseries value across all data
===============================================================================
'''
def compute_mean(path):
    count = 0
    sums = 0

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            samplerate, data = scipy.io.wavfile.read(path)
            if len(data.shape) > 1:
                count+=len(data[:,0])
                sums+= np.sum(data[:,0])

    return sums / count

'''
===============================================================================
compute_st - computes the standard deviation of all timeseries data in directory
@params: path - string - path/directory containing wav files
         avg - float - the mean over the same set of data

@returns: float - standard deviation of timeseries data
===============================================================================
'''
def compute_sd(path, avg):

    count = 0
    sums = 0

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            samplerate, data = scipy.io.wavfile.read(path)

            if len(data.shape) > 1:
                count += len(data[:,0])
                sums += np.sum(np.square(data[:,0] - avg))

    variance = sums/(count-1)
    return np.sqrt(variance)

'''
===============================================================================
normalize_data - Normalizes the timeseries data in a directory according to
                 mean and standard deviation
@params: path - string - path/directory continaing wav files
         out_path - string - path/directory that will contain normalized wavs
         avg - float - mean across all the wav files in directory
         std - float - standard deviation across all wav files in directory

@returns: void
===============================================================================
'''
def normalize_data(path, out_path, avg, std):

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            samplerate, data = scipy.io.wavfile.read(path)

            if len(data.shape)>1:
                data_norm = (data[:,0] - avg) / std
                scipy.io.wavfile.write(out_path + name.split(".")[0]+".wav", samplerate, data_norm)

'''
===============================================================================
normalize_wav_data - Normalizes timeseries data in a directory by first
         computing the mean and standard deviation
@params: waves_path - string - path/directory to wav files
         output_path - string - path/directory that will contain normalized wavs

@returns: void
===============================================================================
'''
def normalize_wav_data(waves_path, output_path):

    print("computing mean...")
    avg = compute_mean(waves_path)
    print(avg)

    print("computing standard deviation...")
    sd = compute_sd(waves_path,avg)
    print(sd)

    print("normalizing data...")
    normalize_data(waves_path, output_path, avg, sd)

    print("Done. Normalized data has been written to "+output_path)

if (__name__ == '__main__'):

    waves_path = 'data/project_waves/train/'
    output_path = 'data/project_waves_norm/train/'

    normalize_wav_data()
