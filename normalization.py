import os
import math
import csv
import scipy.io.wavfile
import numpy as np

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

def normalize_data(path, out_path, avg, std):
    make_dirs()

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            samplerate, data = scipy.io.wavfile.read(path)

            if len(data.shape)>1:
                data_norm = (data[:,0] - avg) / std
                scipy.io.wavfile.write(out_path + name.split(".")[0]+".wav", samplerate, data_norm)

def make_dirs():
    try:
        os.makedirs("data/project_waves_norm")
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs("data/project_waves_norm/train")
    except FileExistsError:
        # directory already exists
        pass


if (__name__ == '__main__'):

    waves_path = 'data/project_waves/train/'
    output_path = 'data/project_waves_norm/train/'

    print("computing mean...")
    avg = compute_mean(waves_path)
    print(avg)

    print("computing standard deviation...")
    sd = compute_sd(waves_path,avg)
    print(sd)

    print("normalizing data...")
    normalize_data(waves_path, output_path, avg, sd)

    print("Done. Normalized data has been written to "+output_path)
