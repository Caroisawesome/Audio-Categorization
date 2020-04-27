import os
import math
import csv
import scipy.io.wavfile
import numpy as np

def compute_mean(path):
    count = 0
    sums = np.zeros(1400)

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            samplerate, data = scipy.io.wavfile.read(path)

            if len(data.shape) > 1:
                song_length=1400
                for i in range(0, song_length):
                    sums[i] += data[i,0]

                count+=1


    return sums / count


def compute_sd(path, avg):

    count = 0
    sums = np.zeros(1400)

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            samplerate, data = scipy.io.wavfile.read(path)

            if len(data.shape) > 1:
                song_length=1400
                for i in range(0, song_length):
                    sums[i] += math.pow((data[i,0] - avg[i]),2)
                count+=1

    variance = sums/(count-1)
    return np.sqrt(variance)

def normalize_data(path, avg, std):
    with open('data/normalized_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                path = os.path.join(root, name)
                samplerate, data = scipy.io.wavfile.read(path)

                if len(data.shape)>1:
                    data_norm = (data[:1400,0] - avg) / std
                    data_norm = np.concatenate(([name.split(".")[0]], data_norm))
                    writer.writerow(data_norm)

if (__name__ == '__main__'):

    waves_path = 'data/project_waves/train/'

    print("computing mean...")
    avg = compute_mean(waves_path)

    print("computing standard deviation...")
    sd = compute_sd(waves_path,avg)

    print("normalizing data...")
    normalize_data(waves_path, avg,sd)

    print("Done. Normalized data has been written to data/normalized_data.csv")
