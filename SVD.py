import scipy.io.wavfile
import math
import numpy as np


if (__name__ == '__main__'):
    samplerate, data = scipy.io.wavfile.read('test.wav')
    (song_length,num_channels) = data.shape

    num_columns = 1024
    num_rows = math.floor(song_length/num_columns)
    num_entries = num_rows*num_columns


    A = np.array( data)
    A = A[:num_entries,0]
    A = np.reshape(A, (num_rows, num_columns))
    u, s, vh = np.linalg.svd(A, full_matrices=True)

    print(s)


    # doto: save all wave files as a spectrogram?
