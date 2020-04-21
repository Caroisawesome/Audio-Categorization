import scipy.io.wavfile
from scipy import signal
import matplotlib.pyplot as plt



def wav_to_spect(filename):

    samplerate, data = scipy.io.wavfile.read(filename)
    num_channels = data.shape[1]

    freq, times, spectrogram = signal.spectrogram(data[:,0], samplerate)

    plt.pcolormesh(times, freq, spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


if (__name__ == '__main__'):

    wav_to_spect('test.wav')


    # doto: save all wave files as a spectrogram?
