from scipy.io import wavfile
from scipy.fftpack import fft,fftfreq

if (__name__ == '__main__'):
    samplerate, data = wavfile.read("data/project_waves_norm/train/00907479")

    num_samples = data.shape[0]

    datafft = fft(data)
    #Get the absolute value of real and complex component:
    fftabs = abs(datafft)

    freqs = fftfreq(num_samples,1/samplerate)

    print(freqs)
    print(fftabs)

    print(freqs.shape)
    print(fftabs.shape)
