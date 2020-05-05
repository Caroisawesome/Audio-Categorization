# Audio Categorization

This project uses Convolutional Neural Networks and Feed Forward Neural Networks to classify the genre of a song. The networks use 2401 classified songs for training. The songs are then converted into spectrograms, timeseries images, and MFCC images which then become inputs for the networks. 

## Data Processing

The mp3 files used for this project are stored in a directory `data/project3/train` that is not tracked by github. The process for preparing the data for the NNs is as follows:

1. Convert mp3 to wav files
2. Normalize wav files
3. Convert wav to image (Spectrogram, Timeseries, or MFCC)

#### Convert mp3 to wav files

Run `python3 util.py 1`

#### Normalize wav files

Run `python3 util.py 2`




## Convolutional Neural Network

## Feed Forward Neural Network