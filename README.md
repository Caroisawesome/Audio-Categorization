# Audio Categorization

This project uses Convolutional Neural Networks and Feed Forward Neural Networks to classify the genre of a song. The networks use 2401 classified songs for training. The songs are then converted into spectrograms, timeseries images, and MFCC images which then become inputs for the networks. 

## Data Processing

The mp3 files used for this project are stored in a directory `data/project3/train` that is not tracked by github. Data that has already been pre-processed and converted into images are stored in the `project_spect` and `project_timeseries` directories. The process for preparing the data for the NNs is as follows:

1. Convert mp3 to wav files
2. Normalize wav files
3. Convert wav to image (Spectrogram, Timeseries, or MFCC)

Once the data has been processed and converted into images, the images can be moved to `project_spect` and `project_timeseries` in order to be read in by the neural networks.

#### Convert mp3 to wav files

Run `python3 util.py 1`.

This command will read mp3 files from `data/project3/train` and will write wav files to `data/project_waves/train`.

#### Normalize wav files

Run `python3 util.py 2`.

This command will read wav files from `data/project_waves/train` and will write normalized wav files to `data/project_waves_norm/train`.

#### Convert wav to Spectrogram Image

Run `python3 util.py 3`.

This command will read normalized wav files from  `data/project_waves_norm/train` and will write spectrogram images to  `data/project_spect/train`.

#### Convert wav to Timeseries Image

Run `python3 util.py 4`.

This command will read normalized wav files from  `data/project_waves_norm/train` and will write timeseries images to  `data/project_timeseries/train`.

#### Convert wav to MFCC Image

Run `python3 util.py 5`


## Convolutional Neural Network

TODO:

## Feed Forward Neural Network

TODO: