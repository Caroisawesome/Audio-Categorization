# Audio Categorization

This project uses Convolutional Neural Networks and Feed Forward Neural Networks to classify the genre of a song. The networks use 2401 classified songs for training. The songs are then converted into spectrograms, timeseries images, and MFCC images which then become inputs for the networks. 

## Data Processing

The mp3 files used for this project are stored in a directory `data/project3/train` and `data/project3/test` that is not tracked by github. Data that has already been pre-processed and converted into images are stored in the `project_spect`, `project_timeseries`, and `project_mfccs` directories. Each of those directory has subdirectories `train` and `test` which represent the training data and the testing data. The process for preparing the data for the NNs is as follows:

1. Convert mp3 to wav files
2. Normalize wav files
3. Convert wav to image (Spectrogram, Timeseries, or MFCC)

Each of these actions can be done on either the training, or the testing data.

Once the data has been processed and converted into images, the images can be moved to `project_spect`, `project_timeseries`, `project_mfccs` in order to be read in by the neural networks.

#### Convert mp3 to wav files

Run `python3 util.py 0 {0: training, 1: testing}`.

This command will read mp3 files from `data/project3/{train|test}` and will write wav files to `data/project_waves/{train|test}`.

#### Normalize wav files

Run `python3 util.py 1 {0: training, 1: testing}`.

This command will read wav files from `data/project_waves/{train|test}` and will write normalized wav files to `data/project_waves_norm/{train|test}`.

#### Convert wav to Spectrogram Image

Run `python3 util.py 2 {0: training, 1: testing}`.

This command will read normalized wav files from  `data/project_waves_norm/{train|test}` and will write spectrogram images to  `data/project_spect/{train|test}`.

#### Convert wav to Timeseries Image

Run `python3 util.py 3 {0: training, 1: testing}`.

This command will read normalized wav files from  `data/project_waves_norm/{train|test}` and will write timeseries images to  `data/project_timeseries/{train|test}`.

#### Convert wav to MFCC Image

Run `python3 util.py 4 {0: training, 1: testing}`.

This command will read normalized wav files from `data/project_waves_norm/{train|test}` and will write MFCC images to `data/project_mfccs/{train|test}`.


## Convolutional Neural Network

TODO:

## Feed Forward Neural Network

TODO: