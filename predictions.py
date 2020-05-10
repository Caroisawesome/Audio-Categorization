import sys
import csv
import os
import numpy as np
import matplotlib.image as mpimg 
import pickle

# Dimensions of data being passed into network
DIM_ROWS = 234 # max 277
DIM_COLS = 200 # max 372
DIM_CHANNELS = 3 # max 3
TOTAL_INPUT_SIZE = DIM_ROWS*DIM_COLS*DIM_CHANNELS

'''
===============================================================================
write_to_csv - Write predictions to a csv file
@params: names - [string] - list of the song ids
         predictions - [int] - list of the genre predictions
         out_path - string - path/name of csv file that will hold predictions

@returns: void
===============================================================================
'''
def write_to_csv(names, predictions,out_path):
    with open(out_path, 'w') as file:
        writer = csv.writer(file)
        for i in range(0, len(names)):
            writer.writerow([names[i], predictions[i]])

'''
===============================================================================
get_test_data - process testing data
@params: in_path - string - path to testing data

@returns: names - [string] - list of song ids that were in directory
          data - ndarray - image data formatted to put into NN
===============================================================================
'''
def get_test_data(in_path):
    t_files = []
    names = []
    for root, dirs, files in os.walk(in_path, topdown=False):
        for name in files:
            path = os.path.join(root,name)
            t_files.append(path)
            name = os.path.splitext(name)[0]
            names.append(name)

    data     = np.zeros((len(t_files),DIM_ROWS,DIM_COLS,DIM_CHANNELS))

    #for path in t_files:
    for i in range(0, len(t_files)):
        image = mpimg.imread(t_files[i])
        np_image = np.array(image)[:DIM_ROWS, :DIM_COLS, :DIM_CHANNELS]
        data[i] = np_image

    return data, names

'''
===============================================================================
make_predictions - make genre predictions based on the test data
@params: classifier - {0|1} - indicates whether to use CNN or FFNN
         in_path - path to test data
         out_path - path/name of csv file to write predictions to

@returns: void
===============================================================================
'''
def make_predictions(classifier, in_path, out_path):

    test_data, names = get_test_data(in_path)
    print("test_data", test_data)
    print("names", names)

    if classifier == 0:
        model = pickle.load(open('CNN_model.p', 'rb'))
    elif classifier == 1:
        model = pickle.load(open('feed_forward_model.p', 'rb'))
    else:
        print("Incorrect classifier. Try again.")

    predictions = model.predict_classes(test_data)
    write_to_csv(names, predictions, out_path)

if __name__ == '__main__':
    # make predictions about data

    predictions_path = 'predictions/'

    paths = [
        'project_spect/test/',
        'project_timeseries/test/',
        'project_mfccs/test/'
    ]

    if len(sys.argv) < 2:
        print("Argument required")
        print("run python3 predictions.py {classifier} {feature}")
        print("Classifiers:")
        print("0: CNN")
        print("1: FFNN")
        print("Features:")
        print("0: spectogram")
        print("1: timeseries")
        print("2: MFCC")

    else:
        classifier = int(sys.argv[1])
        feature = int(sys.argv[2])
        in_path = paths[feature]
        out_path = predictions_path + 'feature_'+str(feature)+'.csv'
        make_predictions(classifier, in_path, out_path)
