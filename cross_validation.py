import statistics as stat
import numpy as np
import feed_forward
import sys
import math

def compute_start_stop_idx(num_items, iteration):
    size_chunk = math.floor(num_items/5)
    start_idx = iteration * size_chunk
    end_idx = start_idx + size_chunk -1
    return start_idx, end_idx

def get_data_subset(data, start,stop, num_items):
    test_data = data[start:stop]
    for i in range(0, start):
        train_data = np.delete(data,i, axis=0)
    for i in range(stop,num_items):
        train_data = np.delete(data, i, axis=0)

    print("shape train", train_data.shape)
    print("stape test", test_data.shape)
    return train_data, test_data

def perform_cross_validation(classifier, feature):
    paths       = ['project_spect/train/', 'project_timeseries/train/', 'project_mfccs']
    pickles     = feed_forward.get_picklefiles()
    data, labels = feed_forward.preprocess_data(feature, pickles, paths)
    num_items = len(labels)

    for i in range(0,5):

        start, stop = compute_start_stop_idx(num_items, i)

        train_data, test_data = get_data_subset(data, start, stop, num_items)
        train_labels, test_labels = get_data_subset(labels, start, stop, num_items)
        accuracies = []


        if classifier == 1:
            loss, accuracy = feed_forward.run_NN(train_data, train_labels, test_data, test_labels)
        elif classifier == 2:
            # TODO!! ADD CNN HERE
            print("5-fold cross validation is not implemented with CNN yet.")
            loss, accuracy = (0,0)

        accuracies.append(accuracy)

    m = stat.mean(accuracies)
    std = stat.stdev(accuracies)
    return (m, std)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Arguments required")
        print("run: python3 cross_validation.py {classifier} {feature}")
        print("classifier: 1: Feed-Forward NN, 2: CNN")
        print("feature:  1: Spectogram, 2: Timeseries, 3: MFCC")
    else:
        classifier = int(sys.argv[1])
        feature = int(sys.argv[2])

        m, sdt = perform_cross_validation(classifier,feature)
        print("Avgerage accruacy: ", m)
        print("Standard devaition: ", std)
