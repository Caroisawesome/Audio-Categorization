import statistics as stat
import feed_forward

def get_cross_validation_data(data, iteration):
    size_chunk = len(data)/5
    start_idx = iteration * size_chunk
    end_idx = start_idx + size_chunk -1

    test_data = data[start_idx, end_idx]
    del data[start_idx,end_idx]

    return data, test_data


def perform_cross_validation(classifier, feature):
    paths       = ['project_spect/train/', 'project_timeseries/train/', 'project_mfccs/train']
    pickles     = get_picklefiles()

    data, labels = feed_forward.process_data(feature, pickles, paths)

    for i in range(0,5):
        data_train, labels_train = get_cross_validation_data(data, i)
        data_test, labels_test = get_cross_validation_data(labels, i)

        accuracies = []

        if classifier == 1:
            loss, accuracy = feed_forward.run_NN(data_train, labels_train, data_test, labels_test)
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
        classifier = sys.argv[1]
        feature = sys.argv[2]

        m, sdt = perform_cross_validation(classifier,feature)
        print("Avgerage accruacy: ", m)
        print("Standard devaition: ", std)
