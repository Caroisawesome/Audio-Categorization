import sys
import csv

def write_to_csv(names, predictions,out_path):
    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(0, len(names)):
            writer.writerow([ names[i], predictions[i]])

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

    for path in t_files:
        image = mpimg.imread(path)
        np_image = np.array(image)[:DIM_ROWS, :DIM_COLS, :DIM_CHANNELS]
        data[i] = np_image

    return data, names

def make_predictions(classifier, in_path, out_path):

    test_data, names = get_test_data(in_path)
    print("test_data", test_data)
    print("names", names)

    if classifier == 0:
        # make predictions with CNN
        model = True
        predictions = []
        print("TODO: set up predictions with CNN")
    elif classifier == 1:
        model = pickle.load(open('feed_forward_model.p', 'rb'))
        predictions = model.predict_classes(test_data)
    else:
        print("Incorrect classifier. Try again.")

    write_to_csv(names,predictions,out_path)

if __name__ == '__main__':
    # make predictions about data

    predictions_path = '/predictions'

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
