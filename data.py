import csv
import numpy as np

def csv_to_list(path, take=None):
    result = []
    with open(path, 'r') as csv_file:
        count = 0
        for row in csv.reader(csv_file):
            count += 1
            result.append(row)
            if take is not None and count >= take:
                return np.asfarray(result)
    return np.asfarray(result)

def split_features_labels(x):
    labels = []
    feature_set = []
    for item in x:
        features = item[1:] / 255. * .99 + .01
        feature_set.append(features)
        label = np.full(10, .01)
        label[int(item[0])] = .99
        labels.append(label)
    return labels, feature_set

def load_sets(limit=None):
    training_set = csv_to_list('data/mnist_train.csv', limit)
    if limit is None:
        test_set = csv_to_list('data/mnist_test.csv')
    else:
        test_set = csv_to_list('data/mnist_test.csv', limit/5)
    train_labels, train_features = split_features_labels(training_set)
    test_labels, test_features = split_features_labels(test_set)
    return train_labels, train_features, test_labels, test_features