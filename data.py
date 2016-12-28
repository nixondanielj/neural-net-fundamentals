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
    data = []
    for item in x:
        features = item[1:] / 255. * .99 + .01
        label = np.full(10, .01)
        label[int(item[0])] = .99
        data.append((features, label))
    return data

class DataLoader:
    training_data = None
    testing_data = None

    def load_sets(self, limit=None):
        if self.training_data is not None:
            if limit is None:
                return self.training_data, self.testing_data
            elif limit <= len(self.training_data):
                return self.training_data[:limit], self.testing_data[:limit/5]
        training_set = csv_to_list('data/mnist_train.csv', limit)
        if limit is None:
            test_set = csv_to_list('data/mnist_test.csv')
        else:
            test_set = csv_to_list('data/mnist_test.csv', limit/5)
        self.training_data = split_features_labels(training_set)
        self.testing_data = split_features_labels(test_set)
        return self.training_data, self.testing_data
