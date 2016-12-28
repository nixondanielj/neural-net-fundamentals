import csv
import numpy as np
import timeit
import network
from data import DataLoader

def print_model(nn):
    print 'model:'
    print '\tlayers:%s' % [l.shape for l in nn.network]
    print '\tlrn_rate:%s' % nn.learning_rate

def test(nn, data):
    total_correct = 0.
    for feats, label in data:
        correct = np.argmax(label)
        prediction = np.argmax(nn.predict(feats))
        total_correct += (correct == prediction)
    return float(total_correct)/len(data)

def train(nn, data):
    print 'training...'
    nn.train(data, 10)
    print 'done'

def train_step_debug(nn, train_labels, train_features):
    for i in range(50):
        prd = nn.predict(train_features[i])
        print 'initial:', prd, np.argmax(prd)
        print 'correct:', np.argmax(train_labels[i])
        nn.train(train_features[i], train_labels[i])
        prd = nn.predict(train_features[i])
        print '2nd attmpt:', prd, np.argmax(prd)

ldr=DataLoader()
def train_and_test(nn, limit=None):
    print 'loading sets'
    training_data, testing_data = ldr.load_sets(limit)
    print_model(nn)
    train(nn, training_data)
    return test(nn, testing_data)

def run_default():
    nn = network.NeuralNetwork(784, 10)
    print 'accuracy: %s' % train_and_test(nn)

def run_timed():
    nn = network.NeuralNetwork(784, 10)
    print 'accuracy: %s' % train_and_test(nn, 1000)

print '%s seconds' % str(timeit.timeit(run_timed, number=1))[:4]