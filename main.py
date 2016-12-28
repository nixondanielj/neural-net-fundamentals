import csv
import numpy as np
import timeit
import network
from data import load_sets

def print_model(nn):
    print 'model:'
    print '\tlayers:%s' % [l.shape for l in nn.network]
    print '\tlrn_rate:%s' % nn.learning_rate

def test(nn, feats, labels):
    total_correct = 0.
    for idx, label in enumerate(labels):
        correct = np.argmax(label)
        prediction = np.argmax(nn.predict(feats[idx]))
        total_correct += (correct == prediction)
    return float(total_correct)/len(labels)

def train(nn, feats, labels):
    print 'training...'
    tenpts = len(feats)/10
    for idx, label in enumerate(train_labels):
        nn.train(train_features[idx], label)
        if idx % tenpts == 0:
            print '%s complete' % (float(idx) / tenpts * 10)
    print 'done'

def train_step_debug(nn, train_labels, train_features):
    for i in range(50):
        prd = nn.predict(train_features[i])
        print 'initial:', prd, np.argmax(prd)
        print 'correct:', np.argmax(train_labels[i])
        nn.train(train_features[i], train_labels[i])
        prd = nn.predict(train_features[i])
        print '2nd attmpt:', prd, np.argmax(prd)

print 'loading sets'
train_labels, train_features, test_labels, test_features = load_sets(10000)
print 'done...'

def train_and_test(nn, limit=None):
    print_model(nn)
    train(nn, train_features, train_labels)
    return test(nn, test_features, test_labels)

def run_default():
    nn = network.NeuralNetwork(784, 10)
    print 'accuracy: %s' % train_and_test(nn)

timeit.timeit(run_default)