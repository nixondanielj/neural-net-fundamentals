import csv
import numpy as np
import timeit
import random
from network import NeuralNetwork
from data import DataLoader
from cross_entropy import CrossEntropyCost

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

def train(nn, training_data, batch_size, epochs=1, test_data=None):
    lr = nn.learning_rate
    for epoch in xrange(1, epochs + 1):
        print 'starting epoch %s' % epoch
        print 'learning rate is %s' % nn.learning_rate
        random.shuffle(training_data)
        mini_batches = [
            training_data[k:k + batch_size]
            for k in xrange(0, len(training_data), batch_size)]
        for batch in mini_batches:
            nn.train_batch(batch)
        nn.learning_rate *= (1.-1./epochs)
        print 'finished epoch %s' % epoch
        print 'accuracy %s%%' % (test(nn, test_data) * 100.)
        print ''
    nn.learning_rate = lr

ldr = DataLoader()
train_data, test_data = ldr.load_sets(10000)
random.shuffle(test_data)

## quadratic
print 'starting quadratic'
nn = NeuralNetwork(784, 10, 100, 3, .3)
print_model(nn)
train(nn, train_data, 20, 10, test_data)
print 'final accuracy %s%%' % (test(nn, test_data) * 100)
print ''

## cross-entropy
print 'starting ce'
nn = NeuralNetwork(784, 10, 100, 3, .3, cost=CrossEntropyCost)
print_model(nn)
train(nn, train_data, 20, 10, test_data)
print 'final accuracy %s%%' % (test(nn, test_data) * 100)
print ''
