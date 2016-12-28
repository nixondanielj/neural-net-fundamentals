import collections
import numpy as np
from sigmoid import Sigmoid
import random

class NeuralNetwork:
    def __init__(self, input_width, output_width, hidden_width=None, depth=3, learning_rate=.1, activation=Sigmoid):
        self.network = []
        self.learning_rate = learning_rate
        self.activation = Sigmoid()
        last_width = input_width
        for layer_idx in range(depth - 2):
            if isinstance(hidden_width, (collections.Sequence, tuple, np.ndarray)):
                width = hidden_width[layer_idx]
            elif isinstance(hidden_width, (int, float)):
                width = hidden_width
            else:
                width = np.abs(input_width-output_width)/(depth-1)
            scale = last_width ** -.5
            layer = np.random.normal(scale=scale, size=(width, last_width))
            self.network.append(layer)
            last_width = width
        scale = last_width ** -.5
        self.network.append(np.random.normal(scale=scale, size=(output_width, last_width)))

    def predict(self, inputs):
        for layer in self.network:
            inputs = self.activation.forward(np.dot(layer, inputs))
        return inputs
    
    def cost_f(self, predictions, targets):
        return targets - predictions

    def _backprop(self, features, labels):
        outputs = [np.array(features, ndmin=2).T]
        for layer in self.network:
            outputs.append(self.activation.forward(np.dot(layer, outputs[-1])))

        labels = np.array(labels, ndmin=2).T
        errors = self.cost_f(outputs[-1], labels)
        weight_deltas = []
        for l_idx, layer in enumerate(self.network[::-1]):
            p_deltas = errors * self.activation.backward(outputs[-l_idx - 1])
            errors = np.dot(layer.T, errors)
            weight_deltas.insert(0, np.dot(p_deltas, outputs[-l_idx - 2].T))
        return weight_deltas

    def train(self, training_data, batch_size, epochs=1):
        for _ in range(epochs):
            #random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + batch_size]
                for k in xrange(0, len(training_data), batch_size)]
            for batch in training_data:
                self.train_batch([batch])

    def train_batch(self, batch):
        raw_deltas = [self._backprop(f, l) for f, l in batch]
        deltas = np.average(raw_deltas, axis=0)
        self.network += self.learning_rate * deltas
