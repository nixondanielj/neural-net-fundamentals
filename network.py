import collections
import numpy as np
from sigmoid import Sigmoid
from quad_cost import QuadraticCost

class NeuralNetwork:
    def __init__(self, input_width, output_width, \
        hidden_width=None, depth=3, learning_rate=.1, \
        activation=Sigmoid, cost=QuadraticCost, \
        reg_rate=0.):
        self.network = []
        self.learning_rate = learning_rate
        self.activation = activation()
        self.cost_f = cost()
        self.reg_rate = reg_rate
        last_width = input_width
        for layer_idx in range(depth - 2):
            if isinstance(hidden_width, (collections.Sequence, tuple, np.ndarray)):
                width = hidden_width[layer_idx]
            elif isinstance(hidden_width, (int, float)):
                width = hidden_width
            else:
                width = np.abs(input_width-output_width)/(depth-1)
            scale = last_width ** -.5
            layer = np.random.normal(scale=scale, size=(last_width, width))
            self.network.append(layer)
            last_width = width
        scale = last_width ** -.5
        self.network.append(np.random.normal(scale=scale, size=(last_width, output_width)))

    def predict(self, inputs):
        for layer in self.network:
            inputs = self.activation.forward(np.dot(inputs, layer))
        return inputs

    def train_batch(self, batch):
        feat_matrix = np.array([f for f, l in batch], ndmin=2)
        label_matrix = np.array([l for f, l in batch], ndmin=2)
        outputs = [feat_matrix]
        zs = [feat_matrix]
        for layer in self.network:
            zs.append(np.dot(outputs[-1], layer))
            outputs.append(self.activation.forward(zs[-1]))

        delta = self.cost_f.delta(zs[-1], outputs[-1], label_matrix)
        w_deltas = [np.dot(outputs[-2].T, delta)]
        for l in xrange(2, len(self.network) + 1):
            delta = np.dot(delta, self.network[-l + 1].T)
            delta *= self.activation.backward(zs[-l])
            w_deltas.append(np.dot(outputs[-l-1].T, delta))
        if self.reg_rate:
            regularizer = 1 - self.learning_rate * self.reg_rate / len(batch)
            self.network = np.multiply(self.network, regularizer)
        self.network -= self.learning_rate * np.array(w_deltas[::-1])/len(batch)
