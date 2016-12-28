import collections
import numpy as np
from sigmoid import Sigmoid

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

    def predict(self, features):
        inputs = features
        for layer in self.network:
            inputs = self.activation.forward(np.dot(layer, inputs))
        return inputs
    
    def cost_f(self, predictions, targets):
        return targets - predictions

    def train(self, features, labels):
        outputs = [np.array(features, ndmin=2).T]
        for layer in self.network:
            outputs.append(self.activation.forward(np.dot(layer, outputs[-1])))

        labels = np.array(labels, ndmin=2).T
        errors = self.cost_f(outputs[-1], labels)
        for l_idx, layer in enumerate(self.network[::-1]):
            l_output = outputs[-l_idx - 1]
            l_input = outputs[-l_idx - 2].T
            p_deltas = errors * self.activation.backward(l_output)
            errors = np.dot(layer.T, errors)
            layer += self.learning_rate * np.dot(p_deltas, l_input)
