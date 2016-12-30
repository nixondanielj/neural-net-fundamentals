import numpy as np

class RELU:
    def forward(self, x):
        return min(x, 1)
    
    def backward(self, x):
        np.maximum(x, 0, x)
        return x