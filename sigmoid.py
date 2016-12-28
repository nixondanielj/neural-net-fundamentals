from scipy.special import expit

class Sigmoid:

    def forward(self, x):
        return expit(x)

    def backward(self, x):
        x = self.forward(x)
        return x * (1. - x)
