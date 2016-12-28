from scipy.special import expit

class Sigmoid:
    def forward(self, x):
        return expit(x)

    def backward(self, x):
        return expit(x) * (1-expit(x))