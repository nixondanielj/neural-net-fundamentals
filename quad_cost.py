import numpy as np
from sigmoid import Sigmoid

class QuadraticCost(object):

    def __init__(self):
        self.sigmoid = Sigmoid()

    def fn(self, a, y):
        return 0.5*np.linalg.norm(a-y)**2

    def delta(self, z, a, y):
        return (a-y) * self.sigmoid.backward(z)
