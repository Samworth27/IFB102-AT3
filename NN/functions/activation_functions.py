import numpy as np


class ActivationFunction:
    def activation(self, x):
        raise NotImplementedError

    def prime(self, x):
        raise NotImplementedError


class Tanh(ActivationFunction):
    def activation(self, x):
        return np.tanh(x)

    def prime(self, x):
        return 1-np.tanh(x)**2

class ReLU(ActivationFunction):
    def activation(self,x):
        return np.maximum(0.0,x)
    
    def prime(self,x):
        return np.where(x<=0,0,1)