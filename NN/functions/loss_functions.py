import numpy as np


class LossFunction:
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    def prime(self, y_true, y_pred):
        raise NotImplementedError


class MSE(LossFunction):
    def loss(self, y_true, y_pred):
        return np.mean(np.power(y_true-y_pred, 2))

    def prime(self, y_true, y_pred):
        return 2*(y_pred-y_true)/y_true.size

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size