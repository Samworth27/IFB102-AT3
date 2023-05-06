import numpy as np

from .layer import Layer
from ..functions.activation_functions import ActivationFunction


class ActivationLayer(Layer):
    def __init__(self, activation_function: ActivationFunction):

        self.activation = activation_function.activation
        self.prime = activation_function.prime
        
    def dropout(self, dropout_rate):
        pass


    def forwards(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output


    def backwards(self, output_error, learning_rate):
        return self.prime(self.input)*output_error