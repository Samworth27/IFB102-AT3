import numpy as np
from .layer import Layer
from .dropout import input_mask, output_mask


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.dropout_mask = np.array([True for _ in range(input_size)])
        self.masked_weights = self.weights

    def __getstate__(self):
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "weights": self.weights,
            "bias": self.bias,
        }

    def __setstate__(self, d):
        self.weights = d["weights"]
        self.bias = d["bias"]
        self.input_size = d["input_size"]
        self.output_size = d["output_size"]
        self.dropout_mask = np.array([True for _ in range(self.input_size)])
        self.masked_weights = self.weights

    def dropout(self, dropout_rate):
        noise = np.random.random((self.dropout_mask.size))
        self.dropout_mask = noise > dropout_rate
        self.masked_weights = input_mask(self.weights, self.dropout_mask)

        # Enable re-rolling if all nodes in a layer are dropped
        if np.all(self.dropout_mask == False):
            self.dropout(dropout_rate)

    def forwards(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backwards(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

    def __str__(self):
        return f"Fully Connected layer {self.weights.shape}"

    def __def__(self):
        return str(self)
