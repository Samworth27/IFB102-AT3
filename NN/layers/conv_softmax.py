from .layer import Layer
import numpy as np


class SoftMax(Layer):
    def __init__(self, input_units, output_units):
        self.weight = np.random.randn(input_units, output_units) / input_units
        self.bias = np.zeros(output_units)

    def forwards(self, image):
        self.original_shape = image.shape
        image_flattened = image.flatten()
        self.flattened_input = image_flattened
        first_output = np.dot(image_flattened, self.weight) + self.bias
        self.output = first_output
        softmax_output = np.exp(first_output) / np.sum(np.exp(first_output), axis=0)
        return softmax_output

    def backwards(self, dE_dY, alpha):
        for i, gradient in enumerate(dE_dY):
            if gradient == 0:
                continue
            transformation_eq = np.exp(self.output)
            S_total = np.sum(transformation_eq)

            dY_dZ = -transformation_eq[i] * transformation_eq / (S_total**2)
            dY_dZ[i] = (
                transformation_eq[i] * (S_total - transformation_eq[i]) / (S_total**2)
            )

            dZ_dw = self.flattened_input
            dZ_db = 1
            dZ_dX = self.weight

            dE_dZ = gradient * dY_dZ

            dE_dw = dZ_dw[np.newaxis].T @ dE_dZ[np.newaxis]
            dE_db = dE_dZ * dZ_db
            dE_dX = dZ_dX @ dE_dZ

            self.weight -= alpha * dE_dw
            self.bias -= alpha * dE_db

            return dE_dX.reshape(self.original_shape)

    def dropout(self, dropout_rate):
        pass