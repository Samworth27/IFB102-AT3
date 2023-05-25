# https://towardsdatascience.com/building-a-convolutional-neural-network-from-scratch-using-numpy-a22808a00a40
from .layer import Layer
import numpy as np

class ConvolutionLayer(Layer):
    def __init__(self, kernel_number, kernel_size):
        self.kernel_number = kernel_number
        self.kernel_size = kernel_size
        self.kernels = np.random.randn(kernel_number, kernel_size, kernel_size)/ (kernel_size**2)

    def patches_generator(self, image):
        image_height, image_width = image.shape
        self.image = image
        for h in range(image_height - self.kernel_size+1):
            for w in range(image_width - self.kernel_size+1):
                patch = image[h:(h+self.kernel_size),w:(w+self.kernel_size)]
                yield patch, h, w

    def forwards(self, image):
        image_height, image_width = image.shape
        convolution_output = np.zeros((image_height-self.kernel_size+1, image_width-self.kernel_size+1, self.kernel_number))
        for patch, h, w in self.patches_generator(image):
            convolution_output[h,w] = np.sum(patch*self.kernels, axis=(1,2))
        return convolution_output

    def backwards(self, output_error, learning_rate):
        dE_dk = np.zeros(self.kernels.shape)
        for patch, h, w in self.patches_generator(self.image):
            for f in range(self.kernel_number):
                dE_dk[f] += patch * output_error[h,w,f]
        self.kernels -= learning_rate * dE_dk
        return dE_dk
    
    def dropout(self, dropout_rate):
        pass