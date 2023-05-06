from .layer import Layer
import numpy as np

class StaticConvolutionLayer(Layer):
    def __init__(self,kernels,padding,stride_size):
        self.kernels = kernels
        self.padding = padding
        self.stride_size = stride_size
            
    @staticmethod
    def output_size(image_size,kernels,padding,stride_size):
        output_size = 0
        i_h, i_w = image_size
        i_h += padding * 2
        i_w += padding * 2
        for kernel in kernels:
            # Calculate the output dimensions
            k_h, k_w = kernel.shape
            o_h = int(((i_h - k_h + 2 * padding) / stride_size) + 1)
            o_w = int(((i_w - k_w + 2 * padding) / stride_size) + 1)
            output_size += (o_h*o_w)
        return output_size
            
    def convolve(self,input, kernel):
        if self.padding > 0:
            input = np.pad(input, self.padding, 'constant')
        
        # Get the dimensions of the input image and kernel
        i_h, i_w = input.shape
        k_h, k_w = kernel.shape
        # Calculate the output dimensions
        o_h = int(((i_h - k_h + 2 * self.padding) / self.stride_size) + 1)
        o_w = int(((i_w - k_w + 2 * self.padding) / self.stride_size) + 1)
        
        # Create an output array of the correct dimensions
        output = np.zeros((o_h, o_w))
        
        # Perform the convolution
        for i in range(0, i_h - k_h + 1, self.stride_size):
            for j in range(0, i_w - k_w + 1, self.stride_size):
                output[i//self.stride_size, j//self.stride_size] = np.sum(input[i:i+k_h, j:j+k_w] * kernel)
        
        return output
       
    def forwards(self,input):
        result = []
        for kernel in self.kernels:
            result.append(self.convolve(input,kernel))
        result = np.array(result)
        return result.reshape((1,-1))
    
    def backwards(self,error,learning_rate):
        pass
    
    def dropout(self,dropout_rate):
        pass