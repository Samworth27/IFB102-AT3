# https://python-course.eu/machine-learning/running-neural-network-with-python.php

import numpy as np
from scipy.stats import truncnorm

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, 
                     (upp - mean) / sd, 
                     loc=mean, 
                     scale=sd)

class Functions:
    
    @classmethod
    def softmax(cls,x):
        e_x = np.exp(x - np.max(x))
        return e_x/ e_x.sum()
    
    @classmethod
    def softmax_derivative(cls, x):
        I = np.eye(x.shape[0])
        return cls.softmax(x) * (I - cls.softmax(x).T)
    
    @classmethod
    @np.vectorize
    def sigmoid(cls,x):
        return 1/(1 + np.exp(-x))
    
    @classmethod
    def sigmoid_derivative(cls,x):
        return cls.sigmoid(x) * (1 - cls.sigmoid(x))
    
    @classmethod
    def sigmoid_pair(cls):
        return (cls.sigmoid,cls.sigmoid_derivative)
    
class Layer:
    def __init__(self, input_nodes,output_nodes,functions):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.activation, self.derivative = functions
        
        n = input_nodes * output_nodes
        rad = 1 / np.sqrt(input_nodes)
        X = truncated_normal(2,1,-rad,rad)
        self.weights = X.rvs(n).reshape((output_nodes, input_nodes))
        
    def forward(self,input):
        self.input = input
        self.output = self.activation(np.dot(self.weights,input))
        return self.output
    
    def backward(self,output_error,learning_rate):
        input_error = np.dot(output_error,self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        
        self.weights -= learning_rate * weights_error
        # self.bias -= learning_rate * output_error
        return input_error
        

class NeuralNetwork:
    def __init__(self, structure, learning_rate, bias = None):
        self.structure = structure
        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matrices()

    
    def create_weight_matrices(self):
        bias_node = 1 if self.bias else 0
        self.layers = []
        
        for i,output_nodes in enumerate(self.structure):
            if i == 0:
                continue
            input_nodes = self.structure[i-1]
            self.layers.append(Layer(input_nodes, output_nodes,Functions.sigmoid_pair()))
            
    def forward(self, input):
        values = np.array(input, ndmin=2).T
        for layer in self.layers:
            values = layer.forward(values)
        print(values)
        
    def backward(self,target,output):
        target_vector = np.array(target_vector, ndmin=2).T
        error = target_vector - output
        for layer in reversed(self.layers):
            delta = layer.backward(error)

    #         in_vector = result_vectors[layer_index - 1]
            
    #         gradient = (np.sum(output_errors, axis = 0).reshape(-1,1) * self.derivative_function(out_vector))
    #         tmp = np.dot(gradient,in_vector.T)
            
            
    #         self.weights[layer_index-1] += self.learning_rate * tmp
    #         output_errors = np.dot(self.weights[layer_index - 1].T, output_errors)
            
    #         if self.bias:
    #             output_errors = output_errors[:-1,:]
        
        
        
        
        
        
        
        
        
        
        
    # def train(self,input_vector,target_vector):
    #     number_layers = len(self.structure)
    #     input_vector = np.array(input_vector, ndmin=2).T
        
    #     layer_index = 0
    #     result_vectors = [input_vector]
    #     while layer_index < number_layers - 1:
    #         in_vector = result_vectors[-1]
    #         if self.bias:
    #             in_vector = np.concatenate((in_vector, [[self.bias]]))
    #             result_vectors[-1] = in_vector
    #         x = np.dot(self.weights[layer_index],in_vector)
    #         out_vector = self.activation_function(x)
    #         result_vectors.append(out_vector)
    #         layer_index += 1
        
    #     layer_index = number_layers - 1
    #     target_vector = np.array(target_vector, ndmin=2).T
    #     output_errors = target_vector - out_vector
    #     while layer_index > 0:
    #         out_vector = result_vectors[layer_index]
    #         in_vector = result_vectors[layer_index - 1]
            
    #         if self.bias and not layer_index == (number_layers - 1):
    #             out_vector = out_vector[:-1,:].copy()
            
    #         gradient = (np.sum(output_errors, axis = 0).reshape(-1,1) * self.derivative_function(out_vector))
    #         tmp = np.dot(gradient,in_vector.T)
            
            
    #         self.weights[layer_index-1] += self.learning_rate * tmp
    #         output_errors = np.dot(self.weights[layer_index - 1].T, output_errors)
            
    #         if self.bias:
    #             output_errors = output_errors[:-1,:]
            
    #         layer_index -= 1

        
    
    # def run(self, input_vector):
        
    #     number_layers = len(self.structure)
    #     if self.bias:
    #         input_vector = np.concatenate((input_vector, [self.bias]))
        
    #     input_vector = np.array(input_vector,ndmin=2).T
        
    #     layer_index = 1
    #     while layer_index < number_layers:
    #         x = np.dot(self.weights[layer_index-1],input_vector)
    #         output_vector = self.activation_function(x)
    #         input_vector = output_vector
    #         if self.bias:
    #             input_vector = np.concatenate((input_vector,[[self.bias]]))
            
    #         layer_index += 1
        
    #     return output_vector
    
    # def evaluate(self,data,labels):
    #     corrects, wrongs = 0,0
    #     # use zip?
    #     for i in range(len(data)):
    #         result = self.run(data[i])
    #         result_max = result.argmax()
    #         if result_max == labels[i].argmax():
    #             corrects += 1
    #         else:
    #             wrongs += 1
    #     return corrects, wrongs

