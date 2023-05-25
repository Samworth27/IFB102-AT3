import os
import pickle
import numpy as np

from .layers import Layer
from NN.functions.loss_functions import LossFunction
from NN.functions.output_functions import OutputFunction



class Network:
    def __init__(self):
        self.layers: list[Layer] = []
        
        self.output_function = None
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

        
    def use_loss(self,loss_function:LossFunction):
        self.loss = loss_function.loss
        self.loss_prime = loss_function.prime
        
    def use_output(self,output_function:OutputFunction):
        self.output_function = output_function.output

    def dropout(self,dropout_rate):
        for layer in self.layers:
            layer.dropout(0)
    
    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []
        # enable any nodes that have been dropped
        self.dropout(0)
        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forwards(output)
            output = self.output_function(output)
            result.append(output)

        return np.array(result)

    # train the network
    def train(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            display_error = 0
            self.dropout(0.3)
            for j in range(samples):
                print(f"{j}/ {samples}\r",end="\r")
                # forward propagation
                output = x_train[j]

                for layer in self.layers:
                    output = layer.forwards(output)

                # compute loss (for display purpose only)
                display_error += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backwards(error, learning_rate)

            # calculate average error on all samples
            display_error /= samples
            print(f"Epoch {i+1}/ {epochs} error: {display_error:.6f}")
            
    def save_network(self, file_name):
        try:
            data = {
                'layers': self.layers,
                'output_function': self.output_function,
                'loss': self.loss,
                'loss_prime': self.loss_prime
            }
            self.dropout(0)
            base_dir = os.path.abspath(os.path.dirname(__file__))
            data_dir = "saved_networks"
            file_name = f"{file_name}.pkl"
            full_file_path = os.path.join(base_dir,data_dir,file_name)
            try:
                with open(full_file_path, 'xb') as file:
                    pickle.dump(data,file)
            except:
                with open(full_file_path,'wb') as file:
                    pickle.dump(data,file)
        except:
            print("Could not save")
        
    
    def load_network(self, file_name):
        base_dir = os.path.abspath(os.path.dirname(__file__))
        data_dir = "saved_networks"
        file_name = f"{file_name}.pkl"
        full_file_path = os.path.join(base_dir,data_dir,file_name)
        with open(full_file_path, 'rb') as file:
            data = pickle.load(file)
        
        self.layers = data['layers']
        self.output_function = data['output_function']
        self.loss = data['loss']
        self.loss_prime = data['loss_prime']