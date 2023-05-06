import numpy as np

class NeuralNetworkLayer:
    
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))
        
    def forward(self, X):
        self.input = X
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    def backward(self, dZ):
        dW = np.dot(self.input.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dA = np.dot(dZ, self.weights.T)
        return dW, db, dA
    
    def activation(self,x):
        return 1/(1 + np.exp(-x))
    
    def derivative(self,x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = NeuralNetworkLayer(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dZ):
        for layer in reversed(self.layers):
            dW, db, dZ = layer.backward(dZ)
            layer.weights -= 0.01 * dW
            layer.bias -= 0.01 * db
            
    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss and print progress
            loss = np.mean((output - y)**2)
            if i % 100 == 0:
                print(f"Epoch {i}: Loss = {loss:.4f}")
            
            # Backward pass
            dZ = output - y
            self.backward(dZ)
            
            # Update learning rate
            learning_rate *= 0.95
    
    def predict(self,input):
        return self.forward(input)