import numpy as np

from NN.network import Network
from NN.functions import activation_functions, loss_functions
from NN.layers import FullyConnectedLayer, ActivationLayer
from NN.layers.activation_layer import ActivationLayer


# XOR training data

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

network = Network()
network.add(FullyConnectedLayer(2,5))
network.add(ActivationLayer(activation_functions.Tanh()))
network.add(FullyConnectedLayer(5,1))
network.add(ActivationLayer(activation_functions.Tanh()))

network.use(loss_functions.MSE())
network.train(x_train,y_train,1000,0.1)
print(network.predict(x_train))

