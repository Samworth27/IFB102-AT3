import numpy as np

from NN.network import Network
from NN.functions import activation_functions, loss_functions, output_functions
from NN.layers import FullyConnectedLayer, ActivationLayer
from NN.layers.activation_layer import ActivationLayer

from NN.functions.one_hot import one_hot

from MNIST_data.mnist_data import get_train_data, get_test_data

x_train, y_train = get_train_data()
x_train = x_train.reshape(x_train.shape[0],1,28*28)
x_train = x_train.astype(np.float32)
x_train /= 255
y_train_oh = one_hot(y_train)

x_test, y_test = get_test_data()
x_test = x_test.reshape(x_test.shape[0],1,28*28)
x_test = x_test.astype(np.float32)
x_test /= 255
y_test_oh = one_hot(y_test)
print("Data loaded")
try:
    network = Network.load_network("net2")
except:
    network = Network()
    network.add(FullyConnectedLayer(28*28,50))
    network.add(ActivationLayer(activation_functions.Tanh()))
    network.add(FullyConnectedLayer(50,50))
    network.add(ActivationLayer(activation_functions.Tanh()))
    network.add(FullyConnectedLayer(50,10))
    network.add(ActivationLayer(activation_functions.Tanh()))

    network.use_loss(loss_functions.MSE())
    network.use_output(output_functions.ArgMax())

train_data = (x_train,y_train_oh)
test_data = (x_test, y_test)
network.train_batch(train_data, test_data,100,0.05,10000)

network.save_network('net2')

