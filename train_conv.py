import numpy as np

from NN.network import Network
from NN.functions import activation_functions, loss_functions, output_functions
from NN.layers import ConvolutionLayer, MaxPooling, SoftMax
from NN.layers.activation_layer import ActivationLayer

from NN.functions.one_hot import one_hot

from MNIST_data.mnist_data import get_train_data, get_test_data

x_train, y_train = get_train_data()
x_train = x_train.reshape(x_train.shape[0],28,28)
x_train = x_train.astype(np.float32)
x_train /= 255
y_train_oh = one_hot(y_train)

x_test, y_test = get_test_data()
x_test = x_test.reshape(x_test.shape[0],28,28)
x_test = x_test.astype(np.float32)
x_test /= 255
y_test_oh = one_hot(y_test)
print("Data loaded")

network = Network()
network.add(ConvolutionLayer(16,3))
network.add(MaxPooling(2))
network.add(SoftMax(13*13*16,10))

network.use_loss(loss_functions.MSE())
network.use_output(output_functions.Pass)

network.train(x_train,y_train_oh,20,0.3)

network.save_network('conv_net1')

