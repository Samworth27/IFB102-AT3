import csv
import matplotlib.pyplot as plt
import numpy as np
import os

from train import NeuralNetwork


targets = {
    'Iris-setosa': [1, 0, 0],
    'Iris-versicolor': [0, 1, 0],
    'Iris-virginica': [0, 0, 1]
}


with open(os.path.join(os.path.dirname(__file__), "iris_csv.csv")) as file:
    data = np.array([i for i in csv.reader(file)][2:])
    np.random.shuffle(data)
    features = np.array([[sep_len, sep_wid, pet_len, pet_wid]
                        for sep_len, sep_wid, pet_len, pet_wid, _ in data]).astype(np.float32)
    labels = np.array([targets[species] for _, _, _, _, species in data])
    
    split = int(len(data)*0.7)
    training_features = features[:split]
    training_labels = labels[:split]
    test_features = features[split:]
    test_labels = labels[split:]
    
test_network = NeuralNetwork([4,50,3],0.3,None)
test_network.forward(training_features[0])