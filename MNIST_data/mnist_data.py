import os
import csv
import numpy as np
import pickle
from .retrieve_data import retrieve_data

base_path = os.path.abspath(os.path.dirname(__file__))
data_dir  = 'MNIST_data'

def read_csv(file):
    if not(os.path.exists('./train.csv') and os.path.exists('./test.csv')):
        retrieve_data()
        
    with open(os.path.join(base_path,file)) as file:
        data_reader = csv.reader(file)
        data = [i for i in data_reader]
        number_samples = len(data)-1
        data = np.array(data[1:]).astype(int)
        x,y = data[:,1:].reshape(number_samples,28,28), data[:,0]
        return x,y
    
def pickle_dump(data,file_name):
    with open(os.path.join(base_path,file_name),'wb') as file:
        pickle.dump(data,file)
    
def read_pickle(file_name):
    with open(os.path.join(base_path,file_name),'rb') as file:
        return pickle.load(file)

def get_train_data()-> tuple[np.array,np.array]:
    try:
        print("Attempting to read train.pkl")
        data = read_pickle('train.pkl')
        print("Data read from train.pkl")
        return data
    except:
        print("Reading data from train.csv")   
        data= read_csv('train.csv')
        print("Data read from train.csv, storing data in train.pkl")
        pickle_dump(data,'train.pkl')
        print("Data stored in train.pkl")
        return data

def get_test_data() -> tuple[np.array,np.array]:
    try:
        print("Attempting to read test.pkl")
        data = read_pickle('test.pkl')
        print("Data read from test.pkl")
        return data
    except:
        print("Reading data from test.csv")   
        data= read_csv('test.csv')
        print("Data read from test.csv, storing data in test.pkl")
        pickle_dump(data,'test.pkl')
        print("Data stored in test.pkl")
        return data