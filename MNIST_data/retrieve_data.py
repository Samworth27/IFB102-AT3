import requests
import os

def retrieve_data():
    folder_dir = os.path.abspath(os.path.dirname(__file__))
    train_url = "https://pjreddie.com/media/files/mnist_train.csv"
    test_url = "https://pjreddie.com/media/files/mnist_test.csv"

    if not os.path.exists(os.path.join(folder_dir,'./train.csv')):
        print("Downloading train.csv")
        with open(os.path.join(folder_dir,'train.csv'), 'xb') as file:
            response = requests.get(train_url)
            file.write(response.content)
    if not os.path.exists(os.path.join(folder_dir,'./test.csv')):
        print("Downloading test.csv")
        with open(os.path.join(folder_dir,'test.csv'), 'xb') as file:
            response = requests.get(test_url)
            file.write(response.content)