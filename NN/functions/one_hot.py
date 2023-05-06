import numpy as np

# labels = np.random.randint(0,10,(100,1))


def one_hot(labels):
    labels = np.array(labels).reshape((-1,1))
    unique_labels = np.unique(labels)
    number_unique_labels = unique_labels.shape[0]
    x = np.where(unique_labels == labels)[1]
    return np.eye(number_unique_labels,dtype=int)[x]