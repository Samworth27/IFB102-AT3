import numpy as np

def input_mask(weights,mask):
    indices = np.where(mask)[0]
    return(weights[indices])
    
def output_mask(weights,mask):
    indices = np.where(mask)[0]
    return(weights[:,indices])
