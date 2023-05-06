import numpy as np

class OutputFunction:
    def output(self,x):
        raise NotImplementedError
    
class ArgMax(OutputFunction):
    def output(self, x):
        x = np.array(x)
        return x.argmax()