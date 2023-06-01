import numpy as np

class OutputFunction:
    def output(self,x):
        raise NotImplementedError
    
class ArgMax(OutputFunction):
    @staticmethod
    def output(x):
        x = np.array(x)
        return x.argmax()
    
class Pass(OutputFunction):
    @staticmethod
    def output(x):
        return x
    
    