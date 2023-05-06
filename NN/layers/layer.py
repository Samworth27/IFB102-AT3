# Base class
class Layer:
    def __init__(self, dropout_rate = 0):
        self.dropout_rate = dropout_rate
        self.input = None
        self.output = None
    
    def dropout(self,dropout_rate):
        raise NotImplementedError

    # computes the output Y of a layer for a given input X
    def forwards(self, input, enabled:list = None):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backwards(self, output_error, learning_rate):
        raise NotImplementedError