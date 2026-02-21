import numpy as np

class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, input):
        
        self.input_shape = input.shape 
        return input.reshape(-1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)