import numpy as np

class ReLU:
    def __init__(self):
        self.input=None

    def forward(self,input):
        self.input=input
        return np.maximum(0,input)
    
    def backward(self,grad_output):
        grad_input=grad_output.copy()
        grad_input[self.input<=0]=0
        return grad_input