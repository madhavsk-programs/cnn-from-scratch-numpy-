import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        
        limit = np.sqrt(1 / input_size)
        self.weights = np.random.uniform(-limit, limit, (output_size, input_size))
        self.bias = np.zeros(output_size)

        self.input = None

    def forward(self, input):
        
        self.input = input
        output = np.dot(self.weights, input) + self.bias
        return output

    def backward(self, grad_output, learning_rate):
        grad_weights = np.outer(grad_output, self.input)
        grad_bias = grad_output
        grad_input = np.dot(self.weights.T, grad_output)

        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input