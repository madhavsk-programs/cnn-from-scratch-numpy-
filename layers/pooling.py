import numpy as np

class MaxPool:
    def __init__(self, pool_size=2, stride=2):

        self.pool_size = pool_size
        self.stride = stride
        self.input = None

    def forward(self, input):
        
        self.input = input
        C, H, W = input.shape
        P = self.pool_size
        S = self.stride

        H_out = (H - P) // S + 1
        W_out = (W - P) // S + 1

        output = np.zeros((C, H_out, W_out))

        for c in range(C):  
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * S
                    h_end = h_start + P
                    w_start = j * S
                    w_end = w_start + P

                    region = input[c, h_start:h_end, w_start:w_end]
                    output[c, i, j] = np.max(region)

        return output

    def backward(self, grad_output):
        
        C, H, W = self.input.shape
        P = self.pool_size
        S = self.stride

        H_out = grad_output.shape[1]
        W_out = grad_output.shape[2]

        grad_input = np.zeros_like(self.input)

        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * S
                    h_end = h_start + P
                    w_start = j * S
                    w_end = w_start + P

                    region = self.input[c, h_start:h_end, w_start:w_end]
                    max_value = np.max(region)

                    for x in range(P):
                        for y in range(P):
                            if region[x, y] == max_value:
                                grad_input[c, h_start + x, w_start + y] += grad_output[c, i, j]

        return grad_input