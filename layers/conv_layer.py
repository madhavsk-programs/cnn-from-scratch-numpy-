import numpy as np

class ConvLayer:
    def __init__(self,num_filters,filter_size,input_channels):
        self.num_filters=num_filters
        self.filter_size=filter_size
        self.input_channels=input_channels

        scale=np.sqrt(2.0/(input_channels*filter_size*filter_size))
        self.filters=np.random.randn(
            num_filters,input_channels,filter_size,filter_size
        )*scale

        self.bias = np.zeros(num_filters)

    def forward(self,input):
        self.input = input
        C, H, W = input.shape
        F = self.filter_size
        
        H_out = H - F + 1
        W_out = W - F + 1
        
        output = np.zeros((self.num_filters, H_out, W_out))

        for f in range(self.num_filters):  
            for i in range(H_out):
                for j in range(W_out):
                    region = input[:, i:i+F, j:j+F] 
                    output[f, i, j] = np.sum(region * self.filters[f]) + self.bias[f]

        return output