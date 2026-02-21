import numpy as np
from layers.conv_layer import ConvLayer
from layers.pooling import MaxPool
from layers.flatten import Flatten
from layers.dense import Dense
from utils.activations import ReLU
from utils.loss import SoftmaxCrossEntropyLoss


class CNN:
    def __init__(self):
  
        self.conv = ConvLayer(num_filters=8, filter_size=3, input_channels=1)
        self.relu = ReLU()
        self.pool = MaxPool(pool_size=2, stride=2)
        self.flatten = Flatten()
        self.dense = Dense(input_size=8 * 13 * 13, output_size=10)
        self.loss_fn = SoftmaxCrossEntropyLoss()

    def forward(self, x, label=None):
      
        x = self.conv.forward(x)
        x = self.relu.forward(x)
        x = self.pool.forward(x)
        x = self.flatten.forward(x)
        logits = self.dense.forward(x)

        if label is not None:
            loss = self.loss_fn.forward(logits, label)
            return logits,loss

        return logits

    def backward(self, learning_rate):
       
        grad = self.loss_fn.backward()
        grad = self.dense.backward(grad, learning_rate)
        grad = self.flatten.backward(grad)
        grad = self.pool.backward(grad)
        grad = self.relu.backward(grad)

        return grad