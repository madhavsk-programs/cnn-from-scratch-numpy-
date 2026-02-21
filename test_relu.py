import numpy as np
from utils.activations import ReLU

conv_output = np.random.randn(8, 26, 26)

relu = ReLU()
activated_output = relu.forward(conv_output)

print("Before ReLU (sample):", conv_output[0, 0, :5])
print("After ReLU (sample):", activated_output[0, 0, :5])
print("Shape:", activated_output.shape)