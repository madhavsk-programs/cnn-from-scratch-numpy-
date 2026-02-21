import numpy as np
from models.cnn import CNN

sample_image = np.random.randn(1, 28, 28)
sample_label = 3 

model = CNN()

logits, loss = model.forward(sample_image, sample_label)

print("Logits shape:", logits.shape)
print("Loss:", loss)
model.backward(learning_rate=0.001)
