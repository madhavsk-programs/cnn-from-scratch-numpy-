import numpy as np

class SoftmaxCrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.labels = None

    def softmax(self, logits):
    
        exp_shifted = np.exp(logits - np.max(logits))
        return exp_shifted / np.sum(exp_shifted)

    def forward(self, logits, label):

        self.labels = label
        self.probs = self.softmax(logits)

        loss = -np.log(self.probs[label] + 1e-9)
        return loss

    def backward(self):
        
        grad = self.probs.copy()
        grad[self.labels] -= 1
        return grad