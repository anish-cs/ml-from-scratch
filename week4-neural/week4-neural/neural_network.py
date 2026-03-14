import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input, output, hidden, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(hidden, input) * 0.01
        self.b1 = np.zeros([hidden,1])
        self.W2 = np.random.randn(output, hidden) * 0.01
        self.b2 = np.zeros([output, 1])
    def relu(self, z):
        return max(0, z)
    
    def relu_derivative(self, z):
        if z >0:
            return 1
        else:
            return 0
        
    def sigmoid(self,z):
        return 1/(1 + np.exp(-z))
    def sigmoid_derivative(self, z):
        self.sigmoid(z)*(1-self.sigmoid(z))

    def forward_pass(self, X):
        z1 = self.W1 @ X + self.b1
        a1 = self.relu(z1)
        z2 = self.W2 @ a1 + self.b2
        a2 = self.sigmoid(z2)
        cache = (z1, a1, z2)
        return a2, cache

nn = NeuralNetwork(input=2, hidden=4, output=1)
print(nn.W1.shape)
print(nn.b1.shape)
print(nn.W2.shape)
print(nn.b2.shape)
