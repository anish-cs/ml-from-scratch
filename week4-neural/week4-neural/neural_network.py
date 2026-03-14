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
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z>0).astype(float)
        
    def sigmoid(self,z):
        return 1/(1 + np.exp(-z))
    def sigmoid_derivative(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

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



X = np.array([[0.5, 0.3], [0.2, 0.8]])
print("input")
print(X.shape)
try:
    output,cache = nn.forward_pass(X)
    print(output.shape)
    print(output)
    print("works")
except Exception as e:
    print(f"error: {e}")