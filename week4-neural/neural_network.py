import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input, output, hidden, lr=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.lr = lr
        self.W1 = np.random.randn(hidden, input) * np.sqrt(2 / input) #Kaiming He initialization using normal distribution
        self.b1 = np.zeros((hidden, 1))
        self.W2 = np.random.randn(output, hidden) * np.sqrt(1 / hidden)
        self.b2 = np.zeros((output, 1))
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z>0).astype(float)
        
    def sigmoid(self,z):
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))

    def forward_pass(self, X):
        z1 = self.W1 @ X + self.b1
        a1 = self.relu(z1)
        z2 = self.W2 @ a1 + self.b2
        a2 = self.sigmoid(z2)
        cache = (z1, a1, z2, a2)
        return a2, cache
    def backward_pass(self, X, y, cache):
        m = X.shape[1] # num of smaples
        z1, a1, z2, a2 = cache #take foward pass values
        dz2 = a2 - y
        dW2 = (1/m) * dz2 @ a1.T
        db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = (self.W2.T @ dz2) * self.relu_derivative(z1)

        dW1 = (1/m) * dz1 @ X.T
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2 
        self.b2 -= self.lr * db2

    def compute_loss(self, y, a2):
        return -np.mean(y * np.log(a2 + 1e-7) + (1-y) * np.log(1 - a2 + 1e-7))
    def train(self, X, y, epochs=20000, verbose=True):
        losses = []
        for i in range(epochs):
            a2, cache = self.forward_pass(X)
            self.backward_pass(X, y, cache)

            if i % 1000 == 0:
                loss = self.compute_loss(y, a2)
                losses.append(loss)
                if verbose:
                    
                    print(f'Iter: {i}, Loss: {loss}')
        return losses
    
    def predict(self, X):
        a2, _ = self.forward_pass(X)
        return (a2 > 0.5).astype(int)
    def predict_prob(self, X):
        a2, _ = self.forward_pass(X)
        return a2



