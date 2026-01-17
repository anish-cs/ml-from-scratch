import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate = 0.01, n=1000):
        self.lr = learning_rate
        self.n = n
        self.weights = None
        self.bias = None
        self.losses = []
    def predict(self, X):

        return X @ self.weights + self.bias
    

if __name__ == "__main__":
    model = LinearRegression()
    model.weights =np.array([2])
    model.bias = 3.0
    X_test = np.array([[1],[2],[3]])
    print('Test predictions:', model.predict(X_test))