import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    
    def __init__(self, lr = 0.1, n = 800):
        self.lr = lr
        self.n_iter = n
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, Z):
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))
    
    def predict_probability(self, X):
        Z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(Z)
    
    def predict(self, X):
        return (self.predict_probability(X) >= 0.5).astype(int)
    def fit(self, X, y):
        n_samp, n_feat = X.shape
        self.weights = np.zeros(n_feat)
        self.bias = 0
        for i in range(self.n_iter):
            y_pred = self.predict_probability(X)
            loss = -(1/n_samp) * np.sum(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
            self.losses.append(loss)
            dw = (1/n_samp)* np.dot(X.T, (y_pred - y))
            db = (1/n_samp)*np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            if i % 100 == 0:
                acc = np.mean((y_pred >= 0.5) == y)
                print(f'Iteration {i}: Loss = {loss}, Acc={acc}')
if __name__ == "__main__":
    np.random.seed(42)

    X0 = np.random.randn(50, 2) + [-2, -2]
    y0 = np.zeros(50)

    X1 = np.random.randn(50, 2) + [2,2]
    y1 = np.ones(50)

    X = np.vstack([X0,X1])
    y = np.concatenate([y0, y1])

    model = LogisticRegression()
    model.fit(X,y)

    y_pred = model.predict(X)
    print(f"\nfinal accuracy: {np.mean(y_pred == y)}")