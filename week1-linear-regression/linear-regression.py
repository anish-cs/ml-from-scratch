import numpy as np
import matplotlib.pyplot as plt
class LinearRegression:
    def __init__(self, learning_rate = 0.1, n=1000):
        self.lr = learning_rate
        self.n = n
        self.weights = None
        self.bias = None
        self.losses = []
    def predict(self, X):
        return X @ self.weights + self.bias
    def fit (self, X, y):
        self.bias = 0
        n_samp, n_feat = X.shape
        self.weights = np.zeros(n_feat)
        for i in range(self.n):
            y_pred = self.predict(X)
            loss = np.mean((y_pred-y)**2)
            self.losses.append(loss)

            dw = (2/n_samp) * X.T @ (y_pred - y) #take partial derivatives of loss function
            db = (2/n_samp) * np.sum(y_pred - y) 

            self.weights += self.lr * dw #negative so it converges.

            self.bias -= self.lr * db
        
    def find_R2(self, w, b):   # R2 = (SSM - SSF)/SSM; trying to implement p value soon for better accuracy
        y_fit = X @ self.weights + self.bias
        SSF = np.sum((y-y_fit)**2)
        SSM = np.sum((y-np.mean(y))**2)
        R2 = 1 - SSF/SSM
        return R2 * 100



if __name__ == "__main__":
    X = np.array([[1],[2],[3],[4]], dtype=float)
    y = 2 * X.flatten() + 3

    model = LinearRegression()
    model.fit(X,y)
    print("Learned weight:", model.weights)
    print("Learned bias:", model.bias)
    print("R square value", model.find_R2(model.weights, model.bias))
    print(model.losses[-1])
   

    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))
    ax1.scatter(X,y,color='blue', label='Data Points')
    ax1.plot(X_line, y_line, color='red', label='Regression line')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title("Linear Regression fit")
    ax1.legend()

    ax2.plot(model.losses)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('MSE')
    ax2.set_title("Loss over time")
    plt.tight_layout()
    plt.show()