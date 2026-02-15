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

def plot_decision_bound(model, X, y):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    plt.scatter(X[y==0, 0], X[y==0, 1], c="red", marker = 'o', label="Class 0", edgecolors='k')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='s', label='class 1', edgecolors='k')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.title('logistic regression decision boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("decision_boundary.png")
    plt.show()
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
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plot_decision_bound(model, X,y)
    plt.subplot(1,2,2)
    plt.plot(model.losses)
    plt.xlabel('Iteration')
    plt.ylabel('Cross-entropy loss')
    plt.title("Training loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("results.png")
    plt.show()