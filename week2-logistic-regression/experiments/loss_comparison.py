import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("..")

np.random.seed(42)

X0 = np.random.randn(50,2) + [-2,-2]
y0 = np.zeros(50)

X1 = np.random.randn(50,2) + [2,2]
y1 = np.ones(50)

X = np.vstack([X0,X1])
y = np.concatenate([y0,y1])

# LOGISTIC REGRESSION using BINARY CROSS-ENTROPY

class LogisticRegressionBCE:
    def __init__(self, lr=0.1, n_iter = 800):
        self.lr = lr
        self.n = n_iter
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, Z):
        Z = np.clip(Z,-500,500)
        return 1 / (1+np.exp(-Z))
    def fit(self,X,y):
        n_samp, n_feat = X.shape
        self.weights = np.zeros(n_feat)
        self.bias = 0

        for i in range(self.n):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)
            loss = -np.mean(y*np.log(y_pred_clip) + (1-y)*np.log(1-y_pred_clip))
            self.losses.append(loss)

            dw = (1/n_samp) * np.dot(X.T, (y_pred - y))
            db = (1/n_samp) * np.sum(y_pred - y)


            self.weights -= self.lr*dw
            self.bias -= self.lr * db
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int)
    
# LOGISTIC REGRESSION using MEAN SQUARED ERROR (WRONG BECAUSE ITS TOO LENIENT)

class LogisticRegressionMSE:
    def __init__(self, lr = 0.1, n_iter = 800):
        self.lr = lr
        self.n = n_iter
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, z):
        z = np.clip(z, -500,500)
        return 1 / (1+np.exp(-z))
    def fit(self, X, y):
        n_samp, n_feat = X.shape
        self.weights = np.zeros(n_feat)
        self.bias = 0

        for i in range(self.n):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            #MSE loss function
            loss = np.mean((y_pred -y)**2)
            self.losses.append(loss)

            errors = y_pred - y
            sigmoid_derivative = y_pred*(1-y_pred)
            dw = (1/n_samp) * np.dot(X.T,errors * sigmoid_derivative)
            db =( 1/n_samp) * np.sum(errors * sigmoid_derivative)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int)
    
print("Cross entropy loss:")
model1 = LogisticRegressionBCE()
model1.fit(X,y)
y_pred1 = model1.predict(X)
acc1 = np.mean(y_pred1 == y)
print("Final accuracy:", acc1)
print('final loss:', model1.losses[-1])
print(f"Iteratiosn to reach 100% acc: {np.where(np.array([np.mean((model1.predict(X)== y)) for _ in range(100)])== 1.0)[0][0] if acc1 == 1.0 else "N/A"}")

print("MSE Loss: ")
model2 = LogisticRegressionMSE()
model2.fit(X,y)
y_pred2 = model2.predict(X)
acc2 =np.mean(y_pred2 == y)
print("final accuracy: ", acc2)
print("final loss: ", model2.losses[-1])

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(model1.losses, label="Cross entropy (correct)", linewidth=2)
plt.plot(model2.losses, label="Mean squared error (wrong)", linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title("loss comparison: Cross entropy vs MSE")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
iterations = [0,100,200,300,400,500,600,700,800]
plt.bar([i-15 for i in range(len(iterations))], [acc1]*len(iterations), width=30, alpha=0.7, label="Cross entropy")
plt.bar([i+15 for i in range(len(iterations))], [acc2]*len(iterations), width=30, alpha=0.7, label="MSE")
plt.ylim([0,1.1])
plt.xlabel('Metric')
plt.ylabel("Final accuracy")
plt.title("Final accuracy comparisons")
plt.xticks([])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('loss_comparison.png')
print("plot saved as loss_comparison.png")
plt.show()
print("\n"+'='*60)
print("Conclusion\nCross-entropy converges faster and is the correct loss\n for classifications problems. MSE works but is not fully optimal")
print('='*60)