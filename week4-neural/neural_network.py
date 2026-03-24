import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input, output, hidden, lr=0.1):
        self.lr = lr
        self.W1 = np.random.randn(hidden, input) * np.sqrt(2/input) #kaimberg initialization using normal distribution
        self.b1 = np.zeros([hidden,1])
        self.W2 = np.random.randn(output, hidden) * np.sqrt(1 / hidden)
        self.b2 = np.zeros([output, 1])
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z>0).astype(float)
        
    def sigmoid(self,z):
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

    def train(self, X, y):
        for i in range(20000):
            a2, cache = self.forward_pass(X)
            self.backward_pass(X, y, cache)

            if i % 1000 == 0:
                loss = -np.mean(y * np.log(a2 + 1e-7) + (1-y) * np.log(1 - a2 + 1e-7))
                print(f'Iter: {i}, Loss: {loss}')





nn = NeuralNetwork(input=2, hidden=32, output=1)
print(nn.W1.shape)
print(nn.b1.shape)
print(nn.W2.shape)
print(nn.b2.shape)




X = np.array([[0, 0, 1, 1], [0,1,0,1]])
y =np.array([[0,1,1,0]])

nn.train(X,y)
output, _ = nn.forward_pass(X)
predictions = (output > 0.5).astype(int)

print("Output: ")
print(output)

print("binary predictions:")
print(predictions)

accuracy = np.mean(predictions == y)
print("Accuracy (%):", accuracy * 100)

def plot_boundary(nn, X, y):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()].T
    Z, _ = nn.forward_pass(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.6)

    plt.scatter(X[0, :], X[1, :], c=y.flatten(), cmap="coolwarm", edgecolors="k")

    plt.title("Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig("results.png", transparent=False)
    print("Saved!")
    plt.show()

plot_boundary(nn, X, y)