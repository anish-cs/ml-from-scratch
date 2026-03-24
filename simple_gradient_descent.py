import numpy as np
import matplotlib.pyplot as plt
#store data to plot gradient descent
w_history = []
loss_history = []
# exmaple data
x = np.array([1,2,3,4,5,6,7])
y = np.array([5,4,6,9,8,10,12])

#initial weights and bias and rate
w = 0.0
b = 0.0
lr = 0.001



#Train for 10 iterations
for i in range(1000):
    #Forward pass
    y_pred = w*x +b

    #loss
    loss = np.mean((y_pred - y) **2)
    #gradients
    dw = (2/len(x)) * np.sum((y_pred - y) * x)
    db = (2/len(x)) * np.sum(y_pred-y)

    w = w - lr * dw
    b = b - lr*db
    w_history.append(w)

    loss_history.append(loss)

    print(f"Iter {i}: w ={w:.3f}, b ={b:.3f}, loss={loss:.3f}")

def find_R2(w,b):
    y_fit = w*x+b
    SSF = np.sum((y-y_fit)**2)
    SSM = np.sum((y-np.mean(y))**2)
    R2 = 1 - SSF/SSM
    return R2 *100


print(f"\nFinal: w={w:.3f} (true: 2.0), b={b:.3f} (true = 0.0)")
print(f"R^2 = {round(find_R2(w,b), 3)}")
plt.plot(loss_history)
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("MSE Loss")
plt.title("Learning Rate Range Test")
plt.show()