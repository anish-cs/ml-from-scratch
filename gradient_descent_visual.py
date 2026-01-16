import numpy as np
import matplotlib.pyplot as plt

#make simple polynomial: f(x) = (x-3)^2
# note: minimum at x = 3
def f(x):
    return (x-3) ** 2

#derivative using power rule = 2(x-3). This is our gradient
def df(x):
    return 2*(x-3)

#gradient descent simulation
x = 0.0
learning_rate = 0.1
# the learning rate is meant to control the steepness of the steps by the gradient
#if learning rate were = to 1.1, the gradient descent diverges
"""
After experimenting with the learning rate, I learned that
higher learning rate makes the descent go faster but if the rate
is too high, the gradient descent diverges and oscillates upward on the curve.
This is because the learning rate makes the next point in the descent overshoot
and skip over the local minimum. 


"""
history = [x]

for i in range(20):
    gradient = df(x)
    x = x - learning_rate*gradient
    #we subtract because the gradient measures the direction of the steepest increase 
    # and we want to decrease so we go the other direction.
    history.append(x)
    print(f"Step {i}: x = {x:.4f}, f(x) = {f(x):.4f}, gradient = {gradient:.4f}")

x_range = np.linspace(-1, 6, 100)
y_range = f(x_range)

plt.figure(figsize=(10,6))
plt.plot(x_range, y_range, 'b-', label='f(x) = (x-3)^2')
plt.plot(history, [f(x) for x in history], 'ro-', label='Gradient descent path')
plt.scatter([3],[0], color='green', s = 200, marker='*', label='Minimum', zorder=5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Visualization')
plt.legend()
plt.grid(True)
plt.savefig('gradient_descent.png')
print('\nPlot saved as gradient_descent.png')
plt.show()