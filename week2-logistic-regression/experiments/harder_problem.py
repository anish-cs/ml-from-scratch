import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from logistic_regression import LogisticRegression

print("\nExperiment: easy vs hard classification\n")

# EASY PROBLEM
np.random.seed(42)
X0_easy = np.random.randn(50,2) + [-2,-2]
X1_easy = np.random.randn(50,2) + [2,2]
X_easy = np.vstack([X0_easy, X1_easy])
y_easy = np.concatenate([np.zeros(50), np.ones(50)])

model_easy = LogisticRegression(lr = 0.1, n = 800)
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    model_easy.fit(X_easy, y_easy)

acc_easy = np.mean(model_easy.predict(X_easy) == y_easy)
print("Classes centered at: (-2,-2) and (2,2)")
print("distance between centers is sqrt(32)")
print(f"final acc: {acc_easy}")
print(f'final loss: {model_easy.losses[-1]}')

# HARD PROBLEM
print("\nHard Problem with Overlapping classes\n")
np.random.seed(42)
X0_hard = np.random.randn(50,2) + [-0.5,-0.5]
X1_hard = np.random.randn(50, 2)+ [0.5, 0.5]
X_hard = np.vstack([X0_hard, X1_hard])
y_hard = np.concatenate([np.zeros(50), np.ones(50)])

model_hard = LogisticRegression(lr=0.1, n=800)
with contextlib.redirect_stdout(io.StringIO()):
    model_hard.fit(X_hard, y_hard)

acc_hard = np.mean(model_hard.predict(X_hard) == y_hard)
print("classes now centered at: (-0.5, -0.5) and (0.5,0.5)")
print("distance between centers is sqrt2 units")
print(f"final accuracy: {acc_hard}")
print(f"final loss: {model_hard.losses[-1]}")

ig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Easy problem - scatter
ax = axes[0, 0]
ax.scatter(X0_easy[:, 0], X0_easy[:, 1], c='red', marker='o', 
           label='Class 0', edgecolors='k', alpha=0.6)
ax.scatter(X1_easy[:, 0], X1_easy[:, 1], c='blue', marker='s', 
           label='Class 1', edgecolors='k', alpha=0.6)
ax.set_title('Easy Problem: Well-Separated Classes')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()
ax.grid(True, alpha=0.3)

# Easy problem - loss curve
ax = axes[0, 1]
ax.plot(model_easy.losses, linewidth=2, color='green')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Easy Problem Loss (Final Acc: {acc_easy:.2%})')
ax.grid(True, alpha=0.3)

# Hard problem - scatter
ax = axes[1, 0]
ax.scatter(X0_hard[:, 0], X0_hard[:, 1], c='red', marker='o', 
           label='Class 0', edgecolors='k', alpha=0.6)
ax.scatter(X1_hard[:, 0], X1_hard[:, 1], c='blue', marker='s', 
           label='Class 1', edgecolors='k', alpha=0.6)
ax.set_title('Hard Problem: Overlapping Classes')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()
ax.grid(True, alpha=0.3)

# Hard problem - loss curve
ax = axes[1, 1]
ax.plot(model_hard.losses, linewidth=2, color='orange')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'Hard Problem Loss (Final Acc: {acc_hard:.2%})')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('easy_vs_hard.png')
print("\nPlot saved as easy_vs_hard.png")
plt.show()

print("\n" + "=" * 60)
print("CONCLUSION:")
print(f"Easy problem (well-separated): {acc_easy:.2%} accuracy")
print(f"Hard problem (overlapping): {acc_hard:.2%} accuracy")
print("\nWhen classes overlap, linear models cannot achieve 100% accuracy.")
print("This is a fundamental limitation, would need nonlinear models (Week 4)")
print("=" * 60)