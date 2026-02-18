import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from logistic_regression import LogisticRegression

np.random.seed(42)
X0 = np.random.randn(50,2) + [-2,-2]
X1 = np.random.randn(50, 2) + [2,2]
X = np.vstack([X0, X1])
y = np.concatenate([np.zeros(50), np.ones(50)])

print("\nExperiment: learning rate sensitivity\n")

learning_rates = [0.1,0.5,1,5,10]
results = []

for lr in learning_rates:
    print(f"testing lr = {lr}")
    model = LogisticRegression(lr = lr, n = 800)

    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X,y)
    y_pred = model.predict(X)
    acc = np.mean(y_pred == y)
    final_loss = model.losses[-1]

    converge_iter = None
    for i in range(0,800,10):
        pass
    results.append({
        "lr": lr,
        "accuracy": acc,
        "final_loss": final_loss,
        "losses": model.losses
    })

    print(f"accuracy: {acc}")
    print(f'final loss: {final_loss}')
    
    if acc < 0.95:
        print("failed to converge")
    elif acc == 1.0 and final_loss <0.01:
        print("great convergence")
    else:
        print("converges")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for r in results:
    label = f"lr={r['lr']}"
    if r['accuracy'] < 0.95:
        label += "FAILED"
    plt.plot(r['losses'], label=label, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Cross-Entropy Loss')
plt.title('Loss Curves for Different Learning Rates')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale helps see differences

plt.subplot(1, 2, 2)
lrs_plot = [r['lr'] for r in results]
accs = [r['accuracy'] for r in results]
colors = ['green' if a == 1.0 else 'orange' if a > 0.9 else 'red' for a in accs]
plt.bar(range(len(lrs_plot)), accs, color=colors, alpha=0.7)
plt.xticks(range(len(lrs_plot)), [f"{lr}" for lr in lrs_plot])
plt.xlabel('Learning Rate')
plt.ylabel('Final Accuracy')
plt.title('Accuracy vs Learning Rate')
plt.ylim([0, 1.1])
plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('learning_rate_comparison.png')
print("\nPlot saved as learning_rate_comparison.png")
plt.show()

print("\n" + "=" * 60)
print("CONCLUSION:")
print(f"Optimal learning rate for this problem: {[r['lr'] for r in results if r['accuracy']==1.0 and r['final_loss']<0.01][0] if any(r['accuracy']==1.0 and r['final_loss']<0.01 for r in results) else 'N/A'}")
print("=" * 60)