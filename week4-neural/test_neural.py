import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

def accuracy(nn, X, y):
    out, _ = nn.forward_pass(X)
    preds = (out > 0.5).astype(int)
    return np.mean(preds == y) * 100

def make_circles(n, noise, seed):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0,2 * np.pi, n)
    r = np.concatenate([np.ones(n//2) * 0.5, np.ones(n//2)])
    labels = np.concatenate([np.zeros(n//2), np.ones(n//2)])
    X =np.vstack([r * np.cos(angles) + rng.normal(0, noise, n), r * np.sin(angles) + rng.normal(0, noise, n)])
    return X, labels.reshape(1,-1)

def make_moons(n, noise, seed):
    rng = np.random.default_rng(seed)
    half = n // 2
    t = np.linspace(0, np.pi, half)
    X0 = np.vstack([np.cos(t), np.sin(t)])
    X1 = np.vstack([1-np.cos(t), 1 - np.sin(t) - 0.5])
    X = np.hstack([X0, X1]) + rng.normal(0, noise, (2,n))
    y = np.hstack([np.zeros(half), np.ones(half)]).reshape(1, -1)
    return X,y

def test_xor():
    print("TEST 1 - XOR")
    X = np.array([[0,0,1,1],
                 [0,1,0,1]])
    y = np.array([0,1,1,0])

    passed = 0
    for seed in range(10):
        np.random.seed(seed)
        nn = NeuralNetwork(input=2, hidden = 4, output = 1, lr = 0.1)
        nn.train(X,y)
        acc = accuracy(nn, X, y)
        ok = acc == 100.0
        passed += ok
    print(f"solved xor: {passed}/10 seeds")
    assert passed >= 7, "xor should solve on most seeds"
    print("passed")

def test_circles():
    print("TEST 2 - concentric circles")
    X, y = make_circles(n=300, noise=0.08,seed=0)
    np.random.seed(42)
    nn = NeuralNetwork(input=2,hidden=32,output=1,lr=0.1)
    nn.train(X,y)
    acc = accuracy(nn, X, y)
    print(f"accuracy: {acc}")
    assert acc > 85, f"expected > 85% on circles, got {acc}"
    print("passed")
    return nn, X, y

def test_moon():
    print("Test 3 moons")
    X, y = make_moons(n=300, noise=0.15, seed= 0)
    np.random.seed(42)
    nn= NeuralNetwork(input=2, hidden=16, output=1, lr=0.1)
    nn.train(X, y)
    acc = accuracy(nn, X, y)
    print(f'accuracy: {acc}')
    assert acc > 85, f"expected 85 on moons, got {acc}"
    print('passed')
    return nn, X, y
def test_hidden_size():
    print("test 4 hidden size")
    X = np.array([[0,0,1,1],
                  [0,1,0,1]])
    y = np.array([0,1,1,0])
    sizes = np.linspace(2,8,6, dtype=int).tolist()
    results = {}
    for h in sizes:
        accs = []
        for seed in range(5):
            np.random.seed(seed)
            nn = NeuralNetwork(input=2, hidden=h, output=1, lr=0.1)
            nn.train(X,y)
            accs.append(accuracy(nn, X, y))

        avg = np.mean(accs)
        results[h] = avg
        print(f"hidden={h:>3} avg acc = {avg}%")
    plt.figure(figsize=(7,4))
    plt.plot(sizes, list(results.values()), marker="o")
    plt.xlabel("hidden units")
    plt.ylabel("accuracy")
    plt.title("hidden vs acc")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("exp_hidden_size.png")
    print("saved!")

def test_lr():
    print('test 5 learnign rate')
    X, y = make_circles(n=300, noise=0.08, seed=0)
    lrs = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    results = {}
    for lr in lrs:
        accs = []
        for seed in range(3):
            np.random.seed(seed)
            nn = NeuralNetwork(input=2, hidden=32, output=1, lr=lr)
            nn.train(X,y)
            accs.append(accuracy(nn, X, y))
        avg = np.mean(accs)
        results[lr] = avg
        print(f"lr={lr:.3f} avg acc = {avg}%")

    plt.figure(figsize=(7,4))
    plt.semilogx(lrs, list(results.values()), marker ="o")
    plt.xlabel("lr ")
    plt.ylabel("accuracy")
    plt.title("lr vs acc")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("exp_lr.png")
    print("saved!")

def test_overfitting():
    print("Test 6")
    
    X = np.array([[0,0,1,1],
                 [0,1,0,1]], dtype=float)
    rng = np.random.default_rng(7)
    y = rng.integers(0,2, (1,4)).astype(float)
    np.random.seed(0)
    nn = NeuralNetwork(input=2, hidden=64, output=1, lr =0.1)
    nn.train(X,y)
    acc = accuracy(nn, X, y)
    print(f" accuracy: {acc}")
    assert acc == 100, "should memorize 4 smaples"
    print("passed")

def plot_boundary(nn, X, y, title, fname):
    x_min, x_max = X[0].min() - 0.3, X[0].max() + 0.3
    y_min, y_max = X[1].min() - 0.3, X[1].max() + 0.3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()].T
    Z, _ = nn.forward_pass(grid)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.6)
    plt.scatter(X[0], X[1], c=y.flatten(), cmap="coolwarm", edgecolors="k", s=20)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"  Saved {fname}")
 
# ── runner ────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    test_xor()
    nn_c, Xc, yc = test_circles()
    nn_m, Xm, ym = test_moon()
    test_hidden_size()
    test_lr()
    test_overfitting()
 
    print("=" * 50)
    print("TEST 7 — Decision boundary plots")
    plot_boundary(nn_c, Xc, yc, "Circles boundary", "boundary_circles.png")
    plot_boundary(nn_m, Xm, ym, "Moons boundary",   "boundary_moons.png")
    print()
 
    print("=" * 50)
    print("ALL TESTS PASSED ✓")
 
