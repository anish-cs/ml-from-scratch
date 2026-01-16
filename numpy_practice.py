import numpy as np

# ===== CREATING ARRAYS =====
# From Python list
a = np.array([1, 2, 3])
print("a:", a)
print("Shape:", a.shape)  # (3,) means 1D array with 3 elements

# 2D array (matrix)
A = np.array([[1, 2], [3, 4]])
print("A:\n", A)
print("Shape:", A.shape)  # (2, 2) means 2 rows, 2 columns

# Special arrays
zeros = np.zeros(5)  # [0, 0, 0, 0, 0]
ones = np.ones((2, 3))  # 2x3 matrix of ones
print("Zeros:", zeros)
print("Ones:\n", ones)

# Random arrays (important for ML!)
random = np.random.randn(3, 2)  # 3x2 matrix, random values from normal distribution
print("Random:\n", random)

# ===== BASIC OPERATIONS =====
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("a + b:", a + b)  # Element-wise: [5, 7, 9]
print("a * b:", a * b)  # Element-wise: [4, 10, 18] NOT dot product!
print("a + 10:", a + 10)  # Broadcasting: [11, 12, 13]
print("a ** 2:", a ** 2)  # Square each: [1, 4, 9]

# ===== MATRIX OPERATIONS =====
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication (THIS IS KEY FOR ML!)
print("Matrix multiply:\n", np.dot(A, B))
# Or use @ operator:
print("Same thing:\n", A @ B)

# Element-wise multiply (different!)
print("Element-wise multiply:\n", A * B)

# Transpose
print("A transpose:\n", A.T)

# ===== DOT PRODUCT =====
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print("Dot product:", np.dot(x, y))  # 1×4 + 2×5 + 3×6 = 32

# ===== USEFUL FUNCTIONS =====
data = np.array([1, 2, 3, 4, 5])
print("Mean:", np.mean(data))  # 3.0
print("Sum:", np.sum(data))  # 15
print("Max:", np.max(data))  # 5
print("Min:", np.min(data))  # 1
print("Std dev:", np.std(data))  # Standard deviation

# ===== INDEXING & SLICING =====
arr = np.array([10, 20, 30, 40, 50])
print("First element:", arr[0])  # 10
print("Last element:", arr[-1])  # 50
print("Slice [1:4]:", arr[1:4])  # [20, 30, 40]

# 2D indexing
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Element at [0,0]:", matrix[0, 0])  # 1
print("Element at [1,2]:", matrix[1, 2])  # 6
print("First row:", matrix[0, :])  # [1, 2, 3]
print("Second column:", matrix[:, 1])  # [2, 5, 8]

# ===== RESHAPING =====
flat = np.array([1, 2, 3, 4, 5, 6])
reshaped = flat.reshape(2, 3)  # 2 rows, 3 columns
print("Reshaped:\n", reshaped)

# ===== BROADCASTING (Advanced but important) =====
# NumPy automatically "stretches" arrays to match shapes
A = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])

# Add b to each row of A
print("A + b:\n", A + b)
# Result:
# [[11, 22, 33],
#  [14, 25, 36]]

# ===== PRACTICE PROBLEM =====
# Recreate the linear regression prediction by hand:
X = np.array([[1], [2], [3]])  # 3 samples, 1 feature
weights = np.array([2.0])  # 1 weight
bias = 3.0

# Prediction: y = X @ weights + bias
predictions = np.dot(X, weights) + bias
print("Predictions:", predictions)  # Should be [5, 7, 9]