import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng

np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# no feautre added
x = np.arange(0, 20, 1)
y = 1 + x ** 2
X = x.reshape(-1, 1)

model_w, model_b = run_gradient_descent_feng(X, y, iterations=1000, alpha=1e-2)

plt.scatter(x, y, marker='x', c='r', label="Actual Value");
plt.title("no feature engineering")
plt.plot(x, X @ model_w + model_b, label="Predicted Value");
plt.xlabel("X");
plt.ylabel("y");
plt.legend();
plt.show()

# ------------------------------------------------- #

# feature added
x = np.arange(0, 20, 1)
y = 1 + x ** 2  # equal to y = 1 + x^2
X = x ** 2  # engineered feature can add  also: np.c_[x,x**2,x**3]
X = X.reshape(-1, 1)  # X should be a 2-D Matrix
model_w, model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-5)

plt.scatter(x, y, marker='x', c='r', label="Actual Value");
plt.title("Added x**2 feature")
plt.plot(x, np.dot(X, model_w) + model_b, label="Predicted Value");
plt.xlabel("x");
plt.ylabel("y");
plt.legend();
plt.show()

# scaling feature
# create target data
x = np.arange(0,20,1)
X = np.c_[x, x**2, x**3]
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")

# add mean_normalization
X = zscore_normalize_features(X)
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")
x = np.arange(0,20,1)
y = x**2

X = np.c_[x, x**2, x**3]
X = zscore_normalize_features(X)

model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha=1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
