import numpy as np

w = np.array([1.0, 2.5, -3.4])  # create an array where 1.0 is x[0] 2.5 is x[1] and so on...
b = 4
x = np.array([10, 20, 30])
# model prediction | Without Vectorization
f = 0
for j in range(len(w)):
    f = f + w[j] * x[j]  # model prediction
f = f + b
# model prediction | Vectorization uses Numpy | Much Faster and shorter
# uses parallel hardware therefore runs much faster than the for loop

f = np.dot(w, x) + b # dot: 1.0 * 10 + 2.5*20 ... w[n]*x[n2] + b
