import numpy as np
import matplotlib.pyplot as plt
from plt_logistic_loss import plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
from plt_logistic_loss import soup_bowl, plt_logistic_squared_error

plt.style.use('deeplearning.mplstyle')

soup_bowl();  # good for linear regression but.

x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)
plt_simple_example(x_train, y_train)
plt.close('all')
plt_logistic_squared_error( # using the normal "logistic" squared  it doesn't have "real" min point
   x_train,
    y_train
)
plt.show()

plt_two_logistic_loss_curves()
plt.close('all')
cst = plt_logistic_cost(x_train,y_train) # this i way better to gradient descent it does not have local minima

