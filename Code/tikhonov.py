import numpy as np
import matplotlib.pyplot as plt
import estimators
import finite_diff
from sample import random_sample_equi

## Generating training / testing data

num_train_samples = 150
num_test_samples = 20
seed = 50

def visualize(fig, func):
  x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -4, 4, num_train_samples, seed = seed)
    
  if fig == 1:
    xeval = np.linspace(-4,4,1000)
    feval = func(xeval)
    degree = 17
    weights = finite_diff.generate_centered_D(degree + 1)
    lam = .1
    tikhonov = estimators.tikhonov(lam, degree, weights)
    tikhonov.fit(x_train, y_train)
    coefs = tikhonov.xstar
    b_hat = tikhonov.predict(x_test) 
    poly = tikhonov.predict(xeval)
    plt.plot(xeval, poly, color = 'blue', label = '$x^*$') 
    plt.plot(x_test, b_hat, 'xr', label = 'tikhonov')
    plt.plot(x_train, y_train, '.', label = 'train data')
    plt.plot(xeval, feval, label = 'f(x)')
    plt.legend()
    plt.show()
    RSS = tikhonov.RSS(x_test, y_test)
    print(RSS)
f = lambda x : np.sin(x) + np.sin(5*x)
visualize(1, f)
