import numpy as np
import matplotlib.pyplot as plt
import estimators
import finite_diff
from sample import random_sample_equi

## Generating training / testing data

num_train_samples = 60
num_test_samples = 20
seed = 50

def visualize(fig, func):
    
  # visualize tikhonov estimator
  if fig == 1:
    x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -3, 3, num_train_samples, seed = seed, std_dev = .7)
    xeval = np.linspace(-3,3,1000)
    feval = func(xeval)
    degree = 13
    weights = finite_diff.generate_centered_D(degree + 1)
    lam = .1
    tikhonov = estimators.tikhonov(lam, degree, weights)
    tikhonov.fit(x_train, y_train)
    coefs = tikhonov.xstar
    b_hat = tikhonov.predict(x_test) 
    poly = tikhonov.predict(xeval)
    plt.plot(xeval, poly, color = 'blue', label = '$x^*$') 
    #plt.plot(x_test, b_hat, 'xr', label = 'tikhonov')
    plt.plot(x_train, y_train, '.', label = 'train data')
    plt.plot(xeval, feval, label = 'f(x)')
    plt.legend()
    plt.show()
  
  #RSS Values vs lambda for a specific seed
  if fig == 2:   
    x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -3, 3, num_train_samples, seed = seed, std_dev = .7) 
    lambdas = np.linspace(0, 20, 100)
    degree = 11
    weights = finite_diff.generate_centered_D(degree + 1)
    RSS_vals = []
    for l in lambdas:
      tikhonov = estimators.tikhonov(l, degree, weights)
      tikhonov.fit(x_train, y_train)
      RSS_vals += [tikhonov.RSS(x_test, y_test)]
    plt.plot(lambdas, RSS_vals)
    plt.title('RSS values for $\lambda \in [0, 20]$')
    plt.show()  

  #RSS values vs degree for specific seed
  if fig == 3:
    x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -3, 3, num_train_samples, seed = seed, std_dev = .7)
    degrees = range(3, 20)
    lam = .1
    RSS_vals = []
    for d in degrees:
      weights = finite_diff.generate_centered_D(d + 1)
      tikhonov = estimators.tikhonov(lam, d, weights)
      tikhonov.fit(x_train, y_train)
      RSS_vals += [tikhonov.RSS(x_test, y_test)]
    plt.plot(degrees, RSS_vals)
    plt.title('RSS vs Degree of $x$')
    plt.show()

  #RSS Values for various seeds
  if fig == 4:
    seeds = np.random.randint(0, 1000, 100)
    seeds = sorted(list(set(seeds)))
    for seed in seeds:
      x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -3, 3, num_train_samples, seed = seed, std_dev = .7)
      lambdas = np.linspace(0, 20, 100)
      degree = 11
      weights = finite_diff.generate_centered_D(degree + 1)
      RSS_vals = []
      for l in lambdas:
        tikhonov = estimators.tikhonov(l, degree, weights)
        tikhonov.fit(x_train, y_train)
        RSS_vals += [tikhonov.RSS(x_test, y_test)]
      plt.plot(lambdas, RSS_vals, alpha = .3)

    plt.title('RSS values vs $\lambda$ for 100 random seeds')
    plt.show()

f = lambda x : np.sin(x) + np.sin(5*x)
visualize(4, f)
