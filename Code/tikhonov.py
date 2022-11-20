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
    seed = 50
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
    seed = 50   
    x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -3, 3, num_train_samples, seed = seed, std_dev = .7) 
    lambdas = np.linspace(0, 20, 1000)
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
    seed = 50
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
    seeds = range(1,101)
    seeds = sorted(list(set(seeds)))
    for seed in seeds:
      x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -3, 3, num_train_samples, seed = seed, std_dev = .7)
      lambdas = np.linspace(0, 20, 1000)
      degree = 11
      weights = finite_diff.generate_centered_D(degree + 1)
      RSS_vals = []
      for l in lambdas:
        tikhonov = estimators.tikhonov(l, degree, weights)
        tikhonov.fit(x_train, y_train)
        RSS_vals += [tikhonov.RSS(x_test, y_test)]
      plt.plot(lambdas, RSS_vals, alpha = .3)

    plt.semilogy()
    plt.title('RSS values vs $\lambda$ for 100 random seeds')
    plt.show()

  #RSS values for various seeds as a single statistical function
  if fig == 5:
    seeds = range(1,101)
    seeds = sorted(list(set(seeds)))

    num_lambdas = 100
    degree = 11
    weights = finite_diff.generate_centered_D(degree + 1)
    lambdas = np.linspace(0, 20, num_lambdas)
    y_evals = np.zeros((len(seeds),num_lambdas))

    for i in range(len(seeds)):
      seed = seeds[i]
      x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -3, 3, num_train_samples, seed = seed, std_dev = .7)
      RSS_vals = []
      for j in range(len(lambdas)):
        l = lambdas[j]
        tikhonov = estimators.tikhonov(l, degree, weights)
        tikhonov.fit(x_train, y_train)
        RSS_val = tikhonov.RSS(x_test,y_test)
        y_evals[i][j] = RSS_val
      
    means = np.mean(y_evals,axis=0)
    stdevs = np.std(y_evals,axis=0)
    
    plt.plot(lambdas, means, color="red", label="Mean")
    plt.fill_between(lambdas, means-stdevs,\
                    means+stdevs,\
                    color="red", alpha=0.25, edgecolor=None, label="Stdev")
    plt.matshow(y_evals)
    plt.legend()
    plt.show()


f = lambda x : np.sin(x) + np.sin(5*x)
#visualize(1, f)
#visualize(2, f)
#visualize(3, f)
#visualize(4, f)
visualize(5, f)
