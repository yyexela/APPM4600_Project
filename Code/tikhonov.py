import numpy as np
import matplotlib.pyplot as plt
import estimators
import finite_diff
from sample import random_sample_equi

## Generating training / testing data

num_train_samples = 60
num_test_samples = 20
seed = 50

color1 = '#FF595E'
color2 = '#1982C4'
color3 = '#6A4C93'
fname = '$sin(x) + sin(5x)$'

def visualize(fig, func, func_name):
    
  # visualize tikhonov estimator
  if fig == 1:
    for degree in range(3,20):
      seed = 4596
      x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -3, 3, num_train_samples, seed = seed, std_dev = .7)
      xeval = np.linspace(-3,3,1000)
      feval = func(xeval)
      #degree = 13
      weights = finite_diff.generate_centered_D(degree + 1)
      lam = .1
      tikhonov = estimators.tikhonov(lam, degree, weights)
      tikhonov.fit(x_train, y_train)
      coefs = tikhonov.xstar
      b_hat = tikhonov.predict(x_test) 
      poly = tikhonov.predict(xeval)
      plt.title(f"Degree {degree}")
      plt.plot(xeval, poly, label = 'Tikhonov Polynomial', color = color2) 
      plt.plot(x_train, y_train, '.', label = 'Training data', color = color3)
      plt.plot(x_test, y_test, '.', label = 'Testing data', color = 'hotpink')
      plt.plot(xeval, feval, label = 'f(x) = ' + func_name, color = color1)
      plt.xlabel('x')
      plt.ylabel('y')
      plt.legend()
      plt.show()
      plt.close()
  
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
    plt.plot(lambdas, RSS_vals, color = color2)
    plt.title('RSS values for $\lambda \in [0, 20]$')
    plt.xlabel('$\lambda$')
    plt.ylabel('RSS')
    plt.show()  

  #RSS values vs degree for specific seed and lambda
  if fig == 3:
    seed = 13
    x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -3, 3, num_train_samples, seed = seed, std_dev = .7)
    degrees = range(3, 20)
    lam = .1
    RSS_vals = []
    for d in degrees:
      weights = finite_diff.generate_centered_D(d + 1)
      tikhonov = estimators.tikhonov(lam, d, weights)
      tikhonov.fit(x_train, y_train)
      RSS_vals += [tikhonov.RSS(x_test, y_test)]
    plt.plot(degrees, RSS_vals, color = color2)
    plt.title(f'RSS vs Degree of $\\beta$ (seed = {seed})')
    plt.xlabel('Degree')
    plt.ylabel('RSS')
    plt.show()
    plt.close()

  #RSS Values for various seeds
  if fig == 4:
    seeds = range(1,101)
    seeds = sorted(list(set(seeds))) 
    best_lambdas = []
    for seed in seeds:
      x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -3, 3, num_train_samples, seed = seed, std_dev = .7)
      lambdas = np.linspace(0, 20, 1000)
      degree = 15
      weights = finite_diff.generate_centered_D(degree + 1)
      RSS_vals = []
      minRSS = float('inf')
      minlam = 0
      for l in lambdas:
        tikhonov = estimators.tikhonov(l, degree, weights)
        tikhonov.fit(x_train, y_train)
        RSS = tikhonov.RSS(x_test, y_test)
        RSS_vals += [RSS]
        if RSS < minRSS:
          minRSS = RSS
          minlam = l
      best_lambdas += [minlam]
      plt.plot(lambdas, RSS_vals, alpha = .3)

    plt.semilogy()
    plt.xlabel('$\lambda$')
    plt.ylabel('RSS')
    plt.title('RSS values vs $\lambda$ for 100 seeds')
    plt.show()
    nonzero = np.count_nonzero(best_lambdas)
    zero = len(best_lambdas) - nonzero
    print(f'Number of seeds where 0 is the best lambda: {zero} \n Number of seeds where best lambda is nonzero: {nonzero}')

  #RSS values for various seeds as a single statistical function
  if fig == 5:
    seeds = range(1,101)
    seeds = sorted(list(set(seeds)))

    num_lambdas = 100
    degree = 11
    weights = finite_diff.generate_centered_D(degree + 1)
    lambdas = np.linspace(0, 20, num_lambdas)
    RSS_vals = np.zeros((len(seeds),num_lambdas))

    for i in range(len(seeds)):
      seed = seeds[i]
      x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -3, 3, num_train_samples, seed = seed, std_dev = .7)
      RSS_vals = []
      for j in range(len(lambdas)):
        l = lambdas[j]
        tikhonov = estimators.tikhonov(l, degree, weights)
        tikhonov.fit(x_train, y_train)
        RSS_val = tikhonov.RSS(x_test,y_test)
        RSS_vals[i][j] = RSS_val
      
    means = np.mean(RSS_vals,axis=0)
    stdevs = np.std(RSS_vals,axis=0)
    
    plt.plot(lambdas, means, color="red", label="Mean")
    plt.fill_between(lambdas, means-stdevs,\
                    means+stdevs,\
                    color="red", alpha=0.25, edgecolor=None, label="Stdev")
    plt.legend()
    plt.show()
    plt.close()

  #RSS values vs degree for specific seed and lambda
  if fig == 6:
    num_seeds = 100
    seeds = list(range(0,num_seeds))
    degrees = list(range(3, 20))
    num_degrees = len(degrees)

    RSS_vals = np.zeros((num_seeds,num_degrees))

    for i in range(num_seeds):
      seed = seeds[i]
      x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -3, 3, num_train_samples, seed = seed, std_dev = .7)
      lam = .1
      RSS_val = []
      for d in degrees:
        weights = finite_diff.generate_centered_D(d + 1)
        tikhonov = estimators.tikhonov(lam, d, weights)
        tikhonov.fit(x_train, y_train)
        RSS_val += [tikhonov.RSS(x_test, y_test)]
      RSS_vals[i,:] = RSS_val
    
    means = np.mean(RSS_vals,axis=0)
    stdevs = np.std(RSS_vals,axis=0)
    
    plt.plot(degrees, means, color="red", label="Mean")
    plt.fill_between(degrees, means-stdevs,\
                    means+stdevs,\
                    color="red", alpha=0.25, edgecolor=None, label="Stdev")
    plt.ylim(0,8000)
    plt.legend()
    plt.show()
    plt.close()

    plt.matshow(RSS_vals)
    plt.show()
    plt.close()

    for i in range(num_seeds):
      plt.plot(degrees, RSS_vals[i])
    plt.show()
    plt.close()



f = lambda x : np.sin(x) + np.sin(5*x)
visualize(1, f, fname)
#visualize(2, f, fname)
#visualize(3, f, fname)
#visualize(4, f, fname)
#visualize(5, f, fname)
#visualize(6, f, fname)
