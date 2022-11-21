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
    seed = 50
    x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -3, 3, num_train_samples, seed = seed, std_dev = .7)
    xeval = np.linspace(-3,3,1000)
    feval = func(xeval)
    degree = 15
    weights = finite_diff.generate_centered_D(degree + 1)
    lam = 1
    tikhonov = estimators.tikhonov(lam, degree, weights)
    tikhonov.fit(x_train, y_train)
    coefs = tikhonov.xstar
    b_hat = tikhonov.predict(x_test) 
    poly = tikhonov.predict(xeval)
    plt.plot(xeval, poly, label = 'Least Squares Polynomial', color = color2) 
    plt.plot(x_train, y_train, '.', label = 'Training data', color = color3)
    plt.plot(xeval, feval, label = 'f(x) = ' + func_name, color = color1)
    plt.title('Degree 15 Tikhonov Estimator with $\lambda = 1$ for seed = 50')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
  
  #RSS Values vs lambda for a specific seed
  if fig == 2:
    seed = 50   
    x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -3, 3, num_train_samples, seed = seed, std_dev = .7) 
    lambdas = np.linspace(0, 20, 1000)
    degree = 15
    weights = finite_diff.generate_centered_D(degree + 1)
    RSS_vals = []
    for l in lambdas:
      tikhonov = estimators.tikhonov(l, degree, weights)
      tikhonov.fit(x_train, y_train)
      RSS_vals += [tikhonov.RSS(x_test, y_test)]
    plt.plot(lambdas, RSS_vals, color = color2)
    plt.title('RSS values of $15^{th}$ Degree Tikhonov Estimator for $\lambda \in [0, 20]$')
    plt.xlabel('$\lambda$')
    plt.ylabel('RSS')
    plt.show()  
    print('lambda that achieves minimum: ', lambdas[np.argmin(RSS_vals)])

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
    plt.plot(degrees, RSS_vals, color = color2)
    plt.title('RSS vs Degree of beta')
    plt.xlabel('Degree')
    plt.ylabel('RSS')
    plt.show()

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
    plt.legend()
    plt.title("Mean and Standard Deviation of RSS vs $\lambda$ for seeds 1-100")
    plt.xlabel("$\lambda$")
    plt.ylabel("Residual Sum of Squares")
    plt.savefig("../Images/Tikhonov_5.pdf")
    plt.show()
    plt.close()
    
    

  #RSS values for various degrees as a single statistical function
  if fig == 6:
    # Make plots of RSS values vs degree (y-axis has mean and standard deviation) for a specific seed

    seed = 50 # Set a constant seed
    num_lambdas = 100 # Number of lambdas to evaluate
    degrees = list(range(1,20)) # Degrees of the polynomials we're fitting
    lambdas = np.linspace(0, 20, num_lambdas) # Range of lambdas we're evaluating at
    y_evals = np.zeros((len(degrees),num_lambdas)) # Matrix which stores out results

    # Iterate over the different polynomial fits
    for i in range(len(degrees)):
      # Set the current degree we're using
      degree = degrees[i]
      # Generate our D matrix for centered difference
      weights = finite_diff.generate_centered_D(degree + 1)
      # Generate our training and testing data
      x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -3, 3, num_train_samples, seed = seed, std_dev = .7)
      # List that we're storing our result in for different lambdas
      RSS_vals = []
      for j in range(len(lambdas)):
        l = lambdas[j]
        # Solve our Tikhonov equation
        tikhonov = estimators.tikhonov(l, degree, weights)
        tikhonov.fit(x_train, y_train)
        RSS_val = tikhonov.RSS(x_test,y_test)
        # Store our result
        y_evals[i][j] = RSS_val
      
    # Calculate the mean and standard deviation for each lamdba
    means = np.mean(y_evals,axis=0)
    stdevs = np.std(y_evals,axis=0)
    
    # Make and save our plot
    plt.plot(lambdas, means, color="red", label="Mean")
    plt.fill_between(lambdas, means-stdevs,\
                    means+stdevs,\
                    color="red", alpha=0.25, edgecolor=None, label="Stdev")
    plt.title("Mean and Standard Deviation of RSS vs $\lambda$ for polynomials of degree 1-20")
    plt.xlabel("$\lambda$")
    plt.ylabel("Residual Sum of Squares")
    plt.legend()
    plt.savefig("../Images/Tikhonov_6.pdf")
    plt.show()
    plt.close()

  #RSS values vs degree for specific seed and lambda
  if fig == 7:
    # Make plots of RSS values vs degree (y-axis has mean and standard deviation) for a specific lambda

    num_seeds = 100 # Number of seeds we'll iterator over
    seeds = list(range(0,num_seeds)) # Make our list of seeds
    degrees = list(range(3, 20)) # Degrees of the polynomials we're fitting
    num_degrees = len(degrees)

    # Make our matrix that we'll store our results in
    RSS_vals = np.zeros((num_seeds,num_degrees))

    # Iterate over our seeds
    for i in range(num_seeds):
      seed = seeds[i]
      # Get our training and testing data
      x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -3, 3, num_train_samples, seed = seed, std_dev = .7)
      # Constant lambda for Tikhonov
      lam = .1
      RSS_val = []
      # Iterate over our degrees
      for d in degrees:
        # Generate our centered difference D matrix
        weights = finite_diff.generate_centered_D(d + 1)
        # Fit our Tikhonov
        tikhonov = estimators.tikhonov(lam, d, weights)
        tikhonov.fit(x_train, y_train)
        # Store our result
        RSS_val += [tikhonov.RSS(x_test, y_test)]
      RSS_vals[i,:] = RSS_val

    # Calculate our mean and standard deviation for each gamma
    means = np.mean(RSS_vals,axis=0)
    stdevs = np.std(RSS_vals,axis=0)

    # Make our plot
    plt.plot(degrees, means, color="red", label="Mean")
    plt.fill_between(degrees, means-stdevs,\
                    means+stdevs,\
                    color="red", alpha=0.25, edgecolor=None, label="Stdev")
    plt.ylim(0,8000)
    plt.title("Mean and Standard Deviation of RSS vs degree polynomial for seeds 0-99")
    plt.xlabel('Degree polynomial')
    plt.ylabel('Residual Sum of Squares')
    plt.legend()
    plt.savefig("../Images/Tikhonov_7.pdf")
    plt.show()
    plt.close()

  if fig == 8:
    # Make plots training/testing data, our original function, and Tikhonov fit

    # Iterate over our polynomials
    for degree in range(15,20):
      # Constant seed which looks nice for plotting purposes
      seed = 4596
      # Get our training and testing data
      x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -3, 3, num_train_samples, seed = seed, std_dev = .7)
      xeval = np.linspace(-3,3,1000)
      feval = func(xeval)
      # Generate our D matrix for centered difference
      weights = finite_diff.generate_centered_D(degree + 1)
      # Constant lambda for Tikhonov
      lam = .1
      # Fit our Tikhonov regularization
      tikhonov = estimators.tikhonov(lam, degree, weights)
      tikhonov.fit(x_train, y_train)
      # Get our predicted polynomial
      poly = tikhonov.predict(xeval)
      # Make our plot
      plt.title(f"Degree {degree} Tikhonov Estimator with $\lambda$={lam} for seed = {seed}")
      plt.plot(xeval, poly, label = 'Tikhonov Polynomial', color = color2) 
      plt.plot(x_train, y_train, '.', label = 'Training data', color = color3)
      plt.plot(x_test, y_test, '.', label = 'Testing data', color = 'hotpink')
      plt.plot(xeval, feval, label = 'f(x) = ' + func_name, color = color1)
      plt.xlabel('x')
      plt.ylabel('y')
      plt.legend()
      plt.savefig(f"../Images/Tikhonov_8_{degree}.pdf")
      plt.show()
      plt.close()

  # Fits for different difference formulas
  if fig == 9: 
    seed = 50
    x_train, y_train, x_test, y_test = random_sample_equi(2*num_train_samples, func, -3, 3, num_train_samples, seed = seed, std_dev = .7)
    xeval = np.linspace(-3,3,1000)
    feval = func(xeval)
    degree = 15
    weights = [(finite_diff.generate_forward_D(degree + 1), 'Forward'), (finite_diff.generate_backwards_D(degree + 1), 'Backwards'), (finite_diff.generate_2nd_centered_D(degree + 1), '2nd Deg. Centered')]
    for w in weights:
      lam = 1
      tikhonov = estimators.tikhonov(lam, degree, w[0])
      tikhonov.fit(x_train, y_train)
      coefs = tikhonov.xstar
      b_hat = tikhonov.predict(x_test)
      poly = tikhonov.predict(xeval)
      plt.plot(xeval, poly, label = 'Tikhonov Polynomial', color = color2)
      plt.plot(x_train, y_train, '.', label = 'Training data', color = color3)
      plt.plot(xeval, feval, label = 'f(x) = ' + func_name, color = color1)
      plt.title('Deg. 15 Tikhonov with $\lambda = .1$ for seed = 50 Using ' + w[1] + ' Diff.')
      plt.xlabel('x')
      plt.ylabel('y')
      plt.legend()
      plt.show()
      plt.close()


f = lambda x : np.sin(x) + np.sin(5*x)
#visualize(1, f, fname)
#visualize(2, f, fname)
#visualize(3, f, fname)
#visualize(4, f, fname)
#visualize(5, f, fname)
#visualize(6, f, fname)
visualize(7, f, fname)
#visualize(8, f, fname)
