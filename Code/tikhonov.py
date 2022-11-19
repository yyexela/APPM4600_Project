import numpy as np
import matplotlib.pyplot as plt
import estimators

## Generating training / testing data

num_train_samples = 80
num_test_samples = 20

x_train = np.random.uniform(-5,5,num_train_samples)
x_test = np.random.uniform(-5,5,num_test_samples)

f = lambda x : np.sin(x) + np.sin(5 * x)
y_train = f(x_train)
y_test = f(x_test)

xeval = np.linspace(-5, 5, 1000)
feval = f(xeval)
degree = 1
weights = [(0,1) for i in range(num_train_samples)]
lambda = .2
tikhonov = estimators.tikhonov(lambda, degree, weights)
tikhonov.fit(x_train, y_train_)


