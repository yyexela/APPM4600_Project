import numpy as np
import matplotlib.pyplot as plt
import estimators
import finite_diff

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
degree = 10
weights = finite_diff.generate_centered_D(degree+1)
lam = .2
tikhonov = estimators.tikhonov(lam, degree, weights)
tikhonov.fit(x_train, y_train)
RSS = tikhonov.RSS(x_test,y_test)
print(RSS)
