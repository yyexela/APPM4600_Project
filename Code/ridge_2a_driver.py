import numpy as np 
import matplotlib.pyplot as plt
import estimators 
from sample import random_sample

#no interval is given of where to sample f, choose [-5, 5)
f = lambda x: 3*x + 2
a = -5
b = 5
number_of_samples = 20
number_of_train_samples = 10
seed = 50

#get random sample, and divide into training and validation data
train_x, train_y, valid_x, valid_y = random_sample(number_of_samples, f, a, b, number_of_train_samples, seed = seed)


degree = 1
gammas_initial = [0,0.1]

fig_initial, ax_initial = plt.subplots(1,2)
rss_initial = []
xeval = np.linspace(-5,5,1000)
for gamma in gammas_initial:
    ridge = estimators.ridge(gamma, degree)
    ridge.fit(train_x, train_y)
    rss_initial.append(ridge.RSS(valid_x, valid_y))
    predict_y = ridge.predict(valid_x)
    ax_initial[0].plot(valid_x, predict_y, label = 'gamma = ' + str(gamma))
ax_initial[0].plot(xeval, f(xeval), label = 'f(x) = 3x + 2')
ax_initial[0].plot(valid_x, valid_y, '.', label = 'Validation Data')
ax_initial[0].legend()
ax_initial[0].set_title('Predicted lines for gamma = 0, 0.1')
ax_initial[0].set_xlabel('x')
ax_initial[0].set_ylabel('y')
ax_initial[1].plot(gammas_initial, rss_initial)
ax_initial[1].set_title('Residual Sum of Squares for gamma = 0, 0.1')
ax_initial[1].set_xlabel('gamma')
ax_initial[1].set_ylabel('Residual Sum of Squares')

plt.show()


gammas_log = [0, 0.1, 1, 10, 100]
fig_log10, ax_log10 = plt.subplots(1,1)
rss_log10 = []
for gamma in gammas_log:
    ridge = estimators.ridge(gamma, degree)
    ridge.fit(train_x, train_y)
    rss_log10.append(ridge.RSS(valid_x, valid_y))
ax_log10.plot(gammas_log, rss_log10)
ax_log10.set_xscale('log')
ax_log10.set_title('Residual Sum of Squares for various gammas')
ax_log10.set_xlabel('gamma')
ax_log10.set_ylabel('Residual Sum of Squares')
plt.show()

gammas_fine = np.linspace(0,10,100)
fig_fine, ax_fine = plt.subplots(1,1)
rss_fine = []
for gamma in gammas_fine:
    ridge = estimators.ridge(gamma, degree)
    ridge.fit(train_x, train_y)
    rss_fine.append(ridge.RSS(valid_x, valid_y))
ax_fine.plot(gammas_fine, rss_fine)
ax_fine.set_title('Residual Sum of Squares for various gammas')
ax_fine.set_xlabel('gamma')
ax_fine.set_ylabel('Residual Sum of Squares')
plt.show()