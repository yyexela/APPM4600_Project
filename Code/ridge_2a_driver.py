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


xeval = np.linspace(-5,5,1000)
degree = 1
gammas_initial = [0,0.1]
fig_initial, ax_initial = plt.subplots(1,1)
rss_initial = []
for gamma in gammas_initial:
    ridge = estimators.ridge(gamma, degree)
    ridge.fit(train_x, train_y)
    rss_initial.append(ridge.RSS(valid_x, valid_y))
    ax_initial.plot(xeval, ridge.predict(xeval), label = 'gamma = ' + str(gamma))
ax_initial.plot(xeval, f(xeval), label = 'f(x) = 3x + 2')
ax_initial.legend()
plt.show()


gammas_log = np.linspace(0,1000, 10000)
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

gammas_fine = np.linspace(0,10,1000)
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

print('The best gamma is ' + str(gammas_fine[np.argmin(rss_fine)]))
