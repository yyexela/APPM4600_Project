import numpy as np 
import matplotlib.pyplot as plt
import estimators 
from sample import random_sample
from sklearn.linear_model import Ridge

#no interval is given of where to sample f, choose [-5, 5)
f = lambda x: 3*x + 2
a = -5
b = 5
number_of_samples = 20
number_of_train_samples = 10
seed = 50

#get random sample, and divide into training and validation data
train_x, train_y, valid_x, valid_y = random_sample(number_of_samples, f, a, b, number_of_train_samples, seed = seed)

fig_sample, ax_sample = plt.subplots(1,1)
fig_sample.set_size_inches(6,6)
xeval = np.linspace(-5,5,1000)
ax_sample.plot(train_x, train_y, 'o', label = 'Train Data')
ax_sample.plot(valid_x, valid_y, 'o', label = 'Validation Data')
ax_sample.plot(xeval, f(xeval), label = 'f(x) = 3x + 2')
ax_sample.set_title('Train and Validation Data')
ax_sample.set_xlabel('x')
ax_sample.set_ylabel('y')
ax_sample.legend()
fig_sample.savefig('2a_sample_data.png')
plt.show()


degree = 1
gammas_initial = [0,10]

rss_initial = []
xeval = np.linspace(-5,5,1000)
for gamma in gammas_initial:
    ridge = estimators.ridge(gamma, degree)
    ridge.fit(train_x, train_y)
    rss_initial.append(ridge.RSS(valid_x, valid_y))
    predict = ridge.predict(valid_x)
    print(predict)

print (rss_initial)

sklearn_0 = Ridge(0)
sklearn_0.fit(train_x.reshape(-1,1), train_y)
predict_sk_0 = sklearn_0.predict(valid_x.reshape(-1,1))

sklearn_1 = Ridge(10)
sklearn_1.fit(train_x.reshape(-1,1), train_y)
predict_sk_1 = sklearn_1.predict(valid_x.reshape(-1,1))

print(predict_sk_0)
print(predict_sk_1)





gammas_log = np.linspace(0,1000,1000)
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

gammas_fine = np.linspace(-5,10,200)
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