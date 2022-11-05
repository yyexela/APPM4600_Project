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
gammas = [0, 0.1, 0.2, 0.4, 1, 1.5]
fig, ax = plt.subplots(1,2)
rss = []
for gamma in gammas:
    ridge = estimators.ridge(gamma, degree)
    ridge.fit(train_x, train_y)
    predict_y = ridge.predict(valid_x)
    rss.append(ridge.RSS(valid_x, valid_y))
    ax[0].plot(valid_x, predict_y, label = 'gamma = ' + str(gamma))
ax[0].plot(valid_x, valid_y,'.', label = 'Validation Data')
ax[0].plot(valid_x, f(valid_x), label = 'f(x) = 3x + 2')
ax[0].legend()
ax[1].plot(gammas, rss)
plt.show()
