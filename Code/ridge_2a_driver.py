import numpy as np 
import matplotlib.pyplot as plt
import estimators 
import sample

#set colors
color1 = '#FF595E'
color2 = '#1982C4'
color3 = '#6A4C93'
colors =[color1, color2, color3]

#no interval is given of where to sample f, choose [-5, 5)
f = lambda x: 3*x + 2
a = -5
b = 5
number_of_samples = 20
number_of_train_samples = 10

xeval = np.linspace(-5,5,1000)
degree = 1
seed = 50
#get random sample, and divide into training and validation data

train_x, train_y, valid_x, valid_y = sample.random_sample_equi(number_of_samples, f, a, b, number_of_train_samples, seed = seed)

gammas_initial = [0,0.1]
linestyles = ['-', '--']
fig_initial, ax_initial = plt.subplots(1,1)
rss_initial = []
counter = 0
for gamma in gammas_initial:
    ridge = estimators.ridge(gamma, degree)
    ridge.fit(train_x, train_y)
    rss_initial.append(ridge.RSS(valid_x, valid_y))
    ax_initial.plot(xeval, ridge.predict(xeval),linestyles[counter], label = '$\gamma$ = ' + str(gamma), color = colors[counter])
    counter += 1
ax_initial.plot(xeval, f(xeval), label = 'f(x) = 3x + 2', color = color3)
ax_initial.plot(train_x, train_y,'.', color ='green', label = 'Training Data')
ax_initial.legend()
ax_initial.set_xlabel('x')
ax_initial.set_ylabel('y')
ax_initial.set_title('Fitted Lines and Real Function for initial $\gamma$s, seed = 50')
print(rss_initial)
plt.show()

gammas_log = np.linspace(0,50,10000)
fig_log10, ax_log10 = plt.subplots(1,1)
rss_log10 = []
for gamma in gammas_log:
    ridge = estimators.ridge(gamma, degree)
    ridge.fit(train_x, train_y)
    rss_log10.append(ridge.RSS(valid_x, valid_y))
ax_log10.plot(gammas_log, rss_log10, color = color2)
ax_log10.set_title('Residual Sum of Squares for various gammas, seed = 50')
ax_log10.set_xlabel('$\gamma$')
ax_log10.set_ylabel('Residual Sum of Squares')
plt.show()


fig, ax = plt.subplots(1,1)
seed = range(1,101)
gammas_best = []
for seed in seed:
    train_x, train_y, valid_x, valid_y = sample.random_sample_equi(number_of_samples, f, a, b, number_of_train_samples, seed = seed)


    gammas_log = np.linspace(0,40,10000)
    rss_log10 = []
    for gamma in gammas_log:
        ridge = estimators.ridge(gamma, degree)
        ridge.fit(train_x, train_y)
        rss_log10.append(ridge.RSS(valid_x, valid_y))
    gammas_best.append(gammas_log[np.argmin(rss_log10)])
    ax.plot(gammas_log, rss_log10, alpha = 0.2)

ax.set_xlabel('$\gamma$')
ax.set_ylabel('log10 of Residual Sum of Squares')
ax.set_title('Residual Sum of Squares vs $\gamma$ for seeds 1-100')
ax.set_yscale('log')
#ax.set_ylim(bottom = 0, top = 200)
plt.show()

nonzero_gammas_count = np.count_nonzero(gammas_best)
print('We had this many 0 gammas' + str(len(gammas_best) - nonzero_gammas_count))
print('We had this many nonzero gammas' + str(nonzero_gammas_count))
print('Percent nonzero ' + str(nonzero_gammas_count/len(gammas_best)))
print('Percent 0 gammas ' +str(1 - (nonzero_gammas_count/len(gammas_best))))
print('Mean Best Gamma = ' + str(np.mean(gammas_best)))

