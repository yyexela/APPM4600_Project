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

xeval = np.linspace(-5,5,100)
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
ax_initial.set_title('Fitted Lines and Real Function for initial $\gamma$\'s, seed = 50')
print(rss_initial)
plt.show()
plt.close()

gammas_log = np.linspace(0,50,1000)
fig_log10, ax_log10 = plt.subplots(1,1)
rss_log10 = []
for gamma in gammas_log:
    ridge = estimators.ridge(gamma, degree)
    ridge.fit(train_x, train_y)
    rss_log10.append(ridge.RSS(valid_x, valid_y))
ax_log10.plot(gammas_log, rss_log10, color = color2)
ax_log10.set_title('Residual Sum of Squares for various $\gamma$\'s, seed = 50')
ax_log10.set_xlabel('$\gamma$')
ax_log10.set_ylabel('Residual Sum of Squares')
plt.show()
plt.close()


fig, ax = plt.subplots(1,1)
seed_list = range(1,101)
gammas = np.linspace(0,50,1000)
mean_std_mat = np.zeros((len(seed_list),len(gammas)))
gammas_best = []
for i in range(len(seed_list)):
    seed = seed_list[i]
    train_x, train_y, valid_x, valid_y = sample.random_sample_equi(number_of_samples, f, a, b, number_of_train_samples, seed = seed)
    rss = []
    for gamma in gammas:
        ridge = estimators.ridge(gamma, degree)
        ridge.fit(train_x, train_y)
        rss.append(ridge.RSS(valid_x, valid_y))
    gammas_best.append(gammas[np.argmin(rss)])
    mean_std_mat[i,:] = rss
    ax.plot(gammas, rss, alpha = 0.2)

ax.set_xlabel('$\gamma$')
ax.set_ylabel('log10 of Residual Sum of Squares')
ax.set_title('Residual Sum of Squares vs $\gamma$ for seeds 1-100')
ax.set_yscale('log')
#ax.set_ylim(bottom = 0, top = 200)
plt.show()
plt.close()

means = np.mean(mean_std_mat,axis=0)
best_mean = np.min(means)
best_gamma = gammas[np.argmin(means)]
stdevs = np.std(mean_std_mat,axis=0)

plt.plot(gammas, means, color="red", label="Mean")
plt.fill_between(gammas, means-stdevs,\
                means+stdevs,\
                color="red", alpha=0.25, edgecolor=None, label="Stdev")
plt.semilogy()
plt.legend()
plt.xlabel('$\gamma$')
plt.ylabel('log10 of Residual Sum of Squares')
plt.title('Mean and Standard Deviation of RSS vs $\gamma$ for seeds 1-100')
plt.show()
plt.close()

plt.plot(gammas, means, color="red", label="Mean")
plt.fill_between(gammas, means-stdevs,\
                means+stdevs,\
                color="red", alpha=0.25, edgecolor=None, label="Stdev")
plt.xlim(0,1.5)
plt.ylim(11.25,12)
plt.legend()
plt.show()
plt.close()

nonzero_gammas_count = np.count_nonzero(gammas_best)
print('Best gamma was ' + str(best_gamma))
print('Min RSS was ' + str(best_mean))
print('We had this many 0 gammas' + str(len(gammas_best) - nonzero_gammas_count))
print('We had this many nonzero gammas' + str(nonzero_gammas_count))
print('Percent nonzero ' + str(nonzero_gammas_count/len(gammas_best)))
print('Percent 0 gammas ' +str(1 - (nonzero_gammas_count/len(gammas_best))))
print('Mean Best Gamma = ' + str(np.mean(gammas_best)))

