import numpy as np 
import matplotlib.pyplot as plt
import estimators 
import sample

# Save or show plots
save_plots = True

#set colors
color1 = '#FF595E'
color2 = '#1982C4'
color3 = '#6A4C93'
colors =[color1, color2, color3]

#no interval is given of where to sample f, choose [-5, 5]
#choose function and number of samples
f = lambda x: x**2
a = -5
b = 5
number_of_samples = 20
number_of_train_samples = 10

xeval = np.linspace(-5,5,1000)
degree = 5
seed = 50
#get random sample, and divide into training and validation data

train_x, train_y, valid_x, valid_y = sample.random_sample_equi(number_of_samples, f, a, b, number_of_train_samples, seed = seed)

#Make Graphs for gammas = 0, 0.1 and calculate RSS, seed = 50
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
ax_initial.plot(xeval, f(xeval), label = 'f(x) = $x^2$', color = color3)
ax_initial.plot(train_x, train_y,'.', color ='green', label = 'Training Data')
ax_initial.legend()
ax_initial.set_xlabel('x')
ax_initial.set_ylabel('y')
print(rss_initial)

if save_plots:
    plt.savefig("../Images/2b_initial_gammas.pdf")
else:
    plt.show()
plt.close()

#make RSS graph for gammas between 0 and 50 for seed = 50
gammas_log = np.linspace(0,50,1000)
fig_log10, ax_log10 = plt.subplots(1,1)
rss_log10 = []
for gamma in gammas_log:
    ridge = estimators.ridge(gamma, degree)
    ridge.fit(train_x, train_y)
    rss_log10.append(ridge.RSS(valid_x, valid_y))
ax_log10.plot(gammas_log, rss_log10, color = color2)
ax_log10.set_xlabel('$\gamma$')
ax_log10.set_ylabel('Residual Sum of Squares')

if save_plots:
    plt.savefig("../Images/2b_seed50_gammas.pdf")
else:
    plt.show()
plt.close()

#Make RSS graph for gammas between 0 and 50 for seeds 1-100
fig, ax = plt.subplots(1,1)
seed_list = range(1,101) #iterate through all seeds
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
ax.set_yscale('log')

if save_plots:
    plt.savefig("../Images/2b_seeds1_100.pdf")
else:
    plt.show()
plt.close()

# Calculate mean and stdev across different seeds
means = np.mean(mean_std_mat,axis=0)
best_mean = np.min(means)
best_gamma = gammas[np.argmin(means)]
stdevs = np.std(mean_std_mat,axis=0)

# Plot our mean and stdev plots
plt.plot(gammas, means, color="red", label="Mean")
plt.fill_between(gammas, means-stdevs,\
                means+stdevs,\
                color="red", alpha=0.25, edgecolor=None, label="Stdev")
plt.semilogy()
plt.xlabel('$\gamma$')
plt.ylabel('log10 of Residual Sum of Squares')
plt.legend()

if save_plots:
    plt.savefig("../Images/2b_seeds1_100mean.pdf")
else:
    plt.show()
plt.close()

nonzero_gammas_count = np.count_nonzero(gammas_best)
print('Best gamma was ' + str(best_gamma))
print('Min RSS was ' + str(best_mean))
print('We had this many 0 gammas ' + str(len(gammas_best) - nonzero_gammas_count))
print('We had this many nonzero gammas ' + str(nonzero_gammas_count))
print('Percent nonzero ' + str(nonzero_gammas_count/len(gammas_best)))
print('Percent 0 gammas ' +str(1 - (nonzero_gammas_count/len(gammas_best))))
print('Mean Best Gamma = ' + str(np.mean(gammas_best)))





