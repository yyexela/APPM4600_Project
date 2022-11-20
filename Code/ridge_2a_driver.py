import numpy as np 
import matplotlib.pyplot as plt
import estimators 
import sample

#no interval is given of where to sample f, choose [-5, 5)
f = lambda x: 3*x + 2
a = -5
b = 5
number_of_samples = 20
number_of_train_samples = 10
seed = range(0,20)

#get random sample, and divide into training and validation data
fig, ax = plt.subplots(1,1)
degree = 1
xeval = np.linspace(-5,5,1000)
gammas_best = []
for seed in seed:
    train_x, train_y, valid_x, valid_y = sample.random_sample_equi(number_of_samples, f, a, b, number_of_train_samples, seed = seed)

    '''
    fig_data, ax_data = plt.subplots(1,1)
    ax_data.plot(train_x, train_y,'.', label = 'Training Data')
    ax_data.plot(valid_x, valid_y,'.', label = 'Validation Data')
    ax_data.plot(xeval, f(xeval), label = 'f(x) = 3x + 2')
    ax_data.legend()
    ax_data.set_title('Training and Validation Data')
    ax_data.set_xlabel('x')
    ax_data.set_ylabel('y')
    '''

    '''
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
    print(rss_initial)
    plt.show()
    '''

    gammas_log = np.linspace(0,10,10000)
    #fig_log10, ax_log10 = plt.subplots(1,1)
    rss_log10 = []
    for gamma in gammas_log:
        ridge = estimators.ridge(gamma, degree)
        ridge.fit(train_x, train_y)
        rss_log10.append(ridge.RSS(valid_x, valid_y))
    gammas_best.append(gammas_log[np.argmin(rss_log10)])
    #ax_log10.plot(gammas_log, rss_log10)
    #ax_log10.set_xscale('log')
    #ax_log10.set_title('Residual Sum of Squares for various gammas')
    #ax_log10.set_xlabel('gamma')
    #ax_log10.set_ylabel('Residual Sum of Squares')
    
    '''
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
    fig_final, ax_final = plt.subplots(1,1)
    ax_final.plot(xeval, f(xeval), label = 'f(x) = 3x + 2')
    ridge = estimators.ridge(gamma = 0, degree = degree)
    ridge.fit(train_x, train_y)
    ax_final.plot(xeval, ridge.predict(xeval),'--', label = '\gamma = 0')
    ridge = estimators.ridge(gammas_fine[np.argmin(rss_fine)], degree)
    ridge.fit(train_x, train_y)
    ax_final.plot(xeval, ridge.predict(xeval),'--', label = '\gamma = ' + str(gammas_fine[np.argmin(rss_fine)]))
    ax_final.legend()
    ax_final.set_title('Models for actual function, no regularization, and best \gamma')
    ax_final.set_xlabel('x')
    ax_final.set_ylabel('y')
    plt.show()
    '''
    ax.plot(gammas_log, rss_log10, alpha = 0.2)


ax.set_ylim(bottom = 0, top = 200)
plt.show()

fig1, ax1 = plt.subplots(1,1)
ax1.hist(gammas_best, bins = 1000)
plt.show()
#print('The best gamma is ' + str(gammas_fine[np.argmin(rss_fine)]))
#print('With RSS of ' + str(np.min(rss_fine)))
