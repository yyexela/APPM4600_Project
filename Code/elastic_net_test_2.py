import matplotlib.pyplot as plt
import numpy as np
from elastic_net import ElasticNet

x_min = -1
x_max = 1
y_min = 0
y_max = 1

f = lambda x: x**2
x_eval = np.linspace(x_min, x_max, 100)
y_eval = f(x_eval)

degree = 4

# Create grid of values for alpha and lambda
#alphas = np.linspace(0,1,5)
#_lambdas = np.linspace(0,1,5)
alphas = np.array([0.5])
_lambdas = np.array([0.25])
coordinate_descent_steps = 100

# Iterate over grid
for alpha in alphas:
    for _lambda in _lambdas:
        # Create ElasticNet class
        en = ElasticNet(x_eval, y_eval, degree, alpha, _lambda, b_init = 1, verbose = False)

        X_s = en.get_X()
        x_eval_s = X_s[:,1]
        X_us = en.unstandardize_X(X_s)
        if True:
            # get X
            print("y_eval")
            print(y_eval)
            print("X standardized:")
            print(X_s)
            print("X unstandardized:")
            print(X_us)
            print("y predictions initial")
            print(en.get_prediction(x_eval))

        # Create the others
        print("Weights before")
        print(en.get_b())
        initial_en = en.get_elastic_net()
        en.iterate_coord_descent(coordinate_descent_steps)
        final_en = en.get_elastic_net()
        print("Weights after")
        print(en.get_b())

        if True:
            print("y predictions final")
            print(en.get_prediction(x_eval))

            print(f"Degree {degree}, initial EN {initial_en:2.2E}, final EN {final_en:2.2E}, alpha {alpha:0.6f}, lambda {_lambda:0.6f}")
            print()

        # Create initial plots
        fig, ax = plt.subplots(1,1,figsize=(10,8), dpi=120, facecolor='white', tight_layout={'pad': 1})

        general_marker_style = dict(markersize = 2, markeredgecolor='black', marker='o', markeredgewidth=0)
        dot_marker_style = dict(markersize = 8, markeredgecolor='black', marker='*', markeredgewidth=0.75)
        data_marker_size = 2
        scatter_marker_size = 20

        ax.scatter(x_eval_s, y_eval, s=scatter_marker_size, color='red', label="Original points normalized")
        ax.scatter(x_eval, y_eval, s=scatter_marker_size, color='blue', label="Original points")
        ax.scatter(x_eval, en.get_prediction(x_eval), s=scatter_marker_size, color='green', label=f"Regularized Fit")
        ax.scatter(x_eval_s, en.get_prediction(x_eval), s=scatter_marker_size, color='purple', label=f"Regularized Fit Normalized")
        #ax.set_ylim(y_min,y_max)
        ax.set_title(f"Degree {degree}, initial EN {initial_en:2.2e}, final EN {final_en:2.2e}, $\\alpha={alpha:0.6f}$, $\lambda={_lambda:0.6f}$")
        ax.legend()
        plt.savefig(f'../Images/EN_alpha{alpha:0.6f}_lambda{_lambda:0.6f}_degree{degree}.pdf')
        plt.close()

