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
alphas = np.linspace(0,1,10)
_lambdas = np.linspace(0,1,10)
#alphas = np.array([0.5])
#_lambdas = np.array([0.5])
coordinate_descent_steps = 100

# Iterate over grid
for alpha in alphas:
    for _lambda in _lambdas:
        # Create ElasticNet class
        en = ElasticNet(x_eval, y_eval, degree, alpha, _lambda, b_init = 1, verbose = False)

        # Create the others
        initial_en = en.get_elastic_net()
        for i in range(coordinate_descent_steps):
            en.iterate_coord_descent(1)
        final_en = en.get_elastic_net()

        print(f"Degree {degree}, initial EN {initial_en:2.2E}, final EN {final_en:2.2E}, alpha {alpha:0.6f}, lambda {_lambda:0.6f}")
        print()
