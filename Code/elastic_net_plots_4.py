from elastic_net_helper import ElasticNetHelper
from prints import *
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2

save_name = "../Images/x^2_matrix.pdf"

# Store RSS for each alpha and lamba
num_a = 101
num_l = 101
num_seeds = 100

alphas = np.linspace(0,1,num_a)
_lambdas = np.linspace(0,1,num_l)
RSS_mat = np.zeros((num_a,num_l))

# Get RSS values, keep track of minimum
min_RSS, min_alpha, min_lambda, min_weights = np.inf, -1, -1, None

small_banner(f"x^2 matrix plot for {num_seeds} seeds", False, True)

for alpha_iter in range(len(alphas)):
    for _lambda_iter in range(len(_lambdas)):
        enp = ElasticNetHelper(f = f , degree = 5, x_min = -5, x_max = 5,\
                                num_evals_x = 20, num_train_x = 10,\
                                noise = 1, verbose = False)

        if alpha_iter == 0 and _lambda_iter == 0:
            enp.print_params()

        alpha = alphas[alpha_iter]
        _lambda = _lambdas[_lambda_iter]

        # Count all RSS for this seed
        RSS_sum = 0

        for seed in np.arange(0,num_seeds,1):
            enp.new_en_solver(alpha = alpha, _lambda = _lambda, seed = seed.item())
            enp.train(10)
            RSS_sum += enp.get_RSS('val')

        this_RSS = RSS_sum/num_seeds
        RSS_mat[alpha_iter][_lambda_iter] = this_RSS

        if this_RSS < min_RSS:
            min_RSS = this_RSS
            min_alpha = alpha
            min_lambda = _lambda
            min_weights = enp.get_weights()

# Print minima
print(f"Minimum RSS is {min_RSS} with alpha {min_alpha} and lambda {min_lambda}")
print(f"Minimum weights were:")
print(min_weights)
print()

# Make plot showing the result

fig, ax = plt.subplots(1,1,figsize=(5,4), dpi=120, facecolor='white', tight_layout={'pad': 1})

# Convert alphas to printable version
alpha_label_locs= np.arange(0,num_a,10)
_lambda_label_locs= np.arange(0,num_l,10)

# Make sure last element is there
if num_a-1 not in alpha_label_locs:
    alpha_label_locs = np.append(alpha_label_locs, [num_a-1])

if num_l-1 not in _lambda_label_locs:
    _lambda_label_locs = np.append(_lambda_label_locs, [num_l-1])

alpha_labels= [f'{1.0 if a >= len(alphas) else alphas[a]:0.2f}' for a in alpha_label_locs]
_lambda_labels= [f'{1.0 if l >= len(_lambdas) else _lambdas[l]:0.2f}' for l in _lambda_label_locs]

a1 = ax.matshow(RSS_mat)
plt.colorbar(a1, label="RSS")
ax.set_xlabel("$\lambda$")
ax.set_ylabel("$\\alpha$")
ax.set_yticks(alpha_label_locs, alpha_labels, rotation=00)
ax.set_xticks(_lambda_label_locs, _lambda_labels, rotation=90)
ax.tick_params(labelbottom = True)
ax.tick_params(labeltop = False)
ax.tick_params(top = False)
#plt.show()
plt.savefig(save_name)
print(f"Saved figure to {save_name}")
