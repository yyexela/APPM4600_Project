from elastic_net_helper import ElasticNetHelper
from prints import *

f = lambda x: 3*x+2

for alpha, _lambda in [(0, 0.1), (1, 0.1), (0.5, 0.1), (0,0)]:

    small_banner(f"3x+2, alpha {alpha}, lambda {_lambda}", False, True)

    enp = ElasticNetHelper(f = f , degree = 1, x_min = -5, x_max = 5,\
                            num_evals_x = 20, num_train_x = 10,\
                            noise = 1, verbose = False)
    enp.print_params()
    print()

    enp.new_en_solver(alpha = alpha, _lambda = _lambda, seed = 0)

    print(f"Before training RSS {enp.get_RSS('train')}")
    print(f"Before training EN {enp.get_elastic_net('train')}")
    print(f"Before validation RSS {enp.get_RSS('val')}")
    print(f"Before validation EN {enp.get_elastic_net('val')}")
    print(f"Before weights")
    print(enp.get_weights())
    print()

    enp.train(10)
    enp.make_plot(num_true_x = 100, title = None, img_name=f"3x+2_a={alpha}_l={_lambda}.pdf")

    print(f"After training RSS {enp.get_RSS('train')}")
    print(f"After training EN {enp.get_elastic_net('train')}")
    print(f"After validation RSS {enp.get_RSS('val')}")
    print(f"After validation EN {enp.get_elastic_net('val')}")
    print(f"After weights")
    print(enp.get_weights())
    print()
