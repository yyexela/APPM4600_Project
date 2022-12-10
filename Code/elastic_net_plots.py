from elastic_net_helper import ElasticNetHelper

enp = ElasticNetHelper(f = lambda x: x**2+2*x-1, degree = 4, x_min = -1, x_max = 1, alpha = 0.5, _lambda = 0.1,\
                        num_evals_x = 200, num_train_x = 100, num_true_x = 200,\
                        seed = 0, img_name = 'test', verbose = False)

print(f"Before training RSS {enp.get_RSS('train')}")
print(f"Before training EN {enp.get_RSS('train')}")
print(f"Before validation RSS {enp.get_RSS('val')}")
print(f"Before validation EN {enp.get_RSS('val')}")
print(f"Before weights")
print(enp.get_weights())

enp.train(10)
enp.make_plot()

print(f"After training RSS {enp.get_RSS('train')}")
print(f"After training EN {enp.get_RSS('train')}")
print(f"After validation RSS {enp.get_RSS('val')}")
print(f"After validation EN {enp.get_RSS('val')}")
print(f"After weights")
print(enp.get_weights())