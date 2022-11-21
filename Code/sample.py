import numpy as np 


def random_sample(number_of_samples, f, a, b, number_of_train_samples, mean = 0, std_dev = 1, seed = None):
    '''
    Take in Number of Samples, Function, end points, how many samples should be training samples, mean, std_dev, and random seed
    x values pulled from a uniform [a,b] distribution, adds Gaussian Noise to y values
    Returns training x, training y, valid x, and valid y 
    '''
 
    rng = np.random.default_rng(seed)
    sample_x = rng.uniform(a,b, (number_of_samples, 1))
    gaussian_noise = rng.normal(mean, std_dev, (number_of_samples, 1))
    sample_y = f(sample_x) + gaussian_noise
    sample_x_y = np.concatenate((sample_x, sample_y), axis = 1)

    train_data = rng.choice(sample_x_y, size = number_of_train_samples, replace = False) 
    valid_data = np.zeros((number_of_samples - number_of_train_samples, 2))

    counter = 0
    for i in range(sample_x_y.shape[0]):
        include = True
        for j in range(train_data.shape[0]):
            if sample_x_y[i,0] == train_data[j,0]:
                include = False
        if include == True:
            valid_data[counter] = sample_x_y[i]
            counter += 1
    
    train_data = train_data[train_data[:,0].argsort()]
    valid_data = valid_data[valid_data[:,0].argsort()]

    train_x = train_data[:,0]
    train_y = train_data[:,1]
    valid_x = valid_data[:,0]
    valid_y = valid_data[:,1]


    return train_x, train_y, valid_x, valid_y
    
def random_sample_equi(number_of_samples, f, a, b, number_of_train_samples, mean = 0, std_dev = 1, seed = None):
    '''
    Take in Number of Samples, Function, end points, how many samples should be training samples, mean, std_dev, and random seed
    x values are equispaced from [a,b], adds Gaussian Noise to y values
    Returns training x, training y, valid x, and valid y 
    '''
 
    rng = np.random.default_rng(seed)
    sample_x = np.linspace(a,b,number_of_samples)
    sample_x = np.reshape(sample_x, (number_of_samples, 1))
    gaussian_noise = rng.normal(mean, std_dev, (number_of_samples, 1))
    sample_y = f(sample_x) + gaussian_noise
    sample_x_y = np.concatenate((sample_x, sample_y), axis = 1)

    train_data = rng.choice(sample_x_y, size = number_of_train_samples, replace = False) 
    valid_data = np.zeros((number_of_samples - number_of_train_samples, 2))

    counter = 0
    for i in range(sample_x_y.shape[0]):
        include = True
        for j in range(train_data.shape[0]):
            if sample_x_y[i,0] == train_data[j,0]:
                include = False
        if include == True:
            valid_data[counter] = sample_x_y[i]
            counter += 1
    
    train_data = train_data[train_data[:,0].argsort()]
    valid_data = valid_data[valid_data[:,0].argsort()]

    train_x = train_data[:,0]
    train_y = train_data[:,1]
    valid_x = valid_data[:,0]
    valid_y = valid_data[:,1]


    return train_x, train_y, valid_x, valid_y