from elastic_net import ElasticNet
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

class ElasticNetHelper:
    def __init__(self, f, degree, x_min, x_max, num_evals_x, num_train_x, noise = 0, verbose = False):
        '''
        `__init__`

        Initialize the plotting helper for the elastic net solver `ElasticNet`
        These parameters are meant to be constant across all generated elastic net solvers

        Parameters

        `f`: Function that we're fitting to
        `degree`: Degree fit we want
        `x_min`, `x_max`: Domain that we're fitting to
        `num_evals_x`: Number of equispaced points in the domain to use for training and validation data
        `num_train_x`: Number of randomly sampled points from `num_evals_x` to use for training, rest is for validation
        `noise`: Noise for each generated elastic net solver
        `verbose`: Optionally turn on (True) or off (False) extra prints (Defaults to 'False')

        Returns

        Nothing
        '''

        # Initialize member variables
        self.f = f
        self.degree = degree
        self.x_min = x_min
        self.x_max = x_max
        self.num_evals_x = num_evals_x
        self.num_train_x = num_train_x
        self.noise = noise
        self.verbose = verbose

        # Create our data
        self.x_eval = np.linspace(x_min, x_max, num_evals_x)
    
    def new_en_solver(self, alpha, _lambda, seed):
        '''
        `new_en_solver`

        Creates a new elastic net solve using the given seed

        Parameters

        `alpha`, `_lambda`: Hyperparameters for `ElasticNet`, see `elastic_net.py` for more details
        `seed`: Numpy seed

        Returns

        Nothing, but sets the current elastic net to this en
        '''

        # Set numpy seed
        np.random.seed(seed)

        # Create our splits
        self.x_train, self.x_val = self.get_split(self.x_eval, self.num_train_x)
        self.y_train = (self.f(self.x_train) + np.random.normal(scale=self.noise, size=self.x_train.shape))
        self.y_val = (self.f(self.x_val) + np.random.normal(scale=self.noise, size=self.x_val.shape))

        if self.verbose:
            print(f"x_train {self.x_train.shape}, x_val {self.x_val.shape}, num_evals_x {self.num_evals_x}")
            print(f"y_train {self.y_train.shape}, y_val {self.y_val.shape}, num_evals_x {self.num_evals_x}")

        # Create the elastic net solve and set it to the current one
        en = ElasticNet(self.x_train, self.y_train, self.degree, alpha, _lambda, verbose=self.verbose)
        self.en = en

    def get_split(self, x_eval, num_train_x):
        '''
        `get_split`

        Given the total data get the training and validation split

        Parameters

        x_eval: Total x-values
        num_train_x: Number of x-values in the training split

        Returns

        x_train: Training split
        x_val: Validation split
        '''

        # Create training split
        x_train = np.random.choice(x_eval, num_train_x, replace=False)
        x_train = np.sort(x_train)

        # Create validation split
        train_idx = 0
        x_val = list()
        for x in x_eval:
            if train_idx < num_train_x and x_train[train_idx] == x:
                train_idx += 1
            else:
                x_val += [x]
        x_val = np.array(x_val)

        return x_train, x_val

    def train(self, j):
        '''
        `train`

        Train the elastic net model j times over each variable

        Parameters

        j: Number of times to iterate over each variable for trainin

        Return Nothing
        '''

        self.en.iterate_coord_descent(j)

    def get_RSS(self, data):
        '''
        `get_RSS`

        Get current elastic net RSS

        Parameters

        data: Either 'val' or 'train' for validation or training data respectively

        Returns RSS
        '''

        if data == 'val':
            return self.en.get_RSS(self.x_val, self.y_val)
        elif data == 'train':
            return self.en.get_RSS(self.x_train, self.y_train)
        else:
            raise Exception("get_RSS: `data` must be either \'val\' or \'train\', but is \'{data}\'")

    def get_elastic_net(self, data):
        '''
        `get_elastic_net`

        Get current elastic net formula that's being minimized

        Parameters

        data: Either 'val' or 'train' for validation or training data respectively

        Returns elastic net formula
        '''

        if data == 'val':
            return self.en.get_elastic_net(self.x_val, self.y_val)
        elif data == 'train':
            return self.en.get_elastic_net(self.x_train, self.y_train)
        else:
            raise Exception("get_elastic_net: `data` must be either \'val\' or \'train\', but is \'{data}\'")
    
    def get_weights(self):
        '''
        `get_weights`

        Returns current elastic net weights
        '''
        return self.en.get_b()
    
    def make_plot(self, num_true_x = 100, title = 'title', x_axis = 'x', y_axis = 'y', img_name = 'PLOT.pdf', img_dir = '../Images', train_data = True, val_data = True, f_plot = True, predict_f_plot = True):
        '''
        `make_plot`

        Makes plot with (optional) training data and (optional) validation data along with (optional) the true function and (optional) the predicted function

        Returns nothing, but makes our plot
        '''

        # Plotting colors
        color1 = '#FF595E'
        color2 = '#1982C4'
        color3 = '#6A4C93'
        color4 = 'green'

        # Construct save path
        save_path = f"{img_dir}/{img_name}"

        # Create helpful data
        true_x = np.linspace(self.x_min, self.x_max, num_true_x)
        true_y = self.f(true_x)

        # Create initial plots
        fig, ax = plt.subplots(1,1,figsize=(5,4), dpi=120, facecolor='white', tight_layout={'pad': 1})

        general_marker_style = dict(markersize = 1, markeredgecolor='black', marker='o', markeredgewidth=0)
        dot_marker_style = dict(markersize = 4, markeredgecolor='black', marker='*', markeredgewidth=0.75)
        data_marker_size = 2
        scatter_marker_size = 10

        # Create original function plot
        if f_plot:
            ax.plot(true_x, true_y, color=color1, label="Original function", **general_marker_style)

        # Create predicted function plot
        if predict_f_plot:
            pred_y = self.en.get_prediction(true_x)
            ax.plot(true_x, pred_y, color=color2, label="Elastic Net function", **general_marker_style)

        # Plot training data
        if train_data:
            ax.scatter(self.x_train, self.y_train, s=scatter_marker_size, color=color3, label=f"Training data")

        # Plot validation data
        if val_data:
            ax.scatter(self.x_val, self.y_val, s=scatter_marker_size, color=color4, label=f"Validation data")

        # Add labels
        if x_axis is not None:
            ax.set_xlabel(f"{x_axis}")
        if y_axis is not None:
            ax.set_ylabel(f"{y_axis}")
        if title is not None:
            ax.set_title(f"{title}")
        ax.legend()

        # Save image
        plt.savefig(f'{save_path}')
        plt.close()
        print(f'Saved to: {save_path}')

    def print_params(self):
        '''
        `print_params`
        
        Make a nice table of all the parameters of the class
        '''

        dict_tmp = dict()
        dict_tmp["Option"] = [
            "self.degree",
            "self.x_min",
            "self.x_max",
            "self.num_evals_x",
            "self.num_train_x",
            "self.noise",
            "self.verbose"
        ]
        dict_tmp["Description"] = [
            self.degree,
            self.x_min,
            self.x_max,
            self.num_evals_x,
            self.num_train_x,
            self.noise,
            self.verbose
        ]
        print(tabulate(dict_tmp, headers="keys", tablefmt="pretty"))