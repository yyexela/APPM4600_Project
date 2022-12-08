import numpy as np
from numpy.linalg import norm

class ElasticNet:
    def __init__(self, x_data, y_data, degree, alpha, _lambda, b_init = 0, verbose = False):
        '''
        `__init__`

        Initialize the Elastic Net solver
        
        Parameters

        x_data:  Numpy array of size (n+1,) for data x-values
        y_data:  Numpy array of size (n+1,) for data y-values
        degree:  Degree polynomial we're fitting to
        alpha:   See "Regularization Paths for Generalized Linear Models via Coordinate Descent" (2010)
        _lamdba: See "Regularization Paths for Generalized Linear Models via Coordinate Descent" (2010)
        verbose: (True) Enable / (False) disable print statements

        Returns

        Nothing
        '''
        # Store initial values
        self.x_data_initial = x_data
        self.y_data = y_data
        self.degree = degree
        self.alpha = alpha
        self._lambda = _lambda
        self.verbose = verbose

        # Standardize x values
        self.x_mean = np.mean(x_data)
        self.x_std = np.std(x_data)
        self.x_data = self.standardize_x(x_data)

        # For ease, store value for number of data points
        self.N = x_data.shape[0]

        # Create our X matrix from the paper
        self.X = self.create_X(self.x_data, degree)

        # Make initial weights values
        # TODO: What should these be initialized to? Currently just doing zero
        self.b = np.ones(degree+1)*b_init
    
    def standardize_x(self, x):
        '''
        `standardize_x`

        Normalizes x-values
        '''
        return (x-self.x_mean)/(self.x_std)

    def unstandardize_x(self, x):
        '''
        `unstandardize_x`

        Undoes normalization for x-values
        '''
        return x*self.x_std + self.x_mean

    def get_x_data(self):
        '''
        `get_x_data`

        Returns normalized x-values
        '''
        return self.x_data

    def get_b(self):
        '''
        `get_b`

        Returns weights
        '''
        return self.b

    def get_prediction(self, x_eval):
        '''
        `get_prediction`

        Given non-standardized x-values, return our prediction for y-values, but using weights trained on standardized x-values

        Parameters

        x_eval: x-values we want our predictions at

        Returns

        y-values at those x-values
        '''

        # Do our transformation to standardized data
        x_eval_standardized = (x_eval - self.x_mean)/(self.x_std)

        # Get the y values for our X's
        X = self.create_X(x_eval_standardized, self.degree)
        y = X @ self.b

        return y

    def step_j(self, j):
        '''
        `step_j`

        Does a coordinate descent step for variable j in beta
        j has to be nonzero since we're not optimizing the intercept
        
        This is equation 5 in
        "Regularization Paths for Generalized Linear Models via Coordinate Descent" (2010)

        Parameters

        j: Index into beta that we're optimizing

        Returns

        Nothing, but updates variable j in the weights
        '''

        if j <= 0:
            raise Exception(f"step_j: j ({j}) must be greater than 0")

        # Solve for y tilde (j) first
        y_tilde = np.sum(self.X*self.b,1)-self.X[:,j]*self.b[j]

        # First calculate sigma from equation (5)
        inner_sum = np.sum(self.X[:,j]*(self.y_data - y_tilde))

        # Then, divide it by N
        param_1 = inner_sum/self.N

        # Get second parameter for the soft-thresholding operator
        param_2 = self._lambda*self.alpha

        # Calculate numerator
        numerator = self.soft_thresholding(param_1, param_2)

        # Calculate denominator
        denominator = 1+self._lambda*(1-self.alpha)

        # Divide to get result
        res = numerator/denominator

        if self.verbose:
            print("y_tilde")
            print(y_tilde)
            print("y_data - y_tilde")
            print(self.y_data - y_tilde)
            print("inner_sum")
            print(inner_sum)
            print("param_1")
            print(param_1)
            print("param_2")
            print(param_2)
            print("numerator")
            print(numerator)
            print("denominator")
            print(denominator)
            print("res")
            print(res)

        self.b[j] = res

    def soft_thresholding(self, z, y):
        '''
        `soft_thresholding`

        This is the soft-thresholding operator, equation 6 in
        "Regularization Paths for Generalized Linear Models via Coordinate Descent" (2010)
        '''
        if y >= abs(z):
            return 0
        elif z > 0:
            return z - y
        else:
            return z + y

    def create_X(self, x_data, degree):
        '''
        `create_X`

        Creates the X matrix in our paper

        Parameters

        x_data: Numpy array of size (n+1,) for data x-values
        degree: Degree polynomial we're fitting to

        Returns

        X matrix as described in our paper
        '''
        X = np.zeros((x_data.shape[0], degree+1))
        for col in range(degree+1):
            X[:,col] = np.power(x_data,col)
        return X

    def get_elastic_net(self):
        '''
        `get_elastic_net`

        Gets the elastic net value, formula 1 in
        "Regularization Paths for Generalized Linear Models via Coordinate Descent" (2010)
        that we're trying to minimize

        Parameters

        None

        Returns

        Elastic net formula that is being minimized (formula 1 in the above paper)
        '''

        # Calculate RSS term
        RSS = sum(np.power(np.sum(self.X*self.b,1)-self.y_data,2))
        RSS = RSS/self.N

        # Calculate regularization term
        P = (1-self.alpha)*norm(self.b[1:],2)**2/2 + self.alpha*norm(self.b[1:],1)

        # Return function elastic net is trying to minimize
        return RSS + P
