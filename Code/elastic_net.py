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

        # For ease, store value for number of data points
        self.N = self.x_data_initial.shape[0]

        # Create our X matrix from the paper
        self.X = self.create_X(self.x_data_initial, self.degree)

        # Standardize x values
        self.X, self.X_means, self.X_stds = self.standardize_X(self.X)

        # Make initial weights values
        # TODO: What should these be initialized to? Currently just doing zero
        self.b = np.ones(degree+1)*b_init

        # Get b0 term
        self.b[0] = self.get_intercept()

    def get_X(self):
        '''
        `get_X`

        Returns the X matrix
        '''
        return self.X

    def get_means(self):
        '''
        `get_means`

        Returns the means for the data matrix
        '''
        return self.X_means

    def get_stds(self):
        '''
        `get_stds`

        Returns the stds for the data matrix
        '''
        self.X_stds

    def standardize_X(self, X):
        '''
        `standardize_X`

        Normalizes X matrix per column
        Each column has mean zero and sum of squares divided by rows as 1

        Parameters

        X matrix which is standardized

        Returns

        Standardized X matrix, the column means, the column standard deviations
        '''

        means = np.mean(X, 0)
        stds = np.std(X, 0)

        X, means, stds = self.standardize_X_ms(X, means, stds)

        return X, means, stds

    def standardize_X_ms(self, X, means, stds):
        '''
        `standardize_X_ms`

        Normalizes X matrix per column given means and stds
        Each column has mean zero and sum of squares divided by rows as 1

        Parameters

        X matrix which is standardized
        Column means which are used in standardization
        Column standard deviations which are used in standardization

        Returns

        Standardized X matrix, the column means used, the column standard deviations used
        '''

        # Remove zeros from standard deviations
        for i in range(len(stds)):
            if stds[i] == 0. :
                stds[i] = 1.
        X = (X - means)/stds

        return X, means, stds

    def unstandardize_X(self, X):
        '''
        `unstandardize_X`

        Un-normalizes X matrix per column using the training means and standard deviations

        Parameters

        X which is to be unstandardized

        Returns

        Unstandardized input matrix X
        '''

        X = X*self.X_stds + self.X_means
        return X

    def get_b(self):
        '''
        `get_b`

        Returns current elastic net weights
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

        # Create our X matrix from the paper
        X = self.create_X(x_eval, self.degree)

        # Standardize x values
        X, _, _ = self.standardize_X_ms(X, self.X_means, self.X_stds)

        # Get b0 term
        self.b[0] = self.get_intercept()

        # Get the y values for our X's
        y = X @ self.b

        return y + self.b[0]
    
    def get_intercept(self):
        '''
        `get_intercept`

        Returns

        Intercept formula obtained from Tyler
        '''

        b0 = sum((self.y_data - self.X@self.b)[1:])/self.N
        return b0

    def iterate_coord_descent(self, n):
        '''
        `iterate_coord_descent`

        Does n steps of coordinate descent for each weight b[1] to b[-1]

        Parameters

        n: Number of steps to do

        Returns

        Nothing, but updates weights in beta
        '''

        for _ in range(n):
            for j in range(1, self.degree+1):
                self.step_j(j)

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

    def get_elastic_net(self, x_data, y_data):
        '''
        `get_elastic_net`

        Gets the elastic net value, formula 1 in
        "Regularization Paths for Generalized Linear Models via Coordinate Descent" (2010)
        that we're trying to minimize

        Parameters

        x_data, y_data: (x,y) points that we're calculating the residual sum of squares for 

        Returns

        Elastic net formula that is being minimized (formula 1 in the above paper)
        '''

        # Calculate RSS term
        RSS = self.get_RSS(x_data, y_data)

        # Calculate regularization term
        P = (1-self.alpha)*norm(self.b[1:],2)**2/2 + self.alpha*norm(self.b[1:],1)

        # Return function elastic net is trying to minimize
        return RSS + P

    def get_RSS(self, x_data, y_data):
        '''
        `get_RSS`

        Gets the Residual Sum of Squares

        Parameters

        x_data, y_data: (x,y) points that we're calculating the residual sum of squares for 

        Returns

        Elastic net formula that is being minimized (formula 1 in the above paper)
        '''

        # Create our X matrix and standardize it
        X = self.create_X(x_data, self.degree)
        X, _, _ = self.standardize_X(X)

        # Calculate RSS term
        RSS = sum(np.power(np.sum(X*self.b,1)-y_data,2))
        RSS = RSS/self.N

        # Return Residual Sum of Squares
        return RSS