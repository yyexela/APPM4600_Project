import numpy as np

class ridge:
    def __init__(self, gamma = 0, degree = 1):
        self.gamma = gamma
        self.degree = degree
        self.E_ridge = None
    
    def construct_A(self, input_x):
        A_matrix = np.ones((input_x.shape[0], self.degree + 1))
        for i in range(A_matrix.shape[1] - 1):
            A_matrix[:,i + 1] = np.power(input_x, i + 1)
        return A_matrix

    def fit(self, train_x, train_y):
        A_matrix = self.construct_A(train_x)
        b = train_y
        self.E_ridge = np.linalg.inv(np.transpose(A_matrix)@A_matrix + self.gamma*np.identity(A_matrix.shape[1]))@np.transpose(A_matrix)@b

    def predict(self, input_x):
        A_test = self.construct_A(input_x)
        b_hat = A_test@self.E_ridge
        return b_hat

    def RSS(self, valid_x, valid_y):
        b_hat = self.predict(valid_x)
        rss = np.sum(np.power((b_hat - valid_y),2))
        return rss



