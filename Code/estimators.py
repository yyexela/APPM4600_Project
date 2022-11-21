import numpy as np
import finite_diff

class ridge:
  def __init__(self, gamma = 0, degree = 1):
    #gamma is regularization constat, degree is degree of polynomial
    self.gamma = gamma
    self.degree = degree
    self.E_ridge = None
    
  # Construct A matrix
  def construct_A(self, input_x):
    A_matrix = np.ones((input_x.shape[0], self.degree + 1))
    for i in range(A_matrix.shape[1] - 1):
      A_matrix[:,i + 1] = np.power(input_x, i + 1)
    return A_matrix

  #Use the closed form solution of the ridge estimator to find solution
  def fit(self, train_x, train_y):
    A_matrix = self.construct_A(train_x)
    b = train_y
    self.E_ridge = np.linalg.inv(np.transpose(A_matrix)@A_matrix + self.gamma*np.identity(A_matrix.shape[1]))@np.transpose(A_matrix)@b

  #Predict y values for given x values after estimator has been fitted
  def predict(self, input_x):
    A_test = self.construct_A(input_x)
    b_hat = A_test@self.E_ridge
    return b_hat
    
  #Caclulate RSS for some validation x and validation y
  def RSS(self, valid_x, valid_y):
    b_hat = self.predict(valid_x)
    rss = np.sum(np.power((b_hat - valid_y),2))
    return rss


class tikhonov:
  ## Weights needs to be a list that will work for implementation of finite_diff.py
  def __init__(self, _lambda, degree, weights):
    self._lambda = _lambda
    self.degree = degree
    self.xstar = None
    self.weight_matrix = weights
	
  def construct_A(self, in_x):
    A = np.ones((in_x.shape[0], self.degree + 1))
    for i in range(self.degree):
      A[:, i + 1] = np.power(in_x, i + 1)
    return A

  def fit(self, train_x, train_y):
    A = self.construct_A(train_x)
    b = train_y
    D = self.weight_matrix
    self.xstar = np.linalg.inv((np.transpose(A) @ A + self._lambda**2 * np.transpose(D) @ D)) @ np.transpose(A) @ b
 
  def predict(self, test_x):
    A_test = self.construct_A(test_x)
    b_hat = A_test @ self.xstar
    return b_hat
  
  def RSS(self, test_x, test_y):
    b_hat = self.predict(test_x)
    rss = np.sum(np.power((b_hat - test_y), 2))
    return rss
