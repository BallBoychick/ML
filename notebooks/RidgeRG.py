import numpy as np


def MAE(predicted, real):
    return np.sum(np.abs(predicted - real))/real.shape[0]

def MSE(predicted, real):
    return np.sum((real-predicted)**2)/real.shape[0]


def RMSE(predicted, real):
    return np.sqrt(MSE(predicted, real))


def MAPE(predicted, real):
    return np.sum(np.abs((predicted - real) / real))/real.shape[0]
# def MAPE(Y_actual,Y_Predicted):
#     mape = np.sum(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
#     return mape

# def R2(predicted, real):
#     return 1 - (MSE(predicted, real) / np.sum((real - np.mean(real))**2))

def R2(predicted, real):
    return 1 - ((MSE(predicted, real) * np.mean(real)) / np.sum((real - predicted)**2))

class RidgeGD():
  def __init__(self, alpha):
    self.thetas = None
    self.loss_history = []
    self.alpha = alpha
 
  def add_ones(self, x):
    return np.c_[np.ones((len(x), 1)), x]
    # return np.concatenate((np.ones((len(x), 1)), x), axis = 1) #код О.Н Канева
  def objective(self, x, y, thetas, n):
    return (np.sum((y - self.h(x, thetas)) ** 2) + self.alpha * np.dot(thetas, thetas)) / (2 * n)
 
  def h(self, x, thetas):
    return np.dot(x, thetas)
 
  def gradient(self, x, y, thetas, n):
    return (np.dot(-x.T, (y - self.h(x, thetas))) + (self.alpha * thetas)) / n
 
  def fit(self, x, y, iter = 2000, learning_rate = 0.05):
    x, y = x.copy(), y.copy()
    x = self.add_ones(x)
 
    thetas, n = np.zeros(x.shape[1]), x.shape[0]
 
    loss_history = []
    for i in range(iter):
      loss_history.append(self.objective(x, y, thetas, n))
      grad = self.gradient(x, y, thetas, n)
      thetas -= learning_rate * grad
    self.thetas = thetas
    self.loss_history = loss_history

  def weights(self):
    return self.thetas

  def predict(self, x):
    x = x.copy()
    x = self.add_ones(x)
    return np.dot(x, self.thetas)