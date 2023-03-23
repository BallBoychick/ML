import statistics
import numpy as np
from statistics import mode
def euclid_distance(a, b):
    return sum(((a-b)**2).T)**0.5/a.shape[0]

class My_Knn_classifier:
    def __init__(self, n_neighbours, distance) -> None:
        self.n_neighbours = n_neighbours
        self.distance = distance

    def fit(self,  X_train, y_train):
         self.X_train = X_train.to_numpy()
         self.y_train = y_train.to_numpy()
         return self

    def predict(self, X_test):
      pred = list()
      for j in X_test.to_numpy():
        distance = euclid_distance(j, self.X_train)
        nearest_neighbor_ids = self.y_train[distance.argsort()[:self.n_neighbours]]
        pred.append(mode(list(nearest_neighbor_ids)))
      return np.array(pred)