import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
from statistics import mean

class Node:

  def __init__(self, data, y_col, depth, methood = None):
    self.data = data
    self.y_col = y_col
    self.depth = depth

    self.X = data.drop(y_col, axis = 1)
    self.y = data[y_col].to_list()
    self.features = list(self.X.columns) #признаки
    # это так позже, когда мы делаем наш поиск, чтобы получить лучшее разделение.  мы можем просто индексировать признаки
    self.counts = self.get_counts(self.y)
    self.gini = self.get_gini(self.counts)

    self.num_samples = len(self.y)

    self.predicted_label = np.argmax(list(self.counts.values()))

    self.left = None
    self.right = None

  def get_counts(self, y):
    return {0: y.count(0), 1: y.count(1), 2: y.count(2)} #use unique

  def get_gini(self, counts): #mega hardcode, need change
    p0 = counts[0] / (counts[0] + counts[1] + counts[2])
    p1 = counts[1] / (counts[0] + counts[1] + counts[2])
    p2 = counts[2] / (counts[0] + counts[1] + counts[2])
    return 1 - (p0 ** 2 + p1 ** 2 + p2**2)

  def get_split_gini(self, features, split_point):
    #разделили на below above
    below = self.data[self.data[features] < split_point]
    above = self.data[self.data[features] > split_point]

    #посчитали, сколько каждого класса в них  
    below_counts = self.get_counts(below[self.y_col].to_list())
    above_counts = self.get_counts(above[self.y_col].to_list())

    below_weight = len(below) / (len(below) + len(above))
    above_weight = 1 - below_weight

    below_gini = self.get_gini(below_counts)
    above_gini = self.get_gini(above_counts)
    #Прирост инфы

    return (below_weight * below_gini) + (above_weight * above_gini)

  def get_opt_split(self):
    opt_feature, opt_split = None, None
    opt_gini_reduction = 0 # We want to maximize this
    for feature in self.features:
      # possible_splits = self.get_possible_splits(feature)
      df_splits = sorted(self.data[feature])
      possible_splits = df_splits[1: -1]
      for split in possible_splits:
        split_gini = self.get_split_gini(feature, split)
        gini_reduction = self.gini - split_gini

        if gini_reduction > opt_gini_reduction:
           opt_gini_reduction = gini_reduction
           opt_feature, opt_split = feature, split

    return opt_feature, opt_split
  
  

  def make_split(self, max_depth, min_samples_split):
    self.opt_feature, self.opt_split = self.get_opt_split()
    if self.opt_feature is not None:
        if (self.depth < max_depth) and (self.num_samples > min_samples_split):
            below = self.data[self.data[self.opt_feature] < self.opt_split]
            above = self.data[self.data[self.opt_feature] > self.opt_split]
            
            self.left = Node(below, 'target', depth = self.depth + 1)
            self.right = Node(above, 'target', depth = self.depth + 1)
            #Reccurence hahahahahah
            self.left.make_split(max_depth, min_samples_split) #????
            self.right.make_split(max_depth, min_samples_split)

class CART:  
  def __init__(self, max_depth, min_samples_split, task = None):
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split

  def train(self, train):
    self.root = Node(train, 'target', depth = 0)
    self.root.make_split(self.max_depth, self.min_samples_split)

  def predict_one(self, row):
    node = self.root
    while node.depth < self.max_depth:
      if node.num_samples <= self.min_samples_split:
        break
      split_feature, split_point = node.opt_feature, node.opt_split

      if split_feature is None:
        break
      else:
        if row[split_feature] < split_point:
          node = node.left
        else:
          node = node.right
      return node.predicted_label

  def predict_many(self, X):
    predictions = []
    for idx, row in X.iterrows():
      predictions.append(self.predict_one(row))
    return predictions