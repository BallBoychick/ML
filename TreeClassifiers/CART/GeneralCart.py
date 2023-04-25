import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
from statistics import mean
from ClassGetMetrics import GetMetrics

class Node:

  def __init__(self, data, y_col, depth, methood, pred, counter):
    self.data = data
    self.y_col = y_col
    self.depth = depth
    self.methood = methood
    self.pred = pred
    self.counter = counter


    self.X = data.drop(y_col, axis = 1)
    self.y = data[y_col].to_list()
    self.features = list(self.X.columns) #признаки
    self.counts = self.counter(self.y)
    self.gini = self.methood(self.counts)

    self.num_samples = len(self.y)
    
    self.predicted_label = pred(self.y)

    self.left = None
    self.right = None

  def get_split_gini(self, features, split_point):
    #разделили на below above
    below = self.data[self.data[features] <= split_point]
    above = self.data[self.data[features] > split_point]

    #посчитали, сколько каждого класса в них  
    below_counts = self.counter(below[self.y_col].to_list())
    above_counts = self.counter(above[self.y_col].to_list())

    below_weight = len(below) / (len(below) + len(above))
    above_weight = 1 - below_weight

    below_gini = self.methood(below_counts)
    above_gini = self.methood(above_counts)
    #Прирост инфы

    return (below_weight * below_gini) + (above_weight * above_gini)

  def get_opt_split(self):
    opt_feature, opt_split = None, None
    opt_gini_reduction = 0
    for feature in self.features:
      df_splits = np.unique(self.X[feature])
      possible_splits = (df_splits [1:] + df_splits [:-1])/2
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
            below = self.data[self.data[self.opt_feature] <= self.opt_split]
            above = self.data[self.data[self.opt_feature] > self.opt_split]
            
            self.left = Node(below, 'Diabetes_012', depth = self.depth + 1, methood=GetMetrics.get_gini, pred=GetMetrics.get_pred_class, counter=GetMetrics.get_counts_class)
            self.right = Node(above, 'Diabetes_012', depth = self.depth + 1, methood=GetMetrics.get_gini, pred=GetMetrics.get_pred_class, counter=GetMetrics.get_counts_class)
            #Reccurence hahahahahah
            self.left.make_split(max_depth, min_samples_split)
            self.right.make_split(max_depth, min_samples_split)

class CART:  
  def __init__(self, max_depth, min_samples_split):
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split

  def train(self, train):
    self.root = Node(train, 'Diabetes_012', depth = 0, methood=GetMetrics.get_gini, pred=GetMetrics.get_pred_class, counter=GetMetrics.get_counts_class)
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
