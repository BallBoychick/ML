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
    # self.gini = self.get_gini(self.counts)
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
    # print("P0", p0, "P1", p1, "P2", p2)
    return 1 - (p0 ** 2 + p1 ** 2 + p2**2)

  def get_possible_splits(self, features): #????
    unique_values = np.unique(self.X[features])
    # if len(unique_values) == 1:
    #   return 0
    return mean(unique_values)

  def get_split_gini(self, features, split_point):
    #разделили на below above
    # if len(self.data[self.data[features]]) > 1
    below = self.data[self.data[features] < split_point]
    above = self.data[self.data[features] > split_point]
    if len(below) and len(above) == 1:
      return 0

    #посчитали, сколько каждого класса в них  
    below_counts = self.get_counts(below[self.y_col].to_list())
    # print("BW", below_counts)
    above_counts = self.get_counts(above[self.y_col].to_list())
    # print("AW", above_counts)

    below_weight = len(below) / (len(below) + len(above))
    above_weight = 1 - below_weight

    below_gini = self.get_gini(below_counts)
    above_gini = self.get_gini(above_counts)
    #Прирост инфы
    # print("PRIROST: ")
    # print("BW", below_weight, "*", "BG", below_gini, "+", "AW", above_weight, "*", "AG", above_gini)

    return (below_weight * below_gini) + (above_weight * above_gini)

  def get_opt_split(self):
    opt_feature, opt_split = None, None
    opt_gini_reduction = 0 # We want to maximize this
    for feature in self.features:
      possible_splits = self.get_possible_splits(feature)
      # for split in possible_splits:
      split_gini = self.get_split_gini(feature, possible_splits)
      gini_reduction = self.gini - split_gini
      # print("gini", self.gini, "minus", split_gini)
      # print("gini_reduction", gini_reduction)

      if gini_reduction > opt_gini_reduction:
          opt_gini_reduction = gini_reduction
        #  print("opt_gini_reduction", opt_gini_reduction)
          opt_feature, opt_split = feature, possible_splits

    return opt_feature, opt_split
  
  

  def make_split(self, max_depth, min_samples_split):
    self.opt_feature, self.opt_split = self.get_opt_split()
    # print("OPT", self.opt_feature)
    if self.opt_feature != None:
        # print("YES")
        # print("D_MD", self.depth, max_depth)
        # print("N_MIN", self.num_samples, min_samples_split)
        if (self.depth < max_depth) or (self.num_samples > min_samples_split)  or (self.num_samples != 2):
            # print("YES2")
            below = self.data[self.data[self.opt_feature] < self.opt_split]
            above = self.data[self.data[self.opt_feature] > self.opt_split]
            # print("B", below)
            # print("A", above)
            
            self.left = Node(below, 'target', depth = self.depth + 1)
            self.right = Node(above, 'target', depth = self.depth + 1)
            # print("DEEEEEEEEEEEEEEEEEPTH", self.depth)
            #Reccurence hahahahahah
            self.left.make_split(max_depth, min_samples_split) #????
            self.right.make_split(max_depth, min_samples_split)


            # if self.methood == "gini":
            #   dshadkhjaks

            # if self.methood == "entorpy":
            #   dsadasd


class CART:  
  def __init__(self, max_depth, min_samples_split, task = None):
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split

  def train(self, train):
    self.root = Node(train, 'target', depth = 0)
    self.root.make_split(self.max_depth, self.min_samples_split)

  def predict_one(self, row):
    node = self.root
    # n = 0
    while node.depth < self.max_depth:
      # print("NM", node.num_samples)
      if node.num_samples <= self.min_samples_split:
        # print('BREAK1')
        break
      # n += 1
      # print("N", n)
      split_feature, split_point = node.opt_feature, node.opt_split

      if split_feature is None:
        # print('BREAK2')
        break
      else:
        if row[split_feature] < split_point:
          node = node.left
          # print(node.left)
          # print("NODE_LEFT", node.depth)
        else:
          node = node.right
          # print(node.right)
          # print("NODE_RIGHT", node.depth)
      return node.predicted_label

  def predict_many(self, X):
    predictions = []
    for idx, row in X.iterrows():
      predictions.append(self.predict_one(row))
    return predictions