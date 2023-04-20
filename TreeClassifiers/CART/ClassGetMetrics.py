import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
from statistics import mean

class GetMetrics:
    @staticmethod
    def get_gini(counts): #methood
        summ = sum(counts)
        return 1 - sum((i / summ) ** 2 for i in counts)

    @staticmethod
    def get_pred_class(y):
        labels = np.unique(y)
        count = [list(y).count(i) for i in labels]
        return labels[np.argmax(count)]

    @staticmethod #methood   
    def get_mse(counts):
        count = mean(counts)
        return sum((i - count) ** 2 for i in counts) / len(counts)
        

    @staticmethod
    def get_pred_regr(y):
        return mean(y)
    @staticmethod    
    def get_counts_class(y):
    # return {0: y.count(0), 1: y.count(1), 2: y.count(2)} #use unique
        unique, counts = np.unique(y, return_counts= True )
        return counts
    
    @staticmethod
    def get_counts_regr(y):
        return y