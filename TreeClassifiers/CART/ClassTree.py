import pandas as pd
import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature #признак(по какому прошло разделение)
        self.threshold = threshold #значение признака на которых делят данные(ПОРОГ)
        self.left = left
        self.right = right
        self.value = value #значение в конечном узле
    
    def is_leaf_node(self):
        return self.value is not None #конечный узел или нет?

class DecisionTree:
    def __init__(self, max_depth=10, min_samples=10):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None
        
    def fit(self, X, y):
        self.tree = self.grow_tree(X, y)
        
    def predict(self, X): #массив значений в конце(т.е массив метокы)
        return np.array([self.travers_tree(x, self.tree) for x in X])
    
    def entropy(self, y):
        # hist = np.bincount(y)
        unique, counts = np.unique(y, return_counts=True)
        hist = counts
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    
    def most_common(self, y): #наиболее встречающаяся метка
        labels = np.unique(y)
        count = [list(y).count(i) for i in labels]
        return labels[np.argmax(count)]
    
    def best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_gain = -1 #прирост информации
        
        for i in range(X.shape[1]): #проходим по всем признакам
            thresholds = np.unique(X[:, i])
            for threshold in thresholds:
                gain = self.information_gain(X[:, i], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = i
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def information_gain(self, X_column, y, threshold):
        if len(np.unique(y)) == 1:
            return 0
        
        n = len(y)
        parent = self.entropy(y)
        #вычисляем энтропию в левом и правом узле
        left_indexes = np.argwhere(X_column <= threshold).flatten()
        right_indexes = np.argwhere(X_column > threshold).flatten()
        
        e_l, n_l = self.entropy(y[left_indexes]), len(left_indexes)
        e_r, n_r = self.entropy(y[right_indexes]), len(right_indexes)
        
        child = (n_l / n) * e_l + (n_r / n) * e_r
        return parent - child

    def grow_tree(self, X, y, depth=0): #рекурсивный метод построения дерева
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        #критерии остановки
        if n_samples <= self.min_samples or depth >= self.max_depth or n_labels == 1:
            return Node(value=self.most_common(y))
        
        best_feature, best_threshold = self.best_split(X, y)
        
        left_indexes = np.argwhere(X[:, best_feature] <= best_threshold).flatten()
        right_indexes = np.argwhere(X[:, best_feature] > best_threshold).flatten()
        
        if len(left_indexes) == 0 or len(right_indexes) == 0:
            return Node(value=self.most_common(y))
        
        left = self.grow_tree(X[left_indexes, :], y[left_indexes], depth + 1)
        right = self.grow_tree(X[right_indexes, :], y[right_indexes], depth + 1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def travers_tree(self, x, tree): #проверка это конечный узел или нет?ы
        if tree.is_leaf_node(): #если да, то просто достаем оттуда предсказание
            return tree.value
            
        if x[tree.feature] <= tree.threshold: #если нет, то смотрим куда идти
            return self.travers_tree(x, tree.left)
        return self.travers_tree(x, tree.right)
