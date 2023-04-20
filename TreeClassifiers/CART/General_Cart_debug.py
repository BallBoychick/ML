# from MyCart import Node, CART
from GeneralCart import Node, CART
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

#-----------------------Debug-Regr-----------------------------------------#
# videogame = '{"console": [0.36, 0.15, 0.34, 0.95, 0.63, 0.11, 0.022, 0.07, 0.14, 0.27, 0.35], "games": [0.35, 0.25, 0.2, 0.121, 0.62, 0.11, 0.02, 0.35,  0.50, 0.16, 0.75], "target": [1.25, 6.84, 8.98, 3.21, 4.4, 5.14, 5.67, 5.22, 3.45, 5.88, 2.22]}'
# dict = json.loads(videogame)
# df2 = pd.DataFrame.from_dict(dict)

# train2 = df2.sample(frac = 0.8, random_state = 69)
# print(train2)
# test2 = df2.drop(train2.index).reset_index(drop = True)
# print(test2)

#-------------------Debug-Class----------------------------#
# videogame = '{"console": [0.36, 0.15, 0.34, 0.95, 0.63, 0.11, 0.022, 0.07, 0.14, 0.27, 0.35], "games": [0.35, 0.25, 0.2, 0.121, 0.62, 0.11, 0.02, 0.35,  0.50, 0.16, 0.75], "target": [0, 1, 2, 2, 0, 2, 2, 0, 1, 2, 2]}'
# dict = json.loads(videogame)
# df2 = pd.DataFrame.from_dict(dict)

# train2 = df2.sample(frac = 0.8, random_state = 69)
# test2 = df2.drop(train2.index).reset_index(drop = True)
# print("TR\n", train2)
# print("Te\n",test2)

#--------------------------Debug-Start--------------------------------------#
# root2 = Node(train2, 'target', depth = 0)
# print(root2.counts)
# print(root2.gini)
# print(root2.predicted_label)
# print(root2.get_split_gini('console', 0.2))
# print(root2.get_opt_split())

#-------------------End-Real-tes----------------------------------#
# # print(root2.get_possible_splits('games'))

# print(root2.get_opt_split())
# # print(root2.left)
# print(root2.make_split(max_depth = 2, min_samples_split = 3))

#------------------Train----------------------------------#
model2 = CART(max_depth = 1, min_samples_split=2)
model2.train(train2)

X_train, X_test, y_train, y_test = train_test_split(train2.drop('target', axis = 1), train2['target'], test_size=1/3)
row2 = X_test
# print("YTr\n", y_train)
# print("YTe\n", y_test)
# print("XT", X_train)
# print(X_test)
print("Predict\n", model2.predict_many(row2))


#---------------Iris_data----------------------------------#
# from sklearn.datasets import load_iris
# iris = load_iris()
# iris = pd.DataFrame(
#     data= np.c_[iris['data'], iris['target']],
#     columns= iris['feature_names'] + ['target']
#     )
# iris_split = iris
# train_iris = iris_split.sample(frac = 0.8, random_state = 69)
# test_iris = iris_split.drop(train_iris.index).reset_index(drop = True)

# model_iris = CART(max_depth = 5, min_samples_split=5)
# model_iris.train(train_iris)
# X_train, X_test, y_train, y_test = train_test_split(train_iris.drop('target', axis = 1), train_iris['target'], test_size=1/3)
# row_iris = X_test
# print("Predict\n", model_iris.predict_many(row_iris))