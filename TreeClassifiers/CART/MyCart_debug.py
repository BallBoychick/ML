from MyCart import Node, CART
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

videogame = '{"console": [0.32, 0.44, 0.22, 0.12, 0.63, 0.111, 0.022, 0.30, 0.61, 0.27, 0.35], "games": [0.34, 0.45, 0.2, 0.121, 0.62, 0.11, 0.02, 0.35,  0.50, 0.16, 0.75], "target": [0, 1, 2, 0, 0, 2, 1, 0, 1, 1, 2]}'
dict = json.loads(videogame)
df2 = pd.DataFrame.from_dict(dict)

train2 = df2.sample(frac = 2/3, random_state = 69)
test2 = df2.drop(train2.index).reset_index(drop = True)


#--------------------------Debug-Start--------------------------------------#
root2 = Node(train2, 'target', depth = 0)
print(root2.counts)
print(root2.get_possible_splits('games'))

print(root2.get_opt_split())
# print(root2.left)
print(root2.make_split(max_depth = 2, min_samples_split = 3))

#------------------Train----------------------------------#
# model2 = CART(max_depth = 2, min_samples_split=2)
# model2.train(train2)

# X_train, X_test, y_train, y_test = train_test_split(train2.drop('target', axis = 1), train2['target'], test_size=1/2)
# row2 = X_test
# print("YTr\n", y_train)
# print("YTe\n", y_test)
# print(X_test)
# print("Predict\n", model2.predict_many(row2))