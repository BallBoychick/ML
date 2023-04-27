from MyNeuralWeb import NeuralWeb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json


videogame = '{"console": [0.36, 0.15, 0.34, 0.95, 0.63, 0.11, 0.022, 0.07, 0.14, 0.27, 0.35], "games": [0.35, 0.25, 0.2, 0.121, 0.62, 0.11, 0.02, 0.35,  0.50, 0.16, 0.75], "target": [0, 1, 2, 2, 0, 2, 2, 0, 1, 2, 2]}'
dict = json.loads(videogame)
df2 = pd.DataFrame.from_dict(dict)
y = df2["target"]
X = df2.drop(["target"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)
# print(X_train.shape[0])
# print(y_train.shape[0])
# print(y_train.to_numpy().shape)
layers_dims = [2, 3]
neur = NeuralWeb(X_train, y_train, layers_dims,  learning_rate=0.0075, num_iterations = 5)
print(neur.initialize) #work
print(neur.iters)
# print(neur.pred)