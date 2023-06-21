from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import numpy as np
from MyClastering import KMeans
import matplotlib.pyplot as plt

# iris = load_iris()
# iris = pd.DataFrame(
#     data= np.c_[iris['data'], iris['target']],
#     columns= iris['feature_names'] + ['target']
#     )

# y = iris["target"]
# X = iris.drop(["target"], axis=1)

# print(X)

iris = load_iris()
X = iris.data
y = iris.target

centroids = X[np.random.choice(150, 3)]
print(centroids)