from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import numpy as np
from MyClastering import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# iris = load_iris()
# X = iris.data
# y = iris.target

# model = KMeans()
# model.fit(X)

# labels = model.labels
# plt.scatter(X[:, 0], X[:, 1], c=labels)
# plt.scatter(model.centroids[:, 0], model.centroids[:, 1], marker='x', s=200, linewidths=3, color='r')
# plt.xlabel('feature 1')
# plt.ylabel('feature 2')
# plt.title('KMeans Clustering')
# plt.show()

X, y = make_blobs(n_samples=100, random_state=10)
model = KMeans()
model.fit(X)

labels = model.labels
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(model.centroids[:, 0], model.centroids[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('KMeans Clustering')
plt.show()