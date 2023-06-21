import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, RocCurveDisplay, auc
import tensorflow as tf
from MyNeuralWeb_Final import NeuralWeb
from sklearn.preprocessing import RobustScaler

from sklearn.datasets import fetch_california_housing

# california = fetch_california_housing()

# cal = pd.DataFrame(
#     data= np.c_[california['data'], california['target']],
#     columns= california['feature_names'] + ['target']
#     )

# y = cal["target"]
# X = cal.drop(["target"], axis=1)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# print("Y", y_test)
# layers_dims = [8, 4, 1]
# neuron = NeuralWeb(layers_dims,  learning_rate=0.0005, num_iterations = 3)
# train = neuron.iterations(X_train, y_train)

# pred_test = neuron.predict_regr(X_test, train[0])

# print("PRED_TEST", pred_test)





df = pd.read_csv('../ML/Data/dataset_regression.csv')
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.isnull().sum().sum()

y = df["price_usd"]
X = df.drop(["price_usd"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)
layers_dims = [29, 4, 1]

neuron = NeuralWeb(layers_dims,  learning_rate=0.0005, num_iterations = 3)
train = neuron.iterations(X_train, y_train)

pred_test = neuron.predict_regr(X_test, train[0])

print("X_trainnnnnnnnnnn\n", X_train.to_numpy())
# print(len(pred_test))
# print(y_test.shape)
print("PRED_TEST", pred_test)
print("Y_test", y_test.to_numpy())
# print(classification_report(y_test, pred_test))