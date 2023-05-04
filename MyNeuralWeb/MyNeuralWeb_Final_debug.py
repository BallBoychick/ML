import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, RocCurveDisplay, auc
import tensorflow as tf
from MyNeuralWeb_Final import NeuralWeb


df = pd.read_csv('../ML/Data/balanced_sclaer_dataset_diabetes.csv')
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.isnull().sum().sum()


y = df["Diabetes_012"]
X = df.drop(["Diabetes_012"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)


layers_dims = [21, 64,  128, 64, 32, 16, 3]

neuron = NeuralWeb(layers_dims,  learning_rate=0.001, num_iterations = 10)
train = neuron.iterations(X_train, y_train)
print(train)

# print(train)

pred_test = neuron.predict(X_test, train[0])
print(classification_report(y_test, pred_test))

