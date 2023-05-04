import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, RocCurveDisplay, auc
import tensorflow as tf
from MyNeuralWeb_Final import NeuralWeb

df = pd.read_csv('../ML/Data/dataset_regression.csv')
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.isnull().sum().sum()

y = df["price_usd"]
X = df.drop(["price_usd"], axis=1)

