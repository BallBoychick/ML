# Файл Predictions.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures


@st.cache_data
def loadByPickle(path):
    return pickle.load(open(path, "rb"))


@st.cache_data
def loadTensor(path):
    return tf.keras.models.load_model(path)


l2r = loadByPickle("./models/PolyRegressionRidge.sav")
br = loadByPickle("./models/BaggingRegressor.sav")
nn = loadTensor("./models/RegressionModelNeuro")

umc = st.checkbox("Использовать мой файл")

if umc:
    uploaded_file = st.file_uploader(
        "Выберите файл .csv", accept_multiple_files=False, type=["csv"]
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.dropna(inplace=True)
        st.write(df)
else:
    df = pd.read_csv("./data/emptyDF.csv")
    st.experimental_data_editor(df)


def predictPoly():
    p = PolynomialFeatures(2)
    X_p = p.fit_transform(df)
    return l2r.predict(X_p)


def predictBR():
    return br.predict(df.to_numpy())


def predictNeuro():
    return nn.predict(df.to_numpy().astype(np.float64)).T.flatten()


st.header("Линейная регрессия второй степени с L2 регуляризацией")

st.markdown("Оценки модели:")
st.markdown("  - MAE: 1701.593947318243")
st.markdown("  - MSE: 7536341.154610522")
st.markdown("  - RMSE: 2745.239726255345")
st.markdown("  - MAPE: 1.0083236256076629")
st.markdown("  - R^2: 0.8167490910841246")

result1 = np.array([])

if st.button("Предсказать", key="poly"):
    result1 = predictPoly()

if result1.any():
    st.success("Модель предсказала {}".format(result1))
else:
    st.info("Здесь будет предсказание")

st.header("Bagging regressor")

st.markdown("Оценки модели:")
st.markdown("  - MAE: 2650.729231088621")
st.markdown("  - MSE: 16209153.435119193")
st.markdown("  - RMSE: 4026.059293542408")
st.markdown("  - MAPE: 1.1580070058242642")
st.markdown("  - R^2: 0.4440883832356515")

result2 = np.array([])

if st.button("Предсказать", key="br"):
    result2 = predictBR()

if result2.any():
    st.success("Модель предсказала {}".format(result2))
else:
    st.info("Здесь будет предсказание")

st.header("Нейронная сеть")

st.markdown("Оценки модели:")
st.markdown("  - MAE: 2380.8796952802204")
st.markdown("  - MSE: 13229954.05725828")
st.markdown("  - RMSE: 3637.3003804000405")
st.markdown("  - MAPE: 1.0966710741455188")
st.markdown("  - R^2: 0.40651277857972923")

result3 = np.array([])

if st.button("Предсказать", key="neuro"):
    result3 = predictNeuro()

if result3.any():
    st.success("Модель предсказала {}".format(result3))
else:
    st.info("Здесь будет предсказание")
