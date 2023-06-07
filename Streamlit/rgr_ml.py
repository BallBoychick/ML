import streamlit as st
import pandas as pd
import pickle 
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def get_data():
    data = pd.read_csv('../Data/balanced_sclaer_dataset_diabetes.csv')
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    print(data.info())
    return data

with open('info.md', 'r',encoding="utf-8") as f:
    markdown_string = f.read()

st.sidebar.title("Решение задачи классификации для предсказания стадии диабета")
st.sidebar.info(
    "Ссылка на датасет: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_012_health_indicators_BRFSS2015.csv"
)
st.sidebar.info("Мой реп "
                "[github](https://github.com/BallBoychick/ML)")

with st.sidebar: 
    selected2 = option_menu(None, ["Информация", "Визуализация", 'Предсказания'], 
        icons=['info', 'bi bi-graph-up-arrow', "bi bi-file-earmark-arrow-down-fill"], 
        menu_icon="cast", default_index=0)
    selected2


if selected2 == "Информация":
    st.markdown(markdown_string, unsafe_allow_html=True)
    video_url = '../mega_diab.mp4'
    st.video(video_url)
    st.write(get_data())
if selected2 == "Визуализация":
    selected3 = option_menu(None, ["Heatmap", "Density plot",  "Boxplot", "Box Plot 2.0"], 
    icons=['bi bi-caret-right-square-fill', 'bi bi-caret-right-square-fill', "bi bi-caret-right-square-fill", 'bi bi-caret-right-square-fill'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
    data = get_data()
    if selected3 == "Heatmap":
        fig, ax = plt.subplots()
        sns.heatmap(data.drop(data.columns[0],axis=1).corr(), ax=ax)
        st.write(fig)
    if selected3 == "Density plot":
        fig, ax = plt.subplots()
        feature = st.selectbox('Выберите признак для построения графика плотности:', data.columns)
        sns.kdeplot(data=data[feature], ax=ax)
        st.write(fig)
    if selected3 == "Boxplot":
        fig = px.box(data.iloc[:10000].drop(["Diabetes_012",data.columns[0]],axis=1))
        st.plotly_chart(fig)
    if selected3 == "Box Plot 2.0":
        group = st.selectbox('Выберите группу для построения Box Plot:', ["Диабет", "Нетдиабета", "ПреДиабет"])
        feature = st.selectbox('Выберите признак для построения графика плотности:', data.columns)
        if group == "Диабет":
            ds = data[data["Diabetes_012"] == 2][feature]
            fig = px.box(ds)
        if group == "Нетдиабета":
            ds = data[data["Diabetes_012"] == 0][feature]
            fig = px.box(ds)
        else:
            ds = data[data["Diabetes_012"] == 1][feature]
            fig = px.box(ds)

        st.plotly_chart(fig)

if selected2 == "Предсказания":
    option = st.selectbox(
    'Выберите модель обучения',
    ('Knn', 'BaggingClassifier', 'Keras'))
    if option == 'Knn':
        model = pickle.load(open('../models/knnpickle_file', 'rb'))
    if option == 'BaggingClassifier':
        model = pickle.load(open('../models/optimal_bagging_file', 'rb'))
    if option == 'Keras':
        model = tf.keras.models.load_model('../models/keras_mode_file.h5')
        
    uploaded_file = st.file_uploader("Choose a file", type="csv")
    
    if st.button('Предсказать'):
        df = pd.read_csv(uploaded_file)
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        y = df["Diabetes_012"]
        X = df.drop(["Diabetes_012"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)
        if option == 'Keras':
            y_pred_arg = [np.argmax(pred) for pred in model.predict(X_test, verbose=None)]
            st.write("Look at this accuracy: ")
            cs2 = accuracy_score(y_test, y_pred_arg)
            st.write(cs2)
            st.write(y_pred_arg)
        else:
            y_pred = model.predict(X_test)
            st.write(y_pred)
            cs = accuracy_score(y_test, y_pred)
            st.write("Look at this accuracy: ")
            st.write(cs)