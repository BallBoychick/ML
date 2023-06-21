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
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

def prepocessing(data):
    
    scaler = joblib.load("../models/scaler_for_class")
    df_majority_hight = data[data.Diabetes_012==0.0]
    df_majority_medium = data[data.Diabetes_012==1.0]
    frames = [df_majority_hight, df_majority_medium]
    df_majority = pd.concat(frames)
    df_minority = data[data.Diabetes_012==2.0]

    df_majority_downsampled = resample(df_majority,  
                                 replace=True,
                                 n_samples=35097,
                                 random_state=123)
 
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    df_down=df_downsampled.sample(frac=1)
    X_down = df_down.drop("Diabetes_012", axis=1)
    Y_down = df_down["Diabetes_012"]

    oversample = SMOTE()
    X_smote, y_smote = oversample.fit_resample(X_down, Y_down)

    scaler.fit(X_smote)

    X_train_smote_norm = pd.DataFrame(scaler.transform(X_smote), columns=X_smote.columns)
    balanced_scaler_dataset = pd.concat([X_train_smote_norm, y_smote], axis = 1)

    return balanced_scaler_dataset

def prepocessing_exp(data):
    scaler = joblib.load("../models/scaler_for_class")
    scaler.fit(data)
    return data

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
    selected2 = option_menu(None, ["Информация", "Визуализация", 'Предсказания', 'Предсказания на своем файле'], 
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
        feature = st.selectbox('Выберите признак для построения графика плотности:', data.columns)
        ds = data[data["Diabetes_012"] == 2][feature]
        ds2 = data[data["Diabetes_012"] == 0][feature]
        ds3 = data[data["Diabetes_012"] == 1][feature]
        fig = px.box(ds)
        fig2 = px.box(ds2)
        fig3 = px.box(ds3)
        st.write("DIABET")
        st.plotly_chart(fig)
        st.write("NODIABET")
        st.plotly_chart(fig2)
        st.write("PREDIABET")
        st.plotly_chart(fig3)

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
    
    df = pd.DataFrame(
    [
    {"HighBP": "0", "HighChol": "0", "CholCheck": "0", "BMI":"22.0", 'Smoker': "0", 'Stroke': "0",
       'HeartDiseaseorAttack' : "0", 'PhysActivity': "1", 'Fruits':"1", 'Veggies': "0",
       'HvyAlcoholConsump': "0", 'AnyHealthcare':"1", 'NoDocbcCost':"0", 'GenHlth':"4",
       'MentHlth':"30", 'PhysHlth': "25", 'DiffWalk': "28", 'Sex': "1", 'Age': "20", 'Education':"3", 'Income':"3",},
    ]
    )
    edited_df = st.experimental_data_editor(df, num_rows="dynamic")
    if st.button('Предсказать'):
        pred = model.predict(prepocessing_exp(edited_df))
        st.write(pred)

if selected2 == "Предсказания на своем файле":

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
        sc_data = prepocessing(df)
        y = sc_data["Diabetes_012"]
        X = sc_data.drop(["Diabetes_012"], axis=1)
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
