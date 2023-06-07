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

# def preprocessing(data):
#     if 'Flight' in data.columns:
#         data = data.drop('id',axis=1)
#         data = data.drop('Flight',axis=1)
#     X = data
#     bn = joblib.load("binary_encoder.joblib")
#     X['Airline'] = bn.fit_transform(X['Airline'])
#     X['AirportFrom'] =  bn.fit_transform(X['Airline'])
#     X['AirportTo'] = bn.fit_transform(X['AirportTo'])
#     scaler = joblib.load("scaler.joblib")

#     scaler.fit(X[["DayOfWeek","Time","Length"]])
#     X[["DayOfWeek","Time","Length"]]= scaler.transform(X[["DayOfWeek","Time","Length"]])
#     return X

def get_data():
    data = pd.read_csv('../Data/balanced_sclaer_dataset_diabetes.csv')
    # for i in data.columns:
    #         data[i] = data[i].astype(float)
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    print(data.info())
    return data

with open('info.md', 'r',encoding="utf-8") as f:
    markdown_string = f.read()

st.sidebar.title("Решение задачи классификации для предсказания задержки рейсов")
st.sidebar.info(
    "Ссылка на датасет: NONE"
)
st.sidebar.info("Мой "
                "NONE")

with st.sidebar: 
    selected2 = option_menu(None, ["Информация", "Визуализация", 'Предсказания'], 
        icons=['info', 'bi bi-graph-up-arrow', "bi bi-file-earmark-arrow-down-fill"], 
        menu_icon="cast", default_index=0)
    selected2


if selected2 == "Информация":
    st.markdown(markdown_string, unsafe_allow_html=True)
# if selected2 == "Визуализация":
#     selected3 = option_menu(None, ["Heatmap", "Диаграмма рассеяния",  "Boxplot", 'ROC Кривая'], 
#     icons=['bi bi-caret-right-square-fill', 'bi bi-caret-right-square-fill', "bi bi-caret-right-square-fill", 'bi bi-caret-right-square-fill'], 
#     menu_icon="cast", default_index=0, orientation="horizontal")
#     data = get_data()
#     if selected3 == "Heatmap":
#         fig, ax = plt.subplots()
#         sns.heatmap(data.drop(data.columns[0],axis=1).corr(), ax=ax)
#         st.write(fig)
#     if selected3 == "Диаграмма рассеяния":
#         datasmal = data.iloc[:10000].drop(data.columns[0],axis=1)
#         fig = ff.create_scatterplotmatrix(datasmal[['Time','Length']], diag='histogram', height=800, width=800)
#         st.plotly_chart(fig)
#     if selected3 == "Boxplot":
#         fig = px.box(data.iloc[:10000].drop(["Delay",data.columns[0]],axis=1))
#         st.plotly_chart(fig)
#     if selected3 == "ROC Кривая":
#         model =  pickle.load(open('../../models/class_model', 'rb'))
#         X_test = pd.read_csv('toapp.csv')
#         y_pred = np.array(model.predict( preprocessing(X_test)) )
#         y_test = np.array(pd.read_csv('y_test.csv'))
#         fpr, tpr, thresholds = roc_curve(y_test, y_pred)
#         roc_auc = auc(fpr, tpr)
#         fig, ax = plt.subplots()
#         ax =plt.plot(fpr, tpr, color='darkorange',
#          label='ROC кривая (area = %0.02f)' % roc_auc)
#         plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('ROC-кривая')
#         plt.legend(loc="lower right")
#         st.pyplot(fig)
#         print('')
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
    # df = pd.DataFrame(
    # [
    #    {"Airline": "UA", "AirportFrom": "IAH", "AirportTo": "CHS", "DayOfWeek":"4","Time":"1195","Length":"131",},
    #  ]
    # )
    # edited_df = st.experimental_data_editor(df, num_rows="dynamic")
    
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

    # if uploaded_file is not None:
    #     upladdata = pd.read_csv(uploaded_file)
    #     print(upladdata)
    #     y_pred = model.predict(preprocessing(upladdata)) 
    #     if len(y_pred.shape)>1:
    #         y_pred = np.argmax(y_pred,axis = 1)
    #     y_pred = pd.DataFrame( y_pred)
    #     y_pred.columns = ['Delay']
    #     ans = pd.concat([upladdata, y_pred],axis =1)
    #     st.write(ans)