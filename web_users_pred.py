import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import joblib
import pickle

import os

# to make this notebook's output stable across runs
np.random.seed(42)


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import pointbiserialr, chi2_contingency, f_oneway, ttest_ind
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import layers


import warnings
warnings.filterwarnings(action='ignore')





st.title('Milestone 1 Phase 2')
st.write("""
Created by Mitra Marona Putra Gurusinga BATCH HCK06


         
Use the sidebar to input customer data.
""")
@st.cache_data
def fetch_data():
    df = pd.read_csv('df_fe.csv')
    return df

df = fetch_data()


avg_transaction_value = st.number_input('Average Transaction Value', 0.0)
avg_frequency_login_days = st.number_input('Average Frequency Login Days', 0.0)
points_in_wallet = st.number_input('Points in Wallet', 0.0)
membership_category = st.selectbox('Membership Category', df['membership_category'].unique())
feedback = st.selectbox('Feedback', df['feedback'].unique())




data = {
    'avg_transaction_value': avg_transaction_value,
    'avg_frequency_login_days': avg_frequency_login_days,
    'points_in_wallet': points_in_wallet,
    'membership_category': membership_category,
    'feedback': feedback,
    }
input = pd.DataFrame(data, index=[0])
review = pd.DataFrame(data, index=['Your Input'])

st.subheader('Customer Data')
st.write('Please make sure your input is right',)
st.write(review.T)

model = joblib.load("churn_ann_pred.pkl")


if st.button('Predict'):
    
    prediction = model.predict(input)
    res_pred = np.where(prediction >= 0.5, 1, 0)


    if res_pred == 1:
        res_pred = 'Churn'
    else:
        res_pred = 'Stay'

    st.write('Based on Your Input, this Customer will: ')
    st.write(res_pred)