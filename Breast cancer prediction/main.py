import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

def load_model():
    data = load_breast_cancer()
    X, y = data.data, data.target
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, data.feature_names, X

model, feature_names, X = load_model()

st.title('Breast Cancer Prediction App')

def user_input_features():
    user_data = {}
    for feature in feature_names:
        user_data[feature] = st.slider(feature, float(min(X[:, feature_names.tolist().index(feature)])), float(max(X[:, feature_names.tolist().index(feature)])), float(np.median(X[:, feature_names.tolist().index(feature)])))
    return pd.DataFrame(user_data, index=[0])

with st.form(key='user_input_form'):
    input_df = user_input_features()
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction')
    breast_cancer_types = np.array(['Malignant', 'Benign'])
    st.write(breast_cancer_types[prediction])

    st.subheader('Prediction Probability')
    fig, ax = plt.subplots()
    ax.pie(prediction_proba[0], labels=breast_cancer_types, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    ax.axis('equal')
    st.pyplot(fig)

    st.subheader('User Input Parameters')
    st.write(input_df)

    st.subheader('Feature Distributions')
    for feature in feature_names:
        fig, ax = plt.subplots()
        sns.histplot(X[:, feature_names.tolist().index(feature)], bins=30, kde=True, ax=ax)
        ax.axvline(input_df[feature].values[0], color='r', linestyle='--')
        ax.set_title(f'Distribution of {feature}')
        st.pyplot(fig)
