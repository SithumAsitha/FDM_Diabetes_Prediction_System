# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:31:03 2023

@author: Sithum
"""
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import warnings
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Suppress warnings
warnings.filterwarnings("ignore")


# loading the saved model
loaded_model = pickle.load(open('C:/Users/Sithum/Desktop/SLIIT/Y3S1/Fundamentals of Data Mining/Mini project/trained_model_using_SVM.sav','rb'))
# loading the saved scaler
loaded_scaler = pickle.load(open('C:/Users/Sithum/Desktop/SLIIT/Y3S1/Fundamentals of Data Mining/Mini project/standard_scaler.sav','rb'))


def diabetes_prediction(input_data):
    

    #changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshaping the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


    #standardizing the input data
    std_data = loaded_scaler.transform(input_data_reshaped)
    print(std_data)

    prediction = loaded_model.predict(std_data)
    print(prediction)

    if prediction[0] == 0:
      return 'The patient is not suffering from diabetes'

    else:
      return 'The patient is suffering from diabetes'
  
    
  
def main():
    
    #giving a title
    st.title('Diabetes Checkup Using Support Vector Machine')
    
    diabetes_df = pd.read_csv('C:/Users/Sithum/Desktop/SLIIT/Y3S1/Fundamentals of Data Mining/Mini project/diabetes_prediction_dataset.csv')
    label_encoder = LabelEncoder()
    diabetes_df['gender'] = label_encoder.fit_transform(diabetes_df['gender'])
    diabetes_df['smoking_history'] = label_encoder.fit_transform(diabetes_df['smoking_history'])
    
    st.subheader('Summary Statistics')
    st.write(diabetes_df.describe())
    
    st.subheader('Data Visualisation')
    # Create scatter plot
    st.subheader('Scatter Plot')
    x_axis = st.selectbox('Select X-Axis:', diabetes_df.columns)
    y_axis = st.selectbox('Select Y-Axis:', diabetes_df.columns)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=diabetes_df, x=x_axis, y=y_axis, ax=ax)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    st.pyplot(fig)
    
    # seperating the data and labels
    X = diabetes_df.drop(columns = 'diabetes', axis=1)
    Y = diabetes_df['diabetes']
    
    standardized_data = loaded_scaler.fit_transform(X)
    X = standardized_data
    Y = diabetes_df['diabetes']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y, random_state = 2)
    
    st.subheader('Accuracy of the model: ')
    st.write(str(accuracy_score(Y_test,loaded_model.predict(X_test))*100)+'%')
    
    st.subheader('Enter Patients Details')
    #getting the input from the user
    
    # Gender selection
    gender = st.radio("Select Gender", ("Female", "Male"))
    gender_code = 1 if gender == "Male" else 0
    
    # Age input
    age = st.text_input("Enter Age", "")
    
    # Hypertension selection
    hypertension = st.radio("Do you have Hypertension?", ("No", "Yes"))
    hypertension_code = 1 if hypertension == "Yes" else 0
    
    # Heart Disease selection
    heart_disease = st.radio("Do you have Heart Disease?", ("No", "Yes"))
    heart_disease_code = 1 if heart_disease == "Yes" else 0
    
    # Smoking history selection
    smoking_options = {
        'No info': 0,
        'Current': 1,
        'Ever': 2,
        'Former': 3,
        'Never': 4,
        'Not current': 5
    }
    smoking_history = st.selectbox("Smoking History", list(smoking_options.keys()))
    smoking_history_code = smoking_options[smoking_history]
    
    # BMI input
    bmi = st.text_input("Enter BMI", "")
    
    # HbA1c level input
    hba1c_level = st.text_input("Enter HbA1c Level", "")
    
    # Blood glucose level input
    blood_glucose_level = st.text_input("Enter Blood Glucose Level", "")
    
    #code for prediction
    diagnosis = ''
    
    # Button to trigger prediction
    if st.button("Checkup Result"):
        diagnosis = diabetes_prediction([gender_code,age,hypertension_code, heart_disease_code,smoking_history_code,bmi,hba1c_level,blood_glucose_level])
        
    st.success(diagnosis)
    
        
if __name__ == '__main__':
    main()



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    