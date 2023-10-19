# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:31:16 2023

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
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from collections import Counter

st.set_page_config(page_title="DiabetaPro",page_icon=":male-doctor:",layout="wide")

# Suppress warnings
warnings.filterwarnings("ignore")


# loading the saved models
loaded_model_SVM = pickle.load(open('C:/Users/Sithum/Desktop/SLIIT/Y3S1/Fundamentals of Data Mining/Mini project/trained_model_using_SVM.sav','rb'))

# loading the saved scaler
loaded_scaler = pickle.load(open('C:/Users/Sithum/Desktop/SLIIT/Y3S1/Fundamentals of Data Mining/Mini project/standard_scaler.sav','rb'))


def diabetes_prediction_SVM(input_data):
    

    #changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshaping the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


    #standardizing the input data
    std_data = loaded_scaler.transform(input_data_reshaped)
    print(std_data)

    prediction = loaded_model_SVM.predict(std_data)
    print(prediction)

    if prediction[0] == 0:
      return 'The patient is not suffering from diabetes'

    else:
      return 'The patient is suffering from diabetes'


    
  
def main():
    
    #sidebar for navigation
    with st.sidebar:
        
        selected = option_menu('Diabetes Prediction System',
                               ['Home','Summary Statistics','Data Visualization','Diabetes Prediction'],
                               icons = ['house','table','bar-chart-line','activity'],
                               default_index=0)
    #Home Page
    if selected == 'Home':
        st.title(":male-doctor::thought_balloon::hospital:")
        st.markdown("<h1 style='color: blue; text-align: center;'>Welcome to DiabetaPro!</h1>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            # Image
            st.image("https://junctionroadfp.com/wp-content/uploads/2022/04/shutterstock_1182539971.jpg", use_column_width=True, caption="Image: Diabetes")
        with col2:
            # Description 
            description = """
            <span style="font-size: 30px;"><strong>Understanding Diabetes:</strong></span>  
            Diabetes is a <strong>chronic medical condition</strong> characterized by elevated blood sugar levels, which can lead to severe health complications if not managed effectively. <strong>Early detection and timely intervention</strong> are crucial for individuals at risk of diabetes.
            
            <span style="font-size: 30px;"><strong>How Our Web App Can Help You:</strong></span>  
            
            <span style="font-size: 24px;"><strong>1. Early Detection and Personalized Insights</strong></span>  
            Our user-friendly interface allows you to input various health parameters, such as age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, blood glucose level.
            
            <span style="font-size: 24px;"><strong>2. Cutting-Edge Machine Learning Algorithms</strong></span>  
            Our web app employs advanced machine learning algorithms, including Support Vector Machine, Logistic Regression, Decision Tree, and Random Forest, to analyze your health data.
            
            <span style="font-size: 24px;"><strong>3. Personalized Risk Assessment</strong></span>  
            You'll receive a personalized prediction of your likelihood of developing diabetes, along with a confidence score.
            
            <span style="font-size: 24px;"><strong>4. Informed Decision-Making</strong></span>  
            Armed with insights from our web app, you can make informed decisions about your lifestyle, diet, and exercise routines to mitigate your diabetes risk.
            
            <span style="font-size: 24px;"><strong>5. Convenient and Secure</strong></span>  
            Accessible from any device with an internet connection, our web app ensures the privacy and security of your data.
            
            <span style="font-size: 24px;"><strong>Empower Yourself with Knowledge</strong></span>  
            Take control of your health by using our Diabetes Prediction Web App. It's time to proactively manage your diabetes risk, make informed choices, and enjoy a healthier future.
            """
            
            st.markdown(description, unsafe_allow_html=True)

    #Summary Statistics Page    
    if selected == 'Summary Statistics':
        diabetes_df = pd.read_csv('C:/Users/Sithum/Desktop/SLIIT/Y3S1/Fundamentals of Data Mining/Mini project/diabetes_prediction_dataset.csv')
        st.markdown("<h1 style='color: blue; text-align: left;'>Here's the dataset that we used to train our models!</h1>", unsafe_allow_html=True)
        fig = go.Figure(data=go.Table(
            header=dict(values=list(diabetes_df[['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level','diabetes']].columns),
                        fill_color='#04E474',align='center'),
            cells=dict(values=[diabetes_df.gender,diabetes_df.age,diabetes_df.hypertension,diabetes_df.heart_disease,diabetes_df.smoking_history,diabetes_df.bmi,diabetes_df.HbA1c_level,diabetes_df.blood_glucose_level,diabetes_df.diabetes],
                       fill_color='#E5ECF6',align='left')))
        fig.update_layout(margin=dict(l=5,r=5,b=10,t=10))
        st.write(fig)
        label_encoder = LabelEncoder()
        diabetes_df['gender'] = label_encoder.fit_transform(diabetes_df['gender'])
        diabetes_df['smoking_history'] = label_encoder.fit_transform(diabetes_df['smoking_history'])
        
        st.markdown("<h1 style='color: blue; text-align: left;'>Summary Statistics of the dataset</h1>", unsafe_allow_html=True)
        st.write(diabetes_df.describe())
    
    #Data Visualization Page
    if selected == 'Data Visualization':
        diabetes_df = pd.read_csv('C:/Users/Sithum/Desktop/SLIIT/Y3S1/Fundamentals of Data Mining/Mini project/diabetes_prediction_dataset.csv')
        st.markdown("<h1 style='color: blue; text-align: center;'>Data Visualization!</h1>", unsafe_allow_html=True)
        
        # Create the histogram
        plt.figure(figsize=(6,4))
        sns.histplot(data=diabetes_df, x='age', hue='diabetes', bins=10, kde=True)
        plt.xlabel('Age of patients')
        plt.ylabel('Frequency')
        plt.legend(title='Diabetes', labels=['No', 'Yes'])
        plt.title('Age Distribution with Diabetes')
        st.pyplot(plt)
        
        
        st.title('Gender vs. Diabetes')
        
        # Calculate counts of 'No Diabetes' and 'Diabetes'
        gender_diabetes_counts = diabetes_df['gender'].groupby(diabetes_df['diabetes']).count()
        
        # Create the pie chart
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(gender_diabetes_counts, labels=['No Diabetes', 'Diabetes'], autopct='%1.1f%%')
        ax.set_title('Gender vs. Diabetes')
        
        # Display the pie chart in Streamlit
        st.pyplot(fig)
   

        
    if selected == 'Diabetes Prediction':
        
        
        st.write('<div style="text-align: center;">', unsafe_allow_html=True)
        diabetes_df = pd.read_csv('C:/Users/Sithum/Desktop/SLIIT/Y3S1/Fundamentals of Data Mining/Mini project/diabetes_prediction_dataset.csv')
        # Removing duplicate records from the dataset
        diabetes_df = diabetes_df.drop_duplicates()
        label_encoder = LabelEncoder()
        diabetes_df['gender'] = label_encoder.fit_transform(diabetes_df['gender'])
        diabetes_df['smoking_history'] = label_encoder.fit_transform(diabetes_df['smoking_history'])
        
        #giving a title
        st.title('Diabetes Checkup:male-doctor:')
        
        # seperating the data and labels
        X = diabetes_df.drop(columns = 'diabetes', axis=1)
        Y = diabetes_df['diabetes']
        
        standardized_data = loaded_scaler.fit_transform(X)
        X = standardized_data
        Y = diabetes_df['diabetes']
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y, random_state = 2)
        #Fitting SMOTE to the training set
        smote = SMOTE(random_state = 27)
        smote_X_train, smote_Y_train = smote.fit_resample(X_train,Y_train)
        
        st.subheader('Accuracy of the checkup: ')
        accuracy = accuracy_score(Y_test, loaded_model_SVM.predict(X_test))
        
        # Create a string with the accuracy score and format it with a green background
        formatted_accuracy = f'<span style="background-color: #04E474; padding: 5px; color: white;">{accuracy*100:.2f}%</span>'
        
        # Display the formatted accuracy score using st.markdown
        st.markdown(formatted_accuracy, unsafe_allow_html=True)

        
        st.subheader('Enter Patients Details:')
        #getting the input from the user
        col1, col2 = st.columns([1, 1])
        with col1:
            
            # Gender selection
            gender = st.radio("Select Gender", ("Female", "Male"))
            gender_code = 1 if gender == "Male" else 0
            
            
            
            # Hypertension selection
            hypertension = st.radio("Do you have Hypertension?", ("No", "Yes"))
            hypertension_code = 1 if hypertension == "Yes" else 0
            
            # Heart Disease selection
            heart_disease = st.radio("Do you have Heart Disease?", ("No", "Yes"))
            heart_disease_code = 1 if heart_disease == "Yes" else 0
        with col2:
            
            # Age input
            age = st.text_input("Enter Age", "")
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
        diagnosis_SVM = ''
        
        # Button to trigger prediction
        if st.button("Checkup Result"):
            diagnosis_SVM = diabetes_prediction_SVM([gender_code,age,hypertension_code, heart_disease_code,smoking_history_code,bmi,hba1c_level,blood_glucose_level])
            
        st.success(diagnosis_SVM)
        st.write('</div>', unsafe_allow_html=True)
    
        
if __name__ == '__main__':
    main()

    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    