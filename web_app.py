# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:51:55 2023

@author: ValKo
"""

import streamlit as st
import pandas as pd
import pickle
import re
from sklearn.preprocessing import OrdinalEncoder

## load trained XGBoost model
with open("best_xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

## preprocess function
def preprocess_data(df):
    # store original index of entire DataFrame
    original_index = df.index
    # drop unnecessary columns
    df = df.drop(["emp_title", "title", "issue_d"], axis=1)

    # xxtract zip code from address
    df['zip_code'] = df['address'].apply(lambda address: re.search(r'\d{5}$', address).group(0) if re.search(r'\d{5}$', address) else None)
    df = df.drop("address", axis=1)

    # combine loan amount and installment
    df['installment_to_loan_ratio'] = df['installment'] / df['loan_amnt']
    df = df.drop(["installment", "loan_amnt"], axis=1)
    df['installment_to_loan_ratio'] *= 100

    # handle earliest credit line
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')
    reference_date = pd.Timestamp('2019-01-01')
    df['years_since_earliest_cr_line'] = (reference_date - df['earliest_cr_line']).dt.days / 365.25
    df = df.drop("earliest_cr_line", axis=1)

    # handle categorical variables
    df = df.drop("grade", axis=1)
    df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
    df = df[df['home_ownership'] != 'OTHER']
    df['application_type'] = df['application_type'].replace(['JOINT', 'DIRECT_PAY'], 'OTHER')
    df['purpose'] = df['purpose'].replace({
        'debt_consolidation': 'debt_management',
        'credit_card': 'debt_management',
        'home_improvement': 'home_related',
        'house': 'home_related',
        'major_purchase': 'major_expenses',
        'car': 'major_expenses',
        'medical': 'major_expenses',
        'wedding': 'major_expenses',
        'small_business': 'business_education',
        'educational': 'business_education',
        'moving': 'lifestyle',
        'vacation': 'lifestyle',
        'renewable_energy': 'lifestyle'
    })

    # ppen to total ratio
    df['open_to_total_ratio'] = df['open_acc'] / (df['total_acc'] + 1e-6)
    df.drop(columns=['open_acc', 'total_acc'], inplace=True, axis=1)

    # handling missing values
    df['emp_length'].fillna(0, inplace=True)
    df['revol_util'].fillna(df['revol_util'].median(), inplace=True)
    df['mort_acc'].fillna(df['mort_acc'].median(), inplace=True)
    df['pub_rec_bankruptcies'].fillna(df['pub_rec_bankruptcies'].median(), inplace=True)

    # binary encoding
    df['term'] = df['term'].map({' 36 months': 0, ' 60 months': 1})
    df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})
    df['initial_list_status'] = df['initial_list_status'].map({'f': 0, 'w': 1})
    df['application_type'] = df['application_type'].map({'INDIVIDUAL': 0, 'OTHER': 1})

    # ordinal encoding
    subgrade_ordering = ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5']
    emp_length_ordering = [0, '< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
    ordinal_encoder = OrdinalEncoder(categories=[subgrade_ordering, emp_length_ordering])
    encoded_values = ordinal_encoder.fit_transform(df[['sub_grade', 'emp_length']])
    df[['sub_grade', 'emp_length']] = encoded_values

    # dummy variables
    columns_to_convert = ['home_ownership', 'verification_status', 'purpose', 'zip_code']
    df[columns_to_convert] = df[columns_to_convert].astype('category')
    df = pd.get_dummies(df, columns=columns_to_convert, drop_first=True, dtype=int)

    # remove correlated features
    df = df.drop('pub_rec', axis=1)

    # keep track of remaining rows after preprocessing
    remaining_index = df.index

    return df, original_index, remaining_index


## Streamlit app
st.title('LendingClub Loan Default Prediction')

# ppload CSV file
uploaded_file = st.file_uploader("Choose a CSV file containing loan cases", type="csv")

if uploaded_file:
    # read uploaded file into a DataFrame
    input_data = pd.read_csv(uploaded_file)
    
    # display uploaded data for user's reference
    st.subheader('Uploaded Data')
    st.write(input_data)
    
    # display shape before preprocessing
    st.write("Shape before preprocessing:", input_data.shape)

    # preprocess input data
    preprocessed_data, original_index, remaining_index = preprocess_data(input_data.copy())

    # display the shape after preprocessing
    st.write("Shape after preprocessing:", preprocessed_data.shape)

    # drop "loan_status" column from preprocessed data
    preprocessed_data_for_prediction = preprocessed_data.drop(columns=['loan_status'])

    # display shape for prediction
    st.write("Shape for prediction:", preprocessed_data_for_prediction.shape)

    # filter original input data using the original index from preprocessing
    input_data_filtered = input_data.loc[remaining_index]

    # make predictions using preprocessed data
    predictions = model.predict(preprocessed_data_for_prediction)

    # create DataFrame with predictions and original index
    predictions_df = pd.DataFrame({'Predicted Loan Status': ['Fully Paid' if pred == 0 else 'Charged Off' for pred in predictions]}, index=remaining_index)

    # merge predictions with filtered original input data
    result_data = pd.concat([input_data_filtered, predictions_df], axis=1)

    # display results
    st.subheader('Predictions')
    st.write(result_data)






