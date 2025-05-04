import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained global model
with open('global_model (4).pkl', 'rb') as file:
    model = pickle.load(file)

# Define input features based on training
feature_names = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]
data['default.payment.next.month'] = 0  # Dummy value
prediction, probability = predict(data)


st.title("Credit Card Default Prediction")

option = st.sidebar.selectbox("Choose Prediction Type", ["Single Prediction", "Batch Prediction"])

def predict(data):
    prediction = model.predict(data)
    probability = model.predict_proba(data)[:, 1]
    return prediction, probability

# ---- SINGLE PREDICTION ----
if option == "Single Prediction":
    st.header("Enter Customer Information")

    input_data = []
    for feature in feature_names:
        value = st.number_input(f"{feature}", step=1.0, format="%.2f")
        input_data.append(value)

    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        prediction, probability = predict(input_array)
        st.write(f"### Prediction: {'Default' if prediction[0]==1 else 'No Default'}")
        st.write(f"### Probability of Default: {probability[0]:.2f}")

# ---- BATCH PREDICTION ----
elif option == "Batch Prediction":
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(data.head())

        if all(col in data.columns for col in feature_names):
            prediction, probability = predict(data[feature_names])
            data['Prediction'] = prediction
            data['Probability'] = probability
            st.write("### Prediction Results")
            st.dataframe(data)

            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", csv, "prediction_results.csv", "text/csv")
        else:
            st.error("Uploaded CSV must contain all required feature columns.")

