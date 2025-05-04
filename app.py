import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
@st.cache_resource
def load_model():
    with open("global_model (4).pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Input feature names (without target)
feature_names = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

def predict(data):
    prediction = model.predict(data)
    probability = model.predict_proba(data)[:, 1]
    return prediction, probability

st.title("Credit Card Default Prediction")

option = st.sidebar.selectbox("Choose Prediction Type", ["Single Prediction", "Batch Prediction"])

if option == "Single Prediction":
    st.header("Enter Customer Information")

    input_data = []
    for feature in feature_names:
        value = st.number_input(f"{feature}", value=0.0)
        input_data.append(value)

    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        prediction, probability = predict(input_array)
        st.write(f"### Prediction: {'Default' if prediction[0]==1 else 'No Default'}")
        st.write(f"### Probability of Default: {probability[0]:.2f}")

elif option == "Batch Prediction":
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file with the required columns", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if 'default payment next month' in df.columns:
            df = df.drop('default payment next month', axis=1)

        if all(col in df.columns for col in feature_names):
            prediction, probability = predict(df[feature_names])
            df['Prediction'] = prediction
            df['Probability'] = probability
            st.write("### Results:")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", csv, "predictions.csv", "text/csv")
        else:
            missing_cols = set(feature_names) - set(df.columns)
            st.error(f"The following required columns are missing: {', '.join(missing_cols)}")
