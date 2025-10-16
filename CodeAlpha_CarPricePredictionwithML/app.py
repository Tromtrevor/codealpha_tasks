import streamlit as st
import pandas as pd
from datetime import datetime as date
from sklearn.preprocessing import LabelEncoder
import joblib

def load_model():
    model = joblib.load('model/price_prediction_model.pkl')
    return model
    
def load_encoders():
    encoders = joblib.load('model/encoders.pkl')
    return encoders

def encoding(data):
    for col, encoder in encoders.items():
        data[col] = encoder.transform(data[col])
    return data

#Loading the model    
model = load_model()
encoders = load_encoders()

df = pd.read_csv('car data.csv')

cars = list(df['Car_Name'].unique())
fuel = list(df['Fuel_Type'].unique())
seller = list(df['Selling_type'].unique())
trans = list(df['Transmission'].unique())
year_now = date.now().year

st.title('CAR PRICE PREDICTION')

#User Input Columns
col1, col2, = st.columns(2)
with col1:
    car_name= st.selectbox('Car Name', cars)
    manufacture= st.selectbox('Manufacture Year', range(2000,year_now))
    transmission= st.selectbox('Transmission', trans)
    fuel_type= st.selectbox('Fuel type', fuel)  
with col2:
    selling_type= st.selectbox('Seller', seller)
    present_price= st.number_input('Current Price')
    driven_kms= st.number_input('Mileage', step=500)
    owner= st.number_input('Previous Owners', step=1)

age= year_now-manufacture

#User input to DataFrame
data = pd.DataFrame({
    'Car_Name': [car_name],
    'Present_Price': [present_price],
    'Driven_kms': [driven_kms],
    'Fuel_Type': [fuel_type],
    'Selling_type': [selling_type],
    'Transmission': [transmission],
    'Owner': owner,
    'Age': age   
    },
    index=[0]
)

#Encoding input data
input_data = encoding(data)

if st.button('Predict Price'):
    with st.spinner('Predicting...'):
        try:
            prediction = model.predict(input_data)[0]
            st.success(f'✅ Predicted Price: *{prediction:.2f}*')

        except Exception as e:
            st.error(f'Prediction failed: {e}')
