import streamlit as st
import joblib
import numpy as np

scalar= joblib.load("scaler.pkl")
model= joblib.load("model.pkl")
st.title("Customer Car Price Estimator App")

st.divider()
st.write("""This app is for getting a price estimation for the customer so a car with the 
         price range given can be advised to the customer.""")

age = st.number_input("Enter the age", min_value=18, max_value=90, value=40, step=1)
salary= st.number_input("Enter the salary", min_value=1000, max_value=99999999, step=5000, value=30000)
networth= st.number_input("Enter the networth",min_value=0, max_value=9999999, step= 20000, value=100000)

X= [age,salary,networth]

calculatebutton = st.button("Calculate")
st.divider()

if calculatebutton:

    st.balloons()
    
    X_2 = np.array(X)

    X_array=scalar.transform([X_2])

    prediction= model.predict(X_array)

    st.write(f"Prediction is {prediction[0][0]:,.2f}")
    st.write("Advice Cars in the Similar Values")

else:
    st.write("Please enter the values and press the calculated button")

