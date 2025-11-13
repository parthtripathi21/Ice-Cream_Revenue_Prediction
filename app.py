import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model/model.pkl", "rb"))

st.title("🍦 Ice Cream Revenue Prediction")
temp = st.number_input("Enter Temperature (°C)", value=30.0)

if st.button("Predict Revenue"):
    result = model.predict(np.array([[temp]]))[0]
    st.success(f"Predicted Revenue: ₹{result:.2f}")
