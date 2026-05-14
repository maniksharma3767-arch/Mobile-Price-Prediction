import streamlit as st
import numpy as np
import pickle
import pandas as pd


model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("features.pkl", "rb"))

st.title("Mobile Price Range Predictor")


important_features = ['ram', 'battery_power', 'px_height', 'px_width', 'mobile_wt']

input_dict = {}

for feature in important_features:
    if feature in ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']:
        val = st.selectbox(feature, ["No", "Yes"])
        val = 1 if val == "Yes" else 0
    else:
        val = st.number_input(feature, value=1.0)

    input_dict[feature] = val


for feature in feature_names:
    if feature not in input_dict:
        input_dict[feature] = 0  

if st.button("Predict"):
    data = pd.DataFrame([input_dict])
    

    data = data[feature_names]
    
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    
    st.success(f"Predicted Price Range: {prediction}")