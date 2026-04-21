import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and features
model = joblib.load('models/model.pkl')
features = joblib.load('models/features.pkl')

# Page config
st.set_page_config(page_title="House Price Predictor", layout="centered")

# Title
st.title("🏠 House Price Prediction App")
st.markdown("Predict house prices using Machine Learning")

st.divider()

# Sidebar Inputs
st.sidebar.header("Enter House Details")

overall_qual = st.sidebar.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Living Area (sq ft)", 500, 5000, 1500)
garage_cars = st.sidebar.slider("Garage Cars", 0, 4, 1)

st.subheader("Prediction")

if st.button("Predict Price"):
    
    input_data = pd.DataFrame(np.zeros((1, len(features))), columns=features)

    if 'OverallQual' in input_data.columns:
        input_data['OverallQual'] = overall_qual
        
    if 'GrLivArea' in input_data.columns:
        input_data['GrLivArea'] = gr_liv_area
        
    if 'GarageCars' in input_data.columns:
        input_data['GarageCars'] = garage_cars

    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Price: ${prediction[0]:,.2f}")

st.divider()

st.markdown("Built with ❤️ using Machine Learning")