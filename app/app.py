import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model + features
model = joblib.load('models/model.pkl')
features = joblib.load('models/features.pkl')

st.title("🏠 House Price Prediction App")

st.write("Enter house details:")

# Example inputs (you can expand later)
overall_qual = st.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.number_input("Living Area (sq ft)", 500, 5000, 1500)
garage_cars = st.slider("Garage Cars", 0, 4, 1)

if st.button("Predict Price"):
    
    # Create full feature vector
    input_data = pd.DataFrame(np.zeros((1, len(features))), columns=features)
    
    # Fill selected features
    if 'OverallQual' in input_data.columns:
        input_data['OverallQual'] = overall_qual
        
    if 'GrLivArea' in input_data.columns:
        input_data['GrLivArea'] = gr_liv_area
        
    if 'GarageCars' in input_data.columns:
        input_data['GarageCars'] = garage_cars

    prediction = model.predict(input_data)
    
    st.success(f"Estimated Price: ${prediction[0]:,.2f}")