import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predict import HousePricePredictor

st.set_page_config(page_title="Home Value Estimator")

st.title("Home Value Estimator")
st.markdown("Enter house details to estimate its market value.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        sqft = st.number_input("Square Footage", min_value=500, max_value=10000, value=2000)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        floors = st.number_input("Floors", min_value=1, max_value=3, value=1)
        
    with col2:
        zipcode = st.selectbox("Zipcode", ['98001', '98002', '98003', '98004', '98005'])
        year_built = st.slider("Year Built", 1900, 2024, 2000)
        condition = st.slider("Condition (1-5)", 1, 5, 3)
        
    submitted = st.form_submit_button("Predict Price")
    
    if submitted:
        try:
            predictor = HousePricePredictor()
            input_data = {
                'sqft': sqft,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'floors': floors,
                'zipcode': zipcode,
                'year_built': year_built,
                'condition': condition
            }
            
            price = predictor.predict(input_data)
            st.success(f"Estimated Price: ${price:,.2f}")
            
        except Exception as e:
            st.error(f"Error: {e}")
