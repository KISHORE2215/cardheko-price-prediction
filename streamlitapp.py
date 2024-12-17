# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# Streamlit page configuration
st.set_page_config(layout="wide", page_title="CarDekho Price Prediction", page_icon="🚗")

# Display logo
st.image(r"C:\Users\HP\OneDrive\Desktop\Guvi\project_3\1626342082_cardekho_logo_startuptalky_jpg-1.jpg")

# Sidebar menu for navigation
page = st.sidebar.selectbox("Select a Page", ["CarDekho-Price Prediction", "User Guide"])

# CarDekho-Price Prediction Page
if page == "CarDekho-Price Prediction":
    st.header(':blue[CarDekho-Price Prediction]')
    
    # Load data
    try:
        df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Guvi\project_3\filtered_data.xls")
    except Exception as e:
        st.error(f"Error loading data: {e}")
    
    # Input fields for car details
    col1, col2 = st.columns(2)

    with col1:
        Ft = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'LPG', 'CNG', 'Electric'])
        Bt = st.selectbox("Body Type", ['Hatchback', 'SUV', 'Sedan', 'MUV', 'Coupe', 'Minivans',
                                        'Convertibles', 'Hybrids', 'Wagon', 'Pickup Trucks'])
        Tr = st.selectbox("Transmission", ['Manual', 'Automatic'])
        Owner = st.selectbox("Owner", [0, 1, 2, 3, 4, 5])
        Brand = st.selectbox("Brand", options=df['Brand'].unique())

        # Filter models dynamically based on the selected brand, body type, and fuel type
        filtered_models = df[(df['Brand'] == Brand) & (df['body type'] == Bt) & (df['Fuel type'] == Ft)]['model'].unique()
        Model = st.selectbox("Model", options=filtered_models)
        
        Model_year = st.selectbox("Model Year", options=sorted(df['modelYear'].unique()))
        IV = st.selectbox("Insurance Validity", ['Third Party', 'Comprehensive', 'Zero Dep', 'Not Available'])
        Km = st.slider("Kilometers Driven", min_value=100, max_value=100000, step=1000)
        ML = st.number_input("Mileage (km/l)", min_value=5, max_value=50, step=1)
        Seats = st.selectbox("Seats", options=sorted(df['Seats'].unique()))
        Color = st.selectbox("Color", options=df['Color'].unique())
        City = st.selectbox("City", options=df['City'].unique())

    with col2:
        Submit = st.button("Predict")

        if Submit:
            try:
                # Load pre-trained model, scaler, and encoder
                with open(r'C:\Users\HP\OneDrive\Desktop\Guvi\project_3\Randomforest_regression.pkl', 'rb') as model_file:
                    final_model = pickle.load(model_file)
                with open(r'C:\Users\HP\OneDrive\Desktop\Guvi\project_3\standard.pkl', 'rb') as scaler_file:
                    scaler = pickle.load(scaler_file)
                with open(r'C:\Users\HP\OneDrive\Desktop\Guvi\project_3\encoder.pkl', 'rb') as encoder_file:
                    encoder = pickle.load(encoder_file)

                # Prepare input data
                cat_cols = ['Fuel type', 'body type', 'transmission', 'Brand', 'model', 'Insurance Validity', 'Color', 'City']
                num_cols = ['ownerNo', 'modelYear', 'Kms Driven', 'Mileage', 'Seats']

                # Create input DataFrame
                input_data = pd.DataFrame([{
                    'Fuel type': Ft,
                    'body type': Bt,
                    'transmission': Tr,
                    'Brand': Brand,
                    'model': Model,
                    'Insurance Validity': IV,
                    'Color': Color,
                    'City': City,
                    'ownerNo': Owner,
                    'modelYear': Model_year,
                    'Kms Driven': Km,
                    'Mileage': ML,
                    'Seats': Seats
                }])

                # Process categorical features
                cat_transformed = encoder.transform(input_data[cat_cols])
                encoder_feature_names = encoder.get_feature_names_out(cat_cols)
                
                # Align feature names with encoder's output
                cat_transformed_df = pd.DataFrame(cat_transformed, columns=encoder_feature_names)
                for col in encoder_feature_names:
                    if col not in cat_transformed_df.columns:
                        cat_transformed_df[col] = 0
                cat_transformed_df = cat_transformed_df[encoder_feature_names]

                # Process numerical features
                num_transformed = scaler.transform(input_data[num_cols])

                # Combine processed features
                final_input = np.hstack((num_transformed, cat_transformed_df.values))

                # Make prediction
                prediction = final_model.predict(final_input)

                # Display result
                st.success(f"The price of the {Brand} car is: **{round(prediction[0], 2)} lakhs**")

            except Exception as e:
                st.error(f"Error during prediction: {e}")

# User Guide Page
elif page == "User Guide":
    st.header('User Guide for Streamlit-based CarDekho Price Prediction Application')
    st.write("""
    ### Welcome to the CarDekho Price Prediction Application!
    
    This guide explains how to use the application to predict car prices. 

    **Steps to Use the App:**
    1. Navigate to the **CarDekho-Price Prediction** page.
    2. Fill in all the car details such as:
       - Fuel Type, Body Type, Transmission, Owner Count, Brand, Model, Year, Insurance Validity, etc.
    3. Press the **Predict** button to get the price estimate.
    4. The app will display the predicted price in lakhs.
    
    **Enjoy using the app! 🚗**
    """)
