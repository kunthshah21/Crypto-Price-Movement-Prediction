import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Crypto Price Movement Prediction",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Crypto Price Movement Prediction")
st.markdown("""
This app predicts whether cryptocurrency prices will go up ⬆️ or down ⬇️ based on market indicators.
Choose your preferred timeframe and input the required data to get a prediction.
""")

# Create directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_dir = "notebooks/models"  # Updated path to the correct location
    try:
        models["daily"] = tf.keras.models.load_model(os.path.join(model_dir, "prediction_daily.keras"))
        models["monthly"] = tf.keras.models.load_model(os.path.join(model_dir, "prediction_monthly.keras"))
        models["yearly"] = tf.keras.models.load_model(os.path.join(model_dir, "prediction_yearly.keras"))
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Load the models
models = load_models()

# Sidebar for model selection
st.sidebar.title("Prediction Settings")
timeframe = st.sidebar.selectbox(
    "Select Prediction Timeframe", 
    ["Daily", "Monthly", "Yearly"]
)

# Main form for data input
st.header("Input Market Data")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    Open_Price = st.number_input("Opening Price", min_value=0.0, format="%.2f")
    High_Price = st.number_input("Highest Price", min_value=0.0, format="%.2f")
    Low_Price = st.number_input("Lowest Price", min_value=0.0, format="%.2f")
    Volume = st.number_input("Trading Volume", min_value=0, step=1000000)
    MA_5 = st.number_input("5-Day Moving Average", min_value=0.0, format="%.2f")
    MA_10 = st.number_input("10-Day Moving Average", min_value=0.0, format="%.2f")

with col2:
    RSI = st.number_input("Relative Strength Index (RSI)", min_value=0.0, max_value=100.0, format="%.2f", value=50.0)
    Volatility = st.number_input("Volatility", min_value=0.0, format="%.4f", value=0.0200)
    Sentiment_Score = st.number_input("Market Sentiment Score", min_value=-1.0, max_value=1.0, format="%.2f", value=0.0)
    Global_Economy = st.number_input("Global Economy Indicator", min_value=0.0, max_value=5.0, format="%.2f", value=2.5)
    Event_Impact = st.number_input("Event Impact Score", min_value=-1.0, max_value=1.0, format="%.2f", value=0.0)

# Prediction button
predict_button = st.button("Predict Price Movement")

# Function to make prediction
def make_prediction(input_data, model_type):
    if models is None:
        st.error("Models failed to load. Please check the model files.")
        return None
    
    # Convert input to numpy array
    input_array = np.array([input_data])  # Shape: (1, 11)
    
    # Standardize the input
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_array)  # Shape: (1, 11)
    
    # Select the appropriate model
    if model_type.lower() == "daily":
        model = models["daily"]
        # The daily model expects only 9 features - exclude the last 2 features
        input_scaled = input_scaled[:, :9]
    elif model_type.lower() == "monthly":
        model = models["monthly"]
    else:  # yearly
        model = models["yearly"]
    
    # Special handling for different models based on their expected input shapes
    if model_type.lower() == "monthly":
        # For monthly model, create a sequence with 60 timesteps
        window_size = 60  # from the monthly notebook
        input_final = np.repeat(input_scaled.reshape(1, 1, input_scaled.shape[1]), window_size, axis=1)
    elif model_type.lower() == "yearly":
        # For yearly model, which expects a sequence of 10 timesteps
        window_size = 10  # from the yearly notebook
        input_final = np.repeat(input_scaled.reshape(1, 1, input_scaled.shape[1]), window_size, axis=1)
    elif model_type.lower() == "daily":
        # For daily model, keep a single timestep but with the 9 features it expects
        input_final = input_scaled.reshape(1, 1, input_scaled.shape[1])
    
    # Make prediction
    prediction = model.predict(input_final, verbose=0)
    return prediction[0][0]  # Return the probability value

# Display prediction results when button is clicked
if predict_button:
    if models:
        # Collect all inputs into a list
        input_data = [
            Open_Price, High_Price, Low_Price, Volume, MA_5, 
            MA_10, RSI, Volatility, Sentiment_Score, Global_Economy, Event_Impact
        ]
        
        # Make prediction based on selected timeframe
        with st.spinner(f"Making {timeframe.lower()} prediction..."):
            prediction = make_prediction(input_data, timeframe.lower())
        
        # Display results
        st.header("Prediction Result")
        
        if prediction is not None:
            # Determine if price will go up (1) or down (0)
            prediction_binary = int(prediction > 0.5)
            confidence = prediction if prediction_binary == 1 else (1-prediction)
            
            # Create columns for result display
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                # Display prediction
                if prediction_binary == 1:
                    st.success(f"### PRICE WILL GO UP ⬆️")
                else:
                    st.error(f"### PRICE WILL GO DOWN ⬇️")
                
                st.metric("Confidence", f"{confidence*100:.2f}%")
                st.metric("Timeframe", timeframe)
            
            with res_col2:
                # Create a simple visualization
                fig, ax = plt.subplots(figsize=(5, 3))
                labels = ['Down', 'Up']
                sizes = [(1-prediction)*100, prediction*100]
                colors = ['#ff9999', '#66b3ff'] if prediction_binary == 1 else ['#66b3ff', '#ff9999']
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                plt.title(f'{timeframe} Price Movement Probability')
                st.pyplot(fig)
    else:
        st.error("Please make sure the model files exist in the 'models' directory.")

# Add information about the features
with st.expander("Feature Information"):
    st.markdown("""
    ## Feature Descriptions
    
    - **Opening Price**: The price at the beginning of the trading period
    - **Highest Price**: The highest price during the trading period
    - **Lowest Price**: The lowest price during the trading period
    - **Trading Volume**: The total number of shares or contracts traded
    - **5-Day Moving Average**: Average price over the last 5 days
    - **10-Day Moving Average**: Average price over the last 10 days
    - **Relative Strength Index (RSI)**: Momentum indicator measuring speed and change of price movements (0-100)
    - **Volatility**: Statistical measure of the dispersion of returns
    - **Market Sentiment Score**: Measure of overall market sentiment (-1 to 1)
    - **Global Economy Indicator**: Numeric value representing economic conditions (0-5)
    - **Event Impact Score**: Impact of significant events on the market (-1 to 1)
    """)

# Add footer
st.markdown("""
---
### Crypto Price Movement Prediction Project
This app uses machine learning models to predict cryptocurrency price movements based on market indicators.
""")
