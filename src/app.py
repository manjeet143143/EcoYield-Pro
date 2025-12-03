import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="EcoYield Pro", page_icon="🌾", layout="centered")

# --- 2. LOAD MODEL (Cached for speed) ---
@st.cache_resource
def load_model():
    with open('models/yield_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/crop_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

model, label_encoder = load_model()

# --- 3. HEADER & SIDEBAR ---
st.title("🌾 EcoYield Pro")
st.markdown("### AI-Based Sustainable Crop Yield Prediction")
st.markdown("Adjust the soil and weather parameters below to predict yield and environmental impact.")

st.sidebar.header("📝 Field Parameters")

# Create a form in the sidebar
with st.sidebar.form("prediction_form"):
    st.subheader("1. Crop Selection")
    # We get the list of crops the model knows (from the encoder)
    crop_options = list(label_encoder.classes_)
    crop_name = st.selectbox("Select Crop Type", crop_options)
    
    st.subheader("2. Soil Nutrients")
    N = st.slider("Nitrogen (N)", 0, 140, 80)
    P = st.slider("Phosphorus (P)", 0, 145, 40)
    K = st.slider("Potassium (K)", 0, 205, 40)
    
    st.subheader("3. Weather Conditions")
    temp = st.number_input("Temperature (°C)", value=25.0, step=0.1)
    hum = st.number_input("Humidity (%)", value=70.0, step=0.1)
    ph = st.number_input("Soil pH", value=6.5, step=0.1)
    rain = st.number_input("Rainfall (mm)", value=200.0, step=1.0)
    
    # The "Predict" Button
    submit_btn = st.form_submit_button("🚀 Predict Yield")

# --- 4. PREDICTION LOGIC ---
if submit_btn:
    # Prepare Data
    crop_num = label_encoder.transform([crop_name])[0]
    input_data = pd.DataFrame([[N, P, K, temp, hum, ph, rain, crop_num]], 
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'crop_num'])
    
    # Get Prediction
    prediction = model.predict(input_data)[0]
    
    # Calculate Sustainability (PFP)
    total_chemicals = N + P + K
    if total_chemicals == 0: total_chemicals = 1
    pfp_score = prediction / total_chemicals

    # Determine Status
    if pfp_score > 0.15:
        status = "SUSTAINABLE (Eco-Friendly)"
        color = "green"
    elif pfp_score > 0.08:
        status = "MODERATE (Standard)"
        color = "orange"
    else:
        status = "UNSUSTAINABLE (Excessive Chemicals)"
        color = "red"

    # --- 5. DISPLAY RESULTS ---
    st.divider()
    
    # Use Columns for a clean layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="🌱 Projected Yield", value=f"{prediction:.2f} Tons/Ha")
    
    with col2:
        st.metric(label="🧪 Chemical Efficiency", value=f"{pfp_score:.4f}", delta=status, delta_color="off")
        st.caption(f"Status: :{color}[{status}]")

    # --- 6. VISUALIZATION ---
    st.subheader("📊 Analysis Report")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Data for chart
    categories = ['Yield Output', 'Chemical Input (scaled)']
    values = [prediction, total_chemicals / 100] # Scaling input down to match yield scale visually
    colors = ['#4CAF50', '#FF5722'] # Green and Deep Orange
    
    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel("Metric Units")
    ax.set_title(f"Efficiency Check for {crop_name}")
    
    # Add numbers on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

    # Send the plot to Streamlit
    st.pyplot(fig)
    
    # Advice Section
    if color == "red":
        st.warning("⚠️ Recommendation: Your chemical usage is too high for this yield. Try reducing Nitrogen by 10% to improve sustainability.")
    elif color == "green":
        st.success("✅ Recommendation: Excellent balance! Maintain current practices.")