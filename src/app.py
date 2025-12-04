import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px

# --- 1. PRO CONFIGURATION ---
st.set_page_config(
    page_title="EcoYield Pro",
    page_icon="🌱",
    layout="wide",  # Uses the full screen width
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .big-font { font-size: 20px !important; font-weight: 500; }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    with open('models/yield_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/crop_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

try:
    model, label_encoder = load_model()
except:
    st.error("⚠️ Model files not found. Please run 'train.py' first.")
    st.stop()

# --- 3. SIDEBAR (CONTROLS) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/6062/6062646.png", width=80)
    st.title("EcoYield Pro")
    st.caption("AI-Driven Precision Agriculture")
    st.divider()

    st.header("1. Crop Details")
    crop_options = list(label_encoder.classes_)
    crop_name = st.selectbox("Target Crop", crop_options, index=crop_options.index('rice') if 'rice' in crop_options else 0)

    st.header("2. Soil Composition")
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Nitrogen (N)", 0, 150, 80)
        K = st.number_input("Potassium (K)", 0, 150, 40)
    with col2:
        P = st.number_input("Phosphorus (P)", 0, 150, 40)
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)

    st.header("3. Weather Data")
    temp = st.slider("Temperature (°C)", 10.0, 50.0, 25.0)
    hum = st.slider("Humidity (%)", 10.0, 100.0, 70.0)
    rain = st.slider("Rainfall (mm)", 0.0, 400.0, 200.0)
    
    st.divider()
    submit_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

# --- 4. MAIN DASHBOARD ---
st.markdown("## 📊 Field Analysis Report")

if submit_btn:
    # Prediction Logic
    crop_num = label_encoder.transform([crop_name])[0]
    input_df = pd.DataFrame([[N, P, K, temp, hum, ph, rain, crop_num]], 
                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'crop_num'])
    
    prediction = model.predict(input_df)[0]
    
    # Logic: PFP (Partial Factor Productivity)
    total_nutrients = N + P + K
    if total_nutrients == 0: total_nutrients = 1
    pfp_score = prediction / (total_nutrients/1000) # Scaling for display logic (Yield in tons vs Input in kg)
    # Simplified Logic for Demo:
    # Yield (Tons) / Input (kg) is tiny. Let's use simple Ratio: Yield / (Input/100)
    efficiency = prediction / (total_nutrients if total_nutrients > 0 else 1)

    # Status Logic
    if efficiency > 0.15:
        status = "Sustainable"
        color = "green"
        msg = "Excellent! High yield with minimal chemical footprint."
    elif efficiency > 0.08:
        status = "Moderate"
        color = "orange"
        msg = "Acceptable. Consider reducing Nitrogen by 10%."
    else:
        status = "Unsustainable"
        color = "red"
        msg = "Warning: Diminishing returns detected. High chemical waste."

    # --- TOP METRICS ROW ---
    m1, m2, m3 = st.columns(3)
    m1.metric("🌱 Projected Yield", f"{prediction:.2f} Tons/Ha", delta="AI Forecast")
    m2.metric("🧪 Chemical Input", f"{total_nutrients} kg/ha", delta="Field Data", delta_color="off")
    m3.metric("🌍 Sustainability Score", f"{efficiency:.4f}", delta=status, delta_color="normal" if color=="green" else "inverse")

    st.divider()

    # --- PROFESSIONAL CHARTS (PLOTLY) ---
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Efficiency Analysis")
        # Interactive Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = efficiency,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sustainability Index", 'font': {'size': 24}},
            delta = {'reference': 0.15, 'increasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [0, 0.3], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#1f77b4"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 0.08], 'color': '#ffcccb'},  # Red Zone
                    {'range': [0.08, 0.15], 'color': '#ffe5b4'}, # Orange Zone
                    {'range': [0.15, 0.3], 'color': '#d0f0c0'}], # Green Zone
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': efficiency}}))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c2:
        st.subheader("AI Recommendation")
        if color == "green":
            st.success(f"**Action:** {msg}")
        elif color == "orange":
            st.warning(f"**Action:** {msg}")
        else:
            st.error(f"**Action:** {msg}")
            
        st.info(f"""
        **Stats for {crop_name}:**
        - Temp: {temp}°C
        - Rainfall: {rain}mm
        - pH: {ph}
        """)

else:
    st.info("👈 Please adjust the field parameters in the sidebar and click 'Run Prediction'")
    # Placeholder image to look nice before prediction
    st.image("https://images.unsplash.com/photo-1625246333195-58197bd47d26?q=80&w=2000&auto=format&fit=crop", caption="Smart Agriculture Dashboard")