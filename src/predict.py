import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt  # <--- NEW LIBRARY

# 1. Load the trained model
print("⏳ Loading the trained AI...")
with open('models/yield_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/crop_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# 2. The Input Function
def get_user_input():
    print("\n--- 🌾 AI SUSTAINABLE CROP PREDICTOR 🌾 ---")
    crop_name = input("Target Crop (e.g., rice, maize, chickpea): ")
    try:
        crop_num = label_encoder.transform([crop_name])[0]
    except:
        print("⚠️ Crop not found. Using 'rice'.")
        crop_name = 'rice'
        crop_num = label_encoder.transform(['rice'])[0]

    N = float(input("Nitrogen (N): "))
    P = float(input("Phosphorus (P): "))
    K = float(input("Potassium (K): "))
    temp = float(input("Temperature (°C): "))
    hum = float(input("Humidity (%): "))
    ph = float(input("Soil pH: "))
    rain = float(input("Rainfall (mm): "))

    return [[N, P, K, temp, hum, ph, rain, crop_num]], N, P, K, crop_name

# 3. Get Data & Predict
user_data, N, P, K, crop_name = get_user_input()
columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'crop_num']
user_df = pd.DataFrame(user_data, columns=columns)

prediction = model.predict(user_df)[0]
total_nutrients = N + P + K
pfp_score = prediction / (total_nutrients if total_nutrients > 0 else 1)

# 4. Print Text Results
print(f"\n🔮 PREDICTED YIELD: {prediction:.2f} Tons/Ha")
print(f"🧪 TOTAL CHEMICALS: {total_nutrients:.2f} kg/ha")
print(f"📊 EFFICIENCY SCORE: {pfp_score:.4f}")

if pfp_score > 0.15:
    status = "Sustainable (Eco-Friendly)"
    color = 'green'
elif pfp_score > 0.08:
    status = "Moderate (Standard)"
    color = 'orange'
else:
    status = "Unsustainable (High Chemical Use)"
    color = 'red'

print(f"✅ STATUS: {status}")

# --- 5. THE VISUAL UPGRADE (New!) ---
print("📈 Generating Graph...")

# Create a bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(['Predicted Yield (Tons)', 'Chemical Input (100kg)'], 
        [prediction, total_nutrients/100], color=['blue', 'red'])

plt.title(f"Analysis for {crop_name}: {status}")
plt.ylabel('Amount')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the text on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.show() # This pops up the window