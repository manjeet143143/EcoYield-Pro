import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
print("⏳ Loading dataset...")
df = pd.read_csv('data/crop_data.csv')

# 2. Data Cleaning & Engineering
# The dataset has 'label' (Crop Name) which is text. We need numbers.
label_encoder = LabelEncoder()
df['crop_num'] = label_encoder.fit_transform(df['label'])

# Note: This dataset is technically for "Classification" (predicting crop name).
# But for your project "Yield Prediction", we will simulate Yield.
# In a real scenario, you would look for a dataset with a 'Yield' column.
# For now, we will Generate a Synthetic Yield based on scientific logic 
# so you can demonstrate the PREDICTION capability.
print("⚙️ Generating Yield Data for training...")
# Random logic: Yield is higher if nutrient balance is good (just for demo)
df['Yield_Tons_Per_Hectare'] = (df['N'] + df['P'] + df['K']) * 0.05 + (df['rainfall'] * 0.02)

# 3. Define Features (X) and Target (y)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'crop_num']]
y = df['Yield_Tons_Per_Hectare']

# 4. Train Model
print("🧠 Training the AI model (Random Forest)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 5. Save the "Brain" and the "Translator"
print("💾 Saving the model to 'models/' folder...")
with open('models/yield_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/crop_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ Training Complete! You can now run the prediction script.")