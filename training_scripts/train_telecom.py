import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# --- ROBUST PATH SETUP ---
# Get the folder where THIS script lives (training_scripts)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to root, then into datasets
DATA_PATH = os.path.join(SCRIPT_DIR, "../datasets/telecom.csv")
ARTIFACTS = os.path.join(SCRIPT_DIR, "../artifacts")

os.makedirs(ARTIFACTS, exist_ok=True)

# Load Data
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ File not found: {DATA_PATH}\nDid you rename 'TelcoCustomerChurn.csv' to 'telecom.csv' and move it to 'datasets/'?")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.replace(" ", "").str.strip()

# Preprocessing
encoders = {}
ui_options = {}

for col in df.select_dtypes(include="object").columns:
    if col != "Churn":
        ui_options[col] = sorted(df[col].dropna().unique().tolist())
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Save UI Config
config_path = os.path.join(ARTIFACTS, "ui_config.json")
try:
    with open(config_path, 'r') as f: config = json.load(f)
except: config = {}
config["telecom"] = {"options": ui_options}
with open(config_path, 'w') as f: json.dump(config, f, indent=2)

# Train
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0)
model.fit(X_train, y_train)

# Metadata
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
metadata = {
    "model_name": "CatBoost Classifier",
    "accuracy": round(acc * 100, 2),
    "samples": len(df),
    "features": len(X.columns)
}

# Save Artifacts
joblib.dump(model, os.path.join(ARTIFACTS, "telecom_model.pkl"))
joblib.dump(X.columns.tolist(), os.path.join(ARTIFACTS, "telecom_features.pkl"))
joblib.dump(encoders, os.path.join(ARTIFACTS, "telecom_encoders.pkl"))
with open(os.path.join(ARTIFACTS, "telecom_metadata.json"), "w") as f:
    json.dump(metadata, f)

print(f"✅ Telecom Model Trained (Accuracy: {acc:.2%})")