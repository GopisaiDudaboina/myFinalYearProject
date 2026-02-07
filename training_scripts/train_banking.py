import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# --- ROBUST PATH SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "../datasets/banking.csv")
ARTIFACTS = os.path.join(SCRIPT_DIR, "../artifacts")

os.makedirs(ARTIFACTS, exist_ok=True)

# Load Data
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ File not found: {DATA_PATH}\nDid you rename 'BankingChurn.csv' to 'banking.csv'?")

df = pd.read_csv(DATA_PATH)
drop_cols = ["CLIENTNUM", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

ui_options = {}
for col in df.select_dtypes(include='object').columns:
    if col != "Attrition_Flag":
        ui_options[col] = sorted(df[col].dropna().unique().tolist())

config_path = os.path.join(ARTIFACTS, "ui_config.json")
try:
    with open(config_path, 'r') as f: config = json.load(f)
except: config = {}
config["bank"] = {"options": ui_options}
with open(config_path, 'w') as f: json.dump(config, f, indent=2)

df["Attrition_Flag"] = df["Attrition_Flag"].apply(lambda x: 1 if "Attrited" in x else 0)
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Attrition_Flag", axis=1)
y = df["Attrition_Flag"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)
model = XGBClassifier(n_estimators=300, max_depth=6, scale_pos_weight=5)
model.fit(X_train, y_train)

# Metadata
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
metadata = {
    "model_name": "XGBoost (Gradient Boosting)",
    "accuracy": round(acc * 100, 2),
    "samples": len(df),
    "features": len(X.columns)
}

joblib.dump(model, os.path.join(ARTIFACTS, "bank_model.pkl"))
joblib.dump(X.columns.tolist(), os.path.join(ARTIFACTS, "bank_features.pkl"))
joblib.dump(scaler, os.path.join(ARTIFACTS, "bank_scaler.pkl"))
with open(os.path.join(ARTIFACTS, "bank_metadata.json"), "w") as f:
    json.dump(metadata, f)

print(f"✅ Banking Model Trained (Accuracy: {acc:.2%})")