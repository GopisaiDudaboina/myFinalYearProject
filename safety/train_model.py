import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = df[["tenure", "MonthlyCharges", "TotalCharges",
         "Contract", "InternetService", "PaymentMethod", "Churn"]]

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna(0, inplace=True)

le_contract = LabelEncoder()
le_internet = LabelEncoder()
le_payment = LabelEncoder()
le_churn = LabelEncoder()

df["Contract"] = le_contract.fit_transform(df["Contract"])
df["InternetService"] = le_internet.fit_transform(df["InternetService"])
df["PaymentMethod"] = le_payment.fit_transform(df["PaymentMethod"])
df["Churn"] = le_churn.fit_transform(df["Churn"])

X = df.drop("Churn", axis=1)
y = df["Churn"]

feature_names = X.columns.tolist()
joblib.dump(feature_names, "models/telecom_features.pkl")   # 🔥 save order

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "models/telecom_model.pkl")
joblib.dump(scaler, "models/telecom_scaler.pkl")

print("✅ Model + scaler + features saved!")
