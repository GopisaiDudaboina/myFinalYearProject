import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier

# ================= LOAD DATA =================
df = pd.read_csv("../data/OnlineRetailChurn.csv", encoding="ISO-8859-1")

# ================= BASIC CLEANING =================
df.dropna(inplace=True)

# Remove cancelled invoices
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Convert date
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Create time features
df["Invoice_Day"] = df["InvoiceDate"].dt.day
df["Invoice_Month"] = df["InvoiceDate"].dt.month
df["Invoice_Hour"] = df["InvoiceDate"].dt.hour

# ================= CREATE CHURN LABEL =================
# Customers who purchased only once = churn
purchase_counts = df.groupby("CustomerID")["InvoiceNo"].nunique()
churn_customers = purchase_counts[purchase_counts == 1].index

df["Churn"] = df["CustomerID"].apply(lambda x: 1 if x in churn_customers else 0)

# ================= SELECT FEATURES =================
df = df[[
    "Quantity",
    "UnitPrice",
    "CustomerID",
    "Country",
    "Invoice_Day",
    "Invoice_Month",
    "Invoice_Hour",
    "Churn"
]]

# ================= ENCODE COUNTRY =================
df = pd.get_dummies(df, columns=["Country"])


# ================= SPLIT X y =================
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ⭐ SAVE FEATURE ORDER (VERY IMPORTANT)
feature_names = X.columns.tolist()
joblib.dump(feature_names, "../models/ecom_features.pkl")

# ================= SCALE =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "../models/ecom_scaler.pkl")

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ================= TRAIN MODEL =================
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    min_gain_to_split=0.01
)


model.fit(X_train, y_train)

# ================= SAVE MODEL =================
joblib.dump(model, "../models/ecom_model.pkl")

print("✅ E-Commerce Churn Model Trained & Saved Successfully!")

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print("Models accuracy: ",accuracy_score(y_pred,y_test))