import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# ================= LOAD DATA =================
df = pd.read_csv("../data/BankingChurn.csv")

print("Dataset Shape:", df.shape)

# ================= DROP ID COLUMNS =================
for col in df.columns:
    if "id" in col.lower():
        df.drop(col, axis=1, inplace=True)

# ================= REMOVE LEAKAGE COLUMNS (IF PRESENT) =================
leakage_cols = [
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"
]

df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)

# ================= TARGET ENCODING =================
df["Attrition_Flag"] = df["Attrition_Flag"].map({
    "Existing Customer": 0,
    "Attrited Customer": 1
})

# ================= ONE HOT ENCODING (NO ENCODER FILE NEEDED) =================
df = pd.get_dummies(df, drop_first=True)

# ================= SPLIT X y =================
X = df.drop("Attrition_Flag", axis=1)
y = df["Attrition_Flag"]

# ⭐ SAVE FEATURE ORDER
feature_names = X.columns.tolist()
joblib.dump(feature_names, "../models/bank_features.pkl")

# ================= SCALE =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "../models/bank_scaler.pkl")

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ================= HANDLE IMBALANCE =================
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# ================= TRAIN XGBOOST =================
model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# ================= EVALUATION =================
y_pred = model.predict(X_test)

print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# ================= SAVE MODEL =================
joblib.dump(model, "../models/bank_model.pkl")

print("\n✅ Bank Churn XGBoost Model Trained & Saved Successfully!")

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print("Models accuracy: ",accuracy_score(y_pred,y_test))