import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

# Load your dataset
df = pd.read_csv("../data/TelcoCustomerChurn.csv")

# Clean column names (remove double spaces)
df.columns = df.columns.str.replace("  ", " ").str.strip()

print("Columns:", df.columns.tolist())

# Encode categorical columns
encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Target
y = df["Churn"]

# Features
X = df.drop("Churn", axis=1)
features = X.columns.tolist()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model (BEST for tabular data)
model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    verbose=0
)

model.fit(X_train, y_train)

# Save everything
joblib.dump(model, "../models/telecom_model.pkl")
joblib.dump(features, "../models/telecom_features.pkl")
joblib.dump(encoders, "../models/telecom_encoders.pkl")

print("✅ Telecom model trained & saved successfully!")

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print("Models accuracy: ",accuracy_score(y_pred,y_test))