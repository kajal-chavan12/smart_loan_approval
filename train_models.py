import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

print("📄 Loading EXCEL dataset...")

# Load Excel dataset
df = pd.read_excel("loan_data.xlsx")

print("✅ Dataset loaded")
print("📊 Shape:", df.shape)
print("📌 Columns:", df.columns.tolist())

# -------------------------------
# CLEAN DATA
# -------------------------------
df = df.drop(columns=["ID"])  # ID not useful for ML

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# -------------------------------
# TARGET COLUMN
# -------------------------------
target_col = "Loan_Approved"
print(f"🎯 Target column: {target_col}")

# -------------------------------
# ENCODE CATEGORICAL DATA
# -------------------------------
label_encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# -------------------------------
# SPLIT DATA
# -------------------------------
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# TRAIN MODEL
# -------------------------------
print("🤖 Training model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("✅ Model trained successfully")

# -------------------------------
# SAVE MODEL & ENCODERS
# -------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/loan_model.pkl")
joblib.dump(label_encoders, "models/encoders.pkl")

print("💾 Model saved → models/loan_model.pkl")
print("🎉 TRAINING COMPLETE")
