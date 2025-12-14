import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("💳 Fraud Detection Dashboard")

# =========================
# Load trained model
# =========================
artifact = joblib.load("models/fraud_model.pkl")
model = artifact["model"]
scaler = artifact["scaler"]

st.success("Model loaded successfully!")

# =========================
# Upload CSV
# =========================
uploaded = st.file_uploader("Upload credit card CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("data/creditcard.csv").sample(200, random_state=42)

st.subheader("Raw Data")
st.dataframe(df.head())

# =========================
# SAME PREPROCESSING AS TRAINING
# =========================
df = df.copy()

# Scale Amount
df["Amount_scaled"] = scaler.transform(df[["Amount"]])
df["Amount_log"] = np.log1p(df["Amount"])

# Drop target
X = df.drop(columns=["Class"], errors="ignore")

# Drop Time column (same as training)
if "Time" in X.columns:
    X = X.drop(columns=["Time"])

# =========================
# 🔥 FIX: Align features with model
# =========================
model_features = model.feature_name()

# Reindex to exact same features & order
X = X.reindex(columns=model_features, fill_value=0)

# =========================
# Prediction
# =========================
probs = model.predict(X, num_iteration=model.best_iteration)
df["fraud_probability"] = probs

# =========================
# Output
# =========================
st.subheader("🚨 Fraud Prediction Results")

st.dataframe(
    df.sort_values("fraud_probability", ascending=False)
      .head(30)
      .reset_index(drop=True)
)
