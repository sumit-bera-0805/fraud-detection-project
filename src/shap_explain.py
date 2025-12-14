import joblib
import shap
import pandas as pd
from data_loader import load_data
from features import transform_features

# Load model
artifact = joblib.load("models/fraud_model.pkl")
model = artifact["model"]

# Load data
X_train, X_test, y_train, y_test = load_data()
X_train, X_test, _ = transform_features(X_train, X_test)

if "Time" in X_test.columns:
    X_test = X_test.drop(columns=["Time"])

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.sample(200, random_state=42))

shap.summary_plot(shap_values, X_test.sample(200, random_state=42))
