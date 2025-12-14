import joblib
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

from data_loader import load_data
from features import transform_features


def train_model():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print("Feature engineering...")
    X_train, X_test, scaler = transform_features(X_train, X_test)

    # Drop Time column
    if "Time" in X_train.columns:
        X_train = X_train.drop(columns=["Time"])
        X_test = X_test.drop(columns=["Time"])

    feature_names = X_train.columns.tolist()  # ✅ save feature order

    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    train_data = lgb.Dataset(X_resampled, label=y_resampled)
    valid_data = lgb.Dataset(X_test, label=y_test)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbosity": -1
    }

    print("Training model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[valid_data]
    )

    print("Saving model...")
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": feature_names  # ✅ VERY IMPORTANT
        },
        "models/fraud_model.pkl"
    )

    print("✅ Model training complete!")


if __name__ == "__main__":
    train_model()
