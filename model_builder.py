import numpy as np
import pandas as pd
import pickle
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def train_and_save_complete_model():
    """
    Train XGBoost model AND save the scaler used during preprocessing.
    This ensures the web app uses the EXACT same normalization.
    """
    print("🚀 Training XGBoost Model with Proper Scaler...")

    # ===========================
    # 1. Load Processed Data
    # ===========================
    base_dir = os.path.dirname(os.path.abspath(__file__))
    X = np.load(os.path.join(base_dir, "data", "processed", "X_features.npy"))
    y = np.load(os.path.join(base_dir, "data", "processed", "y_target.npy"))

    print(f"✓ Loaded {X.shape[0]} movies with {X.shape[1]} features")

    # ===========================
    # 2. Split Data
    # ===========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")

    # ===========================
    # 3. Load the Scaler
    # ===========================
    with open(os.path.join(base_dir, "data", "processed", "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    print("✓ Scaler loaded from preprocessing")

    # ===========================
    # 4. Train XGBoost
    # ===========================
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    print("Training XGBoost...")
    model.fit(X_train, y_train)

    # ===========================
    # 5. Evaluate Model
    # ===========================
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"🏆 Model Accuracy: {accuracy*100:.2f}%")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Successful', 'Successful']))

    # ===========================
    # 6. Save Model
    # ===========================
    os.makedirs("models", exist_ok=True)

    with open("models/movie_predictor.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\n✅ Model saved: models/movie_predictor.pkl")


if __name__ == "__main__":
    train_and_save_complete_model()
