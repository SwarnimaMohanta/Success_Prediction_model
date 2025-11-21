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
    # 3. Load the Scaler (already fitted during preprocessing)
    # ===========================
    with open(os.path.join(base_dir, "data", "processed", "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    
    print("✓ Scaler loaded from preprocessing")
    
    # ===========================
    # 4. Train XGBoost Model
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
    
    # ===========================
    # 7. Test with Sample Data
    # ===========================
    print("\n" + "="*50)
    print("Testing with Sample Movies:")
    print("="*50)
    
    # Test Case 1: High-budget blockbuster (should be successful)
    test_movie_1 = create_test_features(
        budget=150_000_000,
        runtime=141,
        vote_average=9.1,
        vote_count=10000,
        popularity=85.0,
        release_year=2024,
        genres=['Action', 'Adventure', 'Thriller']
    )
    
    # Normalize using scaler
    test_movie_1_normalized = test_movie_1.copy()
    test_movie_1_normalized[:, :13] = scaler.transform(test_movie_1[:, :13])
    
    prob_1 = model.predict_proba(test_movie_1_normalized)[0][1]
    print(f"\n1. High-Budget Action Film (KGF-like):")
    print(f"   Success Probability: {prob_1*100:.2f}%")
    print(f"   Prediction: {'✅ SUCCESSFUL' if prob_1 > 0.5 else '❌ NOT SUCCESSFUL'}")
    
    # Test Case 2: Low-budget indie (may not be successful)
    test_movie_2 = create_test_features(
        budget=2_000_000,
        runtime=95,
        vote_average=6.0,
        vote_count=150,
        popularity=5.0,
        release_year=2024,
        genres=['Drama']
    )
    
    test_movie_2_normalized = test_movie_2.copy()
    test_movie_2_normalized[:, :13] = scaler.transform(test_movie_2[:, :13])
    
    prob_2 = model.predict_proba(test_movie_2_normalized)[0][1]
    print(f"\n2. Low-Budget Indie Drama:")
    print(f"   Success Probability: {prob_2*100:.2f}%")
    print(f"   Prediction: {'✅ SUCCESSFUL' if prob_2 > 0.5 else '❌ NOT SUCCESSFUL'}")
    
    print("\n" + "="*50)
    print("✅ Training Complete!")
    print("="*50)

def create_test_features(budget, runtime, vote_average, vote_count, 
                        popularity, release_year, genres, 
                        release_month=6, is_english=True, num_prod_companies=3):
    """Helper function to create feature vector for testing"""
    
    # Calculate derived features
    num_genres = len(genres)
    star_power = np.log1p(vote_count)
    budget_per_minute = budget / runtime if runtime > 0 else 0
    popularity_per_vote = popularity / (vote_count + 1)
    month_sin = np.sin(2 * np.pi * release_month / 12)
    month_cos = np.cos(2 * np.pi * release_month / 12)
    
    # Create feature array (33 features total)
    features = [
        budget,                    # 0
        runtime,                   # 1
        vote_average,              # 2
        vote_count,                # 3
        popularity,                # 4
        release_year,              # 5
        num_genres,                # 6
        num_prod_companies,        # 7
        star_power,                # 8
        budget_per_minute,         # 9
        popularity_per_vote,       # 10
        month_sin,                 # 11
        month_cos                  # 12
    ]
    
    # Add genre encoding (19 genres)
    all_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
                  'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
                  'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
                  'TV Movie', 'Thriller', 'War', 'Western']
    
    for genre in all_genres:
        features.append(1.0 if genre in genres else 0.0)
    
    # Add language feature
    features.append(1.0 if is_english else 0.0)
    
    return np.array([features], dtype=np.float32)

if __name__ == "__main__":
    train_and_save_complete_model()
