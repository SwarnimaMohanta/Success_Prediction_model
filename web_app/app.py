# web_app/app.py - COMPLETE WORKING VERSION
import streamlit as st
import numpy as np
import sys
import pickle
import traceback

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ---------------------------
# Load Model & Scaler
# ---------------------------
@st.cache_resource
import os
import joblib

def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))

    model_path = os.path.join(project_dir, "models", "movie_predictor.pkl")
    scaler_path = os.path.join(project_dir, "data", "processed", "scaler.pkl")

    print("MODEL PATH:", model_path)
    print("SCALER PATH:", scaler_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


@st.cache_resource
def load_scaler():
    """Load the fitted scaler"""
    possible_paths = [
        'data/processed/scaler.pkl',
        '../data/processed/scaler.pkl',
        'scaler.pkl',
        os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'scaler.pkl')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    scaler = pickle.load(f)
                    return scaler, path
            except Exception as e:
                continue
    
    return None, None

# Load resources
model, model_path = load_model()
scaler, scaler_path = load_scaler()

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Movie Success Predictor", 
    page_icon="🎬", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 3rem 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        max-width: 1200px;
        margin-top: 2rem;
    }

    h1 {
        color:#0d1b2a !important;
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }

    .subtitle {
        color: #555 !important;
        font-size: 1.2rem !important;
        text-align: center;
        margin-bottom: 2rem !important;
        font-weight: 400;
    }

    h3 {
        color: #764ba2 !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        border-radius: 15px !important;
        border: none !important;
        padding: 0.8rem 3rem !important;
        width: 100% !important;
        margin-top: 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 20px rgba(102, 126, 234, 0.5) !important;
    }

    .result-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 1rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }
    
    .result-text {
        color: #333;
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
    }

    .feature-list {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.markdown("<h1>🎬 Movie Success Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict whether a movie will be successful using advanced machine learning</p>", unsafe_allow_html=True)

# Show loading status
if model is None or scaler is None:
    st.error("❌ *Critical Error: Model or Scaler not loaded!*")
    st.info("📁 *Troubleshooting:*")
    st.write("- Ensure models/movie_predictor.pkl exists")
    st.write("- Ensure data/processed/scaler.pkl exists")
    st.write("- Run python train_xgboost.py to generate the model")
    st.stop()
else:
    with st.expander("✅ System Status", expanded=False):
        st.success(f"✅ Model loaded from: {model_path}")
        st.success(f"✅ Scaler loaded from: {scaler_path}")

# ---------------------------
# Preprocessing Function
# ---------------------------
def preprocess_movie_data(movie_data, all_genres, scaler):
    """Convert movie_data to 33-feature array"""
    try:
        # Extract features
        budget = float(movie_data['budget'])
        runtime = float(movie_data['runtime'])
        vote_average = float(movie_data['vote_average'])
        vote_count = float(movie_data['vote_count'])
        popularity = float(movie_data['popularity'])
        release_year = float(movie_data['release_year'])
        release_month = float(movie_data['release_month'])
        
        # Derived features
        num_genres = len(movie_data['genres'])
        num_production_companies = float(movie_data['num_production_companies'])
        star_power = np.log1p(vote_count)
        budget_per_minute = budget / runtime if runtime > 0 else 0
        popularity_per_vote = popularity / (vote_count + 1)
        
        # Cyclical month encoding
        month_sin = np.sin(2 * np.pi * release_month / 12)
        month_cos = np.cos(2 * np.pi * release_month / 12)
        
        # Numerical features (13 features)
        numerical_features = np.array([[
            budget, runtime, vote_average, vote_count, popularity,
            release_year, num_genres, num_production_companies,
            star_power, budget_per_minute, popularity_per_vote,
            month_sin, month_cos
        ]], dtype=np.float32)
        
        # Normalize
        numerical_features_normalized = scaler.transform(numerical_features)
        
        # Build feature vector
        feature_list = numerical_features_normalized[0].tolist()
        
        # Add genre one-hot encoding (19 genres)
        for genre in all_genres:
            feature_list.append(1.0 if genre in movie_data['genres'] else 0.0)
        
        # Add language (1 feature)
        feature_list.append(1.0 if movie_data['is_english'] else 0.0)
        
        return np.array([feature_list], dtype=np.float32)
    
    except Exception as e:
        st.error(f"❌ Preprocessing Error: {e}")
        st.code(traceback.format_exc())
        return None

# ---------------------------
# Input Form
# ---------------------------
with st.form(key="movie_form"):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        title = st.text_input("🎥 Movie Title", "KGF")
    
    st.markdown("<h3>📊 Financial & Technical Details</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        budget = st.number_input("💰 Budget ($)", min_value=0, value=150_000_000, step=1_000_000, format="%d")
        runtime = st.number_input("⏱ Runtime (minutes)", min_value=30, max_value=300, value=120)
    
    with col2:
        vote_average = st.number_input("⭐ Expected Rating (0-10)", min_value=0.0, max_value=10.0, value=9.5, step=0.1)
        vote_count = st.number_input("👥 Vote Count", min_value=0, value=10000, step=100)
    
    with col3:
        popularity = st.number_input("🔥 Popularity Score", min_value=0.0, value=90.0, step=1.0)
        num_production_companies = st.number_input("🏢 Production Companies", min_value=1, max_value=10, value=3)

    st.markdown("<h3>📅 Release Information</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        release_year = st.number_input("📆 Release Year", min_value=1900, max_value=2100, value=2024)
    
    with col2:
        release_month = st.number_input("📅 Release Month (1-12)", min_value=1, max_value=12, value=6)
    
    with col3:
        is_english = st.checkbox("🌍 English Language?", value=True)

    st.markdown("<h3>🎭 Genres (Select multiple)</h3>", unsafe_allow_html=True)
    
    genres_options = [
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
        'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music',
        'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
    ]
    
    genres = st.multiselect(
        "Choose genres that best describe your movie",
        genres_options,
        default=['Action', 'Adventure', 'Thriller']
    )

    submit_button = st.form_submit_button(label="🚀 Predict Success")

# ---------------------------
# Prediction Logic
# ---------------------------
if submit_button:
    # Validate inputs
    if not genres:
        st.error("⚠ *Please select at least one genre!*")
        st.stop()
    
    if runtime <= 0:
        st.error("⚠ *Runtime must be greater than 0!*")
        st.stop()
    
    # Create movie data dictionary
    movie_data = {
        'budget': budget,
        'runtime': runtime,
        'vote_average': vote_average,
        'vote_count': vote_count,
        'popularity': popularity,
        'release_year': release_year,
        'release_month': release_month,
        'is_english': is_english,
        'genres': genres,
        'num_production_companies': num_production_companies
    }

    # Preprocess
    with st.spinner("🔄 Processing your movie data..."):
        X_input = preprocess_movie_data(movie_data, genres_options, scaler)
    
    if X_input is None:
        st.error("❌ Failed to preprocess input data")
        st.stop()
    
    # Validate feature count
    if X_input.shape[1] != 33:
        st.error(f"❌ *Feature mismatch!* Expected 33 features, got {X_input.shape[1]}")
        st.stop()
    
    # Make prediction
    try:
        with st.spinner("🤖 Making prediction..."):
            prediction_prob = model.predict_proba(X_input)[0][1]
            is_successful = prediction_prob >= 0.5
            
            # Calculate confidence (UPDATED for better thresholds)
            if prediction_prob > 0.80 or prediction_prob < 0.20:
                confidence = "High 🎯"
            elif prediction_prob > 0.65 or prediction_prob < 0.35:
                confidence = "Medium 📊"
            else:
                confidence = "Low ⚠"

        # Display Results
        st.markdown("---")
        st.success(f"✅ *Prediction Results for '{title}'*")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Success Probability", f"{prediction_prob*100:.2f}%")
        
        with col2:
            st.metric("Prediction", "✅ SUCCESSFUL" if is_successful else "❌ NOT SUCCESSFUL")
        
        with col3:
            st.metric("Confidence Level", confidence)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Success/Failure Message
        if is_successful:
            st.markdown("""
            <div class='result-card'>
                <p class='result-text'>🎉 Great news! This movie has strong potential for success!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='result-card'>
                <p class='result-text'>⚠ This movie may face challenges. Consider optimizing key factors.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Movie Details Summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class='feature-list' style='color: #333333;'>
                <p style='margin:0; padding-bottom:10px;'><strong>💰 Budget:</strong> ${budget:,}</p>
                <p style='margin:0; padding-bottom:10px;'><strong>⏱ Runtime:</strong> {runtime} minutes</p>
                <p style='margin:0; padding-bottom:10px;'><strong>⭐ Expected Rating:</strong> {vote_average}/10</p>
                <p style='margin:0; padding-bottom:0px;'><strong>🎭 Genres:</strong> {', '.join(genres)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='feature-list' style='color: #333333;'>
                <p style='margin:0; padding-bottom:10px;'><strong>📅 Release:</strong> {release_month}/{release_year}</p>
                <p style='margin:0; padding-bottom:10px;'><strong>🌍 Language:</strong> {'English' if is_english else 'Non-English'}</p>
                <p style='margin:0; padding-bottom:10px;'><strong>🔥 Popularity:</strong> {popularity}</p>
                <p style='margin:0; padding-bottom:0px;'><strong>🏢 Production Co:</strong> {num_production_companies}</p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ *Prediction Error:* {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

# ---------------------------
# Footer
# ---------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888; font-size: 0.9rem;'>Movie Success Predictor v3.0 | Built with Streamlit & XGBoost</p>", unsafe_allow_html=True)
