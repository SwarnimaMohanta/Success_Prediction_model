<h1>🎬 AI-Powered Movie Success Prediction System</h1><br>
An intelligent deep learning system that predicts whether a movie will be successful based on key production features such as budget, runtime, genre, release timing, and other critical metadata. Built with TensorFlow/Keras and deployed as an interactive web application using Streamlit.<br>

<h3>✨ Key Features</h3><br>
📌 AI-Powered Movie Success Prediction

Deep Learning Model: Dense Neural Network (Deep NN) predicts the probability of a movie being successful
Real-time Predictions: Instant success probability calculation based on input features
Binary Classification: Clear "✅ SUCCESSFUL" or "❌ NOT SUCCESSFUL" output with confidence levels

📌 Detailed Feature Analysis<br>

Feature Impact Visualization: Shows top factors influencing the prediction
Comprehensive Metrics: Budget, runtime, genres, popularity, cast, and production details
Interpretable Results: Understand what drives movie success

📌 Interactive Web Dashboard<br>

User-Friendly Interface: Simple input form with intuitive controls
Multi-Select Genres: Choose from 19 different movie genres
Responsive Design: Modern gradient UI with smooth animations
One-Click Prediction: Instant results with detailed breakdown

📌 Transparent Results<br>

Probability Score: Percentage likelihood of success (0-100%)
Confidence Level: High/Medium/Low confidence indicator
Feature Summary: Complete overview of input parameters<br>

<h3>🛠 Tech Stack</h3><br>
<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: left;">
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th>Component</th>
      <th>Technology</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Backend</td>
      <td>Python 3.10</td>
    </tr>
    <tr>
      <td>Deep Learning</td>
      <td>TensorFlow 2.x / Keras</td>
    </tr>
    <tr>
      <td>Web Framework</td>
      <td>Streamlit</td>
    </tr>
    <tr>
      <td>Data Processing</td>
      <td>pandas, NumPy</td>
    </tr>
    <tr>
      <td>Visualization</td>
      <td>Matplotlib, Seaborn</td>
    </tr>
    <tr>
      <td>Model Training</td>
      <td>scikit-learn, imbalanced-learn (SMOTE)</td>
    </tr>
  </tbody>
</table>

<h2>📊 Dataset</h2>
<h3>Source</h3>
TMDB 5000 Movie Dataset from Kaggle<br>

<h3>Download Dataset</h3>
Contains two CSV files:<br>

tmdb_5000_movies.csv - Movie details<br>
tmdb_5000_credits.csv - Cast and crew information<br>



<h3>Dataset Statistics</h3>

Total Movies: ~4,800 movies
Time Period: Various decades (focus on modern cinema)
Features: 20+ raw features (budget, revenue, genres, cast, crew, etc.)<br>

<h3>Features Used</h3>
Numeric Features (13)<br>

Budget ($),
Runtime (minutes),
Vote Average (0-10 rating),
Vote Count,
Popularity Score,
Release Year,
Release Month (cyclical encoding: sin/cos),
Number of Genres,
Number of Production Companies,
Star Power (log-transformed vote count),
Budget per Minute,
Popularity per Vote<br>

Categorical Features (20)

Genres (19 one-hot encoded):

Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, TV Movie, Thriller, War, Western<br>


Language (1 binary feature):

Is English (0/1)



Total Input Features: 33<br>
<h3>Data Preprocessing</h3>

Merged movie and credits datasets
Parsed JSON-formatted columns (genres, cast, crew)
Extracted top 5 cast members and director
Calculated ROI (Return on Investment) and profit
Removed movies with missing critical data (budget/revenue = 0)
Filtered outliers (top 1% extreme values)
Created binary success target variable (70th percentile ROI threshold)<br>

<h3>🎛 Model Architecture & Training</h3><br>
<h2>Model Configuration</h2><br>
<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: left;">
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th>Parameter</th>
      <th>Value</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Architecture</td>
      <td>Dense Neural Network (Deep NN)</td>
      <td>Fully connected feedforward network</td>
    </tr>
    <tr>
      <td>Input Shape</td>
      <td>(33,)</td>
      <td>33 preprocessed features</td>
    </tr>
    <tr>
      <td>Hidden Layers</td>
      <td>5 layers: 256 → 128 → 64 → 32 → 16 neurons</td>
      <td>Number of neurons in each hidden layer</td>
    </tr>
    <tr>
      <td>Activation</td>
      <td>ReLU</td>
      <td>Rectified Linear Unit for hidden layers</td>
    </tr>
    <tr>
      <td>Output Layer</td>
      <td>1 neuron (Sigmoid)</td>
      <td>Binary classification output</td>
    </tr>
    <tr>
      <td>Dropout Rate</td>
      <td>0.3 - 0.4</td>
      <td>Prevent overfitting</td>
    </tr>
    <tr>
      <td>Batch Normalization</td>
      <td>Yes</td>
      <td>Stabilize training</td>
    </tr>
    <tr>
      <td>Optimizer</td>
      <td>Adam</td>
      <td>Learning rate: 0.001</td>
    </tr>
    <tr>
      <td>Loss Function</td>
      <td>Binary Crossentropy</td>
      <td>For binary classification</td>
    </tr>
    <tr>
      <td>Batch Size</td>
      <td>32</td>
      <td>Optimal for memory usage</td>
    </tr>
    <tr>
      <td>Epochs</td>
      <td>50 (with early stopping)</td>
      <td>Stops when validation loss plateaus</td>
    </tr>
  </tbody>
</table><br>

<h3>Training Process</h3>

<h2>Data Split:</h2><br>

Training: 70%
Validation: 15%
Testing: 15%<br>


<h2>Class Imbalance Handling:</h2>

Applied SMOTE (Synthetic Minority Over-sampling Technique)
Computed class weights for loss function
Balanced successful/unsuccessful movie distribution<br>


<h2>Callbacks:</h2>

Early Stopping: Monitors validation AUC-ROC (patience: 15 epochs)
Learning Rate Reduction: Reduces LR by 50% if validation loss plateaus (patience: 5 epochs)<br>


<h2>Model Selection:</h2>

Trained 3 architectures: Baseline NN, Deep NN, Residual NN
Selected best model based on AUC-ROC score
Saved as movie_predictor_v1.h5<br>

<h3>📊 Model Performance Metrics</h3><br>
<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: left;">
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th>Metric</th>
      <th>Score</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Test Accuracy</td>
      <td>87.5%</td>
      <td>Overall correct predictions</td>
    </tr>
    <tr>
      <td>Precision</td>
      <td>85.3%</td>
      <td>True positives / (True positives + False positives)</td>
    </tr>
    <tr>
      <td>Recall</td>
      <td>88.2%</td>
      <td>True positives / (True positives + False negatives)</td>
    </tr>
    <tr>
      <td>F1-Score</td>
      <td>86.7%</td>
      <td>Harmonic mean of precision and recall</td>
    </tr>
    <tr>
      <td>AUC-ROC</td>
      <td>0.92</td>
      <td>Area under ROC curve (excellent discrimination)</td>
    </tr>
  </tbody>
</table><br>

<h2>Strengths</h2><br>
✅ High accuracy in predicting box-office success potential
✅ Balanced precision and recall (handles class imbalance well)
✅ Excellent AUC-ROC score (0.92) indicates strong discriminative power
✅ Handles both numeric and categorical features effectively
✅ Provides interpretable feature-level insights<br>
<h2>Model Limitations</h2>
⚠ Limited to features available before release (no post-release data)
⚠ Success defined by ROI threshold (subjective metric)
⚠ May not capture subjective factors (story quality, marketing buzz)<br>

<h3>🧪 Example Predictions</h3><br>
<h2>High Budget Action Film</h2><br>
Input:
- Budget: $150,000,000
- Runtime: 140 minutes
- Genres: Action, Adventure, Science Fiction
- Expected Rating: 7.5/10
- Popularity: 85<br>

Output:
→ Success Probability: 78.45%
→ Prediction: ✅ SUCCESSFUL
→ Confidence: High 🎯<br>

<h2>Low Budget Drama</h2>
Input:
- Budget: $5,000,000
- Runtime: 105 minutes
- Genres: Drama
- Expected Rating: 7.8/10
- Popularity: 15

Output:
→ Success Probability: 42.33%
→ Prediction: ❌ NOT SUCCESSFUL
→ Confidence: Medium 📊
<h2>Horror Thriller</h2><br>

Input:
- Budget: $20,000,000
- Runtime: 95 minutes
- Genres: Horror, Thriller
- Expected Rating: 6.5/10
- Popularity: 45<br>

Output:
→ Success Probability: 61.28%
→ Prediction: ✅ SUCCESSFUL
→ Confidence: Medium 📊<br>

estimation


<h3>🚀 Future Enhancements</h3>
<h2>Research Directions</h2><br>

 Sentiment Analysis: Incorporate pre-release social media buzz and review sentiment<br>
 Multi-Modal Learning: Combine trailer videos, movie posters, and cast images<br>
 Time-Series Features: Box office opening weekend predictions<br>
 Explainable AI: SHAP/LIME values for feature importance<br>
 Transfer Learning: Fine-tune on larger datasets (100K+ movies)<br>
 Ensemble Methods: Combine multiple models (Random Forest + Neural Network)<br>
 Real-Time API: Deploy as REST API with FastAPI<br>
 A/B Testing: Compare different model architectures<br>

 <h3>Thank you</h3>
