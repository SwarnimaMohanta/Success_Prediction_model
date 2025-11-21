#process.py 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import ast
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

class MovieDataPreprocessor:
    """Preprocess with STRICTER success criteria"""
    
    def __init__(self):
        self.df = pd.read_csv("data/raw/movies_dataset.csv")
        self.scaler = StandardScaler()
        print(f"Loaded dataset with {len(self.df)} movies")
        
    def clean_data(self):
        """Clean and filter"""
        print("\n=== Cleaning Data ===")
        initial_count = len(self.df)
        
        self.df = self.df[self.df['budget'] > 0]
        self.df = self.df[self.df['revenue'] > 0]
        self.df = self.df[self.df['runtime'] > 0]
        
        # Remove extreme outliers
        self.df = self.df[self.df['budget'] < self.df['budget'].quantile(0.99)]
        self.df = self.df[self.df['revenue'] < self.df['revenue'].quantile(0.99)]
        
        self.df['vote_average'].fillna(self.df['vote_average'].median(), inplace=True)
        self.df['vote_count'].fillna(0, inplace=True)
        self.df['director'].fillna('Unknown', inplace=True)
        
        print(f"Removed {initial_count - len(self.df)} movies")
        print(f"Remaining movies: {len(self.df)}")
        
    def create_target_variable(self):
        """
        STRICTER SUCCESS DEFINITION:
        A movie is successful if it meets ALL of these:
        1. ROI > 2.0x (100% profit) AND
        2. Revenue > $50M OR Rating > 7.0 with 1000+ votes
        """
        print(f"\n=== Creating Target Variable (STRICTER) ===")
        
        # Calculate ROI
        self.df['roi'] = self.df.apply(
            lambda row: (row['revenue'] - row['budget']) / row['budget'] 
            if row['budget'] > 0 else np.nan, 
            axis=1
        )
        
        self.df['profit'] = self.df['revenue'] - self.df['budget']
        
        # STRICTER criteria
        has_good_roi = self.df['roi'] > 2.0  # Must DOUBLE investment
        is_blockbuster = self.df['revenue'] > 50_000_000  # OR make $50M+
        is_critically_acclaimed = (self.df['vote_average'] > 7.0) & (self.df['vote_count'] > 1000)
        
        # Success = Good ROI AND (Blockbuster OR Acclaimed)
        self.df['success'] = (has_good_roi & (is_blockbuster | is_critically_acclaimed)).astype(int)
        
        success_count = self.df['success'].sum()
        print(f"Successful movies: {success_count} ({success_count/len(self.df)*100:.1f}%)")
        print(f"Unsuccessful movies: {len(self.df) - success_count} ({(1-success_count/len(self.df))*100:.1f}%)")
        
        print("\nSuccess criteria breakdown:")
        print(f"  ROI > 2.0x: {has_good_roi.sum()} movies")
        print(f"  Revenue > $50M: {is_blockbuster.sum()} movies")
        print(f"  High ratings (>7.0 with 1000+ votes): {is_critically_acclaimed.sum()} movies")
        
    def feature_engineering(self):
        """Engineer features"""
        print("\n=== Feature Engineering ===")
        
        self.df['release_date'] = pd.to_datetime(self.df['release_date'])
        self.df['release_year'] = self.df['release_date'].dt.year
        self.df['release_month'] = self.df['release_date'].dt.month
        self.df['release_quarter'] = self.df['release_date'].dt.quarter
        
        self.df['budget_category'] = pd.cut(self.df['budget'], 
                                           bins=[0, 1e6, 10e6, 50e6, 100e6, 500e6],
                                           labels=['Ultra Low', 'Low', 'Medium', 'High', 'Ultra High'])
        
        self.df['num_genres'] = self.df['genres'].apply(
            lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0
        )
        self.df['num_production_companies'] = self.df['production_companies'].apply(
            lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0
        )
        
        self.df['star_power'] = np.log1p(self.df['vote_count'])
        self.df['budget_per_minute'] = self.df['budget'] / self.df['runtime']
        self.df['popularity_per_vote'] = self.df['popularity'] / (self.df['vote_count'] + 1)
        
        print("Features created")
        
    def encode_categorical_features(self):
        """Encode categorical features"""
        print("\n=== Encoding Categorical Features ===")
        
        all_genres = set()
        for genres_str in self.df['genres']:
            if isinstance(genres_str, str):
                genres = ast.literal_eval(genres_str)
                all_genres.update(genres)
        
        for genre in all_genres:
            self.df[f'genre_{genre}'] = self.df['genres'].apply(
                lambda x: 1 if isinstance(x, str) and genre in ast.literal_eval(x) else 0
            )
        
        print(f"Created {len(all_genres)} genre features")
        
        self.df['is_english'] = (self.df['original_language'] == 'en').astype(int)
        
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['release_month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['release_month'] / 12)
        
    def prepare_final_dataset(self):
        """Prepare with normalization"""
        print("\n=== Preparing Final Dataset ===")
        
        numerical_features = [
            'budget', 'runtime', 'vote_average', 'vote_count', 'popularity',
            'release_year', 'num_genres', 'num_production_companies',
            'star_power', 'budget_per_minute', 'popularity_per_vote',
            'month_sin', 'month_cos'
        ]
        
        genre_features = [col for col in self.df.columns if col.startswith('genre_')]
        
        print(f"\nNormalizing {len(numerical_features)} numerical features...")
        X_numerical = self.df[numerical_features].values
        X_numerical_normalized = self.scaler.fit_transform(X_numerical)
        
        X_genres = self.df[genre_features].values
        X_language = self.df[['is_english']].values
        
        X = np.concatenate([X_numerical_normalized, X_genres, X_language], axis=1)
        y = self.df['success'].values
        
        feature_columns = numerical_features + genre_features + ['is_english']
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        
        return X, y, feature_columns
    
    def save_processed_data(self, X, y, feature_columns):
        """Save data"""
        output_dir = os.path.join('data', 'processed')
        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, 'X_features.npy'), X)
        np.save(os.path.join(output_dir, 'y_target.npy'), y)

        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            f.write('\n'.join(feature_columns))

        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"✓ Scaler saved to: {os.path.join(output_dir, 'scaler.pkl')}")

        self.df.to_csv(os.path.join(output_dir, 'movies_processed.csv'), index=False)
        print(f"\n✓ Processed data saved to: {output_dir}")
        
        # Verification
        print("\n" + "="*70)
        print("VERIFICATION: Sample Movies")
        print("="*70)
        
        # Show some failed movies (low budget, low rating)
        failed = self.df[(self.df['budget'] < 5_000_000) & (self.df['vote_average'] < 6.0)].head(3)
        print("\nLow-budget, low-rating movies (should be unsuccessful):")
        print(failed[['title', 'budget', 'revenue', 'roi', 'vote_average', 'success']].to_string(index=False))
        
        # Show blockbusters (should be successful)
        blockbusters = self.df[self.df['budget'] > 100_000_000].nlargest(3, 'revenue')
        print("\n\nHigh-budget blockbusters (should be successful):")
        print(blockbusters[['title', 'budget', 'revenue', 'roi', 'vote_average', 'success']].to_string(index=False))

if __name__== "__main__":
    preprocessor = MovieDataPreprocessor()
    preprocessor.clean_data()
    preprocessor.create_target_variable()
    preprocessor.feature_engineering()
    preprocessor.encode_categorical_features()
    X, y, feature_columns = preprocessor.prepare_final_dataset()
    preprocessor.save_processed_data(X, y, feature_columns)
    
    print("\n✓ Preprocessing complete with STRICTER criteria!")
    print("\n" + "="*70)
    print("Next step: Run 'python train_xgboost.py'")
    print("="*70)