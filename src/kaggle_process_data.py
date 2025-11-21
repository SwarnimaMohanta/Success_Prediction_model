#process_kaggle_data.py
import pandas as pd
import os
import ast
import numpy as np

class KaggleDataProcessor:
    def __init__(self):
        self.raw_dir = os.path.join('data', 'raw')
        self.movies_path = os.path.join(self.raw_dir, 'tmdb_5000_movies.csv')
        self.credits_path = os.path.join(self.raw_dir, 'tmdb_5000_credits.csv')
        self.output_path = os.path.join(self.raw_dir, 'movies_dataset.csv')


    def load_and_merge(self):
        """Load the two CSVs and merge them into one"""
        print("Loading Kaggle datasets...")
        
        if not os.path.exists(self.movies_path) or not os.path.exists(self.credits_path):
            raise FileNotFoundError(f"❌ Could not find files. Please make sure 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' are in {self.raw_dir}")

        # Load datasets
        df_movies = pd.read_csv(self.movies_path)
        df_credits = pd.read_csv(self.credits_path)

        print(f"Movies shape: {df_movies.shape}")
        print(f"Credits shape: {df_credits.shape}")

        # Merge datasets on ID (movies uses 'id', credits uses 'movie_id')
        print("Merging datasets...")
        df_merged = df_movies.merge(df_credits, left_on='id', right_on='movie_id')
        
        # Drop redundant columns
        df_merged.drop(columns=['movie_id', 'title_y', 'homepage', 'status', 'video'], errors='ignore', inplace=True)
        df_merged.rename(columns={'title_x': 'title'}, inplace=True)
        
        return df_merged

    def safe_extract(self, json_str, key='name', limit=None):
        """Helper to parse JSON strings (e.g., '[{"id": 28, "name": "Action"}]')"""
        try:
            # If it's already a list (rare but possible), just use it
            if isinstance(json_str, list):
                data = json_str
            # Parse string to list of dicts
            else:
                data = ast.literal_eval(json_str)
            
            # Extract the specific key (usually 'name')
            names = [item[key] for item in data]
            
            # Limit the number of items if requested (e.g., top 5 actors)
            if limit:
                return names[:limit]
            return names
        except:
            return []

    def extract_crew_job(self, json_str, job_role, limit=None):
        """Helper specifically for Crew to find Director or Producers"""
        try:
            data = ast.literal_eval(json_str)
            names = [item['name'] for item in data if item['job'] == job_role]
            if limit:
                return names[:limit]
            # If looking for Director, just return the first string, not a list
            if job_role == 'Director':
                return names[0] if names else None
            return names
        except:
            return [] if job_role != 'Director' else None

    def process_data(self, df):
        """Apply transformations to match API format"""
        print("Processing columns (Parsing JSON)... this may take a moment.")

        # 1. Parse JSON columns to simple lists of names
        # Genres
        df['genres'] = df['genres'].apply(lambda x: self.safe_extract(x))
        
        # Keywords (Limit to top 10)
        df['keywords'] = df['keywords'].apply(lambda x: self.safe_extract(x, limit=10))
        
        # Production Companies (Limit to top 3)
        df['production_companies'] = df['production_companies'].apply(lambda x: self.safe_extract(x, limit=3))
        
        # Cast (Limit to top 5 actors)
        df['cast'] = df['cast'].apply(lambda x: self.safe_extract(x, limit=5))

        # 2. Extract Crew (Director & Producers)
        df['director'] = df['crew'].apply(lambda x: self.extract_crew_job(x, 'Director'))
        df['producers'] = df['crew'].apply(lambda x: self.extract_crew_job(x, 'Producer', limit=2))

        # 3. Calculate Success Metrics (ROI & Profit)
        # Avoid division by zero
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
        df['budget'] = pd.to_numeric(df['budget'], errors='coerce').fillna(0)

        df['profit'] = df['revenue'] - df['budget']
        
        # Calculate ROI safely
        df['roi'] = df.apply(lambda row: (row['revenue'] - row['budget']) / row['budget'] 
                             if row['budget'] > 0 else np.nan, axis=1)

        # 4. Select and Order Columns
        # We keep only what we need for the model
        keep_cols = [
            'id', 'title', 'release_date', 'budget', 'revenue', 
            'runtime', 'vote_average', 'vote_count', 'popularity', 
            'original_language', 'genres', 'production_companies', 
            'cast', 'director', 'producers', 'keywords', 'overview', 
            'roi', 'profit'
        ]
        
        # Filter columns that exist
        final_cols = [c for c in keep_cols if c in df.columns]
        return df[final_cols]

    def run(self):
        """Main execution flow"""
        try:
            merged_df = self.load_and_merge()
            processed_df = self.process_data(merged_df)
            
            # Save
            processed_df.to_csv(self.output_path, index=False)
            print(f"\n✅ Success! Processed dataset saved to: {self.output_path}")
            print(f"Total Movies: {len(processed_df)}")
            print("You can now proceed to 'src/preprocess.py'")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__== "__main__":
    processor = KaggleDataProcessor()
    processor.run()