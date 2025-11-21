#make_scaler.py 
import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Create directory
os.makedirs('data/processed', exist_ok=True)

# Create dummy data structure matching your project
typical_data = np.array([
    [40000000, 105, 6.2, 500, 8, 2005, 2.5, 2, 6.2, 400000, 0.02, 0, 1],
    [80000000, 120, 7.0, 2000, 20, 2010, 3, 3, 7.6, 666666, 0.01, 0.5, 0.866]
])

# Fit and save scaler
scaler = StandardScaler()
scaler.fit(typical_data)

with open('data/processed/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Scaler created successfully!")