import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    data = pd.read_csv(path)
    return data

def clean_data(df):
    # Drop unnecessary columns
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    if 'dataset' in df.columns:
        df = df.drop('dataset', axis=1)
        
    # Rename 'num' to 'target' if it exists (UCI dataset standard)
    if 'num' in df.columns:
        df = df.rename(columns={'num': 'target'})
        # Make it binary (0 = no disease, 1,2,3,4 = heart disease present)
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        
    # Rename thalch to thalach if exist
    if 'thalch' in df.columns:
        df = df.rename(columns={'thalch': 'thalach'})
        
    # Handle Categorical Columns and Map them to numerical
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    le = LabelEncoder()
    
    for col in categorical_cols:
        if col in df.columns and df[col].dtype == 'object' or df[col].dtype == 'bool':
            # Fill NaN for categorical with mode
            df[col] = df[col].fillna(df[col].mode()[0])
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            
    # For numerical cols, convert to numeric causing errors to NaN, then fill with mean
    for col in df.columns:
        if col not in categorical_cols and col != 'target':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())
            
    df = df.drop_duplicates()
    return df

def save_processed(df):
    df.to_csv("data/processed/cleaned_heart.csv", index=False)

if __name__ == "__main__":
    print("Loading raw data...")
    data = load_data("data/raw/heart.csv")
    print("Cleaning data...")
    clean = clean_data(data)
    save_processed(clean)
    print("Processed data saved. Shape:", clean.shape)
