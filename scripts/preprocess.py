import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_mice_data(raw_path='../data/raw/mice_protein_expression.csv', 
                         processed_dir='../data/processed',
                         models_dir='../models'):
    """
    Cleans, imputes, and scales the Mice Protein Expression dataset.
    Saves the processed data.
    """
    # Create directories if they don't exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading data from {raw_path}...")
    df_raw = pd.read_csv(raw_path)
    df = df_raw.copy()
    
    # Clean column names
    df.columns = df.columns.str.strip()

    # Identify target and drop metadata
    target_col = df.columns[-1]  # 'class'
    metadata_cols = ['Genotype', 'Treatment', 'Behavior']
    
    if all(col in df.columns for col in metadata_cols):
        df.drop(columns=metadata_cols, inplace=True)
        
    protein_cols = df.columns[:-1]
    
    # Convert protein columns to numeric
    df[protein_cols] = df[protein_cols].apply(pd.to_numeric, errors='coerce')
    
    # Separate X and y
    X = df[protein_cols].copy()
    y_raw = df[target_col].astype(str).copy()
    
    # Impute missing values (Median)
    print("Imputing missing values...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=protein_cols, index=X.index)
    
    # Encode target variable
    print("Encoding target variable...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    # Normalize features (Standardization)
    print("Normalizing features...")
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_imputed)
    X_normalized_df = pd.DataFrame(X_normalized, columns=protein_cols, index=X_imputed.index)
    
    # Save processed data
    processed_df = X_normalized_df.copy()
    processed_df['class_encoded'] = y
    processed_df['class_name'] = y_raw.values
    
    processed_path = os.path.join(processed_dir, 'mice_protein_processed.csv')
    processed_df.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")
    
    return X_normalized_df, y, y_raw, label_encoder

if __name__ == "__main__":
    preprocess_mice_data()