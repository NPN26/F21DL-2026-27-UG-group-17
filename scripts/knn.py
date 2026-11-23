"""
This script trains a KNN classifier on the Mice Protein Expression dataset.
It uses a pipeline for imputation and scaling, tunes the 'k' hyperparameter 
using a validation set, evaluates on a test set, and saves the best model.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


# Configuration
SEED = 42
np.random.seed(SEED) # Set global seed for any underlying stochastic processes

RAW_DATA_PATH = 'data/raw/mice_protein_expression.csv'
MODEL_SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'knn_best_model.joblib')
FIGURES_DIR = 'figures'

def main():
    # Ensure output directories exist
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Data Loading
    print(f"Loading data from {RAW_DATA_PATH}...")
    df_raw = pd.read_csv(RAW_DATA_PATH)
    
    # Clean column names (remove leading/trailing whitespace)
    df_raw.columns = df_raw.columns.str.strip()
    
    # Separate features and target
    metadata_cols = ['Genotype', 'Treatment', 'Behavior']
    drop_cols = [col for col in metadata_cols + ['class'] if col in df_raw.columns]
    
    X = df_raw.drop(columns=drop_cols)
    y = df_raw['class']
    
    # Ensure all feature columns are numeric (coerce errors to NaN for the imputer)
    X = X.apply(pd.to_numeric, errors='coerce')
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # First split: 70% training, 30% remaining
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    
    # Second split: divide remaining 30% equally (15% val, 15% test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    
    print(f"\nDataset Splits:")
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Testing set: {X_test.shape}")
    
    # Hyperparameter Tuning
    k_values = [1, 3, 5, 7, 9, 11, 15]
    
    best_k = 0
    best_f1 = 0.0
    best_model = None
    
    # History to store validation results
    tuning_results = []
    
    print("\n--- Hyperparameter Tuning ---")
    for k in k_values:
        # Create pipeline (Impute -> Scale -> KNN)
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=k))
        ])
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict validation data
        y_val_pred = model.predict(X_val)
        
        # Evaluate model
        accuracy = accuracy_score(y_val, y_val_pred)
        macro_f1 = f1_score(y_val, y_val_pred, average='macro')
        
        print(f"K = {k:2d} | Val Accuracy: {accuracy:.4f} | Val Macro F1: {macro_f1:.4f}")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_k = k
            best_model = model
            
        tuning_results.append((k, accuracy, macro_f1))
        
    # Hyperparameter tuning graph
    plt.figure(figsize=(8, 5))
    tuning_df = pd.DataFrame(tuning_results, columns=['k', 'val_accuracy', 'val_macro_f1'])
    sns.lineplot(data=tuning_df, x='k', y='val_macro_f1', marker='o')
    plt.title('KNN Hyperparameter Tuning (Validation Macro F1)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Validation Macro F1 Score')
    plt.xticks(k_values)
    plt.axvline(best_k, color='r', linestyle='--', label=f'Best k = {best_k}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'knn_hyperparameter_tuning.png'), dpi=150)
    plt.close()
            
    print(f"\n Best K: {best_k} with Validation Macro F1: {best_f1:.4f}")
    
    # 5. Final Evaluation on Test Set
    print("\n--- Test Set Evaluation ---")
    y_test_pred = best_model.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # 6. Save the Model
    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f"Best KNN model pipeline saved to {MODEL_SAVE_PATH}")
    print(f"Hyperparameter tuning plot saved to {os.path.join(FIGURES_DIR, 'knn_hyperparameter_tuning.png')}")

if __name__ == "__main__":
    main()