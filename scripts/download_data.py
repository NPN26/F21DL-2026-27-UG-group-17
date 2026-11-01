import os
import shutil
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
from medmnist import BloodMNIST

def setup_directories():
    """
    Create necessary directories for data storage.
    """
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
def download_mice_protien():
    print("Downloading Mice Protein Expression dataset...")
    # Fetch the Mice Protein Expression dataset from UCI repository 
    mice_protein_expression = fetch_ucirepo(id=342) 
  
    # data (as pandas dataframes) 
    X = mice_protein_expression.data.features 
    y = mice_protein_expression.data.targets 
    
    # Save the data to CSV files
    df = pd.concat([X, y], axis=1)
    df.to_csv('data/raw/mice_protein_expression.csv', index=False)
    
    print("Mice Protein Expression dataset downloaded and saved to 'data/raw/mice_protein_expression.csv'.")

def download_blood_mnist():
    print("Downloading BloodMNIST dataset...")
    # Download the BloodMNIST dataset using medmnist library
    train_blood_mnist = BloodMNIST(split='train', download=True, size=128, root='data/raw')
    val_blood_mnist = BloodMNIST(split='val', download=True, size=128, root='data/raw')
    test_blood_mnist = BloodMNIST(split='test', download=True, size=128, root='data/raw')
    
    print("BloodMNIST dataset downloaded and saved to 'data/raw/BloodMNIST'.")
    
if __name__ == "__main__":
    setup_directories()
    download_mice_protien()
    download_blood_mnist()
    print("Data pipeline completed successfully. All datasets are downloaded and stored in the 'data/raw' directory.")
