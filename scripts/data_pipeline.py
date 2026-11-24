"""
Unified Data Pipeline - Mice Protein Expression
===============================================
This module handles data loading, cleaning, and splitting.

Usage:
    from data_pipeline import get_train_val_test_split, prepare_eda_data, SEED
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Paths resolved relative to THIS file
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "mice_protein_expression.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"

METADATA_COLS = ["Genotype", "Treatment", "Behavior"]
TARGET_COL = "class"

TEST_SIZE = 0.30   # 70% train, 30% temp
VAL_SIZE = 0.50    # temp split in half -> 15% val, 15% test


# Shared loader
def load_raw_data(raw_path: Path | str = RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the raw CSV, strip column whitespace, drop leakage-prone metadata,
    coerce proteins to numeric, and return (X, y).

    Returns dataframe X and series y. Does NOT impute or scale.
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. "
            "See README for the dataset download/ingestion step."
        )

    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.strip()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing. Found: {df.columns.tolist()}")

    # Genotype/Treatment/Behavior
    drop_cols = [c for c in METADATA_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    X = df.drop(columns=[TARGET_COL]).apply(pd.to_numeric, errors="coerce")
    y = df[TARGET_COL].astype(str)

    return X, y


# Dataset splitting
def get_train_val_test_split(
    raw_path: Path | str = RAW_DATA_PATH,
    seed: int = SEED,
    stratify: bool = True,
):
    """
    RAW stratified splits: 70% train / 15% validation / 15% test.

    Do NOT impute or scale here. Wrap SimpleImputer + StandardScaler inside
    your model Pipeline so they are fitted on X_train only.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X, y = load_raw_data(raw_path)
    strat = y if stratify else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=strat
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=VAL_SIZE,
        random_state=seed,
        stratify=(y_temp if stratify else None),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# Unsupervised EDA dataset preparation
def prepare_eda_data(
    raw_path: Path | str = RAW_DATA_PATH,
    save: bool = True,
    filename: str = "mice_protein_processed.csv",
) -> pd.DataFrame:
    """
    Prepare a globally imputed + standardised matrix for EDA, PCA and K-Means.
    Appends `class_name` for colouring plots (never used as a model input).
    """
    X, y_raw = load_raw_data(raw_path)

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X.columns, index=X.index)
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    eda_df = X_scaled.copy()
    print(y)
    eda_df["class_encoded"] = y
    eda_df["class_name"] = y_raw.values

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / filename
        eda_df.to_csv(out_path, index=False)
        joblib.dump(imputer, MODELS_DIR / "eda_imputer.joblib")
        joblib.dump(scaler, MODELS_DIR / "eda_scaler.joblib")
        print(f"dataset written to {out_path}  shape={eda_df.shape}")

    return eda_df


def get_feature_names(raw_path: Path | str = RAW_DATA_PATH) -> list[str]:
    """Convenience helper for feature-importance plots / tables."""
    X, _ = load_raw_data(raw_path)
    return X.columns.tolist()