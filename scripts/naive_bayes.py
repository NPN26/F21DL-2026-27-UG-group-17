"""
Naive Bayes Classifier for Mice Protein Expression Dataset
This script trains a Gaussian Naive Bayes classifier on the Mice Protein Expression dataset.
It uses a pipeline for imputation and scaling, tunes the 'var_smoothing' hyperparameter
using a validation set, evaluates on a test set, and saves the best model along with diagnostic plots.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_pipeline import FIGURES_DIR, MODELS_DIR, SEED, get_train_val_test_split

MODEL_PATH = MODELS_DIR / "naive_bayes_best_model.joblib"
TUNING_PLOT_PATH = FIGURES_DIR / "naive_bayes_tuning_curve.png"
CM_PLOT_PATH = FIGURES_DIR / "naive_bayes_confusion_matrix.png"


def make_pipeline(var_smoothing: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # fitted on TRAIN only
        ("scaler", StandardScaler()),                   # fitted on TRAIN only
        ("nb", GaussianNB(var_smoothing=var_smoothing)),
    ])


def main():
    np.random.seed(SEED)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data — identical splits across all R3/R4 models
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_split()
    labels = sorted(y_train.unique())
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Features: {X_train.shape[1]} proteins | Classes: {len(labels)}")

    # 2. Tune var_smoothing on the validation set (test stays untouched)
    print("\n--- Hyperparameter tuning (var_smoothing) ---")
    smoothings = np.logspace(-9, 0, 10)  # 1e-9 (sklearn default) -> 1e0
    rows = []
    best_f1, best_vs, best_model = -1.0, None, None

    for vs in smoothings:
        model = make_pipeline(vs)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        rows.append({"var_smoothing": vs, "val_accuracy": val_acc, "val_macro_f1": val_f1})
        print(f"var_smoothing={vs:.0e} | val acc: {val_acc:.4f} | val macro F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_vs, best_model = val_f1, vs, model

    tuning = pd.DataFrame(rows)
    print(f"\nBest var_smoothing: {best_vs:.0e} (val macro F1: {best_f1:.4f})")

    # Tuning curve (log x-axis)
    plt.figure(figsize=(7, 4))
    plt.semilogx(tuning["var_smoothing"], tuning["val_macro_f1"], marker="o")
    plt.axvline(best_vs, color="r", ls="--", label=f"Best ({best_vs:.0e})")
    plt.title("Validation Macro F1 vs. var_smoothing")
    plt.xlabel("var_smoothing (log scale)")
    plt.ylabel("Validation Macro F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(TUNING_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Final evaluation on the test set
    y_pred = best_model.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}\n")
    print(classification_report(y_test, y_pred, labels=labels, digits=3))

    # 4. Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Naive Bayes — Test Confusion Matrix (var_smoothing={best_vs:.0e})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CM_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Save the best pipeline (imputer + scaler + NB in one artifact)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved plots -> {TUNING_PLOT_PATH}")
    print(f"               {CM_PLOT_PATH}")


if __name__ == "__main__":
    main()