"""
This script trains a Decision Tree classifier on the Mice Protein Expression dataset.
It uses a pipeline for imputation, tunes the 'max_depth' hyperparameter using a validation set, 
evaluates on a test set, and saves the best model along with diagnostic plots.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from data_pipeline import FIGURES_DIR, MODELS_DIR, SEED, get_train_val_test_split

MODEL_PATH = MODELS_DIR / "decision_tree_best_model.joblib"
TUNING_PLOT_PATH = FIGURES_DIR / "decision_tree_tuning_curve.png"
DIAGNOSTICS_PLOT_PATH = FIGURES_DIR / "decision_tree_diagnostics.png"


def make_pipeline(depth: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # fitted on TRAIN only
        ("tree", DecisionTreeClassifier(max_depth=depth, random_state=SEED)),
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

    # 2. Tune max_depth on the validation set (test stays untouched)
    print("\n--- Hyperparameter tuning (max_depth) ---")
    rows = []
    best_f1, best_depth, best_model = -1.0, None, None

    for depth in range(2, 16):
        model = make_pipeline(depth)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        rows.append({"max_depth": depth, "val_accuracy": val_acc, "val_macro_f1": val_f1})
        print(f"depth={depth:2d} | val acc: {val_acc:.4f} | val macro F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_depth, best_model = val_f1, depth, model

    tuning = pd.DataFrame(rows)
    print(f"\nBest depth: {best_depth} (val macro F1: {best_f1:.4f})")

    # Tuning curve
    plt.figure(figsize=(7, 4))
    plt.plot(tuning["max_depth"], tuning["val_macro_f1"], marker="o")
    plt.axvline(best_depth, color="r", ls="--", label=f"Best depth ({best_depth})")
    plt.title("Validation Macro F1 vs. Tree Depth")
    plt.xlabel("max_depth")
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

    # 4. Feature importance + confusion matrix
    importances = pd.Series(
        best_model.named_steps["tree"].feature_importances_, index=X_train.columns
    )
    top10 = importances.sort_values(ascending=False).head(10)
    print("Top 10 proteins (Gini importance):")
    print(top10.to_string())

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(x=top10.values, y=top10.index, hue=top10.index,
                palette="viridis", legend=False, ax=ax[0])
    ax[0].set_title(f"Top 10 Proteins (depth={best_depth})")
    ax[0].set_xlabel("Gini importance")
    ax[0].set_ylabel("")

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[1],
                xticklabels=labels, yticklabels=labels)
    ax[1].set_title("Test Set Confusion Matrix")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(DIAGNOSTICS_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Save the best pipeline (imputer + tree in one artifact)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved plots -> {TUNING_PLOT_PATH}")
    print(f"               {DIAGNOSTICS_PLOT_PATH}")


if __name__ == "__main__":
    main()