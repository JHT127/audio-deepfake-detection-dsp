"""
scripts/run_pipeline.py
-----------------------
End-to-end pipeline: load pre-computed frames -> extract features ->
train/evaluate classifier -> print metrics -> save all plots.

Run from repo root:
    python scripts/run_pipeline.py

Prerequisites:
    - data/split.csv  (created by scripts/make_split.py)
    - data/processed/{real,fake}/*.npy  (created by scripts/process_dataset.py)

Outputs (all in results/figures/):
    features.csv              -- extracted feature matrix (for inspection)
    confusion_matrix.png      -- confusion matrix plot
    feature_importance.png    -- Random Forest feature ranking
"""

import os
import sys

# Allow `from src.xxx import ...` when run from repo root or scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.features import extract_features, get_feature_names
from src.classify import (
    train_and_evaluate,
    feature_importance_report,
    print_evaluation_report,
)

SPLIT_CSV   = "data/split.csv"
PROCESSED   = "data/processed"
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def load_feature_matrix(split_df):
    """
    Iterate over split.csv, load each .npy frame array, extract features.

    Returns:
        X      : np.ndarray, shape (n_files, num_features)
        y      : np.ndarray, shape (n_files,)  1=real, 0=fake
        splits : list of str  "train" | "test"
    """
    feature_names = get_feature_names()
    X, y, splits = [], [], []
    n = len(split_df)

    for i, row in split_df.iterrows():
        npy_path = os.path.join(PROCESSED, row["label"], row["filename"])
        if not os.path.exists(npy_path):
            print(f"  WARNING: missing {npy_path} — skipping")
            continue

        frames = np.load(npy_path)          # (num_frames, 400)
        feat   = extract_features(frames)   # (14,)

        X.append(feat)
        y.append(1 if row["label"] == "real" else 0)
        splits.append(row["split"])

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}] features extracted …")

    return np.array(X), np.array(y), splits


def save_features_csv(X, y, splits, split_df, out_path):
    """Save feature matrix to CSV for inspection."""
    feature_names = get_feature_names()
    df = pd.DataFrame(X, columns=feature_names)
    df.insert(0, "label",    ["real" if yi == 1 else "fake" for yi in y])
    df.insert(1, "split",    splits)
    df.insert(2, "filename", split_df["filename"].values[:len(X)])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Features saved: {out_path}  ({df.shape})")


def main():
    print("=" * 60)
    print("  Audio Deepfake Detection — End-to-End Pipeline")
    print("=" * 60)

    print("\n[1/5] Reading data/split.csv …")
    split_df = pd.read_csv(SPLIT_CSV)
    print(f"      {len(split_df)} rows  "
          f"({(split_df.label=='real').sum()} real, "
          f"{(split_df.label=='fake').sum()} fake)")

    print(f"\n[2/5] Extracting {len(get_feature_names())} DSP features …")
    X, y, splits = load_feature_matrix(split_df)
    print(f"      Feature matrix: {X.shape}")

    features_csv = os.path.join(RESULTS_DIR, "features.csv")
    save_features_csv(X, y, splits, split_df, features_csv)

    print("\n[3/5] Splitting train / test …")
    splits_arr  = np.array(splits)
    train_mask  = splits_arr == "train"
    test_mask   = splits_arr == "test"

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]
    print(f"      Train: {X_train.shape[0]} samples   "
          f"Test: {X_test.shape[0]} samples")

    print("\n[4/5] Training SVM (RBF kernel) and evaluating …")
    metrics = train_and_evaluate(X_train, y_train, X_test, y_test,
                                 save_dir=FIGURES_DIR)
    print_evaluation_report(metrics)
    print(f"  Confusion matrix plot: {metrics['cm_plot_path']}")

    print("[5/5] Computing feature importance (Random Forest) …")
    importance = feature_importance_report(
        X_train, y_train,
        feature_names=get_feature_names(),
        save_dir=FIGURES_DIR,
    )
    print("  Feature importances (descending):")
    for name, imp in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"    {name:<35s}  {imp:.4f}  {bar}")
    print(f"\n  Feature importance plot: {os.path.join(FIGURES_DIR, 'feature_importance.png')}")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
