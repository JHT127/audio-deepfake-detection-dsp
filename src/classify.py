"""
Classification and evaluation module — Person C.

Task 7: Train a simple classifier on features from src/features.py.
Task 9: Evaluate and report metrics.

Expected input from Person B:
    extract_features(frames) -> 1D np.ndarray, shape (num_features,)

Usage pattern:
    import pandas as pd, numpy as np
    from src.features import extract_features

    split_df = pd.read_csv("data/split.csv")
    X, y, splits = [], [], []
    for _, row in split_df.iterrows():
        frames = np.load(f"data/processed/{row.label}/{row.filename}")
        X.append(extract_features(frames))
        y.append(1 if row.label == "real" else 0)
        splits.append(row.split)

    X = np.array(X)   # shape (300, num_features)
    y = np.array(y)   # shape (300,)

Required metrics (Task 9):
    accuracy, precision, recall, F1-score, confusion matrix
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Train classifier and return evaluation metrics.

    Args:
        X_train, X_test: feature matrices, shape (n_samples, num_features).
        y_train, y_test: label vectors (1=real, 0=fake).

    Returns:
        dict of metrics: accuracy, precision, recall, f1, confusion_matrix.
    """
    raise NotImplementedError("Person C: implement classifier here")