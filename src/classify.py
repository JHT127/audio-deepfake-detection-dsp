"""Classifier training and evaluation."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, classification_report,
)


def _build_classifier():
    """Return RBF-kernel SVM classifier."""
    return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       save_dir="results/figures"):
    """Train SVM classifier and evaluate. Returns metrics dict."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    clf = _build_classifier()
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")

    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fake (0)", "Real (1)"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — SVM (RBF)\nAudio Deepfake Detection")
    fig.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "accuracy":         acc,
        "precision":        prec,
        "recall":           rec,
        "f1":               f1,
        "roc_auc":          auc,
        "confusion_matrix": cm,
        "y_pred":           y_pred,
        "classifier":       clf,
        "scaler":           scaler,
        "cm_plot_path":     cm_path,
    }


def feature_importance_report(X_train, y_train, feature_names,
                               save_dir="results/figures"):
    """Train Random Forest and return feature importances dict."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y_train)

    importances = rf.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]

    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(importances)),
           importances[sorted_idx],
           color="steelblue")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in sorted_idx],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean decrease in impurity")
    ax.set_title("Feature Importance — Random Forest\nAudio Deepfake Detection")
    fig.tight_layout()
    imp_path = os.path.join(save_dir, "feature_importance.png")
    fig.savefig(imp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {feature_names[i]: float(importances[i]) for i in range(len(importances))}


def print_evaluation_report(metrics):
    """Print evaluation metrics."""
    print("\n" + "=" * 50)
    print("  CLASSIFICATION RESULTS — Audio Deepfake Detection")
    print("=" * 50)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print()
    print("  Confusion matrix (rows=true, cols=pred):")
    print("          Fake   Real")
    cm = metrics["confusion_matrix"]
    print(f"  Fake  [  {cm[0,0]:3d}    {cm[0,1]:3d}  ]")
    print(f"  Real  [  {cm[1,0]:3d}    {cm[1,1]:3d}  ]")
    print("=" * 50)
    print(f"  Confusion-matrix plot: {metrics.get('cm_plot_path', 'N/A')}")
    print()
