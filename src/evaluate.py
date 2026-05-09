import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "results", "models")
METRICS_DIR = os.path.join(PROJECT_ROOT, "results", "metrics")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "results", "plots")


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary", pos_label=1),
        "recall": recall_score(y_test, y_pred, average="binary", pos_label=1),
        "f1": f1_score(y_test, y_pred, average="binary", pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def evaluate_all_models(models_dir, X_test, y_test):
    rows = []
    for fname in sorted(os.listdir(models_dir)):
        if not fname.endswith(".joblib"):
            continue
        model_name = fname[:-len(".joblib")]
        model = joblib.load(os.path.join(models_dir, fname))
        metrics = evaluate_model(model, X_test, y_test)

        cm = metrics.pop("confusion_matrix")
        cm_path = os.path.join(PLOTS_DIR, f"cm_{model_name}.png")
        plot_confusion_matrix(cm, model_name, cm_path)

        rows.append({"model": model_name, **metrics})

    df = pd.DataFrame(rows, columns=["model", "accuracy", "precision", "recall", "f1", "roc_auc"])
    return df


def plot_roc_curve_single(model, X_test, y_test, model_name, save_path):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curves_combined(models_dict, X_test, y_test, save_path):
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#17becf",
    ]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")

    for (model_name, model), color in zip(models_dict.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{model_name} (AUC = {auc:.4f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All 8 Classifiers")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, model_name, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No disease", "Disease"],
        yticklabels=["No disease", "Disease"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
