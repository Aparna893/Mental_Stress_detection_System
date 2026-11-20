"""
common.py - shared code for centralized, federated and DP-federated experiments.
Every experiment script imports from here so that all three models use the SAME
features, architecture, metrics and folder layout.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import GroupKFold

# ---------------------------------------------------------------- config
SEED = 42
DATA_PATH = Path("data/participant_hrv_34_features.csv")
RUNS_DIR = Path("runs")

FEATURES = [
    "MEAN_RR", "MEDIAN_RR", "SDRR", "RMSSD", "SDSD", "SDRR_RMSSD", "HR",
    "pNN25", "pNN50", "SD1", "SD2", "KURT", "SKEW", "MEAN_REL_RR",
    "MEDIAN_REL_RR", "SDRR_REL_RR", "RMSSD_REL_RR", "SDSD_REL_RR",
    "SDRR_RMSSD_REL_RR", "KURT_REL_RR", "SKEW_REL_RR", "VLF", "VLF_PCT",
    "LF", "LF_PCT", "LF_NU", "HF", "HF_PCT", "HF_NU", "TP", "LF_HF",
    "HF_LF", "sampen", "higuci",
]

# One architecture setting for ALL three models (change it here only).
ARCH = dict(filters=16, kernel=2, lstm_units=16, dense=32,
            l2=0.001, conv_dropout=0.2, lstm_dropout=0.1, dense_dropout=0.2,
            lr=0.001)


# ---------------------------------------------------------------- data
def load_data(path=DATA_PATH):
    """Load the participant-level HRV file and validate it."""
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + ["participant", "label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(df[FEATURES].to_numpy()).all():
        raise ValueError("Non-finite feature values found.")
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError("Labels must be 0/1.")
    return df


def participant_folds(df, n_splits=5):
    """Yield (fold, train_df, test_df) where test participants are NEVER in train."""
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr, te) in enumerate(gkf.split(df, df["label"], groups=df["participant"])):
        yield fold, df.iloc[tr].copy(), df.iloc[te].copy()


# ---------------------------------------------------------------- model
def build_model():
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Input, Reshape, Conv1D, Dropout,
                                         Bidirectional, LSTM, Dense)
    from tensorflow.keras.regularizers import l2
    a = ARCH
    n = len(FEATURES)
    model = Sequential([
        Input(shape=(n,)),
        Reshape((n, 1)),
        Conv1D(a["filters"], a["kernel"], activation="relu",
               kernel_regularizer=l2(a["l2"])),
        Dropout(a["conv_dropout"]),
        Bidirectional(LSTM(a["lstm_units"], dropout=a["lstm_dropout"],
                           recurrent_dropout=a["lstm_dropout"])),
        Dense(a["dense"], activation="relu", kernel_regularizer=l2(a["l2"])),
        Dropout(a["dense_dropout"]),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(a["lr"]),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ---------------------------------------------------------------- metrics
def evaluate(y_true, prob, threshold=0.5):
    """All metrics from probabilities. Includes majority baseline and AUC."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).ravel()
    pred = (prob >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred),
        "auc": roc_auc_score(y_true, prob) if len(set(y_true)) == 2 else float("nan"),
        "majority_baseline": max(np.mean(y_true), 1 - np.mean(y_true)),
        "n_test": int(len(y_true)),
        "predicted_stress_fraction": float(pred.mean()),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in out.items()}


# ---------------------------------------------------------------- outputs
def make_run_dir(name):
    """runs/<name>_<timestamp>/ so no experiment overwrites another."""
    d = RUNS_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
