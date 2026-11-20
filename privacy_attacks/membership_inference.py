import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "federated_model.keras"
CLIENT_PATH = ROOT / "models" / "fl_clients.pkl"
TEST_PATH = ROOT / "models" / "fl_test_data.pkl"

RESULT_ROOT = ROOT / "privacy_attacks" / "results"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = RESULT_ROOT / f"mia_run_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

RANDOM_STATE = 42


# --------------------------------------------------
# LOAD EXISTING DATA AND MODEL (READ-ONLY)
# --------------------------------------------------

print("Loading federated model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Loading participant training clients...")
clients = joblib.load(CLIENT_PATH)

print("Loading held-out test data...")
test_data = joblib.load(TEST_PATH)

member_X = np.concatenate(
    [clients[pid]["X"] for pid in clients],
    axis=0
)
member_y = np.concatenate(
    [clients[pid]["y"] for pid in clients],
    axis=0
)

nonmember_X = np.asarray(test_data["X"])
nonmember_y = np.asarray(test_data["y"])

if member_X.ndim != 2 or nonmember_X.ndim != 2:
    raise ValueError("Expected two-dimensional feature arrays.")

if member_X.shape[1] != 34 or nonmember_X.shape[1] != 34:
    raise ValueError("Expected exactly 34 HRV features.")

if len(member_X) != len(member_y):
    raise ValueError("Member feature and label counts do not match.")

if len(nonmember_X) != len(nonmember_y):
    raise ValueError("Non-member feature and label counts do not match.")

print("Member samples:", len(member_X))
print("Non-member samples:", len(nonmember_X))


# --------------------------------------------------
# EXTRACT MODEL OUTPUTS
# --------------------------------------------------

def get_attack_features(X, y):
    probabilities = model.predict(X, verbose=0).reshape(-1)

    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    y = np.asarray(y).astype(int).reshape(-1)

    if len(probabilities) != len(y):
        raise ValueError("Prediction and label counts do not match.")

    entropy = -(
        probabilities * np.log(probabilities)
        + (1 - probabilities) * np.log(1 - probabilities)
    )

    losses = -(
        y * np.log(probabilities)
        + (1 - y) * np.log(1 - probabilities)
    )

    predicted_labels = (probabilities >= 0.5).astype(int)
    correctness = (predicted_labels == y).astype(int)

    return np.column_stack([
        probabilities,
        entropy,
        losses,
        correctness
    ]), probabilities


print("Extracting member prediction features...")
member_features, member_probabilities = get_attack_features(
    member_X, member_y
)

print("Extracting non-member prediction features...")
nonmember_features, nonmember_probabilities = get_attack_features(
    nonmember_X, nonmember_y
)


# --------------------------------------------------
# CREATE BALANCED ATTACK DATASET
# --------------------------------------------------

sample_count = min(len(member_features), len(nonmember_features))

rng = np.random.default_rng(RANDOM_STATE)

member_indices = rng.choice(
    len(member_features),
    size=sample_count,
    replace=False
)

nonmember_indices = rng.choice(
    len(nonmember_features),
    size=sample_count,
    replace=False
)

X_attack = np.vstack([
    member_features[member_indices],
    nonmember_features[nonmember_indices]
])

# 1 = member, 0 = non-member
y_attack = np.concatenate([
    np.ones(sample_count, dtype=int),
    np.zeros(sample_count, dtype=int)
])

attack_probabilities = np.concatenate([
    member_probabilities[member_indices],
    nonmember_probabilities[nonmember_indices]
])

X_train, X_test, y_train, y_test, indices_train, indices_test = (
    train_test_split(
        X_attack,
        y_attack,
        np.arange(len(y_attack)),
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y_attack
    )
)


# --------------------------------------------------
# TRAIN ATTACK CLASSIFIER
# --------------------------------------------------

attack_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ))
])

print("Training membership inference classifier...")
attack_model.fit(X_train, y_train)

predictions = attack_model.predict(X_test)
membership_scores = attack_model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(
        y_test, predictions, zero_division=0
    )),
    "recall": float(recall_score(
        y_test, predictions, zero_division=0
    )),
    "f1_score": float(f1_score(
        y_test, predictions, zero_division=0
    )),
    "roc_auc": float(roc_auc_score(
        y_test, membership_scores
    )),
}

cm = confusion_matrix(y_test, predictions)

print("\nMEMBERSHIP INFERENCE RESULTS")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nConfusion matrix:")
print(cm)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

pd.DataFrame({
    "actual_membership": y_test,
    "predicted_membership": predictions,
    "attack_membership_probability": membership_scores,
    "target_model_probability": attack_probabilities[indices_test],
}).to_csv(
    OUTPUT_DIR / "attack_predictions.csv",
    index=False
)

pd.DataFrame(
    cm,
    index=["Actual non-member", "Actual member"],
    columns=["Predicted non-member", "Predicted member"]
).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

with (OUTPUT_DIR / "metrics.json").open("w") as f:
    json.dump(metrics, f, indent=4)

with (OUTPUT_DIR / "classification_report.txt").open("w") as f:
    f.write(classification_report(
        y_test,
        predictions,
        target_names=["Non-member", "Member"],
        zero_division=0
    ))

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, membership_scores)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Membership Inference Attack - ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mia_roc_curve.png", dpi=300)
plt.close()

experiment_info = {
    "experiment": "Membership Inference Attack",
    "model": str(MODEL_PATH),
    "member_source": str(CLIENT_PATH),
    "nonmember_source": str(TEST_PATH),
    "member_samples_available": int(len(member_X)),
    "nonmember_samples_available": int(len(nonmember_X)),
    "balanced_samples_per_class": int(sample_count),
    "attack_train_samples": int(len(X_train)),
    "attack_test_samples": int(len(X_test)),
    "attack_features": [
        "prediction_probability",
        "prediction_entropy",
        "binary_cross_entropy_loss",
        "prediction_correctness"
    ],
    "attack_classifier": "StandardScaler + LogisticRegression",
    "random_state": RANDOM_STATE,
    "metrics": metrics,
    "interpretation_note": (
        "This is an empirical supervised membership inference "
        "evaluation using stored participant-client training records "
        "and held-out test records. Results depend on the provenance "
        "of these saved datasets and do not establish a formal "
        "differential privacy guarantee."
    )
}

with (OUTPUT_DIR / "experiment_info.json").open("w") as f:
    json.dump(experiment_info, f, indent=4)

print("\nMEMBERSHIP INFERENCE EXPERIMENT COMPLETED")
print("Results saved in:", OUTPUT_DIR)