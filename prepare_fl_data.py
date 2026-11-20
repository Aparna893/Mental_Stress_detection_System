
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "data" / "comparison_train.csv"
TEST_PATH = BASE_DIR / "data" / "comparison_test.csv"

MODEL_DIR = BASE_DIR / "models"

SCALER_PATH = MODEL_DIR / "fl_scaler.pkl"
CLIENTS_PATH = MODEL_DIR / "fl_clients.pkl"
VAL_DATA_PATH = MODEL_DIR / "fl_validation_data.pkl"
TEST_DATA_PATH = MODEL_DIR / "fl_test_data.pkl"

VALIDATION_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# HRV FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "MEAN_RR", "MEDIAN_RR", "SDRR", "RMSSD",
    "SDSD", "SDRR_RMSSD", "HR", "pNN25",
    "pNN50", "SD1", "SD2", "KURT", "SKEW",
    "MEAN_REL_RR", "MEDIAN_REL_RR", "SDRR_REL_RR",
    "RMSSD_REL_RR", "SDSD_REL_RR",
    "SDRR_RMSSD_REL_RR", "KURT_REL_RR",
    "SKEW_REL_RR", "VLF", "VLF_PCT", "LF",
    "LF_PCT", "LF_NU", "HF", "HF_PCT",
    "HF_NU", "TP", "LF_HF", "HF_LF",
    "sampen", "higuci"
]

REQUIRED_COLUMNS = FEATURE_COLUMNS + [
    "participant", "label"
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING ORIGINAL TRAINING AND TESTING DATA")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

for name, data in [
    ("Training", train_df),
    ("Testing", test_df)
]:
    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{name} data is missing columns: {missing}"
        )

print("Original training samples:", len(train_df))
print("Original testing samples:", len(test_df))


# ============================================================
# VALIDATE DATA
# ============================================================

for name, data in [
    ("Training", train_df),
    ("Testing", test_df)
]:
    features = data[FEATURE_COLUMNS].to_numpy(dtype=float)

    if not np.isfinite(features).all():
        raise ValueError(
            f"{name} data contains missing or infinite features."
        )

    if not set(data["label"].unique()).issubset({0, 1}):
        raise ValueError(
            f"{name} data contains invalid labels."
        )

    if data["participant"].isnull().any():
        raise ValueError(
            f"{name} data contains missing participant IDs."
        )

print("Data validation successful.")


# ============================================================
# PARTICIPANT-WISE TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 60)
print("CREATING PARTICIPANT-WISE VALIDATION SPLIT")
print("=" * 60)

train_parts = []
validation_parts = []

participants = sorted(
    train_df["participant"].unique(),
    key=lambda x: int(str(x).replace("p", ""))
)

for participant in participants:

    participant_df = train_df[
        train_df["participant"] == participant
    ].copy()

    # Stratify when each class has at least 2 samples.
    class_counts = participant_df["label"].value_counts()

    stratify_labels = (
        participant_df["label"]
        if len(class_counts) == 2
        and class_counts.min() >= 2
        else None
    )

    participant_train, participant_val = train_test_split(
        participant_df,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_labels
    )

    train_parts.append(participant_train)
    validation_parts.append(participant_val)

    print(
        f"{participant}: "
        f"train={len(participant_train)}, "
        f"validation={len(participant_val)}"
    )

new_train_df = pd.concat(
    train_parts,
    ignore_index=True
)

validation_df = pd.concat(
    validation_parts,
    ignore_index=True
)

print("\nNew training samples:", len(new_train_df))
print("Validation samples:", len(validation_df))


# ============================================================
# FIT SCALER ON NEW TRAINING DATA ONLY
# ============================================================

print("\n" + "=" * 60)
print("SCALING HRV FEATURES")
print("=" * 60)

scaler = StandardScaler()

new_train_df.loc[:, FEATURE_COLUMNS] = scaler.fit_transform(
    new_train_df[FEATURE_COLUMNS]
)

validation_df.loc[:, FEATURE_COLUMNS] = scaler.transform(
    validation_df[FEATURE_COLUMNS]
)

# Original test data also uses the same training-fitted scaler.
test_df.loc[:, FEATURE_COLUMNS] = scaler.transform(
    test_df[FEATURE_COLUMNS]
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(scaler, SCALER_PATH)

print("Scaler fitted on new training data only.")
print("Validation and test data transformed using that scaler.")


# ============================================================
# CREATE PARTICIPANT-SPECIFIC FEDERATED CLIENTS
# ============================================================

print("\n" + "=" * 60)
print("CREATING FEDERATED CLIENTS")
print("=" * 60)

clients = {}

for participant in participants:

    participant_df = new_train_df[
        new_train_df["participant"] == participant
    ].copy()

    X = participant_df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )

    y = participant_df["label"].to_numpy(
        dtype=np.int64
    )

    clients[participant] = {
        "X": X,
        "y": y
    }

    print(
        f"{participant}: "
        f"samples={len(y)}, "
        f"no_stress={(y == 0).sum()}, "
        f"stress={(y == 1).sum()}"
    )


# ============================================================
# PREPARE VALIDATION DATA
# ============================================================

X_val = validation_df[FEATURE_COLUMNS].to_numpy(
    dtype=np.float32
)

y_val = validation_df["label"].to_numpy(
    dtype=np.int64
)

val_participants = validation_df["participant"].to_numpy()

validation_data = {
    "X": X_val,
    "y": y_val,
    "participant": val_participants
}


# ============================================================
# PREPARE FINAL TEST DATA
# ============================================================

X_test = test_df[FEATURE_COLUMNS].to_numpy(
    dtype=np.float32
)

y_test = test_df["label"].to_numpy(
    dtype=np.int64
)

test_participants = test_df["participant"].to_numpy()

test_data = {
    "X": X_test,
    "y": y_test,
    "participant": test_participants
}


# ============================================================
# SAVE ALL PREPARED DATA
# ============================================================

joblib.dump(clients, CLIENTS_PATH)
joblib.dump(validation_data, VAL_DATA_PATH)
joblib.dump(test_data, TEST_DATA_PATH)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE")
print("=" * 60)

print("New training samples:", len(new_train_df))
print("Validation samples:", len(validation_df))
print("Final test samples:", len(test_df))
print("HRV features:", len(FEATURE_COLUMNS))
print("Federated clients:", len(clients))

print("\nValidation class distribution:")
print(
    pd.Series(y_val)
    .value_counts()
    .sort_index()
    .rename(index={0: "No Stress", 1: "Stress"})
)

print("\nTest class distribution:")
print(
    pd.Series(y_test)
    .value_counts()
    .sort_index()
    .rename(index={0: "No Stress", 1: "Stress"})
)

print("\nSaved files:")
print(SCALER_PATH)
print(CLIENTS_PATH)
print(VAL_DATA_PATH)
print(TEST_DATA_PATH)

print("\nReady for validation-based federated training.")
