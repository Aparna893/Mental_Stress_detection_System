import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/participant_hrv_34_features.csv"
SCALER_PATH = "models/fl_scaler.pkl"


# ============================================================
# 34 HRV FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "MEAN_RR",
    "MEDIAN_RR",
    "SDRR",
    "RMSSD",
    "SDSD",
    "SDRR_RMSSD",
    "HR",
    "pNN25",
    "pNN50",
    "SD1",
    "SD2",
    "KURT",
    "SKEW",
    "MEAN_REL_RR",
    "MEDIAN_REL_RR",
    "SDRR_REL_RR",
    "RMSSD_REL_RR",
    "SDSD_REL_RR",
    "SDRR_RMSSD_REL_RR",
    "KURT_REL_RR",
    "SKEW_REL_RR",
    "VLF",
    "VLF_PCT",
    "LF",
    "LF_PCT",
    "LF_NU",
    "HF",
    "HF_PCT",
    "HF_NU",
    "TP",
    "LF_HF",
    "HF_LF",
    "sampen",
    "higuci"
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING PARTICIPANT HRV DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = (
    FEATURE_COLUMNS
    + ["participant", "label"]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

print("34 HRV feature columns found.")
print("Participant column found.")
print("Label column found.")


# ============================================================
# CHECK MISSING / INFINITE VALUES
# ============================================================

if df[FEATURE_COLUMNS].isnull().sum().sum() > 0:
    raise ValueError(
        "Missing values found in HRV features."
    )

if not df[FEATURE_COLUMNS].map(
    lambda x: pd.api.types.is_number(x)
).all().all():
    raise ValueError(
        "Non-numeric values found in HRV features."
    )

print("No missing feature values.")


# ============================================================
# CHECK LABELS
# ============================================================

valid_labels = {0, 1}

actual_labels = set(
    df["label"].unique()
)

if not actual_labels.issubset(valid_labels):
    raise ValueError(
        f"Unexpected labels found: {actual_labels}"
    )

print("\nClass distribution:")
print(df["label"].value_counts().sort_index())


# ============================================================
# FIT STANDARD SCALER
# ============================================================

print("\n" + "=" * 60)
print("FITTING STANDARD SCALER")
print("=" * 60)

scaler = StandardScaler()

df[FEATURE_COLUMNS] = scaler.fit_transform(
    df[FEATURE_COLUMNS]
)

joblib.dump(
    scaler,
    SCALER_PATH
)

print("Scaler saved to:")
print(SCALER_PATH)


# ============================================================
# CREATE PARTICIPANT CLIENTS
# ============================================================

clients = {}

for participant in sorted(
    df["participant"].unique(),
    key=lambda x: int(x.replace("p", ""))
):

    participant_df = df[
        df["participant"] == participant
    ].copy()

    X = participant_df[
        FEATURE_COLUMNS
    ].values

    y = participant_df[
        "label"
    ].values

    clients[participant] = {
        "X": X,
        "y": y
    }


# ============================================================
# DISPLAY CLIENT INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("FEDERATED CLIENTS")
print("=" * 60)

print(
    f"Total clients: {len(clients)}"
)

for participant, data in clients.items():

    X = data["X"]
    y = data["y"]

    no_stress = (y == 0).sum()
    stress = (y == 1).sum()

    print(
        f"{participant}: "
        f"samples={len(y)}, "
        f"no_stress={no_stress}, "
        f"stress={stress}"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREPARATION COMPLETE")
print("=" * 60)

print("Total samples:", len(df))
print("Total features:", len(FEATURE_COLUMNS))
print("Total clients:", len(clients))
print("Scaler:", SCALER_PATH)

print("\nReady for federated learning.")