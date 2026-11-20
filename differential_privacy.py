import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Reshape,
    Conv1D,
    Dropout,
    Bidirectional,
    LSTM,
    Dense
)
from tensorflow.keras.regularizers import l2


# ================================================================
# CONFIGURATION
# ================================================================

TRAIN_PATH = "data/comparison_train.csv"
TEST_PATH = "data/comparison_test.csv"
SCALER_PATH = "models/dp_federated_scaler.pkl"

MODEL_PATH = "models/dp_federated_model.keras"
WEIGHTS_PATH = "models/dp_federated_weights.weights.h5"

RANDOM_STATE = 42

NUM_ROUNDS = 5
LOCAL_EPOCHS = 2
BATCH_SIZE = 16

# Differential Privacy parameters
CLIPPING_NORM = 0.5
NOISE_MULTIPLIER = 0.1


np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ================================================================
# 34 HRV FEATURES
# ================================================================

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


# ================================================================
# CNN-BiLSTM MODEL
# ================================================================

def create_model():

    model = Sequential([
        Input(shape=(len(FEATURE_COLUMNS),)),

        Reshape((len(FEATURE_COLUMNS), 1)),

        Conv1D(
            filters=16,
            kernel_size=2,
            activation="relu",
            kernel_regularizer=l2(0.01)
        ),

        Dropout(0.5),

        Bidirectional(
            LSTM(
                16,
                dropout=0.5,
                recurrent_dropout=0.5
            )
        ),

        Dense(
            32,
            activation="relu",
            kernel_regularizer=l2(0.01)
        ),

        Dropout(0.5),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ================================================================
# DIFFERENTIAL PRIVACY: CLIP CLIENT UPDATE
# ================================================================

def clip_update(local_weights, global_weights, clipping_norm):

    # Calculate client update
    updates = []

    for local_w, global_w in zip(local_weights, global_weights):
        updates.append(local_w - global_w)

    # Calculate L2 norm of complete client update
    total_norm = np.sqrt(
        sum(np.sum(update ** 2) for update in updates)
    )

    # Clip update if norm exceeds clipping norm
    clipping_factor = min(
        1.0,
        clipping_norm / (total_norm + 1e-12)
    )

    clipped_updates = [
        update * clipping_factor
        for update in updates
    ]

    return clipped_updates, total_norm


# ================================================================
# DIFFERENTIAL PRIVACY: ADD GAUSSIAN NOISE
# ================================================================

def add_gaussian_noise(
    clipped_updates,
    clipping_norm,
    noise_multiplier
):

    noisy_updates = []

    noise_std = clipping_norm * noise_multiplier

    for update in clipped_updates:

        noise = np.random.normal(
            loc=0.0,
            scale=noise_std,
            size=update.shape
        )

        noisy_update = update + noise

        noisy_updates.append(noisy_update)

    return noisy_updates



# ================================================================
# LOAD COMMON COMPARISON DATASET
# ================================================================

print("\n" + "=" * 70)
print("LOADING COMMON COMPARISON DATASET")
print("=" * 70)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training dataset:", train_df.shape)
print("Testing dataset:", test_df.shape)

# Verify required columns
required_columns = FEATURE_COLUMNS + ["participant", "label"]

for col in required_columns:
    if col not in train_df.columns or col not in test_df.columns:
        raise ValueError(f"Missing required column: {col}")

# Get participants
participants = sorted(
    train_df["participant"].unique(),
    key=lambda x: int(str(x).replace("p", ""))
)

test_participants = set(test_df["participant"].unique())

if set(participants) != test_participants:
    raise ValueError("Training and testing participants do not match.")

# ================================================================
# CREATE FEDERATED CLIENTS USING THE EXISTING COMMON SPLIT
# ================================================================

train_parts = {}
test_parts = {}

for participant in participants:

    client_train = train_df[
        train_df["participant"] == participant
    ].copy()

    client_test = test_df[
        test_df["participant"] == participant
    ].copy()

    if client_train.empty or client_test.empty:
        raise ValueError(
            f"Missing training or testing samples for {participant}"
        )

    train_parts[participant] = {
        "X": client_train[FEATURE_COLUMNS].to_numpy(),
        "y": client_train["label"].astype(int).to_numpy()
    }

    test_parts[participant] = {
        "X": client_test[FEATURE_COLUMNS].to_numpy(),
        "y": client_test["label"].astype(int).to_numpy()
    }

# ================================================================
# FIT SCALER ONLY ON COMMON TRAINING DATA
# ================================================================

all_training_X = np.vstack([
    client["X"]
    for client in train_parts.values()
])

scaler = StandardScaler()

scaler.fit(all_training_X)

joblib.dump(
    scaler,
    SCALER_PATH
)

print("\nScaler fitted only on common training data.")
print("Saved:", SCALER_PATH)

# ================================================================
# SCALE CLIENT DATA
# ================================================================

for participant in participants:

    train_parts[participant]["X"] = scaler.transform(
        train_parts[participant]["X"]
    )

    test_parts[participant]["X"] = scaler.transform(
        test_parts[participant]["X"]
    )

# ================================================================
# VERIFY COMMON DATASET
# ================================================================

total_train = sum(
    len(train_parts[p]["y"])
    for p in participants
)

total_test = sum(
    len(test_parts[p]["y"])
    for p in participants
)

print("\nDP Federated dataset verification:")
print("Number of clients:", len(participants))
print("Total training samples:", total_train)
print("Total testing samples:", total_test)

if total_train != 1401 or total_test != 359:
    raise ValueError(
        "Dataset sizes do not match the common comparison dataset."
    )

print("\nDP Federated preprocessing completed successfully!")



# ================================================================
# DISPLAY CLIENT INFORMATION
# ================================================================

print("\n" + "=" * 70)
print("FEDERATED CLIENTS")
print("=" * 70)

for participant in participants:

    print(
        f"{participant}: "
        f"train={len(train_parts[participant]['y'])}, "
        f"test={len(test_parts[participant]['y'])}"
    )


# ================================================================
# INITIAL GLOBAL MODEL
# ================================================================

global_model = create_model()

global_weights = global_model.get_weights()


# ================================================================
# DIFFERENTIAL PRIVACY FEDERATED LEARNING
# ================================================================

print("\n" + "=" * 70)
print("STARTING FEDERATED LEARNING WITH DIFFERENTIAL PRIVACY")
print("=" * 70)

print(f"Rounds: {NUM_ROUNDS}")
print(f"Local epochs: {LOCAL_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Clipping norm: {CLIPPING_NORM}")
print(f"Noise multiplier: {NOISE_MULTIPLIER}")


for round_num in range(NUM_ROUNDS):

    print("\n" + "-" * 70)
    print(f"ROUND {round_num + 1}/{NUM_ROUNDS}")
    print("-" * 70)

    client_updates = []
    client_sizes = []

    for participant in participants:

        print(f"Training Client {participant}...")

        X_client = train_parts[participant]["X"]
        y_client = train_parts[participant]["y"]

        # Create local model
        local_model = create_model()

        # Start from current global model
        local_model.set_weights(global_weights)

        # Local training
        local_model.fit(
            X_client,
            y_client,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        local_weights = local_model.get_weights()

        # --------------------------------------------------------
        # STEP 1: CLIP CLIENT UPDATE
        # --------------------------------------------------------

        clipped_updates, original_norm = clip_update(
            local_weights,
            global_weights,
            CLIPPING_NORM
        )

        # --------------------------------------------------------
        # STEP 2: ADD GAUSSIAN NOISE
        # --------------------------------------------------------

        noisy_updates = add_gaussian_noise(
            clipped_updates,
            CLIPPING_NORM,
            NOISE_MULTIPLIER
        )

        client_updates.append(noisy_updates)
        client_sizes.append(len(y_client))

        print(
            f"{participant}: "
            f"{len(y_client)} samples | "
            f"update norm={original_norm:.4f}"
        )


    # ============================================================
    # WEIGHTED FEDAVG OF NOISY UPDATES
    # ============================================================

    total_samples = sum(client_sizes)

    new_global_weights = []

    for layer_index in range(len(global_weights)):

        aggregated_update = np.zeros_like(
            global_weights[layer_index]
        )

        for client_index in range(len(client_updates)):

            weight = (
                client_sizes[client_index]
                / total_samples
            )

            aggregated_update += (
                client_updates[client_index][layer_index]
                * weight
            )

        # Apply aggregated noisy update
        new_weight = (
            global_weights[layer_index]
            + aggregated_update
        )

        new_global_weights.append(new_weight)


    global_weights = new_global_weights

    global_model.set_weights(global_weights)


    # ============================================================
    # GLOBAL MODEL EVALUATION
    # ============================================================

    X_test_global = np.vstack([
        test_parts[p]["X"]
        for p in participants
    ])

    y_test_global = np.concatenate([
        test_parts[p]["y"]
        for p in participants
    ])

    loss, accuracy = global_model.evaluate(
        X_test_global,
        y_test_global,
        verbose=0
    )

    print(
        f"\nRound {round_num + 1} "
        f"Test Accuracy: {accuracy * 100:.2f}%"
    )


# ================================================================
# FINAL EVALUATION
# ================================================================

print("\n" + "=" * 70)
print("FINAL DP FEDERATED MODEL EVALUATION")
print("=" * 70)

probabilities = global_model.predict(
    X_test_global,
    verbose=0
).ravel()

predictions = (
    probabilities >= 0.5
).astype(int)

accuracy = accuracy_score(
    y_test_global,
    predictions
)

print(
    f"\nFinal Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test_global,
        predictions,
        target_names=[
            "No Stress",
            "Stress"
        ]
    )
)


# ================================================================
# SAVE DP MODEL
# ================================================================

global_model.save(MODEL_PATH)

global_model.save_weights(
    WEIGHTS_PATH
)

print("\nDP Federated model saved:")
print(MODEL_PATH)

print("\nDP Federated weights saved:")
print(WEIGHTS_PATH)

print("\n" + "=" * 70)
print("DIFFERENTIAL PRIVACY FEDERATED LEARNING COMPLETE")
print("=" * 70)