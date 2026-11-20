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


# ============================================================
# CONFIGURATION
# ============================================================

CLIENTS_PATH = "models/fl_clients.pkl"
TEST_DATA_PATH = "models/fl_test_data.pkl"
SCALER_PATH = "models/fl_scaler.pkl"

MODEL_PATH = "models/federated_model.keras"
WEIGHTS_PATH = "models/federated_weights.weights.h5"

RANDOM_STATE = 42

NUM_ROUNDS = 5
LOCAL_EPOCHS = 2
BATCH_SIZE = 16

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# FEATURES
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
# CNN-BILSTM HYBRID MODEL
# ============================================================

def create_model():

    model = Sequential([
        Input(shape=(len(FEATURE_COLUMNS),)),

        Reshape(
            (len(FEATURE_COLUMNS), 1)
        ),

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



# ============================================================
# LOAD PREPARED FEDERATED DATA
# ============================================================

print("=" * 65)
print("LOADING PREPARED FEDERATED DATA")
print("=" * 65)

clients = joblib.load(CLIENTS_PATH)
test_data = joblib.load(TEST_DATA_PATH)

# Training clients
train_parts = clients

# Separate test data
X_test_global = test_data["X"]
y_test_global = test_data["y"]
test_participant_ids = test_data["participant"]

participants = sorted(
    train_parts.keys(),
    key=lambda x: int(str(x).replace("p", ""))
)

# Validate the loaded data
if len(participants) != 23:
    raise ValueError(
        f"Expected 23 clients, found {len(participants)}"
    )

if len(X_test_global) != len(y_test_global):
    raise ValueError("Test feature and label counts do not match.")

if len(X_test_global) != len(test_participant_ids):
    raise ValueError("Test participant count does not match.")

if X_test_global.shape[1] != len(FEATURE_COLUMNS):
    raise ValueError("Unexpected number of test features.")

for participant in participants:

    X = train_parts[participant]["X"]
    y = train_parts[participant]["y"]

    if len(X) != len(y):
        raise ValueError(
            f"Training feature and label counts do not match for {participant}"
        )

    if X.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Unexpected feature count for {participant}"
        )

# Create test partitions for reporting by participant
test_parts = {}

for participant in participants:

    mask = test_participant_ids == participant

    test_parts[participant] = {
        "X": X_test_global[mask],
        "y": y_test_global[mask]
    }

    if len(test_parts[participant]["y"]) == 0:
        raise ValueError(
            f"No test samples found for {participant}"
        )

total_train = sum(
    len(train_parts[p]["y"])
    for p in participants
)

total_test = len(y_test_global)

if total_train != 1401 or total_test != 359:
    raise ValueError(
        f"Unexpected dataset sizes: "
        f"train={total_train}, test={total_test}"
    )

print("Number of clients:", len(participants))
print("Training samples:", total_train)
print("Testing samples:", total_test)
print("Number of HRV features:", len(FEATURE_COLUMNS))

print("\n" + "=" * 65)
print("FEDERATED CLIENT SPLITS")
print("=" * 65)

for participant in participants:

    print(
        f"{participant}: "
        f"train={len(train_parts[participant]['y'])}, "
        f"test={len(test_parts[participant]['y'])}"
    )

print("\nPrepared federated data loaded successfully!")


# ============================================================
# INITIAL GLOBAL MODEL
# ============================================================

global_model = create_model()

global_weights = global_model.get_weights()

print("\n" + "=" * 65)
print("STARTING FEDERATED LEARNING")
print("=" * 65)


# ============================================================
# FEDERATED TRAINING
# ============================================================

for round_num in range(NUM_ROUNDS):

    print(
        f"\nROUND {round_num + 1}/{NUM_ROUNDS}"
    )

    client_weights = []
    client_sizes = []

    for participant, client_data in train_parts.items():

        X_client = client_data["X"]
        y_client = client_data["y"]

        local_model = create_model()

        local_model.set_weights(
            global_weights
        )

        local_model.fit(
            X_client,
            y_client,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        client_weights.append(
            local_model.get_weights()
        )

        client_sizes.append(
            len(y_client)
        )

        print(
            f"{participant}: "
            f"{len(y_client)} samples trained"
        )


    # ========================================================
    # WEIGHTED FEDAVG
    # ========================================================

    total_samples = sum(client_sizes)

    new_global_weights = []

    for layer_weights in zip(*client_weights):

        weighted_layer = np.zeros_like(
            layer_weights[0]
        )

        for weights, sample_count in zip(
            layer_weights,
            client_sizes
        ):

            weighted_layer += (
                weights *
                (sample_count / total_samples)
            )

        new_global_weights.append(
            weighted_layer
        )

    global_weights = new_global_weights

    global_model.set_weights(
        global_weights
    )


    # ========================================================
    # ROUND EVALUATION
    # ========================================================

    X_test_global = np.vstack([
        test_parts[p]["X"]
        for p in test_parts
    ])

    y_test_global = np.concatenate([
        test_parts[p]["y"]
        for p in test_parts
    ])

    loss, accuracy = global_model.evaluate(
        X_test_global,
        y_test_global,
        verbose=0
    )

    print(
        f"Round {round_num + 1} "
        f"Test Accuracy: "
        f"{accuracy * 100:.2f}%"
    )


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "=" * 65)
print("FINAL GLOBAL MODEL EVALUATION")
print("=" * 65)

X_test_global = np.vstack([
    test_parts[p]["X"]
    for p in test_parts
])

y_test_global = np.concatenate([
    test_parts[p]["y"]
    for p in test_parts
])

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


# ============================================================
# SAVE MODEL
# ============================================================

global_model.save(
    MODEL_PATH
)

global_model.save_weights(
    WEIGHTS_PATH
)

print("\nFederated model saved:")
print(MODEL_PATH)

print("\nFederated weights saved:")
print(WEIGHTS_PATH)

print("\nFEDERATED LEARNING COMPLETE")