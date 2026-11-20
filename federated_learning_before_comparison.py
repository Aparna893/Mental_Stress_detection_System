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

DATA_PATH = "data/participant_hrv_34_features.csv"

SCALER_PATH = "models/federated_scaler.pkl"
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
# LOAD DATA
# ============================================================

print("=" * 65)
print("LOADING PARTICIPANT DATA")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

print("Dataset:", df.shape)
print("Participants:", df["participant"].nunique())


# ============================================================
# SPLIT EACH PARTICIPANT
# ============================================================

train_parts = {}
test_parts = {}

for participant in sorted(
    df["participant"].unique(),
    key=lambda x: int(x.replace("p", ""))
):

    client_df = df[
        df["participant"] == participant
    ].copy()

    X = client_df[FEATURE_COLUMNS].values
    y = client_df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    train_parts[participant] = {
        "X": X_train,
        "y": y_train
    }

    test_parts[participant] = {
        "X": X_test,
        "y": y_test
    }


# ============================================================
# FIT SCALER ONLY ON TRAINING DATA
# ============================================================

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

print("\nScaler fitted ONLY on training samples.")
print("Saved:", SCALER_PATH)


# ============================================================
# SCALE CLIENT DATA
# ============================================================

for participant in train_parts:

    train_parts[participant]["X"] = scaler.transform(
        train_parts[participant]["X"]
    )

    test_parts[participant]["X"] = scaler.transform(
        test_parts[participant]["X"]
    )


# ============================================================
# CLIENT INFORMATION
# ============================================================

print("\n" + "=" * 65)
print("FEDERATED CLIENT SPLITS")
print("=" * 65)

total_train = 0
total_test = 0

for participant in train_parts:

    n_train = len(
        train_parts[participant]["y"]
    )

    n_test = len(
        test_parts[participant]["y"]
    )

    total_train += n_train
    total_test += n_test

    print(
        f"{participant}: "
        f"train={n_train}, test={n_test}"
    )

print("\nTotal training samples:", total_train)
print("Total testing samples:", total_test)


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