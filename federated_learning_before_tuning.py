
import os
import numpy as np
import tensorflow as tf
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

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

os.makedirs("models", exist_ok=True)


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


# ============================================================
# LOAD PREPARED FEDERATED DATA
# ============================================================

print("=" * 65)
print("LOADING PREPARED FEDERATED DATA")
print("=" * 65)

for path in [CLIENTS_PATH, TEST_DATA_PATH, SCALER_PATH]:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required prepared file not found: {path}"
        )

clients = joblib.load(CLIENTS_PATH)
test_data = joblib.load(TEST_DATA_PATH)
scaler = joblib.load(SCALER_PATH)

# Check the saved scaler
if scaler.n_features_in_ != len(FEATURE_COLUMNS):
    raise ValueError("Scaler feature count does not match model features.")

# Training clients
train_parts = clients

# Separate held-out test data
X_test_global = np.asarray(test_data["X"], dtype=np.float32)
y_test_global = np.asarray(test_data["y"], dtype=int)
test_participant_ids = np.asarray(
    test_data["participant"]
).astype(str)

participants = sorted(
    train_parts.keys(),
    key=lambda x: int(str(x).replace("p", ""))
)

# ============================================================
# DATA VALIDATION
# ============================================================

if len(participants) != 23:
    raise ValueError(
        f"Expected 23 clients, found {len(participants)}"
    )

if X_test_global.ndim != 2:
    raise ValueError("Test features must be a 2D array.")

if X_test_global.shape[1] != len(FEATURE_COLUMNS):
    raise ValueError("Unexpected number of test features.")

if not (
    len(X_test_global)
    == len(y_test_global)
    == len(test_participant_ids)
):
    raise ValueError("Test feature, label and participant counts differ.")

if not np.isfinite(X_test_global).all():
    raise ValueError("Test data contains NaN or infinite values.")

if not np.isin(y_test_global, [0, 1]).all():
    raise ValueError("Test labels must contain only 0 and 1.")

for participant in participants:

    X = np.asarray(
        train_parts[participant]["X"],
        dtype=np.float32
    )

    y = np.asarray(
        train_parts[participant]["y"],
        dtype=int
    )

    train_parts[participant]["X"] = X
    train_parts[participant]["y"] = y

    if X.ndim != 2 or X.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Invalid feature shape for {participant}: {X.shape}"
        )

    if len(X) != len(y) or len(y) == 0:
        raise ValueError(
            f"Invalid training data for {participant}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"Non-finite training features for {participant}"
        )

    if not np.isin(y, [0, 1]).all():
        raise ValueError(
            f"Invalid labels for {participant}"
        )

    if not np.any(test_participant_ids == str(participant)):
        raise ValueError(
            f"No test records found for {participant}"
        )

# Confirm participant IDs in the test data
if set(test_participant_ids) != set(map(str, participants)):
    raise ValueError(
        "Training and testing participant IDs do not match."
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

print("\nDATA VALIDATION SUCCESSFUL")

print("Number of clients:", len(participants))
print("Training samples:", total_train)
print("Testing samples:", total_test)
print("HRV features:", len(FEATURE_COLUMNS))


# ============================================================
# CLIENT INFORMATION
# ============================================================

print("\n" + "=" * 65)
print("FEDERATED CLIENT SPLITS")
print("=" * 65)

for participant in participants:

    client_train = train_parts[participant]["y"]

    client_test_count = np.sum(
        test_participant_ids == str(participant)
    )

    print(
        f"{participant}: "
        f"train={len(client_train)}, "
        f"test={client_test_count}"
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

print("Communication rounds:", NUM_ROUNDS)
print("Local epochs:", LOCAL_EPOCHS)
print("Batch size:", BATCH_SIZE)


# ============================================================
# FEDERATED TRAINING
# ============================================================

for round_num in range(NUM_ROUNDS):

    print(f"\nROUND {round_num + 1}/{NUM_ROUNDS}")

    client_weights = []
    client_sizes = []

    for participant, client_data in train_parts.items():

        X_client = client_data["X"]
        y_client = client_data["y"]

        # Create a fresh local model
        local_model = create_model()

        # Receive the current global weights
        local_model.set_weights(global_weights)

        # Train using only this participant's data
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
    # WEIGHTED FEDAVG AGGREGATION
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
                weights * (sample_count / total_samples)
            )

        new_global_weights.append(weighted_layer)

    global_weights = new_global_weights

    global_model.set_weights(global_weights)

    # ========================================================
    # ROUND EVALUATION
    # ========================================================

    loss, accuracy = global_model.evaluate(
        X_test_global,
        y_test_global,
        verbose=0
    )

    print(
        f"Round {round_num + 1} "
        f"Test Accuracy: {accuracy * 100:.2f}%"
    )


# ============================================================
# FINAL GLOBAL MODEL EVALUATION
# ============================================================

print("\n" + "=" * 65)
print("FINAL GLOBAL MODEL EVALUATION")
print("=" * 65)

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

precision = precision_score(
    y_test_global,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test_global,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test_global,
    predictions,
    zero_division=0
)

print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")
print(f"Stress Precision: {precision * 100:.2f}%")
print(f"Stress Recall: {recall * 100:.2f}%")
print(f"Stress F1-score: {f1 * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test_global,
        predictions,
        labels=[0, 1],
        target_names=["No Stress", "Stress"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test_global,
        predictions,
        labels=[0, 1]
    )
)


# ============================================================
# SAVE GLOBAL MODEL
# ============================================================

global_model.save(MODEL_PATH)

global_model.save_weights(WEIGHTS_PATH)

print("\nFederated model saved:")
print(MODEL_PATH)

print("\nFederated weights saved:")
print(WEIGHTS_PATH)

print("\nFEDERATED LEARNING COMPLETE")
