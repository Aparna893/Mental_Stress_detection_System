
import os
import gc
import numpy as np
import tensorflow as tf
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
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
VALIDATION_PATH = "models/fl_validation_data.pkl"
TEST_DATA_PATH = "models/fl_test_data.pkl"
SCALER_PATH = "models/fl_scaler.pkl"

MODEL_PATH = "models/federated_model.keras"
WEIGHTS_PATH = "models/federated_weights.weights.h5"

RANDOM_STATE = 42

NUM_ROUNDS = 5
LOCAL_EPOCHS = 2
BATCH_SIZE = 16

LEARNING_RATE = 0.001

np.random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)

os.makedirs("models", exist_ok=True)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "MEAN_RR", "MEDIAN_RR", "SDRR", "RMSSD",
    "SDSD", "SDRR_RMSSD", "HR", "pNN25",
    "pNN50", "SD1", "SD2", "KURT",
    "SKEW", "MEAN_REL_RR", "MEDIAN_REL_RR",
    "SDRR_REL_RR", "RMSSD_REL_RR", "SDSD_REL_RR",
    "SDRR_RMSSD_REL_RR", "KURT_REL_RR",
    "SKEW_REL_RR", "VLF", "VLF_PCT", "LF",
    "LF_PCT", "LF_NU", "HF", "HF_PCT",
    "HF_NU", "TP", "LF_HF", "HF_LF",
    "sampen", "higuci"
]


# ============================================================
# CNN-BILSTM MODEL
# ============================================================

def create_model():

    model = Sequential([
        Input(shape=(len(FEATURE_COLUMNS),)),

        Reshape((len(FEATURE_COLUMNS), 1)),

        Conv1D(
            filters=16,
            kernel_size=2,
            activation="relu",
            kernel_regularizer=l2(0.001)
        ),

        Dropout(0.2),

        Bidirectional(
            LSTM(
                16,
                dropout=0.1,
                recurrent_dropout=0.1
            )
        ),

        Dense(
            32,
            activation="relu",
            kernel_regularizer=l2(0.001)
        ),

        Dropout(0.2),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# LOAD PREPARED DATA
# ============================================================

print("=" * 65)
print("LOADING PREPARED FEDERATED DATA")
print("=" * 65)

for path in [
    CLIENTS_PATH,
    VALIDATION_PATH,
    TEST_DATA_PATH,
    SCALER_PATH
]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

clients = joblib.load(CLIENTS_PATH)
validation_data = joblib.load(VALIDATION_PATH)
test_data = joblib.load(TEST_DATA_PATH)
scaler = joblib.load(SCALER_PATH)

if scaler.n_features_in_ != len(FEATURE_COLUMNS):
    raise ValueError("Scaler feature count does not match model features.")

train_parts = clients

X_val = np.asarray(
    validation_data["X"], dtype=np.float32
)
y_val = np.asarray(
    validation_data["y"], dtype=int
)

X_test = np.asarray(
    test_data["X"], dtype=np.float32
)
y_test = np.asarray(
    test_data["y"], dtype=int
)

val_participant_ids = np.asarray(
    validation_data["participant"]
).astype(str)

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

for name, X, y, ids in [
    ("Validation", X_val, y_val, val_participant_ids),
    ("Testing", X_test, y_test, test_participant_ids)
]:
    if X.ndim != 2 or X.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f"Invalid {name} feature shape: {X.shape}")

    if not (len(X) == len(y) == len(ids)):
        raise ValueError(f"{name} feature, label and ID counts differ.")

    if not np.isfinite(X).all():
        raise ValueError(f"{name} data contains non-finite features.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{name} labels must be 0 or 1.")

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"{name} data must contain both classes.")

total_train = 0

for participant in participants:

    X = np.asarray(
        train_parts[participant]["X"],
        dtype=np.float32
    )

    y = np.asarray(
        train_parts[participant]["y"],
        dtype=int
    )

    if X.ndim != 2 or X.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Invalid training feature shape for {participant}"
        )

    if len(X) != len(y) or len(y) == 0:
        raise ValueError(
            f"Invalid training data for {participant}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"Non-finite features for {participant}"
        )

    if not np.isin(y, [0, 1]).all():
        raise ValueError(
            f"Invalid labels for {participant}"
        )

    train_parts[participant]["X"] = X
    train_parts[participant]["y"] = y

    if not np.any(val_participant_ids == str(participant)):
        raise ValueError(
            f"No validation records found for {participant}"
        )

    if not np.any(test_participant_ids == str(participant)):
        raise ValueError(
            f"No test records found for {participant}"
        )

    total_train += len(y)

if set(val_participant_ids) != set(map(str, participants)):
    raise ValueError("Validation and training participant IDs do not match.")

if set(test_participant_ids) != set(map(str, participants)):
    raise ValueError("Test and training participant IDs do not match.")

if total_train != 1111 or len(y_val) != 290 or len(y_test) != 359:
    raise ValueError(
        f"Unexpected data sizes: train={total_train}, "
        f"validation={len(y_val)}, test={len(y_test)}"
    )

print("\nDATA VALIDATION SUCCESSFUL")
print("Clients:", len(participants))
print("Training samples:", total_train)
print("Validation samples:", len(y_val))
print("Testing samples:", len(y_test))
print("HRV features:", len(FEATURE_COLUMNS))


# ============================================================
# CLIENT INFORMATION
# ============================================================

print("\n" + "=" * 65)
print("FEDERATED CLIENT INFORMATION")
print("=" * 65)

for participant in participants:

    train_count = len(train_parts[participant]["y"])

    val_count = np.sum(
        val_participant_ids == str(participant)
    )

    test_count = np.sum(
        test_participant_ids == str(participant)
    )

    print(
        f"{participant}: "
        f"train={train_count}, "
        f"validation={val_count}, "
        f"test={test_count}"
    )


# ============================================================
# INITIAL GLOBAL MODEL
# ============================================================

global_model = create_model()
global_weights = global_model.get_weights()

best_val_mcc = -float("inf")
best_val_loss = float("inf")
best_round = 0
best_weights = None

print("\n" + "=" * 65)
print("STARTING VALIDATION-AWARE FEDERATED LEARNING")
print("=" * 65)

print("Communication rounds:", NUM_ROUNDS)
print("Local epochs:", LOCAL_EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Validation selection metric: MCC")
print("Tie-break metric: validation loss")


# ============================================================
# FEDERATED TRAINING
# ============================================================

for round_num in range(NUM_ROUNDS):

    print(f"\nROUND {round_num + 1}/{NUM_ROUNDS}")

    client_weights = []
    client_sizes = []

    for participant in participants:

        client_data = train_parts[participant]

        X_client = client_data["X"]
        y_client = client_data["y"]

        local_model = create_model()
        local_model.set_weights(global_weights)

        local_model.fit(
            X_client,
            y_client,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            shuffle=True
        )

        client_weights.append(
            local_model.get_weights()
        )

        client_sizes.append(len(y_client))

        print(
            f"{participant}: "
            f"{len(y_client)} samples trained"
        )

        del local_model

    # --------------------------------------------------------
    # WEIGHTED FEDAVG
    # --------------------------------------------------------

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

    del client_weights
    gc.collect()

    print(f"Round {round_num + 1} aggregation completed.")

    # --------------------------------------------------------
    # VALIDATION AFTER EACH ROUND
    # --------------------------------------------------------

    val_loss, val_accuracy = global_model.evaluate(
        X_val,
        y_val,
        verbose=0
    )

    val_probabilities = global_model.predict(
        X_val,
        verbose=0
    ).ravel()

    val_predictions = (
        val_probabilities >= 0.5
    ).astype(int)

    val_precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    val_recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    val_f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    val_mcc = matthews_corrcoef(
        y_val,
        val_predictions
    )

    print("\nVALIDATION RESULTS")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Validation Precision: {val_precision * 100:.2f}%")
    print(f"Validation Recall: {val_recall * 100:.2f}%")
    print(f"Validation F1: {val_f1 * 100:.2f}%")
    print(f"Validation MCC: {val_mcc:.4f}")

    # --------------------------------------------------------
    # SELECT BEST GLOBAL MODEL
    # --------------------------------------------------------

    is_better = (
        val_mcc > best_val_mcc
        or (
            np.isclose(val_mcc, best_val_mcc)
            and val_loss < best_val_loss
        )
    )

    if is_better:

        best_val_mcc = val_mcc
        best_val_loss = val_loss
        best_round = round_num + 1

        best_weights = [
            weight.copy()
            for weight in global_weights
        ]

        print(
            f"New best model selected from round {best_round}."
        )

    else:
        print("Best model remains from an earlier round.")


# ============================================================
# RESTORE BEST VALIDATION MODEL
# ============================================================

if best_weights is None:
    raise RuntimeError("No best model weights were selected.")

global_model.set_weights(best_weights)

print("\n" + "=" * 65)
print("BEST VALIDATION MODEL SELECTED")
print("=" * 65)

print("Best communication round:", best_round)
print(f"Best validation MCC: {best_val_mcc:.4f}")
print(f"Best validation loss: {best_val_loss:.4f}")


# ============================================================
# FINAL TEST EVALUATION - ONCE ONLY
# ============================================================

print("\n" + "=" * 65)
print("FINAL TEST EVALUATION")
print("=" * 65)

probabilities = global_model.predict(
    X_test,
    verbose=0
).ravel()

predictions = (
    probabilities >= 0.5
).astype(int)

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

mcc = matthews_corrcoef(y_test, predictions)

print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")
print(f"Stress Precision: {precision * 100:.2f}%")
print(f"Stress Recall: {recall * 100:.2f}%")
print(f"Stress F1-score: {f1 * 100:.2f}%")
print(f"MCC: {mcc:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No Stress", "Stress"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )
)

print("\nPrediction Distribution:")
print("No Stress predictions:", np.sum(predictions == 0))
print("Stress predictions:", np.sum(predictions == 1))

print("\nProbability Distribution:")
print("Minimum:", np.min(probabilities))
print("Maximum:", np.max(probabilities))
print("Mean:", np.mean(probabilities))


# ============================================================
# SAVE SELECTED GLOBAL MODEL
# ============================================================

global_model.save(MODEL_PATH)
global_model.save_weights(WEIGHTS_PATH)

print("\nBest federated model saved:")
print(MODEL_PATH)

print("\nBest federated weights saved:")
print(WEIGHTS_PATH)

print("\nVALIDATION-AWARE FEDERATED LEARNING COMPLETE")
