import os, math, joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Reshape, Conv1D, Dropout, Bidirectional, LSTM, Dense
from tensorflow.keras.regularizers import l2

# Participant-level central DP-FedAvg.
# Assumptions: fixed 23-client cohort; replace-one participant adjacency;
# fixed public client weights; trusted aggregator; no secure aggregation.
# The final released model is protected. The server still sees individual updates.

CLIENTS_PATH = "models/fl_clients.pkl"
VALIDATION_PATH = "models/fl_validation_data.pkl"
TEST_PATH = "models/fl_test_data.pkl"
SCALER_PATH = "models/fl_scaler.pkl"
MODEL_PATH = "models/dp_federated_model.keras"
WEIGHTS_PATH = "models/dp_federated_weights.weights.h5"
RESULTS_PATH = "models/dp_federated_results.pkl"

RANDOM_STATE = 42
NUM_ROUNDS = 5
LOCAL_EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 0.001
CLIPPING_NORM = 0.5
NOISE_MULTIPLIER = 2.0
DELTA = 1e-5
ORDERS = np.concatenate([np.arange(1.1, 10.0, 0.1), np.arange(10, 64), np.arange(64, 129, 2)]).astype(float)

FEATURE_COLUMNS = [
    "MEAN_RR", "MEDIAN_RR", "SDRR", "RMSSD", "SDSD", "SDRR_RMSSD",
    "HR", "pNN25", "pNN50", "SD1", "SD2", "KURT", "SKEW",
    "MEAN_REL_RR", "MEDIAN_REL_RR", "SDRR_REL_RR", "RMSSD_REL_RR",
    "SDSD_REL_RR", "SDRR_RMSSD_REL_RR", "KURT_REL_RR", "SKEW_REL_RR",
    "VLF", "VLF_PCT", "LF", "LF_PCT", "LF_NU", "HF", "HF_PCT",
    "HF_NU", "TP", "LF_HF", "HF_LF", "sampen", "higuci"
]

np.random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)
os.makedirs("models", exist_ok=True)

def create_model():
    model = Sequential([
        Input(shape=(len(FEATURE_COLUMNS),)),
        Reshape((len(FEATURE_COLUMNS), 1)),
        Conv1D(16, 2, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Bidirectional(LSTM(16, dropout=0.1, recurrent_dropout=0.1)),
        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model

def get_xy(obj, name):
    if not isinstance(obj, dict) or "X" not in obj or "y" not in obj:
        raise ValueError(f"{name} must contain dictionary keys 'X' and 'y'.")
    X = np.asarray(obj["X"], dtype=np.float32)
    y = np.asarray(obj["y"], dtype=int)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f"{name}: invalid feature shape {X.shape}")
    if len(X) != len(y) or len(y) == 0 or not np.isfinite(X).all():
        raise ValueError(f"{name}: invalid or non-finite X/y.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{name}: labels must be 0/1.")
    return X, y

def clip_update(local_weights, global_weights, bound):
    updates = [np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
               for a, b in zip(local_weights, global_weights)]
    norm = math.sqrt(sum(float(np.sum(u ** 2)) for u in updates))
    scale = min(1.0, bound / (norm + 1e-12))
    return [u * scale for u in updates], norm

def privacy_epsilon(z, rounds, delta):
    # Basic RDP composition for Gaussian mechanisms; conservative conversion.
    eps = rounds * ORDERS / (2.0 * z * z) + math.log(1.0 / delta) / (ORDERS - 1.0)
    i = int(np.argmin(eps))
    return float(eps[i]), float(ORDERS[i])

print("=" * 65)
print("LOADING PREPARED DATA FOR PARTICIPANT-LEVEL DP-FEDAVG")
print("=" * 65)
for p in [CLIENTS_PATH, VALIDATION_PATH, TEST_PATH, SCALER_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required prepared file not found: {p}")

clients = joblib.load(CLIENTS_PATH)
val_obj = joblib.load(VALIDATION_PATH)
test_obj = joblib.load(TEST_PATH)
scaler = joblib.load(SCALER_PATH)

if not isinstance(clients, dict):
    raise TypeError("fl_clients.pkl must contain a dictionary of clients.")
participants = sorted(clients, key=lambda x: int(str(x).replace("p", "")))
if len(participants) != 23:
    raise ValueError(f"Expected 23 clients; found {len(participants)}.")
if scaler.n_features_in_ != len(FEATURE_COLUMNS):
    raise ValueError("Scaler feature count does not match 34 HRV features.")

for pid in participants:
    clients[pid]["X"], clients[pid]["y"] = get_xy(clients[pid], f"Client {pid}")
X_val, y_val = get_xy(val_obj, "Validation data")
X_test, y_test = get_xy(test_obj, "Test data")

n_train = sum(len(clients[p]["y"]) for p in participants)
if (n_train, len(X_val), len(X_test)) != (1111, 290, 359):
    raise ValueError(f"Expected split 1111/290/359; got {n_train}/{len(X_val)}/{len(X_test)}.")

sizes = np.array([len(clients[p]["y"]) for p in participants], dtype=float)
weights = sizes / sizes.sum()

# For fixed weights and replace-one client adjacency, changing one clipped
# client update has weighted-mean sensitivity <= 2*C*max(weight_i).
sensitivity = 2.0 * CLIPPING_NORM * float(weights.max())
noise_std = NOISE_MULTIPLIER * sensitivity
epsilon, best_order = privacy_epsilon(NOISE_MULTIPLIER, NUM_ROUNDS, DELTA)

print("\nDATA VALIDATION SUCCESSFUL")
print("Clients:", len(participants), "| Train:", n_train,
      "| Validation:", len(X_val), "| Test:", len(X_test))
print("\nPRIVACY SETUP")
print("Unit: one complete participant/client")
print("Adjacency: replace-one participant, fixed 23-client cohort")
print("Rounds:", NUM_ROUNDS, "| Full participation: 23/23 each round")
print("Clipping norm:", CLIPPING_NORM, "| Noise multiplier:", NOISE_MULTIPLIER)
print("Aggregate sensitivity bound:", sensitivity)
print("Gaussian noise std per aggregate coordinate:", noise_std)
print("Delta:", DELTA)
print(f"Composed epsilon (RDP upper bound): {epsilon:.4f} at order {best_order:.2f}")
print("Assumption: trusted aggregator; individual client updates are visible to server.")
print("No secure aggregation is implemented.")

model = create_model()
global_weights = [np.asarray(w, dtype=np.float64) for w in model.get_weights()]
round_logs = []

print("\nSTARTING PARTICIPANT-LEVEL DP-FEDAVG")
print("Validation is diagnostic only; no round selection. Test is evaluated once.")

for r in range(NUM_ROUNDS):
    print(f"\nROUND {r + 1}/{NUM_ROUNDS}")
    aggregate = [np.zeros_like(w, dtype=np.float64) for w in global_weights]
    norms = []

    for pid, weight in zip(participants, weights):
        local = create_model()
        local.set_weights([w.astype(np.float32) for w in global_weights])
        local.fit(clients[pid]["X"], clients[pid]["y"],
                  epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE,
                  verbose=0, shuffle=True)
        clipped, original_norm = clip_update(local.get_weights(), global_weights, CLIPPING_NORM)
        norms.append(original_norm)
        for j, u in enumerate(clipped):
            aggregate[j] += weight * u
        print(f"{pid}: {len(clients[pid]['y'])} records | update norm={original_norm:.4f}")
        del local

    # One Gaussian vector is added to the weighted aggregate each round.
    noisy_update = [
        layer + np.random.normal(0.0, noise_std, size=layer.shape)
        for layer in aggregate
    ]
    global_weights = [w + u for w, u in zip(global_weights, noisy_update)]
    model.set_weights([w.astype(np.float32) for w in global_weights])

    vp = model.predict(X_val, verbose=0).ravel()
    vy = (vp >= 0.5).astype(int)
    log = {
        "round": r + 1,
        "validation_accuracy": float(accuracy_score(y_val, vy)),
        "validation_precision": float(precision_score(y_val, vy, zero_division=0)),
        "validation_recall": float(recall_score(y_val, vy, zero_division=0)),
        "validation_f1": float(f1_score(y_val, vy, zero_division=0)),
        "validation_mcc": float(matthews_corrcoef(y_val, vy)),
        "mean_unclipped_client_update_norm": float(np.mean(norms))
    }
    round_logs.append(log)
    print(f"Validation accuracy: {log['validation_accuracy']*100:.2f}%")
    print(f"Validation precision: {log['validation_precision']*100:.2f}%")
    print(f"Validation recall: {log['validation_recall']*100:.2f}%")
    print(f"Validation F1: {log['validation_f1']*100:.2f}%")
    print(f"Validation MCC: {log['validation_mcc']:.4f}")

print("\nFINAL TEST EVALUATION (ONCE)")
prob = model.predict(X_test, verbose=0).ravel()
pred = (prob >= 0.5).astype(int)
acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred, zero_division=0)
rec = recall_score(y_test, pred, zero_division=0)
f1 = f1_score(y_test, pred, zero_division=0)
mcc = matthews_corrcoef(y_test, pred)
cm = confusion_matrix(y_test, pred, labels=[0, 1])

print(f"Accuracy: {acc*100:.2f}%")
print(f"Stress precision: {prec*100:.2f}%")
print(f"Stress recall: {rec*100:.2f}%")
print(f"Stress F1: {f1*100:.2f}%")
print(f"MCC: {mcc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, pred, labels=[0, 1],
                            target_names=["No Stress", "Stress"], zero_division=0))
print("Confusion matrix:\n", cm)

model.save(MODEL_PATH)
model.save_weights(WEIGHTS_PATH)
results = {
    "privacy_unit": "participant/client",
    "adjacency": "replace-one participant in fixed 23-client cohort",
    "trusted_aggregator_assumption": True,
    "secure_aggregation": False,
    "clients": len(participants), "train": n_train,
    "validation": len(X_val), "test": len(X_test),
    "rounds": NUM_ROUNDS, "local_epochs": LOCAL_EPOCHS,
    "clipping_norm": CLIPPING_NORM, "noise_multiplier": NOISE_MULTIPLIER,
    "delta": DELTA, "epsilon_rdp_upper_bound": epsilon,
    "best_rdp_order": best_order, "aggregate_sensitivity": sensitivity,
    "aggregate_noise_std": noise_std,
    "test_accuracy": float(acc), "test_precision_stress": float(prec),
    "test_recall_stress": float(rec), "test_f1_stress": float(f1),
    "test_mcc": float(mcc), "confusion_matrix": cm.tolist(),
    "round_logs": round_logs
}
joblib.dump(results, RESULTS_PATH)
print("\nSaved:", MODEL_PATH, WEIGHTS_PATH, RESULTS_PATH)
print("PARTICIPANT-LEVEL DP-FEDAVG COMPLETE")
