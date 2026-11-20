import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CLIENT_PATH = ROOT / "models" / "fl_clients.pkl"
RESULT_ROOT = ROOT / "secure_aggregation" / "results"

RANDOM_STATE = 42
MASK_SCALE = 1.0

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = RESULT_ROOT / f"secure_agg_run_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

rng = np.random.default_rng(RANDOM_STATE)


# --------------------------------------------------
# LOAD EXISTING CLIENT DATA (READ-ONLY)
# --------------------------------------------------

print("Loading existing federated clients...")
clients = joblib.load(CLIENT_PATH)

client_names = list(clients.keys())
num_clients = len(client_names)

if num_clients < 2:
    raise ValueError("At least two clients are required.")

print("Number of clients:", num_clients)


# --------------------------------------------------
# SIMULATE CLIENT MODEL UPDATES
# --------------------------------------------------

# Each client update is represented by a 34-dimensional
# vector. This is a controlled simulation, not a real
# neural-network gradient or weight update.

updates = {}
sample_counts = {}

for client_name, client_data in clients.items():
    X = np.asarray(client_data["X"], dtype=np.float64)
    y = np.asarray(client_data["y"], dtype=np.float64)

    if X.ndim != 2 or X.shape[1] != 34:
        raise ValueError(f"Unexpected feature shape for {client_name}")

    if len(X) != len(y):
        raise ValueError(f"Feature/label mismatch for {client_name}")

    # Simulated local update derived from each client's
    # local feature and label statistics.
    feature_mean = np.mean(X, axis=0)
    label_rate = np.mean(y)

    update = (
        feature_mean * 0.01
        + label_rate * 0.01
        + rng.normal(0, 0.001, size=34)
    )

    updates[client_name] = update
    sample_counts[client_name] = len(X)


# --------------------------------------------------
# ORDINARY FEDAVG AGGREGATION
# --------------------------------------------------

total_samples = sum(sample_counts.values())

fedavg = np.zeros(34, dtype=np.float64)

for client_name in client_names:
    weight = sample_counts[client_name] / total_samples
    fedavg += weight * updates[client_name]


# --------------------------------------------------
# PAIRWISE MASK GENERATION
# --------------------------------------------------

# For every pair (i, j), client i adds a random mask
# and client j subtracts the same mask.
#
# Therefore, all pairwise masks cancel during summation.

masked_updates = {
    name: updates[name].copy()
    for name in client_names
}

pair_count = 0

for i in range(num_clients):
    for j in range(i + 1, num_clients):
        client_i = client_names[i]
        client_j = client_names[j]

        pair_mask = rng.normal(
            loc=0.0,
            scale=MASK_SCALE,
            size=34
        )

        masked_updates[client_i] += pair_mask
        masked_updates[client_j] -= pair_mask

        pair_count += 1


# --------------------------------------------------
# SERVER AGGREGATION
# --------------------------------------------------

# The server receives masked updates and aggregates them.
# The masks cancel because every pair has opposite signs.

secure_sum = np.zeros(34, dtype=np.float64)

for client_name in client_names:
    secure_sum += masked_updates[client_name]

secure_aggregate = secure_sum / total_samples


# --------------------------------------------------
# VERIFICATION
# --------------------------------------------------

# In this simulation, the masks are added to raw updates.
# The weighted FedAvg equivalent is obtained by applying
# sample weights to the unmasked client updates.
#
# For a valid weighted secure aggregation, clients must
# mask their weighted contributions.

weighted_masked_updates = {}

for client_name in client_names:
    weight = sample_counts[client_name] / total_samples
    weighted_masked_updates[client_name] = (
        weight * updates[client_name]
    )

# Apply pairwise masks to weighted contributions.
for i in range(num_clients):
    for j in range(i + 1, num_clients):
        client_i = client_names[i]
        client_j = client_names[j]

        pair_mask = rng.normal(
            loc=0.0,
            scale=MASK_SCALE,
            size=34
        )

        weighted_masked_updates[client_i] += pair_mask
        weighted_masked_updates[client_j] -= pair_mask

secure_fedavg = np.sum(
    np.array(list(weighted_masked_updates.values())),
    axis=0
)

absolute_error = np.abs(secure_fedavg - fedavg)
max_error = float(np.max(absolute_error))
mean_error = float(np.mean(absolute_error))

print("\nSECURE AGGREGATION VERIFICATION")
print("Pairwise masks generated:", pair_count)
print("Maximum absolute error:", max_error)
print("Mean absolute error:", mean_error)

if not np.allclose(secure_fedavg, fedavg, atol=1e-10):
    raise RuntimeError("Secure aggregation verification failed.")

print("Verification successful: secure FedAvg matches ordinary FedAvg.")


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

pd.DataFrame({
    "feature_index": np.arange(1, 35),
    "ordinary_fedavg": fedavg,
    "secure_fedavg": secure_fedavg,
    "absolute_difference": absolute_error
}).to_csv(
    OUTPUT_DIR / "aggregation_comparison.csv",
    index=False
)

pd.DataFrame({
    "participant": client_names,
    "sample_count": [
        sample_counts[name] for name in client_names
    ]
}).to_csv(
    OUTPUT_DIR / "client_sample_counts.csv",
    index=False
)


# Save the simulated local updates separately.
update_rows = []

for name in client_names:
    row = {"participant": name}
    for i, value in enumerate(updates[name]):
        row[f"update_{i + 1}"] = float(value)
    update_rows.append(row)

pd.DataFrame(update_rows).to_csv(
    OUTPUT_DIR / "simulated_client_updates.csv",
    index=False
)

metrics = {
    "experiment": "Pairwise Masked Secure Aggregation Simulation",
    "number_of_clients": int(num_clients),
    "total_client_samples": int(total_samples),
    "number_of_pairwise_masks": int(pair_count),
    "update_dimension": 34,
    "mask_scale": MASK_SCALE,
    "maximum_absolute_difference": max_error,
    "mean_absolute_difference": mean_error,
    "verification_passed": True,
    "aggregation_method": "Sample-count weighted FedAvg",
    "security_scope": (
        "Educational simulation of additive pairwise masking. "
        "Not a production cryptographic secure aggregation protocol."
    )
}

with (OUTPUT_DIR / "metrics.json").open("w") as f:
    json.dump(metrics, f, indent=4)

# Comparison plot
plt.figure(figsize=(9, 5))
plt.plot(fedavg, label="Ordinary FedAvg", marker="o", markersize=3)
plt.plot(secure_fedavg, label="Secure FedAvg", linestyle="--")
plt.xlabel("Feature index")
plt.ylabel("Aggregated update value")
plt.title("Ordinary FedAvg vs Secure Aggregation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "aggregation_comparison.png",
    dpi=300
)
plt.close()

print("\nSECURE AGGREGATION SIMULATION COMPLETED")
print("Results saved in:", OUTPUT_DIR)