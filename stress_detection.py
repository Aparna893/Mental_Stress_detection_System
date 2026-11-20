import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================================
# 1. LOAD DATA
# ============================================================

# ============================================================
# 1. LOAD COMMON COMPARISON DATASET
# ============================================================

print("Loading common comparison datasets...")

train_df = pd.read_csv("data/comparison_train.csv")
test_df = pd.read_csv("data/comparison_test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


# ============================================================
# 2. DEFINE 34 HRV FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "MEAN_RR", "MEDIAN_RR", "SDRR", "RMSSD", "SDSD",
    "SDRR_RMSSD", "HR", "pNN25", "pNN50", "SD1",
    "SD2", "KURT", "SKEW", "MEAN_REL_RR",
    "MEDIAN_REL_RR", "SDRR_REL_RR", "RMSSD_REL_RR",
    "SDSD_REL_RR", "SDRR_RMSSD_REL_RR", "KURT_REL_RR",
    "SKEW_REL_RR", "VLF", "VLF_PCT", "LF", "LF_PCT",
    "LF_NU", "HF", "HF_PCT", "HF_NU", "TP",
    "LF_HF", "HF_LF", "sampen", "higuci"
]

# Verify that all expected features exist
missing_features = [
    col for col in FEATURE_COLUMNS
    if col not in train_df.columns or col not in test_df.columns
]

if missing_features:
    raise ValueError(f"Missing HRV features: {missing_features}")

if len(FEATURE_COLUMNS) != 34:
    raise ValueError("Expected exactly 34 HRV features.")


# ============================================================
# 3. VERIFY LABELS
# ============================================================

if "label" not in train_df.columns or "label" not in test_df.columns:
    raise ValueError("The comparison datasets must contain a label column.")

X_train = train_df[FEATURE_COLUMNS].copy()
y_train = train_df["label"].astype(int).to_numpy()

X_test = test_df[FEATURE_COLUMNS].copy()
y_test = test_df["label"].astype(int).to_numpy()

if set(np.unique(y_train)) != {0, 1}:
    raise ValueError("Training data must contain labels 0 and 1.")

if not set(np.unique(y_test)).issubset({0, 1}):
    raise ValueError("Testing data contains invalid labels.")


# ============================================================
# 4. STANDARD SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

print("\nScaler fitted only on training data.")


# ============================================================
# 5. CONVERT DATA
# ============================================================

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

print("\nFinal centralized data:")
print("HRV features:", len(FEATURE_COLUMNS))
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nTraining label distribution:")
print(pd.Series(y_train).value_counts().sort_index())

print("\nTesting label distribution:")
print(pd.Series(y_test).value_counts().sort_index())

print("\nCentralized preprocessing completed!")

# ============================================================
# 10. DISPLAY FINAL DATA
# ============================================================

print("\nFinal data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# 11. DISPLAY CLASS DISTRIBUTION
# ============================================================

print("\nTraining class distribution:")

unique, counts = np.unique(y_train, return_counts=True)

for label, count in zip(unique, counts):
    if label == 0:
        print("No Stress (0):", count)
    else:
        print("Stress (1):", count)


print("\nTest class distribution:")

unique, counts = np.unique(y_test, return_counts=True)

for label, count in zip(unique, counts):
    if label == 0:
        print("No Stress (0):", count)
    else:
        print("Stress (1):", count)


print("\nPreprocessing completed successfully!")
# ============================================================
# 12. CNN-BiLSTM MODEL
# ============================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv1D,
    LSTM,
    Bidirectional,
    Dropout,
    Reshape
)
from tensorflow.keras.regularizers import l2


def create_model():
    model = Sequential([
        # Input layer
        Input(shape=(X_train.shape[1],)),

        # Convert 34 HRV features into CNN sequence format
        Reshape((X_train.shape[1], 1)),

        # CNN feature extraction
        Conv1D(
            filters=16,
            kernel_size=2,
            activation="relu",
            kernel_regularizer=l2(0.01)
        ),

        Dropout(0.5),

        # BiLSTM
        Bidirectional(
            LSTM(
                16,
                dropout=0.5,
                recurrent_dropout=0.5
            )
        ),

        # Fully connected layer
        Dense(
            32,
            activation="relu",
            kernel_regularizer=l2(0.01)
        ),

        Dropout(0.5),

        # Binary classification
        Dense(1, activation="sigmoid")
    ])

    return model


# ============================================================
# 13. CREATE MODEL
# ============================================================

model = create_model()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 14. DISPLAY MODEL
# ============================================================

print("\nCNN-BiLSTM Model:")
model.summary()


# ============================================================
# 15. CENTRALIZED TRAINING
# ============================================================

print("\nStarting centralized training...")

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# ============================================================
# 16. EVALUATE CENTRALIZED MODEL
# ============================================================

print("\nEvaluating centralized model...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nCentralized Test Loss:", test_loss)
print("Centralized Test Accuracy:", test_accuracy)


# ============================================================
# 17. SAVE CENTRALIZED MODEL
# ============================================================

model.save("models/centralized_model.keras")

print("\nCentralized model saved successfully!")
print("Location: models/centralized_model.keras")