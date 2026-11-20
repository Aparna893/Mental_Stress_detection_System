import json
import joblib
import numpy as np
import tensorflow as tf

from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "federated_model.keras"
CLIENT_DATA_PATH = ROOT / "models" / "fl_clients.pkl"
TEST_DATA_PATH = ROOT / "models" / "fl_test_data.pkl"

RESULTS_ROOT = ROOT / "lora_experiment" / "results"

RANDOM_STATE = 42
EPOCHS = 20
BATCH_SIZE = 32

LORA_RANK = 4
LORA_ALPHA = 8.0
LEARNING_RATE = 0.001


# --------------------------------------------------
# CUSTOM LORA DENSE LAYER
# --------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="StressSense")
class LoRADense(tf.keras.layers.Layer):

    def __init__(
        self,
        original_kernel,
        original_bias=None,
        rank=4,
        alpha=8.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.original_kernel = np.asarray(
            original_kernel, dtype=np.float32
        )

        self.original_bias = (
            None if original_bias is None
            else np.asarray(original_bias, dtype=np.float32)
        )

        self.rank = rank
        self.alpha = alpha

    def build(self, input_shape):

        input_dim = int(input_shape[-1])
        output_dim = self.original_kernel.shape[1]

        self.base_kernel = self.add_weight(
            name="base_kernel",
            shape=(input_dim, output_dim),
            initializer=tf.keras.initializers.Constant(
                self.original_kernel
            ),
            trainable=False,
        )

        if self.original_bias is not None:
            self.base_bias = self.add_weight(
                name="base_bias",
                shape=(output_dim,),
                initializer=tf.keras.initializers.Constant(
                    self.original_bias
                ),
                trainable=False,
            )
        else:
            self.base_bias = None

        self.lora_A = self.add_weight(
            name="lora_A",
            shape=(input_dim, self.rank),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.lora_B = self.add_weight(
            name="lora_B",
            shape=(self.rank, output_dim),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs):

        base_output = tf.linalg.matmul(
            inputs, self.base_kernel
        )

        low_rank_update = tf.linalg.matmul(
            tf.linalg.matmul(inputs, self.lora_A),
            self.lora_B,
        )

        output = base_output + (
            self.alpha / self.rank
        ) * low_rank_update

        if self.base_bias is not None:
            output = output + self.base_bias
        output = tf.nn.relu(output)
        return output

    def get_config(self):

        config = super().get_config()

        config.update({
            "original_kernel": self.original_kernel.tolist(),
            "original_bias": (
                None if self.original_bias is None
                else self.original_bias.tolist()
            ),
            "rank": self.rank,
            "alpha": self.alpha,
        })

        return config


# --------------------------------------------------
# LOAD EXISTING DATA
# --------------------------------------------------

def load_data():

    print("\nLoading existing model and datasets...")

    base_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    clients = joblib.load(CLIENT_DATA_PATH)
    test_data = joblib.load(TEST_DATA_PATH)

    X_parts = []
    y_parts = []

    for participant_id, participant_data in clients.items():

        X_parts.append(
            np.asarray(participant_data["X"], dtype=np.float32)
        )

        y_parts.append(
            np.asarray(participant_data["y"]).reshape(-1)
        )

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0).astype(np.float32)

    X_test = np.asarray(
        test_data["X"], dtype=np.float32
    )

    y_test = np.asarray(
        test_data["y"]
    ).reshape(-1).astype(np.float32)

    if X.shape[1] != 34:
        raise ValueError(
            f"Expected 34 input features, received {X.shape[1]}"
        )

    print("Training samples:", len(X))
    print("Test samples:", len(X_test))
    print("Input features:", X.shape[1])

    return base_model, X, y, X_test, y_test


# --------------------------------------------------
# BUILD LORA-ADAPTED MODEL
# --------------------------------------------------

def build_lora_model(base_model):

    print("\nBuilding LoRA-adapted model...")

    if len(base_model.layers) != 7:
        raise ValueError(
            "Unexpected model architecture. "
            "Expected the inspected seven-layer model."
        )

    original_dense = base_model.get_layer("dense")
    output_dense = base_model.get_layer("dense_1")

    # Freeze the original model layers.
    for layer in base_model.layers:
        layer.trainable = False

    # New input.
    inputs = tf.keras.Input(shape=(34,), name="hrv_input")

    # Reuse original feature-extraction layers.
    x = inputs

    for layer in base_model.layers[:4]:
        x = layer(x)

    # Replace the original Dense operation with LoRA.
    lora_layer = LoRADense(
        original_kernel=original_dense.get_weights()[0],
        original_bias=(
            original_dense.get_weights()[1]
            if original_dense.use_bias else None
        ),
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        name="lora_dense",
    )

    x = lora_layer(x)

    # Reuse original dropout and output layers.
    x = base_model.get_layer("dropout_1")(x)
    outputs = output_dense(x)

    adapted_model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="StressSense_LoRA",
    )

    adapted_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return adapted_model


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate_predictions(y_true, probabilities):

    predictions = (
        probabilities.reshape(-1) >= 0.5
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities.reshape(-1))
        ),
        "classification_report": classification_report(
            y_true,
            predictions,
            zero_division=0,
        ),
    }


# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

def main():

    print("=" * 60)
    print("STRESSSENSE - LORA EFFICIENT TRAINING EXPERIMENT")
    print("=" * 60)

    # Create a unique output directory.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = RESULTS_ROOT / f"lora_run_{timestamp}"

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    # Load original model and data.
    base_model, X, y, X_test, y_test = load_data()

    # Split training data.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Baseline evaluation.
    print("\nEvaluating original federated model...")

    baseline_probabilities = base_model.predict(
        X_test,
        verbose=0
    )

    baseline_metrics = evaluate_predictions(
        y_test,
        baseline_probabilities
    )

    # Build LoRA model.
    lora_model = build_lora_model(base_model)

    # Count trainable and total parameters.
    trainable_parameters = int(
        sum(
            np.prod(weight.shape)
            for weight in lora_model.trainable_weights
        )
    )

    total_parameters = int(lora_model.count_params())

    print("\nTotal model parameters:", total_parameters)
    print("Trainable LoRA parameters:", trainable_parameters)

    # Train only LoRA adapter matrices.
    print("\nStarting LoRA training...")

    history = lora_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )

    # Evaluate adapted model.
    print("\nEvaluating LoRA-adapted model...")

    lora_probabilities = lora_model.predict(
        X_test,
        verbose=0
    )

    lora_metrics = evaluate_predictions(
        y_test,
        lora_probabilities
    )

    # Save training history.
    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(
        output_dir / "training_history.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(history_data, file, indent=4)

    # Save metrics.
    results = {
        "experiment": "LoRA adaptation of StressSense",
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history.history["loss"]),
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_parameter_percentage": (
            100.0 * trainable_parameters / total_parameters
        ),
        "baseline_metrics": baseline_metrics,
        "lora_metrics": lora_metrics,
    }

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)

    # Save adapter matrices separately.
    lora_layer = lora_model.get_layer("lora_dense")

    np.savez(
        output_dir / "lora_adapter_weights.npz",
        lora_A=lora_layer.lora_A.numpy(),
        lora_B=lora_layer.lora_B.numpy(),
    )

    # Save the adapted model.
    lora_model.save(
        output_dir / "lora_adapted_model.keras"
    )

    # Save prediction comparison.
    np.savetxt(
        output_dir / "prediction_comparison.csv",
        np.column_stack([
            y_test,
            baseline_probabilities.reshape(-1),
            lora_probabilities.reshape(-1),
        ]),
        delimiter=",",
        header="actual_label,baseline_probability,lora_probability",
        comments="",
    )

    # Print summary.
    print("\n" + "=" * 60)
    print("LORA EXPERIMENT COMPLETED")
    print("=" * 60)

    print("\nBASELINE MODEL:")
    for key, value in baseline_metrics.items():
        if key != "classification_report":
            print(f"{key}: {value:.4f}")

    print("\nLORA MODEL:")
    for key, value in lora_metrics.items():
        if key != "classification_report":
            print(f"{key}: {value:.4f}")

    print("\nTrainable parameters:", trainable_parameters)
    print("Total parameters:", total_parameters)

    print("\nResults saved in:")
    print(output_dir)


if __name__ == "__main__":
    main()