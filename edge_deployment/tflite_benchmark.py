
import json
import time
import joblib
import numpy as np
import tensorflow as tf

from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "federated_model.keras"
TEST_DATA_PATH = ROOT / "models" / "fl_test_data.pkl"

RESULTS_ROOT = ROOT / "edge_deployment" / "results"

WARMUP_RUNS = 10
BENCHMARK_RUNS = 100


# --------------------------------------------------
# LOAD MODEL AND TEST DATA
# --------------------------------------------------

def load_resources():

    print("\nLoading original federated model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    test_data = joblib.load(TEST_DATA_PATH)

    X_test = np.asarray(
        test_data["X"], dtype=np.float32
    )

    y_test = np.asarray(
        test_data["y"]
    ).reshape(-1).astype(int)

    print("Test samples:", len(X_test))
    print("Input shape:", X_test.shape)
    print("Model input:", model.input_shape)

    if X_test.shape[1] != 34:
        raise ValueError("Expected 34 input features.")

    return model, X_test, y_test


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(y_true, probabilities):

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
            roc_auc_score(
                y_true, probabilities.reshape(-1)
            )
        ),
    }


# --------------------------------------------------
# CONVERT TO TENSORFLOW LITE
# --------------------------------------------------



def convert_model(model, output_path, quantized=False):

    print("\nConverting:", output_path.name)

    # Build a temporary inference model with batch size fixed to 1.
    fixed_input = tf.keras.Input(
        batch_size=1,
        shape=(model.input_shape[-1],),
        dtype=tf.float32
    )

    fixed_output = model(fixed_input, training=False)

    inference_model = tf.keras.Model(
        inputs=fixed_input,
        outputs=fixed_output
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(
        inference_model
    )

    # Use built-in operations first.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]

    if quantized:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    try:
        tflite_model = converter.convert()

    except Exception as error:
        print("Built-in conversion failed.")
        print("Trying Select TF Ops fallback...")
        print(str(error)[:500])

        converter = tf.lite.TFLiteConverter.from_keras_model(
            inference_model
        )

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]

        converter._experimental_lower_tensor_list_ops = False

        if quantized:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

    with open(output_path, "wb") as file:
        file.write(tflite_model)

    print("Saved:", output_path)
    print("Size:", round(len(tflite_model) / 1024, 2), "KB")

    return len(tflite_model)

# --------------------------------------------------
# TFLITE INFERENCE
# --------------------------------------------------

def run_tflite(model_path, X_test):

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    input_dtype = input_details[0]["dtype"]

    predictions = []

    for sample in X_test:

        input_data = sample.reshape(1, -1).astype(
            input_dtype
        )

        interpreter.set_tensor(
            input_index, input_data
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_index
        )

        predictions.append(
            float(output.reshape(-1)[0])
        )

    return np.asarray(predictions, dtype=np.float32)


# --------------------------------------------------
# LATENCY BENCHMARK
# --------------------------------------------------

def benchmark_keras(model, sample):

    sample = sample.reshape(1, -1).astype(np.float32)

    for _ in range(WARMUP_RUNS):
        model(sample, training=False)

    timings = []

    for _ in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        model(sample, training=False)

        end = time.perf_counter()

        timings.append((end - start) * 1000)

    return {
        "mean_ms": float(np.mean(timings)),
        "median_ms": float(np.median(timings)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
    }


def benchmark_tflite(model_path, sample):

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    input_data = sample.reshape(1, -1).astype(
        input_details[0]["dtype"]
    )

    for _ in range(WARMUP_RUNS):

        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        interpreter.get_tensor(output_index)

    timings = []

    for _ in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        interpreter.get_tensor(output_index)

        end = time.perf_counter()

        timings.append((end - start) * 1000)

    return {
        "mean_ms": float(np.mean(timings)),
        "median_ms": float(np.median(timings)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
    }


# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

def main():

    print("=" * 60)
    print("STRESSSENSE - TENSORFLOW LITE EDGE BENCHMARK")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = RESULTS_ROOT / f"tflite_run_{timestamp}"

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    model, X_test, y_test = load_resources()

    # Original Keras predictions.
    print("\nRunning original Keras model...")

    keras_predictions = model.predict(
        X_test,
        verbose=0
    ).reshape(-1)

    keras_metrics = calculate_metrics(
        y_test, keras_predictions
    )

    # Float32 conversion.
    float_path = output_dir / "stresssense_float32.tflite"

    float_size = convert_model(
        model,
        float_path,
        quantized=False
    )

    float_predictions = run_tflite(
        float_path, X_test
    )

    float_metrics = calculate_metrics(
        y_test, float_predictions
    )

    # Dynamic-range quantization.
    quant_path = output_dir / "stresssense_dynamic_quant.tflite"

    quant_size = None
    quant_predictions = None
    quant_metrics = None
    quant_error = None

    try:

        quant_size = convert_model(
            model,
            quant_path,
            quantized=True
        )

        quant_predictions = run_tflite(
            quant_path, X_test
        )

        quant_metrics = calculate_metrics(
            y_test, quant_predictions
        )

        quant_error = {
            "mean_absolute_error": float(
                np.mean(
                    np.abs(
                        keras_predictions - quant_predictions
                    )
                )
            ),
            "max_absolute_error": float(
                np.max(
                    np.abs(
                        keras_predictions - quant_predictions
                    )
                )
            ),
        }

    except Exception as error:

        print("\nDynamic-range quantization failed:")
        print(str(error))

    # Prediction agreement.
    float_error = {
        "mean_absolute_error": float(
            np.mean(
                np.abs(
                    keras_predictions - float_predictions
                )
            )
        ),
        "max_absolute_error": float(
            np.max(
                np.abs(
                    keras_predictions - float_predictions
                )
            )
        ),
    }

    # Latency benchmark.
    print("\nBenchmarking single-sample inference...")

    sample = X_test[0]

    latency = {
        "keras_float32": benchmark_keras(
            model, sample
        ),
        "tflite_float32": benchmark_tflite(
            float_path, sample
        ),
    }

    if quant_size is not None:
        latency["tflite_dynamic_quant"] = benchmark_tflite(
            quant_path, sample
        )

    # Model size comparison.
    model_sizes = {
        "keras_model_kb": float(
            MODEL_PATH.stat().st_size / 1024
        ),
        "tflite_float32_kb": float(
            float_size / 1024
        ),
        "tflite_dynamic_quant_kb": (
            None if quant_size is None
            else float(quant_size / 1024)
        ),
    }

    # Save predictions.
    columns = [
        y_test,
        keras_predictions,
        float_predictions,
    ]

    header = [
        "actual_label",
        "keras_probability",
        "tflite_float32_probability",
    ]

    if quant_predictions is not None:
        columns.append(quant_predictions)
        header.append("tflite_dynamic_quant_probability")

    np.savetxt(
        output_dir / "prediction_comparison.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(header),
        comments="",
    )

    # Save results.
    results = {
        "experiment": "StressSense TensorFlow Lite edge deployment",
        "test_samples": int(len(X_test)),
        "input_features": 34,
        "benchmark_runs": BENCHMARK_RUNS,
        "warmup_runs": WARMUP_RUNS,
        "keras_metrics": keras_metrics,
        "tflite_float32_metrics": float_metrics,
        "tflite_dynamic_quant_metrics": quant_metrics,
        "float32_prediction_error": float_error,
        "dynamic_quant_prediction_error": quant_error,
        "model_sizes": model_sizes,
        "latency": latency,
    }

    with open(
        output_dir / "benchmark_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)

    # Save readable summary.
    print("\n" + "=" * 60)
    print("EDGE DEPLOYMENT RESULTS")
    print("=" * 60)

    print("\nKERAS MODEL:")
    for key, value in keras_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nTFLITE FLOAT32 MODEL:")
    for key, value in float_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nPREDICTION ERROR - FLOAT32:")
    print(float_error)

    if quant_metrics is not None:
        print("\nTFLITE DYNAMIC QUANTIZATION:")
        for key, value in quant_metrics.items():
            print(f"{key}: {value:.4f}")

        print("\nQUANTIZED PREDICTION ERROR:")
        print(quant_error)

    print("\nMODEL SIZES:")
    print(model_sizes)

    print("\nLATENCY:")
    for name, values in latency.items():
        print(name, values)

    print("\nResults saved in:")
    print(output_dir)


if __name__ == "__main__":
    main()
