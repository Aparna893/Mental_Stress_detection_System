
from pathlib import Path
from datetime import datetime
import json

import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "federated_model.keras"
SCALER_PATH = ROOT / "models" / "federated_scaler.pkl"
DATA_PATH = ROOT / "data" / "participant_hrv_34_features.csv"

OUTPUT_ROOT = ROOT / "explainability" / "results"

# A new timestamped directory prevents overwriting old results.
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = OUTPUT_ROOT / f"shap_run_{RUN_ID}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)


# --------------------------------------------------
# 2. HRV FEATURES
# --------------------------------------------------

FEATURES = [
    "MEAN_RR", "MEDIAN_RR", "SDRR", "RMSSD", "SDSD",
    "SDRR_RMSSD", "HR", "pNN25", "pNN50", "SD1", "SD2",
    "KURT", "SKEW", "MEAN_REL_RR", "MEDIAN_REL_RR",
    "SDRR_REL_RR", "RMSSD_REL_RR", "SDSD_REL_RR",
    "SDRR_RMSSD_REL_RR", "KURT_REL_RR", "SKEW_REL_RR",
    "VLF", "VLF_PCT", "LF", "LF_PCT", "LF_NU", "HF",
    "HF_PCT", "HF_NU", "TP", "LF_HF", "HF_LF",
    "sampen", "higuci"
]

RANDOM_STATE = 42
BACKGROUND_SIZE = 40
EXPLAIN_SIZE = 20


# --------------------------------------------------
# 3. LOAD EXISTING MODEL AND SCALER
# --------------------------------------------------

def load_artifacts():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not SCALER_PATH.exists():
        raise FileNotFoundError(SCALER_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    scaler = joblib.load(SCALER_PATH)

    if model.input_shape[-1] != len(FEATURES):
        raise ValueError("Model feature count mismatch.")

    if scaler.n_features_in_ != len(FEATURES):
        raise ValueError("Scaler feature count mismatch.")

    print("Federated model loaded.")
    print("Federated scaler loaded.")

    return model, scaler


# --------------------------------------------------
# 4. LOAD DATASET
# --------------------------------------------------

def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + ["label"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURES].apply(
        pd.to_numeric,
        errors="raise"
    )

    y = pd.to_numeric(
        df["label"],
        errors="raise"
    ).astype(int)

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains missing or infinite features.")

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary: 0 or 1.")

    print("Dataset loaded:", X.shape)
    print("Label distribution:")
    print(y.value_counts().sort_index())

    return X, y


# --------------------------------------------------
# 5. MODEL PREDICTION FUNCTION
# --------------------------------------------------

def make_predict_function(model, scaler):

    def predict_probability(data):

        data = np.asarray(data, dtype=np.float32)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        dataframe = pd.DataFrame(
            data,
            columns=FEATURES
        )

        scaled = scaler.transform(dataframe)

        predictions = model.predict(
            scaled,
            verbose=0
        ).reshape(-1)

        return predictions

    return predict_probability


# --------------------------------------------------
# 6. GENERATE SHAP EXPLANATIONS
# --------------------------------------------------

def run_shap():

    model, scaler = load_artifacts()

    X, y = load_dataset()

    # Use raw HRV values as the explainer input.
    # The prediction function applies the existing scaler.
    X_train, X_explain, _, y_explain = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    background = X_train.sample(
        n=min(BACKGROUND_SIZE, len(X_train)),
        random_state=RANDOM_STATE
    )

    explain_data = X_explain.sample(
        n=min(EXPLAIN_SIZE, len(X_explain)),
        random_state=RANDOM_STATE
    )

    print("Background samples:", len(background))
    print("Samples to explain:", len(explain_data))

    predict_fn = make_predict_function(model, scaler)

    # SHAP permutation explainer.
    masker = shap.maskers.Independent(
        background.to_numpy(),
        max_samples=len(background)
    )

    explainer = shap.Explainer(
        predict_fn,
        masker,
        algorithm="permutation",
        feature_names=FEATURES
    )

    print("Calculating SHAP values...")

    shap_values = explainer(
        explain_data.to_numpy(),
        max_evals=2 * len(FEATURES) + 1,
        batch_size=16
    )

    print("SHAP calculation completed.")

    # --------------------------------------------------
    # 7. SAVE NUMERICAL RESULTS
    # --------------------------------------------------

    values = np.asarray(shap_values.values)

    if values.ndim == 3:
        values = values[..., 0]

    if values.shape != explain_data.shape:
        raise ValueError(
            f"Unexpected SHAP shape: {values.shape}"
        )

    importance = np.abs(values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Mean_Absolute_SHAP": importance
    }).sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    importance_df.to_csv(
        OUTPUT_DIR / "shap_feature_importance.csv",
        index=False
    )

    shap_df = pd.DataFrame(
        values,
        columns=FEATURES
    )

    shap_df.to_csv(
        OUTPUT_DIR / "shap_values.csv",
        index=False
    )

    predictions = predict_fn(explain_data.to_numpy())

    prediction_df = pd.DataFrame({
        "Actual_Label": y_explain.loc[explain_data.index].to_numpy(),
        "Predicted_Probability": predictions,
        "Predicted_Label": (predictions >= 0.5).astype(int)
    })

    prediction_df.to_csv(
        OUTPUT_DIR / "explained_predictions.csv",
        index=False
    )

    # --------------------------------------------------
    # 8. GLOBAL SUMMARY PLOT
    # --------------------------------------------------

    plt.figure()

    shap.plots.beeswarm(
        shap_values,
        max_display=20,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # 9. FEATURE IMPORTANCE PLOT
    # --------------------------------------------------

    plt.figure(figsize=(10, 8))

    shap.plots.bar(
        shap_values,
        max_display=20,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # 10. INDIVIDUAL WATERFALL PLOT
    # --------------------------------------------------

    plt.figure()

    shap.plots.waterfall(
        shap_values[0],
        max_display=15,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_waterfall.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # 11. SAVE EXPERIMENT INFORMATION
    # --------------------------------------------------

    metadata = {
        "experiment": "SHAP Explainability",
        "model": str(MODEL_PATH),
        "scaler": str(SCALER_PATH),
        "dataset": str(DATA_PATH),
        "number_of_features": len(FEATURES),
        "background_samples": len(background),
        "explained_samples": len(explain_data),
        "explainer": "PermutationExplainer",
        "threshold": 0.5,
        "random_state": RANDOM_STATE
    }

    with open(
        OUTPUT_DIR / "experiment_info.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nSHAP EXPERIMENT COMPLETED")
    print("Results saved in:")
    print(OUTPUT_DIR)

    print("\nTop 10 influential features:")
    print(importance_df.head(10).to_string(index=False))


# --------------------------------------------------
# 12. MAIN
# --------------------------------------------------

if __name__ == "__main__":

    try:
        run_shap()

    except Exception as error:
        print("\nSHAP EXPERIMENT FAILED")
        print(type(error).__name__, ":", error)
        raise
