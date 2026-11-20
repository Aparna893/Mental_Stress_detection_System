
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load test dataset
test_df = pd.read_csv("data/comparison_test.csv")

# 2. Separate features and labels
X_test = test_df.drop(
    columns=["participant", "ElapsedTime", "condition", "label"]
)
y_test = test_df["label"].astype(int)

# 3. Model and scaler paths
models = {
    "Centralized CNN-BiLSTM": (
        "models/centralized_model.keras",
        "models/scaler.pkl"
    ),
    "Federated CNN-BiLSTM": (
        "models/federated_model.keras",
        "models/federated_scaler.pkl"
    ),
    "DP Federated CNN-BiLSTM": (
        "models/dp_federated_model.keras",
        "models/dp_federated_scaler.pkl"
    )
}

results = []
predictions = {}

os.makedirs("comparison_results", exist_ok=True)

# 4. Evaluate each model
for name, (model_path, scaler_path) in models.items():

    print(f"\nEvaluating: {name}")

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    X_scaled = scaler.transform(X_test)

    # Reshape for CNN-BiLSTM
    X_reshaped = X_scaled.reshape(
        X_scaled.shape[0], X_scaled.shape[1], 1
    )

    y_prob = model.predict(X_reshaped, verbose=0)

    # Convert probabilities into class predictions
    if y_prob.ndim == 2 and y_prob.shape[1] > 1:
        y_pred = np.argmax(y_prob, axis=1)
    else:
        y_pred = (y_prob.reshape(-1) >= 0.5).astype(int)

    predictions[name] = y_pred

    # Calculate metrics
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, y_pred, zero_division=0
        )
    })

    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No Stress", "Stress"],
        zero_division=0
    ))

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Stress", "Stress"]
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        f"comparison_results/{filename}_confusion_matrix.png"
    )
    plt.close()

# 5. Create comparison table
results_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "comparison_results/model_comparison.csv",
    index=False
)

# 6. Generate metric comparison graph
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1),
    rot=15
)

plt.title("Performance Comparison of Three Models")
plt.ylabel("Score")
plt.xlabel("Model")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig("comparison_results/model_comparison_graph.png")
plt.show()

print("\nAll comparison results saved in comparison_results/")