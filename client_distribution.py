
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the common training dataset
df = pd.read_csv("data/comparison_train.csv")

# Count No Stress and Stress samples for each participant
distribution = pd.crosstab(
    df["participant"],
    df["label"]
)

# Ensure both classes are present as columns
distribution = distribution.reindex(columns=[0, 1], fill_value=0)

# Rename columns
distribution.columns = ["No Stress", "Stress"]

# Calculate total samples and stress percentage
distribution["Total"] = distribution["No Stress"] + distribution["Stress"]

distribution["Stress Percentage"] = (
    distribution["Stress"] / distribution["Total"] * 100
).round(2)

# Display results
print("\nCLASS DISTRIBUTION ACROSS FEDERATED CLIENTS")
print(distribution.to_string())

# Save results
os.makedirs("comparison_results", exist_ok=True)

distribution.to_csv(
    "comparison_results/client_distribution.csv"
)

# Plot client-wise class distribution
distribution[["No Stress", "Stress"]].plot(
    kind="bar",
    figsize=(13, 6)
)

plt.title("Class Distribution Across 23 Federated Clients")
plt.xlabel("Federated Clients (Participants)")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.legend(title="Class")
plt.tight_layout()

plt.savefig(
    "comparison_results/client_class_distribution.png"
)

plt.show()

# Overall class distribution
print("\nOVERALL TRAINING CLASS DISTRIBUTION")
print(df["label"].value_counts().sort_index())
