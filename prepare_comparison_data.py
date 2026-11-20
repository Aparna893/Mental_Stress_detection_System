
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/participant_hrv_34_features.csv")
OUTPUT_DIR = Path("data")
RANDOM_STATE = 42
TEST_SIZE = 0.20

df = pd.read_csv(DATA_PATH)

# Validate the labels before splitting.
expected_labels = {
    "no stress": 0,
    "interruption": 1,
    "time pressure": 1,
}

conditions = df["condition"].astype(str).str.strip().str.lower()
expected = conditions.map(expected_labels)

if expected.isna().any():
    raise ValueError("Unexpected condition found.")

if not expected.equals(df["label"].astype(int)):
    raise ValueError("Condition and binary label mappings disagree.")

# Keep the same participant-specific held-out rows
# for all three experiments.
train_indices = []
test_indices = []

for participant in sorted(
    df["participant"].unique(),
    key=lambda x: int(str(x).replace("p", ""))
):
    participant_indices = df.index[
        df["participant"] == participant
    ].to_numpy()

    y = df.loc[participant_indices, "label"].to_numpy()

    train_idx, test_idx = train_test_split(
        participant_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    train_indices.extend(train_idx.tolist())
    test_indices.extend(test_idx.tolist())

train_df = df.loc[train_indices].copy()
test_df = df.loc[test_indices].copy()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_path = OUTPUT_DIR / "comparison_train.csv"
test_path = OUTPUT_DIR / "comparison_test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

# Verify that the two splits do not overlap.
assert set(train_indices).isdisjoint(test_indices)
assert len(train_df) + len(test_df) == len(df)

print("Comparison split created successfully.")
print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))
print("Participants:", df["participant"].nunique())

print("\nTraining labels:")
print(train_df["label"].value_counts().sort_index())

print("\nTesting labels:")
print(test_df["label"].value_counts().sort_index())

print("\nTraining conditions:")
print(train_df["condition"].value_counts())

print("\nTesting conditions:")
print(test_df["condition"].value_counts())

print("\nSaved files:")
print(train_path)
print(test_path)
