
import pandas as pd
from pathlib import Path

train_path = Path("data/train.csv")
test_path = Path("data/test.csv")

for path in (train_path, test_path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

print("Loading datasets...")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("\nTRAIN SHAPE:", train.shape)
print("TEST SHAPE :", test.shape)

print("\nTRAIN COLUMNS:")
print(train.columns.tolist())

print("\nTEST COLUMNS:")
print(test.columns.tolist())

print("\nCOLUMNS MATCH:", train.columns.tolist() == test.columns.tolist())

required = {"condition", "datasetId"}

for name, df in [("TRAIN", train), ("TEST", test)]:
    print(f"\n--- {name} CHECKS ---")

    missing_columns = required - set(df.columns)
    if missing_columns:
        print("Missing required columns:", missing_columns)
        continue

    print("\nCondition counts:")
    print(df["condition"].value_counts(dropna=False))

    print("\ndatasetId counts:")
    print(df["datasetId"].value_counts(dropna=False))

    print("\nMissing values:", int(df.isna().sum().sum()))

    labels = df["condition"].astype(str).str.strip().str.lower()
    print("\nConditions outside expected categories:")
    print(sorted(set(labels) -
                 {"no stress", "interruption", "time pressure"}))

if set(train.columns) != set(test.columns):
    print("\nWARNING: Train and test columns differ.")

if "condition" in train and "condition" in test:
    train_conditions = set(
        train["condition"].dropna().astype(str).str.strip().str.lower()
    )
    test_conditions = set(
        test["condition"].dropna().astype(str).str.strip().str.lower()
    )
    print("\nTrain conditions:", train_conditions)
    print("Test conditions :", test_conditions)
    print("Condition sets match:", train_conditions == test_conditions)

print("\nAudit complete.")
