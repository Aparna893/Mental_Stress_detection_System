
import pandas as pd

# Read only 100 rows instead of the full dataset
df = pd.read_csv("data/train.csv", nrows=100)

print("Shape of sample:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nData types:")
print(df.dtypes)

print("\nMissing values in sample:")
print(df.isnull().sum())

# Save a small sample for sharing
df.to_csv("train_sample.csv", index=False)

print("\nSaved train_sample.csv successfully!")
