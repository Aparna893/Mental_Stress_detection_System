import pandas as pd

DATA_PATH = "data/participant_hrv_34_features.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nParticipants:")
print(df["participant"].unique())
print("Number of participants:", df["participant"].nunique())

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nSamples per participant:")
print(df["participant"].value_counts().sort_index())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nFirst 5 rows:")
print(df.head())