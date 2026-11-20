
import pandas as pd

path = "data/participant_hrv_34_features.csv"
df = pd.read_csv(path)

features = [
    "MEAN_RR", "MEDIAN_RR", "SDRR", "RMSSD", "SDSD",
    "SDRR_RMSSD", "HR", "pNN25", "pNN50", "SD1", "SD2",
    "KURT", "SKEW", "MEAN_REL_RR", "MEDIAN_REL_RR",
    "SDRR_REL_RR", "RMSSD_REL_RR", "SDSD_REL_RR",
    "SDRR_RMSSD_REL_RR", "KURT_REL_RR", "SKEW_REL_RR",
    "VLF", "VLF_PCT", "LF", "LF_PCT", "LF_NU",
    "HF", "HF_PCT", "HF_NU", "TP", "LF_HF", "HF_LF",
    "sampen", "higuci"
]

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nMissing required features:")
print(sorted(set(features) - set(df.columns)))

print("\nMissing values:", int(df.isna().sum().sum()))

print("\nLabel counts:")
print(df["label"].value_counts(dropna=False).sort_index())

print("\nCondition counts:")
print(df["condition"].value_counts(dropna=False))

print("\nLabel versus condition:")
print(pd.crosstab(df["condition"], df["label"], dropna=False))

print("\nParticipant count:", df["participant"].nunique())
print("\nSamples per participant:")
print(df.groupby("participant").size().to_string())

print("\nLabels per participant:")
print(pd.crosstab(df["participant"], df["label"]))

print("\nAudit complete.")
