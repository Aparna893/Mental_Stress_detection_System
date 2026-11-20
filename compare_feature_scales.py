import pandas as pd
import numpy as np

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

old = pd.concat([
    pd.read_csv("data/train.csv"),
    pd.read_csv("data/test.csv")
], ignore_index=True)

new = pd.read_csv("data/participant_hrv_34_features.csv")

print("FEATURE SCALE COMPARISON")
print("-" * 100)

for col in features:
    old_values = pd.to_numeric(old[col], errors="coerce").dropna()
    new_values = pd.to_numeric(new[col], errors="coerce").dropna()

    old_mean = old_values.mean()
    new_mean = new_values.mean()

    if abs(new_mean) > 1e-12:
        ratio = old_mean / new_mean
    else:
        ratio = np.nan

    print(
        f"{col:22s} "
        f"Old Mean: {old_mean:14.7f} "
        f"New Mean: {new_mean:14.7f} "
        f"Old/New: {ratio:14.4f}"
    )

print("\nCONDITION DISTRIBUTION IN PARTICIPANT DATASET")
print(new["condition"].value_counts())

print("\nPARTICIPANT INFORMATION")
print("Unique participants:", new["participant"].nunique())
print("Unique IDs:", sorted(new["participant"].unique()))

