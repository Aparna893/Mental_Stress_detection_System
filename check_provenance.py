import pandas as pd

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

print("Loading datasets...")
participant = pd.read_csv("data/participant_hrv_34_features.csv")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Convert HRV features to numeric values
for df in [participant, train, test]:
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def compare(source, source_name):
    print(f"\n--- Comparing with {source_name} ---")

    left = participant[features].round(8).copy()
    right = source[features].round(8).copy()

    left["_participant_row"] = range(len(left))
    right["_source_row"] = range(len(right))

    matches = left.merge(
        right,
        on=features,
        how="inner"
    )

    matched_rows = matches["_participant_row"].nunique()

    print("Participant records:", len(participant))
    print("Exact matching records (rounded to 8 decimals):", matched_rows)
    print("Unmatched participant records:", len(participant) - matched_rows)

compare(train, "train.csv")
compare(test, "test.csv")

print("\n--- Feature range comparison: RMSSD ---")
print("Original train:", train["RMSSD"].describe().to_dict())
print("Original test:", test["RMSSD"].describe().to_dict())
print("Participant:", participant["RMSSD"].describe().to_dict())

print("\nProvenance check completed.")
