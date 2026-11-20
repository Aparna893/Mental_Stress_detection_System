from common import *

df = load_data()
print("Data:", df.shape, "participants:", df["participant"].nunique())

for fold, tr, te in participant_folds(df, 5):
    overlap = set(tr["participant"]) & set(te["participant"])
    print(f"Fold {fold}: train={len(tr)} test={len(te)} overlap={overlap}")

model = build_model()
print("Model parameters:", model.count_params())