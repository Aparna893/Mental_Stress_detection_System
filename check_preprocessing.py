
import ast
from pathlib import Path

files = [
    "stress_detection.py",
    "federated_learning.py",
    "differential_privacy.py",
    "compare_fl_dp.py",
]

keywords = [
    "condition", "label", "get_dummies",
    "LabelEncoder", "replace(", "map(",
    "StandardScaler", "MinMaxScaler",
    "fit_transform", "transform(",
    "load_model", "evaluate(", "predict(",
    "accuracy", "train_test_split",
]

for filename in files:
    path = Path(filename)
    print(f"\n{'=' * 65}")
    print(f"FILE: {filename}")
    print("=" * 65)

    if not path.exists():
        print("File not found.")
        continue

    lines = path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()

    for i, line in enumerate(lines, start=1):
        if any(k.lower() in line.lower() for k in keywords):
            print(f"{i}: {line.rstrip()}")
