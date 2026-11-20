from pathlib import Path

sections = {
    "stress_detection.py": [(220, 420)],
    "federated_learning.py": [(380, 520)],
    "differential_privacy.py": [(457, 829)],
}

for filename, ranges in sections.items():
    path = Path(filename)

    print("\n" + "=" * 75)
    print("FILE:", filename)
    print("=" * 75)

    if not path.exists():
        print("File not found!")
        continue

    lines = path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()

    for start, end in ranges:
        print(f"\n--- Lines {start} to {end} ---")

        for i in range(start - 1, min(end, len(lines))):
            print(f"{i + 1}: {lines[i]}")