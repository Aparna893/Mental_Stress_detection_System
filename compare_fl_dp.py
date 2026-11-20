import re
import subprocess
import sys


# ============================================================
# READ SETTINGS FROM EXISTING FILES
# ============================================================

def read_value(filename, pattern, name):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    match = re.search(pattern, text, re.MULTILINE)

    if not match:
        raise ValueError(
            f"Could not find {name} in {filename}"
        )

    return match.group(1)


# ============================================================
# READ CENTRALIZED SETTINGS
# ============================================================

centralized_epochs = int(
    read_value(
        "stress_detection.py",
        r"epochs\s*=\s*(\d+)",
        "epochs"
    )
)


# ============================================================
# READ FL SETTINGS
# ============================================================

num_rounds = int(
    read_value(
        "federated_learning.py",
        r"NUM_ROUNDS\s*=\s*(\d+)",
        "NUM_ROUNDS"
    )
)

local_epochs = int(
    read_value(
        "federated_learning.py",
        r"LOCAL_EPOCHS\s*=\s*(\d+)",
        "LOCAL_EPOCHS"
    )
)

batch_size = int(
    read_value(
        "federated_learning.py",
        r"BATCH_SIZE\s*=\s*(\d+)",
        "BATCH_SIZE"
    )
)


# ============================================================
# READ DP SETTINGS
# ============================================================

dp_rounds = int(
    read_value(
        "differential_privacy.py",
        r"NUM_ROUNDS\s*=\s*(\d+)",
        "NUM_ROUNDS"
    )
)

dp_local_epochs = int(
    read_value(
        "differential_privacy.py",
        r"LOCAL_EPOCHS\s*=\s*(\d+)",
        "LOCAL_EPOCHS"
    )
)

dp_batch_size = int(
    read_value(
        "differential_privacy.py",
        r"BATCH_SIZE\s*=\s*(\d+)",
        "BATCH_SIZE"
    )
)

clipping_norm = float(
    read_value(
        "differential_privacy.py",
        r"CLIPPING_NORM\s*=\s*([0-9.]+)",
        "CLIPPING_NORM"
    )
)

noise_multiplier = float(
    read_value(
        "differential_privacy.py",
        r"NOISE_MULTIPLIER\s*=\s*([0-9.]+)",
        "NOISE_MULTIPLIER"
    )
)


# ============================================================
# DISPLAY SETTINGS
# ============================================================

print("\n" + "=" * 70)
print("SETTINGS DETECTED")
print("=" * 70)

print(f"Centralized Epochs  : {centralized_epochs}")
print(f"FL Rounds           : {num_rounds}")
print(f"FL Local Epochs     : {local_epochs}")
print(f"FL Batch Size       : {batch_size}")
print(f"DP-FL Rounds        : {dp_rounds}")
print(f"DP-FL Local Epochs  : {dp_local_epochs}")
print(f"DP-FL Batch Size    : {dp_batch_size}")
print(f"DP Clipping Norm    : {clipping_norm}")
print(f"DP Noise Multiplier : {noise_multiplier}")


# ============================================================
# RUN A PYTHON FILE
# ============================================================

def run_script(filename):

    print("\n" + "=" * 70)
    print(f"RUNNING: {filename}")
    print("=" * 70)

    process = subprocess.run(
        [sys.executable, filename],
        capture_output=True,
        text=True
    )

    if process.returncode != 0:

        print(process.stdout)
        print(process.stderr)

        raise RuntimeError(
            f"{filename} failed."
        )

    return process.stdout


# ============================================================
# RUN CENTRALIZED
# ============================================================

centralized_output = run_script(
    "stress_detection.py"
)


# ============================================================
# RUN FEDERATED LEARNING
# ============================================================

fl_output = run_script(
    "federated_learning.py"
)


# ============================================================
# RUN DP-FEDERATED LEARNING
# ============================================================

dp_output = run_script(
    "differential_privacy.py"
)


# ============================================================
# EXTRACT CENTRALIZED TRAINING ACCURACY
# ============================================================

centralized_accuracies = []

for line in centralized_output.splitlines():

    # Keras verbose=1 normally prints:
    # accuracy: 0.xxxx

    if "accuracy:" in line.lower():

        matches = re.findall(
            r"accuracy:\s*([0-9]*\.?[0-9]+)",
            line,
            re.IGNORECASE
        )

        for value in matches:

            accuracy = float(value)

            if accuracy <= 1:
                accuracy *= 100

            centralized_accuracies.append(
                accuracy
            )


# Keep only the requested number of epochs
centralized_accuracies = (
    centralized_accuracies[:centralized_epochs]
)


# ============================================================
# EXTRACT FL ROUND ACCURACIES
# ============================================================

fl_matches = re.findall(
    r"Round\s+(\d+)\s+Test Accuracy:\s*"
    r"([0-9]*\.?[0-9]+)%",
    fl_output,
    re.IGNORECASE
)

fl_accuracies = [
    float(match[1])
    for match in fl_matches
]


# ============================================================
# EXTRACT DP-FL ROUND ACCURACIES
# ============================================================

dp_matches = re.findall(
    r"Round\s+(\d+)\s+Test Accuracy:\s*"
    r"([0-9]*\.?[0-9]+)%",
    dp_output,
    re.IGNORECASE
)

dp_accuracies = [
    float(match[1])
    for match in dp_matches
]


# ============================================================
# PRINT CENTRALIZED RESULTS
# ============================================================

print("\n\n" + "=" * 70)
print("CENTRALIZED MODEL ACCURACY")
print("=" * 70)

if centralized_accuracies:

    for i, accuracy in enumerate(
        centralized_accuracies,
        start=1
    ):

        print(
            f"{i}/{centralized_epochs} = "
            f"{accuracy:.2f}%"
        )

else:

    print(
        "No centralized epoch accuracy found."
    )


# ============================================================
# PRINT FL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FEDERATED LEARNING ACCURACY")
print("=" * 70)

if fl_accuracies:

    for i, accuracy in enumerate(
        fl_accuracies,
        start=1
    ):

        print(
            f"{i}/{num_rounds} = "
            f"{accuracy:.2f}%"
        )

else:

    print(
        "No FL round accuracy found."
    )


# ============================================================
# PRINT DP-FL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("DP-FEDERATED LEARNING ACCURACY")
print("=" * 70)

if dp_accuracies:

    for i, accuracy in enumerate(
        dp_accuracies,
        start=1
    ):

        print(
            f"{i}/{dp_rounds} = "
            f"{accuracy:.2f}%"
        )

else:

    print(
        "No DP-FL round accuracy found."
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETED")
print("=" * 70)