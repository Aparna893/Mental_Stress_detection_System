import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score

RUNS = Path("runs")
OUT = Path("final_results")
OUT.mkdir(exist_ok=True)
METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "auc"]


def load_all(prefix, need, keys):
    frames, used = [], []
    for d in sorted(RUNS.glob(prefix + "_2*")):
        f = d / "results.json"
        if not f.exists():
            continue
        r = json.load(open(f))
        if any(r["args"].get(k) != v for k, v in need.items()):
            continue
        t = pd.DataFrame(r["per_fold"]).drop(
            columns=["confusion_matrix", "test_participants"], errors="ignore")
        t["run"] = d.name
        frames.append(t)
        used.append(d)
    if not frames:
        raise SystemExit(f"No finished run found for '{prefix}' with {need}")
    t = pd.concat(frames, ignore_index=True)
    t = t.drop_duplicates(subset=keys, keep="last")
    kept = t[["run"] + keys].drop_duplicates()
    ps = []
    for d in used:
        p = pd.read_csv(d / "test_probabilities.csv")
        p["run"] = d.name
        ps.append(p)
    p = pd.concat(ps, ignore_index=True).merge(kept, on=["run"] + keys, how="inner")
    print(f"{prefix:20s} runs={sorted(set(t['run']))} seeds={sorted(t['seed'].unique())} "
          f"rows={len(t)}")
    return t, p


cen, cen_p = load_all("centralized", {"epochs": 20, "folds": 5}, ["seed", "fold"])
fed, fed_p = load_all("federated_natural",
                      {"rounds": 10, "local_epochs": 2, "folds": 5}, ["seed", "fold"])
iid, iid_p = load_all("federated_iid",
                      {"rounds": 10, "local_epochs": 2, "folds": 5}, ["seed", "fold"])
dp, dp_p = load_all("dp_federated", {"rounds": 10, "clip": 0.5, "folds": 5},
                    ["noise_multiplier", "seed", "fold"])

groups = {"Centralized": cen, "FedAvg (natural)": fed, "FedAvg (IID)": iid}
eps_of = {}
for z in sorted(dp["noise_multiplier"].unique()):
    label = f"DP-FedAvg z={z:g}"
    groups[label] = dp[dp["noise_multiplier"] == z]
    e = groups[label]["epsilon"].dropna()
    eps_of[label] = float(e.iloc[0]) if len(e) else np.nan

rows = []
for label, t in groups.items():
    row = {"model": label, "epsilon": eps_of.get(label, np.nan), "n_folds": len(t)}
    for k in METRICS:
        row[k] = f"{t[k].mean():.3f} +/- {t[k].std(ddof=1):.3f}"
    row["majority_baseline"] = round(t["majority_baseline"].mean(), 3)
    rows.append(row)
table = pd.DataFrame(rows)
print("\nFINAL COMPARISON (mean +/- std over folds)")
print(table.to_string(index=False))
table.to_csv(OUT / "final_comparison.csv", index=False)

print("\nPAIRED COMPARISON vs FedAvg (natural), matched by seed and fold")
base = groups["FedAvg (natural)"]
for label, t in groups.items():
    if label == "FedAvg (natural)":
        continue
    m = base.merge(t, on=["seed", "fold"], suffixes=("_b", "_o"))
    for k in ["auc", "accuracy"]:
        d = m[k + "_o"] - m[k + "_b"]
        p = stats.ttest_rel(m[k + "_o"], m[k + "_b"]).pvalue if d.std() > 0 else float("nan")
        print(f"{label:20s} {k:9s} mean diff {d.mean():+.3f}  paired t-test p={p:.3f}  n={len(d)}")
print("(with 5 folds per seed these tests have very little power; treat p-values as rough)")

dp_sweep = [(l, eps_of[l]) for l in eps_of if np.isfinite(eps_of[l])]
dp_sweep.sort(key=lambda x: x[1])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, k, title in [(axes[0], "auc", "AUC"), (axes[1], "accuracy", "Accuracy")]:
    xs = [e for _, e in dp_sweep]
    ys = [groups[l][k].mean() for l, _ in dp_sweep]
    es = [groups[l][k].std(ddof=1) for l, _ in dp_sweep]
    ax.errorbar(xs, ys, yerr=es, fmt="o-", capsize=3, label="DP-FedAvg")
    for (l, e), y in zip(dp_sweep, ys):
        ax.annotate(l.split("=")[1], (e, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.axhline(base[k].mean(), color="green", ls="--", label="FedAvg, no DP")
    ax.axhline(0.5 if k == "auc" else base["majority_baseline"].mean(),
               color="gray", ls=":", label="chance" if k == "auc" else "majority baseline")
    ax.set_xscale("log")
    ax.set_xlabel("ε (log scale, lower = more private)")
    ax.set_ylabel(title)
    ax.set_title(f"{title} vs privacy budget (labels = noise multiplier z)")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "privacy_utility.png", dpi=200)
plt.close()

plt.figure(figsize=(5.5, 5))
curves = [("Centralized", cen_p), ("FedAvg (natural)", fed_p)]
for z in [1.0, 2.0]:
    curves.append((f"DP-FedAvg z={z:g}", dp_p[np.isclose(dp_p["noise_multiplier"], z)]))
for label, p in curves:
    if len(p) == 0:
        continue
    fpr, tpr, _ = roc_curve(p["y_true"], p["prob"])
    plt.plot(fpr, tpr, label=f"{label} (pooled AUC {roc_auc_score(p['y_true'], p['prob']):.2f})")
plt.plot([0, 1], [0, 1], "k:", label="chance")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC on held-out participants (pooled over folds)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "roc_curves.png", dpi=200)
plt.close()

print("\nSaved to", OUT.resolve())