import argparse
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import (FEATURES, ARCH, SEED, load_data, participant_folds,
                    build_model, evaluate, make_run_dir, save_json)

METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "auc"]


def set_seed(seed):
    import tensorflow as tf
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED])
    args = ap.parse_args()

    df = load_data()
    run_dir = make_run_dir("centralized")
    print("Run folder:", run_dir)

    rows, prob_rows = [], []
    t0 = time.time()

    for seed in args.seeds:
        for fold, train_df, test_df in participant_folds(df, args.folds):
            set_seed(seed)

            scaler = StandardScaler().fit(train_df[FEATURES])
            X_tr = scaler.transform(train_df[FEATURES]).astype("float32")
            X_te = scaler.transform(test_df[FEATURES]).astype("float32")
            y_tr = train_df["label"].to_numpy(dtype=int)
            y_te = test_df["label"].to_numpy(dtype=int)

            model = build_model()
            model.fit(X_tr, y_tr, epochs=args.epochs,
                      batch_size=args.batch_size, verbose=0, shuffle=True)
            prob = model.predict(X_te, verbose=0).ravel()

            m = evaluate(y_te, prob)
            m.update(seed=seed, fold=fold,
                     n_train=len(train_df),
                     test_participants=sorted(test_df["participant"].unique()))
            rows.append(m)

            prob_rows.append(pd.DataFrame({
                "seed": seed, "fold": fold,
                "participant": test_df["participant"].to_numpy(),
                "y_true": y_te, "prob": prob}))

            print(f"seed {seed} fold {fold}: "
                  f"acc={m['accuracy']:.3f} auc={m['auc']:.3f} "
                  f"mcc={m['mcc']:.3f} recall={m['recall']:.3f} "
                  f"(baseline {m['majority_baseline']:.3f}, "
                  f"pred-stress {m['predicted_stress_fraction']:.2f})")

    table = pd.DataFrame(rows)
    summary = {k: {"mean": float(table[k].mean()), "std": float(table[k].std(ddof=1))}
               for k in METRICS + ["majority_baseline"]}

    print("\nMEAN +/- STD over folds" + (" and seeds" if len(args.seeds) > 1 else ""))
    for k in METRICS + ["majority_baseline"]:
        print(f"  {k:18s} {summary[k]['mean']:.3f} +/- {summary[k]['std']:.3f}")
    print(f"Total time: {time.time() - t0:.0f}s")

    save_json({"experiment": "centralized", "args": vars(args), "arch": ARCH,
               "summary": summary, "per_fold": rows,
               "seconds": time.time() - t0},
              run_dir / "results.json")
    pd.concat(prob_rows).to_csv(run_dir / "test_probabilities.csv", index=False)
    table.drop(columns=["confusion_matrix", "test_participants"]).to_csv(
        run_dir / "per_fold_metrics.csv", index=False)
    print("Saved to", run_dir)


if __name__ == "__main__":
    main()