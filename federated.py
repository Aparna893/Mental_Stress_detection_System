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


def make_clients(train_df, partition, seed):
    """natural = one client per participant; iid = same number of clients, rows shuffled."""
    if partition == "natural":
        return [g for _, g in train_df.groupby("participant")]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(train_df))
    n_clients = train_df["participant"].nunique()
    return [train_df.iloc[p] for p in np.array_split(idx, n_clients)]


def local_train(model, global_w, X, y, epochs, batch_size):
    import tensorflow as tf
    model.set_weights(global_w)
    # reset Adam state in place (no recompile), then restore the learning rate
    opt = model.optimizer
    if getattr(opt, "built", False):
        for v in opt.variables:
            v.assign(tf.zeros_like(v))
        opt.learning_rate.assign(ARCH["lr"])
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=True)
    print(".", end="", flush=True)
    return model.get_weights()


def fedavg(weight_lists, sizes):
    total = float(sum(sizes))
    return [sum(w[i] * (n / total) for w, n in zip(weight_lists, sizes))
            for i in range(len(weight_lists[0]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--partition", choices=["natural", "iid"], default="natural")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED])
    args = ap.parse_args()

    import tensorflow as tf

    df = load_data()
    run_dir = make_run_dir(f"federated_{args.partition}")
    print("Run folder:", run_dir)

    rows, prob_rows, round_logs = [], [], []
    t0 = time.time()

    for seed in args.seeds:
        for fold, train_df, test_df in participant_folds(df, args.folds):
            tf.keras.backend.clear_session()
            set_seed(seed)

            scaler = StandardScaler().fit(train_df[FEATURES])
            X_te = scaler.transform(test_df[FEATURES]).astype("float32")
            y_te = test_df["label"].to_numpy(dtype=int)

            clients = []
            for c in make_clients(train_df, args.partition, seed):
                clients.append((scaler.transform(c[FEATURES]).astype("float32"),
                                c["label"].to_numpy(dtype=int)))
            sizes = [len(y) for _, y in clients]

            global_model = build_model()
            worker = build_model()  # one reusable local model
            gw = global_model.get_weights()

            for r in range(args.rounds):
                local_ws = []
                for X, y in clients:
                    local_ws.append(local_train(worker, gw, X, y,
                                                args.local_epochs, args.batch_size))
                gw = fedavg(local_ws, sizes)
                global_model.set_weights(gw)
                print(f"  fold {fold} round {r + 1}/{args.rounds} "
                      f"done ({time.time() - t0:.0f}s)", flush=True)

                # diagnostic only: NOT used for model selection
                prob_r = global_model.predict(X_te, verbose=0).ravel()
                mr = evaluate(y_te, prob_r)
                round_logs.append({"seed": seed, "fold": fold, "round": r + 1,
                                   **{k: mr[k] for k in ["accuracy", "auc", "mcc", "recall"]}})

            prob = global_model.predict(X_te, verbose=0).ravel()
            m = evaluate(y_te, prob)
            m.update(seed=seed, fold=fold, n_clients=len(clients),
                     n_train=int(sum(sizes)),
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
                  f"pred-stress {m['predicted_stress_fraction']:.2f}, "
                  f"clients {len(clients)})")

    table = pd.DataFrame(rows)
    summary = {k: {"mean": float(table[k].mean()), "std": float(table[k].std(ddof=1))}
               for k in METRICS + ["majority_baseline"]}

    print(f"\nFEDERATED ({args.partition}) MEAN +/- STD over folds")
    for k in METRICS + ["majority_baseline"]:
        print(f"  {k:18s} {summary[k]['mean']:.3f} +/- {summary[k]['std']:.3f}")
    print(f"Total time: {time.time() - t0:.0f}s")

    save_json({"experiment": f"federated_{args.partition}", "args": vars(args),
               "arch": ARCH, "summary": summary, "per_fold": rows,
               "round_logs": round_logs, "seconds": time.time() - t0},
              run_dir / "results.json")
    pd.concat(prob_rows).to_csv(run_dir / "test_probabilities.csv", index=False)
    pd.DataFrame(round_logs).to_csv(run_dir / "round_logs.csv", index=False)
    table.drop(columns=["confusion_matrix", "test_participants"]).to_csv(
        run_dir / "per_fold_metrics.csv", index=False)
    print("Saved to", run_dir)


if __name__ == "__main__":
    main()