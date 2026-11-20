import argparse
import math
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import (FEATURES, ARCH, SEED, load_data, participant_folds,
                    build_model, evaluate, make_run_dir, save_json)
from federated import make_clients, local_train

METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "auc"]
ORDERS = np.concatenate([np.arange(1.1, 10.0, 0.1),
                         np.arange(10, 64),
                         np.arange(64, 129, 2)]).astype(float)


def set_seed(seed):
    import tensorflow as tf
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def epsilon(z, rounds, delta):
    """RDP composition of `rounds` Gaussian mechanisms, converted to (eps, delta)."""
    if z <= 0:
        return float("inf")
    eps = rounds * ORDERS / (2.0 * z * z) + math.log(1.0 / delta) / (ORDERS - 1.0)
    return float(eps.min())


def clip_update(local_w, global_w, bound):
    upd = [l.astype(np.float64) - g.astype(np.float64)
           for l, g in zip(local_w, global_w)]
    norm = math.sqrt(sum(float(np.sum(u ** 2)) for u in upd))
    scale = min(1.0, bound / (norm + 1e-12))
    return [u * scale for u in upd], norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, nargs="+",
                    default=[0.0, 0.1, 0.3, 1.0, 2.0])
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED])
    args = ap.parse_args()

    import tensorflow as tf

    df = load_data()
    run_dir = make_run_dir("dp_federated")
    print("Run folder:", run_dir)

    all_rows, prob_rows, sweep = [], [], []
    t0 = time.time()

    for z in args.noise:
        eps = epsilon(z, args.rounds, args.delta)
        eps_txt = "inf (clipping only)" if math.isinf(eps) else f"{eps:.2f}"
        print(f"\n=== noise multiplier {z} | epsilon = {eps_txt} "
              f"(delta={args.delta}, {args.rounds} rounds) ===", flush=True)
        rows = []

        for seed in args.seeds:
            for fold, train_df, test_df in participant_folds(df, args.folds):
                tf.keras.backend.clear_session()
                set_seed(seed)
                rng = np.random.RandomState(seed * 1000 + fold)

                scaler = StandardScaler().fit(train_df[FEATURES])
                X_te = scaler.transform(test_df[FEATURES]).astype("float32")
                y_te = test_df["label"].to_numpy(dtype=int)

                clients = []
                for c in make_clients(train_df, "natural", seed):
                    clients.append((scaler.transform(c[FEATURES]).astype("float32"),
                                    c["label"].to_numpy(dtype=int)))
                sizes = np.array([len(y) for _, y in clients], dtype=float)
                weights = sizes / sizes.sum()

                # replace-one sensitivity of the weighted aggregate
                sensitivity = 2.0 * args.clip * float(weights.max())
                noise_std = z * sensitivity

                global_model = build_model()
                worker = build_model()
                gw = [w.astype(np.float64) for w in global_model.get_weights()]
                clipped_fracs = []

                for r in range(args.rounds):
                    agg = [np.zeros_like(w) for w in gw]
                    n_clipped = 0
                    gw32 = [w.astype(np.float32) for w in gw]
                    for (X, y), wt in zip(clients, weights):
                        lw = local_train(worker, gw32, X, y,
                                         args.local_epochs, args.batch_size)
                        upd, norm = clip_update(lw, gw, args.clip)
                        n_clipped += int(norm > args.clip)
                        for j, u in enumerate(upd):
                            agg[j] += wt * u
                    if noise_std > 0:
                        agg = [a + rng.normal(0.0, noise_std, size=a.shape)
                               for a in agg]
                    gw = [w + a for w, a in zip(gw, agg)]
                    clipped_fracs.append(n_clipped / len(clients))
                    print(f"  z={z} fold {fold} round {r + 1}/{args.rounds} "
                          f"done ({time.time() - t0:.0f}s)", flush=True)

                global_model.set_weights([w.astype(np.float32) for w in gw])
                prob = global_model.predict(X_te, verbose=0).ravel()
                prob = np.nan_to_num(prob, nan=0.5)

                m = evaluate(y_te, prob)
                m.update(noise_multiplier=z, seed=seed, fold=fold,
                         epsilon=None if math.isinf(eps) else eps,
                         clip=args.clip, noise_std=noise_std,
                         mean_clipped_fraction=float(np.mean(clipped_fracs)))
                rows.append(m)
                all_rows.append(m)
                prob_rows.append(pd.DataFrame({
                    "noise_multiplier": z, "seed": seed, "fold": fold,
                    "participant": test_df["participant"].to_numpy(),
                    "y_true": y_te, "prob": prob}))

                print(f"z={z} fold {fold}: acc={m['accuracy']:.3f} "
                      f"auc={m['auc']:.3f} mcc={m['mcc']:.3f} "
                      f"(baseline {m['majority_baseline']:.3f}, "
                      f"pred-stress {m['predicted_stress_fraction']:.2f}, "
                      f"clipped {m['mean_clipped_fraction']:.2f})", flush=True)

        t = pd.DataFrame(rows)
        sweep.append({
            "noise_multiplier": z,
            "epsilon": None if math.isinf(eps) else eps,
            **{f"{k}_mean": float(t[k].mean()) for k in METRICS},
            **{f"{k}_std": float(t[k].std(ddof=1)) for k in METRICS},
            "pred_stress_mean": float(t["predicted_stress_fraction"].mean()),
            "clipped_fraction_mean": float(t["mean_clipped_fraction"].mean()),
        })

    summary = pd.DataFrame(sweep)
    print("\nPRIVACY-UTILITY SWEEP (mean over folds)")
    show = summary[["noise_multiplier", "epsilon", "accuracy_mean", "auc_mean",
                    "mcc_mean", "pred_stress_mean", "clipped_fraction_mean"]]
    print(show.round(3).to_string(index=False))
    print(f"Total time: {time.time() - t0:.0f}s")

    save_json({"experiment": "dp_federated", "args": vars(args), "arch": ARCH,
               "sweep": sweep, "per_fold": all_rows,
               "seconds": time.time() - t0}, run_dir / "results.json")
    summary.to_csv(run_dir / "sweep_summary.csv", index=False)
    pd.concat(prob_rows).to_csv(run_dir / "test_probabilities.csv", index=False)
    print("Saved to", run_dir)


if __name__ == "__main__":
    main()