"""MNIST 7x7 budget sweep (60 outer epochs) under the drift-neutral operator
(Appendix F).

Same protocol as the density-agnostic 60-epoch sweep of Table 6 (budget_sweep_mnist.run_one
with N_EPOCH = 60: POP = 30, ELITISM = 4, RESEED = 6, alpha = 0, rho0 = 0.10,
3 seeds), with the mutation operator replaced by the drift-neutral one
(swap / remove / add = 0.40 / 0.30 / 0.30, the operator of the drift-neutral
runs of operator_drift_400ep.py).

Default budgets are the seven not covered by the 400-epoch drift-neutral runs
(which give B = 512, 2048, 8192 when read at epoch 60).

Reported per run: rho_pop_avg_last5 = population-mean density averaged over
epochs 56-60, and the held-out accuracy of the top-ranked individual at
epoch 60 (the outer loop selects on that split).

Output: results/mnist_sweep_drift_neutral_60ep_results.pkl (saved after
every run, resumable).
Env overrides: B_LIST_OVERRIDE ("128,256"), SEEDS_OVERRIDE, N_JOBS_OVERRIDE
(default 8, the setting of the stored runs), N_EPOCH_OVERRIDE.
Usage:  python src/budget_sweep_mnist_drift_neutral.py [--smoke]
"""
import os, pickle, sys, tempfile, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_sweep_mnist as bs
import operator_drift_400ep as od

# Mutation happens in the parent process inside bs.train_evolutive, so
# replacing the module-level operator governs every mutation.
bs.random_modif_quantum = od.symmetric_modif

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE   = "--smoke" in sys.argv
B_LIST  = [128, 256, 1024, 4096, 16384, 32768, 65536]
SEEDS   = [0, 1, 2]
N_EPOCH = 60
N_JOBS  = int(os.environ.get("N_JOBS_OVERRIDE", 8))
if os.environ.get("B_LIST_OVERRIDE"):
    B_LIST = [int(x) for x in os.environ["B_LIST_OVERRIDE"].split(",")]
if os.environ.get("SEEDS_OVERRIDE"):
    SEEDS = [int(x) for x in os.environ["SEEDS_OVERRIDE"].split(",")]
if os.environ.get("N_EPOCH_OVERRIDE"):
    N_EPOCH = int(os.environ["N_EPOCH_OVERRIDE"])
RESULTS = os.path.join(ROOT, "results", "mnist_sweep_drift_neutral_60ep_results.pkl")
if SMOKE:
    B_LIST, SEEDS, N_EPOCH = [64], [0], 3
    RESULTS = os.path.join(tempfile.gettempdir(), "budget_sweep_mnist_drift_neutral_smoke.pkl")


def run_one(seed, B):
    """budget_sweep_mnist.run_one with an explicit epoch count and worker count."""
    bs.set_all_seeds(seed)
    init_pop = [bs.random_ini(density=bs.F_TARGET) for _ in range(bs.POP_SIZE)]
    t0 = time.time()
    best, pop, avg_d, hist = bs.train_evolutive(
        init_pop, n_epoch=N_EPOCH, alpha=bs.ALPHA, B=B, n_jobs=N_JOBS,
        base_seed=seed, init_density=bs.F_TARGET)
    return {"ds": "mnist", "operator": "drift_neutral_0.40_0.30_0.30", "seed": seed, "B": B,
            "N_EPOCH": N_EPOCH, "n_jobs": N_JOBS, "elapsed": time.time() - t0,
            "rho_pop_avg_last5": float(np.mean(avg_d[-5:])),
            "rho_best_final": float(hist["best"]["density"][-1]),
            "acc_best_final": float(hist["best"]["acc_te"][-1]),
            "avg_density_traj": avg_d,
            "best_density_traj": hist["best"]["density"],
            "best_acc_traj": hist["best"]["acc_te"]}


def summarize(records):
    print("\nSummary (mean +/- std over seeds, ddof=0):", flush=True)
    for B in sorted({r["B"] for r in records}):
        v = [r for r in records if r["B"] == B]
        rho = np.array([r["rho_pop_avg_last5"] for r in v]); acc = np.array([r["acc_best_final"] for r in v])
        print(f"  B={B:6d} (2^{int(np.log2(B))})  n={len(v)}  rho(ep 56-60) = {rho.mean():.3f} +/- {rho.std():.3f}"
              f"   best acc = {acc.mean():.2f}", flush=True)


def main():
    print(f"MNIST 60-epoch sweep, drift-neutral operator"
          f"\n  B_LIST={B_LIST} seeds={SEEDS} N_EPOCH={N_EPOCH} n_jobs={N_JOBS}"
          f"\n  results -> {RESULTS if SMOKE else os.path.relpath(RESULTS, ROOT)}", flush=True)
    bs.activate_dataset()
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    records = []
    if os.path.exists(RESULTS) and not SMOKE:
        with open(RESULTS, "rb") as f:
            records = pickle.load(f)["records"]
        print(f"Resumed: {len(records)} records.", flush=True)
    for B in B_LIST:
        for seed in SEEDS:
            if any(r["B"] == B and r["seed"] == seed and r["N_EPOCH"] == N_EPOCH for r in records):
                continue
            r = run_one(seed, B)
            records.append(r)
            print(f"  B={B:6d} seed={seed}  rho(ep 56-60)={r['rho_pop_avg_last5']:.4f}  "
                  f"acc={r['acc_best_final']:.4f}  ({r['elapsed']:.0f}s)", flush=True)
            with open(RESULTS, "wb") as f:
                pickle.dump({"records": records,
                             "config": dict(POP_SIZE=bs.POP_SIZE, ELITISM=bs.ELITISM, RESEED=bs.RESEED,
                                            ALPHA=bs.ALPHA, F_TARGET=bs.F_TARGET, N_EPOCH=N_EPOCH)}, f)
    summarize(records)


if __name__ == "__main__":
    main()
