"""400-epoch MNIST runs under the two mutation operators (Figure 6, Appendix D).

Long-horizon runs of the MNIST 7x7 pruning loop of budget_sweep_mnist.py
(~970-parameter masked MLP, N = 30, E = 4, R = 6, alpha = 0, rho0 = 0.10,
400 outer epochs, inner CMA-ES at budget B), with one of two mutation
operators:

  density_agnostic  swap / remove / add = 0.40 / 0.20 / 0.40, the operator of
                    the unbiased runs (expected growth +0.2 active entries per
                    flip); B = 512 (seeds 0, 1, 2) and B = 8192 (seeds 0, 1).
                    Output: results/mechanism/expA_pretest_results.pkl
  drift_neutral     swap / remove / add = 0.40 / 0.30 / 0.30 (expected change
                    zero); B = 512, 2048, 8192, seeds 0, 1, 2.
                    Output: results/mechanism/expG_results_g-s<seed>.pkl
                    (one file per seed)

Their selection-off null is selection_off_null.py (the grey band of
Figure 6). Epochs run in blocks of 20 through bs.train_evolutive with the
population carried over, so the dynamics are those of one continuous run;
only the inner-loop seed stream changes per block. Each run saves when it
finishes and is skipped on restart.

Env overrides: OPERATOR (density_agnostic | drift_neutral), B_LIST_OVERRIDE
("512,8192"), SEEDS_OVERRIDE ("0,1"), SEED, N_EPOCH_OVERRIDE, N_JOBS_OVERRIDE.
Usage:  OPERATOR=drift_neutral python src/operator_drift_400ep.py [--smoke]
"""
import os, sys, time, pickle, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_sweep_mnist as bs

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results", "mechanism")
SMOKE       = "--smoke" in sys.argv
OPERATOR    = os.environ.get("OPERATOR", "density_agnostic")
assert OPERATOR in ("density_agnostic", "drift_neutral"), OPERATOR

bs.N_JOBS = int(os.environ.get("N_JOBS_OVERRIDE", 8))
N_EPOCH   = 6 if SMOKE else 400
BLOCK     = 3 if SMOKE else 20
if os.environ.get("N_EPOCH_OVERRIDE"):
    N_EPOCH = int(os.environ["N_EPOCH_OVERRIDE"])

if OPERATOR == "density_agnostic":
    RUNS = [(512, 0), (512, 1), (512, 2), (8192, 0), (8192, 1)]
else:
    RUNS = [(B, s) for s in (0, 1, 2) for B in (512, 2048, 8192)]
if SMOKE:
    RUNS = [(64, 0)]
if any(os.environ.get(k) for k in ("B_LIST_OVERRIDE", "SEEDS_OVERRIDE", "SEED")):
    b_list = sorted({B for B, _ in RUNS})
    seeds = sorted({s for _, s in RUNS})
    if os.environ.get("B_LIST_OVERRIDE"):
        b_list = [int(x) for x in os.environ["B_LIST_OVERRIDE"].split(",")]
    if os.environ.get("SEEDS_OVERRIDE"):
        seeds = [int(x) for x in os.environ["SEEDS_OVERRIDE"].split(",")]
    if os.environ.get("SEED"):
        seeds = [int(os.environ["SEED"])]
    RUNS = [(B, s) for s in seeds for B in b_list]


def symmetric_modif(model, n_flips=None, min_density=bs.MIN_DENSITY):
    """Drift-neutral operator: swap 40 / remove 30 / add 30."""
    model = model.clone_arch()
    total = sum(m.numel() for m in model.masks)
    if n_flips is None:
        n_flips = max(1, int(0.02 * total))
    min_active = max(1, int(min_density * total))
    for _ in range(n_flips):
        active_count = sum(int(m.data.sum().item()) for m in model.masks)
        a = np.random.random()
        if a < 0.4 and 0 < active_count < total:
            removed = bs._flip_one(model, current=1.0, target=0.0)
            if removed is None: continue
            added = bs._flip_one(model, current=0.0, target=1.0)
            if added is None:
                model.masks[removed[0]].data[removed[1]] = 1.0
        elif a < 0.7 and active_count > min_active:
            bs._flip_one(model, current=1.0, target=0.0)
        else:
            if active_count < total:
                bs._flip_one(model, current=0.0, target=1.0)
    return model


# Mutation happens in the parent process inside bs.train_evolutive, so
# replacing the module-level operator governs every mutation; workers only
# run score().
if OPERATOR == "drift_neutral":
    bs.random_modif_quantum = symmetric_modif


def results_path(seed):
    if SMOKE:
        return os.path.join(tempfile.gettempdir(), f"operator_drift_400ep_{OPERATOR}_smoke.pkl")
    if OPERATOR == "density_agnostic":
        return os.path.join(RESULTS_DIR, "expA_pretest_results.pkl")
    return os.path.join(RESULTS_DIR, f"expG_results_g-s{seed}.pkl")


def run_one_chunked(seed, B):
    """bs.run_one in BLOCK-epoch chunks with progress logging."""
    bs.set_all_seeds(seed)
    pop = [bs.random_ini(density=bs.F_TARGET) for _ in range(bs.POP_SIZE)]
    avg_d_all, best_d_all, best_acc_all = [], [], []
    t0 = time.time()
    for blk in range(N_EPOCH // BLOCK):
        best, pop, avg_d, hist = bs.train_evolutive(
            pop, n_epoch=BLOCK, alpha=bs.ALPHA, B=B,
            n_jobs=bs.N_JOBS, base_seed=seed * 100 + blk,
            init_density=bs.F_TARGET)
        avg_d_all.extend(avg_d)
        best_d_all.extend(hist["best"]["density"])
        best_acc_all.extend(hist["best"]["acc_te"])
        ep = (blk + 1) * BLOCK
        print(f"    B={B:5d} seed={seed}  epoch {ep:3d}/{N_EPOCH}  "
              f"avg_rho={avg_d[-1]:.4f}  best_acc={hist['best']['acc_te'][-1]:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    rec = {
        "ds": "mnist" if OPERATOR == "density_agnostic" else "mnist-sym",
        "seed": seed, "B": B, "elapsed": time.time() - t0, "N_EPOCH": N_EPOCH,
        "rho_pop_avg_last5": float(np.mean(avg_d_all[-5:])),
        "rho_pop_avg_last50": float(np.mean(avg_d_all[-50:])),
        "rho_best_final": float(best_d_all[-1]),
        "acc_best_final": float(best_acc_all[-1]),
        "avg_density_traj": avg_d_all,
        "best_density_traj": best_d_all,
        "best_acc_traj": best_acc_all,
    }
    if OPERATOR == "drift_neutral":
        del rec["rho_pop_avg_last5"]
    return rec


def load_records(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f).get("records", [])
    return []


def save_records(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    config = dict(POP_SIZE=bs.POP_SIZE, ELITISM=bs.ELITISM, RESEED=bs.RESEED)
    if OPERATOR == "density_agnostic":
        payload = {"records": records, "runs": RUNS, "N_EPOCH": N_EPOCH,
                   "config": dict(config, ALPHA=bs.ALPHA, F_TARGET=bs.F_TARGET, BLOCK=BLOCK)}
    else:
        payload = {"records": records, "mutation": "symmetric_40_30_30",
                   "config": dict(config, N_EPOCH=N_EPOCH)}
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def summarize(records):
    print(f"\nSummary ({OPERATOR}; population-mean density):", flush=True)
    for B in sorted({r["B"] for r in records}):
        rs = [r for r in records if r["B"] == B]
        last50 = [r["rho_pop_avg_last50"] for r in rs]
        final = [r["avg_density_traj"][-1] for r in rs]
        print(f"  B={B:5d}  n={len(rs)}  last 50 epochs = {np.mean(last50):.4f} "
              f"+/- {np.std(last50):.4f}   final epoch = {np.mean(final):.4f} "
              f"+/- {np.std(final):.4f}", flush=True)


def main():
    print(f"400-epoch MNIST runs, {OPERATOR} operator"
          f"\n  runs (B, seed): {RUNS}"
          f"\n  N_EPOCH={N_EPOCH} (blocks of {BLOCK}), POP={bs.POP_SIZE}, "
          f"n_jobs={bs.N_JOBS}, smoke={SMOKE}", flush=True)
    bs.activate_dataset()
    all_records = []
    for idx, (B, seed) in enumerate(RUNS, 1):
        path = results_path(seed)
        records = load_records(path)
        if any(r["B"] == B and r["seed"] == seed for r in records):
            print(f"[{idx}/{len(RUNS)}] B={B} seed={seed}  (done — skipped)", flush=True)
        else:
            print(f"[{idx}/{len(RUNS)}] B={B} seed={seed}  starting ...", flush=True)
            r = run_one_chunked(seed, B)
            records.append(r)
            save_records(path, records)
            print(f"[{idx}/{len(RUNS)}] B={B:5d} seed={seed}  DONE  "
                  f"rho_last50={r['rho_pop_avg_last50']:.4f}  "
                  f"acc={r['acc_best_final']:.4f}  ({r['elapsed']:.0f}s)  -> "
                  f"{path if SMOKE else os.path.relpath(path, ROOT)}", flush=True)
        all_records += [r for r in records if r["B"] == B and r["seed"] == seed]
    summarize(all_records)


if __name__ == "__main__":
    main()
