"""Estimator variance against density and budget on the teacher–student task
(Figure 4(b); the variance-collapse budgets drawn in Figure 3; the 2,700-run
grid of Section 4.1).

Measures the quantity Proposition 1 assumes, in the environment of
teacher_student.py: the spread of the best fitness that one inner CMA-ES run
finds from a cold start (what a reseeded individual sees), as a function of
the active dimension and the budget. There is no outer loop.

Grid: rho in {0.05, 0.10, 0.20, 0.40, 0.70, 1.00}
      x B in {2^7 .. 2^15}
      x 50 repetitions (fresh random mask and fresh weights each repetition).

Per (rho, B) cell: mean and variance of the best noisy fitness (what
selection compares) and of the clean MSE (one noiseless evaluation).

Output: results/mechanism/expI_results_i.pkl.
Env overrides: N_JOBS_OVERRIDE, REPS_OVERRIDE.
Usage:  python src/variance_grid.py [--smoke]
"""
import os, sys, time, pickle, tempfile, warnings
import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_sweep_mnist as bs
import teacher_student as ts     # teacher, data, noisy-MSE inner loop

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE  = "--smoke" in sys.argv
N_JOBS = int(os.environ.get("N_JOBS_OVERRIDE", 14))
RHOS   = [0.1, 1.0] if SMOKE else [0.05, 0.10, 0.20, 0.40, 0.70, 1.00]
B_LIST = [64] if SMOKE else [2**k for k in range(7, 16)]     # 128 .. 32768
REPS   = 3 if SMOKE else 50
if os.environ.get("REPS_OVERRIDE"):
    REPS = int(os.environ["REPS_OVERRIDE"])
RESULTS = (os.path.join(tempfile.gettempdir(), "variance_grid_smoke.pkl") if SMOKE
           else os.path.join(ROOT, "results", "mechanism", "expI_results_i.pkl"))


def one_rep(rho, B, rep):
    warnings.filterwarnings("ignore")
    seed = int(1e6 * rho) + 31 * B + rep
    bs.set_all_seeds(seed)
    model = bs.random_ini(density=rho)              # cold start
    model = ts.train_inner_cmaes_mse(model, B, seed)
    rng = np.random.default_rng(seed + 777)
    with torch.no_grad():
        noise = torch.from_numpy(
            rng.standard_normal(ts.Y_VA.shape).astype(np.float32)) * ts.NOISE_STD
        noisy = F.mse_loss(model(ts.X_VA), ts.Y_VA + noise).item()
        clean = F.mse_loss(model(ts.X_VA), ts.Y_VA).item()
    dims = int(sum(int(m.data.sum().item()) for m in model.masks))
    return {"rho": rho, "B": B, "rep": rep, "dims": dims,
            "noisy_fit": float(-noisy), "clean_mse": float(clean)}


def main():
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    print(f"Variance grid (teacher-student)"
          f"\n  rhos={RHOS}  B={B_LIST}  reps={REPS}  n_jobs={N_JOBS}"
          f"\n  results -> {RESULTS if SMOKE else os.path.relpath(RESULTS, ROOT)}", flush=True)
    tasks = [(r, B, k) for r in RHOS for B in B_LIST for k in range(REPS)]
    print(f"{len(tasks)} inner runs total", flush=True)
    t0 = time.time()
    recs = Parallel(n_jobs=N_JOBS)(
        delayed(one_rep)(r, B, k) for (r, B, k) in tasks)
    print(f"All reps done in {time.time()-t0:.0f}s", flush=True)
    print("\nVar[f_hat] table (noisy fitness; rows rho, cols B):", flush=True)
    header = "  rho\\B  " + "  ".join(f"{B:>8d}" for B in B_LIST)
    print(header, flush=True)
    cells = {}
    for r in RHOS:
        row = []
        for B in B_LIST:
            xs = [x["noisy_fit"] for x in recs
                  if abs(x["rho"] - r) < 1e-9 and x["B"] == B]
            cs = [x["clean_mse"] for x in recs
                  if abs(x["rho"] - r) < 1e-9 and x["B"] == B]
            cells[(r, B)] = {"n": len(xs),
                             "mean_fit": float(np.mean(xs)),
                             "var_fit": float(np.var(xs)),
                             "mean_clean": float(np.mean(cs)),
                             "var_clean": float(np.var(cs))}
            row.append(f"{np.var(xs):8.2e}")
        print(f"  {r:5.2f}  " + "  ".join(row), flush=True)
    with open(RESULTS, "wb") as f:
        pickle.dump({"records": recs, "cells": cells,
                     "config": dict(RHOS=RHOS, B_LIST=B_LIST, REPS=REPS,
                                    NOISE_STD=ts.NOISE_STD,
                                    RHO_TEACHER=ts.RHO_TEACHER)}, f)
    print(f"Saved {RESULTS if SMOKE else os.path.relpath(RESULTS, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
