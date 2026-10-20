"""Population-size x elitism grid on the teacher-student benchmark (Appendix D).

teacher_student.run_one, unchanged, at a fixed budget with the
population size N and the number of elites E varied:

  N in {15, 30, 60}, E in {2, 4, 8}, reseeds R = 0.2 N (3, 6, 12),
  B = 4096, rho0 = 0.10, drift-neutral operator, 3 seeds  ->  27 runs.

Everything else (teacher, noise, inner CMA-ES, plateau criterion, 300-epoch
cap) is as in the main sweep. As in the main sweep, the R mutated copies are
made from the top R/2 ranks; these extend beyond the elite set when E < R/2.

The cell N = 30, E = 4, R = 6 is the configuration of the main sweep. If the
stored runs of that sweep (results/mechanism/expE_results_e10-s*.pkl) are
present, they are reused for seeds 0-2 rather than recomputed (set
REUSE_MAIN_SWEEP=0 to recompute them).

For reference the selection-off null of every cell (same outer loop with a
random ranking and no inner loop) is computed as well; it takes seconds.

Output: results/mechanism/grid_population_elitism_results.pkl (saved after
every run, resumable).
Env overrides: N_LIST_OVERRIDE, E_LIST_OVERRIDE, SEEDS_OVERRIDE, B_OVERRIDE,
N_JOBS_OVERRIDE (default 8, the setting of the stored runs), N_EPOCH_OVERRIDE,
REUSE_MAIN_SWEEP.
Usage:  python src/population_elitism_grid.py [--smoke]
"""
import glob, os, pickle, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("N_JOBS_OVERRIDE", "8")
import budget_sweep_mnist as bs
import teacher_student as ts

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = "--smoke" in sys.argv

N_LIST, E_LIST, SEEDS = [15, 30, 60], [2, 4, 8], [0, 1, 2]
B, RHO_INIT = 4096, 0.10
if os.environ.get("N_LIST_OVERRIDE"):
    N_LIST = [int(x) for x in os.environ["N_LIST_OVERRIDE"].split(",")]
if os.environ.get("E_LIST_OVERRIDE"):
    E_LIST = [int(x) for x in os.environ["E_LIST_OVERRIDE"].split(",")]
if os.environ.get("SEEDS_OVERRIDE"):
    SEEDS = [int(x) for x in os.environ["SEEDS_OVERRIDE"].split(",")]
if os.environ.get("B_OVERRIDE"):
    B = int(os.environ["B_OVERRIDE"])
REUSE = os.environ.get("REUSE_MAIN_SWEEP", "1") == "1"
RESULTS = os.path.join(ROOT, "results", "mechanism", "grid_population_elitism_results.pkl")
NULL_EPOCHS = 120
if SMOKE:          # teacher_student.py has already shortened its own run (6 epochs)
    N_LIST, E_LIST, SEEDS, B, REUSE, NULL_EPOCHS = [15], [2], [0], 64, False, 6
    RESULTS = os.path.join(tempfile.gettempdir(), "population_elitism_grid_smoke.pkl")


def reseeds(N):
    return int(round(0.2 * N))


def set_cell(N, E):
    ts.POP_SIZE, ts.ELITISM, ts.RESEED = N, E, reseeds(N)


def null_run(N, E, seed, n_epoch=NULL_EPOCHS):
    """Outer loop of ts.run_one with a random ranking and no inner loop."""
    R = reseeds(N)
    bs.set_all_seeds(seed)
    circuits = [bs.random_ini(density=RHO_INIT) for _ in range(N)]
    avg = []
    for _ in range(n_epoch):
        order = [int(i) for i in np.random.permutation(N)]
        lo, hi = E, N - R
        picks = set(np.random.choice(range(lo, hi), size=R, replace=False).tolist())
        for i in range(1, R + 1):
            circuits[order[-i]] = bs.random_ini(density=RHO_INIT)
        for pr, i in zip([k // 2 for k in range(R)], sorted(picks)):
            circuits[order[i]] = ts.symmetric_modif(circuits[order[pr]])
        for i in range(lo, hi):
            if i not in picks:
                circuits[order[i]] = ts.symmetric_modif(circuits[order[i]])
        avg.append(float(np.mean([m.density() for m in circuits])))
    return float(np.mean(avg[-40:]))


def main_sweep_records():
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "mechanism", "expE_results_e10-s*.pkl"))):
        for r in pickle.load(open(p, "rb"))["records"]:
            if r["B"] == B and abs(r["rho_init"] - RHO_INIT) < 1e-9 and r["seed"] in SEEDS:
                out.append(dict(r, N=bs.POP_SIZE, E=bs.ELITISM, R=bs.RESEED, source="main sweep (stored)"))
    return out


def summarize(records, nulls):
    print(f"\nrho* at B={B}, rho0={RHO_INIT} (mean +/- std over seeds, ddof=0); "
          f"selection-off null in brackets:", flush=True)
    print("   N \\ E " + "".join(f"{E:>26d}" for E in E_LIST), flush=True)
    for N in N_LIST:
        row = []
        for E in E_LIST:
            v = [r["rho_eq_last40"] for r in records if r["N"] == N and r["E"] == E]
            nv = nulls.get((N, E))
            row.append(f"{np.mean(v):.3f} +/- {np.std(v):.3f} (n={len(v)}) [{nv:.3f}]" if v and nv is not None
                       else f"{'-':>26s}")
        print(f"   {N:3d}    " + "  ".join(row), flush=True)


def main():
    print(f"Population x elitism grid: N={N_LIST} E={E_LIST} seeds={SEEDS} B={B} rho0={RHO_INIT} "
          f"n_jobs={ts.N_JOBS}\n  results -> {RESULTS if SMOKE else os.path.relpath(RESULTS, ROOT)}", flush=True)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    records, nulls = [], {}
    if os.path.exists(RESULTS) and not SMOKE:
        with open(RESULTS, "rb") as f:
            d = pickle.load(f)
        records, nulls = d["records"], d.get("nulls", {})
        print(f"Resumed: {len(records)} records.", flush=True)

    def save():
        with open(RESULTS, "wb") as f:
            pickle.dump({"records": records, "nulls": nulls,
                         "config": dict(B=B, RHO_INIT=RHO_INIT, N_LIST=N_LIST, E_LIST=E_LIST,
                                        RESEED_RULE="R = round(0.2 N)", NOISE_STD=ts.NOISE_STD,
                                        PLATEAU_SLOPE=ts.PLATEAU_SLOPE, N_EPOCH_CAP=ts.N_EPOCH)}, f)

    done = lambda N, E, s: any(r["N"] == N and r["E"] == E and r["seed"] == s for r in records)
    if REUSE and bs.POP_SIZE in N_LIST and bs.ELITISM in E_LIST and reseeds(bs.POP_SIZE) == bs.RESEED:
        for r in main_sweep_records():
            if not done(r["N"], r["E"], r["seed"]):
                records.append(r)
                print(f"  reused stored main-sweep run N={r['N']} E={r['E']} seed={r['seed']}: "
                      f"rho*={r['rho_eq_last40']:.4f}", flush=True)
    for N in N_LIST:
        for E in E_LIST:
            if (N, E) not in nulls:
                nulls[(N, E)] = float(np.mean([null_run(N, E, s) for s in SEEDS]))
    save()
    for N in N_LIST:                      # cheapest cells first
        for E in E_LIST:
            for seed in SEEDS:
                if done(N, E, seed):
                    continue
                set_cell(N, E)
                r = ts.run_one(seed, B, RHO_INIT)
                r.update(N=N, E=E, R=reseeds(N), source="this script")
                records.append(r)
                print(f"  N={N:2d} E={E} R={reseeds(N):2d} seed={seed}  rho*={r['rho_eq_last40']:.4f}  "
                      f"plateaued={r['plateaued']} (ep {r['epochs_run']})  "
                      f"cleanMSE={r['best_clean_mse_traj'][-1]:.5f}  ({r['elapsed']:.0f}s)", flush=True)
                save()
    summarize(records, nulls)


if __name__ == "__main__":
    main()
