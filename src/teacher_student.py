"""Teacher–student budget sweep (Section 4.1, Figure 3, Figure 4(a), Appendix B).

Proposition 1 assumes that competing topologies share the same noiseless
optimum: extra dimensions add estimation variance but no fitness. On UCI and
MNIST extra weights also buy accuracy, so this benchmark builds the regime
the proposition describes:

  - Teacher: a sparse MaskedMLP with the 49 -> 16 -> 10 architecture of the
    MNIST experiments, PyTorch default initialisation (never trained), a
    random 10% mask (95 of 970 parameters active), frozen (seed 1234).
  - Student fitness: negative MSE to the teacher's outputs on random inputs
    with fresh Gaussian noise (sigma = 0.10) at every evaluation, so the
    fitness never saturates. 2,048 inputs for the inner loop, a separate
    1,024 for the selection score and the clean error.
  - Any active parameter beyond the teacher's support adds search dimensions
    but no attainable fitness.
  - Mutation: the drift-neutral operator (swap / remove / add =
    0.40 / 0.30 / 0.30), so that operator drift cannot stand in for the
    budget effect (selection_off_null.py measures its null).

Outer loop as in budget_sweep_mnist.py (N = 30, E = 4, R = 6, alpha = 0).
After at least 120 epochs a run stops when the least-squares slope of the
population-mean density over the last 40 epochs is below the plateau
threshold in absolute value; rho* is the population-mean density averaged
over those 40 epochs.

Run sets (RUN_SET, one seed per invocation via SEED; output
results/mechanism/expE_results_<run set>-s<seed>.pkl):
  e10    rho0 = 0.10, B = 2^7 .. 2^14, threshold 2e-4, cap 300 epochs
  e10x   rho0 = 0.10, B = 2^15,        threshold 2e-4, cap 300 epochs
  e30    rho0 = 0.30, B = 2^7 .. 2^14, threshold 2e-4, cap 300 epochs
  e30x   rho0 = 0.30, B = 2^10 .. 2^12, threshold 5e-5, cap 600 epochs
         (the strict-criterion re-runs)
Without RUN_SET the e10 settings run for all five seeds into
expE_results_local.pkl.

Env overrides: RUN_SET, SEED, B_LIST_OVERRIDE ("128,256"), SEEDS_OVERRIDE
("0,1"), RHO_INIT_OVERRIDE, N_EPOCH_OVERRIDE, PLATEAU_SLOPE_OVERRIDE,
N_JOBS_OVERRIDE.
Usage:  python src/teacher_student.py [--smoke]
"""
import os, sys, time, pickle, tempfile, warnings
import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_sweep_mnist as bs

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results", "mechanism")

SMOKE        = "--smoke" in sys.argv
N_JOBS       = int(os.environ.get("N_JOBS_OVERRIDE", 14))
TEACHER_SEED = 1234
RHO_TEACHER  = 0.10
NOISE_STD    = 0.10
N_TRAIN, N_VAL = 2048, 1024
POP_SIZE, ELITISM, RESEED = bs.POP_SIZE, bs.ELITISM, bs.RESEED

RUN_SETS = {
    "e10":  dict(B_LIST=[2**k for k in range(7, 15)], RHO_INIT=0.10, N_EPOCH=300, PLATEAU_SLOPE=2e-4),
    "e10x": dict(B_LIST=[2**15],                       RHO_INIT=0.10, N_EPOCH=300, PLATEAU_SLOPE=2e-4),
    "e30":  dict(B_LIST=[2**k for k in range(7, 15)], RHO_INIT=0.30, N_EPOCH=300, PLATEAU_SLOPE=2e-4),
    "e30x": dict(B_LIST=[2**10, 2**11, 2**12],        RHO_INIT=0.30, N_EPOCH=600, PLATEAU_SLOPE=5e-5),
}
RUN_SET = os.environ.get("RUN_SET", "e10")
assert RUN_SET in RUN_SETS, f"RUN_SET must be one of {sorted(RUN_SETS)}"
_cfg = RUN_SETS[RUN_SET]

B_LIST   = [64] if SMOKE else _cfg["B_LIST"]
SEEDS    = [0] if SMOKE else [0, 1, 2, 3, 4]
RHO_INIT = _cfg["RHO_INIT"]
N_EPOCH  = 6 if SMOKE else _cfg["N_EPOCH"]
PLATEAU_MIN_EPOCHS = 120
PLATEAU_SLOPE      = float(os.environ.get("PLATEAU_SLOPE_OVERRIDE", _cfg["PLATEAU_SLOPE"]))
if os.environ.get("B_LIST_OVERRIDE"):
    B_LIST = [int(x) for x in os.environ["B_LIST_OVERRIDE"].split(",")]
if os.environ.get("SEEDS_OVERRIDE"):
    SEEDS = [int(x) for x in os.environ["SEEDS_OVERRIDE"].split(",")]
if os.environ.get("SEED"):
    SEEDS = [int(os.environ["SEED"])]
if os.environ.get("RHO_INIT_OVERRIDE"):
    RHO_INIT = float(os.environ["RHO_INIT_OVERRIDE"])
if os.environ.get("N_EPOCH_OVERRIDE"):
    N_EPOCH = int(os.environ["N_EPOCH_OVERRIDE"])
if SMOKE:
    RESULTS = os.path.join(tempfile.gettempdir(), "teacher_student_smoke.pkl")
elif "RUN_SET" in os.environ and len(SEEDS) == 1:
    RESULTS = os.path.join(RESULTS_DIR, f"expE_results_{RUN_SET}-s{SEEDS[0]}.pkl")
else:
    RESULTS = os.path.join(RESULTS_DIR, "expE_results_local.pkl")

# ---------------- Teacher and data (identical across all runs) -------------

def build_teacher():
    g = torch.Generator().manual_seed(TEACHER_SEED)
    torch.manual_seed(TEACHER_SEED)          # teacher weights via default init
    teacher = bs.MaskedMLP()
    for m in teacher.masks:
        m.data = (torch.rand(m.shape, generator=g) < RHO_TEACHER).float()
    X_tr = torch.randn(N_TRAIN, bs.IN_DIM, generator=g)
    X_va = torch.randn(N_VAL, bs.IN_DIM, generator=g)
    with torch.no_grad():
        y_tr = teacher(X_tr).detach()
        y_va = teacher(X_va).detach()
    return teacher, X_tr, y_tr, X_va, y_va

TEACHER, X_TR, Y_TR, X_VA, Y_VA = build_teacher()

# ---------------- Drift-neutral mutation (swap 40 / remove 30 / add 30) ----

def symmetric_modif(model, n_flips=None, min_density=bs.MIN_DENSITY):
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

# ---------------- Noisy-MSE inner loop and scoring -------------------------

def train_inner_cmaes_mse(model, budget, seed):
    import cma
    init_vec, layout = bs._collect_active_params(model)
    if len(init_vec) == 0:
        return model
    rng = np.random.default_rng(seed)
    def loss_fn(x):
        bs._apply_vec(model, x, layout)
        with torch.no_grad():
            noise = torch.from_numpy(
                rng.standard_normal(Y_TR.shape).astype(np.float32)) * NOISE_STD
            v = F.mse_loss(model(X_TR), Y_TR + noise).item()
        return float(v) if not np.isnan(v) else 1e6
    opts = {"maxfevals": budget, "verbose": -9, "seed": int(seed) % (2**31),
            "tolx": 1e-8, "tolfun": 1e-8}
    es = cma.CMAEvolutionStrategy(init_vec.tolist(), bs.INNER_SIGMA0, opts)
    es.optimize(loss_fn)
    best_x = es.result.xbest if es.result.xbest is not None else init_vec
    bs._apply_vec(model, best_x, layout)
    return model


def score_ts(model, B, seed):
    warnings.filterwarnings("ignore")
    model = train_inner_cmaes_mse(model, B, seed)
    rng = np.random.default_rng(seed + 777)
    with torch.no_grad():
        noise = torch.from_numpy(
            rng.standard_normal(Y_VA.shape).astype(np.float32)) * NOISE_STD
        noisy_val = F.mse_loss(model(X_VA), Y_VA + noise).item()
        clean_val = F.mse_loss(model(X_VA), Y_VA).item()
    n_active = int(sum(int(m.data.sum().item()) for m in model.masks))
    s = -noisy_val
    if np.isnan(s): s = -1e6
    return s, clean_val, n_active, model

# ---------------- Outer loop with plateau detection ------------------------

def run_one(seed, B, rho_init):
    bs.set_all_seeds(seed)
    circuits = [bs.random_ini(density=rho_init) for _ in range(POP_SIZE)]
    avg_d, best_clean, pop_records = [], [], []
    t0 = time.time()
    n_epochs_run = 0
    for epoch in range(N_EPOCH):
        evaluated = Parallel(n_jobs=N_JOBS)(
            delayed(score_ts)(m, B, seed * 100_000 + epoch * POP_SIZE + i)
            for i, m in enumerate(circuits))
        scored = []
        for i, (s, clean, n_act, trained) in enumerate(evaluated):
            scored.append((s, i, clean, n_act))
            circuits[i] = trained
        scored.sort(reverse=True)
        best_clean.append(scored[0][2])
        pop_records.append([(t[3], float(t[0])) for t in scored])
        max_s = scored[0][0]
        lo, hi = ELITISM, POP_SIZE - RESEED
        weights = np.array([max_s - t[0] + 1e-6 for t in scored[lo:hi]])
        probs = weights / weights.sum()
        picks = set(np.random.choice(range(lo, hi), size=RESEED,
                                     replace=False, p=probs).tolist())
        for i in range(1, RESEED + 1):
            circuits[scored[-i][1]] = bs.random_ini(density=rho_init)
        parent_ranks = [k // 2 for k in range(RESEED)]
        for pr, i in zip(parent_ranks, sorted(picks)):
            circuits[scored[i][1]] = symmetric_modif(circuits[scored[pr][1]])
        for i in range(lo, hi):
            if i not in picks:
                circuits[scored[i][1]] = symmetric_modif(circuits[scored[i][1]])
        avg_d.append(float(np.mean([m.density() for m in circuits])))
        n_epochs_run = epoch + 1
        if (epoch + 1) % 20 == 0:
            print(f"    B={B:5d} seed={seed} rho0={rho_init:.2f}  "
                  f"epoch {epoch+1:3d}  avg_rho={avg_d[-1]:.4f}  "
                  f"best_cleanMSE={best_clean[-1]:.5f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        if n_epochs_run >= PLATEAU_MIN_EPOCHS:
            xs = np.arange(40); ys = np.array(avg_d[-40:])
            slope = float(np.polyfit(xs, ys, 1)[0])
            if abs(slope) < PLATEAU_SLOPE:
                print(f"    B={B:5d} seed={seed}  PLATEAU at epoch "
                      f"{n_epochs_run} (slope {slope:+.6f})", flush=True)
                break
    return {"B": B, "seed": seed, "rho_init": rho_init,
            "rho_teacher": RHO_TEACHER, "noise_std": NOISE_STD,
            "elapsed": time.time() - t0, "epochs_run": n_epochs_run,
            "plateaued": n_epochs_run < N_EPOCH,
            "rho_eq_last40": float(np.mean(avg_d[-40:])),
            "avg_density_traj": avg_d, "best_clean_mse_traj": best_clean,
            "pop_records": pop_records}


def already_done(records, B, seed, rho_init):
    return any(r["B"] == B and r["seed"] == seed and
               abs(r["rho_init"] - rho_init) < 1e-9 for r in records)


def main():
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    print(f"Teacher-student budget sweep ({RUN_SET})"
          f"\n  B_LIST={B_LIST}  seeds={SEEDS}  rho_init={RHO_INIT}"
          f"\n  rho_teacher={RHO_TEACHER}  noise_std={NOISE_STD}  "
          f"N_EPOCH<={N_EPOCH} (plateau-stopped, slope {PLATEAU_SLOPE:g})  n_jobs={N_JOBS}"
          f"\n  results -> {os.path.relpath(RESULTS, ROOT)}", flush=True)
    records = []
    if os.path.exists(RESULTS):
        with open(RESULTS, "rb") as f:
            records = pickle.load(f).get("records", [])
        print(f"Resumed: {len(records)} records.", flush=True)
    runs = [(B, s) for B in B_LIST for s in SEEDS]
    for idx, (B, seed) in enumerate(runs, 1):
        if already_done(records, B, seed, RHO_INIT):
            print(f"[{idx}/{len(runs)}] B={B} seed={seed}  (done — skipped)",
                  flush=True)
            continue
        print(f"[{idx}/{len(runs)}] B={B} seed={seed} rho0={RHO_INIT} ...",
              flush=True)
        r = run_one(seed, B, RHO_INIT)
        records.append(r)
        print(f"[{idx}/{len(runs)}] B={B:5d} seed={seed}  DONE  "
              f"rho_eq={r['rho_eq_last40']:.4f}  plateaued={r['plateaued']} "
              f"(ep {r['epochs_run']})  cleanMSE={r['best_clean_mse_traj'][-1]:.5f}  "
              f"({r['elapsed']:.0f}s)", flush=True)
        with open(RESULTS, "wb") as f:
            pickle.dump({"records": records,
                         "config": dict(POP_SIZE=POP_SIZE, ELITISM=ELITISM,
                                        RESEED=RESEED, RHO_TEACHER=RHO_TEACHER,
                                        TEACHER_SEED=TEACHER_SEED,
                                        NOISE_STD=NOISE_STD,
                                        PLATEAU_SLOPE=PLATEAU_SLOPE)}, f)
    print("\nSummary — rho_eq (last 40 epochs) per B:", flush=True)
    for B in B_LIST:
        vals = [r["rho_eq_last40"] for r in records
                if r["B"] == B and abs(r["rho_init"] - RHO_INIT) < 1e-9]
        if vals:
            print(f"  B={B:5d}  rho_eq = {np.mean(vals):.4f} "
                  f"+/- {np.std(vals):.4f}  (n={len(vals)})", flush=True)
    print(f"Saved {os.path.relpath(RESULTS, ROOT)} ({len(records)} records).", flush=True)


if __name__ == "__main__":
    main()
