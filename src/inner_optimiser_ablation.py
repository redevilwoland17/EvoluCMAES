"""Inner-optimiser ablation on MNIST 7x7 pruning (Appendix C, Figure 5).

Does the equilibrium density depend on the inner optimiser, or only on the
budget? The outer loop and data are those of budget_sweep_mnist.py (MNIST
7x7, ~970-parameter masked MLP, N = 30, E = 4, R = 6, alpha = 0,
density-agnostic operator swap / remove / add = 0.40 / 0.20 / 0.40,
rho0 = 0.10, 60 outer epochs); the inner loop is one of four optimisers at
the same evaluation budget B:

  cmaes   — bs.train_inner_cmaes (the default inner loop)
  random  — B Gaussian samples around the initial weights, keep the best
  fdsgd   — finite-difference SGD (SPSA two-sided estimator), 2 evals/step
  adam    — full-batch Adam on the active weights, 1 forward query per step

Budget metering: one evaluation = one forward pass used to score a candidate
(the cross-entropy on the training batch). cmaes: maxfevals = B. random: B
samples. fdsgd: floor(B/2) steps. adam: B steps; the backward pass gives
Adam gradient information at no cost under this metering, which favours
Adam.

At B = 2048 the dense-model accuracy (~0.84) is not reached by CMA-ES (best
accuracy ~0.74 in the 60-epoch sweep), so inner-loop fitness still separates
topologies. The outer loop selects on the held-out split on which accuracy
is reported. The Adam arm at B = 256 uses B_OVERRIDE=256
OPTIMIZERS_OVERRIDE=adam with its own RESULTS_PATH_OVERRIDE.

Output: results/mechanism/expB_inner_ablation_mnist_results.pkl (saved after
every run, resumable).
Env overrides: B_OVERRIDE, OPTIMIZERS_OVERRIDE ("adam,cmaes"), SEEDS_OVERRIDE,
N_EPOCH_OVERRIDE, N_JOBS_OVERRIDE, RESULTS_PATH_OVERRIDE.
Usage:  python src/inner_optimiser_ablation.py [--smoke]
"""
import os, sys, time, pickle, tempfile, warnings
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_sweep_mnist as bs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = "--smoke" in sys.argv

bs.N_JOBS  = int(os.environ.get("N_JOBS_OVERRIDE", 8))
B          = 60 if SMOKE else 2048
N_EPOCH    = 3 if SMOKE else 60
SEEDS      = [0] if SMOKE else [0, 1, 2]
if os.environ.get("SEEDS_OVERRIDE"):
    SEEDS = [int(s) for s in os.environ["SEEDS_OVERRIDE"].split(",")]
OPTIMIZERS = ["cmaes", "random", "fdsgd", "adam"]
ADAM_LR    = 0.01          # as in the Adam inner loop of NNpruning_experiment_2
RESULTS_PATH = (os.path.join(tempfile.gettempdir(), "inner_optimiser_ablation_smoke.pkl") if SMOKE
                else os.path.join(ROOT, "results", "mechanism", "expB_inner_ablation_mnist_results.pkl"))
if os.environ.get("B_OVERRIDE"):
    B = int(os.environ["B_OVERRIDE"])
if os.environ.get("OPTIMIZERS_OVERRIDE"):
    OPTIMIZERS = os.environ["OPTIMIZERS_OVERRIDE"].split(",")
if os.environ.get("N_EPOCH_OVERRIDE"):
    N_EPOCH = int(os.environ["N_EPOCH_OVERRIDE"])
if os.environ.get("RESULTS_PATH_OVERRIDE"):
    RESULTS_PATH = os.environ["RESULTS_PATH_OVERRIDE"]


# ---------------- Inner optimisers (all consume <= B fitness evals) --------

def _loss_fn(model, layout, X, y):
    def loss(x):
        bs._apply_vec(model, x, layout)
        with torch.no_grad():
            v = F.cross_entropy(model(X), y).item()
        return float(v) if not np.isnan(v) else 1e6
    return loss


def train_inner_cmaes(model, X, y, budget, seed=0):
    return bs.train_inner_cmaes(model, X, y, budget, seed=seed)


def train_inner_random(model, X, y, budget, seed=0):
    init_vec, layout = bs._collect_active_params(model)
    if len(init_vec) == 0:
        return model
    loss = _loss_fn(model, layout, X, y)
    rng = np.random.default_rng(seed)
    best_x, best_v = init_vec.copy(), loss(init_vec)
    n_evals = 1
    while n_evals < budget:
        x = init_vec + rng.normal(0, bs.INNER_SIGMA0, size=len(init_vec))
        v = loss(x); n_evals += 1
        if v < best_v:
            best_v, best_x = v, x.copy()
    bs._apply_vec(model, best_x, layout)
    return model


def train_inner_fdsgd(model, X, y, budget, seed=0, a=0.05, c=0.05):
    """SPSA-style two-sided finite-difference SGD; 2 fitness evals per step."""
    init_vec, layout = bs._collect_active_params(model)
    n = len(init_vec)
    if n == 0:
        return model
    loss = _loss_fn(model, layout, X, y)
    rng = np.random.default_rng(seed)
    theta = init_vec.copy()
    best_x, best_v = theta.copy(), loss(theta)
    n_evals, k = 1, 0
    while n_evals + 2 <= budget:
        ak = a / (k + 1 + 10) ** 0.602
        ck = c / (k + 1) ** 0.101
        delta = rng.choice([-1.0, 1.0], size=n)
        f_p = loss(theta + ck * delta); n_evals += 1
        f_m = loss(theta - ck * delta); n_evals += 1
        grad = (f_p - f_m) / (2.0 * ck * delta)
        theta = theta - ak * grad
        v = min(f_p, f_m)
        if v < best_v:
            best_v = v
            best_x = (theta + (ck * delta if f_p < f_m else -ck * delta)).copy()
        k += 1
    bs._apply_vec(model, best_x, layout)
    return model


def train_inner_adam(model, X, y, budget, seed=0, lr=ADAM_LR):
    """Full-batch Adam on active parameters; 1 forward fitness query per step.

    Gradients are masked so inactive entries receive no update (parity with
    the vectorised optimizers, which only touch active entries); weights are
    re-masked after each step. Best-seen parameters are restored at the end,
    matching CMA-ES's return-xbest semantics.
    """
    torch.manual_seed(seed)
    params = model.params
    opt = torch.optim.Adam(params, lr=lr)
    best_v, best_state = float("inf"), None
    for _ in range(budget):
        opt.zero_grad()
        out = F.cross_entropy(model(X), y)
        v = float(out.item())
        if not np.isnan(v) and v < best_v:
            best_v = v
            best_state = [p.detach().clone() for p in params]
        out.backward()
        with torch.no_grad():
            for p, m in zip(params, model.masks):
                if p.grad is not None:
                    p.grad.mul_(m.data)
        opt.step()
        with torch.no_grad():
            for p, m in zip(params, model.masks):
                p.data.mul_(m.data)
    if best_state is not None:
        with torch.no_grad():
            for p, s in zip(params, best_state):
                p.data.copy_(s)
    return model


INNER_FNS = {
    "cmaes":  train_inner_cmaes,
    "random": train_inner_random,
    "fdsgd":  train_inner_fdsgd,
    "adam":   train_inner_adam,
}


# ---------------- Outer loop with optimiser routing ------------------------

def score(model, alpha, B, seed, optimizer, Xtr, ytr, Xte, yte):
    warnings.filterwarnings("ignore")
    model = INNER_FNS[optimizer](model, Xtr, ytr, budget=B, seed=seed)
    acc_tr = bs.eval_accuracy(model, Xtr, ytr)
    acc_te = bs.eval_accuracy(model, Xte, yte)
    density = model.density()
    s = acc_te - alpha * (density - bs.F_TARGET) ** 2
    if np.isnan(s): s = -1e6
    return s, acc_tr, acc_te, density, model


def train_evolutive(init_pop, n_epoch, alpha, B, optimizer, base_seed=0):
    """bs.train_evolutive with the inner optimiser routed via `optimizer`;
    population dynamics (selection, elitism, reseed, mutation) identical."""
    from joblib import Parallel, delayed
    history = {k: {"acc_te": [], "density": []} for k in ("best",)}
    avg_density = []
    pop_size = len(init_pop)
    circuits = init_pop
    Xtr, ytr, Xte, yte = bs.X_tr, bs.y_tr, bs.X_te, bs.y_te
    t0 = time.time()
    for epoch in range(n_epoch):
        evaluated = Parallel(n_jobs=bs.N_JOBS)(
            delayed(score)(m, alpha, B,
                           base_seed * 10000 + epoch * pop_size + i, optimizer,
                           Xtr, ytr, Xte, yte)
            for i, m in enumerate(circuits))
        scored = []
        for i, (s, atr, ate, d, trained) in enumerate(evaluated):
            scored.append((s, i, atr, ate, d))
            circuits[i] = trained
        scored.sort(reverse=True)
        history["best"]["acc_te"].append(scored[0][3])
        history["best"]["density"].append(scored[0][4])
        max_s = scored[0][0]
        lo, hi = bs.ELITISM, pop_size - bs.RESEED
        weights = np.array([max_s - t[0] + 1e-6 for t in scored[lo:hi]])
        probs = weights / weights.sum()
        picks = set(np.random.choice(range(lo, hi), size=bs.RESEED,
                                     replace=False, p=probs).tolist())
        for i in range(1, bs.RESEED + 1):
            circuits[scored[-i][1]] = bs.random_ini(density=bs.F_TARGET)
        parent_ranks = [k // 2 for k in range(bs.RESEED)]
        for pr, i in zip(parent_ranks, sorted(picks)):
            circuits[scored[i][1]] = bs.random_modif_quantum(circuits[scored[pr][1]])
        for i in range(lo, hi):
            if i not in picks:
                circuits[scored[i][1]] = bs.random_modif_quantum(circuits[scored[i][1]])
        avg_density.append(float(np.mean([m.density() for m in circuits])))
        if (epoch + 1) % 10 == 0 or epoch == n_epoch - 1:
            print(f"      epoch {epoch+1:3d}/{n_epoch}  avg_rho={avg_density[-1]:.4f}  "
                  f"best_acc={history['best']['acc_te'][-1]:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return circuits, avg_density, history


def run_one(optimizer, seed):
    bs.set_all_seeds(seed)
    init_pop = [bs.random_ini(density=bs.F_TARGET) for _ in range(bs.POP_SIZE)]
    t0 = time.time()
    pop, avg_d, hist = train_evolutive(init_pop, N_EPOCH, bs.ALPHA, B,
                                       optimizer, base_seed=seed)
    return {
        "ds": "mnist", "optimizer": optimizer, "seed": seed, "B": B,
        "elapsed": time.time() - t0,
        "rho_pop_avg_last5": float(np.mean(avg_d[-5:])),
        "rho_best_final": float(hist["best"]["density"][-1]),
        "acc_best_final": float(hist["best"]["acc_te"][-1]),
        "avg_density_traj": avg_d,
        "best_density_traj": hist["best"]["density"],
        "best_acc_traj": hist["best"]["acc_te"],
    }


def already_done(records, optimizer, seed):
    return any(r["optimizer"] == optimizer and r["seed"] == seed for r in records)


def summarize(records):
    print("\n============ Inner-optimiser ablation (MNIST, B=%d, %d epochs) ============"
          % (B, N_EPOCH), flush=True)
    for opt in OPTIMIZERS:
        rs = [r for r in records if r["optimizer"] == opt]
        if not rs: continue
        rho = [r["rho_pop_avg_last5"] for r in rs]
        acc = [r["acc_best_final"] for r in rs]
        print(f"  {opt:7s}  n={len(rs)}  rho_eq = {np.mean(rho):.4f} +/- "
              f"{np.std(rho):.4f}   best_acc = {np.mean(acc):.4f} +/- "
              f"{np.std(acc):.4f}", flush=True)


def main():
    print(f"Inner-optimiser ablation on MNIST pruning"
          f"\n  optimizers: {OPTIMIZERS}   B={B}   seeds={SEEDS}"
          f"\n  N_EPOCH={N_EPOCH}, POP={bs.POP_SIZE}, n_jobs={bs.N_JOBS}, "
          f"smoke={SMOKE}"
          f"\n  saves incrementally to {os.path.relpath(RESULTS_PATH, ROOT)}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(RESULTS_PATH)), exist_ok=True)
    bs.activate_dataset()

    records = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "rb") as f:
            records = pickle.load(f).get("records", [])
        print(f"Resumed: {len(records)} records already saved.", flush=True)

    runs = [(o, s) for o in OPTIMIZERS for s in SEEDS]
    for idx, (opt, seed) in enumerate(runs, 1):
        if already_done(records, opt, seed):
            print(f"[{idx:2d}/{len(runs)}] {opt} seed={seed}  (already done — skipped)",
                  flush=True)
            continue
        print(f"[{idx:2d}/{len(runs)}] {opt} seed={seed}  starting ...", flush=True)
        r = run_one(opt, seed)
        records.append(r)
        print(f"[{idx:2d}/{len(runs)}] {opt:7s} seed={seed}  DONE  "
              f"rho_eq={r['rho_pop_avg_last5']:.4f}  "
              f"rho_best={r['rho_best_final']:.4f}  "
              f"acc={r['acc_best_final']:.4f}  ({r['elapsed']:.0f}s)", flush=True)
        with open(RESULTS_PATH, "wb") as f:
            pickle.dump({"records": records, "optimizers": OPTIMIZERS,
                         "seeds": SEEDS, "B": B,
                         "config": dict(POP_SIZE=bs.POP_SIZE, ELITISM=bs.ELITISM,
                                        RESEED=bs.RESEED, ALPHA=bs.ALPHA,
                                        F_TARGET=bs.F_TARGET, N_EPOCH=N_EPOCH,
                                        ADAM_LR=ADAM_LR)}, f)

    summarize(records)
    print(f"\nSaved {os.path.relpath(RESULTS_PATH, ROOT)} ({len(records)} records).", flush=True)


if __name__ == "__main__":
    main()
