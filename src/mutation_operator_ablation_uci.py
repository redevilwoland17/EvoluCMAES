"""UCI mutation-operator ablation at alpha = 0 (Table 2, unbiased rows under the
drift-neutral operator; Appendix E density traces).

Script version of the alpha = 0 runs of notebooks/NNpruning_experiment_4.ipynb.
Everything is as in the notebook (POP = 100, ELITISM = 10, RESEED = 20, 40
outer epochs with the same early stop, CMA-ES inner loop of 150 evaluations,
initial density 0.05, 70/30 split with random_state = 42, selection on the
held-out split) except the swap / remove / add probabilities of the mutation,
which are selectable:

  OPERATOR=drift_neutral     0.40 / 0.30 / 0.30   (default)
  OPERATOR=density_agnostic  0.40 / 0.20 / 0.40   (the notebook's operator)

Output: results/nnpruning4_<operator>_results.pkl (saved after every run,
resumable). One record per (dataset, seed) with the per-epoch population-mean
density, the history of the best individual and the wall-clock time.

Env overrides: OPERATOR, DATASETS_OVERRIDE ("iris,wine"), SEEDS_OVERRIDE
("0,1"), N_JOBS_OVERRIDE (default 8, the setting of the stored runs).
Usage:  python src/mutation_operator_ablation_uci.py [--smoke]
"""
import os, pickle, random, sys, tempfile, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cma
from joblib import Parallel, delayed
from sklearn.datasets import load_wine, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = "--smoke" in sys.argv

DATASETS  = ["iris", "wine", "cancer"]
HIDDEN    = [64, 32]
F_TARGET  = 0.05
SEEDS     = [0, 1, 2, 3, 4]

POP_SIZE      = 100
N_EPOCH_MULTI = 40
ELITISM       = 10
RESEED        = 20
INNER_BUDGET  = 150
INNER_SIGMA0  = 0.3
MIN_DENSITY   = 0.005
ALPHA_NATURAL = 0.0

# swap / remove / add: (P_SWAP, P_SWAP + P_REMOVE) thresholds on a uniform draw
OPERATORS = {"drift_neutral": (0.4, 0.7), "density_agnostic": (0.4, 0.6)}
OPERATOR  = os.environ.get("OPERATOR", "drift_neutral")
T_SWAP, T_REMOVE = OPERATORS[OPERATOR]
N_JOBS    = int(os.environ.get("N_JOBS_OVERRIDE", 8))
if os.environ.get("DATASETS_OVERRIDE"):
    DATASETS = os.environ["DATASETS_OVERRIDE"].split(",")
if os.environ.get("SEEDS_OVERRIDE"):
    SEEDS = [int(s) for s in os.environ["SEEDS_OVERRIDE"].split(",")]
RESULTS_FILE = os.path.join(ROOT, "results", f"nnpruning4_{OPERATOR}_results.pkl")
if SMOKE:
    DATASETS, SEEDS, N_EPOCH_MULTI = ["iris"], [0], 3
    RESULTS_FILE = os.path.join(tempfile.gettempdir(), f"mutation_operator_ablation_uci_{OPERATOR}_smoke.pkl")


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(name):
    d = {"iris": load_iris, "wine": load_wine, "cancer": load_breast_cancer}[name]()
    X, y = d.data, d.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    sc = StandardScaler().fit(X_tr)
    X_tr = torch.tensor(sc.transform(X_tr), dtype=torch.float32)
    X_te = torch.tensor(sc.transform(X_te), dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    y_te = torch.tensor(y_te, dtype=torch.long)
    dims = [int(X_tr.shape[1])] + HIDDEN + [int(len(set(d.target.tolist())))]
    return (X_tr, y_tr, X_te, y_te), dims


class MaskedMLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = list(dims)
        self.layers = nn.ModuleList([nn.Linear(self.dims[i], self.dims[i+1]) for i in range(len(self.dims)-1)])
        self.masks = nn.ParameterList([
            nn.Parameter(torch.ones_like(l.weight), requires_grad=False)
            for l in self.layers
        ])

    def forward(self, x):
        for i, l in enumerate(self.layers):
            w = l.weight * self.masks[i]
            x = F.linear(x, w, l.bias)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

    def density(self):
        total = sum(m.numel() for m in self.masks)
        active = sum(float(m.sum().item()) for m in self.masks)
        return active / total

    def clone_arch(self):
        new = MaskedMLP(self.dims)
        for i, l in enumerate(self.layers):
            new.layers[i].weight.data.copy_(l.weight.data)
            new.layers[i].bias.data.copy_(l.bias.data)
            new.masks[i].data.copy_(self.masks[i].data)
        return new


# ----- Mutation: swap / remove / add with fixed probabilities ---------------
def _flip_one_active(model):
    li = np.random.randint(len(model.masks))
    m = model.masks[li]
    active = torch.where(m.data.flatten() > 0.5)[0]
    if len(active) == 0:
        return None
    flat = int(active[np.random.randint(len(active))].item())
    idx = np.unravel_index(flat, m.shape)
    m.data[idx] = 0.0
    return (li, idx)


def _flip_one_inactive(model):
    li = np.random.randint(len(model.masks))
    m = model.masks[li]
    inactive = torch.where(m.data.flatten() < 0.5)[0]
    if len(inactive) == 0:
        return None
    flat = int(inactive[np.random.randint(len(inactive))].item())
    idx = np.unravel_index(flat, m.shape)
    m.data[idx] = 1.0
    return (li, idx)


def random_modif_quantum(model, n_flips=None, min_density=MIN_DENSITY,
                         t_swap=None, t_remove=None):
    t_swap = T_SWAP if t_swap is None else t_swap
    t_remove = T_REMOVE if t_remove is None else t_remove
    model = model.clone_arch()
    total = sum(m.numel() for m in model.masks)
    if n_flips is None:
        n_flips = max(1, int(0.02 * total))
    min_active = max(1, int(min_density * total))
    for _ in range(n_flips):
        active_count = sum(int(m.data.sum().item()) for m in model.masks)
        a = np.random.random()
        if a < t_swap and active_count > 0 and active_count < total:
            removed = _flip_one_active(model)
            if removed is None:
                continue
            added = _flip_one_inactive(model)
            if added is None:
                model.masks[removed[0]].data[removed[1]] = 1.0
        elif a < t_remove and active_count > min_active:
            _flip_one_active(model)
        else:
            if active_count < total:
                _flip_one_inactive(model)
    return model


def random_ini(density, dims):
    model = MaskedMLP(dims)
    for i in range(len(model.masks)):
        model.masks[i].data = (torch.rand_like(model.masks[i]) < density).float()
    return model


# ----- CMA-ES inner loop -----------------------------------------------------
def _collect_active_params(model):
    init_vals = []
    layout = []
    for li, l in enumerate(model.layers):
        mask = model.masks[li].data
        rows, cols = torch.where(mask > 0.5)
        for r, c in zip(rows.tolist(), cols.tolist()):
            init_vals.append(float(l.weight.data[r, c].item()))
            layout.append(("w", li, r, c))
    for li, l in enumerate(model.layers):
        for bi in range(l.bias.numel()):
            init_vals.append(float(l.bias.data[bi].item()))
            layout.append(("b", li, bi))
    return np.array(init_vals, dtype=np.float64), layout


def _apply_vec(model, x, layout):
    with torch.no_grad():
        for k, item in enumerate(layout):
            if item[0] == "w":
                _, li, r, c = item
                model.layers[li].weight.data[r, c] = float(x[k])
            else:
                _, li, bi = item
                model.layers[li].bias.data[bi] = float(x[k])
        for li, l in enumerate(model.layers):
            l.weight.data.mul_(model.masks[li])


def train_inner_cmaes(model, X, y, budget=INNER_BUDGET, sigma0=INNER_SIGMA0, seed=None):
    init_vec, layout = _collect_active_params(model)
    if len(init_vec) == 0:
        return model

    def loss_fn(x):
        _apply_vec(model, x, layout)
        with torch.no_grad():
            l = F.cross_entropy(model(X), y).item()
        return float(l) if not np.isnan(l) else 1e6

    opts = {"maxfevals": budget, "verbose": -9, "seed": int(seed) if seed is not None else 0,
            "tolx": 1e-8, "tolfun": 1e-8}
    es = cma.CMAEvolutionStrategy(init_vec.tolist(), sigma0, opts)
    es.optimize(loss_fn)
    best_x = es.result.xbest if es.result.xbest is not None else init_vec
    _apply_vec(model, best_x, layout)
    return model


def eval_accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=1)
        return float((pred == y).float().mean().item())


def run_inner_eval(model, f_target, alpha, seed, data):
    """Inner loop + score of one individual (runs in a worker process)."""
    warnings.filterwarnings("ignore")
    X_tr, y_tr, X_te, y_te = data
    try:
        model = train_inner_cmaes(model, X_tr, y_tr, seed=seed)
        acc_tr = eval_accuracy(model, X_tr, y_tr)
        acc_te = eval_accuracy(model, X_te, y_te)
        density = model.density()
        s = acc_te - alpha * (density - f_target) ** 2
        if np.isnan(s):
            s = -1e6
        return s, acc_tr, acc_te, density, model
    except Exception as e:
        print(f"eval failed: {e}")
        return -1e6, 0.0, 0.0, 1.0, model


# ----- Outer loop ------------------------------------------------------------
def train_evolutive(init_pop, n_epoch, f_target, alpha, data, dims,
                    n_jobs=N_JOBS, elitism=ELITISM, reseed=RESEED,
                    min_epoch=20, patience=10, tol=1e-4, base_seed=0,
                    init_density=F_TARGET):
    history = {k: {"acc_tr": [], "acc_te": [], "density": [], "loss": []}
               for k in ("best", "second", "third", "mid")}
    avg_density = []
    pop_size = len(init_pop)
    circuit_list = init_pop
    for epoch in range(n_epoch):
        evaluated = Parallel(n_jobs=n_jobs)(
            delayed(run_inner_eval)(m, f_target, alpha,
                                    base_seed * 10000 + epoch * pop_size + i, data)
            for i, m in enumerate(circuit_list))
        scored = []
        for i, (s, atr, ate, d, trained) in enumerate(evaluated):
            scored.append((s, i, atr, ate, d))
            circuit_list[i] = trained
        scored.sort(reverse=True)
        for key, rank in (("best", 0), ("second", 1), ("third", 2),
                          ("mid", len(scored) // 2)):
            history[key]["acc_tr"].append(scored[rank][2])
            history[key]["acc_te"].append(scored[rank][3])
            history[key]["density"].append(scored[rank][4])
            history[key]["loss"].append(-scored[rank][0])

        max_s = scored[0][0]
        lo, hi = elitism, pop_size - reseed
        weights = np.array([max_s - t[0] + 1e-6 for t in scored[lo:hi]])
        probs = weights / weights.sum()
        picks = set(np.random.choice(range(lo, hi), size=reseed,
                                     replace=False, p=probs).tolist())

        for i in range(1, reseed + 1):
            circuit_list[scored[-i][1]] = random_ini(init_density, dims)

        parent_ranks = [k // 2 for k in range(reseed)]
        assert parent_ranks[-1] < elitism

        for pr, i in zip(parent_ranks, sorted(picks)):
            parent = circuit_list[scored[pr][1]]
            circuit_list[scored[i][1]] = random_modif_quantum(parent)
        for i in range(lo, hi):
            if i not in picks:
                circuit_list[scored[i][1]] = random_modif_quantum(circuit_list[scored[i][1]])

        avg_density.append(float(np.mean([m.density() for m in circuit_list])))

        if epoch + 1 >= min_epoch:
            recent = history["best"]["loss"][-patience:]
            if len(recent) == patience and (max(recent) - min(recent)) < tol:
                break

    return avg_density, history


def run_seed(name, seed, alpha=ALPHA_NATURAL, init_density=F_TARGET):
    data, dims = load_dataset(name)
    set_all_seeds(seed)
    init_pop = [random_ini(init_density, dims) for _ in range(POP_SIZE)]
    t0 = time.time()
    avg_d, hist = train_evolutive(init_pop, N_EPOCH_MULTI, F_TARGET, alpha, data, dims,
                                  base_seed=seed, init_density=init_density)
    return {"dataset": name, "seed": seed, "operator": OPERATOR,
            "p_swap_remove_add": (T_SWAP, round(T_REMOVE - T_SWAP, 2), round(1 - T_REMOVE, 2)),
            "dims": dims, "elapsed": time.time() - t0, "epochs_run": len(avg_d),
            "acc_te_best_final": hist["best"]["acc_te"][-1],
            "density_best_final": hist["best"]["density"][-1],
            "avg_d_final": avg_d[-1], "avg_d": avg_d, "hist": hist}


def summarize(records):
    print(f"\nSummary ({OPERATOR}; mean +/- std over seeds, std with ddof=0):")
    for ds in DATASETS:
        rs = [r for r in records if r["dataset"] == ds]
        if not rs:
            continue
        acc = np.array([r["acc_te_best_final"] for r in rs])
        db = np.array([r["density_best_final"] for r in rs])
        da = np.array([r["avg_d_final"] for r in rs])
        print(f"  {ds:7s} n={len(rs)}  acc = {acc.mean():.3f} +/- {acc.std():.3f}   "
              f"density(best) = {db.mean():.3f} +/- {db.std():.3f}   "
              f"density(pop. mean) = {da.mean():.3f} +/- {da.std():.3f}", flush=True)


def main():
    print(f"UCI alpha=0 runs, operator = {OPERATOR} "
          f"(swap/remove/add = {T_SWAP:.2f}/{T_REMOVE - T_SWAP:.2f}/{1 - T_REMOVE:.2f})"
          f"\n  datasets={DATASETS} seeds={SEEDS} POP={POP_SIZE} E={ELITISM} R={RESEED}"
          f"\n  results -> {RESULTS_FILE if SMOKE else os.path.relpath(RESULTS_FILE, ROOT)}", flush=True)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    records = []
    if os.path.exists(RESULTS_FILE) and not SMOKE:
        with open(RESULTS_FILE, "rb") as f:
            records = pickle.load(f)["records"]
        print(f"Resumed: {len(records)} records.", flush=True)
    for ds in DATASETS:
        for s in SEEDS:
            if any(r["dataset"] == ds and r["seed"] == s for r in records):
                continue
            r = run_seed(ds, s)
            records.append(r)
            print(f"  [{ds}] seed={s}  done in {r['elapsed']:.0f}s, acc_te={r['acc_te_best_final']:.4f}, "
                  f"density={r['density_best_final']:.3f}, avg_d_final={r['avg_d_final']:.3f} "
                  f"({r['epochs_run']} epochs)", flush=True)
            with open(RESULTS_FILE, "wb") as f:
                pickle.dump({"records": records, "operator": OPERATOR,
                             "config": dict(POP_SIZE=POP_SIZE, ELITISM=ELITISM, RESEED=RESEED,
                                            N_EPOCH=N_EPOCH_MULTI, INNER_BUDGET=INNER_BUDGET,
                                            F_TARGET=F_TARGET, ALPHA=ALPHA_NATURAL)}, f)
    summarize(records)


if __name__ == "__main__":
    main()
