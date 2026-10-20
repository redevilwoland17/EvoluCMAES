"""Pruning at 95% sparsity on small tabular datasets, with per-method wall-clock
(Appendix G; the per-method wall-clock of Section 4.2).

Script version of notebooks/NNpruning_experiment_2.ipynb (Table 1):
EvoluCMAES with an Adam inner loop (80 epochs, lr 0.01), biased mutation,
score penalty 500 (rho - 0.05)^2, POP = 30, ELITISM = 5, RESEED = 10, up to 60
outer epochs with the notebook's early stop, against five baselines (random,
magnitude, IMP, SNIP, SynFlow) at density 0.05, five seeds, 70/30 split with
random_state = 42. A timer is placed around every method.

Datasets: iris, wine, cancer (scikit-learn, as in Table 1) and four further
small UCI datasets fetched from OpenML by id: seeds (1499), glass (41),
ionosphere (59), sonar (40).

Two protocols (PROTOCOL env variable):

  table1      the Table 1 protocol. The outer loop selects on the held-out
              30% split and accuracy is reported on that same split, so the
              EvoluCMAES accuracy is biased upwards. Default datasets: all
              seven (the three Table 1 datasets give the Section 4.2 timings).
  validation  the 70% training part is split again 80/20 (random_state = 42,
              stratified). EvoluCMAES trains its inner loop on the 80% and
              selects on the 20%; accuracy is reported on the untouched 30%
              test split. The baselines are trained on the full 70% as in
              table1 (they never use a selection split), so their numbers
              are identical under both protocols. Default datasets: the four
              further ones.

Output: results/tabular_<protocol>_results.pkl (saved after every method run,
resumable). One record per (dataset, method, seed).

Env overrides: PROTOCOL, DATASETS_OVERRIDE ("seeds,glass"), SEEDS_OVERRIDE,
METHODS_OVERRIDE ("EvoluCMAES,SNIP"), N_JOBS_OVERRIDE (default 8, the
setting of the reported EvoluCMAES timings; the baselines run in a single
process).
Usage:  PROTOCOL=validation python src/tabular_benchmarks.py [--smoke]
"""
import os, pickle, random, sys, tempfile, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.datasets import load_wine, load_iris, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = "--smoke" in sys.argv

SKLEARN  = {"iris": load_iris, "wine": load_wine, "cancer": load_breast_cancer}
OPENML   = {"seeds": 1499, "glass": 41, "ionosphere": 59, "sonar": 40}
HIDDEN   = [64, 32]
ALPHA    = 500.0
F_TARGET = 0.05
INNER_EPOCHS = 80
INNER_LR     = 0.01
SEEDS = [0, 1, 2, 3, 4]
N_EPOCH_MULTI = 60
POP_MULTI = 30
ELITISM, RESEED = 5, 10
METHODS = ["EvoluCMAES", "Magnitude", "IMP", "SNIP", "SynFlow", "Random"]

PROTOCOL = os.environ.get("PROTOCOL", "table1")
assert PROTOCOL in ("table1", "validation")
DATASETS = (list(SKLEARN) if PROTOCOL == "table1" else []) + list(OPENML)
N_JOBS = int(os.environ.get("N_JOBS_OVERRIDE", 8))
if os.environ.get("DATASETS_OVERRIDE"):
    DATASETS = os.environ["DATASETS_OVERRIDE"].split(",")
if os.environ.get("SEEDS_OVERRIDE"):
    SEEDS = [int(s) for s in os.environ["SEEDS_OVERRIDE"].split(",")]
if os.environ.get("METHODS_OVERRIDE"):
    METHODS = os.environ["METHODS_OVERRIDE"].split(",")
RESULTS_FILE = os.path.join(ROOT, "results", f"tabular_{PROTOCOL}_results.pkl")
if SMOKE:
    DATASETS, SEEDS, N_EPOCH_MULTI = ["iris"], [0], 3
    RESULTS_FILE = os.path.join(tempfile.gettempdir(), f"tabular_benchmarks_{PROTOCOL}_smoke.pkl")


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(name, protocol=PROTOCOL):
    """Returns the splits as tensors and the layer sizes.

    train: the 70% part (baselines train here). test: the 30% part.
    fit / sel: what the EvoluCMAES inner loop trains on and what its outer
    loop selects on (train / test under 'table1'; 80% / 20% of train under
    'validation').
    """
    if name in SKLEARN:
        d = SKLEARN[name]()
        X, y = d.data, d.target
    else:
        d = fetch_openml(data_id=OPENML[name], as_frame=True, parser="auto")
        X = d.data.to_numpy(dtype=np.float64)
        y = LabelEncoder().fit_transform(np.asarray(d.target).astype(str))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    sc = StandardScaler().fit(X_tr)
    t = lambda A: torch.tensor(sc.transform(A), dtype=torch.float32)
    l = lambda a: torch.tensor(a, dtype=torch.long)
    data = {"train": (t(X_tr), l(y_tr)), "test": (t(X_te), l(y_te))}
    if protocol == "validation":
        X_fit, X_sel, y_fit, y_sel = train_test_split(X_tr, y_tr, test_size=0.2,
                                                      random_state=42, stratify=y_tr)
        data["fit"], data["sel"] = (t(X_fit), l(y_fit)), (t(X_sel), l(y_sel))
    else:
        data["fit"], data["sel"] = data["train"], data["test"]
    dims = [int(X.shape[1])] + HIDDEN + [int(len(set(np.asarray(y).tolist())))]
    return data, dims


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


# ----- Density-driven (biased) mutation --------------------------------------
def random_modif(model, n_flips=None, f_target=None):
    model = model.clone_arch()
    total = sum(m.numel() for m in model.masks)
    if n_flips is None:
        n_flips = max(1, int(0.02 * total))
    if f_target is not None:
        current_d = model.density()
        p_add = float(np.clip(0.5 - 4.0 * (current_d - f_target), 0.05, 0.95))
    else:
        p_add = 0.5
    for _ in range(n_flips):
        li = np.random.randint(len(model.masks))
        m = model.masks[li]
        if np.random.random() < p_add:
            inactive = torch.where(m.data.flatten() < 0.5)[0]
            if len(inactive) == 0:
                continue
            flat = int(inactive[np.random.randint(len(inactive))].item())
        else:
            active = torch.where(m.data.flatten() > 0.5)[0]
            if len(active) == 0:
                continue
            flat = int(active[np.random.randint(len(active))].item())
        idx = np.unravel_index(flat, m.shape)
        m.data[idx] = 1.0 - m.data[idx]
    return model


def random_ini(density, dims):
    model = MaskedMLP(dims)
    for i in range(len(model.masks)):
        model.masks[i].data = (torch.rand_like(model.masks[i]) < density).float()
    return model


# ----- Inner loop and score --------------------------------------------------
def train_inner_adam(model, X, y, epochs=INNER_EPOCHS, lr=INNER_LR):
    opt = torch.optim.Adam([p for l in model.layers for p in (l.weight, l.bias)], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        with torch.no_grad():
            for i, l in enumerate(model.layers):
                if l.weight.grad is not None:
                    l.weight.grad.mul_(model.masks[i])
        opt.step()
        with torch.no_grad():
            for i, l in enumerate(model.layers):
                l.weight.data.mul_(model.masks[i])
    return model


def eval_accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=1)
        return float((pred == y).float().mean().item())


def run_inner_eval(model, f_target, fit, sel):
    """Inner loop + score of one individual (runs in a worker process)."""
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        model = train_inner_adam(model, fit[0], fit[1])
        acc_fit = eval_accuracy(model, fit[0], fit[1])
        acc_sel = eval_accuracy(model, sel[0], sel[1])
        density = model.density()
        s = acc_sel - ALPHA * (density - f_target) ** 2
        if np.isnan(s):
            s = -1e6
        return s, acc_fit, acc_sel, density, model
    except Exception as e:
        print(f"eval failed: {e}")
        return -1e6, 0.0, 0.0, 1.0, model


def train_evolutive(init_pop, n_epoch, f_target, data, dims,
                    n_jobs=N_JOBS, elitism=ELITISM, reseed=RESEED,
                    min_epoch=20, patience=10, tol=1e-4):
    history = {k: {"acc_tr": [], "acc_te": [], "density": [], "loss": []}
               for k in ("best", "second", "third", "mid")}
    avg_density = []
    pop_size = len(init_pop)
    circuit_list = init_pop
    for epoch in range(n_epoch):
        evaluated = Parallel(n_jobs=n_jobs)(
            delayed(run_inner_eval)(m, f_target, data["fit"], data["sel"]) for m in circuit_list)
        scored = []
        for i, (s, atr, ate, d, trained) in enumerate(evaluated):
            scored.append((s, i, atr, ate, d))
            circuit_list[i] = trained
        scored.sort(reverse=True)
        for key, rank in (("best", 0), ("second", 1), ("third", 2),
                          ("mid", len(scored) // 2)):
            history[key]["acc_tr"].append(scored[rank][2])
            history[key]["acc_te"].append(scored[rank][3])   # accuracy on the selection split
            history[key]["density"].append(scored[rank][4])
            history[key]["loss"].append(-scored[rank][0])

        max_s = scored[0][0]
        lo, hi = elitism, pop_size - reseed
        weights = np.array([max_s - t[0] + 1e-6 for t in scored[lo:hi]])
        probs = weights / weights.sum()
        picks = set(np.random.choice(range(lo, hi), size=reseed,
                                     replace=False, p=probs).tolist())
        for i in range(1, reseed + 1):
            circuit_list[scored[-i][1]] = random_ini(f_target, dims)

        parent_ranks = [k // 2 for k in range(reseed)]
        assert parent_ranks[-1] < elitism
        for pr, i in zip(parent_ranks, sorted(picks)):
            parent = circuit_list[scored[pr][1]]
            circuit_list[scored[i][1]] = random_modif(parent, f_target=f_target)
        for i in range(lo, hi):
            if i not in picks:
                circuit_list[scored[i][1]] = random_modif(circuit_list[scored[i][1]], f_target=f_target)

        avg_density.append(float(np.mean([m.density() for m in circuit_list])))

        if epoch + 1 >= min_epoch:
            recent = history["best"]["loss"][-patience:]
            if len(recent) == patience and (max(recent) - min(recent)) < tol:
                break

    best_model = circuit_list[scored[0][1]]      # an elite: left untouched above
    return best_model, avg_density, history


# ----- Baselines (trained on the 70% training part) --------------------------
def train_dense(dims, tr, epochs=200, lr=0.01, seed=0):
    set_all_seeds(seed)
    model = MaskedMLP(dims)
    train_inner_adam(model, tr[0], tr[1], epochs=epochs, lr=lr)
    return model


def magnitude_pruning(dims, tr, target_density=0.3, epochs=200, lr=0.01, seed=0):
    model = train_dense(dims, tr, epochs=epochs, lr=lr, seed=seed)
    all_w = torch.cat([l.weight.abs().flatten() for l in model.layers])
    k = int(target_density * all_w.numel())
    if k == 0:
        thresh = float(all_w.max().item()) + 1.0
    else:
        thresh = float(torch.topk(all_w, k).values[-1].item())
    for i, l in enumerate(model.layers):
        model.masks[i].data = (l.weight.abs() >= thresh).float()
    train_inner_adam(model, tr[0], tr[1], epochs=epochs, lr=lr)
    return model


def random_pruning(dims, tr, target_density=0.3, epochs=200, lr=0.01, seed=0):
    set_all_seeds(seed)
    model = MaskedMLP(dims)
    for i in range(len(model.masks)):
        model.masks[i].data = (torch.rand_like(model.masks[i]) < target_density).float()
    train_inner_adam(model, tr[0], tr[1], epochs=epochs, lr=lr)
    return model


def snip_pruning(dims, tr, target_density=0.3, epochs=200, lr=0.01, seed=0):
    set_all_seeds(seed)
    model = MaskedMLP(dims)
    for l in model.layers:
        l.weight.requires_grad_(True)
    out = model(tr[0])
    loss = F.cross_entropy(out, tr[1])
    grads = torch.autograd.grad(loss, [l.weight for l in model.layers])
    saliencies = torch.cat([torch.abs(l.weight * g).flatten() for l, g in zip(model.layers, grads)])
    k = int(target_density * saliencies.numel())
    thresh = float(torch.topk(saliencies, k).values[-1].item()) if k > 0 else float("inf")
    for i, l in enumerate(model.layers):
        model.masks[i].data = (torch.abs(l.weight * grads[i]) >= thresh).float()
    train_inner_adam(model, tr[0], tr[1], epochs=epochs, lr=lr)
    return model


def synflow_pruning(dims, tr, target_density=0.3, epochs=200, lr=0.01, seed=0, n_iter=100):
    set_all_seeds(seed)
    model = MaskedMLP(dims)
    with torch.no_grad():
        weights_backup = [l.weight.data.clone() for l in model.layers]
        bias_backup = [l.bias.data.clone() for l in model.layers]
        for l in model.layers:
            l.weight.data = l.weight.data.abs()
            l.bias.data = l.bias.data.abs()
    input_ones = torch.ones(1, dims[0])
    end_sparsity = 1.0 - target_density
    for it in range(n_iter):
        out = model(input_ones)
        loss = out.sum()
        grads = torch.autograd.grad(loss, [l.weight for l in model.layers])
        saliencies = torch.cat([torch.abs(l.weight * g * m).flatten()
                                for l, g, m in zip(model.layers, grads, model.masks)])
        sparsity = end_sparsity * ((it + 1) / n_iter) ** 3
        keep = int((1 - sparsity) * saliencies.numel())
        if keep > 0:
            thresh = float(torch.topk(saliencies, keep).values[-1].item())
            for i, l in enumerate(model.layers):
                w_sal = torch.abs(l.weight * grads[i] * model.masks[i])
                model.masks[i].data = (w_sal >= thresh).float()
    with torch.no_grad():
        for l, w, b in zip(model.layers, weights_backup, bias_backup):
            l.weight.data = w
            l.bias.data = b
    train_inner_adam(model, tr[0], tr[1], epochs=epochs, lr=lr)
    return model


def imp_pruning(dims, tr, target_density=0.3, epochs=100, lr=0.01, seed=0, n_rounds=5):
    set_all_seeds(seed)
    model = MaskedMLP(dims)
    train_inner_adam(model, tr[0], tr[1], epochs=epochs, lr=lr)
    for r in range(1, n_rounds + 1):
        target_r = 1.0 - (1.0 - target_density) * (r / n_rounds)
        all_w = torch.cat([l.weight.abs().flatten() for l in model.layers])
        k = int(target_r * all_w.numel())
        thresh = float(torch.topk(all_w, k).values[-1].item()) if k > 0 else float("inf")
        for i, l in enumerate(model.layers):
            model.masks[i].data = (l.weight.abs() >= thresh).float()
        train_inner_adam(model, tr[0], tr[1], epochs=epochs, lr=lr)
    return model


BASELINES = {"Magnitude": magnitude_pruning, "IMP": imp_pruning, "SNIP": snip_pruning,
             "SynFlow": synflow_pruning, "Random": random_pruning}


# ----- One (dataset, method, seed) run, timed --------------------------------
def run_method(name, method, seed, data, dims, f_target=F_TARGET):
    rec = {"dataset": name, "method": method, "seed": seed, "protocol": PROTOCOL, "dims": dims}
    t0 = time.time()
    if method == "EvoluCMAES":
        set_all_seeds(seed)
        init_pop = [random_ini(f_target, dims) for _ in range(POP_MULTI)]
        best, avg_d, hist = train_evolutive(init_pop, N_EPOCH_MULTI, f_target, data, dims)
        rec.update(elapsed=time.time() - t0, epochs_run=len(avg_d), n_jobs=N_JOBS,
                   acc_sel=hist["best"]["acc_te"][-1],
                   acc_test=eval_accuracy(best, *data["test"]),
                   density=hist["best"]["density"][-1],
                   hist_best=hist["best"], avg_d=avg_d)
    else:
        m = BASELINES[method](dims, data["train"], target_density=f_target, seed=seed)
        rec.update(elapsed=time.time() - t0, acc_test=eval_accuracy(m, *data["test"]),
                   density=m.density())
    return rec


def summarize(records):
    print(f"\nSummary, protocol = {PROTOCOL} (mean +/- std over seeds, ddof=0; "
          f"wall-clock per run in seconds):")
    for ds in DATASETS:
        for method in METHODS:
            rs = [r for r in records if r["dataset"] == ds and r["method"] == method]
            if not rs:
                continue
            acc = np.array([r["acc_test"] for r in rs]); den = np.array([r["density"] for r in rs])
            el = np.array([r["elapsed"] for r in rs])
            extra = ""
            if method == "EvoluCMAES" and PROTOCOL == "validation":
                sel = np.array([r["acc_sel"] for r in rs])
                extra = f"   (acc on selection split {sel.mean():.3f} +/- {sel.std():.3f})"
            print(f"  {ds:10s} {method:10s} n={len(rs)}  test acc = {acc.mean():.3f} +/- {acc.std():.3f}   "
                  f"density = {den.mean():.3f}   time = {el.mean():7.1f} +/- {el.std():.1f}{extra}", flush=True)


def main():
    print(f"Tabular pruning benchmarks, protocol = {PROTOCOL}"
          f"\n  datasets={DATASETS} seeds={SEEDS} methods={METHODS}"
          f"\n  results -> {RESULTS_FILE if SMOKE else os.path.relpath(RESULTS_FILE, ROOT)}", flush=True)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    records = []
    if os.path.exists(RESULTS_FILE) and not SMOKE:
        with open(RESULTS_FILE, "rb") as f:
            records = pickle.load(f)["records"]
        print(f"Resumed: {len(records)} records.", flush=True)
    for ds in DATASETS:
        data, dims = load_dataset(ds)
        print(f"{ds}: dims={dims} train={len(data['train'][0])} fit={len(data['fit'][0])} "
              f"sel={len(data['sel'][0])} test={len(data['test'][0])}", flush=True)
        for method in METHODS:
            for s in SEEDS:
                if any(r["dataset"] == ds and r["method"] == method and r["seed"] == s for r in records):
                    continue
                r = run_method(ds, method, s, data, dims)
                records.append(r)
                print(f"  [{ds}] {method:10s} seed={s}  acc_test={r['acc_test']:.4f}  "
                      f"density={r['density']:.3f}  ({r['elapsed']:.1f}s)", flush=True)
                with open(RESULTS_FILE, "wb") as f:
                    pickle.dump({"records": records, "protocol": PROTOCOL,
                                 "config": dict(POP=POP_MULTI, ELITISM=ELITISM, RESEED=RESEED,
                                                N_EPOCH=N_EPOCH_MULTI, INNER_EPOCHS=INNER_EPOCHS,
                                                INNER_LR=INNER_LR, ALPHA=ALPHA, F_TARGET=F_TARGET)}, f)
    summarize(records)


if __name__ == "__main__":
    main()
