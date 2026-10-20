"""Budget sweep on downsampled MNIST.

Companion to the UCI budget sweep. Same paper §5 design (B/8 ... 8B, 5 seeds,
α=0, unbiased qmut, NNpruning_experiment_4 outer loop), but on a harder
dataset and at a parameter scale where the inner-loop convergence horizon sits
inside the swept B-range.

Configuration:
  - MNIST images downsampled 28x28 → 7x7 (49 dims)
  - MLP 49 → 16 → 10  (970 dense params)
  - At ρ_init=0.10 the active params are ~97; convergence horizon ~10k evals.
    8B = 1200 sits at ~10% of horizon for ρ=0.10 — well inside the variance-gap
    regime that Corollary 1 predicts.
  - TRAIN_SUBSET=1024, VAL_SUBSET=1024  (forward pass ≈ 0.5 ms)

Output:  budget_sweep_mnist_results.pkl + fig_budget_sweep_mnist.pdf
"""
import os, time, pickle, warnings, random
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
import cma
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----- Config (paper-faithful sweep grid) -----
F_TARGET     = 0.10
ALPHA        = 0.0
B_BASELINE   = 150
B_LIST       = [B_BASELINE // 8, B_BASELINE // 4, B_BASELINE // 2,
                B_BASELINE, 2*B_BASELINE, 4*B_BASELINE, 8*B_BASELINE]
SEEDS        = [0, 1, 2, 3, 4]
POP_SIZE     = 30
N_EPOCH      = 15
ELITISM      = 4
RESEED       = 6
INNER_SIGMA0 = 0.3
MIN_DENSITY  = 0.005
TRAIN_SUBSET = 1024
VAL_SUBSET   = 1024
N_JOBS       = 8              # leave 8 cores free so the laptop stays responsive
DOWNSAMPLE   = 4              # 28x28 → 7x7

X_tr = y_tr = X_te = y_te = None


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _load_mnist_downsampled():
    import torchvision
    from torchvision import transforms
    tx = transforms.Compose([transforms.ToTensor()])
    root = os.path.expanduser("~/.cache/mnist")
    os.makedirs(root, exist_ok=True)
    tr = torchvision.datasets.MNIST(root=root, train=True, transform=tx, download=True)
    te = torchvision.datasets.MNIST(root=root, train=False, transform=tx, download=True)
    Xtr = torch.stack([tr[i][0] for i in range(len(tr))]).float()
    ytr = torch.tensor([tr[i][1] for i in range(len(tr))], dtype=torch.long)
    Xte = torch.stack([te[i][0] for i in range(len(te))]).float()
    yte = torch.tensor([te[i][1] for i in range(len(te))], dtype=torch.long)
    # downsample 28x28 → (28/k)x(28/k) via average pooling
    k = DOWNSAMPLE
    Xtr = F.avg_pool2d(Xtr, k).flatten(1)            # → (N, 49)
    Xte = F.avg_pool2d(Xte, k).flatten(1)
    # standardise per-feature
    mean = Xtr.mean(0, keepdim=True); std = Xtr.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    return Xtr, ytr, Xte, yte


def activate_dataset():
    global X_tr, y_tr, X_te, y_te
    Xtr, ytr, Xte, yte = _load_mnist_downsampled()
    rng = np.random.RandomState(0)
    tr_idx = rng.choice(len(Xtr), TRAIN_SUBSET, replace=False)
    te_idx = rng.choice(len(Xte), VAL_SUBSET, replace=False)
    X_tr = Xtr[tr_idx]
    y_tr = ytr[tr_idx]
    X_te = Xte[te_idx]
    y_te = yte[te_idx]


# ----- Small MLP ~970 params -----
IN_DIM  = (28 // DOWNSAMPLE) ** 2     # 49
HIDDEN  = 16
OUT_DIM = 10

class MaskedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, OUT_DIM)
        self.params = [self.fc1.weight, self.fc1.bias,
                       self.fc2.weight, self.fc2.bias]
        self.masks = nn.ParameterList([
            nn.Parameter(torch.ones_like(p), requires_grad=False) for p in self.params])

    def forward(self, x):
        w1 = self.fc1.weight * self.masks[0]; b1 = self.fc1.bias * self.masks[1]
        w2 = self.fc2.weight * self.masks[2]; b2 = self.fc2.bias * self.masks[3]
        x = F.relu(F.linear(x, w1, b1))
        return F.linear(x, w2, b2)

    def density(self):
        total = sum(m.numel() for m in self.masks)
        active = sum(float(m.sum().item()) for m in self.masks)
        return active / total

    def clone_arch(self):
        new = MaskedMLP()
        for src, dst in zip(self.params, new.params):
            dst.data.copy_(src.data)
        for i in range(len(self.masks)):
            new.masks[i].data.copy_(self.masks[i].data)
        return new


# ----- Mutation (same unbiased qmut as UCI) -----
def _flip_one(model, current=1.0, target=0.0):
    li = np.random.randint(len(model.masks))
    m = model.masks[li]
    flat = m.data.flatten()
    cand = torch.where((flat > 0.5) if current == 1.0 else (flat < 0.5))[0]
    if len(cand) == 0: return None
    pick = int(cand[np.random.randint(len(cand))].item())
    idx = np.unravel_index(pick, m.shape)
    m.data[idx] = target
    return (li, idx)


def random_modif_quantum(model, n_flips=None, min_density=MIN_DENSITY):
    model = model.clone_arch()
    total = sum(m.numel() for m in model.masks)
    if n_flips is None:
        n_flips = max(1, int(0.02 * total))
    min_active = max(1, int(min_density * total))
    for _ in range(n_flips):
        active_count = sum(int(m.data.sum().item()) for m in model.masks)
        a = np.random.random()
        if a < 0.4 and 0 < active_count < total:
            removed = _flip_one(model, current=1.0, target=0.0)
            if removed is None: continue
            added = _flip_one(model, current=0.0, target=1.0)
            if added is None:
                model.masks[removed[0]].data[removed[1]] = 1.0
        elif a < 0.6 and active_count > min_active:
            _flip_one(model, current=1.0, target=0.0)
        else:
            if active_count < total:
                _flip_one(model, current=0.0, target=1.0)
    return model


def random_ini(density=F_TARGET):
    model = MaskedMLP()
    for i in range(len(model.masks)):
        model.masks[i].data = (torch.rand_like(model.masks[i]) < density).float()
    return model


def _collect_active_params(model):
    init_vals = []
    layout = []
    for p, m in zip(model.params, model.masks):
        idx = torch.where(m.data.flatten() > 0.5)[0]
        layout.append((p, idx, p.shape))
        if len(idx):
            init_vals.extend(p.data.flatten()[idx].tolist())
    return np.asarray(init_vals, dtype=np.float64), layout


def _apply_vec(model, x, layout):
    x_t = torch.as_tensor(x, dtype=torch.float32)
    pos = 0
    with torch.no_grad():
        for p, idx, shape in layout:
            n = idx.numel()
            if n == 0: continue
            flat = p.data.flatten()
            flat[idx] = x_t[pos:pos+n]
            p.data.copy_(flat.view(shape))
            pos += n
        for p, m in zip(model.params, model.masks):
            p.data.mul_(m.data)


def train_inner_cmaes(model, X, y, budget, sigma0=INNER_SIGMA0, seed=None):
    init_vec, layout = _collect_active_params(model)
    n = len(init_vec)
    if n == 0:
        return model
    def loss_fn(x):
        _apply_vec(model, x, layout)
        with torch.no_grad():
            v = F.cross_entropy(model(X), y).item()
        return float(v) if not np.isnan(v) else 1e6
    opts = {"maxfevals": budget, "verbose": -9,
            "seed": int(seed) if seed is not None else 0,
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


def score(model, alpha, B, seed, Xtr, ytr, Xte, yte):
    warnings.filterwarnings("ignore")
    model = train_inner_cmaes(model, Xtr, ytr, budget=B, seed=seed)
    acc_tr = eval_accuracy(model, Xtr, ytr)
    acc_te = eval_accuracy(model, Xte, yte)
    density = model.density()
    s = acc_te - alpha * (density - F_TARGET) ** 2
    if np.isnan(s): s = -1e6
    return s, acc_tr, acc_te, density, model


def train_evolutive(init_pop, n_epoch, alpha, B, n_jobs=N_JOBS, base_seed=0,
                    init_density=F_TARGET):
    history = {k: {"acc_te": [], "density": [], "loss": []}
               for k in ("best", "second", "third", "mid")}
    avg_density = []
    pop_size = len(init_pop)
    circuits = init_pop
    Xtr, ytr, Xte, yte = X_tr, y_tr, X_te, y_te
    for epoch in range(n_epoch):
        evaluated = Parallel(n_jobs=n_jobs)(
            delayed(score)(m, alpha, B,
                           base_seed * 10000 + epoch * pop_size + i,
                           Xtr, ytr, Xte, yte)
            for i, m in enumerate(circuits))
        scored = []
        for i, (s, atr, ate, d, trained) in enumerate(evaluated):
            scored.append((s, i, atr, ate, d))
            circuits[i] = trained
        scored.sort(reverse=True)
        for key, rank in (("best", 0), ("second", 1), ("third", 2),
                          ("mid", len(scored)//2)):
            history[key]["acc_te"].append(scored[rank][3])
            history[key]["density"].append(scored[rank][4])
            history[key]["loss"].append(-scored[rank][0])
        max_s = scored[0][0]
        lo, hi = ELITISM, pop_size - RESEED
        weights = np.array([max_s - t[0] + 1e-6 for t in scored[lo:hi]])
        probs = weights / weights.sum()
        picks = set(np.random.choice(range(lo, hi), size=RESEED,
                                     replace=False, p=probs).tolist())
        for i in range(1, RESEED + 1):
            circuits[scored[-i][1]] = random_ini(density=init_density)
        parent_ranks = [k // 2 for k in range(RESEED)]
        for pr, i in zip(parent_ranks, sorted(picks)):
            circuits[scored[i][1]] = random_modif_quantum(circuits[scored[pr][1]])
        for i in range(lo, hi):
            if i not in picks:
                circuits[scored[i][1]] = random_modif_quantum(circuits[scored[i][1]])
        avg_density.append(float(np.mean([m.density() for m in circuits])))
    best_model = circuits[scored[0][1]]
    return best_model, circuits, avg_density, history


def run_one(seed, B):
    set_all_seeds(seed)
    init_pop = [random_ini(density=F_TARGET) for _ in range(POP_SIZE)]
    t0 = time.time()
    best, pop, avg_d, hist = train_evolutive(
        init_pop, n_epoch=N_EPOCH, alpha=ALPHA, B=B, base_seed=seed,
        init_density=F_TARGET)
    elapsed = time.time() - t0
    return {
        "ds": "mnist", "seed": seed, "B": B, "elapsed": elapsed,
        "rho_pop_avg_last5": float(np.mean(avg_d[-5:])),
        "rho_best_final": float(hist["best"]["density"][-1]),
        "acc_best_final": float(hist["best"]["acc_te"][-1]),
        "avg_density_traj": avg_d,
        "best_density_traj": hist["best"]["density"],
        "best_acc_traj": hist["best"]["acc_te"],
    }


def main():
    print("Loading MNIST (downsampled 28x28 → 7x7) ...", flush=True)
    activate_dataset()
    n_params = sum(p.numel() for p in MaskedMLP().params)
    print(f"  X_tr {X_tr.shape}  X_te {X_te.shape}", flush=True)
    print(f"  Total parameters: {n_params}", flush=True)
    out = []
    runs = [(B, s) for B in B_LIST for s in SEEDS]
    print(f"{len(runs)} runs total ({len(B_LIST)} budgets × {len(SEEDS)} seeds)\n", flush=True)
    for idx, (B, seed) in enumerate(runs, 1):
        r = run_one(seed, B)
        out.append(r)
        print(f"[{idx:2d}/{len(runs)}] B={B:5d} seed={seed}  "
              f"rho_eq={r['rho_pop_avg_last5']:.4f}  "
              f"rho_best={r['rho_best_final']:.4f}  "
              f"acc={r['acc_best_final']:.4f}  ({r['elapsed']:.0f}s)",
              flush=True)
        with open("budget_sweep_mnist_results.pkl", "wb") as f:
            pickle.dump({"records": out, "B_list": B_LIST, "seeds": SEEDS,
                         "config": dict(POP_SIZE=POP_SIZE, N_EPOCH=N_EPOCH,
                                        ELITISM=ELITISM, RESEED=RESEED,
                                        F_TARGET=F_TARGET, ALPHA=ALPHA,
                                        TRAIN_SUBSET=TRAIN_SUBSET, VAL_SUBSET=VAL_SUBSET,
                                        IN_DIM=IN_DIM, HIDDEN=HIDDEN, OUT_DIM=OUT_DIM,
                                        DOWNSAMPLE=DOWNSAMPLE)}, f)
    print(f"\nSaved budget_sweep_mnist_results.pkl ({len(out)} records)", flush=True)
    plot(out)


def plot(records):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    means, stds, Bs = [], [], []
    for B in B_LIST:
        rs = [r["rho_pop_avg_last5"] for r in records if r["B"] == B]
        if not rs: continue
        means.append(np.mean(rs)); stds.append(np.std(rs))
        Bs.append(B)
    means = np.array(means); stds = np.array(stds)
    ax.errorbar(Bs, means, yerr=stds, marker='o', color='#205070',
                capsize=3, linewidth=2, markersize=7, label='pop-mean ρ (mean ± std)')
    for seed in SEEDS:
        rs = sorted([(r["B"], r["rho_pop_avg_last5"]) for r in records if r["seed"] == seed])
        if rs:
            xs, ys = zip(*rs)
            ax.plot(xs, ys, marker='.', color='#205070', alpha=0.3, linestyle=':')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Inner CMA-ES budget $B$ per individual')
    ax.set_ylabel('Equilibrium density $\\rho$ (pop. avg, last 5 epochs)')
    ax.set_title('MNIST downsampled 7×7 + small MLP (~970 params): $\\rho$ vs $B$')
    ax.axhline(F_TARGET, color='gray', linestyle='--', alpha=0.4, label=f'init $\\rho={F_TARGET}$')
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.4, label='dense $\\rho=1.0$')
    ax.grid(True, alpha=0.3); ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig("fig_budget_sweep_mnist.pdf", bbox_inches='tight')
    plt.savefig("fig_budget_sweep_mnist.png", dpi=140, bbox_inches='tight')
    print("Saved fig_budget_sweep_mnist.pdf and .png")


if __name__ == "__main__":
    main()
