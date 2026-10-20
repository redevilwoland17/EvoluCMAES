"""Direct measurement of the convergence horizon as a function of ρ.

Probes the Proposition's *mechanism* without running the outer evolutionary
loop. For each fixed mask density ρ:
  1. Sample several random masks of that density.
  2. Run inner CMA-ES on each mask with a *very* large per-individual budget
     B_max, recording the trajectory of best test accuracy as a function of
     fitness-evals consumed.
  3. Plot mean trajectory per ρ — small ρ should plateau early, large ρ
     plateaus much later (or doesn't, within B_max).

The variance-gap claim of Proposition 1 says: at fixed B, smaller ρ has
*lower variance* in ̂fB because it has already converged, while larger ρ
is still climbing. This experiment makes that visible directly: the *seed
spread* (band) at fixed B is the variance, and the *trajectory shape* shows
how convergence proceeds.

Setup:
  - MNIST downsampled 7×7 (49 dims), MLP 49→16→10 (970 params)
  - ρ ∈ {0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80}
  - 5 random masks per ρ
  - B_max = 50_000 fitness evaluations per inner CMA-ES run
  - Per-eval ≈ 0.21 ms → per-run ≈ 10 s, total ≈ 6 min

Output:  convergence_horizon_mnist_results.pkl
         fig_convergence_horizon_mnist.pdf
"""
import os, time, pickle, random, warnings
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
import cma
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse infra from the budget sweep
import budget_sweep_mnist as bs

RHO_LIST     = [0.05, 0.20, 0.50, 1.00]   # 4 networks: small (target) + 3 bigger
SEEDS        = [0, 1, 2, 3, 4]
B_MAX        = 50_000
INNER_SIGMA0 = 0.3
RECORD_EVERY = 5            # log best-acc-so-far every K evals
N_JOBS       = 8


def fixed_mask_model(rho, seed):
    """Sample one fixed mask of (approximate) density ρ; init params at random."""
    rng = np.random.RandomState(seed)
    m = bs.MaskedMLP()
    for i, mask in enumerate(m.masks):
        flat = mask.data.flatten()
        n = flat.numel()
        k = max(1, int(round(rho * n)))
        idx = rng.choice(n, size=k, replace=False)
        new = torch.zeros_like(flat)
        new[idx] = 1.0
        mask.data = new.view_as(mask.data)
    return m


def run_one(rho, seed):
    """Run inner CMA-ES at fixed mask, record (n_eval, best_test_acc) trajectory."""
    warnings.filterwarnings("ignore")
    bs.set_all_seeds(seed)
    bs.activate_dataset()
    Xtr, ytr, Xte, yte = bs.X_tr, bs.y_tr, bs.X_te, bs.y_te
    m = fixed_mask_model(rho, seed)
    init_vec, layout = bs._collect_active_params(m)
    n_active = len(init_vec)
    if n_active == 0:
        return {'rho': rho, 'seed': seed, 'evals': [0], 'acc': [0.1],
                'final_acc': 0.1, 'n_active': 0}

    best_acc = -np.inf
    evals_log, acc_log = [], []
    n_eval = [0]

    def loss(x):
        bs._apply_vec(m, x, layout)
        with torch.no_grad():
            l = F.cross_entropy(m(Xtr), ytr).item()
        n_eval[0] += 1
        # Record best test acc seen so far at sparse intervals
        if n_eval[0] % RECORD_EVERY == 0 or n_eval[0] == 1:
            with torch.no_grad():
                acc = float((m(Xte).argmax(dim=1) == yte).float().mean().item())
            nonlocal best_acc
            if acc > best_acc:
                best_acc = acc
            evals_log.append(n_eval[0])
            acc_log.append(best_acc)
        return float(l) if not np.isnan(l) else 1e6

    es = cma.CMAEvolutionStrategy(
        init_vec.tolist(), INNER_SIGMA0,
        {'maxfevals': B_MAX, 'verbose': -9, 'tolx': 1e-10, 'tolfun': 1e-10,
         'seed': int(seed)},
    )
    es.optimize(loss)
    # final eval at best-x
    best_x = es.result.xbest if es.result.xbest is not None else init_vec
    bs._apply_vec(m, best_x, layout)
    with torch.no_grad():
        final_acc = float((m(Xte).argmax(dim=1) == yte).float().mean().item())
    return {
        'rho': rho, 'seed': seed,
        'n_active': n_active,
        'evals': evals_log, 'acc': acc_log,
        'final_acc': final_acc,
    }


def main():
    print(f"Convergence-horizon experiment on MNIST 7×7 + small MLP",
          f"\n  ρ ∈ {RHO_LIST}",
          f"\n  {len(SEEDS)} seeds × {len(RHO_LIST)} ρ values = {len(RHO_LIST)*len(SEEDS)} runs",
          f"\n  B_max = {B_MAX} per run",
          flush=True)
    runs = [(rho, s) for rho in RHO_LIST for s in SEEDS]
    t0 = time.time()
    out = Parallel(n_jobs=N_JOBS)(
        delayed(run_one)(rho, s) for rho, s in runs)
    print(f"All {len(out)} runs done ({time.time()-t0:.0f}s).", flush=True)

    with open("convergence_horizon_mnist_results.pkl", "wb") as f:
        pickle.dump({'records': out, 'rho_list': RHO_LIST, 'seeds': SEEDS,
                     'B_max': B_MAX, 'record_every': RECORD_EVERY}, f)
    print(f"Saved convergence_horizon_mnist_results.pkl", flush=True)
    plot(out)


def plot(records):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=np.log2(min(RHO_LIST)), vmax=np.log2(max(RHO_LIST)))

    # Common geometric eval grid for interpolation
    eval_grid = np.unique(np.geomspace(RECORD_EVERY, B_MAX, 300).astype(int))

    # ----- Build per-(ρ, seed) max-acc-so-far trajectories -----
    summary = {}        # rho → {'mean': arr_len_grid, 'std', 'max_acc_per_seed': list}
    for rho in RHO_LIST:
        rs = [r for r in records if r['rho'] == rho]
        accs_grid, max_accs = [], []
        n_active = rs[0]['n_active'] if rs else None
        for r in rs:
            ev = np.asarray(r['evals']); ac = np.asarray(r['acc'])
            if len(ev) == 0: continue
            # acc[] is already cumulative max; just interpolate
            accs_grid.append(np.interp(eval_grid, ev, ac,
                                       left=ac[0], right=ac[-1]))
            max_accs.append(float(ac[-1]))   # max-so-far at end == max overall
        if not accs_grid:
            continue
        accs_grid = np.stack(accs_grid)
        summary[rho] = {
            'eval_grid': eval_grid,
            'mean': accs_grid.mean(0), 'std': accs_grid.std(0),
            'all_traj': accs_grid,
            'max_acc_per_seed': max_accs,
            'mean_max_acc': float(np.mean(max_accs)),
            'n_active': n_active,
        }

    # ----- (A) Trajectories: best-test-acc-so-far vs B per ρ -----
    ax = axes[0]
    for rho in RHO_LIST:
        if rho not in summary: continue
        s = summary[rho]
        ax.plot(eval_grid, s['mean'], color=cmap(norm(np.log2(rho))),
                linewidth=2, label=f"ρ={rho:.2f} (d={s['n_active']}, max_acc={s['mean_max_acc']:.3f})")
        ax.fill_between(eval_grid, s['mean'] - s['std'], s['mean'] + s['std'],
                        color=cmap(norm(np.log2(rho))), alpha=0.15)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Inner CMA-ES budget $B$ (fitness evals)')
    ax.set_ylabel('Best test accuracy reached within $B$ evals (max-so-far)')
    ax.set_title('(A) Convergence trajectories per ρ')
    ax.grid(True, alpha=0.3); ax.legend(loc='lower right', fontsize=7)
    ax.set_ylim(0, 0.85)

    # ----- (B) B_required(ρ) for targets defined by small-ρ max-acc -----
    ax2 = axes[1]
    SOURCE_RHOS = [RHO_LIST[0]]                   # smallest ρ sets the target
    target_set = []
    for src_rho in SOURCE_RHOS:
        if src_rho not in summary: continue
        target_set.append((src_rho, summary[src_rho]['mean_max_acc']))

    print("\n=== B_required(ρ) for targets set by small-ρ max accuracy ===")
    print(f"{'target ρ_src':<12} {'target acc':>12}  " +
          "".join(f"  ρ={r:.2f}" for r in RHO_LIST))
    for src_rho, target_acc in target_set:
        Bs_to_reach = []
        rhos_used = []
        row = []
        for rho_q in RHO_LIST:
            if rho_q < src_rho: row.append("    -"); continue
            if rho_q not in summary:
                row.append("    -")
                continue
            # average across seeds: B_required = first B at which seed traj hits target
            B_per_seed = []
            for traj in summary[rho_q]['all_traj']:
                idx = np.where(traj >= target_acc)[0]
                if len(idx) == 0:
                    B_per_seed.append(B_MAX)   # didn't reach within B_max
                else:
                    B_per_seed.append(int(eval_grid[idx[0]]))
            B_med = int(np.median(B_per_seed))
            Bs_to_reach.append(B_med); rhos_used.append(rho_q)
            row.append(f"{B_med:6d}")
        print(f"  ρ_src={src_rho:.2f}  acc={target_acc:.3f}    " + "  ".join(row))
        if Bs_to_reach:
            ax2.plot(rhos_used, Bs_to_reach, marker='o', linewidth=2,
                     label=f'target = max_acc(ρ={src_rho:.2f}) = {target_acc:.3f}')

    ax2.set_xlabel('Mask density ρ (test density, ≥ source ρ)')
    ax2.set_ylabel('Median budget $B^{*}$ needed to reach target accuracy')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=2)
    ax2.set_title('(B) Convergence horizon $B^{*}(\\rho)$ for small-ρ targets')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='upper left', fontsize=8)

    fig.suptitle('Convergence horizon on MNIST 7×7 + small MLP (970 params), 5 seeds')
    plt.tight_layout()
    plt.savefig("fig_convergence_horizon_mnist.pdf", bbox_inches='tight')
    plt.savefig("fig_convergence_horizon_mnist.png", dpi=140, bbox_inches='tight')
    print("Saved fig_convergence_horizon_mnist.pdf and .png", flush=True)


if __name__ == "__main__":
    main()
