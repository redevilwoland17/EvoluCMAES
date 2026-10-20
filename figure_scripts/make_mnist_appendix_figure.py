"""Paper-ready 2-panel figure for the MNIST appendix.

Panel A: equilibrium ρ vs B from the full 60-epoch sweep — combines
  - budget_sweep_mnist_60epochs_results.pkl                     (B in {2^7..2^11})
  - budget_sweep_mnist_60epochs_extended_low_results.pkl        (B in {2^12..2^14})
  - budget_sweep_mnist_60epochs_extended_high_results.pkl       (B in {2^15..2^16})

Panel B: convergence trajectories per fixed ρ from convergence_horizon_mnist.
"""
import pickle, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import data  # noqa: E402
import matplotlib as mpl

mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 8

# Load the three sweep halves and merge
records = []
for path in ['budget_sweep_mnist_60epochs_results.pkl',
             'budget_sweep_mnist_60epochs_extended_low_results.pkl',
             'budget_sweep_mnist_60epochs_extended_high_results.pkl']:
    with open(data(path), 'rb') as f:
        d = pickle.load(f)
    records.extend(d['records'])

with open(data('convergence_horizon_mnist_results.pkl'), 'rb') as f:
    ch = pickle.load(f)
ch_records = ch['records']
RHO_LIST = ch['rho_list']
B_MAX = ch['B_max']
RECORD_EVERY = ch['record_every']


def agg_rho_eq(records):
    Bs = sorted({r['B'] for r in records})
    means, stds = [], []
    for B in Bs:
        rho = [r['rho_pop_avg_last5'] for r in records if r['B'] == B]
        means.append(np.mean(rho)); stds.append(np.std(rho))
    return np.array(Bs), np.array(means), np.array(stds)


B_arr, means, stds = agg_rho_eq(records)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# (A) ρ_eq vs B
ax = axes[0]
ax.errorbar(B_arr, means, yerr=stds, marker='o', color='#205070', capsize=3,
            linewidth=2, markersize=7,
            label=r'$\rho_\mathrm{eq}$ (mean $\pm$ std, 3 seeds)')
seeds = sorted({r['seed'] for r in records})
for seed in seeds:
    pts = sorted([(r['B'], r['rho_pop_avg_last5']) for r in records if r['seed'] == seed])
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker='.', color='#205070', alpha=0.3, linestyle=':')
ax.set_xscale('log', base=2)
ax.set_xlabel(r'Inner CMA-ES budget $B$ per individual')
ax.set_ylabel(r'Equilibrium density $\rho_\mathrm{eq}$')
ax.axhline(0.10, color='gray', linestyle='--', alpha=0.4, label=r'init $\rho=0.10$')
ax.set_title(r'(A) Budget sweep: $\rho_\mathrm{eq}$ vs $B$')
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=8)
ax.set_ylim(0.10, 0.27)

# (B) Convergence trajectories per fixed ρ
ax = axes[1]
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=np.log2(min(RHO_LIST)), vmax=np.log2(max(RHO_LIST)))
eval_grid = np.unique(np.geomspace(RECORD_EVERY, B_MAX, 300).astype(int))

for rho in RHO_LIST:
    rs = [r for r in ch_records if r['rho'] == rho]
    accs_grid = []
    for r in rs:
        ev = np.asarray(r['evals']); ac = np.asarray(r['acc'])
        if len(ev) == 0: continue
        accs_grid.append(np.interp(eval_grid, ev, ac, left=ac[0], right=ac[-1]))
    if not accs_grid: continue
    accs_grid = np.stack(accs_grid)
    mean = accs_grid.mean(0); std = accs_grid.std(0)
    n_active = rs[0]['n_active']
    color = cmap(norm(np.log2(rho)))
    ax.plot(eval_grid, mean, color=color, linewidth=2,
            label=fr'$\rho={rho:.2f}$ ($d={n_active}$, ceil $={mean[-1]:.2f}$)')
    ax.fill_between(eval_grid, mean - std, mean + std, color=color, alpha=0.15)
ax.set_xscale('log', base=2)
ax.set_xlabel(r'Inner CMA-ES budget $B$ (fitness evals)')
ax.set_ylabel(r'Best test accuracy reached within $B$')
ax.set_title(r'(B) Convergence horizon per fixed $\rho$')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=8)
ax.set_ylim(0, 0.85)

plt.suptitle(r'MNIST $7\times 7$ + small MLP (970 params): budget sweep')
plt.tight_layout()
plt.savefig('fig_budget_sweep_mnist_appendix.pdf', bbox_inches='tight')
plt.savefig('fig_budget_sweep_mnist_appendix.png', dpi=140, bbox_inches='tight')
print("Saved fig_budget_sweep_mnist_appendix.pdf / .png")

# Also print the merged numbers for the table
print('\n=== Merged 60-epoch sweep across 10 budgets ===')
print(f"{'B':>7} {'mean ρ':>8} {'std ρ':>8} {'mean acc':>9} {'n seeds'}")
for B, m, s in zip(B_arr, means, stds):
    accs = [r['acc_best_final'] for r in records if r['B'] == B]
    print(f"{B:>7} {m:>8.4f} {s:>8.4f} {np.mean(accs):>9.3f} {len(accs)}")
