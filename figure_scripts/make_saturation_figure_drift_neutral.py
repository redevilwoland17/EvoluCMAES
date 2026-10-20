"""Figure 8: density traces of the UCI alpha = 0 runs under the drift-neutral
operator (Appendix E; the counterpart of Figure 7, fig_natural_unbiased_density_drift).

Per-epoch population-mean density on the three UCI datasets, mean over seeds
with a +/- 1 standard deviation band, from the results of
src/mutation_operator_ablation_uci.py (nnpruning4_drift_neutral_results.pkl).
Runs stop early at different epochs, so later epochs average over the seeds
still running.

Data are read through _paths.data(); the figure is written to
fig_natural_driftneutral_density_drift.pdf (vector) and .png in the current
directory, like the other figure scripts.
Usage:  python figure_scripts/make_saturation_figure_drift_neutral.py
"""
import os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths

mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 9

OUT = os.getcwd()
NAME = "fig_natural_driftneutral_density_drift"
DATASETS = ['iris', 'wine', 'cancer']
COLORS = {'iris': '#cc4422', 'wine': '#7733aa', 'cancer': '#226688'}

with open(_paths.data("nnpruning4_drift_neutral_results.pkl"), 'rb') as f:
    records = pickle.load(f)['records']
traces = {ds: [r['avg_d'] for r in sorted(records, key=lambda r: r['seed']) if r['dataset'] == ds]
          for ds in DATASETS}

fig, ax = plt.subplots(figsize=(7, 4.5))
for ds in DATASETS:
    avg_d_list = traces[ds]
    max_len = max(len(t) for t in avg_d_list)
    arr = np.full((len(avg_d_list), max_len), np.nan)      # runs stop early at different epochs
    for i, t in enumerate(avg_d_list):
        arr[i, :len(t)] = t
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    ep = np.arange(1, max_len + 1)
    ax.plot(ep, mean, color=COLORS[ds], linewidth=2, label=f"{ds.capitalize()} ({len(avg_d_list)} seeds)")
    ax.fill_between(ep, mean - std, mean + std, color=COLORS[ds], alpha=0.15)

ax.axhline(0.05, color='red', linestyle=':', alpha=0.6,
           label=r'initial density $\rho_0 = 0.05$ ($=\rho_\mathrm{target}$ of the biased runs)')
ax.set_xlabel('Outer-loop epoch')
ax.set_ylabel(r'Average population density $\rho$')
ax.set_title(r"Drift-neutral mutation (0.40/0.30/0.30), $\alpha = 0$: density across epochs")
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=9)
ax.set_ylim(0, 0.30)
plt.tight_layout()
plt.savefig(os.path.join(OUT, NAME + '.png'), dpi=140, bbox_inches='tight')
plt.savefig(os.path.join(OUT, NAME + '.pdf'), bbox_inches='tight')
print(f"Saved {NAME}.pdf / .png")
