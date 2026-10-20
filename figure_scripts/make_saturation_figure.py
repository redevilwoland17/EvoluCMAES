"""Figure 7: fig_natural_unbiased_density_drift.png/.pdf (Appendix E).

Per-epoch average population density under the density-agnostic operator
(qmut, swap/remove/add 0.40/0.20/0.40), α=0, three UCI datasets,
5 seeds each. Source pickle: nnpruning4_results.pkl (alpha_natural_qmut group).
"""
import pickle, numpy as np
import torch.nn as nn
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
mpl.rcParams['legend.fontsize'] = 9


# Stub class for unpickling
class MaskedMLP(nn.Module):
    def __init__(self, dims=None, **kw):
        super().__init__()
        self.dims = list(dims) if dims is not None else None


import sys
sys.modules['__main__'].MaskedMLP = MaskedMLP

with open(data('nnpruning4_results.pkl'), 'rb') as f:
    r4 = pickle.load(f)

DATASETS = ['iris', 'wine', 'cancer']
COLORS = {'iris': '#cc4422', 'wine': '#7733aa', 'cancer': '#226688'}

fig, ax = plt.subplots(figsize=(7, 4.5))

for ds in DATASETS:
    # avg_d is per-seed list of per-epoch population-mean densities
    avg_d_list = r4['alpha_natural_qmut'][ds]['avg_d']
    # Pad/align to common epoch length
    max_len = max(len(t) for t in avg_d_list)
    arr = np.full((len(avg_d_list), max_len), np.nan)
    for i, t in enumerate(avg_d_list):
        arr[i, :len(t)] = t
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    ep = np.arange(1, max_len + 1)
    ax.plot(ep, mean, color=COLORS[ds], linewidth=2, label=ds.capitalize())
    ax.fill_between(ep, mean - std, mean + std, color=COLORS[ds], alpha=0.15)

ax.axhline(0.05, color='red', linestyle=':', alpha=0.6,
           label=r'$\rho_\mathrm{target}=0.05$ (biased reference)')
ax.set_xlabel('Outer-loop epoch')
ax.set_ylabel(r'Average population density $\rho$')
ax.set_title(r'Unbiased mutation, $\alpha = 0$: density drift across epochs')
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=9)
ax.set_ylim(0, 0.30)

plt.tight_layout()
plt.savefig('fig_natural_unbiased_density_drift.png', dpi=140, bbox_inches='tight')
plt.savefig('fig_natural_unbiased_density_drift.pdf', bbox_inches='tight')
print("Saved fig_natural_unbiased_density_drift.png / .pdf")
