"""Re-render fig_nnpruning_parsimony.pdf to match v4 §3.4.1.

Compares:
  - α=0 biased mutation (nnpruning3_results.pkl, group 'alpha_natural')   — STARS
  - α=0 unbiased mutation (nnpruning4_results.pkl, group 'alpha_natural_qmut') — CIRCLES

Per (dataset, seed) we plot one marker at (final-density, final-test-acc).
"""
import pickle, sys
import numpy as np
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import data  # noqa: E402

# Stub for unpickling MaskedMLP instances (we don't use them, but they must load)
class MaskedMLP(nn.Module):
    def __init__(self, dims=None):
        super().__init__()
        self.dims = list(dims) if dims is not None else None
sys.modules['__main__'].MaskedMLP = MaskedMLP

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'legend.fontsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

DATASETS = ['iris', 'wine', 'cancer']
COLOR    = {'iris': '#1F77B4', 'wine': '#2CA02C', 'cancer': '#C44536'}


def collect(file, group):
    """Return {dataset: {'rho': [..5..], 'acc': [..5..]}}."""
    d = pickle.load(open(data(file), 'rb'))
    g = d[group]
    out = {}
    for ds in DATASETS:
        rho, acc = [], []
        for seed_hist in g[ds]['hist']:
            best = seed_hist['best']
            rho.append(float(best['density'][-1]))
            acc.append(float(best['acc_te'][-1]))
        out[ds] = {'rho': rho, 'acc': acc}
    return out


biased   = collect('nnpruning3_results.pkl', 'alpha_natural')
unbiased = collect('nnpruning4_results.pkl', 'alpha_natural_qmut')

# Per-group summary
def summarise(label, data):
    print(f'\n=== {label} ===')
    print(f'{"dataset":>10s}  {"rho":>16s}  {"acc":>16s}')
    for ds in DATASETS:
        r = data[ds]['rho']; a = data[ds]['acc']
        print(f'{ds:>10s}  {np.mean(r):.3f} ± {np.std(r):.3f}    '
              f'{np.mean(a):.3f} ± {np.std(a):.3f}')

summarise('alpha=0 BIASED  (nnpruning3 / alpha_natural)',     biased)
summarise('alpha=0 UNBIASED (nnpruning4 / alpha_natural_qmut)', unbiased)


# Plot
fig, ax = plt.subplots(figsize=(6.5, 4))
for ds in DATASETS:
    ax.scatter(unbiased[ds]['rho'], unbiased[ds]['acc'],
               marker='o', s=55, color=COLOR[ds],
               edgecolor='black', linewidth=0.6,
               label=fr'{ds.title()} ($\alpha=0$, unbiased)', zorder=3)
    ax.scatter(biased[ds]['rho'], biased[ds]['acc'],
               marker='*', s=180, color=COLOR[ds],
               edgecolor='black', linewidth=0.8,
               label=fr'{ds.title()} ($\alpha=0$, biased)', zorder=4)

ax.axvline(0.05, color='red', ls=':', alpha=0.55, label=r'$\rho_\mathrm{target}=0.05$')

ax.set_xlabel(r'Final density $\rho$')
ax.set_ylabel('Test accuracy')
ax.set_title(r'Natural sparsity: dimensionality pressure $\rightarrow$ compact networks')
all_rho = sum([unbiased[ds]['rho'] + biased[ds]['rho'] for ds in DATASETS], [])
all_acc = sum([unbiased[ds]['acc'] + biased[ds]['acc'] for ds in DATASETS], [])
ax.set_xlim(min(all_rho) * 0.85, max(all_rho) * 1.10)
ax.set_ylim(min(all_acc) - 0.02, 1.01)
ax.legend(loc='lower right', ncol=2, fontsize=8, framealpha=0.85)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('fig_nnpruning_parsimony.pdf', bbox_inches='tight')
plt.savefig('fig_nnpruning_parsimony.png', bbox_inches='tight', dpi=150)
print('\nSaved fig_nnpruning_parsimony.{pdf,png}')
