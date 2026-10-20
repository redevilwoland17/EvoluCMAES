"""Generate fig_pareto.pdf (E91 library cells against the noiseless Acín envelope).

Run from any directory; the figure is written to the current directory:
    python figure_scripts/make_paper_figures.py
"""
import math, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import data  # noqa: E402
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
                     'legend.fontsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})

# ─── FIG 3: E91 library Pareto with Acín envelope at p=0 ───
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
S_grid = np.linspace(0, 2*math.sqrt(2), 200)
def acin(S):
    if S < 2: return 1.0
    if S > 2*math.sqrt(2): return 0.0
    arg = (S/2)**2 - 1
    x = 0.5 + 0.5*math.sqrt(arg)
    return -x*math.log2(x) - (1-x)*math.log2(1-x) if 0 < x < 1 else 0.0
ax.plot(S_grid, [acin(s) for s in S_grid], 'k--', label=r'Acín envelope (noiseless)', alpha=0.7)

# Library cells at p=0 (from e91_library_pathB_7levels.pkl)
try:
    lib = pickle.load(open(data('e91_library_pathB_7levels.pkl'), 'rb'))
    p0_S, p0_chi = [], []
    for (p_anc, lvl), entry in lib.items():
        if abs(p_anc) < 1e-6:
            p0_S.append(entry['S_at_p']); p0_chi.append(entry['chi_at_p'])
    ax.scatter(p0_S, p0_chi, color='#3478B5', s=60, label='EvoluCMAES library cells (at $p=0$)', zorder=5)
except Exception as e: print(f'library plot: {e}')

# Mark DET_THRESH
ax.axvline(2.444, color='red', ls=':', alpha=0.6, label='DET_THRESH (E91)')
# Mark library DP
ax.scatter([2.448], [0.350], marker='*', s=180, color='#2A9D8F', zorder=6, label='Library DP (OU-averaged)')

ax.set_xlabel(r'CHSH violation $S$')
ax.set_ylabel(r"Eve's information $\chi(A:E)$")
ax.set_title('E91 Pareto: EvoluCMAES library vs Acín envelope (at $p=0$)')
ax.set_xlim(1.5, 2.85)
ax.set_ylim(0, 1.05)
ax.legend(loc='upper right', fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig_pareto.pdf', bbox_inches='tight')
plt.savefig('fig_pareto.png', bbox_inches='tight', dpi=150)
print('Saved fig_pareto')
