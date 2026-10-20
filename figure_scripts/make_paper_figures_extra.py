"""Generate fig_bb84_training.pdf (BB84 Rainbow-DQN, strict checkpoint score).

Run from any directory; the figure is written to the current directory:
    python figure_scripts/make_paper_figures_extra.py
"""
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import data  # noqa: E402

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'legend.fontsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family': 'serif',
})


# ─── FIG 4: BB84 Rainbow training curve ───
def fig_bb84_training():
    e7 = pickle.load(open(data('exp7_results_rainbow_v3.pkl'), 'rb'))
    h = e7.get('history', [])
    if not h:
        print('No exp_7 history — skipping')
        return
    steps = [r['step'] for r in h]
    chi   = [r.get('f_ae', r.get('chi', 0)) for r in h]
    det   = [r.get('det', 0)*100 for r in h]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].plot(steps, chi, color='#2CA02C', lw=1.5, label='Rainbow-DQN')
    axes[0].axhline(0.7434, color='black', ls='--', alpha=0.6, label='Oracle')
    axes[0].axhline(0.5000, color='gray',  ls=':',  alpha=0.5, label='Identity')
    axes[0].set_xlabel('Training steps'); axes[0].set_ylabel(r'$F_{AE}$')
    axes[0].set_title('BB84 Rainbow-DQN: $F_{AE}$ over training')
    axes[0].legend(loc='lower right', fontsize=9); axes[0].grid(alpha=0.25)

    axes[1].plot(steps, det, color='#C44536', lw=1.5)
    axes[1].set_xlabel('Training steps'); axes[1].set_ylabel('Detection rate (\\%)')
    axes[1].set_title('Detection rate during training')
    axes[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig('fig_bb84_training.pdf', bbox_inches='tight')
    plt.savefig('fig_bb84_training.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved fig_bb84_training')


if __name__ == '__main__':
    fig_bb84_training()
