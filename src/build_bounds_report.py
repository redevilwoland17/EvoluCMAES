"""Build unified report comparing analytical and numerical Eve bounds for BB84 and E91."""
import os, math, pickle
import numpy as np

print('=' * 72)
print('  EVE EPISODE-MEAN UPPER BOUNDS — Per-step adaptive Eve, Holevo info')
print('  Attack class: SMART COLLECTIVE (adaptive, IID per-round, joint readout)')
print('=' * 72)


def load(path):
    if not os.path.exists(path): return None
    with open(path, 'rb') as f: return pickle.load(f)


# ─── E91 ───
print('\n[E91 / DIQKD bilateral depolarizing — DET_THRESH = 2.4440]')
print('-' * 72)
print('  ANALYTICAL CEILINGS:')
print('    Acín-BGMPS upper bound at S=DET_THRESH:  χ ≤ 0.6069')

em = load('episode_mean_bounds.pkl')
if em:
    print(f'    Acín-LP episode-mean (analytical):       χ ≤ {em["e91"]["chi_avg"]:.4f}')

print()
print('  NUMERICAL CEILINGS (16-dim Hilbert space, 2-qubit Eve ancilla):')

direct = load('npa_e91_direct.pkl')
if direct:
    print(f'    Direct SLSQP per-anchor χ_max(S≥DET_THRESH):')
    for p, chi in zip(direct['P_ANCHORS'], [direct['pareto_per_p'][i][5][1] if len(direct['pareto_per_p'][i]) > 5 else None for i in range(len(direct['P_ANCHORS']))]):
        if chi is not None:
            print(f'      p={p:.4f}: χ ≈ {chi:.4f}')
    if 'episode_mean_lp' in direct:
        lp = direct['episode_mean_lp']
        print(f'    Episode-mean LP (warm-started SLSQP):    χ ≤ {lp["chi_avg"]:.4f}')
        print(f'    (S_avg achieved = {lp["S_avg"]:.4f}, target {direct["DET_THRESH"]:.4f})')
else:
    print('    [npa_e91_direct.pkl not yet available]')

ext = load('npa_e91_extended.pkl')
if ext:
    print(f'    Extended (random init only):             χ ≤ {ext["episode_mean_lp"]["chi_avg"]:.4f} (local optima)')

print()
print(f'  LIBRARY-BASED:')
print(f'    Library DP (engineered):                 χ = 0.360')
print(f'    Library LP (engineered, mean-field):     χ = 0.347')


# ─── BB84 ───
print('\n[BB84 / prepare-and-measure bit-flip — F_THRESHOLD = 0.87]')
print('-' * 72)
print('  ANALYTICAL CEILINGS:')
print('    Cerf cloner upper bound at F_AB=F_THRESHOLD:  F_AE ≤ 0.8363  (clean-input cloner)')
if em:
    print(f'    Cerf-LP episode-mean (channel-cap-respecting): F_AE ≤ {em["bb84"]["fae_avg"]:.4f}')

print()
print('  NUMERICAL CEILINGS (4x4 Eve attack unitary):')

bbd = load('npa_bb84_direct.pkl')
if bbd:
    if 'episode_mean_lp' in bbd:
        lp = bbd['episode_mean_lp']
        print(f'    Episode-mean LP (direct SLSQP):           F_AE ≤ {lp["FE_avg"]:.4f}')
        print(f'    (F_AB_avg = {lp["F_avg"]:.4f}, target {bbd["F_THRESHOLD"]})')
        # Per-anchor F_AE at F_AB=F_THRESHOLD
        print(f'    Per-anchor F_AE at F_AB=F_THRESHOLD:')
        for i, p in enumerate(bbd['P_ANCHORS']):
            pts = bbd['pareto_per_p'][i]
            # Find F_AB closest to 0.87
            best = min(pts, key=lambda x: abs(x[0] - 0.87))
            print(f'      p_bf={p:.4f}: F_AE ≈ {best[1]:.4f} at F_AB={best[0]:.4f}')
else:
    print('    [npa_bb84_direct.pkl not yet available]')

bbe = load('npa_bb84_extended.pkl')
if bbe:
    print(f'    Extended (random init):                   F_AE ≤ {bbe["episode_mean_lp"]["FE_avg"]:.4f} (local optima)')

print()
print(f'  LIBRARY-BASED:')
print(f'    Library DP (engineered):                 F_AE = 0.7527')


# ─── Final summary ───
print('\n' + '=' * 72)
print('  SUMMARY TABLE — episode-mean ceilings')
print('=' * 72)

def fmt(v): return f'{v:.4f}' if v is not None else '   N/A '

e91_acin_lp = em['e91']['chi_avg'] if em else None
e91_num = direct['episode_mean_lp']['chi_avg'] if direct and 'episode_mean_lp' in direct else None
bb_cerf_lp = em['bb84']['fae_avg'] if em else None
bb_num = bbd['episode_mean_lp']['FE_avg'] if bbd and 'episode_mean_lp' in bbd else None

print(f'  {"":40s}  {"E91 (χ)":>10s}  {"BB84 (F_AE)":>14s}')
print('  ' + '-' * 68)
print(f'  {"Analytical envelope (Acín / Cerf)":40s}  {fmt(e91_acin_lp):>10s}  {fmt(bb_cerf_lp):>14s}')
print(f'  {"Numerical SLSQP (channel-respecting)":40s}  {fmt(e91_num):>10s}  {fmt(bb_num):>14s}')
print(f'  {"Library DP (engineered)":40s}  {0.360:>10.4f}  {0.7527:>14.4f}')

print('\nINTERPRETATION:')
print('- Analytical = Acín/Cerf applied to the post-channel state with episode-mean')
print('  constraint. Upper bound for ANY Eve strategy in the smart-collective class.')
print('- Numerical = direct optimization over Eve\'s prep state (E91) or attack')
print('  unitary (BB84) at each anchor. Tighter bound that respects what the channel')
print('  actually permits Eve to construct.')
print('- Library DP = optimal policy over engineered library actions only.')
print('- Gap (Numerical - Library) tells how much room remains for richer-ansatz Eve.')
