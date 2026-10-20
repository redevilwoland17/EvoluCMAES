"""Recover E91 LP from log file (after PchipInterpolator crash)."""
import re, math, pickle
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d

S_SEC = 2.4240
DET_THRESH = S_SEC + 0.02
P_CLIFF = 1 - math.sqrt(S_SEC / (2*math.sqrt(2)))
P_MU, P_KAPPA, P_SIGMA = 0.03, 0.1, 0.015

def acin(S):
    if S <= 2.0: return 1.0
    if S >= 2*math.sqrt(2): return 0.0
    arg = (S/2.0)**2 - 1.0
    x = 0.5 + 0.5*math.sqrt(arg)
    if x <= 0 or x >= 1: return 0.0
    return -x*math.log2(x) - (1-x)*math.log2(1-x)


# Parse log
log_path = 'npa_e91_direct.log'
with open(log_path) as f:
    text = f.read()

# Split by anchors
anchors = re.split(r'=== p=([\d.]+) ===', text)[1:]  # alternating: p_str, body, p_str, body, ...
data = {}
for i in range(0, len(anchors), 2):
    p = float(anchors[i])
    body = anchors[i+1]
    pts = []
    for m in re.finditer(r'S_target → S=([\d.]+), χ=([\d.]+),', body):
        S = float(m.group(1)); chi = float(m.group(2))
        if S > 0:  # filter failed SLSQP
            pts.append((S, chi))
    data[p] = pts

P_ANCHORS = sorted(data.keys())
print(f'Recovered {len(P_ANCHORS)} anchors')
for p in P_ANCHORS:
    print(f'  p={p:.4f}: {len(data[p])} valid points')


sigma_inf = P_SIGMA / math.sqrt(2*P_KAPPA)
mu_p = np.array([np.exp(-0.5*((p-P_MU)/sigma_inf)**2) for p in P_ANCHORS])
mu_p /= mu_p.sum()

# Build envelopes (dedup, monotone)
envs = []
for p in P_ANCHORS:
    pts = sorted(data[p], key=lambda x: x[0])
    # Dedup
    keep = [0]
    for i in range(1, len(pts)):
        if pts[i][0] > pts[keep[-1]][0] + 1e-6:
            keep.append(i)
    pts = [pts[i] for i in keep]
    S_arr = np.array([p[0] for p in pts])
    chi_arr = np.array([p[1] for p in pts])
    chi_env = np.maximum.accumulate(chi_arr[::-1])[::-1]
    f = interp1d(S_arr, chi_env, bounds_error=False, fill_value=(chi_env[0], 0.0))
    envs.append((S_arr, chi_env, f))


def opt_at_p(i, lam):
    S_arr, chi_env, f = envs[i]
    S_grid = np.linspace(S_arr[0], S_arr[-1], 500)
    chi_grid = f(S_grid)
    chi_grid[np.isnan(chi_grid)] = 0.0
    objs = chi_grid + lam * S_grid
    best_idx = np.argmax(objs)
    return S_grid[best_idx], chi_grid[best_idx]

def cr(lam):
    return sum(mu_p[i]*opt_at_p(i, lam)[0] for i in range(len(P_ANCHORS))) - DET_THRESH

f_lo = cr(0); f_hi = cr(20)
print(f'\nf(λ=0) = {f_lo:.4f}, f(λ=20) = {f_hi:.4f}')
if f_lo*f_hi > 0:
    lam_star = 0 if f_lo >= 0 else 20
else:
    lam_star = brentq(cr, 0, 20, xtol=1e-5)

S_star = np.array([opt_at_p(i, lam_star)[0] for i in range(len(P_ANCHORS))])
chi_star = np.array([opt_at_p(i, lam_star)[1] for i in range(len(P_ANCHORS))])
S_avg = float(np.sum(mu_p * S_star))
chi_avg = float(np.sum(mu_p * chi_star))

print(f'\n=== EPISODE-MEAN LP RESULT (E91) ===')
print(f'  λ* = {lam_star:.4f}')
print(f'  Per-anchor allocation:')
print(f'  {"p":>8s}  {"μ":>6s}  {"S*":>8s}  {"χ*":>8s}  {"Acín(S*)":>10s}')
for i, p in enumerate(P_ANCHORS):
    print(f'  {p:>8.4f}  {mu_p[i]:>6.3f}  {S_star[i]:>8.4f}  {chi_star[i]:>8.4f}  {acin(S_star[i]):>10.4f}')
print(f'\n  Episode-mean S = {S_avg:.4f} (target {DET_THRESH:.4f})')
print(f'  Episode-mean χ = {chi_avg:.4f}')
print(f'  Per-step OU-mean (constant S=DET_THRESH at every p):')
chi_thresh_avg = sum(mu_p[i]*opt_at_p(i, 100)[1] for i in range(len(P_ANCHORS)))  # forces S near max
# Actually compute per-step: at each anchor, the chi at S=DET_THRESH or nearest above
chi_perstep = []
for i, p in enumerate(P_ANCHORS):
    S_arr, chi_env, f = envs[i]
    if S_arr[-1] < DET_THRESH:
        chi_perstep.append(0)
    else:
        chi_perstep.append(float(f(DET_THRESH)))
chi_perstep_avg = sum(mu_p[i]*chi_perstep[i] for i in range(len(P_ANCHORS)))
print(f'  Per-step (S=DET_THRESH at every p): χ_avg = {chi_perstep_avg:.4f}')
print(f'  Acín(DET_THRESH): 0.6069')
print(f'  Library DP: 0.360')

with open('npa_e91_direct.pkl', 'wb') as f:
    pickle.dump({
        'P_ANCHORS': np.array(P_ANCHORS),
        'mu_p': mu_p,
        'pareto_per_p': [data[p] for p in P_ANCHORS],
        'episode_mean_lp': {
            'lambda_star': lam_star, 'S_per_p': S_star, 'chi_per_p': chi_star,
            'S_avg': S_avg, 'chi_avg': chi_avg,
        },
        'DET_THRESH': DET_THRESH,
        'chi_perstep_avg': chi_perstep_avg,
    }, f)
print('\nSaved npa_e91_direct.pkl')
