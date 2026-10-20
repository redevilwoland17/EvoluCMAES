"""Direct SLSQP solve: max χ s.t. S = S_target at each (p, S_target).
Then build the Pareto curve at each p, then solve episode-mean LP.

Uses 2-qubit Eve ancilla and library warm-starts.
"""
import math, pickle, time, os, sys
import numpy as np
import torch
from scipy.optimize import minimize, brentq
from scipy.interpolate import interp1d, PchipInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_default_dtype(torch.float64)

def vprint(*a, **k): print(*a, **{**k, 'flush': True})

S_SEC = 2.4240
DET_THRESH = S_SEC + 0.02
P_CLIFF = 1 - math.sqrt(S_SEC / (2*math.sqrt(2)))
P_MU, P_KAPPA, P_SIGMA = 0.03, 0.1, 0.015
P_MIN, P_MAX = 0.0, P_CLIFF - 0.01

import importlib.util
spec = importlib.util.spec_from_file_location('m_ext', 'npa_e91_extended.py')
m_ext = importlib.util.module_from_spec(spec)
sys.modules['m_ext'] = m_ext
spec.loader.exec_module(m_ext)
evaluate_post = m_ext.evaluate_post


def acin_chi(S):
    if S <= 2.0: return 1.0
    if S >= 2*math.sqrt(2): return 0.0
    arg = (S/2.0)**2 - 1.0
    x = 0.5 + 0.5*math.sqrt(arg)
    if x <= 0 or x >= 1: return 0.0
    return -x*math.log2(x) - (1-x)*math.log2(1-x)


from evoluspsa_e91 import circuit_to_unitary
lib = pickle.load(open('e91_library_pathB.pkl','rb'))

def lib_state_16(cell):
    U = circuit_to_unitary(cell['circuit'])
    psi_8 = (U @ torch.zeros(8, dtype=torch.complex128).index_fill(0, torch.tensor([0]), 1.0)).numpy().astype(complex)
    psi_16 = np.zeros(16, dtype=complex)
    for i in range(8):
        a = i // 4; b = (i // 2) % 2; e1 = i % 2
        psi_16[a*8 + b*4 + e1*2 + 0] = psi_8[i]
    return psi_16

def psi_to_params(psi):
    return np.concatenate([psi.real, psi.imag])

def params_to_psi(params):
    psi = params[:16] + 1j*params[16:]
    n = np.linalg.norm(psi)
    if n < 1e-9: return None
    return psi / n


def neg_chi(params, p_val):
    psi = params_to_psi(params)
    if psi is None: return 1e10
    try:
        S, chi = evaluate_post(psi, p_val)
        return -chi
    except: return 1e10

def s_at(params, p_val):
    psi = params_to_psi(params)
    if psi is None: return -1e10
    try:
        S, chi = evaluate_post(psi, p_val)
        return S
    except: return -1e10


def max_chi_at_s_target(p_val, S_target, n_inits=20, tol=1e-3):
    """SLSQP: max χ s.t. S = S_target."""
    best_chi = -np.inf; best_S = 0
    # Library warm-starts
    inits = []
    for cell in lib.values():
        inits.append(psi_to_params(lib_state_16(cell)))
    # Random extras
    rng = np.random.default_rng(int(p_val*10000) + int(S_target*1000))
    for _ in range(n_inits - len(inits)):
        inits.append(rng.normal(0, 0.5, 32))
    for p0 in inits:
        try:
            res = minimize(
                neg_chi, p0, args=(p_val,),
                method='SLSQP',
                constraints=[{'type': 'eq', 'fun': lambda x, p=p_val, s=S_target: s_at(x, p) - s}],
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            psi = params_to_psi(res.x)
            if psi is None: continue
            S_actual, chi = evaluate_post(psi, p_val)
            if abs(S_actual - S_target) < tol and chi > best_chi:
                best_chi = chi; best_S = S_actual
        except Exception:
            continue
    return best_chi if best_chi > -np.inf else 0.0, best_S


def pareto_curve_at_p(p_val, S_targets):
    """Return list of (S, chi) on the Pareto curve at fixed p."""
    pts = []
    for S_t in S_targets:
        chi, S = max_chi_at_s_target(p_val, S_t, n_inits=20)
        pts.append((S, chi))
    return pts


def solve_episode_mean_lp(pareto_per_p, P_GRID, mu, thresh):
    pareto_envs = []
    for pts in pareto_per_p:
        sorted_pts = sorted(pts, key=lambda x: x[0])
        S_arr = np.array([p[0] for p in sorted_pts])
        chi_arr = np.array([p[1] for p in sorted_pts])
        # Use PCHIP interp for smooth concave-like envelope
        chi_env = np.maximum.accumulate(chi_arr[::-1])[::-1]
        f = PchipInterpolator(S_arr, chi_env, extrapolate=False)
        pareto_envs.append((S_arr, chi_env, f))

    def opt_at_p(i, lam):
        S_arr, chi_env, f = pareto_envs[i]
        S_grid = np.linspace(S_arr[0], S_arr[-1], 500)
        chi_grid = f(S_grid)
        chi_grid[np.isnan(chi_grid)] = 0.0
        objs = chi_grid + lam * S_grid
        best_idx = np.argmax(objs)
        return S_grid[best_idx], chi_grid[best_idx]

    def cr(lam):
        return sum(mu[i]*opt_at_p(i, lam)[0] for i in range(len(P_GRID))) - thresh

    f_lo = cr(0); f_hi = cr(20)
    if f_lo*f_hi > 0:
        lam_star = 0 if f_lo >= 0 else 20
    else:
        lam_star = brentq(cr, 0, 20, xtol=1e-5)
    S_star = np.array([opt_at_p(i, lam_star)[0] for i in range(len(P_GRID))])
    chi_star = np.array([opt_at_p(i, lam_star)[1] for i in range(len(P_GRID))])
    return {'lambda_star': lam_star, 'S_per_p': S_star, 'chi_per_p': chi_star,
            'S_avg': float(np.sum(mu*S_star)), 'chi_avg': float(np.sum(mu*chi_star))}


if __name__ == '__main__':
    P_ANCHORS = np.array([0.0, 0.012, 0.025, 0.038, 0.051, P_CLIFF - 0.01])
    sigma_inf = P_SIGMA / math.sqrt(2*P_KAPPA)
    mu_p = np.array([np.exp(-0.5*((p-P_MU)/sigma_inf)**2) for p in P_ANCHORS])
    mu_p /= mu_p.sum()

    # Pareto sweep targets — dense around DET_THRESH plus low/high anchors
    S_TARGETS = [2.0, 2.20, 2.30, 2.40, 2.43, DET_THRESH, 2.46, 2.50, 2.55, 2.65, 2.75, 2.80]
    vprint(f'P_ANCHORS = {P_ANCHORS}')
    vprint(f'S_TARGETS = {S_TARGETS}')
    vprint(f'OU weights = {mu_p.round(3)}\n')

    pareto_per_p = []
    t0 = time.time()
    for p_val in P_ANCHORS:
        vprint(f'=== p={p_val:.4f} ===')
        t_p = time.time()
        pts = pareto_curve_at_p(float(p_val), S_TARGETS)
        pareto_per_p.append(pts)
        for S, chi in pts:
            ac = acin_chi(S) if S > 0 else 1.0
            vprint(f'  S_target → S={S:.4f}, χ={chi:.4f}, Acín={ac:.4f}, gap={ac-chi:+.4f}')
        feasible = [(S,chi) for S,chi in pts if S >= DET_THRESH - 1e-3]
        chi_thresh = max((c for s,c in feasible), default=0.0)
        vprint(f'  → χ_max(S≥DET_THRESH) ≈ {chi_thresh:.4f}  ({time.time()-t_p:.1f}s)\n')

    vprint('=' * 70)
    vprint('  EPISODE-MEAN LP (direct SLSQP Pareto)')
    vprint('=' * 70)
    lp = solve_episode_mean_lp(pareto_per_p, P_ANCHORS, mu_p, DET_THRESH)
    vprint(f'  λ* = {lp["lambda_star"]:.4f}')
    for i, p in enumerate(P_ANCHORS):
        ac = acin_chi(lp['S_per_p'][i])
        vprint(f'    p={p:.4f}: S*={lp["S_per_p"][i]:.4f}  χ*={lp["chi_per_p"][i]:.4f}  Acín={ac:.4f}')
    vprint(f'\n  Episode-mean S = {lp["S_avg"]:.4f} (target {DET_THRESH:.4f})')
    vprint(f'  Episode-mean χ = {lp["chi_avg"]:.4f}')
    vprint(f'  Acín analytical: 0.6069')
    vprint(f'  Library DP:      0.360')
    vprint(f'\nTotal: {(time.time()-t0)/60:.1f} min')

    with open('npa_e91_direct.pkl', 'wb') as f:
        pickle.dump({'P_ANCHORS': P_ANCHORS, 'mu_p': mu_p, 'S_TARGETS': S_TARGETS,
                     'pareto_per_p': pareto_per_p, 'episode_mean_lp': lp,
                     'DET_THRESH': DET_THRESH}, f)
    vprint('Saved npa_e91_direct.pkl')
