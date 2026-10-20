"""E91 numerical Pareto with library-warm-started optimization.

Key fix: random init wasn't finding Acín-saturating states. Initialize from
library cells (which we know reach near-Acín at their anchor) plus random
restarts.

Uses 2-qubit Eve ancilla (16-dim Hilbert space).
"""
import math, pickle, time, os, sys
import numpy as np
import torch
from scipy.optimize import minimize, brentq
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_default_dtype(torch.float64)

def vprint(*a, **k): print(*a, **{**k, 'flush': True})

# Constants
S_SEC = 2.4240
DET_THRESH = S_SEC + 0.02
P_CLIFF = 1 - math.sqrt(S_SEC / (2*math.sqrt(2)))
P_MU, P_KAPPA, P_SIGMA = 0.03, 0.1, 0.015
P_MIN, P_MAX = 0.0, P_CLIFF - 0.01

# Use the existing 16-dim evaluator
import importlib.util
spec = importlib.util.spec_from_file_location('m_ext', 'npa_e91_extended.py')
m_ext = importlib.util.module_from_spec(spec)
sys.modules['m_ext'] = m_ext
spec.loader.exec_module(m_ext)

evaluate_post = m_ext.evaluate_post   # (psi_16, p) → (S, chi)


def acin_chi(S):
    arg = (S/2.0)**2 - 1.0
    if arg < 0: return 1.0
    x = 0.5 + 0.5*math.sqrt(arg)
    if x <= 0 or x >= 1: return 0.0
    return -x*math.log2(x) - (1-x)*math.log2(1-x)


# Load library
from evoluspsa_e91 import circuit_to_unitary, STD_SETTINGS
lib = pickle.load(open('e91_library_pathB.pkl','rb'))

def lib_state_at_p(lib_cell):
    """Return embedded |psi⟩ in 16-dim from library cell."""
    U = circuit_to_unitary(lib_cell['circuit'])
    psi_8 = U @ torch.zeros(8, dtype=torch.complex128).index_fill(0, torch.tensor([0]), 1.0)
    psi_8 = psi_8.numpy().astype(complex)
    psi_16 = np.zeros(16, dtype=complex)
    for i in range(8):
        a = i // 4; b = (i // 2) % 2; e1 = i % 2
        psi_16[a*8 + b*4 + e1*2 + 0] = psi_8[i]
    return psi_16


def psi_to_params(psi):
    """Inverse of params_to_psi."""
    return np.concatenate([psi.real, psi.imag])


def params_to_psi(params):
    psi = params[:16] + 1j*params[16:]
    n = np.linalg.norm(psi)
    if n < 1e-9: return None
    return psi / n


def neg_lagrangian(params, p_val, lam):
    psi = params_to_psi(params)
    if psi is None: return 1e10
    try:
        S, chi = evaluate_post(psi, p_val)
        return -(chi + lam * S)
    except Exception:
        return 1e10


def pareto_at_p_warm(p_val, n_lambdas=12, n_random_restarts=10):
    """Pareto with both library warm-starts and random restarts."""
    lambdas = np.concatenate([np.linspace(0, 0.5, n_lambdas//2),
                              np.linspace(0.6, 4.0, n_lambdas - n_lambdas//2)])

    # Library warm-starts: each library cell at its anchor and ALL anchors
    warm_inits = []
    for key, cell in lib.items():
        psi16 = lib_state_at_p(cell)
        warm_inits.append(psi_to_params(psi16))
    # Add a small perturbation around each
    perturbed = []
    for w in warm_inits:
        rng = np.random.default_rng(42)
        perturbed.append(w + 0.01 * rng.normal(size=32))
    warm_inits = warm_inits + perturbed

    pts = []
    for lam in lambdas:
        best_obj = -np.inf; best_S = 0; best_chi = 0
        # Library warm starts
        for w in warm_inits:
            res = minimize(neg_lagrangian, w, args=(p_val, lam),
                           method='L-BFGS-B', options={'maxiter': 300, 'gtol': 1e-9})
            psi = params_to_psi(res.x)
            if psi is None: continue
            S, chi = evaluate_post(psi, p_val)
            obj = chi + lam*S
            if obj > best_obj:
                best_obj = obj; best_S = S; best_chi = chi
        # Random restarts
        for restart in range(n_random_restarts):
            rng = np.random.default_rng(restart * 1000 + int(p_val*10000) + int(lam*100))
            p0 = rng.normal(0, 0.5, 32)
            res = minimize(neg_lagrangian, p0, args=(p_val, lam),
                           method='L-BFGS-B', options={'maxiter': 300, 'gtol': 1e-9})
            psi = params_to_psi(res.x)
            if psi is None: continue
            S, chi = evaluate_post(psi, p_val)
            obj = chi + lam*S
            if obj > best_obj:
                best_obj = obj; best_S = S; best_chi = chi
        pts.append((best_S, best_chi, lam))
    return pts


def solve_episode_mean_lp(pareto_per_p, P_GRID, mu, thresh):
    pareto_envs = []
    for pts in pareto_per_p:
        sorted_pts = sorted(pts, key=lambda x: x[0])
        S_arr = np.array([p[0] for p in sorted_pts])
        chi_arr = np.array([p[1] for p in sorted_pts])
        chi_env = np.maximum.accumulate(chi_arr[::-1])[::-1]
        f = interp1d(S_arr, chi_env, bounds_error=False, fill_value=(chi_env[0], 0.0))
        pareto_envs.append((S_arr, chi_env, f))

    def opt_at_p(i, lam):
        S_arr, chi_env, f = pareto_envs[i]
        S_grid = np.linspace(S_arr[0], S_arr[-1], 200)
        chi_grid = f(S_grid)
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
    return {'lambda_star': lam_star,
            'S_per_p': S_star, 'chi_per_p': chi_star,
            'S_avg': float(np.sum(mu*S_star)),
            'chi_avg': float(np.sum(mu*chi_star))}


if __name__ == '__main__':
    P_ANCHORS = np.array([0.0, 0.012, 0.025, 0.038, 0.051, P_CLIFF - 0.01])
    sigma_inf = P_SIGMA / math.sqrt(2*P_KAPPA)
    mu_p = np.array([np.exp(-0.5*((p-P_MU)/sigma_inf)**2) for p in P_ANCHORS])
    mu_p /= mu_p.sum()
    vprint(f'P_ANCHORS = {P_ANCHORS}')
    vprint(f'OU weights = {mu_p.round(3)}')
    vprint(f'2-qubit Eve, library warm-start + 10 random restarts per λ\n')

    pareto_per_p = []
    t0 = time.time()
    for p_val in P_ANCHORS:
        vprint(f'=== p={p_val:.4f} ===')
        t_p = time.time()
        pts = pareto_at_p_warm(float(p_val))
        pareto_per_p.append(pts)
        # Show
        for S, chi, lam in pts:
            ac = acin_chi(S) if S >= 0 else 0
            vprint(f'  λ={lam:.3f}: S={S:.4f}, χ={chi:.4f}, Acín(S)={ac:.4f}, gap={ac-chi:+.4f}')
        feasible = [(S,chi) for (S,chi,lam) in pts if S >= DET_THRESH - 1e-3]
        chi_thresh = max((c for s,c in feasible), default=0.0)
        vprint(f'  → χ_max(S≥DET_THRESH) = {chi_thresh:.4f}  ({time.time()-t_p:.1f}s)\n')

    vprint('=' * 70)
    vprint('  EPISODE-MEAN LP (warm-started empirical Pareto)')
    vprint('=' * 70)
    lp = solve_episode_mean_lp(pareto_per_p, P_ANCHORS, mu_p, DET_THRESH)
    vprint(f'  λ* = {lp["lambda_star"]:.4f}')
    vprint(f'  Per-p:')
    for i, p in enumerate(P_ANCHORS):
        vprint(f'    p={p:.4f}: S*={lp["S_per_p"][i]:.4f}  χ*={lp["chi_per_p"][i]:.4f}')
    vprint(f'\n  Episode-mean S = {lp["S_avg"]:.4f} (target {DET_THRESH:.4f})')
    vprint(f'  Episode-mean χ = {lp["chi_avg"]:.4f}')
    vprint(f'  Acín analytical: 0.6069')
    vprint(f'  Library DP:      0.360')
    vprint(f'\nTotal: {(time.time()-t0)/60:.1f} min')

    with open('npa_e91_warmstart.pkl', 'wb') as f:
        pickle.dump({'P_ANCHORS': P_ANCHORS, 'mu_p': mu_p,
                     'pareto_per_p': pareto_per_p, 'episode_mean_lp': lp,
                     'DET_THRESH': DET_THRESH}, f)
    vprint('Saved npa_e91_warmstart.pkl')
