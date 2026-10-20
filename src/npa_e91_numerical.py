"""Numerical (S, χ) Pareto envelope for E91 with bilateral depolarizing.

For each anchor p, maximize χ_obs over all pre-channel ρ_ABE (PSD trace-1) under the
constraint S_obs ≥ S_target. This is the Pareto envelope for "any unitary every step"
Eve attacks (collective). Compare to Acín bound applied to S_obs.

Output: pkl with (p, S_target, χ_max_numerical, χ_acin_at_S_obs).
"""
import os, math, pickle, time
import numpy as np
import torch
from scipy.optimize import minimize, brentq
from scipy.special import xlogy

torch.set_default_dtype(torch.float64)

import pathB_lib as L
from pathB_lib import compute_holevo_and_S_noisy, P_INTERP, P_CLIFF, P_MU, P_KAPPA, P_SIGMA
from evoluspsa_e91 import STD_SETTINGS, secure_key_rate

S_SEC = brentq(lambda s: secure_key_rate(s) - 1e-6, 2.01, 2.8)
DET_THRESH = S_SEC + 0.02

C = torch.complex128

def acin_bound(S):
    """Analytical Acín-Brunner-Gisin upper bound on χ(A:E) given CHSH S."""
    arg = (S/2.0)**2 - 1.0
    if arg < 0: return 1.0
    x = 0.5 + 0.5*math.sqrt(arg)
    if x <= 0 or x >= 1: return 0.0
    return -x*math.log2(x) - (1-x)*math.log2(1-x)


# Parametrize 8x8 ρ_ABE pre-channel: ρ = L L^† / Tr(L L^†), L = lower-triangular complex
# 8x8 lower triangular = 8 real diagonal + 8*7/2 = 28 complex off-diag = 36+28=64 real params.
# Total 72 real params; we'll search a structured subspace via random restarts.

def params_to_rho(p):
    """Reshape param vector → density matrix via Cholesky-style construction."""
    # Use 8x8 complex L where L[i,i] is real, L[i,j] (i>j) is complex.
    # Real params: 8 diag + 28*2 = 64 → total 72 floats.
    L = torch.zeros(8, 8, dtype=C)
    idx = 0
    for i in range(8):
        L[i, i] = float(p[idx])  # real diagonal
        idx += 1
    for i in range(8):
        for j in range(i):
            re = float(p[idx]); im = float(p[idx+1]); idx += 2
            L[i, j] = complex(re, im)
    rho = L @ L.conj().T
    tr = torch.trace(rho).real
    if tr.item() < 1e-10:
        return None
    return rho / tr


def evaluate_post_channel(rho_abe, p, settings):
    """Apply bilateral Λ_p to ρ_ABE, then compute (S_obs, χ_obs)."""
    # Reuse compute_holevo_and_S_noisy logic but starting from a density matrix instead of a unitary.
    # We'll inline the depolarize + observable computation.
    from pathB_lib import depolarize_ab
    from evoluspsa_e91 import obs as _pauli_obs, von_neumann_entropy
    rho = depolarize_ab(rho_abe, p)
    a1 = torch.tensor(settings[0], dtype=torch.float64)
    a2 = torch.tensor(settings[1], dtype=torch.float64)
    b1 = torch.tensor(settings[2], dtype=torch.float64)
    b2 = torch.tensor(settings[3], dtype=torch.float64)
    rho_AB = torch.zeros(4, 4, dtype=C)
    for a in range(2):
        for b in range(2):
            for ap in range(2):
                for bp in range(2):
                    rho_AB[a*2+b, ap*2+bp] = (
                        rho[a*4+b*2+0, ap*4+bp*2+0] + rho[a*4+b*2+1, ap*4+bp*2+1]
                    )
    def E_corr(ai, bj):
        return torch.trace(rho_AB @ torch.kron(_pauli_obs(ai), _pauli_obs(bj))).real
    S = E_corr(a1, b1) + E_corr(a1, b2) + E_corr(a2, b1) - E_corr(a2, b2)
    rho_E = torch.zeros(2, 2, dtype=C)
    for e in range(2):
        for ep in range(2):
            for a in range(2):
                for b in range(2):
                    rho_E[e, ep] += rho[a*4+b*2+e, a*4+b*2+ep]
    rho_E_cond, probs = [], []
    for x in range(2):
        rho_ex = torch.zeros(2, 2, dtype=C)
        for e in range(2):
            for ep in range(2):
                for b in range(2):
                    rho_ex[e, ep] += rho[x*4+b*2+e, x*4+b*2+ep]
        px = torch.trace(rho_ex).real.clamp(min=1e-12)
        probs.append(px); rho_E_cond.append(rho_ex / px)
    chi = von_neumann_entropy(rho_E)
    for x in range(2):
        chi = chi - probs[x] * von_neumann_entropy(rho_E_cond[x])
    return float(S.item()), float(chi.item())


def negative_lagrangian(params, p_val, lam):
    """Negative Lagrangian L = -χ - λ·(S - threshold). Maximize χ s.t. S ≥ thresh.
    For Pareto, parametrize: max [χ + lam·S]."""
    rho = params_to_rho(params)
    if rho is None: return 1e10
    try:
        S, chi = evaluate_post_channel(rho, p_val, STD_SETTINGS)
        return -(chi + lam * S)
    except Exception:
        return 1e10


def search_pareto_at_p(p_val, n_lam=8, n_restarts=10, n_steps=200):
    """Sweep λ, return (S, χ) frontier points."""
    lams = np.concatenate([np.linspace(0, 0.3, n_lam//2),
                           np.linspace(0.4, 2.0, n_lam - n_lam//2)])
    pts = []
    for lam in lams:
        best_obj = -np.inf; best_pt = None
        for restart in range(n_restarts):
            rng = np.random.default_rng(restart * 1000 + int(p_val*10000) + int(lam*100))
            p0 = rng.normal(0, 0.3, 64)
            res = minimize(
                negative_lagrangian, p0, args=(p_val, lam),
                method='L-BFGS-B', options={'maxiter': n_steps, 'gtol': 1e-7}
            )
            if -res.fun > best_obj:
                best_obj = -res.fun
                rho = params_to_rho(res.x)
                S, chi = evaluate_post_channel(rho, p_val, STD_SETTINGS)
                best_pt = (S, chi, lam)
        if best_pt is not None:
            pts.append(best_pt)
    return pts


if __name__ == '__main__':
    # OU stationary
    sigma_inf = P_SIGMA / math.sqrt(2*P_KAPPA)
    P_ANCHORS = np.array([0.0, 0.012, 0.025, 0.038, 0.051, P_CLIFF - 0.01])
    print(f'P_ANCHORS = {P_ANCHORS}')
    print(f'DET_THRESH = {DET_THRESH:.4f}')
    print(f'Acín(DET_THRESH) = {acin_bound(DET_THRESH):.4f}\n')

    results = {}
    t0 = time.time()
    for p_val in P_ANCHORS:
        print(f'\n=== p = {p_val:.4f} ===')
        t_p = time.time()
        pts = search_pareto_at_p(float(p_val), n_lam=10, n_restarts=15, n_steps=150)
        # Also: max χ subject to S ≥ DET_THRESH
        feasible = [(S, chi) for (S, chi, lam) in pts if S >= DET_THRESH]
        chi_at_thresh = max([chi for (S, chi) in feasible]) if feasible else 0.0
        # Acín applied to S_obs achievable at this p
        # Eve constrained to S_obs ≥ DET_THRESH; for honest pair S_obs = (1-p)²·2√2.
        s_honest = (1 - p_val)**2 * 2*math.sqrt(2)
        chi_acin_naive = acin_bound(min(DET_THRESH, s_honest))
        # Acín applied to pre-channel: at p, Eve's pre-channel must have S_pre = DET_THRESH/(1-p)²
        if p_val < 1:
            s_pre_required = DET_THRESH / (1-p_val)**2
            chi_acin_prechannel = acin_bound(s_pre_required) if s_pre_required <= 2*math.sqrt(2) else 0.0
        else:
            chi_acin_prechannel = 0.0
        results[float(p_val)] = {
            'pareto_pts': pts, 'chi_at_DET_THRESH': chi_at_thresh,
            'chi_acin_post': chi_acin_naive,
            'chi_acin_pre': chi_acin_prechannel,
            's_honest': s_honest,
        }
        print(f'  Numerical χ_max(S=DET_THRESH) = {chi_at_thresh:.4f}')
        print(f'  Acín(DET_THRESH) = {chi_acin_naive:.4f}  (post-channel, loose)')
        print(f'  Acín(S_pre=DET/{(1-p_val)**2:.3f}²) = {chi_acin_prechannel:.4f}  (pre-channel ceiling, ignoring Λ_A on χ)')
        print(f'  ({time.time()-t_p:.1f}s)')
    print(f'\nTotal: {(time.time()-t0)/60:.1f} min')

    # OU-weighted average
    mu_p = np.array([np.exp(-0.5*((p-P_MU)/sigma_inf)**2) for p in P_ANCHORS])
    mu_p /= mu_p.sum()
    chi_lp_numerical = sum(mu_p[i] * results[float(p)]['chi_at_DET_THRESH']
                           for i, p in enumerate(P_ANCHORS))
    chi_lp_acin_post = sum(mu_p[i] * results[float(p)]['chi_acin_post']
                            for i, p in enumerate(P_ANCHORS))
    print(f'\n== Continuous LP ceiling (OU-averaged) ==')
    print(f'  Numerical (NPA-style):  χ ≤ {chi_lp_numerical:.4f}')
    print(f'  Acín(S_obs):            χ ≤ {chi_lp_acin_post:.4f}  (overestimate)')
    print(f'  Library DP (reference):  χ = 0.360')
    print(f'  Library LP (reference):  χ = 0.347')

    with open('npa_e91_bounds.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'P_ANCHORS': P_ANCHORS,
            'mu_p': mu_p,
            'DET_THRESH': DET_THRESH,
            'chi_lp_numerical': chi_lp_numerical,
            'chi_lp_acin_post': chi_lp_acin_post,
        }, f)
    print(f'\nSaved npa_e91_bounds.pkl')
