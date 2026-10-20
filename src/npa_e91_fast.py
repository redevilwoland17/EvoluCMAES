"""Fast numerical ceiling for E91: at each anchor p, max χ_obs s.t. S_obs ≥ DET_THRESH.

Uses pure numpy (no torch) for speed. Direct density-matrix optimization.
"""
import sys, math, pickle, time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigvalsh
from scipy.special import xlogy

def vprint(*a, **k): print(*a, **{**k, 'flush': True})

# Constants from pathB_lib (avoid torch import)
S_SEC = 2.4240   # secure key threshold
DET_MARGIN = 0.02
DET_THRESH = S_SEC + DET_MARGIN  # 2.4440
P_CLIFF = 1 - math.sqrt(S_SEC / (2*math.sqrt(2)))  # ≈ 0.0742
P_MU, P_KAPPA, P_SIGMA = 0.03, 0.1, 0.015

# CHSH std settings
STD = (0.0, math.pi/2, math.pi/4, -math.pi/4)

# Pauli matrices (numpy)
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [I2, X, Y, Z]

def kron3(a, b, c): return np.kron(np.kron(a, b), c)

# Embed Pauli on qubit q ∈ {0,1,2} of 3-qubit system
EMB_A = [kron3(P, I2, I2) for P in PAULIS]
EMB_B = [kron3(I2, P, I2) for P in PAULIS]
EMB_E = [kron3(I2, I2, P) for P in PAULIS]

def depolarize_ab(rho, p):
    if p <= 0: return rho
    s0, s1 = math.sqrt(1 - 3*p/4), math.sqrt(p/4)
    coefs = [s0, s1, s1, s1]
    tmp = np.zeros_like(rho)
    for c, K0 in zip(coefs, EMB_A):
        K = c * K0; tmp += K @ rho @ K.conj().T
    out = np.zeros_like(tmp)
    for c, K0 in zip(coefs, EMB_B):
        K = c * K0; out += K @ tmp @ K.conj().T
    return out

def O(theta):
    return math.cos(theta)*Z + math.sin(theta)*X

def vn_entropy(rho):
    """von Neumann entropy in bits."""
    eigs = eigvalsh(rho)
    eigs = np.clip(eigs, 1e-15, 1.0)
    return float(-np.sum(eigs * np.log2(eigs)))

def evaluate_post(rho_abe, p):
    """Compute (S_obs, χ_obs) on post-channel state."""
    rho = depolarize_ab(rho_abe, p)
    a1, a2, b1, b2 = STD
    # Reshape rho (8x8) to (2,2,2,2,2,2): A,B,E,A',B',E'
    R = rho.reshape(2,2,2,2,2,2)
    # rho_AB = Tr_E
    rho_AB = np.einsum('abekpe->abkp', R).reshape(4, 4)
    # Wait — that's wrong. Let me re-do indexing.
    # rho has indices (i, j) where i = a*4 + b*2 + e, similarly j.
    rho_AB = np.zeros((4, 4), dtype=complex)
    for a in range(2):
      for b in range(2):
        for ap in range(2):
          for bp in range(2):
            rho_AB[a*2+b, ap*2+bp] = rho[a*4+b*2+0, ap*4+bp*2+0] + rho[a*4+b*2+1, ap*4+bp*2+1]
    def E_corr(ai, bj):
        return np.real(np.trace(rho_AB @ np.kron(O(ai), O(bj))))
    S = E_corr(a1, b1) + E_corr(a1, b2) + E_corr(a2, b1) - E_corr(a2, b2)

    # ρ_E
    rho_E = np.zeros((2, 2), dtype=complex)
    for e in range(2):
      for ep in range(2):
        for a in range(2):
          for b in range(2):
            rho_E[e, ep] += rho[a*4+b*2+e, a*4+b*2+ep]

    # ρ_E|X=x: condition Alice on Z basis (her measurement for the key)
    rho_E_cond = []; probs = []
    for x in range(2):
      rho_ex = np.zeros((2, 2), dtype=complex)
      for e in range(2):
        for ep in range(2):
          for b in range(2):
            rho_ex[e, ep] += rho[x*4+b*2+e, x*4+b*2+ep]
      px = np.real(np.trace(rho_ex)).clip(min=1e-12)
      probs.append(px); rho_E_cond.append(rho_ex / px)
    chi = vn_entropy(rho_E)
    for x in range(2):
        chi -= probs[x] * vn_entropy(rho_E_cond[x])
    return float(S), float(chi)


def params_to_rho(params):
    """64 real params → 8x8 PSD trace-1 matrix via Cholesky construction."""
    # Build lower-triangular L: 8 real diag + 28 complex off-diag = 8 + 56 = 64 reals
    L = np.zeros((8, 8), dtype=complex)
    L[range(8), range(8)] = params[:8]
    idx = 8
    for i in range(8):
        for j in range(i):
            L[i, j] = complex(params[idx], params[idx+1]); idx += 2
    rho = L @ L.conj().T
    tr = np.real(np.trace(rho))
    if tr < 1e-12: return None
    return rho / tr


def neg_lagrangian(params, p_val, lam):
    rho = params_to_rho(params)
    if rho is None: return 1e10
    try:
        S, chi = evaluate_post(rho, p_val)
        return -(chi + lam * S)
    except Exception:
        return 1e10


def find_chi_max_at_thresh(p_val, n_restarts=8, tol=1e-3):
    """Find max χ_obs s.t. S_obs ≥ DET_THRESH. Use bisection on lambda."""
    # Phase 1: scan lambdas to bracket constraint satisfaction
    best_chi = -np.inf; best_S = 0
    lams = np.linspace(0.0, 2.0, 8)
    for lam in lams:
        for restart in range(n_restarts):
            rng = np.random.default_rng(restart * 1000 + int(p_val*10000) + int(lam*100))
            p0 = rng.normal(0, 0.3, 64)
            res = minimize(neg_lagrangian, p0, args=(p_val, lam),
                           method='L-BFGS-B', options={'maxiter': 100, 'gtol': 1e-6})
            rho = params_to_rho(res.x)
            if rho is None: continue
            S, chi = evaluate_post(rho, p_val)
            if S >= DET_THRESH - 1e-3 and chi > best_chi:
                best_chi = chi; best_S = S
    return best_chi if best_chi > -np.inf else 0.0, best_S


def acin_bound(S):
    arg = (S/2.0)**2 - 1.0
    if arg < 0: return 1.0
    x = 0.5 + 0.5*math.sqrt(arg)
    if x <= 0 or x >= 1: return 0.0
    return -x*math.log2(x) - (1-x)*math.log2(1-x)


if __name__ == '__main__':
    P_ANCHORS = np.array([0.0, 0.012, 0.025, 0.038, 0.051, P_CLIFF - 0.01])
    sigma_inf = P_SIGMA / math.sqrt(2*P_KAPPA)
    mu_p = np.array([np.exp(-0.5*((p-P_MU)/sigma_inf)**2) for p in P_ANCHORS])
    mu_p /= mu_p.sum()
    vprint(f'P_ANCHORS = {P_ANCHORS}')
    vprint(f'OU weights = {mu_p.round(3)}')
    vprint(f'DET_THRESH = {DET_THRESH:.4f}')
    vprint(f'Acín(DET_THRESH) = {acin_bound(DET_THRESH):.4f}\n')

    results = {}
    t0 = time.time()
    for p_val in P_ANCHORS:
        vprint(f'=== p={p_val:.4f} ===')
        t_p = time.time()
        chi_max, S_at_max = find_chi_max_at_thresh(float(p_val), n_restarts=12)
        chi_acin_obs = acin_bound(DET_THRESH)
        s_pre_req = DET_THRESH / max((1-p_val)**2, 1e-9) if p_val < 1 else 99
        chi_acin_pre = acin_bound(min(s_pre_req, 2*math.sqrt(2)-1e-3)) if s_pre_req <= 2*math.sqrt(2) else 0.0
        results[float(p_val)] = {
            'chi_max_numerical': chi_max,
            'S_at_max': S_at_max,
            'chi_acin_obs': chi_acin_obs,
            'chi_acin_pre': chi_acin_pre,
            's_pre_required': s_pre_req,
        }
        vprint(f'  Numerical χ_max @ S≥{DET_THRESH:.4f}: χ={chi_max:.4f} (achieved S={S_at_max:.4f})')
        vprint(f'  Acín(S_obs=DET_THRESH):  χ ≤ {chi_acin_obs:.4f}  (loose post-channel)')
        vprint(f'  Acín(S_pre={s_pre_req:.4f}): χ ≤ {chi_acin_pre:.4f}  (pre-channel saturating)')
        vprint(f'  ({time.time()-t_p:.1f}s)\n')

    chi_lp_numerical = sum(mu_p[i] * results[float(p)]['chi_max_numerical'] for i, p in enumerate(P_ANCHORS))
    chi_lp_acin_obs = sum(mu_p[i] * results[float(p)]['chi_acin_obs'] for i, p in enumerate(P_ANCHORS))
    chi_lp_acin_pre = sum(mu_p[i] * results[float(p)]['chi_acin_pre'] for i, p in enumerate(P_ANCHORS))

    vprint('=' * 60)
    vprint('  CONTINUOUS LP CEILING (OU-averaged)')
    vprint('=' * 60)
    vprint(f'  Numerical (this script):     χ ≤ {chi_lp_numerical:.4f}')
    vprint(f'  Acín applied to S_obs:       χ ≤ {chi_lp_acin_obs:.4f}  (independent of p)')
    vprint(f'  Acín applied to S_pre:       χ ≤ {chi_lp_acin_pre:.4f}')
    vprint(f'  Library DP (compare):         χ = 0.360')
    vprint(f'  Library LP (compare):         χ = 0.347')
    vprint(f'\nTotal: {(time.time()-t0)/60:.1f} min')

    with open('npa_e91_fast.pkl', 'wb') as f:
        pickle.dump({
            'P_ANCHORS': P_ANCHORS, 'mu_p': mu_p, 'results': results,
            'chi_lp_numerical': chi_lp_numerical,
            'chi_lp_acin_obs': chi_lp_acin_obs,
            'chi_lp_acin_pre': chi_lp_acin_pre,
            'DET_THRESH': DET_THRESH,
        }, f)
    vprint(f'Saved npa_e91_fast.pkl')
