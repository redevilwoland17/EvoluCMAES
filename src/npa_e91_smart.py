"""Smart E91 ceiling: search over a richer Eve attack ansatz than density-matrix Cholesky.

Strategy: Eve's pre-channel state is parametrized as a Stinespring-purified state
|ψ_ABE⟩ in C^8. We sample 16 real parameters (8 complex amplitudes, normalized).
This gives a pure state — for a fully purified attack this is general enough.

The bilateral channel + measurement pipeline is the same as numpy E91 evaluator.
"""
import sys, math, pickle, time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import eigvalsh

def vprint(*a, **k): print(*a, **{**k, 'flush': True})

S_SEC = 2.4240
DET_THRESH = S_SEC + 0.02
P_CLIFF = 1 - math.sqrt(S_SEC / (2*math.sqrt(2)))
P_MU, P_KAPPA, P_SIGMA = 0.03, 0.1, 0.015

STD = (0.0, math.pi/2, math.pi/4, -math.pi/4)
I2 = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
PAULIS = [I2, X, Y, Z]

def kron3(a,b,c): return np.kron(np.kron(a,b),c)

EMB_A = np.stack([kron3(P, I2, I2) for P in PAULIS])
EMB_B = np.stack([kron3(I2, P, I2) for P in PAULIS])

def depolarize_ab(rho, p):
    if p <= 0: return rho
    s0, s1 = math.sqrt(1 - 3*p/4), math.sqrt(p/4)
    coefs = np.array([s0, s1, s1, s1])
    tmp = np.zeros_like(rho)
    for i in range(4):
        K = coefs[i] * EMB_A[i]
        tmp = tmp + K @ rho @ K.conj().T
    out = np.zeros_like(tmp)
    for i in range(4):
        K = coefs[i] * EMB_B[i]
        out = out + K @ tmp @ K.conj().T
    return out

def O(theta):
    return math.cos(theta)*Z + math.sin(theta)*X

# Pre-compute kron for CHSH measurement operators
def kron2(a,b): return np.kron(a,b)
M_CHSH = []
for ai in [STD[0], STD[1]]:
    for bj in [STD[2], STD[3]]:
        M_CHSH.append(kron2(O(ai), O(bj)))
# M_CHSH order: (a0,b0), (a0,b1), (a1,b0), (a1,b1)
# CHSH = E_00 + E_01 + E_10 - E_11

def vn_entropy(rho):
    eigs = eigvalsh(rho)
    eigs = np.clip(eigs, 1e-15, 1.0)
    return float(-np.sum(eigs * np.log2(eigs)))

def evaluate_post_pure(psi, p):
    """psi: 8-dim complex (already normalized). Compute (S_obs, χ_obs) post-channel."""
    rho_pre = np.outer(psi, psi.conj())
    rho = depolarize_ab(rho_pre, p)

    # ρ_AB = Tr_E (8x8 → 4x4): trace over E qubit (3rd qubit, indices flip last bit)
    rho_AB = np.zeros((4, 4), dtype=complex)
    for a in range(2):
      for b in range(2):
        for ap in range(2):
          for bp in range(2):
            rho_AB[a*2+b, ap*2+bp] = rho[a*4+b*2+0, ap*4+bp*2+0] + rho[a*4+b*2+1, ap*4+bp*2+1]
    E00 = np.real(np.trace(rho_AB @ M_CHSH[0]))
    E01 = np.real(np.trace(rho_AB @ M_CHSH[1]))
    E10 = np.real(np.trace(rho_AB @ M_CHSH[2]))
    E11 = np.real(np.trace(rho_AB @ M_CHSH[3]))
    S = E00 + E01 + E10 - E11

    rho_E = np.zeros((2,2), dtype=complex)
    for e in range(2):
      for ep in range(2):
        for a in range(2):
          for b in range(2):
            rho_E[e,ep] += rho[a*4+b*2+e, a*4+b*2+ep]
    rho_E_cond = []; probs = []
    for x in range(2):
      rho_ex = np.zeros((2,2), dtype=complex)
      for e in range(2):
        for ep in range(2):
          for b in range(2):
            rho_ex[e,ep] += rho[x*4+b*2+e, x*4+b*2+ep]
      px = max(np.real(np.trace(rho_ex)), 1e-12)
      probs.append(px); rho_E_cond.append(rho_ex / px)
    chi = vn_entropy(rho_E)
    for x in range(2):
        chi -= probs[x] * vn_entropy(rho_E_cond[x])
    return float(S), float(chi)


def params_to_psi(params):
    """16 reals → 8-dim complex unit vector."""
    psi = params[:8] + 1j*params[8:]
    n = np.linalg.norm(psi)
    if n < 1e-9: return None
    return psi / n


def neg_lagrangian(params, p_val, lam):
    psi = params_to_psi(params)
    if psi is None: return 1e10
    try:
        S, chi = evaluate_post_pure(psi, p_val)
        return -(chi + lam * S)
    except Exception:
        return 1e10


def find_chi_max_at_thresh(p_val, n_restarts=20):
    best_chi = -np.inf; best_S = 0
    lams = np.concatenate([np.linspace(0, 0.5, 6), np.linspace(0.6, 2.5, 8)])
    for lam in lams:
        for restart in range(n_restarts):
            rng = np.random.default_rng(restart * 1000 + int(p_val*10000) + int(lam*100))
            p0 = rng.normal(0, 0.5, 16)
            res = minimize(neg_lagrangian, p0, args=(p_val, lam),
                           method='L-BFGS-B', options={'maxiter': 200, 'gtol': 1e-7})
            psi = params_to_psi(res.x)
            if psi is None: continue
            S, chi = evaluate_post_pure(psi, p_val)
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
        chi_max, S_at_max = find_chi_max_at_thresh(float(p_val), n_restarts=20)
        chi_acin_obs = acin_bound(DET_THRESH)
        s_pre_req = DET_THRESH / max((1-p_val)**2, 1e-9)
        chi_acin_pre = acin_bound(min(s_pre_req, 2*math.sqrt(2)-1e-3)) if s_pre_req <= 2*math.sqrt(2) else 0.0
        results[float(p_val)] = {
            'chi_max_numerical': chi_max,
            'S_at_max': S_at_max,
            'chi_acin_obs': chi_acin_obs,
            'chi_acin_pre': chi_acin_pre,
        }
        vprint(f'  Numerical (pure-state) χ_max @ S≥{DET_THRESH:.4f}: χ={chi_max:.4f} (S={S_at_max:.4f})')
        vprint(f'  Acín(S_obs):  χ ≤ {chi_acin_obs:.4f}')
        vprint(f'  Acín(S_pre={s_pre_req:.4f}): χ ≤ {chi_acin_pre:.4f}')
        vprint(f'  ({time.time()-t_p:.1f}s)\n')

    chi_lp_numerical = sum(mu_p[i] * results[float(p)]['chi_max_numerical'] for i, p in enumerate(P_ANCHORS))
    chi_lp_acin_obs = sum(mu_p[i] * results[float(p)]['chi_acin_obs'] for i, p in enumerate(P_ANCHORS))
    chi_lp_acin_pre = sum(mu_p[i] * results[float(p)]['chi_acin_pre'] for i, p in enumerate(P_ANCHORS))

    vprint('=' * 60)
    vprint('  CONTINUOUS LP CEILING (OU-averaged)')
    vprint('=' * 60)
    vprint(f'  Numerical (pure-state):       χ ≤ {chi_lp_numerical:.4f}')
    vprint(f'  Acín(S_obs=DET_THRESH):       χ ≤ {chi_lp_acin_obs:.4f}')
    vprint(f'  Acín(S_pre=DET_THRESH/(1-p)²): χ ≤ {chi_lp_acin_pre:.4f}')
    vprint(f'  Library DP (compare):           χ = 0.360')
    vprint(f'  Library LP (compare):           χ = 0.347')
    vprint(f'\nTotal: {(time.time()-t0)/60:.1f} min')

    with open('npa_e91_smart.pkl', 'wb') as f:
        pickle.dump({
            'P_ANCHORS': P_ANCHORS, 'mu_p': mu_p, 'results': results,
            'chi_lp_numerical': chi_lp_numerical,
            'chi_lp_acin_obs': chi_lp_acin_obs,
            'chi_lp_acin_pre': chi_lp_acin_pre,
            'DET_THRESH': DET_THRESH,
        }, f)
    vprint(f'Saved npa_e91_smart.pkl')
