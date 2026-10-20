"""Extended E91 ceiling with 2-qubit Eve ancilla — tests whether the 1-qubit Eve spec is restricting Eve.

ρ_ABE^pre lives in (A=2)·(B=2)·(E=4) = 16-dim Hilbert space.
Pure-state Eve (Stinespring purification): |ψ⟩ ∈ ℂ^16 → 32 real params.

Bilateral depolarizing on A and B (E left untouched), then compute (S_obs, χ_obs).
Search per-anchor (S, χ) Pareto curve via Lagrangian sweep.
Then solve episode-mean LP using the empirical Pareto.
"""
import math, pickle, time
import numpy as np
from scipy.optimize import minimize, brentq, minimize_scalar
from scipy.linalg import eigvalsh

def vprint(*a, **k): print(*a, **{**k, 'flush': True})

# Constants
S_SEC = 2.4240
DET_THRESH = S_SEC + 0.02
P_CLIFF = 1 - math.sqrt(S_SEC / (2*math.sqrt(2)))
P_MU, P_KAPPA, P_SIGMA = 0.03, 0.1, 0.015
P_MIN, P_MAX = 0.0, P_CLIFF - 0.01

STD = (0.0, math.pi/2, math.pi/4, -math.pi/4)
I2 = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
PAULIS = [I2, X, Y, Z]

# Eve has 2 qubits, so total Hilbert space = A(2) ⊗ B(2) ⊗ E(4) = 16-dim
def emb_a(P):
    return np.kron(np.kron(P, I2), np.eye(4, dtype=complex))
def emb_b(P):
    return np.kron(np.kron(I2, P), np.eye(4, dtype=complex))
EMB_A = [emb_a(P) for P in PAULIS]
EMB_B = [emb_b(P) for P in PAULIS]

def depolarize_ab(rho, p):
    if p <= 0: return rho
    s0, s1 = math.sqrt(1 - 3*p/4), math.sqrt(p/4)
    coefs = [s0, s1, s1, s1]
    tmp = np.zeros_like(rho)
    for i in range(4):
        K = coefs[i] * EMB_A[i]
        tmp = tmp + K @ rho @ K.conj().T
    out = np.zeros_like(tmp)
    for i in range(4):
        K = coefs[i] * EMB_B[i]
        out = out + K @ tmp @ K.conj().T
    return out

def O_op(theta):
    return math.cos(theta)*Z + math.sin(theta)*X

# CHSH measurement: kron of single-qubit observables
M_CHSH = []
for ai in [STD[0], STD[1]]:
    for bj in [STD[2], STD[3]]:
        M_CHSH.append(np.kron(O_op(ai), O_op(bj)))

def vn_entropy(rho):
    eigs = eigvalsh(rho)
    eigs = np.clip(eigs, 1e-15, 1.0)
    return float(-np.sum(eigs * np.log2(eigs)))


def evaluate_post(psi, p):
    """psi: 16-dim complex unit vector. (A,B,E1,E2) qubits."""
    rho_pre = np.outer(psi, psi.conj())
    rho = depolarize_ab(rho_pre, p)

    # Indices: rho[i, j] where i, j ∈ {0..15}; layout: (a,b,e1,e2) → i = a·8 + b·4 + e1·2 + e2.
    # ρ_AB = Tr_E (sum over (e1, e2) diagonal).
    rho_AB = np.zeros((4, 4), dtype=complex)
    for a in range(2):
      for b in range(2):
        for ap in range(2):
          for bp in range(2):
            s = 0+0j
            for e1 in range(2):
              for e2 in range(2):
                s += rho[a*8+b*4+e1*2+e2, ap*8+bp*4+e1*2+e2]
            rho_AB[a*2+b, ap*2+bp] = s
    E00 = np.real(np.trace(rho_AB @ M_CHSH[0]))
    E01 = np.real(np.trace(rho_AB @ M_CHSH[1]))
    E10 = np.real(np.trace(rho_AB @ M_CHSH[2]))
    E11 = np.real(np.trace(rho_AB @ M_CHSH[3]))
    S = E00 + E01 + E10 - E11

    # ρ_E = Tr_AB (sum over (a, b) diagonal). 4x4 matrix.
    rho_E = np.zeros((4, 4), dtype=complex)
    for e in range(4):
      for ep in range(4):
        for a in range(2):
          for b in range(2):
            rho_E[e, ep] += rho[a*8+b*4+e, a*8+b*4+ep]

    # ρ_E|X=x: condition Alice on Z basis.
    rho_E_cond = []; probs = []
    for x in range(2):
      rho_ex = np.zeros((4, 4), dtype=complex)
      for e in range(4):
        for ep in range(4):
          for b in range(2):
            rho_ex[e, ep] += rho[x*8+b*4+e, x*8+b*4+ep]
      px = max(np.real(np.trace(rho_ex)), 1e-12)
      probs.append(px); rho_E_cond.append(rho_ex / px)

    chi = vn_entropy(rho_E)
    for x in range(2):
        chi -= probs[x] * vn_entropy(rho_E_cond[x])
    return float(S), float(chi)


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


def pareto_at_p(p_val, n_lambdas=12, n_restarts=20):
    """Return list of (S, chi) Pareto points by sweeping lambda."""
    lambdas = np.concatenate([np.linspace(0, 0.5, n_lambdas//2),
                              np.linspace(0.6, 3.0, n_lambdas - n_lambdas//2)])
    pts = []
    for lam in lambdas:
        best_obj = -np.inf; best_S = 0; best_chi = 0
        for restart in range(n_restarts):
            rng = np.random.default_rng(restart * 1000 + int(p_val*10000) + int(lam*100))
            p0 = rng.normal(0, 0.5, 32)
            res = minimize(neg_lagrangian, p0, args=(p_val, lam),
                           method='L-BFGS-B', options={'maxiter': 200, 'gtol': 1e-7})
            psi = params_to_psi(res.x)
            if psi is None: continue
            S, chi = evaluate_post(psi, p_val)
            obj = chi + lam*S
            if obj > best_obj:
                best_obj = obj; best_S = S; best_chi = chi
        pts.append((best_S, best_chi, lam))
    return pts


def solve_episode_mean_lp(pareto_per_p, P_GRID, mu, thresh):
    """Given empirical Pareto points per p, solve episode-mean LP via Lagrangian.
    pareto_per_p[i] = list of (S, chi, lam) at P_GRID[i].
    """
    # For each p, build chi_max(S) interpolant (concave envelope).
    from scipy.interpolate import interp1d
    pareto_envs = []
    for pts in pareto_per_p:
        # Sort by S, take running max chi
        sorted_pts = sorted(pts, key=lambda x: x[0])
        S_arr = np.array([p[0] for p in sorted_pts])
        chi_arr = np.array([p[1] for p in sorted_pts])
        # Upper envelope: chi(S) is non-increasing for S above some pivot;
        # take running max from the right (larger S → lower chi)
        chi_env = np.maximum.accumulate(chi_arr[::-1])[::-1]
        f = interp1d(S_arr, chi_env, bounds_error=False, fill_value=(chi_env[0], 0.0))
        pareto_envs.append((S_arr, chi_env, f))

    # At each p, given lambda, find optimal S* maximizing [chi(S) + lam*S]
    def opt_at_p(i, lam):
        S_arr, chi_env, f = pareto_envs[i]
        # Eve picks max [chi(S) + lam*S] over the empirical S range
        S_grid = np.linspace(S_arr[0], S_arr[-1], 200)
        chi_grid = f(S_grid)
        objs = chi_grid + lam * S_grid
        best_idx = np.argmax(objs)
        return S_grid[best_idx], chi_grid[best_idx]

    def constraint_residual(lam):
        S_avg = sum(mu[i] * opt_at_p(i, lam)[0] for i in range(len(P_GRID)))
        return S_avg - thresh

    f_lo = constraint_residual(0.0)
    f_hi = constraint_residual(20.0)
    if f_lo * f_hi > 0:
        lam_star = 0.0 if f_lo >= 0 else 20.0
    else:
        lam_star = brentq(constraint_residual, 0.0, 20.0, xtol=1e-5)

    S_star = np.array([opt_at_p(i, lam_star)[0] for i in range(len(P_GRID))])
    chi_star = np.array([opt_at_p(i, lam_star)[1] for i in range(len(P_GRID))])
    return {
        'lambda_star': lam_star,
        'S_per_p': S_star, 'chi_per_p': chi_star,
        'S_avg': float(np.sum(mu * S_star)),
        'chi_avg': float(np.sum(mu * chi_star)),
    }


if __name__ == '__main__':
    # Coarse anchor grid for Pareto sweep (computationally expensive)
    P_ANCHORS = np.array([0.0, 0.012, 0.025, 0.038, 0.051, P_CLIFF - 0.01])
    sigma_inf = P_SIGMA / math.sqrt(2*P_KAPPA)
    mu_p = np.array([np.exp(-0.5*((p-P_MU)/sigma_inf)**2) for p in P_ANCHORS])
    mu_p /= mu_p.sum()

    vprint(f'P_ANCHORS = {P_ANCHORS}')
    vprint(f'OU weights = {mu_p.round(3)}')
    vprint(f'DET_THRESH = {DET_THRESH:.4f}')
    vprint(f'2-qubit Eve ancilla (16-dim Hilbert space)\n')

    pareto_per_p = []
    t0 = time.time()
    for p_val in P_ANCHORS:
        vprint(f'=== p={p_val:.4f} ===')
        t_p = time.time()
        pts = pareto_at_p(float(p_val), n_lambdas=10, n_restarts=15)
        pareto_per_p.append(pts)
        # Show summary
        feasible = [(S, chi) for (S, chi, lam) in pts if S >= DET_THRESH - 1e-3]
        if feasible:
            chi_at_thresh = max(chi for (S, chi) in feasible)
        else:
            chi_at_thresh = 0.0
        vprint(f'  Pareto sample (S, χ) at p={p_val:.4f}:')
        for S, chi, lam in pts:
            vprint(f'    λ={lam:.3f}: (S={S:.4f}, χ={chi:.4f})')
        vprint(f'  → χ_max(S≥DET_THRESH) = {chi_at_thresh:.4f}  ({time.time()-t_p:.1f}s)\n')

    vprint('=' * 70)
    vprint('  EPISODE-MEAN LP SOLVED ON EMPIRICAL PARETO (2-qubit Eve)')
    vprint('=' * 70)
    lp_result = solve_episode_mean_lp(pareto_per_p, P_ANCHORS, mu_p, DET_THRESH)
    vprint(f'  λ* = {lp_result["lambda_star"]:.4f}')
    vprint(f'  Per-p choice:')
    vprint(f'    {"p":>8s}  {"μ(p)":>8s}  {"S*(p)":>8s}  {"χ*(p)":>8s}')
    for i, p in enumerate(P_ANCHORS):
        vprint(f'    {p:>8.4f}  {mu_p[i]:>8.4f}  {lp_result["S_per_p"][i]:>8.4f}  {lp_result["chi_per_p"][i]:>8.4f}')
    vprint(f'\n  Episode-mean S = {lp_result["S_avg"]:.4f} (target {DET_THRESH:.4f})')
    vprint(f'  Episode-mean χ = {lp_result["chi_avg"]:.4f}')
    vprint(f'  Acín analytical ceiling: 0.6069')
    vprint(f'  Library DP (compare):    0.360')
    vprint(f'\nTotal: {(time.time()-t0)/60:.1f} min')

    with open('npa_e91_extended.pkl', 'wb') as f:
        pickle.dump({'P_ANCHORS': P_ANCHORS, 'mu_p': mu_p,
                     'pareto_per_p': pareto_per_p,
                     'episode_mean_lp': lp_result,
                     'DET_THRESH': DET_THRESH}, f)
    vprint('Saved npa_e91_extended.pkl')
