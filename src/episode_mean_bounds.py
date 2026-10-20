"""Analytical envelopes on the episode mean for BB84 and E91.

The abort condition constrains the EPISODE MEAN of the monitored statistic
(an OU-weighted average over the noise level p), not each round. By linearity
of expectation, max ∫μ(p)·r(p) s.t. ∫μ(p)·s(p) ≥ threshold decouples per p via
a Lagrange multiplier.

Per-p envelopes:
  E91:  χ(S) ≤ Acín(S) = h(1/2 + 1/2·sqrt((S/2)^2 - 1))   (collective attacks,
        any dimension; independent of p). Channel cap: S_max(p) = (1-p)^2·2√2.
  BB84: F_AE(F_AB) ≤ 1/2 + sqrt(F_AB(1-F_AB))             (individual attacks,
        noiseless closed form of Fuchs et al. 1997 / Cerf 2000).
        Channel cap: the bit flip of rate p acts on the travelling qubit before
        Eve's interaction, so the honest (passive) fidelity averaged over the two
        bases is F_AB = 1 - p/2, and no attack can present a larger F_AB.

Because both envelopes are concave and decreasing above the threshold, the
episode-mean ceilings coincide with the per-round envelopes evaluated at the
threshold whenever the channel cap does not bind: F_AE <= 0.8363 (BB84, at
F_AB = 0.87) and chi <= 0.6069 (E91, at S = 2.444).
"""
import math, pickle, time
import numpy as np
from scipy.optimize import brentq, minimize_scalar

def vprint(*a, **k): print(*a, **{**k, 'flush': True})

# ─── Constants ───
S_SEC = 2.4240
DET_THRESH = S_SEC + 0.02
P_CLIFF_E91 = 1 - math.sqrt(S_SEC / (2*math.sqrt(2)))
P_MU_E, P_KAPPA_E, P_SIGMA_E = 0.03, 0.1, 0.015
P_MIN_E, P_MAX_E = 0.0, P_CLIFF_E91 - 0.01

F_THRESHOLD = 0.87
P_MU_B, P_KAPPA_B, P_SIGMA_B = 0.10, 0.1, 0.03
P_MIN_B, P_MAX_B = 0.01, 0.30


def acin_chi(S):
    arg = (S/2.0)**2 - 1.0
    if arg < 0: return 1.0   # below LHV: trivially χ ≤ 1
    x = 0.5 + 0.5*math.sqrt(arg)
    if x <= 0 or x >= 1: return 0.0
    return -x*math.log2(x) - (1-x)*math.log2(1-x)

def cerf_fae(F_AB):
    if F_AB <= 0.5: return 1.0
    return 0.5 + math.sqrt(F_AB * (1 - F_AB))


# ────────────────────────────────────────────────────────────────────
#  E91 episode-mean ceiling
# ────────────────────────────────────────────────────────────────────
def e91_episode_mean_ceiling():
    # OU stationary distribution on a fine grid
    sigma_inf = P_SIGMA_E / math.sqrt(2*P_KAPPA_E)
    P_GRID = np.linspace(P_MIN_E, P_MAX_E, 200)
    mu = np.exp(-0.5*((P_GRID - P_MU_E)/sigma_inf)**2)
    mu /= mu.sum()  # normalized PMF on the grid

    # Channel cap on max achievable S_obs at each p
    s_max = lambda p: (1 - p)**2 * 2*math.sqrt(2)

    # At each p, per-step Pareto: χ(S) = Acín(S) for S in [2, min(s_max(p), 2√2)].
    # (Below S=2, Eve has classical correlations ⇒ trivial χ=1 — but we include S∈[0, s_max] for completeness.)
    # Lagrangian: at each p, max_S [Acín(S) + λ·S]  with S ∈ [0, s_max(p)].

    def opt_S_at_p(p, lam):
        cap = s_max(p)
        # Search S in [0, cap]
        # Acín(S) is decreasing in S for S > 2; below S=2 it's flat at 1.
        # Lagrangian's slope: d(χ)/dS + λ = 0 → ∂Acín/∂S = -λ.
        # For S > 2: ∂χ/∂S < 0. Optimal S satisfies |∂χ/∂S| = λ.
        # Use scipy minimize on -[Acín(S) + λS].
        def obj(S): return -(acin_chi(S) + lam*S)
        res = minimize_scalar(obj, bounds=(0.0, cap), method='bounded', options={'xatol': 1e-7})
        S_star = res.x
        chi_star = acin_chi(S_star)
        return S_star, chi_star

    # Find λ such that ∫ μ S*(λ) = DET_THRESH
    def constraint_residual(lam):
        S_avg = sum(mu[i] * opt_S_at_p(P_GRID[i], lam)[0] for i in range(len(P_GRID)))
        return S_avg - DET_THRESH

    lam_lo, lam_hi = 0.0, 50.0
    f_lo = constraint_residual(lam_lo)
    f_hi = constraint_residual(lam_hi)
    vprint(f'  E91 lambda search: f(lam={lam_lo})={f_lo:.4f}  f(lam={lam_hi})={f_hi:.4f}')
    if f_lo * f_hi > 0:
        # No bracket — meaning constraint always satisfied (infeasibility on the high end?)
        if f_lo >= 0:
            lam_star = 0.0
        else:
            lam_star = lam_hi
    else:
        lam_star = brentq(constraint_residual, lam_lo, lam_hi, xtol=1e-5)
    vprint(f'  E91 optimal λ* = {lam_star:.4f}')

    # Compute final ceiling
    S_per_p = np.array([opt_S_at_p(P_GRID[i], lam_star)[0] for i in range(len(P_GRID))])
    chi_per_p = np.array([opt_S_at_p(P_GRID[i], lam_star)[1] for i in range(len(P_GRID))])
    S_avg = float(np.sum(mu * S_per_p))
    chi_avg = float(np.sum(mu * chi_per_p))
    return {
        'P_GRID': P_GRID, 'mu': mu,
        'S_per_p': S_per_p, 'chi_per_p': chi_per_p,
        's_max_per_p': np.array([s_max(p) for p in P_GRID]),
        'lambda_star': lam_star,
        'S_avg': S_avg, 'chi_avg': chi_avg,
    }


# ────────────────────────────────────────────────────────────────────
#  BB84 episode-mean ceiling
# ────────────────────────────────────────────────────────────────────
def bb84_episode_mean_ceiling():
    sigma_inf = P_SIGMA_B / math.sqrt(2*P_KAPPA_B)
    P_GRID = np.linspace(P_MIN_B, P_MAX_B, 200)
    mu = np.exp(-0.5*((P_GRID - P_MU_B)/sigma_inf)**2)
    mu /= mu.sum()

    # Honest (passive) F_AB at bit-flip rate p, averaged over the Z and X bases:
    # the flip errs only in Z, so F_AB = 1 - p/2 (see bb84_lib.compute_fidelities).
    fab_max = lambda p: 1 - p / 2

    def opt_F_at_p(p, lam):
        cap = fab_max(p)
        # Maximize Cerf(F) + λ·F over F ∈ [0.5, cap]
        def obj(F): return -(cerf_fae(F) + lam*F)
        res = minimize_scalar(obj, bounds=(0.5, max(cap, 0.501)), method='bounded',
                              options={'xatol': 1e-7})
        F_star = res.x
        return F_star, cerf_fae(F_star)

    def constraint_residual(lam):
        F_avg = sum(mu[i] * opt_F_at_p(P_GRID[i], lam)[0] for i in range(len(P_GRID)))
        return F_avg - F_THRESHOLD

    lam_lo, lam_hi = 0.0, 50.0
    f_lo = constraint_residual(lam_lo)
    f_hi = constraint_residual(lam_hi)
    vprint(f'  BB84 lambda search: f(lam={lam_lo})={f_lo:.4f}  f(lam={lam_hi})={f_hi:.4f}')
    if f_lo * f_hi > 0:
        if f_lo >= 0: lam_star = 0.0
        else: lam_star = lam_hi
    else:
        lam_star = brentq(constraint_residual, lam_lo, lam_hi, xtol=1e-5)
    vprint(f'  BB84 optimal λ* = {lam_star:.4f}')

    F_per_p = np.array([opt_F_at_p(P_GRID[i], lam_star)[0] for i in range(len(P_GRID))])
    fae_per_p = np.array([opt_F_at_p(P_GRID[i], lam_star)[1] for i in range(len(P_GRID))])
    F_avg = float(np.sum(mu * F_per_p))
    fae_avg = float(np.sum(mu * fae_per_p))
    return {
        'P_GRID': P_GRID, 'mu': mu,
        'F_per_p': F_per_p, 'fae_per_p': fae_per_p,
        'fab_max_per_p': np.array([fab_max(p) for p in P_GRID]),
        'lambda_star': lam_star,
        'F_avg': F_avg, 'fae_avg': fae_avg,
    }


if __name__ == '__main__':
    vprint('=' * 70)
    vprint('  EPISODE-MEAN CEILING for any-unitary-every-step Eve (collective)')
    vprint('=' * 70)

    vprint('\n--- E91 (DIQKD bilateral depolarizing) ---')
    e91 = e91_episode_mean_ceiling()
    vprint(f'  Acín envelope used; channel-cap S_max(p) = (1-p)²·2√2')
    vprint(f'  Resulting episode-mean S = {e91["S_avg"]:.4f} (target DET_THRESH={DET_THRESH:.4f})')
    vprint(f'  Resulting episode-mean χ = {e91["chi_avg"]:.4f}')
    # Show per-p
    P = e91['P_GRID']; S = e91['S_per_p']; X = e91['chi_per_p']
    sample_ix = [0, len(P)//5, 2*len(P)//5, 3*len(P)//5, 4*len(P)//5, len(P)-1]
    vprint('\n  Per-p (sample):')
    vprint(f'    {"p":>8s}  {"μ(p)":>8s}  {"S*(p)":>8s}  {"χ*(p)":>8s}  {"S_max":>8s}')
    for i in sample_ix:
        vprint(f'    {P[i]:>8.4f}  {e91["mu"][i]:>8.4f}  {S[i]:>8.4f}  {X[i]:>8.4f}  {e91["s_max_per_p"][i]:>8.4f}')

    vprint('\n--- BB84 (prepare-and-measure bit-flip) ---')
    bb = bb84_episode_mean_ceiling()
    vprint(f'  Cerf envelope used; channel-cap F_AB_max(p) = 1 - p_bf/2')
    vprint(f'  Resulting episode-mean F_AB = {bb["F_avg"]:.4f} (target F_THRESHOLD={F_THRESHOLD:.4f})')
    vprint(f'  Resulting episode-mean F_AE = {bb["fae_avg"]:.4f}')
    P = bb['P_GRID']; F = bb['F_per_p']; FE = bb['fae_per_p']
    sample_ix = [0, len(P)//5, 2*len(P)//5, 3*len(P)//5, 4*len(P)//5, len(P)-1]
    vprint('\n  Per-p (sample):')
    vprint(f'    {"p_bf":>8s}  {"μ(p)":>8s}  {"F_AB*":>8s}  {"F_AE*":>8s}  {"F_AB_max":>10s}')
    for i in sample_ix:
        vprint(f'    {P[i]:>8.4f}  {bb["mu"][i]:>8.4f}  {F[i]:>8.4f}  {FE[i]:>8.4f}  {bb["fab_max_per_p"][i]:>10.4f}')

    # Library DP references reported in the paper (Table 3): BB84 fine-grid DP
    # (libraries/bb84_strat_dp_hires.pkl, 0% detection over 200 episodes) and
    # E91 coarse-grid DP (results/e91_exp7_results.pkl, 7% detection).
    LIB_DP_BB84 = 0.7527
    LIB_DP_E91 = 0.3540
    vprint('\n' + '=' * 70)
    vprint('  COMPARISON with the library DP references')
    vprint('=' * 70)
    vprint(f'  E91:  episode-mean χ ≤ {e91["chi_avg"]:.4f}  (Acín envelope, collective attacks)')
    vprint(f'        Library DP (coarse grid): χ = {LIB_DP_E91:.4f}')
    vprint(f'        Separation:               {e91["chi_avg"] - LIB_DP_E91:+.4f}')
    vprint(f'')
    vprint(f'  BB84: episode-mean F_AE ≤ {bb["fae_avg"]:.4f}  (noiseless closed form, individual attacks)')
    vprint(f'        Library DP (fine grid):   F_AE = {LIB_DP_BB84:.4f}')
    vprint(f'        Separation:               {bb["fae_avg"] - LIB_DP_BB84:+.4f}')

    with open('episode_mean_bounds.pkl', 'wb') as f:
        pickle.dump({'e91': e91, 'bb84': bb,
                     'DET_THRESH': DET_THRESH, 'F_THRESHOLD': F_THRESHOLD}, f)
    vprint('\nSaved episode_mean_bounds.pkl')
