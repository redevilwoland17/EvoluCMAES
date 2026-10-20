"""NPA relaxation for Eve's guessing probability in the CHSH scenario (noiseless).

Uses ncpol2sdpa to compute an NPA upper bound on P_guess(A0|E) at a fixed CHSH
value S. The guessing probability is converted into an upper bound on the
Holevo quantity through the min-entropy, H(A|E) >= H_min(A|E) = -log2 P_guess,
so that chi(A:E) = 1 - H(A|E) <= 1 + log2 P_guess. This bound holds for any
dimension and is looser than the Acin envelope, which is tight for collective
attacks; the script is an implementation check against that envelope and does
not bound the noisy (bilateral depolarising) problem of the paper.

Note: Fano's inequality, H(A|E) <= h(P_err), gives a LOWER bound on chi and
cannot be used to bound Eve's information from above.
"""
import sys, math, pickle, time
import numpy as np
from scipy.optimize import brentq
from ncpol2sdpa import generate_measurements, projective_measurement_constraints, SdpRelaxation

print = lambda *a, **k: __builtins__.print(*a, **{**k, 'flush': True})

from evoluspsa_e91 import secure_key_rate
S_SEC = brentq(lambda s: secure_key_rate(s) - 1e-6, 2.01, 2.8)
DET_THRESH = S_SEC + 0.02

def acin_bound(S):
    arg = (S/2.0)**2 - 1.0
    if arg < 0: return 1.0
    x = 0.5 + 0.5*math.sqrt(arg)
    if x <= 0 or x >= 1: return 0.0
    return -x*math.log2(x) - (1-x)*math.log2(1-x)


def npa_p_guess(S_target, level=2):
    """NPA bound: max P_guess(A0|E) subject to CHSH = S_target.
    Eve has 2-outcome measurement, guessing Alice's bit a∈{0,1}.

    Uses ncpol2sdpa convention: 2-outcome → 1 free operator A[x][0], with
    P(a=1) = 1 - A[x][0].
    """
    A = generate_measurements([2, 2], 'A')
    B = generate_measurements([2, 2], 'B')
    E = generate_measurements([2], 'E')

    subs = projective_measurement_constraints(A, B, E)

    # ±1-valued observables: O = 1 - 2·P(a=1) = 2·A[x][0] - 1
    obs_A0 = 2*A[0][0] - 1
    obs_A1 = 2*A[1][0] - 1
    obs_B0 = 2*B[0][0] - 1
    obs_B1 = 2*B[1][0] - 1
    S_op = obs_A0*obs_B0 + obs_A0*obs_B1 + obs_A1*obs_B0 - obs_A1*obs_B1

    # P_guess(A0|E) when Eve outputs e ∈ {0,1} as her guess of Alice's a:
    # = ⟨P(A0=0) · E_e=0⟩ + ⟨P(A0=1) · E_e=1⟩
    # = ⟨A[0][0] · E[0][0]⟩ + ⟨(1-A[0][0]) · (1-E[0][0])⟩
    P_guess = A[0][0]*E[0][0] + (1 - A[0][0])*(1 - E[0][0])

    flat_ops = A[0] + A[1] + B[0] + B[1] + E[0]
    sdp = SdpRelaxation(flat_ops, normalized=True, verbose=0)
    sdp.get_relaxation(level=level, objective=-P_guess,
                       momentequalities=[S_op - S_target],
                       substitutions=subs)
    sdp.solve(solver='cvxpy')
    if sdp.primal is None:
        raise RuntimeError(f'SDP failed to solve, status={sdp.status}')
    return -sdp.primal


if __name__ == '__main__':
    print(f'DET_THRESH = {DET_THRESH:.4f}')
    print(f'Acín bound at DET_THRESH: χ ≤ {acin_bound(DET_THRESH):.4f}\n')

    results = {}
    for S_target in [2.0, 2.20, 2.40, 2.444, 2.50, 2.60, 2.70, 2*math.sqrt(2) - 1e-3]:
        print(f'\n=== S_target = {S_target:.4f} ===')
        t0 = time.time()
        try:
            p_guess = npa_p_guess(S_target, level=2)
            # Holevo upper bound via the min-entropy: chi(A:E) <= 1 + log2 P_guess.
            p_clip = min(max(p_guess, 0.5), 1.0)
            chi_upper = 1.0 + math.log2(p_clip)
            acin = acin_bound(S_target)
            print(f'  NPA-2: P_guess={p_guess:.4f} → χ ≤ {chi_upper:.4f}  (min-entropy)')
            print(f'  Acín:  χ ≤ {acin:.4f}  (collective attacks, tight)')
            results[S_target] = {'p_guess': p_guess, 'chi_minentropy_upper': chi_upper, 'acin': acin}
        except Exception as e:
            print(f'  FAILED: {e}')
            import traceback; traceback.print_exc()
        print(f'  ({time.time()-t0:.1f}s)')

    with open('npa_e91_sdp_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print('\nSaved npa_e91_sdp_results.pkl')
