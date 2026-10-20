"""Build E91 Path B library with finer S-target grid (7 levels × 4 anchors).

Same machinery as the 4-level / 6-anchor builds. Levels span above-threshold
through well-below-threshold to give DP / BC finer control.

Output: e91_library_pathB_7levels.pkl
"""
import os, math, pickle, time
import numpy as np
import torch
from scipy.optimize import brentq

import evoluspsa_e91 as _e91
from evoluspsa_e91 import circuit_to_unitary, STD_SETTINGS, secure_key_rate

torch.set_default_dtype(torch.float64)

S_SEC      = brentq(lambda s: secure_key_rate(s) - 1e-6, 2.01, 2.8)
DET_MARGIN = 0.02
DET_THRESH = S_SEC + DET_MARGIN
P_CLIFF    = 1 - np.sqrt(S_SEC / (2 * np.sqrt(2)))

# 7 levels (vs 4 in original Path B): finer Pareto sampling
S_ABS_TARGETS = {
    'safer':   2.55,   # above threshold
    'safer2':  2.50,   # tight above
    'safe':    2.43,   # at threshold
    'medium1': 2.30,   # between
    'medium':  2.20,   # below
    'medium2': 2.00,   # near LHV
    'agg':     1.80,   # well below
}

P_GRID = np.linspace(0.0, P_CLIFF - 0.01, 4)   # same 4 anchors as original

FEAS_MARGIN    = 0.01
N_SEEDS_LIB    = 2
LIB_ALPHA      = 100.0
LIB_BETA       = 0.0
LIB_POP_SIZE   = 30
LIB_N_EPOCHS   = 25
LIB_SPSA_ITERS = 40
LIB_PATH       = 'e91_library_pathB_7levels.pkl'

# Noise-aware physics
C = torch.complex128
_I2 = torch.eye(2, dtype=C)
_X = torch.tensor([[0, 1], [1, 0]], dtype=C)
_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=C)
_Z = torch.tensor([[1, 0], [0, -1]], dtype=C)
_PAULIS = [_I2, _X, _Y, _Z]

def _embed_on(G, which):
    ops = [_I2, _I2, _I2]; ops[which] = G
    return torch.kron(torch.kron(ops[0], ops[1]), ops[2])
_EMB_A = [_embed_on(G, 0) for G in _PAULIS]
_EMB_B = [_embed_on(G, 1) for G in _PAULIS]

def depolarize_ab(rho_abe, p):
    if p <= 0: return rho_abe
    s0, s1 = math.sqrt(1 - 3*p/4), math.sqrt(p/4)
    coefs = [s0, s1, s1, s1]
    tmp = torch.zeros_like(rho_abe)
    for c, K0 in zip(coefs, _EMB_A):
        K = c * K0; tmp = tmp + K @ rho_abe @ K.conj().T
    out = torch.zeros_like(tmp)
    for c, K0 in zip(coefs, _EMB_B):
        K = c * K0; out = out + K @ tmp @ K.conj().T
    return out

from evoluspsa_e91 import obs as _pauli_obs, von_neumann_entropy

def compute_holevo_and_S_noisy(U, settings, p):
    psi = U @ torch.zeros(8, dtype=C).index_fill(0, torch.tensor([0]), 1.0)
    rho = torch.outer(psi, psi.conj())
    rho = depolarize_ab(rho, p)
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
                        rho[a*4+b*2+0, ap*4+bp*2+0] + rho[a*4+b*2+1, ap*4+bp*2+1])
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
    return S.item(), chi.item()


_CURRENT_P = 0.0
def _fitness_noisy(individual, s_target, alpha=_e91.ALPHA, beta=_e91.BETA):
    circuit, settings = individual
    try:
        U = circuit_to_unitary(circuit)
        S, chi = compute_holevo_and_S_noisy(U, settings, _CURRENT_P)
        return chi - alpha*(S - s_target)**2 - beta*len(circuit), S, chi
    except Exception:
        return -1e6, 0.0, 0.0
_e91.fitness = _fitness_noisy

def run_evoluspsa_at_p(s_target, p, **kwargs):
    global _CURRENT_P
    _CURRENT_P = float(p)
    return _e91.run_evoluspsa(s_target, **kwargs)


def s_honest(p): return (1 - p)**2 * 2 * math.sqrt(2)
def feasible(target, p): return target <= s_honest(p) - FEAS_MARGIN

cells_to_build = [(float(p), lvl, t)
                  for p in P_GRID for lvl, t in S_ABS_TARGETS.items()
                  if feasible(t, float(p))]
print(f'P_GRID = {[f"{p:.4f}" for p in P_GRID]}')
print(f'S targets ({len(S_ABS_TARGETS)} levels): {S_ABS_TARGETS}')
print(f'\nFeasibility:')
for p in P_GRID:
    sh = s_honest(float(p))
    row = [f'p={p:.4f}', f's_honest={sh:.4f}', '|']
    for lvl, t in S_ABS_TARGETS.items():
        row.append(f'{lvl}={"BUILD" if feasible(t, float(p)) else "skip "}')
    print('  '.join(row))
print(f'\n{len(cells_to_build)} cells to build (out of {len(P_GRID)*len(S_ABS_TARGETS)} possible)\n', flush=True)


if os.path.exists(LIB_PATH):
    library = pickle.load(open(LIB_PATH, 'rb'))
    print(f'Loaded cached {LIB_PATH}: {len(library)} cells')
else:
    library = {}
    t0 = time.time()
    for i, (p, lvl, s_tgt) in enumerate(cells_to_build, 1):
        seeds = []
        for seed in range(N_SEEDS_LIB):
            r = run_evoluspsa_at_p(
                s_tgt, p, seed=seed*1000+hash(lvl)%997,
                pop_size=LIB_POP_SIZE, n_epochs=LIB_N_EPOCHS,
                spsa_iters=LIB_SPSA_ITERS, alpha=LIB_ALPHA, beta=LIB_BETA,
                variable_settings=False, verbose=False)
            seeds.append(r)
        best = max(seeds, key=lambda r: r['best_fitness'])
        library[(p, lvl)] = {
            'circuit': best['best_circuit'], 'settings': best['best_settings'],
            'S_at_p': best['best_S'], 'chi_at_p': best['best_chi'],
            'n_gates': best['best_n_gates'], 's_target': s_tgt,
        }
        with open(LIB_PATH, 'wb') as f:
            pickle.dump(library, f)
        print(f'[{i}/{len(cells_to_build)}] p={p:.3f} {lvl:>8s}: '
              f'S={best["best_S"]:.4f} χ={best["best_chi"]:.4f} ({(time.time()-t0)/60:.1f} min total)', flush=True)
    print(f'\nSaved {len(library)} entries to {LIB_PATH} in {(time.time()-t0)/60:.1f} min')
