"""Rebuild Path B library with 6 noise anchors (denser around P_MU and near cliff).

Mirrors rl_e91_library_noise_pathB.ipynb library construction exactly, but extends
P_GRID from 4 → 6 anchors. Output: e91_library_pathB_6anchors.pkl
"""
import os, math, pickle, time
import numpy as np
import torch
from scipy.optimize import brentq

import evoluspsa_e91 as _e91
from evoluspsa_e91 import (
    circuit_to_unitary, STD_SETTINGS, secure_key_rate,
)

torch.set_default_dtype(torch.float64)

# ── Thresholds (same as notebook) ──
S_SEC      = brentq(lambda s: secure_key_rate(s) - 1e-6, 2.01, 2.8)
DET_MARGIN = 0.02
DET_THRESH = S_SEC + DET_MARGIN
P_CLIFF    = 1 - np.sqrt(S_SEC / (2 * np.sqrt(2)))

# ── Path B targets (same) ──
S_ABS_TARGETS = {
    'safer':  2.55,
    'safe':   2.43,
    'medium': 2.20,
    'agg':    1.80,
}

# ── 6 anchors instead of 4: denser near P_MU=0.03 and near P_CLIFF ──
P_GRID = np.array([0.0, 0.012, 0.025, 0.038, 0.051, P_CLIFF - 0.01])  # last ≈ 0.0642

# ── CMA-ES knobs (same as notebook) ──
FEAS_MARGIN    = 0.01
N_SEEDS_LIB    = 2
LIB_ALPHA      = 100.0
LIB_BETA       = 0.0
LIB_POP_SIZE   = 30
LIB_N_EPOCHS   = 25
LIB_SPSA_ITERS = 40
LIB_VARIABLE_SETTINGS = False
LIB_PATH       = 'e91_library_pathB_6anchors.pkl'

# ── Noise-aware physics (verbatim from notebook) ──
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
    s0, s1 = math.sqrt(1 - 3 * p / 4), math.sqrt(p / 4)
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
    return S.item(), chi.item()


# ── Monkey-patch evoluspsa_e91.fitness (same as notebook) ──
_CURRENT_P = 0.0
def _fitness_noisy(individual, s_target, alpha=_e91.ALPHA, beta=_e91.BETA):
    circuit, settings = individual
    try:
        U = circuit_to_unitary(circuit)
        S, chi = compute_holevo_and_S_noisy(U, settings, _CURRENT_P)
        return chi - alpha * (S - s_target)**2 - beta * len(circuit), S, chi
    except Exception:
        return -1e6, 0.0, 0.0
_e91.fitness = _fitness_noisy

def run_evoluspsa_at_p(s_target, p, **kwargs):
    global _CURRENT_P
    _CURRENT_P = float(p)
    return _e91.run_evoluspsa(s_target, **kwargs)


# ── Decide which cells to build ──
def s_honest(p): return (1 - p)**2 * 2 * math.sqrt(2)
def feasible(target, p): return target <= s_honest(p) - FEAS_MARGIN

cells_to_build = []
print(f'P_CLIFF    = {P_CLIFF:.5f}')
print(f'P_GRID (6) = {[f"{p:.4f}" for p in P_GRID]}')
print(f'DET_THRESH = {DET_THRESH:.4f}\n')
print(f"{'p':>7}  {'s_honest':>9}  | " + "  ".join(f'{lvl:>6s}={t:.2f}' for lvl, t in S_ABS_TARGETS.items()))
print('-' * 80)
for p in P_GRID:
    row = [f'{p:>7.4f}', f'{s_honest(float(p)):>9.4f}', '|']
    for lvl, target in S_ABS_TARGETS.items():
        if feasible(target, float(p)):
            cells_to_build.append((float(p), lvl, target)); row.append(f'{lvl:>6s}=BUILD')
        else:
            row.append(f'{lvl:>6s}=skip ')
    print('  '.join(row))
print(f'\n{len(cells_to_build)} cells to build (out of {len(P_GRID)*len(S_ABS_TARGETS)} possible)\n')


# ── Build ──
if os.path.exists(LIB_PATH):
    print(f'Loading cached {LIB_PATH}')
    with open(LIB_PATH, 'rb') as f:
        library = pickle.load(f)
else:
    library = {}
    t0 = time.time()
    for i, (p, lvl, s_tgt) in enumerate(cells_to_build, 1):
        sh = s_honest(p)
        seeds = []
        for seed in range(N_SEEDS_LIB):
            print(f'[{i}/{len(cells_to_build)}] p={p:.3f}  lvl={lvl:>6}  '
                  f's_honest={sh:.3f}  s_target={s_tgt:.3f}  seed={seed}')
            r = run_evoluspsa_at_p(
                s_tgt, p, seed=seed * 1000 + hash(lvl) % 997,
                pop_size=LIB_POP_SIZE, n_epochs=LIB_N_EPOCHS,
                spsa_iters=LIB_SPSA_ITERS, alpha=LIB_ALPHA, beta=LIB_BETA,
                variable_settings=LIB_VARIABLE_SETTINGS, verbose=False,
            )
            seeds.append(r)
            print(f'   → S={r["best_S"]:.4f}  χ={r["best_chi"]:.4f}  gates={r["best_n_gates"]}')
        best = max(seeds, key=lambda r: r['best_fitness'])
        library[(p, lvl)] = {
            'circuit':  best['best_circuit'],
            'settings': best['best_settings'],
            'S_at_p':   best['best_S'],
            'chi_at_p': best['best_chi'],
            'n_gates':  best['best_n_gates'],
            's_target': s_tgt,
        }
        with open(LIB_PATH, 'wb') as f:   # checkpoint after each cell
            pickle.dump(library, f)
    print(f'\nSaved {len(library)} entries to {LIB_PATH} in {(time.time() - t0)/60:.1f} min')

print(f'\nLibrary size: {len(library)} entries')
print(f'Max gates across cells: {max(e["n_gates"] for e in library.values())}')
