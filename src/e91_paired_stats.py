"""Per-episode reconstruction of the E91 test evaluation and paired statistics.

The 200-episode E91 test of notebooks/QKD_experiment_7_E91.ipynb stores, for
each policy, the 10,000 actions it took (200 episodes x 50 rounds) and the
aggregate summary, but not the per-episode chi values. Because the channel
trajectory of PathBEnv depends only on the episode seed (20_000 + episode) and
not on the actions, the per-episode values can be recovered exactly by
replaying the stored actions on the same noise trajectories:

    chi_ep = mean_t chi(a_t, p_t),   S_ep = mean_t S(a_t, p_t),

with chi(a, p) and S(a, p) interpolated on the 16-point grid P_INTERP, as in
PathBEnv. This script re-implements, in NumPy only, the physics of
src/pathB_lib.py (bilateral depolarising channel after Eve's preparation,
CHSH value and Holevo quantity chi(A:E) for Alice's Z-basis bit) and the OU
dynamics of PathBEnv, checks that the replay reproduces every aggregate in
results/e91_exp7_results.pkl, and prints the paired comparisons reported in
the appendix table tab:e91_paired (two-sided Wilcoxon signed-rank tests and
normal-approximation paired 95% intervals, the convention of
figure_scripts/make_rl_method_figures.py).

Run from the release root:

    python src/e91_paired_stats.py
"""
import math
import pickle

import numpy as np
from scipy import stats
from scipy.optimize import brentq

LIB_PATH = 'libraries/e91_library_pathB_7levels.pkl'
RES_PATH = 'results/e91_exp7_results.pkl'
EPISODE_LEN = 50
TEST_EPS = 200
SEED0 = 20_000

# ---------------------------------------------------------------- thresholds
def h_bin(x):
    if x <= 0 or x >= 1:
        return 0.0
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def secure_key_rate(S):
    s2 = (S / 2) ** 2
    if s2 <= 1:
        return 0.0
    return max(0.0, 1 - h_bin(0.5 + 0.5 * math.sqrt(s2 - 1))
               - h_bin((1 - S / (2 * math.sqrt(2))) / 2))


S_SEC = brentq(lambda s: secure_key_rate(s) - 1e-6, 2.01, 2.8)
DET_THRESH = S_SEC + 0.02
P_CLIFF = 1 - np.sqrt(S_SEC / (2 * np.sqrt(2)))
P_INTERP = np.linspace(0.0, P_CLIFF - 0.005, 16)
P_MU, P_KAPPA, P_SIGMA = 0.03, 0.1, 0.015
P_MIN, P_MAX = 0.0, P_CLIFF - 0.01

# ---------------------------------------------------------------- physics
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
N_QUBITS = 3


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


def embed_1q(m, q):
    ops = [I2, I2, I2]
    ops[q] = m
    return kron3(*ops)


def embed_cx(c, t):
    U = np.zeros((8, 8), dtype=complex)
    for s in range(8):
        bits = [(s >> (N_QUBITS - 1 - q)) & 1 for q in range(N_QUBITS)]
        nb = bits.copy()
        nb[t] ^= bits[c]
        U[sum(nb[q] << (N_QUBITS - 1 - q) for q in range(N_QUBITS)), s] = 1.0
    return U


def gate(gt, param):
    if gt == 'rx':
        c, s = math.cos(param / 2), math.sin(param / 2)
        return np.array([[c, -1j * s], [-1j * s, c]])
    if gt == 'ry':
        c, s = math.cos(param / 2), math.sin(param / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    if gt == 'rz':
        return np.diag([np.exp(-1j * param / 2), np.exp(1j * param / 2)])
    if gt == 'h':
        return H
    raise ValueError(gt)


def circuit_to_unitary(circuit):
    U = np.eye(8, dtype=complex)
    for gt, qubits, param in circuit:
        gt = str(gt)
        G = embed_cx(qubits[0], qubits[1]) if gt == 'cx' else embed_1q(gate(gt, param), qubits[0])
        U = G @ U
    return U


EMB_A = [embed_1q(P, 0) for P in (I2, X, Y, Z)]
EMB_B = [embed_1q(P, 1) for P in (I2, X, Y, Z)]


def depolarize_ab(rho, p):
    if p <= 0:
        return rho
    s0, s1 = math.sqrt(1 - 3 * p / 4), math.sqrt(p / 4)
    coefs = [s0, s1, s1, s1]
    tmp = sum((c * K) @ rho @ (c * K).conj().T for c, K in zip(coefs, EMB_A))
    return sum((c * K) @ tmp @ (c * K).conj().T for c, K in zip(coefs, EMB_B))


def pauli_obs(phi):
    return math.cos(phi) * Z + math.sin(phi) * X


def vn_entropy(rho):
    ev = np.clip(np.linalg.eigvalsh(rho).real, 1e-12, None)
    return float(-np.sum(ev * np.log2(ev)))


def holevo_and_S_noisy(U, settings, p):
    psi = U[:, 0]
    rho = depolarize_ab(np.outer(psi, psi.conj()), p)
    r = rho.reshape(2, 2, 2, 2, 2, 2)                  # a b e, a' b' e'
    rho_AB = np.einsum('abeABe->abAB', r).reshape(4, 4)
    a1, a2, b1, b2 = settings
    E = lambda x, y: np.trace(rho_AB @ np.kron(pauli_obs(x), pauli_obs(y))).real
    S = E(a1, b1) + E(a1, b2) + E(a2, b1) - E(a2, b2)
    rho_E = np.einsum('abeabf->ef', r)
    chi = vn_entropy(rho_E)
    for x in range(2):
        rho_ex = np.einsum('bebf->ef', r[x, :, :, x, :, :])
        px = max(np.trace(rho_ex).real, 1e-12)
        chi -= px * vn_entropy(rho_ex / px)
    return float(S), float(chi)


def build_tables(library):
    phi_plus_e0 = np.zeros(8, dtype=complex)
    phi_plus_e0[0] = phi_plus_e0[6] = 1 / math.sqrt(2)
    U_id = np.eye(8, dtype=complex)
    U_id[:, 0] = phi_plus_e0                           # only column 0 is used
    unitaries = [U_id]
    settings = [np.array([0.0, np.pi / 2, np.pi / 4, -np.pi / 4])]
    for _, entry in library.items():
        unitaries.append(circuit_to_unitary(entry['circuit']))
        settings.append(np.array(entry['settings'], dtype=float))
    S_TBL = np.zeros((len(unitaries), len(P_INTERP)))
    CHI_TBL = np.zeros_like(S_TBL)
    for a, (U, st) in enumerate(zip(unitaries, settings)):
        for j, p in enumerate(P_INTERP):
            S_TBL[a, j], CHI_TBL[a, j] = holevo_and_S_noisy(U, st, float(p))
    return S_TBL, CHI_TBL


# ---------------------------------------------------------------- dynamics
def p_trajectory(seed):
    """Noise levels seen in rounds 0..49 of PathBEnv after reset(seed)."""
    rng = np.random.Generator(np.random.PCG64(seed))   # gymnasium np_random
    p = float(np.clip(P_MU + P_SIGMA * rng.standard_normal(), P_MIN, P_MAX))
    decay = np.exp(-P_KAPPA)
    noise_std = P_SIGMA * np.sqrt((1 - np.exp(-2 * P_KAPPA)) / (2 * P_KAPPA))
    ps = []
    for _ in range(EPISODE_LEN):
        ps.append(p)
        p = float(np.clip(P_MU + (p - P_MU) * decay + noise_std * rng.standard_normal(), P_MIN, P_MAX))
    return np.array(ps)


def replay(actions, S_TBL, CHI_TBL):
    acts = np.asarray(actions).reshape(TEST_EPS, EPISODE_LEN)
    chi_ep, S_ep = np.zeros(TEST_EPS), np.zeros(TEST_EPS)
    for ep in range(TEST_EPS):
        ps = p_trajectory(SEED0 + ep)
        chi_ep[ep] = np.mean([np.interp(p, P_INTERP, CHI_TBL[a]) for a, p in zip(acts[ep], ps)])
        S_ep[ep] = np.mean([np.interp(p, P_INTERP, S_TBL[a]) for a, p in zip(acts[ep], ps)])
    return chi_ep, S_ep


def paired(x, y):
    d = x - y
    half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))        # normal approximation, as in the figures
    return d.mean(), d.mean() - half, d.mean() + half, stats.wilcoxon(x, y).pvalue


if __name__ == '__main__':
    library = pickle.load(open(LIB_PATH, 'rb'))
    res = pickle.load(open(RES_PATH, 'rb'))['results']
    S_TBL, CHI_TBL = build_tables(library)
    per_ep = {}
    print(f'{"policy":<14}{"mean chi":>10}{"stored":>10}{"det":>7}{"stored":>8}')
    for name, rec in res.items():
        chi_ep, S_ep = replay(rec['actions'], S_TBL, CHI_TBL)
        per_ep[name] = chi_ep
        det = float(np.mean(S_ep < DET_THRESH))
        s = rec['summary']
        print(f'{name:<14}{chi_ep.mean():>10.4f}{s["mean_chi"]:>10.4f}{det:>7.3f}{s["det_rate"]:>8.3f}')
    print('\nPaired comparisons (n = 200 episodes, shared channel realisations)')
    for a, b in [('Rainbow-DQN', 'MaskablePPO'), ('Rainbow-DQN', 'Oracle'), ('Rainbow-DQN', 'DP')]:
        m, lo, hi, pv = paired(per_ep[a], per_ep[b])
        print(f'  {a} - {b:<12} mean {m:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  Wilcoxon p = {pv:.1e}')
