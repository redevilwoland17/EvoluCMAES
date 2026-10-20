"""Shared BB84 physics + env + policies, analog of pathB_lib.py for E91.

Mirrors the BB84 setup of QKD_experiment_5/6/7 but factored into a module so we can
run the same analytic toolkit (LP, DP, WIN-sweep, BC, hybrid) we built for E91.
"""
import math, pickle
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from qiskit.quantum_info import Operator

# ──── Constants (BB84 exp_5 values) ────
OU_MU       = 0.10
OU_KAPPA    = 0.1
OU_SIGMA    = 0.03
P_MIN       = 0.01
P_MAX       = 0.30
F_THRESHOLD = 0.87
EPISODE_LEN = 50
HIST_LEN    = 3
WIN_FAB     = 10
LIB_PATHS   = {
    "safer":  "evolu_grid_safer_bitflip.pkl",
    "safe":   "evolu_grid_bitflip.pkl",
    "medium": "evolu_grid_medium_bitflip.pkl",
    "agg":    "evolu_grid_hard_bitflip.pkl",
}

# Interpolation grid for (F_AB, F_AE) tables
P_INTERP = np.linspace(P_MIN, P_MAX, 100)


# ──── BB84 Physics (from exp_5 cell 6) ────
_I2 = torch.eye(2, dtype=torch.complex64)
_X  = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
_ket0 = torch.tensor([1, 0], dtype=torch.complex64)
_rho_eve = torch.outer(_ket0, _ket0.conj())

_LABELS = {
    '0': torch.tensor([1, 0], dtype=torch.complex64),
    '1': torch.tensor([0, 1], dtype=torch.complex64),
    '+': torch.tensor([1, 1], dtype=torch.complex64) / np.sqrt(2),
    '-': torch.tensor([1, -1], dtype=torch.complex64) / np.sqrt(2),
}

_bb84_data = []
for label, psi in _LABELS.items():
    proj = torch.outer(psi, psi.conj())
    rho_clean = torch.kron(proj, _rho_eve)
    rho_flip  = torch.kron(_X @ proj @ _X.conj().T, _rho_eve)
    proj_A = torch.kron(proj, _I2)
    proj_E = torch.kron(_I2, proj)
    _bb84_data.append((rho_clean, rho_flip, proj_A, proj_E))


def compute_fidelities(U, p):
    """BB84 Eve fidelity F_AE and Bob fidelity F_AB at bit-flip noise rate p."""
    f_ab = torch.zeros((), dtype=torch.float32)
    f_ae = torch.zeros((), dtype=torch.float32)
    for rho_clean, rho_flip, proj_A, proj_E in _bb84_data:
        rho_in  = (1 - p) * rho_clean + p * rho_flip
        rho_out = U @ rho_in @ U.conj().T
        f_ab = f_ab + torch.trace(proj_A @ rho_out).real
        f_ae = f_ae + torch.trace(proj_E @ rho_out).real
    return float(f_ab / 4), float(f_ae / 4)


def load_library_and_actions():
    """Load BB84 library and build (F_AB, F_AE) tables.

    Library structure: 4 levels × 6 noise anchors + 1 identity = 25 actions.
    Returns: library_dict, action_U (list), action_meta (list),
             FAB_TBL [N x len(P_INTERP)], FAE_TBL [N x len(P_INTERP)]
    """
    grids = {}
    for name, path in LIB_PATHS.items():
        with open(path, "rb") as f:
            grids[name] = pickle.load(f)

    def _extract_best_U(res):
        best_list, _, _, history_list, _ = res
        loss = np.array([h["best"]["loss"][-1] for h in history_list])
        i = int(np.argmin(loss))
        return torch.tensor(Operator(best_list[i]).data, dtype=torch.complex64)

    ps_sorted = sorted(grids["safe"].keys())  # [0.02, 0.05, 0.10, 0.15, 0.20, 0.25]

    action_U = [torch.eye(4, dtype=torch.complex64)]   # action 0 = identity
    action_meta = [{'label': 'identity', 'level': None, 'p_anchor': None}]

    for name in LIB_PATHS:
        for p in ps_sorted:
            U = _extract_best_U(grids[name][p])
            action_U.append(U)
            action_meta.append({'label': f'{name}_p{p}', 'level': name, 'p_anchor': p})

    N = len(action_U)
    FAB_TBL = np.zeros((N, len(P_INTERP)))
    FAE_TBL = np.zeros((N, len(P_INTERP)))
    for a in range(N):
        for j, p in enumerate(P_INTERP):
            fab, fae = compute_fidelities(action_U[a], float(p))
            FAB_TBL[a, j] = fab
            FAE_TBL[a, j] = fae

    library = {(name, p): None for name in LIB_PATHS for p in ps_sorted}  # placeholder
    return library, action_U, action_meta, FAB_TBL, FAE_TBL, ps_sorted


# ──── OU dynamics ────
def ou_step(p, rng):
    decay = np.exp(-OU_KAPPA)
    noise_std = OU_SIGMA * np.sqrt((1 - np.exp(-2 * OU_KAPPA)) / (2 * OU_KAPPA))
    p_next = OU_MU + (p - OU_MU) * decay + noise_std * rng.standard_normal()
    return float(np.clip(p_next, P_MIN, P_MAX))


# ──── Env ────
class BB84LibraryEnv(gym.Env):
    """BB84 library env. Mirrors exp_5's LibraryEnv but parametrized for ablations."""

    def __init__(self, FAB_TBL, FAE_TBL,
                 dense_coef=10.0, det_penalty=30.0,
                 margin_coef=3.0, margin_tol=0.01,
                 augmented_obs=False):
        super().__init__()
        self.FAB_TBL = FAB_TBL
        self.FAE_TBL = FAE_TBL
        self.N = FAB_TBL.shape[0]
        self.dense_coef = dense_coef
        self.det_penalty = det_penalty
        self.margin_coef = margin_coef
        self.margin_tol = margin_tol
        self.augmented_obs = augmented_obs
        self.action_space = spaces.Discrete(self.N)
        obs_dim = 9 if augmented_obs else 8
        self.observation_space = spaces.Box(
            low=np.float32(-1.0), high=np.float32(2.0),
            shape=(obs_dim,), dtype=np.float32,
        )

    def F_AB_at(self, a, p):
        return float(np.interp(p, P_INTERP, self.FAB_TBL[a]))
    def F_AE_at(self, a, p):
        return float(np.interp(p, P_INTERP, self.FAE_TBL[a]))

    def _obs(self):
        recent = self.f_ab_ep[-WIN_FAB:] if self.f_ab_ep else [1.0]
        base = [
            self.p,
            self.p_hist[-1] if len(self.p_hist) >= 1 else self.p,
            self.p_hist[-2] if len(self.p_hist) >= 2 else self.p,
            self.p_hist[-3] if len(self.p_hist) >= 3 else self.p,
            self.last_action / max(self.N - 1, 1),
            float(np.mean(recent)),
            float(np.mean(self.f_ae_ep)) if self.f_ae_ep else 0.5,
            (EPISODE_LEN - self.t) / EPISODE_LEN,
        ]
        if self.augmented_obs:
            cum_F = float(np.sum(self.f_ab_ep))
            base.append(cum_F / EPISODE_LEN)
        return np.array(base, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.p = float(np.clip(OU_MU + OU_SIGMA * self.np_random.standard_normal(), P_MIN, P_MAX))
        self.p_hist = []
        self.last_action = 0
        self.f_ab_ep = []
        self.f_ae_ep = []
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        a = int(action)
        f_ab = self.F_AB_at(a, self.p)
        f_ae = self.F_AE_at(a, self.p)
        self.p_hist.append(self.p)
        if len(self.p_hist) > HIST_LEN:
            self.p_hist.pop(0)
        self.last_action = a
        self.f_ab_ep.append(f_ab)
        self.f_ae_ep.append(f_ae)
        self.t += 1

        reward = f_ae - 0.5
        running_fab = float(np.mean(self.f_ab_ep))
        if running_fab < F_THRESHOLD:
            reward -= self.dense_coef * (F_THRESHOLD - running_fab)
        else:
            excess = running_fab - F_THRESHOLD
            if excess > self.margin_tol:
                reward -= self.margin_coef * (excess - self.margin_tol)

        done = self.t >= EPISODE_LEN
        detected = done and running_fab < F_THRESHOLD
        if detected:
            reward -= self.det_penalty

        self.p = ou_step(self.p, self.np_random)
        return self._obs(), float(reward), done, False, {'f_ab': f_ab, 'f_ae': f_ae, 'detected': detected}


# ──── Policies ────
def identity_policy(obs, env):
    return 0


def make_oracle_policy(window=WIN_FAB):
    """Greedy max-F_AE subject to projected windowed mean ≥ F_THRESHOLD."""
    def fn(obs, env):
        p_now = float(obs[0])
        past = env.f_ab_ep[-(window - 1):] if window > 1 else []
        sum_past = float(np.sum(past)) if past else 0.0
        wn = len(past) + 1
        best_a, best_fae = -1, -np.inf
        for a in range(env.N):
            f_ab_a = env.F_AB_at(a, p_now)
            projected = (sum_past + f_ab_a) / wn
            f_ae_a = env.F_AE_at(a, p_now)
            if projected >= F_THRESHOLD and f_ae_a > best_fae:
                best_fae = f_ae_a
                best_a = a
        if best_a < 0:
            # No feasible — pick max-F_AB
            best_a = int(np.argmax([env.F_AB_at(a, p_now) for a in range(env.N)]))
        return best_a
    return fn


def make_static_safer_policy(action_meta):
    """Pick the safer cell whose anchor is closest to current p."""
    safer_actions = {m['p_anchor']: i for i, m in enumerate(action_meta) if m['level'] == 'safer'}
    p_list = sorted(safer_actions)
    def fn(obs, env):
        if not p_list:
            return 0
        p_now = float(obs[0])
        return safer_actions[min(p_list, key=lambda v: abs(v - p_now))]
    return fn


def run_episode(env, policy_fn, env_seed):
    obs, _ = env.reset(seed=env_seed)
    done, actions = False, []
    while not done:
        a = policy_fn(obs, env)
        actions.append(a)
        obs, r, done, _, info = env.step(a)
    return {
        'f_ae': float(np.mean(env.f_ae_ep)),
        'f_ab': float(np.mean(env.f_ab_ep)),
        'actions': actions,
        'detected': float(np.mean(env.f_ab_ep)) < F_THRESHOLD,
    }


def summarize_runs(rollouts):
    f_aes = [r['f_ae'] for r in rollouts]
    f_abs = [r['f_ab'] for r in rollouts]
    dets = sum(r['detected'] for r in rollouts)
    return {
        'mean_f_ae': float(np.mean(f_aes)),
        'std_f_ae':  float(np.std(f_aes)),
        'mean_f_ab': float(np.mean(f_abs)),
        'std_f_ab':  float(np.std(f_abs)),
        'det_rate':  dets / len(rollouts),
    }
