"""Shared physics + env + policies for Path B experiments.

Used by rl_e91_library_noise_pathB.ipynb and rl_e91_pathB_diagnostic.ipynb.
"""
import math, pickle
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from scipy.optimize import brentq

import evoluspsa_e91 as _e91
from evoluspsa_e91 import (
    circuit_to_unitary, STD_SETTINGS, obs as _pauli_obs,
    von_neumann_entropy, secure_key_rate,
)

# ──── Thresholds ────
S_SEC      = brentq(lambda s: secure_key_rate(s) - 1e-6, 2.01, 2.8)
DET_MARGIN = 0.02
DET_THRESH = S_SEC + DET_MARGIN
P_CLIFF    = 1 - np.sqrt(S_SEC / (2 * np.sqrt(2)))

P_GRID  = np.linspace(0.0, P_CLIFF - 0.01, 4)
P_INTERP = np.linspace(0.0, P_CLIFF - 0.005, 16)

# ──── Env defaults ────
P_MU, P_KAPPA, P_SIGMA = 0.03, 0.1, 0.015
P_MIN, P_MAX = 0.0, P_CLIFF - 0.01
EPISODE_LEN = 50
WIN_S = 10
HIST_LEN = 3

# ──── Physics ────
C = torch.complex128
_I2 = torch.eye(2, dtype=C)
_X = torch.tensor([[0, 1], [1, 0]], dtype=C)
_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=C)
_Z = torch.tensor([[1, 0], [0, -1]], dtype=C)
_PAULIS = [_I2, _X, _Y, _Z]

def _embed_on(G, which):
    ops = [_I2, _I2, _I2]
    ops[which] = G
    return torch.kron(torch.kron(ops[0], ops[1]), ops[2])

_EMB_A = [_embed_on(G, 0) for G in _PAULIS]
_EMB_B = [_embed_on(G, 1) for G in _PAULIS]

def depolarize_ab(rho_abe, p):
    if p <= 0:
        return rho_abe
    s0, s1 = math.sqrt(1 - 3 * p / 4), math.sqrt(p / 4)
    coefs = [s0, s1, s1, s1]
    tmp = torch.zeros_like(rho_abe)
    for c, K0 in zip(coefs, _EMB_A):
        K = c * K0
        tmp = tmp + K @ rho_abe @ K.conj().T
    out = torch.zeros_like(tmp)
    for c, K0 in zip(coefs, _EMB_B):
        K = c * K0
        out = out + K @ tmp @ K.conj().T
    return out

def compute_holevo_and_S_noisy(U, settings, p):
    psi = U @ torch.zeros(8, dtype=C).index_fill(0, torch.tensor([0]), 1.0)
    rho = torch.outer(psi, psi.conj())
    rho = depolarize_ab(rho, p)

    a1, a2, b1, b2 = (torch.tensor(settings[i], dtype=torch.float64) for i in range(4))

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
        probs.append(px)
        rho_E_cond.append(rho_ex / px)

    chi = von_neumann_entropy(rho_E)
    for x in range(2):
        chi = chi - probs[x] * von_neumann_entropy(rho_E_cond[x])
    return S.item(), chi.item()


def s_honest(p):
    return (1 - p)**2 * 2 * math.sqrt(2)


def make_U_star(seed=0):
    """Identity-ish action: prepares honest |Φ+⟩_AB ⊗ |0⟩_E."""
    g = torch.Generator().manual_seed(seed)
    phi_plus_e0 = torch.zeros(8, dtype=C)
    phi_plus_e0[0] = 1 / math.sqrt(2)
    phi_plus_e0[6] = 1 / math.sqrt(2)
    M = torch.randn(8, 8, dtype=C, generator=g) + 1j * torch.randn(8, 8, dtype=C, generator=g)
    M[:, 0] = phi_plus_e0
    Q, _ = torch.linalg.qr(M)
    Q[:, 0] = phi_plus_e0
    return Q


def load_library_and_actions(lib_path='e91_library_pathB.pkl'):
    """Load library pkl, build action_U + action_meta + (S, χ) tables."""
    with open(lib_path, 'rb') as f:
        library = pickle.load(f)

    U_star = make_U_star()
    action_U = [U_star]
    action_settings = [np.array(STD_SETTINGS)]
    action_meta = [{'label': 'identity', 'lvl': None, 'p_anchor': None, 's_target': None}]

    for (p_anchor, lvl), entry in library.items():
        action_U.append(circuit_to_unitary(entry['circuit']))
        action_settings.append(np.array(entry['settings']))
        action_meta.append({
            'label': f'{lvl}_p{p_anchor:.3f}',
            'lvl': lvl,
            'p_anchor': p_anchor,
            's_target': entry['s_target'],
        })

    N = len(action_U)
    S_TBL = np.zeros((N, len(P_INTERP)))
    CHI_TBL = np.zeros((N, len(P_INTERP)))
    for a in range(N):
        for j, p in enumerate(P_INTERP):
            S, chi = compute_holevo_and_S_noisy(action_U[a], action_settings[a], float(p))
            S_TBL[a, j] = S
            CHI_TBL[a, j] = chi

    return library, action_U, action_settings, action_meta, S_TBL, CHI_TBL


# ──── Env ────
class PathBEnv(gym.Env):
    """Path B library env. Reward shape configurable for ablations."""

    def __init__(self, S_TBL, CHI_TBL,
                 viol_coef=3.0, det_penalty=10.0,
                 chi_coef=1.0, margin_coef=2.0, margin_tol=0.02,
                 augmented_obs=False):
        super().__init__()
        self.S_TBL = S_TBL
        self.CHI_TBL = CHI_TBL
        self.N = S_TBL.shape[0]
        self.viol_coef = viol_coef
        self.det_penalty = det_penalty
        self.chi_coef = chi_coef
        self.margin_coef = margin_coef
        self.margin_tol = margin_tol
        self.augmented_obs = augmented_obs
        self.action_space = spaces.Discrete(self.N)
        obs_dim = 9 if augmented_obs else 7
        self.observation_space = spaces.Box(
            low=np.float32(-3.0), high=np.float32(5.0),
            shape=(obs_dim,), dtype=np.float32,
        )

    def S_at(self, a, p):
        return float(np.interp(p, P_INTERP, self.S_TBL[a]))
    def chi_at(self, a, p):
        return float(np.interp(p, P_INTERP, self.CHI_TBL[a]))

    def _obs(self):
        recent_S = self.S_ep[-WIN_S:] if self.S_ep else [s_honest(self.p)]
        running_S = float(np.mean(recent_S))
        base = [
            self.p,
            self.p_hist[-1] if len(self.p_hist) >= 1 else self.p,
            self.p_hist[-2] if len(self.p_hist) >= 2 else self.p,
            self.last_action / max(self.N - 1, 1),
            running_S,
            float(np.mean(self.chi_ep)) if self.chi_ep else 0.0,
            (EPISODE_LEN - self.t) / EPISODE_LEN,
        ]
        if self.augmented_obs:
            # Headroom: how far above DET_THRESH we are right now (windowed mean), and total episode-mean room.
            headroom_window = running_S - DET_THRESH
            ep_running = float(np.mean(self.S_ep)) if self.S_ep else s_honest(self.p)
            headroom_episode = ep_running - DET_THRESH
            base.extend([headroom_window, headroom_episode])
        return np.array(base, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.p = float(np.clip(P_MU + P_SIGMA * self.np_random.standard_normal(), P_MIN, P_MAX))
        self.p_hist = []
        self.last_action = 0
        self.S_ep = []
        self.chi_ep = []
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        a = int(action)
        S = self.S_at(a, self.p)
        chi = self.chi_at(a, self.p)
        self.p_hist.append(self.p)
        if len(self.p_hist) > HIST_LEN:
            self.p_hist.pop(0)
        self.last_action = a
        self.S_ep.append(S)
        self.chi_ep.append(chi)
        self.t += 1

        s_h = s_honest(self.p)
        reward = self.chi_coef * chi
        running_S = float(np.mean(self.S_ep[-WIN_S:]))
        if running_S < DET_THRESH:
            reward -= self.viol_coef * (DET_THRESH - running_S)
        excess = running_S - s_h
        if excess > self.margin_tol:
            reward -= self.margin_coef * (excess - self.margin_tol)

        done = self.t >= EPISODE_LEN
        detected = done and float(np.mean(self.S_ep)) < DET_THRESH
        if detected:
            reward -= self.det_penalty

        decay = np.exp(-P_KAPPA)
        noise_std = P_SIGMA * np.sqrt((1 - np.exp(-2 * P_KAPPA)) / (2 * P_KAPPA))
        self.p = float(np.clip(
            P_MU + (self.p - P_MU) * decay + noise_std * self.np_random.standard_normal(),
            P_MIN, P_MAX,
        ))
        return self._obs(), reward, done, False, {'S': S, 'chi': chi, 'detected': detected}


# ──── Policies ────
def identity_policy(obs, env):
    return 0

def make_static_safer_policy(action_meta):
    SAFER = {m['p_anchor']: i for i, m in enumerate(action_meta) if m['lvl'] == 'safer'}
    safer_p_list = sorted(SAFER) if SAFER else []
    def fn(obs, env):
        if not safer_p_list:
            return 0
        p_now = float(obs[0])
        return SAFER[min(safer_p_list, key=lambda v: abs(v - p_now))]
    return fn

def make_oracle_policy():
    """Greedy-feasible Oracle: max χ subject to projected windowed mean ≥ DET_THRESH."""
    def fn(obs, env):
        p_now = float(obs[0])
        past = env.S_ep[-(WIN_S - 1):]
        sum_past = float(np.sum(past)) if past else 0.0
        wn = len(past) + 1
        best_a, best_chi = 0, -np.inf
        for a in range(env.N):
            S_a = env.S_at(a, p_now)
            projected = (sum_past + S_a) / wn
            chi_a = env.chi_at(a, p_now)
            if projected >= DET_THRESH and chi_a > best_chi:
                best_chi = chi_a
                best_a = a
        if best_chi < 0:
            best_a = int(np.argmax([env.S_at(a, p_now) for a in range(env.N)]))
        return best_a
    return fn


def make_cumulative_oracle_policy():
    """Cumulative Oracle: max χ s.t. exact episode-mean projection ≥ DET_THRESH.

    At step t, with cumulative S_so_far and T-t steps left, picks the action a whose
    S(a, p) leaves enough budget for the maximum-S action to be played for all remaining
    steps and still average ≥ DET_THRESH:

        (S_so_far + S(a,p) + (T-t-1)·S_max(p)) / T  ≥  DET_THRESH

    where S_max(p) = max_a' S(a', p). This is tight: assumes Eve will play the safest
    cell for the rest of the episode if forced.
    """
    def fn(obs, env):
        p_now = float(obs[0])
        S_so_far = float(np.sum(env.S_ep))
        T = EPISODE_LEN
        t_done = env.t                     # steps already taken before this one
        steps_left_after = T - t_done - 1  # after taking this action
        S_max_at_p = max(env.S_at(a, p_now) for a in range(env.N))
        budget_target = T * DET_THRESH

        best_a, best_chi = -1, -np.inf
        for a in range(env.N):
            S_a = env.S_at(a, p_now)
            projected_total = S_so_far + S_a + steps_left_after * S_max_at_p
            if projected_total >= budget_target and env.chi_at(a, p_now) > best_chi:
                best_chi = env.chi_at(a, p_now)
                best_a = a
        if best_a < 0:
            # No feasible: take max-S action to minimize damage
            best_a = int(np.argmax([env.S_at(a, p_now) for a in range(env.N)]))
        return best_a
    return fn


# ──── OU dynamics ────
def ou_step(p, rng):
    """One-step OU update used by PathBEnv. Reproduces env's transition exactly."""
    decay = np.exp(-P_KAPPA)
    noise_std = P_SIGMA * np.sqrt((1 - np.exp(-2 * P_KAPPA)) / (2 * P_KAPPA))
    p_next = P_MU + (p - P_MU) * decay + noise_std * rng.standard_normal()
    return float(np.clip(p_next, P_MIN, P_MAX))


def run_episode(env, policy_fn, env_seed):
    obs, _ = env.reset(seed=env_seed)
    done, actions, ps, Ss, chis, running_Ss = False, [], [obs[0]], [], [], []
    while not done:
        a = policy_fn(obs, env)
        actions.append(a)
        obs, r, done, _, info = env.step(a)
        ps.append(obs[0])
        Ss.append(info['S'])
        chis.append(info['chi'])
        running_Ss.append(float(np.mean(env.S_ep[-WIN_S:])))
    return {
        'chi':   float(np.mean(env.chi_ep)),
        'S':     float(np.mean(env.S_ep)),
        'actions': actions,
        'p_traj': np.array(ps),
        'S_traj': np.array(Ss),
        'chi_traj': np.array(chis),
        'running_S_traj': np.array(running_Ss),
        'detected': float(np.mean(env.S_ep)) < DET_THRESH,
    }


def summarize_runs(rollouts):
    chis = [r['chi'] for r in rollouts]
    Ss = [r['S'] for r in rollouts]
    dets = sum(r['detected'] for r in rollouts)
    return {
        'mean_chi': float(np.mean(chis)),
        'std_chi':  float(np.std(chis)),
        'mean_S':   float(np.mean(Ss)),
        'det_rate': dets / len(rollouts),
    }
