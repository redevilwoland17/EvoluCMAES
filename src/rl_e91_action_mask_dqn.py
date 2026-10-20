"""Test whether action masking or DQN can get PPO to use the granular library cells.

Two approaches:
A) MaskablePPO with constraint-aware action mask: at each step mask actions whose
   projected windowed-mean S < DET_THRESH (i.e., would violate the constraint).
   This is essentially Oracle's feasibility filter applied as a hard mask.

B) DQN — value-based learning over the same 27-action library. Off-policy + Q-values
   may handle constrained discrete spaces better than policy-gradient.

Both compared to vanilla PPO (collapsed at 0.208) and BC (matches DP at 0.350).
"""
import os, math, pickle, time, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pathB_lib as L
from pathB_lib import (PathBEnv, DET_THRESH, P_INTERP, P_MU, P_KAPPA, P_SIGMA,
                       P_MIN, P_MAX, EPISODE_LEN, WIN_S, run_episode, summarize_runs)
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
import gymnasium as gym
from gymnasium import spaces

torch.set_default_dtype(torch.float32)
def vprint(*a, **k): print(*a, **{**k, 'flush': True})

# Load 7-level library (26 cells)
LIB_PATH = 'e91_library_pathB_7levels.pkl'
library = pickle.load(open(LIB_PATH, 'rb'))
vprint(f'Loaded {len(library)} cells from {LIB_PATH}')

# Build (S, χ) tables for the 7-level library
import importlib.util
spec = importlib.util.spec_from_file_location('build', 'pathB_lib7levels_build.py')
m = importlib.util.module_from_spec(spec)
sys.modules['build'] = m

# Load + compute tables (use evoluspsa machinery)
import torch as _t; _t.set_default_dtype(_t.float64)
from evoluspsa_e91 import circuit_to_unitary, STD_SETTINGS

# Build the same physics function as the notebook
import math as _m
C_dt = torch.complex128
_I2 = torch.eye(2, dtype=C_dt)
_X = torch.tensor([[0,1],[1,0]], dtype=C_dt)
_Y = torch.tensor([[0,-1j],[1j,0]], dtype=C_dt)
_Z = torch.tensor([[1,0],[0,-1]], dtype=C_dt)
def _embed(G, w):
    ops = [_I2, _I2, _I2]; ops[w] = G
    return torch.kron(torch.kron(ops[0], ops[1]), ops[2])
_EMB_A = [_embed(P, 0) for P in [_I2, _X, _Y, _Z]]
_EMB_B = [_embed(P, 1) for P in [_I2, _X, _Y, _Z]]
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
    psi = U @ torch.zeros(8, dtype=C_dt).index_fill(0, torch.tensor([0]), 1.0)
    rho = torch.outer(psi, psi.conj())
    rho = depolarize_ab(rho, p)
    a1, a2, b1, b2 = (torch.tensor(s, dtype=torch.float64) for s in settings)
    rho_AB = torch.zeros(4, 4, dtype=C_dt)
    for a in range(2):
        for b in range(2):
            for ap in range(2):
                for bp in range(2):
                    rho_AB[a*2+b, ap*2+bp] = (
                        rho[a*4+b*2+0, ap*4+bp*2+0] + rho[a*4+b*2+1, ap*4+bp*2+1])
    def E_corr(ai, bj):
        return torch.trace(rho_AB @ torch.kron(_pauli_obs(ai), _pauli_obs(bj))).real
    S = E_corr(a1, b1) + E_corr(a1, b2) + E_corr(a2, b1) - E_corr(a2, b2)
    rho_E = torch.zeros(2, 2, dtype=C_dt)
    for e in range(2):
      for ep in range(2):
        for a in range(2):
          for b in range(2):
            rho_E[e, ep] += rho[a*4+b*2+e, a*4+b*2+ep]
    rho_E_cond, probs = [], []
    for x in range(2):
      rho_ex = torch.zeros(2, 2, dtype=C_dt)
      for e in range(2):
        for ep in range(2):
          for b in range(2):
            rho_ex[e, ep] += rho[x*4+b*2+e, x*4+b*2+ep]
      px = torch.trace(rho_ex).real.clamp(min=1e-12)
      probs.append(px); rho_E_cond.append(rho_ex / px)
    chi = von_neumann_entropy(rho_E)
    for x in range(2):
        chi = chi - probs[x] * von_neumann_entropy(rho_E_cond[x])
    return float(S.item()), float(chi.item())

# Build action U set
phi_plus_e0 = torch.zeros(8, dtype=C_dt)
phi_plus_e0[0] = 1 / math.sqrt(2); phi_plus_e0[6] = 1 / math.sqrt(2)
M_init = torch.randn(8, 8, dtype=C_dt) + 1j * torch.randn(8, 8, dtype=C_dt)
M_init[:, 0] = phi_plus_e0
Q, _ = torch.linalg.qr(M_init); Q[:, 0] = phi_plus_e0
U_star = Q

action_U = [U_star]
action_settings = [np.array(STD_SETTINGS)]
action_meta = [{'label': 'identity', 'lvl': None, 'p_anchor': None, 's_target': None}]
for (p_anchor, lvl), entry in library.items():
    action_U.append(circuit_to_unitary(entry['circuit']))
    action_settings.append(np.array(entry['settings']))
    action_meta.append({'label': f'{lvl}_p{p_anchor:.3f}', 'lvl': lvl,
                        'p_anchor': p_anchor, 's_target': entry['s_target']})
N_ACTIONS = len(action_U)
vprint(f'N_ACTIONS = {N_ACTIONS}')

S_TBL = np.zeros((N_ACTIONS, len(P_INTERP)))
CHI_TBL = np.zeros((N_ACTIONS, len(P_INTERP)))
for a in range(N_ACTIONS):
    for j, p in enumerate(P_INTERP):
        S, chi = compute_holevo_and_S_noisy(action_U[a], action_settings[a], float(p))
        S_TBL[a, j] = S; CHI_TBL[a, j] = chi
torch.set_default_dtype(torch.float32)
vprint('Tables computed')


# ── Maskable env: PathBEnv with action_mask method ──
class MaskedPathBEnv(PathBEnv):
    def __init__(self, **kwargs):
        super().__init__(S_TBL, CHI_TBL, **kwargs)
    def action_masks(self):
        """At step t: mask actions where projected windowed mean would dip below DET_THRESH."""
        past = self.S_ep[-(WIN_S - 1):] if hasattr(self, 'S_ep') else []
        sum_past = float(np.sum(past)) if past else 0.0
        n = len(past) + 1
        mask = np.zeros(self.N, dtype=bool)
        for a in range(self.N):
            S_a = self.S_at(a, self.p)
            projected = (sum_past + S_a) / n
            if projected >= DET_THRESH - 0.001:   # small slack
                mask[a] = True
        if not mask.any():
            # Force highest-S action when all infeasible
            best = int(np.argmax([self.S_at(a, self.p) for a in range(self.N)]))
            mask[best] = True
        return mask


def evaluate_model(model, n_episodes=50, masked=False):
    env_kwargs = dict(viol_coef=3.0, det_penalty=30.0, augmented_obs=True)
    env = MaskedPathBEnv(**env_kwargs) if masked else PathBEnv(S_TBL, CHI_TBL, **env_kwargs)
    runs = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=20_000+ep); done = False
        chis, Ss, actions = [], [], []
        while not done:
            if masked:
                mask = env.action_masks()
                a, _ = model.predict(obs, action_masks=mask, deterministic=True)
            else:
                a, _ = model.predict(obs, deterministic=True)
            a = int(a); actions.append(a)
            obs, _, done, _, info = env.step(a)
            chis.append(info['chi']); Ss.append(info['S'])
        runs.append({'chi': np.mean(chis), 'S': np.mean(Ss),
                     'detected': np.mean(Ss) < DET_THRESH, 'actions': actions})
    return summarize_runs(runs), runs


# ── (A) MaskablePPO ──
vprint('\n=== (A) MaskablePPO with feasibility mask ===')
env_kwargs = dict(viol_coef=3.0, det_penalty=30.0, augmented_obs=True)

class EvalMPCB(BaseCallback):
    def __init__(self):
        super().__init__()
        self.best_score = -1e9; self.history = []
    def _on_step(self):
        if self.num_timesteps % 25_000 == 0:
            try:
                summary, _ = evaluate_model(self.model, n_episodes=20, masked=True)
                m, d = summary['mean_chi'], summary['det_rate']
                self.history.append({'step': self.num_timesteps, 'chi': m, 'det': d})
                score = m - 2.0 * d
                if score > self.best_score and d <= 0.05:
                    self.best_score = score
                    self.model.save('best_mppo')
                if self.num_timesteps % 100_000 == 0:
                    vprint(f'    step={self.num_timesteps:>7,}: χ={m:.4f} det={d*100:.1f}%')
            except Exception as e:
                vprint(f'    eval error: {e}')
        return True

train_env = MaskedPathBEnv(**env_kwargs)
cb_mppo = EvalMPCB()
mppo = MaskablePPO('MlpPolicy', train_env, verbose=0, device='cpu',
                   learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10,
                   gamma=0.99, ent_coef=0.10, clip_range=0.2, seed=42,
                   policy_kwargs=dict(net_arch=[128, 128]))
t0 = time.time()
mppo.learn(total_timesteps=500_000, callback=cb_mppo)
vprint(f'  MaskablePPO trained in {(time.time()-t0)/60:.1f} min')
if os.path.exists('best_mppo.zip'):
    mppo = MaskablePPO.load('best_mppo')
mppo_summary, mppo_runs = evaluate_model(mppo, n_episodes=50, masked=True)
acts = np.concatenate([r['actions'] for r in mppo_runs])
vprint(f'  MaskablePPO: χ={mppo_summary["mean_chi"]:.4f} ± {mppo_summary["std_chi"]:.4f}  '
       f'S={mppo_summary["mean_S"]:.4f}  det={mppo_summary["det_rate"]*100:.1f}%  '
       f'actions={len(np.unique(acts))}')


# ── (B) DQN ──
vprint('\n=== (B) DQN ===')
class EvalDQNCB(BaseCallback):
    def __init__(self):
        super().__init__()
        self.best_score = -1e9; self.history = []
    def _on_step(self):
        if self.num_timesteps % 25_000 == 0:
            try:
                summary, _ = evaluate_model(self.model, n_episodes=20, masked=False)
                m, d = summary['mean_chi'], summary['det_rate']
                self.history.append({'step': self.num_timesteps, 'chi': m, 'det': d})
                score = m - 2.0 * d
                if score > self.best_score and d <= 0.05:
                    self.best_score = score
                    self.model.save('best_dqn_7lvl')
                if self.num_timesteps % 100_000 == 0:
                    vprint(f'    step={self.num_timesteps:>7,}: χ={m:.4f} det={d*100:.1f}%')
            except Exception as e:
                vprint(f'    eval error: {e}')
        return True

train_env_dqn = PathBEnv(S_TBL, CHI_TBL, **env_kwargs)
cb_dqn = EvalDQNCB()
dqn = DQN('MlpPolicy', train_env_dqn, verbose=0, device='cpu',
          learning_rate=1e-4, buffer_size=200_000, learning_starts=20_000,
          batch_size=128, gamma=0.99, train_freq=4, target_update_interval=2_000,
          exploration_fraction=0.5, exploration_final_eps=0.05,
          seed=42, policy_kwargs=dict(net_arch=[256, 256]))
t0 = time.time()
dqn.learn(total_timesteps=500_000, callback=cb_dqn)
vprint(f'  DQN trained in {(time.time()-t0)/60:.1f} min')
if os.path.exists('best_dqn_7lvl.zip'):
    dqn = DQN.load('best_dqn_7lvl')
dqn_summary, dqn_runs = evaluate_model(dqn, n_episodes=50, masked=False)
acts = np.concatenate([r['actions'] for r in dqn_runs])
vprint(f'  DQN: χ={dqn_summary["mean_chi"]:.4f} ± {dqn_summary["std_chi"]:.4f}  '
       f'S={dqn_summary["mean_S"]:.4f}  det={dqn_summary["det_rate"]*100:.1f}%  '
       f'actions={len(np.unique(acts))}')


vprint('\n' + '=' * 60)
vprint('  SUMMARY')
vprint('=' * 60)
vprint(f'  Vanilla PPO (baseline):     χ=0.208  det=0%  actions=2  ← collapsed')
vprint(f'  MaskablePPO (feasibility):  χ={mppo_summary["mean_chi"]:.4f}  det={mppo_summary["det_rate"]*100:.1f}%')
vprint(f'  DQN:                        χ={dqn_summary["mean_chi"]:.4f}  det={dqn_summary["det_rate"]*100:.1f}%')
vprint(f'  BC (Oracle):                χ=0.3496  det=4%   actions=13')
vprint(f'  DP:                         χ=0.3497  det=4%   actions=19')

with open('rl_e91_mask_dqn_results.pkl', 'wb') as f:
    pickle.dump({'mppo': mppo_summary, 'dqn': dqn_summary,
                 'mppo_history': cb_mppo.history,
                 'dqn_history': cb_dqn.history}, f)
vprint('\nSaved rl_e91_mask_dqn_results.pkl')
