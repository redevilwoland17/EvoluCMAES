"""E91 port of QKD_experiment_6: RecurrentPPO + MPC with OU prediction.

Minimal changes from BB84 exp_6:
  - `LibraryEnv` → `PathBEnv` (E91, S/χ instead of F_AB/F_AE)
  - `F_THRESHOLD = 0.87` → `DET_THRESH = S_SEC + 0.02 ≈ 2.444`
  - BB84 OU (μ=0.10, σ=0.03) → E91 OU (μ=0.03, σ=0.015)
  - BB84 25-action library → Path B 16-action library
  - Reward coefficients scaled (DENSE=10/0.87 → VIOL=3.5, DET_PEN=30, MARGIN=3)
  - 500k training steps (down from 3M, since 16 vs 25 actions)

Tests whether memory (LSTM) + model-based planning (MPC over OU rollouts) closes
the RL-vs-DP gap (currently 0.315 PPO baseline → 0.356 DP optimal).
"""
import os, math, pickle, time
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO

import pathB_lib as L
from pathB_lib import (
    DET_THRESH, P_INTERP, EPISODE_LEN, WIN_S, P_MU, P_KAPPA, P_SIGMA, P_MIN, P_MAX,
    PathBEnv, run_episode, summarize_runs,
    make_oracle_policy, make_static_safer_policy, identity_policy,
)

torch.set_default_dtype(torch.float64)
library, action_U, action_settings, action_meta, S_TBL, CHI_TBL = L.load_library_and_actions()
N = S_TBL.shape[0]
print(f'Library N={N}, DET_THRESH={DET_THRESH:.4f}')

# Hyperparameters (BB84 exp_6 values where possible)
EPISODE_LEN_LOCAL = EPISODE_LEN
DENSE_COEF   = 3.5         # was 10.0 in BB84; scaled (BB84 F gap max ≈ 0.87, E91 S gap max ≈ 2.44)
DET_PENALTY  = 30.0
MARGIN_COEF  = 3.0
MARGIN_TOL   = 0.02
TRAIN_STEPS  = 500_000     # BB84 used 3M; smaller action space + we've seen plateau
ENT_COEF     = 0.10
LR_INIT      = 3e-4
N_EVAL       = 20
TOP_K        = 3
MASTER_SEED  = 42

MPC_HORIZON  = 5
MPC_SAMPLES  = 30
MPC_VIOL_COEF = 20.0
MPC_SAFETY_MARGIN = 0.02


# ──── E91 env with v1-tuned coefficients ────
def make_env():
    return PathBEnv(S_TBL, CHI_TBL, viol_coef=DENSE_COEF, det_penalty=DET_PENALTY,
                    margin_coef=MARGIN_COEF, margin_tol=MARGIN_TOL)

env_probe = make_env()
print(f'obs_dim = {env_probe.observation_space.shape[0]}, action_space = {env_probe.action_space.n}')


# ──── Training callback for RecurrentPPO ────
class BestModelCallback(BaseCallback):
    def __init__(self, save_prefix='ppo_e91_recurrent_best', k=3, eval_freq=10_000, n_eval=20):
        super().__init__()
        self.save_prefix = save_prefix
        self.k = k
        self.eval_freq = eval_freq
        self.n_eval = n_eval
        self.top = []
        self.best_score = -np.inf
        self.best_step = 0
        self.history = []

    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            env = make_env()
            chis, Ss, dets = [], [], 0
            for ep in range(self.n_eval):
                obs, _ = env.reset(seed=10_000 + ep)
                done = False
                lstm_states = None
                episode_starts = np.ones((1,), dtype=bool)
                while not done:
                    a, lstm_states = self.model.predict(
                        obs, state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=True)
                    episode_starts = np.zeros((1,), dtype=bool)
                    obs, r, done, _, info = env.step(a)
                chis.append(np.mean(env.chi_ep))
                Ss.append(np.mean(env.S_ep))
                if info.get('detected'):
                    dets += 1
            mchi, mS, drate = float(np.mean(chis)), float(np.mean(Ss)), dets / self.n_eval
            self.history.append({'step': self.num_timesteps, 'chi': mchi, 'S': mS, 'det': drate})
            score = mchi - 2.0 * drate
            if len(self.top) < self.k or score > min(s for s, _, _ in self.top):
                path = f'{self.save_prefix}_top{self.num_timesteps}'
                self.model.save(path)
                self.top.append((score, self.num_timesteps, path))
                self.top.sort(reverse=True)
                if len(self.top) > self.k:
                    _, _, drop_path = self.top.pop()
                    try: os.remove(drop_path + '.zip')
                    except Exception: pass
            if score > self.best_score:
                self.best_score = score
                self.best_step = self.num_timesteps
                self.model.save(self.save_prefix)
            if self.num_timesteps % 50_000 == 0:
                print(f'  [Eval @ {self.num_timesteps:>7,}] χ={mchi:.4f}  S={mS:.4f}  det={drate*100:.1f}%')
        return True


# ──── Train RecurrentPPO ────
print('\n=== Training RecurrentPPO (LSTM policy) ===')
torch.set_default_dtype(torch.float32)

REC_PATH = 'pathB_strat_recurrent.pkl'
if os.path.exists(REC_PATH):
    print(f'Loading cached {REC_PATH}')
    with open(REC_PATH, 'rb') as f:
        rec_results = pickle.load(f)
else:
    train_env = make_env()
    cb = BestModelCallback(save_prefix='ppo_e91_recurrent_best', k=TOP_K, n_eval=N_EVAL)
    lr_schedule = lambda progress: LR_INIT * (0.2 + 0.8 * progress)
    model_rec = RecurrentPPO('MlpLstmPolicy', train_env, verbose=0, device='cpu',
                             learning_rate=lr_schedule, n_steps=128, batch_size=64, n_epochs=10,
                             gamma=0.99, ent_coef=ENT_COEF, clip_range=0.2, seed=MASTER_SEED,
                             policy_kwargs=dict(lstm_hidden_size=64, enable_critic_lstm=True))
    t0 = time.time()
    model_rec.learn(total_timesteps=TRAIN_STEPS, callback=cb)
    print(f'  Trained in {(time.time()-t0)/60:.1f} min')
    model_rec = RecurrentPPO.load('ppo_e91_recurrent_best', device='cpu')

    # Eval
    test_env = make_env()
    runs = []
    for ep in range(50):
        obs, _ = test_env.reset(seed=20_000 + ep)
        done, acts = False, []
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        while not done:
            a, lstm_states = model_rec.predict(obs, state=lstm_states,
                                                 episode_start=episode_starts, deterministic=True)
            episode_starts = np.zeros((1,), dtype=bool)
            acts.append(int(a))
            obs, r, done, _, info = test_env.step(a)
        runs.append({'chi': float(np.mean(test_env.chi_ep)), 'S': float(np.mean(test_env.S_ep)),
                     'detected': float(np.mean(test_env.S_ep)) < DET_THRESH, 'actions': acts})
    rec_results = {'history': cb.history, 'runs': runs, 'summary': summarize_runs(runs)}
    with open(REC_PATH, 'wb') as f:
        pickle.dump(rec_results, f)

s = rec_results['summary']
print(f'\nRecurrentPPO: χ={s["mean_chi"]:.4f} ± {s["std_chi"]:.4f}  S={s["mean_S"]:.4f}  det={s["det_rate"]*100:.1f}%')


# ──── MPC policy: Monte Carlo OU rollouts ────
print('\n=== MPC policy (Monte Carlo OU rollouts) ===')

decay = np.exp(-P_KAPPA)
noise_std = P_SIGMA * np.sqrt((1 - np.exp(-2 * P_KAPPA)) / (2 * P_KAPPA))

# Fallback averages (for steps beyond MPC horizon): use windowed-Oracle estimate
S_FALLBACK = 2.50    # rough mean S Eve achieves at OU-mean p
CHI_FALLBACK = 0.30  # rough mean χ

def mpc_policy_factory(horizon=MPC_HORIZON, n_samples=MPC_SAMPLES, viol_coef=MPC_VIOL_COEF,
                       safety_margin=MPC_SAFETY_MARGIN):
    def policy(obs, env):
        p_now = float(obs[0])
        steps_done = len(env.S_ep)
        remaining = EPISODE_LEN_LOCAL - steps_done
        if remaining <= 0:
            return 0
        h = min(horizon, remaining - 1)

        rng = np.random.default_rng(42 + steps_done)
        if h > 0:
            p_paths = np.zeros((n_samples, h))
            for i in range(n_samples):
                p = p_now
                for k in range(h):
                    p = P_MU + (p - P_MU) * decay + noise_std * rng.standard_normal()
                    p = float(np.clip(p, P_MIN, P_MAX))
                    p_paths[i, k] = p
        else:
            p_paths = np.zeros((n_samples, 0))

        thresh = DET_THRESH + safety_margin

        best_a, best_score = 0, -np.inf
        for a in range(N):
            S_a = env.S_at(a, p_now)
            chi_a = env.chi_at(a, p_now)
            S_so_far_sum = float(np.sum(env.S_ep))
            chis_future = np.zeros(n_samples)
            Ss_future = np.zeros(n_samples)
            for i in range(n_samples):
                chi_cum = chi_a
                S_cum = S_so_far_sum + S_a
                # For future steps, follow a "safer" reference (closest safer cell at future p)
                for k in range(h):
                    p_future = p_paths[i, k]
                    # Pick the cell with max χ s.t. S(cell, p_future) ≥ thresh (greedy oracle)
                    best_future_chi = -np.inf; best_future_S = 0
                    for aa in range(N):
                        S_aa = env.S_at(aa, p_future)
                        chi_aa = env.chi_at(aa, p_future)
                        if S_aa >= thresh and chi_aa > best_future_chi:
                            best_future_chi = chi_aa; best_future_S = S_aa
                    if best_future_chi < 0:
                        best_future_chi = 0; best_future_S = max(env.S_at(aa, p_future) for aa in range(N))
                    chi_cum += best_future_chi
                    S_cum += best_future_S
                steps_after = remaining - 1 - h
                if steps_after > 0:
                    chi_cum += steps_after * CHI_FALLBACK
                    S_cum += steps_after * S_FALLBACK
                chis_future[i] = chi_cum / remaining
                Ss_future[i] = S_cum / EPISODE_LEN_LOCAL

            exp_chi = float(np.mean(chis_future))
            viol_rate = float(np.mean(Ss_future < DET_THRESH))
            score = exp_chi - viol_coef * viol_rate
            if score > best_score:
                best_score = score; best_a = a
        return best_a
    return policy


mpc_policy = mpc_policy_factory()

print('Evaluating MPC...')
mpc_runs = []
for ep in range(50):
    test_env = make_env()
    obs, _ = test_env.reset(seed=20_000 + ep)
    done, acts = False, []
    while not done:
        a = mpc_policy(obs, test_env)
        acts.append(a)
        obs, r, done, _, info = test_env.step(a)
    mpc_runs.append({'chi': float(np.mean(test_env.chi_ep)), 'S': float(np.mean(test_env.S_ep)),
                     'detected': float(np.mean(test_env.S_ep)) < DET_THRESH, 'actions': acts})
mpc_summary = summarize_runs(mpc_runs)
print(f'MPC: χ={mpc_summary["mean_chi"]:.4f} ± {mpc_summary["std_chi"]:.4f}  '
      f'S={mpc_summary["mean_S"]:.4f}  det={mpc_summary["det_rate"]*100:.1f}%')

with open('pathB_strat_mpc.pkl', 'wb') as f:
    pickle.dump({'runs': mpc_runs, 'summary': mpc_summary}, f)


# ──── Final summary ────
print('\n' + '='*60)
print(' EXP_6 E91 SUMMARY ')
print('='*60)
print(f'  DP hi-res ceiling:       χ=0.3556, det=0.0%')
print(f'  WIN=50 windowed Oracle:  χ=0.3499, det=0.0%')
print(f'  RecurrentPPO (LSTM):     χ={rec_results["summary"]["mean_chi"]:.4f}, det={rec_results["summary"]["det_rate"]*100:.1f}%')
print(f'  MPC (OU rollouts):       χ={mpc_summary["mean_chi"]:.4f}, det={mpc_summary["det_rate"]*100:.1f}%')
print(f'  PPO baseline:            χ=0.3147, det=0.0%')
print(f'  Rainbow-lite:            χ=0.3395, det=6.0%')
