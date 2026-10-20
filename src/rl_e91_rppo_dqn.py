"""RecurrentPPO (LSTM) and DQN on E91 with augmented obs."""
import os, math, pickle, time, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pathB_lib as L
from pathB_lib import (PathBEnv, DET_THRESH, run_episode, summarize_runs)
from sb3_contrib import RecurrentPPO
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

torch.set_default_dtype(torch.float32)
def vprint(*a, **k): print(*a, **{**k, 'flush': True})

library, action_U, action_settings, action_meta, S_TBL, CHI_TBL = L.load_library_and_actions()
N = S_TBL.shape[0]
vprint(f'Library: {N} actions')

env_kwargs = dict(viol_coef=3.0, det_penalty=30.0, augmented_obs=True)


def evaluate_model(model, n_episodes=50, recurrent=False):
    env = PathBEnv(S_TBL, CHI_TBL, **env_kwargs)
    runs = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=20_000+ep); done = False
        actions = []; chis = []; Ss = []
        if recurrent:
            lstm_state = None; episode_start = np.array([True])
        while not done:
            if recurrent:
                a, lstm_state = model.predict(obs, state=lstm_state,
                                              episode_start=episode_start, deterministic=True)
                episode_start = np.array([False])
            else:
                a, _ = model.predict(obs, deterministic=True)
            a = int(a); actions.append(a)
            obs, r, done, _, info = env.step(a)
            chis.append(info['chi']); Ss.append(info['S'])
        runs.append({'chi': np.mean(chis), 'S': np.mean(Ss),
                     'detected': np.mean(Ss) < DET_THRESH, 'actions': actions})
    return summarize_runs(runs), runs


# ── RecurrentPPO ──
vprint('\n=== RecurrentPPO ===')
class RPPOCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.best_score = -1e9; self.history = []
    def _on_step(self):
        if self.num_timesteps % 25_000 == 0:
            try:
                summary, _ = evaluate_model(self.model, n_episodes=20, recurrent=True)
                m, d = summary['mean_chi'], summary['det_rate']
                self.history.append({'step': self.num_timesteps, 'chi': m, 'det': d})
                score = m - 2.0 * d
                if score > self.best_score and d <= 0.05:
                    self.best_score = score
                    self.model.save('best_rppo')
                if self.num_timesteps % 100_000 == 0:
                    vprint(f'    step={self.num_timesteps:>7,}: χ={m:.4f} det={d*100:.1f}%')
            except Exception as e:
                vprint(f'    eval error: {e}')
        return True

train_env = PathBEnv(S_TBL, CHI_TBL, **env_kwargs)
cb_rppo = RPPOCallback()
rppo = RecurrentPPO('MlpLstmPolicy', train_env, verbose=0, device='cpu',
                    learning_rate=3e-4, n_steps=128, batch_size=64, n_epochs=10,
                    gamma=0.99, ent_coef=0.05, clip_range=0.2, seed=42,
                    policy_kwargs=dict(lstm_hidden_size=128,
                                       net_arch=dict(pi=[128, 128], vf=[128, 128])))
t0 = time.time()
rppo.learn(total_timesteps=500_000, callback=cb_rppo)
vprint(f'  Trained in {(time.time()-t0)/60:.1f} min')
if os.path.exists('best_rppo.zip'):
    rppo = RecurrentPPO.load('best_rppo')
rppo_summary, rppo_runs = evaluate_model(rppo, recurrent=True)
actions = np.concatenate([r['actions'] for r in rppo_runs])
n_uniq = len(np.unique(actions))
vprint(f'  RecurrentPPO: χ={rppo_summary["mean_chi"]:.4f} ± {rppo_summary["std_chi"]:.4f}  '
       f'S={rppo_summary["mean_S"]:.4f}  det={rppo_summary["det_rate"]*100:.1f}%  '
       f'actions={n_uniq}')


# ── DQN ──
vprint('\n=== DQN ===')
class DQNCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.best_score = -1e9; self.history = []
    def _on_step(self):
        if self.num_timesteps % 25_000 == 0:
            try:
                summary, _ = evaluate_model(self.model, n_episodes=20)
                m, d = summary['mean_chi'], summary['det_rate']
                self.history.append({'step': self.num_timesteps, 'chi': m, 'det': d})
                score = m - 2.0 * d
                if score > self.best_score and d <= 0.05:
                    self.best_score = score
                    self.model.save('best_dqn')
                if self.num_timesteps % 100_000 == 0:
                    vprint(f'    step={self.num_timesteps:>7,}: χ={m:.4f} det={d*100:.1f}%')
            except Exception as e:
                vprint(f'    eval error: {e}')
        return True

cb_dqn = DQNCallback()
dqn = DQN('MlpPolicy', train_env, verbose=0, device='cpu',
          learning_rate=1e-4, buffer_size=200_000, learning_starts=20_000,
          batch_size=128, gamma=0.99, train_freq=4, target_update_interval=2_000,
          exploration_fraction=0.5, exploration_final_eps=0.05,
          seed=42, policy_kwargs=dict(net_arch=[256, 256]))
t0 = time.time()
dqn.learn(total_timesteps=500_000, callback=cb_dqn)
vprint(f'  Trained in {(time.time()-t0)/60:.1f} min')
if os.path.exists('best_dqn.zip'):
    dqn = DQN.load('best_dqn')
dqn_summary, dqn_runs = evaluate_model(dqn)
actions = np.concatenate([r['actions'] for r in dqn_runs])
n_uniq = len(np.unique(actions))
vprint(f'  DQN: χ={dqn_summary["mean_chi"]:.4f} ± {dqn_summary["std_chi"]:.4f}  '
       f'S={dqn_summary["mean_S"]:.4f}  det={dqn_summary["det_rate"]*100:.1f}%  '
       f'actions={n_uniq}')


vprint('\n' + '=' * 60)
vprint('  SUMMARY')
vprint('=' * 60)
vprint(f'  RecurrentPPO: χ={rppo_summary["mean_chi"]:.4f} det={rppo_summary["det_rate"]*100:.1f}%')
vprint(f'  DQN:          χ={dqn_summary["mean_chi"]:.4f} det={dqn_summary["det_rate"]*100:.1f}%')
vprint(f'  DP target:    χ=0.356  det=0%')

with open('rl_e91_rppo_dqn_results.pkl', 'wb') as f:
    pickle.dump({
        'rppo_summary': rppo_summary, 'rppo_history': cb_rppo.history,
        'dqn_summary': dqn_summary,   'dqn_history': cb_dqn.history,
    }, f)
vprint('Saved rl_e91_rppo_dqn_results.pkl')
