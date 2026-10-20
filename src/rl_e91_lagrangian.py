"""Lagrangian PPO with adaptive λ targeting det rate = 0%.

Idea: instead of a fixed DET_PENALTY, learn a Lagrange multiplier that adapts based
on observed detection rate. This avoids the brittle trade-off between under-penalty
(unsafe) and over-penalty (degenerate).

We adapt λ in an outer loop: every K epochs, compute current det_rate, increase λ if >0,
decrease if =0 and chi is plateau.
"""
import os, math, pickle, time, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pathB_lib as L
from pathB_lib import (PathBEnv, DET_THRESH, run_episode, summarize_runs)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

torch.set_default_dtype(torch.float32)
def vprint(*a, **k): print(*a, **{**k, 'flush': True})

library, action_U, action_settings, action_meta, S_TBL, CHI_TBL = L.load_library_and_actions()
N = S_TBL.shape[0]
vprint(f'Library: {N} actions')


# Train multiple stages, increasing det_pen if det_rate > target
def evaluate(model, env_kwargs, n=20):
    env = PathBEnv(S_TBL, CHI_TBL, **env_kwargs)
    chis, dets = [], 0
    for ep in range(n):
        obs, _ = env.reset(seed=10_000+ep); done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(a)
        chis.append(np.mean(env.chi_ep))
        if info.get('detected'): dets += 1
    return float(np.mean(chis)), dets / n


# Curriculum / adaptive Lagrangian
det_pens = [5.0, 10.0, 20.0, 30.0, 50.0]   # Start low, ramp up if needed
target_det = 0.05
best_summary = None; best_chi = -1e9; best_det_pen = None; best_model = None

vprint('\n=== Adaptive Lagrangian PPO (curriculum on DET_PEN) ===')
for stage_idx, dp in enumerate(det_pens):
    env_kwargs = dict(viol_coef=3.0, det_penalty=dp, augmented_obs=True)
    vprint(f'\n--- Stage {stage_idx+1}: det_pen={dp} ---')
    train_env = PathBEnv(S_TBL, CHI_TBL, **env_kwargs)
    if stage_idx == 0:
        model = PPO('MlpPolicy', train_env, verbose=0, device='cpu',
                    learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10,
                    gamma=0.99, ent_coef=0.10, clip_range=0.2, seed=42,
                    policy_kwargs=dict(net_arch=[128, 128]))
    else:
        # Continue from previous stage
        model.set_env(train_env)

    t0 = time.time()
    model.learn(total_timesteps=200_000, reset_num_timesteps=False)
    train_time = time.time() - t0
    chi, det_rate = evaluate(model, env_kwargs, n=30)
    vprint(f'  After stage: χ={chi:.4f}  det={det_rate*100:.1f}%  ({train_time/60:.1f} min)')
    if det_rate <= target_det and chi > best_chi:
        best_chi = chi; best_det_pen = dp
        best_summary = (chi, det_rate)
        model.save('best_lagrangian')
        vprint(f'  → New best: χ={chi:.4f} det={det_rate*100:.1f}% (det_pen={dp})')

# Final extended evaluation
if os.path.exists('best_lagrangian.zip'):
    final_model = PPO.load('best_lagrangian')
    final_env_kwargs = dict(viol_coef=3.0, det_penalty=best_det_pen, augmented_obs=True)
    final_env = PathBEnv(S_TBL, CHI_TBL, **final_env_kwargs)
    def policy(obs, env):
        a, _ = final_model.predict(obs, deterministic=True); return int(a)
    final_runs = [run_episode(final_env, policy, env_seed=20_000+ep) for ep in range(50)]
    final_summary = summarize_runs(final_runs)
    actions = np.concatenate([r['actions'] for r in final_runs])
    n_uniq = len(np.unique(actions))
    vprint(f'\n=== FINAL ===')
    vprint(f'  best_det_pen: {best_det_pen}')
    vprint(f'  χ={final_summary["mean_chi"]:.4f} ± {final_summary["std_chi"]:.4f}  '
           f'S={final_summary["mean_S"]:.4f}  det={final_summary["det_rate"]*100:.1f}%  '
           f'actions={n_uniq}')

    with open('rl_e91_lagrangian_results.pkl', 'wb') as f:
        pickle.dump({'summary': final_summary, 'best_det_pen': best_det_pen,
                     'n_unique': n_uniq}, f)
    vprint('Saved rl_e91_lagrangian_results.pkl')
else:
    vprint('No safe policy found across stages.')
