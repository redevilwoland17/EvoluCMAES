"""PPO with augmented observation (cum_S headroom features) + larger net + longer training.

Hypothesis: degenerate collapse to single safer cell is because vanilla PPO obs
lacks state info needed for buffer-and-spend planning.
"""
import os, math, pickle, time, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pathB_lib as L
from pathB_lib import (PathBEnv, DET_THRESH, run_episode, summarize_runs,
                       make_oracle_policy, identity_policy)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

torch.set_default_dtype(torch.float32)

def vprint(*a, **k): print(*a, **{**k, 'flush': True})

library, action_U, action_settings, action_meta, S_TBL, CHI_TBL = L.load_library_and_actions()
N = S_TBL.shape[0]
vprint(f'Library: {N} actions')

VARIANTS = [
    {'name': 'ppo_aug_64x64',     'augmented_obs': True,  'net': [64, 64],          'viol': 3.0, 'det_pen': 30.0, 'ent': 0.10, 'lr': 3e-4, 'steps': 500_000},
    {'name': 'ppo_aug_128x128',   'augmented_obs': True,  'net': [128, 128],        'viol': 3.0, 'det_pen': 30.0, 'ent': 0.10, 'lr': 3e-4, 'steps': 500_000},
    {'name': 'ppo_aug_256_3',     'augmented_obs': True,  'net': [256, 256, 256],   'viol': 3.0, 'det_pen': 30.0, 'ent': 0.05, 'lr': 1e-4, 'steps': 750_000},
    {'name': 'ppo_aug_lowpen',    'augmented_obs': True,  'net': [128, 128],        'viol': 3.0, 'det_pen': 10.0, 'ent': 0.10, 'lr': 3e-4, 'steps': 500_000},
]


class EvalCB(BaseCallback):
    def __init__(self, env_kwargs, eval_freq=20_000, n_eval=20, det_tol=0.05, savename='best'):
        super().__init__()
        self.env_kwargs = env_kwargs; self.eval_freq = eval_freq; self.n_eval = n_eval
        self.det_tol = det_tol; self.savename = savename
        self.best_score = -1e9; self.history = []
    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            env = PathBEnv(S_TBL, CHI_TBL, **self.env_kwargs)
            chis = []; dets = 0
            for ep in range(self.n_eval):
                obs, _ = env.reset(seed=10_000+ep); done = False
                while not done:
                    a, _ = self.model.predict(obs, deterministic=True)
                    obs, r, done, _, info = env.step(a)
                chis.append(np.mean(env.chi_ep))
                if info.get('detected'): dets += 1
            mchi = float(np.mean(chis)); drate = dets/self.n_eval
            self.history.append({'step': self.num_timesteps, 'chi': mchi, 'det': drate})
            score = mchi - 2.0 * drate
            if score > self.best_score and drate <= self.det_tol:
                self.best_score = score
                self.model.save(self.savename)
            if self.num_timesteps % 100_000 == 0:
                vprint(f'    step={self.num_timesteps:>7,}: χ={mchi:.4f} det={drate*100:.1f}%')
        return True


def evaluate(model, env_kwargs, n_episodes=50):
    env = PathBEnv(S_TBL, CHI_TBL, **env_kwargs)
    def policy(obs, env):
        a, _ = model.predict(obs, deterministic=True); return int(a)
    runs = [run_episode(env, policy, env_seed=20_000+ep) for ep in range(n_episodes)]
    return summarize_runs(runs), runs


results = {}
for v in VARIANTS:
    vprint(f'\n=== {v["name"]} ===')
    env_kwargs = dict(viol_coef=v['viol'], det_penalty=v['det_pen'],
                       augmented_obs=v['augmented_obs'])
    train_env = PathBEnv(S_TBL, CHI_TBL, **env_kwargs)
    cb = EvalCB(env_kwargs, savename=f'best_{v["name"]}')
    model = PPO('MlpPolicy', train_env, verbose=0, device='cpu',
                learning_rate=v['lr'], n_steps=1024, batch_size=64, n_epochs=10,
                gamma=0.99, ent_coef=v['ent'], clip_range=0.2, seed=42,
                policy_kwargs=dict(net_arch=v['net']))
    t0 = time.time()
    model.learn(total_timesteps=v['steps'], callback=cb)
    train_time = time.time() - t0
    if os.path.exists(f'best_{v["name"]}.zip'):
        model = PPO.load(f'best_{v["name"]}')
        vprint(f'  Loaded best checkpoint')
    summary, runs = evaluate(model, env_kwargs)
    actions = np.concatenate([r['actions'] for r in runs])
    n_unique = len(np.unique(actions))
    vprint(f'  RESULT: χ={summary["mean_chi"]:.4f} ± {summary["std_chi"]:.4f}  '
           f'S={summary["mean_S"]:.4f}  det={summary["det_rate"]*100:.1f}%  '
           f'actions={n_unique}  ({train_time/60:.1f} min)')
    results[v['name']] = {
        'config': v, 'summary': summary, 'history': cb.history,
        'n_unique': n_unique, 'action_dist': np.bincount(actions, minlength=N).tolist(),
        'train_time': train_time,
    }

vprint('\n' + '=' * 60)
vprint('  SUMMARY')
vprint('=' * 60)
vprint(f'{"variant":>20s}  {"χ":>8s}  {"std":>8s}  {"S":>8s}  {"det%":>6s}  {"acts":>6s}')
for name, r in results.items():
    s = r['summary']
    vprint(f'{name:>20s}  {s["mean_chi"]:.4f}  {s["std_chi"]:.4f}  {s["mean_S"]:.4f}  '
           f'{s["det_rate"]*100:>5.1f}%  {r["n_unique"]:>6d}')

with open('rl_e91_ppo_aug_results.pkl', 'wb') as f:
    pickle.dump(results, f)
vprint('\nSaved rl_e91_ppo_aug_results.pkl')
