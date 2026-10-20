"""DAgger: BC pretrain on Oracle, then PPO fine-tune.

Hypothesis: PPO can't discover Oracle's diverse multi-cell policy via random exploration
+ sparse detection signal. Pre-training as imitation gives PPO a good starting point.
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

env_kwargs = dict(viol_coef=3.0, det_penalty=30.0, augmented_obs=True)


# ── Step 1: Collect Oracle dataset ──
vprint('\n[1] Collecting Oracle rollouts (1000 episodes)...')
oracle_policy = make_oracle_policy()
obs_buf, act_buf = [], []
env = PathBEnv(S_TBL, CHI_TBL, **env_kwargs)
n_ep = 1000
t0 = time.time()
for ep in range(n_ep):
    obs, _ = env.reset(seed=70_000+ep); done = False
    while not done:
        a = oracle_policy(obs, env)
        obs_buf.append(obs.copy()); act_buf.append(a)
        obs, _, done, _, _ = env.step(a)
vprint(f'  Collected {len(obs_buf)} (obs, action) pairs ({time.time()-t0:.1f}s)')

X = torch.tensor(np.array(obs_buf), dtype=torch.float32)
y = torch.tensor(np.array(act_buf), dtype=torch.long)


# ── Step 2: BC pretraining ──
vprint('\n[2] BC pretrain on Oracle dataset...')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class BCNet(nn.Module):
    def __init__(self, obs_dim=9, n_actions=N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )
    def forward(self, x): return self.net(x)

bc_net = BCNet(obs_dim=X.shape[1], n_actions=N)
opt = torch.optim.Adam(bc_net.parameters(), lr=1e-3)
loader = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=True)
t0 = time.time()
for epoch in range(50):
    losses = []; correct = 0; total = 0
    for xb, yb in loader:
        logits = bc_net(xb); loss = F.cross_entropy(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        correct += (logits.argmax(-1) == yb).sum().item(); total += yb.numel()
    if epoch % 10 == 0:
        vprint(f'  epoch {epoch}: loss={np.mean(losses):.4f}  acc={correct/total*100:.1f}%')
vprint(f'  BC trained in {time.time()-t0:.1f}s')


# ── Step 3: Evaluate BC alone ──
def eval_policy(policy_fn, n=50, name=''):
    env = PathBEnv(S_TBL, CHI_TBL, **env_kwargs)
    runs = [run_episode(env, policy_fn, env_seed=20_000+ep) for ep in range(n)]
    summary = summarize_runs(runs)
    actions = np.concatenate([r['actions'] for r in runs])
    n_uniq = len(np.unique(actions))
    vprint(f'  {name}: χ={summary["mean_chi"]:.4f} ± {summary["std_chi"]:.4f}  '
           f'S={summary["mean_S"]:.4f}  det={summary["det_rate"]*100:.1f}%  actions={n_uniq}')
    return summary, runs

vprint('\n[3] Eval BC alone:')
def bc_policy(obs, env):
    with torch.no_grad():
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        return int(bc_net(x).argmax().item())
bc_summary, bc_runs = eval_policy(bc_policy, name='BC alone')


# ── Step 4: Initialize PPO from BC weights ──
vprint('\n[4] Init PPO from BC weights, fine-tune 250k steps...')
train_env = PathBEnv(S_TBL, CHI_TBL, **env_kwargs)
model = PPO('MlpPolicy', train_env, verbose=0, device='cpu',
            learning_rate=1e-4, n_steps=1024, batch_size=64, n_epochs=10,
            gamma=0.99, ent_coef=0.05, clip_range=0.2, seed=42,
            policy_kwargs=dict(net_arch=[128, 128]))

# Copy BC weights to PPO policy net (best-effort: assumes matching net_arch)
ppo_pi = model.policy.mlp_extractor.policy_net
bc_layers = [m for m in bc_net.net if isinstance(m, nn.Linear)]
ppo_layers = [m for m in ppo_pi if isinstance(m, nn.Linear)]
copied = 0
for bc_l, ppo_l in zip(bc_layers[:-1], ppo_layers):  # skip last bc layer (action head goes elsewhere)
    if bc_l.weight.shape == ppo_l.weight.shape:
        ppo_l.weight.data.copy_(bc_l.weight.data)
        ppo_l.bias.data.copy_(bc_l.bias.data)
        copied += 1
# Action head:
last_bc = bc_layers[-1]
if last_bc.weight.shape == model.policy.action_net.weight.shape:
    model.policy.action_net.weight.data.copy_(last_bc.weight.data)
    model.policy.action_net.bias.data.copy_(last_bc.bias.data)
    copied += 1
vprint(f'  Copied {copied} layers from BC into PPO')


class EvalCB(BaseCallback):
    def __init__(self):
        super().__init__()
        self.best_score = -1e9; self.history = []
    def _on_step(self):
        if self.num_timesteps % 20_000 == 0:
            env = PathBEnv(S_TBL, CHI_TBL, **env_kwargs)
            chis = []; dets = 0
            for ep in range(20):
                obs, _ = env.reset(seed=10_000+ep); done = False
                while not done:
                    a, _ = self.model.predict(obs, deterministic=True)
                    obs, r, done, _, info = env.step(a)
                chis.append(np.mean(env.chi_ep))
                if info.get('detected'): dets += 1
            mchi = float(np.mean(chis)); drate = dets/20
            self.history.append({'step': self.num_timesteps, 'chi': mchi, 'det': drate})
            score = mchi - 2.0 * drate
            if score > self.best_score and drate <= 0.05:
                self.best_score = score
                self.model.save('best_dagger_ppo')
            if self.num_timesteps % 50_000 == 0:
                vprint(f'    step={self.num_timesteps:>7,}: χ={mchi:.4f} det={drate*100:.1f}%')
        return True

cb = EvalCB()
t0 = time.time()
model.learn(total_timesteps=250_000, callback=cb)
vprint(f'  Trained in {(time.time()-t0)/60:.1f} min')

if os.path.exists('best_dagger_ppo.zip'):
    model = PPO.load('best_dagger_ppo')
    vprint('  Loaded best checkpoint')


# ── Step 5: Evaluate ──
vprint('\n[5] Final eval (BC pretrain + PPO finetune):')
def dagger_policy(obs, env):
    a, _ = model.predict(obs, deterministic=True); return int(a)
final_summary, final_runs = eval_policy(dagger_policy, name='DAgger PPO')

# Action distribution
actions = np.concatenate([r['actions'] for r in final_runs])
counts = np.bincount(actions, minlength=N)
vprint(f'\n  Action distribution (top 10):')
top = np.argsort(counts)[::-1][:10]
labels = ['identity'] + [f'{m["lvl"]}_p{m["p_anchor"]:.3f}' for m in action_meta[1:]]
for a in top:
    if counts[a] > 0:
        vprint(f'    [{a:>2d}] {labels[a]:<18s}: {counts[a]/len(actions)*100:.1f}%')

with open('rl_e91_dagger_results.pkl', 'wb') as f:
    pickle.dump({
        'bc_summary': bc_summary,
        'final_summary': final_summary,
        'history': cb.history,
        'config': dict(env_kwargs),
    }, f)
vprint('\nSaved rl_e91_dagger_results.pkl')
