"""Iterative DAgger: BC v_n → roll out BC v_n → query Oracle on visited states → retrain BC v_{n+1}.

Standard DAgger algorithm. Addresses BC's distributional shift when its rollout
visits states the Oracle's training data didn't cover.

Goal: push from BC=0.344 toward DP=0.356 with det=0%.
"""
import os, math, pickle, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pathB_lib as L
from pathB_lib import (PathBEnv, DET_THRESH, run_episode, summarize_runs,
                       make_oracle_policy, make_cumulative_oracle_policy)

torch.set_default_dtype(torch.float32)
def vprint(*a, **k): print(*a, **{**k, 'flush': True})

library, action_U, action_settings, action_meta, S_TBL, CHI_TBL = L.load_library_and_actions()
N = S_TBL.shape[0]
vprint(f'Library: {N} actions')

env_kwargs = dict(viol_coef=3.0, det_penalty=30.0, augmented_obs=True)


# Try cumulative-oracle (planning-based) — usually tighter than greedy oracle
oracle_greedy = make_oracle_policy()
try:
    oracle_cum = make_cumulative_oracle_policy()
except Exception:
    oracle_cum = oracle_greedy
oracle = oracle_greedy  # E91: greedy beats cumulative (0.342 vs 0.248). Cumulative overspends.

# ── Test oracles ──
def eval_policy(policy_fn, n=50, name='', recurrent=False):
    env = PathBEnv(S_TBL, CHI_TBL, **env_kwargs)
    runs = [run_episode(env, policy_fn, env_seed=20_000+ep) for ep in range(n)]
    summary = summarize_runs(runs)
    actions = np.concatenate([r['actions'] for r in runs])
    n_uniq = len(np.unique(actions))
    vprint(f'  {name:>30s}: χ={summary["mean_chi"]:.4f} ± {summary["std_chi"]:.4f}  '
           f'S={summary["mean_S"]:.4f}  det={summary["det_rate"]*100:.1f}%  actions={n_uniq}')
    return summary, runs

vprint('\n[Oracle baselines]')
eval_policy(oracle_greedy, name='Greedy oracle')
eval_policy(oracle_cum,    name='Cumulative oracle')


# ── BCNet ──
class BCNet(nn.Module):
    def __init__(self, obs_dim=9, n_actions=N, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x): return self.net(x)


def collect_rollouts(policy_fn, n_episodes, seed_offset=0):
    env = PathBEnv(S_TBL, CHI_TBL, **env_kwargs)
    obs_buf, act_buf = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset+ep); done = False
        while not done:
            a = policy_fn(obs, env)
            obs_buf.append(obs.copy()); act_buf.append(a)
            obs, _, done, _, _ = env.step(a)
    return np.array(obs_buf), np.array(act_buf)


def collect_with_oracle_label(policy_fn, n_episodes, seed_offset=0):
    """Roll out policy_fn; at each visited state also record Oracle's action."""
    env = PathBEnv(S_TBL, CHI_TBL, **env_kwargs)
    obs_buf, act_buf = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset+ep); done = False
        while not done:
            a_oracle = oracle(obs, env)
            obs_buf.append(obs.copy()); act_buf.append(a_oracle)
            a_play = policy_fn(obs, env)
            obs, _, done, _, _ = env.step(a_play)
    return np.array(obs_buf), np.array(act_buf)


def train_bc(X, y, n_epochs=80, lr=1e-3, hidden=256):
    bc = BCNet(obs_dim=X.shape[1], hidden=hidden)
    opt = torch.optim.Adam(bc.parameters(), lr=lr)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=256, shuffle=True)
    for epoch in range(n_epochs):
        for xb, yb in loader:
            logits = bc(xb); loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return bc

def bc_policy(bc):
    def fn(obs, env):
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            return int(bc(x).argmax().item())
    return fn


# ── DAgger loop ──
vprint('\n[DAgger iterations]')

# Iter 0: BC on pure Oracle rollouts
vprint('\n--- Iter 0: BC on Oracle rollouts (1000 ep) ---')
t0 = time.time()
X_all, y_all = collect_rollouts(oracle, n_episodes=1000, seed_offset=70_000)
vprint(f'  Collected {len(X_all)} (obs, action) from Oracle ({time.time()-t0:.1f}s)')
bc = train_bc(X_all, y_all, n_epochs=80, hidden=256)
summary, _ = eval_policy(bc_policy(bc), name='BC iter 0', n=50)

# DAgger iterations: roll out BC, label with Oracle, augment dataset
N_DAGGER_ITERS = 4
for it in range(1, N_DAGGER_ITERS + 1):
    vprint(f'\n--- DAgger iter {it} ---')
    t0 = time.time()
    X_new, y_new = collect_with_oracle_label(bc_policy(bc),
                                              n_episodes=500,
                                              seed_offset=80_000 + it*2000)
    vprint(f'  Collected {len(X_new)} new states (BC rollout + Oracle label)')
    X_all = np.concatenate([X_all, X_new])
    y_all = np.concatenate([y_all, y_new])
    vprint(f'  Total dataset: {len(X_all)} pairs')
    bc = train_bc(X_all, y_all, n_epochs=60, hidden=256)
    summary, _ = eval_policy(bc_policy(bc), name=f'BC iter {it}', n=50)
    vprint(f'  ({time.time()-t0:.1f}s)')

# Final extended evaluation
vprint('\n[FINAL]')
final_summary, final_runs = eval_policy(bc_policy(bc), name='Final iterative-DAgger BC', n=200)
actions = np.concatenate([r['actions'] for r in final_runs])
counts = np.bincount(actions, minlength=N)
vprint('\n  Action distribution (top 10):')
top = np.argsort(counts)[::-1][:10]
labels = ['identity'] + [f'{m["lvl"]}_p{m["p_anchor"]:.3f}' for m in action_meta[1:]]
for a in top:
    if counts[a] > 0:
        vprint(f'    [{a:>2d}] {labels[a]:<18s}: {counts[a]/len(actions)*100:.1f}%')

torch.save(bc.state_dict(), 'bc_iter_dagger.pt')
with open('rl_e91_dagger_iter_results.pkl', 'wb') as f:
    pickle.dump({
        'final_summary': final_summary,
        'config': dict(env_kwargs),
        'n_dagger_iters': N_DAGGER_ITERS,
    }, f)
vprint('\nSaved rl_e91_dagger_iter_results.pkl + bc_iter_dagger.pt')
