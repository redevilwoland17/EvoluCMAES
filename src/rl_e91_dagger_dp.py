"""DAgger of DP (the analytical optimal): the tightest possible imitation target.

DP achieves χ=0.356 vs Oracle's 0.342. If BC can imitate DP, we close the last gap.
"""
import os, math, pickle, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pathB_lib as L
from pathB_lib import (PathBEnv, DET_THRESH, run_episode, summarize_runs)

torch.set_default_dtype(torch.float32)
def vprint(*a, **k): print(*a, **{**k, 'flush': True})

library, action_U, action_settings, action_meta, S_TBL, CHI_TBL = L.load_library_and_actions()
N = S_TBL.shape[0]
vprint(f'Library: {N} actions')

env_kwargs = dict(viol_coef=3.0, det_penalty=30.0, augmented_obs=True)


# ── Load DP policy from cached pkl ──
dp_data = pickle.load(open('pathB_strat_dp_hires.pkl','rb'))
PI = dp_data['PI']
N_SBIN = dp_data['N_SBIN']
S_BINS = np.linspace(0, 50*2*math.sqrt(2)+5, N_SBIN)
P_INTERP_LIB = L.P_INTERP

def dp_policy(obs, env):
    p_idx = int(np.argmin(np.abs(P_INTERP_LIB - float(obs[0]))))
    s_idx = int(np.argmin(np.abs(S_BINS - float(np.sum(env.S_ep)))))
    return int(PI[env.t, p_idx, s_idx])


def eval_policy(policy_fn, n=50, name=''):
    env = PathBEnv(S_TBL, CHI_TBL, **env_kwargs)
    runs = [run_episode(env, policy_fn, env_seed=20_000+ep) for ep in range(n)]
    summary = summarize_runs(runs)
    actions = np.concatenate([r['actions'] for r in runs])
    n_uniq = len(np.unique(actions))
    vprint(f'  {name:>30s}: χ={summary["mean_chi"]:.4f} ± {summary["std_chi"]:.4f}  '
           f'S={summary["mean_S"]:.4f}  det={summary["det_rate"]*100:.1f}%  actions={n_uniq}')
    return summary, runs

vprint('\n[Verify DP baseline]')
eval_policy(dp_policy, name='DP', n=200)


# ── BCNet ──
class BCNet(nn.Module):
    def __init__(self, obs_dim=9, n_actions=N, hidden=256, depth=3):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, n_actions))
        self.net = nn.Sequential(*layers)
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


def collect_with_label(policy_fn, label_fn, n_episodes, seed_offset=0):
    env = PathBEnv(S_TBL, CHI_TBL, **env_kwargs)
    obs_buf, act_buf = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset+ep); done = False
        while not done:
            a_label = label_fn(obs, env)
            obs_buf.append(obs.copy()); act_buf.append(a_label)
            a_play = policy_fn(obs, env)
            obs, _, done, _, _ = env.step(a_play)
    return np.array(obs_buf), np.array(act_buf)


def train_bc(X, y, n_epochs=80, lr=1e-3, hidden=256, depth=3):
    bc = BCNet(obs_dim=X.shape[1], hidden=hidden, depth=depth)
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


# ── BC of DP, then iter ──
vprint('\n[Iterative DAgger of DP]')
vprint('\n--- Iter 0: BC on DP rollouts (2000 ep) ---')
t0 = time.time()
X_all, y_all = collect_rollouts(dp_policy, n_episodes=2000, seed_offset=70_000)
vprint(f'  Collected {len(X_all)} (obs, action) ({time.time()-t0:.1f}s)')
bc = train_bc(X_all, y_all, n_epochs=100, hidden=256, depth=3)
eval_policy(bc_policy(bc), name='BC iter 0', n=100)

N_ITERS = 5
for it in range(1, N_ITERS + 1):
    vprint(f'\n--- DAgger iter {it} ---')
    t0 = time.time()
    X_new, y_new = collect_with_label(bc_policy(bc), dp_policy,
                                       n_episodes=500, seed_offset=80_000 + it*2000)
    X_all = np.concatenate([X_all, X_new])
    y_all = np.concatenate([y_all, y_new])
    vprint(f'  Total dataset: {len(X_all)} pairs')
    bc = train_bc(X_all, y_all, n_epochs=80, hidden=256, depth=3)
    eval_policy(bc_policy(bc), name=f'BC iter {it}', n=100)
    vprint(f'  ({time.time()-t0:.1f}s)')

vprint('\n[FINAL]')
final_summary, final_runs = eval_policy(bc_policy(bc), name='Final BC-of-DP iter-DAgger', n=200)
actions = np.concatenate([r['actions'] for r in final_runs])
counts = np.bincount(actions, minlength=N)
vprint('\n  Action distribution (top 12):')
top = np.argsort(counts)[::-1][:12]
labels = ['identity'] + [f'{m["lvl"]}_p{m["p_anchor"]:.3f}' for m in action_meta[1:]]
for a in top:
    if counts[a] > 0:
        vprint(f'    [{a:>2d}] {labels[a]:<18s}: {counts[a]/len(actions)*100:.1f}%')

torch.save(bc.state_dict(), 'bc_iter_dagger_dp.pt')
with open('rl_e91_dagger_dp_results.pkl', 'wb') as f:
    pickle.dump({'final_summary': final_summary, 'config': dict(env_kwargs)}, f)
vprint('\nSaved rl_e91_dagger_dp_results.pkl + bc_iter_dagger_dp.pt')
