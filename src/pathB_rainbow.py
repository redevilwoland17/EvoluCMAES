"""Rainbow-DQN-lite for Path B / E91 (port of QKD_experiment_7.ipynb's components).

Components:
  - Double DQN (separate online + target networks)
  - Dueling architecture (V + advantage stream)
  - N-step returns (n=3)
  - Smooth-L1 loss + grad clipping
  - ε-greedy exploration with linear decay

No prioritized replay (omitted for simplicity; vanilla uniform sampling).
No action masking (all 16 library actions are always valid in PathBEnv).

Trained on the same library as the prior PPO/DQN runs, evaluated on the same 50 test seeds.
"""
import os, math, random, pickle, time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pathB_lib as L
from pathB_lib import (
    DET_THRESH, P_INTERP, EPISODE_LEN, WIN_S, P_MU, P_KAPPA, P_SIGMA, P_MIN, P_MAX,
    PathBEnv, run_episode, summarize_runs,
)

# Hyperparameters
HIDDEN_DIM     = 128
Q_LR           = 5e-4
GAMMA          = 0.99
N_STEP         = 3
BATCH_SIZE     = 128
BUFFER_SIZE    = 100_000
TRAIN_STEPS    = 500_000
WARMUP_STEPS   = 2_000
TARGET_UPDATE  = 1_000
TRAIN_FREQ     = 4
EPS_START      = 1.0
EPS_END        = 0.02
EPS_DECAY      = 200_000
EVAL_FREQ      = 10_000
N_EVAL         = 20
TOP_K          = 3
MASTER_SEED    = 42

DEVICE = torch.device("cpu")

random.seed(MASTER_SEED); np.random.seed(MASTER_SEED); torch.manual_seed(MASTER_SEED)
torch.set_default_dtype(torch.float64)

# Library
library, action_U, action_settings, action_meta, S_TBL, CHI_TBL = L.load_library_and_actions()
N = S_TBL.shape[0]
print(f'Library N={N}, DET_THRESH={DET_THRESH:.4f}')


# ──── Env (use the standard PathBEnv with v1-tuned shaping) ────
def make_env():
    return PathBEnv(S_TBL, CHI_TBL, viol_coef=3.0, det_penalty=10.0)

env_probe = make_env()
OBS_DIM = env_probe.observation_space.shape[0]


# ──── Dueling Q-Network ────
torch.set_default_dtype(torch.float32)

class DuelingQNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N, hidden=HIDDEN_DIM):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.value_head = nn.Linear(hidden, 1)
        self.adv_head = nn.Linear(hidden, n_actions)
    def forward(self, obs):
        z = self.feat(obs)
        v = self.value_head(z)
        adv = self.adv_head(z)
        return v + adv - adv.mean(dim=-1, keepdim=True)
    def act(self, obs_np, epsilon=0.0):
        if np.random.random() < epsilon:
            return int(np.random.randint(N))
        with torch.no_grad():
            obs = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q = self.forward(obs)
            return int(q.argmax(dim=-1).item())


# ──── Replay buffer with n-step returns ────
class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE, obs_dim=OBS_DIM, n_step=N_STEP, gamma=GAMMA):
        self.capacity = capacity
        self.obs       = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts      = np.zeros(capacity, dtype=np.int64)
        self.rews      = np.zeros(capacity, dtype=np.float32)
        self.dones     = np.zeros(capacity, dtype=np.float32)
        self.size = 0; self.idx = 0
        self.n_step = n_step; self.gamma = gamma
        self.nstep_queue = deque(maxlen=n_step)

    def _store(self, obs, act, rew, next_obs, done):
        i = self.idx
        self.obs[i] = obs; self.acts[i] = act; self.rews[i] = rew
        self.next_obs[i] = next_obs; self.dones[i] = float(done)
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add(self, obs, act, rew, next_obs, done):
        self.nstep_queue.append((obs, act, rew, next_obs, done))
        if len(self.nstep_queue) == self.n_step:
            o0, a0, _, _, _ = self.nstep_queue[0]
            R = 0.0; d = False; last_next_obs = next_obs
            for k, tr in enumerate(self.nstep_queue):
                _, _, r_k, no_k, d_k = tr
                R += (self.gamma ** k) * r_k
                last_next_obs = no_k
                if d_k:
                    d = True; break
            self._store(o0, a0, R, last_next_obs, d)
        if done:
            while len(self.nstep_queue) > 1:
                self.nstep_queue.popleft()
                o0, a0, _, _, _ = self.nstep_queue[0]
                R = 0.0; d = True; last_next_obs = next_obs
                for k, tr in enumerate(self.nstep_queue):
                    _, _, r_k, no_k, _ = tr
                    R += (self.gamma ** k) * r_k
                    last_next_obs = no_k
                self._store(o0, a0, R, last_next_obs, d)
            self.nstep_queue.clear()

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (torch.tensor(self.obs[idx]), torch.tensor(self.acts[idx]),
                torch.tensor(self.rews[idx]), torch.tensor(self.next_obs[idx]),
                torch.tensor(self.dones[idx]))


# ──── Training ────
q_online = DuelingQNet().to(DEVICE)
q_target = DuelingQNet().to(DEVICE)
q_target.load_state_dict(q_online.state_dict())
q_target.eval()
opt = torch.optim.Adam(q_online.parameters(), lr=Q_LR)
buffer = ReplayBuffer()
env = make_env()

def train_step():
    if buffer.size < BATCH_SIZE:
        return 0.0
    obs, acts, rews, next_obs, dones = buffer.sample(BATCH_SIZE)
    obs = obs.to(DEVICE); next_obs = next_obs.to(DEVICE)
    acts = acts.to(DEVICE); rews = rews.to(DEVICE); dones = dones.to(DEVICE)
    q_pred = q_online(obs).gather(1, acts.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_acts = q_online(next_obs).argmax(dim=-1)               # Double DQN
        q_next = q_target(next_obs).gather(1, next_acts.unsqueeze(1)).squeeze(1)
        target = rews + (GAMMA ** N_STEP) * (1 - dones) * q_next
    loss = F.smooth_l1_loss(q_pred, target)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(q_online.parameters(), 10.0)
    opt.step()
    return float(loss.item())


def eval_agent(n_eval=N_EVAL, seed_base=10_000):
    chis, Ss, dets = [], [], 0
    eval_env = make_env()
    for ep in range(n_eval):
        obs, _ = eval_env.reset(seed=seed_base + ep)
        done = False
        while not done:
            a = q_online.act(obs, epsilon=0.0)
            obs, _, done, _, _ = eval_env.step(a)
        chis.append(np.mean(eval_env.chi_ep))
        Ss.append(np.mean(eval_env.S_ep))
        if np.mean(eval_env.S_ep) < DET_THRESH:
            dets += 1
    return float(np.mean(chis)), float(np.mean(Ss)), dets / n_eval


print(f'Starting Rainbow training: {TRAIN_STEPS:,} steps (warmup {WARMUP_STEPS:,})')
history = []
top_k = []
best_score = -np.inf; best_step = 0

obs, _ = env.reset(seed=MASTER_SEED)
step = 0; n_eps = 0
t0 = time.time()

while step < TRAIN_STEPS:
    epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END) * step / EPS_DECAY)
    if step < WARMUP_STEPS:
        a = int(np.random.randint(N))
    else:
        a = q_online.act(obs, epsilon=epsilon)
    next_obs, reward, done, _, _ = env.step(a)
    buffer.add(obs, a, reward, next_obs, done)
    obs = next_obs; step += 1
    if done:
        n_eps += 1
        obs, _ = env.reset(seed=MASTER_SEED + n_eps)
    if step >= WARMUP_STEPS and step % TRAIN_FREQ == 0:
        train_step()
    if step % TARGET_UPDATE == 0:
        q_target.load_state_dict(q_online.state_dict())
    if step % EVAL_FREQ == 0 and step >= WARMUP_STEPS:
        mchi, mS, drate = eval_agent()
        score = mchi - 2.0 * drate
        history.append({'step': step, 'chi': mchi, 'S': mS, 'det': drate,
                        'score': score, 'epsilon': epsilon})
        state = {'q_online': q_online.state_dict()}
        if len(top_k) < TOP_K or score > min(s for s, _, _ in top_k):
            path = f'pathB_rainbow_top{step}'
            torch.save(state, f'{path}.pt')
            top_k.append((score, step, path))
            top_k.sort(reverse=True)
            if len(top_k) > TOP_K:
                _, _, drop_path = top_k.pop()
                try: os.remove(f'{drop_path}.pt')
                except Exception: pass
        if score > best_score:
            best_score = score; best_step = step
            torch.save(state, 'pathB_rainbow_best.pt')
        if step % 50_000 == 0:
            print(f'  [Step {step:7,}] χ={mchi:.4f}  S={mS:.4f}  det={drate*100:5.1f}%  '
                  f'score={score:+.4f}  ε={epsilon:.3f}')

print(f'\nTrained in {(time.time()-t0)/60:.1f} min')
print(f'Best: step={best_step}, score={best_score:.4f}')


# Final eval on the standard test seed range (20_000 + ep)
print('\nFinal evaluation on test seeds 20_000..20_049')
q_online.load_state_dict(torch.load('pathB_rainbow_best.pt')['q_online'])
q_online.eval()

def rainbow_policy(obs, e):
    return q_online.act(obs, epsilon=0.0)

test_env = make_env()
runs = []
for ep in range(50):
    obs, _ = test_env.reset(seed=20_000 + ep)
    done, acts = False, []
    while not done:
        a = rainbow_policy(obs, test_env)
        acts.append(a)
        obs, r, done, _, info = test_env.step(a)
    runs.append({
        'chi': float(np.mean(test_env.chi_ep)),
        'S':   float(np.mean(test_env.S_ep)),
        'detected': float(np.mean(test_env.S_ep)) < DET_THRESH,
        'actions': acts,
    })
summary = summarize_runs(runs)
print(f'\nRainbow-lite: χ={summary["mean_chi"]:.4f} ± {summary["std_chi"]:.4f}  '
      f'S={summary["mean_S"]:.4f}  det={summary["det_rate"]*100:.1f}%')

# Save
result = {'history': history, 'runs': runs, 'summary': summary, 'top_k': top_k,
          'best_step': best_step, 'best_score': best_score}
with open('pathB_strat_rainbow.pkl', 'wb') as f:
    pickle.dump(result, f)
print('Saved pathB_strat_rainbow.pkl')

print('\n=== Summary ===')
print(f'  LP ceiling:         χ=0.3466')
print(f'  WIN=50 (best safe): χ=0.3499')
print(f'  Rainbow-lite:       χ={summary["mean_chi"]:.4f}, det={summary["det_rate"]*100:.1f}%')
print(f'  PPO baseline:       χ=0.3147')
