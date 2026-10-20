"""BB84 PPO + Rainbow with the new toolkit (mirror of E91 setup) for fair comparison
to old BB84 exp_5/7 results."""
import os, math, pickle, time, random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import bb84_lib as B
from bb84_lib import (F_THRESHOLD, P_INTERP, EPISODE_LEN, BB84LibraryEnv,
                       run_episode, summarize_runs)

torch.set_num_threads(4)
torch.set_default_dtype(torch.float32)
library, action_U, action_meta, FAB_TBL, FAE_TBL, ps_sorted = B.load_library_and_actions()
N = FAB_TBL.shape[0]

# ──── PPO baseline ────
print('=== BB84 PPO baseline (500k steps, mirror of E91 setup) ===')
PPO_PATH = 'bb84_strat_ppo.pkl'
if os.path.exists(PPO_PATH):
    print(f'Loading cached {PPO_PATH}')
    ppo_results = pickle.load(open(PPO_PATH, 'rb'))
else:
    class _CB(BaseCallback):
        def __init__(self, eval_freq=20_000, n_eval=20):
            super().__init__()
            self.eval_freq, self.n_eval = eval_freq, n_eval
            self.history = []
        def _on_step(self):
            if self.num_timesteps % self.eval_freq == 0:
                e = BB84LibraryEnv(FAB_TBL, FAE_TBL)
                faes, fabs, dets = [], [], 0
                for ep in range(self.n_eval):
                    obs, _ = e.reset(seed=10_000+ep); done = False
                    while not done:
                        a, _ = self.model.predict(obs, deterministic=True)
                        obs, r, done, _, info = e.step(a)
                    faes.append(np.mean(e.f_ae_ep)); fabs.append(np.mean(e.f_ab_ep))
                    if np.mean(e.f_ab_ep) < F_THRESHOLD: dets += 1
                self.history.append({'step': self.num_timesteps,
                                     'f_ae': float(np.mean(faes)),
                                     'f_ab': float(np.mean(fabs)),
                                     'det': dets/self.n_eval})
                if self.num_timesteps % 100_000 == 0:
                    print(f'  [Eval @ {self.num_timesteps:>7,}] F_AE={np.mean(faes):.4f}  F_AB={np.mean(fabs):.4f}  det={dets/self.n_eval*100:.1f}%')
            return True

    env = BB84LibraryEnv(FAB_TBL, FAE_TBL)
    cb = _CB()
    model = PPO('MlpPolicy', env, verbose=0, device='cpu',
                learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10,
                gamma=0.99, ent_coef=0.05, clip_range=0.2, seed=42,
                policy_kwargs=dict(net_arch=[64, 64]))
    t0 = time.time()
    model.learn(total_timesteps=500_000, callback=cb)
    print(f'  Trained in {(time.time()-t0)/60:.1f} min')

    test_env = BB84LibraryEnv(FAB_TBL, FAE_TBL)
    runs = []
    for ep in range(50):
        obs, _ = test_env.reset(seed=20_000+ep); done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, info = test_env.step(a)
        runs.append({'f_ae': float(np.mean(test_env.f_ae_ep)),
                     'f_ab': float(np.mean(test_env.f_ab_ep)),
                     'detected': float(np.mean(test_env.f_ab_ep)) < F_THRESHOLD,
                     'actions': []})
    ppo_results = {'history': cb.history, 'runs': runs, 'summary': summarize_runs(runs)}
    model.save('bb84_ppo_baseline')
    with open(PPO_PATH, 'wb') as f:
        pickle.dump(ppo_results, f)

s = ppo_results['summary']
print(f'NEW BB84 PPO: F_AE={s["mean_f_ae"]:.4f}  F_AB={s["mean_f_ab"]:.4f}  det={s["det_rate"]*100:.1f}%')


# ──── Rainbow-lite ────
print('\n=== BB84 Rainbow-lite (500k steps, same as E91 setup) ===')
RB_PATH = 'bb84_strat_rainbow.pkl'
if os.path.exists(RB_PATH):
    print(f'Loading cached {RB_PATH}')
    rb_results = pickle.load(open(RB_PATH, 'rb'))
else:
    HIDDEN = 128; LR = 5e-4; GAMMA = 0.99; N_STEP = 3
    BATCH = 128; BUF = 100_000; STEPS = 500_000; WARM = 2_000
    TGT_UPDATE = 1_000; TRAIN_FREQ = 4
    EPS_S, EPS_E, EPS_DEC = 1.0, 0.02, 200_000
    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    env = BB84LibraryEnv(FAB_TBL, FAE_TBL)
    OBS_DIM = env.observation_space.shape[0]

    class DuelingQNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat = nn.Sequential(nn.Linear(OBS_DIM, HIDDEN), nn.ReLU(),
                                      nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
            self.value_head = nn.Linear(HIDDEN, 1)
            self.adv_head = nn.Linear(HIDDEN, N)
        def forward(self, obs):
            z = self.feat(obs); v = self.value_head(z); adv = self.adv_head(z)
            return v + adv - adv.mean(dim=-1, keepdim=True)
        def act(self, obs, eps=0.0):
            if np.random.random() < eps: return int(np.random.randint(N))
            with torch.no_grad():
                return int(self.forward(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).argmax(-1).item())

    class Buf:
        def __init__(self):
            self.obs = np.zeros((BUF, OBS_DIM), dtype=np.float32)
            self.next_obs = np.zeros((BUF, OBS_DIM), dtype=np.float32)
            self.acts = np.zeros(BUF, dtype=np.int64)
            self.rews = np.zeros(BUF, dtype=np.float32)
            self.dones = np.zeros(BUF, dtype=np.float32)
            self.size, self.idx = 0, 0
            self.q = deque(maxlen=N_STEP)
        def _store(self, o, a, r, no, d):
            i = self.idx
            self.obs[i] = o; self.acts[i] = a; self.rews[i] = r
            self.next_obs[i] = no; self.dones[i] = float(d)
            self.idx = (i+1) % BUF; self.size = min(self.size+1, BUF)
        def add(self, o, a, r, no, d):
            self.q.append((o, a, r, no, d))
            if len(self.q) == N_STEP:
                o0, a0, _, _, _ = self.q[0]
                R = 0; dn = False; lno = no
                for k, tr in enumerate(self.q):
                    R += (GAMMA**k)*tr[2]; lno = tr[3]
                    if tr[4]: dn = True; break
                self._store(o0, a0, R, lno, dn)
            if d:
                while len(self.q) > 1:
                    self.q.popleft()
                    o0, a0, _, _, _ = self.q[0]
                    R = 0; lno = no
                    for k, tr in enumerate(self.q):
                        R += (GAMMA**k)*tr[2]; lno = tr[3]
                    self._store(o0, a0, R, lno, True)
                self.q.clear()
        def sample(self, n):
            idx = np.random.randint(0, self.size, n)
            return (torch.tensor(self.obs[idx]), torch.tensor(self.acts[idx]),
                    torch.tensor(self.rews[idx]), torch.tensor(self.next_obs[idx]),
                    torch.tensor(self.dones[idx]))

    qo = DuelingQNet(); qt = DuelingQNet(); qt.load_state_dict(qo.state_dict()); qt.eval()
    opt = torch.optim.Adam(qo.parameters(), lr=LR)
    buf = Buf()
    history = []; best_score = -np.inf

    obs, _ = env.reset(seed=SEED); step = 0; n_eps = 0
    t0 = time.time()
    while step < STEPS:
        eps = max(EPS_E, EPS_S - (EPS_S-EPS_E)*step/EPS_DEC)
        a = int(np.random.randint(N)) if step < WARM else qo.act(obs, eps)
        no, r, done, _, _ = env.step(a)
        buf.add(obs, a, r, no, done)
        obs = no; step += 1
        if done:
            n_eps += 1; obs, _ = env.reset(seed=SEED+n_eps)
        if step >= WARM and step % TRAIN_FREQ == 0 and buf.size >= BATCH:
            o, a, r, no, d = buf.sample(BATCH)
            qp = qo(o).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                na = qo(no).argmax(-1)
                qn = qt(no).gather(1, na.unsqueeze(1)).squeeze(1)
                tgt = r + (GAMMA**N_STEP)*(1-d)*qn
            loss = F.smooth_l1_loss(qp, tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(qo.parameters(), 10.0)
            opt.step()
        if step % TGT_UPDATE == 0:
            qt.load_state_dict(qo.state_dict())
        if step % 20_000 == 0 and step >= WARM:
            ee = BB84LibraryEnv(FAB_TBL, FAE_TBL)
            faes, fabs, dets = [], [], 0
            for epp in range(20):
                oo, _ = ee.reset(seed=10_000+epp); dd = False
                while not dd:
                    aa = qo.act(oo, 0.0); oo, _, dd, _, _ = ee.step(aa)
                faes.append(np.mean(ee.f_ae_ep)); fabs.append(np.mean(ee.f_ab_ep))
                if np.mean(ee.f_ab_ep) < F_THRESHOLD: dets += 1
            mfae, mfab, drate = float(np.mean(faes)), float(np.mean(fabs)), dets/20
            score = mfae - 2.0*drate
            history.append({'step': step, 'f_ae': mfae, 'f_ab': mfab, 'det': drate, 'score': score})
            if score > best_score:
                best_score = score
                torch.save({'q': qo.state_dict()}, 'bb84_rainbow_best.pt')
            if step % 100_000 == 0:
                print(f'  [Step {step:>7,}] F_AE={mfae:.4f}  F_AB={mfab:.4f}  det={drate*100:.1f}%  ε={eps:.3f}')

    print(f'  Trained in {(time.time()-t0)/60:.1f} min')
    qo.load_state_dict(torch.load('bb84_rainbow_best.pt')['q']); qo.eval()
    test_env = BB84LibraryEnv(FAB_TBL, FAE_TBL)
    runs = []
    for ep in range(50):
        obs, _ = test_env.reset(seed=20_000+ep); done = False
        while not done:
            a = qo.act(obs, 0.0); obs, r, done, _, info = test_env.step(a)
        runs.append({'f_ae': float(np.mean(test_env.f_ae_ep)),
                     'f_ab': float(np.mean(test_env.f_ab_ep)),
                     'detected': float(np.mean(test_env.f_ab_ep)) < F_THRESHOLD,
                     'actions': []})
    rb_results = {'history': history, 'runs': runs, 'summary': summarize_runs(runs)}
    with open(RB_PATH, 'wb') as f:
        pickle.dump(rb_results, f)

s = rb_results['summary']
print(f'NEW BB84 Rainbow: F_AE={s["mean_f_ae"]:.4f}  F_AB={s["mean_f_ab"]:.4f}  det={s["det_rate"]*100:.1f}%')


# ──── Final summary ────
print('\n' + '='*60)
print('  BB84 results comparison (NEW toolkit vs OLD exp_5/7)')
print('='*60)
print(f'{"Method":<32}  F_AE     det     source')
print('-' * 65)
print(f'{"DP true ceiling":<32}  0.7527    0.0%   NEW (this session)')
print(f'{"LP stationary":<32}  0.7524    0.0%   NEW')
print(f'{"Oracle WIN=50":<32}  0.7514    0.0%   NEW')
print(f'{"NEW Rainbow (500k)":<32}  {rb_results["summary"]["mean_f_ae"]:.4f}    {rb_results["summary"]["det_rate"]*100:.1f}%   NEW (this session)')
print(f'{"Old Rainbow exp_7 (1M, lucky)":<32}  0.7434    0.0%   OLD exp_7')
print(f'{"NEW PPO (500k)":<32}  {ppo_results["summary"]["mean_f_ae"]:.4f}    {ppo_results["summary"]["det_rate"]*100:.1f}%   NEW (this session)')
print(f'{"Old PPO exp_5 (500k)":<32}  0.7226    2.0%   OLD exp_5')
print(f'{"Static safer":<32}  0.7193    0.0%   NEW')
print(f'{"Identity":<32}  0.5000    0.0%   ref')
