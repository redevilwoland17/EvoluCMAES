"""Regenerate the RL comparison figures from like-for-like evaluations.

Outputs (written to the current directory):
    fig_bb84_rl_methods.pdf  -- BB84: paired difference of each policy from the
                                greedy oracle evaluated on the same 50 episodes,
                                with 95% confidence intervals.
    fig_e91_rl_methods.pdf   -- E91: mean Holevo information per policy with
                                +/- 1 standard error, grouped by evaluation run
                                (each run evaluates all of its policies on one
                                shared set of channel realisations).
    fig_e91_rl_training.pdf  -- E91 Rainbow-DQN evaluation curve with the oracle
                                and dynamic-programming references taken from the
                                same 200-episode evaluation (exp_7).

Run from the release root:
    python figure_scripts/make_rl_method_figures.py
"""
import os
import pickle

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load(name):
    with open(os.path.join(RES, name), "rb") as f:
        return pickle.load(f)


def paired_ci(x, y):
    """Mean and 95% normal-approximation CI of the paired difference x - y."""
    d = np.asarray(x, float) - np.asarray(y, float)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return m, 1.96 * se, len(d)


# --------------------------------------------------------------------------
# BB84: paired differences from the same-episode greedy oracle
# --------------------------------------------------------------------------
bb84_rows = []  # (label, mean diff, ci half-width, detections, n)
e5 = load("exp5_results.pkl")["results"]
e6 = load("exp6_results_v5.pkl")["results"]
e7s = load("exp7_results_rainbow_v3.pkl")["results"]
e7t = load("exp7_results_rainbow_v2.pkl")["results"]

for label, run, key in [
    ("PPO, unmasked (run A)", e5, "RL (PPO)"),
    ("RecurrentPPO, unmasked (run B)", e6, "RL top-1 (step 2560000)"),
    ("MPC (run B)", e6, "MPC base"),
    ("Noise-indexed lookup (run C)", e7s, "Static safe"),
    ("Rainbow-DQN, strict (run C)", e7s, "Rainbow top-1 (step 185000)"),
    ("Rainbow-DQN, tolerated (run D)", e7t, "Rainbow top-1 (step 750000)"),
]:
    m, ci, n = paired_ci(run[key]["f_ae"], run["Oracle"]["f_ae"])
    bb84_rows.append((label, m, ci, int(run[key]["det"]), n))

fig, ax = plt.subplots(figsize=(5.4, 2.5))
ys = np.arange(len(bb84_rows))[::-1]
for y, (label, m, ci, det, n) in zip(ys, bb84_rows):
    colour = "#2E7D32" if "Rainbow" in label else ("#777777" if "lookup" in label else "#1F5FA8")
    ax.errorbar(m, y, xerr=ci, fmt="o", color=colour, capsize=3, ms=4)
    ax.text(0.031, y, f"det. {100 * det / n:.0f}%", va="center", fontsize=7.5)
ax.axvline(0.0, color="black", lw=0.8, ls="--")
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in bb84_rows])
ax.set_xlabel(r"$F_{AE}$ minus greedy oracle on the same 50 episodes")
ax.set_xlim(-0.035, 0.045)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig("fig_bb84_rl_methods.pdf", bbox_inches="tight")
plt.close(fig)

print("BB84 paired differences (mean, 95% CI half-width, detections/n):")
for row in bb84_rows:
    print(f"  {row[0]:34s} {row[1]:+.4f} +/- {row[2]:.4f}  det {row[3]}/{row[4]}")


# --------------------------------------------------------------------------
# E91: grouped means with standard errors
# --------------------------------------------------------------------------
def e91_stats(entry):
    if "summary" in entry:
        s = entry["summary"]
        n = len(entry["actions"]) // 50
        return s["mean_chi"], s["std_chi"] / np.sqrt(n), s["det_rate"], n
    chi = np.asarray(entry["chi"], float)
    det = entry["det"]
    det_rate = float(np.mean(det)) if np.ndim(det) else float(det) / len(chi)
    return chi.mean(), chi.std(ddof=1) / np.sqrt(len(chi)), det_rate, len(chi)


r5 = load("e91_exp5_results.pkl")["results"]
r6 = load("e91_exp6_results.pkl")["results"]
r7 = load("e91_exp7_results.pkl")["results"]
groups = [
    ("Run A (50 episodes)", r5, [("PPO, unmasked", "PPO"), ("BC + iterated DAgger", "BC (Oracle)"),
                                 ("Greedy oracle", "Oracle"), ("DP, coarse grid", "DP")]),
    ("Run B (100 episodes)", r6, [("RecurrentPPO, unmasked", "RecurrentPPO"), ("MPC", "MPC"),
                                  ("Greedy oracle", "Oracle"), ("DP, coarse grid", "DP")]),
    ("Run C (200 episodes)", r7, [("MaskablePPO", "MaskablePPO"), ("Rainbow-DQN", "Rainbow-DQN"),
                                  ("Greedy oracle", "Oracle"), ("DP, coarse grid", "DP")]),
]

fig, ax = plt.subplots(figsize=(5.4, 3.8))
y = 0
yticks, ylabels = [], []
print("\nE91 (mean, s.e., detection rate, n):")
for gname, run, rows in groups:
    ax.text(0.002, y + 0.75, gname, fontsize=7.5, style="italic", va="center")
    y -= 0.5
    for label, key in rows:
        m, se, det, n = e91_stats(run[key])
        colour = "#2E7D32" if "Rainbow" in label else ("#555555" if ("oracle" in label or "DP" in label) else "#1F5FA8")
        ax.barh(y, m, xerr=se, color=colour, alpha=0.85, capsize=2, height=0.7)
        ax.text(m + se + 0.004, y, f"{m:.4f}" + (f"  (det. {100 * det:.0f}%)" if det > 0 else ""),
                va="center", fontsize=7)
        yticks.append(y)
        ylabels.append(label)
        print(f"  {gname:22s} {label:24s} {m:.4f} {se:.4f} {det:.2f} {n}")
        y -= 1
    y -= 0.6
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
ax.set_xlabel(r"Mean Holevo information $\chi(A{:}E)$ per round")
ax.set_xlim(0.0, 0.45)
ax.set_ylim(y + 0.4, 1.3)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig("fig_e91_rl_methods.pdf", bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------------------------------
# E91: training curve with same-run references
# --------------------------------------------------------------------------
e7 = load("e91_exp7_results.pkl")
hist = e7["rainbow_history"]
steps = np.array([h["step"] for h in hist])
chis = np.array([h["chi"] for h in hist])
dets = np.array([h["det"] for h in hist]) * 100
oracle = r7["Oracle"]["summary"]["mean_chi"]
dp = r7["DP"]["summary"]["mean_chi"]

fig, axes = plt.subplots(2, 1, figsize=(5.4, 3.3), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
axes[0].plot(steps, chis, color="#2E7D32", lw=1.6, marker="o", ms=2.5, label="Rainbow-DQN (20-episode evaluations)")
axes[0].axhline(oracle, color="black", ls="--", lw=1, label=f"Greedy oracle ({oracle:.4f})")
axes[0].axhline(dp, color="black", ls=":", lw=1, label=f"DP, coarse grid ({dp:.4f}, 7% det.)")
axes[0].axhline(0.0, color="grey", lw=1, alpha=0.6, label="Passive (identity)")
axes[0].set_ylabel(r"$\chi(A{:}E)$")
axes[0].set_ylim(-0.02, 0.42)
axes[0].legend(loc="lower right", fontsize=7)
axes[0].grid(alpha=0.3)
axes[1].plot(steps, dets, color="#2E7D32", lw=1.4)
axes[1].set_ylabel("Det. (%)")
axes[1].set_ylim(-1, 8)
axes[1].set_xlabel("Training steps")
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig_e91_rl_training.pdf", bbox_inches="tight")
plt.close(fig)
print("\nSaved fig_bb84_rl_methods.pdf, fig_e91_rl_methods.pdf, fig_e91_rl_training.pdf")
