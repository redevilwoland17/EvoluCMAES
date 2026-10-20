"""Figures 3-6 (the controlled mechanism tests) from results/mechanism/.

Figure 3  fig_teacher_student_equilibria.pdf   teacher-student rho*(B) for both
          initial densities, with the variance-collapse budgets
Figure 4  fig_teacher_student_fit_quality.pdf  (a) best-individual clean error at
          equilibrium, (b) clean error after one cold-start inner run
Figure 5  fig_inner_optimiser_mnist.pdf        inner-optimiser ablation, MNIST
Figure 6  fig_operator_drift_null.pdf          400-epoch MNIST runs under both
          operators against the selection-off null

Data are read through _paths.data(); like the other figure scripts, this one
writes the PDFs (vector) to the current directory.
Usage:  python figure_scripts/make_mechanism_figures.py
"""
import glob, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths

OUT = os.getcwd()


def data(name):
    return _paths.data(f"mechanism/{name}")

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # slots 1-4
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11,
})


def load(pattern):
    recs = []
    paths = sorted(glob.glob(str(data(pattern))))
    assert paths, f"no result files match {pattern}"
    for p in paths:
        recs += pickle.load(open(p, "rb"))["records"]
    return recs


def agg(recs, key="rho_eq_last40", by="B", flt=lambda r: True):
    out = {}
    for r in recs:
        if flt(r):
            out.setdefault(r[by], []).append(r[key])
    Bs = sorted(out)
    return (np.array(Bs), np.array([np.mean(out[b]) for b in Bs]),
            np.array([np.std(out[b]) for b in Bs]))


# ---------------------------------------------------------------- Figure 3
e10 = load("expE_results_e10-s*.pkl") + load("expE_results_e10x-s*.pkl")
e30 = load("expE_results_e30-s*.pkl")
e30x = load("expE_results_e30x-s*.pkl")
# strict-criterion re-runs: keep only engaged seeds (clean MSE < 0.05)
e30x_ok = [r for r in e30x if r["best_clean_mse_traj"][-1] < 0.05]

fig, ax = plt.subplots(figsize=(7.6, 4.6))
B1, m1, s1 = agg(e10)
B3, m3, s3 = agg(e30)
Bx, mx, sx = agg(e30x_ok)

ax.axhline(0.10, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
ax.annotate("teacher density 0.10", xy=(150, 0.102), fontsize=8.5,
            color=INK2, va="bottom")

for Bv, lbl in ((6000, "Var[$\\hat f$]($\\rho{=}0.2$) collapses"),
                (24000, "Var[$\\hat f$]($\\rho{=}0.4$) collapses")):
    ax.axvline(Bv, color=INK2, linewidth=0.9, linestyle=":", zorder=1)
    ax.annotate(lbl, xy=(Bv, 0.335), fontsize=7.5, color=INK2,
                rotation=90, ha="right", va="top")

ax.errorbar(B1, m1, yerr=s1, color=C1, marker="o", markersize=6.5,
            linewidth=2, capsize=3, zorder=4, label="init $\\rho_0=0.10$ (sparse basin)")
ax.errorbar(B3, m3, yerr=s3, color=C2, marker="s", markersize=6.5,
            linewidth=2, capsize=3, zorder=4, label="init $\\rho_0=0.30$ (dense basin)")
ax.errorbar(Bx, mx, yerr=sx, color=C2, marker="s", markersize=7.5,
            markerfacecolor=SURF, linewidth=0, capsize=3, zorder=5,
            label="dense basin, strict criterion")

ax.annotate("monotone Corollary-1 shift:\n0.054 → 0.097 over 3 octaves",
            xy=(11500, 0.036), fontsize=8.5, color=C1, ha="center", va="center")
ax.annotate("hysteresis gap:\ntwo stable equilibria",
            xy=(7800, 0.185), fontsize=8.5, color=INK2, ha="right", va="center")
ax.annotate("", xy=(8192, 0.252), xytext=(8192, 0.078),
            arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.1))

ax.set_xscale("log", base=2)
ax.set_xlabel("Inner-loop budget $B$ (fitness evaluations per individual)")
ax.set_ylabel("Equilibrium density $\\rho^*$ (pop. mean, plateaued)")
ax.set_title("Teacher–student neutral-capacity task: budget-gated equilibria (5 seeds/cell)")
ax.set_ylim(0.02, 0.36)
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_teacher_student_equilibria.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 5
d = pickle.load(open(data("expB_inner_ablation_mnist_results.pkl"), "rb"))
order = ["cmaes", "random", "fdsgd", "adam"]
names = ["CMA-ES", "Random", "FD-SGD", "Adam"]
cols = [C1, C2, C3, C4]
rho = {o: [r["rho_pop_avg_last5"] for r in d["records"] if r["optimizer"] == o] for o in order}
acc = {o: [r["acc_best_final"] for r in d["records"] if r["optimizer"] == o] for o in order}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.6))
x = np.arange(4)
lo = min(np.mean(rho[o]) for o in order)
hi = max(np.mean(rho[o]) for o in order)
a1.axhspan(lo - 0.01, hi + 0.01, color=GRID, zorder=0)
a1.annotate("shared band 0.19–0.22\n(dense $= 1.0$, off scale)", xy=(1.5, 0.245),
            ha="center", fontsize=8.5, color=INK2)
for i, o in enumerate(order):
    a1.errorbar(i, np.mean(rho[o]), yerr=np.std(rho[o]), color=cols[i],
                marker="o", markersize=9, capsize=4, linewidth=0, elinewidth=2)
    a1.annotate(f"{np.mean(rho[o]):.3f}", xy=(i, np.mean(rho[o]) - 0.028),
                ha="center", fontsize=8.5, color=INK)
    a2.errorbar(i, np.mean(acc[o]), yerr=np.std(acc[o]), color=cols[i],
                marker="o", markersize=9, capsize=4, linewidth=0, elinewidth=2)
    a2.annotate(f"{np.mean(acc[o]):.3f}", xy=(i, np.mean(acc[o]) - 0.045),
                ha="center", fontsize=8.5, color=INK)
a2.axhline(0.84, color=INK2, linewidth=1.1, linestyle=(0, (4, 3)))
a2.annotate("dense-model ceiling", xy=(0.0, 0.845), fontsize=8, color=INK2, va="bottom")
for a, ylab, ttl in ((a1, "Equilibrium density $\\rho^*$", "Same equilibrium band…"),
                     (a2, "Best test accuracy", "…at very different optimiser strength")):
    a.set_xticks(x); a.set_xticklabels(names, fontsize=9)
    a.set_ylabel(ylab); a.set_title(ttl, fontsize=10)
a1.set_ylim(0.10, 0.30); a2.set_ylim(0.45, 0.92)
fig.suptitle("Inner-optimiser ablation — MNIST 7×7 pruning, matched evaluation budget B=2048 (3 seeds)",
             fontsize=10.5, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_inner_optimiser_mnist.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 6
qa = pickle.load(open(data("expA_pretest_results.pkl"), "rb"))["records"]
sg = load("expG_results_g-s*.pkl")
nul = pickle.load(open(data("expF_selection_null_results.pkl"), "rb"))["records"]
BCOL = {512: C1, 2048: C2, 8192: C3}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
for a, recs, ttl in ((a1, qa, "Density-agnostic operator (net growth bias)"),
                     (a2, sg, "Drift-neutral operator")):
    ntraj = min(len(r["avg_density_traj"]) for r in nul)
    band = np.array([r["avg_density_traj"][:ntraj] for r in nul])
    a.fill_between(range(ntraj), band.min(0), band.max(0), color=GRID, zorder=0)
    for B in (512, 2048, 8192):
        v = [r for r in recs if r["B"] == B]
        if not v: continue
        n = min(len(r["avg_density_traj"]) for r in v)
        tr = np.array([r["avg_density_traj"][:n] for r in v])
        a.plot(range(n), tr.mean(0), color=BCOL[B], linewidth=2,
               label=f"B={B}")
        a.fill_between(range(n), tr.mean(0) - tr.std(0), tr.mean(0) + tr.std(0),
                       color=BCOL[B], alpha=0.18, linewidth=0)
    a.set_title(ttl, fontsize=10)
    a.set_xlabel("Outer-loop epoch")
    a.set_xlim(0, 400)
a1.annotate("selection-off drift null", xy=(200, 0.125), fontsize=8, color=INK2,
            ha="center")
a1.set_ylabel("Population-mean density $\\rho$")
a1.legend(loc="upper left", fontsize=8.5)
a2.legend(loc="upper left", fontsize=8.5)
a1.set_ylim(0.0, 0.62)
fig.suptitle("MNIST 7×7, 400-epoch horizon: two mutation operators against the selection-off null (2–3 seeds/curve)",
             fontsize=10.5, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_operator_drift_null.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 4
# (a) Fit quality of the best individual at equilibrium against B, both
#     initial densities: does the low-B plateau come from a search that fails?
# (b) One cold-start inner run without the outer loop (variance grid, mean
#     clean MSE per fixed rho): the budget one individual needs at each density.
e30_all = e30 + e30x
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.9))

for recs, col, mk, lbl in ((e10, C1, "o", "init $\\rho_0=0.10$"),
                           (e30_all, C2, "s", "init $\\rho_0=0.30$")):
    by = {}
    for r in recs:
        by.setdefault(r["B"], []).append(r["best_clean_mse_traj"][-1])
    Bs = sorted(by)
    for B in Bs:                                   # per-seed dots (bimodality)
        a1.plot([B] * len(by[B]), by[B], mk, color=col, markersize=4,
                alpha=0.35, markeredgewidth=0)
    a1.plot(Bs, [np.median(by[B]) for B in Bs], color=col, marker=mk,
            markersize=6.5, linewidth=2, label=lbl + " (median)")
a1.set_xscale("log", base=2); a1.set_yscale("log")
a1.set_xlabel("Inner-loop budget $B$")
a1.set_ylabel("Final clean MSE (best individual)")
a1.set_title("(a) With the outer loop: fit quality at equilibrium", fontsize=10)
a1.annotate("dense-init runs at low $B$:\n11 of 15 seeds stay near init\n(median MSE$\\approx$0.16)",
            xy=(200, 0.05), fontsize=8, color=C2)
a1.annotate("sparse basin fits well\nat every budget", xy=(2500, 0.00115),
            fontsize=8, color=C1, ha="center")
a1.legend(fontsize=8, loc="center right")

cells = pickle.load(open(data("expI_results_i.pkl"), "rb"))["cells"]
RHOS_I = [0.05, 0.10, 0.20, 0.40, 0.70, 1.00]
BS_I = sorted({B for (_, B) in cells})
seq = ["#bcd7f2", "#8fbceb", "#5f9fdf", "#3a7fce", "#2563ac", "#174a85"]
for c, r0 in zip(seq, RHOS_I):
    ys = [cells[(r0, B)]["mean_clean"] for B in BS_I]
    a2.plot(BS_I, ys, color=c, marker="o", markersize=4.5, linewidth=1.8,
            label=f"$\\rho={r0:.2f}$")
a2.set_xscale("log", base=2); a2.set_yscale("log")
a2.set_xlabel("Inner-loop budget $B$")
a2.set_ylabel("Clean MSE after ONE inner run")
a2.set_title("(b) No outer loop: single individual, cold start", fontsize=10)
a2.legend(fontsize=7.5, ncol=2, loc="lower left", framealpha=0.9)
fig.suptitle("Teacher–student: optimisation quality vs. budget — population equilibria (a) and per-density convergence without evolution (b)",
             fontsize=10, y=1.03)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_teacher_student_fit_quality.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved Figures 3-6 to", OUT)
