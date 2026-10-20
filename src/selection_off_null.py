"""Selection-off null of the MNIST / teacher–student outer loop (Appendix D;
the grey band of Figure 6).

What does the population-mean density do under the outer loop's structural
steps alone (elite slots, reseeding at the initial density, replacement by
mutated copies of top-ranked parents, in-place mutation) when selection
carries no information? The loop is that of bs.train_evolutive with the
ranking replaced by a uniformly random permutation at every epoch and the
inner loop skipped (with a random ranking, trained weights cannot affect
survival, so the budget B plays no role).

Two mutation operators, three seeds each, N = 30, E = 4, R = 6, rho0 = 0.10,
400 epochs:
  'qmut'      — density-agnostic: swap 40% / remove 20% / add 40%
                (expected growth +0.2 active entries per flip)
  'symmetric' — drift-neutral: swap 40% / remove 30% / add 30%

Because the inner loop is skipped, the null depends on the operator, N, E, R
and rho0 but not on the task, so it serves both the MNIST runs and the
teacher–student benchmark.

Output: results/mechanism/expF_selection_null_results.pkl. Runs in a few
minutes; no dataset, no training.
Env overrides (the defaults reproduce the stored file): RHO_INIT_OVERRIDE,
MUTATIONS_OVERRIDE ("symmetric"), RESULTS_TAG (writes
expF_selection_null_results_<tag>.pkl).
Usage:  python src/selection_off_null.py
"""
import os, sys, time, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_sweep_mnist as bs

N_EPOCH   = 400
SEEDS     = [0, 1, 2]
POP_SIZE  = bs.POP_SIZE      # 30
ELITISM   = bs.ELITISM       # 4
RESEED    = bs.RESEED        # 6
RHO_INIT  = bs.F_TARGET      # 0.10
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS   = os.path.join(ROOT, "results", "mechanism", "expF_selection_null_results.pkl")
if os.environ.get("RHO_INIT_OVERRIDE"):
    RHO_INIT = float(os.environ["RHO_INIT_OVERRIDE"])
if os.environ.get("RESULTS_TAG"):
    RESULTS = os.path.join(ROOT, "results", "mechanism",
                           f"expF_selection_null_results_{os.environ['RESULTS_TAG']}.pkl")


def symmetric_modif(model, n_flips=None, min_density=bs.MIN_DENSITY):
    """bs.random_modif_quantum with drift-unbiased weights 40/30/30."""
    model = model.clone_arch()
    total = sum(m.numel() for m in model.masks)
    if n_flips is None:
        n_flips = max(1, int(0.02 * total))
    min_active = max(1, int(min_density * total))
    for _ in range(n_flips):
        active_count = sum(int(m.data.sum().item()) for m in model.masks)
        a = np.random.random()
        if a < 0.4 and 0 < active_count < total:
            removed = bs._flip_one(model, current=1.0, target=0.0)
            if removed is None: continue
            added = bs._flip_one(model, current=0.0, target=1.0)
            if added is None:
                model.masks[removed[0]].data[removed[1]] = 1.0
        elif a < 0.7 and active_count > min_active:
            bs._flip_one(model, current=1.0, target=0.0)
        else:
            if active_count < total:
                bs._flip_one(model, current=0.0, target=1.0)
    return model


MUTATIONS = {"qmut": bs.random_modif_quantum, "symmetric": symmetric_modif}
if os.environ.get("MUTATIONS_OVERRIDE"):
    MUTATIONS = {k: MUTATIONS[k] for k in os.environ["MUTATIONS_OVERRIDE"].split(",")}


def run_null(seed, mutation_fn):
    bs.set_all_seeds(seed)
    circuits = [bs.random_ini(density=RHO_INIT) for _ in range(POP_SIZE)]
    avg = []
    for epoch in range(N_EPOCH):
        order = np.random.permutation(POP_SIZE)          # information-free "ranking"
        scored = [(0.0, int(i)) for i in order]
        lo, hi = ELITISM, POP_SIZE - RESEED
        picks = set(np.random.choice(range(lo, hi), size=RESEED,
                                     replace=False).tolist())
        for i in range(1, RESEED + 1):
            circuits[scored[-i][1]] = bs.random_ini(density=RHO_INIT)
        parent_ranks = [k // 2 for k in range(RESEED)]
        for pr, i in zip(parent_ranks, sorted(picks)):
            circuits[scored[i][1]] = mutation_fn(circuits[scored[pr][1]])
        for i in range(lo, hi):
            if i not in picks:
                circuits[scored[i][1]] = mutation_fn(circuits[scored[i][1]])
        avg.append(float(np.mean([m.density() for m in circuits])))
    return avg


def main():
    print(f"Selection-off null (POP={POP_SIZE}, E={ELITISM}, "
          f"R={RESEED}, rho_init={RHO_INIT}, {N_EPOCH} epochs)", flush=True)
    records = []
    for mut_name, mut_fn in MUTATIONS.items():
        for seed in SEEDS:
            t0 = time.time()
            traj = run_null(seed, mut_fn)
            records.append({"mutation": mut_name, "seed": seed,
                            "avg_density_traj": traj,
                            "rho_last50": float(np.mean(traj[-50:]))})
            cps = {c: round(float(np.mean(traj[c-10:c])), 4)
                   for c in (60, 100, 200, 300, 400)}
            print(f"  {mut_name:9s} seed={seed}  rho_last50="
                  f"{records[-1]['rho_last50']:.4f}  checkpoints={cps}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
    print("\nSummary (rho over last 50 epochs):", flush=True)
    for mut_name in MUTATIONS:
        vals = [r["rho_last50"] for r in records if r["mutation"] == mut_name]
        print(f"  {mut_name:9s}  {np.mean(vals):.4f} +/- {np.std(vals):.4f}",
              flush=True)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "wb") as f:
        pickle.dump({"records": records,
                     "config": dict(N_EPOCH=N_EPOCH, POP_SIZE=POP_SIZE,
                                    ELITISM=ELITISM, RESEED=RESEED,
                                    RHO_INIT=RHO_INIT)}, f)
    print(f"Saved {os.path.relpath(RESULTS, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
