"""Probe whether 15 outer-loop epochs is enough to reach the stationary ρ.

Hypothesis: at slow-selection regimes (real datasets where the ρ-shift signal
is small), 15 epochs may not be enough for the equilibrium to settle. Run
N_EPOCH=60 at fixed B=2^14 and compare to the 15-epoch sweep result.

Setup: same as budget_sweep_mnist.py except
    N_EPOCH = 60   (vs. 15 baseline)
    B = 16384      (= 2^14)
    seeds 0,1,2
    n_jobs = 4     (low core count so it co-exists with extended sweep)

Per-(B, seed) cost ≈ 30 pop × 60 epoch × 16384 × 0.21 ms / 4 cores ≈ 26 min.
3 seeds → ~80 min.

Output:  budget_sweep_mnist_60epochs_results.pkl
"""
import os, time, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_sweep_mnist as bs

# Overrides
bs.N_EPOCH = 60
bs.N_JOBS  = 4         # leave 12 cores for extended sweep + system

B_LIST = [128, 256, 512, 1024, 2048]   # 2^7 ... 2^11
SEEDS = [0, 1, 2]


def main():
    print(f"60-epoch probe at B in {B_LIST}", flush=True)
    print(f"  N_EPOCH={bs.N_EPOCH}, POP={bs.POP_SIZE}, n_jobs={bs.N_JOBS}", flush=True)
    bs.activate_dataset()
    out = []
    runs = [(B, s) for B in B_LIST for s in SEEDS]
    for idx, (B, seed) in enumerate(runs, 1):
        r = bs.run_one(seed, B)
        out.append(r)
        avg_traj = r['avg_density_traj']
        first5 = sum(avg_traj[:5]) / 5
        mid5 = sum(avg_traj[len(avg_traj)//2 - 2:len(avg_traj)//2 + 3]) / 5
        last5 = sum(avg_traj[-5:]) / 5
        print(f"[{idx:2d}/{len(runs)}] B={B:5d} seed={seed}  "
              f"rho_eq={r['rho_pop_avg_last5']:.4f}  acc={r['acc_best_final']:.4f}  "
              f"first5={first5:.4f} mid5={mid5:.4f} last5={last5:.4f}  "
              f"({r['elapsed']:.0f}s)", flush=True)
        with open("budget_sweep_mnist_60epochs_results.pkl", "wb") as f:
            pickle.dump({"records": out, "B_LIST": B_LIST, "N_EPOCH": bs.N_EPOCH,
                         "SEEDS": SEEDS}, f)
    print("Saved budget_sweep_mnist_60epochs_results.pkl", flush=True)


if __name__ == "__main__":
    main()
