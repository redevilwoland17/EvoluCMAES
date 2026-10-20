"""Extended MNIST budget sweep — high-B half (B in {2^15, 2^16}).

Companion to budget_sweep_mnist_60epochs.py (which covers B in {2^7..2^11})
and budget_sweep_mnist_60epochs_extended_low.py (which covers B in
{2^12, 2^13, 2^14}). Same setup as budget_sweep_mnist_60epochs.py:
N_EPOCH=60, POP=30, ELITISM=4, RESEED=6, unbiased qmut, alpha=0,
MNIST 7x7, MLP 49->16->10 (~970 params). Saves results incrementally to
budget_sweep_mnist_60epochs_extended_high_results.pkl after every (B, seed);
resumes if killed.

Usage:  python budget_sweep_mnist_60epochs_extended_high.py
"""
import os, time, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_sweep_mnist as bs

bs.N_EPOCH = 60
bs.N_JOBS  = 8

B_LIST = [2**k for k in range(15, 17)]   # 32768, 65536
SEEDS  = [0, 1, 2]
RESULTS_PATH = "budget_sweep_mnist_60epochs_extended_high_results.pkl"


def already_done(records, B, seed):
    return any(r['B'] == B and r['seed'] == seed for r in records)


def main():
    print(f"Extended MNIST sweep — high-B half",
          f"\n  B-range: {B_LIST}",
          f"\n  seeds:   {SEEDS}",
          f"\n  N_EPOCH: {bs.N_EPOCH}, POP={bs.POP_SIZE}, n_jobs={bs.N_JOBS}",
          f"\n  saves incrementally to {RESULTS_PATH}",
          flush=True)
    bs.activate_dataset()

    out = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "rb") as f:
            out = pickle.load(f).get("records", [])
        print(f"Resumed: {len(out)} records already saved.", flush=True)

    # Run smaller B and earlier seeds first (cheaper sanity-check signal first).
    runs = sorted([(B, s) for B in B_LIST for s in SEEDS])
    for idx, (B, seed) in enumerate(runs, 1):
        if already_done(out, B, seed):
            print(f"[{idx:2d}/{len(runs)}] B={B:6d} seed={seed}  (already done — skipped)",
                  flush=True)
            continue
        r = bs.run_one(seed, B)
        out.append(r)
        avg = r['avg_density_traj']
        first5 = sum(avg[:5]) / 5; last5 = sum(avg[-5:]) / 5
        print(f"[{idx:2d}/{len(runs)}] B={B:6d} seed={seed}  "
              f"rho_eq={r['rho_pop_avg_last5']:.4f}  "
              f"rho_best={r['rho_best_final']:.4f}  acc={r['acc_best_final']:.4f}  "
              f"(first5={first5:.4f} last5={last5:.4f})  "
              f"({r['elapsed']:.0f}s)",
              flush=True)
        with open(RESULTS_PATH, "wb") as f:
            pickle.dump({"records": out, "B_LIST": B_LIST, "SEEDS": SEEDS,
                         "N_EPOCH": bs.N_EPOCH, "POP_SIZE": bs.POP_SIZE,
                         "half": "high"}, f)
    print(f"\nSaved {RESULTS_PATH} ({len(out)} records).", flush=True)


if __name__ == "__main__":
    main()
