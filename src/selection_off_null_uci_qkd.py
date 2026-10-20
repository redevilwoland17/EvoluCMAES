"""Selection-off nulls for the UCI and QKD mutation operators (Appendix D).

Same construction as selection_off_null.py: keep the structure of the outer loop
(elite slots, reseeding, replacement by mutated copies of top-ranked parents,
in-place mutation of the rest) but make the ranking a uniformly random
permutation and skip the inner loop. Whatever the population-mean size does
then is due to the proposals and the reseeding alone.

UCI pruning (size = mask density):
  table2   N = 100, E = 10, R = 20, rho0 = 0.05, 40 epochs, with the
           density-agnostic (0.40/0.20/0.40), drift-neutral (0.40/0.30/0.30)
           and density-driven (target 0.05) operators
  table1   N = 30, E = 5, R = 10, rho0 = 0.05, 60 epochs, density-driven
  on the three network shapes (iris, wine, cancer).

QKD circuit search (size = gate count):
  bb84     replace / remove / add = 0.40 / 0.20 / 0.40 on two-qubit circuits,
           N = 100, E = 10, R = 20, five initial gates, 40 epochs
  e91      replace / remove / insert / perturb = 0.35 / 0.20 / 0.25 / 0.20 on
           three-qubit gate lists, N = 30, 8 elites, 12 reseeds, six initial
           gates, 25 epochs

The mutation operators are the ones used in the experiments: the UCI ones are
imported from mutation_operator_ablation_uci.py (swap / remove / add) and
tabular_benchmarks.py (density-driven); the two QKD ones are copied unchanged
from the BB84 circuit search and from evoluspsa_e91.py, with the E91 settings
of the library build in notebooks/QKD_experiment_5_E91.ipynb.

Output: results/mechanism/selection_null_uci_qkd_results.pkl. Takes about 15
minutes on one core, most of it in the UCI part; no dataset, no training.
Env overrides: SEEDS_OVERRIDE ("0,1,2"), PARTS_OVERRIDE ("uci,bb84,e91").
Usage:  python src/selection_off_null_uci_qkd.py [--smoke]
"""
import os, pickle, random, sys, tempfile, time
from copy import deepcopy
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutation_operator_ablation_uci as t2   # swap/remove/add operator, MaskedMLP
import tabular_benchmarks as t1               # density-driven operator

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = "--smoke" in sys.argv

SEEDS = [0, 1, 2]
PARTS = ["uci", "bb84", "e91"]
if os.environ.get("SEEDS_OVERRIDE"):
    SEEDS = [int(s) for s in os.environ["SEEDS_OVERRIDE"].split(",")]
if os.environ.get("PARTS_OVERRIDE"):
    PARTS = os.environ["PARTS_OVERRIDE"].split(",")
RESULTS = os.path.join(ROOT, "results", "mechanism", "selection_null_uci_qkd_results.pkl")
if SMOKE:
    SEEDS = [0]
    RESULTS = os.path.join(tempfile.gettempdir(), "selection_off_null_uci_qkd_smoke.pkl")
UCI_DIMS = {"iris": [4, 64, 32, 3], "wine": [13, 64, 32, 3], "cancer": [30, 64, 32, 2]}
F_TARGET = 0.05


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def null_loop(pop, n_epoch, elitism, reseed, new_individual, mutate, size):
    """Outer loop of the pruning and BB84 searches with an information-free ranking."""
    n = len(pop)
    traj = [float(np.mean([size(c) for c in pop]))]          # epoch 0 = initial population
    for _ in range(n_epoch):
        order = [int(i) for i in np.random.permutation(n)]   # random "ranking"
        lo, hi = elitism, n - reseed
        picks = set(np.random.choice(range(lo, hi), size=reseed, replace=False).tolist())
        for i in range(1, reseed + 1):
            pop[order[-i]] = new_individual()
        parent_ranks = [k // 2 for k in range(reseed)]
        for pr, i in zip(parent_ranks, sorted(picks)):
            pop[order[i]] = mutate(pop[order[pr]])
        for i in range(lo, hi):
            if i not in picks:
                pop[order[i]] = mutate(pop[order[i]])
        traj.append(float(np.mean([size(c) for c in pop])))
    return traj


# ---------------------------------------------------------------- UCI pruning
UCI_OPERATORS = {
    "density-agnostic 0.40/0.20/0.40": lambda m: t2.random_modif_quantum(m, t_swap=0.4, t_remove=0.6),
    "drift-neutral 0.40/0.30/0.30":    lambda m: t2.random_modif_quantum(m, t_swap=0.4, t_remove=0.7),
    "density-driven (target 0.05)":    lambda m: t1.random_modif(m, f_target=F_TARGET),
}
UCI_SETTINGS = {   # name: (N, E, R, epochs, operators)
    "table2": (100, 10, 20, 40, list(UCI_OPERATORS)),
    "table1": (30, 5, 10, 60, ["density-driven (target 0.05)"]),
}
if SMOKE:
    UCI_SETTINGS = {k: (N, E, R, 3, ops) for k, (N, E, R, _, ops) in UCI_SETTINGS.items()}


def run_uci():
    recs = []
    for setting, (N, E, R, n_epoch, ops) in UCI_SETTINGS.items():
        for op in ops:
            for ds, dims in UCI_DIMS.items():
                for seed in SEEDS:
                    set_all_seeds(seed)
                    make = (lambda d=dims: t2.random_ini(F_TARGET, d)) if "driven" not in op \
                        else (lambda d=dims: t1.random_ini(F_TARGET, d))
                    pop = [make() for _ in range(N)]
                    traj = null_loop(pop, n_epoch, E, R, make, UCI_OPERATORS[op], lambda m: m.density())
                    recs.append({"part": "uci", "setting": setting, "operator": op, "dataset": ds,
                                 "seed": seed, "N": N, "E": E, "R": R, "size": "density", "traj": traj})
                v = [r["traj"][-1] for r in recs[-len(SEEDS):]]
                print(f"  uci/{setting:6s} {op:32s} {ds:6s} density after {n_epoch} epochs = "
                      f"{np.mean(v):.4f} +/- {np.std(v):.4f}", flush=True)
    return recs


# ---------------------------------------------------------------------- BB84
def run_bb84(n_epoch=40, N=100, E=10, R=20):
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit import CircuitInstruction
    from qiskit.circuit.library import HGate, RXGate, RYGate, RZGate, MCXGate

    def create_base_circuit(n_q=2, n_c=3):
        return QuantumCircuit(QuantumRegister(n_q, "q"), ClassicalRegister(n_c, "c"))

    def random_gate(n_q):
        theta = np.random.uniform(0, 2*np.pi)
        if n_q == 2:
            pool = [RXGate(theta).control(1), RYGate(theta).control(1),
                    RZGate(theta).control(1), MCXGate(1)]
        else:
            pool = [RXGate(theta), RYGate(theta), RZGate(theta), HGate()]
        return pool[np.random.randint(len(pool))]

    def random_modif(circuit):
        circuit = circuit.copy()
        n = len(circuit.data)
        nq_circ = circuit.num_qubits
        a = np.random.random()
        if a < 0.4 and n > 2:
            idx = np.random.randint(0, n)
            old_nq = len(circuit.data[idx].qubits)
            gate = random_gate(old_nq)
            qs = [circuit.qubits[q] for q in
                  np.random.choice(nq_circ, size=old_nq, replace=False)]
            circuit.data[idx] = CircuitInstruction(gate, qs)
            return circuit
        if a < 0.6 and n > 3:
            circuit.data.pop(np.random.randint(0, n))
            return circuit
        nq = np.random.choice([1, 2])
        qs = list(np.random.choice(nq_circ, size=nq, replace=False))
        circuit.append(random_gate(nq), qs)
        return circuit

    def random_ini(circuit, n_gates=5):
        for _ in range(n_gates):
            circuit = circuit.copy()
            nq = np.random.choice([1, 2])
            qs = list(np.random.choice(circuit.num_qubits, size=nq, replace=False))
            circuit.append(random_gate(nq), qs)
        return circuit

    recs = []
    for seed in SEEDS:
        set_all_seeds(seed)
        make = lambda: random_ini(create_base_circuit())
        pop = [make() for _ in range(N)]
        traj = null_loop(pop, n_epoch, E, R, make, random_modif, lambda c: len(c.data))
        recs.append({"part": "bb84", "setting": "library search", "seed": seed, "N": N, "E": E, "R": R,
                     "operator": "replace/remove/add 0.40/0.20/0.40", "size": "gates", "traj": traj})
    v = [r["traj"][-1] for r in recs]
    print(f"  bb84   mean gate count after {n_epoch} epochs = {np.mean(v):.2f} +/- {np.std(v):.2f} "
          f"(initial 5)", flush=True)
    return recs


# ----------------------------------------------------------------------- E91
RX, RY, RZ, H, CX = 'rx', 'ry', 'rz', 'h', 'cx'
ALL_1Q_GATES = [RX, RY, RZ, H]
PARAM_GATES = {RX, RY, RZ}
N_QUBITS = 3
INIT_GATES = 6


def random_gate_1q():
    gt = np.random.choice(ALL_1Q_GATES)
    param = np.random.uniform(0, 2*np.pi) if gt in PARAM_GATES else None
    qubit = np.random.randint(0, N_QUBITS)
    return (gt, [qubit], param)


def random_gate_2q():
    q0, q1 = np.random.choice(N_QUBITS, size=2, replace=False).tolist()
    return (CX, [q0, q1], None)


def random_gate_any():
    return random_gate_1q() if np.random.random() < 0.6 else random_gate_2q()


def random_circuit(n_gates=INIT_GATES):
    return [random_gate_any() for _ in range(n_gates)]


def mutate(circuit):
    circuit = deepcopy(circuit)
    n = len(circuit)
    action = np.random.random()
    if action < 0.35 and n > 2:
        idx = np.random.randint(0, n)
        circuit[idx] = random_gate_any()
    elif action < 0.55 and n > 3:
        circuit.pop(np.random.randint(0, n))
    elif action < 0.80:
        circuit.insert(np.random.randint(0, n+1), random_gate_any())
    else:
        pidx = [i for i, g in enumerate(circuit) if g[2] is not None]
        if pidx:
            idx = np.random.choice(pidx)
            gt, qubits, param = circuit[idx]
            circuit[idx] = (gt, qubits, param + np.random.normal(0, 0.3))
    return circuit


def run_e91(n_epoch=25, N=30, n_elite=8, n_replace=12):
    recs = []
    for seed in SEEDS:
        np.random.seed(seed)
        population = [random_circuit(INIT_GATES) for _ in range(N)]
        traj = [float(np.mean([len(c) for c in population]))]
        for _ in range(n_epoch):
            population = [population[i] for i in np.random.permutation(N)]   # random "ranking"
            new_pop = [deepcopy(population[i]) for i in range(n_elite)]
            for _ in range(n_replace):
                new_pop.append(random_circuit(INIT_GATES))
            for i in range(N - n_elite - n_replace):
                if np.random.random() < 0.5:
                    parent = deepcopy(population[np.random.randint(0, n_elite)])
                else:
                    parent = deepcopy(population[min(n_elite + i, N - 1)])
                new_pop.append(mutate(parent))
            population = new_pop
            traj.append(float(np.mean([len(c) for c in population])))
        recs.append({"part": "e91", "setting": "library search", "seed": seed, "N": N, "E": n_elite,
                     "R": n_replace, "operator": "replace/remove/insert/perturb 0.35/0.20/0.25/0.20",
                     "size": "gates", "traj": traj})
    v = [r["traj"][-1] for r in recs]
    print(f"  e91    mean gate count after {n_epoch} epochs = {np.mean(v):.2f} +/- {np.std(v):.2f} "
          f"(initial 6)", flush=True)
    return recs


def main():
    print(f"Selection-off nulls, parts={PARTS}, seeds={SEEDS}", flush=True)
    t0 = time.time(); records = []
    for part, fn in (("uci", run_uci), ("bb84", run_bb84), ("e91", run_e91)):
        if part in PARTS:
            records += fn(n_epoch=3) if SMOKE and part != "uci" else fn()
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "wb") as f:
        pickle.dump({"records": records, "seeds": SEEDS}, f)
    print(f"Saved {RESULTS if SMOKE else os.path.relpath(RESULTS, ROOT)} "
          f"({len(records)} records, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
