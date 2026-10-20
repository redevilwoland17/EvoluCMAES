"""Supplementary inner-optimiser check on BB84 (three seeds).

This script runs the same outer evolutionary loop on a BB84 cloning task with
three inner optimisers (random search, CMA-ES and a finite-difference/SPSA-style
gradient estimate) at a matched evaluation budget per individual. It is a
lightweight three-seed check; the five-seed comparison reported in the paper
(Table "tab:bb84_inner_loop") comes from the BB84 experiment-4 notebook, which is
listed in README.md.

Setup:
  - Outer loop: identical to qkd_experiment_4_e91.run_evolutive --
    population, elite, replace, MAP-Elites archive, mutate.
  - BB84 physics: 2-qubit Eve circuit (travelling qubit + 1 ancilla). A bit flip
    of rate p_bf acts on the travelling qubit BEFORE Eve's interaction, following
    Decker et al. (npj Quantum Inf. 11, 140, 2025); fidelities as in
    bb84_lib.compute_fidelities.
  - Fitness: f = F_AE  -  alpha * max(0, F_AB_target - F_AB)^2
    Eve maximises F_AE while keeping F_AB above F_AB_target.
  - Inner optimisers: random search, CMA-ES and fdsgd (an SPSA-style
    finite-difference gradient estimate).
  - All three inner optimisers run at the same fitness-eval budget B per
    individual (same convention as Experiment #2 and the budget sweep).

Output:  bb84_random_vs_cmaes_results.pkl
         fig_bb84_random_vs_cmaes.pdf

Usage:  python bb84_random_vs_cmaes.py
"""
import os, time, pickle, warnings, copy
from copy import deepcopy
import numpy as np
import torch
import cma
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
C = torch.complex128

# Pre-build BB84 BB84-state operators in complex128 (bb84_lib uses complex64,
# incompatible with our complex128 unitaries — so we re-derive locally).
_I2_loc = torch.eye(2, dtype=C)
_X_loc  = torch.tensor([[0, 1], [1, 0]], dtype=C)
_ket0_loc = torch.tensor([1, 0], dtype=C)
_rho_eve_loc = torch.outer(_ket0_loc, _ket0_loc.conj())
_LABELS_loc = {
    '0': torch.tensor([1, 0], dtype=C),
    '1': torch.tensor([0, 1], dtype=C),
    '+': torch.tensor([1, 1], dtype=C) / np.sqrt(2),
    '-': torch.tensor([1, -1], dtype=C) / np.sqrt(2),
}
_bb84_data_loc = []
for label, psi in _LABELS_loc.items():
    proj = torch.outer(psi, psi.conj())
    rho_clean = torch.kron(proj, _rho_eve_loc)
    rho_flip  = torch.kron(_X_loc @ proj @ _X_loc.conj().T, _rho_eve_loc)
    proj_A = torch.kron(proj, _I2_loc)
    proj_E = torch.kron(_I2_loc, proj)
    _bb84_data_loc.append((rho_clean, rho_flip, proj_A, proj_E))


def compute_fidelities(U, p):
    """BB84 (F_AB, F_AE) at bit-flip rate p, complex128 throughout."""
    f_ab = torch.zeros((), dtype=torch.float64)
    f_ae = torch.zeros((), dtype=torch.float64)
    for rho_clean, rho_flip, proj_A, proj_E in _bb84_data_loc:
        rho_in  = (1 - p) * rho_clean + p * rho_flip
        rho_out = U @ rho_in @ U.conj().T
        f_ab = f_ab + torch.trace(proj_A @ rho_out).real
        f_ae = f_ae + torch.trace(proj_E @ rho_out).real
    return float(f_ab / 4), float(f_ae / 4)

# -------- Config (paper-faithful) ---------------------------------
N_QUBITS      = 2                           # travelling qubit + 1 Eve ancilla
P_BF          = 0.10                        # bit-flip rate (mid-range, OU mean)
F_AB_TARGET   = 0.90                        # safe target above 0.87 threshold
ALPHA         = 100.0                       # F_AB-penalty weight
INNER_BUDGET  = 150
OPTIMIZERS    = ["cmaes", "random", "fdsgd"]
SEEDS         = [0, 1, 2]
INIT_GATES    = 6
POP_SIZE      = 30
N_EPOCHS      = 12
N_ELITE       = 4
N_REPLACE     = 6

RX, RY, RZ, H, CX = 'rx', 'ry', 'rz', 'h', 'cx'
PARAM_GATES = {RX, RY, RZ}
ALL_1Q_GATES = [RX, RY, RZ, H]

# -------- Gate matrices (2-qubit) ---------------------------------
I2 = torch.eye(2, dtype=C)
H_mat = torch.tensor([[1, 1], [1, -1]], dtype=C) / np.sqrt(2)


def rx_mat(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    return torch.stack([torch.stack([c + 0j, -1j * s]),
                        torch.stack([-1j * s, c + 0j])])


def ry_mat(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    return torch.stack([torch.stack([c + 0j, -s + 0j]),
                        torch.stack([s + 0j, c + 0j])])


def rz_mat(t):
    return torch.diag(torch.stack([torch.exp(-1j * t / 2), torch.exp(1j * t / 2)]))


def gate_matrix(gate_type, param=0.0):
    if gate_type == RX: return rx_mat(torch.tensor(param))
    if gate_type == RY: return ry_mat(torch.tensor(param))
    if gate_type == RZ: return rz_mat(torch.tensor(param))
    if gate_type == H:  return H_mat
    raise ValueError(f"Unknown gate: {gate_type}")


def embed_1q(mat, qubit):
    ops = [I2] * N_QUBITS
    ops[qubit] = mat
    out = ops[0]
    for op in ops[1:]:
        out = torch.kron(out, op)
    return out


def embed_cx(control, target):
    dim = 2 ** N_QUBITS
    U = torch.zeros(dim, dim, dtype=C)
    for state in range(dim):
        bits = [(state >> (N_QUBITS - 1 - q)) & 1 for q in range(N_QUBITS)]
        new_bits = bits.copy()
        new_bits[target] ^= bits[control]
        new_state = sum(new_bits[q] << (N_QUBITS - 1 - q) for q in range(N_QUBITS))
        U[new_state, state] = 1.0
    return U


CX_CACHE = {(c, t): embed_cx(c, t) for c in range(N_QUBITS) for t in range(N_QUBITS) if c != t}


def circuit_to_unitary(circuit):
    U = torch.eye(2 ** N_QUBITS, dtype=C)
    for (gt, qubits, param) in circuit:
        if gt == CX:
            G = CX_CACHE[(qubits[0], qubits[1])]
        elif gt == H:
            G = embed_1q(H_mat, qubits[0])
        elif gt in PARAM_GATES:
            G = embed_1q(gate_matrix(gt, param), qubits[0])
        else:
            raise ValueError(f"Unknown gate: {gt}")
        U = G @ U
    return U


# -------- Circuit representation (BB84) ---------------------------
def random_gate_1q():
    gt = np.random.choice(ALL_1Q_GATES)
    param = np.random.uniform(0, 2 * np.pi) if gt in PARAM_GATES else None
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
    """Same primitives as evoluspsa_e91.mutate (the paper's symmetric add/remove/swap)."""
    circuit = deepcopy(circuit)
    n = len(circuit)
    a = np.random.random()
    if a < 0.35 and n > 2:
        idx = np.random.randint(0, n)
        circuit[idx] = random_gate_any()
    elif a < 0.55 and n > 3:
        circuit.pop(np.random.randint(0, n))
    elif a < 0.80:
        circuit.insert(np.random.randint(0, n + 1), random_gate_any())
    else:
        pidx = [i for i, g in enumerate(circuit) if g[2] is not None]
        if pidx:
            idx = np.random.choice(pidx)
            gt, qubits, param = circuit[idx]
            circuit[idx] = (gt, qubits, param + np.random.normal(0, 0.3))
    return circuit


def random_individual(n_gates=INIT_GATES):
    return random_circuit(n_gates)


def mutate_individual(individual):
    return mutate(individual)


def extract_all_params(individual):
    return np.array([g[2] for g in individual if g[2] is not None], dtype=np.float64)


def inject_all_params(individual, params):
    new_circuit = deepcopy(individual)
    idx = 0
    for i, (gt, qubits, param) in enumerate(new_circuit):
        if param is not None:
            new_circuit[i] = (gt, qubits, float(params[idx]))
            idx += 1
    return new_circuit


# -------- BB84 fitness --------------------------------------------
def fitness(individual, p_bf=P_BF, f_ab_target=F_AB_TARGET, alpha=ALPHA):
    """f = F_AE  -  alpha * max(0, F_AB_target - F_AB)^2.
    Eve wants to maximise F_AE without dragging F_AB below F_AB_target."""
    try:
        U = circuit_to_unitary(individual)
        f_ab, f_ae = compute_fidelities(U, p_bf)
        violation = max(0.0, f_ab_target - f_ab)
        f = f_ae - alpha * violation ** 2
        return f, f_ab, f_ae
    except Exception:
        return -1e6, 0.0, 0.0


# -------- Pluggable inner optimisers (all consume B fitness evals) ---
def refine_cmaes(individual, p_bf, f_ab_target, alpha, B, sigma=0.3, seed=0):
    params = extract_all_params(individual)
    if len(params) == 0:
        f, f_ab, f_ae = fitness(individual, p_bf, f_ab_target, alpha)
        return individual, f, f_ab, f_ae, 1
    es = cma.CMAEvolutionStrategy(
        params.tolist(), sigma,
        {'maxfevals': B, 'verbose': -9, 'tolx': 1e-6, 'tolfun': 1e-6, 'seed': int(seed)},
    )
    best_f, best_params = -1e6, params.copy()
    best_fab, best_fae = 0.0, 0.0
    n_eval = 0
    while not es.stop():
        sols = es.ask()
        vals = []
        for s in sols:
            f, f_ab, f_ae = fitness(inject_all_params(individual, np.asarray(s)),
                                    p_bf, f_ab_target, alpha)
            vals.append(-f); n_eval += 1
            if f > best_f:
                best_f, best_params = f, np.asarray(s).copy()
                best_fab, best_fae = f_ab, f_ae
        es.tell(sols, vals)
    return inject_all_params(individual, best_params), best_f, best_fab, best_fae, n_eval


def refine_random(individual, p_bf, f_ab_target, alpha, B, sigma=0.3, seed=0):
    """B Gaussian perturbations around init params; keep best."""
    params = extract_all_params(individual)
    if len(params) == 0:
        f, f_ab, f_ae = fitness(individual, p_bf, f_ab_target, alpha)
        return individual, f, f_ab, f_ae, 1
    rng = np.random.default_rng(seed)
    f, f_ab, f_ae = fitness(individual, p_bf, f_ab_target, alpha)
    best_params, best_f, best_fab, best_fae = params.copy(), f, f_ab, f_ae
    n_eval = 1
    while n_eval < B:
        cand = params + rng.normal(0, sigma, size=len(params))
        f, f_ab, f_ae = fitness(inject_all_params(individual, cand),
                                p_bf, f_ab_target, alpha)
        n_eval += 1
        if f > best_f:
            best_f, best_params = f, cand.copy()
            best_fab, best_fae = f_ab, f_ae
    return inject_all_params(individual, best_params), best_f, best_fab, best_fae, n_eval


def refine_fdsgd(individual, p_bf, f_ab_target, alpha, B, sigma=0.3,
                 a=0.05, c=0.05, seed=0):
    """SPSA-style finite-difference SGD: 2 evals per step → B/2 steps."""
    params = extract_all_params(individual)
    n = len(params)
    if n == 0:
        f, f_ab, f_ae = fitness(individual, p_bf, f_ab_target, alpha)
        return individual, f, f_ab, f_ae, 1
    rng = np.random.default_rng(seed)
    theta = params.copy()
    f, f_ab, f_ae = fitness(individual, p_bf, f_ab_target, alpha)
    best_f, best_params, best_fab, best_fae = f, theta.copy(), f_ab, f_ae
    n_eval = 1
    k = 0
    while n_eval + 2 <= B:
        ak = a / (k + 1 + 10) ** 0.602
        ck = c / (k + 1) ** 0.101
        delta = rng.choice([-1.0, 1.0], size=n)
        f_p, fab_p, fae_p = fitness(inject_all_params(individual, theta + ck * delta),
                                    p_bf, f_ab_target, alpha)
        f_m, fab_m, fae_m = fitness(inject_all_params(individual, theta - ck * delta),
                                    p_bf, f_ab_target, alpha)
        n_eval += 2
        # SPSA-style ascent (we maximise, so flip sign)
        grad = (f_p - f_m) / (2.0 * ck * delta)
        theta = theta + ak * grad
        if f_p > best_f:
            best_f = f_p; best_params = (theta + ck * delta).copy()
            best_fab, best_fae = fab_p, fae_p
        if f_m > best_f:
            best_f = f_m; best_params = (theta - ck * delta).copy()
            best_fab, best_fae = fab_m, fae_m
        k += 1
    return inject_all_params(individual, best_params), best_f, best_fab, best_fae, n_eval


REFINE = {"cmaes": refine_cmaes, "random": refine_random, "fdsgd": refine_fdsgd}


# -------- MAP-Elites archive (mirror qkd_experiment_4_e91) -----
def descriptor(individual):
    n_gates = len(individual)
    n_cx = sum(1 for g in individual if g[0] == CX)
    return (min(n_gates, 20), min(n_cx, 6))


# -------- Outer loop ---------------------------------------------
def run_evolutive(optimizer, seed, p_bf=P_BF, f_ab_target=F_AB_TARGET,
                  alpha=ALPHA, B=INNER_BUDGET,
                  pop_size=POP_SIZE, n_epochs=N_EPOCHS,
                  n_elite=N_ELITE, n_replace=N_REPLACE,
                  map_elites=True, verbose=False):
    np.random.seed(seed)
    refine = REFINE[optimizer]
    population = [random_individual(INIT_GATES) for _ in range(pop_size)]
    archive = {}
    history = {'best_fae': [], 'best_fab': [], 'best_f': [],
               'avg_gates': [], 'best_gates': [], 'archive_size': []}
    total_evals = 0
    global_best, global_best_f = None, -1e6

    for epoch in range(n_epochs):
        results = []
        for ind in population:
            refined, f, f_ab, f_ae, ne = refine(ind, p_bf, f_ab_target, alpha, B,
                                                 seed=seed * 1000 + epoch * pop_size + len(results))
            total_evals += ne
            results.append((refined, f, f_ab, f_ae))
            if map_elites:
                d = descriptor(refined)
                if d not in archive or archive[d][1] < f:
                    archive[d] = (refined, f, f_ab, f_ae)
        results.sort(key=lambda x: x[1], reverse=True)
        population = [r[0] for r in results]
        if results[0][1] > global_best_f:
            global_best_f = results[0][1]
            global_best = (deepcopy(results[0][0]), results[0][1], results[0][2], results[0][3])
        history['best_fae'].append(results[0][3])
        history['best_fab'].append(results[0][2])
        history['best_f'].append(results[0][1])
        history['avg_gates'].append(float(np.mean([len(r[0]) for r in results])))
        history['best_gates'].append(len(results[0][0]))
        history['archive_size'].append(len(archive))
        if verbose:
            print(f'  ep {epoch+1:2d}: F_AE={results[0][3]:.4f}  F_AB={results[0][2]:.4f}  '
                  f'gates={len(results[0][0])}  archive={len(archive)}')

        # Evolve
        new_pop = [deepcopy(population[i]) for i in range(n_elite)]
        for _ in range(n_replace):
            new_pop.append(random_individual(INIT_GATES))
        fill = pop_size - n_elite - n_replace
        if map_elites and archive:
            keys = list(archive.keys())
            for _ in range(fill):
                k = keys[np.random.randint(0, len(keys))]
                new_pop.append(mutate_individual(deepcopy(archive[k][0])))
        else:
            for i in range(fill):
                idx = min(n_elite + i, pop_size - 1)
                new_pop.append(mutate_individual(deepcopy(population[idx])))
        population = new_pop

    return {'best_individual': global_best[0], 'best_fitness': global_best[1],
            'best_F_AB': global_best[2], 'best_F_AE': global_best[3],
            'archive': archive, 'history': history, 'total_evals': total_evals,
            'optimizer': optimizer}


def run_one(optimizer, seed):
    t0 = time.time()
    out = run_evolutive(optimizer=optimizer, seed=seed)
    elapsed = time.time() - t0
    return {
        'optimizer': optimizer, 'seed': seed, 'elapsed': elapsed,
        'best_F_AE': float(out['best_F_AE']),
        'best_F_AB': float(out['best_F_AB']),
        'best_fitness': float(out['best_fitness']),
        'best_n_gates': len(out['best_individual']),
        'best_circuit': out['best_individual'],
        'archive_size': len(out['archive']),
        'history': out['history'],
        'total_evals': out['total_evals'],
    }


def main():
    out = []
    runs = [(opt, s) for opt in OPTIMIZERS for s in SEEDS]
    print(f"BB84 random vs CMA-ES vs FD-SGD comparison",
          f"\n  p_bf = {P_BF}, F_AB target = {F_AB_TARGET}, B = {INNER_BUDGET}",
          f"\n  pop = {POP_SIZE}, n_epochs = {N_EPOCHS}, n_elite={N_ELITE}, n_replace={N_REPLACE}",
          f"\n  {len(runs)} runs total\n", flush=True)
    for idx, (opt, seed) in enumerate(runs, 1):
        r = run_one(opt, seed)
        out.append(r)
        print(f"[{idx:2d}/{len(runs)}] opt={opt:7s} seed={seed} "
              f"F_AE={r['best_F_AE']:.4f}  F_AB={r['best_F_AB']:.4f}  "
              f"gates={r['best_n_gates']}  archive={r['archive_size']}  "
              f"evals={r['total_evals']:6d}  ({r['elapsed']:.0f}s)", flush=True)
        with open("bb84_random_vs_cmaes_results.pkl", "wb") as f:
            pickle.dump({'records': out, 'optimizers': OPTIMIZERS, 'seeds': SEEDS,
                         'config': dict(P_BF=P_BF, F_AB_TARGET=F_AB_TARGET, ALPHA=ALPHA,
                                        INNER_BUDGET=INNER_BUDGET,
                                        POP_SIZE=POP_SIZE, N_EPOCHS=N_EPOCHS,
                                        N_ELITE=N_ELITE, N_REPLACE=N_REPLACE,
                                        INIT_GATES=INIT_GATES)}, f)
    print(f"\nSaved bb84_random_vs_cmaes_results.pkl ({len(out)} records)", flush=True)
    plot(out)


def plot(records):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    color = {"cmaes": "#2266cc", "random": "#cc6622", "fdsgd": "#22aa66"}
    # Left: F_AE bars across optimisers
    ax = axes[0]
    means_fae, stds_fae, all_fae = [], [], []
    for opt in OPTIMIZERS:
        rs = [r['best_F_AE'] for r in records if r['optimizer'] == opt]
        means_fae.append(np.mean(rs)); stds_fae.append(np.std(rs)); all_fae.append(rs)
    x = np.arange(len(OPTIMIZERS))
    for i, opt in enumerate(OPTIMIZERS):
        ax.bar(i, means_fae[i], yerr=stds_fae[i], capsize=4, color=color[opt],
               alpha=0.85, width=0.6, label=opt)
        for v in all_fae[i]:
            ax.plot(i, v, 'k.', markersize=4, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(OPTIMIZERS)
    ax.set_ylabel(r'Final $F_{AE}$ (best individual)')
    ax.set_title(rf'BB84,  $p_\mathrm{{bf}} = {P_BF}$,  $F_{{AB}}^\mathrm{{target}} = {F_AB_TARGET}$')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, label='Identity (0.5)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=8)

    # Right: F_AE vs F_AB scatter
    ax = axes[1]
    for opt in OPTIMIZERS:
        for r in records:
            if r['optimizer'] != opt: continue
            ax.scatter(r['best_F_AB'], r['best_F_AE'], color=color[opt],
                       s=60, alpha=0.7, edgecolor='black')
    ax.axvline(F_AB_TARGET, color='gray', linestyle='--', alpha=0.4,
               label=f'F_AB target = {F_AB_TARGET}')
    ax.axvline(0.87, color='red', linestyle=':', alpha=0.4, label='F_AB threshold = 0.87')
    ax.set_xlabel('Final $F_{AB}$'); ax.set_ylabel('Final $F_{AE}$')
    ax.set_title('Best-individual ($F_{AB}$, $F_{AE}$) per (opt, seed)')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle('NeurIPS Falsification Experiment #3: '
                 'random search vs CMA-ES on BB84')
    plt.tight_layout()
    plt.savefig("fig_bb84_random_vs_cmaes.pdf", bbox_inches='tight')
    plt.savefig("fig_bb84_random_vs_cmaes.png", dpi=140, bbox_inches='tight')
    print("Saved fig_bb84_random_vs_cmaes.pdf and .png")


if __name__ == "__main__":
    main()
