"""
EvoluSPSA for E91 Source Control — Pure PyTorch Implementation
==============================================================

Hybrid evolutionary-SPSA optimisation of eavesdropping circuits on E91.
Eve prepares |ψ_ABE⟩ and distributes qubits to Alice, Bob; keeps E.

Objective: max χ(A:E) subject to S ≈ s_target
Fitness:   χ(A:E) - α*(S - s_target)² - β*n_gates

Two modes:
  - Fixed settings:    standard CHSH angles, Eve optimises circuit only
  - Variable settings: Eve co-optimises circuit + 4 CHSH measurement angles

Circuit representation: list of (gate_type, qubits, param) tuples
Gate set: RX, RY, RZ (1-qubit), H (1-qubit), CX (2-qubit)

Run: python evoluspsa_e91.py
"""

import torch
import numpy as np
import json
import time
from copy import deepcopy

torch.set_default_dtype(torch.float64)
C = torch.complex128

# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════
POP_SIZE = 60
N_EPOCHS = 50
SPSA_ITERS = 80
SPSA_A = 0.15
SPSA_C = 0.1
ALPHA = 100.0
BETA = 0.0
N_ELITE = 8
N_REPLACE = 12
INIT_GATES = 6
N_QUBITS = 3
VARIABLE_SETTINGS = False

RX, RY, RZ, H, CX = 'rx', 'ry', 'rz', 'h', 'cx'
PARAM_GATES = {RX, RY, RZ}
ALL_1Q_GATES = [RX, RY, RZ, H]
STD_SETTINGS = np.array([0.0, np.pi/2, np.pi/4, -np.pi/4])

# ══════════════════════════════════════════════════════════════
# Gate matrices
# ══════════════════════════════════════════════════════════════
I2 = torch.eye(2, dtype=C)
Z_mat = torch.tensor([[1,0],[0,-1]], dtype=C)
X_mat = torch.tensor([[0,1],[1,0]], dtype=C)
H_mat = torch.tensor([[1,1],[1,-1]], dtype=C) / np.sqrt(2)

def rx_mat(t):
    c, s = torch.cos(t/2), torch.sin(t/2)
    return torch.stack([torch.stack([c+0j, -1j*s]), torch.stack([-1j*s, c+0j])])
def ry_mat(t):
    c, s = torch.cos(t/2), torch.sin(t/2)
    return torch.stack([torch.stack([c+0j, -s+0j]), torch.stack([s+0j, c+0j])])
def rz_mat(t):
    return torch.diag(torch.stack([torch.exp(-1j*t/2), torch.exp(1j*t/2)]))
def gate_matrix(gate_type, param=0.0):
    if gate_type == RX: return rx_mat(torch.tensor(param))
    if gate_type == RY: return ry_mat(torch.tensor(param))
    if gate_type == RZ: return rz_mat(torch.tensor(param))
    if gate_type == H:  return H_mat
    raise ValueError(f"Unknown gate: {gate_type}")

# ══════════════════════════════════════════════════════════════
# Circuit representation
# ══════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════
# Individual = (circuit, settings)
# ══════════════════════════════════════════════════════════════
def random_individual(n_gates=INIT_GATES, variable_settings=False):
    circuit = random_circuit(n_gates)
    settings = STD_SETTINGS + np.random.normal(0, 0.2, 4) if variable_settings else STD_SETTINGS.copy()
    return (circuit, settings)

def mutate_individual(individual, variable_settings=False):
    circuit, settings = individual
    return (mutate(circuit), settings.copy())

def extract_all_params(individual, variable_settings=False):
    circuit, settings = individual
    gate_params = np.array([g[2] for g in circuit if g[2] is not None], dtype=np.float64)
    return np.concatenate([gate_params, settings]) if variable_settings else gate_params

def inject_all_params(individual, params, variable_settings=False):
    circuit, settings = individual
    if variable_settings:
        gate_params, new_settings = params[:-4], params[-4:].copy()
    else:
        gate_params, new_settings = params, settings.copy()
    new_circuit = deepcopy(circuit)
    idx = 0
    for i, (gt, qubits, param) in enumerate(new_circuit):
        if param is not None:
            new_circuit[i] = (gt, qubits, float(gate_params[idx]))
            idx += 1
    return (new_circuit, new_settings)

# ══════════════════════════════════════════════════════════════
# Circuit to unitary
# ══════════════════════════════════════════════════════════════
def embed_1q(mat, qubit):
    ops = [I2]*N_QUBITS
    ops[qubit] = mat
    result = ops[0]
    for op in ops[1:]:
        result = torch.kron(result, op)
    return result

def embed_cx(control, target):
    dim = 2**N_QUBITS
    U = torch.zeros(dim, dim, dtype=C)
    for state in range(dim):
        bits = [(state >> (N_QUBITS-1-q)) & 1 for q in range(N_QUBITS)]
        new_bits = bits.copy()
        new_bits[target] ^= bits[control]
        new_state = sum(new_bits[q] << (N_QUBITS-1-q) for q in range(N_QUBITS))
        U[new_state, state] = 1.0
    return U

CX_CACHE = {(c,t): embed_cx(c,t) for c in range(N_QUBITS) for t in range(N_QUBITS) if c != t}

def circuit_to_unitary(circuit):
    U = torch.eye(2**N_QUBITS, dtype=C)
    for (gt, qubits, param) in circuit:
        if gt == CX:     G = CX_CACHE[(qubits[0], qubits[1])]
        elif gt == H:    G = embed_1q(H_mat, qubits[0])
        elif gt in PARAM_GATES: G = embed_1q(gate_matrix(gt, param), qubits[0])
        else: raise ValueError(f"Unknown gate: {gt}")
        U = G @ U
    return U

# ══════════════════════════════════════════════════════════════
# E91 Holevo and CHSH
# ══════════════════════════════════════════════════════════════
def obs(phi):
    return torch.cos(phi)*Z_mat + torch.sin(phi)*X_mat

def von_neumann_entropy(rho):
    eigs = torch.linalg.eigvalsh(rho).real.clamp(min=1e-12)
    return -torch.sum(eigs * torch.log2(eigs))

def compute_holevo_and_S(U, settings):
    a1, a2 = torch.tensor(settings[0]), torch.tensor(settings[1])
    b1, b2 = torch.tensor(settings[2]), torch.tensor(settings[3])

    psi = U @ torch.zeros(8, dtype=C).index_fill(0, torch.tensor([0]), 1.0)
    rho = torch.outer(psi, psi.conj())

    rho_AB = torch.zeros(4, 4, dtype=C)
    for a in range(2):
        for b in range(2):
            for ap in range(2):
                for bp in range(2):
                    rho_AB[a*2+b, ap*2+bp] = rho[a*4+b*2+0, ap*4+bp*2+0] + rho[a*4+b*2+1, ap*4+bp*2+1]

    def E_corr(ai, bj):
        return torch.trace(rho_AB @ torch.kron(obs(ai), obs(bj))).real
    S = E_corr(a1,b1) + E_corr(a1,b2) + E_corr(a2,b1) - E_corr(a2,b2)

    rho_E = torch.zeros(2, 2, dtype=C)
    for e in range(2):
        for ep in range(2):
            for a in range(2):
                for b in range(2):
                    rho_E[e, ep] += rho[a*4+b*2+e, a*4+b*2+ep]

    rho_E_cond, probs = [], []
    for x in range(2):
        rho_ex = torch.zeros(2, 2, dtype=C)
        for e in range(2):
            for ep in range(2):
                for b in range(2):
                    rho_ex[e, ep] += rho[x*4+b*2+e, x*4+b*2+ep]
        px = torch.trace(rho_ex).real.clamp(min=1e-12)
        probs.append(px)
        rho_E_cond.append(rho_ex / px)

    chi = von_neumann_entropy(rho_E)
    for x in range(2):
        chi = chi - probs[x] * von_neumann_entropy(rho_E_cond[x])
    return S.item(), chi.item()

def fitness(individual, s_target, alpha=ALPHA, beta=BETA):
    circuit, settings = individual
    try:
        U = circuit_to_unitary(circuit)
        S, chi = compute_holevo_and_S(U, settings)
        return chi - alpha*(S - s_target)**2 - beta*len(circuit), S, chi
    except Exception:
        return -1e6, 0.0, 0.0

# ══════════════════════════════════════════════════════════════
# SPSA
# ══════════════════════════════════════════════════════════════
def spsa_optimise(individual, s_target, variable_settings=False,
                  n_iters=SPSA_ITERS, a=SPSA_A, c_val=SPSA_C):
    params = extract_all_params(individual, variable_settings)
    n_params = len(params)
    if n_params == 0:
        f, S, chi = fitness(individual, s_target)
        return individual, f, S, chi, 1

    # Per-parameter scaling: settings get smaller perturbations than gate params
    if variable_settings and n_params > 4:
        n_gate = n_params - 4
        # Gate params: scale 1.0, settings: scale 0.3 (smaller steps)
        scale = np.concatenate([np.ones(n_gate), 0.3 * np.ones(4)])
    else:
        scale = np.ones(n_params)

    best_params, best_f, best_S, best_chi = params.copy(), -1e6, 0.0, 0.0
    n_evals = 0

    for k in range(n_iters):
        ak = a / (k+1+10)**0.602
        ck = c_val / (k+1)**0.101
        delta = np.random.choice([-1,1], size=n_params).astype(np.float64)
        scaled_delta = delta * scale

        f_plus, _, _ = fitness(inject_all_params(individual, params + ck*scaled_delta, variable_settings), s_target)
        f_minus, _, _ = fitness(inject_all_params(individual, params - ck*scaled_delta, variable_settings), s_target)
        n_evals += 2

        grad = (f_plus - f_minus) / (2*ck*scaled_delta)
        params = params + ak * grad * scale

        # Clip settings to reasonable range (±π from standard)
        if variable_settings and n_params > 4:
            params[-4:] = np.clip(params[-4:], STD_SETTINGS - np.pi, STD_SETTINGS + np.pi)

        f_cur, S_cur, chi_cur = fitness(inject_all_params(individual, params, variable_settings), s_target)
        n_evals += 1
        if f_cur > best_f:
            best_f, best_params, best_S, best_chi = f_cur, params.copy(), S_cur, chi_cur

    return inject_all_params(individual, best_params, variable_settings), best_f, best_S, best_chi, n_evals

# ══════════════════════════════════════════════════════════════
# Acin bound
# ══════════════════════════════════════════════════════════════
def h_bin(x):
    if x <= 0 or x >= 1: return 0.0
    return -x*np.log2(x) - (1-x)*np.log2(1-x)

def acin_holevo_bound(S):
    s2 = (S/2)**2
    return 1.0 if s2 <= 1 else h_bin(0.5 + 0.5*np.sqrt(s2-1))

def secure_key_rate(S):
    s2 = (S/2)**2
    if s2 <= 1: return 0.0
    return max(0.0, 1 - h_bin(0.5+0.5*np.sqrt(s2-1)) - h_bin((1-S/(2*np.sqrt(2)))/2))

# ══════════════════════════════════════════════════════════════
# Main evolutionary loop
# ══════════════════════════════════════════════════════════════
def run_evoluspsa(s_target, seed=42, pop_size=POP_SIZE, n_epochs=N_EPOCHS,
                  spsa_iters=SPSA_ITERS, alpha=ALPHA, beta=BETA,
                  n_elite=N_ELITE, n_replace=N_REPLACE,
                  variable_settings=VARIABLE_SETTINGS, verbose=True):
    np.random.seed(seed)
    population = [random_individual(INIT_GATES, variable_settings) for _ in range(pop_size)]
    history = {"best_chi":[], "best_S":[], "best_fitness":[],
               "median_chi":[], "median_S":[], "avg_gates":[], "best_gates":[]}
    total_evals = 0
    global_best_f, global_best_individual = -1e6, None
    global_best_S, global_best_chi = 0.0, 0.0

    for epoch in range(n_epochs):
        results = []
        for ind in population:
            best_ind, f, S, chi, n_evals = spsa_optimise(ind, s_target, variable_settings, spsa_iters)
            results.append((best_ind, f, S, chi, len(best_ind[0]), n_evals))
            total_evals += n_evals

        results.sort(key=lambda x: x[1], reverse=True)
        population = [r[0] for r in results]
        fitnesses = [r[1] for r in results]
        S_vals = [r[2] for r in results]
        chi_vals = [r[3] for r in results]
        gate_counts = [r[4] for r in results]

        if results[0][1] > global_best_f:
            global_best_f = results[0][1]
            global_best_individual = deepcopy(results[0][0])
            global_best_S, global_best_chi = results[0][2], results[0][3]

        mid = pop_size // 2
        history["best_chi"].append(chi_vals[0])
        history["best_S"].append(S_vals[0])
        history["best_fitness"].append(fitnesses[0])
        history["median_chi"].append(chi_vals[mid])
        history["median_S"].append(S_vals[mid])
        history["avg_gates"].append(float(np.mean(gate_counts)))
        history["best_gates"].append(gate_counts[0])

        if verbose and (epoch+1) % 5 == 0:
            acin = acin_holevo_bound(S_vals[0])
            s_str = ""
            if variable_settings:
                s = population[0][1]
                s_str = f"  set=[{s[0]:.2f},{s[1]:.2f},{s[2]:.2f},{s[3]:.2f}]"
            print(f"  Ep {epoch+1:3d}: χ={chi_vals[0]:.4f} S={S_vals[0]:.4f} "
                  f"gap={acin-chi_vals[0]:.5f} g={gate_counts[0]} "
                  f"avg_g={np.mean(gate_counts):.1f}{s_str}")

        # Evolutionary step
        new_pop = [deepcopy(population[i]) for i in range(n_elite)]
        for _ in range(n_replace):
            new_pop.append(random_individual(INIT_GATES, variable_settings))
        for i in range(pop_size - n_elite - n_replace):
            if np.random.random() < 0.5:
                parent = deepcopy(population[np.random.randint(0, n_elite)])
            else:
                idx = min(n_elite + i, pop_size - 1)
                parent = deepcopy(population[idx])
            new_pop.append(mutate_individual(parent, variable_settings))
        population = new_pop

    best_circuit, best_settings = global_best_individual
    acin = acin_holevo_bound(global_best_S)
    return {
        "best_S": global_best_S, "best_chi": global_best_chi, "best_fitness": global_best_f,
        "best_circuit": [(g[0], g[1], g[2]) for g in best_circuit],
        "best_settings": best_settings.tolist(),
        "best_n_gates": len(best_circuit),
        "acin_bound": acin, "gap": acin - global_best_chi,
        "total_evals": total_evals, "variable_settings": variable_settings,
        "history": history,
    }

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from scipy.optimize import brentq
    S_threshold = brentq(lambda s: secure_key_rate(s) - 1e-6, 2.01, 2.8)
    print(f"Secure key threshold: S ≈ {S_threshold:.4f}\n")

    N_TARGETS = 6
    N_SEEDS = 3
    RUN_FIXED = True
    RUN_VARIABLE = True
    s_targets = np.linspace(S_threshold + 0.02, 2*np.sqrt(2) - 0.02, N_TARGETS)

    all_results = {"fixed": {}, "variable": {}}
    t_start = time.time()

    for i, s_target in enumerate(s_targets):
        acin = acin_holevo_bound(s_target)
        print(f"\n{'='*65}")
        print(f"Target {i+1}/{N_TARGETS}: S = {s_target:.4f}  (Acín = {acin:.5f})")
        print(f"{'='*65}")

        for mode in ["fixed", "variable"]:
            if mode == "fixed" and not RUN_FIXED: continue
            if mode == "variable" and not RUN_VARIABLE: continue
            var_set = (mode == "variable")
            print(f"\n  --- {mode} settings ---")

            seed_results = []
            for seed in range(N_SEEDS):
                print(f"\n  Seed {seed}:")
                result = run_evoluspsa(s_target, seed=seed*1000+i*100+42,
                                       variable_settings=var_set)
                seed_results.append(result)
                s_str = ""
                if var_set:
                    s = result["best_settings"]
                    s_str = f"  set=[{s[0]:.2f},{s[1]:.2f},{s[2]:.2f},{s[3]:.2f}]"
                print(f"  → χ={result['best_chi']:.4f} S={result['best_S']:.4f} "
                      f"gap={result['gap']:.5f} g={result['best_n_gates']} "
                      f"ev={result['total_evals']}{s_str}")

            all_results[mode][f"target_{i}"] = {
                "s_target": float(s_target),
                "acin_bound": acin,
                "seeds": [{k:v for k,v in r.items() if k != "history"} for r in seed_results],
                "chi_mean": float(np.mean([r["best_chi"] for r in seed_results])),
                "chi_std": float(np.std([r["best_chi"] for r in seed_results])),
                "gap_mean": float(np.mean([r["gap"] for r in seed_results])),
                "gates_mean": float(np.mean([r["best_n_gates"] for r in seed_results])),
                "evals_mean": float(np.mean([r["total_evals"] for r in seed_results])),
            }

    elapsed = time.time() - t_start
    print(f"\n{'='*75}\nSUMMARY ({elapsed/60:.1f} min)\n{'='*75}")
    for mode in ["fixed", "variable"]:
        if not all_results[mode]: continue
        print(f"\n  {mode.upper()} SETTINGS:")
        print(f"  {'S':>7} {'χ':>14} {'Gap':>8} {'Gates':>6} {'Evals':>8} {'Acín':>7}")
        print(f"  {'-'*58}")
        for key in sorted(all_results[mode].keys()):
            r = all_results[mode][key]
            print(f"  {r['s_target']:.4f} {r['chi_mean']:.4f}±{r['chi_std']:.4f} "
                  f"{r['gap_mean']:.5f} {r['gates_mean']:.1f} {r['evals_mean']:.0f} {r['acin_bound']:.4f}")

    with open("evoluspsa_e91_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: evoluspsa_e91_results.json")