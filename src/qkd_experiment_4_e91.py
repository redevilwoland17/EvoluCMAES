"""
QKD_experiment_4 for E91.

Optimizer-agnostic evolutive-training framework + MAP-Elites + head-to-head
SPSA / CMA-ES / Random comparison, ported from BB84 exp4 to E91.

Produces `e91_exp4_library.pkl` — a library keyed by (s_target, safety_level)
with circuits, metadata, and MAP-Elites archive. Format compatible with
rl_e91_library_noise.ipynb if you swap LIB_PATH.

Run: python qkd_experiment_4_e91.py
"""
import os, time, pickle, json
from copy import deepcopy
import numpy as np
import torch
import cma

import evoluspsa_e91 as E91
from evoluspsa_e91 import (
    random_individual, mutate_individual, extract_all_params, inject_all_params,
    circuit_to_unitary, compute_holevo_and_S, acin_holevo_bound,
    POP_SIZE, INIT_GATES, ALPHA, BETA, STD_SETTINGS,
)


# ──────────────────────────────────────────────────────────────────────
# Fitness (reuse E91's)
# ──────────────────────────────────────────────────────────────────────
def fitness(individual, s_target, alpha=ALPHA, beta=BETA):
    circuit, settings = individual
    try:
        U = circuit_to_unitary(circuit)
        S, chi = compute_holevo_and_S(U, settings)
        return chi - alpha * (S - s_target) ** 2 - beta * len(circuit), S, chi
    except Exception:
        return -1e6, 0.0, 0.0


# ──────────────────────────────────────────────────────────────────────
# Optimizer-agnostic per-topology refinement
# ──────────────────────────────────────────────────────────────────────
def refine_spsa(individual, s_target, n_iters=60, a=0.15, c_val=0.1):
    """2 evaluations per iteration."""
    params = extract_all_params(individual, variable_settings=False)
    if len(params) == 0:
        f, S, chi = fitness(individual, s_target)
        return individual, f, S, chi, 1
    best_params, best_f, best_S, best_chi = params.copy(), -1e6, 0.0, 0.0
    n_eval = 0
    for k in range(n_iters):
        ak = a / (k + 1 + 10) ** 0.602
        ck = c_val / (k + 1) ** 0.101
        delta = np.random.choice([-1, 1], size=len(params)).astype(np.float64)
        f_p, _, _ = fitness(inject_all_params(individual, params + ck * delta, False), s_target)
        f_m, _, _ = fitness(inject_all_params(individual, params - ck * delta, False), s_target)
        n_eval += 2
        grad = (f_p - f_m) / (2 * ck * delta)
        params = params + ak * grad
        f, S, chi = fitness(inject_all_params(individual, params, False), s_target)
        n_eval += 1
        if f > best_f:
            best_f, best_params, best_S, best_chi = f, params.copy(), S, chi
    return inject_all_params(individual, best_params, False), best_f, best_S, best_chi, n_eval


def refine_cmaes(individual, s_target, n_iters=12, popsize=12, sigma=0.3):
    """CMA-ES: popsize evaluations per generation."""
    params = extract_all_params(individual, variable_settings=False)
    if len(params) == 0:
        f, S, chi = fitness(individual, s_target)
        return individual, f, S, chi, 1
    es = cma.CMAEvolutionStrategy(
        params.tolist(), sigma,
        {'popsize': popsize, 'maxiter': n_iters, 'verbose': -9,
         'tolx': 1e-4, 'tolfun': 1e-4, 'tolstagnation': 5},
    )
    best_f, best_params, best_S, best_chi = -1e6, params.copy(), 0.0, 0.0
    n_eval = 0
    while not es.stop():
        sols = es.ask()
        vals = []
        for s in sols:
            f, S, chi = fitness(inject_all_params(individual, np.asarray(s), False), s_target)
            vals.append(-f)
            n_eval += 1
            if f > best_f:
                best_f, best_params, best_S, best_chi = f, np.asarray(s).copy(), S, chi
        es.tell(sols, vals)
    return inject_all_params(individual, best_params, False), best_f, best_S, best_chi, n_eval


def refine_random(individual, s_target, n_iters=150, sigma=0.3):
    """Uniform random search over parameter space around the current point."""
    params = extract_all_params(individual, variable_settings=False)
    best_params = params.copy()
    best_f, best_S, best_chi = fitness(individual, s_target)
    n_eval = 1
    for _ in range(n_iters):
        pert = np.random.randn(len(params)) * sigma
        f, S, chi = fitness(inject_all_params(individual, params + pert, False), s_target)
        n_eval += 1
        if f > best_f:
            best_f, best_params, best_S, best_chi = f, (params + pert).copy(), S, chi
    return inject_all_params(individual, best_params, False), best_f, best_S, best_chi, n_eval


OPTIMIZERS = {
    'spsa':   refine_spsa,
    'cmaes':  refine_cmaes,
    'random': refine_random,
}


# ──────────────────────────────────────────────────────────────────────
# MAP-Elites descriptor
# ──────────────────────────────────────────────────────────────────────
def descriptor(individual):
    """Bin circuits by (n_gates, n_cx) — two cheap structural features."""
    circuit, _ = individual
    n_gates = len(circuit)
    n_cx = sum(1 for g in circuit if g[0] == 'cx')
    return (min(n_gates, 20), min(n_cx, 6))


# ──────────────────────────────────────────────────────────────────────
# Main evolutive loop with pluggable optimizer + MAP-Elites archive
# ──────────────────────────────────────────────────────────────────────
def run_evolutive(
    s_target, optimizer='spsa', seed=0,
    pop_size=30, n_epochs=15, n_elite=6, n_replace=10,
    verbose=False, map_elites=True,
):
    np.random.seed(seed)
    refine = OPTIMIZERS[optimizer]

    population = [random_individual(INIT_GATES, False) for _ in range(pop_size)]
    archive = {}  # descriptor -> (individual, fitness, S, chi)
    history = {'best_chi': [], 'best_S': [], 'best_fitness': [], 'archive_size': []}
    total_evals = 0
    global_best = None
    global_best_f = -1e6

    for epoch in range(n_epochs):
        results = []
        for ind in population:
            refined_ind, f, S, chi, ne = refine(ind, s_target)
            total_evals += ne
            results.append((refined_ind, f, S, chi))
            if map_elites:
                d = descriptor(refined_ind)
                if d not in archive or archive[d][1] < f:
                    archive[d] = (refined_ind, f, S, chi)

        results.sort(key=lambda x: x[1], reverse=True)
        population = [r[0] for r in results]
        if results[0][1] > global_best_f:
            global_best_f = results[0][1]
            global_best = (deepcopy(results[0][0]), results[0][1], results[0][2], results[0][3])

        history['best_chi'].append(results[0][3])
        history['best_S'].append(results[0][2])
        history['best_fitness'].append(results[0][1])
        history['archive_size'].append(len(archive))

        if verbose:
            print(f'  ep {epoch + 1:2d}: χ={results[0][3]:.4f}  S={results[0][2]:.4f}  '
                  f'archive={len(archive)}  evals={total_evals}')

        # Evolve
        new_pop = [deepcopy(population[i]) for i in range(n_elite)]
        for _ in range(n_replace):
            new_pop.append(random_individual(INIT_GATES, False))
        # Seed remaining slots from archive (if MAP-Elites) or from top
        fill = pop_size - n_elite - n_replace
        if map_elites and archive:
            keys = list(archive.keys())
            for _ in range(fill):
                k = keys[np.random.randint(0, len(keys))]
                parent = deepcopy(archive[k][0])
                new_pop.append(mutate_individual(parent, False))
        else:
            for i in range(fill):
                idx = min(n_elite + i, pop_size - 1)
                new_pop.append(mutate_individual(deepcopy(population[idx]), False))
        population = new_pop

    return {
        'best_individual': global_best[0],
        'best_fitness':    global_best[1],
        'best_S':          global_best[2],
        'best_chi':        global_best[3],
        'archive':         archive,
        'history':         history,
        'total_evals':     total_evals,
        'optimizer':       optimizer,
    }


# ──────────────────────────────────────────────────────────────────────
# Head-to-head comparison at a single target
# ──────────────────────────────────────────────────────────────────────
def head_to_head(s_target, seeds=(0, 1, 2)):
    print(f'\n=== Head-to-head at s_target = {s_target:.3f} '
          f'(Acín bound χ = {acin_holevo_bound(s_target):.4f}) ===')
    out = {}
    for opt in ['spsa', 'cmaes', 'random']:
        print(f'\n  [{opt}]')
        runs = []
        for s in seeds:
            t0 = time.time()
            r = run_evolutive(s_target, optimizer=opt, seed=s, n_epochs=12, pop_size=20,
                              n_elite=4, n_replace=6, verbose=False)
            runs.append(r)
            print(f'    seed={s}: χ={r["best_chi"]:.4f}  S={r["best_S"]:.4f}  '
                  f'gates={len(r["best_individual"][0])}  '
                  f'archive={len(r["archive"])}  evals={r["total_evals"]}  '
                  f'time={time.time()-t0:.1f}s')
        chis = [r['best_chi'] for r in runs]
        Ss   = [r['best_S']   for r in runs]
        evs  = [r['total_evals'] for r in runs]
        print(f'    mean χ = {np.mean(chis):.4f} ± {np.std(chis):.4f} '
              f'|  mean S = {np.mean(Ss):.4f}  |  mean evals = {np.mean(evs):.0f}')
        out[opt] = runs
    return out


# ──────────────────────────────────────────────────────────────────────
# Full sweep to produce a library pkl (compatible with RL notebook format)
# ──────────────────────────────────────────────────────────────────────
def build_library(s_targets, levels, optimizer='spsa', seeds_per_cell=2):
    lib = {}
    n_total = len(s_targets) * len(levels) * seeds_per_cell
    done = 0
    for s in s_targets:
        for lvl in levels:
            # Level defines an s_target offset (mirror of rl_e91_library.ipynb convention)
            effective = float(np.clip(s + lvl, 2.44, 2.82))
            runs = []
            for seed in range(seeds_per_cell):
                done += 1
                print(f'  [{done}/{n_total}] anchor={s:.3f}  lvl={lvl:+.2f}  '
                      f'eff_target={effective:.3f}  seed={seed}  opt={optimizer}')
                r = run_evolutive(effective, optimizer=optimizer, seed=seed * 100,
                                  n_epochs=10, pop_size=20, n_elite=4, n_replace=6)
                runs.append(r)
                print(f'     → χ={r["best_chi"]:.4f}  S={r["best_S"]:.4f}  '
                      f'archive={len(r["archive"])}')
            best = max(runs, key=lambda r: r['best_fitness'])
            lib[(float(s), float(lvl))] = {
                'circuit':  best['best_individual'][0],
                'settings': STD_SETTINGS.tolist(),
                'S':        best['best_S'],
                'chi':      best['best_chi'],
                'archive':  [(k, v[1], v[2], v[3]) for k, v in best['archive'].items()],
                'eff_target': effective,
            }
    return lib


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    t_start = time.time()

    print('=' * 68)
    print('Part 1: head-to-head SPSA vs CMA-ES vs Random at one target')
    print('=' * 68)
    h2h = head_to_head(s_target=2.55, seeds=(0, 1, 2))

    print('\n' + '=' * 68)
    print('Part 2: small sweep with the winning optimizer')
    print('=' * 68)
    # Pick the winner by mean best χ
    mean_chi = {opt: float(np.mean([r['best_chi'] for r in runs])) for opt, runs in h2h.items()}
    winner = max(mean_chi, key=mean_chi.get)
    print(f'\nMean best χ: {mean_chi}')
    print(f'Winner: {winner}')

    s_targets = [2.50, 2.60, 2.70]
    levels = [+0.05, 0.0, -0.05]
    print(f'\nBuilding library: {len(s_targets)} targets × {len(levels)} levels × 2 seeds '
          f'using {winner}...')
    lib = build_library(s_targets, levels, optimizer=winner, seeds_per_cell=2)

    with open('e91_exp4_library.pkl', 'wb') as f:
        pickle.dump({'lib': lib, 'head_to_head_meta': {opt: mean_chi[opt] for opt in mean_chi}}, f)
    print(f'\nSaved {len(lib)} library entries to e91_exp4_library.pkl')

    print(f'\n{"=" * 68}')
    print(f'TOTAL time: {(time.time() - t_start) / 60:.1f} min')
    print('=' * 68)
