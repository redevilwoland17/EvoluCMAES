# EvoluCMAES: code and data release

Code, notebooks, stored results, libraries and figures for

> Marcel Mordarski, Benjamin Gras, Abdelrahman Shehata, Daniel Budina and Roberto Bondesan.
> **Sparsity for Free: A Budget-Induced Equilibrium in Joint Topology–Parameter Search.**
> Advances in Neural Information Processing Systems (NeurIPS), 2026.

Table, figure and appendix numbers below refer to the camera-ready version of the paper.

## Quick start

```bash
pip install -r requirements.txt
python verify.py                      # recomputes the reported values (NumPy and SciPy only)
python src/e91_paired_stats.py        # per-episode E91 statistics (Appendix I, Table 8)
```

`verify.py` loads the stored results in `results/` and `libraries/`, recomputes
every value it can check and prints the value in the paper next to it: Tables
1–8 and 11–14, the detection thresholds and analytical envelopes, the BB84
library ratios of Section 4.3, the paired BB84 and E91 statistics, the values
quoted in Section 4.1 and Appendices B–G, and the compute totals of Appendix L
(250 values). It also recomputes every row of the 13 CSV tables in `results/`
and `results/mechanism/` from the result pickles. Its exit status is the number
of mismatches. The pruning pickles contain PyTorch modules, which `verify.py`
replaces by stubs, so PyTorch is not needed for verification.

## Directory layout

```
src/                Python sources: framework, QKD physics and environments, RL trainers,
                    bound computations, and the scripts of the controlled mechanism tests
notebooks/          Jupyter notebooks that produced the tables (executed, with outputs)
results/            Stored evaluation results (pickles; CSV summaries of the runs added
                    for the camera-ready version; compute_hours.csv)
results/mechanism/  Results of the controlled mechanism tests (Section 4.1, Appendices B–D)
libraries/          EvoluCMAES libraries (RL action sets) and the BB84 dynamic-programming solution
figures/            The figures of the paper (PDF; PNG where the paper embeds a PNG)
figure_scripts/     Scripts that redraw the figures from results/ and libraries/
verify.py           Recomputes the reported values from the stored results
requirements.txt    Python dependencies
LICENSE             MIT
```

Figure scripts can be run from any directory, for example
`python figure_scripts/make_rl_method_figures.py`; they find the data in
`results/` and `libraries/` and write the figure to the current directory.

## Paper item → source

### Main text

| Paper item | Source |
|---|---|
| Figure 1 (paint-coverage schematic) | TikZ in the paper source; contains no data |
| Figure 2 and Algorithm 1 (two-loop framework) | TikZ and `algorithmic` in the paper source; implementation in `src/qkd_experiment_4_e91.py` (`run_evolutive`), `src/evoluspsa_e91.py` (E91, SPSA inner loop), `src/budget_sweep_mnist.py` (MNIST and teacher–student) and the pruning notebooks |
| Section 4.1, Figure 3 (teacher–student benchmark) | `src/teacher_student.py` → `results/mechanism/expE_results_{e10,e10x,e30,e30x}-s{0..4}.pkl`; variance-collapse budgets from `src/variance_grid.py` → `results/mechanism/expI_results_i.pkl`; `figure_scripts/make_mechanism_figures.py` |
| Section 4.1, inner-optimiser values and Adam at B = 256 | `src/inner_optimiser_ablation.py` → `results/mechanism/expB_inner_ablation_mnist_results.pkl`, `expB_adam_B256_results.pkl` |
| Section 4.1, selection-off nulls at ρ0 = 0.10 and 0.30 | `src/selection_off_null.py` → `results/mechanism/expF_selection_null_results.pkl`, `expF_selection_null_results_rho030.pkl` |
| Table 1 (UCI pruning at ρ = 0.05) | `notebooks/NNpruning_experiment_2.ipynb` → `results/nnpruning2_results.pkl` (the EvoluCMAES row is stored under the key `EvoluPrune`) |
| Table 2, biased rows | `notebooks/NNpruning_experiment_3.ipynb` → `results/nnpruning3_results.pkl` (group `alpha_natural`) |
| Table 2, unbiased (drift-neutral) rows | `src/mutation_operator_ablation_uci.py` → `results/nnpruning4_drift_neutral_results.pkl`; `results/table2_drift_neutral_rows.csv` |
| Table 2 caption, density-agnostic values | `notebooks/NNpruning_experiment_4.ipynb` → `results/nnpruning4_results.pkl` (group `alpha_natural_qmut`) |
| Section 4.2, per-method wall-clock and the further tabular datasets | `src/tabular_benchmarks.py` → `results/tabular_table1_results.pkl`, `results/tabular_validation_results.pkl` |
| Section 4.3, BB84 library (24 cells, 4–6 gates; ratios to the cloning value) | cells built from the `evolu_grid_*_bitflip.pkl` grids (not yet in this release); per-cell fidelities in the stored output of `notebooks/QKD_experiment_5.ipynb` (cell 9); physics in `src/bb84_lib.py` |
| Section 4.3, E91 library (26 cells, 5–9 gates) | `src/pathB_lib7levels_build.py` with `src/evoluspsa_e91.py` (driver: `notebooks/evoluspsa_e91.ipynb`) → `libraries/e91_library_pathB_7levels.pkl` |
| Sections 3.3, 4.2 and 4.3, selection-off nulls of the UCI and QKD searches | `src/selection_off_null_uci_qkd.py` → `results/mechanism/selection_null_uci_qkd_results.pkl` |
| Detection threshold S_sec ≈ 2.4240, threshold 2.444, p_cliff ≈ 0.0742 | `secure_key_rate()` in `src/evoluspsa_e91.py`; recomputed in `verify.py` |
| Table 3, BB84 strict agent, Oracle, Identity (first set of 50 episodes) | `notebooks/QKD_experiment_7.ipynb` → `results/exp7_results_rainbow_v3.pkl` |
| Table 3, BB84 tolerated agent and noise-indexed lookup (second set of 50 episodes; Oracle 0.7429) | `notebooks/QKD_experiment_7.ipynb`, lower-κ run → `results/exp7_results_rainbow_v2.pkl` |
| Table 3, BB84 DP reference 0.7527 (fine grid, 200 episodes, no detection) | `libraries/bb84_strat_dp_hires.pkl` (`summary`) |
| Table 3, E91 rows (200 episodes) | `notebooks/QKD_experiment_7_E91.ipynb` → `results/e91_exp7_results.pkl` |
| Table 3 caption, envelopes F_AE ≤ 0.836 and χ ≤ 0.607 | `src/episode_mean_bounds.py` → `results/episode_mean_bounds.pkl` |
| Section 4.4, paired BB84 statistics (+0.0047, ±0.0011, 46 of 50, sign test) | recomputed in `verify.py` from the two `exp7_results_rainbow_v*.pkl` files |

### Appendices

| Paper item | Source |
|---|---|
| Appendix A (proof sketch) | no code |
| Appendix B, Figure 4 (teacher–student protocol and fit quality; 2,700 cold-start inner runs) | `src/teacher_student.py`, `src/variance_grid.py` → `results/mechanism/expE_results_*.pkl`, `expI_results_i.pkl`; `teacher_student_runs.csv`, `teacher_student_cells.csv`, `variance_grid_cells.csv`; `figure_scripts/make_mechanism_figures.py` |
| Appendix C, Figure 5 (inner-optimiser ablation on MNIST, Adam at B = 256) | `src/inner_optimiser_ablation.py` → `results/mechanism/expB_*.pkl`; `inner_optimiser_runs.csv`; `figure_scripts/make_mechanism_figures.py` |
| Appendix D, Figure 6 (400-epoch MNIST runs under both operators, selection-off nulls) | `src/operator_drift_400ep.py` → `results/mechanism/expA_pretest_results.pkl` (density-agnostic), `expG_results_g-s{0,1,2}.pkl` (drift-neutral); `src/selection_off_null.py` → `expF_selection_null_results*.pkl`; `mnist_400epoch_runs.csv`, `selection_off_null_runs.csv`; `figure_scripts/make_mechanism_figures.py` |
| Appendix D, Table 4 (selection-off nulls of the UCI and QKD searches) | `src/selection_off_null_uci_qkd.py` → `results/mechanism/selection_null_uci_qkd_results.pkl`; `selection_off_null_uci_qkd.csv` |
| Appendix D, Table 5 (population size × elitism grid) | `src/population_elitism_grid.py` → `results/mechanism/grid_population_elitism_results.pkl`; `population_elitism_grid_cells.csv` |
| Appendix E, Figure 7 (UCI density drift, density-agnostic operator) | `results/nnpruning4_results.pkl`; `figure_scripts/make_saturation_figure.py` |
| Appendix E, Figure 8 (UCI density drift, drift-neutral operator) | `results/nnpruning4_drift_neutral_results.pkl`; `results/uci_density_traces_drift_neutral.csv`; `figure_scripts/make_saturation_figure_drift_neutral.py` |
| Appendix E, Figure 9 (density against accuracy) | `results/nnpruning3_results.pkl`, `results/nnpruning4_results.pkl`; `figure_scripts/make_parsimony_figure.py` |
| Appendix F, Table 6, density-agnostic column (B = 2^7 … 2^11) | `src/budget_sweep_mnist_60epochs.py` → `results/budget_sweep_mnist_60epochs_results.pkl` |
| Appendix F, Table 6, density-agnostic column (B = 2^12 … 2^14 and 2^15, 2^16) | `src/budget_sweep_mnist_60epochs_extended_low.py`, `..._extended_high.py` → the corresponding `results/*_extended_*_results.pkl` |
| Appendix F, Table 6, drift-neutral column | `src/budget_sweep_mnist_drift_neutral.py` → `results/mnist_sweep_drift_neutral_60ep_results.pkl`; B = 2^9, 2^11 and 2^13 read at epoch 60 of the 400-epoch runs (`results/mechanism/expG_results_g-s*.pkl`); `results/mnist_sweep_drift_neutral_cells.csv` |
| Appendix F, Figure 10 (panel A sweep, panel B convergence horizons) | the three density-agnostic sweep pickles and `src/convergence_horizon_mnist.py` → `results/convergence_horizon_mnist_results.pkl`; `figure_scripts/make_mnist_appendix_figure.py` |
| Appendix G, Table 7 (further tabular datasets), Table 1 re-run | `src/tabular_benchmarks.py` → `results/tabular_validation_results.pkl`, `results/tabular_table1_results.pkl`; `tabular_benchmarks_{validation,table1}_protocol.csv` |
| Appendix H, Figure 11 (QKD primer) | TikZ in the paper source |
| Appendix I, Figure 12 (BB84 training, strict score) | `results/exp7_results_rainbow_v3.pkl` (`history`); `figure_scripts/make_paper_figures_extra.py` |
| Appendix I, Figure 13 (BB84 paired differences from Oracle) | `results/exp5_results.pkl`, `exp6_results_v5.pkl`, `exp7_results_rainbow_v2.pkl`, `exp7_results_rainbow_v3.pkl`; `figure_scripts/make_rl_method_figures.py` |
| Appendix I, Table 8 (paired E91 statistics) | `src/e91_paired_stats.py`: replays the stored actions of `results/e91_exp7_results.pkl` on the seeded channel trajectories and reproduces every stored aggregate |
| Appendix I, Figures 14 and 15 (E91 training curve, E91 methods) | `results/e91_exp5_results.pkl`, `e91_exp6_results.pkl`, `e91_exp7_results.pkl`; `figure_scripts/make_rl_method_figures.py` |
| Appendix J, Table 9 (BB84 inner loops: SPSA, CMA-ES, random search; five seeds) | BB84 experiment-4 notebook, not yet in this release; the pluggable-optimiser framework is in `src/qkd_experiment_4_e91.py`. `src/bb84_random_vs_cmaes.py` → `results/bb84_random_vs_cmaes_results.pkl` is a separate three-seed check (CMA-ES, random search, finite differences at p = 0.1) that the paper does not report |
| Appendix K, Table 10 (BB84 ↔ E91 mapping) | no code |
| Appendix K, Tables 11–13 (E91 ablations) | `notebooks/QKD_experiment_5_E91.ipynb`, `_6_E91.ipynb`, `_7_E91.ipynb` → `results/e91_exp5_results.pkl`, `e91_exp6_results.pkl`, `e91_exp7_results.pkl` |
| Appendix K, Table 14 (4 against 7 levels) | 4 levels: `results/pathB_strat_dp.pkl` (DP), `results/e91_pathB_v3_results.pkl` (Oracle, PPO); 7 levels: `results/e91_exp5_results.pkl`. The 4-level BC value (0.3459) is not stored in the bundled files |
| Appendix K.7, Figure 16 (library against the Acín envelope) | `libraries/e91_library_pathB_7levels.pkl`; `figure_scripts/make_paper_figures.py` |
| Appendix K.7, numerical Pareto search and NPA check | `src/npa_e91_numerical.py`, `npa_e91_warmstart.py`, `npa_e91_smart.py`, `npa_e91_extended.py`, `npa_e91_fast.py`, `npa_e91_direct.py`; NPA implementation check (noiseless, min-entropy bound) in `src/npa_e91_sdp.py` |
| Appendix L (compute) | `results/compute_hours.csv` for the runs of Section 4.1, Appendices B–D and the camera-ready additions; wall-clock estimates below |

## Reproducing the experiments

1. Pruning, Tables 1 and 2 and Appendix E: run `notebooks/NNpruning_experiment_2.ipynb`, `_3.ipynb` and `_4.ipynb`, and `python src/mutation_operator_ablation_uci.py` for the drift-neutral rows of Table 2.
2. MNIST sweep, Appendix F: `python src/budget_sweep_mnist_60epochs.py`, the two `_extended_*.py` scripts and `python src/convergence_horizon_mnist.py`, then `python figure_scripts/make_mnist_appendix_figure.py`; `python src/budget_sweep_mnist_drift_neutral.py` for the drift-neutral column.
3. E91 library: `python src/pathB_lib7levels_build.py` (several hours; uses `src/evoluspsa_e91.py`).
4. RL on BB84, Table 3 and Appendix I: `notebooks/QKD_experiment_5.ipynb`, `_6.ipynb`, `_7.ipynb` (these load the BB84 grids `evolu_grid_*_bitflip.pkl`, which are not yet in this release).
5. RL on E91, Table 3 and Appendix K: `notebooks/QKD_experiment_5_E91.ipynb`, `_6_E91.ipynb`, `_7_E91.ipynb`, then `python src/e91_paired_stats.py`.
6. Envelopes: `python src/episode_mean_bounds.py` (writes `episode_mean_bounds.pkl` to the current directory).
7. Controlled mechanism tests and the runs added for the camera-ready version: see the next section.

The notebooks import modules from `src/` and expect the pickles they read in
their working directory: start Jupyter with `PYTHONPATH=src` and copy the
files a notebook names from `results/` and `libraries/` next to it. Random seeds are set in each
script and notebook (for example, the E91 test episodes use environment seeds
20,000 + episode index).

### Controlled mechanism tests and camera-ready additions

Run every command from the release root. Each script saves its result under
`results/` (the mechanism tests under `results/mechanism/`) and skips runs
already present there, so an interrupted run resumes when the same command is
repeated; delete a file to recompute it. Every script except
`selection_off_null.py` accepts `--smoke` for a short check that writes only
to the system temporary directory, and `N_JOBS_OVERRIDE` sets the number of
worker processes (each needs up to about 0.8 GB of memory). The docstring of
each script lists its other settings.

```bash
# Teacher–student sweep: one seed per command, four run sets (Appendix B)
for s in 0 1 2 3 4; do
  for set in e10 e10x e30 e30x; do RUN_SET=$set SEED=$s python src/teacher_student.py; done
done

python src/variance_grid.py                       # Figure 4(b)
python src/inner_optimiser_ablation.py            # Appendix C, Figure 5
B_OVERRIDE=256 OPTIMIZERS_OVERRIDE=adam \
  RESULTS_PATH_OVERRIDE=results/mechanism/expB_adam_B256_results.pkl \
  python src/inner_optimiser_ablation.py          # Appendix C, Adam at B = 256
python src/selection_off_null.py                  # Appendix D nulls, rho0 = 0.10
RHO_INIT_OVERRIDE=0.30 MUTATIONS_OVERRIDE=symmetric RESULTS_TAG=rho030 \
  python src/selection_off_null.py                # Appendix D null, rho0 = 0.30
OPERATOR=density_agnostic python src/operator_drift_400ep.py   # Figure 6, left
OPERATOR=drift_neutral    python src/operator_drift_400ep.py   # Figure 6, right

python src/mutation_operator_ablation_uci.py      # Table 2 drift-neutral rows, Appendix E
python src/budget_sweep_mnist_drift_neutral.py    # Appendix F, drift-neutral column
python src/selection_off_null_uci_qkd.py          # Appendix D, Table 4
python src/population_elitism_grid.py             # Appendix D, Table 5
PROTOCOL=table1     python src/tabular_benchmarks.py   # Section 4.2 timings, Appendix G
PROTOCOL=validation python src/tabular_benchmarks.py   # Appendix G, Table 7

python figure_scripts/make_mechanism_figures.py              # Figures 3-6
python figure_scripts/make_saturation_figure_drift_neutral.py  # Figure 8
```

The run sets of `teacher_student.py` are `e10` (ρ0 = 0.10, B = 2^7…2^14),
`e10x` (ρ0 = 0.10, B = 2^15), `e30` (ρ0 = 0.30, B = 2^7…2^14) and `e30x`
(ρ0 = 0.30, B = 2^10…2^12 under the stricter plateau criterion 5 × 10⁻⁵ with
a 600-epoch cap). `population_elitism_grid.py` reuses the stored `e10` runs at
B = 2^12 for its N = 30, E = 4 cell (`REUSE_MAIN_SWEEP=0` recomputes them).
The MNIST and teacher–student scripts import `src/budget_sweep_mnist.py`
(model, mutation operator, inner CMA-ES and outer loop); the UCI scripts are
script versions of `notebooks/NNpruning_experiment_2.ipynb` and
`NNpruning_experiment_4.ipynb`. The MNIST scripts download MNIST to
`~/.cache/mnist` on first use, and `tabular_benchmarks.py` fetches the four
further datasets from OpenML (cached by scikit-learn in `~/scikit_learn_data`).

### Reproducibility

All runs are seeded, and with the same library versions they reproduce the
stored results exactly, with one exception. pycma treats a seed of 0 as a
request for a clock-based seed, and in the CMA-ES runs the inner loop of the
first individual in the first epoch of every seed-0 run gets inner seed 0. A
seed-0 run can therefore depart from the stored one after its first epoch and
then agrees with it only statistically. Seeds 1 and above, the selection-off
nulls, the variance grid and the Adam inner loops (`tabular_benchmarks.py`, the
Adam arms of `inner_optimiser_ablation.py`) are unaffected.

`tabular_benchmarks.py` with `PROTOCOL=table1` re-implements
`notebooks/NNpruning_experiment_2.ipynb` and reproduces 17 of the 18 cells of
Table 1 to three decimals; IMP on Cancer differs in one seed, giving a mean of
0.952 against the 0.951 of Table 1, which reports the notebook's results.

## Wall-clock

| Experiment (16-core x86-64 workstation, CPU only) | Approximate cost |
|---|---|
| UCI pruning, Tables 1 and 2 | 6 CPU-hours |
| MNIST budget sweep, Appendix F, density-agnostic column | see Appendix L of the paper |
| BB84 library (24 cells) | 20 CPU-hours |
| E91 library (26 cells, two seeds per cell) | 60–100 CPU-hours |
| BB84 RL (strict and tolerated Rainbow-DQN; PPO, RecurrentPPO and MPC baselines) | 24 + 8–12 CPU-hours |
| E91 RL ablations | 30–50 CPU-hours |
| Pareto search and NPA check | 4–8 CPU-hours |
| `verify.py`, `src/e91_paired_stats.py` | under a minute each |

The controlled mechanism tests (Section 4.1, Appendices B–D) ran on lab
workstations with 14 or 20 worker processes per job and on a laptop (Intel
Core i7-10875H, 8 cores), 41.5 h of wall-clock time and about 615 core-hours
in all; the runs added for the camera-ready version ran on the laptop, mostly
with 8 worker processes, about 45 h and about 354 core-hours.
`results/compute_hours.csv` lists the wall-clock time and worker count of each
run set. With 8 worker processes on the laptop, the drift-neutral MNIST sweep
takes about 22 hours (mostly B = 2^15 and 2^16) and the N × E grid about 20
hours; the longest single jobs elsewhere are the B = 2^15 teacher–student runs
and the 400-epoch runs at B = 8192, roughly 1–3.5 hours each. The UCI runs
take about 1.5 hours in all, the tabular benchmarks under half an hour, and
`selection_off_null_uci_qkd.py` about 15 minutes on one core;
`selection_off_null.py` takes a few minutes and needs no data. Appendix L of
the paper gives the full breakdown.

## Not yet in this release

The following items are reported in the paper but their code or data are not
part of this bundle. They are to be added before the public release:

- the BB84 library grids `evolu_grid_safer_bitflip.pkl`, `evolu_grid_bitflip.pkl`, `evolu_grid_medium_bitflip.pkl` and `evolu_grid_hard_bitflip.pkl` (Section 4.3; read by the BB84 RL notebooks);
- the BB84 experiment-4 notebook behind Table 9 (Appendix J).

## Licence

MIT; see `LICENSE`.
