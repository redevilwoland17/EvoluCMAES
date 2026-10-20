"""Re-derive the numbers of the camera-ready paper from the bundled result files.

Run from the release root:

    python verify.py

For each reported value the script prints the value recomputed from results/,
libraries/ or notebook outputs, the value printed in the paper, and OK or
MISMATCH at the precision the paper reports. It needs NumPy and SciPy only;
the pruning pickles contain PyTorch modules, which are replaced by stubs on
loading, so PyTorch does not have to be installed. The exit status is the
number of mismatches.

The second half of the script checks the controlled mechanism tests and the
runs added for the camera-ready version (Section 4.1, Table 2, Appendices B-G)
against the stored results in results/mechanism/ and results/: it recomputes
every row of the CSV tables there and re-derives the values quoted in the text.
The BB84 inner-loop table of Appendix J is not checked, because the notebook
and grids behind it are not part of this release (see README.md).
"""
import json
import math
import os
import pickle
import sys

import numpy as np
from scipy import stats
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))
N_FAIL = 0


# ------------------------------------------------------------------ helpers
class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.state = state

    def __call__(self, *a, **k):
        return _Stub()


def _stub_fn(*a, **k):
    return None


class _StubUnpickler(pickle.Unpickler):
    """Loads pickles that reference torch or __main__ classes without importing them."""

    def find_class(self, module, name):
        if module.startswith('torch') or module == '__main__':
            return _stub_fn if name[:1].islower() else _Stub
        return super().find_class(module, name)

    def persistent_load(self, pid):
        return None


def load(rel):
    with open(os.path.join(ROOT, rel), 'rb') as f:
        return _StubUnpickler(f).load()


def hdr(s):
    print('\n' + '=' * 78 + f'\n  {s}\n' + '=' * 78)


def check(label, value, paper, nd):
    """Compare at the paper's precision: nd decimals (rounded half up/even tolerated)."""
    global N_FAIL
    ok = abs(round(float(value), nd) - paper) <= 0.5 * 10 ** (-nd) + 1e-12
    if not ok:
        N_FAIL += 1
    print(f'  {label:<52} {float(value):>10.{nd + 1}f}   paper {paper:<9}  {"OK" if ok else "MISMATCH"}')


# ======================================================================
hdr('Table 1 (tab:nnpruning): UCI pruning at rho = 0.05, five seeds')
r2 = load('results/nnpruning2_results.pkl')
PAPER_T1 = {  # (iris, wine, cancer, mean), mean +- std per dataset
    'EvoluPrune': ((0.933, 0.000), (0.996, 0.007), (0.986, 0.003), 0.972),
    'Magnitude': ((0.938, 0.022), (0.411, 0.044), (0.958, 0.009), 0.769),
    'IMP': ((0.911, 0.020), (0.885, 0.095), (0.951, 0.009), 0.916),
    'SNIP': ((0.920, 0.011), (0.959, 0.032), (0.959, 0.022), 0.946),
    'SynFlow': ((0.902, 0.011), (0.970, 0.009), (0.966, 0.008), 0.946),
    'Random': ((0.449, 0.231), (0.659, 0.206), (0.758, 0.162), 0.622),
}
for m, (*per_ds, mean_p) in PAPER_T1.items():
    means = []
    for ds, (mu_p, sd_p) in zip(['iris', 'wine', 'cancer'], per_ds):
        acc = np.array([rec['best']['acc_te'][-1] for rec in r2[ds][m][3]], float)
        means.append(acc.mean())
        check(f'{m} {ds} accuracy (mean)', acc.mean(), mu_p, 3)
        check(f'{m} {ds} accuracy (std)', acc.std(), sd_p, 3)
    check(f'{m} mean over datasets', np.mean(means), mean_p, 3)
print('  (EvoluPrune is the EvoluCMAES (biased) row; its accuracy is measured on the split used for selection.)')

# ======================================================================
hdr('Table 2 (tab:parsimony_ablation): alpha = 0 mutation ablation, five seeds')
r3 = load('results/nnpruning3_results.pkl')
r4 = load('results/nnpruning4_results.pkl')
PAPER_T2 = {
    'biased': {'iris': (0.978, 0.014, 0.057, 0.002), 'wine': (1.000, 0.000, 0.055, 0.004),
               'cancer': (0.979, 0.006, 0.052, 0.002)},
    # density-agnostic operator (0.40/0.20/0.40): values given in the caption of Table 2;
    # the accuracy standard deviation is not printed there (None)
    'density-agnostic (caption)': {'iris': (0.996, None, 0.153, 0.032), 'wine': (1.000, None, 0.120, 0.010),
                                   'cancer': (0.991, None, 0.156, 0.010)},
}
for label, group in {'biased': r3['alpha_natural'], 'density-agnostic (caption)': r4['alpha_natural_qmut']}.items():
    for ds in ['iris', 'wine', 'cancer']:
        acc = np.array([rec['best']['acc_te'][-1] for rec in group[ds]['hist']], float)
        den = np.array([rec['best']['density'][-1] for rec in group[ds]['hist']], float)
        a, sa, d, sd = PAPER_T2[label][ds]
        check(f'{label} {ds} accuracy', acc.mean(), a, 3)
        if sa is not None:
            check(f'{label} {ds} accuracy std', acc.std(), sa, 3)
        check(f'{label} {ds} density', den.mean(), d, 3)
        check(f'{label} {ds} density std', den.std(), sd, 3)
print('  (The drift-neutral rows of Table 2 are checked below, under "Additional runs".)')

# ======================================================================
hdr('Appendix F (tab:mnist_budget_sweep): MNIST 7x7 budget sweep, three seeds')
recs = []
for fn in ['results/budget_sweep_mnist_60epochs_results.pkl',
           'results/budget_sweep_mnist_60epochs_extended_low_results.pkl',
           'results/budget_sweep_mnist_60epochs_extended_high_results.pkl']:
    recs.extend(load(fn)['records'])
PAPER_MNIST = {128: (0.182, 0.016, 0.58), 256: (0.182, 0.012, 0.66), 512: (0.197, 0.018, 0.69),
               1024: (0.204, 0.002, 0.71), 2048: (0.208, 0.002, 0.74), 4096: (0.213, 0.005, 0.79),
               8192: (0.201, 0.008, 0.80), 16384: (0.219, 0.006, 0.84), 32768: (0.224, 0.016, 0.84),
               65536: (0.209, 0.010, 0.84)}
for B, (rho_p, sd_p, acc_p) in PAPER_MNIST.items():
    rs = [r for r in recs if r['B'] == B]
    rho = np.array([r['rho_pop_avg_last5'] for r in rs])
    acc = np.array([r['acc_best_final'] for r in rs])
    check(f'B = {B} density', rho.mean(), rho_p, 3)
    check(f'B = {B} density std', rho.std(), sd_p, 3)
    check(f'B = {B} best accuracy', acc.mean(), acc_p, 2)

# ======================================================================
hdr('Table 3 (tab:rl_headline), BB84 rows and the paired BB84 statistics')
v3 = load('results/exp7_results_rainbow_v3.pkl')['results']   # first set of 50 test episodes
v2 = load('results/exp7_results_rainbow_v2.pkl')['results']   # second set of 50 test episodes
dp_bb = load('libraries/bb84_strat_dp_hires.pkl')['summary']  # fine-grid DP, 200 episodes
fae = lambda d: float(np.mean(d['f_ae']))
DP_BB = dp_bb['mean_f_ae']
rows = [('Identity', fae(v3['Identity']), 0.5000, 66),
        ('Static safe (second set)', fae(v2['Static safe']), 0.7197, 96),
        ('Oracle (greedy)', fae(v3['Oracle']), 0.7434, 99),
        ('Rainbow-DQN strict, step 185k', fae(v3['Rainbow top-1 (step 185000)']), 0.7434, 99),
        ('Rainbow-DQN tolerated, step 750k (second set)', fae(v2['Rainbow top-1 (step 750000)']), 0.7477, 99)]
for name, val, paper, pct in rows:
    check(f'{name}: F_AE', val, paper, 4)
    check(f'{name}: % of DP', 100 * val / DP_BB, pct, 0)
check('DP reference (fine grid, 200 episodes): F_AE', DP_BB, 0.7527, 4)
check('DP reference: detection rate (%)', 100 * dp_bb['det_rate'], 0.0, 1)
tol = v2['Rainbow top-1 (step 750000)']
check('tolerated: detection rate (%)', 100 * tol['det'] / len(tol['f_ae']), 4.0, 1)
check('greedy Oracle on the second set', fae(v2['Oracle']), 0.7429, 4)
check('per-episode std of F_AE (strict)', np.std(v3['Rainbow top-1 (step 185000)']['f_ae'], ddof=1), 0.03, 2)
check('abstract: strict agent, % of DP', 100 * fae(v3['Rainbow top-1 (step 185000)']) / DP_BB, 98.8, 1)
d = np.array(tol['f_ae'], float) - np.array(v2['Oracle']['f_ae'], float)
half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))   # normal approximation, as in the figures
check('tolerated - Oracle, paired mean', d.mean(), 0.0047, 4)
check('tolerated - Oracle, paired 95% half-width', half, 0.0011, 4)
check('episodes in which tolerated > Oracle (of 50)', (d > 0).sum(), 46, 0)
p_sign = stats.binomtest(int((d > 0).sum()), int((d != 0).sum())).pvalue
print(f'  two-sided sign test p = {p_sign:.1e}   paper ~4e-10')

# ======================================================================
hdr('Table 3, E91 rows; Appendix K (tab:exp5/6/7_e91, tab:granularity)')
chi = lambda d: float(d['summary']['mean_chi']) if 'summary' in d else float(np.mean(d['chi']))
e7 = load('results/e91_exp7_results.pkl')['results']
DP_E = chi(e7['DP'])
for name, key, paper, pct in [('Identity', 'Identity', 0.000, 0), ('Oracle (greedy)', 'Oracle', 0.3507, 99),
                              ('Rainbow-DQN', 'Rainbow-DQN', 0.3484, 98), ('DP (coarse grid)', 'DP', 0.3540, 100)]:
    check(f'E91 {name}: chi', chi(e7[key]), paper, 4)
    check(f'E91 {name}: % of DP', 100 * chi(e7[key]) / DP_E, pct, 0)
check('E91 DP: detection rate (%)', 100 * e7['DP']['summary']['det_rate'], 7.0, 1)
check('E91 Rainbow-DQN per-episode std of chi', e7['Rainbow-DQN']['summary']['std_chi'], 0.065, 3)
check('abstract: E91 agent, % of DP', 100 * chi(e7['Rainbow-DQN']) / DP_E, 98.4, 1)
for tag, fname, paper in [
        ('exp5', 'results/e91_exp5_results.pkl',
         {'PPO': 0.2563, 'BC (Oracle)': 0.3496, 'DP': 0.3497, 'Oracle': 0.3473, 'Identity': 0.0000}),
        ('exp6', 'results/e91_exp6_results.pkl',
         {'RecurrentPPO': 0.2801, 'MPC': 0.3511, 'DP': 0.3520, 'Oracle': 0.3495, 'Identity': 0.0000}),
        ('exp7', 'results/e91_exp7_results.pkl',
         {'MaskablePPO': 0.3467, 'Rainbow-DQN': 0.3484, 'DP': 0.3540, 'Oracle': 0.3507, 'Identity': 0.0000})]:
    res = load(fname)['results']
    for m, p in paper.items():
        check(f'tab:{tag}_e91 {m}: chi', chi(res[m]), p, 4)
pa4 = load('results/e91_pathB_v3_results.pkl')['results']
check('tab:granularity 4-level DP', load('results/pathB_strat_dp.pkl')['summary']['mean_chi'], 0.3477, 4)
check('tab:granularity 4-level Oracle', chi(pa4['Oracle']), 0.3415, 4)
check('tab:granularity 4-level vanilla PPO', chi(pa4['RL (PPO)']), 0.2520, 4)
print('  tab:granularity 4-level BC (0.3459): not stored in the bundled 4-level result file.')

# ======================================================================
hdr('Appendix I (tab:e91_paired): paired per-episode statistics on E91')
import e91_paired_stats as eps  # noqa: E402

lib = load('libraries/e91_library_pathB_7levels.pkl')
S_TBL, CHI_TBL = eps.build_tables(lib)
per_ep = {}
for name in ['MaskablePPO', 'Rainbow-DQN', 'Oracle', 'DP']:
    per_ep[name], _ = eps.replay(e7[name]['actions'], S_TBL, CHI_TBL)
    check(f'replayed mean chi, {name}', per_ep[name].mean(), round(chi(e7[name]), 4), 4)
for b, (m_p, lo_p, hi_p) in {'MaskablePPO': (0.0017, 0.0012, 0.0022), 'Oracle': (-0.0023, -0.0028, -0.0018),
                             'DP': (-0.0056, -0.0065, -0.0047)}.items():
    m, lo, hi, pv = eps.paired(per_ep['Rainbow-DQN'], per_ep[b])
    check(f'Rainbow-DQN - {b}: mean difference', m, m_p, 4)
    check(f'Rainbow-DQN - {b}: 95% lower', lo, lo_p, 4)
    check(f'Rainbow-DQN - {b}: 95% upper', hi, hi_p, 4)
    print(f'  Rainbow-DQN - {b}: Wilcoxon p = {pv:.1e}')

# ======================================================================
hdr('Detection thresholds and analytical envelopes (Section 4.3, Table 3, Appendices H and K)')
h = lambda x: 0.0 if x <= 0 or x >= 1 else -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def key_rate(S):
    s2 = (S / 2) ** 2
    return 0.0 if s2 <= 1 else max(0.0, 1 - h(0.5 + 0.5 * math.sqrt(s2 - 1)) - h((1 - S / (2 * math.sqrt(2))) / 2))


S_sec = brentq(lambda s: key_rate(s) - 1e-6, 2.01, 2.8)
check('E91 S_sec', S_sec, 2.4240, 4)
check('E91 detection threshold S_sec + 0.02', S_sec + 0.02, 2.444, 3)
check('E91 p_cliff = 1 - sqrt(S_sec / 2 sqrt 2)', 1 - math.sqrt(S_sec / (2 * math.sqrt(2))), 0.0742, 4)
check('Acin envelope at the threshold', h(0.5 + 0.5 * math.sqrt(((S_sec + 0.02) / 2) ** 2 - 1)), 0.607, 3)
check('BB84 cloning envelope at F_AB = 0.87', 0.5 + math.sqrt(0.87 * 0.13), 0.836, 3)
em = load('results/episode_mean_bounds.pkl')
check('episode-mean ceiling, BB84 (episode_mean_bounds.pkl)', em['bb84']['fae_avg'], 0.836, 3)
check('episode-mean ceiling, E91 (episode_mean_bounds.pkl)', em['e91']['chi_avg'], 0.607, 3)
check('honest BB84 F_AB at the 11% Shor-Preskill error rate', 1 - 0.11, 0.89, 2)

# ======================================================================
hdr('Section 4.3: BB84 library cells against the noiseless cloning value')
nb = json.load(open(os.path.join(ROOT, 'notebooks/QKD_experiment_5.ipynb')))
out = ''.join(''.join(o.get('text', '')) for o in nb['cells'][9].get('outputs', []))
cells = []
for line in out.splitlines():
    parts = line.replace('=', ' ').split()
    if len(parts) > 6 and parts[1] == 'p' and parts[3] == 'saved':
        p, fab, fae_ = float(parts[2].rstrip(':')), float(parts[5]), float(parts[7])
        cells.append((parts[0], p, fab, fae_, fae_ / (0.5 + math.sqrt(fab * (1 - fab)))))
print(f'  {len(cells)} library cells read from the stored output of notebooks/QKD_experiment_5.ipynb')
for p_ref, lo_p, hi_p in [(0.02, 98.0, 99.0), (0.25, 70, 87)]:
    r = [100 * c[4] for c in cells if abs(c[1] - p_ref) < 1e-9]
    nd = 1 if p_ref == 0.02 else 0
    check(f'p = {p_ref}: lowest ratio to the cloning value (%)', min(r), lo_p, nd)
    check(f'p = {p_ref}: highest ratio to the cloning value (%)', max(r), hi_p, nd)

# ===========================================================================
# Controlled mechanism tests: Figures 3-6, Appendices B and D
# (results/mechanism/). Every row of the CSV tables there is recomputed from
# the result pickles, and the values quoted in the text are re-derived.
# ===========================================================================
import csv as _csv
import glob as _glob
import os as _os
import pickle as _pickle
import numpy as _np

_MECH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "results", "mechanism")
_mech_fail = []


def _mload(name):
    with open(_os.path.join(_MECH, name), "rb") as f:
        return _pickle.load(f)


def _mrecs(pattern):
    out = []
    for p in sorted(_glob.glob(_os.path.join(_MECH, pattern))):
        with open(p, "rb") as f:
            out += [dict(r, _file=_os.path.basename(p)) for r in _pickle.load(f)["records"]]
    return out


def _mcsv(name):
    with open(_os.path.join(_MECH, name)) as f:
        return list(_csv.DictReader(f))


def _same(computed, text):
    """Does `computed` round to the value printed as `text` in a CSV?"""
    text = text.strip()
    if isinstance(computed, str):
        return computed == text
    if text == "":
        return computed in (None, "", 0) or (isinstance(computed, float) and _np.isnan(computed))
    v = float(text)
    if "e" in text.lower():
        digits = len(text.lower().split("e")[0].replace("-", "").replace(".", "").lstrip("0"))
        return abs(computed - v) <= 0.5 * 10 ** (_np.floor(_np.log10(abs(v))) - digits + 1) + 1e-15
    dec = len(text.split(".")[1]) if "." in text else 0
    return abs(computed - v) <= 0.5 * 10 ** (-dec) + 1e-9


def _check_table(name, rows, recompute, keys):
    bad = 0
    for row in rows:
        got = recompute(row)
        for k in keys:
            if not _same(got[k], row[k]):
                bad += 1
                _mech_fail.append(f"{name}: {k} for {[row[c] for c in list(row)[:4]]}: "
                                  f"recomputed {got[k]!r}, CSV {row[k]!r}")
    print(f"  {name:32s} {len(rows):3d} rows x {len(keys)} columns: "
          f"{'all match' if bad == 0 else f'{bad} MISMATCHES'}")


def _claim(label, value, paper, ok):
    print(f"  {label:58s} {value:>18s}   (paper {paper})" + ("" if ok else "   MISMATCH"))
    if not ok:
        _mech_fail.append(f"{label}: {value} vs paper {paper}")


hdr("Controlled mechanism tests: Figures 3-6, Appendices B and D")

# ---- teacher-student runs (Figure 3, Figure 4(a), Appendix B) ----------------
ts_sets = {s: _mrecs(f"expE_results_{s}-s*.pkl") for s in ("e10", "e10x", "e30", "e30x")}
for s, rr in ts_sets.items():
    for r in rr:
        r["run_set"] = s
ts_all = [r for rr in ts_sets.values() for r in rr]
_cap = {"e10": 300, "e10x": 300, "e30": 300, "e30x": 600}
_thr = {"e10": 2e-4, "e10x": 2e-4, "e30": 2e-4, "e30x": 5e-5}
_open = lambda r: r["run_set"] == "e30x" and r["best_clean_mse_traj"][-1] < 0.05


def _ts_run(row):
    r = next(x for x in ts_sets[row["run_set"]]
             if x["B"] == int(row["B"]) and x["seed"] == int(row["seed"]))
    return {"rho_init": r["rho_init"], "epochs_run": r["epochs_run"], "epoch_cap": _cap[row["run_set"]],
            "plateau_slope_threshold": _thr[row["run_set"]], "plateaued": int(r["plateaued"]),
            "rho_star_mean_last40_epochs": r["rho_eq_last40"],
            "final_best_individual_clean_mse": r["best_clean_mse_traj"][-1],
            "plotted_as_open_square_fig3": (1 if _open(r) else 0) if r["run_set"] == "e30x" else "",
            "wall_clock_s": r["elapsed"]}


rows = _mcsv("teacher_student_runs.csv")
_check_table("teacher_student_runs.csv", rows, _ts_run,
             ["rho_init", "epochs_run", "epoch_cap", "plateau_slope_threshold", "plateaued",
              "rho_star_mean_last40_epochs", "final_best_individual_clean_mse",
              "plotted_as_open_square_fig3", "wall_clock_s"])
_series = {
    "Fig3 filled circles": ts_sets["e10"] + ts_sets["e10x"],
    "Fig3 filled squares": ts_sets["e30"],
    "Fig3 open squares": [r for r in ts_sets["e30x"] if _open(r)],
    "not plotted": ts_sets["e30x"],
}


def _ts_cell(row):
    rr = [r for r in _series[row["series"].split(":")[0]] if r["B"] == int(row["B"])]
    rho = _np.array([r["rho_eq_last40"] for r in rr])
    mse = _np.array([r["best_clean_mse_traj"][-1] for r in rr])
    return {"log2_B": _np.log2(int(row["B"])), "n_seeds": len(rr), "rho_star_mean": rho.mean(),
            "sd_ddof0_as_plotted": rho.std(), "sd_ddof1": rho.std(ddof=1) if len(rr) > 1 else _np.nan,
            "median_final_clean_mse": _np.median(mse), "min_clean_mse": mse.min(), "max_clean_mse": mse.max()}


cells = _mcsv("teacher_student_cells.csv")
_check_table("teacher_student_cells.csv", cells, _ts_cell,
             ["log2_B", "n_seeds", "rho_star_mean", "sd_ddof0_as_plotted", "sd_ddof1",
              "median_final_clean_mse", "min_clean_mse", "max_clean_mse"])

circ = {int(c["B"]): _ts_cell(c)["rho_star_mean"] for c in cells if c["series"].startswith("Fig3 filled circles")}
lo_band = [circ[2**k] for k in range(7, 12)]
_claim("Fig 3: sparse-basin rho* for B = 2^7 .. 2^11", f"{min(lo_band):.3f}-{max(lo_band):.3f}", "~0.09",
       round(min(lo_band), 2) >= 0.08 and round(max(lo_band), 2) <= 0.10)
_claim("Fig 3: sparse-basin rho* at B = 2^12", f"{circ[2**12]:.3f}", "0.054", f"{circ[2**12]:.3f}" == "0.054")
_claim("Fig 3: sparse-basin rho* at B = 2^15", f"{circ[2**15]:.3f}", "0.097", f"{circ[2**15]:.3f}" == "0.097")
_claim("Fig 3: sparse-basin rho* rises monotonically 2^12 -> 2^15",
       str(all(circ[2**k] < circ[2**(k + 1)] for k in range(12, 15))), "True",
       all(circ[2**k] < circ[2**(k + 1)] for k in range(12, 15)))
main_runs = ts_sets["e10"] + ts_sets["e10x"] + ts_sets["e30"]
ep = [r["epochs_run"] for r in main_runs]
_claim("App B: runs stopped on the plateau criterion", f"{sum(r['plateaued'] for r in main_runs)} of {len(main_runs)}",
       "all 85", len(main_runs) == 85 and all(r["plateaued"] for r in main_runs))
_claim("App B: runs stopping at epoch 120 / latest stop", f"{ep.count(120)} / {max(ep)}", "72 / 173",
       ep.count(120) == 72 and max(ep) == 173)
epx = [r["epochs_run"] for r in ts_sets["e30x"]]
_claim("App B: strict re-runs (B = 2^10..2^12) stopped at epochs", f"{min(epx)}-{max(epx)}", "120-178",
       (min(epx), max(epx)) == (120, 178) and all(r["plateaued"] for r in ts_sets["e30x"]))
sp = [r["best_clean_mse_traj"][-1] for r in ts_sets["e10"] + ts_sets["e10x"]]
_claim("App B / Fig 4(a): sparse-init clean error, all budgets", f"{min(sp):.4f}-{max(sp):.4f}", "near 0.003",
       0.001 <= min(sp) and max(sp) <= 0.005)
med = [_np.median([r["best_clean_mse_traj"][-1] for r in ts_sets["e30"] if r["B"] == B]) for B in (2**7, 2**8, 2**9)]
_claim("App B / Fig 4(a): dense-init median clean error, B <= 2^9", f"{min(med):.3f}-{max(med):.3f}", "0.157-0.166",
       f"{min(med):.3f}-{max(med):.3f}" == "0.157-0.166")
dl = [r["best_clean_mse_traj"][-1] for r in ts_sets["e30"] if r["B"] <= 2**9]
_learn = [x for x in dl if x < 0.05]
_stay = [x for x in dl if x >= 0.05]
_claim("App B / Fig 3: dense-init seeds that do not optimise (clean error >= 0.05), B <= 2^9",
       f"{len(_stay)} of {len(dl)}", "11 of 15", (len(_stay), len(dl)) == (11, 15))
_claim("App B: clean error of those seeds", f"{min(_stay):.2f}-{max(_stay):.2f}", "0.13-0.22",
       f"{min(_stay):.2f}-{max(_stay):.2f}" == "0.13-0.22")
_claim("App B: clean error of the other dense-init seeds, B <= 2^9", f"{min(_learn):.3f}-{max(_learn):.3f}",
       "0.003-0.005", f"{min(_learn):.3f}-{max(_learn):.3f}" == "0.003-0.005")
op = [_ts_cell(c)["rho_star_mean"] for c in cells if c["series"].startswith("Fig3 open squares")]
al = [_ts_cell(c)["rho_star_mean"] for c in cells if c["series"].startswith("not plotted")]
_claim("App B: strict rho*, seeds that optimise (open squares)", f"{min(op):.3f}-{max(op):.3f}", "0.258-0.265",
       f"{min(op):.3f}-{max(op):.3f}" == "0.258-0.265")
_claim("App B: strict rho*, all five seeds", f"{min(al):.3f}-{max(al):.3f}", "0.266-0.277",
       f"{min(al):.3f}-{max(al):.3f}" == "0.266-0.277")
sq = [_ts_cell(c)["rho_star_mean"] for c in cells if c["series"].startswith("Fig3 filled squares")]
_claim("Sec 4.1: dense-basin rho* (filled and open squares)", f"{min(sq + op):.2f}-{max(sq + op):.2f}", "0.26-0.29",
       f"{min(sq + op):.2f}-{max(sq + op):.2f}" == "0.26-0.29")
_nl10 = _np.mean([_np.mean(r["avg_density_traj"][80:120]) for r in _mload("expF_selection_null_results.pkl")["records"]
                  if r["mutation"] == "symmetric"])
_nl30 = _np.mean([_np.mean(r["avg_density_traj"][80:120])
                  for r in _mload("expF_selection_null_results_rho030.pkl")["records"]])
gaps = [_nl10 - x for x in circ.values()] + [_nl30 - x for x in sq + op]
_claim("Sec 4.1: nulls (0.102, 0.300) above both basins by", f"{min(gaps):.3f}-{max(gaps):.3f}", "0.005-0.05",
       f"{min(gaps):.3f}" == "0.005" and f"{max(gaps):.2f}" == "0.05" and min(gaps) > 0)

# ---- variance grid (Figure 4(b), Appendix B) ----------------------------------
vg = _mload("expI_results_i.pkl")


def _vg_cell(row):
    rho, B = float(row["rho"]), int(row["B"])
    rr = [r for r in vg["records"] if abs(r["rho"] - rho) < 1e-9 and r["B"] == B]
    fit = _np.array([r["noisy_fit"] for r in rr]); cl = _np.array([r["clean_mse"] for r in rr])
    return {"log2_B": _np.log2(B), "n_cold_starts": len(rr), "mean_active_params": _np.mean([r["dims"] for r in rr]),
            "var_noisy_fitness_ddof0": fit.var(), "mean_noisy_fitness": fit.mean(),
            "mean_clean_mse_as_plotted_fig4b": vg["cells"][(rho, B)]["mean_clean"],
            "median_clean_mse": _np.median(cl), "var_clean_mse_ddof0": cl.var()}


_check_table("variance_grid_cells.csv", _mcsv("variance_grid_cells.csv"), _vg_cell,
             ["log2_B", "n_cold_starts", "mean_active_params", "var_noisy_fitness_ddof0", "mean_noisy_fitness",
              "mean_clean_mse_as_plotted_fig4b", "median_clean_mse", "var_clean_mse_ddof0"])
_claim("Sec 4.1 / App B: cold-start inner runs in the grid", f"{len(vg['records']):,}", "2,700",
       len(vg["records"]) == 2700 and all(c["n"] == 50 for c in vg["cells"].values()))
v = [c["var_fit"] for c in vg["cells"].values()]
_claim("App B: span of Var[f_B] over the grid (orders of magnitude)", f"{_np.log10(max(v) / min(v)):.1f}",
       "nearly five", 4.5 <= _np.log10(max(v) / min(v)) < 5)

# ---- inner-optimiser ablation (Figure 5, Appendix C) ---------------------------
ib = _mload("expB_inner_ablation_mnist_results.pkl")["records"]
ib256 = _mload("expB_adam_B256_results.pkl")["records"]


def _ib_run(row):
    r = next(x for x in ib + ib256 if x["optimizer"] == row["inner_optimiser"] and x["B"] == int(row["B"])
             and x["seed"] == int(row["seed"]))
    return {"B": r["B"], "outer_epochs": len(r["avg_density_traj"]),
            "rho_pop_mean_last5_epochs": r["rho_pop_avg_last5"],
            "best_individual_acc_final_epoch": r["acc_best_final"], "wall_clock_s": r["elapsed"]}


_check_table("inner_optimiser_runs.csv", _mcsv("inner_optimiser_runs.csv"), _ib_run,
             ["B", "outer_epochs", "rho_pop_mean_last5_epochs", "best_individual_acc_final_epoch", "wall_clock_s"])
m_rho = {o: _np.mean([r["rho_pop_avg_last5"] for r in ib if r["optimizer"] == o]) for o in ("cmaes", "random", "fdsgd", "adam")}
m_acc = {o: _np.mean([r["acc_best_final"] for r in ib if r["optimizer"] == o]) for o in m_rho}
_claim("Fig 5 / App C: rho* range over the four optimisers", f"{min(m_rho.values()):.3f}-{max(m_rho.values()):.3f}",
       "0.186-0.215", f"{min(m_rho.values()):.3f}-{max(m_rho.values()):.3f}" == "0.186-0.215")
_claim("Fig 5 / App C: best accuracy, random search / Adam", f"{m_acc['random']:.3f} / {m_acc['adam']:.3f}",
       "0.544 / 0.847", f"{m_acc['random']:.3f} / {m_acc['adam']:.3f}" == "0.544 / 0.847")
a_rho = _np.array([r["rho_pop_avg_last5"] for r in ib256]); a_acc = _np.array([r["acc_best_final"] for r in ib256])
_claim("App C: Adam at B = 256, rho* (three seeds)", f"{a_rho.mean():.3f} +/- {a_rho.std():.3f}", "0.222 +/- 0.010",
       len(ib256) == 3 and f"{a_rho.mean():.3f} +/- {a_rho.std():.3f}" == "0.222 +/- 0.010")
_claim("App C: Adam at B = 256, best accuracy", f"{a_acc.mean():.3f} +/- {a_acc.std():.3f}", "0.842 +/- 0.006",
       f"{a_acc.mean():.3f} +/- {a_acc.std():.3f}" == "0.842 +/- 0.006")

# ---- 400-epoch operator runs and selection-off nulls (Figure 6, Appendix D) ---
da = [dict(r, op="density-agnostic 0.40/0.20/0.40") for r in _mload("expA_pretest_results.pkl")["records"]]
dn = [dict(r, op="drift-neutral 0.40/0.30/0.30") for r in _mrecs("expG_results_g-s*.pkl")]


def _op_run(row):
    r = next(x for x in da + dn if x["op"] == row["operator_swap/remove/add"]
             and x["B"] == int(row["B"]) and x["seed"] == int(row["seed"]))
    t = r["avg_density_traj"]
    return {"epochs": len(t), "rho_mean_epochs_56-60": _np.mean(t[55:60]), "best_acc_epoch_60": r["best_acc_traj"][59],
            "rho_mean_last50_epochs": r["rho_pop_avg_last50"], "rho_epoch_400": t[-1],
            "best_acc_epoch_400": r["best_acc_traj"][-1], "wall_clock_s": r["elapsed"]}


_check_table("mnist_400epoch_runs.csv", _mcsv("mnist_400epoch_runs.csv"), _op_run,
             ["epochs", "rho_mean_epochs_56-60", "best_acc_epoch_60", "rho_mean_last50_epochs",
              "rho_epoch_400", "best_acc_epoch_400", "wall_clock_s"])
nl = [dict(r, rho0=0.10) for r in _mload("expF_selection_null_results.pkl")["records"]]
nl30 = [dict(r, rho0=_mload("expF_selection_null_results_rho030.pkl")["config"]["RHO_INIT"])
        for r in _mload("expF_selection_null_results_rho030.pkl")["records"]]
_nname = {"qmut": "density-agnostic 0.40/0.20/0.40", "symmetric": "drift-neutral 0.40/0.30/0.30"}


def _null_run(row):
    r = next(x for x in nl + nl30 if _nname[x["mutation"]] == row["operator_swap/remove/add"]
             and abs(x["rho0"] - float(row["rho_init_and_reseed_density"])) < 1e-9 and x["seed"] == int(row["seed"]))
    t = r["avg_density_traj"]
    return {"rho_init_and_reseed_density": r["rho0"], "rho_mean_epochs_81-120": _np.mean(t[80:120]),
            "rho_mean_last50_of_400": r["rho_last50"], "min_after_epoch_50": min(t[50:]),
            "max_after_epoch_50": max(t[50:])}


_check_table("selection_off_null_runs.csv", _mcsv("selection_off_null_runs.csv"), _null_run,
             ["rho_init_and_reseed_density", "rho_mean_epochs_81-120", "rho_mean_last50_of_400",
              "min_after_epoch_50", "max_after_epoch_50"])
_claim("Fig 6: seeds per curve (density-agnostic 512 / 8192; drift-neutral)",
       f"{sum(r['B'] == 512 for r in da)} / {sum(r['B'] == 8192 for r in da)}; "
       f"{min(sum(r['B'] == B for r in dn) for B in (512, 2048, 8192))}", "3 / 2; 3",
       (sum(r["B"] == 512 for r in da), sum(r["B"] == 8192 for r in da)) == (3, 2)
       and all(sum(r["B"] == B for r in dn) == 3 for B in (512, 2048, 8192)))
e400 = lambda rr, B: _np.array([r["avg_density_traj"][-1] for r in rr if r["B"] == B])
_claim("App D: density-agnostic density at epoch 400, B = 512 / 8192",
       f"{e400(da, 512).mean():.2f} / {e400(da, 8192).mean():.2f}", "about 0.56 / 0.49",
       f"{e400(da, 512).mean():.2f} / {e400(da, 8192).mean():.2f}" == "0.56 / 0.49")
qn = _np.array([r["rho_last50"] for r in nl if r["mutation"] == "qmut"])
_claim("App D: density-agnostic selection-off null", f"{qn.mean():.3f} +/- {qn.std():.3f}", "0.112 +/- 0.001",
       f"{qn.mean():.3f} +/- {qn.std():.3f}" == "0.112 +/- 0.001")
for B, paper in ((512, "0.194 +/- 0.033"), (2048, "0.232 +/- 0.030"), (8192, "0.201 +/- 0.028")):
    x = e400(dn, B)
    _claim(f"App D: drift-neutral density at epoch 400, B = {B}", f"{x.mean():.3f} +/- {x.std():.3f}", paper,
           f"{x.mean():.3f} +/- {x.std():.3f}" == paper)
sn = _np.array([r["rho_last50"] for r in nl if r["mutation"] == "symmetric"])
_claim("App D: drift-neutral selection-off null (last 50 of 400)", f"{sn.mean():.3f} +/- {sn.std():.3f}",
       "0.102 +/- 0.002", f"{sn.mean():.3f} +/- {sn.std():.3f}" == "0.102 +/- 0.002")
s81 = _np.array([_np.mean(r["avg_density_traj"][80:120]) for r in nl if r["mutation"] == "symmetric"])
_claim("App D: teacher-student null at rho0 = 0.10 (epochs 81-120)", f"{s81.mean():.3f} +/- {s81.std():.3f}",
       "0.102 +/- 0.001", f"{s81.mean():.3f} +/- {s81.std():.3f}" == "0.102 +/- 0.001")
s30 = _np.array([_np.mean(r["avg_density_traj"][80:120]) for r in nl30])
_claim("App D: teacher-student null at rho0 = 0.30 (epochs 81-120)", f"{s30.mean():.3f}", "0.300",
       len(nl30) == 3 and f"{s30.mean():.3f}" == "0.300")

# ===========================================================================
# Additional runs: Table 2 drift-neutral rows (results/), Appendix E traces,
# Appendix F drift-neutral sweep, Appendix D UCI/QKD nulls and N x E grid
# (results/mechanism/), Appendix G and the Section 4.2 wall-clock times.
# ===========================================================================
_RES = _os.path.dirname(_MECH)


def _rload(name):
    with open(_os.path.join(_RES, name), "rb") as f:
        return _pickle.load(f)


def _rcsv(name):
    with open(_os.path.join(_RES, name)) as f:
        return list(_csv.DictReader(f))


_pm = lambda x, nd=3: f"{_np.mean(x):.{nd}f} +/- {_np.std(x):.{nd}f}"
_span = lambda xs, nd=3: f"{min(xs):.{nd}f}-{max(xs):.{nd}f}"

hdr("Additional runs: Table 2 drift-neutral rows, Appendices D-G")

# ---- Table 2 rows and Appendix E traces (drift-neutral operator) ---------------
t2 = _rload("nnpruning4_drift_neutral_results.pkl")["records"]
_t2 = lambda ds: sorted([r for r in t2 if r["dataset"] == ds], key=lambda r: r["seed"])


def _t2_row(row):
    rr = _t2(row["dataset"])
    a = [r["acc_te_best_final"] for r in rr]; db = [r["density_best_final"] for r in rr]
    dp = [r["avg_d_final"] for r in rr]
    return {"seeds": len(rr), "test_acc_mean": _np.mean(a), "test_acc_std": _np.std(a),
            "density_best_mean": _np.mean(db), "density_best_std": _np.std(db),
            "density_popmean_mean": _np.mean(dp), "density_popmean_std": _np.std(dp),
            "epochs_run_per_seed": " ".join(str(r["epochs_run"]) for r in rr)}


_check_table("table2_drift_neutral_rows.csv", _rcsv("table2_drift_neutral_rows.csv"), _t2_row,
             ["seeds", "test_acc_mean", "test_acc_std", "density_best_mean", "density_best_std",
              "density_popmean_mean", "density_popmean_std", "epochs_run_per_seed"])


def _t2_trace(row):
    e = int(row["epoch"])
    v = [r["avg_d"][e - 1] for r in _t2(row["dataset"]) if len(r["avg_d"]) >= e]
    return {"seeds_running": len(v), "popmean_density_mean": _np.mean(v), "popmean_density_std": _np.std(v)}


_check_table("uci_density_traces_drift_neutral.csv", _rcsv("uci_density_traces_drift_neutral.csv"), _t2_trace,
             ["seeds_running", "popmean_density_mean", "popmean_density_std"])
for ds, acc, rho in (("iris", "0.987 +/- 0.011", "0.066 +/- 0.008"), ("wine", "0.996 +/- 0.007", "0.067 +/- 0.007"),
                     ("cancer", "0.982 +/- 0.004", "0.055 +/- 0.004")):
    rr = _t2(ds)
    got = (_pm([r["acc_te_best_final"] for r in rr]), _pm([r["density_best_final"] for r in rr]))
    _claim(f"Table 2: drift-neutral row, {ds} (accuracy; final density)", "; ".join(got), f"{acc}; {rho}",
           len(rr) == 5 and got == (acc, rho))
for ds, paper in (("iris", "0.064 +/- 0.006"), ("wine", "0.064 +/- 0.005"), ("cancer", "0.056 +/- 0.003")):
    got = _pm([r["avg_d_final"] for r in _t2(ds)])
    _claim(f"App E: drift-neutral population-mean density at the last epoch, {ds}", got, paper, got == paper)
start = [_np.mean([r["avg_d"][0] for r in _t2(ds)]) for ds in ("iris", "wine", "cancer")]
_claim("App E: drift-neutral traces start at (epoch 1, three datasets)", _span(start), "0.050",
       _span(start) == "0.050-0.050")

# ---- Appendix F: MNIST sweep under the drift-neutral operator ------------------
sw = _rload("mnist_sweep_drift_neutral_60ep_results.pkl")["records"]
g4 = _mrecs("expG_results_g-s*.pkl")


def _sw_cell(row):
    B = int(row["B"])
    if row["source"] == "60-epoch run":
        rr = [r for r in sw if r["B"] == B]
        rho = [r["rho_pop_avg_last5"] for r in rr]; acc = [r["acc_best_final"] for r in rr]
        wall = sum(r["elapsed"] for r in rr) / 3600
    else:
        rr = [r for r in g4 if r["B"] == B]
        rho = [_np.mean(r["avg_density_traj"][55:60]) for r in rr]; acc = [r["best_acc_traj"][59] for r in rr]
        wall = ""
    return {"log2_B": _np.log2(B), "seeds": len(rr), "rho_epochs56_60_mean": _np.mean(rho),
            "rho_epochs56_60_std": _np.std(rho), "best_acc_epoch60_mean": _np.mean(acc),
            "best_acc_epoch60_std": _np.std(acc), "wall_clock_h_sum_over_seeds": wall}


cells_f = _rcsv("mnist_sweep_drift_neutral_cells.csv")
_check_table("mnist_sweep_drift_neutral_cells.csv", cells_f, _sw_cell,
             ["log2_B", "seeds", "rho_epochs56_60_mean", "rho_epochs56_60_std", "best_acc_epoch60_mean",
              "best_acc_epoch60_std", "wall_clock_h_sum_over_seeds"])
_appf = {7: ("0.126 +/- 0.011", "0.57"), 8: ("0.124 +/- 0.013", "0.60"), 9: ("0.128 +/- 0.005", "0.61"),
         10: ("0.134 +/- 0.012", "0.65"), 11: ("0.136 +/- 0.007", "0.70"), 12: ("0.143 +/- 0.009", "0.74"),
         13: ("0.139 +/- 0.007", "0.78"), 14: ("0.162 +/- 0.009", "0.81"), 15: ("0.156 +/- 0.005", "0.80"),
         16: ("0.147 +/- 0.007", "0.80")}
for c in cells_f:
    k = int(c["log2_B"]); got = _sw_cell(c)
    val = (f"{got['rho_epochs56_60_mean']:.3f} +/- {got['rho_epochs56_60_std']:.3f}", f"{got['best_acc_epoch60_mean']:.2f}")
    _claim(f"App F: drift-neutral sweep at B = 2^{k} (rho, epochs 56-60; best accuracy)", "; ".join(val),
           "; ".join(_appf[k]), got["seeds"] == 3 and val == _appf[k])
sn_null = _np.mean([r["rho_last50"] for r in nl if r["mutation"] == "symmetric"])
lowest = min(_sw_cell(c)["rho_epochs56_60_mean"] for c in cells_f)
_claim("App F: every budget above the drift-neutral null", f"{lowest:.3f} > {sn_null:.3f}", "0.102",
       lowest > sn_null and f"{sn_null:.3f}" == "0.102")

# ---- Appendix D: selection-off nulls of the UCI and QKD operators ----------------
uq = _mload("selection_null_uci_qkd_results.pkl")["records"]
_uqk = lambda r: (r["part"], r.get("setting", ""), r["operator"], r.get("dataset", ""))


def _uq_row(row):
    rr = [r for r in uq if _uqk(r) == (row["part"], row["setting"], row["operator"], row["dataset"])]
    t0 = [r["traj"][0] for r in rr]; tf = [r["traj"][-1] for r in rr]; t10 = [_np.mean(r["traj"][-10:]) for r in rr]
    return {"N": rr[0]["N"], "E": rr[0]["E"], "R": rr[0]["R"], "size_measure": rr[0]["size"],
            "epochs": len(rr[0]["traj"]) - 1, "seeds": len(rr), "initial_mean": _np.mean(t0),
            "final_mean": _np.mean(tf), "final_std": _np.std(tf),
            "last10_epochs_mean": _np.mean(t10), "last10_epochs_std": _np.std(t10)}


_check_table("selection_off_null_uci_qkd.csv", _mcsv("selection_off_null_uci_qkd.csv"), _uq_row,
             ["N", "E", "R", "size_measure", "epochs", "seeds", "initial_mean", "final_mean", "final_std",
              "last10_epochs_mean", "last10_epochs_std"])
_uci = lambda setting, op: [_np.mean([r["traj"][-1] for r in uq if r["part"] == "uci" and r["setting"] == setting
                                      and r["operator"].startswith(op) and r["dataset"] == ds])
                            for ds in ("iris", "wine", "cancer")]
for setting, op, paper in (("table2", "density-agnostic", "0.063-0.065"), ("table2", "drift-neutral", "0.049-0.051"),
                           ("table2", "density-driven", "0.050-0.051"), ("table1", "density-driven", "0.050-0.051")):
    got = _span(_uci(setting, op))
    _claim(f"App D: UCI null, {setting} loop, {op} (three network shapes)", got, paper, got == paper)
for part, start, paper, paper10 in (("bb84", "5", "5.75 +/- 0.14", "5.57 +/- 0.05"),
                                    ("e91", "6", "6.03 +/- 0.10", "6.05 +/- 0.05")):
    rr = [r for r in uq if r["part"] == part]
    got = (f"{_np.mean([r['traj'][0] for r in rr]):.0f}", _pm([r["traj"][-1] for r in rr], 2),
           _pm([_np.mean(r["traj"][-10:]) for r in rr], 2))
    _claim(f"App D: {part.upper()} null, gates (start; final; last 10 epochs)", "; ".join(got),
           f"{start}; {paper}; {paper10}", got == (start, paper, paper10))

# ---- Appendix D: population size x elitism grid ----------------------------------
gd = _mload("grid_population_elitism_results.pkl")


def _gd_cell(row):
    N, E = int(row["N"]), int(row["E"])
    rr = sorted([r for r in gd["records"] if r["N"] == N and r["E"] == E], key=lambda r: r["seed"])
    rho = [r["rho_eq_last40"] for r in rr]
    return {"R": rr[0]["R"], "B": gd["config"]["B"], "rho0": gd["config"]["RHO_INIT"], "seeds": len(rr),
            "rho_star_mean": _np.mean(rho), "rho_star_std": _np.std(rho), "selection_off_null": gd["nulls"][(N, E)],
            "best_clean_mse_median": _np.median([r["best_clean_mse_traj"][-1] for r in rr]),
            "epochs_run_per_seed": " ".join(str(r["epochs_run"]) for r in rr),
            "runs_plateaued": sum(bool(r["plateaued"]) for r in rr),
            "wall_clock_h_sum_over_seeds": sum(r["elapsed"] for r in rr) / 3600}


cells_g = _mcsv("population_elitism_grid_cells.csv")
_check_table("population_elitism_grid_cells.csv", cells_g, _gd_cell,
             ["R", "B", "rho0", "seeds", "rho_star_mean", "rho_star_std", "selection_off_null",
              "best_clean_mse_median", "epochs_run_per_seed", "runs_plateaued", "wall_clock_h_sum_over_seeds"])
_grid = {(15, 2): "0.073 +/- 0.013", (15, 4): "0.061 +/- 0.007", (15, 8): "0.081 +/- 0.010",
         (30, 2): "0.051 +/- 0.002", (30, 4): "0.054 +/- 0.001", (30, 8): "0.050 +/- 0.002",
         (60, 2): "0.053 +/- 0.002", (60, 4): "0.053 +/- 0.003", (60, 8): "0.053 +/- 0.006"}
for c in cells_g:
    got = _gd_cell(c); key = (int(c["N"]), int(c["E"]))
    val = f"{got['rho_star_mean']:.3f} +/- {got['rho_star_std']:.3f}"
    _claim(f"App D: grid rho* at N = {key[0]}, E = {key[1]}, R = {got['R']}", val, _grid[key],
           got["seeds"] == 3 and val == _grid[key])
_claim("App D: grid selection-off nulls", _span(list(gd["nulls"].values())), "0.101-0.104",
       _span(list(gd["nulls"].values())) == "0.101-0.104")
ep_g = [r["epochs_run"] for r in gd["records"]]
_claim("App D: grid runs meeting the plateau rule; epochs", f"{sum(bool(r['plateaued']) for r in gd['records'])} of "
       f"{len(ep_g)}; {min(ep_g)}-{max(ep_g)}", "27 of 27; 120-170",
       len(ep_g) == 27 and all(r["plateaued"] for r in gd["records"]) and (min(ep_g), max(ep_g)) == (120, 170))

# ---- Appendix G and the Section 4.2 wall-clock times ------------------------------
tb = {p: _rload(f"tabular_{p}_results.pkl")["records"] for p in ("table1", "validation")}
_tb = lambda p, ds, m: sorted([r for r in tb[p] if r["dataset"] == ds and r["method"] == m], key=lambda r: r["seed"])
_METHODS = ("Random", "Magnitude", "IMP", "SNIP", "SynFlow", "EvoluCMAES")
_FURTHER = ("seeds", "glass", "ionosphere", "sonar")


def _tb_row(proto):
    def f(row):
        rr = _tb(proto, row["dataset_key"], row["method"]); ev = row["method"] == "EvoluCMAES"
        a = [r["acc_test"] for r in rr]; t = [r["elapsed"] for r in rr]
        sel = [r["acc_sel"] for r in rr] if ev and proto == "validation" else None
        return {"network": str(rr[0]["dims"]), "seeds": len(rr), "test_acc_mean": _np.mean(a), "test_acc_std": _np.std(a),
                "density_mean": _np.mean([r["density"] for r in rr]), "wall_clock_s_per_run_mean": _np.mean(t),
                "wall_clock_s_per_run_std": _np.std(t), "worker_processes": rr[0]["n_jobs"] if ev else 1,
                "epochs_run_per_seed": " ".join(str(r["epochs_run"]) for r in rr) if ev else "",
                "selection_split_acc_mean": _np.mean(sel) if sel else "",
                "selection_split_acc_std": _np.std(sel) if sel else ""}
    return f


for proto in ("table1", "validation"):
    _check_table(f"tabular_benchmarks_{proto}_protocol.csv", _rcsv(f"tabular_benchmarks_{proto}_protocol.csv"),
                 _tb_row(proto), ["network", "seeds", "test_acc_mean", "test_acc_std", "density_mean",
                                  "wall_clock_s_per_run_mean", "wall_clock_s_per_run_std", "worker_processes",
                                  "epochs_run_per_seed", "selection_split_acc_mean", "selection_split_acc_std"])
acc = lambda p, ds, m: [r["acc_test"] for r in _tb(p, ds, m)]
_appg = {"seeds": "0.543 0.857 0.860 0.838 0.867 0.883 0.975", "glass": "0.428 0.612 0.662 0.668 0.662 0.662 0.732",
         "ionosphere": "0.674 0.919 0.902 0.919 0.913 0.896 0.970", "sonar": "0.692 0.825 0.733 0.775 0.816 0.727 0.902"}
for ds in _FURTHER:
    got = " ".join(f"{_np.mean(acc('validation', ds, m)):.3f}" for m in _METHODS) + \
          f" {_np.mean(acc('table1', ds, 'EvoluCMAES')):.3f}"
    same_bl = all(_np.allclose(acc("validation", ds, m), acc("table1", ds, m)) for m in _METHODS[:-1])
    _claim(f"App G: test accuracy, {ds} (5 baselines, EvoluCMAES val / Table 1)", got, _appg[ds],
           got == _appg[ds] and same_bl)
means = [_np.mean([_np.mean(acc("validation", ds, m)) for ds in _FURTHER]) for m in _METHODS] + \
        [_np.mean([_np.mean(acc("table1", ds, "EvoluCMAES")) for ds in _FURTHER])]
got = " ".join(f"{x:.3f}" for x in means)
_claim("App G: mean over the four datasets", got, "0.584 0.803 0.789 0.800 0.814 0.792 0.895",
       got == "0.584 0.803 0.789 0.800 0.814 0.792 0.895")
got = " ".join(f"{_np.mean([r['acc_sel'] for r in _tb('validation', ds, 'EvoluCMAES')]):.3f}" for ds in _FURTHER)
_claim("App G: EvoluCMAES accuracy on its selection split", got, "1.000 0.707 0.988 0.952",
       got == "1.000 0.707 0.988 0.952")
for proto, paper in (("table1", "24.9 29.0 37.2 43.1"), ("validation", "16.6 16.5 24.5 36.4")):
    got = " ".join(f"{_np.mean([r['elapsed'] for r in _tb(proto, ds, 'EvoluCMAES')]):.1f}" for ds in _FURTHER)
    _claim(f"App G: EvoluCMAES wall-clock per run (s), {proto} protocol", got, paper, got == paper)
bl = [_np.mean([r["elapsed"] for r in _tb(p, ds, m)]) for p in tb for ds in _FURTHER for m in _METHODS[:-1]]
_claim("App G: baseline wall-clock per run (s)", _span(bl, 1), "0.1-0.5", _span(bl, 1) == "0.1-0.5")
den = [_np.mean([r["density"] for r in _tb(p, ds, m)]) for p in tb for ds in dict.fromkeys(r["dataset"] for r in tb[p])
       for m in _METHODS]
_claim("App G: final density, every method and dataset", _span(den), "0.050-0.054", _span(den) == "0.050-0.054")
got = "; ".join(_pm(acc("table1", ds, "EvoluCMAES")) for ds in ("iris", "wine", "cancer"))
_claim("App G: Table 1 datasets re-run, EvoluCMAES", got, "0.933 +/- 0.000; 0.996 +/- 0.007; 0.986 +/- 0.003",
       got == "0.933 +/- 0.000; 0.996 +/- 0.007; 0.986 +/- 0.003")
_t1name = {"EvoluCMAES": "EvoluPrune"}
_t1ok = [(m, ds) for m in _METHODS for i, ds in enumerate(("iris", "wine", "cancer"))
         if f"{_np.mean(acc('table1', ds, m)):.3f} {_np.std(acc('table1', ds, m)):.3f}"
         == "{:.3f} {:.3f}".format(*PAPER_T1[_t1name.get(m, m)][i])]
_claim("App G: Table 1 cells reproduced by the re-run (mean and std)", f"{len(_t1ok)} of 18", "17 of 18",
       len(_t1ok) == 17 and ("IMP", "cancer") not in _t1ok)
got = f"{_np.mean(acc('table1', 'cancer', 'IMP')):.3f}"
_claim("App G: IMP on Cancer in the re-run (Table 1: 0.951)", got, "0.952", got == "0.952")
got = " ".join(f"{_np.mean([r['elapsed'] for r in _tb('table1', ds, 'EvoluCMAES')]):.1f}" for ds in ("iris", "wine", "cancer"))
_claim("Sec 4.2: EvoluCMAES wall-clock per run (s), Iris / Wine / Cancer", got, "19.5 22.2 30.8", got == "19.5 22.2 30.8")
bl3 = [_np.mean([r["elapsed"] for r in _tb("table1", ds, m)]) for ds in ("iris", "wine", "cancer") for m in _METHODS[:-1]]
_claim("Sec 4.2: baseline wall-clock per run (s), Table 1 datasets", _span(bl3, 1), "0.1-0.6", _span(bl3, 1) == "0.1-0.6")

# ---- Appendix L: compute (results/compute_hours.csv) ------------------------------
ch = _rcsv("compute_hours.csv")
_tot = [i for i, r in enumerate(ch) if r["experiment"].startswith("TOTAL")]
blk_a, blk_b = ch[:_tot[0]], ch[_tot[0] + 1:_tot[1]]
_sum = lambda rr, k: sum(float(r[k]) for r in rr)
_ok_tot = all(abs(_sum(b, k) - float(ch[i][k])) < 0.051 for b, i in ((blk_a, _tot[0]), (blk_b, _tot[1]))
              for k in ("wall_clock_h_sum_of_run_times", "core_h_wall_x_workers"))
# hours are stored to two decimals, so hours x workers matches the core-hours to within workers x 0.005
_ok_rows = all(r["worker_processes"].strip() == "8 / 1" or abs(float(r["wall_clock_h_sum_of_run_times"]) *
               float(r["worker_processes"]) - float(r["core_h_wall_x_workers"]))
               <= 0.005 * float(r["worker_processes"]) + 0.05 for r in ch if r not in [ch[i] for i in _tot])
_claim("App L: compute_hours.csv rows and totals add up", str(_ok_tot and _ok_rows), "True", _ok_tot and _ok_rows)
vg_row = [r for r in blk_a if r["experiment"].startswith("Variance grid")]
h_a = _sum(blk_a, "wall_clock_h_sum_of_run_times") - _sum(vg_row, "wall_clock_h_sum_of_run_times")
c_a = _sum(blk_a, "core_h_wall_x_workers") - _sum(vg_row, "core_h_wall_x_workers")
got = f"{h_a:.1f} h, {5 * round(c_a / 5):.0f} core-h; grid {_sum(vg_row, 'wall_clock_h_sum_of_run_times'):.1f} h, " \
      f"{_sum(vg_row, 'core_h_wall_x_workers'):.0f} core-h"
_claim("App L: Section 4.1 and Appendices B-D (without; variance grid)", got,
       "40.8 h, about 605 core-h; grid 0.7 h, about 10 core-h", got == "40.8 h, 605 core-h; grid 0.7 h, 10 core-h")
mn = [r for r in blk_b if r["experiment"].startswith("MNIST 60-epoch sweep")]
gr = [r for r in blk_b if r["experiment"].startswith("Population x elitism grid")]
got = f"{_sum(blk_b, 'wall_clock_h_sum_of_run_times'):.0f} h, {_sum(blk_b, 'core_h_wall_x_workers'):.0f} core-h; " \
      f"sweep {_sum(mn, 'wall_clock_h_sum_of_run_times'):.1f} h, grid {_sum(gr, 'wall_clock_h_sum_of_run_times'):.1f} h"
_claim("App L: runs added for the camera-ready version", got, "45 h, 354 core-h; sweep 22.4 h, grid 19.7 h",
       got == "45 h, 354 core-h; sweep 22.4 h, grid 19.7 h")

print(f"\n  Checks of the new sections: {'all passed' if not _mech_fail else f'{len(_mech_fail)} FAILED'}")
for msg in _mech_fail:
    print("    -", msg)

N_FAIL += len(_mech_fail)

print('\n' + '=' * 78)
print(f'  DONE: {N_FAIL} mismatch(es)')
print('=' * 78)
sys.exit(N_FAIL)
