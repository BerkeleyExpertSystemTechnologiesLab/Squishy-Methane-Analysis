# C3: choosing a second scan from the first scan's readings

Shulun Gao | Software/ML

This branch contains an opt-in, finite Bayesian scanning prototype and a small
service interface. The scanner already knows how to execute a physical path and
assimilate timestamped readings. This work asks which path to request next, and
whether the first request's readings should change the second request.

The original scanner, estimator, gate and default quickstart are unchanged.
The native controller has not accepted this service. Hardware timing and an
end-to-end patrol integration remain pending.

## Run one software example

From a fresh checkout, in Windows PowerShell:

```powershell
git clone --filter=blob:none --sparse --single-branch --branch shulun/c3-active-sensing https://github.com/BerkeleyExpertSystemTechnologiesLab/Squishy-Methane-Analysis.git
cd Squishy-Methane-Analysis
git sparse-checkout set PlumeScanner
cd PlumeScanner
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts/run-c3-demo.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_c3_active_sensing.py tests/test_bayes_two_stage.py tests/test_c2_r2.py tests/test_c2_ordered_column_scan.py tests/test_sensing_core.py tests/test_evidence_gate.py tests/test_sensing_adapters.py
```

On macOS/Linux use `python3` to create the environment and `.venv/bin/python`
for subsequent commands. These platforms have not been exercised for this release.
The module entry point `python -m plume_focus.c3_demo` also works after installation.

The example writes `outputs/c3-software-demo/RESULTS.json`. It refuses to overwrite
an existing directory; use `--output outputs/c3-software-demo-2` for another output.
This is one fixed fixture with a disclosed source and fixed simulator seeds,
chosen before the release run. The planner only receives delivered readings,
compiled paths, priors and declared contracts. Simulator truth stays in the demo.

The output contains the real FULL prefix, both requested physical paths,
exposure-specific readings, completion receipts, planning charges, history odds,
new evidence and selection records. No performance percentage is inferred from it.
It uses the existing research-only `R2Counterfactual` executor. The example's 3 s
terminal reservation is illustrative; it neither measures nor executes HOME+FULL.

## What is implemented

`bayes_two_stage.py` preserves the frozen research implementation:

- A finite mixture over source positions, amplitudes, onset times and shared
  old-source/wind nuisance components.
- Moving-footprint Simpson exposure predictions for historical and proposed scans.
- Predictive likelihood before assimilation, with a shared Gaussian offset.
- Small Monte Carlo two-step trees. Adaptive second actions are optimized within
  each hypothetical first-reading branch; an open plan commits one second action.
- A matched-first committed control that isolates use of the first scan's readings.

`c3_active_sensing.py` adds a separate opt-in interface:

- `FullHistoryBank.from_history`: checks reading/exposure identities and availability,
  preserves historical model odds for design, then starts a separate new-evidence
  counter at READY. Histories can span multiple physical plans.
- `GuardedTwoStagePlanner.choose`: refuses service during a committed scan, native
  FULL obligation, disallowed gate or due coverage; reserves declared terminal
  time and compiles from the decision time plus its planning charge.
- `ingest`: assimilates delivered readings as they arrive, without waiting for the
  completion receipt. Readings and exposure slots cannot be counted twice.
- `finish`: requires matching, complete receipts and readings before opening the
  second decision; subsequent decisions use the actual receipt pose and exposure.
- Explicit hypothesis, candidate, forecast and wall-time acceptance limits.
- Optional, explicitly declared entropy stopping and predictive-NLL review limits.
  Neither has a tuned default or a false-alarm guarantee. A model warning waits
  for the committed action's receipt before handing back control.

The native permission and coverage/terminal contract are caller assertions. They
need a real receiving team's adapter; this module does not grant FOCUS, execute
recovery FULL, prove a return duration or interrupt an active path. Wall-time is
checked before returning an action; this is a bounded finite menu, not a hard
real-time interrupt. Model-time planning fees and measured wall time are separate.

## Two posterior conventions, kept separate

The historical study conditions weights *within* H0 and H1 on the full history,
then starts the post-READY comparison with equal H odds. It asks about conditional
new evidence. It is not the full-history posterior over whether a new source exists.

The new interface uses:

`design log odds = historical log odds + post-READY log Bayes factor`.

Historical readings are not counted again as new confirmation evidence. Existing
study results therefore do not validate the new full-history interface's detection
performance. They remain valid records of their original conditional experiment.
No threshold calibrated for the old convention is promoted to a false-alarm
guarantee for a new service or repeated online triggers.

## The result that limits adoption

The completed earlier study used 12 development, 24 calibration and 24 independent
test bases. The 24 test bases contained 12 positives and 12 negatives, with four
paired methods per base; 96 method rows are not 96 independent worlds.

| Test outcome | Fixed two-region F | Adaptive two-step A |
| --- | ---: | ---: |
| Confirmed positives | 7/12 | 7/12 |
| Observed false alarms | 0/12 | 0/12 |
| Restricted mean post-READY confirmation delay | 7.912 s | 9.067 s |
| Common-window field MSE | 0.00643511 | 0.00657093 |

Unconfirmed positives contribute the full 16 s horizon to the delay. The prototype
charged planning within a 12 s action budget. It uses an additive, dimensionless,
steady-after-READY toy kernel, not a validated detector or emission-rate model.
These budgets, estimator and onset conditions differ from original C3 REFRESH;
do not merge their denominators or compare their numerical scores as one study.

A changed the second scan in 14/19 executable test cases. Against its matched-first
committed control, its mean MSE was descriptively lower, but the reported interval
crossed zero. Against F, A gained two confirmations and lost two. A larger source/
onset model also predicted the common development observations less well. This
does not establish a stable feedback benefit or justify replacing the simple rule.

Frozen `historical_TEST_RESULTS.csv`, `historical_ANALYSIS.json`, threshold and
tariff records are copied from that study, not generated by this release demo.
`PROVENANCE.json` records exact source hashes. The original independent artifact
consumer checked joint versus sequential evidence and causal loss integrals;
this release does not rerun its development/calibration/test queue. The release
contains the model, service, fixed CLI and summary tables; it is not a complete
raw-archive or full historical study reproduction package. The release
contains the model, service, fixed CLI and summary tables; it is not a complete
raw-archive or full historical study reproduction package.

**Decision: keep the simple rule and the native scanner defaults.** Share this
branch as reusable research code, not an adopted faster or more accurate controller.

## Verification scope

New tests compare full-history weights with an independently computed joint
Gaussian density, check visibility and duplicate exposure rejection, native
obligations, coverage/terminal reservations, planning charges, streaming ingest,
receipt identity/completeness, actual-pose replanning and matched first actions.
The seven frozen Bayesian tests and relevant C2/scanner/gate tests are retained.

The release acceptance command passed **157 tests** on Windows, Python 3.13.5.
Tested dependency versions are recorded in `configs/c3-active-sensing/constraints.txt`.
The fixed CLI fixture completed two real simulated actions with three readings
each. Its historical posterior was already highly concentrated: this fixture
demonstrates wiring, not improved discovery or a useful second scan. The original
scanner quickstart and source compilation were also checked; see `RELEASE_CHECKS.json`.
If Windows prevents pytest's default temporary-directory creation, choose a new
workspace-local directory with `--basetemp outputs/pytest-c3-check-1`.

The inherited full suite has a known `NameError: report is not defined` in
`tests/test_recorded_qualification.py::test_v2_checker_rejects_symlinked_evidence_snapshot`.
It is unchanged from the baseline. The release acceptance command above is the
scoped check; it does not claim that the entire inherited suite is green.

## Research background

- [Foster et al. (2021), Sec. 3](https://proceedings.mlr.press/v139/foster21a/foster21a.pdf):
  conditional information and adaptive design policies. No DAD network is trained.
- [Hainy et al. (2022), Sec. 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC8924111/):
  model-label entropy and discrimination losses. No classifier method is imported.
- [Ott et al. (2024), Sec. 4](https://arxiv.org/html/2402.08841v2):
  budgeted paths and repeated decisions. Its linear-model bounds do not transfer.
- [Tang et al. (2026)](https://proceedings.mlr.press/v300/tang26d.html):
  model misspecification can undermine information-driven selection. No risk bound
  from that work is claimed for this scanner.

Bayes updates, information gain and two-step planning are existing methods. The
project contribution is the executable comparison, timing/evidence interfaces and
explicit negative result. Further algorithm research should first audit predictive
support and onset/transport mismatch, rather than expand a planner until it wins.
