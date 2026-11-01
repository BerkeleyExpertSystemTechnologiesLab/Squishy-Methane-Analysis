# Scan-order study: a runnable P0-V example

## What this branch adds

Branch `shulun/scan-order-study` derives directly from the team's
`plume_scanner_new`, software release baseline
`5cac633f47001d3aa455f8b6f3cc00b11c550ed3`. This is also the historical
experiment baseline; it was re-fetched for this release. It does not derive from
`source_prior_fixed_evidence`. That branch's README organization is a delivery
reference only; none of its research implementation is copied.

The team already supplied the physical scanner compiler, signal estimator,
evidence gate, acquisition messages, synthetic plume generator, causal replay,
and baseline quickstart. Shulun's added research code compiles directed complete
columns, audits/replays those routes, saves a common READY state, generates P0-V
paired synthetic worlds, and compares fixed routes with paid planning delay.
The portable adapter and CLI in this release expose the preselected `b00_L`
comparison. B remains the default. L_up is called only through research code.

The six imported research modules are exact copies from local producing/delivery
commit `f888fd4a4fcd56ac3638937dbf94443e8795ecae`; their hashes and source
roles are in [scan-order-provenance.json](scan-order-provenance.json).
No historical local Git commits are transplanted. No external implementation
is imported; mathematical references are retained in the routing modules.

## Install and run

Requires an installed Python 3.10+ on PATH (`python --version` should show Python
3). Windows release validation uses Python 3.13.5. If the Windows launcher
registers only an older Python, use the installed Python 3 executable instead
of `py -3`. No GPU, private data, cloud service, footage or old virtual environment
is needed. Run these commands in a new directory:

```powershell
git clone --filter=blob:none --sparse --single-branch --branch shulun/scan-order-study https://github.com/BerkeleyExpertSystemTechnologiesLab/Squishy-Methane-Analysis.git
cd Squishy-Methane-Analysis
git sparse-checkout set PlumeScanner
cd PlumeScanner
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts/run-scan-order-demo.py
```

On macOS/Linux, use `python3 -m venv .venv` and `.venv/bin/python` for the
interpreter path; the remaining arguments are the same. Activation is optional.
Those platforms have not been run in this release verification.

The existing scanner quickstart is unchanged:

```powershell
.\.venv\Scripts\python.exe scripts/run-scanner-demo.py
```

## What to expect

The terminal prints `b00_L`, the shared simulation window, both column orders and
directions, each paid action duration and reading count, each reconstruction MSE,
and `gain = MSE(B) - MSE(L_up)`. A positive gain means L_up has lower MSE; zero
means a tie; a negative gain favors B. Values are computed, not loaded from a
result table. This one example is not expected to reproduce the historical
study's mean percentage improvement.

`outputs/scan-order-demo/` contains `RESULTS.json`, both `*_route.json` files,
both `*_readings.json` files, `PREFIX.json`, `DEMO_CONFIG.json` and
`evaluation.npz` (common times, duration weights and both causal MSE series).
Routes remain editable JSON. Coordinates are normalized `(y,x)`, with x to the
right and y downward. L_up visits columns from left to right and scans each
upward (decreasing y); it is not a calibrated pan-tilt command.

The command preserves existing output. To repeat, use a fresh directory:

```powershell
.\.venv\Scripts\python.exe scripts/run-scan-order-demo.py --output outputs/scan-order-demo-2
```

## Frozen measurement and scoring boundary

`configs/scan-order-study/demo.json` selects the first left-source P0-V base
(`b00_L`, seed 2136000) before any result is inspected. The original `make_pairs`
generator and team configuration build the world; no selection by gain occurs.
The original complete FULL creates the same observation history, posterior,
detector state, pose and requested region for both routes. Each route then
uses the original synthetic source at its own actual positions and exposure
times through `FakeScanner`. Readings are not reordered from a saved scan.

Original declared planning fees are B=0.13 s and L_up=0.14 s, carried from the
frozen P0-V tariff. They consume the original **8 s action budget** before real
recompilation; fallback/rejection is still charged. These are simulation inputs,
not current-machine profiling, instrument delays or latency guarantees. The
world continues through that delay. Source truth is used only by simulation and
offline evaluation, never passed into route selection or the estimator.

Both routes are evaluated over the same **12 s from the original decision**,
including planning, readout and post-receipt idle. Only available readings enter
the causal estimator. The time-weighted whole-field MSE uses the original
32x48 uniform field projection and 0.1 s midpoint integration split at the
original four B/M/F/C routes' exposure-end/receipt events. M/R_up are compiled
only as integration-grid references; only B/L_up receive fresh scan executions
and MSE evaluation in this demo. Neither route changes the selected region,
estimator or gate.
The output unit is synthetic signal squared, not ppm or kg/h. Compute wall time
is reported separately and is not added to the simulation clock.

## Tests and limitations

Run all added tests and the relevant inherited scanner regressions:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_scan_order_demo.py tests/test_c2_r2.py tests/test_c2_ordered_column_scan.py tests/test_scanner_demo.py tests/test_sensing_core.py tests/test_evidence_gate.py tests/test_scanner_calibration.py tests/test_sensing_adapters.py
```

For focused iteration:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_scan_order_demo.py tests/test_c2_r2.py tests/test_c2_ordered_column_scan.py
```

Checks cover physical-plan parity, route audit/recompile, fresh route-specific
sampling, strict public-call prefix replay, paid time, budget rejection, whole
window integration and output preservation. No test requires private logs or
developer paths. Baseline controller/gate/estimator/configuration files remain
unchanged. Numerical values can vary slightly with dependency/platform versions.

The complete inherited suite can also be run with `python -m pytest -q`.
On environments permitting file symlinks, the baseline test
`test_v2_checker_rejects_symlinked_evidence_snapshot` in
`tests/test_recorded_qualification.py` references an undefined `report` after
its symlink check and raises `NameError`. This exact test is already present
at baseline `5cac633`; this branch does not edit or suppress it. On Windows
environments without file-symlink capability, that and the other platform
symlink tests skip normally. The release acceptance command above exercises
the added research code and relevant baseline scanner behavior independently
of that unrelated recorded-data qualification test. If pytest cannot access
the system temporary directory, add `--basetemp outputs/pytest-local` (use a
fresh output path when preserving previous verification artifacts).

Release verification on Windows / Python 3.13.5: the acceptance command above
passes **159 tests**; the original scanner quickstart completes with public
replay passed (one FULL, two FOCUS, explicit terminal budget shortfall).
The complete suite in a symlink-capable environment produced 652 passes and
the one inherited `NameError` described above. Dependencies used were
NumPy 2.5.4, SciPy 1.18.1, headless OpenCV 5.0.0.93 and pytest 9.2.0.
This is software verification, not a repeat of the full historical study.

**Demo versus full study:** this release supplies a runnable single-case demo,
not a complete reproduction command for the historical 24-base, two-layout,
four-rule, noise-repeat study. Although its deterministic paired-world generator
is included as a dependency, the full study's freeze/profiling/numerical-control/
consumer orchestration and historical output bundle are deliberately not shipped.
Do not interpret running this demo as repeating that study, obtaining its average
gain, or validating any held-out or field claim.

This source-anchored synthetic single-action comparison does not establish
unknown-source discovery, general route superiority, closed-loop recovery,
end-to-end real-time benefit, calibrated uncertainty or hardware performance.
It is an opt-in research example; no default promotion is proposed.
