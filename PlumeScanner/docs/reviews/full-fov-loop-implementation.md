# Lean full-FoV loop implementation

Status: complete, 2026-09-16. Approved plan: [full-FoV loop](full-fov-loop-plan.md).
Start `84cf70d`; branch `codex/full-fov-loop-plan`. Runtime checkpoint `90fba10`,
observer/API checkpoint `da531b6`. Both checkpoints were pushed to the explicitly
authorized GitHub remote. Generated traces/media remain local; preexisting
untracked `output/` is preserved.

## What changed

The intended operating sequence is now implemented independently of historical
allocation experiments: mandatory complete full-FoV startup, then supported
signal → focus, otherwise return home → another complete full scan. A fresh
coverage obligation is created on every full rescan. Only actual readings clear
cells; partial execution cannot unlock focus. Home motion, detector timing and
receipt latency are charged. If the remaining session cannot complete the full
scan, the result says so instead of substituting age blocks.

The current inferred mean must exceed .15 to focus. The region uses the existing
Gaussian .95 upper signal quantile, forecast to half the focused time ceiling;
ranked prefixes retain mechanical closure and a four-reading minimum. Focus has
a 12 s ceiling; full scans use actual compiled time. The field can expand, move,
stay or shrink. There is no recursive cropping, 90% gate, prior-startup bypass,
periodic service detour, covariance inflation or belief reset.

The Gaussian estimator, zero-inflow transport, source assumptions, process noise
and physical profiles remain fixed. This is not an estimator calibration round.

## Lean representation and compatibility

`FullFovSession` and `FullFovPlan` are independent minimal messages, rather than
subclasses carrying historical mass/risk/service settings. `FullFovController`
owns the decision and full-scan state. Shared `AcquisitionController` owns causal
evidence/lifecycle handling. The neutral physical compiler owns motion and
acquisition costs; the old compiler wraps exactly that geometry for historical
messages. Historical policy dispatch is selected once in `HistoricalPolicy`.

Inference/planning do not import simulation, rendering, filesystem access or
evaluator truth. The adapter records public calls and separately evaluates
reconstruction. The [implementation guide](../sensing-core.md) maps modules and
provides commands; `examples/full_fov_scanner.py` shows two API cycles.

This reduces coupling in the active runtime, not the total repository line
count. Historical implementations, regression tests and experiment evidence
remain intentionally available. No claim of real-time compute feasibility is made.
The simulator still reuses the historical synthetic-source configuration, including
legacy generator fields. Those are outside the controller; the explicit instrument
profile alone controls new-loop acquisition. This round does not rewrite that
generator or strip fields from the frozen source manifests.

## Declared mechanism and noise study

Local root: `outputs/lean-full-fov/`. The declaration freezes 18 seed-0 runs:
coarse/doubled profiles × continuous/blank/cessation × noise 0/.01/.05. Every run
starts at time zero, lasts 360 s including final idle time, and requires complete
startup. The first full scan takes 33.575 s / 129.5875 s respectively. Evaluation
uses the existing 120×160 spatial mesh and .05 s spacing; losses are descriptive,
not a newly refined superiority comparison.

Simulated noise and assumed likelihood noise are explicitly separate. Assumed
standard deviation stays .05. Zero-noise readings equal clean exposure-integrated
responses exactly. The blank case uses source-free inference and zero signal;
the cessation case removes truth at 180 s while leaving source-anchored inference
unchanged. This abrupt control is not a physical source-shutdown model.

Zero-noise results (all six public-call/causal-loss replays passed):

| Profile/case | Complete full scans | Focus scans | J/T |
| --- | ---: | ---: | ---: |
| Coarse continuous | 1 | 62 | .01503319 |
| Coarse blank | 10 | 0 | 0 |
| Coarse cessation | 5 | 33 | .00871271 |
| Doubled continuous | 1 | 20 | .01703467 |
| Doubled blank | 2 | 0 | 0 |
| Doubled cessation | 2 | 6 | .00984581 |

These runs demonstrate the required transitions, including cessation returning
to full scanning. They do not establish that selected focused measurements are
optimal. The zero-noise continuous views still show a long estimated tail beyond
the simulated compact plume. That is distinct from added detector noise and
remains model/calibration work; the new loop does not hide it.

Cessation is not fully resolved: the unchanged source-anchored estimator rebuilds
false signal during full scans. In the coarse zero-noise control it falls below
the gate at 187.7 s, completes a full rescan, then rises to a .254 maximum mean
and focuses again at 222.0 s; this alternation repeats. The doubled control
returns to full scanning at 189.3 s and refocuses at 320.2 s with a .296 maximum
mean. Truth has been zero since 180 s in both. The loop follows its inferred
signal; fixing false support requires separately justified estimator work.

All 18 executions and exact public-call/causal-loss replays passed. Noise-stage losses:

| Profile/case | J/T, noise .01 | J/T, noise .05 | Charged seconds, noise 0/.01/.05 |
| --- | ---: | ---: | --- |
| Coarse continuous | .01503967 | .01506416 | 359.925 / 359.925 / 359.975 |
| Coarse blank | .000000931 | .000023268 | 349.925 / 349.925 / 349.925 |
| Coarse cessation | .00872600 | .00877314 | 343.125 / 343.725 / 344.825 |
| Doubled continuous | .01703440 | .01706693 | 359.9375 / 359.9375 / 359.9125 |
| Doubled blank | .000000772 | .000019302 | 260.7625 / 260.7625 / 260.7625 |
| Doubled cessation | .00985025 | .00992806 | 332.0875 / 332.0875 / 332.1375 |

Full/focus counts match the zero-noise table across all noise levels, although
paths and spending can differ. Added-noise RMS is .009914–.010118 for declared
.01 and .049862–.050405 for declared .05; it is exactly zero in zero-noise runs.
All blank cases have zero focused scans. Both cessation profiles retain false
re-focusing (five coarse / one doubled post-180 s focus starts at each noise
level). Nonblank loss changes are below 1% relative to zero noise in this seed.
This suggests injected measurement noise is not the main explanation for these
visible failures; it is not a general noise-calibration claim. Common horizons
and ceilings are not exactly matched expenditure or an adaptive-superiority test.
The time-averaged signed enhancement-index error is positive in all nonblank
runs (.001105–.006978 in synthetic retrieved-scalar units), consistent with
retained excess signal. The largest absolute blank index error is .000257.
These are field-index errors, not calibrated emission-rate errors. Exact per-run
values and configuration/execution hashes are in the verification JSON.

## Verification and inspection

Focused tests cover mandatory startup despite a positive prior, strict gate
crossings, fresh rescan coverage, partial home return, missing evidence, charged
timing, delayed-batch replay, atomic duplicate conflicts, actual overruns,
unchanged active paths, expanding/moving regions, serialization and public API
replay. Renderer checks use supplied causal state and preserve measured panels
and operational state across playback settings.

Exact plan/posterior hashes match `84cf70d` over two-cycle arithmetic fixtures for
legacy, v1, v2, zero-motion v3, v4, zero-motion v4, v5 and zero-motion v5. Full
regression passes 489 tests with five existing Windows file/directory-symlink
skips; compilation, import-boundary
and diff checks pass. Verification logs are recorded locally.

Physical-time videos use 1× source time, separate compressed videos one frame per
completed scan. Existing six-panel layout is preserved. Labels distinguish
FULL/FOCUS/RETURN HOME, clean measurements, uncertainty and normalized excess.
The excess panel is diagnostic, not sensor noise or the eligibility surface.
Video writers decode every frame and verify the execution-file hash is unchanged.
Visual review covers representative keyframes; it is not a claim of watching
every encoded frame in real time.

The canonical closeout selection is 14 videos: physical-time (1×, 2 FPS) and
compressed-summary views for the six zero-noise cases plus coarse continuous
noise .05. `inspection/review.json` identifies the manually inspected frames.
Extra 4 FPS/1× and 2 FPS/4× blank variants preserve the exact same execution
hash, with 1,441 / 721 / 181 physical frames across the three observer settings.
Intermediate renders remain local. The terminal footer now labels idle periods
explicitly instead of retaining the previous scan's active mode.

Reproduction commands are in [the callable-core guide](../sensing-core.md).
The canonical observer command is
`python scripts/render-full-fov-demo.py --name NAME --fps 2 --variant reviewed`.
Verification commands used:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check-full-fov-compatibility.py
.\.venv\Scripts\python.exe -m compileall -q src scripts examples
git diff --check
```

The long doubled blank idle interval is included in J/T through 360 s. Its
repeated evaluator forecasts are expensive offline; no truncation or physical
time discount is applied. Recorded `offline_seconds` covers the execution/loss
loop, excluding separately built truth quadrature and media encoding. It is not
a hardware latency benchmark.

## Provenance and closeout

The declaration SHA256 is
`1a3d2d38bbf5ae82d5fd3ba1aa1df5a7b468c7c7900f9ab27c3ec19954ad1a7a`.
The local `outputs/lean-full-fov/artifact-manifest.json` lists 184 artifacts and
the runtime/adapter source hashes; its SHA256 is
`9a138acc4f9e35a82e82b6cf0077b14679be81ad85a06dc5d3cdad95a92e837e`.
The manifest records starting revision `84cf70d`, runtime revision `90fba10`
and verification revision `6b5f796`. Intermediate observer/verification commits
are `da531b6`, `9d506c6` and `d3a80c6`; inference and allocation remained unchanged
through the study. The final documentation revision and matching pushed remote
tip are recorded in local `closeout.json` after the closeout commit.

Resolved: mandatory full startup, fresh full-rescan completion, charged home
return, separation of full-scan/focus budgets, minimal independent policy/plan
messages, shared acquisition/mechanical code, explicit measurement-noise labels,
terminal idle labels, causal replay, and historical behavior compatibility.

## Remaining work and interpretation

Implementation and inspection are complete. Reconstruction superiority and
calibrated bounds remain unproven.
Historical `not_demonstrated` findings remain unchanged. Source/lifetime and
forecast calibration are the smallest next analysis stage after this corrected
loop; changing local drift, hardware validation and networking remain outside it.
