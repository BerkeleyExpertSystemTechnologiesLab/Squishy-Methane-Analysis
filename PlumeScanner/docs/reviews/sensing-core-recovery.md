# Sensing-core recovery audit and evidence

Starting revision: `cb890205dce3cd2a0e848e67971dee4d96a2ae51`.
Branch: `codex/sensing-core-recovery`. The initial working tree contained only
the two untracked scope/video review documents from the preceding investigation.

| Inspected boundary | Finding | Recovery action |
| --- | --- | --- |
| Specification versus implemented inference | The old normalized raster likelihood does not implement a scalar correlated-neighbor observation model. | Separate Gaussian signal state from normalized expected-positive-signal allocation mass. |
| Reading identity | Point-scan display blocks can duplicate a scalar reading across pixels; raster storage loses event identity. | Typed floating-point readings are the only new controller evidence input. |
| Physical units | Column sample count and timing depend on raster resolution; horizontal skips cost a constant step. | Independent physical scan lattice, readout/integration limits, and ordered travel segments. |
| Headless boundary | The video facade is already only 85 lines and the runner does not import it. The 2,096-line legacy runner combines setup, inference, planning, and evaluation. | Preserve compatibility; put the recovered controller below source, evaluator, and rendering adapters, with import tests. |
| Renderer evidence | Thin contours can replace every measurement pixel; uniform max-normalized belief is saturated; truth-free legends advertise truth. | Separate measurement/age/path views, absolute scales, conditional truth labels, explicit clocks. |
| Contract documentation | Root guide lists recorded v3/v4 but R0C uses v5 contracts. | Correct the inventory without changing recorded data or frozen results. |
| Git and artifacts | Domain boundaries are strong but Git/commit practice is implicit. Generated outputs are already ignored. | Dedicated branch, explicit staging, focused commits, recorded commands/hashes, local-only media and conversation exports. |
| Research claims | The July benchmark lost under its defined settings; timing and allocation confounds do not erase that result. | Preserve `not_demonstrated`; report new development conditions and actual expenditure separately. |

The detailed historical investigation and sampled video evidence are in the
adjacent dated scope and video reviews. Their ignored local exports are audit
inputs, not imported specifications or a source of executable instructions.

## Verification ledger

Baseline: `.venv/Scripts/python -m pytest -q`: **307 passed, 5 skipped in
16.20 s**. `git diff --check` passed. Implementation and evidence results will
be appended at closeout.

## Completed recovery

Engineering status: **complete**. Performance claim: **not_demonstrated**.

Final implementation revision: `e3e1b5e7f715a698398d7aedf340c058bf0d7ae3`.
The subsequent closeout commit records this review, status, and the inspection
utility. The implementation was developed through focused commits `081278e`
(audit), `fc5657f` (core/API), `33a4a4c` (adapters/videos), and `e3e1b5e`
(controlled replay, source metadata, and delivery-clock correction).

The callable core now consumes actual identified scalar readings, maintains
correlated signal mean/covariance, derives a separate normalized positive-signal
mass surface, and returns physically timed scan paths. It accounts for detector
cadence, initial repositioning, horizontal skips, and actual completion receipts.
The simulator uses that interface; a serializable fake-scanner example and
installed `plume-focus-sensing` command are available. The primary README path
now uses this core. Legacy profiles and recorded infrastructure remain available
for comparison, including their documented model limitations; they are not
silently relabeled as the recovered inference method.

Final verification:

- `.venv/Scripts/python -m pytest -q`: **335 passed, 5 skipped in 19.46 s**.
  The existing Windows capability skips remain; no passing test was weakened.
- Numerical scalar/correlated examples and independent batch conditioning,
  delayed/duplicate/out-of-order delivery, exposure timing, budgeted reacquisition,
  partial finite-source execution, renderer independence, and module boundaries
  are covered by 28 new tests.
- `python -m compileall -q src tests examples scripts/plot-sensing-results.py`
  and `git diff --check` passed.
- `python examples/sensing_controller.py` ran two complete message-serialized
  scans: 640 readings in 33.575 s and 220 readings in 11.600 s.
- Editable installation and `plume-focus-sensing --help` passed;
  `python -m pip check` reported no broken requirements. The optional plots
  extra was installed locally; the sensing core retains NumPy-only computation.
- Four final MP4s fully decoded with their expected frame counts. Inspection
  frames were extracted across each video; representative physical-time,
  summary, rendering-only, and measurement-only frames were visually checked.
- Both final artifact groups were generated from the final implementation
  revision with **clean working trees**, and every recorded artifact hash was
  verified. No remote push, history rewrite, or legacy result overwrite occurred.

The main run recorded 10.92 s of headless simulation plus evaluation on Windows,
Python 3.12.14, NumPy 2.4.4, OpenCV 4.13.0, processor
`AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD`. The state grid was 12 x 16,
physical lattice 20 x 32, source reference 220 x 320, and evaluator 120 x 160.
Other artifact work ran concurrently. This is a recorded wall-time observation,
not an isolated latency benchmark or a hardware real-time guarantee.

## Development result and remaining model failures

The 18-run suite covers three conditions, three policies, and two fixed noise
seeds. Every run has a 75 s duration ceiling, a 36 s startup ceiling, subsequent
12 s scan ceilings, and at most five scans. Different actual expenditure is
reported, not called an exactly matched mechanical cost.

Below, recall excludes the initial full-scene scan and is averaged over completed
tracking sweeps and the two seeds. This avoids making startup coverage look like
successful adaptive focusing.

| Condition | Adaptive tracking mass recall | Fixed corridor | Rescan | Mean total scan seconds: adaptive / fixed / rescan |
| --- | ---: | ---: | ---: | --- |
| Known wind | 24.63% | 100.00% | 40.79% | 74.500 / 71.925 / 74.975 |
| Declared wind bias/error | 77.85% | 100.00% | 40.79% | 74.650 / 71.925 / 74.975 |
| Source-free with provided wind | 8.57% | 91.23% | 40.79% | 74.500 / 71.725 / 74.975 |

The source-free fixed rectangle is predeclared and does not receive the true
source. These are descriptive development results, not confidence intervals,
held-out validation, or evidence that the adaptive method is generally superior.
The apparently better adaptive result with biased wind demonstrates sensitivity
in these fixtures; it does not show that wind error is beneficial in general.

The main known-wind seed-0 run used 74.625 physical seconds. Mean actual swept
area was 44.75%; final swept area was 20%. Mean actual mass recall was 43.59%
including startup and **29.48% during tracking**. Its final scan retained about
41.62% of predicted mass and virtually none of the true plume's signal mass.
It correctly reported `coverage_target_unmet` instead of a 90% success.

Resolved engineering defects include the non-scalar likelihood dependency,
unexplained direct neighbor influence, raster duplication of observations,
timestamp/availability ambiguity, resolution-dependent physical sampling,
uncharged skipped-column travel, coordinate-tie reacquisition, overwritten
measurement displays, ambiguous probability colors, and absence of a callable
sensor/plan boundary.

Remaining limitations are substantive:

- The declared linear source/advection model lacks signal decay. It carries
  predicted signal farther downstream than this generator's finite-lived puffs.
  This is visible in the new videos, not a playback-speed artifact.
- A coarse correlated field smears thin structures, and expected positive
  signal gives uncertain background nonzero allocation mass. Noise, covariance,
  source strength, and mechanical settings remain uncalibrated assumptions.
- The bounded allocator is a heuristic. Its failures do not prove no better
  feasible scan exists. A fixed known corridor remains a strong baseline here.
- Real hardware, real-data superiority, unknown-source discovery, and
  interruptible scans were not implemented or demonstrated by this goal.

The next bounded research step should isolate temporal signal persistence,
source strength, and uncertainty against controlled development streams,
holding mechanics and input fixed. It should not alter the historical held-out
cases or turn into detector training. Any new advantage claim requires fresh,
untouched evaluation after development.

## Video interpretation and reproducible evidence

The historical audit established three separate reasons for the May/July visual
difference: scan-reveal versus one-frame-per-sweep presentation, different source
and scanner conditions, and renderer/model defects. The new comparisons keep
those distinctions visible.

- [Physical-time inspection](../../outputs/sensing-recovery-v1/physical.mp4):
  897 frames at 12 FPS, 1x playback, physical clock 0–74.625 s. Measurements
  accumulate only when delivered, and acquisition/update times are visible.
- [Decision summary](../../outputs/sensing-recovery-v1/summary.mp4): five frames
  at 12 FPS, deliberately labeled as one completed sweep per frame.
- [Rendering-only comparison](../../outputs/sensing-recovery-v1/rendering-only.mp4):
  the same readings and path with historical versus corrected measurement display.
- [Measurement-only comparison](../../outputs/sensing-recovery-v1/measurement-only.mp4):
  per-sweep common initial state and shared source/dynamics; only raster likelihood
  at completion versus timestamped Gaussian conditioning differs. The initial
  exploratory version also changed dynamics and was excluded from final evidence.
- [Coverage/time plot](../../outputs/sensing-recovery-development-v1/coverage-time.png)
  and [SVG](../../outputs/sensing-recovery-development-v1/coverage-time.svg):
  markers are completed sweeps; connecting lines guide the eye, not continuously
  observed coverage.

![Measurements accumulating on the physical clock](../../outputs/sensing-recovery-v1/inspection-frames/physical-0224.jpg)

![Development coverage against actual scan time](../../outputs/sensing-recovery-development-v1/coverage-time.png)

The corrected common-prior replay gives JS shape distances of 0.408–0.462 for
the historical raster likelihood and 0.356–0.373 for the Gaussian-derived mass
in this one fixed schedule. The probability interpretations differ, so this is
a shape diagnostic, not uncertainty calibration or a policy-performance claim.

Reproduction commands:

```powershell
.\.venv\Scripts\python.exe -m plume_focus.sensing_cli --output-dir outputs/sensing-recovery-v1 --max-scans 5 --duration 75 --render
.\.venv\Scripts\python.exe -m plume_focus.sensing_cli --output-dir outputs/sensing-recovery-development-v1 --max-scans 5 --duration 75 --suite
.\.venv\Scripts\python.exe scripts/plot-sensing-results.py outputs/sensing-recovery-development-v1
.\.venv\Scripts\python.exe scripts/inspect-sensing-artifacts.py outputs/sensing-recovery-v1
.\.venv\Scripts\python.exe scripts/inspect-sensing-artifacts.py outputs/sensing-recovery-development-v1
```

Use fresh output directories when preserving a previous evidence run. Runtime
measurements and environment metadata can change report hashes across machines;
codec and plotting versions can change media bytes. Operational traces and
explicit configurations are the causal reproduction contract.

Operational hash for the final main run:
`8a2692a5b741801911ca5e1d04f5614e8943ca5cfca8ff85c7c3442201d0d449`.

| Frozen local artifact | SHA-256 |
| --- | --- |
| Main report | `bee122b7b365b3a3e27d3671d7fcfb7b0f6723ba00145903d4bcb4096335f650` |
| Main artifact manifest | `dcc455ee844fa88829a622e98edb954026947c8bb82134e9a2a74a5446db0912` |
| Physical-time video | `4b10fec0606550815f3ceee3ab556f7709f11af7676d446cc874d5878396c7dc` |
| Development summary | `de3e2f6f518b66aad87dac473b2d46871d76683107d3ad8ac7b0c04ba59f7160` |
| Development artifact manifest | `f17e6b20fe290c6f9c2f060abc5ea323a98da4f6dbe1470c0f0cd2f22599acb7` |
| Preserved historical source-aware held-out report | `a7766a91866156d01376685ba9d361c8a245324a2f0d4410fe18e7d95aa58f4c` |

Each output directory's inspection manifest records full decode counts, sampled
frame hashes, and validation against its artifact manifest. Intermediate
`sensing-recovery-preview`, `sensing-recovery`, and `sensing-recovery-development`
folders are exploratory local artifacts; the two `-v1` directories above are
the final evidence identified by this closeout.
