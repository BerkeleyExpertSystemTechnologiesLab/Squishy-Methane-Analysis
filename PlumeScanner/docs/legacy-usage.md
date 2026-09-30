# Historical usage reference

Preserved from the reviewed checkpoint `1bdf3c6`. These commands include older
APIs and recorded-data workflows. Start with the [current README](../README.md)
for installation and the evidence-gated scanner demo. Historical negative findings
retain their original scope.

---

# Plume Focus

Plume Focus is a CPU-based prototype for mechanically constrained active
sensing:

> Given partial, time-skewed plume measurements and a short-horizon motion
> estimate, which physically feasible region should the scanner measure next?

The eventual objective is **plume reconstruction and strength estimation under
resource constraints**. Adaptive scanning should collect useful coverage before
an intermittent, moving plume changes enough to distort the assembled image.
Shorter focused scans may improve temporal fidelity, but missing signal can
worsen the estimate; the benefit must be tested with a fixed instrument profile.
The motivation is to reduce reliance on conventional OGI and human visual
interpretation. Comparative sensor accuracy, cost, and calibrated strength
estimation remain unproven. See the [scope clarification](../SYSTEM_GOAL.md#eventual-purpose-user-scope-clarification).

The project is a simulation and planning prototype, not a calibrated gas
detector or production safety system.

The persistent delivery goal, completion gates, stage ledger, and evidence
history are maintained in [`SYSTEM_GOAL.md`](../SYSTEM_GOAL.md). `AGENTS.md`
continues to define the project boundaries and development rules.
The actively maintained [`CHANGELOG.md`](../CHANGELOG.md) summarizes broad
checkpoints from the original May work through the v1 implementation,
with implementation status, negative findings, and links to detailed evidence.
The subordinate recorded-data accession, sensor-emulation, and evidence
workflow is tracked in
[`RECORDED_DATA_WORK_PLAN.md`](../RECORDED_DATA_WORK_PLAN.md); it cannot advance
the parent system gates without the evidence required by `SYSTEM_GOAL.md`.

## Frozen formulation and implemented scan policy

The current mechanism demonstration uses the independent
[lean full-FoV loop](../docs/sensing-core.md#lean-full-fov-loop-current-mechanism-demonstration):
complete startup scan, then signal-supported focus, otherwise charged return
home and another complete scan. It retains Gaussian inference and fixed scanner
mechanics. Full scans are not restricted by the 12 s focus ceiling. Historical
v1–v5 modes remain reproducible; this experimental loop is not a new policy freeze.
See `examples/full_fov_scanner.py` for the three-operation API and the guide for
declared zero/.01/.05 simulated-noise demonstrations.
The [implementation review](../docs/reviews/full-fov-loop-implementation.md) records
18 verified runs, 489 passing tests, and the remaining false re-focusing caused
by retained source-anchored inference after signal cessation.

The latest opt-in [evidence-gated candidate](../docs/reviews/evidence-gated-focus.md)
uses delivered readings to permit or retain focus around the calibrated estimator.
Its frozen 104-run study supports reduced false focusing, with mixed reconstruction
results: blank, cessation and intermittency map error worsen. The
[cleanup review](../docs/reviews/evidence-gated-focus-cleanup.md) covers acquisition
timing, resumption scoring and diagnostic/report corrections. Gate bounds describe
measurement error under the declared noise model; they are not calibrated field
confidence. Source-activity inference and physical strength calibration remain open.

The [v1 specification](../docs/specs/reconstruction-policy-v1.md) fixes the immediate
target as a time-referenced map of **inferred beam-integrated concentration
enhancement** and evaluates reconstruction over a common physical horizon.
The first policy retains the 90% expected positive-signal rule, preserves its
selected cells through mechanical column conversion, and reports budget
shortfalls with separately charged reacquisition. Repeated trimming is not a
requirement. Confidence bounds are a diagnostic; alternative allocation remains
experimental.

Every experiment requires an explicit [instrument profile](../docs/specs/instrument-profile-v1.md).
No numerical reference hardware is selected. Beam size, physical pitch, cadence,
integration, and mechanical speeds stay fixed within comparisons; display
resolution and playback are independent.

The explicit v1 implementation now preserves selection and fixed instrument timing;
its [verification and 36-run comparison](../docs/reviews/policy-v1-conformance.md) are
complete. That checkpoint passed 366 tests with five Windows capability skips.
Coarse adaptive runs use age fallback and lose to the fixed corridor; narrow
startup remains incomplete. Legacy recovery commands retain their original
column packing/timing.
See the [worked checks](../docs/specs/freeze-v1-worked-examples.md) and
[original implementation handoff](../docs/reviews/policy-v1-freeze.md).
The current adaptive claim remains `not_demonstrated`.

A subsequent [mechanism diagnosis](../docs/reviews/mechanism-revision.md) tests an
[experimental v2 candidate](../docs/specs/mechanism-grounded-policy-v2.md): reportable
signal above a declared synthetic threshold, short vertical surveillance segments,
and a tentative 40 x 64 physical profile. V1 remains frozen and reproducible.
Its 43-run development study has mixed results and does not establish superiority.
That checkpoint passed 382 tests with five Windows capability skips. The review
includes inspected videos, size/age diagnostics, and unresolved model and budget issues.

The [persistent wind candidate](../docs/specs/persistent-wind-motion-v3.md) now carries
conditional Gaussian fields and learned weights across scans, preserving v2
allocation and mechanics. Its [development review](../docs/reviews/persistent-wind-prediction.md)
records 403 passing tests, 84 completed comparisons and 10 inspected videos.
Hypothesis-count refinement is numerically unresolved. Source-free full-strength
map loss improves, but its enhancement-index MSE worsens; main adaptive mean loss
also worsens. Changing local drift and correcting an inherited boundary defect
remain required model work. No reconstruction superiority or new freeze is claimed.

The [v4 acquisition candidate](../docs/specs/reconstruction-value-policy-v4.md) now
compares expected reconstruction risk across feasible scan patterns instead of
requiring 90% mass. It retains both estimators and fixed mechanics. Its 96-record
comparison and 20-video inspection are complete; 430 tests pass with five existing
Windows skips. Gaussian results are mixed. The fixed mixture remains numerically
unresolved, with doubled biased-wind map loss worsening by about 100%, despite a
27% improvement in the full-strength source-free diagnostic. No general advantage
or real-time feasibility is established. See the [implementation guide](../docs/sensing-core.md) and
[completed review](../docs/reviews/reconstruction-value-policy-v4.md). Historical
negative findings and the model limitations above remain intact.

## Primary sensing demonstration and callable API

The explicit frozen-v1 entry points are:

```powershell
.\.venv\Scripts\python.exe -m plume_focus.sensing_cli --run-config configs/sensing-v1/coarse.json --output-dir outputs/sensing-v1-coarse --render
.\.venv\Scripts\python.exe examples/sensing_controller_v1.py --run-config configs/sensing-v1/coarse.json
```

The second synthetic configuration, `configs/sensing-v1/narrow.json`, preserves a
220 x 320 physical lattice with an 11-second column pass at 20 Hz. Its full-scene
startup exceeds the demonstration horizon. These fixtures are not hardware references.
Full-suite and reconstruction commands are in the [sensing guide](../docs/sensing-core.md).

The following commands reproduce the historical recovery demonstration:

The recovered core maintains a correlated Gaussian signal estimate and
uncertainty, then derives a normalized expected-signal map for a **90% mass**
scan target. It consumes timestamped scalar readings, accounts for physical
travel and detector cadence, and selects the next path after the current sweep
finishes. A 90% mass target is not a whole-plume containment guarantee.

```powershell
.\.venv\Scripts\python.exe -m plume_focus.sensing_cli --output-dir outputs/sensing-recovery --max-scans 5 --duration 75 --render
.\.venv\Scripts\python.exe examples/sensing_controller.py
```

The first command produces physical-time and compressed-summary videos from
the same operational trace, plus matched rendering and measurement diagnostics.
The second demonstrates the serializable controller interface through a fake
scanner. Installed packages also expose `plume-focus-sensing`.

See the [callable API, mathematical contract, assumptions, and reproduction
guide for the existing recovery implementation](../docs/sensing-core.md) and
[recovery audit](../docs/reviews/sensing-core-recovery.md).
The current dynamics remain uncalibrated and can carry signal too far downstream;
the adaptive performance claim remains `not_demonstrated`. Recorded-data
expansion remains paused after conformance closeout. Legacy
commands, profiles, and historical evidence remain available below.

## Legacy causal experiment contract

The legacy video/recorded pipeline is causal and scan-driven:

1. A physical scan starts at an explicit time and samples a moving field.
2. The scan completes before its likelihood is used.
3. The planner updates its posterior from completed scan values and masks.
4. Motion comes from external wind or successive completed scan posteriors.
5. The planner predicts through the expected duration of the next scan.
6. The next scan starts only after the previous scan completes.
7. Offline evaluation scores the actual planned mechanical sweep at its start,
   midpoint, and completion time.

Finite real videos never reuse their final frame as unavailable future sensor
data. If a scan or prediction target cannot fit inside the clip, the run stops
with an explicit `termination_reason`.

Full-frame vision remains available only as an explicitly labeled comparison:

```powershell
plume-focus-video --scenario plume --evidence-mode full-frame
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the tests:

```powershell
pytest
```

## Legacy causal methane example

The default methane model uses a mechanical column sweep. Vertical pass time
and horizontal step time are physical seconds, independent of video playback.

This causal smoke configuration uses a fast synthetic scanner and verifies the
timeline, scan-evidence, reporting, and rendering contracts. Its high recall is
not a model-performance benchmark: a fixed equal-area source corridor performs
as well on this deterministic scenario.

```powershell
plume-focus-video `
  --scenario methane `
  --max-scans 8 `
  --vertical-pass-seconds 0.005 `
  --horizontal-step-seconds 0 `
  --column-mass-fraction 1 `
  --row-mass-fraction 1 `
  --max-sweep-columns 320 `
  --output outputs/causal-methane.mp4 `
  --report outputs/examples/causal-methane.json
```

For a known persistent source, plume propagation is source-aware. Existing
belief is advected and diffused while new belief is emitted continuously along
every intermediate wind position over the prediction horizon. The default
emission rate is an explicit `0.24/s` model parameter, not a calibrated
concentration or generator-truth input. Both wind components orient the source
prior and emission trail.

The ordinary slower defaults are intentionally conservative. When accumulated
motion and scan-age uncertainty exceed the field of view, the policy selects a
full-scene reacquisition instead of silently capping uncertainty and continuing
with a narrow ROI.

Source-free tracking begins with a uniform full-scene prior:

```powershell
plume-focus-video --scenario methane --source-free --max-scans 8
```

Source-free mode does not receive a hidden source or emission trail. It retains
the detached-belief advection model until source inference is implemented.

For the synthetic methane generator, the supplied wind is the exact generator
wind and is reported as `oracle_known_wind_px_per_second`. It is an oracle
baseline, not a noisy sensor estimate.

The legacy file-backed demo accepts a video and optional wind components:

```powershell
plume-focus-video `
  --scenario methane `
  --input .\path\to\methane.mp4 `
  --methane-wind-dy -0.02 `
  --methane-wind-dx 1.02
```

Without external wind, scan mode estimates motion from successive completed
scan posteriors. The first scan uses zero motion and a broad/reacquisition
policy because two completed observations are not yet available.

This legacy `--input` path has no asset hashes, timestamp provenance,
calibration provenance, frozen split, or independent truth annotation. Its
reports are therefore labeled `legacy_unmanifested_input` and are not final
acceptance evidence.

## Recorded-data evidence contract

Use the separate recorded workflow for accession-audited, provenance-checked
inputs. The repository does not ship GasVid videos, derived masks, or a
leak-off calibration clip; obtain external assets under their applicable terms
before running these commands. The complete schema and evidence boundaries are
documented in
[`docs/recorded-data-contract.md`](../docs/recorded-data-contract.md).

Current GasVid status is `acquired_unverified`, not evidence-ready. A
2026-07-31 primary-source/provider audit found no explicit dataset licence or
permission for local processing, derivatives, publication, or redistribution,
so `rights_status` remains `unverified` and GasVid preparation stays blocked.
The local audit covers the full 31-file accession; this must not be confused
with VideoGasNet's later 25-file modeled subset. Five apparent capture groups
and the 18/13 filename-to-pad mapping are strong inferences only. Class 0 is
described as non-leak/background but also has a nonzero table value, while
`MOV_2568` and `MOV_2582` have non-nominal 157- and 360-second marker gaps.

The bounded R0B-Q metadata-only qualification substage is engineering-complete
at `54893a9`. Parent R0B remains `blocked` on owner permission and authoritative
assignment/interval evidence. The separate licensed R0C engineering smoke is
complete with a valid negative focusing result. The R0B-Q gate
provides a typed qualification
manifest/report, independently scoped permissions, per-asset evidence states,
explicit interval conflicts, a stable
`qualification_id`, a normalized data-root-relative `asset_root`, exact
template/check commands, and preparation-spec-v3 hash binding. The FGSTP paper
reports manual masks for a selected 14-video study, but their files, code,
exact mapping, and rights are not established; labels are not needed for a
truth-free raw causal pilot.

The preparation-spec-v3 `local-raw-pilot` path is deliberately narrower than
the general recorded workflow: it accepts only bounded `raw_recorded` windows,
uses no calibration source or structure-suppressed emulation, and keeps
evaluation truth at `none`. Qualification checking verifies canonical bytes,
the audit identity/hash, every confined
`asset_root/qualified-relative-path`, and every asset SHA-256 before a selected
preparation media path/hash may bind to it.

The enriched canonical GasVid qualification has SHA-256
`560f3bad93d3c5b72340ecf30ee174400ef1805b0ea9698c727b143b994c18ed`.
Its verified check exited `2`, returned `eligible=false` with 36 blockers, and
produced report SHA-256
`f330e64a26cfec5807bcc034190d5cdc8bb4c7a992f25a80ed11223cea519487`.
That is an intentional fail-closed result: no interval decode/extraction,
derived frame/mask/calibration, causal pilot, or performance run occurred.

R0C in [`RECORDED_DATA_WORK_PLAN.md`](../RECORDED_DATA_WORK_PLAN.md) used one Utah
State ground controlled-propane video under CC BY 4.0 for a raw, truth-free,
no-wind causal smoke without cleaning or policy tuning. Schema-v5 manifests
separate planner scenario from provider-backed gas identity; qualification-v2,
preparation-v4, and the frozen suite bind exact permissions and bytes. All
eight scans were causal, 96.34% of repeated-coordinate samples changed, and
beliefs stayed finite and normalized. However, requested-mask area remained
89.54–100% and every requested bounding ROI was full-height and nearly
full-width, so R0C closed as
`engineering_not_demonstrated`. This verifies plumbing, not methane
performance, plume recall, or Stage 4/G3 completion. The next proposed stage is
the development-only R0D measurement-separability and typed-normalization
contract; it is not active.

GasVid correspondence remains a separate path. The draft inquiry must first be
aligned with the checker's per-file group/split, interval, and permission
fields and given a hashed follow-up/closure protocol. It then requires the
user's finalized signature and authorization to send. There is no GasVid pilot
or evidence-ready suite yet.

```powershell
# Hash and fully decode an external accession; this is not sequence evidence
plume-focus-recorded-data audit `
  --root data/recorded/assets/external-accession `
  --accession-manifest data/recorded/accessions/accession.json `
  --report data/recorded/assets/dataset-audit.json

# R0B-Q interface is engineering-complete at 54893a9.
# Qualification path arguments are relative to --root unless an absolute path
# is supplied; every absolute path must still resolve inside --root.
plume-focus-recorded-data qualification-template `
  --root data/recorded `
  --dataset-audit assets/dataset-audit.json `
  --qualification-id gasvid-qualification-v1 `
  --asset-root assets/external-accession `
  --output metadata/gasvid-qualification-v1.json

# Exit 0 = eligible, 2 = valid but blocked, 1 = invalid/write failure.
plume-focus-recorded-data qualification-check `
  --root data/recorded `
  --qualification metadata/gasvid-qualification-v1.json `
  --report outputs/gasvid-qualification-v1.check.json

# New licensed diagnostics use qualification v2. First freeze the exact
# purpose, operations, and project-selected asset identities; then create and
# enrich the separate provider-fact/permission qualification. Content-hashed
# evidence references must be confined local snapshots under --root.
plume-focus-recorded-data project-preregistration-template `
  --root data/recorded `
  --dataset-audit metadata/dataset-audit.json `
  --preregistration-id external-raw-diagnostic-v1 `
  --purpose diagnostic_raw_pilot `
  --created-at-utc <ISO_8601_UTC_TIMESTAMP> `
  --operation decode_analysis `
  --operation render_local_media `
  --development-asset <AUDIT_RELATIVE_MEDIA_PATH> `
  --output metadata/project-preregistration.json

# Build deterministic truth-free manifests for a rights-cleared accession.
# Do not use this as a bypass for the blocked GasVid qualification result.
# Whole assets remain schema v3; preparation-spec v2 window requests emit
# explicit schema-v4 manifests. The implemented v3 local-raw-pilot extension
# additionally binds the qualification manifest path/hash.
plume-focus-recorded-data prepare `
  --root data/recorded `
  --spec data/recorded/metadata/preparation-spec.json `
  --profile data/recorded/metadata/preparation-profile.json `
  --output-manifests data/recorded/manifests `
  --report data/recorded/metadata/preparation-report.json

# Freeze one split using the exact preparation-report byte hash
plume-focus-recorded-data freeze `
  --root data/recorded `
  --preparation-report metadata/preparation-report.json `
  --preparation-report-sha256 <SHA256_OF_PREPARATION_REPORT> `
  --suite-id held-out-v1 `
  --split held-out `
  --frozen-at-utc <ISO_8601_UTC_TIMESTAMP> `
  --pre-change-model-commit <FULL_40_CHARACTER_COMMIT> `
  --freeze-provenance "<WHO_FROZE_IT_AND_WHY>" `
  --output manifests/held-out-suite-v1.json `
  --freeze-record metadata/held-out-suite-v1.freeze.json

# Verify the canonical freeze record and preparation/freeze lineage
plume-focus-recorded validate `
  --freeze-record metadata/held-out-suite-v1.freeze.json `
  --root data/recorded

# Verify one sequence and every declared SHA-256 hash
plume-focus-recorded validate `
  --manifest data/recorded/manifests/development-001.json

# Verify a frozen development or held-out suite
plume-focus-recorded validate `
  --suite data/recorded/manifests/held-out-suite-v1.json

# Verify media, calibration, and capture-group roles across frozen splits
plume-focus-recorded validate-pair `
  --development-suite data/recorded/manifests/development-suite-v1.json `
  --held-out-suite data/recorded/manifests/held-out-suite-v1.json

# Run causally; rendering remains optional
plume-focus-recorded run `
  --manifest data/recorded/manifests/development-001.json `
  --max-scans 8 `
  --report outputs/recorded-development-001.json `
  --output-video outputs/recorded-development-001.mp4
```

Paths passed to `freeze`, and the freeze-record path passed to `validate`, are
relative to `--root` unless they are absolute. Sequence/suite validation paths
are resolved directly from the current working directory. Freeze-record
validation checks preparation/freeze lineage; suite validation separately
decodes and checks runtime sequence assets. Acceptance review requires both.

Default sequence schema v3 supports honest truth-free causal input and binds
structure-suppressed emulation to a canonical fixed-calibration artifact.
Explicit schema v4 adds provenance-bound temporal windows for raw, truth-free,
no-wind runs while leaving v1-v3 manifests unchanged. Schema v5 is the
gas-aware, qualification-bound recorded contract: methane maps to the methane
planner, while propane or unspecified hydrocarbon maps to the generic plume
planner. V5 is currently raw-input-only until calibration-source project roles
are modeled. It verifies decode authority before media access and requires the
preregistered `render_local_media` operation before video rendering. The
artifact record
hashes the exact audited leak-off source, capture-group role, preparation spec,
complete emulation profile, producing commit, shape, sample count, and
generated bytes. Runtime loading does not refit calibration or decode the
leak-off video.

Preparation is dataset-neutral and no-overwrite: every audited byte identity
must have a declared capture-group disposition, excluded groups require an
objective non-performance reason, and rerunning the exact hashed audit,
preparation spec, profile, and output paths is byte-idempotent when decoded
content is unchanged. Audit schema v2 records both an OpenCV version/build
signature and the exact decoded grayscale pixel-stream hash for every video.
Preparation and runtime loading reject a decoded-content mismatch rather than
silently accepting codec-dependent pixels. Raw OGI and
structure-suppressed tracks stay separate. Emulation uses fixed per-pixel
median/MAD calibration, requires `uint8_identity`, and rejects temporal
smoothing until a stateful sensor response exists.

Truth-free v3/v4 runs report spatial metrics as unavailable rather than
deriving labels from sensor frames. A later lossless uint8 truth sidecar may be graded
`silver_pseudo` or `gold_human_reviewed`; silver remains diagnostic-only and
truth is evaluator-only. The current preparation/freezer accepts only its
exact truth-free output; deterministic truth attachment and refreezing remain
the planned R2 workflow. The recorded scanner remains
`counterfactual_virtual`: its area and time are modeled, not measured
hardware.

The recorded loader decodes the finite backing video into a timestamped source,
but the headless planner receives only mechanically requested coordinates and
timestamps. It cannot access evaluation truth or future samples. Schemas v1
and v2 remain loadable for reproduction; v1 is
`legacy_unclassified_evidence`, while v2 retains its legacy runtime
calibration path. Successful hash validation establishes identity and
consistency, not independent collection, held-out integrity, hardware
performance, or real sensor calibration.

## Fair methane benchmark

Any recorded methane fixture is a causal smoke test, not model
acceptance. The separate headless benchmark compares policies on fresh
timelines with identical initial evidence, selected area, column/row counts,
and physical scan duration:

```powershell
# Fast harness check; not benchmark evidence
plume-focus-benchmark --suite development --smoke

# Development and frozen held-out manifests
plume-focus-benchmark --suite development
plume-focus-benchmark --suite held-out
```

The benchmark uses a known source, constant episode wind, 12 decisions, and
three exact column budgets. Generator truth and planner motion are separate:
primary cases use noisy, biased, or missing measured wind. Reversed and
near-zero winds are labeled stress diagnostics and do not enter the claim
gate. The oracle policy is also excluded from the claim.

The source-aware policy-v2 held-out run used the unchanged v1 manifest with 24
primary and 6 stress episodes. All 540
policy/episode/budget results were present; timeline integrity passed; maximum
area and duration differences between matched policies were both exactly zero.
Mean interval probability-mass recall was:

| Budget | Area | Duration | Adaptive | Static corridor | Constant velocity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small | 2.39% | 9.09 s | 8.54% | 46.13% | 17.61% |
| Medium | 6.82% | 24.77 s | 6.10% | 84.01% | 5.76% |
| Large | 17.73% | 62.68 s | 61.17% | 100.00% | 50.19% |

The result is `not_demonstrated`, which is a valid completed benchmark
outcome. Averaged equally across budgets, adaptive minus the fixed source
corridor was `-0.5144`, with a paired bootstrap 95% interval of
`[-0.5529, -0.4758]`. Source-aware dynamics improved the small-budget adaptive
recall from `1.04%` to `8.54%`, but did not alter the medium- or large-budget
failure. The exact-wind policy reached only `0.34%` at the small budget because
its narrower forecast under-covered the plume. Source continuity is corrected;
uncertainty calibration and fixed-budget selection still need improvement
before a performance claim is credible.

The full local evidence files are generated artifacts:

- development manifest
  `914a31f920a379eae8e4ac11bdfa12cab57190aa71547826051c563243cbf758`;
- held-out manifest
  `da4e87e3665973ca270905889b14919081b359aa0e47bb077c65ec924b896ecd`;
- held-out report SHA-256
  `a7766a91866156d01376685ba9d361c8a245324a2f0d4410fe18e7d95aa58f4c`.

Reports use `benchmark_schema_version: 1` semantics under the JSON key
`report_schema_version`, with `policy_version: "2"` and
`frozen_model_commit: "104f55c"`. They include episode specifications, seeds,
the complete uncalibrated model profile, per-case metrics, recall-cost curves,
paired confidence intervals, stress diagnostics, oracle disclosures, and
integrity gates.

## Model profiles and hardcoded assumptions

Every causal run resolves a validated `ModelProfile` containing plume
dynamics, sparse-measurement likelihood, uncertainty growth, scan-policy
settings, source inclusion, and static-corridor geometry. The complete resolved
profile and its provenance are serialized into the JSON report.

Generated fixtures use the explicitly named `synthetic_demo_v1` compatibility
profile. File-backed video uses `uncalibrated_video_v1` unless the caller
supplies another profile programmatically. Neither profile is a real-sensor
calibration.

The `0.45`, `0.22`, and `0.08` attention cutoffs are relative-density rendering
and diagnostic values only. They are not calibrated confidence values and do
not affect mechanical scan selection.

## Evidence, timing, and scan controls

- `--evidence-mode scan` is the default for every scenario.
- `--evidence-mode full-frame` is an explicit instantaneous-image baseline.
- `--max-scans` limits physically completed scan/update cycles.
- `--max-frames` remains as a deprecated compatibility alias for
  `--max-scans`.
- `--scan-rate-hz` controls point visits per second in point mode.
- `--sample-stride-px` controls one point sample footprint.
- `--vertical-pass-seconds` controls a full-height column traversal.
- `--horizontal-step-seconds` controls movement between column axes.
- `--column-stride-px` controls horizontal column spacing.
- `--frame-stride` subsamples real-video/full-frame refresh without changing
  physical scan seconds.
- `--success-coverage` sets the required completion-time plume probability-mass
  recall; the default is `0.9`.

The planner distinguishes:

- requested belief mask;
- bounding envelope used for readability;
- mechanically traversed sweep mask;
- pixels actually sampled;
- attention field used for visualization;
- offline true plume support.

Column row bands are contiguous. A scanner cannot skip travel between two
selected rows and then report only the endpoints as swept area.

## Headless experiments

Benchmark and analysis code can run the causal engine without creating files or
loading the rendering adapter:

```python
from plume_focus.experiment_runner import ExperimentRequest, run_experiment

result = run_experiment(
    ExperimentRequest(scenario="methane", max_scans=8)
)
print(result.report.termination_reason)
```

Pass an `ExperimentObserver` to receive `ExperimentContext` and
`ExperimentStep` lifecycle events. The MP4 workflow is one such observer-based
adapter; the core runner never creates a `VideoWriter` or writes JSON.

## Report schema

Video proof reports use `report_schema_version: 3`. Schema v3 retains the
schema-v2 metric boundaries and adds observer-independent operational evidence.
Important sections include:

- `model_profile`: complete declared and physically resolved behavior settings;
- `data_provenance`: source class, asset hashes, split, calibration, and claim
  eligibility;
- `evaluation_truth`: truth basis and explicit policy-isolation declaration;
- `wind_evidence`: causal lookup counts, measurement age, and fallback counts;
- `timeline_integrity`: non-overlap status, elapsed experiment time, and target
  horizon;
- `operational_trace`: each completed scan's causal times, requested,
  mechanical, sampled, and next ROIs, evidence-time span, and reacquisition
  decision;
- `belief_integrity`: explicit finite, nonnegative, and normalized checks over
  every reported posterior and predictive distribution;
- `sample_variation`: repeated-coordinate intensity comparisons derived only
  from causal scan samples, so changing recorded evidence is distinguishable
  from a changing scan schedule;
- `evidence_mode` and `motion_source`;
- `scan_duration_summary`: initial, mean, maximum, and final physical duration;
- `actual_sweep_evaluation`: actual mechanical-mask recall, IoU, and thresholded
  success;
- `attention_diagnostics`: broad visualization-mask metrics, kept separate from
  scan success;
- `termination_reason` and `oracle_inputs`.

Recorded reports also carry
`data_provenance.evidence_contract.media_attribution`: the manifest source
credit, accession source and licence URIs, declared analysis changes, and any
annotated-rendering change. Keep the JSON report beside a redistributed video;
the video alone is not a complete attribution artifact.

The canonical smoke-test report is
`outputs/examples/causal-methane.json`. Generated MP4 and preview files are
ignored and should be recreated locally from the command above.

## Package map

- `plume_focus.experiment`: finite/continuous time sources and causal event
  invariants.
- `plume_focus.experiment_runner`: headless sequential orchestration and the
  observer lifecycle.
- `plume_focus.experiment_contracts`: shared scan and report contracts.
- `plume_focus.video_sources`: synthetic and file-backed frame sources.
- `plume_focus.scan_execution`: point and mechanical column scans.
- `plume_focus.experiment_metrics`: observation and distribution metrics.
- `plume_focus.plume_dynamics`: advection, diffusion, source emission, and
  source-oriented initialization.
- `plume_focus.model_profiles`: validated named model, likelihood, uncertainty,
  and scan-policy configurations.
- `plume_focus.scan_policies`: causal policy contracts and exact-cost mask
  allocation.
- `plume_focus.benchmark_cases`: deterministic development and held-out
  manifests.
- `plume_focus.benchmark_runner`: headless matched-policy evaluation.
- `plume_focus.benchmark_statistics`: paired bootstrap comparisons.
- `plume_focus.recorded_dataset`: deterministic external-accession hash and
  full-decode auditing before sequence construction.
- `plume_focus.recorded_dataset_cli`: thin audit, prepare, and freeze command
  dispatcher.
- `plume_focus.recorded_contracts`: typed recorded manifests and runtime
  evidence interfaces.
- `plume_focus.recorded_preparation_contracts`: strict preparation, frozen
  calibration, and preparation-report contracts.
- `plume_focus.recorded_preparation`: deterministic truth-free manifest and
  fixed-calibration construction.
- `plume_focus.recorded_freeze`: canonical suite and companion lineage-record
  publication plus preparation/freeze-lineage validation.
- `plume_focus.recorded_role_isolation`: cross-split media, calibration, and
  capture-role checks.
- `plume_focus.recorded_manifest_io`: strict JSON parsing and canonical hashes.
- `plume_focus.recorded_assets`: confined asset paths, clocks, normalization,
  wind sidecars, and hash verification.
- `plume_focus.recorded_providers`: causal wind and evaluation-only truth
  providers.
- `plume_focus.recorded_sources`: public loading and suite-validation façade.
- `plume_focus.recorded_workflow`: optional video and JSON adapter for
  validated recorded sequences.
- `plume_focus.sensor_emulation`: fixed leak-off median/MAD calibration and
  causal structure-suppressed timeline adapter.
- `plume_focus.video_rendering`: optional OpenCV rendering observer.
- `plume_focus.video_workflow`: MP4 and JSON filesystem adapter.
- `plume_focus.bayes`: likelihood, posterior, compatibility wrappers, and
  credible regions.
- `plume_focus.scanner`: reusable single-pixel scan primitives.
- `plume_focus.video`: compatibility façade and CLI entry point.
- `plume_focus.motion`: motion representations and legacy full-frame estimators.
- `plume_focus.predictor`: bounded and smooth attention primitives.

## Current limitations

- Source-aware dynamics use a constant 2-D wind vector, one persistent source,
  a simplified Gaussian diffusion model, and an uncalibrated emission rate.
- Spatially varying, rotating, intermittent, and turbulent flow are not
  represented.
- The synthetic exact-wind baseline is optimistic and is clearly labeled.
- Real sensor calibration, hardware control, concentration estimation,
  turbulence calibration, unknown-source discovery, multi-source separation,
  and safety guarantees are not implemented.
- Full-frame truth is available to the offline evaluator and renderer only; it
  must not enter scan-mode planning.

See `AGENTS.md` for the project’s claim boundaries and development rules.
