# Plume Scanner

Adaptive point sensing for reconstructing moving methane plumes under mechanical
scan and acquisition-time constraints. A single sensor measures positions sequentially;
timestamped readings and a supplied wind estimate guide where to look next. The
eventual goal is plume reconstruction and strength estimation. This is a research
prototype with synthetic demonstrations.

This research branch, `source_prior_fixed_evidence`, extends the scanner from
[`plume_scanner_new` at `5cac633`](https://github.com/BerkeleyExpertSystemTechnologiesLab/Squishy-Methane-Analysis/tree/5cac633f47001d3aa455f8b6f3cc00b11c550ed3/PlumeScanner).
The [source-assumption experiment guide](docs/source-assumptions-demo.md) identifies
the baseline, synthetic input/output quantities, timing and spatial assumptions,
and added comparisons. The baseline controller and scan policy are unchanged.
Run commands from `PlumeScanner/`. The Python package name is `plume_focus`.

## Install

Requires **Python 3.10+**; verified on Python 3.12. The quickstart needs no GPU or
external footage. Dependencies are NumPy, SciPy and headless OpenCV; `.[dev]`
also installs pytest. Matplotlib is optional through `.[plots]`.

Fetch this project without downloading the repository's other research datasets:

```sh
git clone --filter=blob:none --sparse --single-branch --branch source_prior_fixed_evidence https://github.com/BerkeleyExpertSystemTechnologiesLab/Squishy-Methane-Analysis.git
cd Squishy-Methane-Analysis
git sparse-checkout set PlumeScanner
cd PlumeScanner
```

For an existing checkout, switch to `source_prior_fixed_evidence` and enter `PlumeScanner/`.
Then create an environment and install the package.

**Windows PowerShell** (activation is optional):

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

**macOS / Linux**:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Below, replace the Windows interpreter path with `.venv/bin/python` on macOS/Linux.
After activating the environment, `python` also works.

## Run the current scanner

Run the supplied **40 s synthetic, source-anchored, known-wind** demonstration:

```powershell
.\.venv\Scripts\python.exe scripts/run-scanner-demo.py
```

The default [configuration](configs/evidence-gated-focus/coarse-demo.json) uses the
opt-in `evidence_gated_focus_loop_v3_candidate`, the frozen calibrated synthetic
estimator, a 20 x 32 physical lattice and zero added simulated noise. Assumed sensor
noise remains .05. The short demo's horizon differs from the 360 s study.

The command completes startup coverage, executes focused sweeps when readings
support them, verifies public-call replay and prints scan counts, termination and
synthetic reconstruction loss. Plans, readings, receipts and configuration are saved
to `outputs/quickstart/runs/demo.json`. Physical scan time and offline compute time
are separate. A budget shortfall is explicit when another full scan cannot fit.

Existing output is preserved. To repeat, choose a fresh directory:

```powershell
.\.venv\Scripts\python.exe scripts/run-scanner-demo.py --output outputs/quickstart-2
```

Render the first run in the existing six-panel view:

```powershell
.\.venv\Scripts\python.exe scripts/render-full-fov-demo.py --output outputs/quickstart --name demo --fps 4 --speed 1
```

Under `outputs/quickstart/inspection/demo/`, `physical.mp4` follows physical time
and `summary.mp4` shows one frame per completed scan. Playback never changes
acquisition or inference. Open these in a video player; headless OpenCV has no GUI.

For the source-assumption study and a simpler side-by-side visual demonstration,
see [Source-assumption experiments](docs/source-assumptions-demo.md).
It compares estimated maps from identical readings with correct and incorrect
fixed source hints, and explains prior, posterior and injection in plain language.

## Controller API


The three operations are `plan_next`, `ingest_readings` and `finish_scan`. An active
path stays fixed until its receipt closes the scan. Evidence uses actual acquisition
and availability times; new planning waits for completion.

```python
import json
from pathlib import Path

from plume_focus.controller import SensingController
from plume_focus.full_fov_execution import decode_config
from plume_focus.sensing_messages_v2 import ScanRequestV1

config = decode_config(json.loads(
    Path("configs/evidence-gated-focus/coarse-demo.json").read_text()
))
controller = SensingController.from_session(config.session)
plan = controller.plan_next(
    0.0, config.session.home_yx,
    ScanRequestV1(config.duration_seconds), wind=config.wind,
)
# An adapter executes plan and supplies SensorReading messages and a receipt:
# controller.ingest_readings(readings, available_at_seconds=delivery_time)
# controller.finish_scan(receipt)
```

For an executable two-cycle example using `FakeScanner`:

```powershell
.\.venv\Scripts\python.exe examples/full_fov_scanner.py configs/evidence-gated-focus/coarse-demo.json
```

Coordinates are normalized `(y, x)` FOV fractions; time is seconds from the session
origin. Motor mapping, sensor retrieval calibration and hardware control belong to
the adapter. The [API guide](docs/sensing-core.md) covers messages and causal checks.

## Verify changes

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src scripts examples tests
```

The submission passes **554 tests with five platform skips**, including the new
demo CLI. The reviewed scanner checkpoint replayed 104 frozen study executions
with unchanged reconstruction loss. Tests need no external footage. Symlink
tests skip where that capability is unavailable. If the system temporary directory
is restricted, use a fresh workspace directory with `--basetemp outputs/pytest-local`.

## Evidence and limitations

Delivered scalar readings govern entry into and retention of focus around the
unchanged calibrated estimator. Unsupported readings trigger a charged return home
and full-field scan. The predicted signal field chooses where to focus; the gate
does not erase false model signal.

The frozen 104-run study supports reduced false focusing in blank, cessation and
intermittent fixtures, while preserving continuous/wind-error/weak behavior.
**Reconstruction remains mixed**: blank, cessation and intermittency map error
worsen. Gate bounds describe measurement error under the declared noise model,
not calibrated field confidence. Real sensor calibration, strength estimation,
unknown-source discovery and production hardware validation remain open. The
candidate stays opt-in; general reconstruction advantage is not demonstrated.

The [study review](docs/reviews/evidence-gated-focus.md) and
[cleanup review](docs/reviews/evidence-gated-focus-cleanup.md) record results,
corrections and limits. Versioned specs and configuration hashes define experiments;
do not tune against frozen held-out results. A new study can be declared with
`scripts/run-evidence-gated-focus.py --phase declare --root outputs/new-study`;
execution is substantially larger than the quickstart. Original generated study
records and videos remain local and are excluded from this submission.

## Repository guide

| Path | Purpose |
|---|---|
| `src/plume_focus/` | Inference, acquisition, planning and adapters |
| `examples/` | Executable public-API examples |
| `configs/` | Reference profiles and study declarations |
| `scripts/` | Demo, evaluation, replay and rendering commands |
| `tests/` | Unit, acceptance, replay and report regressions |
| `docs/specs/` | Versioned formulation and policy contracts |
| `docs/reviews/` | Verification, research evidence and failures |
| `outputs/` | Local generated artifacts; ignored except a canonical example |

[SYSTEM_GOAL.md](SYSTEM_GOAL.md) tracks research status and
[CHANGELOG.md](CHANGELOG.md) records checkpoints. Older commands and recorded-data
workflows remain in the [historical usage reference](docs/legacy-usage.md) and
[recorded-data work plan](RECORDED_DATA_WORK_PLAN.md). External datasets and their
access requirements are separate from this synthetic quickstart.
