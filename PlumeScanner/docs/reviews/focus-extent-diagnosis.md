# Focus extent and overlapping-mechanism diagnosis

Completed 2026-09-18. Start `44be6a9`; branch `codex/focus-extent-diagnosis`;
diagnostic execution code `9e52042`. See the [declared protocol](../specs/focus-extent-diagnosis.md).
This round changes observers and tests only. Operating inference, policy,
mechanics, physical profiles and video layout remain unchanged. The claim stays
`not_demonstrated`.

## What explains the visible region?

The readings and clean synthetic plume are localized. The inferred mean already
has a long, weak downwind tail before the upper uncertainty bound is applied.
Thresholding that tail selects many background cells. The main discrepancy in
the first decisions is **downwind length**, not uniformly excessive width in
both axes. Nominal wind conversion is exactly consistent with the simulator;
the model already uses velocity multiplied by elapsed time.

At the first completed full scan, zero added noise, forecast six seconds ahead:

| Physical profile | Mean >= .15 | Upper 95% >= .15 | Truth >= .15 | Requested / swept |
| --- | ---: | ---: | ---: | ---: |
| Coarse 20 x 32 | 44 | 54 | 14 | 54 / 54 |
| Doubled 40 x 64 | 167 | 262 | 59 | 189 / 189 |

These are physical lattice cell counts, not probabilities. The doubled policy
cannot execute the whole eligible set within its ceiling. Mechanical closure
adds no cells in any of the 42 inspected plans; this excludes closure enlargement
as the explanation in these saved cases, not in every possible geometry.

Coarse current-mean downwind extent is .626 FOV versus truth .253. At +6 s it
is .784 versus .251. Doubled current-mean extent is .720 versus .266, becoming
.876 versus .267 at +6 s. Crosswind extent is comparable or smaller than truth
in these examples. A single generic ROI-width number would conceal that distinction.

## Hypothesis evidence

| Explanation | Evidence | Classification and limitation |
| --- | --- | --- |
| Missing amplitude scaling in the Gaussian update | Independent joint conditioning agrees; readings .1/.3/1 yield projected peaks .0917/.2750/.9166 in the controlled baseline | Contradicted as an arithmetic bug. Does not establish calibrated inferred amplitudes after propagation. |
| Source injection and persistence sustain the tail | Future source-off reduces first coarse mean cells 44 to 24; source-off throughout replay nearly erases first post-startup signal. After cessation/rescan it removes false gate activation | Supported contribution. Removing the source outright is not a satisfactory fix: it also loses continuing signal during long gaps. |
| Fixed spatial representation affects amplitude/width | Halving correlation length changes the .3-reading above-threshold region from 700 to 356 cells on the fixed 120 x 160 diagnostic mesh; doubling inference resolution gives 536 | Supported sensitivity, not proof that either alternative improves autonomous reconstruction. |
| Uncertainty alone causes the excess | Coarse upper bound adds 10 background cells and zero signal cells at first +6 s forecast; doubled adds 89 background and six signal cells. Mean is already long | Secondary contribution. Bounds sometimes add useful signal as well; wholesale removal is not justified. |
| Diffusion and process uncertainty duplicate one effect | Process-off leaves mean exactly unchanged; diffusion-off changes mean and covariance. First coarse upper mask: baseline 54, diffusion-off 54, process-off 52, both-off 53 | Distinct mathematical roles, with overlapping possible physical assumptions. Future-only ablation does not remove historical spread. No demonstrated duplicate state update. |
| Forecast horizon worsens geometry | Coarse upper region grows 41 current cells to 54 at +6 s and 65 at +12 s. Actual first focus duration is 4.55 s, so the fixed +6 s lookahead extends beyond its completion | Supported additional contribution. Current mean is already too long; horizon change alone cannot repair it. |
| Added simulated noise creates the tail | First coarse forecast mean cells stay 44 at noise 0/.01/.05; doubled 167/167/168 | Not supported as the main cause in these descriptive seed-0 fixtures. Assumed likelihood noise remains .05. |
| Wind calibration has a unit/sign mismatch | Converted simulator velocity equals supplied nominal velocity with maximum difference zero across all 18 configurations | Contradicted for these fixtures. Lateral deformation and signal lifetime still differ from the estimator. |

The source cessation result is particularly direct. At 222.025 s, after a fresh
coarse full rescan of the blank scene, maximum current mean is .253745. Identical
evidence replay without source input gives .0000331. At 320.1625 s on the doubled
profile, the corresponding values are .296207 and approximately 5.8e-26. Both
baseline values cross the .15 focus gate. This is not caused by simulated noise.

However, source-off is not uniformly better. At the first continuous coarse
forecast, instantaneous MSE changes from .015664 to .021077; doubled changes
from .016423 to .021930. At the later doubled continuous snapshot, source-off
still has 391 above-threshold mean cells versus baseline 388 and higher MSE.
Historical measured signal, transport and representation also matter. These are
fixed-evidence instantaneous forecast errors, **not causal J/T from new executions**.

## Amplitude and representation

Conditioning adds `P h / (h' P h + r)` times the reading innovation. There is no
separate fixed-height Gaussian pasted around each hit. Reading magnitude scales
the correction; the covariance sets its spatial shape. A stronger reading also
expands the area exceeding an absolute cutoff even when correlation length is fixed.

The initial model source width is .035 FOV, while the simulator's newborn puff
standard deviations are approximately .0082 vertically and .0076 horizontally.
The 12 x 16 inference grid has pitches .0833/.0625 FOV. These are distinct
assumptions, not interchangeable sensor pixel sizes. The controlled narrow
Gaussian packet already reconstructs a peak of .4165 versus continuous truth
.989 at time zero, before transport. This diagnoses representation error rather
than an assimilation amplitude multiplier.

Doubling the footprint in the particular centered reading probe has identical
weights: both footprints remain within the same bilinear interpolation patch.
That local equality does not make sensor footprint modeling redundant generally.

Independent packet references separate persistent translation from exponential
fading (52/12 s time constant) and finite 5.5 s lifetime. At +6 s a fading packet
is absent, but the unchanged model retains a .1474 peak. The continuous-source
controls likewise expose different balance between injection and loss. These are
analytic reference fixtures, not fitted substitutes for the simulator. They retain
their initial representation error and do not establish a calibrated decay rate.

## Redundancy audit

| System | Actual role | Finding |
| --- | --- | --- |
| Spatial covariance | Correlated likelihood correction and uncertainty | Necessary distinct role; spatial scale requires calibration. |
| Gaussian source input | Adds mean signal without new evidence | Active, consequential assumption; not a second likelihood update. |
| Advection / diffusion | Transport / spread of mean and covariance | Active; numerical interpolation also smooths. No duplicate transport found. |
| Correlated process noise | Adds covariance, not mean | Active. May represent some of the same unresolved physics as diffusion, but the present experiment does not identify a removable duplicate. |
| Footprint / projection | Measurement averaging / field representation | Distinct operators. Coarse representation can make small footprint differences invisible. |
| Upper quantile | Converts latent uncertainty into eligibility | Active allocation operation, not an additional Gaussian update. |
| Expected-excess map | Normalized diagnostic visualization | Not used for lean selection. Bright normalized pixels do not imply strong absolute signal. |
| Historical mass/risk/service/mixture policies | Compatibility execution modes | Not stacked into the lean path. Removing them would not repair the observed tail. |
| Read-only forecast methods | Clone and project state for different consumers | Some repeated code/computation, but no cumulative evidence or transport. Optional later helper extraction must preserve event partitioning. |
| Per-reading BeliefView / mass computation | Shared acquisition event compatibility | Extra derived work for a lean adapter that ignores returned events; possible later opt-in/lazy optimization, not an inference error. |
| Legacy source scan fields and shared survey timestamp | Historical configuration/state compatibility | Inactive for lean allocation. Candidates for adapter cleanup, not the physical cause. |

No production mechanism was removed. Reducing uncertainty because it looks broad
would conceal calibration failure; overlap in a picture is not proof of redundancy.

## Verification and artifacts

All 18 input runs produce 42 predeclared snapshots. Fixed-evidence replay matches
public-controller mean/covariance with maximum error zero through each last
selected event. Source-off replay preserves the same reading identities, delivery
calls, acquisition ordering, wind changes and propagation boundaries. Alternative
models are not required to reproduce historical allocation decisions.

Six forecast variants cover 0/1/3/6/12 s and focused midpoint/completion. Physical
mask compilation keeps complete masks and explicitly reports feasibility. Saved
arrays include clean exposure readings, actual values and ages; unobserved values
are NaN only in numerical arrays, not invented zeros or nonstandard JSON.

All 84 doubled spatial-mesh comparisons (120 x 160 to 240 x 320, current/+6 s)
meet the 1% instantaneous-loss check and one-physical-pitch width check. This is
projection/evaluation convergence, not inference-grid or temporal-model convergence.
Sub-pitch geometry differences remain unresolved. No new full-session loss claim
or independent-pixel confidence claim is made.

Verification: **506 tests pass, five existing Windows symlink skips**; compilation,
import-boundary tests and Git diff checks pass. All 184 historical artifacts and
recorded runtime source hashes remain byte-identical. There are 28 separate
diagnostic figures and 91 hashed diagnosis artifacts. Existing video layouts are
unchanged; reviewed frames at coarse continuous 36 s, doubled continuous 135.5 s
and coarse cessation return-home 188 s agree with the traced state meanings.
Compressed-summary frames reviewed at 33.575 s (coarse continuous), 141.5125 s
(doubled continuous) and 343.125 s (coarse cessation) retain their explicit
summary labels. One presentation ambiguity remains: the footer's gate maximum
belongs to the planning decision, whereas the signal panel shows the current
posterior. For example, startup's gate maximum stays zero in its completion
frame. This is not evidence that assimilation failed; the new diagnostic figures
label current and forecast states separately. Historical media are not rewritten.
Representative new figures reviewed: coarse first focus event 642, doubled first
focus profiles event 2562, and coarse false refocus event 3614. Figure labels
separate future executed cells from evidence available at the decision.

Reproduction from the repository root (PowerShell):

```powershell
.\.venv\Scripts\python.exe scripts/diagnose-focus-extent.py --phase declare
.\.venv\Scripts\python.exe scripts/diagnose-focus-extent.py --phase controls
.\.venv\Scripts\python.exe scripts/diagnose-focus-extent.py --phase run --zero-only --workers 2
.\.venv\Scripts\python.exe scripts/diagnose-focus-extent.py --phase run --workers 2
.\.venv\Scripts\python.exe scripts/report-focus-extent.py
.\.venv\Scripts\python.exe scripts/verify-focus-extent.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
git diff --check
```

Results live in local `outputs/focus-extent/`: declaration, controls, per-run
records/arrays, summary, figures, artifact manifest and verification. The execution
code digest is `030650a4a2c4341632bad19f5c01c33865e49c6299fae5a5146a652bda2faace`.
Early zero-noise records contain the pre-commit HEAD `44be6a9`; their exact code
bytes are verified against committed implementation `9e52042`, so that HEAD alone
must not be mistaken for their full producing source. Final revision, current
artifact manifest hash and remote confirmation are recorded in local closeout.

## Smallest justified next stage

Calibrate **source input and signal persistence on fixed evidence and paths**,
including both continuous emission and cessation. First require the declared
field representation to resolve the reference peak sufficiently to distinguish
source/lifetime error from spatial approximation. Test a separately versioned loss
or lifetime model; do not change the allocator at the same time. Check peak,
downwind tail, misses and current-time error together. Reject a narrower model
that merely deletes continuing signal or hides uncertainty.

Resolved: missing amplitude multiplier, nominal wind unit mismatch and mechanical
closure enlargement are not supported explanations for the inspected behavior.
Supported: source/persistence contribution, representation sensitivity and extra
uncertainty-driven background selection. Unresolved: calibration of these terms,
local deformation, optimal forecast timing and autonomous reconstruction advantage.
Code cleanup can follow separately, with unchanged replay hashes; it is not the
scientific fix. No new policy is frozen.
