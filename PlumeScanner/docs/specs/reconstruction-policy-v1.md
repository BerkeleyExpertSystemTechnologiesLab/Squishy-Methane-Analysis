# Reconstruction formulation and scan policy v1

Status: **frozen specification**, 2026-09-13 UTC. This is a specification freeze,
not a claim of implementation conformance or adaptive superiority.

Identifiers: `beam_reconstruction_v1`, `mass90_column_v1`, and
`instrument_profile_v1`. These identifiers are separate from the existing
controller's transport `schema_version=1`. The [implementation review](../reviews/policy-v1-freeze.md)
records the gaps. [Worked examples](freeze-v1-worked-examples.md) check the
definitions independently of the production allocator.

## 1. Purpose, evidence, and version boundary

The purpose is reconstruction of an evolving, intermittent plume from a costly
single remote detector under mechanical travel and acquisition constraints,
eventually supporting a physically defined plume-strength estimate. The
immediate product is a time-referenced angular/projected map of **inferred
beam-integrated concentration enhancement**. The detector supplies a scalar
already retrieved from its response. Raw spectroscopy inversion is outside the
controller. A two-dimensional map does not identify gas location along a beam,
and its spatial sum is not automatically plume mass or an emission rate.

The Gaussian estimator is an explicit replacement model, not recovered May
code. Standard conditioning is established mathematics. The selected transport
approximation, 90% rule, fallback schedule, and their usefulness for this
instrument are modeling and policy choices. Adaptive superiority, calibrated
uncertainty, improved hardware accuracy, and lower system cost are unproved.

The primary objective is

\[
J(\pi)=E\left[\int_0^T \sum_j w_j
  (x_j(t)-\hat x_j^\pi(t))^2\,dt\right].
\]

Here `T` is a common physical monitoring horizon, positive spatial quadrature
weights `w` sum to one and remain fixed, and the expectation is over the declared
scenario/noise distribution. `x` is signed enhancement above a declared
background; do not clip it before computing reconstruction error. At time `t`,
the reported estimate may use only evidence delivered by `t`. A delayed
measurement can improve a present estimate; it cannot improve an already
published past estimate for this score. Retrospective smoothing is a separate
product with its own latency and metric.

The policy below is a reproducible heuristic for this objective. It does not
minimize `J` exactly. Shorter scans may improve freshness but may also miss signal.
Coverage, scan area, or duration alone cannot establish success.

## 2. Measurement and finite Bayesian state

Use a finite grid of Gaussian enhancement coefficients `X_t ~ N(mu_t, P_t)` on
the declared field of view. The initial implementation uses a configurable
12 x 16 inference grid. This size is a numerical fixture, not a beam size or
hardware requirement. Freeze the inference discretization and approximation
settings within each comparison; change display resolution independently.

The instrument profile declares concentration-path units and the retrieval
convention, including any one-way/round-trip correction already applied. Define
the observation after fixed calibration/background correction as

\[
y_i=(v_i-o_i)/g_i=h_i^T X_{t_i}+\epsilon_i,
\qquad \epsilon_i\sim N(0,r_i),\quad r_i=(s_i/g_i)^2>0.
\]

`v_i` is the supplied retrieved scalar, not a raw camera pixel. Offset/background
`o_i`, gain `g_i>0`, and error standard deviation `s_i` have declared units and
frozen provenance. The v1 demonstration uses fixed synthetic calibration;
there is no per-sweep percentile scaling. Any background map or path-dependent
correction is fixed before acquisition, supplied by the adapter, and identified
in the run manifest. Preserve the original scalar in the evidence record.

`h_i` is a normalized spatial average of bilinear basis functions over the actual
beam footprint. One footprint produces one observation, even if it covers many
inference or display cells. The current four-point rectangular quadrature and
exposure-midpoint approximation are the initial numerical choices. Clip and
renormalize an edge footprint only in a profile explicitly using the declared
clipped-FOV sensor model; do not silently ignore gas outside the modeled FOV in
a hardware profile. Other edge/temporal response models require a named version.

The physical reading generally averages over an exposure interval and moving
beam. Treating it as a midpoint observation is an approximation that must be
recorded and assessed against integration duration, motion, and retrieval lag.
It does not remove scan-time distortion. Longer filtered or temporally correlated
retrievals are not conforming independent midpoint evidence without an explicit
adapter/error-model justification. A sensor's retrieval prior must not be counted
again as independent information.

For each accepted reading, first predict to its acquisition midpoint, then use

\[
K_i=\frac{P^-h_i}{h_i^TP^-h_i+r_i},\qquad
\mu^+=\mu^-+K_i(y_i-h_i^T\mu^-),\qquad
P^+=P^- - \frac{(P^-h_i)(P^-h_i)^T}{h_i^TP^-h_i+r_i}.
\]

The joint covariance supplies neighborhood influence; independent occupancy
probabilities are not substituted for it. Preserve signed means. Symmetrize
roundoff and check positive semidefiniteness within a declared numerical
tolerance; do not hide material negative eigenvalues by clipping.

The v1 likelihood assumes independent residual errors conditional on the state
and fixed calibration. A repeated independent reading reduces uncertainty; a
duplicate delivery does not. Missing, invalid, or rejected observations produce
no likelihood. A valid zero or negative enhancement is evidence. Saturation and
shared calibration bias require explicit treatment rather than synthetic zeros
or falsely independent repetitions.

## 3. Evolution and causal lifecycle

Between acquisitions apply the declared linear dynamics

\[
X_{t+\Delta}=F_{t,\Delta}X_t+u_{t,\Delta}+\eta,
\quad \mu'=F\mu+u,\quad P'=FPF^T+Q,\quad Q\succeq0.
\]

The initial numerical model composes bilinear advection and absorbing-boundary
diffusion, with a squared-exponential prior kernel and correlated process noise
`Q = process_std^2 * dt * kernel` per substep. Record the substep rule, spatial
correlation, diffusion coefficient, and all units. Advection velocity is in the
declared FOV coordinate system; an anemometer vector needs an explicit projection
model before it becomes that velocity. A fixed 2-D velocity is a short-horizon
approximation to the projected field, not a recovered 3-D transport law.

Only source-anchored mode may supply nonzero continuous `u` from a declared
source prior. Source-free mode has `u=0`; full-scene startup describes initial
acquisition and does not grant source knowledge. The current no-decay transport
can retain synthetic signal too long. Preserve this limitation and measure it;
the freeze does not assert that diffusion/process noise repairs source lifetime.
The scalar wind-error bound used by the scheduler is not automatically an
uncertainty-calibrated covariance model. Any mapping from wind error to `Q` must
be separately specified in the model configuration.

Use seconds from one monotonic session origin. The causal contract is:

1. `plan_next(now, position, constraints, wind)` uses only available evidence;
   the new scan cannot overlap the previous actual execution or precede receipt
   availability. Charge travel from the actual final position.
2. The accepted path stays fixed while its scan is open. Readings carry unique
   session identities, scan identity, actual position/footprint, exposure interval
   or midpoint plus duration, scalar value, and availability time. Profile and
   calibration identities are bound through the session manifest.
3. Delivery calls are chronological. Assimilate evidence in acquisition order.
   Delayed out-of-order data may replay the **open** sweep from its saved starting
   state. Replay becomes visible at its delivery time, not in past snapshots.
   Equivalent evidence and dynamics yield the same completed posterior for batch
   and streaming delivery; intermediate estimates and `J` can differ with latency.
4. Exact duplicate payloads are idempotent, including after closure. Conflicting
   identities reject atomically. New nonduplicate evidence for a closed scan is
   rejected in v1; cross-scan smoothing is not silently promised.
5. `finish_scan` records actual completion, actual final position, and receipt
   availability. All accepted exposures must lie within actual execution. The
   controller advances knowledge to receipt availability and enables the next
   decision. Missing planned readings and partial execution are recorded, not
   fabricated. Actual overrun is reported as an execution deviation.

Hold the causal wind estimate through the open sweep; new estimates apply at the
next decision. For a gap before that decision, use the previously available
estimate, not a wind measurement that arrived at the end of the gap. Evaluator
truth, future source values, source internals, rendering, and files are not
controller inputs. Preserve the existing three-operation Python boundary and
versioned JSON messages; new mandatory fields require an explicit transport
version or session wrapper, not a silent reinterpretation of old messages.

## 4. Expected-mass selection

Let physical lattice center `j` have predictive marginal mean `a_j` and standard
deviation `s_j`, obtained by bilinear projection of the finite Gaussian state.
Compute projected variances with the full covariance, not just its diagonal.
Use fixed positive quadrature weights `w_j` summing to one on this lattice.

For a declared enhancement threshold `tau`, set

\[
e_j=E[(Z_j-\tau)_+]
 =(a_j-\tau)\Phi(d_j)+s_j\phi(d_j),\quad d_j=(a_j-\tau)/s_j,
\qquad m_j=w_je_j,\quad b_j=m_j/\sum_k m_k.
\]

For `s_j=0`, use `e_j=max(a_j-tau,0)`. An expected total at or below the declared
positive `mass_floor` is `no_meaningful_predicted_signal`, with mass ratios
unavailable. Never normalize an empty field or call it 100% covered.

`m` is an expected positive-signal index, not physical plume mass. `b` is a scan
weight distribution, not `P(Z_j>tau)` and not a joint plume-containment probability.
Its uncertainty dependence is already present in `s_j phi(d_j)`.

For nonuniform tracking, sort physical cells by descending `m_j`, break exact
ties by row-major index, and take the smallest prefix with cumulative `b >= 0.90`.
Extend that prefix if necessary to the declared minimum sample count. Mechanical
conversion preserves **every** selected cell. No smoothing that removes cells,
previous-ROI restriction, mandatory shrink factor, or budget-based subset packing
belongs to `mass90_column_v1`.

Apply the recorded normalized-mass tolerance to cumulative threshold comparisons
as well as the final coverage check; floating-point `0.35+0.30+0.25` must not
force an extra cell solely because its representation falls just below `0.90`.

The retained predicted fraction `rho(S,H)` sums `b_j(H)` for physical quadrature
centers inside the union of the planned **acquiring column sweep** footprints.
Repositioning is excluded. Geometric rasterization uses this fixed lattice, not
display pixels. Separately report requested cells, geometric sweep footprint,
midpoint sample footprints, and reading count. This discrete coverage proxy is
not a claim that every location was independently measured or that all those
readings represent the field at the common forecast time.

## 5. Exact column policy and constraint handling

The [instrument contract](instrument-profile-v1.md) fixes legal sample locations,
beam size, cadence, actual sweep speed, travel model, and integration. For each
selected column, close vertical gaps from the lowest to highest selected row
and acquire every intervening row. Columns without selected cells are traversed
only as required for repositioning. Choose the nearer horizontal end of the
active columns (left on a tie), then visit them monotonically. At each column,
start from the nearer vertical endpoint (top on a tie). Reposition horizontally
then vertically. Charge the initial move, skipped columns, all vertical travel,
settling specified in the profile, and acquiring passes.

For a legal constant-speed column profile with vertical pitch `p_y`, actual
sweep speed `v_y`, commanded cadence `f`, and integration `e`, require
`f = v_y / p_y`, `f <= detector_capacity_hz`, and `0 < e <= 1/f`.
A span of `n` rows takes `n*p_y/v_y = n/f` seconds, with midpoint exposures at
half-slot offsets. The compiler must reject inconsistent settings instead of
using `max(travel_time, readout_time)` to silently lower speed. Reposition and
settling remain additional charged time. The separate point-scan profile uses
travel plus declared fixed dwell slots; no column timing claim transfers to it.

### Decision procedure

The policy configuration explicitly records `gamma=0.90`, minimum samples,
iteration limit (v1 default 5), time tolerance (default `1e-8 s`), mass tolerance
(default `1e-10` in normalized mass), uniform-density tolerance (default `0.02`),
`mass_floor`, `stale_after_seconds`, and `max_prediction_error_fov`. The last
three have units and must be chosen before an experiment; there are no hidden
hardware or reacquisition-time defaults. Record startup mode and the initial
scanner pose. All variants in a comparison share these safety/resource settings.

1. At decision time `t`, let `B` be the permitted duration for this cycle, capped
   by the remaining common monitoring/budget horizon. Startup may explicitly
   request a full-scene sweep; if that does not fit, execute a labeled reacquisition
   portion as below. A configured prior startup may begin tracking immediately.
   Record `startup_full_scene` separately from tracking. Partial startup remains
   incomplete until every lattice cell has a valid visit since session start;
   continue age-based portions until then rather than claiming a completed full
   scene from the first partial sweep. There is no startup completion guarantee
   when the budget or missing data prevents it.
   A nonpositive remaining budget or minimum sample count exceeding the lattice
   capacity returns `no_legal_action`; invalid numeric parameters reject before
   forecasting. Apply the recorded time tolerance to budget comparisons.
2. Forecast without hypothetical readings to `t+B`. Use reacquisition when total
   expected signal is at/below `mass_floor`, expected signal density is uniform,
   any cell is overdue, or the declared wind-error displacement bound
   `error_speed * B >= max_prediction_error_fov`. Uniform means the spatially
   weighted coefficient of variation of `e_j` is at most `0.02` (or the recorded
   tolerance), not that unequal quadrature weights are identical. Untouched cells
   become overdue after `stale_after_seconds` from session start.
   Both wind-error quantities use Euclidean distance in normalized FOV coordinates;
   an external wind measurement must supply a justified mapped error bound.
3. Otherwise initialize `H=B`. On each bounded iteration, forecast to `t+H`,
   select the 90% prefix, compile its full column closure from the current pose,
   and obtain duration `D`. Re-evaluate that path's mass at **its own** completion
   `t+D`. Save the candidate, its selection horizon, duration, retained fraction,
   and constraint checks. Set `H=D` for the next iteration. If a later forecast
   is at/below the mass floor, stop with that reason and use reacquisition.
4. Stop on identical requested cells and duration to the preceding iteration,
   time fixed point `abs(D-H) <= tolerance` (using the incoming `H`), a repeated
   nonconsecutive signature (cycle), or the iteration limit. A candidate is valid
   only if it meets the actual-duration budget, minimum readings, and mass target
   at its own completion time. Choose the shortest valid visited candidate,
   breaking ties by the lexicographic requested-cell tuple. This is bounded
   candidate selection, not a proof of globally optimal coverage or travel.
5. For the returned candidate, independently check whether recomputing the
   selection at its duration yields the same requested cells. Report this as
   `converged`; preserve iteration-limit/cycle diagnostics even if a valid
   nonconverged candidate is executable. If no visited candidate is valid,
   return an explicit tracking shortfall with each failed constraint, then use
   the declared reacquisition fallback. Do not delete selected cells and retain
   the original target label. An oversized candidate is diagnostic, not an
   executable plan. No result asserts global infeasibility from this search.

### Age-based reacquisition

Maintain, per physical lattice cell, the most recent valid reading midpoint
whose actual pointing lies in that cell. A wide footprint does not fabricate
independent cell visits. For priority, never-visited cells rank before visited
cells; for the overdue trigger their age is time since session start.

Rank columns by their oldest cell timestamp, oldest first (never visited is
older than any timestamp), then cyclic column order starting at rotation `r`.
Start with an empty mask. In that priority order, try adding each **whole-height
column** and compile the full trial from the current pose. Keep the addition
only when its actual duration fits `B`. Consider each column once; never shorten
a column or change cadence/footprint to make it fit. Execute the resulting plan
only if it meets the minimum reading count. Otherwise return `no_legal_action`
with no active scan. Advance `r=(r+1) mod number_of_columns` after each completed
execution; update ages only from actual valid readings.

Every fallback uses the same travel and acquisition accounting. Its status is
`reacquisition`, accompanied by the trigger, any tracking failure, and its own
numeric mass retention (or null). Even if that value exceeds 90%, it is not a
successful converged tracking decision. Reacquisition is a budgeted coverage
schedule, not guaranteed unknown-source discovery or recovery from missing data.
An age threshold shorter than achievable scene-refresh time can force continual
reacquisition; flag this configuration limitation instead of weakening the rule.

## 6. Confidence diagnostic; allocation extensions remain separate

Keep the baseline unchanged by the diagnostic. At the chosen candidate's
completion time, form a simultaneous Gaussian band over the **fixed physical
quadrature grid** with `N` points:

\[
\ell_j=a_j-\kappa s_j,\quad u_j=a_j+\kappa s_j,\qquad
\kappa=\Phi^{-1}(1-\delta/(2N)),\quad\delta=0.05\ \text{by default}.
\]

For the fixed retained set `R`, let

\[
L_{in}=\sum_{j\in R}w_j(\ell_j-\tau)_+,\quad
U_{out}=\sum_{j\notin R}w_j(u_j-\tau)_+,
\quad c=\frac{L_{in}}{L_{in}+U_{out}}.
\]

The union bound needs no independence assumption. For positive total signal,
`c` is a conservative retained-fraction lower bound on the simultaneous band
event. Report null if its denominator is zero, and report whether total signal
is demonstrably above `mass_floor` using the sum of all lower positive bounds.
An uninformative bound is `not_certified`, not proof that coverage fails. Use
this single specified construction in v1; no posterior-draw selection tuning.

The band applies at one time to the finite modeled field, not continuously along
the sweep or jointly over all decisions. Profile/grid resolution and posterior
misspecification affect it. A 95% model-conditional band is not a hardware
coverage guarantee. `gamma` is a signal fraction; `delta` is a posterior tail
parameter. Keep both distinct in reports.

UCB scores, boundary sampling, and reconstruction-variance reduction are
separately named allocation experiments described in the
[research appendix](../research/confidence-bounds-scan-allocation-appendix.md).
They do not change the estimate, likelihood, or baseline target. Any supplementary
scanning consumes the same resource budget. Do not transfer a bandit regret
guarantee to this reconstruction objective.

## 7. Evaluation and falsification

Use a predeclared spatial evaluation grid/weights and physical time mesh shared
by all policies. Approximate `J` by left-endpoint rectangle quadrature of causal
estimates at `t_0,...,t_{K-1}` with weights `t_{k+1}-t_k`; include prior-only startup,
receipt latency, idle periods, and evolution between sweeps. Declare mesh
resolution and check temporal/spatial convergence on development fixtures before
freezing an evaluation suite. Report `J/T` with squared-enhancement units as well
as `J`; never normalize by the number of completed scans or only successful ROIs.
Predictions after a finite truth source ends have unavailable truth, not a reused
last frame. Choose `T` within common available truth; otherwise mark the run
incomplete and do not compare its truncated loss to full-horizon losses.

Secondary metrics are signed spatial-mean enhancement error, expected/actual
positive-signal coverage, uncertainty calibration, reading age/acquisition-time
spread, missed signal, actual elapsed/travel/acquisition time, sample count, and
fallback/shortfall frequency. A spatial mean or integral is a **signal index**
until physical geometry establishes a stronger interpretation. Truth-free runs
report reconstruction/strength error as unavailable, not zero.

For the initial strength-related index use `I(t)=sum_j w_j*x_j(t)` and its
estimate `sum_j w_j*hat_x_j(t)`. Score their squared difference with the same
time quadrature as `J`; do not call this signed mean enhancement an emission
rate or replace the primary map error with it. Equal spatial means can conceal
incorrect plume shape and position.

Compare `mass90_column_v1`, a fixed declared source corridor, and the same
age-based rescan schedule under identical instrument/model/startup settings and
common budget ceilings and monitoring horizons. A full-scene feasible rescan is
an additional reference. Source-free corridor geometry cannot use a hidden
source. Use the same continuous scenario and causal wind realization; adaptive
paths naturally obtain different readings. Noise coupling/seeds are frozen and
recorded. Use **identical recorded readings** only for estimator or rendering
replays, not by forcing different policies to pretend they sampled the same
locations. Report actual expenditure; common ceilings are not exactly matched
costs.

Development cases include continuous clean signal with known wind, wind error,
source-free startup, moving/expanding plumes, and source exhaustion. First verify
mechanical/model conformance, then run the comparison. A claim of adaptive benefit
is falsified on a declared suite when reconstruction loss does not improve under
the stated comparison criterion, or a supposed gain relies on extra resources,
future evidence, or changed instrument settings. Do not tune against the frozen
historical held-out data. Both historical benchmark and recovery development
results remain `not_demonstrated`.

## 8. Freeze and extension rules

The objective, evidence semantics, likelihood/dynamics class, mass definition,
selection/closure/order rules, bounded-search behavior, fallback, diagnostic, and
evaluation definitions above are frozen v1. Numerical instrument, model, and
policy configurations must be explicit and immutable per experiment. A change
to an algorithmic rule requires a new named policy/formulation version; changing
a declared parameter creates a new configuration hash, not a silent retuning.

No hardware reference is selected by this document. Hardware calibration,
correlated retrieval-error models, extended exposure likelihoods, spatially
varying transport, source inference, 3-D inversion, physical emissions,
networking/drivers, and within-sweep replanning remain extensions. Renderer
changes are limited to verified semantic/label defects when implementation
proceeds; no redesign is required by this specification.

Mathematical background: [Gaussian conditioning](https://gaussianprocess.org/gpml/chapters/RW2.pdf)
and [Bayesian filtering](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
The [research synthesis](../research/adaptive-beam-plume-formulation.md) records
the application literature and its limits. These sources motivate components;
they do not validate this particular scan policy.
