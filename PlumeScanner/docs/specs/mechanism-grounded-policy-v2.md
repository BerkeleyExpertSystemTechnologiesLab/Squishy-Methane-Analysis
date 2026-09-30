# Mechanism-grounded sensing revision: experimental v2

Status: **implemented experimental candidate; development comparison complete**.
This is a separately named revision; it does not rewrite frozen v1 or establish
a hardware profile, calibrated uncertainty, or an adaptive performance claim.
The [completed review](../reviews/mechanism-revision.md) records mixed results,
weak-signal limits and unresolved formulation decisions. This is not a new freeze.

## Mechanisms before symbols

The instrument reports one retrieved beam-integrated scalar at a time. It can
point in two axes and acquire a vertical segment between chosen endpoints.
A column scan does not require either endpoint to be on the FOV boundary.
Beam dimensions, pitch, commanded cadence, integration and speeds are fixed
within an experiment. Only the next physical region/path changes.

The reconstruction remains a signed enhancement field at physical time t,
conditioned on readings delivered by t. Keep the existing Gaussian observation
and advection/diffusion mathematics in this first revision, so changed allocation
can be isolated from changed reconstruction physics. Unknown concentration along
the beam, real retrieval calibration, finite plume lifetime and uncertain angular
wind mapping remain limitations. Do not invent a decay rate to improve the demo.

Three different sources of uncertainty must be named: actual instrument/retrieval
error injected by a simulator or supplied by hardware; assumed observation error
R used in conditioning; and evolution error Q/prior uncertainty used in prediction.
They are independently configured. Reducing Q merely to obtain a tight region
would make uncertainty unjustified, even if it made scanning look successful.

## Evidence, actionable signal and surveillance

Maintain the Gaussian posterior `(mu, P)` exactly as in v1. Preserve signed
means and distinguish missing readings from valid zeros. All posterior outputs
and the primary full-field squared reconstruction loss remain available.

Introduce a separately declared **minimum reportable enhancement** `a`, in the
retrieved scalar's units, above the background correction already used by the
estimator. For tracking use

`m_j(a) = w_j E[(X_j - a)_+ | delivered evidence]`.

Select the descending 90% prefix of these weights, then preserve every selected
cell through column closure. This means 90% of the model's expected excess above
the reportable threshold. It is not 90% of all gas, a probability of containing
the plume, or an emission estimate. The old zero-threshold mass stays a separately
reported diagnostic. A null or nearly uniform decision surface causes surveillance.
Tracking also requires at least one predicted mean above a (equivalently, a
Gaussian marginal exceedance probability above one half). Otherwise return an
unavailable tracking ratio instead of normalizing a field made only of small
uncertainty tails. This is an explicit action-readiness rule, not a declaration
that no plume exists. The zero-threshold ablation retains v1 normalization.
The fixed-corridor comparator executes its declared region without this adaptive
signal-readiness or mass gate; it shares periodic service and physical constraints.

The reason for making a explicit is mechanical: at zero mean and zero threshold,
each uncertain background cell contributes `sigma/sqrt(2*pi)`. Covering 90% of the
sum can therefore become a background survey regardless of measured plume evidence.
A signal reporting threshold suppresses weak excursions without deleting their
posterior uncertainty or claiming the cells were observed empty.

The initial numerical fixture fixes `a=0.15` synthetic concentration-path units,
three times its declared independent blank-reading standard deviation 0.05.
For an ideal zero-mean Gaussian blank this corresponds to a one-reading upper-tail
probability about 0.00135. This is a synthetic blank-response convention, **not**
a calibrated plume detection probability or multiple-comparison guarantee.
Real hardware needs background/path correction, blank and known-signal response,
retrieval lag and detection/reporting requirements before selecting a. The value
is declared before closed-loop comparisons; 2/4-sigma threshold sensitivity is
diagnostic and must not be used to choose the best evaluation result.

This retains the Gaussian expected-excess definition instead of asserting that
uncertainty has no measurement value. It can still over-allocate if Q is wrong,
or miss weak diffuse plumes below a. Full-field reconstruction error and weak-signal
stress cases must expose those failures. No UCB or lower-bound allocator is added.

## Legal two-axis actions and clocks

Tracking keeps v1 descending-prefix, deterministic column ordering and bounded
own-completion forecasting. It may change horizontal support and vertical endpoints.
The compiler charges acquisition once, plus nonacquiring transit, settling and
detector recovery. A failed full tracking selection is retained in diagnostics;
it is never relabelled as a smaller successful tracking selection.

Surveillance uses fixed-height **vertical blocks within columns**, not whole
columns. The initial block height is 0.20 FOV: four rows at 20 x 32 or eight at
40 x 64. Edge blocks clip to the lattice. Blocks are ordered by oldest actual
cell visit, then cyclic block order. Try each once; keep additions whose complete
compiled closure fits the actual remaining budget. Adjacent/gapped blocks in a
column use the existing compiler's contiguous closure, and that extra travel and
acquisition is charged. The resulting mask may grow in both directions. Actual
visits alone update ages. No widened beam or fictitious peripheral observation.

An overdue cell does not force every subsequent scan into all-background service.
Instead a declared wall-time surveillance interval (initially 30 s) limits how
long tracking can run between surveillance decisions. Due surveillance executes
one charged block schedule; completion resets that service clock. This is a
periodic service rule, not a guarantee that every cell is refreshed within 30 s.
Report full age distribution and unmet refresh requirements. No meaningful signal,
uniform decision mass, wind-error displacement, or failed tracking also invokes
the same two-axis surveillance action. No action is explicit when no block fits.

The previous ROI is not the next decision's search domain: each mass selection
uses the entire physical lattice. There is no recursive multiplication of ROI
area by 0.9. Nevertheless, selective observation can reinforce an incorrect small
region, and a four-reading floor alone does not protect reconstruction. Record
successive requested/swept areas, expansion and contraction, minimum-size visits,
surveillance readings outside the last tracking region, and unvisited/old cells.
Test that new evidence can grow or move the region. Alternating sizes can also
indicate unstable forecasting; neither oscillation nor small area is a success metric.

Plans remain fixed until completion. Delivery calls, duplicates, delayed open-scan
replay, actual overruns, and the public three-operation lifecycle are unchanged.
Full-scene startup, when requested, still requires all cells. Prior startup permits
surveillance/tracking without that prerequisite; it must not claim full-scene
coverage or source discovery. Compare policies under the same startup assumption.

## Instrument resolution and implementation

The 40 x 64 profile is a user-requested **tentative physical fixture**. With the
same 20 Hz capacity, vertical acquiring speed must be 0.5 FOV/s to retain 1/40
sampling pitch; a full column takes 2 s. The profile explicitly chooses the same
vertical transit speed. Horizontal speed remains 0.625 FOV/s. The detector is not
silently accelerated to keep the original video speed. Full acquisition alone is
128 s, before travel. Thus full-scene startup cannot fit a 75 s experiment.
Display and inference resolution remain independent. Do not call this real hardware.

Repository mapping:

| Mechanism | Ownership |
|---|---|
| Blank/retrieval noise vs assumed R vs Q | explicit v2 run/session configuration; simulator injects its own noise value |
| Signed current field and uncertainty | unchanged `signal_belief` |
| Reportable excess and two-axis decisions | new `sensing_policy_v2`; preserved v1 module |
| Fixed beam and physical timing | existing `physical_scan` profile compiler |
| Last surveillance receipt, actual visits and causal lifecycle | existing `controller` with a v2 dispatch |
| Evidence-only replay, true field used for scoring | existing adapters/evaluator with versioned v2 configuration decoding |
| Observer presentation | existing six panels; only threshold/policy labels change |

Policy identifiers distinguish threshold+segment revision, zero-threshold+segment
ablation, fixed corridor with shared segment surveillance, and pure segment rescan.
The main allocation comparison keeps observation/process/prior noise unchanged.
Model and signal threshold changes are not disguised as an allocation-only result.

## Bridge to theory and falsification

Gaussian conditioning supplies neighboring-cell influence and a causal current
estimate; it does not decide where to look. [GPML chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf)
supports this inference component. Mechanically legal scan segments and their
actual acquisition times supply the action set.

[FAST scanning microscopy](https://www.nature.com/articles/s41467-023-40339-1)
connects acquired evidence, reconstruction and physical travel. Its stationary
material experiments and learned utility do not prove performance for moving gas.
The transferred principle is to evaluate complete executable acquisition actions.

[Goal-oriented experimental design](https://arxiv.org/abs/1802.06517) connects
measurements to uncertainty in a scientific target. For weighted squared map
error, a Gaussian candidate observation reduces predicted covariance by
`C C^T / v`, and expected map risk by `C^T W C / v`. This explains why signal
amount and information value differ. The current 90% rule remains an interpretable
coverage surrogate, not an optimizer of reconstruction loss. A future one-sweep
risk calculation must value timestamps and redundant observations jointly.

Evaluate the same causal full-horizon J and J/T, signed mean-enhancement index
error, actual time, full-signal recall, reportable-signal recall, background
allocation, scan heights, tracking/fallback frequency and ages. Keep unknown truth
null. Include initial/idle time, amplitude sensitivity, wind error and source-free
operation. Comparisons share source/noise coupling, mechanics and ceilings; actual
expenditure can differ. Lower acquisition time alone is not success.

Falsify the proposed fixes if vertical focusing never executes, background weight
still dominates, surveillance starves part of the scene, weak plumes are discarded,
or lower scan time does not improve reconstruction. Preserve negative outcomes.
