# Source assumptions: a worksheet-aligned explanation and demo

## Start with the problem

**If our assumptions about the methane source are wrong, how does that affect
our interpretation of the measurements?**

For the first demo, make that question concrete:

**Can an incorrect source-location hint distort the methane map, even when the
sensor readings are unchanged?**

This belongs to the worksheet's statistical sensor policy and belief-update
workstream. Source identification supplies information to this component; the
current exercise tests how that information affects reconstruction. It does not
replace the source-identification team's work or establish a better scan policy.

## Three terms to explain before showing results

- **Prior:** our estimate and uncertainty before incorporating a new reading.
  For source location, this could mean "the source might be anywhere" or "it is
  probably near this piece of equipment." For the plume field, it is the map we
  predict before receiving the next reading. These are different quantities.
- **Posterior:** the updated estimate and uncertainty after incorporating the
  reading. A new observation can change the field map and, when the model allows
  it, the relative plausibility of different source locations. It does not
  automatically make either estimate correct.
- **Injection:** the source term in the model: newly added simulated signal near
  an assumed source during each prediction step. It is not a new sensor reading,
  not a Bayesian probability, and not a calibrated physical emission rate.

The general loop is:

```text
Previous estimate + source/wind assumptions
                  |
                  v
Predict the map before the next reading (prior)
                  |
           add the sensor reading
                  |
                  v
Update map and uncertainty (posterior)
                  |
                  v
Choose the next scan -> execute -> receive the next reading
```

The demonstration freezes the recorded scan path and changes only the assumed
source location. The two hints are fixed point assumptions, so their locations
cannot update: the field maps update, but this demo does not animate a source
posterior learning its way from the wrong hint to the correct source. The maps
show posterior/predictive means, not the complete uncertainty distribution.


## Run the visual demo

From `PlumeScanner/`, after installing `.[dev,plots]` as described in the README:

```sh
python scripts/demo-source-assumptions.py --output outputs/source-assumptions-demo
```

This generates and replay-checks the existing 40-second demo, then feeds its same
readings to two field estimators. To use an already saved record:

```sh
python scripts/demo-source-assumptions.py \
  --record outputs/2026-10-28-baseline-01/runs/demo.json \
  --output outputs/source-assumptions-replay
```

For an uninstalled source checkout, set `PYTHONPATH=src`. Choose a new output
directory for each invocation. PNG, GIF, numerical frames, copied evidence and a
hash manifest are saved locally. MP4 is also generated if FFmpeg is available
through the system or the optional `imageio-ffmpeg` package. The GIF requires
only the plotting dependencies. Playback shows 40 simulation seconds in about
20 seconds; it does not alter measurement timing.

What to point at:

1. Left: simulated truth, available only to the evaluator.
2. Middle: the map inferred with a correct fixed source hint.
3. Right: the map inferred with an offset fixed source hint.
4. Bottom: the identical delivered scalar readings used by both estimators.

The first frame starts with the same empty estimated maps. As the simulator and
filter evolve, injection at the assumed source and sparse sequential readings
both influence the estimates. The correct-hint map is not perfect because other
model discrepancies remain. The full scan takes about 33.6 seconds; the video
must not be presented as an instant full-scene measurement.

## Suggested three-page presentation

**1. Problem and role**

- We use sequential readings to estimate a changing methane map.
- My question: can incorrect source information distort that map?
- I am testing the estimation step before changing scanning decisions.

**2. Explain the loop and show the demo**

- Prior = what we expect before a reading.
- Posterior = what we estimate after using the reading.
- Injection = new simulated signal added near the assumed source.
- Same readings; correct versus incorrect source hints; compare the maps.

**3. Finding and next step**

- The same sensor readings can produce different maps under different source assumptions.
- In this saved example, the incorrect fixed hint increased time-averaged map
  error by about 52%; this is one synthetic example, not a general performance claim.
- Next: test how to use uncertain source information without trusting it too strongly.
- Team question: what source information can we obtain, and how reliable is it?

Put the detailed model-consistency and width/amount plots in backup. They explain
follow-up diagnosis after the audience understands the map-reconstruction task.

## Evidence and code map

- Existing demo: `scripts/run-scanner-demo.py`,
  `configs/evidence-gated-focus/coarse-demo.json`.
- Field estimator with source hypotheses: `src/plume_focus/source_prior_belief.py`.
- Same-evidence comparison: `scripts/compare-source-priors.py` and
  [its review](reviews/source-prior-fixed-evidence-v1.md).
- Matching-model control: `scripts/model-consistent-source-priors.py` and
  [its review](reviews/model-consistent-source-priors-v1.md).
- One-factor controls: `scripts/one-factor-source-mismatch.py` and
  [its review](reviews/one-factor-source-mismatch-v1.md).
- Conserved-width controls: `scripts/conserved-width-source-control.py` and
  [its review](reviews/conserved-source-width-v1.md).
- Presentation observer: `scripts/demo-source-assumptions.py`.

The numerical findings are from the frozen October 28 local runs. Their archived
configs refer to ignored local evidence and SHA-256 hashes; cloning the code does
not download those artifacts. The visual demo above is self-contained. To rerun
a research study with newly generated evidence, copy its config, replace the
record path and SHA-256 with those of the new record, and record new provenance;
do not edit the archived configs or silently describe a new record as the frozen
one. Dependent refinements also need new parent-output paths. Code and protocols
are versioned; generated videos, arrays and chat/worksheet documents stay local.
