# Source input and representation audit — 2026-11-01

## Outcome

The historical fixed-peak source changes discrete total input when its location
moves relative to the grid. On the 12x16 grid, the original width .035 FOV gives
discrete/analytic-continuous input ratios .934663–1.062822 over the declared
interior positions. At half width .0175, the range is .129145–2.640123. These are
source-vector quadrature differences before transport or observation, not inferred
source-location errors or attribution of the earlier reconstruction failure.

Two opt-in source-input conventions now separate amount normalization from spatial
representation. Both keep the in-FOV total fixed; cell integration also gives
consistent source-cell masses when fine cells are aggregated. They do not change
the baseline scanner, prove improved reconstruction, or establish calibrated
physical emissions. Joint location/amount inference is the following stage.

## Definitions

Follow the [frozen protocol](../specs/source-input-audit-v1.md) and
[`source-input-audit-v1.json`](../../configs/source-prior-study/source-input-audit-v1.json).
Baseline revision is `5cac633f47001d3aa455f8b6f3cc00b11c550ed3`; implementation
starts from `1043f5856d795e4b36e4e9bc41da401b348b05a3`.

The legacy vector samples a peak-scaled Gaussian at grid centers. Set a new total
`Q = 0.002305094376073324` once, equal to the legacy spatial mean at the original
source position, width and 12x16 grid. This is synthetic input integrated over
the unit FOV per second. Its numerical scale matches that one reference total,
without making its spatial shape identical or converting it into methane mass.

- `point_sampled_fov_normalized_v1` preserves the sampled shape and rescales its
  mean to `Q`.
- `cell_integrated_fov_normalized_v1` uses analytic Gaussian probabilities for
  each cell, divided by cell area and total in-FOV Gaussian probability.

Both alternatives condition source input on the FOV: outside Gaussian mass is
redistributed inside. Thus they intentionally retain `Q` near edges. A physical
open-boundary convention would be different. Keep boundary cases separate from
the initial interior-source inference experiment.

## Declared probes

Four input grids (12x16, 24x32, 48x64, 96x128), four widths (.5/1/1.5/2 times .035)
and six named positions give 96 named profiles. A 17x17 source-position sweep
inside [.35,.65]^2 gives another 4,624 profiles. The sweep samples alignment with
grid cells; these are not independent physical scenes or statistical trials.
Grid refinement compares each input grid with one doubled on each axis.
Grids above 24x32 are source-input probes only, not covariance-filter executions.

The legacy comparison uses the analytic continuous Gaussian integral over the
same FOV. It therefore separates quadrature error from actual boundary truncation.

| Width / baseline | 12x16 legacy total / analytic total, min–max | 24x32 min–max |
| --- | ---: | ---: |
| .5 | .129145–2.640123 | .934663–1.061994 |
| 1 | .934663–1.062822 | .999998–1.000002 |
| 1.5 | .999207–1.000755 | approximately 1 |
| 2 | .999998–1.000002 | approximately 1 |

For both normalized profiles, maximum relative error in total input over all
named and phase probes is `7.77e-16`, below the predeclared `5e-13` tolerance.
Maximum integrated coarse/fine cell-mass L1 discrepancy divided by `Q` is
`2.71e-16`, below the predeclared `1e-12` tolerance.

## Remaining representation error

Input centroid is computed using each cell's center and its input mass, and
compared with the analytic truncated-Gaussian centroid. This is a quadrature
diagnostic, not the estimated source location in a reconstructed field.

For half width on 12x16, the interior position sweep's 95th-percentile centroid
error is .029213 FOV for point normalization and .011531 FOV for cell integration.
On 24x32 these become .002850 and .000408 FOV. Neither makes a narrow source fully
resolved on the original coarse grid.

Cell integration is not uniformly better for every centroid diagnostic. At
source (.02,.02) with the original width, centroid error is .006320 FOV for point
normalization and .010292 FOV for integration; at the FOV corner it is .015182
versus .017150 FOV. Integrated cell *masses* are correct, but representing all of
a boundary cell's mass at its center can give a worse first moment. Preserve these
counterexamples. Do not convert an input consistency result into a general accuracy
claim or treat the existing transport as a finite-volume solver.

## Verification

49 targeted tests pass: 32 new source-input checks plus existing conservation,
source-bank and likelihood checks. Tests include independent numerical quadrature
of selected Gaussian cells, coarse/fine additivity, amount linearity, zero input,
interior/edge sources, invalid parameters and legacy normalized-shape agreement.
Compilation and whitespace checks pass.

An initial relative-only shape assertion failed for vanishing Gaussian tails
(~1e-150 and smaller; relative roundoff about 1.5e-13). The shape assertion now
also uses an absolute tolerance of 1e-15 relative to a unit peak. Conservation
and refinement tolerances were not changed.

The historical baseline source vector reproduces exactly. Its full configuration
digest remains `2c43534d49a0596c3fbeae372c90d01333d3a42492ffc1e82ae924b02b276446`.
Existing source formulas, model, controller, policies and archived evidence are
unchanged; only the new audit explicitly imports the new source-input module.

## Saved evidence and reproduction

Local artifacts are in `outputs/2026-11-01-source-input-audit-01/`: copied configs,
named/phase results, numerical phase arrays, verification, code/protocol snapshots,
execution log, provenance, hashes and two inspected figures. A separate closeout
records environment, tests and the producing commit. No new field or observation
histories are generated by this source-input audit.

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/plume-input-audit-mpl .venv/bin/python scripts/audit-source-input.py --config configs/source-prior-study/source-input-audit-v1.json --output outputs/<fresh-input-audit>
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_source_input.py tests/test_conserved_source_control.py tests/test_source_prior_belief.py tests/test_source_likelihood_quadrature.py
```

The next comparison should use the same declared source-input profile in a
matched truth generator and in the location-only, joint location/amount and
oracle-amount estimators. Keep wind, width, decay and the physical history fixed.
That experiment must separately assess field reconstruction, prediction and
uncertainty; this audit does not establish its outcome.
