# Source input and representation audit v1

## Purpose and reference

This is the first stage of a location/strength-confounding experiment. Establish
the meaning and numerical representation of source input before changing inference.
Baseline: `plume_scanner_new` at `5cac633f47001d3aa455f8b6f3cc00b11c550ed3`;
implementation starts from `1043f5856d795e4b36e4e9bc41da401b348b05a3`.
Settings are frozen in `configs/source-prior-study/source-input-audit-v1.json`.
The baseline's source formula, estimator and archived experiments remain unchanged.

## Three input conventions

The historical source is `peak * exp(-distance^2 / (2 sigma^2))`, sampled at
grid-cell centers. Its spatial mean is the discrete input integrated over the
unit FOV per second. At fixed peak, that total depends on width, grid alignment
and clipping at the FOV boundary.

Two opt-in alternatives use a fixed total `Q` in synthetic signal times normalized
FOV area per second. For equal-area cells, `mean(input_vector) = Q`:

1. `point_sampled_fov_normalized_v1` divides the historical sampled shape by its
   discrete mean and multiplies by `Q`. This separates amount conservation from
   the choice of spatial representation; center-sampling aliasing remains.
2. `cell_integrated_fov_normalized_v1` uses the product of normal-CDF differences
   at each cell's edges. Divide cell probability by its area and by the Gaussian
   probability inside the FOV, then multiply by `Q`. Integrating adjacent fine
   cells reproduces coarse-cell source mass up to roundoff.

For both alternatives the total means input *inside* the FOV. Near a boundary,
outside Gaussian mass is redistributed inside. This convention is useful for an
amount-controlled synthetic experiment; it is not a claim about physical methane
emission or an open-boundary source. Use interior sources for the initial main
experiment and retain boundary cases as separate diagnostics.

Choose `Q` once as the baseline source vector's spatial mean at its original
position/width/12x16 grid. This gives a convenient scale, not physical calibration
or equal source shape. Total input is not the evolving field's mass, observed
signal, signal-to-noise ratio or a calibrated gas emission rate.

## Audit and frozen checks

Compare the original and both alternatives over the declared widths, named
source positions and interior sub-cell phase sweep. Report totals and source-input
centroids. For the historical input, also compare total against the analytic
continuous Gaussian integral truncated at the same FOV. This separates center
sampling from the genuine change in retained Gaussian mass at a boundary.

New-profile relative total error must be below `5e-13`; integrated coarse/fine
cell-mass L1 discrepancy relative to `Q` must be below `1e-12`. Report centroid
quadrature error against the analytic truncated Gaussian centroid, without
requiring it to vanish on an under-resolved grid. This is not an inferred
source-location error or an uncertainty-calibration test.

The integrated input alone does not make the existing advection/interpolation
operators a finite-volume solver. It does not fix transport, observation or
field representation errors. Grids 48x64 and 96x128 probe only source input;
they are not runnable covariance grids for the current limited-state estimator.

## Following stage

Use explicitly named source inputs consistently in a matched generator and
candidate estimators; hold wind, width, decay and the acquisition schedule fixed.
Compare location-only, joint location/amount, and oracle-amount controls on shared
evidence. That inference experiment is separate from this input audit.
