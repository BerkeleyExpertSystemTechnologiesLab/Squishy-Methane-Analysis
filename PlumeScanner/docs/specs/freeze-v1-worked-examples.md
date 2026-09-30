# Worked checks for the v1 freeze

These are mathematical and policy-definition examples, not new plume-performance
experiments. The standalone [arithmetic script](../../scripts/check-policy-v1-spec.py)
does not import the production planner or certify its conformance. Run:

```powershell
.\.venv\Scripts\python.exe scripts/check-policy-v1-spec.py --output outputs/policy-v1-freeze/worked-checks.json
```

The [synthetic 4 x 4 instrument](instrument-profile-v1.example.json) has a fixed
0.125 x 0.125 FOV footprint, vertical sweep/transit speed 0.25 FOV/s, horizontal
transit speed 0.5 FOV/s, 1 Hz commanded cadence, 2 Hz capacity, 0.5 s integration,
and zero settling. These values make arithmetic simple; none is a proposed
reference hardware value. Starting pose is `(0,0.125)` unless stated otherwise.

## Bayesian updates and evidence timing

With zero mean, `P=[[1,0.6],[0.6,1]]`, `h=[1,0]`, reading `2`, and noise variance
`0.25`, the gain is `[0.8,0.48]`. The posterior is

```text
mu = [1.6, 0.96]
P  = [[0.2, 0.12], [0.12, 0.712]]
```

The neighbor changes through covariance. Replacing the prior covariance with
the identity leaves the unobserved cell unchanged. Starting instead at means
`[1,1]` and measuring zero gives means `[0.2,0.52]`; zero is useful evidence,
not missing data.

Two independent scalar readings of `2`, each with noise variance `0.25`, from
prior `N(0,1)` give posterior mean `16/9` and variance `1/9`. Re-delivering one
reading's identity must instead keep the one-reading mean `1.6` and variance
`0.2`. The existing controller duplicate/conflict test checks delivery behavior;
the arithmetic script checks the independent-observation result.

A single footprint `h=[0.5,0.5,0]`, identity prior covariance, reading `2`, and
noise variance `0.5` yields means `[1,1,0]` and covariance
`[[0.75,-0.25,0],[-0.25,0.75,0],[0,0,1]]`. Treating its two covered cells as two
independent readings would produce a different, unjustified posterior.

The dynamic check uses three unit-time transitions with
`F=[[0.9,0.1],[0,0.8]]`, `Q=0.02 I`, the correlated prior above, a reading at
time 1 with `h=[1,0], y=1.5, r=0.25`, and one at time 2 with
`h=[0.25,0.75], y=0.2, r=0.4`. It compares sequential filtering/prediction with
an independent batch calculation from the augmented joint prior of
`(X_1,X_2,X_3)`. At time 3 both give

```text
mu = [0.9603247388, 0.2576961593]
P  = [[0.1737410864, 0.0569989739],
      [0.0569989739, 0.1470044830]]
maximum absolute difference = 2.22e-16 on the verification environment
```

Equal completed posteriors do not imply equal online reconstruction. In a
one-cell illustrative field with truth `1`, a prior mean of `0` and a delivered
reading that changes it to `0.8`, the causal squared-error integral on `[0,4]`
is `1.12` for delivery at time 1 and `3.04` for delivery at time 3. Their final
means are identical. A batch must not repaint its improved estimate into earlier
times when computing the primary score.

## Region stability, change, and physical closure

Indices below are row-major on the fixed 4 x 4 physical lattice. Listed values
are normalized predicted masses at the candidate reference time; other cells
have zero. These isolate selection semantics, not calibrated plume dynamics.

| Case | Nonzero masses | Required selected cells |
| --- | --- | --- |
| Stationary | `0:0.55, 4:0.35, 15:0.10` | `[0,4]` on every identical decision |
| Expansion | `0:0.35, 4:0.30, 3:0.25, 15:0.10` | `[0,3,4]` |
| Return to earlier field | Same as stationary | `[0,4]`; shrink is allowed when justified |
| Displacement | `3:0.55, 7:0.35, 0:0.10` | `[3,7]`; previous ROI is not a restriction |

The same input does not repeatedly trim the selection. The normalized-mass
tolerance prevents floating-point representation of `0.35+0.30+0.25` from
spuriously selecting a fourth cell.

For `0:0.23, 3:0.23, 12:0.23, 15:0.23, 6:0.08`, the selection contains the four
corner cells and 92% mass. Column closure fills rows between the selected
endpoints, yielding eight readings in two full columns. The acquiring passes
take `4+4=8 s`; moving from column 0 to column 3 costs `0.75/0.5=1.5 s` even though
intermediate columns are not sampled. Total: **9.5 s**. Readings occur at
`0.5,1.5,2.5,3.5,6,7,8,9 s`, with fixed 1 Hz within each pass.

A 6 s budget therefore produces a tracking shortfall. Trimming to one of these
columns would retain only 46% mass. The v1 policy keeps the 9.5 s candidate as a
diagnostic, does not execute it over budget, and may execute a separately labeled
reacquisition fallback. Changing a display from 120 x 160 to 480 x 640 does not
enter this calculation. Doubling the commanded sweep speed without changing
cadence violates the instrument relation and is rejected; the compiler must not
silently reduce speed back to the old value.

## Budgeted reacquisition and uniform uncertainty

With a 4.6 s reacquisition budget, initially unvisited cells, and the same scanner,
the first four completed reacquisitions visit columns `[0,1,2,3]`. Their durations
are `[4,4.5,4.5,4.5] s`: **17.5 s** total, including 1.5 s horizontal travel.
No beam enlargement or faster sweep creates this coverage. Smaller budgets can
prevent a full column or the necessary repositioning; the policy reports that
limitation rather than promising scene recovery. The independent script checks
this simple all-valid-reading schedule; missing readings and partial receipts
need production conformance checks.

For uniform marginals `N(0,0.2^2)` and threshold zero, each cell has expected
positive signal density `0.0797884561`. The means being zero does not make
expected positive signal zero. This field triggers the uniform-density
reacquisition rule; it is not high confidence of an everywhere-present plume.

## Expected fraction and posterior confidence

Let cell A equal `10` exactly, cell B be `N(1,1)`, threshold zero, and use equal
spatial weights. Selecting A retains **90.225709% of expected positive signal**.
The probability that A retains at least 90% of the realized positive signal is
only **54.423588%**, since the event is `B <= 10/9`.

The frozen diagnostic uses a simultaneous two-sided band on both cells, with
`delta=0.05`, `N=2`, and `kappa=2.241402728`. Its conservative retained-fraction
bound is **75.520700%**, so it does not certify 90% retention. The appendix's
approximately 79.1% value uses a different, one-sided bound on the single omitted
cell; neither value is an allocation change or a hardware guarantee.

## Existing integration coverage and open checks

The full existing suite at the freeze starting implementation passed **335 tests
with 5 Windows capability skips**. Its 28 sensing cases include sequential/batch
conditioning, duplicate conflicts, delayed open-scan replay, path locking, actual
completion overruns, finite source exhaustion, serialized messages, fake-scanner
equivalence, truth isolation, physical travel/readout, and renderer independence.

Those tests preserve current behavior. They do not establish the new
coverage-preserving allocator, required profile loader, common-time reconstruction
evaluator, explicit age trigger, or confidence diagnostic. The
[handoff](../reviews/policy-v1-freeze.md) lists the needed conformance tests and
does not count the standalone arithmetic examples as production acceptance.
