# Three-estimator paired comparison v1

Completed locally on 2026-11-01, starting from `4b17d15`. This analysis derives
paired contrasts from the already executed [matched study](joint-source-amount-v1.md).
It generates no new readings, inference runs or scan policies. The original
103-file closeout manifest is verified before and after analysis.

## Method definitions

| Role | Location | Source input amount |
| --- | --- | --- |
| Existing-style control | Unknown; uniform spatial prior | Fixed at default Q0 |
| Candidate | Unknown; same spatial prior | Jointly inferred, uniform multiplier [.25,1.75] |
| Ideal-information control | Unknown; same spatial prior | Supplied the true scenario amount |

The existing-style control retains the fixed-amount assumption while allowing
unknown location. It is not the unchanged historical production estimator, whose
source assumptions differ, and it does not stand for the entire scanner baseline.
The ideal-information control knows amount only: source position, random field
perturbations and measurement noise are still unknown. It is not a zero-error
oracle or a guaranteed lower bound on each realized seed's error.

All methods receive the same 708 readings within each history and use the same
matched model, timing, field grid, footprint and frozen source-informed path.
The underlying experiment has three amounts, ten paired noise seeds per amount
and three additional deterministic controls. Statistics below use the ten noisy
histories; deterministic controls remain in the original study report.

## Reconstruction and the value of knowing amount

Primary score is mean latent-grid MSE at 36 delivered-reading checkpoints, not
physical-time-integrated J/T. Values below are means over the ten paired seeds.

| True amount | Fixed default | Candidate | True-amount control | Candidate change vs fixed | Candidate excess vs true-amount control |
| --- | ---: | ---: | ---: | ---: | ---: |
| .5 Q0 | .00258382 | .00175833 | .00174679 | -31.95% | +.66% |
| 1 Q0 | .00178416 | .00179999 | .00178416 | +.89% | +.89% |
| 1.5 Q0 | .00239813 | .00185166 | .00182286 | -22.79% | +1.58% |

Supplying the true amount reduces fixed-control MSE by 32.39% at low input and
23.99% at high input. Joint inference recovers 98.62% and 94.99% of those observed
reductions, respectively. This is `(fixed-candidate)/(fixed-oracle)` using scenario
mean MSE, not a percentage of explained physical error. It is undefined nominally
because fixed and true-amount controls are exactly identical there. Ratios are not
clamped and do not imply an oracle bound on individual realizations.

Candidate MSE is lower than fixed MSE on 10/10 seeds at low input, 1/10 nominally
and 10/10 at high input. It is lower than true-amount MSE on 2/10, 1/10 and 3/10
seeds, respectively. Descriptive standard errors and every paired difference are
saved in `comparison.json`; no new significance/generalization claim is made.

## Location and amount interpretation

| True amount | Fixed mean location error | Candidate mean location error | True-amount mean location error | Candidate mean amount |
| --- | ---: | ---: | ---: | ---: |
| .5 Q0 | .140881 | .013374 | .011909 | .544649 Q0 |
| 1 Q0 | .006757 | .007318 | .006757 | 1.040264 Q0 |
| 1.5 Q0 | .004714 | .005050 | .004738 | 1.538107 Q0 |

Location errors refer to the posterior mean at t=40 s in normalized FOV distance.
The candidate improves low-input localization in every seed. At high input it
slightly worsens mean location error while improving reconstruction in every seed.
The original no-realized-noise low-input control also displaces the fixed-amount
source toward the boundary. This supports possible compensation through position,
but does not imply all strength errors create large location shifts. Joint amount
recovery is not exact, and nominal input shows a small cost of estimating amount.

## Reproduction, checks and limits

```sh
MPLCONFIGDIR=/tmp/plume-amount-mpl PYTHONPATH=src OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/summarize-three-estimators.py \
  --input outputs/2026-11-01-joint-source-amount-01 \
  --output outputs/2026-11-01-three-estimator-comparison-01
```

Use a new destination if rerunning. Output includes copied fine-grid input scores,
analysis configuration with source hashes, per-seed contrasts, a comparison PNG,
the producer script, verification and a SHA-256 manifest. All input hashes remain
unchanged; method order/seed pairing and nominal fixed/oracle equality are checked.
Independent recomputation agrees with the original summary. The prior inference
run's 49 passing tests and 99/99 final refinement checks remain its verification;
this derived analysis does not rerun that experiment. Syntax and diff checks pass.

This is evidence for one source geometry, matched synthetic source definition,
specified priors and fixed source-informed path. It does not establish physical
emission estimation, robustness to other model errors, independent-scene accuracy,
source activity estimation or better next-scan selection. The apparent benefit of
true amount is an observed information-control comparison, not a universal upper
bound. No GitHub upload or slide preparation is part of this checkpoint.
