# Scanner baseline calibration

Declared 2026-09-19 from `b5b48e6`, branch `codex/scanner-baseline-calibration`.
Robotics and simulator selection are deferred. Preserve the full-scan/focus/full-
rescan loop, historical bindings, instrument settings and renderer layout.

Stage A: vertical margins 0/.05/.10 FOV per side, rounded outward to physical
rows, clipped at the boundary, applied to every ranked prefix before compilation.
Padding is charged and never removed. Rank/threshold/quantile and full fallback
are unchanged. Test both profiles, continuous/blank/cessation, noise 0/.05: 36
development executions. Blank retains the declared asset location and model source
assumption. Select the smallest nonzero margin with >=95% geometric bright-core
coverage during continuous focus intervals and aggregate continuous J/T <=105%
of margin zero. Require coverage separately per profile/noise case; no focus or
no core is unavailable, never a passing 100%. Otherwise retain zero.

Bright core uses clean evaluator truth strictly above max(.15, half instantaneous
peak). Geometric coverage is the time average of the fraction inside the committed
sweep mask at common quadrature times during focus. Actual readings are scored
separately using clean exposure values and instantaneous clean peak; sample age,
guard/background readings, misses, duration, latency and actual spending remain
separate metrics. New settings/session/plan messages are separately named; zero
margin/no decay reproduces historical paths and posterior values.

Stage B: freeze margin; replay its 12 development histories. Test half-lives
null (infinite)/3/6/12 s and source sigma .0175/.035 FOV. Fit a shared source rate
in [0,.30] by bounded least squares of causal integrated uniform field loss.
Mean is affine in source rate on fixed evidence: evaluate two endpoint means with
shared covariance. No other coefficients are fitted. For a transport step A,
attenuation a=exp(-lambda*dt), add source (1-a)/lambda and covariance process
(1-a*a)/(2*lambda), with continuous dt limits at lambda zero. Covariance transported
by a*A. Runtime and evaluator use the same step partition. Include unchanged
rate .15/no-decay/sigma .035 as an explicit selection candidate. Use 1e-10
absolute / 1% relative aggregate-loss numerical ties; favor unchanged, then longer
half-life, then original source width, then smaller rate. Report all near ties.

Freeze held-out manifests BEFORE calibration: profiles coarse/double; scenarios
continuous, wind error (.003,-.004) FOV/s, weak .3, blank with source assumption,
cessation at180, intermittent [0,120)/[240,360); seeds101/102 with simulated noise
.01/.05 respectively. Source phases from default_rng(seed).uniform(0,2*pi,3),
shared across scenarios at that seed. Controls are synthetic signal windows, not
a physical valve-shutdown model. Run baseline, frozen calibrated and repeated
full scans:72 executions of360s. Full-scan comparator uses the original estimator;
the baseline uses original settings. Shared truth/noise coupling, distinct actual
readings on different paths. No held-out retuning.

Evaluate with120x160/.05s; refine representative nonblank and blank executions
on each profile at240x320/.025s, then480x640/.0125s only if loss change exceeds
max(1e-10,1%). Report unresolved at limit. Controlled representation probes compare
12x16 with24x32; inference resolution stays fixed in executions. No posterior
coverage calibration or superiority claim follows from narrower scans.

Acceptance: arithmetic, zero-mode compatibility, padding/edge/cadence constraints,
delayed/duplicate evidence, overruns, immutable plans, source lifecycle, serialized
replay, truth isolation, display independence, full tests/imports/compilation/diff.
Review existing-layout physical and summary videos. Document whether blank stays
FULL, continuous sustains focus, and cessation followed by a background full scan
avoids false refocus. Finish honestly if calibration fails; no additional source
state inference, resets or threshold changes. Freeze hashes, commands, revisions,
artifacts, results and synchronized project documentation; commit and push.

Completed result: see [the review](../reviews/scanner-baseline-calibration.md).
Margin tuning fails and retains zero. The fitted model improves reconstruction in
the declared tests but fails blank/cessation behavior; it remains an experimental
opt-in binding. This closeout does not alter the predeclared protocol above.
