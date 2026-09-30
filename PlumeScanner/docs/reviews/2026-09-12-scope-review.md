**Repository scope review — May versus July 2026**

Reviewed 12 September 2026 (America/Los_Angeles). Working tree reviewed at `cb890205dce3cd2a0e848e67971dee4d96a2ae51`. This is an assessment and proposed recovery plan; it does not change the implementation, accepted project scope, or historical benchmark outcome.

**Assessment**

Your concern is substantially supported as a change in research emphasis and in how the results were interpreted. The strongest defensible conclusion is: **July built a much larger experimental and recorded-data system around a measurement update that still does not implement the local, uncertainty-aware single-reading update you describe.** The video evidence adds a serious presentation defect: narrow sampled columns can have all their measurements painted over by their outlines.

However, “May was correct and July broke it” is not supported. May already had an incomplete neighbor update, misleading probability language, overlapping physical scans, and attention-based coverage scores. July corrected several of those experimental defects. A wholesale rollback would restore them.

The recommended response is to recover the original mathematical purpose, preserve July’s useful timing/evaluation components, and make the measurement update and scanner assumptions the next bounded task. Pause expansion of recorded-data infrastructure while doing that.

**1. What the historical record actually says**

I examined all 50 reachable commits, branch tips, the reflog, the only unreachable commit (an amended July 24 refactor), seven local research conversations, historical source files, saved JSON evidence, and nine rendered videos. The videos were fully decoded; five time-spaced frames per video were visually inspected. This is frame-based visual review, not a claim of continuous audiovisual playback.

| Period / checkpoint | Repository evidence | Conversation alignment |
|---|---|---|
| April 29 | Work later captured in the initial May commit | You proposed coarse 2-D evidence, temporal smear from a single-pixel scan, external wind as a motion proxy, uncertainty bounds, and sparse spatial interpolation. You then explicitly requested Bayesian prior → likelihood → posterior → next ROI. |
| April 29 mechanical clarification | Column sweep code in the May snapshot | You specified fast vertical traverses and horizontal stepping, with vertical passes potentially taking a couple of seconds. Column sweeping was your clarification, not a July invention. |
| May 4, 01:40 PDT: `1fcca43` | Initial commit; local `master` and `origin/master` still point here | Contains Bayesian likelihood, wind propagation, credible masks, mechanical column budgets, and videos. It is a saved snapshot, not the whole of May’s work. |
| Later May 4 | Changes remained uncommitted until July | In “Review recovered chat,” you explicitly said the repository had been saved before a major revision and requested Plan A: remove the source from mapping. The chat reports implementation of source-free tracking with known wind. |
| May 4, 03:09–03:36 PDT | No replacement local-neighbor rule committed | “Create sparse plume graph” contains the exact discussion of converting a reading into belief, constants 0.05/0.95/0.35, normalization, 93% mass selection, and unsampled neighbors. |
| May 16–18 | Instructions later included in the July WIP checkpoint | Discussion concerned compute/pitch comparisons; on May 18 you requested the missing `AGENT.md` and exclusion principles. |
| July 24: `a49558d` | “checkpoint source-free tracking WIP” | This captures earlier May source-free work and instructions. Dating everything in this commit as a new July idea would misattribute its origin. |
| July 24: `607b6f7` → `d167f1d` | Causal scan scheduling, actual-sweep evaluation, headless runner, module separation | You requested a critical review, then correctness repairs and cleanup of `video.py`. |
| July 24–25: `4fb910c` → `ad12498` | Matched-cost benchmark; source-aware propagation | You questioned the high success scores, requested fair comparisons, then identified wind propagation that detached belief from a continuously emitting source. |
| July 25: `fbfbb78` | `SYSTEM_GOAL.md` | You explicitly approved the broader deliverable plan, including comparative evidence at equal cost and known-source as the primary proof target. This was an accepted expansion, although it can now be reconsidered. |
| July 25–31: `2260d1b` → `cb89020` | Recorded-input contracts, calibration/provenance, preparation, freezing, qualification, licensed propane diagnostic | You explored GasVid as a substitute for unavailable hardware and clarified that the actual sensor should have less scene structure. |
| July 31, after the last commit | Further analysis and a local comparison video | You challenged the synthetic benchmark’s operating regime and resolution dependence. The assistant retracted its earlier broad interpretation. This correction was not carried through into the persistent status documents. |

The main comparison is May versus July, following your correction that there was little June work. No June commit or June project conversation appeared in the inspected local records. This does not establish that no conversation or copy exists elsewhere.

Evidence: [original conversation](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/01-original-april29.md>), [April–May continuation](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/02-april29-may04.md>), [May source-free revision](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/03-may04-recovery.md>), [May mathematical discussion](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/04-may04-bayes-neighbors.md>), [July conversation](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/07-july-review-and-development.md>). These are visible-message exports; historical assistant statements are treated as claims to check. The shared ChatGPT page referenced by the first conversation could not be fetched, so any earlier material unique to that page remains a gap.

**2. The recoverable mathematics, and the gap**

The original problem is recoverable without adding a neural detector:

> Given delayed scalar measurements from selected locations, a motion/wind estimate, and a physical time budget, update spatial belief and choose the next feasible scan.

There are three separate operations: measurement conditioning, motion prediction, and scan selection. Spatial smoothing cannot substitute for defining all three.

**What May implemented, and current code largely retains.** For sampled mask \(M\), intensity readings \(y_i\), and globally normalized spatial weights \(b_i^-\):

\[
a=Q_{.05}(y_M),\qquad c=Q_{.99}(y_M),
\]
\[
z_i=\mathbf1_{i\in M}\operatorname{clip}\left(
\frac{y_i-a}{\max(c-a,10^{-9})},0,1\right),\qquad
\widetilde z=G_{\sigma=2}*z,
\]
\[
L_i=\begin{cases}
0.05+0.95\widetilde z_i&i\in M,\\
0.35&i\notin M,
\end{cases}
\qquad
b_i^+=\frac{b_i^-L_i}{\sum_j b_j^-L_j}.
\]

May then shifted the posterior by a wind vector, blurred it, optionally mixed 2% source prior, and renormalized. July made the propagation depend on elapsed seconds, oriented the prior with both wind components, and added a continuous source-emission contribution. Those are substantive dynamics changes; the central likelihood rule remains the same by default.

See [May likelihood source](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/may04/bayes.py:72>), [current likelihood](<C:/Users/05mus/Documents/New project 2/src/plume_focus/bayes.py:67>), [current measurement constants](<C:/Users/05mus/Documents/New project 2/src/plume_focus/model_profiles.py:43>), and [current propagation](<C:/Users/05mus/Documents/New project 2/src/plume_focus/plume_dynamics.py:99>).

**The neighbor issue was explicitly identified on May 4.** The assistant first described a point reading spreading probabilistic evidence nearby. When you asked what the code actually did, it acknowledged that unsampled neighbors stay at 0.35. The code agrees with that correction: blur influences sampled cells through surrounding entries, but its result is not assigned to unsampled neighbors. Prediction diffusion later spreads belief; that is a different operation from updating neighbors using a new measurement. The separate inverse-distance reconstruction module is also not the Bayesian likelihood path.

I ran the same probes on the exported May function and the current function:

| Probe | May and current result | Meaning |
|---|---:|---|
| One sampled cell with value 255 | Likelihood 0.05 | With one sample, both percentiles equal 255; normalized evidence becomes zero. |
| One bright and one distant dark sample, default blur | Bright-cell likelihood 0.0878006 | Zero-filled unsampled space dilutes isolated evidence. Without blur, the bright likelihood is 1.0. |
| Unsampled cell next to the bright reading | 0.35 | No direct positive neighbor evidence. |
| Distant unsampled cell | 0.35 | Same as the neighbor; their posterior odds remain equal under a uniform prior. |

These probes show incompatibility with an intended per-reading update. They do not imply every dense sweep demonstration fails. [Reproducible probe and results](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/belief-probes.json>).

There is also a probability-definition issue. A grid with \(\sum_i b_i=1\) is a distribution of relative spatial mass, or could represent the location of one latent item. It is not automatically the collection of probabilities that plume is present at each pixel. Multiple pixels can simultaneously contain plume; their occupancy probabilities need not sum to one. A 93% region of normalized intensity mass is not a 93% probability that the entire plume lies inside the region. Mechanical column/row trimming can further remove mass from the initial credible mask.

Nor is “change unsampled likelihood from 0.35 to 1” a complete Bayesian repair. Likelihood is defined up to a common scale: dividing *all* likelihoods by 0.35 leaves the posterior unchanged. The missing ingredient is a justified observation model and its relative likelihoods, including what non-observation means for the chosen latent state.

**A coherent formulation to recover next — proposal, not recovered implementation.** Let \(X(t)\) denote a latent sensor-signal field. A measurement is a location/footprint \(h_n\), acquisition time \(t_n\), value \(y_n\), and noise model. Then:

\[
p_n^-(X)=\int p(X\mid X',v,\Delta t_n)\,p_{n-1}^+(X')\,dX',
\qquad
p_n^+(X)\propto p(y_n\mid X,h_n)\,p_n^-(X).
\]

A rule for neighboring pixels must come from a declared spatial dependency. For example, under a Gaussian field and \(y=X_i+\epsilon,\ \epsilon\sim N(0,r)\), Gaussian conditioning gives

\[
\mu_j^+=\mu_j^-+
\frac{P_{ji}^-}{P_{ii}^-+r}(y-\mu_i^-),\qquad
P_{jk}^+=P_{jk}^--
\frac{P_{ji}^-P_{ik}^-}{P_{ii}^-+r}.
\]

The measured cell and correlated neighbors update; uncorrelated cells do not. This is an example of a fully specified local influence rule, not a recommendation to build a dense GP system or evidence of an original contribution. A sparse approximation would need separate justification. The conditioning identity is standard; see [Rasmussen and Williams, Chapter 2, equations 2.23–2.24](https://gaussianprocess.org/gpml/chapters/RW2.pdf).

Alternatively, define per-cell binary occupancy \(q_i=P(X_i>\theta\mid D)\). Its measurement update needs plume-present and plume-absent likelihoods; neighbor propagation still needs an explicit dependency model. The May conversations do not settle which interpretation you intended. That semantic choice should precede another architecture change.

For delayed batches, retain each sample’s time and condition at its acquisition time, or explicitly approximate transport to a common decision time. The current runner advances the prior to scan completion but feeds an untransported mixed-time intensity map to the likelihood; that function has no timestamp argument. Thus July fixes inter-scan causality without fully solving within-scan temporal conditioning. The current decision point is one completed sweep, not each pixel. Streaming updates can be processed in acquisition order after completion without changing the existing scan cadence; intra-sweep replanning is a separate choice.

Preserve the original coverage/cost objective:

\[
S^\star=\arg\min_{S\in\mathcal F}T(S)
\quad\text{subject to}\quad
\operatorname{predictedMassRecall}(S,t+T(S))\ge\alpha,\quad T(S)\le B.
\]

Here \(\mathcal F\) contains feasible scan paths, \(B\) is the time limit, and the recall quantity must have a stated meaning. If no feasible scan meets the target, report that outcome. An information-gain policy would be another possible objective, not what the current belief-mass ranking already computes.

**3. What changed in July, and which conclusions survive**

The source package grew from **10 modules / 3,336 lines** at the May commit to **44 modules / 22,427 lines** today. Recorded-data and sensor-emulation modules account for **13,177 lines, about 59%** of current source. These are physical line counts including comments, not a quality measure. They substantiate where the implementation effort went.

| Component | Assessment | Recovery decision |
|---|---|---|
| Sequential scans, finite-input bounds, evaluator/planner separation | Corrects May defects central to the original task | Keep |
| Actual-sweep metrics; requested, swept, sampled distinctions | Essential to testing mechanical benefit | Keep |
| Headless runner and optional renderer | Useful separation; avoids coupling experiments to videos | Keep |
| Source-aware dynamics and explicit profiles | Useful, but remain uncalibrated assumptions | Keep as replaceable models; verify short-horizon behavior |
| Per-scan percentile likelihood and normalized density | Still leaves the intended local measurement model unresolved | Prioritize replacement or explicit restriction to its valid use |
| Exact-budget benchmark allocator | Valuable control, but mishandles reacquisition and omits parts of real travel cost | Repair before using it to explain policy quality |
| Recorded provenance/calibration/qualification infrastructure | Useful for later validation, but currently dominates the project | Preserve; pause expansion during core recovery |
| Raw OGI experiment | Different input regime from the intended cleaner scalar signal | Keep as a labeled diagnostic; do not let it dictate the core sensing model |

May’s old loop starts a new scan for every video index, even when a scan lasts tens or hundreds of seconds, and scores against the next video frame. Finite inputs clamp out-of-range sampling to the last frame. Its coverage uses the attention field, not the executed next sweep. Restoring that loop would restore apparent responsiveness without establishing causal sensing. [Historical loop](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/may04/video.py:1323>).

The benchmark’s negative result is real **for its defined configurations**. The source-aware held-out report matches the hash recorded in `SYSTEM_GOAL.md`. Adaptive minus static-corridor interval mass recall is −0.5144, with paired 95% interval [−0.5529, −0.4758]. Its claim status remains `not_demonstrated`. [Saved report](<C:/Users/05mus/Documents/New project 2/outputs/source-aware-methane-fair-benchmark-held-out-v1.json>).

But the benchmark does not isolate why it differs from the successful demo:

| Evidence artifact | Mechanics / conditions | Reported result |
|---|---|---|
| July source-aware 46-cycle demo | Mean scan 0.1504 s; full-height pass 0.005 s; zero horizontal step time; exact wind; variable mask | 100% completion mass recall |
| Source-aware held-out benchmark | Scan 9.086 / 24.768 / 62.677 s; full-height pass 1 s; horizontal step 0.05 s; noisy/biased/missing wind; exact budgets | Interval mass recall 8.54% / 6.10% / 61.17% |
| July 31 matched fast development diagnostic | Same episode and noisy wind for both policies; 60×80; scan 0.1091 s | Completion mass recall: adaptive 80.92%, static 95.12% |

The successful “fast” settings are counterfactual, not measured hardware capabilities. In particular, a 5 ms full-height pass was not established by your April description of passes taking up to a couple of seconds. Speeding the simulation up until it succeeds would not resolve this.

Two July problems are independently visible in code:

- When uncertainty triggers reacquisition, the adaptive policy returns a uniform field. The runner checks exact budget before reacquisition, so that field is sent to stable top-column ranking rather than a broad scan. A small probe selects columns [0,1,2,3,20] when the source is at column 20. It repeatedly chooses a coordinate-based pattern instead of implementing reacquisition. [Planner branch](<C:/Users/05mus/Documents/New project 2/src/plume_focus/experiment_runner.py:1157>), [allocator](<C:/Users/05mus/Documents/New project 2/src/plume_focus/scan_policies.py:295>).
- Synthetic methane uses an unbounded analytical timeline, with each puff active for 66 frames (5.5 s at 12 FPS). The benchmark therefore runs a continuous source far longer than a nominal 72-frame clip. This is not a reset into independent snapshots. An analytical continuous stream is not inherently invalid; it is a different input contract from replaying a fixed finite video. Nor does individual puff lifetime imply the continuously emitting plume ceases to exist.

The late July assistant’s blanket characterization of the benchmark as invalid acceptance evidence was therefore too categorical. What is justified is narrowing its interpretation, documenting the regime and reacquisition failure, and designing a new development experiment that changes one factor at a time. The faster diagnostic still loses to the static baseline. We cannot infer that appropriate timing will make the adaptive policy superior.

Column timing remains tied to pixel resolution:
\[
T=\sum_{c\in C}t_v\frac{R_c}{H}+(|C|-1)t_h.
\]
It charges vertical span and a fixed horizontal move per selected column, but not a calibrated distance-dependent slew or separately constrained detector integration cadence. A single detector may scan continuously; however, finer vertical pixels currently create more perfect samples within the same pass. Separate belief-grid resolution, beam footprint, scan lattice, readout cadence, and physical travel before interpreting hardware cost.

**4. What the rendered videos add**

[Video manifest, decode counts and frame samples](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/video-manifest.json>).

| Rendered artifact | What is visibly happening | What it establishes |
|---|---|---|
| May tilted adapted scan reveal | 380 frames / 31.67 s. Live plume on the left; measurements progressively appear in a fixed selected sweep on the right. | Clear communication of sequential sensing, but only one already selected scan, not a per-pixel policy-update demonstration. |
| May full-scene proof | 46 frames / 3.83 s. First scan label says 336 s; later labels are about 33–37 s, while the plume moves between adjacent rendered frames. | The visual narrative hides the old time mismatch; it cannot establish a causal feedback loop. |
| May source-free proof | Broad initial field, then a smaller scan; early disconnected cyan fragments remain visible. | Source-free work already existed in May; sharp-looking overlays do not prove correct neighbor inference or sweep recall. |
| July wind-saturation comparison | The next mask remains fixed when planner wind changes; predictive field is saturated across the image. | Supports the uncertainty/reacquisition diagnosis; not merely an unattractive overlay. |
| July source-aware 46-cycle demo | A broad cyan mechanical region covers the plume while colored belief peaks can be much smaller or separated. | Belief heatmap, mechanical coverage, and plume truth convey different quantities. 100% sweep recall does not mean a calibrated belief field. |
| July raw propane OGI | 8 frames / 0.535 s for 36.22 s of modeled scans. Yellow stripes dominate the scan panel; red wash covers nearly the whole right panel. | The rendered artifact is inadequate for inspecting the measurements or scan progression. It is also a raw, source-free, wind-absent input, unlike the clean known-wind demonstration. |
| July 31 fast paired diagnostic | 180 frames / 15 s. Adaptive selection initially includes detached columns; static corridor stays compact around the visible plume. | More useful comparative visualization, and a visible residual allocation problem even without reacquisition. Still one development case. |

**Confirmed renderer bug:** `project_single_pixel_scan` draws a one-pixel contour around every sampled strip. For isolated one-pixel-wide columns, the contour paints the entire strip. I changed **all 192 sampled values from 0 to 255** in a synthetic stride-4 scan; the rendered scan projection changed by **zero pixels**. Every sampled pixel was the outline color. Thus the yellow-striped OGI view cannot be used to judge whether the underlying sampled values are informative. This is stronger than a cosmetic complaint. [Code](<C:/Users/05mus/Documents/New project 2/src/plume_focus/video_rendering.py:155>), [probe result](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/render-probes.json>).

The same contour drawing already appears in [May’s renderer](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/may04/video.py:928>). It is an inherited defect made especially consequential by the sparse OGI columns, not evidence that July introduced every visual problem.

The red wash has another explanation: the runner divides predictive belief by its maximum for display. A uniform, highly uncertain field becomes 1 everywhere and is drawn at maximum color intensity. Red is relative density, not calibrated confidence. The legend also advertises “true plume envelope” even in the truth-free OGI run, and the header says “Next Video Frame” while the causal runner targets next-scan completion. [Display normalization](<C:/Users/05mus/Documents/New project 2/src/plume_focus/experiment_runner.py:1784>), [renderer labels](<C:/Users/05mus/Documents/New project 2/src/plume_focus/video_rendering.py:296>).

The OGI report has no truth and zero evaluated predictions; its zero recall fields are not measured 0% recall. Its mean requested mask fraction is 91.05%, whereas swept fraction is 23.36% because columns are skipped. The smaller sweep therefore does not establish successful focusing. [OGI report](<C:/Users/05mus/Documents/New project 2/data/recorded/outputs/usu-r0c-v1-rendered.json>).

This visual evidence strengthens the case for restoring a readable scan-progress demonstration alongside the mathematical repair. It does not justify restoring the old causal errors. The [visual evidence companion](<C:/Users/05mus/Documents/New project 2/docs/reviews/2026-09-12-video-evidence.md>) includes extracted frames and links to the original videos.

**5. Practical recovery options**

| Option | Advantage | Cost / risk | Assessment |
|---|---|---|---|
| Reset the repository to May | Small, familiar demonstration | Restores timing, metric, likelihood, and resolution problems; loses useful infrastructure | Do not use as the implementation path |
| Continue the current recorded-data roadmap unchanged | Builds on existing ingestion work | Leaves the central model and visual diagnosis unresolved | Poor next priority |
| Recover a focused sensing core within the current repository | Keeps causal execution and evidence tools while correcting the original mathematical gap | Requires a clearly defined belief and observation model | Recommended |
| Extract a minimal successor after the repair is specified | Could simplify long-term maintenance | Premature copying creates another file-management split and duplicated assumptions | Consider later, only with measured dependency burden |

The saved May revision is still recoverable from Git. Earlier May local edits are identified in chat and were checkpointed in July. No reset, branch deletion, repository move, or rewrite is necessary to begin recovery.

A concrete next task could be:

> Re-establish a timestamped single-pixel Bayesian sensing demonstrator with an explicit spatial update rule, physically stated scan limits, and a faithful scan-progress visualization, retaining existing causal execution and historical evidence.

Recommended order and completion conditions:

1. **Settle the state semantics and measurement rule.** Decide whether the stored quantity is relative spatial mass, per-cell occupancy, or a latent sensor signal with uncertainty. Define sensor footprint, reading units, likelihood/noise model, and the rule governing neighboring influence. Complete when a worked numerical one-reading example and its uncertainty update are unambiguous.
2. **Repair measurement conditioning in isolation.** Preserve sample timestamps; separate absent measurements from measured zeros; eliminate dependence on having a contrast range within each scan. Demonstrate positive and negative readings, one-point and sparse-column inputs, correlated neighbors, distant cells, stale samples, and repeated data. Complete when these cases match the declared model; otherwise the mathematical implementation fails.
3. **Repair the visual inspection path.** Display measured intensity separately from age and mask outlines, show the actual clock and accumulating samples, mark posterior-update moments, distinguish prior/posterior/prediction, and omit truth labels when unavailable. Complete when changing sensor values visibly changes the measurement panel and a viewer can identify when the policy learns.
4. **Fix the physical experiment definition.** Use declared angular/spatial scan positions and detector cadence, with explicit mapping to the belief grid. Retain point and column modes as distinct models. Complete when the same physical scan has approximately invariant timing and coverage under a change in numerical resolution, or the remaining dependency is explicitly quantified.
5. **Reproduce and test the intended stream on development cases.** Hold the continuous input and mechanics fixed while comparing measurement update, wind, uncertainty, and allocation separately. Include a static corridor and a simple rescan baseline; make reacquisition feasible and charge its cost, or terminate as infeasible. Complete with coverage/time curves and failure cases, even if no adaptive advantage appears.
6. **Reconcile the durable documentation, then return to external data.** Append the corrected interpretation of the July benchmark and the late July visual evidence. Preserve `not_demonstrated` and all historical results. If improving the policy, the already inspected held-out cases cannot function as a fresh untouched test; freeze new evaluation cases after development.

The research chain should remain explicit:

**Problem:** delayed sparse sensing of a moving plume under scan-time limits.\
**Claim to test:** the declared local update plus wind prediction permits a smaller/faster feasible scan while retaining the chosen coverage target.\
**Experiment:** continuous scalar-signal streams, controlled observation noise and wind error, stated physical timing, and matched comparisons.\
**Falsification:** local updates fail their numerical examples, scan duration erases useful predictability, coverage is insufficient at feasible cost, or a simple static policy performs as well or better in the intended regime.

A static corridor winning on a stable known-source plume is an admissible outcome. Adding source discovery, segmentation training, or arbitrary new dynamics merely to obtain an advantage would change the research question.

**Verification and limits**

The focused existing suites for Bayes, scanning, scan execution, policies, dynamics, and the headless runner passed: **49 tests in 3.99 s**. The additional belief and renderer probes above were run against current code; the belief probe also used the May snapshot. Nine local videos were fully decoded and sampled for visual inspection. Historical benchmark reports were inspected, not rerun or retuned.

Current `SYSTEM_GOAL.md` records 307 passing tests and five platform skips from its earlier checkpoint; I did not rerun the entire recorded-data suite in this review. Passing software tests does not resolve the mathematical or hardware assumptions.

No tracked implementation, existing status document, historical result, Git branch, or remote was changed. This review is a new uncommitted document. Conversation exports, historical source/video copies, contact sheets, and reproducible probes are local artifacts under [the review evidence directory](<C:/Users/05mus/Documents/New project 2/outputs/scope-review-2026-09-12/history-manifest.json>), which is ignored by the repository’s existing output policy. Local `origin/master` was inspected; the remote server was not refreshed.
