# Appendix: Confidence Bounds and Scan Allocation

Research background. The subsequent [v1 formulation and policy freeze](../specs/reconstruction-policy-v1.md)
specifies the initial diagnostic and keeps allocation extensions experimental.
The alternatives below are preserved for later assessment.

## Role within the current formulation

Upper confidence bounds are worth testing as a small allocation extension. They can help revisit uncertain signal outside the current region, identify unresolved plume boundaries, or assess how much signal a proposed sweep might miss. These uses address different questions. The recommended first extension preserves the existing Bayesian estimator, completed-sweep decisions, fixed instrument profile, and 90% predicted-mass baseline.

Classical GP-UCB selects an observation using posterior mean plus a multiple of posterior standard deviation. Its original objective is high cumulative function value, with regret guarantees under specified function, kernel, and noise assumptions. That differs from reconstructing a moving field or estimating its total strength.[^15] A UCB-inspired score can be useful here without importing those guarantees.

At a declared future time, let the latent projected enhancement in cell j have mean mu_j and standard deviation sigma_j. A possible optimistic signal score is

$$U_j=\mu_j+\kappa\sigma_j,\qquad s_j^{\mathrm{UCB}}=w_j(U_j-\tau)_+.$$

Here w_j is the cell's area or solid-angle weight, tau the enhancement threshold, and kappa an exploration parameter. Use this score to propose feasible regions or supplementary visits. A weak but poorly known region receives more attention than its mean alone would justify. Correlation remains in the estimator and should also inform the value of a complete sweep.

The current expected-positive-signal rule already responds to uncertainty. For a Gaussian marginal with positive standard deviation,

$$m_j=w_j[(\mu_j-\tau)\Phi(a_j)+\sigma_j\phi(a_j)],\qquad a_j=(\mu_j-\tau)/\sigma_j.$$

Phi and phi are the standard normal CDF and density. With zero standard deviation, use w_j times the positive part of mu_j-tau. Consequently, adding a UCB bonus changes an already uncertainty-dependent policy; it is not simply adding previously absent uncertainty. Keep the original mass rule and the optimistic rule as distinct comparison variants.

**Interpretation matters.** Normalizing UCB scores gives a distribution of optimistic weights. Retaining 90% of those weights is not the same as retaining 90% of expected signal mass, and neither is a 90% posterior coverage probability. Continue to report original predicted-mass retention even when another score generates candidates.

The standard deviation should describe uncertainty in the latent field at the relevant time. Adding irreducible readout noise to the exploration bonus can favor a noisy location precisely because it is hard to measure. Expected information gain must account for that noise in the observation model. An exploration bonus belongs in action selection; it must not inflate the reported concentration estimate or count as new evidence.

<!-- pagebreak -->

## Predicted mass and probability of coverage

The 90% rule and a confidence requirement involve two separate parameters. Let gamma=0.90 be the desired signal fraction and let delta be a tolerated posterior failure probability. For a candidate sweep S, define its retained cell weights by a fixed mechanical-footprint mapping. For clarity, the following equations use a binary retained set R(S) on the inference grid; fractional footprint coverage can replace that binary mapping.

At a common reference time, define random retained and omitted enhancement:

$$M_{\mathrm{in}}=\sum_{j\in R(S)}w_j(x_j-\tau)_+,\qquad M_{\mathrm{out}}=\sum_{j\notin R(S)}w_j(x_j-\tau)_+.$$

The existing criterion is a ratio of posterior expectations:

$$\frac{\mathbb{E}[M_{\mathrm{in}}\mid D_t]}{\mathbb{E}[M_{\mathrm{in}}+M_{\mathrm{out}}\mid D_t]}\geq\gamma.$$

A stronger, different criterion asks for posterior probability at least 1-delta that the realized retained fraction exceeds gamma, when total enhancement is meaningfully positive. It is a joint spatial event, not a collection of separate pixel confidence statements. Work on Gaussian excursion sets makes this distinction between marginal and simultaneous spatial statements explicit.[^18] That literature supplies relevant probability machinery; the mass-ratio criterion here is a proposed adaptation.

**Worked illustration, not an experiment.** Consider two equal-weight cells, threshold zero, no motion, cell A exactly 10, and cell B distributed as N(1,1), where the second parameter is variance. A sweep retaining only A captures 10/(10+1.083315)=90.2257% of predicted positive mass. Nevertheless, its probability of retaining at least 90% of the realized positive mass is only 54.4236%: the event requires B to be at most 10/9. A 95% coverage-confidence requirement would reject this candidate.

For this example, a one-sided 95% upper bound on omitted positive signal is 2.644854. A conservative retained-fraction bound is therefore 10/(10+2.644854), approximately 79.1%. The original 90% selection has not made an algebraic error; it answers a weaker question.

This stronger requirement may demand a larger or slower sweep. Its usefulness depends on whether the resulting reconstruction is better under the same physical budget. It should first be a diagnostic alongside the original rule, not a compulsory expansion step.

Retained spatial mass at a reference time is also not identical to signal actually encountered along a moving scan. The latter depends on every acquisition time. Neither coverage quantity establishes amplitude accuracy; keep the reconstruction objective and actual execution trace from the parent formulation.

<!-- pagebreak -->

## Conservative bounds and joint posterior assessment

A practical diagnostic can start with simultaneous bounds on the finite projected field. Suppose lower and upper values ell_j and u_j satisfy posterior probability at least 1-delta that all cells lie between their bounds at the specified time. For a Gaussian posterior, one conservative construction is

$$\ell_j=\mu_j-\kappa\sigma_j,\quad u_j=\mu_j+\kappa\sigma_j,\quad \kappa=\Phi^{-1}(1-\delta/(2N)).$$

N is the number of latent cells, not display pixels. This follows from marginal Gaussian tails and a union bound; it does not require independence. It is a model-conditional statement at one time, not a universal frequentist guarantee. A fitted or misspecified covariance need not provide its nominal coverage.

Define a lower bound on retained signal and an upper bound on omitted signal:

$$L_{\mathrm{in}}=\sum_{j\in R(S)}w_j(\ell_j-\tau)_+,\qquad U_{\mathrm{out}}=\sum_{j\notin R(S)}w_j(u_j-\tau)_+.$$

On the simultaneous event, and for positive total mass, monotonicity gives

$$\frac{M_{\mathrm{in}}}{M_{\mathrm{in}}+M_{\mathrm{out}}}\geq\frac{L_{\mathrm{in}}}{L_{\mathrm{in}}+U_{\mathrm{out}}}.$$

Thus a sufficient condition for gamma coverage is U_out no greater than (1-gamma)/gamma times L_in. A zero lower denominator gives no usable certificate. The construction is deliberately conservative: wide intervals across many cells may make it uninformative. It illustrates why normalizing upper bounds on both retained and omitted mass is not a conservative coverage calculation; the retained numerator needs a lower bound.

For less conservative assessment, draw correlated complete fields from the predictive posterior and evaluate each candidate's retained fraction on every draw. Joint sampling preserves shared plume displacement and spatial dependence; independent per-cell draws can distort the tail risk. For candidates evaluated at several times, use the joint space-time forecast. A statement about the entire sweep requires an explicitly defined temporal event or discretization, not a bound checked only at completion.

Near zero signal, report the probability that total enhancement exceeds a declared meaningful floor M_min. Conditional coverage given that event and an absolute bound on missed enhancement are useful complementary outputs. Do not treat a ratio dominated by an almost-empty field as a successful plume reconstruction.

A Monte Carlo fraction is an estimate, not an exact posterior guarantee. Validate the selected candidate with independent posterior draws and report simulation error; otherwise searching many candidates can exploit sampling noise. Physical calibration error is a separate problem that more posterior draws cannot fix.

Use this assessment to expose risk and compare feasible alternatives. If the target cannot be certified within the budget, report that fact. A failed certificate need not mean the scan truly fails, especially when the bounding method is conservative.

<!-- pagebreak -->

## Boundary learning and reconstruction value

**Boundary sampling.** Lower and upper bounds distinguish cells confidently above a threshold, confidently below it, and still unresolved. Gotovos et al. develop this idea for GP level-set estimation, including batch selection, and compare it with the straddle heuristic.[^16] A simple proposed boundary score is

$$s_j^{\mathrm{boundary}}=w_j(\kappa\sigma_j-|\mu_j-\tau|)_+.$$

This favors uncertain cells whose intervals span the threshold. It can help determine whether a plume has expanded beyond the current region. It is less suited to measuring an already confirmed plume's amplitude, so boundary sampling should not become the sole strength-estimation policy. In a moving plume, a cell classified below threshold must be allowed to become unresolved again; permanent elimination based on past intervals is inappropriate.

**Reduction in uncertainty.** The parent formulation's reconstruction-value calculation is more directly aligned with the scientific objective. For a complete sweep with vector observations Y_S, let C_HS be the predictive cross-covariance between the target field and those observations, and V_S their covariance including measurement error. In a fixed linear Gaussian approximation,

$$\Delta_{\mathrm{map}}(S)=\operatorname{tr}(W C_{HS}V_S^{-1}C_{HS}^{T}).$$

For a linear strength index with coefficient vector a, use

$$\Delta_{\mathrm{strength}}(S)=a^T C_{HS}V_S^{-1}C_{HS}^{T}a.$$

These expressions handle redundant correlated readings. Use actual candidate acquisition times when constructing the covariances, and keep all candidates' target time or monitoring window consistent.

TRUVAR is a useful adjacent method: it reduces uncertainty in a relevant set, truncating the incentive to refine points already sufficiently resolved. It supports unequal costs and noise. Its theorem assumes a variance-reduction submodularity condition; its movement-dependent cost experiment goes beyond the fixed point-cost setup of that theorem.[^17] Transfer the idea of reducing relevant uncertainty per charged cost, not an unverified performance guarantee.

For this scanner, compare legal complete sweeps within the budget. Marginal information gain per added travel time can be a candidate-construction heuristic; it does not establish a globally optimal route. A large uncertainty caused mainly by future process noise may be difficult to reduce through present observations. Cross-covariance to the future target captures that limitation more faithfully than a raw uncertainty bonus.

**Posterior sampling.** Thompson sampling provides another exploration mechanism by acting according to a sampled plausible model.[^21] A proposed adaptation would sample a correlated plume forecast and generate a feasible region from it. That is a randomized candidate generator, not a coverage certificate or automatically a reconstruction-optimal policy. Evaluate its candidate against the full posterior and record random seeds. This is a secondary comparison after the simpler bound and uncertainty-reduction variants.

<!-- pagebreak -->

## Integration and comparison protocol

The extension fits between prediction and mechanical-plan selection. Assimilation still consumes the inferred beam-integrated readings with their timing and uncertainty. Prediction supplies the latent mean, covariance, and age information. Candidate generation can use mass, optimistic signal, or boundary scores. Candidate assessment then reports expected mass, coverage risk, expected reconstruction value, and physical cost separately.

Within one proposed batch, account for the information expected from already selected readings before assigning value to further readings. GP-BUCB explicitly addresses redundancy under delayed feedback and the overconfidence that can arise from treating unobserved outcomes as available evidence.[^19] Here, planning covariance calculations remain hypothetical; the live posterior changes only through available readings and declared state evolution. The selected execution path remains fixed until completion.

Time-varying GP bandit work models an evolving function and distinguishes gradual forgetting from periodic resets.[^20] The applicable lesson is that old evidence cannot imply permanent certainty about a changing location. The present state-space dynamics already provide a place for age, wind error, and process uncertainty. Adding a second arbitrary forgetting rule should require evidence that the existing evolution is inadequate.

| Candidate comparison | Intended role |
| --- | --- |
| Original 90% mass policy | Preserve the current allocation baseline. |
| UCB-weighted proposal | Test whether uncertain omitted regions deserve more visits. |
| Original mass policy with risk assessment | Expose possible missed signal without initially changing actions. |
| Original core plus useful supplementary visits | Test boundary or reconstruction-value sampling within the same total budget. |
| Joint coverage-constrained candidate | Test the cost and benefit of a stronger posterior criterion. |

The recommended first addition is **risk assessment plus one exploration variant**, not simultaneous adoption of every method. If retaining the original core exhausts the budget, supplementary visits are infeasible; there is no free exploration allowance. A 90% mass target does not mean that 10% of scan time remains available. Use an explicitly named fallback if a combined candidate cannot meet the target.

Freeze mechanics, estimator, calibration, scenarios, and physical evaluation times across each comparison. Test a missed secondary lobe, plume displacement, an exhausted source, a very noisy region, and a broad uncertain background. Compare reconstruction error and missed enhancement against actual charged time, coverage-certificate frequency, and empirical bound violations. Tune exploration constants on development cases; reserve separate cases for assessing coverage and performance.

Uniform uncertainty can give UCB nearly equal scores everywhere. It does not solve startup or rediscovery by itself. Keep the charged age-based schedule and deterministic rotation as a baseline, and distinguish avoidable allocation failure from uncertainty that the sensor cannot resolve within the budget.

This appendage proposes allocation and reporting alternatives. It does not change sensor footprint, scan speed, inference evidence, renderer, or within-sweep behavior. An improvement is demonstrated only if it improves reconstruction or a defined strength estimate at an honestly reported cost.

<!-- pagebreak -->

## Appendix sources

15. Niranjan Srinivas, Andreas Krause, Sham M. Kakade, and Matthias Seeger. **Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design.** ICML, 2010, pp. 1015-1022. [Author-hosted paper](https://www.dna.caltech.edu/Papers/Srinivas-2010-regret-ICML.pdf). Algorithm 1 and section 4 establish the UCB rule and its assumptions. Regret concerns optimization of the modeled function, not plume reconstruction.

16. Alkis Gotovos, Nathalie Casati, Gregory Hitz, and Andreas Krause. **Active Learning for Level Set Estimation.** IJCAI, 2013, pp. 1344-1350. [Proceedings paper](https://www.ijcai.org/Proceedings/13/Papers/202.pdf). Confidence-based classification, batch sampling, and the straddle comparison inform boundary acquisition. The time-invariant target assumption requires adaptation.

17. Ilija Bogunovic, Jonathan Scarlett, Andreas Krause, and Volkan Cevher. **Truncated Variance Reduction: A Unified Approach to Bayesian Optimization and Level-Set Estimation.** NeurIPS, 2016. [Proceedings paper](https://proceedings.neurips.cc/paper/2016/file/ce78d1da254c0843eb23951ae077ff5f-Paper.pdf). Sections 2-4 provide the selection rule, theorem assumptions, and cost experiment. No theorem is transferred to the scanner's route-dependent costs.

18. David Bolin and Finn Lindgren. **Excursion and contour uncertainty regions for latent Gaussian models.** Journal of the Royal Statistical Society: Series B 77(1), 85-106, 2015. [DOI](https://doi.org/10.1111/rssb.12055); [author paper](https://arxiv.org/pdf/1211.3946). Supports distinguishing marginal from joint spatial probability. The positive-mass ratio bound in this appendix is a direct derivation, not a quoted result from this paper.

19. Thomas Desautels, Andreas Krause, and Joel W. Burdick. **Parallelizing Exploration-Exploitation Tradeoffs in Gaussian Process Bandit Optimization.** Journal of Machine Learning Research 15, 4053-4103, 2014. [Journal paper](https://jmlr.org/papers/volume15/desautels14a/desautels14a.pdf). Sections 3-4 address batch redundancy, pending feedback, and confidence inflation. Parallel experiments differ from a moving sequential sweep.

20. Ilija Bogunovic, Jonathan Scarlett, and Volkan Cevher. **Time-Varying Gaussian Process Bandit Optimization.** AISTATS, PMLR 51, 314-323, 2016. [Proceedings](https://proceedings.mlr.press/v51/bogunovic16.html); [paper](https://proceedings.mlr.press/v51/bogunovic16.pdf). Uses a declared Markov evolution and studies forgetting. Its dynamics and regret objective are not assumed to describe this plume.

21. Daniel J. Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, and Zheng Wen. **A Tutorial on Thompson Sampling.** Foundations and Trends in Machine Learning 11(1), 1-96, 2018. [DOI](https://doi.org/10.1561/2200000070); [author manuscript](https://arxiv.org/pdf/1707.02038). General posterior-sampling principle; the correlated plume-candidate adaptation is a proposal.

[^15]: Srinivas et al., [GP-UCB](https://www.dna.caltech.edu/Papers/Srinivas-2010-regret-ICML.pdf), 2010, algorithm 1 and section 4.
[^16]: Gotovos et al., [Level-set estimation](https://www.ijcai.org/Proceedings/13/Papers/202.pdf), 2013, algorithm and experimental comparisons.
[^17]: Bogunovic et al., [TRUVAR](https://proceedings.neurips.cc/paper/2016/file/ce78d1da254c0843eb23951ae077ff5f-Paper.pdf), 2016, sections 2-4.
[^18]: Bolin and Lindgren, [Spatial excursion uncertainty](https://arxiv.org/pdf/1211.3946), 2015, joint versus marginal probability.
[^19]: Desautels et al., [GP-BUCB](https://jmlr.org/papers/volume15/desautels14a/desautels14a.pdf), 2014, sections 3-4.
[^20]: Bogunovic et al., [Time-varying GP optimization](https://proceedings.mlr.press/v51/bogunovic16.pdf), 2016.
[^21]: Russo et al., [Thompson sampling tutorial](https://arxiv.org/pdf/1707.02038), 2018.
