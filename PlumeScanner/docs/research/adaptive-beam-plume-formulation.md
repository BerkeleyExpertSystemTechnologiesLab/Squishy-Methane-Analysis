# Adaptive Beam Scanning for Plume Reconstruction

Research background. The subsequent [v1 formulation and policy freeze](../specs/reconstruction-policy-v1.md)
is the authoritative specification; this memo preserves the research reasoning
and does not establish current implementation conformance.

Companion appendix: [Confidence Bounds and Scan Allocation](confidence-bounds-scan-allocation-appendix.md) examines UCB weighting, joint coverage bounds, and supplementary sampling within this formulation.

## Recommendation

Use the adjacent literature selectively. Its most useful contributions are a way to value measurements by their effect on reconstruction, a disciplined treatment of time, and examples of acquisition policies that respect scientific instruments. None of these requires changing the reference sensor footprint, scan speed, or sweep pattern. A defensible next formulation can retain Bayesian spatial updates and the original 90% predicted-mass region as a candidate policy while making its reconstruction purpose explicit.

The application is a mechanically steered sensor reporting a concentration value inferred from its response and integrated along a beam. The instrument is used sequentially because the intended measurement capability is costly or impractical to deploy as a full array. The scientific objective is to reconstruct an intermittent, evolving plume and eventually estimate a defined measure of its strength. This economic premise motivates the design; comparative hardware cost and accuracy have not yet been established for a specified instrument.

The literature divides into three useful groups. Adaptive scanning microscopy offers measurement-selection objectives and instrument integration. Dynamic single-pixel imaging examines spatial detail versus temporal freshness. Direct gas-sensing research provides the closest observation models and exposes the assumptions needed to infer concentrations or emissions from line-integral data. Their results should inform distinct parts of the formulation, with transfer conditions checked separately.

The strongest conceptual change is to distinguish **signal amount**, **uncertainty**, and **expected measurement value**. A bright region can already be well measured; a weaker region can dominate uncertainty in total plume signal. Conversely, an uncertain region can be irrelevant to the quantity being estimated. The scan policy therefore needs a stated scientific loss, even if its first implementation uses a simple mass-selection heuristic.

The recommended immediate target is a **time-referenced map of beam-integrated concentration enhancement**. A three-dimensional concentration field and an emission rate are further inference products with additional requirements. The report below develops a proposed formulation for the immediate target, not a claim that the existing controller implements or validates it.

The two initial papers are useful, but direct methane research is at least as important. In particular, neither generic robot navigation nor an off-the-shelf learned scanner should determine the measurement contract. No implementation change, new experimental result, or novelty claim follows from this assessment.

<!-- pagebreak -->

## Measurement and reconstruction target

Let q denote the scanner's angular direction and c(r,t) the physical gas concentration. An idealized one-way concentration-path integral is

$$z(q,t)=\int_0^{L(q)} c(o+\ell d(q),t)\,d\ell.$$

Here o is the optical origin, d(q) the beam direction, and L(q) the path endpoint. If c is a volume mixing ratio in ppm and length is in metres, z has units ppm m. A reported path average instead has units ppm and divides the integral by path length. A round-trip instrument may report a value already corrected to a one-way convention. These conventions must come from the instrument's retrieval definition; the controller must not silently apply another factor or divide by an assumed plume depth.

The controller consumes the **retrieved scalar**, not the raw optical signal. A general approximate observation model is

$$y_i=\int w_i(s)\,z(q_i(s),s)\,ds+\beta_i+\epsilon_i.$$

The normalized temporal response w_i includes integration or filtering when known; q_i(s) includes movement during the reading; beta_i is residual retrieval/background bias; epsilon_i represents remaining error. A finite beam can be included by averaging z over its angular footprint. A midpoint sample is an approximation to this model and must be checked against response time and plume motion. Missing, rejected, or saturated readings are not measured zero concentration.

For the immediate reconstruction, estimate the beam-integrated enhancement x(q,t)=z(q,t)-z_bg(q,t) on a declared angular field of view. Background can depend on path length and scene geometry. A positive excess map is useful for finding a plume, but thresholding must not erase uncertainty or silently become a calibrated mass estimate. Sensor-reported uncertainty, calibration identity, acquisition interval, availability time, and actual pointing belong with each scalar value.

**Identifiability.** One scalar integral does not identify where gas lies along that beam. Even a dense angular scan from one fixed origin generally cannot uniquely recover an arbitrary three-dimensional gas field. Transport assumptions, additional viewpoints, range-resolved measurements, or strong source restrictions can add information, but they change the inference problem. Direct 3-D Bayesian gas tomography explicitly identifies limited-angle rank deficiency and supplies transport dynamics to constrain it.[^7]

A displayed two-dimensional concentration-path map is therefore a valid reconstruction target in its own right. Its angular sum is not automatically plume mass: conversion requires justified spatial geometry and units. Emission-rate estimation additionally needs a transport or flux relationship. NIST's quantification overview separates retrieved enhancements from emissions and identifies wind and plume geometry as consequential inputs.[^12]

<!-- pagebreak -->

## Adaptive acquisition in neighboring instruments

**FAST scanning microscopy: borrow the objective and cycle structure.** Kandel et al. select measurements using estimated reduction in reconstruction error, group points into batches, and then optimize their travel route. Their demonstration concerns a material image, with a learned predictor of measurement utility; it does not establish an evolving-plume observation model. The authors explicitly identify excessive motor travel as a limitation of selecting candidates without incorporating movement cost, and suggest movement-aware or line-based selection.[^1]

For this scanner, the transferable idea is to assess a complete feasible sweep by what it teaches us. The predictor, point ordering, and trained image model are optional implementation choices. A faithful adaptation would compare legal column sweeps from the current pose, including all necessary repositioning. It would not select arbitrary desirable pixels and assume that their subsequent mechanical conversion preserves utility or coverage.

**Foveated single-pixel imaging: borrow temporal allocation, preserve the hardware distinction.** Phillips et al. concentrate rapid acquisition on changing image regions and combine observations over longer periods where the scene remains stable. Their detector measures whole-scene mask correlations through patterned optics. Peripheral monitoring and variable effective cell sizes are enabled by that optical architecture; they are not equivalent to a fixed-footprint beam visiting points. The paper also distinguishes real-time acquisition/display from more expensive postprocessed reconstruction.[^2]

For plume sensing, the useful question is how long evidence remains informative locally. A proposed adaptation is to revisit likely changing plume regions more frequently while allocating an explicit, charged background surveillance schedule. Sparse visits do not measure a larger footprint or provide a full peripheral image. Reusing older evidence needs a motion or persistence model; it cannot be treated as new evidence.

**Line-constrained mass-spectrometry imaging: borrow legal action geometry.** Helminiak et al. formulate line-bounded candidate acquisition to accommodate instrument movement restrictions.[^3] Hu et al. subsequently report experimental linewise integration with a 2.3-fold throughput improvement, while their larger pointwise gains are simulation results.[^4] This is a concrete warning against treating pointwise sampling results as equivalent to column-sweep performance.

The appropriate transfer is the principle of scoring actions the instrument can actually execute. Their tissue models and training data do not establish a plume policy. In this context, “dynamic sampling” can mean adaptive choice while imaging a fixed object; it must not be confused with reconstruction of a physically evolving gas distribution.

<!-- pagebreak -->

## Direct gas-sensing evidence

**Bayesian beam-integral reconstruction.** Voss et al. combine multi-directional laser concentration observations with Bayesian state estimation in a two-dimensional transport model.[^6] Vänskä et al. extend this line of work to three-dimensional transport and source estimation, explicitly discussing limitations of the earlier 2-D approximation.[^7] These are close precedents for separating the observation operator from the evolving state. Their estimation results do not, by themselves, establish the benefit of adaptively choosing the next scan region.

Scheel et al. demonstrate a farm deployment with 16 retro-reflector paths and use a fixed-lag extended Kalman smoother. They also exhibit uncertainty miscalibration under wind-model error, including cases where credible intervals do not contain the true simulated emission rate as often as intended.[^8] The useful lessons are to preserve observation timing, distinguish filtering from retrospective smoothing, and test transport-model errors. Their sparse reflector geometry and source priors differ from a dense angular plume image.

**Mechanically scanned TDLAS imaging.** Senyuan Wang et al. combine a standoff gas sensor, pan-tilt unit, rangefinder, visible camera, and wind measurements for imaging and quantification.[^10] Xiachun Wang et al. describe a scanning TDLAS imager with correction for scanning hysteresis and flux inference using simulated wind.[^11] These are direct application neighbors. Publicly accessible abstracts support those descriptions, but do not establish that either implements the proposed Bayesian adaptive sweep policy or solves rapid plume deformation under the reference mechanics.

**A counterweight to “faster is always better.”** Di Wang et al. study path-integral methane mapping with several optical geometries and discuss scanning-duration effects. Their controlled gas-bag analysis recommends sufficient measurement time to reduce motion-induced vibration error.[^9] This supports separating mechanical disturbance and detector response from the changing plume itself. It does not justify increasing or decreasing the reference dwell time during policy comparisons.

Three distinct errors should therefore be represented or measured separately: gas evolution during a sweep; sensor filtering or response lag that displaces a reading relative to pointing; and mechanical pointing or settling error. A lag correction can improve a static target image without repairing nonrigid plume evolution. Conversely, a shorter region scan can reduce temporal spread while leaving calibration bias untouched.

The application has substantial direct precedent. The unresolved question for this project is the performance of a particular adaptive acquisition method under its specified instrument and plume conditions. It is not reasonable to infer either novelty or futility from a broad literature label.

<!-- pagebreak -->

## Proposed Bayesian state formulation

The following is a proposed synthesis, not a transcription of one paper or a decision to replace the current implementation. Represent the projected enhancement field by a finite vector x_t. Keep its discretization independent of the physical beam footprint, readout spacing, and display resolution. The field can initially be coarse, provided approximation error and physical units remain explicit.

Below, y_i denotes the retrieved value after declared background correction; the original reading remains in the evidence record. Residual uncertain bias belongs in the state or error model. The general sequential model is

$$x_{k+1}=F_k x_k+u_k+\eta_k,\qquad y_i=h_i^T x_{t_i}+\epsilon_i.$$

F_k describes a declared short-horizon evolution; u_k is a justified input; eta_k represents process error. The vector h_i maps the projected field to one reported reading, including the beam footprint and, where needed, temporal response through an augmented state. If the latent state were instead physical three-dimensional concentration, h_i would contain ray-length weights across many voxels. These are different models, even when both produce one scalar.

For a fixed linear Gaussian approximation, posterior mean and covariance can be updated by standard conditioning. This supplies a mathematically explicit neighborhood mechanism without requiring independent cell-occupancy probabilities. Gaussian-process regression provides the corresponding correlated-observation framework, and filtering theory supplies the temporal recursion.[^13][^14]

$$K_i=\frac{P_i^- h_i}{h_i^T P_i^- h_i+r_i},$$
$$\mu_i^+=\mu_i^-+K_i(y_i-h_i^T\mu_i^-),\qquad P_i^+=P_i^--K_i h_i^T P_i^-.$$

This choice remains provisional. A Gaussian enhancement model can represent signed deviations around background, but does not guarantee positive absolute concentration. Retrieval noise may be biased, correlated between adjacent samples, heteroscedastic, or affected by common calibration error. Repeated readings reduce independent noise; they do not remove a shared bias. If an instrument exports its own posterior estimate, its retrieval prior must not be counted again as independent evidence.

For a projected map, physical wind is not automatically a uniform angular displacement. Plume range, depth, viewing geometry, and line-of-sight integration affect the apparent motion. A simple image-space advection proxy is acceptable as a declared approximation and test case. It should not be presented as a direct consequence of a measured wind vector without the required mapping.

Keep source assumptions explicit. A known-source input can be evaluated as one mode. It must not inject the true source into an unknown-source run. Finite release, plume departure, background intrusion, and model decay should be represented in development cases rather than repaired through display normalization.

The operational decision uses only observations available at that time. Measurements arriving later can improve a retrospective reconstruction or a current estimate through a declared delayed-data update. They cannot retrospectively improve a scan decision already executed. Smoothing has scientific value, but its output time and latency must remain visible.

<!-- pagebreak -->

## Measurement value and time

The most useful additional theoretical reference is goal-oriented experimental design. Attia, Alexanderian, and Saibaba select observations to reduce uncertainty in a specified prediction or quantity of interest; their transport example distinguishes vector reconstruction from a scalar summary.[^5] This fits the distinction between mapping a plume and estimating its strength, while leaving the instrument's mechanics to be supplied separately.

Let D_t contain the evidence actually available at the decision time. Let S be an ordered sweep belonging to the feasible action set F_theta of a fixed instrument profile theta. Its cost T_theta(S,p_t) includes the move from the current position, vertical travel, horizontal repositioning, settling, and acquisition subject to their actual overlap rules. A maximum duration B limits admissible sweeps.

An initial common-deadline objective is

$$S^*=\underset{S\in\mathcal{F}_{\theta},\ T_{\theta}(S,p_t)\leq B}{\operatorname{argmin}}\ \mathbb{E}[L(x_{t+H},\hat x_{t+H}(D_t,Y_S))\mid D_t],$$

where the horizon H is fixed across candidates and H is at least B. The expectation averages possible readings under the declared model; it does not inspect their future realized values. Each candidate has its own acquisition times, and only data delivered by the evaluation deadline may contribute. A simple loss is area- or solid-angle-weighted squared error over the declared reconstruction domain.

For a defined scalar strength functional g(x), add a separately scaled error term for g, or report it as a distinct objective. Its weighting is a scientific choice, not a free optimization knob. An angular integral can be a reproducible signal index, but it becomes physical mass or flux only with justified geometry and conversion factors. A nonlinear emission functional may need numerical uncertainty propagation rather than the linear formula below.

A terminal score alone does not value early delivery. For continuous monitoring, compare policies over the same physical window using a time-integrated loss,

$$J(\pi)=\mathbb{E}\left[\int_t^{t+H} L(x_s,\hat x_s^{\pi})\,ds\mid D_t\right].$$

Here pi specifies repeated completed-sweep decisions, each held fixed during execution. This is a formulation of what matters, not a demand for a general POMDP solver. A bounded candidate comparison or one-sweep lookahead can be evaluated first. If only the next sweep is compared, subsequent behavior within the window must be held to the same declared continuation rule.

Both objectives prevent an accidental comparison between a fast policy scored on an early, difficult plume and a slow policy scored after that plume has disappeared. Coverage, completion time, missed signal, and reconstruction error should also be reported individually; a single aggregate score can hide the reason for failure.

<!-- pagebreak -->

## An interpretable measurement-value calculation

For a fixed Gaussian approximation, the benefit of a candidate reading has a simple expression. At a common target time H, write P_H for predicted covariance, C_Hi for the covariance between x_H and the proposed observation y_i, and v_i for the observation variance including noise. Conditioning gives

$$P_H^+=P_H-\frac{C_{Hi}C_{Hi}^T}{v_i}.$$

For weighted squared reconstruction error, the expected reduction is

$$\Delta_i^{\mathrm{map}}=\frac{C_{Hi}^T W C_{Hi}}{v_i}.$$

For a linear strength index g(x)=a^T x, it is

$$\Delta_i^{\mathrm{strength}}=\frac{(a^T C_{Hi})^2}{v_i}.$$

These are direct consequences of Gaussian conditioning, stated here as an analytical illustration. They show how neighbor correlation and the chosen target enter measurement value. Dynamics enter through C_Hi: a very old observation may have little relation to the current target field. Joint sweep value must account for correlations between readings; adding independent point scores can double-count redundant information.

Consider two independent projected cells, equal measurement cost, no evolution, observation variance 0.25, and a total-signal target x_A+x_B. Use the deliberately hypothetical values below.

| Candidate | Mean signal | Prior variance | Variance reduction from one reading |
| --- | --- | --- | --- |
| Cell A | 100 | 0.04 | 0.00552 |
| Cell B | 10 | 4.00 | 3.76471 |

The calculation is P_jj squared divided by P_jj+0.25. Cell A contains 90.9% of the mean signal, but cell B offers far more reduction in uncertainty about total signal. This is not an experimental plume result or a proposed sensor calibration. It demonstrates why highest predicted signal and greatest measurement value can disagree.

There is an equally important limitation: with a fixed linear Gaussian model and a fixed quadratic loss, covariance reduction does not depend on the realized measurement amplitude. Such a criterion can become a largely predetermined uncertainty-coverage schedule. A plume-following policy may additionally require a signal-relevant loss, learned model parameters, a justified non-Gaussian model, or an explicit exploration/tracking balance. These choices must be named rather than hidden inside “likelihood.”

Separate three quantities currently grouped under “likelihood”: the measurement likelihood p(y_i|x), posterior presence probability Pr(x_j>tau|D), and normalized predicted signal mass used for selection. Presence probabilities need not sum to one. For a declared threshold tau and cell weight w_j, the mass rule is

$$m_j=w_j\mathbb{E}[(x_j-\tau)_+\mid D],\qquad b_j=m_j/\sum_k m_k.$$

The 90% rule selects a region whose b_j sum reaches 0.90 when total mass is positive. It can remain a baseline; it guarantees neither reconstruction accuracy nor whole-plume containment. Recompute regions from evidence without requiring each to be smaller than its predecessor. Near-zero mass requires an explicit search or surveillance rule.

<!-- pagebreak -->

## Transfer decisions and fixed instrument contract

The following decisions are recommendations for formulation and comparison. They do not authorize a new implementation stage.

| Idea | Decision | Adaptation required |
| --- | --- | --- |
| Expected reconstruction improvement | Use | Define the observable, loss, and common evaluation time first. |
| Goal-specific uncertainty | Use | Keep map accuracy and the chosen strength quantity distinguishable. |
| Line or sweep candidate actions | Use | Score complete paths allowed by the reference scanner. |
| Local evidence freshness | Adapt | Propagate or discount stale evidence using an explicit temporal model. |
| Peripheral surveillance | Adapt | Pay for every beam visit and all connecting travel. |
| Learned image-utility predictor | Defer | First establish a reliable measurement model and interpretable baseline. |
| Patterned-optics foveation | Do not transfer directly | Its observation operator and variable cells differ from this instrument. |
| Multi-view gas tomography | Separate extension | Requires geometry and assumptions beyond a projected image. |
| Retrospective smoothing | Optional reconstruction output | Preserve latency and keep decision-time evidence causal. |

Freeze physical beam footprint, angular sampling lattice, integration/readout settings, allowed sweep pattern, speed limits, and positioning/settling rules for the reference experiment. Keep them configurable between named profiles. Also hold the source scenario, calibration, and evidence interface constant when comparing one algorithmic change. Display resolution and playback speed remain presentation choices.

The existing May point and column demonstrations represent different profiles; they should not be blended into a new instrument with superficially similar timing numbers. A historical profile may contain physically incomplete accounting. Preserve its reproducible behavior as historical evidence, and disclose any accounting correction as a separate profile change. Recovering a profile does not require retaining a causality defect.

Choose feasible complete sweeps early in candidate generation. An arbitrary mask followed by aggressive time-budget packing can discard the very region the inference selected. A coverage-constrained candidate should be checked after mechanical conversion. If no candidate retains the required mass within B, expose that incompatibility; evaluate an explicitly named fallback instead of preserving a nominal 90% claim.

Repeated trimming is unnecessary in this formulation. A stable plume can retain its region; new evidence can move or enlarge it. Reacquisition is a resource allocation decision with a visible cost. The sensor's spatial footprint does not enlarge merely because the planner visits fewer positions or the renderer displays larger blocks.

<!-- pagebreak -->

## Experiments that can falsify the proposal

The scientific claim should be conditional: with this instrument and these declared plume regimes, adaptive acquisition improves a stated reconstruction or strength measure relative to simple scanning policies. The claim fails if apparent gains disappear after correcting timing, geometry, resource use, or access to future information. Negative results identify the regime or assumption that needs reconsideration; they do not justify silently changing the sensor.

Begin with a forward model that produces beam-integrated concentration values at actual acquisition times. A synthetic projected field is sufficient for an initial algorithm test if it is labeled as such. Before claiming physical concentration or emissions, add a distinct three-dimensional forward projection with independently specified geometry. Use different discretizations or dynamics in generation and inference to expose model mismatch, rather than validating only a model against itself.

| Development case | Question or failure condition |
| --- | --- |
| Static plume, known response | Does inference recover amplitudes and uncertainty without needing motion compensation? |
| Moving finite plume, known motion | Does reduced scan-time spread improve reconstruction at common times? |
| Birth, cessation, and displacement | Does a focused region miss new signal or continue reconstructing an exhausted plume? |
| Wind/range error and shape change | Do prediction errors produce confident but incorrect reconstructions? |
| Instrument lag and pointing error | Can static-target displacement be separated from actual plume movement? |
| Background or calibration drift | Does repeated evidence incorrectly eliminate shared uncertainty? |
| Longer actual execution | Do timing overruns affect prediction and evaluation honestly? |

Use a full legal raster or sweep, a fixed corridor where its prior is justified, a uniform or age-based rescan schedule, and the original 90% proposal as baselines. Add one reconstruction-value candidate only after the shared model is working. A known-source corridor is an appropriate comparator in a known-source setting; it must not be secretly given the source in an unknown-source comparison.

Evaluate error over a common physical window and record time to a stated usable reconstruction. Include absolute and normalized amplitude error, integrated-signal error with explicit units, missed enhancement, and uncertainty calibration. Report truth-free runs with these metrics unavailable when truth is absent. A visually sharp image is not sufficient evidence of quantitative accuracy.

Apply common budget ceilings, but record actual travel and acquisition expenditure. Equal ceilings do not imply equal cost. Compare error-versus-time trajectories and the set of achievable cost/error tradeoffs; only label expenditures exactly matched when they are. A change of FPS must leave these results unchanged.

The first useful outcome may be a boundary: focusing helps only when plume persistence exceeds a certain relationship to sweep duration and prediction uncertainty remains small enough. That is a meaningful application result if the boundary is measured under fixed conditions. No universal monotonic relationship between smaller region, shorter scan, and better reconstruction is assumed.

<!-- pagebreak -->

## Repository implications and open decisions

The callable plan/readings/completion boundary remains compatible with this formulation. It already expresses the correct broad separation between inference and completed-sweep decisions. The evidence messages should eventually identify retrieved concentration units, one-way or round-trip convention, actual beam direction, temporal support, quality flags, and any calibration or range metadata needed to interpret the scalar. These are recommendations for a later reviewed change, not a claim that every field is present today.

The current coarse Gaussian model is one candidate estimator. Its mathematical consistency does not establish that its projected dynamics, source input, or uncertainty match the instrument. Likewise, expected positive signal is one allocation surface; its uncertainty contribution should not be mistaken for independent evidence of plume presence. Keep the estimator and policy independently replaceable so the 90% rule can be compared using the same reconstruction model.

The historical benchmark's **not_demonstrated** result remains evidence about its tested model and scanner profile. This proposed formulation neither overturns that negative result nor demonstrates a new policy's advantage.

A familiar physical-time view can show the evolving reference scene, accumulated reported measurements, scanner progression, and the inferred field at a declared time. Optional uncertainty or diagnostic panels should support inspection without becoming a different scientific demonstration. A reconstructed current image and a historical measurement mosaic are different products and need different labels. Both are useful for explaining temporal distortion.

Four decisions remain before a new implementation plan: the instrument's exact retrieved quantity and response timing; the angular/physical geometry of the field of view; the meaning and units of the strength target; and which fixed mechanical profile will serve as the reference. A fifth is whether output is required immediately or whether a declared reconstruction latency is acceptable. These choices determine the objective and evidence needed more than the choice between named optimization libraries.

The practical sequence is to specify those quantities, reproduce the reference acquisition, validate an estimator on shared evidence, and then compare a small set of legal next-sweep policies. Retain the original mass-selection idea during that comparison. Adopt a more elaborate policy only if it improves the stated scientific objective within the same instrument constraints.

## Evidence limits

The assessment supports selective use of existing methods, not a claim of an unexplored field or proven adaptive superiority. FAST and the foveated paper were examined beyond their abstracts; the latter's 2016 preprint was checked against its 2017 journal record. Direct gas work spans different geometries, source assumptions, and temporal scales. The 2025 and 2026 pan-tilt papers are assessed at the accessible abstract level, so their full scan algorithms and comparison protocols remain unresolved here.

The strongest supported direction is **Bayesian reconstruction of the reported beam-integrated field, with sweep selection judged by reconstruction value under fixed mechanics and explicit time**. Whether that method improves plume-strength estimation is an experiment. This memo defines how to make that experiment informative without substituting a different instrument or claiming calibration from synthetic images.

<!-- pagebreak -->

## Sources

1. Saugat Kandel et al. **Demonstration of an AI-driven workflow for autonomous high-resolution scanning microscopy.** Nature Communications 14, 5501, 7 September 2023. [Publisher](https://www.nature.com/articles/s41467-023-40339-1). Relevant sections: workflow, discussion, SLADS-Net methods. Full main text examined. Basis for reconstruction utility, batching, and the travel-cost limitation.

2. David B. Phillips et al. **Adaptive foveated single-pixel imaging with dynamic supersampling.** Science Advances 3(4), e1601782, 21 April 2017. [Journal record and text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5400451/); [2016 author preprint and supplement](https://arxiv.org/html/1607.08236v1). Relevant sections: optics, gaze control, temporal fusion. Full preprint and indexed journal text examined; hardware-specific gains are not transferred.

3. David Helminiak, Hang Hu, Julia Laskin, and Dong Hye Ye. **Deep Learning Approach for Dynamic Sparse Sampling for High-Throughput Mass Spectrometry Imaging.** Electronic Imaging 2021, 2901-2907. [Publisher article](https://doi.org/10.2352/ISSN.2470-1173.2021.15.COIMG-290); [author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC8553253/). Indexed methods, including line-bounded acquisition, examined. Supports legal action geometry, not moving-gas reconstruction.

4. Hang Hu et al. **High-Throughput Mass Spectrometry Imaging with Dynamic Sparse Sampling.** ACS Measurement Science Au 2(5), 466-474, published online 15 August 2022. [Publisher](https://pubs.acs.org/doi/10.1021/acsmeasuresciau.2c00031). Publisher abstract and indexed methods examined. Experimental linewise improvement is distinguished from simulated pointwise improvement.

5. Ahmed Attia, Alen Alexanderian, and Arvind K. Saibaba. **Goal-Oriented Optimal Design of Experiments for Large-Scale Bayesian Linear Inverse Problems.** Inverse Problems 34, 095009, 2018. [DOI](https://doi.org/10.1088/1361-6420/aad210); [author paper](https://arxiv.org/pdf/1802.06517). Sections on goal-oriented covariance criteria and the contaminant-transport example examined. Supports task-specific utility; fixed sensor placement is not a timed mechanical sweep.

6. Antti Voss, Elias Vänskä, Damien Weidmann, Aki Pulkkinen, and Aku Seppänen. **Multi-open-path laser dispersion spectroscopy combined with Bayesian state estimation for localizing and quantifying methane emissions.** Atmospheric Environment: X 22, 100260, April 2024. [Publisher](https://www.sciencedirect.com/science/article/pii/S2590162124000273). Publisher abstract and institutional metadata examined. Model details are not inferred beyond accessible evidence.

7. Elias Vänskä, Damien Weidmann, and Aku Ursin. **Greenhouse gas emission mapping and quantification based on 3D transport modeling and Bayesian state estimation.** Inverse Problems 41, 095001, 3 September 2025. [Publisher](https://doi.org/10.1088/1361-6420/adfb29). Indexed publisher abstract, introduction, and conclusions examined. Supports rank-deficient tomography and transport-constrained inference, not adaptive allocation superiority.

<!-- pagebreak -->

## Sources continued

8. Kenneth Scheel, Elias Vänskä, Damien Weidmann, and Aku Ursin. **Dynamic quantification of methane emissions at facility scale using laser tomography: demonstration of a farm deployment.** Atmospheric Measurement Techniques 19, 2343-2367, 10 April 2026. [Publisher](https://amt.copernicus.org/articles/19/2343/2026/). Full main text examined, especially sections 2.1-2.3 and 3.4-3.5. Basis for retrieved-data uncertainty, smoothing latency, and model-error limits.

9. Di Wang et al. **Vertical Distribution Mapping for Methane Fugitive Emissions Using Laser Path-Integral Sensing in Non-Cooperative Open Paths.** Sensors 24(4), 1307, 18 February 2024. [Publisher](https://www.mdpi.com/1424-8220/24/4/1307); [author article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10892198/). Abstract and indexed scan-duration/gas-bag passages examined. Detailed numerical results are not generalized to a moving plume.

10. Senyuan Wang et al. **Active methane gas imaging and leakage quantification based on standoff TDLAS and pan-tilt units.** Sensors and Actuators B: Chemical 436, 137698, issue dated 1 August 2025. [Publisher](https://www.sciencedirect.com/science/article/pii/S0925400525004733). Publisher abstract examined. Full scan policy and experimental protocol not independently assessed.

11. Xiachun Wang et al. **Dynamic Imaging and Inverse Quantification Method of Methane Gas Cloud with Laser Scanning Technology.** Environmental Science & Technology 60(14), 10688-10703, issue dated 14 April 2026. [Publisher abstract](https://pubs.acs.org/doi/abs/10.1021/acs.est.5c16972); [PubMed record](https://pubmed.ncbi.nlm.nih.gov/41923449/). Abstract-level assessment. Publisher lists online publication 2 April; PubMed records article date 1 April. This discrepancy does not change the technical assessment.

12. John Worden, Paul Green, Annmarie Eldering, and Evan Sherwin. **Common Practices for Quantifying Methane Emissions from Plumes Detected by Remote Sensing.** NIST IR 8575, May 2025. [Report](https://nvlpubs.nist.gov/nistpubs/ir/2025/NIST.IR.8575.pdf). Sections 2.1-2.2 and quantification discussion examined. Its satellite-oriented geometry is not assumed to match a ground-based scanner.

13. Carl Edward Rasmussen and Christopher K. I. Williams. **Gaussian Processes for Machine Learning**, chapter 2, Regression. MIT Press, 2006. [Author-hosted chapter](https://gaussianprocess.org/gpml/chapters/RW2.pdf). Gaussian conditioning supports the illustrative covariance algebra; it does not establish the appropriate plume prior or dynamics.

14. Simo Särkkä and Lennart Svensson. **Bayesian Filtering and Smoothing**, second edition. Cambridge University Press, 2023. [Institutional record](https://research.aalto.fi/en/publications/bayesian-filtering-and-smoothing-3/); [author-hosted book](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf). General reference for temporal state estimation; no claim of a plume-specific solution or transferred numerical result.

[^1]: Kandel et al., [FAST scanning microscopy](https://www.nature.com/articles/s41467-023-40339-1), 2023, workflow and discussion.
[^2]: Phillips et al., [Adaptive foveated single-pixel imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC5400451/), 2017; author preprint sections I-III.
[^3]: Helminiak et al., [Dynamic sparse sampling](https://pmc.ncbi.nlm.nih.gov/articles/PMC8553253/), 2021, line-bounded acquisition.
[^4]: Hu et al., [High-throughput mass spectrometry imaging](https://pubs.acs.org/doi/10.1021/acsmeasuresciau.2c00031), 2022.
[^5]: Attia et al., [Goal-oriented experimental design](https://arxiv.org/pdf/1802.06517), 2018, sections 3-4.
[^6]: Voss et al., [Bayesian methane state estimation](https://www.sciencedirect.com/science/article/pii/S2590162124000273), 2024, abstract.
[^7]: Vänskä et al., [3D gas transport and Bayesian estimation](https://doi.org/10.1088/1361-6420/adfb29), 2025.
[^8]: Scheel et al., [Dynamic methane quantification](https://amt.copernicus.org/articles/19/2343/2026/), 2026, sections 2.2 and 3.4-3.5.
[^9]: Di Wang et al., [Vertical methane distribution mapping](https://www.mdpi.com/1424-8220/24/4/1307), 2024.
[^10]: Senyuan Wang et al., [Standoff TDLAS imaging](https://www.sciencedirect.com/science/article/pii/S0925400525004733), 2025, abstract.
[^11]: Xiachun Wang et al., [Dynamic laser-scanning gas imaging](https://pubs.acs.org/doi/abs/10.1021/acs.est.5c16972), 2026, abstract.
[^12]: Worden et al., [NIST IR 8575](https://nvlpubs.nist.gov/nistpubs/ir/2025/NIST.IR.8575.pdf), 2025, sections 2.1-2.2.
[^13]: Rasmussen and Williams, [Gaussian Processes for Machine Learning, chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf), 2006.
[^14]: Särkkä and Svensson, [Bayesian Filtering and Smoothing](https://research.aalto.fi/en/publications/bayesian-filtering-and-smoothing-3/), second edition, 2023.
