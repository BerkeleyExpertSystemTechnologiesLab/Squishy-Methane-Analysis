# Recorded Data Work Plan

Last updated: 2026-07-31

Parent authority: [`SYSTEM_GOAL.md`](SYSTEM_GOAL.md)

Starting branch: `codex/recorded-ogi-sensor-emulation`

Starting commit: `17c7f8c`

Workflow status: `blocked` on external evidence

Active stage: none. R0B is blocked on owner permission and authoritative
assignment/interval evidence. R0C is closed with result
`engineering_not_demonstrated`: its recorded-input plumbing passed, but the
unchanged raw profile requested approximately the full frame throughout.

External-data status: mixed — GasVid `acquired_unverified`; USU R0C
`licensed_diagnostic_acquired`

Claim status: `not_demonstrated`

Engineering checkpoint: R1 through `48c7d8d`; R0B window contract through
`366730d`; R0B-Q qualification gate through `54893a9`; pre-accession R0C
contract through `523df93`; R0C evidence through `a8a65b0`

Engineering verification: 307 passed, 5 Windows capability skips;
`compileall` and `git diff --check` passed

This is the durable execution tracker for recorded-data preparation and
evaluation. It is subordinate to `SYSTEM_GOAL.md`, which remains authoritative
for system completion gates and claim status. In particular, System Goal Stage
4 and G3 remain `blocked` until independently recorded development and frozen
held-out sequences have been acquired, verified, and run. Closing R0 as
blocked here does not change that status.

## Workflow contract

> Establish a provenance-preserving, dataset-neutral workflow that converts
> externally recorded OGI sequences into reproducible causal virtual-scan
> episodes, while keeping raw plume dynamics, sensor emulation, and offline
> evaluation truth separate.

GasVid is Candidate 1, not a privileged code path. The workflow must emit the
existing recorded sequence and frozen-suite contracts so later candidates can
be substituted without changing the causal runner.

### Three-layer evidence design

| Layer | Role | Planner access | Claim meaning |
| --- | --- | --- | --- |
| Raw OGI video | Preserves externally recorded plume motion, scene structure, compression, and camera instability | Requested samples only in separately labeled diagnostic runs | Robustness to OGI imagery; not a clean gas sensor |
| Structure-suppressed sensor emulation | Primary virtual-scanner candidate; fixed per-pixel background/noise calibration subtracts estimated stable structure and produces non-negative relative evidence | Sampled coordinates and available timestamps only | Planning over an OGI-derived cleaner-sensor hypothesis; not validated hardware equivalence |
| Pseudo plume masks | Offline weighted evaluation truth and disagreement analysis | Never | Diagnostic silver evidence; not concentration, gas mass, or independently verified truth |

The primary sensor-emulation transform is conceptually:

```text
fixed calibration maps (background, noise, polarity)
    + sampled raw OGI value at time t
    -> standardized signed residual
    -> frozen polarity mapping
    -> non-negative gas-evidence amplitude
```

Background and noise maps may be computed from a declared calibration interval
selected without planner results or held-out outcomes. At experiment time, the
virtual scanner must evaluate the transform only at requested samples. A
full-frame transform may be used offline to build evaluation artifacts or an
explicitly labeled perception upper bound, never silently as planner evidence.

### Claim boundaries

Evidence from this workflow may support:

- causal planner execution on external plume dynamics through a declared
  virtual scanner;
- comparison of scan policies at exact matched modeled mechanical cost;
- robustness findings for raw OGI and structure-suppressed sensor inputs;
- diagnostic plume-coverage findings when the evaluator uses silver masks.

It cannot by itself support:

- physical scanner, pan-tilt, latency, or hardware-readiness claims;
- synchronized wind or anemometer calibration unless such data have verified
  acquisition provenance;
- methane concentration, mass-flow, or radiometric measurement claims;
- source discovery when release location or a source corridor is supplied;
- a final `supported` claim from pseudo masks;
- safety, field-deployment, or operational leak-detection guarantees.

Modeled scanner time must be labeled `counterfactual_virtual` and reported
separately from video decode or rendering time. OGI intensity is not gas
concentration. No full-frame, future-frame, pseudo-mask, or evaluation-only
signal may enter the causal planning path.

## Source and accession registry

Primary sources must be retained with access dates and local snapshots or
content hashes where redistribution permits. A publication URL does not prove
dataset access or derivative rights.

| ID | Source | Primary URL | Intended use | Verification |
| --- | --- | --- | --- | --- |
| S1 | Original GasVid / GasNet paper, *Machine vision for natural gas methane emissions detection using an infrared camera* | https://svl.stanford.edu/gibson2/assets/Wangetal.2019.pdf | Original acquisition design and class-0/background description | Primary PDF checked 2026-07-31; no dataset licence or reuse permission found in the paper |
| S2 | Stanford VideoGasNet publication record | https://eao.stanford.edu/publications/journal-articles/videogasnet-deep-learning-natural-gas-methane-leak-classification | Bibliographic provenance for the later classification study | Stanford record checked 2026-07-31 |
| S3 | FLIR GF320 manufacturer specification | https://www.flir.com/products/gf320 | Camera capability and recording-format boundary | Manufacturer page verified 2026-07-26 |
| S4 | METEC official curated-data project | https://metec.colostate.edu/curated-dataset/ | Check for separately accessioned facility and meteorological data | Project verified 2026-07-26; no dataset join inferred |
| S5 | LangGas primary paper | https://openaccess.thecvf.com/content/CVPR2025W/PBVS/papers/Guo_LangGas_Introducing_Language_in_Selective_Zero-Shot_Background_Subtraction_for_Semi-Transparent_CVPRW_2025_paper.pdf | Later GasVid segmentation/annotation methodology | Paper verified 2026-07-26; confirms GasVid segmentation evaluation is qualitative for lack of ground truth |
| S6 | FGSTP primary paper | https://arxiv.org/abs/2505.00295 | Later manually labeled GasVid segmentation study | Primary paper checked 2026-07-31; it reports 14 selected videos (9 train, 5 test), but no authoritative mask/code accession or reuse terms have been established |
| S7 | Data Cards paper | https://research.google/pubs/data-cards-purposeful-and-transparent-dataset-documentation-for-responsible-ai/ | Dataset documentation structure | `pending` |
| S8 | FAIR principles | https://doi.org/10.1038/sdata.2016.18 | Asset identity, provenance, and reuse practice | `pending` |
| S9 | Public GasVid provider folder | https://drive.google.com/drive/folders/1JKEMtCGPSq2IqGk4uXZP9A6gr8fyJsGC | Exact media accession and native logging sheet | Accessed 2026-07-26 and rechecked 2026-07-31; 31 MP4 files plus one Google Sheet; no licence or reuse terms were visible and the provider owner is hidden in the public UI |
| S10 | LangGas implementation repository | https://github.com/weathon/Lang-Gas#test-on-gasvid | Route evidence linking S9 as the original-author download | Route verified 2026-07-26; this is not a GasVid rights grant |
| S11 | VideoGasNet primary publisher record | https://www.sciencedirect.com/science/article/pii/S0360544221017643 | Paper-defined modeled subset, pad roles, and 3-minute class convention | Primary record checked 2026-07-31; the study uses 25 of the 31 recordings after excluding five 18.6 m files and the single 8.8 m file |
| S12 | Utah State University, *Optical Gas Imaging Survey of Oil and Gas Wells in the Uinta Basin, Utah* | https://digitalcommons.usu.edu/all_datasets/82/ | Rights-clean recorded-OGI engineering fallback | Institutional dataset record checked 2026-07-31; CC BY 4.0; seven substantive downloadable OGI videos with per-file MD5s; controlled clips are propane and the field example is identified only as hydrocarbon |
| S13 | Lyman et al., *Aerial and ground-based optical gas imaging survey of Uinta Basin oil and gas wells* | https://online.ucpress.edu/elementa/article/doi/10.1525/elementa.381/112514/Aerial-and-ground-based-optical-gas-imaging-survey | Acquisition, camera-mode, controlled-release, and meteorological context for S12 | Primary paper checked 2026-07-31; ground GF320 footage used a stabilized high-sensitivity mode and measured meteorology, but no frame-synchronous wind/video join is established |
| S14 | CSU Mountain Scholar, Martinez thesis and supplementary natural-gas OGI videos | https://mountainscholar.org/items/ef0f6244-482c-44ad-9d7e-98dbc9d2fe86/full | Methane-relevant fallback review | Repository record checked 2026-07-31; supplementary ZIP contains videos, but the item is explicitly in copyright and grants no general derivative permission |
| S15 | BASE Lab Controlled Diet Methane Dataset | https://huggingface.co/datasets/baselab/controlled-diet-methane-dataset | Later measurement/normalization and spatial-mask diagnostics | Dataset card checked 2026-07-31; CC BY 4.0; 4,885 methane OGI images and masks, but no authoritative source-video clock or contiguous temporal sequence is published |
| S16 | CARB Aliso Canyon infrared video record | https://ww2.arb.ca.gov/aliso-canyon-natural-gas-leak-infrared-camera-images-and-videos | Real methane sequence candidate requiring separate rights review | Government page checked 2026-07-31; downloadable methane footage has useful event timing, but no explicit derivative licence is attached and CARB's site terms warn that some media require owner permission |

### Candidate ledger

| Candidate | Description | Authoritative accession | Rights/terms | Local assets | Evidence grade | Status |
| --- | --- | --- | --- | --- | --- | --- |
| C1 `gasvid` | Externally recorded controlled-release OGI video described by S1/S11 | `gasvid-public-google-drive-20260726` from S9, linked by S10 | Primary-paper, publisher, provider-folder, and route audit found no explicit dataset licence, local-processing permission, derivative permission, or redistribution grant | 31 audited MP4s plus the provider logging CSV; raw bytes remain ignored | `unverified_external` | `blocked` on rights and evidence preparation |
| C2 `usu-uinta-ogi` | Independently recorded GF320 OGI videos from field hydrocarbon observations and controlled propane releases | S12, DOI `10.26078/BQPE-JT65` | CC BY 4.0 on the institutional dataset record; adaptation and redistribution allowed with attribution | None locally | `licensed_external_engineering` | `selected` for a raw engineering smoke only; not methane evidence |
| C3 `csu-metec-martinez` | Controlled natural-gas OGI supplementary videos with release-rate and wind context | S14, DOI `10.25675/3.05018` | In copyright; local derivative and publication rights not established | None locally | `unverified_external` | `blocked` pending permission |
| C4 `controlled-diet-methane-images` | Methane OGI PNG frames with segmentation masks from nine recordings | S15 | CC BY 4.0 | None locally | `licensed_external_images` | `standby` for measurement/spatial diagnostics; not a causal sequence |
| C5 `carb-aliso` | Long real-methane incident and no-leak follow-up footage | S16 | Downloadable, but no explicit derivative licence and site terms require rights caution | None locally | `unverified_external` | `blocked` pending permission and independence design |

### Accession and rights ledger

| Candidate | Access date | Provider/URI | Terms or license | Derivatives allowed | Redistribution allowed | Correspondence/evidence | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | 2026-07-26; primary-source recheck 2026-07-31 | S9 public Google Drive folder; linkage recorded by S10 | No explicit dataset terms or permission located in S1, S2, S9, S10, or S11; `rights_status=unverified` | `unknown` for interval extraction, calibration products, masks, figures, and rendered clips | `unknown` for raw and derived assets; record separately | tracked accession manifest; audit SHA-256 `246fc11d...a189`; provider-sheet SHA-256 `1869c151...db55`; primary URLs S1/S2/S9/S10/S11 | Local identity/decode audit only; do not create frame derivatives, prepare sequences, publish media, redistribute, or claim eligibility until the required permission scope is verified |
| C2 | Primary-source review 2026-07-31 | S12 institutional dataset record and S13 paper | CC BY 4.0 | Yes, with attribution | Yes, with attribution | DOI, institutional licence statement, published per-file MD5s, and paper acquisition description | Eligible for accession/qualification as an engineering-only recorded-OGI candidate; gas identity and evidence role remain separate metadata |
| C3 | Primary-source review 2026-07-31 | S14 institutional repository | In copyright; use beyond applicable legal exceptions requires rights-holder permission | `unknown` | `unknown` | repository `dc.rights` and `dcterms.rights.dpla` records | Do not download for transformation or construct derivatives under the current evidence contract |
| C4 | Primary-source review 2026-07-31 | S15 dataset repository | CC BY 4.0 | Yes, with attribution | Yes, with attribution | dataset card and licence | May support image-domain measurement/spatial diagnostics; cannot be reconstructed into a temporal sequence from filenames |
| C5 | Primary-source review 2026-07-31 | S16 CARB record and CARB conditions of use | No asset-specific derivative licence found | `unknown` | `unknown` | dated government media page and site conditions | Do not prepare or publish derivatives without a clearer rights record |

If access is granted without redistribution rights, raw and derived media remain
external and ignored; only code, small manifests, commands, and hashes may be
tracked. If rights cannot be established, C1 becomes `blocked` or
`not_demonstrated`; the workflow must remain usable with another candidate.

## Asset inventory and lineage

A local ignored GasVid accession is present and byte/decode-audited. Presence
and decode validity do not resolve reuse rights, capture grouping, leak-rate
semantics, spatial truth, or independence. The accession contains the full 31
recordings described by the dataset papers. It must not be silently reduced to
the later 25-file VideoGasNet modeling subset: S11 excluded the five 18.6 m
recordings and the single 8.8 m recording for that model, not because those six
files failed the local identity/decode audit.

### Acquisition inventory

| Asset ID | Candidate | Original filename | Acquisition/capture group | Source pad/site | Distance | Release interval/rate label | Shape/FPS/frames | Polarity/background notes | SHA-256 | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gasvid-mp4-corpus` | C1 | 31 files, `MOV_1237.mp4` through the provider inventory ending at `MOV_2583.mp4` | Provider rows form five strong candidate capture sweeps by date/time and repeated distance order, but the five group identities are inferred rather than provider-declared | The 18 `25xx` versus 13 `12xx`/`14xx` count match to the paper's pad counts is strong inference only; no filename-to-pad assignment is authoritative | Provider sheet: 4.6–18.6 m plus one 8.8 m file | Start times for native `Leak 0` and `Leak 1` clips; no authoritative duration, rate, wind, or spatial mask supplied | All 240×320; 14.8745–14.9216 FPS; 696,065 decoded frames | Raw compressed OGI display imagery; no clean-sensor or concentration claim | Individual hashes in audit `246fc11d...a189` | 31/31 `valid`; rights blocked; full accession distinct from S11's 25-file modeled subset |
| `gasvid-logging` | C1 | `GasVid Logging File` exported as `metadata/gasvid_logging.csv` | 31 rows keyed by video number; five candidate sweeps remain an inference | Pad/site field absent | Per-video value | `Leak 0`/`Leak 1` start timestamps only. `MOV_2568` has a 157 s gap and `MOV_2582` a 360 s gap instead of the nominal 180 s; these are unresolved anomalies, not automatic exclusions | CSV, 1,537 bytes | Metadata only; not planner evidence, interval duration, class truth, or spatial truth | `1869c151d59d712981ffcf6316ffe3919aeba2296448203ca1f5ceb4ab02db55` | acquired; rights unverified |

S1 describes the initial class-0 segment as non-leak/background, while its
class table assigns class 0 a nonzero `5.3 g CH4/h` value. Until the authors or
an authoritative companion record resolve this contradiction, `Leak 0`,
`class 0`, `background`, and `no release` must not be treated as equivalent
machine-readable facts. The provider timestamps are event markers, not proof
that every interval is exactly three minutes.

S6 reports a separately and manually labeled GasVid segmentation experiment,
but it selected only 14 videos and does not establish an accessible, hashed
mask corpus, source-code release, filename/frame mapping, licence, or
permission for this project. Those labels are not required for the first
truth-free raw causal pilot; they would matter only for a later, separately
rights-audited spatial evaluation track.

### Derived-asset lineage

| Derived ID | Parent hashes | Transformation/profile | Producing command | Producing commit | Output hash | Storage/rights | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `planned` |

Raw files are immutable. Every derived field, timestamp sidecar, calibration
map, pseudo mask, manifest, and report must identify its parents, resolved
profile, producing command, commit, and SHA-256. Rebuilding from identical
parents and configuration must reproduce identical content hashes.

## Cleaning and exclusion register

“Cleaned GasVid” is not an acceptable evidence label. The derived corpus name is
`gasvid_active_scan_silver_v1` only after its manifests are frozen.

| Operation | Classification | Rule |
| --- | --- | --- |
| Deterministic decode or content-preserving transcode | Allowed | Preserve frame order and pixel coordinates; record codec and hashes |
| Timestamp reconstruction | Allowed with caveat | Use verified container timing or a hashed sidecar; record that timing is reconstructed |
| Fixed coordinate-preserving crop | Allowed with caveat | Declare the crop once; do not choose it from plume masks or planner outcomes |
| Objective corrupt-frame or corrupt-file exclusion | Allowed | Record the exact integrity failure before benchmarking |
| Declared stable/transition interval extraction | Allowed with caveat | Retain transitions as a separately labeled stress stratum |
| Fixed per-pixel background/noise calibration | Allowed for sensor emulation | Development/calibration evidence only; freeze before held-out use |
| Causal temporal integration | Unsupported | Requires a future stateful scanner-response lifecycle; profile-v1 non-null values are rejected |
| Per-frame min/max normalization | Denied | Uses whole-frame context and destroys comparable amplitude |
| Per-frame histogram equalization | Denied | Changes temporal amplitude and can inject unscanned context |
| Truth-guided crop, inversion, threshold, or exclusion | Denied | Leaks evaluation truth into input preparation |
| Future-frame background or denoising at experiment time | Denied | Violates causal availability |
| Filling missing frames by freezing or looking ahead | Denied | Conceals unavailable samples |
| Removing difficult clips based on planner performance | Denied | Invalidates the frozen evaluation |
| Treating pseudo masks as sensor input | Denied | Evaluation-only information |

Every exception or new operation requires a decision-log row before it is used.
Negative and excluded cases remain visible in the inventory.

## Split and freeze ledger

Splits are by complete acquisition/capture group, never by individual frames or
overlapping clips. The paper-defined acquisition grouping is the initial
GasVid proposal and must be reconciled against the actual accession before use.

| Suite | Candidate/groups | Purpose | Freeze prerequisites | Manifest hash | Frozen at/model commit | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `gasvid_development_v1` | Paper development acquisition group, identity `pending` | Sensor-emulation and pseudo-truth calibration; policy development | Accession, rights, inventory, grouping, raw hashes | `pending` | `pending` | `planned` |
| `gasvid_heldout_v1` | Paper test acquisition group, identity `pending` | Untouched diagnostic acceptance | Development/held-out independence review and pre-change model commit | `pending` | `pending` | `planned` |
| `gasvid_transition_stress_v1` | Transition periods retained within their parent split | Intermittency and failure analysis | No group leakage; fixed extraction rule | `pending` | `pending` | `planned` |

Held-out media, annotations, and reports must not inform measurement transforms,
pseudo-truth parameters, model settings, or scan policy. A frozen suite may be
superseded only by appending a new version and rationale; its historical hashes
must not be overwritten.

## Sensor-emulation profile and calibration ledger

The public profile is dataset-neutral and behavior-changing. Every resolved
value, unit, provenance record, and calibration asset hash must be serialized
into the sequence manifest and report.

Proposed profile name: `structure_suppressed_gas_sensor_v1`

Primary output contract: non-negative gas-evidence amplitude. Signed residuals
may be retained as diagnostics but are not the primary planner input.

| Field/group | Meaning | Units/options | Provenance rule | Current value/status |
| --- | --- | --- | --- | --- |
| Background map | Fixed scene/equipment structure estimate | native decoded intensity | Separate audited leak-off asset; `development_only` or `independent_pre_run` | Per-pixel median implemented; no real calibration fitted |
| Noise map | Robust per-pixel instability scale | native intensity; positive floor required | Same audited calibration evidence | Normal-consistent MAD plus typed floor implemented; uncalibrated |
| Polarity | Maps signed residual toward gas evidence | `bright`, `dark`, `two_sided_magnitude` | Selected on development evidence only; never per frame | Typed and implemented; no real selection |
| Standardization | Residual divided by robust noise | dimensionless | Formula, floor, profile, and resolved calibration hash serialized | Implemented |
| Non-negative mapping | Produces gas-evidence amplitude | scale per noise sigma; fixed `[0, 255]` interface | Development-calibrated; no held-out tuning | Implemented as `uint8_relative_gas_evidence_v1`; scale uncalibrated |
| Spatial filtering | Optional local sensor response | pixels | Must operate only on mechanically available neighboring samples | Disabled until justified |
| Temporal integration | Future clean-sensor response | seconds | Requires a stateful scanner-response lifecycle | Unsupported in profile v1; non-null values rejected |
| Missing calibration behavior | Explicit failure/fallback | `reject` or labeled raw-OGI diagnostic | Must not synthesize a background silently | Default `reject` for primary track |
| Output units | Honest interface label | `relative_gas_evidence` | Never concentration or radiance | Implemented as a relative uint8-like interface |

Calibration assets belong to the development evidence and must be frozen before
held-out execution. A full-frame background map may be precomputed from
declared calibration data, but current full frames cannot be consumed to
renormalize sampled measurements.

## Evaluation-truth grades

| Grade | Definition | Allowed use | Claim eligibility |
| --- | --- | --- | --- |
| `none` | No spatial evaluation truth | Causal execution and qualitative inspection | Implemented in schema v3 and raw windowed schema v4; spatial metrics are unavailable, not zero-valued evidence |
| `silver_pseudo` | Weighted agreement from frozen automated proposal methods | Diagnostic metrics, method disagreement, failure discovery | At best `diagnostic_only_pseudo_truth`; unresolved rights produce `blocked_unverified_data_rights` |
| `gold_human_reviewed` | Frozen spatial labels reviewed independently of planner output under a documented protocol | Candidate independent evaluation, subject to provenance review | May be considered for G3/G5/G6; not automatically sufficient |

For `silver_pseudo`, individual proposal masks and an agreement-weight map are
stored separately. Proposed baseline methods and every threshold/version must be
fitted on development data and frozen before held-out generation. Pseudo masks
may be built from full frames offline because they are evaluator-only. Changing,
removing, or replacing truth must not change any planned scan.

Pseudo truth represents visible plume evidence, not methane concentration,
probability mass in physical units, or complete gas support. All reports using
it must include the pseudo-truth limitation. With verified rights, the derived
claim eligibility is `diagnostic_only_pseudo_truth`; unverified or restricted
rights produce the stricter `blocked_unverified_data_rights` label.

## Engineering checkpoint

The following reusable engineering implementation is complete even though no
external-data or performance stage is complete:

- commit `f0748ac`: strict, dataset-neutral accession manifest plus full
  hash/decode audit CLI; the artifact is always `not_sequence_evidence`;
- commits `1466faa` and `484b063`: fixed median/MAD structure suppression,
  explicit bright/dark/two-sided polarity, immutable profiles/calibration, and
  removal of spatial-order-dependent temporal smoothing;
- commit `ac38940`: strict sequence schema v2, exact shared audit parsing,
  hashed audit/capture lineage, audited calibration membership and scope,
  lossless silver/gold truth, verified truth-protocol assets, derived claim
  limitations, counterfactual-only scanner evidence, sample-only planner
  access in headless scan mode, and capture-group/source-media split isolation.
- commit `afacbdc`: canonical platform-neutral fixed-calibration artifacts
  with checksum validation and no-overwrite publication;
- commit `7eb26be`: sequence schema v3, truth-free causal execution, strict
  calibration result records, and semantic binding across source, profile,
  preparation inputs, producing commit, shape, sample count, and output bytes;
- commit `a074113`: deterministic dataset-neutral preparation with objective
  exclusion coverage, raw/emulated track separation, canonical reports, and
  on-demand sampled sensor emulation rather than a derived planner video;
- commit `b0eefe8`: deterministic suite freezing with explicit UTC time,
  full pre-change commit, exact preparation-report hash, and a companion
  lineage record;
- commit `c8099af`: audit/prepare/freeze CLI integration, end-to-end fixture
  coverage, and cross-suite media/calibration/capture-role isolation.
- commit `4eb339f`: audit-schema-v2 decoder build and decoded-pixel hashes,
  exact active-asset use, objective duplicate-byte handling, and strict
  freeze-record validation across audit, source media, calibration, prepared
  manifests, and suite membership; runtime sequence assets remain a separate
  suite-validation check.
- commit `067e06a`: accession-audit validity now enforces the recorded
  runtime's minimum three-frame input contract.
- commit `48c7d8d`: freeze CLI output distinguishes the non-evidence companion
  record from the claimless suite publication.

Verification: 208 tests passed and one Windows-only symlink capability test
was skipped. These are engineering fixtures and do not themselves provide
recorded evidence. The separately acquired GasVid accession is audited below,
but it is not a prepared corpus, frozen suite, or performance result.

## Stage ledger

Exactly one stage in this tracker may be `active`. Stage outcomes and negative
results are append-only.

### R0 — Candidate accession and rights audit

Status: `blocked`

Objective:

- establish the authoritative C1 accession, terms, completeness, and identity;
- verify primary-source facts against actual files;
- determine whether derivation and local evaluation are permitted;
- record unresolved label, timing, wind, and acquisition metadata without
  inference.

End conditions:

- authoritative provider/URI and access date recorded;
- terms, derivative use, and redistribution decision recorded;
- every obtained raw file hashed before transformation;
- file inventory reconciled with the original publication;
- acquisition/capture groups and native label semantics resolved or explicitly
  marked unknown;
- a viability result of `complete`, `not_demonstrated`, or `blocked` appended.

R0 does not end merely because a paper is available.

Result:

- primary publication and camera/dataset context were verified;
- S9 and S10 establish a concrete public provider route, and the local
  accession contains 31 MP4s totaling 2,489,878,899 bytes;
- the strict audit fully decoded all 696,065 frames and found all 31 assets
  valid; the report is `not_sequence_evidence`;
- the provider logging sheet supplies per-video distance, capture time, and
  start times for native `Leak 0` and `Leak 1` clips, but no dense plume masks,
  wind, concentration truth, or explicit leak-rate field;
- no explicit license, derivative-use terms, or redistribution terms were
  established, and capture-group/label-duration reconciliation remains open;
- R0 therefore remains `blocked`, now on rights and metadata sufficiency rather
  than data availability.

### R0B — GasVid qualification and raw-pilot contract

Status: `blocked` on owner permission and authoritative assignment/interval
provenance. The bounded R0B-Q engineering substage is complete at `54893a9`;
no R0B engineering substage is active. The separate R0C stage is complete with
a valid negative focusing result.

Objective:

- obtain an authoritative rights and metadata decision without weakening the
  existing preparation gate;
- reconcile filename-to-pad membership, native interval semantics, and the
  paper-defined acquisition split using primary evidence rather than counts or
  filename inference;
- add a dataset-neutral, provenance-bound temporal-window contract so one
  declared interval can be decoded and run without loading or silently mixing
  an entire multi-condition recording;
- if and only if rights and grouping are verified, freeze and run one
  truth-free raw-grayscale development pilot without policy tuning.

Implementation boundaries:

- sequence manifests store resolved inclusive/exclusive frame bounds, source
  timestamps, parent and decoded-window hashes, and selection-metadata lineage;
- preparation specifications store the requested time bounds and expected
  frame indices, then verify them once against the authoritative frame clock;
- schema-v1 through schema-v3 sequences retain their existing whole-asset
  behavior;
- the first windowed track is raw `uint8_identity` input only; same-file
  calibration, pseudo truth, and structure suppression remain later stages;
- the planner receives requested samples only. Full frames may be decoded by
  the source adapter but never enter the planning path.

Current evidence and blockers:

- schema-v4 temporal-window preparation, loading, validation, experiment
  execution, and freeze lineage are implemented at `366730d`; adversarial
  tests bind requested times to the authoritative parent clock and reject
  hash, path-alias, and symlink substitution;
- the primary paper confirms 31 24-minute grayscale videos, eight nominal
  three-minute conditions, the first-15/last-5-second trimming convention,
  pad-2 development and pad-1 test roles, and an initial class-0 segment
  described as non-leak;
- the accession contains 18 `25xx` and 13 `12xx`/`14xx` files, but matching
  those counts to pad identities remains an inference and must not be frozen;
- `MOV_2559` has a technically valid proposed `3:44–6:24` interval, but its
  development role and native label semantics are not yet authoritative;
- no explicit GasVid licence or derivative-use grant was found. The prepared
  inquiry is tracked at
  `data/recorded/accessions/GASVID_RIGHTS_INQUIRY.md` and remains unsent;
- no synchronized wind, pixel source/corridor, or spatial truth is available.

End conditions:

- written terms or correspondence explicitly permit the required local
  processing and derivative construction, with redistribution recorded
  separately;
- all 31 files have provenance-backed capture-group and split assignments;
- temporal-window preparation, loading, validation, and freeze lineage pass
  compatibility, tamper, causality, and truth-isolation tests;
- one outcome-blind development interval runs reproducibly with raw grayscale,
  truth `none`, wind `none`, source-free full-scene startup, and a
  `counterfactual_virtual` scanner;
- JSON and optional video hashes are recorded while generated artifacts remain
  untracked;
- the stage closes `blocked`, rather than remaining indefinitely active, if
  permission or minimally sufficient metadata cannot be established after the
  documented inquiry process.

R0B completion does not complete Stage 4/G3 and does not change the
`not_demonstrated` system claim.

GasVid continuation plan:

1. Do not send the current draft yet. First align its requested per-file
   `paper_group_id`, `capture_group_id`, split, interval, and permission fields
   with the machine qualification gate; define a hashed response record, one
   follow-up, and an explicit close date before sending. User identity and
   authorization are required only after that revision is ready.
2. Separate a label-free diagnostic-pilot decision from an evidence-suite
   decision. The former needs verified rights and authoritative provenance for
   its selected asset/window; the latter still requires complete all-asset
   group/split isolation and resolved semantics. Neither gate may be weakened
   after viewing an outcome.
3. If authoritative evidence makes the relevant gate eligible, run exactly
   the preregistered raw pilot. Otherwise close GasVid as `blocked` at the
   recorded deadline and retain all negative qualification history.

No pilot, evidence-ready suite, or performance result exists at this
checkpoint.

### R0B-Q — Metadata-only qualification contract

Status: `complete` (engineering)

Evidence commit: `54893a9`

Starting branch: `codex/recorded-ogi-sensor-emulation`

Starting commit: `0600338`

Objective:

- turn the current paper/provider audit into a strict, reviewable
  qualification artifact before any GasVid frame interval is extracted;
- distinguish authoritative source facts, provider-row facts, strong
  inferences, contradictions, and unknowns on a per-asset basis;
- represent permission as separately evidenced scopes instead of one broad
  rights boolean; and
- make later preparation depend on the exact accepted qualification manifest,
  without advancing Stage 4, G3, or the system claim.

Implemented and verified boundaries:

- add a typed, canonical metadata-only qualification manifest and generated
  report. The input records source observations and proposed interpretations;
  the report records validation results, unresolved conflicts, and a
  preparation decision;
- give every manifest a stable non-empty `qualification_id` and an explicit
  `asset_root`: an existing, normalized, data-root-relative POSIX directory
  with no absolute path, `..`, or symlink traversal. Each qualified asset path
  is interpreted relative to that declared root rather than guessed from the
  audit or qualification-file location;
- the dataset-neutral permission contract has exactly six independent scopes:
  `local_decode_analysis`, `bounded_interval_extraction`,
  `local_derivative_construction`, `publish_aggregate_metrics`,
  `publish_frames_or_clips`, and `redistribution`. Every scope is `allowed`,
  `unknown`, or `denied`, cites its own evidence, and defaults to `unknown`
  rather than inheriting from public access;
- every audited asset binds its path and byte hash plus a generic
  `assignment_status` (`unresolved`, `inferred`, or `verified`), nullable
  `paper_group_id`, `capture_group_id`, and `split`, and assignment evidence.
  Inferred assignments remain reviewable but cannot satisfy a verified
  preparation prerequisite. GasVid provider-row keys, pad hypotheses, and
  model-subset disposition remain tracker evidence, not mandatory fields in
  the reusable qualification schema;
- interval records preserve the provider's literal `Leak 0`/`Leak 1` event
  timestamps, the nominal paper convention, the unresolved class-0
  contradiction, and the `MOV_2568`/`MOV_2582` anomalies. Qualification must
  not silently manufacture durations or convert event names into plume truth;
- add exact `qualification-template` and `qualification-check` CLI actions.
  The template emits a canonical pending skeleton without decoding media; the
  check verifies canonical qualification bytes, the exact audit identity/hash,
  exact audit asset coverage, and every existing confined asset at the full
  `<data-root>/<asset_root>/<relative_path>` path and declared SHA-256 before
  producing its canonical report;
- introduce preparation-spec v3 only as a binding extension: it names the
  exact qualification manifest/hash and required permission scopes, and
  preparation rejects a missing, changed, noncanonical, ineligible, or
  under-scoped manifest. Selected media must match the full
  `asset_root/relative_path` and SHA-256 identity plus its verified capture
  group and split;
- preparation-spec v3 is only for the `local-raw-pilot` purpose: every
  sequence is a bounded raw `raw_recorded` window with no calibration source,
  no structure-suppressed emulation, and evaluation truth `none`. Spatial
  labels are not an input or prerequisite;
- R0B-Q may read and hash metadata already present in the accession audit and
  provider sheet, but it must not decode new frame intervals, emit extracted
  frames/video, create masks or calibration maps, run the planner, or publish
  a GasVid-derived artifact.

Frozen inputs and forbidden evidence:

- accession audit SHA-256 `246fc11d...a189`, provider-sheet SHA-256
  `1869c151...db55`, and primary-source observations S1, S2, S6, S9, S10,
  and S11 are the starting qualification evidence;
- filename prefixes, five apparent capture sweeps, pad membership, and split
  role are not frozen facts and must remain inferences until authoritative
  evidence resolves them;
- FGSTP paper figures or performance values do not establish that its masks,
  mapping files, code, or permissions are available; and
- no spatial label, planner result, visual quality judgment, or desired split
  outcome may influence qualification.

End conditions:

- a strict schema and canonical parser/writer represent source observations,
  scoped permissions, per-asset evidence states, interval semantics,
  contradictions, and qualification decisions without free-form values
  controlling preparation;
- the template/check CLI round trip is deterministic and test-covered,
  including stale-hash, missing-asset, ambiguous-pad, class-0-conflict,
  interval-anomaly, canonical-byte, path-confinement, full-path/hash, and
  permission-scope failures. Exit status is `0` for an eligible checked
  qualification, `2` for a valid but blocked/ineligible check, and `1` for an
  invalid request/artifact or write failure;
- all 31 audited files are accounted for, while the full accession and S11's
  25-file modeled subset remain explicitly distinct;
- the five candidate capture groups and filename-to-pad mapping are reported
  as strong inference only, represented by `assignment_status=inferred` where
  included in an asset assignment, and not converted to verified split
  assignments;
- preparation-spec v3 binds the exact accepted qualification manifest and
  fails closed when any permission needed by the requested operation is not
  `allowed` by authorizing evidence. It remains raw-window-only, consumes no
  calibration source, and requires no spatial labels;
- no frame-derived artifact is created and no GasVid causal pilot is run;
- exact commands, tests, resulting artifact hashes, and implementation commit
  are appended only after verification; and
- R0B remains blocked unless independent correspondence or formal terms verify
  the necessary permissions and authoritative metadata. Stage 4/G3 stay
  `blocked`, and the claim stays `not_demonstrated`.

Current result:

- the interface and preparation-spec-v3 binding passed the full test, compile,
  and diff checks and were committed at `54893a9`; R0B-Q engineering is
  complete;
- canonical `data/recorded/metadata/gasvid-qualification-v1.json` has SHA-256
  `560f3bad93d3c5b72340ecf30ee174400ef1805b0ea9698c727b143b994c18ed`;
- the canonical qualification check exited `2`, wrote the ignored report at
  `data/recorded/outputs/gasvid-qualification-v1.check.json` with SHA-256
  `f330e64a26cfec5807bcc034190d5cdc8bb4c7a992f25a80ed11223cea519487`,
  and reported `eligible=false` with 36 blockers;
- those blockers are the 31 per-asset `assignment_inferred` results,
  `dataset_audit.rights_blocked`, unresolved native interval semantics, and
  unknown local decode, interval-extraction, and derivative permissions. The
  report's 18 proposed development assets remain inferred, not verified split
  evidence; and
- qualification and checking performed metadata/canonical-byte and file-byte
  hash validation only. No media interval was decoded or extracted, no frame,
  mask, calibration, or other derivative was created, and no causal pilot or
  performance evaluation ran.

### R0C — Licensed raw-OGI engineering smoke — Complete

Status: `complete` (result `engineering_not_demonstrated`)

Overarching objective:

> Establish on one explicitly licensed, independently recorded OGI video that
> the causal recorded-input and counterfactual mechanical-scan loop can consume
> changing raw grayscale observations reproducibly, without labels, cleaning,
> wind, hidden full frames, or policy tuning. This is an engineering
> feasibility result, not methane validation or scan-policy performance
> evidence.

Why this is next:

- C1 GasVid is externally blocked and its current inquiry/checker contract
  needs alignment before correspondence is sent;
- C2 is available under explicit CC BY 4.0 terms and therefore lets the
  existing dataset-neutral pipeline be exercised without weakening rights
  checks;
- the conversation's immediate uncertainty is whether real video behaves
  through the current scanner loop, while labels are unnecessary for that
  question; and
- the current recorded manifest hardcodes `scenario="methane"`, so a propane
  pilot cannot honestly use it until planner mode and gas identity are
  separated; and
- introducing normalization before observing the raw-input failure mode would
  confound input plumbing with a new measurement model.

Preregistered candidate and role:

- use S12 `Ground_26feb2018_propane.mp4` as the first choice: the institutional
  record publishes it as a substantive ground GF320 controlled-propane video,
  size 64,946 kB and MD5 `94f1897d786ff52de4b34a270bd86de7`;
- retain `Ground_28feb2018_propane.mp4` only as a declared fallback if the
  first asset fails download or decode identity before any planner outcome is
  viewed;
- record gas as `propane`, evidence grade as
  `licensed_external_engineering`, and scanner as
  `counterfactual_virtual`; never relabel it methane or hardware evidence; and
- use the full first-choice asset rather than selecting a visually favorable
  interval. Any runtime cap must be fixed from decode/compute constraints
  before rendering or inspecting plume behavior.

Preregistered R0C-v1 execution settings, fixed before media inspection:

- run exactly `max_scans=8` unless the finite source terminates earlier;
- use source-free full-scene startup and mechanical column mode;
- use `scan_rate_hz=20`, `sample_stride_px=2`,
  `column_stride_px=4`, `vertical_pass_seconds=0.05`, and
  `horizontal_step_seconds=0.01`;
- retain the generic `uncalibrated_video_v1("plume")` scan-policy values:
  column/row mass `1.0`, minimum one column/row, and no maximum-column cap;
- label every duration as a fast counterfactual virtual-scanner setting, not
  measured hardware timing; and
- if these settings yield a degenerate result, retain it as the R0C-v1 result.
  Any normalization or scanner-policy comparison is a separately
  preregistered follow-on, never an in-place retune.

Implementation boundaries:

Pre-accession engineering checkpoint:

- manifest schema v5, qualification schema v2, preparation schema v4, and
  report schema v3 are implemented with legacy compatibility;
- qualification evidence snapshots, provider gas identity, exact project
  selection, decode authority, and local-render authority are hash-bound and
  rechecked before the corresponding operation;
- the combined suite passes with 307 tests and 5 Windows capability skips;
  and
- commit `523df93` froze this contract before any selected media was
  downloaded, decoded, viewed, or used for tuning.

1. Before media accession, add a backward-compatible, dataset-neutral contract
   revision:
   - the next recorded-manifest schema separates planner scenario from a typed
     gas-identity/provenance field; C2 resolves to generic `plume` planning and
     `propane` gas identity, while existing schema-v1 through schema-v4 methane
     manifests retain their exact behavior;
   - qualification schema v2 declares an operation purpose such as
     `diagnostic_raw_pilot` or `evidence_suite` and derives permission and
     asset-coverage requirements from that purpose;
   - provider-declared capture-group facts remain authoritative source
     metadata, while a project split is a separate content-hashed
     preregistration decision made before outcome inspection; and
   - correspondence or owner metadata that changes eligibility requires a
     content hash and stable reference. Historical v1 artifacts remain valid
     and append-only.
2. Prove the contract fails if a propane asset would serialize as methane,
   if a project split is presented as provider metadata, if a diagnostic pilot
   silently expands into an evidence suite, or if requested outputs require an
   undeclared permission.
3. Accession only the selected media, S12/S13 provenance, the CC BY 4.0 terms,
   provider MD5, local SHA-256, and decoder/pixel identity. Raw bytes and
   generated artifacts remain ignored.
4. Use the dataset-neutral audit → qualification → prepare → validate
   → freeze path. Add no `usu` branch to the causal runner and no video-specific
   policy constant.
5. Freeze raw `uint8_identity` input, truth `none`, wind `none`, calibration
   `none`, source-free full-scene startup, planner scenario `plume`, gas
   identity `propane`, and `uncalibrated_video_v1("plume")`. Mark the run's
   evidence role as `uncalibrated_engineering_smoke`; do not rename this into a
   calibrated or methane default. No pseudo mask, leak label, source point, or
   full current frame may enter planning.
6. Run the headless causal experiment first; render one local video only from
   the frozen report/run. Record commands, commit, resolved manifest/profile,
   JSON/video hashes, and attribution. Do not tune after viewing the output.
7. Report only operational facts: decoded clock and frame variation, requested
   versus completed scan timing, sampled/swept fractions, causal sample
   availability, ROI/reacquisition sequence, termination reason, and whether
   the belief remained finite and normalized. Plume recall, superiority, and
   methane claims are unavailable without independent truth.

Required tests:

- exact accession/hash and CC-licence qualification;
- backward compatibility for schema-v1 through schema-v4 methane records and
  strict `plume`/`propane` serialization for C2;
- purpose-aware qualification tests that distinguish source metadata from a
  preregistered project split and diagnostic selection from suite completeness;
- finite-source behavior rejects unavailable future samples and never freezes
  the last frame;
- a sample-access sentinel proves planner execution cannot read unrequested
  full-frame pixels or evaluator truth;
- report serialization keeps `propane`, `raw_ogi`, no-wind, no-truth, and
  counterfactual scanner labels intact;
- renderer output cannot alter the scan schedule or headless report; and
- the focused tests, full suite, `compileall`, and `git diff --check` pass.

R0C-v1 result, retained without tuning:

- the official 66,504,835-byte asset matched provider MD5
  `94f1897d786ff52de4b34a270bd86de7` and local SHA-256
  `da3f7702376eeaa6c2d6b37a835b874965d8f49b8b60f56227af8f42bd6b75fd`;
- audit SHA-256 `58b91a710636f483283b6d99190bf4ea3d849cb9abc7ab71f04b1a1e8388737a`
  records 2,067 decoded 240×320 frames, nominal 14.946166324 fps, and
  decoded-content SHA-256
  `4ef5f4d8a5045cdc59acd723b871e575225287abfe302657e5edb7bcf6467a71`;
- qualification SHA-256
  `5c7c09678d7f45025fe00d0bed5f597bdbe32203ae7c27ec0f6be24b2527b2c2`
  is eligible for the preregistered `decode_analysis` and
  `render_local_media` operations; preparation-report SHA-256 is
  `610da2b217ce6bf4c18deeef61b2c28af28d8f20eecfd49ab303881e2c767037`;
- the frozen development suite validates with canonical suite hash
  `c63650e9d89ad6467f8c632bedb84b1e6540315c42c07ceb56a86055fbd74479`
  and freeze-record SHA-256
  `d92403a5589a4901fe7439553adf0ae098a6457fa9ccd90d3c6167df6304c0b6`;
- eight sequential scans completed in 36.2213 modeled seconds and terminated
  honestly at `max_scans_reached`; 96.34% of 124,326 repeated-coordinate
  comparisons changed, with mean absolute change 18.888 raw units;
- all 16 checked posterior/predictive distributions were finite,
  non-negative, and normalized within `2.23e-16`; no truth, source location,
  future frame, or full-frame planner evidence was used;
- headless and rendered operational traces, belief-integrity summaries,
  sample-variation summaries, timelines, termination, and scan counts match
  exactly. Headless report SHA-256 is
  `08f973d09699d72af2526dcebb8ab4d9e08cc1c2ac75feaaa53c988291b3740e`,
  rendered report SHA-256 is
  `a554f9c2ce013d14e0f0c9cbd3558080d550bd040ac61a8f89702edec92beccb`,
  and the local eight-frame video SHA-256 is
  `8e1202a3e85a2fdcb21d6919899ae74244d8a194fbaa5405e990747ba9bdf27c`;
  and
- the adaptive result is nevertheless behaviorally weak: requested area was
  89.54–100% (mean 91.05%), only two full-height requested ROIs occurred, and
  mean uncertainty was 146.63 px. Column stride reduced the modeled swept area
  to 22.32–25% (mean 23.36%), but this is not evidence that the planner found
  or covered the plume. With no spatial truth, recall is unavailable.

The result is therefore `engineering_not_demonstrated`, not `invalid`:
licensed recorded bytes flowed causally through the scanner and reporting
path, but raw OGI display intensity did not produce a useful focused request.
This does not weaken the frozen run or justify tuning its policy in place.

R0C-v1 reproduction commands from repository root:

```powershell
.\.venv\Scripts\plume-focus-recorded.exe validate `
  --freeze-record metadata/usu-uinta-ogi/usu-r0c-v1-development.freeze.json `
  --root data/recorded

.\.venv\Scripts\plume-focus-recorded.exe validate `
  --suite data/recorded/manifests/usu-r0c-v1-development-suite.json

.\.venv\Scripts\plume-focus-recorded.exe run `
  --manifest data/recorded/manifests/usu-ground-26feb2018-propane-raw-r0c-v1.json `
  --max-scans 8 `
  --report data/recorded/outputs/usu-r0c-v1-headless.json

.\.venv\Scripts\plume-focus-recorded.exe run `
  --manifest data/recorded/manifests/usu-ground-26feb2018-propane-raw-r0c-v1.json `
  --max-scans 8 `
  --report data/recorded/outputs/usu-r0c-v1-rendered.json `
  --output-video data/recorded/outputs/usu-r0c-v1-rendered.mp4 `
  --render-scale 2
```

Known tooling gaps retained for the next engineering stage:

- `run` accepts a sequence manifest but cannot atomically validate and execute
  from a freeze record/suite;
- renderer equivalence currently requires an external report comparison; and
- R0C focus diagnostics have no preregistered numeric classifier. Until such a
  classifier exists, report the raw mask, envelope, row/column, and duration
  values and make the stage decision explicitly from its frozen end condition.

R0C end conditions:

- `engineering_feasible`: the licensed asset, manifests, and hashes are
  reproducible; the report says `plume`/`propane` rather than methane; at least
  two sequential scans complete against changing recorded frames; all
  causal/truth-isolation checks pass; the source ends honestly; and a report
  plus optional linked video can be regenerated;
- `engineering_not_demonstrated`: the run is valid and reproducible but the
  existing raw profile yields degenerate belief, repeated full-scene
  reacquisition, or otherwise non-useful scan behavior; retain that negative
  result and use it to justify a later typed normalization experiment; or
- `invalid`: rights, identity, timing, causality, or truth isolation fails. Fix
  only that invalidity and rerun the unchanged preregistered pilot.

R0C does not complete Stage 4/G3 because it is one propane engineering
sequence with no held-out pair, frame-synchronous wind, or spatial truth. It
does not reactivate Stage 5 or change the system claim from
`not_demonstrated`.

Follow-on work, not part of R0C:

- align the GasVid inquiry and qualification schema so diagnostic-pilot and
  evidence-suite purposes are explicit, eligibility-changing correspondence
  is content-hashed, interval contradictions have typed resolutions, and the
  versioned rights audit is regenerated before qualification;
- obtain the user's final signature and send authorization only after that
  alignment; and
- if raw OGI is valid but dominated by background structure or unstable
  pixels, preregister a separate raw-versus-fixed-normalization comparison
  using development or independent leak-off calibration only. Normalization
  must remain a typed measurement transform, never a hidden planner heuristic.

### R0D — OGI measurement separability and normalization contract — Planned

Status: `planned`; not active

Overarching objective:

> Determine on development-only recorded OGI whether static scene structure,
> camera/display drift, and per-pixel instability can be separated from a
> planner-usable gas signal by a fixed, causal, typed measurement model. Keep
> scan policy and mechanical cost unchanged. Without independent spatial
> truth, report measurement and focusing diagnostics only, never plume
> coverage or superiority.

Execution plan:

1. Freeze R0C-v1 as the immutable raw baseline. Add focused report-comparison
   and measurement-diagnostic commands rather than another one-off script. The
   diagnostic measures spatial structure, temporal drift, repeated-coordinate
   noise, saturation, and camera/global brightness change. Full-frame offline
   analysis is permitted only in this development diagnostic and must be
   unavailable to the planner.
2. Define the calibration contract before implementing a transform. Prefer an
   independently recorded leak-off/background capture with matching camera
   mode. If none can be proven, explicitly reject a persistent-gas-erasing
   same-sequence background estimate rather than inventing a clean baseline.
3. Implement at most one selected transform behind a typed measurement-profile
   protocol. It may use fixed calibration plus requested samples and their
   causal history; it may not use the current unscanned frame, evaluation
   truth, frame-global min/max, histogram equalization, or policy-specific
   constants. Serialize units, calibration hashes, and every parameter.
4. First replay the exact frozen R0C sample schedule to isolate the measurement
   change. Then, only if the replay is finite and stable, run the unchanged
   adaptive policy at the exact R0C mechanical settings. Keep raw and
   transformed outputs side by side; do not optimize on ROI contraction alone.
5. Record temporal stability, belief entropy/normalization, uncertainty,
   reacquisition, requested and swept area, and failure cases. ROI contraction
   is diagnostic; no coverage metric is available until independently derived
   truth is attached through a separate evaluator-only contract.

R0D end conditions:

- `complete`: the raw diagnostic and one justified typed transform (or an
  explicit no-transform decision) are reproducible; planner sample-access and
  renderer-independence tests pass; the raw R0C result remains unchanged; and
  all values have units and calibration provenance;
- `not_demonstrated`: no fixed causal transform improves signal stability
  without erasing persistent plume evidence or depending on unscanned pixels;
  retain the result and require a better calibration capture; or
- `blocked`: no independently defensible background/calibration source exists
  and same-sequence estimation would make the claimed sensor abstraction
  circular.

R0D cannot change Stage 4/G3 or the system claim. It is a measurement-contract
stage, not a fair policy benchmark and not real-sensor calibration.

Recommended next bounded Goal Mode objective:

> Complete R0D-A by implementing and running a reproducible development-only
> OGI measurement-separability diagnostic, then make an evidence-backed
> calibration decision. Do not change the scan policy or implement a
> normalization transform until the diagnostic establishes what calibration
> source and causal information that transform may legitimately use.

R0D-A ends when the raw diagnostic command, tests, report schema, command and
artifact hashes, and calibration-source decision are committed. It may close
`complete`, `not_demonstrated`, or `blocked`; only a `complete` decision that
names a defensible calibration source may activate R0D-B. Goal Mode is useful
for this bounded audit-and-decision stage because it must preserve the frozen
R0C baseline across code, evidence artifacts, and documentation, but it should
not be an indefinite roadmap goal.

### R1 — Deterministic derivative construction

Status: `complete` (engineering)

External-corpus result: `blocked` with R0

Objective:

- implement dataset-neutral audit and preparation adapters;
- preserve immutable raw media and complete transformation lineage;
- produce deterministic clock, frozen-calibration, and recorded-sequence
  manifests without changing the causal runner. The primary emulated track is
  an on-demand sampled transform, not a required derived full-frame video.

End conditions:

- the exact hashed audit, preparation spec/profile, decoded content, and
  output paths reproduce identical derived hashes;
- frame order, geometry, timing caveats, and exclusions are verified;
- every exclusion has a non-performance reason;
- raw OGI and emulated-sensor assets remain distinct;
- focused tests pass without importing preparation behavior into
  `plume_focus.video`.

Result:

- all engineering end conditions pass on small generated video fixtures;
- repeated builds are byte-identical and different-content overwrites fail;
- preparation rejects a decoder-signature or decoded-calibration-pixel
  mismatch with the accession audit;
- audit validity rejects clips shorter than the runtime's three-frame minimum;
- every audited byte identity requires a declared capture-group disposition,
  every non-excluded identity is actually used, duplicate bytes require an
  objective exclusion, and every excluded group requires a typed
  non-performance reason;
- freeze-record validation rechecks the preparation/freeze lineage instead of
  trusting a write-only companion record; suite validation separately checks
  decoded media and runtime sidecars;
- the real C1 raw corpus is now locally acquired and audited, but deterministic
  derivative preparation is intentionally not run because the audit is
  rights-blocked; no GasVid-derived evidence claim was created.

### R2 — Silver pseudo-truth construction

Status: `planned`

Objective:

- generate frozen, evaluation-only proposal masks and agreement weights;
- quantify disagreement and conduct a stratified manual spot audit;
- preserve truth isolation from preparation and planning.

End conditions:

- proposals, profiles, parents, versions, and hashes are reproducible;
- every truth frame aligns with its sensor timestamp and coordinates;
- truth substitution invariance proves planned scans are unchanged;
- all silver reports are automatically diagnostic-only;
- poor-agreement strata remain reported rather than excluded.

### R3 — OGI-derived sensor-emulation contract

Status: `planned`

Engineering status: `complete` through `48c7d8d`

Evidence/calibration status: `planned`

The typed runtime and isolation tests are complete, but R3 remains a
data-bearing stage. It cannot complete until its profile is fitted and frozen
from rights-audited development or independent pre-run calibration evidence,
then exercised on an actual recorded sequence without consulting held-out
outcomes.

Objective:

- implement the validated structure-suppressed sensor profile;
- make the emulated field the primary virtual-scanner input;
- keep raw OGI and full-frame-derived tracks explicitly diagnostic.

End conditions:

- background, noise, polarity, rectification, and clipping are typed,
  validated, provenance-backed, and reportable; temporal integration remains
  explicitly rejected unless a stateful scanner-response contract is added;
- the runtime transform uses only fixed calibration assets and mechanically
  sampled current values; any future history must be scanner-state history,
  never unobserved full-frame history;
- renderer, pseudo truth, full current frames, future frames, and held-out
  outcomes cannot affect planning;
- bright/dark and unstable-pixel fixtures behave predictably;
- missing calibration fails or enters an explicitly labeled raw diagnostic
  mode;
- a rights-audited leak-off asset is used to fit the real calibration;
- residual polarity, noise floor, and output scale are selected only from
  declared development or independent pre-run evidence and then frozen;
- at least one real development sequence runs with the resolved calibration
  hash, sample count, profile, and limitations serialized in its report.

### R4 — Frozen silver benchmark

Status: `planned`

Objective:

- compare adaptive and declared non-oracle baselines at exact matched
  counterfactual scan cost;
- report acquisition-level uncertainty and failure strata;
- preserve the existing system thresholds without promoting silver evidence to
  final proof.

End conditions:

- development and held-out suite hashes were frozen before eligible tuning;
- all matched cases are present and cost equality is exact;
- statistics resample acquisition/capture groups, not frames;
- raw, sensor-emulated, and upper-bound tracks remain separately labeled;
- outcome is appended as `not_demonstrated` or diagnostic support, while the
  overall system claim remains unchanged.

### R5 — Gold-evidence upgrade

Status: `planned`

Objective:

- if silver evidence justifies the cost, preregister a modest human-reviewed
  spatial-label subset independent of planner output;
- freeze its annotation protocol, review decisions, suites, and hashes.

End conditions:

- complete independent review and adjudication provenance is present;
- annotators did not select frames or boundaries from planner outcomes;
- sequence-level split isolation remains intact;
- the parent `SYSTEM_GOAL.md` gates are updated only from evidence that actually
  satisfies their requirements.

R5 completion alone does not establish hardware, wind, concentration, or safety
readiness.

## Commands and evidence placeholders

Implemented R0B-Q command surface at `54893a9`:

All qualification path arguments are interpreted relative to `--root` unless
absolute, and absolute paths must still resolve inside that data root. The
declared `--asset-root` is always a normalized relative POSIX directory.

```powershell
plume-focus-recorded-data qualification-template `
  --root <recorded-data-root> `
  --dataset-audit <dataset-audit.json> `
  --qualification-id <stable-qualification-id> `
  --asset-root <normalized-data-root-relative-asset-directory> `
  --output <qualification.json>

plume-focus-recorded-data qualification-check `
  --root <recorded-data-root> `
  --qualification <qualification.json> `
  --report <qualification-check.json>
```

`qualification-template` returns `0` after an idempotent canonical
pending write and `1` for an invalid request or conflicting no-overwrite
target. `qualification-check` returns `0` only when eligible, `2` when the
artifact is valid but blockers remain, and `1` when validation or report
writing fails. Both artifacts remain `not_sequence_evidence`; eligibility
authorizes only the raw local pilot contract and does not change any claim.

Implemented dataset-neutral audit, preparation, and freeze:

```powershell
plume-focus-recorded-data audit `
  --root <external-video-root> `
  --accession-manifest <accession.json> `
  --report <audit-report.json>

plume-focus-recorded-data prepare `
  --root <recorded-data-root> `
  --spec <preparation-spec.json> `
  --profile <preparation-profile.json> `
  --output-manifests <recorded-data-root>/manifests `
  --report <recorded-data-root>/metadata/preparation-report.json

plume-focus-recorded-data freeze `
  --root <recorded-data-root> `
  --preparation-report <preparation-report.json> `
  --preparation-report-sha256 <exact-byte-sha256> `
  --suite-id <suite-id> `
  --split <development-or-held-out> `
  --frozen-at-utc <explicit-utc-time> `
  --pre-change-model-commit <full-40-character-commit> `
  --freeze-provenance <provenance> `
  --output <frozen-suite.json> `
  --freeze-record <freeze-record.json>
```

Implemented recorded validation and execution:

```powershell
plume-focus-recorded validate --manifest <sequence-manifest.json>
plume-focus-recorded validate --suite <frozen-suite.json>
plume-focus-recorded validate `
  --freeze-record <freeze-record.json> `
  --root <recorded-data-root>
plume-focus-recorded validate-pair `
  --development-suite <development-suite.json> `
  --held-out-suite <held-out-suite.json>
plume-focus-recorded run `
  --manifest <sequence-manifest.json> `
  --max-scans <count> `
  --report <report.json> `
  --output-video <optional-video.mp4>
```

| Evidence | Command/commit | SHA-256 or result | Status |
| --- | --- | --- | --- |
| C1 source/access review | S1-S6 plus S9-S11 checked through 2026-07-31 | Concrete provider folder and route established; original and VideoGasNet paper facts separated; no explicit dataset licence or permission located | `complete` for access/fact audit / `blocked` for rights |
| Raw C1 inventory | `plume-focus-recorded-data audit --root data/recorded/assets/external-accession --accession-manifest data/recorded/accessions/accession.json --report data/recorded/assets/dataset-audit.json` | report `246fc11ded363272bce0cec44d7cdfef3ebbab2bd637cdade474b16a1797a189`; 31/31 valid; 696,065 frames; `rights_blocked=true` | `complete` identity/decode / `blocked` rights |
| Native C1 logging sheet | provider Google Sheet exported to ignored `metadata/gasvid_logging.csv` | `1869c151d59d712981ffcf6316ffe3919aeba2296448203ca1f5ceb4ab02db55`; 31 rows; clip timing labels only | `complete` acquisition / `blocked` semantic reconciliation |
| Development suite | `pending` | `pending` | `planned` |
| Held-out suite | `pending` | `pending` | `planned` |
| Sensor-emulation profile | `1466faa`, `484b063`, `ac38940` | Engineering fixtures only; no real calibration | `complete` engineering / `planned` evidence |
| Silver truth profile | `pending` | `pending` | `planned` |
| R4 development report | `pending` | `pending` | `planned` |
| R4 held-out report | `pending` | `pending` | `planned` |
| Deterministic preparation/freeze | `7eb26be`, `a074113`, `b0eefe8`, `c8099af`, `4eb339f`, `067e06a`, `48c7d8d` | Canonical fixture artifacts, verified preparation/freeze lineage, separate runtime suite validation, audit/runtime validity alignment, and explicit claim boundaries; no external data | `complete` engineering |
| Full test suite | `.venv\Scripts\python -m pytest -q` at `48c7d8d` | 208 passed, 1 skipped | `complete` engineering |
| R0B temporal-window contract | `366730d` | Raw truth-free schema-v4 windows bind requested times, resolved parent frames, parent/selection/window hashes, and exact freeze/runtime provenance; no GasVid sequence prepared | `complete` engineering / `blocked` external pilot |
| R0B full test suite | `.venv\Scripts\python -m pytest -q` at `366730d` | 244 passed, 2 capability skips; compile and diff checks passed | `complete` engineering |
| R0B-Q qualification contract | `54893a9` | Typed metadata-only qualification, scoped permissions, per-asset evidence state, interval conflict/anomaly report, template/check CLI, and preparation-spec-v3 binding verified; no frame derivatives | `complete` engineering / external qualification blocked |
| R0B-Q full test suite | `.venv\Scripts\python -m pytest -q` at `54893a9` | 280 passed, 4 Windows symlink-capability skips; `compileall` and `git diff --check` passed | `complete` engineering verification |
| Enriched GasVid qualification | `data/recorded/metadata/gasvid-qualification-v1.json` | `560f3bad93d3c5b72340ecf30ee174400ef1805b0ea9698c727b143b994c18ed`; canonical; 31 assets; rights/permissions and interval semantics remain unresolved | `complete` metadata record / `blocked` pilot eligibility |
| GasVid qualification check | `plume-focus-recorded-data qualification-check --root data/recorded --qualification metadata/gasvid-qualification-v1.json --report outputs/gasvid-qualification-v1.check.json` | exited `2`; report `f330e64a26cfec5807bcc034190d5cdc8bb4c7a992f25a80ed11223cea519487`; `eligible=false`; 36 blockers | valid negative qualification / no pilot |
| R0C pre-accession contract | `523df9307246fb8365710f8ff2e9fd8784fbdd1c` | Schema-v5 manifest, qualification-v2, preparation-v4, report-v3 operational traces, exact operation authorization, provider-backed propane identity, and renderer independence; no selected media inspected | `complete` engineering prerequisite / R0C run closed at `a8a65b0` |
| R0C pre-accession verification | `.venv\Scripts\python -m pytest -q` at `523df93` | 307 passed, 5 Windows capability skips; `compileall` and `git diff --check` passed | `complete` engineering verification |
| R0C licensed accession | `a8a65b0`; `data/recorded/accessions/usu-uinta-ogi-ground-propane-20260731.json`; audit `58b91a710636f483283b6d99190bf4ea3d849cb9abc7ab71f04b1a1e8388737a` | Provider MD5 matched; one 66,504,835-byte asset; 2,067/2,067 frames decoded at 240×320; rights verified CC BY 4.0 | `complete` identity/decode/rights |
| R0C qualification and freeze | Qualification `5c7c09678d7f45025fe00d0bed5f597bdbe32203ae7c27ec0f6be24b2527b2c2`; check `c9c7c6a8f1939d2cb31c8a63527b7f5387458267cec00db88c150293ebab061f`; preparation report `610da2b217ce6bf4c18deeef61b2c28af28d8f20eecfd49ab303881e2c767037`; suite canonical hash `c63650e9d89ad6467f8c632bedb84b1e6540315c42c07ceb56a86055fbd74479`; freeze record `d92403a5589a4901fe7439553adf0ae098a6457fa9ccd90d3c6167df6304c0b6` | Exact development-only `plume`/`propane`, raw/no-wind/no-truth lineage eligible and valid | `complete` engineering lineage / not Stage 4B evidence pair |
| R0C raw execution | Headless `08f973d09699d72af2526dcebb8ab4d9e08cc1c2ac75feaaa53c988291b3740e`; rendered `a554f9c2ce013d14e0f0c9cbd3558080d550bd040ac61a8f89702edec92beccb`; local video `8e1202a3e85a2fdcb21d6919899ae74244d8a194fbaa5405e990747ba9bdf27c` | Eight causal scans and changing samples; renderer trace equal; requested area mean 91.05%; no spatial metrics available | `engineering_not_demonstrated` |

Generated media, masks, calibration arrays, and full reports are evidence
artifacts, not source of truth. Keep them untracked unless a later rights and
size review explicitly approves a small canonical fixture.

## Decision log

| Date | Decision | Rationale | Rejected alternative | Commit/evidence |
| --- | --- | --- | --- | --- |
| 2026-07-26 | Use a dataset-neutral recorded-data workflow with GasVid as Candidate 1 | Prevents dataset-specific assumptions from entering the causal engine | Building a `gasvid` experiment runner | `102f283` |
| 2026-07-26 | Separate raw OGI, structure-suppressed sensor emulation, and pseudo truth | The expected physical reading is cleaner than OGI imagery, while pseudo masks serve a different evaluation role | Treating an enhanced or segmented full frame as the sensor | `1466faa` |
| 2026-07-26 | Make non-negative relative gas evidence the primary emulated-sensor contract | Matches the current planner's likelihood interface without claiming concentration | Per-frame signed/polarity switching | `1466faa` |
| 2026-07-26 | Deny per-frame min/max and histogram normalization | They consume unscanned frame context and destabilize amplitude over time | Cosmetic normalization selected frame by frame | contract decision |
| 2026-07-26 | Treat pseudo masks as silver and evaluation-only | They are useful diagnostics but are neither sensor readings nor independent dense truth | Calling generated masks ground truth | `ac38940` |
| 2026-07-26 | Keep parent Stage 4/G3 blocked while R0 was investigated | No external sequence has been acquired or verified | Advancing system gates from a dataset proposal | tracker evidence |
| 2026-07-26 | Close R0 as blocked | Publications were verified, but authoritative data access, terms, and local asset identity were not | Treating a paper URL as a dataset accession | no external asset |
| 2026-07-26 | Reject temporal smoothing in sensor profile v1 | A batch EMA over spatial samples was scan-order-dependent and lacked a persistent hardware response state | Retaining a superficially causal but physically ambiguous filter | `484b063` |
| 2026-07-26 | Require exact audit, calibration, truth-protocol, and split lineage | Every behavior-changing external input must be confined, hashed, typed, and covered by rights/split checks | Trusting partial lookalike JSON or free-text evidence grades | `ac38940` |
| 2026-07-26 | Permit only counterfactual virtual scanner evidence in recorded-video v2 | The current adapter samples full recorded media through a mechanical model and has no measured hardware scan stream | Allowing a manifest to self-label modeled timing as measured hardware | `ac38940` |
| 2026-07-26 | Make schema v3 truth-free during preparation | Calibration and causal ingestion must not depend on pseudo labels or future annotation work | Requiring a fake truth sidecar to run a new recording | `7eb26be` |
| 2026-07-26 | Freeze canonical calibration values instead of refitting compressed leak-off video at load time | Exact derived bytes, profile, source, and producing commit must be reproducible | Decoder-dependent runtime calibration | `afacbdc`, `7eb26be` |
| 2026-07-26 | Keep sensor emulation sampled and on demand | The cleaner expected sensor interface should affect mechanically requested evidence without becoming hidden full-frame planner access | Precomputing a transformed video as the primary planner input | `a074113` |
| 2026-07-26 | Require objective disposition of every audited byte identity | Silent omission can bias a prepared corpus even when included files are hashed correctly | Ignoring unassigned audit assets | `a074113` |
| 2026-07-26 | Bind compressed-video use to decoded pixel hashes | OpenCV metadata alone cannot prove identical decoded sensor values | Assuming the same version string guarantees identical pixels | `4eb339f` |
| 2026-07-26 | Validate companion freeze records as preparation/freeze lineage | Acceptance review must recheck audit, source, calibration, prepared manifests, and suite membership, then run suite validation separately | Treating a write-only record as verification | `4eb339f` |
| 2026-07-26 | Require audit-valid clips to satisfy the runtime frame minimum | A valid audit must not approve input that the loader immediately rejects | Calling every nonempty decode runnable | `067e06a` |
| 2026-07-26 | Keep suite publication distinct from freeze-record eligibility | The companion record is non-evidence, while the suite freezes references without carrying a claim field | Printing one ambiguous claim label for both artifacts | `48c7d8d` |
| 2026-07-26 | Reopen R0 on a concrete public GasVid provider route while retaining the rights block | Public access resolves availability, not license, derivative permission, spatial truth, or split independence | Continuing to call the corpus absent, or treating downloadability as permission | accession manifest plus audit `246fc11d...a189` |
| 2026-07-31 | Keep window selection provenance-bound and the first pilot raw/truth-free | The recorded runner needs bounded inputs without importing guessed split roles, pseudo labels, same-file calibration, or hidden full frames | Dataset-specific slicing in the runner or using native clip labels as spatial truth | `366730d` |
| 2026-07-31 | Insert metadata-only qualification before any GasVid derivation | The rights audit found public access but no dataset licence or permission, while filename/pad and interval meanings still contain inference and contradiction | Letting public downloadability, filename counts, or paper-wide prose authorize preparation | S1/S2/S6/S9/S10/S11 audit; accession/report hashes above |
| 2026-07-31 | Keep the 31-file accession distinct from the 25-file VideoGasNet modeled subset | S11's six exclusions are modeling choices (five 18.6 m assets and the single 8.8 m asset), not local audit failures or permission decisions | Silently dropping valid accession files to reproduce one later model |
| 2026-07-31 | Treat FGSTP labels as a separate candidate evidence source, not a prerequisite for the raw pilot | S6 reports manual masks for a selected 14-video experiment, but their files, code, mapping, and rights are not established; causal execution needs grayscale samples, not spatial truth | Blocking raw causal execution on labels or claiming unavailable labels as ground truth |
| 2026-07-31 | Treat qualification-check exit `2` as a valid blocked result | The canonical artifact passed structural, canonical-byte, audit, path, and file-hash checks, while 36 declared evidence blockers correctly prevented eligibility | Weakening the checker, promoting inferred assignments, or running the pilot despite `eligible=false` | qualification `560f3bad...c18ed`; check `f330e64a...19487` |
| 2026-07-31 | Close R0B-Q engineering and leave R0B externally blocked | Commit `54893a9` completes the qualification machinery, while owner permission and authoritative assignment/interval evidence remain outside the codebase | Keeping an engineering stage active during an external wait or calling a valid blocked check pilot readiness | `54893a9` |
| 2026-07-31 | Use the owner inquiry as the next decision point | A recorded response either permits an eligible raw truth-free no-wind pilot or supports closing GasVid blocked and pivoting under the same contract | Running before eligibility, weakening blockers, or waiting indefinitely without a recorded disposition | `data/recorded/accessions/GASVID_RIGHTS_INQUIRY.md` |
| 2026-07-31 | Supersede immediate sending with inquiry/checker alignment | The draft does not yet request every machine-required per-file group field and has no hashed follow-up/closure protocol | Sending an inquiry whose answer cannot deterministically clear the gate | R0C planning audit |
| 2026-07-31 | Select C2 for R0C raw recorded-OGI engineering smoke after a species-neutral contract revision | USU publishes substantive ground GF320 OGI video under CC BY 4.0; a typed `plume`/`propane` record can answer the label-free raw-input plumbing question without using the current methane-only manifest dishonestly | Waiting idle, adding a USU runner branch, treating downloadable but unlicensed methane footage as reusable, or relabeling propane as methane | S12/S13 primary-source and schema audit |
| 2026-07-31 | Commit the R0C contract before accession | A pre-media commit makes the no-tuning boundary independently auditable and binds every permitted decode/render action to exact evidence | Viewing the selected recording before freezing policy, profile, lineage, and report semantics | `523df93` |
| 2026-07-31 | Close R0C with a valid negative focusing result | Recorded bytes changed causally and beliefs remained normalized, but the raw-profile requested ROI stayed near full-frame; column stride is not evidence of plume coverage | Calling the run `engineering_feasible` merely because plumbing worked, or tuning the frozen settings after viewing | R0C report hashes above |
| 2026-07-31 | Make R0D a development-only measurement separability contract | Expected hardware readout is cleaner than OGI display video, so static structure and pixel instability must be diagnosed before another policy benchmark | Changing adaptive-policy constants, using frame-global cosmetic normalization, or making a coverage claim without truth | R0C raw negative plus conversation sensor premise |

## Evidence log

| Date | Stage | Commit/artifact/command | Result |
| --- | --- | --- | --- |
| 2026-07-26 | R0 | Starting commit `17c7f8c` | External data absent; accession and rights not yet verified; no evidence-grade change |
| 2026-07-26 | Engineering | `102f283` | Persistent recorded-data workflow and claim boundaries committed |
| 2026-07-26 | Engineering | `1466faa`, `484b063` | Fixed median/MAD sensor emulation added; ambiguous temporal smoothing removed |
| 2026-07-26 | Engineering | `f0748ac` | Dataset-neutral accession audit and CLI added |
| 2026-07-26 | Engineering | `ac38940` | Schema-v2 production path, lossless truth, derived claim grades, and paired-suite checks added |
| 2026-07-26 | Engineering review | `.venv\Scripts\python -m pytest -q` | 151 passed, 1 skipped; no external evidence run |
| 2026-07-26 | R0 | No authoritative asset/rights record | Result `blocked`; parent Stage 4/G3 and claim remain unchanged |
| 2026-07-26 | R1 engineering | `afacbdc`, `7eb26be`, `a074113`, `b0eefe8`, `c8099af`, `4eb339f`, `067e06a`, `48c7d8d` | Canonical calibration, schema v3, deterministic prepare/freeze, decoded-pixel binding, preparation/freeze validation plus separate suite validation, audit/runtime validity alignment, explicit claim boundaries, CLI, and role isolation implemented; no external evidence |
| 2026-07-26 | Engineering review | `.venv\Scripts\python -m pytest -q` | 208 passed, 1 skipped; audit → prepare → validate → freeze → freeze-record validation fixture passed |
| 2026-07-26 | R0 accession | `data/recorded/accessions/accession.json`; local ignored provider assets | 31 MP4s plus 31-row logging CSV acquired; 2,489,878,899 video bytes |
| 2026-07-26 | R0 audit | report SHA-256 `246fc11ded363272bce0cec44d7cdfef3ebbab2bd637cdade474b16a1797a189` | 31/31 videos valid; 696,065 decoded frames; all 240×320; `rights_blocked=true`; claim eligibility `not_sequence_evidence` |
| 2026-07-31 | R0B engineering | `366730d` | Schema-v4 raw media-window contract, parent-clock/index reconciliation, exact decoded-window and selection lineage, deterministic prepare/freeze verification, two-scan causal execution test, and rights-inquiry draft implemented; no external pilot |
| 2026-07-31 | R0B verification | `.venv\Scripts\python -m pytest -q` | 244 passed, 2 capability skips; `compileall` and `git diff --check` passed; rights, pad/split provenance, and pilot remain unresolved |
| 2026-07-31 | R0B-Q primary-source audit | S1/S2/S6/S9/S10/S11, checked through 2026-07-31 | No explicit GasVid dataset licence, local-processing permission, derivative permission, or redistribution grant found; `rights_status` remains `unverified` |
| 2026-07-31 | R0B-Q metadata reconciliation | provider sheet `1869c151...db55`; audit `246fc11d...a189` | Full 31-file accession confirmed; five apparent capture sweeps and 18/13 filename-to-pad mapping are strong inference only; class-0 meaning conflicts with the nonzero table entry; `MOV_2568` and `MOV_2582` have 157 s and 360 s Leak-0-to-Leak-1 gaps |
| 2026-07-31 | R0B-Q label-source review | S6 https://arxiv.org/abs/2505.00295 | Paper reports frame-by-frame labels on a 14-video selection (9 train, 5 test); mask/code accession, exact file/frame mapping, and reuse rights remain unverified; labels are unnecessary for a truth-free raw causal pilot |
| 2026-07-31 | R0B-Q engineering closeout | `54893a9`; `.venv\Scripts\python -m pytest -q`; `.venv\Scripts\python -m compileall -q src tests`; `git diff --check` | Qualification gate committed; 280 passed, 4 Windows symlink-capability skips; compile and diff checks passed; parent R0B remains blocked and no pilot ran |
| 2026-07-31 | R0B-Q canonical qualification | `data/recorded/metadata/gasvid-qualification-v1.json` | SHA-256 `560f3bad93d3c5b72340ecf30ee174400ef1805b0ea9698c727b143b994c18ed`; enriched metadata only; no decode, derivative, label, calibration, or pilot |
| 2026-07-31 | R0B-Q qualification check | `plume-focus-recorded-data qualification-check --root data/recorded --qualification metadata/gasvid-qualification-v1.json --report outputs/gasvid-qualification-v1.check.json` | exited `2`; report SHA-256 `f330e64a26cfec5807bcc034190d5cdc8bb4c7a992f25a80ed11223cea519487`; `eligible=false`; 36 blockers; no pilot run |
| 2026-07-31 | Recorded fallback source/rights audit | S12-S16 | No reviewed source combines derivative-friendly rights, methane-confirmed temporal OGI, and Stage 4B-ready grouping/truth; C2 is rights-clean for engineering only, C3/C5 remain rights-blocked, and C4 is image-only |
| 2026-07-31 | R0C pre-accession contract | `523df9307246fb8365710f8ff2e9fd8784fbdd1c`; `.venv\Scripts\python -m pytest -q`; `.venv\Scripts\python -m compileall -q src tests`; `git diff --check` | Contract prerequisite frozen before selected-media access; 307 passed, 5 Windows capability skips; R0C accession and run remain pending |
| 2026-07-31 | R0C licensed accession/freeze | `a8a65b0`; asset/audit/qualification/preparation/suite/freeze hashes in the evidence table | Exact provider MD5 and local SHA-256 matched; qualification eligible; preparation, sequence, freeze record, and development suite all validated |
| 2026-07-31 | R0C raw headless/rendered run | `a8a65b0`; `.venv\Scripts\plume-focus-recorded run --manifest data/recorded/manifests/usu-ground-26feb2018-propane-raw-r0c-v1.json --max-scans 8 ...`; report/video hashes above | Operational plumbing passed and rendered trace matched; near-full-frame requested mask retained as `engineering_not_demonstrated`; no claim-status change |

## Update protocol

At stage start:

1. verify that no other stage is `active`;
2. record the branch, starting commit, frozen inputs, and forbidden evidence;
3. copy the objective and end conditions into the bounded execution goal;
4. append the status change and rationale to the decision log.

At stage closeout:

1. run the stated integrity tests and reproduction commands;
2. record exact commands, commits, manifest hashes, asset hashes, and reports;
3. append negative, invalid, and excluded results without deletion;
4. evaluate every end condition from current evidence;
5. update `SYSTEM_GOAL.md` only when its independent gate is actually met;
6. activate the next stage only when work begins.

Frozen artifacts and historical log rows are append-only. Corrections create a
new version and a superseding row; they never rewrite the old record. Allowed
stage statuses are `planned`, `active`, `complete`, `not_demonstrated`, and
`blocked`. Claim status remains separately `supported`,
`not_demonstrated`, or `invalid`.
