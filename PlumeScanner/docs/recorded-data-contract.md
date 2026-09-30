# Recorded Data Contract

This contract describes provenance-checked inputs for
`plume-focus-recorded`. Sequence-manifest schema v3 remains the compatibility
default for whole recorded assets. Explicit schema v4 adds provenance-bound
temporal windows for raw, truth-free, no-wind runs. Explicit schema v5 keeps
the planner scenario separate from a typed gas identity, requires an eligible
hash-bound qualification-v2 lineage, and permits either a whole audited asset
or the same provenance-bound window form. V3 provides
truth-free causal execution and canonical frozen calibration lineage on top of
the classified dataset, sensor, truth, and scanner boundaries introduced in
v2. Schemas v1-v4 retain their prior field sets and semantics: v1 derives
`legacy_unclassified_evidence`, v2 retains its legacy runtime calibration path,
and none of v1-v4 accepts the v5 `gas_identity` field.

Dataset qualification schema v2 and preparation-spec schema v4 are separate,
explicit opt-ins. Qualification v2 separates provider facts from an
outcome-blind project preregistration; preparation v4 binds the exact eligible
qualification, operation plan, and selected assets before decode. Qualification
schema v1 and preparation schemas v1-v3 retain their existing behavior.
Experiment report schema v3 preserves the existing metrics while adding an
observer-independent operational trace, belief-integrity checks, sampled-value
variation, and recorded-media attribution.

The contract proves file identity and enforces causal access. It does not prove
that a dataset is independent, that pseudo labels are correct, or that virtual
scan timing matches hardware.

The R0B-Q interface below is implemented and verified at `54893a9`; its
engineering substage is `complete`. The verified interface must not be
confused with an eligible GasVid pilot. Parent R0B remains `blocked` on owner
permission and authoritative assignment/interval evidence. Preparation-spec
v1/v2 remains legacy-compatible, and no GasVid asset may be prepared while its
rights and qualification blockers remain unresolved.

## Current asset status

The tracked repository does not contain GasVid videos, GasVid-derived masks,
or a leak-off calibration clip. A workspace may contain ignored, locally
audited external assets, but they remain unusable for preparation while their
rights audit is blocked. Users must obtain external assets under applicable
terms and record their source, access, rights, and hash evidence. The candidate
ledger and primary-source URLs are maintained in
[`RECORDED_DATA_WORK_PLAN.md`](../RECORDED_DATA_WORK_PLAN.md); a publication
URL is not treated as a dataset accession or rights grant.

As of the primary-source recheck on 2026-07-31, no explicit GasVid dataset
licence, local decode/analysis permission, interval-extraction permission,
derivative-use permission, publication permission, or redistribution grant was
found in the original paper, VideoGasNet publisher/Stanford records, public
provider folder, or linked download route. The accession therefore remains
`rights_status=unverified`; silence and public downloadability do not change
that state.

The ignored local accession contains all 31 validly decoded recordings. This
is not the same scope as the later VideoGasNet modeled subset of 25, which
excluded five 18.6 m recordings and the single 8.8 m recording. Provider rows
form five strong candidate capture sweeps and filename counts match the
paper's pad counts, but neither the five groups nor filename-to-pad membership
is authoritative. The paper's description of class 0 as non-leak/background
also conflicts with its nonzero class-0 table value. Provider markers for
`MOV_2568` and `MOV_2582` are 157 and 360 seconds apart rather than the nominal
180 seconds. These observations must remain explicit conflicts/inferences,
not repaired metadata.

FGSTP (https://arxiv.org/abs/2505.00295) reports manual frame labels for a
selected 14-video GasVid experiment (9 training and 5 testing videos), but an
authoritative mask/code accession, exact filename/frame mapping, licence, and
reuse permission have not been established. Those labels are not required to
run a truth-free raw causal pilot; they are a possible later spatial-evaluation
source only after a separate accession and rights audit.

## Dataset accession audit

Audit an external accession before constructing sequence manifests. The audit
hashes and fully decodes every supported video below `--root`; it does not
create labels, choose splits, transform media, or create claim evidence.

The strict accession declaration has this shape:

```json
{
  "dataset_accession_schema_version": 1,
  "dataset_id": "<DATASET_ID>",
  "accession_id": "<STABLE_ACCESSION_ID>",
  "source_uri": "<AUTHORITATIVE_SOURCE_URI>",
  "accessed_at_utc": "<ISO_8601_UTC_TIMESTAMP>",
  "rights_status": "unverified",
  "rights_provenance": "<HOW_RIGHTS_STATUS_WAS_ESTABLISHED>",
  "rights_uri": null
}
```

`rights_status` is `verified`, `unverified`, or `restricted`. A verified
accession requires a non-null `rights_uri`. The field must remain present and
null for an accession without a rights URI.

Run the audit with the accession manifest explicitly supplied:

```powershell
plume-focus-recorded-data audit `
  --root data/recorded/assets/external-accession `
  --accession-manifest data/recorded/accessions/accession.json `
  --report data/recorded/assets/dataset-audit.json
```

The report records the accession-manifest hash, per-file byte hash, complete
decode result, shape, frame count, nominal clock information, codec metadata,
OpenCV version/build signature, exact ordered decoded grayscale-frame hash, and
rights state. Sequence loading parses this exact audit schema strictly; partial
hand-written lookalikes are rejected. Its claim label is always
`not_sequence_evidence`. The command returns a nonzero status if any candidate
video fails the audit, including a clip shorter than the runtime's minimum
three frames.

## Metadata-only qualification contracts

### Qualification schema v1 (`local-raw-pilot`; R0B-Q complete)

Qualification is mandatory after byte/decode audit for the
preparation-spec-v3 `local-raw-pilot` workflow. It is not retroactively
required by the legacy-compatible v1/v2 preparation schemas. Qualification
reads canonical audit and metadata artifacts only; it must not decode a new
interval, write frames or video, construct calibration maps or masks, run the
planner, or produce a performance artifact.

The canonical qualification manifest/report binds:

- a stable non-empty `qualification_id` and explicit `asset_root`;
- `asset_root` as an existing normalized data-root-relative POSIX directory.
  Absolute paths, `..`, path aliases, and symlink traversal are rejected, and
  every qualified `relative_path` is interpreted below this root;
- the exact dataset/accession identity and audit byte hash;
- primary or owner evidence records with stable reference, observation time,
  optional content hash, evidence kind, and concise supported fact;
- exactly six independent permission decisions:
  `local_decode_analysis`, `bounded_interval_extraction`,
  `local_derivative_construction`, `publish_aggregate_metrics`,
  `publish_frames_or_clips`, and `redistribution`. Each scope is `allowed`,
  `unknown`, or `denied` and cites its own authorizing evidence; public access
  cannot authorize another scope;
- every audited path and byte hash, plus a generic `assignment_status`
  (`unresolved`, `inferred`, or `verified`), nullable `paper_group_id`,
  `capture_group_id`, and `split`, and assignment evidence. An inferred
  assignment may be recorded for review but cannot satisfy a verified
  preparation prerequisite. Dataset-specific pad names, provider-row keys,
  and model-subset disposition belong in evidence/tracker records rather than
  mandatory schema fields;
- literal provider interval markers, the nominal paper convention, class-0
  conflict state, trimming convention, and per-asset exceptions including
  `MOV_2568` and `MOV_2582`; and
- a deterministic blocker list and operation-specific eligibility decision,
  always labeled `not_sequence_evidence`.

The command surface implemented at `54893a9` is:

All path arguments in these two commands are data-root-relative unless an
absolute path is supplied, and every absolute input/output path must still
resolve inside `--root`. `--asset-root` is never absolute.

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

The template materializes exact audit coverage with all permissions and
assignment/interval metadata at fail-closed values. It does not decode media.
The check requires canonical qualification JSON, revalidates the exact audit
identity and hash, requires exact path/hash coverage of the audit, and verifies
every actual asset at
`<data-root>/<asset_root>/<qualified-relative-path>` with path confinement,
no symlink traversal, and full SHA-256 matching. It then applies permission,
authoritative-evidence, interval, assignment, and group/split blockers and
writes a canonical no-overwrite report containing `qualification_id`,
`asset_root`, qualification path/hash, audit reference, counts, eligibility,
and blockers.

`qualification-template` exits `0` on an idempotent successful write and `1`
for an invalid request or conflicting target. `qualification-check` exits `0`
only for an eligible qualification, `2` for a structurally valid but blocked
or ineligible qualification, and `1` for invalid input or report-write failure.
Neither a zero exit nor the check report is sequence or performance evidence.

### Qualification schema v2 (purpose- and operation-aware)

Qualification schema v2 is an explicit opt-in. It does not change schema-v1
payloads, the default schema-v1 template, the fixed v1 `local-raw-pilot`
purpose, or the v1 checker bytes and blockers. V2 adds a separate canonical,
content-hashed `recorded_project_preregistration` artifact. This artifact is
created before outcome inspection and binds:

- one purpose: `diagnostic_raw_pilot` or `evidence_suite`;
- an exact dataset-audit path and SHA-256;
- the operations the project intends to perform; and
- exact project asset assignments by relative path and byte SHA-256.

`diagnostic_raw_pilot` may select a subset of the audit, but every listed asset
must be assigned `development`; unselected audit assets are omitted rather than
called held out or excluded. `evidence_suite` must exactly partition every
audited identity into non-empty `development` and `held-out` sets, with optional
`excluded` assignments carrying non-performance exclusion reasons. A project
split is an outcome-blind project decision, not a fact asserted by the dataset
provider.

Create the separate preregistration before creating the v2 qualification:

```powershell
plume-focus-recorded-data project-preregistration-template `
  --root <recorded-data-root> `
  --dataset-audit <dataset-audit.json> `
  --preregistration-id <stable-preregistration-id> `
  --purpose diagnostic_raw_pilot `
  --created-at-utc <ISO_8601_UTC_TIMESTAMP> `
  --operation decode_analysis `
  --development-asset <AUDIT_RELATIVE_PATH> `
  --output <project-preregistration.json>

plume-focus-recorded-data qualification-template `
  --root <recorded-data-root> `
  --dataset-audit <dataset-audit.json> `
  --qualification-id <stable-qualification-id> `
  --asset-root <normalized-data-root-relative-asset-directory> `
  --schema-version 2 `
  --purpose diagnostic_raw_pilot `
  --project-preregistration <project-preregistration.json> `
  --output <qualification-v2.json>
```

Repeat `--operation`, `--development-asset`, and, for an evidence suite,
`--held-out-asset` as needed. An excluded evidence-suite asset uses
`--excluded-asset RELATIVE_PATH=NON_PERFORMANCE_REASON`. All paths are confined
to `--root`, artifacts are canonical/no-overwrite, and neither command decodes
the media. `qualification-check` is the same command surface for v1 and v2 and
returns the same exit classes: `0` eligible, `2` valid but blocked, and `1`
invalid.

If a local proof video is part of the preregistered diagnostic, add
`--operation render_local_media` when creating the preregistration. Do not add
that operation after inspecting run outcomes merely to enable a favorable
render.

The v2 qualification template copies the exact audit/preregistration
references, creates one unresolved provider-fact row per audited asset, and
leaves permissions and native interval semantics fail-closed. It is a starting
artifact to enrich with authoritative evidence, not an eligibility shortcut.

V2 derives required permission scopes from typed operations; the project does
not supply an editable list of permission requirements:

| Preregistered operation | Required permission scope |
| --- | --- |
| `decode_analysis` | `local_decode_analysis` |
| `extract_bounded_window` | `bounded_interval_extraction` |
| `interpret_provider_intervals` | `bounded_interval_extraction` |
| `construct_local_derivative` | `local_derivative_construction` |
| `render_local_media` | `local_derivative_construction` |
| `publish_metrics` | `publish_aggregate_metrics` |
| `publish_media` | `publish_frames_or_clips` |
| `redistribute` | `redistribution` |

Every preregistration includes `decode_analysis`. Permission decisions remain
independent and fail closed. In v2, every non-unknown permission decision must
cite content-hashed formal terms or dataset-owner correspondence. Publication
or redistribution permission never authorizes local decoding, extraction, or
derivative construction.

For every v2 evidence record with a `content_sha256`, `reference` is a
normalized, confined path to a local evidence snapshot below the recorded-data
root. The checker rejects missing files, symlink/path escapes, and SHA-256
mismatches. A URL may be recorded inside that frozen snapshot; a mutable URL by
itself is not the bytes being authorized.

The v2 qualification itself exactly covers the accession audit with
`provider_assets`. Each provider fact contains the byte identity, optional
paper and capture-group IDs, optional typed gas identity, a fact status, and
evidence references; it does not contain a project split. A selected asset must
have verified provider facts, a capture-group ID, and a gas identity backed by
content-hashed dataset-owner metadata or correspondence. Unselected facts in a
diagnostic subset may remain unresolved. For an evidence suite, active paper or
capture groups may not cross project splits. Native interval semantics block
only when `interpret_provider_intervals` was preregistered; selecting an
independently declared bounded window does not silently opt into provider
interval labels.

The v2 checker revalidates the canonical qualification, complete provider-fact
coverage, actual asset bytes, every declared evidence-snapshot hash, exact audit
and preregistration hashes, purpose, operation-derived permission scopes,
selected counts, and purpose-specific blockers. Its report remains
`not_sequence_evidence`. Eligibility establishes that the declared operations
and selection satisfy this metadata contract; it does not establish label
quality, data independence, planner performance, or hardware validity.

Preparation-spec v3 is reserved for the binding extension. It must reference
the exact accepted qualification manifest and SHA-256 for the fixed
`local-raw-pilot` purpose. Eligibility requires authorizing evidence for
`local_decode_analysis`, `bounded_interval_extraction`, and
`local_derivative_construction`; publication and redistribution decisions are
reported independently and do not authorize or block a private raw pilot.
Preparation must reject a missing, changed, noncanonical, ineligible, or
under-scoped qualification, as well as selected media whose complete
`asset_root/relative_path` plus SHA-256 identity, verified capture group, or
split differs from the preparation request.

Preparation-spec v3 is raw-window-only: every sequence has a media-window
request, uses `raw_recorded` input, has no calibration source or
structure-suppressed emulation, and carries evaluation truth `none`. Spatial
labels are neither consumed nor required for causal execution. V1/v2
preparation must remain reproducible, but those compatibility paths must not
be used to bypass the documented GasVid R0B-Q process.

The engineering verification completed with 280 passing tests and 4 Windows
symlink-capability skips; `compileall` and `git diff --check` also passed. The
canonical enriched GasVid qualification is
`metadata/gasvid-qualification-v1.json` relative to `data/recorded`, SHA-256
`560f3bad93d3c5b72340ecf30ee174400ef1805b0ea9698c727b143b994c18ed`.
Running:

```powershell
plume-focus-recorded-data qualification-check `
  --root data/recorded `
  --qualification metadata/gasvid-qualification-v1.json `
  --report outputs/gasvid-qualification-v1.check.json
```

exited `2` and produced an ignored report with SHA-256
`f330e64a26cfec5807bcc034190d5cdc8bb4c7a992f25a80ed11223cea519487`,
`eligible=false`, and 36 blockers: 31 inferred asset assignments, blocked
accession rights, unresolved interval semantics, and three unknown permissions
needed by the local raw pilot. This was a valid negative qualification, not a
failed checker. Qualification/checking performed canonical metadata and file
hash validation only: no interval decode or extraction, frame/mask/calibration
derivative, causal pilot, or performance run occurred.

R0B-Q engineering is complete at `54893a9`. The negative GasVid qualification
does not establish permission, run the raw pilot, complete Stage 4/G3, or
change the `not_demonstrated` claim. No engineering substage is active while
R0B awaits external evidence.

Do not send the current inquiry draft yet: it must first request the exact
per-file group/split and interval fields consumed by the checker and define a
content-hashed response, one follow-up, and an explicit close date. A later
qualification revision must distinguish a selected-asset label-free diagnostic
pilot from a complete evidence-suite decision without weakening either after
outcomes are seen.

R0C exercised this dataset-neutral contract on one explicitly CC-BY-4.0 Utah
State controlled-propane OGI video using sequence schema v5, qualification v2,
preparation v4, and report v3. The byte/decode, qualification, preparation,
freeze, causal headless run, and rendered replay all passed; the headless and
rendered operational traces are identical. The raw requested-mask area
nevertheless remained 89.54–100%, with a full-height, nearly full-width
bounding ROI, so the stage result is
`engineering_not_demonstrated`. This verifies recorded-input plumbing only; it
is not a GasVid result, methane evidence, a policy-performance result, or an
evidence-ready suite. Exact commands, hashes, and the R0D follow-on plan are in
`RECORDED_DATA_WORK_PLAN.md`.

## Required layout

Sequence manifests must be under a directory named `manifests`. Asset paths
are resolved relative to its parent:

```text
data/recorded/
|-- accessions/
|   `-- accession.json
|-- metadata/
|   |-- evidence/
|   |   |-- terms-snapshot.html
|   |   `-- provider-metadata-snapshot.json
|   |-- project-preregistration.json
|   |-- dataset-qualification.json
|   |-- dataset-qualification-check.json
|   |-- preparation-spec.json
|   |-- preparation-profile.json
|   |-- preparation-report.json
|   `-- held-out-suite-v1.freeze.json
|-- assets/
|   |-- dataset-audit.json
|   |-- external-accession/
|   |   |-- sensor.mp4
|   |   `-- leak-off.mp4
|   |-- prepared/
|   |   `-- <preparation-id>/
|   |       |-- calibration-<hash>.plmcal
|   |       `-- <sequence-id>-calibration.json
|   |-- truth.npz
|   |-- truth-protocol.json
|   |-- timestamps.csv
|   `-- wind.csv
`-- manifests/
    |-- development-001.json
    `-- development-suite-v1.json
```

This is a layout example, not a declaration that these assets exist. Absolute
paths, symbolic-link escapes, and `..` traversal are rejected. Asset SHA-256
values are hashes of file bytes.

## Deterministic preparation and suite freezing

Preparation inputs are strict, hashed JSON artifacts under the recorded-data
root:

- the preparation spec binds the exact dataset audit, full producing commit,
  capture-group roles, objective exclusion records, truth-free schema-v3 or
  schema-v5 templates, and any separate calibration source. Preparation-spec
  v2 requires a preregistered media-window request that resolves a schema-v3
  template to a schema-v4 output. The R0B-Q preparation-spec-v3 extension
  requires those window requests plus the exact qualification-v1 path/hash and
  permits only the raw, uncalibrated `local-raw-pilot` described above.
  Preparation-spec v4 instead requires an exact qualification-v2 path/hash and
  permits each sequence to select either a whole asset (`media_window_request`:
  `null`) or a declared window;
- the preparation profile selects either `raw_recorded` or
  `structure_suppressed_emulation` and serializes every behavior-changing
  emulation value;
- every audited byte identity must belong to exactly one declared capture
  group. Unused identities must be placed in an `excluded` group with a typed
  non-performance reason; duplicate paths with identical bytes require the
  `duplicate_byte_asset` exclusion.

Run preparation with:

```powershell
plume-focus-recorded-data prepare `
  --root data/recorded `
  --spec data/recorded/metadata/preparation-spec.json `
  --profile data/recorded/metadata/preparation-profile.json `
  --output-manifests data/recorded/manifests `
  --report data/recorded/metadata/preparation-report.json
```

Writers are canonical and no-overwrite: rerunning identical inputs is
idempotent, while a different artifact at the same path is rejected. Emulated
preparation decodes only the declared leak-off calibration source, writes a
platform-neutral fixed-calibration artifact, and records its source hash,
capture group, scope, fit algorithm, profile hash, shape, sample count,
producing commit, and output hash. It does not precompute a transformed planner
video. Dataset-audit schema v2 records the OpenCV version/build signature and
the exact ordered grayscale decoded-frame hash for every asset. Preparation
requires the same decoder signature and verifies the calibration pixel stream;
runtime sequence loading verifies its media pixel stream. A different decoder
output therefore requires a new audit instead of becoming an implicit
calibration or sensor-input change. Audit schema v1 remains parseable for
historical reproduction but cannot drive deterministic preparation.

Preparation-spec v4 re-runs qualification v2 before any preparation decode. It
requires the exact qualification and project-preregistration bytes and audit
identity. V4 currently accepts a `raw_recorded` preparation profile only;
qualification v2 does not yet model calibration-source project roles for the
structure-suppressed track.

For each prepared sequence, the complete selected media path and SHA-256 must
match an active preregistered assignment, a provider-fact entry, and the audit;
the manifest split must equal the project assignment, and the manifest capture
group must equal the verified provider capture group. A schema-v5 template's
gas identity must equal the verified provider gas identity, and its
`dataset_qualification` path, hash, and purpose must equal the preparation's
qualification lineage. The set of prepared media identities must exactly cover
the preregistration's active assignments.

V4 also derives the operations it actually performs: `decode_analysis` is
always required and `extract_bounded_window` is additionally required if any
sequence requests a window. Each must already appear in the exact
preregistration, whose operation list in turn determines the required
permission scopes. Preparation does not infer permission from a licence name or
silently add an unregistered operation. A schema-v5 template remains schema v5
after preparation, whether it names the whole asset or a window; a legacy
schema-v3 template retains the established schema-v3/schema-v4 output behavior.
Preparation schemas v1-v3 are unchanged and cannot consume a qualification-v2
artifact through the v4 binding path.

Freeze one split only from the exact preparation-report byte hash:

```powershell
plume-focus-recorded-data freeze `
  --root data/recorded `
  --preparation-report metadata/preparation-report.json `
  --preparation-report-sha256 <SHA256_OF_PREPARATION_REPORT> `
  --suite-id held-out-v1 `
  --split held-out `
  --frozen-at-utc <ISO_8601_UTC_TIMESTAMP> `
  --pre-change-model-commit <FULL_40_CHARACTER_COMMIT> `
  --freeze-provenance "<WHO_FROZE_IT_AND_WHY>" `
  --output manifests/held-out-suite-v1.json `
  --freeze-record metadata/held-out-suite-v1.freeze.json

plume-focus-recorded validate `
  --freeze-record metadata/held-out-suite-v1.freeze.json `
  --root data/recorded

plume-focus-recorded validate `
  --suite data/recorded/manifests/held-out-suite-v1.json
```

All non-absolute paths accepted by `freeze`, and the freeze-record path passed
to `validate`, are relative to `--root`. Sequence/suite manifest paths are
resolved directly from the current working directory.

The companion freeze record binds the preparation report and exact suite
bytes. Its dedicated validation path rechecks canonical JSON, the audit,
preparation spec/profile/report, raw source media, calibration source,
calibration artifact, prepared manifests, and exact split membership. It does
not decode sequence media or validate timestamp, wind, or truth sidecars; run
`validate --suite` separately for the runtime sequence assets. Conversely,
suite validation does not establish preparation lineage, so acceptance review
requires both checks.

The freeze record itself is labeled `not_sequence_evidence`. The suite
manifest has no claim-eligibility field; it freezes sequence references but
does not create a performance claim. The current freezer accepts the exact
truth-free preparation outputs. A future R2 truth-attachment artifact must
extend this lineage explicitly before truth-bearing prepared suites can be
frozen; editing a prepared manifest in place is rejected.

## Explicit sequence-manifest schema v5 gas identity

Schema v5 makes gas identity an explicit evidence field instead of encoding it
in the planner's scenario name. It adds these required strict fields:

```json
{
  "manifest_schema_version": 5,
  "scenario": "plume",
  "gas_identity": {
    "identity": "propane",
    "provenance": "Controlled propane release declared by the primary dataset record."
  },
  "dataset_qualification": {
    "path": "metadata/dataset-qualification-v2.json",
    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "purpose": "diagnostic_raw_pilot"
  }
}
```

Allowed identities and planner mappings are fixed:

| `gas_identity.identity` | Required planner `scenario` |
| --- | --- |
| `methane` | `methane` |
| `propane` | `plume` |
| `hydrocarbon_unspecified` | `plume` |

The `scenario` field selects the planner/model profile; it is not a species
label. A propane or unspecified-hydrocarbon recording therefore cannot be
serialized as methane merely to reuse the methane planner configuration. Gas
identity provenance must be non-empty, and the complete v5 manifest, including
that provenance, is retained in run provenance. This typing does not establish
gas concentration, species detection, or equivalence between gases.

`dataset_qualification.path` is normalized and data-root-relative,
`dataset_qualification.sha256` binds the exact bytes, and
`dataset_qualification.purpose` is `diagnostic_raw_pilot` or `evidence_suite`.
Before any media decode, the loader re-hashes that artifact, requires
qualification schema v2, re-runs its checker, requires eligibility and purpose
agreement, and checks that its audit lineage equals the manifest's. It then
verifies the exact project-preregistration bytes, requires
`decode_analysis`, and matches the selected media path/hash, project split,
provider capture group, and provider gas identity. The resolved qualification,
operations, and selected provider/project facts are retained in run provenance.

The recorded workflow separately requires preregistered `render_local_media`
before constructing a video renderer. Because qualification checking has
already derived and verified that operation's permission scope, adding
`--output-video` cannot silently turn a decode-only preregistration into a local
media derivative.

V5 currently requires `raw_recorded` input because qualification v2 does not
yet model calibration-source project roles. It accepts a whole audited media
asset when `media_window` is absent. When
`media_window` is present, v5 uses the same parent/window byte hashes,
selection-metadata lineage, half-open interval, and rebased clock contract as
v4. A windowed v5 input remains limited to `raw_recorded`, `uint8_identity`,
truth-free evaluation, and `wind.mode=none`. Whole-asset v5 input does not gain
the latter three window-specific restrictions, but it remains raw and all
ordinary classified lineage, scanner, truth, calibration, and causal-access
checks still apply. Runtime additionally requires `extract_bounded_window`
when a v5 window is present; a decode-only preregistration cannot bypass the
operation gate with a hand-authored manifest.

Schema v5 is opt-in; schema v3 remains the compatibility default. Schemas
v1-v4 remain methane-only, reject both v5 fields, and keep their established
whole-asset/window, runtime, and canonical-payload behavior.

## Explicit sequence-manifest schema v4 windows

Schema v4 represents one immutable half-open interval within an audited parent
video. It is deliberately limited to `raw_recorded`, truth-free input with
`wind.mode=none`; aligned windowed wind and same-parent calibration require
separate future contracts.

The preparation-spec v2 request records:

- requested source start/end seconds;
- expected inclusive start and exclusive end frame indices; and
- a confined selection-metadata path/hash plus selection provenance.

Preparation resolves the complete authoritative parent clock with
`searchsorted(..., side="left")`, rejects index drift and out-of-range requests,
decodes only the declared interval into memory, and emits a resolved
`media_window` containing:

- the requested bounds and exact resolved source-frame range;
- first and last included source timestamps;
- parent-media and decoded-window SHA-256 hashes; and
- the immutable selection-metadata lineage.

Runtime verifies the parent bytes, selection metadata, decoded window, and
source timestamps before rebasing the window timeline to zero. Reports retain
the original source-frame/time provenance. Schema v1-v3 manifests continue to
load whole assets and do not gain a `media_window` field.

The three-frame runtime minimum also applies to a window. Evaluation truth, if
added by a future contract, must already match the window length; the loader
never guesses how to slice a truth sidecar.

## Default sequence-manifest schema v3

Schema v3 permits a truth-free manifest:

```json
{
  "evidence_grade": "none",
  "annotation_provenance": "Truth is attached by a separate evidence workflow."
}
```

Such a run is eligible only for causal/qualitative engineering. Reports set
`spatial_metrics_available` to false and must be interpreted using that flag
and the metric sample count; legacy numeric placeholders are not evaluated
zeros.

For `structure_suppressed_emulation`, v3 replaces runtime leak-off fitting with
a `frozen_calibration` object containing:

- canonical calibration `path` and `sha256`;
- audited `source_path`, `source_sha256`, `source_provenance`, scope, and full
  producing commit;
- exact preparation spec/profile paths and hashes; and
- a calibration-record path/hash that independently binds the fit algorithm,
  source capture group, complete profile, shape, sample count, and artifact.

The loader verifies every link and loads the canonical calibration without
decoding the leak-off media. Scanner, wind, source-assumption, and later truth
changes do not alter calibration identity; acquisition, clock,
normalization, media, split, or capture-group changes invalidate it.

## Legacy classified sequence-manifest schema v2

Schema v2 introduced four evidence boundaries to the common recorded-sequence
fields:

- `dataset_lineage` references a hashed dataset audit, its accession IDs, and
  the exact audited source-video hash. The source hash must equal the sequence
  media hash.
- `capture_group` identifies the acquisition-level unit used to prevent one
  capture from leaking across development and held-out splits.
- `sensor_input` declares either raw recorded intensity or a
  structure-suppressed sensor emulation.
- Scanner and truth evidence receive explicit grades from which
  `claim_eligibility` is derived. A manifest cannot self-declare that label.

The following is a complete schema-v2 field-shape example. Replace every
example identity, provenance string, path, and 64-character placeholder hash
with values from the actual accession. The audit referenced here must contain
the matching dataset/accession IDs and exact source and calibration asset
hashes:

```json
{
  "manifest_schema_version": 2,
  "sequence_id": "development-001",
  "split": "development",
  "split_provenance": "Assigned by acquisition group before policy development.",
  "scenario": "methane",
  "dataset_lineage": {
    "dataset_id": "external-ogi-candidate",
    "accession_id": "external-ogi-candidate-20260726",
    "audit_path": "assets/dataset-audit.json",
    "audit_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "source_asset_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  },
  "capture_group": {
    "capture_group_id": "acquisition-session-001",
    "grouping_provenance": "All clips from this capture session are one split unit."
  },
  "media": {
    "path": "assets/external-accession/sensor.mp4",
    "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "capture_provenance": "Externally recorded OGI clip; see accession record."
  },
  "sensor_input": {
    "kind": "structure_suppressed_emulation",
    "provenance": "Fixed leak-off calibration used to emulate cleaner relative gas evidence.",
    "emulation_profile": {
      "profile_id": "structure-suppressed-development-v1",
      "residual_mode": "two_sided_magnitude",
      "provenance": "Selected using development evidence only.",
      "noise_floor_raw_units": 1.0,
      "output_units_per_noise_sigma": 32.0,
      "smoothing_time_constant_seconds": null,
      "profile_schema_version": 1
    },
    "leak_off_calibration": {
      "path": "assets/external-accession/leak-off.mp4",
      "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      "provenance": "Separate leak-off capture recorded before policy execution.",
      "calibration_scope": "development_only"
    }
  },
  "frame_clock": {
    "mode": "constant_fps",
    "fps": 15.0,
    "provenance": "Container rate verified against the acquisition record."
  },
  "normalization": {
    "mode": "uint8_identity",
    "provenance": "Decoded single-channel uint8 display intensity is preserved."
  },
  "calibration": {
    "sensor_id": "external-ogi-camera-001",
    "signal_units": "compressed_8bit_ogi_display_intensity",
    "provenance": "Non-radiometric video; no concentration calibration claimed."
  },
  "wind": {
    "mode": "none",
    "units": "px_per_second",
    "provenance": "No synchronized wind measurement is available.",
    "conversion_provenance": "not applicable"
  },
  "source_assumption": {
    "mode": "known_source",
    "provenance": "Release point declared from the acquisition protocol before evaluation.",
    "source_point": [120.0, 40.0],
    "initial_roi": [20, 80, 180, 220]
  },
  "scanner": {
    "provenance": "Counterfactual column scanner fixed before held-out execution.",
    "evidence_kind": "counterfactual_virtual",
    "scan_mode": "column",
    "scan_rate_hz": 20.0,
    "sample_stride_px": 8,
    "vertical_pass_seconds": 1.0,
    "horizontal_step_seconds": 0.05,
    "column_stride_px": 1,
    "column_mass_fraction": 0.8,
    "row_mass_fraction": 0.85,
    "minimum_sweep_columns": 30,
    "minimum_sweep_rows": 56,
    "maximum_sweep_columns": 120
  },
  "evaluation_truth": {
    "path": "assets/truth.npz",
    "sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    "truth_kind": "probability_weight",
    "storage_format": "lossless_npz_uint8",
    "evidence_grade": "silver_pseudo",
    "method_id": "frozen-pseudo-truth-method-v1",
    "protocol_path": "assets/truth-protocol.json",
    "protocol_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    "annotation_provenance": "Generated offline under the separately hashed frozen protocol."
  },
  "model_profile_id": "uncalibrated_video_v1"
}
```

The example is structurally complete, but it is not runnable until every
placeholder has been replaced and all referenced files and audit entries agree.
Unknown or omitted keys fail strict parsing.

### Raw and structure-suppressed input

Raw input is declared without an emulation profile:

```json
{
  "kind": "raw_recorded",
  "provenance": "<WHY_RAW_RECORDED_VALUES_ARE_USED>"
}
```

`structure_suppressed_emulation` treats raw OGI as the external plume-dynamics
source while presenting the virtual scanner with a hypothesized cleaner
relative-evidence field. It uses a fixed per-pixel median background and robust
MAD noise scale fitted from the separately hashed leak-off clip. Bright, dark,
and two-sided residual modes are explicit. Output is clipped to a nonnegative
uint8-like range. This is a versioned emulation hypothesis, not evidence that
the transform reproduces an eventual hardware sensor.

The leak-off clip must be distinct from the evaluation video and truth by both
path and hash, and its hash must appear in the same strict accession audit as
the source video so its rights state cannot be omitted. Declare its
`calibration_scope` as `development_only` or `independent_pre_run`.
Structure-suppressed input currently requires `uint8_identity`; upstream
linear or per-frame normalization would make the emulation units ambiguous.
Per-frame min-max normalization and histogram equalization are not part of
this contract.

Temporal integration is reserved for a future scanner-response adapter and is
deliberately unsupported in the current emulation profile. Set
`smoothing_time_constant_seconds` to `null`. This avoids treating unobserved
video frames as sensor history and postpones cross-scan sensor state until a
hardware timing contract exists.

### Truth storage and evidence grade

Classified v2/v3 truth must be a lossless NPZ containing exactly one array named
`frames`, with dtype `uint8` and shape `(frames, rows, cols)`. It must match the
sensor frame count and decoded shape.

- `silver_pseudo` identifies frozen algorithmic or ensemble pseudo labels.
  It remains diagnostic regardless of metric value. With verified rights its
  claim label is `diagnostic_only_pseudo_truth`; unverified or restricted
  rights produce the stricter `blocked_unverified_data_rights` label.
- `gold_human_reviewed` identifies truth produced under the separately hashed
  review protocol. The confined `protocol_path` is hash-verified during
  loading. The grade still requires external protocol and freeze review and
  does not prove hardware scanning or dataset independence.

`binary_support` uses pixels at least `128/255` as support.
`probability_weight` normalizes nonzero uint8 weights per frame. Truth is
available only through the offline evaluator and is never supplied to the
planner, sensor emulator, or scan policy.

### Scanner evidence

The recorded-video adapter currently permits only
`counterfactual_virtual`. Scan area and time are produced by the declared
mechanical model over recorded frames; they were not measured from the camera
or a physical scanner. `measured_hardware` is reserved for a future adapter
with measured scan evidence and is rejected by this workflow.

The recorded loader decodes finite backing media into a timestamp-addressable
source. In headless scan mode, the causal engine then requests only declared
mechanical sample coordinates and timestamps from that source; planner code
does not receive a full-frame accessor. Schema v3 does not derive truth from
sensor frames. Explicit full-frame baselines and renderers remain separately
labeled.

## Common sequence fields

Every classified schema-v2 through schema-v5 sequence also declares:

- sequence ID, `development` or `held-out` split, and split provenance;
- `uncalibrated_video_v1` model profile and a planner scenario: v2-v4 require
  `methane`, while v5 requires the gas-identity mapping above;
- sensor video, capture provenance, and byte hash;
- constant FPS or a hashed timestamp CSV;
- sensor identity, signal units, and calibration-status provenance;
- absent, constant, or timestamped image-plane wind, except that current
  windowed v4/v5 inputs require absent wind until window-aligned wind is
  defined;
- known-source, known-corridor, or source-free assumption; and
- complete point- or column-scan settings and their provenance.

For irregular capture times:

```json
{
  "mode": "timestamp_csv",
  "path": "assets/timestamps.csv",
  "sha256": "<SHA256_OF_TIMESTAMP_CSV>",
  "provenance": "<CLOCK_EXTRACTION_AND_SYNCHRONIZATION_PROVENANCE>"
}
```

The CSV columns are:

```csv
frame_index,timestamp_seconds
0,0.0
1,0.083
2,0.169
```

Indices are contiguous from zero, timestamps start at zero and increase
strictly, and the row count matches the decoded frame count. Lookup always
selects the latest capture not later than the requested time.

When synchronized wind is unavailable:

```json
{
  "mode": "none",
  "units": "px_per_second",
  "provenance": "<WHY_SYNCHRONIZED_WIND_IS_UNAVAILABLE>",
  "conversion_provenance": "not applicable"
}
```

Constant image-plane wind uses exactly these fields:

```json
{
  "mode": "constant",
  "units": "px_per_second",
  "provenance": "<MEASUREMENT_PROVENANCE>",
  "conversion_provenance": "<PHYSICAL_OR_IMAGE_TO_PIXEL_CONVERSION>",
  "dy_px_per_second": 2.0,
  "dx_px_per_second": 5.0
}
```

Timestamped wind uses exactly these fields:

```json
{
  "mode": "time_series_csv",
  "units": "px_per_second",
  "provenance": "<MEASUREMENT_PROVENANCE>",
  "conversion_provenance": "<PHYSICAL_OR_IMAGE_TO_PIXEL_CONVERSION>",
  "path": "assets/wind.csv",
  "sha256": "<SHA256_OF_WIND_CSV>",
  "max_age_seconds": 0.5
}
```

Its CSV header is exact:

```csv
timestamp_seconds,dy_px_per_second,dx_px_per_second
0.0,2.0,5.0
0.5,2.5,4.7
```

Missing or stale wind falls back to motion from successive completed scan
posteriors when available; the first decision uses explicit zero motion.
Source-free describes only the source-location assumption: it may use declared
constant or timestamped external wind, scan-derived fallback, or initial zero
motion. It remains plume tracking without a supplied source point, not
unknown-source discovery.

Source assumptions use one of these exact field sets:

```json
{
  "mode": "known_source",
  "provenance": "<PREDECLARED_SOURCE_PROVENANCE>",
  "source_point": [120.0, 40.0],
  "initial_roi": [20, 80, 180, 220]
}
```

```json
{
  "mode": "known_corridor",
  "provenance": "<PREDECLARED_CORRIDOR_PROVENANCE>",
  "initial_roi": [20, 80, 180, 220]
}
```

```json
{
  "mode": "source_free",
  "provenance": "<WHY_NO_SOURCE_LOCATION_ENTERS_PLANNING>"
}
```

`source_point` is `[row, column]`. `initial_roi` is inclusive
`[min_row, min_col, max_row, max_col]`. A source-free declaration may not
contain either field.

For scanner settings, both `point` and `column` modes use the complete field
set shown in the v2 example. `scan_mode` is the only mode selector,
`maximum_sweep_columns` may be an integer or `null`, and classified schemas require
`evidence_kind: counterfactual_virtual`. All timing, stride, and mass fields
remain mandatory even when a selected scan mode does not consume every one,
so report serialization and cross-mode comparisons remain stable.

## Schema-v1 legacy compatibility

Strict schema-v1 manifests remain loadable so prior runs can be reproduced.
Their scanner, sensor input, and truth evidence have no classification fields,
truth remains a hashed video, and validation derives
`legacy_unclassified_evidence`. Loading v1 does not upgrade its evidence.

The previous example remains valid as a legacy manifest shape:

```json
{
  "manifest_schema_version": 1,
  "sequence_id": "development-001",
  "split": "development",
  "split_provenance": "Assigned before policy calibration under a declared protocol.",
  "scenario": "methane",
  "media": {
    "path": "assets/external-accession/sensor.mp4",
    "sha256": "<SHA256_OF_SENSOR_VIDEO>",
    "capture_provenance": "<ACQUISITION_PROVENANCE>"
  },
  "frame_clock": {
    "mode": "constant_fps",
    "fps": 12.0,
    "provenance": "<CLOCK_PROVENANCE>"
  },
  "normalization": {
    "mode": "fixed_linear",
    "provenance": "<LEGACY_NORMALIZATION_PROVENANCE>",
    "raw_min": 0.0,
    "raw_max": 255.0
  },
  "calibration": {
    "sensor_id": "<SENSOR_ID>",
    "signal_units": "<DECLARED_SIGNAL_UNITS>",
    "provenance": "<CALIBRATION_STATUS_PROVENANCE>"
  },
  "wind": {
    "mode": "none",
    "units": "px_per_second",
    "provenance": "<WIND_AVAILABILITY_PROVENANCE>",
    "conversion_provenance": "not applicable"
  },
  "source_assumption": {
    "mode": "known_corridor",
    "provenance": "<PREDECLARED_CORRIDOR_PROVENANCE>",
    "initial_roi": [60, 20, 180, 220]
  },
  "scanner": {
    "provenance": "<LEGACY_SCANNER_FIXTURE_PROVENANCE>",
    "scan_mode": "column",
    "scan_rate_hz": 20.0,
    "sample_stride_px": 8,
    "vertical_pass_seconds": 1.0,
    "horizontal_step_seconds": 0.05,
    "column_stride_px": 1,
    "column_mass_fraction": 0.8,
    "row_mass_fraction": 0.85,
    "minimum_sweep_columns": 30,
    "minimum_sweep_rows": 56,
    "maximum_sweep_columns": 120
  },
  "evaluation_truth": {
    "path": "assets/truth.mp4",
    "sha256": "<SHA256_OF_LEGACY_TRUTH_VIDEO>",
    "truth_kind": "binary_support",
    "annotation_provenance": "<LEGACY_ANNOTATION_PROVENANCE>"
  },
  "model_profile_id": "uncalibrated_video_v1"
}
```

## Experiment report schema v3

Current experiment execution emits report schema v3. It retains the schema-v2
metric meanings and the separation among requested masks, mechanical sweeps,
sampled pixels, attention diagnostics, and offline truth. The added evidence is
operational and observer-independent; enabling the video renderer does not
change the causal schedule, belief checks, or sampled-value summary.

`operational_trace.schedule` contains one entry per completed scan. Each entry
records:

- scan start, completion, decision, and next-target times;
- the inclusive bounding regions for the requested mask, actual mechanical
  sweep, sampled pixels, and planned next scan;
- domain-separated SHA-256 values for the requested, swept, sampled, and next
  masks;
- a hash of sampled coordinates and their per-sample source-clock times;
- a separate hash, count, minimum, and maximum for the causal sampled uint8
  values;
- requested, swept, and sampled area fractions and effective sample points;
- the range and count of evidence timestamps;
- reacquisition/convergence state and uncertainty radius; and
- posterior/predictive probability sums plus finite, nonnegative, and
  normalized flags.

The trace labels source frame indices as
`not_available_at_causal_runner_boundary`; it does not reconstruct or guess
indices that the causal scan interface did not expose. Mask and value hashes
are integrity fingerprints, not accuracy or uniqueness scores.

`belief_integrity` aggregates finite, nonnegative, and normalization checks over
the posterior and predictive distributions, including the declared absolute
tolerance, maximum probability-sum error, and minimum probability. Its status is
`evaluated` or `not_evaluated`; `evaluated` is not a synonym for verified or
passing. Read `all_finite`, `all_nonnegative`, and `all_normalized` for the
outcome. A false integrity flag is visible evidence of a numerical problem;
true flags do not prove that the belief is calibrated or physically correct.

`sample_variation` compares only same-coordinate values actually present in
causal `sampled_mask` evidence at strictly later source-clock observations. It
reports comparisons, changed values/fraction, mean absolute change, and maximum
absolute change, explicitly marks full-frame evidence mode, and sets
`variation_observed` from whether at least one compared value changed. Its
status is also only `evaluated` or `not_evaluated`, never `verified`. This can
show that a run consumed changing temporal measurements rather than a frozen
image. It is not a plume label, spatial recall metric, gas calibration, or proof
that the changes are plume-caused. A run with no valid repeated-coordinate
comparisons reports `not_evaluated` and `variation_observed=false` instead of
inventing variation.

For a classified recorded input,
`data_provenance.evidence_contract.media_attribution` carries the manifest's
source credit, dataset/accession identity, source URI, rights status, licence
URI and provenance, and a list of analysis changes. When an annotated proof
video is requested, the report marks `rendered_derivative=true` and records the
time-compressed three-panel rendering change. The distribution condition says
to retain the JSON report as the attribution-and-changes companion to any
rendered derivative. This metadata supports attribution; it does not expand the
underlying licence or establish that redistribution is permitted.

Historical schema-v2 reports remain historical evidence artifacts and do not
acquire v3 trace fields retroactively. Report schema v3 likewise does not raise
a sequence's claim eligibility or make a truth-free engineering run a
performance result.

## Validation, execution, and frozen suites

Validate one sequence and every referenced hash:

```powershell
plume-focus-recorded validate `
  --manifest data/recorded/manifests/development-001.json
```

Run causally; video rendering is optional:

```powershell
plume-focus-recorded run `
  --manifest data/recorded/manifests/development-001.json `
  --max-scans 8 `
  --report outputs/recorded-development-001.json `
  --output-video outputs/recorded-development-001.mp4
```

Validation returns the canonical manifest hash, resolved evidence grades,
derived claim eligibility, and every active limitation. Copy the canonical
hash into a suite reference; do not hash formatted JSON manually.

Suite schema v1 remains the current suite container for all supported sequence
schema versions:

```json
{
  "suite_schema_version": 1,
  "suite_id": "<SUITE_ID>",
  "split": "held-out",
  "frozen_at_utc": "<ISO_8601_UTC_TIMESTAMP>",
  "frozen_before_model_commit": "<PRE_CHANGE_GIT_COMMIT>",
  "freeze_provenance": "<WHO_FROZE_THE_SUITE_AND_WHY>",
  "sequences": [
    {
      "path": "held-out-001.json",
      "manifest_sha256": "<CANONICAL_HASH_FROM_SEQUENCE_VALIDATION>"
    }
  ]
}
```

Validate the suite:

```powershell
plume-focus-recorded validate `
  --suite data/recorded/manifests/held-out-suite-v1.json
```

Validate development and held-out suites together before acceptance. This
reruns both suite validations and rejects cross-role reuse of source media,
calibration media, or acquisition/calibration capture groups. Legacy v2
calibration sources lack a capture-group field, so their byte isolation is
enforced while group-level completeness is reported as false:

```powershell
plume-focus-recorded validate-pair `
  --development-suite data/recorded/manifests/development-suite-v1.json `
  --held-out-suite data/recorded/manifests/held-out-suite-v1.json
```

Hashes and frozen manifests establish exactly which declared bytes were used.
They do not establish independent collection or untouched held-out status
without the external protocol record. A successful GasVid-like silver run can
complete a diagnostic engineering stage, but it cannot complete the
independent-evidence, hardware-readiness, or overarching system gates in
`SYSTEM_GOAL.md`.

The completed truth-free schema-v5 controlled-propane smoke has eligible
rights, intact qualification lineage, changing sampled values, normalized
beliefs, and a valid rendered derivative. Its near-full-frame raw requests are
a negative focusing diagnostic, not a coverage result. Without independent
spatial truth it cannot measure plume coverage, and propane footage cannot be
reported as methane validation.
