# Evidence-gated focus review and cleanup

Started 2026-09-29 from `98aeba97a9e0d18e77ac018bf1384a4c0b624e71` on
`codex/evidence-gated-focus`; working branch `codex/evidence-gated-cleanup`.
The user requested one review/cleanup round before company-repository transfer,
then clarified that the company URL or upload-instruction request will come later.
This round prepares a local commit and does not push to the configured personal remote.

Scope is the four open findings in the
[partial checkpoint](evidence-gated-focus.md#open-review-findings-and-partial-checkpoint),
plus directly related missing-evidence reporting and documentation cleanup.
Estimator, transport, scanner mechanics, instrument profiles, policy settings and
scientific question stay fixed. No new experiment, default promotion, model-signal
suppression, source-activity inference or robotics integration is included.

## Repairs

1. Scheduled-slot mapping checks stable reading/scan identity and reported position.
   Actual acquisition time is retained and validated by the shared acquisition
   lifecycle rather than compared for equality with the plan. Tests cover delayed
   acquisition, actual overruns, entry and renewal, identity/position mismatch,
   duplicate slots, pre-start exposures and receipts ending before an exposure.
2. Resumption scoring filters individual acquisition midpoints at the change
   boundary, including scans already underway. A 200–260 s scan with its first
   qualifying post-resumption reading at 250 s now scores 10 s after resumption at
   240 s. Pre-resumption hits cannot count. Post-change slots in straddling full scans
   are reported explicitly; a short remaining horizon does not imply no opportunity
   in an ongoing scan. The frozen criterion requiring enough time for a fresh full
   scan remains distinct and unchanged.
3. `gate_evidence_actions` records each preceding evidence action's identities,
   alpha, slots, multiplier and cutoff. Next-action allocation retains its separate
   existing fields. Multi-action entry and fresh renewal are covered through JSON
   round trips and renderer text. Old plans decode with absent optional provenance;
   replay compares every original field and checks provenance on new plans.
4. The normal report producer places its criterion verdict after aggregate fields,
   preventing `complete` from replacing `supported`. Passing, loss-failing,
   coverage-failing and unavailable-data cases are tested. Missing coverage or an
   absent profile reports unavailable; empty development evidence cannot establish
   that unsupported focus forces a full scan.

The renderer retains the six-panel layout, with a corrected evidence/next-alpha
label and text scaled to fit its existing header. A separate 40 s coarse continuous
engineering smoke produced and fully decoded two videos (9 physical-time frames,
3 summary frames). The reviewed first-focus summary at 36.775 s displays evidence
alpha .025, M 640 and cutoff .3556, separately from next alpha .05. Rendering leaves
the smoke execution hash unchanged. This smoke is a layout check, not study evidence.

## Verification and closeout

Engineering status: **complete for the bounded review and cleanup**. Producing
implementation commit `b1684f2eed85ace43d1312e92cb9925aa61ae8a2`.
The final full suite passes **553 tests, five existing platform skips**; the focused
gate/report/full-FoV checks pass 49. Compilation and `git diff --check` pass.
All **104 original executions** pass public-plan replay, causal-loss replay within
the existing 1e-12 tolerance, exact original-metric checks and frozen source-hash
checks under the cleanup implementation. No historical file was rewritten.

Normal report generation now classifies continuous preservation as `supported`.
All seven predeclared behavioral checks remain supported and reconstruction remains
`mixed`. Twelve candidate/control intermittency summaries change when ongoing scans
are included. The coarse candidate first qualifies a reading .10-.95 s after
resumption, within the existing full scan; the old 26 s score described the later
focused action. `reacquisition_focus_plan` is false at these earlier reading times,
so a measurement hit is not mislabeled as a newly selected focused plan.
The doubled candidate still has no qualifying hit despite **566 post-resumption
slots** in its ongoing full scan. A fresh 129.5875 s scan cannot fit in the remaining
120 s, so it stays outside the frozen full-restart criterion, but it did have an
ongoing-scan opportunity. That limitation is explicit rather than hidden by the
budget explanation. The added `reacquisition_opportunity` diagnostic describes
full-scan opportunity; reading latency independently scores FULL and FOCUS scans.

Local provenance:

| Artifact | SHA-256 |
|---|---|
| `outputs/evidence-gated-cleanup-final-2026-09-29/report.json` | `d79f0a4fa1090ac113e5dc3e79e1397bc7adb6269ad0e16dc0bf8c9164fda159` |
| Review operational code | `7d2f5c0bc1a7120ad6e13a0a082248bc6107fdbf7e2e083082ea0cbf3ea352c9` |
| Additive review script | `0c75069209efd867e17acf158328b460bc5b94371dbaf6aa5a9b28b59699f3f9` |
| Original producing code | `e6d6b22d0634390dbcb686ea7a7b92e2177f9bb2b987ab22c36d377daf64a5b4` |
| Smoke execution | `1d0b789fb3f472cf10b715094fb509e304a57a97cb2c211e806c809bfa8f354b` |
| Smoke media manifest | `e43c4df1540cfae17c88dc7be54a74c98484e39109fdb13650d6711ba2321a2c` |
| Smoke visual review | `5c33fe12cbf72e79178655a712d14e0e560263fc8acabc9fcef9c42112af30c2` |

The review report carries per-record source hashes, old/new change metrics and
declaration/freeze/report/inventory/held-out-manifest hashes. Media evidence remains
under `outputs/evidence-gated-cleanup-media-2026-09-29/`; its manifest records both
video hashes and full decode counts. Selected frames were inspected at 0, 36.775
and 39.925 s. Preliminary interrupted review output remains local as well.

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp outputs/review-cleanup-pytest
.\.venv\Scripts\python.exe -m pytest -q tests/test_evidence_gate.py tests/test_evidence_gate_report.py tests/test_full_fov_loop.py
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
$env:OPENBLAS_NUM_THREADS='1'
.\.venv\Scripts\python.exe scripts/review-evidence-gated-focus.py --output outputs/evidence-gated-cleanup-final-2026-09-29 --workers 3
git diff --check
```

The additive review command refuses an existing destination or a destination
overlapping the frozen evidence root. It verifies frozen source hashes and producing
code/configuration/freeze lineage, replays public plans and causal reconstruction
loss, checks the original metrics, and writes recomputed change diagnostics and
criteria separately. It requires saved truth caches and never generates caches or
writes reports in the frozen root. Review-code provenance is distinct from the
original producing-code provenance. Generated reports, traces and media stay local.

The scientific claim remains `not_demonstrated`: the historical reconstruction
outcome is mixed, gate bounds are conditional measurement-error bounds, and false
posterior source signal remains. Engineering cleanup cannot establish calibrated
field confidence, reconstruction superiority or strength estimation.
