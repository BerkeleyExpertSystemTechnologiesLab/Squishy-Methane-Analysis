# Company submission

User-authorized destination on 2026-09-29:
`BerkeleyExpertSystemTechnologiesLab/Squishy-Methane-Analysis`, new branch
`plume_scanner_new`. The authenticated account has push access and is WinChurchill;
the submission must be attributed to that user.

Start from reviewed source `1bdf3c670a9698024b7bd1f59fa66ae2f45b0d02` on local
`codex/company-submission`. The isolated company branch starts from main
`94ed07fc96592b7cb073f59b88f9d98bd6e1ff37`, follows the existing `PlumeScanner/`
directory convention, and preserves all existing research directories. It imports
only the committed current source tree, not the source repository's unrelated Git
history, untracked work, generated evidence, recordings, caches or environments.

The root README gives direct checkout/setup/run instructions; the package README
covers a complete 40 s source-anchored, known-wind synthetic example, replayable
JSON, physical-time and summary video rendering, the public API and test commands.
Older instructions remain in `docs/legacy-usage.md`. The new single-run wrapper
calls the unchanged causal runner and verifies its public trace; it refuses to
overwrite an existing execution. The two-cycle example dispatches all supported
run-configuration versions. A tested complete demo configuration uses the frozen
coarse instrument, estimator and evidence-gate settings with a shorter horizon;
it is not additional scientific comparison evidence.

Scientific status remains `not_demonstrated`; the 104-run result remains mixed,
with no calibrated field confidence, strength estimate or hardware validation.
The cleanup's original records and their hashes remain local and unchanged.

Pre-submission auditing found that Git's prior text normalization stored different
bytes for several frozen JSON files than the working copies used to declare their
hashes. `*.json -text` now preserves actual frozen bytes across checkouts; source
profiles, context-package hashes and parsed configurations remain unchanged.

Verification complete in the isolated company checkout with a fresh Python
3.12.14 environment installed using `python -m pip install -e '.[dev]'`:

- `python -m pytest -q --basetemp outputs/company-submission-tests`: 554 passed,
  five Windows symlink skips, 47.16 s.
- `python -m compileall -q src scripts examples tests`: passes.
- `python scripts/run-scanner-demo.py`: public replay passes, FULL x 1 / FOCUS x 2,
  40 s horizon, `incomplete_full_scan_budget`, synthetic J/T
  .016568662718721257. Configuration hash
  `2c43534d49a0596c3fbeae372c90d01333d3a42492ffc1e82ae924b02b276446`.
- `python scripts/render-full-fov-demo.py --output outputs/quickstart --name demo
  --fps 4 --speed 1`: 161 physical frames and three summary frames decode;
  startup and focus keyframes visually checked. Rendering leaves execution bytes
  unchanged: SHA-256
  `e287f21cc17dc88af8e296c85b638f5cfa6bdf9733ec0c029114a57cf3c1c50f`.
- The two-cycle example completes FULL then FOCUS. README local links resolve.
- All 271 package files match source Git blobs after normal text EOL conversion;
  frozen JSON and provider evidence retain their exact bytes. Explicit staging
  contains only `PlumeScanner/`, root README and the scripts ignore exception.
  Existing historical Markdown hard breaks and redundant blank EOF lines are
  cleaned so the imported tree also passes whitespace checking.

The installed module resolves to the isolated company package, not the original
editable installation. Fresh dependencies are NumPy 2.5.3, SciPy 1.18.1, OpenCV
headless 5.0.0.93 and pytest 9.1.1. Local audit and render manifests remain in
ignored `outputs/`. Publishing is pending and limited to the requested branch,
with no force push or merge. Source/submission producing commits, GitHub author
attribution and remote equality will be recorded at closeout. No pull request is
requested.
