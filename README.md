# Squishy Methane Analysis — Source-Assumption Experiments

This branch, `source_prior_fixed_evidence`, adds controlled source-assumption
experiments to the existing scanner in [`PlumeScanner/`](PlumeScanner/).
The question is how incorrect information about a methane source changes the
map reconstructed from sequential measurements.

The baseline is [`plume_scanner_new` at `5cac633`](https://github.com/BerkeleyExpertSystemTechnologiesLab/Squishy-Methane-Analysis/tree/5cac633f47001d3aa455f8b6f3cc00b11c550ed3/PlumeScanner).
The experiments reuse its synthetic measurements, timing, scan path and Gaussian
field estimator; they do not replace its scan policy. See the
[experiment guide](PlumeScanner/docs/source-assumptions-demo.md) for the exact
configuration, input/output contract, simulation assumptions and comparisons.

The scanner is a research demonstrator. The current candidate remains opt-in;
synthetic reconstruction results are mixed, and calibrated gas strength,
hardware performance and leak-detection guarantees are not established.

## Get the scanner

Use a sparse checkout to avoid downloading the repository's unrelated datasets:

```sh
git clone --filter=blob:none --sparse --single-branch --branch source_prior_fixed_evidence https://github.com/BerkeleyExpertSystemTechnologiesLab/Squishy-Methane-Analysis.git
cd Squishy-Methane-Analysis
git sparse-checkout set PlumeScanner
cd PlumeScanner
```

## Install and run

Python 3.10 or newer is required; the submission is tested with Python 3.12.
Run these commands from `PlumeScanner/`.

Windows PowerShell:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe scripts/run-scanner-demo.py
```

macOS/Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python scripts/run-scanner-demo.py
```

The 40-second synthetic demo verifies public-call replay and writes
`outputs/quickstart/runs/demo.json`. It refuses to overwrite an existing run;
use `--output outputs/quickstart-2` for another run.

Optional videos and verification (Windows; use `.venv/bin/python` on macOS/Linux):

```powershell
.venv\Scripts\python.exe scripts/render-full-fov-demo.py --output outputs/quickstart --name demo --fps 4 --speed 1
.venv\Scripts\python.exe -m pytest -q
```

See the [scanner README](PlumeScanner/README.md) for the public API, configuration,
output interpretation and research evidence. The
[submission review](PlumeScanner/docs/reviews/company-submission.md) records the
cleanup and verification. Generated runs, media, environments and caches stay local.
