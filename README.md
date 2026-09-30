# Squishy Methane Analysis — Plume Scanner

This branch, `plume_scanner_new`, contains the reviewed active-scanning prototype
in [`PlumeScanner/`](PlumeScanner/). It chooses sequential point measurements to
reconstruct a changing plume under scanner motion and acquisition-time constraints.
The repository's other research projects are preserved from company `main`.

The scanner is a research demonstrator. The current candidate remains opt-in;
synthetic reconstruction results are mixed, and calibrated gas strength,
hardware performance and leak-detection guarantees are not established.

## Get the scanner

Use a sparse checkout to avoid downloading the repository's unrelated datasets:

```sh
git clone --filter=blob:none --sparse --single-branch --branch plume_scanner_new https://github.com/BerkeleyExpertSystemTechnologiesLab/Squishy-Methane-Analysis.git
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
