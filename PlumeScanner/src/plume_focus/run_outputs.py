"""Run-directory guard, separate from intentionally updated within-run progress."""
from pathlib import Path


def require_fresh_output(path):
    output=Path(path)
    if output.is_symlink() or getattr(output,"is_junction",lambda:False)():
        raise ValueError("A run output directory must not be a link.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a fresh output directory; previous evidence is preserved: {output}")
    output.mkdir(parents=True,exist_ok=True)
    return output
