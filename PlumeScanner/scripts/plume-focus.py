"""Run the Plume Focus project from the repository checkout.

This wrapper adds `src/` to the import path so the project can be run before it
is installed as an editable package.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from plume_focus.demo import main


if __name__ == "__main__":
    main()
