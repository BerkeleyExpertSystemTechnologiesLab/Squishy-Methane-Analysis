"""Run the OpenCV moving-object proof demo from the repository checkout."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from plume_focus.video import main


if __name__ == "__main__":
    main()
