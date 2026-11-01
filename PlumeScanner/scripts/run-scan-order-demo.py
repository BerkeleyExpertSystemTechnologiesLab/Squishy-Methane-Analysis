"""Run the opt-in, preselected C2 P0-V b00_L comparison."""
import argparse
import os
from pathlib import Path

# Per-process setting only. No persistent machine or instrument settings change.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from plume_focus.scan_order_demo import run_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/scan-order-demo"))
    args = parser.parse_args()
    result = run_demo(Path(__file__).resolve().parents[1], args.output)
    print("C2 P0-V demo:", result["variant"], "(synthetic, not the full 24-base study)")
    print("Common simulation window (s):", result["window_seconds"], "; action budget (s):", result["action_budget_seconds"])
    for name, r in result["results"].items():
        print(f"{name}: columns={r['column_order']}; down={r['directions_down']}")
        print(f"  MSE={r['mse']:.12g}; paid action={r['action_seconds']} s; readings={r['readings']}")
    print(f"gain = MSE(B) - MSE(L_up) = {result['gain']:.12g} synthetic signal^2")
    print("Positive: L_up lower MSE; zero: tie; negative: B lower MSE. B remains default.")
    print("Results:", args.output / "RESULTS.json")


if __name__ == "__main__":
    main()
