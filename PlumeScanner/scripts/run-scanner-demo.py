"""Run one self-contained synthetic scanner demo and verify its public trace."""
import argparse
from collections import Counter
import json
from pathlib import Path

from plume_focus.full_fov_execution import decode_config, run_full_fov, verify_replay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evidence-gated-focus/coarse-demo.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/quickstart"))
    args = parser.parse_args(argv)
    config = decode_config(json.loads(args.config.read_text(encoding="utf-8")))
    target = args.output / "runs" / "demo.json"
    if target.exists():
        parser.error(f"{target} already exists; choose a fresh --output directory.")
    result = run_full_fov(config)
    verify_replay(result)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another run using this destination.
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"record": str(target), "configuration_hash": config.digest,
        "policy": config.policy, "simulated_horizon_seconds": config.duration_seconds,
        "scans": dict(Counter(scan["plan"]["mode"] for scan in result["scans"])),
        "termination": result["termination"], "synthetic_J_over_T": result["J_over_T"],
        "public_replay": "passed"}, indent=2))


if __name__ == "__main__":
    main()
