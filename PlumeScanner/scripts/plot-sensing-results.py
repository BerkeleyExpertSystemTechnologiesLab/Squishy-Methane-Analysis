"""Plot exported development reports; requires the optional 'plots' extra."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, layout="constrained")
    colors = {"adaptive": "#167a9f", "static_corridor": "#a35618", "rescan": "#587940"}
    for axis, case in zip(axes, ("known-wind", "wind-error", "source-free")):
        for policy, color in colors.items():
            report = json.loads((args.directory / f"{case}-{policy}-seed-0.json").read_text())
            scans = report["scan_schedule"]
            axis.plot([s["actual_completion_seconds"] for s in scans],
                      [s["actual_sweep_mass_recall"] for s in scans], "o-", color=color,
                      label=policy.replace("_", " "), linewidth=1.6, markersize=4)
        axis.set_title(case.replace("-", " "))
        axis.set_xlabel("Actual scan completion time (seconds)")
        axis.set_ylim(-.04, 1.04)
        axis.set_xlim(0, 77)
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Actual sweep signal-mass recall")
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Development seed 0: coverage against actual charged scan time\nCommon budget ceilings; differing expenditure. No adaptive-advantage claim.", fontsize=11)
    fig.savefig(args.directory / "coverage-time.png", dpi=180)
    fig.savefig(args.directory / "coverage-time.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
