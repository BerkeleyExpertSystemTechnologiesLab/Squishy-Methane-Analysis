"""Summarize completed nuisance diagnostics without changing frozen execution."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plume_focus.source_amount_nuisance import METHODS, save_json, sha256

COLORS = {"fixed": "#697586", "joint": "#1565b0", "correct_dynamics": "#24835d"}
LABELS = {"fixed": "Fixed amount", "joint": "Joint location + amount", "correct_dynamics": "Correct dynamics, fixed amount"}


def diagnostic(row, method, time):
    return next(r for r in row["diagnostics"] if r["method"] == method and r["time"] == time)


def aggregate(condition, rows):
    selected = [r for r in rows if r["metadata"]["split"] == "evaluation" and r["metadata"]["seed"] is not None]
    result = dict(condition=condition, histories=len(selected),
                  JT=np.mean([r["JT"] for r in selected], axis=0).tolist(),
                  forecasts=np.mean([r["forecasts"] for r in selected], axis=0).tolist())
    result["joint_JT_change_percent"] = 100 * (result["JT"][1] / result["JT"][0] - 1)
    result["joint_forecast_change_percent"] = [100 * (v[1] / v[0] - 1) for v in result["forecasts"]]
    result["joint_lower_JT_count"] = sum(r["JT"][1] < r["JT"][0] for r in selected)
    joint = [diagnostic(r, "joint", 40.) for r in selected]
    result.update(amount_mean=float(np.mean([r["amount_mean"] for r in joint])),
                  amount_mean_range=[min(r["amount_mean"] for r in joint), max(r["amount_mean"] for r in joint)],
                  amount_absolute_error=float(np.mean([abs(r["amount_mean"] - 1) for r in joint])),
                  amount_intervals_excluding_truth=sum(not r["true_amount_inside95"] for r in joint),
                  max_amount_edge_mass=max(r["amount_edge_mass"] for r in joint))
    result["coverage"] = [dict(method=method, time=t,
        source_count=sum(diagnostic(r, method, t)["inside95"][-2] for r in selected),
        downwind_count=sum(diagnostic(r, method, t)["inside95"][-1] for r in selected),
        global_count=sum(sum(diagnostic(r, method, t)["inside95"][:-2]) for r in selected),
        source_denominator=len(selected), global_denominator=5 * len(selected))
        for method in METHODS for t in (20., 40., 46., 52.)]
    return result


def family_items(summaries, factor):
    return sorted([r for r in summaries if r["condition"]["factor"] in ("matched", factor)],
                  key=lambda r: r["condition"]["multiplier"])


def finish(fig, path, note):
    fig.text(.5, .015, note, ha="center", va="bottom", fontsize=9, color="#475467")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plots(config, allrows, summaries, study, out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "axes.labelsize": 10})
    evaluation_scenes = [s for s in config["scenes"] if s["split"] == "evaluation"]
    n = summaries[0]["histories"]
    seed_count = len({s for scene in evaluation_scenes for s in scene["seeds"] if s is not None})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    for ri, factor in enumerate(("wind_speed", "half_life")):
        family = family_items(summaries, factor)
        x = [r["condition"]["multiplier"] for r in family]
        for ci, title in enumerate(("Monitoring, 0-40 s", "Forecast, +6 s", "Forecast, +12 s")):
            ax = axes[ri, ci]
            for mi, method in enumerate(METHODS):
                y = [r["JT"][mi] if ci == 0 else r["forecasts"][ci - 1][mi] for r in family]
                ax.plot(x, y, "o-", color=COLORS[method], label=LABELS[method], markersize=4)
            ax.set_title(title)
            ax.set_xlabel("True wind speed / nominal" if factor == "wind_speed" else "True half-life / nominal")
            ax.set_ylabel("Map MSE (synthetic signal squared)")
            ax.set_xticks(x)
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Can amount flexibility compensate for wrong dynamics?", fontsize=17)
    fig.tight_layout(rect=(0, .065, 1, .96))
    finish(fig, out / "reconstruction-and-forecast.png",
           f"Means of {n} evaluation combinations ({len(evaluation_scenes)} source positions x {seed_count} reused noise seeds). "
           "True input = 1 Q0. No confidence intervals.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    for ax, factor in zip(axes, ("wind_speed", "half_life")):
        for summary in family_items(summaries, factor):
            name = summary["condition"]["name"]
            x = summary["condition"]["multiplier"]
            rows = [r for r in allrows[name] if r["metadata"]["split"] == "evaluation" and r["metadata"]["seed"] is not None]
            for jitter, row in zip(np.linspace(-.025, .025, len(rows)), rows):
                d = diagnostic(row, "joint", 40.)
                low, high = d["amount_interval95"]
                ax.plot([x + jitter] * 2, [low, high], color=COLORS["joint"], alpha=.3, lw=1)
                ax.scatter(x + jitter, d["amount_mean"], s=15, color=COLORS["joint"], alpha=.5)
            ax.scatter(x, summary["amount_mean"], s=65, color=COLORS["joint"], edgecolors="white", zorder=5)
        ax.axhline(1, color="black", linestyle="--", label="True amount = 1 Q0")
        ax.set_xlabel("True wind speed / nominal" if factor == "wind_speed" else "True half-life / nominal")
        ax.set_xticks([.5, .75, 1, 1.25, 1.5])
        ax.set_ylim(.2, 1.8)
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Inferred input / Q0 at 40 s")
    fig.suptitle("Does inferred strength move when true strength does not?", fontsize=17)
    fig.tight_layout(rect=(0, .13, 1, .94))
    finish(fig, out / "inferred-amount.png",
           "Small dots/bars: each history's posterior mean / central 95% model interval. Large dots: average posterior mean.\n"
           "Bars are not confidence intervals for the ensemble mean. Uniform amount prior: [0.25, 1.75] Q0.")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharey=True)
    for ri, factor in enumerate(("wind_speed", "half_life")):
        family = family_items(summaries, factor)
        for ax, t in zip(axes[ri], (40., 52.)):
            for method in METHODS:
                counts = [next(v["source_count"] for v in r["coverage"] if v["method"] == method and v["time"] == t) for r in family]
                ax.plot([r["condition"]["multiplier"] for r in family], counts, "o-", color=COLORS[method], label=LABELS[method])
            ax.set_title("At end of monitoring" if t == 40 else "After 12 s without new readings")
            ax.set_xlabel("True wind speed / nominal" if factor == "wind_speed" else "True half-life / nominal")
            ax.set_ylabel(f"Included cases (out of {n})")
            ax.set_xticks([.5, .75, 1, 1.25, 1.5])
            ax.set_ylim(-.3, n + .3)
            ax.set_yticks(range(n + 1))
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("95% predictive intervals: membership at the source footprint", fontsize=16)
    fig.tight_layout(rect=(0, .085, 1, .95))
    finish(fig, out / "predictive-coverage.png",
           "Source-relative footprint is evaluator-only. Correlated descriptive counts; no general calibration claim.\n"
           "Global and downwind footprint diagnostics are retained separately in summary.json.")

    # Prespecified worked example: development mean control, true wind 1.5x nominal.
    name = "wind_150"
    row = next(r for r in allrows[name] if r["metadata"]["split"] == "development" and r["metadata"]["seed"] is None)
    level = row["level"]
    folder = study / name / f"source-{level[0]}-amount-{level[1]}"
    info = json.loads((folder / "results.json").read_text())
    local = info["indices"].index(row["history"])
    maps = np.load(folder / "fields.npz")["40.0"][local]
    det = np.load(study / name / "environment/deterministic.npy", mmap_mode="r")
    truth = det[round(40 / config["clock_tick_seconds"]), row["metadata"]["scene_index"]]
    fields = [truth, *maps]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    lo, hi = min(f.min() for f in fields), max(f.max() for f in fields)
    for ax, field, title in zip(axes, fields, ("Simulation truth", *[LABELS[m] for m in METHODS])):
        im = ax.imshow(field.reshape(config["grid_shape"]), origin="lower", extent=(0, 1, 0, 1),
                       vmin=lo, vmax=hi, cmap="magma", interpolation="nearest")
        y, x = row["metadata"]["source_yx"]
        ax.plot(x, y, "+", color="cyan", markersize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x / FOV")
    axes[0].set_ylabel("y / FOV")
    fig.subplots_adjust(top=.78, bottom=.25, right=.91, wspace=.25)
    cax = fig.add_axes((.93, .28, .012, .42))
    fig.colorbar(im, cax=cax, label="Synthetic signal")
    amount = diagnostic(row, "joint", 40.)["amount_mean"]
    fig.suptitle(f"Prespecified mean control: wind 1.5x; true amount 1 Q0; joint estimate {amount:.3f} Q0", fontsize=14)
    finish(fig, out / "worked-example.png",
           "40 s snapshot; identical readings across methods; shared color scale. Cross: true source, evaluation only. No random perturbations.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads((args.study / "config.json").read_text())
    verification = json.loads((args.study / "verification.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    summaries, allrows = [], {}
    for condition in config["conditions"]:
        name = condition["name"]
        rows = json.loads((args.study / name / "selected-results.json").read_text())
        allrows[name] = rows
        summaries.append(aggregate(condition, rows))
    save_json(args.output / "summary.json", dict(summaries=summaries, verification=verification))
    flat = []
    for summary in summaries:
        flat.append(dict(condition=summary["condition"]["name"], n=summary["histories"],
                         fixed_JT=summary["JT"][0], joint_JT=summary["JT"][1], correct_dynamics_JT=summary["JT"][2],
                         joint_JT_change_percent=summary["joint_JT_change_percent"],
                         joint_forecast_6_change_percent=summary["joint_forecast_change_percent"][0],
                         joint_forecast_12_change_percent=summary["joint_forecast_change_percent"][1],
                         joint_amount_mean=summary["amount_mean"],
                         wrong_amount_interval_count=summary["amount_intervals_excluding_truth"],
                         max_amount_edge_mass=summary["max_amount_edge_mass"]))
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    plots(config, allrows, summaries, args.study, args.output)
    shutil.copy2(__file__, args.output / Path(__file__).name)
    save_json(args.output / "sha256.json", {p.name: sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()})
    print(json.dumps(flat, indent=2))


if __name__ == "__main__":
    main()
