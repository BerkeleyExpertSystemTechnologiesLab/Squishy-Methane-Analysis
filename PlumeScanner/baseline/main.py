# baseline/main.py
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .grid import Grid
from .motion import MotionModel
from .policy.raster import RasterBaselinePolicy, RasterConfig
from .env.synthetic import SyntheticPlumeEnv
from .runner import run_baseline
from .traceio import write_jsonl


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plume_scanner Phase-1 baseline",
        description="Uniform raster baseline over a 2D discretized grid (x wraps, y bounded).",
    )

    # Grid
    p.add_argument("--w", type=int, default=120, help="Grid width (wrap dimension, x).")
    p.add_argument("--h", type=int, default=30, help="Grid height (bounded dimension, y).")

    # Policy
    p.add_argument("--dwell", type=float, default=0.12, help="Dwell seconds per cell.")
    p.add_argument("--start_x", type=int, default=0, help="Start x index.")
    p.add_argument("--start_y", type=int, default=0, help="Start y index.")

    # Motion model
    p.add_argument("--x_rate", type=float, default=45.0, help="x slew rate in cells/s.")
    p.add_argument("--y_rate", type=float, default=25.0, help="y slew rate in cells/s.")
    p.add_argument("--settle", type=float, default=0.02, help="Settle time (s) added per move.")
    p.add_argument(
        "--motion_mode",
        type=str,
        default="max",
        choices=["max", "sum"],
        help="Move-time composition: max(axis) or sum(axis).",
    )

    # Runner / evaluation
    p.add_argument("--max_time", type=float, default=120.0, help="Max simulated time (s).")
    p.add_argument("--detect_thr", type=float, default=0.55, help="Detection threshold on measurement.")
    p.add_argument("--detect_k", type=int, default=2, help="Consecutive hits for detection.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--trace",
        type=str,
        default="",
        help="Optional: path to write trace .jsonl (e.g., trace.jsonl).",
    )

    # Env selection (right now only synthetic, but this makes it easy to extend)
    p.add_argument("--env", type=str, default="synthetic", choices=["synthetic"], help="Environment type.")

    # Synthetic env params (so you can stress-test baseline)
    p.add_argument("--amp", type=float, default=1.0, help="Synthetic plume amplitude.")
    p.add_argument("--noise", type=float, default=0.08, help="Synthetic noise std.")
    p.add_argument("--sigma_x", type=float, default=6.0, help="Synthetic plume sigma in x (cells).")
    p.add_argument("--sigma_y", type=float, default=2.5, help="Synthetic plume sigma in y (cells).")
    p.add_argument("--drift_x", type=float, default=0.10, help="Synthetic drift in x (cells/s).")
    p.add_argument("--drift_y", type=float, default=0.00, help="Synthetic drift in y (cells/s).")
    p.add_argument("--bg", type=float, default=0.02, help="Synthetic background offset.")
    p.add_argument("--gt_thr_frac", type=float, default=0.25, help="GT plume if true_val > frac*amp.")

    return p


def make_env(args, grid: Grid):
    if args.env == "synthetic":
        return SyntheticPlumeEnv(
            grid=grid,
            amp=args.amp,
            noise_std=args.noise,
            sigma_x_cells=args.sigma_x,
            sigma_y_cells=args.sigma_y,
            drift_x_cells_per_s=args.drift_x,
            drift_y_cells_per_s=args.drift_y,
            background=args.bg,
            gt_threshold_frac=args.gt_thr_frac,
        )
    raise ValueError(f"Unknown env: {args.env}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # --- Construct components ---
    grid = Grid(width=args.w, height=args.h, wrap_x=True)

    motion = MotionModel(
        x_rate_cells_per_s=args.x_rate,
        y_rate_cells_per_s=args.y_rate,
        settle_s=args.settle,
        mode=args.motion_mode,
    )

    env = make_env(args, grid)

    policy = RasterBaselinePolicy(
        grid=grid,
        cfg=RasterConfig(dwell_s=args.dwell, start=(args.start_x, args.start_y)),
    )

    # --- Run ---
    metrics, trace = run_baseline(
        env=env,
        grid=grid,
        motion=motion,
        policy=policy,
        max_time_s=args.max_time,
        detect_threshold=args.detect_thr,
        detect_consecutive=args.detect_k,
        seed=args.seed,
        record_trace=bool(args.trace),
    )
    if args.trace and trace is not None:
    # Put a meta record at the top of trace for viz (synthetic GT overlay)
        meta = {
            "env": args.env,
            "grid_w": args.w,
            "grid_h": args.h,
            "seed": args.seed,

            # synthetic params (needed to reconstruct full plume)
            "amp": args.amp,
            "noise_std": args.noise,
            "sigma_x_cells": args.sigma_x,
            "sigma_y_cells": args.sigma_y,
            "drift_x_cells_per_s": args.drift_x,
            "drift_y_cells_per_s": args.drift_y,
            "background": args.bg,
            "gt_threshold_frac": args.gt_thr_frac,

            # initial center after reset (SyntheticPlumeEnv keeps this attr)
            "center0": list(getattr(env, "center0", (None, None))),
        }
    trace.insert(0, {"_meta": meta})


    # --- Save trace if requested ---
    if args.trace:
        trace_path = Path(args.trace)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        if trace is None:
            raise RuntimeError("record_trace requested but trace is None (unexpected).")
        write_jsonl(str(trace_path), trace)
        print(f"[ok] saved trace: {trace_path.resolve()}  (rows={len(trace)})")

    # --- Print summary ---
    print("\n=== Phase 1 Baseline: Uniform Raster (2D grid, x wraps) ===")
    print(f"grid:          {grid.width} x {grid.height}  (wrap_x={grid.wrap_x})")
    print(f"policy:        dwell={args.dwell:.3f}s start=({args.start_x},{args.start_y})")
    print(f"motion:        mode={args.motion_mode} x_rate={args.x_rate} y_rate={args.y_rate} settle={args.settle}")
    print(f"env:           {args.env}")
    if args.env == "synthetic":
        print(
            f"synthetic:     amp={args.amp} noise={args.noise} sigma_x={args.sigma_x} sigma_y={args.sigma_y} "
            f"drift_x={args.drift_x} drift_y={args.drift_y} bg={args.bg}"
        )
    print("\n--- metrics ---")
    print(f"detection_time_s: {metrics.detection_time_s}")
    print(f"full_scan_time_s: {metrics.full_scan_time_s}")
    print(f"total_time_s:     {metrics.total_time_s:.3f}")
    print(f"total_motion_s:   {metrics.total_motion_s:.3f}")
    print(f"total_dwell_s:    {metrics.total_dwell_s:.3f}")
    print(f"steps:            {metrics.steps}")
    print()


if __name__ == "__main__":
    main()
