# baseline/runner.py
from __future__ import annotations
from typing import Optional, List, Dict, Tuple
from .grid import Grid, Cell
from .motion import MotionModel
from .metrics import Metrics
from .env.base import ScanEnv
from .policy.raster import RasterBaselinePolicy


def run_baseline(
    env: ScanEnv,
    grid: Grid,
    motion: MotionModel,
    policy: RasterBaselinePolicy,
    *,
    max_time_s: float = 120.0,
    detect_threshold: float = 0.55,
    detect_consecutive: int = 2,
    seed: int = 0,
    record_trace: bool = False,
) -> Tuple[Metrics, Optional[List[Dict]]]:
    env.reset(seed=seed)

    visited = [[False] * grid.width for _ in range(grid.height)]
    visited_count = 0
    total_cells = grid.width * grid.height

    t = 0.0
    total_motion = 0.0
    total_dwell = 0.0
    steps = 0

    detection_time: Optional[float] = None
    full_scan_time: Optional[float] = None
    above = 0

    trace: Optional[List[Dict]] = [] if record_trace else None

    actions = policy.actions()

    # Initialize previous cell as first target to avoid a big artificial first move
    first_cell, _ = next(actions)
    prev: Cell = grid.normalize(first_cell)

    # Include the first action
    pending_first = True
    first_pair = (prev, policy.cfg.dwell_s)

    while t < max_time_s:
        if pending_first:
            cell, dwell_s = first_pair
            pending_first = False
        else:
            cell, dwell_s = next(actions)
            cell = grid.normalize(cell)

        # Motion
        move_t = motion.move_time(grid, prev, cell)
        t_after_move = t + move_t

        total_motion += move_t

        # Measurement + dwell
        meas, info = env.step(cell, dwell_s, t_after_move)
        t_after_dwell = t_after_move + dwell_s

        total_dwell += dwell_s

        # Full scan coverage
        x, y_idx = cell
        if not visited[y_idx][x]:
            visited[y_idx][x] = True
            visited_count += 1
            if visited_count == total_cells and full_scan_time is None:
                full_scan_time = t_after_dwell

        # Detection (simple threshold)
        if meas >= detect_threshold:
            above += 1
        else:
            above = 0
        if detection_time is None and above >= detect_consecutive:
            detection_time = t_after_dwell

        # Record trace row
        if trace is not None:
            trace.append(
                {
                    "step": steps,
                    "t_s": t_after_dwell,
                    "x": int(x),
                    "y": int(y_idx),
                    "move_t": float(move_t),
                    "dwell_s": float(dwell_s),
                    "meas": float(meas),
                    # optional GT if env provides it
                    "true_val": float(info["true_val"]) if "true_val" in info else None,
                    "gt_is_plume": bool(info["gt_is_plume"]) if "gt_is_plume" in info else None,
                }
            )

        # Commit time + step
        t = t_after_dwell
        prev = cell
        steps += 1

        # Phase 1: often stop when full scan is complete (and we detected)
        if full_scan_time is not None and detection_time is not None:
            break

    metrics = Metrics(
        detection_time_s=detection_time,
        full_scan_time_s=full_scan_time,
        total_time_s=t,
        total_motion_s=total_motion,
        total_dwell_s=total_dwell,
        steps=steps,
    )
    return metrics, trace
