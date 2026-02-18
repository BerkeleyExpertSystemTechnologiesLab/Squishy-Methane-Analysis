# baseline/viz.py
from __future__ import annotations
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def _wrap_dx(x: np.ndarray, cx: int, w: int) -> np.ndarray:
    dx = (x - cx) % w
    dx = np.where(dx > (w // 2), dx - w, dx)  # shortest signed wrap distance
    return dx


def synth_truth_field(meta: Dict, t_s: float) -> np.ndarray:
    """Reconstruct full synthetic plume field at time t_s from meta."""
    w = int(meta["grid_w"])
    h = int(meta["grid_h"])

    amp = float(meta["amp"])
    sigx = float(meta["sigma_x_cells"])
    sigy = float(meta["sigma_y_cells"])
    drift_x = float(meta["drift_x_cells_per_s"])
    drift_y = float(meta["drift_y_cells_per_s"])
    bg = float(meta["background"])

    c0x, c0y = meta["center0"]
    c0x = int(c0x)
    c0y = int(c0y)

    cx = (c0x + drift_x * t_s) % w
    cy = np.clip(c0y + drift_y * t_s, 0.0, h - 1.0)

    cx_i = int(np.round(cx)) % w

    xs = np.arange(w)[None, :]          # shape (1, w)
    ys = np.arange(h)[:, None]          # shape (h, 1)

    dx = _wrap_dx(xs, cx_i, w)          # shape (1, w) broadcast
    dy = ys - cy                        # shape (h, 1) broadcast

    val = amp * np.exp(-0.5 * ((dx / sigx) ** 2 + (dy / sigy) ** 2))
    return (bg + val).astype(np.float32)


def animate_trace(
    trace: List[Dict],
    grid_w: int,
    grid_h: int,
    *,
    interval_ms: int = 30,
    max_frames: Optional[int] = None,
    view: str = "dual",          # "counts", "meas", "dual", "dual_truth"
    meas_ema_alpha: float = 0.35
) -> None:
    if not trace:
        raise ValueError("Empty trace.")

    # Pull meta record if present
    meta = None
    if isinstance(trace[0], dict) and "_meta" in trace[0]:
        meta = trace[0]["_meta"]
        trace = trace[1:]

    n = len(trace) if max_frames is None else min(len(trace), max_frames)

    counts = np.zeros((grid_h, grid_w), dtype=np.int32)
    meas_map = np.full((grid_h, grid_w), np.nan, dtype=np.float32)

    # --- Layout ---
    if view in ("dual", "dual_truth"):
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, axL = plt.subplots(1, 1, figsize=(6, 5))
        axR = None

    # --- Left panel ---
    im_left = None
    im_vis = None
    ptL = None

    if view == "counts":
        im_left = axL.imshow(counts, origin="lower", aspect="auto")
        axL.set_title("Visit counts")
        (ptL,) = axL.plot([], [], marker="o", linestyle="")
    elif view == "meas":
        im_left = axL.imshow(np.ma.masked_invalid(meas_map), origin="lower", aspect="auto")
        axL.set_title("Measured signal (EMA)")
        (ptL,) = axL.plot([], [], marker="o", linestyle="")
    elif view == "dual":
        im_left = axL.imshow((counts > 0).astype(int), origin="lower", aspect="auto", vmin=0, vmax=1)
        axL.set_title("Visited mask")
        (ptL,) = axL.plot([], [], marker="o", linestyle="")
    elif view == "dual_truth":
        if meta is None or meta.get("env") != "synthetic":
            raise ValueError("dual_truth requires synthetic trace with a _meta row (regen trace with updated main.py).")
        # Ground truth plume field
        truth0 = synth_truth_field(meta, t_s=0.0)
        im_left = axL.imshow(truth0, origin="lower", aspect="auto")
        axL.set_title("Ground truth plume (synthetic)")
        # Overlay visited mask as semi-transparent gray
        im_vis = axL.imshow(
            (counts > 0).astype(int),
            origin="lower",
            aspect="auto",
            cmap="gray",
            alpha=0.25,
            vmin=0,
            vmax=1,
        )
        (ptL,) = axL.plot([], [], marker="o", linestyle="")
        # Fix truth scale so it doesn’t wash out
        bg = float(meta["background"])
        amp = float(meta["amp"])
        im_left.set_clim(bg, bg + amp)
    else:
        raise ValueError("Invalid view.")

    axL.set_xlabel("x (wrap)")
    axL.set_ylabel("y (bounded)")

    # --- Right panel (dual only) ---
    im_meas = None
    ptR = None
    if view in ("dual", "dual_truth") and axR is not None:
        im_meas = axR.imshow(np.ma.masked_invalid(meas_map), origin="lower", aspect="auto")
        axR.set_title("Measured signal (EMA)")
        axR.set_xlabel("x (wrap)")
        axR.set_ylabel("y (bounded)")
        (ptR,) = axR.plot([], [], marker="o", linestyle="")

    def init():
        artists = []
        if im_left is not None:
            artists.append(im_left)
        if im_vis is not None:
            artists.append(im_vis)
        if ptL is not None:
            ptL.set_data([], [])
            artists.append(ptL)
        if im_meas is not None:
            im_meas.set_data(np.ma.masked_invalid(meas_map))
            artists.append(im_meas)
        if ptR is not None:
            ptR.set_data([], [])
            artists.append(ptR)
        return tuple(artists)

    def update(frame: int):
        row = trace[frame]
        x = int(row["x"])
        y = int(row["y"])
        meas = float(row["meas"])
        t_s = float(row["t_s"])

        counts[y, x] += 1

        # Update left
        if view == "counts":
            im_left.set_data(counts)
        elif view == "meas":
            # update meas_map
            if np.isnan(meas_map[y, x]):
                meas_map[y, x] = meas
            else:
                meas_map[y, x] = (1.0 - meas_ema_alpha) * meas_map[y, x] + meas_ema_alpha * meas
            im_left.set_data(np.ma.masked_invalid(meas_map))
        elif view == "dual":
            im_left.set_data((counts > 0).astype(int))
        elif view == "dual_truth":
            truth = synth_truth_field(meta, t_s=t_s)
            im_left.set_data(truth)
            im_vis.set_data((counts > 0).astype(int))

        ptL.set_data([x], [y])

        # Update right (meas)
        if im_meas is not None and ptR is not None:
            if np.isnan(meas_map[y, x]):
                meas_map[y, x] = meas
            else:
                meas_map[y, x] = (1.0 - meas_ema_alpha) * meas_map[y, x] + meas_ema_alpha * meas
            im_meas.set_data(np.ma.masked_invalid(meas_map))
            ptR.set_data([x], [y])

            vmin = np.nanmin(meas_map)
            vmax = np.nanmax(meas_map)
            if np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax:
                im_meas.set_clim(vmin, vmax)

        fig.suptitle(f"step={frame}  t={t_s:.2f}s  cell=({x},{y})  meas={meas:.3f}")
        artists = [im_left, ptL]
        if im_vis is not None:
            artists.append(im_vis)
        if im_meas is not None:
            artists.append(im_meas)
        if ptR is not None:
            artists.append(ptR)
        return tuple(a for a in artists if a is not None)

    anim = FuncAnimation(fig, update, frames=n, init_func=init, interval=interval_ms, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()


def main():
    p = argparse.ArgumentParser("Visualize scan trace (jsonl)")
    p.add_argument("--trace", type=str, required=True, help="Path to trace.jsonl")
    p.add_argument("--w", type=int, required=True, help="grid width")
    p.add_argument("--h", type=int, required=True, help="grid height")
    p.add_argument("--interval", type=int, default=30, help="ms between frames")
    p.add_argument("--max_frames", type=int, default=0, help="0=all")
    p.add_argument("--view", type=str, default="dual", choices=["counts", "meas", "dual", "dual_truth"])
    p.add_argument("--ema_alpha", type=float, default=0.35, help="EMA alpha for meas view")
    args = p.parse_args()

    from .traceio import read_jsonl
    trace = read_jsonl(args.trace)

    max_frames = None if args.max_frames == 0 else args.max_frames
    animate_trace(
        trace,
        args.w,
        args.h,
        interval_ms=args.interval,
        max_frames=max_frames,
        view=args.view,
        meas_ema_alpha=args.ema_alpha,
    )


if __name__ == "__main__":
    main()
