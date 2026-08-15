#!/usr/bin/env python
"""Crop + fade gate for generated LTX clips.

Crop a generated LTX clip to its first N frames (drops the end-of-clip fade
LTX appends), then reject clips that fade at EITHER end:

  - fade-OUT: last 1 s much darker than the clip body.
  - fade-IN : first 1 s much darker than the clip body — i.e. the clip opens
    on black and ramps into the scene. Under I2V the first frame was pinned by
    the conditioning image so this never happened; under pure T2V it does, and
    those clips must be dropped just like fade-outs.

Both are measured as a ratio against a ROBUST body reference (median luma over
the middle 60 % of the clip), NOT against each other. A fade-IN clip has
end/start >> 1, so an end/start ratio check passes it. Measuring each end
against the body median catches both, and crucially does NOT punish uniformly
dim scenes (night / dusk): there the body median is also dim, so both ratios
stay ~ 1 and the clip is kept.

Exit code: 0 = kept, 1 = rejected (faded at either end), 2 = error.
Prints a JSON metrics line to stdout. `ratio` = end-vs-body (fade-out) ratio.

Usage:
    crop_and_check_fade.py --src raw_361f.mp4 --dst kept_241f.mp4 \\
        --crop_frames 241 --fade_threshold 0.7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np


def measure_fade(frames: list[np.ndarray], window: int = 24) -> dict:
    """Per-end brightness vs a robust body reference.

    window=24 ~ 1 s @ 24 fps. body reference = median luma over the middle
    60 % of the clip (robust to a dark head or tail). start_ratio / end_ratio
    < threshold => that end fades.
    """
    lumas = [float(f.mean()) for f in frames]
    n = len(lumas)
    w = window if n >= 2 * window else max(1, n // 4)
    start = float(np.mean(lumas[:w]))
    end = float(np.mean(lumas[-w:]))
    lo, hi = int(n * 0.2), int(n * 0.8)
    body_slice = lumas[lo:hi] if hi > lo else lumas
    body = float(np.median(body_slice))
    ref = max(body, 1e-6)
    return {
        "start_luma": start,
        "end_luma": end,
        "body_luma": body,
        "start_ratio": start / ref,   # < thr => fade-IN (opens on black)
        "end_ratio": end / ref,       # < thr => fade-OUT (ends on black)
        "ratio": end / max(start, 1e-6),  # legacy key, logged for continuity
        "window": w,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--crop_frames", type=int, required=True)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--fade_threshold", type=float, default=0.7,
                    help="Reject if start_ratio or end_ratio < this (vs body median).")
    ap.add_argument("--no_filter", action="store_true",
                    help="Always write dst regardless of fade (still report metrics).")
    args = ap.parse_args()

    if not args.src.exists():
        print(json.dumps({"error": f"missing src {args.src}"}))
        return 2

    reader = iio.get_reader(str(args.src))
    frames: list[np.ndarray] = []
    for f in reader:
        frames.append(np.asarray(f, dtype=np.uint8))
        if len(frames) >= args.crop_frames:
            break
    reader.close()

    if len(frames) < args.crop_frames:
        print(json.dumps({
            "src": str(args.src),
            "error": f"got {len(frames)} frames < crop_frames {args.crop_frames}",
        }))
        return 2

    metrics = measure_fade(frames)
    thr = args.fade_threshold
    fade_in = metrics["start_ratio"] < thr
    fade_out = metrics["end_ratio"] < thr
    keep = args.no_filter or not (fade_in or fade_out)

    metrics["src"] = str(args.src)
    metrics["dst"] = str(args.dst) if keep else None
    metrics["kept"] = bool(keep)
    metrics["fade_in"] = bool(fade_in)
    metrics["fade_out"] = bool(fade_out)
    metrics["threshold"] = thr
    print(json.dumps(metrics))

    if not keep:
        return 1

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(
        str(args.dst), fps=int(args.fps), codec="libx264",
        macro_block_size=1, quality=None,
        ffmpeg_params=["-crf", "10", "-pix_fmt", "yuv420p"],
    )
    for f in frames:
        writer.append_data(f)
    writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
