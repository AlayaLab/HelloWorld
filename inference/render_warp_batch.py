#!/usr/bin/env python
"""Batch Pi3X warp render: load Pi3X ONCE, render many (image, poses) jobs.

Companion to render_warp.py (which renders a single pair and reloads Pi3X each
call). This driver instantiates the Pi3X renderer once and loops over a JSON
job list, so a batch of N items pays the model-load cost once instead of N
times. Reuses render_warp.py's image/mp4/mask helpers verbatim.

Must run inside the warp conda env (same as render_warp.py).

Input JSON (--jobs):
    {
      "height": 352, "width": 640, "num_frames": 241, "fps": 24,
      "translation_scale": 0.1,
      "jobs": [ {"name": "...", "image": "...", "poses": ".../camera_poses.npz",
                 "out_dir": ".../warp"}, ... ]
    }

Per job writes <out_dir>/{warp.mp4, mask.npz, meta.json} (cache-aware) and
records {status, seconds, warp_mp4, mask_npz} into the --results JSON. A
per-job failure is recorded and the rest continue (exit code stays 0).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
# render_warp imports the Pi3X package + defines these helpers at module load.
from render_warp import (  # noqa: E402
    Pi3XWarpRenderer,
    Pi3XWarpRendererConfig,
    load_first_frame,
    write_mask_npz,
    write_warp_mp4,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, type=Path, help="Job-list JSON (see module docstring).")
    ap.add_argument("--results", required=True, type=Path, help="Where to write the per-job results JSON.")
    ap.add_argument("--device", default="cuda")
    # Pi3X tuning — defaults match render_warp.py.
    ap.add_argument("--invisible-fill", choices=["mean_first_frame", "black"], default="mean_first_frame")
    ap.add_argument("--mesh-depth-rtol", type=float, default=0.03)
    ap.add_argument("--mesh-normal-tol-deg", type=float, default=5.0)
    ap.add_argument("--target-fill-radius", type=int, default=1)
    args = ap.parse_args()

    spec = json.loads(args.jobs.read_text())
    jobs = spec["jobs"]
    H, W = int(spec["height"]), int(spec["width"])
    NF, FPS = int(spec["num_frames"]), int(spec["fps"])
    TS = float(spec["translation_scale"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    renderer = Pi3XWarpRenderer(Pi3XWarpRendererConfig(
        mesh_depth_rtol=float(args.mesh_depth_rtol),
        mesh_normal_tol_deg=float(args.mesh_normal_tol_deg),
        target_fill_radius=int(args.target_fill_radius),
    ))
    print(f"[render_batch] Pi3X loaded once; {len(jobs)} job(s) to render", flush=True)

    results = []
    for i, j in enumerate(jobs):
        name = j.get("name", f"job{i}")
        out_dir = Path(j["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
        out_mp4, out_mask, out_meta = out_dir / "warp.mp4", out_dir / "mask.npz", out_dir / "meta.json"
        rec = {"name": name, "out_dir": str(out_dir), "warp_mp4": str(out_mp4), "mask_npz": str(out_mask)}
        t0 = time.time()
        try:
            if out_mp4.exists() and out_mask.exists() and out_meta.exists():
                rec.update(status="ok", cached=True, seconds=0.0)
                print(f"[render_batch] [{i + 1}/{len(jobs)}] cache hit: {name}", flush=True)
                results.append(rec); continue

            image = load_first_frame(Path(j["image"]), H, W).to(device)
            poses = np.asarray(np.load(j["poses"])["camera_poses"], dtype=np.float32)
            if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
                raise ValueError(f"bad pose shape {poses.shape}, expected [T,4,4]")
            if poses.shape[0] != NF:
                raise ValueError(f"pose count {poses.shape[0]} != num_frames {NF}")

            rendered = renderer.render(
                image_tensor=image, camera_poses=poses, height=H, width=W, num_frames=NF,
                device=device, translation_scale=TS,
                invisible_fill_mode=str(args.invisible_fill), target_fill_radius=int(args.target_fill_radius),
            )
            write_warp_mp4(out_mp4, rendered["warp_video"], fps=FPS)
            write_mask_npz(out_mask, rendered["warp_visibility_mask"])
            out_meta.write_text(json.dumps({
                "image": str(j["image"]), "poses": str(j["poses"]),
                "height": H, "width": W, "num_frames": NF, "fps": FPS,
                "translation_scale": TS, "invisible_fill": str(args.invisible_fill),
                "warp_mp4": str(out_mp4), "mask_npz": str(out_mask),
                "mask_mean_visible": float(rendered["warp_visibility_mask"].mean().item()),
            }, indent=2))
            rec.update(status="ok", cached=False, seconds=round(time.time() - t0, 1))
            print(f"[render_batch] [{i + 1}/{len(jobs)}] OK {name}  {rec['seconds']}s", flush=True)
        except Exception as e:  # noqa: BLE001 — record per-job failure, keep going
            rec.update(status="fail", error=f"{type(e).__name__}: {e}", seconds=round(time.time() - t0, 1))
            print(f"[render_batch] [{i + 1}/{len(jobs)}] FAIL {name}: {rec['error']}", file=sys.stderr, flush=True)
        results.append(rec)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps({"results": results}, indent=2) + "\n")
    n_ok = sum(r["status"] == "ok" for r in results)
    print(f"[render_batch] done: {n_ok}/{len(results)} ok. results -> {args.results}", flush=True)


if __name__ == "__main__":
    main()
