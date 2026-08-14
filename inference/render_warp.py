#!/usr/bin/env python
"""Render Pi3X-based camera warp for a single (image, trajectory) pair.

Reads one input image and one camera_poses.npz (OpenCV c2w, [T, 4, 4]), runs
the Pi3X camera-warp renderer at the requested resolution, and writes:

  <out_dir>/warp.mp4       - RGB warp video, T frames @ fps, libx264 crf=10
  <out_dir>/mask.npz       - visibility mask, key="visibility", float32 [T, H, W]
  <out_dir>/meta.json      - small sidecar (shapes, fps, source paths)

The mp4 round-trip is lossy but cheap. Stage-2 of LTX-2.3 is a refinement
pass and largely smooths over VAE quantization noise; warp guidance from the
re-decoded mp4 stays well within that floor.

Must run inside the warp conda env (which provides the Pi3X renderer and CUDA
stack). Not importable from the LTX env.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

# Ensure the Pi3X warp-renderer package is importable (see README).
REPO_ROOT = Path(os.environ.get("WARP_REPO_ROOT", str(Path(__file__).resolve().parent.parent / "third_party" / "warp-as-history")))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("XFORMERS_DISABLED", "1")

from warp_as_history.camera_warp import (  # noqa: E402
    Pi3XWarpRenderer,
    Pi3XWarpRendererConfig,
    center_crop_resize_first_frame,
)


def load_first_frame(image_path: Path, height: int, width: int) -> torch.Tensor:
    """Load image, center-crop+resize to (H, W), return [1, 3, H, W] in [-1, 1]."""
    img = Image.open(image_path).convert("RGB")
    img = center_crop_resize_first_frame(img, height, width)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC in [0, 1]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
    t = t * 2.0 - 1.0  # [-1, 1]
    return t.unsqueeze(0)  # [1, 3, H, W]


def write_warp_mp4(path: Path, video: torch.Tensor, fps: int) -> None:
    """video: [1, 3, T, H, W] in [-1, 1] -> H.264 mp4 (crf=10)."""
    assert video.ndim == 5 and video.shape[0] == 1 and video.shape[1] == 3, video.shape
    arr = video[0].detach().float().cpu().clamp(-1.0, 1.0)
    arr = ((arr + 1.0) * 127.5).round().to(torch.uint8)
    arr = arr.permute(1, 2, 3, 0).numpy()  # T H W C uint8
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path),
        fps=int(fps),
        codec="libx264",
        macro_block_size=1,
        quality=None,
        ffmpeg_params=["-crf", "10", "-pix_fmt", "yuv420p"],
    ) as w:
        for frame in arr:
            w.append_data(frame)


def write_mask_npz(path: Path, mask: torch.Tensor) -> None:
    """mask: [1, 1, T, H, W] in [0, 1] -> .npz with key 'visibility' float32 [T, H, W]."""
    assert mask.ndim == 5 and mask.shape[0] == 1 and mask.shape[1] == 1, mask.shape
    m = mask[0, 0].detach().float().cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, visibility=m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path, help="Input first-frame image (jpg/png).")
    ap.add_argument("--poses", required=True, type=Path, help="camera_poses.npz from build_warp_trajectory.py.")
    ap.add_argument("--height", required=True, type=int, help="Render height (LTX stage-1 H = full_H // 2).")
    ap.add_argument("--width", required=True, type=int, help="Render width (LTX stage-1 W = full_W // 2).")
    ap.add_argument("--num_frames", required=True, type=int, help="Frames to render; must match LTX num_frames.")
    ap.add_argument("--fps", type=int, default=24, help="Output mp4 fps (must match LTX frame_rate).")
    ap.add_argument("--translation_scale", type=float, default=0.1,
                    help="Forwarded to Pi3X renderer; default 0.1 * first_frame_depth_median.")
    ap.add_argument("--invisible-fill", choices=["mean_first_frame", "black"], default="mean_first_frame",
                    help="Pi3X invisible_fill_mode. Default 'mean_first_frame' fills disocclusion regions with the "
                         "first frame's mean colour - in social scenes that's a brownish skin-tone-ish average that "
                         "VAE/LTX faithfully refines into brown grid textures. 'black' (=0 in [-1,1]) encodes to a "
                         "low-magnitude latent the model is more likely to ignore.")
    ap.add_argument("--mesh-depth-rtol", type=float, default=0.03,
                    help="Pi3X mesh_depth_rtol. Tighten (e.g. 0.01) to break mesh more aggressively at depth "
                         "discontinuities so NPC edges don't stretch as halo.")
    ap.add_argument("--mesh-normal-tol-deg", type=float, default=5.0,
                    help="Pi3X mesh_normal_tol_deg. Tighten (e.g. 2.0) to break mesh at sharper orientation jumps.")
    ap.add_argument("--target-fill-radius", type=int, default=1,
                    help="Pi3X target_fill_radius. Set 0 to disable post-splat neighbour extension (halos stop "
                         "bleeding outward 1 px).")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = args.out_dir / "warp.mp4"
    out_mask = args.out_dir / "mask.npz"
    out_meta = args.out_dir / "meta.json"

    if out_mp4.exists() and out_mask.exists() and out_meta.exists():
        print(f"[render_warp] cache hit -> {args.out_dir}", flush=True)
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    image = load_first_frame(args.image, args.height, args.width).to(device)

    poses_npz = np.load(args.poses)
    poses = np.asarray(poses_npz["camera_poses"], dtype=np.float32)  # [T, 4, 4]
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"Bad pose array shape {poses.shape}, expected [T, 4, 4].")
    if poses.shape[0] != args.num_frames:
        # build_warp_trajectory writes whatever --total we ask for; insist on a match so the
        # LTX denoiser gets a per-frame visibility mask aligned with its T_lat after VAE encode.
        raise ValueError(
            f"Pose count {poses.shape[0]} != num_frames {args.num_frames}; rebuild trajectory with --total {args.num_frames}."
        )

    renderer = Pi3XWarpRenderer(Pi3XWarpRendererConfig(
        mesh_depth_rtol=float(args.mesh_depth_rtol),
        mesh_normal_tol_deg=float(args.mesh_normal_tol_deg),
        target_fill_radius=int(args.target_fill_radius),
    ))
    rendered = renderer.render(
        image_tensor=image,
        camera_poses=poses,
        height=int(args.height),
        width=int(args.width),
        num_frames=int(args.num_frames),
        device=device,
        translation_scale=float(args.translation_scale),
        invisible_fill_mode=str(args.invisible_fill),
        target_fill_radius=int(args.target_fill_radius),
    )
    warp_video = rendered["warp_video"]            # [1, 3, T, H, W] in [-1, 1]
    warp_mask = rendered["warp_visibility_mask"]   # [1, 1, T, H, W] in [0, 1]

    write_warp_mp4(out_mp4, warp_video, fps=args.fps)
    write_mask_npz(out_mask, warp_mask)
    out_meta.write_text(
        json.dumps(
            {
                "image": str(args.image),
                "poses": str(args.poses),
                "height": int(args.height),
                "width": int(args.width),
                "num_frames": int(args.num_frames),
                "fps": int(args.fps),
                "translation_scale": float(args.translation_scale),
                "invisible_fill": str(args.invisible_fill),
                "mesh_depth_rtol": float(args.mesh_depth_rtol),
                "mesh_normal_tol_deg": float(args.mesh_normal_tol_deg),
                "target_fill_radius": int(args.target_fill_radius),
                "warp_mp4": str(out_mp4),
                "mask_npz": str(out_mask),
                "mask_mean_visible": float(warp_mask.mean().item()),
            },
            indent=2,
        )
    )
    print(f"[render_warp] wrote {out_mp4} ({warp_video.shape[2]} frames) and {out_mask}", flush=True)


if __name__ == "__main__":
    main()
