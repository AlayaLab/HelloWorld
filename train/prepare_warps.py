#!/usr/bin/env python
"""Warp-prep: turn generated T2V clips into (gt, warp, mask) training triples.

Per clip:

    decode .mp4
      -> temporal linear-resample to (clip_frames @ fps)
      -> spatial center-crop+resize to height x width
      -> Pi3X estimate camera poses over the WHOLE clip
      -> render warp from frame-0 along those estimated poses
      -> save {gt.mp4, warp.mp4, mask.npz, meta.json}

Inputs come from CLI only (no hardcoded experiment paths):
  --raw-dir      directory of generated `<scene_id>__<family>.mp4` clips
  --prompts-tsv  the 5-column TSV from build_prompts.py
                 (scene_id, family, gen_prompt, train_prompt, expected_count)
  --out-root     where to write clip_XXXX_.../ + manifest.tsv

The train_prompt (col 4) is looked up per (scene_id, family) and written as the
per-clip `prompt` in meta.json — it's camera-neutral, so the LoRA reads motion
from the warp, not the text. gen_prompt (col 3) is intentionally ignored here.

--slice-num / --slice-total split work across GPUs (each worker handles items
where (i % slice_total) == slice_num); per-slice manifests are merged by the
launcher (run_train.sh).

Runs in the warp conda env (the Pi3X renderer + CUDA stack). Pi3X is GPU-heavy
and loaded/freed per clip.

Usage:
    # Single-GPU:
    python prepare_warps.py --raw-dir data/raw --prompts-tsv prompts.tsv \\
        --out-root data/prepared

    # Dual-GPU (split across GPU 0 and 1):
    CUDA_VISIBLE_DEVICES=0 python prepare_warps.py --raw-dir data/raw \\
        --prompts-tsv prompts.tsv --out-root data/prepared --slice-num 0 --slice-total 2 &
    CUDA_VISIBLE_DEVICES=1 python prepare_warps.py --raw-dir data/raw \\
        --prompts-tsv prompts.tsv --out-root data/prepared --slice-num 1 --slice-total 2 &
    wait
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

# Ensure the Pi3X warp-renderer package is importable (see helloworld README).
REPO_ROOT = Path(os.environ.get(
    "WARP_REPO_ROOT",
    str(Path(__file__).resolve().parent.parent / "third_party" / "warp-as-history"),
))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("XFORMERS_DISABLED", "1")

from warp_as_history.camera_warp import (  # noqa: E402
    Pi3XWarpRenderer,
    Pi3XWarpRendererConfig,
    center_crop_resize_first_frame,
    se3_inverse,
)

# Constant generator tag kept in clip_dir names / manifest for compatibility
# with the trainer's manifest schema (its `generator` column is optional).
GENERATOR_TAG = "ltx"


# ---------------------------------------------------------------- I/O helpers
def load_and_resample_clip(
    video_path: Path,
    target_frames: int,
    target_fps: int,
    target_h: int,
    target_w: int,
) -> list[Image.Image]:
    """Decode video, temporally linear-resample to (target_frames @ target_fps)
    over the source's playback duration, then center-crop+resize each frame to
    target_h x target_w."""
    reader = imageio.get_reader(str(video_path))
    src_meta = reader.get_meta_data()
    src_fps = float(src_meta["fps"])
    src_frames_np: list[np.ndarray] = []
    for arr in reader:
        src_frames_np.append(np.asarray(arr, dtype=np.uint8))
    reader.close()
    src_T = len(src_frames_np)
    if src_T == 0:
        raise ValueError(f"empty video: {video_path}")

    src_duration = (src_T - 1) / src_fps
    target_t = np.linspace(0.0, src_duration, target_frames, dtype=np.float64)
    src_t = np.arange(src_T, dtype=np.float64) / src_fps

    out: list[Image.Image] = []
    for t in target_t:
        idx_right = int(np.clip(np.searchsorted(src_t, t, side="right"), 1, src_T - 1))
        idx_left = idx_right - 1
        denom = src_t[idx_right] - src_t[idx_left]
        alpha = 0.0 if denom <= 0 else float((t - src_t[idx_left]) / denom)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if alpha == 0.0:
            f = src_frames_np[idx_left]
        elif alpha == 1.0:
            f = src_frames_np[idx_right]
        else:
            f = ((1.0 - alpha) * src_frames_np[idx_left].astype(np.float32)
                 + alpha * src_frames_np[idx_right].astype(np.float32))
            f = np.clip(f, 0.0, 255.0).astype(np.uint8)
        pil = Image.fromarray(f).convert("RGB")
        pil = center_crop_resize_first_frame(pil, int(target_h), int(target_w))
        out.append(pil)
    return out


def pil_to_minus1_1(pil: Image.Image) -> torch.Tensor:
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return (t * 2.0 - 1.0).unsqueeze(0)


def gt_clip_to_tensor(frames: list[Image.Image]) -> torch.Tensor:
    per_frame = [pil_to_minus1_1(f) for f in frames]
    return torch.stack([t[0] for t in per_frame], dim=1).unsqueeze(0)


def write_mp4(path: Path, video_minus1_1: torch.Tensor, fps: int) -> None:
    assert video_minus1_1.ndim == 5 and video_minus1_1.shape[:2] == (1, 3), video_minus1_1.shape
    arr = video_minus1_1[0].detach().float().cpu().clamp(-1.0, 1.0)
    arr = ((arr + 1.0) * 127.5).round().to(torch.uint8)
    arr = arr.permute(1, 2, 3, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path), fps=int(fps), codec="libx264",
        macro_block_size=1, quality=None,
        ffmpeg_params=["-crf", "10", "-pix_fmt", "yuv420p"],
    ) as w:
        for frame in arr:
            w.append_data(frame)


def write_mask_npz(path: Path, mask: torch.Tensor) -> None:
    assert mask.ndim == 5 and mask.shape[:2] == (1, 1), mask.shape
    m = mask[0, 0].detach().float().cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, visibility=m)


# ---------------------------------------------------------------- Pose math
def relative_poses_from_geometry(geometry: dict, source_idx: int,
                                  target_indices: list[int]) -> np.ndarray:
    keyframe_geoms = geometry["keyframe_geometries"]
    source_pose = np.asarray(keyframe_geoms[int(source_idx)]["source_pose"], dtype=np.float32)
    target_world = np.stack(
        [np.asarray(keyframe_geoms[int(i)]["source_pose"], dtype=np.float32) for i in target_indices],
        axis=0,
    )
    source_inv = se3_inverse(source_pose[None])[0]
    return np.einsum("ij,tjk->tik", source_inv.astype(np.float32, copy=False), target_world).astype(np.float32)


# ---------------------------------------------------------------- Per-clip
def render_clip_warp(
    renderer: Pi3XWarpRenderer,
    geometry: dict,
    source_idx: int,
    target_indices: list[int],
    height: int,
    width: int,
    device: torch.device,
) -> dict:
    sub_geom = dict(geometry)
    sub_keyframes = [geometry["keyframe_geometries"][int(source_idx)]]
    latest = sub_keyframes[-1]
    sub_geom["keyframe_geometries"] = sub_keyframes
    sub_geom["intrinsic"] = latest["intrinsic"]
    sub_geom["keyframe_count"] = 1
    sub_geom["preserve_pi3x_keyframe_points"] = True
    sub_geom["render_height"] = latest["render_height"]
    sub_geom["render_width"] = latest["render_width"]
    sub_geom["source_pose"] = latest["source_pose"]
    sub_geom["source_rgb_u8"] = latest["source_rgb_u8"]

    relative = relative_poses_from_geometry(geometry, source_idx, target_indices)
    return renderer.render_from_geometry(
        sub_geom, relative,
        height=int(height), width=int(width), device=device,
        invisible_fill_mode="mean_first_frame",
    )


# ---------------------------------------------------------------- Source enumeration
def parse_clip_name(name: str) -> tuple[str, str] | None:
    """'<scene_id>__<family>.mp4' -> (scene_id, family). Returns None on mismatch.

    scene_id is anything before '__'; family is anything (non-empty) after.
    """
    if not name.endswith(".mp4"):
        return None
    stem = name[:-4]
    if "__" not in stem:
        return None
    scene_id, _, family = stem.partition("__")
    if not scene_id or not family:
        return None
    return scene_id, family


def load_prompts(prompts_tsv: Path) -> dict[tuple[str, str], str]:
    """(scene_id, family) -> train_prompt (5-column TSV: scene_id, family,
    gen_prompt, train_prompt, expected_count). The train_prompt is
    camera-neutral — it's what the LoRA sees as text conditioning. gen_prompt
    (col 3) is only used during LTX T2V data generation and is ignored here."""
    out: dict[tuple[str, str], str] = {}
    for i, line in enumerate(prompts_tsv.read_text().splitlines()):
        if not line.strip():
            continue
        cells = line.split("\t")
        if i == 0:
            # header: scene_id, family, gen_prompt, train_prompt, expected_count
            assert cells[:4] == ["scene_id", "family", "gen_prompt", "train_prompt"], \
                f"Unexpected header: {cells}"
            continue
        if len(cells) < 4:
            continue
        out[(cells[0], cells[1])] = cells[3]   # use train_prompt (col 4): appearance+scene+action, camera-neutral
    return out


# ---------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, type=Path,
                    help="Directory of generated <scene_id>__<family>.mp4 clips.")
    ap.add_argument("--prompts-tsv", required=True, type=Path,
                    help="5-column TSV from build_prompts.py.")
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--height", type=int, default=352)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--clip-frames", type=int, default=241,
                    help="Output frame count (must be (8k+1) for LTX). Default 241 = 10 s @ 24 fps.")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--test-ids", nargs="*", default=[],
                    help="scene_ids to RESERVE for eval (excluded from prepared output). Default: none.")
    ap.add_argument("--slice-num", type=int, default=0,
                    help="This worker handles items where (i %% slice_total) == slice_num.")
    ap.add_argument("--slice-total", type=int, default=1,
                    help="Total worker count. 1 = single-GPU run.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.clip_frames % 8 != 1:
        raise SystemExit(f"--clip-frames must be (8k+1) for LTX, got {args.clip_frames}")
    if not args.prompts_tsv.exists():
        raise SystemExit(f"missing {args.prompts_tsv}")
    if not args.raw_dir.exists():
        raise SystemExit(f"missing --raw-dir {args.raw_dir}")
    if not (0 <= args.slice_num < args.slice_total):
        raise SystemExit(f"--slice-num {args.slice_num} not in [0, {args.slice_total})")

    args.out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    prompts = load_prompts(args.prompts_tsv)
    test_ids = set(args.test_ids)
    print(f"[prep] {len(prompts)} prompts; test_ids={sorted(test_ids)}; "
          f"slice={args.slice_num}/{args.slice_total}", flush=True)

    # Enumerate items (sorted by filename for deterministic slicing).
    items: list[dict] = []
    for p in sorted(args.raw_dir.glob("*.mp4")):
        parsed = parse_clip_name(p.name)
        if parsed is None:
            continue
        scene_id, family = parsed
        if scene_id in test_ids:
            continue
        prompt = prompts.get((scene_id, family))
        if prompt is None:
            print(f"[prep] WARN: no prompt for ({scene_id}, {family}); skipping {p.name}", flush=True)
            continue
        items.append({
            "generator": GENERATOR_TAG,
            "source": p,
            "image_id": scene_id,
            "family": family,
            "prompt": prompt,
        })

    if not items:
        raise SystemExit("[prep] no training clips matched. Check --raw-dir / --prompts-tsv / --test-ids.")

    # Apply slicing AFTER sorting -> each (slice_num, slice_total) deterministic.
    my_items = [it for i, it in enumerate(items) if (i % args.slice_total) == args.slice_num]
    print(f"[prep] {len(items)} total clips; this worker handles {len(my_items)}.", flush=True)

    renderer = Pi3XWarpRenderer(Pi3XWarpRendererConfig())

    # Per-worker manifest to avoid races; merge after both workers finish.
    if args.slice_total > 1:
        manifest_path = args.out_root / f"manifest.slice{args.slice_num}of{args.slice_total}.tsv"
    else:
        manifest_path = args.out_root / "manifest.tsv"
    manifest_rows = [
        "clip_id\tclip_dir\tgenerator\timage_id\tfamily\theight\twidth\tnum_frames\tfps\tsource\tprompt"
    ]
    target_indices = list(range(args.clip_frames))

    for local_idx, it in enumerate(my_items):
        # Use full-set global index for clip_id so worker outputs don't collide.
        clip_id = items.index(it)
        clip_dir = args.out_root / f"clip_{clip_id:04d}_{it['generator']}_{it['image_id']}_{it['family']}"
        gt_path = clip_dir / "gt.mp4"
        warp_path = clip_dir / "warp.mp4"
        mask_path = clip_dir / "mask.npz"
        meta_path = clip_dir / "meta.json"

        if (not args.overwrite
            and gt_path.exists() and warp_path.exists()
            and mask_path.exists() and meta_path.exists()):
            print(f"[prep] skip {clip_id:04d} {it['image_id']} {it['family']} (cached)",
                  flush=True)
        else:
            clip_dir.mkdir(parents=True, exist_ok=True)
            print(f"[prep] {local_idx + 1}/{len(my_items)} (global {clip_id:04d}) {it['image_id']} {it['family']} <- {it['source'].name}",
                  flush=True)

            frames = load_and_resample_clip(
                it["source"], target_frames=args.clip_frames, target_fps=args.fps,
                target_h=args.height, target_w=args.width,
            )
            assert len(frames) == args.clip_frames, f"got {len(frames)} frames"

            gt_tensor = gt_clip_to_tensor(frames)
            write_mp4(gt_path, gt_tensor, fps=args.fps)

            image_tensors = [pil_to_minus1_1(f).to(device) for f in frames]
            try:
                geometry = renderer.estimate_keyframe_geometry(image_tensors, device=device)
            except Exception as exc:
                print(f"[prep]   PI3X FAIL: {exc}; skipping", flush=True)
                if gt_path.exists():
                    gt_path.unlink()
                continue

            try:
                rendered = render_clip_warp(
                    renderer, geometry, source_idx=0, target_indices=target_indices,
                    height=args.height, width=args.width, device=device,
                )
            except Exception as exc:
                print(f"[prep]   RENDER FAIL: {exc}; skipping", flush=True)
                if gt_path.exists():
                    gt_path.unlink()
                continue
            warp_video = rendered["warp_video"]
            warp_mask = rendered["warp_visibility_mask"]
            if warp_video.shape[2] != args.clip_frames:
                print(f"[prep]   bad warp T={warp_video.shape[2]}, skipping", flush=True)
                if gt_path.exists():
                    gt_path.unlink()
                continue

            write_mp4(warp_path, warp_video, fps=args.fps)
            write_mask_npz(mask_path, warp_mask)
            meta_path.write_text(json.dumps({
                "clip_id": int(clip_id),
                "generator": it["generator"],
                "image_id": it["image_id"],
                "family": it["family"],
                "prompt": it["prompt"],
                "source": str(it["source"]),
                "height": int(args.height),
                "width": int(args.width),
                "num_frames": int(args.clip_frames),
                "fps": int(args.fps),
                "mask_mean_visible": float(warp_mask.mean().item()),
                "gt": str(gt_path),
                "warp": str(warp_path),
                "mask": str(mask_path),
            }, indent=2))

            renderer._pi3x_runtime = None
            torch.cuda.empty_cache()

        manifest_rows.append("\t".join(map(str, [
            clip_id, clip_dir, it["generator"], it["image_id"], it["family"],
            args.height, args.width, args.clip_frames, args.fps, it["source"],
            it["prompt"].replace("\t", " "),
        ])))

    manifest_path.write_text("\n".join(manifest_rows) + "\n")
    print(f"[prep] wrote {len(my_items)} clip rows -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
