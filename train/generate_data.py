#!/usr/bin/env python
"""Batched T2V generator — ONE long-lived process per GPU.

Replaces the naive "one subprocess per clip per retry" loop, which reloaded
the 22B LTX checkpoint AND the Faster R-CNN detector on every single attempt
(most of the per-clip wall time was load, not denoise). Here we:

  * build DistilledPipeline ONCE, with a StateDictRegistry so the checkpoint
    state-dicts cache in RAM (the framework still moves weights to the `meta`
    device between calls to free VRAM, but they're rebuilt from the cached
    state-dict — no disk re-read);
  * load the person detector ONCE (resident);
  * loop over this process's slice of rows, running the seed-retry loop with
    both quality gates (fade + people-count) as in-process function calls.

Launched once per GPU by run_train.sh (CUDA_VISIBLE_DEVICES pins the device,
so this process always sees cuda:0). Each process writes its own manifest
shard; the launcher merges them.

Gates (in-process):
  1. fade — crop GEN_FRAMES->FRAME_NUM, reject fade-in/out  (crop_and_check_fade.measure_fade)
  2. people — first/mid/last frame main-person count == expected (count_people_filter.count_main_people)
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import imageio.v2 as iio
import torch

# in-process reuse of the two gate implementations (same dir)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crop_and_check_fade as cf       # measure_fade
import count_people_filter as cp       # count_main_people + _model

from ltx_core.loader.registry import StateDictRegistry
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import OffloadMode


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def write_clip(frames, dst, fps):
    dst.parent.mkdir(parents=True, exist_ok=True)
    w = iio.get_writer(str(dst), fps=int(fps), codec="libx264",
                       macro_block_size=1, quality=None,
                       ffmpeg_params=["-crf", "10", "-pix_fmt", "yuv420p"])
    for f in frames:
        w.append_data(f)
    w.close()


@torch.inference_mode()   # CRITICAL for a long-lived process: without it,
def main():                # autograd state accumulates across calls -> VRAM OOM.
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-tsv", required=True, type=Path)
    ap.add_argument("--row-start", type=int, required=True)  # 1-based incl header=1
    ap.add_argument("--row-end", type=int, required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--tmp-dir", required=True, type=Path)
    ap.add_argument("--manifest-shard", required=True, type=Path)
    ap.add_argument("--gpu-label", default="0")
    ap.add_argument("--distilled", required=True)
    ap.add_argument("--upscaler", required=True)
    ap.add_argument("--gemma", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--gen-frames", type=int, default=361)
    ap.add_argument("--frame-num", type=int, default=241)
    ap.add_argument("--frame-rate", type=int, default=24)
    ap.add_argument("--max-seeds", type=int, default=4)
    ap.add_argument("--person-score", type=float, default=0.9)
    ap.add_argument("--person-min-area", type=float, default=0.015)
    ap.add_argument("--fade-threshold", type=float, default=0.7)
    ap.add_argument("--positive-nudge", default=" Well-lit throughout, bright final frame, no fade out.")
    ap.add_argument("--no-fade-gate", action="store_true",
                    help="disable the fade filter (keep clips that fade in/out)")
    ap.add_argument("--no-cast-gate", action="store_true",
                    help="disable the subject/people-count filter")
    ap.add_argument("--offload-mode", choices=[m.value for m in OffloadMode], default="none",
                    help="VRAM policy for the Gemma encoder + 22B transformer. "
                         "'none' (default) keeps all weights on GPU (~78 GB, fastest). "
                         "'cpu' streams weights from CPU RAM (~5 GB VRAM + ~36 GB RAM, "
                         "much slower) — use to fit a busy/shared GPU. 'disk' streams "
                         "from disk (lowest RAM, slowest).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = f"{args.width}*{args.height}"

    # read assigned rows (skip header at row 1)
    rows = []
    with open(args.prompts_tsv) as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines, start=1):
        if i < max(2, args.row_start) or i > args.row_end:
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        scene_id, family, gen_prompt, _train, exp = parts[:5]
        # Columns 6-7 (subject_type, gate) are present once build_prompts.py is
        # run with --include-nonhuman; default to a human/person scene so older
        # 5-column TSVs still work unchanged.
        gate = parts[6] if len(parts) > 6 else "person"
        rows.append((scene_id, family, gen_prompt, int(exp), gate))
    log(f"GPU {args.gpu_label}: {len(rows)} rows (lines {args.row_start}..{args.row_end})")

    # ---- load BOTH models once ----
    log(f"GPU {args.gpu_label}: building DistilledPipeline (cached registry) ...")
    t0 = time.time()
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=args.distilled,
        spatial_upsampler_path=args.upscaler,
        gemma_root=args.gemma,
        loras=(),
        registry=StateDictRegistry(),   # cache state-dicts in RAM across calls
        offload_mode=OffloadMode(args.offload_mode),
    )
    cp._model(device)                   # preload Faster R-CNN once (resident)
    log(f"GPU {args.gpu_label}: models ready in {time.time()-t0:.0f}s")

    tiling = TilingConfig.default()
    chunks = get_video_chunks_number(args.gen_frames, tiling)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_shard.parent.mkdir(parents=True, exist_ok=True)
    mf = open(args.manifest_shard, "a")

    def emit(scene, family, status, dest, seed, attempts, secs, ratio):
        mf.write("\t".join([
            time.strftime("%F %T"), str(args.gpu_label), scene, family,
            f"T2V_attempts={attempts}", str(dest), str(seed), str(secs),
            str(args.frame_num), size, "", status, f"{ratio}" if ratio is not None else "NA",
        ]) + "\n")
        mf.flush()

    for scene, family, gen_prompt, expected, gate in rows:
        name = f"{scene}__{family}"
        dest = args.out_dir / f"{name}.mp4"
        if dest.exists():
            log(f"GPU {args.gpu_label}: SKIP {name} (exists)")
            continue
        raw_tmp = args.tmp_dir / f"{name}_raw.mp4"
        prompt = gen_prompt + args.positive_nudge

        status, ratio, seed, counts = "fail_gen", None, None, None
        t_all = time.time()
        for attempt in range(1, args.max_seeds + 1):
            seed = random.randint(1, 2**31 - 1)
            log(f"GPU {args.gpu_label}: GEN {name} attempt={attempt}/{args.max_seeds} seed={seed} exp={expected} gate={gate}")
            try:
                video, _audio = pipeline(
                    prompt=prompt, seed=seed,
                    height=args.height, width=args.width,
                    num_frames=args.gen_frames, frame_rate=args.frame_rate,
                    images=[], tiling_config=tiling,
                )
                encode_video(video=video, fps=args.frame_rate, audio=None,
                             output_path=str(raw_tmp), video_chunks_number=chunks)
            except Exception as e:  # noqa: BLE001
                status = "fail_gen"
                log(f"GPU {args.gpu_label}:   attempt {attempt} gen ERROR: {type(e).__name__}: {e}")
                continue

            # read generated frames once
            r = iio.get_reader(str(raw_tmp))
            frames = []
            for fr in r:
                frames.append(np.asarray(fr, dtype=np.uint8))
                if len(frames) >= args.gen_frames:
                    break
            r.close()
            if len(frames) < args.frame_num:
                status = "fail_gen"
                log(f"GPU {args.gpu_label}:   attempt {attempt} too few frames ({len(frames)})")
                continue
            crop = frames[: args.frame_num]

            # ---- gate 1: fade (in or out) ----
            fm = cf.measure_fade(crop)
            ratio = round(fm["ratio"], 3)
            if not args.no_fade_gate and (
                    fm["start_ratio"] < args.fade_threshold or fm["end_ratio"] < args.fade_threshold):
                status = "fade_dropped"
                log(f"GPU {args.gpu_label}:   attempt {attempt} DROP fade "
                    f"(start={fm['start_ratio']:.2f} end={fm['end_ratio']:.2f})")
                continue

            # ---- gate 2: subject / cast cleanliness (first/mid/last) ----
            # Subject-aware (see count_people_filter.verdict_subject):
            #   person   -> count people (original rule);
            #   <animal> -> require zero humans + the animal present/not-exceeded;
            #   noperson -> require zero humans;
            #   none     -> skip (human-shaped non-human; detector hits subject).
            n = len(crop)
            idxs = sorted({0, n // 2, n - 1})
            if args.no_cast_gate:
                gate = "none"
            if gate == "none":
                person_counts, target_counts = None, None
            else:
                person_counts = [cp.count_class(crop[i], device, args.person_score,
                                                args.person_min_area, cp.COCO_PERSON)
                                 for i in idxs]
                if gate in ("person", "noperson"):
                    target_counts = person_counts if gate == "person" else None
                else:
                    tid = cp.coco_id(gate)
                    target_counts = [cp.count_class(crop[i], device, args.person_score,
                                                    args.person_min_area, tid)
                                     for i in idxs]
            counts = {"person": person_counts, "target": target_counts}
            if not cp.verdict_subject(person_counts, target_counts, expected, gate):
                status = "people_dropped"
                log(f"GPU {args.gpu_label}:   attempt {attempt} DROP cast gate={gate} "
                    f"exp={expected} person={person_counts} target={target_counts}")
                continue

            # passed both gates -> write cropped clip
            write_clip(crop, dest, args.frame_rate)
            status = "ok"
            log(f"GPU {args.gpu_label}: OK {name} attempt={attempt} seed={seed} fade={ratio} counts={counts}")
            break

        secs = int(time.time() - t_all)
        if raw_tmp.exists():
            raw_tmp.unlink()
        emit(scene, family, status, dest if status == "ok" else "", seed,
             attempt, secs, ratio)
        # release any per-clip VRAM the pipeline left reserved before the next clip
        torch.cuda.empty_cache()
        gc.collect()

    mf.close()
    log(f"GPU {args.gpu_label}: done.")


if __name__ == "__main__":
    main()
