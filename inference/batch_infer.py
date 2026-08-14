#!/usr/bin/env python
"""Batch inference: load the model once, run many jobs from one JSON.

Reads an input JSON describing a shared `config`, per-item `defaults`, and a
list of `items`, and runs in THREE phases (not interleaved per item) so the
GPU is never shared between the warp renderer and the resident LTX model:

  Phase A  build every trajectory, then render every Pi3X warp in ONE warp-env
           subprocess (Pi3X loaded once; LTX not loaded yet, so the renderer
           has the GPU to itself)
  Phase B  load the audio-video pipeline ONCE, then run every inference
           back-to-back -> gen.mp4                  (shared pipeline)
  Phase C  keyboard UI overlay for every item       (ui-env subprocess)

Interleaving warp (a GPU-heavy subprocess) before each inference while ~40 GB
of LTX sits resident measurably slows the inference; the phase split avoids
that and also loads the model only once.

Writes an output JSON recording, per item, the status and the paths of the
generated video (and the UI video / warp). Each item's artefacts go to
`<output_root>/<name>/` with the same layout as run_helloworld.sh. A per-item
failure in any phase is recorded and the remaining items continue.

Run (ltx env):
    python batch_infer.py --input examples_batch.json --output results.json --gpu 0

Each item needs exactly these inputs:
  image             the first-frame scene image
  text_prompt       full prompt (scene + character + action + spoken phrase)
  camera_poses      the camera trajectory: an .npz with `camera_poses`,
                    one 4x4 camera-to-world matrix per frame
  interaction_time  [start, end] seconds — when the action/speech happens
  interaction_prompt / interaction_speech
                    the action / spoken phrase, verbatim substrings of text_prompt
  seed              random seed

Instead of `camera_poses` you may give `pose`, a WASD/arrow shorthand that is
compiled into a trajectory for you (see build_pose_trajectory.py). `phases`
(optional, auto-derived from `pose`) is the key-press timeline used only for
the keyboard-overlay video; items without it skip the overlay.

Input JSON schema (all config/defaults fields are optional — sensible
defaults shown in DEFAULT_CONFIG / DEFAULT_ITEM below):

    {
      "config":   { "output_root": "outputs/batch", "lora_ckpt": "...", ... },
      "defaults": { "enable_audio": true, "interaction_prompt": "wave hello", ... },
      "items": [
        { "name": "garden", "image": "scene.png", "text_prompt": "... wave hello ... 'Hi!' ...",
          "camera_poses": "camera_poses.npz", "phases": "phases.json",
          "interaction_time": [4, 7], "interaction_speech": "'Hi!'", "seed": 1234 },
        { "name": "street", "image": "scene2.png", "text_prompt": "...",
          "pose": "w-3, left-14, right-28, left-11, w-3" }
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# Defaults (mirror run_helloworld.sh)
# -----------------------------------------------------------------------------
HF_ROOT = os.environ.get("HF_ROOT", os.path.expanduser("~/.cache/huggingface/hub"))
DEFAULT_CONFIG = {
    "output_root": str(HERE / "outputs" / "batch"),
    "num_frames": 241,
    "frame_rate": 24,
    "width": 1280,
    "height": 704,
    "cond_attn_strength": 0.3,
    "vis_threshold": 0.1,
    "ramp": 12,
    # The released HelloWorld LoRA. It is trained with the "No other people
    # appear in the scene." clause, so ending text_prompt with that clause
    # actually suppresses hallucinated extra characters (the base model is
    # CFG-less, so the constraint only works because training included it).
    "lora_ckpt": str(HERE.parent / "checkpoints" / "helloworld_lora_v1.safetensors"),
    "lora_strength": 1.0,
    "distilled_ckpt": f"{HF_ROOT}/Lightricks/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors",
    "upscaler_ckpt": f"{HF_ROOT}/Lightricks/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "gemma_root": f"{HF_ROOT}/google/gemma-3-12b-it-qat-q4_0-unquantized",
    "warp_py": os.environ.get("WARP_PY", "python"),
    "ui_py": os.environ.get("UI_PY", sys.executable),
    "ffmpeg": os.environ.get("FFMPEG", "ffmpeg"),
}
DEFAULT_ITEM = {
    "interaction_time": [4, 7],          # seconds
    "interaction_prompt": "wave hello",
    "interaction_speech": "'Hello!'",
    "enable_audio": True,
    "enable_videotemporalmask": True,
    "enable_audiotemporalmask": True,
    "with_ui": True,     # also write gen_ui.mp4 (keyboard overlay; needs phases)
    "with_warp": True,   # keep warp/warp.mp4 (the camera condition) in the output
    "seed": 1234,
    # Optional per-frame speed/amplitude knobs (see build_pose_trajectory.py).
    # Absent -> the builder's own defaults. Set in `defaults` to apply to every
    # item, or per item to override. yaw/pitch in deg/frame, fwd/strafe in
    # c2w units/frame.
    # "yaw_rate": 0.125, "pitch_rate": 0.125, "fwd_rate": 0.004167, "strafe_rate": 0.004167,
}

# Per-frame speed knobs that, if present on an item, are forwarded to the
# trajectory builder (otherwise its defaults apply).
RATE_KEYS = ("yaw_rate", "pitch_rate", "fwd_rate", "strafe_rate")

# Pi3X warp calibration (maps pose-translation units to scene-metric motion at
# the scale the warp/LoRA was rendered with). LOCKED constant — NOT a speed
# knob; use the fwd_rate / strafe_rate item fields to change dolly/strafe.
WARP_TRANSLATION_SCALE = 0.1


def secs_to_frames(t, frame_rate, num_frames):
    f = int(round(float(t) * float(frame_rate)))
    return max(0, min(f, num_frames))


def render_warps_batch(cfg, warp_jobs: list[dict], scratch_dir: Path) -> dict:
    """Render every item's Pi3X warp in ONE warp-env subprocess (Pi3X loaded
    once). `warp_jobs` = [{name, image, poses, out_dir}, ...]. Returns a dict
    keyed by out_dir -> {status, seconds, warp_mp4, mask_npz, error?}."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    jobs_json = scratch_dir / "_warp_jobs.json"
    results_json = scratch_dir / "_warp_results.json"
    jobs_json.write_text(json.dumps({
        "height": cfg["height"] // 2, "width": cfg["width"] // 2,
        "num_frames": cfg["num_frames"], "fps": cfg["frame_rate"],
        "translation_scale": WARP_TRANSLATION_SCALE,
        "jobs": warp_jobs,
    }, indent=2))
    cmd = [
        cfg["warp_py"], str(HERE / "render_warp_batch.py"),
        "--jobs", str(jobs_json), "--results", str(results_json),
    ]
    subprocess.run(cmd, check=True)  # inherits CUDA_VISIBLE_DEVICES
    out = json.loads(results_json.read_text())
    return {r["out_dir"]: r for r in out["results"]}


def make_ui(cfg, item_dir: Path, gen_mp4: Path, phases_json: Path,
            f_start: int, f_end: int, with_audio: bool) -> Path:
    """Overlay the keyboard UI in the ui conda env (subprocess)."""
    out = item_dir / "gen_ui.mp4"
    cmd = [
        cfg["ui_py"], str(HERE / "make_ui.py"),
        "--in-mp4", str(gen_mp4), "--out-mp4", str(out),
        "--phases", str(phases_json),
        "--f-start", str(f_start), "--f-end", str(f_end),
        "--num-frames", str(cfg["num_frames"]),
        "--ffmpeg", cfg["ffmpeg"],
    ]
    if with_audio:
        cmd.append("--with-audio")
    subprocess.run(cmd, check=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="Input JSON (config/defaults/items).")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output results JSON (default: <output_root>/results.json).")
    ap.add_argument("--gpu", type=int, default=None,
                    help="Physical GPU index. Overrides config.gpu. Set before model load.")
    args = ap.parse_args()

    spec = json.loads(args.input.read_text())
    cfg = {**DEFAULT_CONFIG, **spec.get("config", {})}
    item_defaults = {**DEFAULT_ITEM, **spec.get("defaults", {})}
    items = spec.get("items", [])
    if not items:
        raise SystemExit("input JSON has no 'items'.")

    # GPU must be fixed BEFORE torch is imported (infer.py imports torch).
    gpu = args.gpu if args.gpu is not None else cfg.get("gpu", 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import logging
    logging.getLogger().setLevel(logging.INFO)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from build_pose_trajectory import build_and_write
    from infer import build_pipeline, infer_one
    from ltx_pipelines.utils.args import (
        LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP, resolve_path,
    )

    out_root = Path(cfg["output_root"])

    # Build the per-item job list (merged config, paths, F-window).
    jobs = []
    for idx, raw in enumerate(items):
        it = {**item_defaults, **raw}
        name = it.get("name") or f"item_{idx:03d}"
        item_dir = Path(it["output_dir"]) if it.get("output_dir") else (out_root / name)
        item_dir.mkdir(parents=True, exist_ok=True)
        jobs.append({
            "it": it,
            "name": name,
            "item_dir": item_dir,
            "f_start": secs_to_frames(it["interaction_time"][0], cfg["frame_rate"], cfg["num_frames"]),
            "f_end": secs_to_frames(it["interaction_time"][1], cfg["frame_rate"], cfg["num_frames"]),
            "stage_s": {},
            "rec": {"name": name, "output_dir": str(item_dir), "status": "pending"},
            "ok": True,   # cleared on first failing stage; later stages skip it
        })

    def fail(job, stage, exc):
        job["ok"] = False
        job["rec"].update({"status": "fail", "failed_stage": stage,
                           "error": f"{type(exc).__name__}: {exc}"})
        print(f"[batch] [FAIL:{stage}] {job['name']}: {job['rec']['error']}", file=sys.stderr, flush=True)

    # ---- Phase A: build EVERY trajectory (in-process), then render all Pi3X
    # warps in ONE warp-env subprocess (Pi3X loaded once; LTX not loaded yet so
    # the renderer has the GPU to itself). ----
    print(f"\n[batch] === Phase A: trajectories + batched warp for {len(jobs)} item(s) ===", flush=True)
    for j in jobs:
        it = j["it"]
        try:
            traj_dir = j["item_dir"] / "trajectory"
            if it.get("camera_poses"):
                # Trajectory supplied directly (one 4x4 c2w per frame).
                traj_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(it["camera_poses"], traj_dir / "camera_poses.npz")
                if it.get("phases"):  # key-press timeline, only for the UI overlay
                    shutil.copyfile(it["phases"], traj_dir / "phases.json")
            elif it.get("pose"):
                # WASD/arrow shorthand -> trajectory + phases.json.
                rate_kwargs = {k: float(it[k]) for k in RATE_KEYS if it.get(k) is not None}
                if it.get("frames_per_unit") is not None:
                    rate_kwargs["frames_per_unit"] = int(it["frames_per_unit"])
                build_and_write(it["pose"], cfg["num_frames"], traj_dir, verbose=False, **rate_kwargs)
            else:
                raise ValueError("item needs either 'camera_poses' (.npz) or 'pose' (WASD string)")
            j["traj_dir"] = traj_dir
            j["warp_dir"] = j["item_dir"] / "warp"
        except Exception as e:  # noqa: BLE001
            fail(j, "trajectory", e)

    # Warp REUSE: the Pi3X warp depends only on (image, trajectory) — NOT on the
    # LoRA — so a warp already rendered (by another method, or a prior run) is
    # reusable. Skip rendering any item whose warp.mp4 + mask.npz already exist.
    # (Also makes this stage resumable, and lets a new method reuse the shared
    # warps: pre-seed <item>/warp/{warp.mp4,mask.npz} from an existing method.)
    warp_jobs = []
    for j in jobs:
        if not j["ok"]:
            continue
        wmp4, wnpz = j["warp_dir"] / "warp.mp4", j["warp_dir"] / "mask.npz"
        if wmp4.is_file() and wnpz.is_file():
            j["warp_mp4"], j["mask_npz"] = wmp4, wnpz
            j["stage_s"]["warp"] = 0.0
            j["warp_reused"] = True
        else:
            warp_jobs.append(
                {"name": j["name"], "image": j["it"]["image"],
                 "poses": str(j["traj_dir"] / "camera_poses.npz"),
                 "out_dir": str(j["warp_dir"])})
    n_reused = sum(1 for j in jobs if j.get("warp_reused"))
    if n_reused:
        print(f"[batch] warp reuse: {n_reused} item(s) already have warp.mp4+mask.npz "
              f"(warp is LoRA-independent); rendering the remaining {len(warp_jobs)}.", flush=True)
    if warp_jobs:
        ts = time.time()
        try:
            warp_res = render_warps_batch(cfg, warp_jobs, out_root)
        except Exception as e:  # noqa: BLE001 — whole warp pass failed; fail all warp jobs
            warp_res = {}
            for j in jobs:
                if j["ok"] and not j.get("warp_reused"):
                    fail(j, "warp", e)
        for j in jobs:
            if not j["ok"] or j.get("warp_reused"):
                continue
            r = warp_res.get(str(j["warp_dir"]))
            if r is None or r.get("status") != "ok":
                fail(j, "warp", RuntimeError(r.get("error") if r else "no warp result"))
                continue
            j["warp_mp4"], j["mask_npz"] = Path(r["warp_mp4"]), Path(r["mask_npz"])
            j["stage_s"]["warp"] = r.get("seconds")
        print(f"[batch] warp pass: {round(time.time() - ts, 1)}s total (Pi3X loaded once)", flush=True)

    # ---- Phase B: load the model ONCE, then run ALL inferences back-to-back
    # (no warp subprocess interleaved). ----
    todo = [j for j in jobs if j["ok"]]
    if todo:
        print(f"\n[batch] === Phase B: load model on GPU {gpu}, then infer {len(todo)} item(s) ===", flush=True)
        loras = (LoraPathStrengthAndSDOps(
            resolve_path(cfg["lora_ckpt"]), float(cfg["lora_strength"]), LTXV_LORA_COMFY_RENAMING_MAP),)
        pipeline = build_pipeline(
            distilled_checkpoint_path=cfg["distilled_ckpt"],
            spatial_upsampler_path=cfg["upscaler_ckpt"],
            gemma_root=cfg["gemma_root"],
            loras=loras,
        )
        print("[batch] pipeline ready.", flush=True)
        for j in todo:
            it, name = j["it"], j["name"]
            print(f"[batch] [infer] {name}", flush=True)
            try:
                ts = time.time()
                gen_mp4 = j["item_dir"] / "gen.mp4"
                infer_one(
                    pipeline,
                    prompt=it["text_prompt"],
                    image=it["image"],
                    warp_mp4=j["warp_mp4"],
                    warp_mask=j["mask_npz"],
                    output_path=gen_mp4,
                    interaction_prompt=it["interaction_prompt"],
                    interaction_speech=it.get("interaction_speech"),
                    interaction_window=(j["f_start"], j["f_end"]),
                    interaction_ramp=cfg["ramp"],
                    enable_audio=bool(it["enable_audio"]),
                    enable_videotemporalmask=bool(it["enable_videotemporalmask"]),
                    enable_audiotemporalmask=bool(it["enable_audiotemporalmask"]),
                    num_frames=cfg["num_frames"],
                    frame_rate=cfg["frame_rate"],
                    width=cfg["width"],
                    height=cfg["height"],
                    seed=int(it["seed"]),
                    cond_attn_strength=cfg["cond_attn_strength"],
                    vis_threshold=cfg["vis_threshold"],
                )
                j["gen_mp4"] = gen_mp4
                j["stage_s"]["infer"] = round(time.time() - ts, 1)
            except Exception as e:  # noqa: BLE001
                fail(j, "infer", e)

    # ---- Phase C: UI overlays (ui-env subprocess; CPU + ffmpeg). Needs the
    # key-press timeline (phases.json) — items with a bare trajectory and no
    # `phases` skip the overlay. ----
    todo = [j for j in jobs if j["ok"] and j["it"].get("with_ui")
            and (j["traj_dir"] / "phases.json").is_file()]
    if todo:
        print(f"\n[batch] === Phase C: UI overlay for {len(todo)} item(s) ===", flush=True)
        for j in todo:
            try:
                ts = time.time()
                j["gen_ui_mp4"] = make_ui(
                    cfg, j["item_dir"], j["gen_mp4"], j["traj_dir"] / "phases.json",
                    j["f_start"], j["f_end"], with_audio=bool(j["it"]["enable_audio"]),
                )
                j["stage_s"]["ui"] = round(time.time() - ts, 1)
            except Exception as e:  # noqa: BLE001
                fail(j, "ui", e)

    # ---- Finalize: params sidecar + results record per item. The warp is
    # always rendered (it is the model's camera condition); with_warp only
    # controls whether it is kept as an output artefact. ----
    results = []
    for j in jobs:
        it, rec, s = j["it"], j["rec"], j["stage_s"]
        if j["ok"] and not it.get("with_warp", True):
            shutil.rmtree(j["warp_dir"], ignore_errors=True)
            j["warp_mp4"] = None
        (j["item_dir"] / "params.txt").write_text(
            json.dumps({**cfg, **it, "interaction_window_frames": [j["f_start"], j["f_end"]]}, indent=2) + "\n"
        )
        if j["ok"]:
            rec.update({
                "status": "ok",
                "gen_mp4": str(j.get("gen_mp4")),
                "gen_ui_mp4": (str(j["gen_ui_mp4"]) if j.get("gen_ui_mp4") else None),
                "warp_mp4": (str(j["warp_mp4"]) if j.get("warp_mp4") else None),
            })
            print(f"[batch] [OK] {j['name']}  warp {s.get('warp')}s + infer {s.get('infer')}s "
                  f"+ ui {s.get('ui')}s", flush=True)
        rec.update({
            "interaction_window_frames": [j["f_start"], j["f_end"]],
            "stage_seconds": s,
        })
        results.append(rec)

    out_json = args.output or (out_root / "results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    n_ok = sum(r["status"] == "ok" for r in results)
    out_json.write_text(json.dumps(
        {"gpu": gpu, "n_total": len(results), "n_ok": n_ok, "results": results}, indent=2) + "\n")
    print(f"\n[batch] done: {n_ok}/{len(results)} ok. results -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
