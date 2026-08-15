#!/usr/bin/env python
"""Train the warp-conditioned social-interaction LoRA on LTX-2.3.

Inference-aware LoRA training on a multi-scene dataset of (gt, warp, mask)
triples. Each prepared clip carries its own camera-neutral `prompt` in its
`meta.json`; we cache the Gemma-encoded (video_context, context_mask) for each
unique prompt once, then look up the right cached pair per sampled clip.

The training step mixes the encoded warp into the noisy latent within the
visible-mask region (strength = --warp-lambda-max, which MUST match the
inference value) and simulates the inference post-process re-injection, so the
LoRA learns to "subtract" the Pi3X warp grid from the x_0 prediction.

CLI only — no hardcoded paths.

Data layout (--data-root): manifest.tsv + clip_*/{gt.mp4, warp.mp4, mask.npz,
meta.json}. All clips must share (H, W, T, fps). The manifest's per-clip
`level` column groups motion families (the warp-prep stage's `family` column,
renamed to `level` by the visibility filter).

Example:
  python train_lora.py \
    --distilled-checkpoint /path/to/ltx-2.3-22b-distilled-1.1.safetensors \
    --gemma-root /path/to/gemma-3-12b-it-... \
    --data-root data/filtered \
    --output-dir runs/lora \
    --warp-lambda-max 0.5 --max-steps 2000 --save-every 500 --log-every 10 \
    --lr 1e-4 --lora-rank 32
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file

from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.helpers import video_latent_from_file

from ltx_trainer.model_loader import (
    load_embeddings_processor,
    load_text_encoder,
    load_transformer,
    load_video_vae_encoder,
)
from ltx_trainer.timestep_samplers import ShiftedLogitNormalTimestepSampler
from ltx_core.model.transformer.modality import Modality


# --------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--distilled-checkpoint", required=True, type=Path)
    p.add_argument("--gemma-root", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path,
                   help="A.3 data dir: manifest.tsv + clip_*/{gt,warp,mask,meta}.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--warp-lambda-max", type=float, default=0.5,
                   help="Per-token visible-anchor strength; must match inference value.")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--vis-threshold", type=float, default=0.1)
    # LoRA — same defaults as v2 (which matched A.2/W-a-H)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lora-target-modules",
                   default="attn1.to_q,attn1.to_k,attn1.to_v,attn1.to_out.0")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    return p.parse_args()


# --------------------------------------------------------------------------- data
def load_clip_manifest(data_root: Path) -> list[dict]:
    """Read A.3 manifest.tsv + each clip's meta.json (for the per-clip prompt)."""
    manifest = data_root / "manifest.tsv"
    if not manifest.exists():
        raise SystemExit(f"missing {manifest}; run prepare_data_a3.py / merge_data.py first")
    rows: list[dict] = []
    header = None
    for i, line in enumerate(manifest.read_text().splitlines()):
        if not line.strip():
            continue
        if i == 0:
            header = line.split("\t")
            continue
        cells = line.split("\t")
        row = dict(zip(header, cells))
        clip_dir = Path(row["clip_dir"])
        meta_path = clip_dir / "meta.json"
        if not meta_path.exists():
            raise SystemExit(f"missing meta.json at {meta_path}")
        meta = json.loads(meta_path.read_text())
        prompt = meta.get("prompt") or row.get("prompt")
        if not prompt:
            raise SystemExit(f"clip {row['clip_id']} has no prompt (meta.json or manifest)")
        rows.append({
            "clip_id": int(row["clip_id"]),
            "clip_dir": clip_dir,
            "gt": clip_dir / "gt.mp4",
            "warp": clip_dir / "warp.mp4",
            "mask": clip_dir / "mask.npz",
            "meta": meta_path,
            "prompt": prompt,
            "image_id": row.get("image_id", meta.get("image_id", "")),
            "generator": row.get("generator", meta.get("generator", "")),
            "level": row.get("level", meta.get("level", "")),
            "height": int(row["height"]),
            "width": int(row["width"]),
            "num_frames": int(row["num_frames"]),
            "fps": int(row["fps"]),
        })
    if not rows:
        raise SystemExit(f"no clips listed in {manifest}")
    return rows


@torch.inference_mode()
def encode_video_to_latent(
    encoder, video_path: Path, num_frames: int, height: int, width: int, fps: int,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    return video_latent_from_file(
        video_encoder=encoder, file_path=str(video_path),
        output_shape=VideoPixelShape(batch=1, frames=num_frames,
                                      height=height, width=width, fps=fps),
        device=device, dtype=dtype,
    )


def load_visibility_to_latent(
    mask_path: Path, latent_shape: tuple[int, int, int, int, int],
    device: torch.device, dtype: torch.dtype, threshold: float,
) -> torch.Tensor:
    data = np.load(mask_path)
    if "visibility" not in data.files:
        raise KeyError(f"{mask_path} missing 'visibility' key")
    vis = data["visibility"].astype(np.float32)
    if vis.ndim != 3:
        raise ValueError(f"visibility must be [T, H, W], got {vis.shape}")
    vis5 = torch.from_numpy(vis)[None, None].to(device).float()
    B, _C, T_lat, H_lat, W_lat = latent_shape
    resampled = F.interpolate(vis5, size=(T_lat, H_lat, W_lat),
                               mode="trilinear", align_corners=False)
    binary = (resampled.clamp(0.0, 1.0) > float(threshold)).to(dtype=dtype)
    return binary.expand(B, 1, T_lat, H_lat, W_lat).contiguous()


def build_video_positions(
    patchifier: VideoLatentPatchifier,
    latent_shape: VideoLatentShape, fps: int,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    scale = SpatioTemporalScaleFactors.default()
    latent_coords = patchifier.get_patch_grid_bounds(
        output_shape=latent_shape, device=device,
    )
    positions = get_pixel_coords(
        latent_coords=latent_coords,
        scale_factors=scale,
        causal_fix=True,
    ).float()
    positions[:, 0, ...] = positions[:, 0, ...] / float(fps)
    return positions.to(dtype)


@torch.inference_mode()
def encode_prompt_once(
    text_encoder, embeddings_processor, prompt: str,
    device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states, raw_attention_mask = text_encoder.encode(prompt, padding_side="left")
    out = embeddings_processor.process_hidden_states(
        hidden_states, raw_attention_mask, padding_side="left",
    )
    video_context = out.video_encoding.to(device=device, dtype=dtype)
    binary_mask = out.attention_mask.to(device=device)
    return video_context, binary_mask


# --------------------------------------------------------------------------- training step (verbatim v2)
def train_step(
    *,
    transformer,
    timestep_sampler,
    target_latent: torch.Tensor,
    warp_latent: torch.Tensor,
    visibility: torch.Tensor,
    video_context: torch.Tensor,
    context_mask: torch.Tensor,
    positions: torch.Tensor,
    patchifier: VideoLatentPatchifier,
    dtype: torch.dtype,
    device: torch.device,
    lambda_max: float,
) -> tuple[torch.Tensor, dict]:
    """v2 training step (inference-aware loss). Identical to the A.2 v2 trainer;
    kept identical to the released checkpoint's training run.
    """
    B, C, T_lat, H_lat, W_lat = target_latent.shape
    assert B == 1
    lam = float(lambda_max)

    frame0_5d = torch.zeros(B, 1, T_lat, H_lat, W_lat, dtype=dtype, device=device)
    frame0_5d[:, :, 0] = 1.0

    vis_anchored = visibility.clone().to(dtype=dtype)
    vis_anchored[:, :, 0] = 0.0
    s_5d = vis_anchored * lam

    denoise_mask_5d = (1.0 - frame0_5d) * (1.0 - s_5d)

    target_tokens     = patchifier.patchify(target_latent.to(dtype))
    warp_tokens       = patchifier.patchify(warp_latent.to(dtype))
    s_tok             = patchifier.patchify(s_5d)
    frame0_tok        = patchifier.patchify(frame0_5d)
    denoise_mask_tok  = patchifier.patchify(denoise_mask_5d)

    sigma = timestep_sampler.sample_for(target_tokens).to(dtype=dtype)
    sigma_b11 = sigma.view(-1, 1, 1)
    sigma_per_token = sigma_b11 * denoise_mask_tok

    noise = torch.randn_like(target_tokens)
    natural_noisy = (1.0 - sigma_b11) * target_tokens + sigma_b11 * noise
    mixed_noisy   = (1.0 - s_tok) * natural_noisy + s_tok * warp_tokens
    input_tokens  = (1.0 - frame0_tok) * mixed_noisy + frame0_tok * target_tokens

    clean_baseline_5d = torch.zeros_like(target_latent)
    clean_baseline_5d[:, :, 0] = target_latent[:, :, 0]
    clean_baseline = patchifier.patchify(clean_baseline_5d.to(dtype))
    clean_polluted = (1.0 - s_tok) * clean_baseline + s_tok * warp_tokens

    per_token_timesteps = sigma_per_token.squeeze(-1).to(dtype=dtype)
    video = Modality(
        enabled=True,
        latent=input_tokens,
        sigma=sigma,
        timesteps=per_token_timesteps,
        positions=positions.to(dtype=dtype),
        context=video_context,
        context_mask=context_mask,
    )
    pred_video, _ = transformer(video=video, audio=None, perturbations=None)
    pred_v = pred_video

    pred_x0 = input_tokens - sigma_per_token * pred_v
    final_x0 = denoise_mask_tok * pred_x0 + (1.0 - denoise_mask_tok) * clean_polluted

    loss_mask = (1.0 - frame0_tok)
    per_token_sqerr = (final_x0 - target_tokens).pow(2)
    weighted = per_token_sqerr * loss_mask
    denom = loss_mask.sum().clamp(min=1.0) * float(per_token_sqerr.shape[-1])
    loss = weighted.sum() / denom

    with torch.no_grad():
        vis_mask  = ((1.0 - frame0_tok) * (s_tok > 0).to(dtype))
        free_mask = ((1.0 - frame0_tok) * (s_tok == 0).to(dtype))
        vis_denom  = vis_mask.sum().clamp(min=1.0) * float(per_token_sqerr.shape[-1])
        free_denom = free_mask.sum().clamp(min=1.0) * float(per_token_sqerr.shape[-1])
        loss_visible = (per_token_sqerr * vis_mask).sum() / vis_denom
        loss_free    = (per_token_sqerr * free_mask).sum() / free_denom
    stats = {
        "sigma": float(sigma.detach().mean().item()),
        "lambda_max": lam,
        "vis_frac_raw": float(visibility.mean().item()),
        "loss_visible": float(loss_visible.detach().item()),
        "loss_free": float(loss_free.detach().item()),
    }
    return loss, stats


# --------------------------------------------------------------------------- save LoRA
def save_lora_safetensors(transformer, path: Path, dtype: torch.dtype = torch.bfloat16) -> None:
    base = transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer
    state_dict = get_peft_model_state_dict(base)
    state_dict = {k.replace("base_model.model.", "", 1): v for k, v in state_dict.items()}
    state_dict = {f"diffusion_model.{k}": v.to(dtype) if isinstance(v, torch.Tensor) else v
                  for k, v in state_dict.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(path))


# --------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # ---- load clips (each carries its own prompt) ----
    clips = load_clip_manifest(args.data_root)
    print(f"[train_v2_a3] {len(clips)} clips from {args.data_root}  "
          f"λ_max={args.warp_lambda_max}", flush=True)
    H = clips[0]["height"]; W = clips[0]["width"]; T = clips[0]["num_frames"]; fps = clips[0]["fps"]
    for c in clips:
        assert (c["height"], c["width"], c["num_frames"], c["fps"]) == (H, W, T, fps), \
            f"all clips must share (H, W, T, fps); got mismatch at {c['clip_dir']}"

    pixel_shape = VideoPixelShape(batch=1, frames=T, height=H, width=W, fps=fps)
    v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
    latent_shape = (1, v_shape.channels, v_shape.frames, v_shape.height, v_shape.width)
    print(f"[train_v2_a3] (H, W, T) = ({H}, {W}, {T}) -> latent {latent_shape}", flush=True)

    patchifier = VideoLatentPatchifier(patch_size=1)

    # ---- text encoder: cache encoding of each UNIQUE prompt, then free Gemma ----
    print("[train_v2_a3] loading text encoder + embeddings processor...", flush=True)
    text_encoder = load_text_encoder(
        gemma_model_path=str(args.gemma_root), device="cuda", dtype=dtype,
    )
    embeddings_processor = load_embeddings_processor(
        checkpoint_path=str(args.distilled_checkpoint), device="cuda", dtype=dtype,
    )
    unique_prompts = sorted({c["prompt"] for c in clips})
    print(f"[train_v2_a3] {len(unique_prompts)} unique prompts among {len(clips)} clips", flush=True)
    prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for i, p in enumerate(unique_prompts):
        vc, cm = encode_prompt_once(text_encoder, embeddings_processor, p, device, dtype)
        prompt_cache[p] = (vc, cm)
        if (i + 1) % 10 == 0 or i == len(unique_prompts) - 1:
            print(f"[train_v2_a3]   encoded {i + 1}/{len(unique_prompts)} prompts", flush=True)
    del text_encoder, embeddings_processor
    torch.cuda.empty_cache()

    positions = build_video_positions(patchifier, v_shape, fps, device, dtype)
    print(f"[train_v2_a3] positions={tuple(positions.shape)}", flush=True)

    # ---- VAE encoder: pre-encode ALL clips (target + warp) and mask ----
    print("[train_v2_a3] loading video VAE encoder...", flush=True)
    video_encoder = load_video_vae_encoder(
        checkpoint_path=str(args.distilled_checkpoint), device="cuda", dtype=dtype,
    )
    print(f"[train_v2_a3] pre-encoding {len(clips)} clips...", flush=True)
    encoded_clips: list[dict] = []
    for c in clips:
        target_latent = encode_video_to_latent(
            video_encoder, c["gt"], T, H, W, fps, device, dtype,
        )
        warp_latent = encode_video_to_latent(
            video_encoder, c["warp"], T, H, W, fps, device, dtype,
        )
        if tuple(target_latent.shape) != latent_shape:
            raise RuntimeError(f"target_latent shape {target_latent.shape} != expected {latent_shape}")
        if tuple(warp_latent.shape) != latent_shape:
            raise RuntimeError(f"warp_latent shape {warp_latent.shape} != expected {latent_shape}")
        visibility = load_visibility_to_latent(
            c["mask"], latent_shape, device, dtype, threshold=args.vis_threshold,
        )
        encoded_clips.append({
            "clip_id": c["clip_id"],
            "image_id": c["image_id"],
            "generator": c["generator"],
            "level": c["level"],
            "prompt": c["prompt"],
            "target": target_latent,
            "warp": warp_latent,
            "vis": visibility,
        })
        if c["clip_id"] % 20 == 0 or c["clip_id"] == clips[-1]["clip_id"]:
            print(f"[train_v2_a3]   clip {c['clip_id']:04d} ({c['generator']}_{c['image_id']}_{c['level']}) "
                  f"encoded vis_frac={visibility.float().mean().item():.3f}", flush=True)
    del video_encoder
    torch.cuda.empty_cache()

    # ---- transformer + LoRA ----
    print("[train_v2_a3] loading transformer (frozen base)...", flush=True)
    transformer = load_transformer(
        checkpoint_path=str(args.distilled_checkpoint), device="cuda", dtype=dtype,
    )
    if args.gradient_checkpointing and hasattr(transformer, "gradient_checkpointing_enable"):
        transformer.gradient_checkpointing_enable()

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=int(args.lora_rank), lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=target_modules, init_lora_weights=True,
    )
    transformer = get_peft_model(transformer, lora_config)
    trainable = [p for p in transformer.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in transformer.parameters())
    print(f"[train_v2_a3] LoRA trainable params: {n_train/1e6:.2f}M / total {n_total/1e9:.2f}B "
          f"({100.0 * n_train / n_total:.3f}%)", flush=True)

    start_step = 0
    if args.resume_from is not None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file as _load_file
        import re as _re
        sd = _load_file(str(args.resume_from))
        sd = {k.replace("diffusion_model.", "", 1): v for k, v in sd.items()}
        set_peft_model_state_dict(transformer.get_base_model(), sd)
        m = _re.search(r"step(\d+)", args.resume_from.name)
        if not m:
            raise SystemExit(f"--resume-from filename has no step number: {args.resume_from.name}")
        start_step = int(m.group(1))
        print(f"[train_v2_a3] resumed from step {start_step} ({args.resume_from})", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    timestep_sampler = ShiftedLogitNormalTimestepSampler()

    transformer.train()
    losses_log: list[dict] = []
    train_start = time.perf_counter()
    rng = random.Random(args.seed)
    for _ in range(start_step):
        rng.randrange(len(encoded_clips))

    for step in range(start_step, args.max_steps):
        clip = encoded_clips[rng.randrange(len(encoded_clips))]
        optimizer.zero_grad(set_to_none=True)

        video_context, context_mask = prompt_cache[clip["prompt"]]

        loss, stats = train_step(
            transformer=transformer,
            timestep_sampler=timestep_sampler,
            target_latent=clip["target"], warp_latent=clip["warp"],
            visibility=clip["vis"],
            video_context=video_context, context_mask=context_mask,
            positions=positions, patchifier=patchifier,
            dtype=dtype, device=device,
            lambda_max=float(args.warp_lambda_max),
        )
        loss.backward()
        if float(args.max_grad_norm) > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(args.max_grad_norm))
        else:
            grad_norm = None
        optimizer.step()

        record = {
            "step": int(step),
            "clip_id": int(clip["clip_id"]),
            "image_id": clip["image_id"],
            "level": clip["level"],
            "loss": float(loss.detach().cpu()),
            "lr": float(args.lr),
            "grad_norm": float(grad_norm.detach().cpu()) if grad_norm is not None else None,
            "elapsed_s": time.perf_counter() - train_start,
            **stats,
        }
        losses_log.append(record)

        if args.log_every > 0 and ((step + 1) % args.log_every == 0 or step == 0):
            print(json.dumps(record), flush=True)
            (args.output_dir / "train_loss.json").write_text(json.dumps(losses_log, indent=2))

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            ckpt = args.output_dir / f"helloworld_lora_step{step + 1:05d}.safetensors"
            save_lora_safetensors(transformer, ckpt, dtype=dtype)
            print(json.dumps({"event": "checkpoint", "step": step + 1, "path": str(ckpt)}), flush=True)

    final_ckpt = args.output_dir / "helloworld_lora_final.safetensors"
    save_lora_safetensors(transformer, final_ckpt, dtype=dtype)
    (args.output_dir / "train_loss.json").write_text(json.dumps(losses_log, indent=2))
    (args.output_dir / "train_config.json").write_text(json.dumps({
        "argv": sys.argv,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "num_clips": len(clips),
        "num_unique_prompts": len(unique_prompts),
        "latent_shape": list(latent_shape),
        "stage1_resolution": [H, W],
        "num_frames": T,
        "fps": fps,
        "lora_target_modules": target_modules,
    }, indent=2))
    print(json.dumps({"event": "train_done", "final_ckpt": str(final_ckpt),
                       "steps": args.max_steps,
                       "elapsed_s": time.perf_counter() - train_start}), flush=True)


if __name__ == "__main__":
    main()
