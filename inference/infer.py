#!/usr/bin/env python
"""Unified inference driver.

One clean entry point for the method (a camera-trajectory world model with
viewpoint-directed social interaction). Wraps the audio-video temporal-mask
pipeline (`TemporalAttentionMaskAVICLoraPipeline`) and exposes the four
ablation toggles as orthogonal flags so any combination runs through a single
code path:

    --enable_audio                output carries a co-generated speech track
    --enable_videotemporalmask    F-window gates the video action via the
                                  text cross-attention temporal mask
    --enable_audiotemporalmask    F-window gates the spoken phrase via the
                                  audio cross-attention temporal mask

Interaction (F-key) control:
    --interaction-prompt   substring of --prompt naming the *video* action
                           (e.g. "wave hello") gated by the video temporal mask
    --interaction-speech   substring of --prompt naming the *spoken* phrase
                           (e.g. "'Hello!'") gated by the audio temporal mask
    --interaction-window   START END  (pixel frames) the F-key high window
    --interaction-ramp     N pixel-frames of linear ramp at each window edge

Camera control is supplied separately as a pre-rendered Pi3X warp
(--warp-mp4 / --warp-mask), built from a POSE string by
build_pose_trajectory.py + render_warp.py. This driver never sees the POSE.

Toggle mechanics (all on the single AV pipeline):
  - video mask OFF  -> all-ones latent-frame weights (the mask never bites;
    action timing then comes from the prompt narrative alone).
  - audio mask OFF  -> audio_action_substring=None (audio ungated).
  - audio       OFF -> the co-generated audio track is dropped at encode time
    (the video is identical; LTX always jointly denoises the audio latents).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

HERE = Path(__file__).resolve().parent
LIB = HERE / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number  # noqa: E402
from ltx_core.types import VideoLatentShape, VideoPixelShape  # noqa: E402
from ltx_pipelines.utils.args import (  # noqa: E402
    ImageConditioningInput,
    default_2_stage_distilled_arg_parser,
    detect_checkpoint_path,
)
from ltx_pipelines.utils.constants import detect_params  # noqa: E402
from ltx_pipelines.utils.helpers import assert_resolution  # noqa: E402
from ltx_pipelines.utils.media_io import encode_video  # noqa: E402
from ltx_pipelines.utils.types import OffloadMode  # noqa: E402

# Self-contained pipeline modules (see lib/).
from mixed_velocity import build_latent_frame_weights  # noqa: E402
from temporal_attention_mask import TemporalAttentionMaskWindow  # noqa: E402
from temporal_attention_mask_av import TemporalAttentionMaskAVICLoraPipeline  # noqa: E402


# -----------------------------------------------------------------------------
# Arg surface
# -----------------------------------------------------------------------------
def add_warp_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("warp (camera trajectory)")
    g.add_argument("--warp-mp4", required=True, type=Path,
                   help="Pi3X warp video (stage-1 resolution; same fps as --frame-rate).")
    g.add_argument("--warp-mask", required=True, type=Path,
                   help="Visibility mask npz (key='visibility', [T,H,W] float32 in [0,1]).")
    g.add_argument("--cond-attn-strength", type=float, default=0.3,
                   help="Scalar on the per-token visibility attention mask. Locked operating "
                        "point is 0.3: geometry transmits, Pi3X pixel texture does not.")
    g.add_argument("--vis-threshold", type=float, default=0.1,
                   help="Binarise the visibility mask at this threshold before use "
                        "(training default 0.1). Set to a negative value to keep raw.")
    g.add_argument("--skip-stage-2", action="store_true",
                   help="Emit half-res Stage-1 output (faster iteration).")


def add_interaction_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("interaction (F-key social action)")
    g.add_argument("--interaction-prompt", type=str, required=True,
                   help="Substring of --prompt naming the VIDEO action (e.g. 'wave hello'). "
                        "Gated in time by the video temporal mask when --enable_videotemporalmask.")
    g.add_argument("--interaction-speech", type=str, default=None,
                   help="Substring of --prompt naming the SPOKEN phrase (e.g. \"'Hello!'\"). "
                        "Gated by the audio temporal mask when --enable_audio + "
                        "--enable_audiotemporalmask. Omit to leave speech ungated.")
    g.add_argument("--interaction-window", type=int, nargs=2, metavar=("START", "END"),
                   required=True,
                   help="Pixel-frame window [START, END) where the F-key is high.")
    g.add_argument("--interaction-ramp", type=int, default=12, metavar="N",
                   help="Linear ramp width (pixel frames) at each window edge (default 12 ~ 0.5s).")
    g.add_argument("--enable_audio", action="store_true",
                   help="Keep the co-generated speech track in the output.")
    g.add_argument("--enable_videotemporalmask", action="store_true",
                   help="Gate the video action token in time via the text cross-attention temporal mask.")
    g.add_argument("--enable_audiotemporalmask", action="store_true",
                   help="Gate the spoken phrase in time via the audio cross-attention mask.")


# -----------------------------------------------------------------------------
# Visibility npz loader  (inlined for self-containment)
# -----------------------------------------------------------------------------
def load_visibility_pixel_mask(
    mask_path: Path,
    num_frames: int,
    stage_1_height: int,
    stage_1_width: int,
    device: torch.device,
    dtype: torch.dtype,
    vis_threshold: float | None,
) -> torch.Tensor:
    """`mask.npz` -> (1, 1, num_frames, stage_1_h, stage_1_w) tensor in [0, 1]."""
    data = np.load(mask_path)
    if "visibility" not in data.files:
        raise KeyError(f"{mask_path} missing 'visibility' key; got {list(data.files)}")
    vis = data["visibility"].astype(np.float32)  # [T, H, W]
    if vis.ndim != 3:
        raise ValueError(f"visibility must be [T, H, W], got {vis.shape}")
    mask = torch.from_numpy(vis)[None, None].to(device=device, dtype=torch.float32)
    target = (int(num_frames), int(stage_1_height), int(stage_1_width))
    if mask.shape[2:] != target:
        mask = F.interpolate(mask, size=target, mode="trilinear", align_corners=False)
    mask = mask.clamp(0.0, 1.0)
    if vis_threshold is not None and vis_threshold > 0.0:
        mask = (mask > float(vis_threshold)).to(dtype=torch.float32)
    return mask.to(dtype=dtype)


# -----------------------------------------------------------------------------
# Pipeline construction + a single inference  (reused by batch_infer.py)
# -----------------------------------------------------------------------------
def build_pipeline(
    *,
    distilled_checkpoint_path,
    spatial_upsampler_path,
    gemma_root,
    loras=(),
    quantization=None,
    torch_compile=False,
    offload_mode=OffloadMode.NONE,
) -> TemporalAttentionMaskAVICLoraPipeline:
    """Instantiate the audio-video pipeline once. The expensive model load
    happens here; reuse the returned object across many `infer_one` calls.

    NOTE: offload_mode must be the OffloadMode.NONE *enum*, not Python None —
    the pipeline tests `offload_mode != OffloadMode.NONE` to decide whether to
    shuffle weights CPU<->GPU, so passing None silently enables offloading and
    makes every inference ~1.5x slower."""
    if offload_mode is None:
        offload_mode = OffloadMode.NONE
    return TemporalAttentionMaskAVICLoraPipeline(
        distilled_checkpoint_path=distilled_checkpoint_path,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=tuple(loras) if loras else (),
        quantization=quantization,
        torch_compile=torch_compile,
        offload_mode=offload_mode,
    )


@torch.inference_mode()
def infer_one(
    pipeline: TemporalAttentionMaskAVICLoraPipeline,
    *,
    prompt: str,
    image: str,
    warp_mp4: Path,
    warp_mask: Path,
    output_path: Path,
    interaction_prompt: str,
    interaction_window: tuple[int, int],
    interaction_speech: str | None = None,
    interaction_ramp: int = 12,
    enable_audio: bool = True,
    enable_videotemporalmask: bool = True,
    enable_audiotemporalmask: bool = True,
    num_frames: int = 241,
    frame_rate: float = 24.0,
    width: int = 1280,
    height: int = 704,
    seed: int = 1234,
    cond_attn_strength: float = 0.3,
    vis_threshold: float | None = 0.1,
    skip_stage_2: bool = False,
    enhance_prompt: bool = False,
) -> Path:
    """Run one (image, warp, prompt, interaction) job on an already-loaded
    `pipeline` and write `output_path`. Returns `output_path`."""
    warp_mp4, warp_mask, output_path = Path(warp_mp4), Path(warp_mask), Path(output_path)
    assert_resolution(height=height, width=width, is_two_stage=True)
    if not warp_mp4.exists():
        raise FileNotFoundError(f"warp mp4 missing: {warp_mp4}")
    if not warp_mask.exists():
        raise FileNotFoundError(f"warp mask missing: {warp_mask}")

    # ---- Validate interaction substrings live in the prompt ----
    if interaction_prompt not in prompt:
        raise ValueError(
            f"interaction_prompt {interaction_prompt!r} is not a substring of the prompt. "
            f"The temporal mask localises the action by verbatim substring match."
        )
    audio_mask_on = bool(enable_audio and enable_audiotemporalmask)
    if audio_mask_on:
        if interaction_speech is None:
            raise ValueError("audio temporal mask requires interaction_speech (a substring of the prompt).")
        if interaction_speech not in prompt:
            raise ValueError(f"interaction_speech {interaction_speech!r} is not a substring of the prompt.")
    if enable_audiotemporalmask and not enable_audio:
        logging.warning("[infer] audio temporal mask ignored (audio is disabled).")

    win_start, win_end = int(interaction_window[0]), int(interaction_window[1])
    if not (0 <= win_start <= win_end <= num_frames):
        raise ValueError(
            f"interaction_window={win_start},{win_end} must satisfy 0 <= START <= END <= num_frames ({num_frames})"
        )
    window = TemporalAttentionMaskWindow(
        start=win_start, end=win_end, ramp=int(interaction_ramp), max_weight=1.0,
    )

    # ---- Latent-frame weights for the video temporal mask ----
    stage_1_h, stage_1_w = height // 2, width // 2
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    stage_1_pix = VideoPixelShape(
        batch=1, frames=num_frames, width=stage_1_w, height=stage_1_h, fps=frame_rate,
    )
    f_lat = VideoLatentShape.from_pixel_shape(stage_1_pix).frames

    if enable_videotemporalmask:
        frame_weights = build_latent_frame_weights(
            num_pixel_frames=num_frames, num_latent_frames=f_lat, window=window,
        )
        logging.info(
            f"[infer] video mask ON: window=[{win_start},{win_end}) ramp={interaction_ramp} "
            f"-> F_lat={f_lat} active={(frame_weights > 0).sum().item()}/{f_lat}"
        )
    else:
        # All-ones weights -> the mask never bites (action timing from prompt only).
        frame_weights = torch.ones(f_lat, dtype=torch.float32)
        logging.info("[infer] video mask OFF: action timing via prompt narrative only.")

    audio_action_substring = interaction_speech if audio_mask_on else None
    audio_window = window if audio_mask_on else None
    logging.info(
        f"[infer] audio mask {'ON' if audio_mask_on else 'OFF'} (enable_audio={enable_audio})"
    )

    cond_attn_mask = load_visibility_pixel_mask(
        mask_path=warp_mask,
        num_frames=num_frames,
        stage_1_height=stage_1_h,
        stage_1_width=stage_1_w,
        device=pipeline.device,
        dtype=torch.bfloat16,
        vis_threshold=(None if (vis_threshold is None or vis_threshold < 0) else vis_threshold),
    )

    video, audio = pipeline(
        prompt=prompt,
        action_substring=interaction_prompt,
        frame_weights=frame_weights,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        images=[ImageConditioningInput(path=str(image), frame_idx=0, strength=1.0)],
        video_conditioning=[(str(warp_mp4), 1.0)],
        enhance_prompt=enhance_prompt,
        tiling_config=tiling_config,
        conditioning_attention_strength=float(cond_attn_strength),
        skip_stage_2=bool(skip_stage_2),
        conditioning_attention_mask=cond_attn_mask,
        audio_action_substring=audio_action_substring,
        audio_window=audio_window,
    )

    if not enable_audio:
        audio = None  # drop the co-generated speech track -> silent output.

    output_path.parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        video=video,
        fps=frame_rate,
        audio=audio,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
    )
    logging.info(f"[infer] wrote {output_path}")
    return output_path


# -----------------------------------------------------------------------------
# Single-run CLI
# -----------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    pipeline = build_pipeline(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        torch_compile=args.compile,
        offload_mode=args.offload_mode,
    )
    # The base parser allows multiple --image; this driver uses the first as the
    # scene first-frame (the rest, if any, are ignored — warp drives the motion).
    image_path = args.images[0].path if args.images else None
    if image_path is None:
        raise ValueError("an --image first-frame is required.")
    infer_one(
        pipeline,
        prompt=args.prompt,
        image=image_path,
        warp_mp4=args.warp_mp4,
        warp_mask=args.warp_mask,
        output_path=args.output_path,
        interaction_prompt=args.interaction_prompt,
        interaction_speech=args.interaction_speech,
        interaction_window=tuple(args.interaction_window),
        interaction_ramp=args.interaction_ramp,
        enable_audio=args.enable_audio,
        enable_videotemporalmask=args.enable_videotemporalmask,
        enable_audiotemporalmask=args.enable_audiotemporalmask,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        width=args.width,
        height=args.height,
        seed=args.seed,
        cond_attn_strength=args.cond_attn_strength,
        vis_threshold=args.vis_threshold,
        skip_stage_2=bool(args.skip_stage_2),
        enhance_prompt=args.enhance_prompt,
    )


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    checkpoint_path = detect_checkpoint_path(distilled=True)
    params = detect_params(checkpoint_path)
    parser = default_2_stage_distilled_arg_parser(params=params)
    add_warp_args(parser)
    add_interaction_args(parser)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(args)


if __name__ == "__main__":
    main()
