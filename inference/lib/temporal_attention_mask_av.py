"""Extend the temporal cross-attention mask to the audio modality.

The video temporal mask gates the *video* cross-attention on a substring of
`--prompt` (typically `"wave hello"`) so the visual wave only lands inside the
pixel-frame F window. Without an audio gate, the spoken `'Hi!'/'Hello!'`
reliably appears but anchors to the model's default ~4 s slot regardless of the
F window. This module mirrors the same mask onto the audio modality so the
spoken word is gated the same way:

- The audio cross-attention (`audio_attn2` at
  `packages/ltx-core/src/ltx_core/model/transformer/transformer.py:274`)
  reads `audio.context_mask` exactly like the video path reads
  `video.context_mask` — same `_apply_text_cross_attention` call.
- Audio text encoding (`audio_encoding` from `EmbeddingsProcessor`) goes
  through `Embeddings1DConnector` with the *same* shape and padding
  convention as `video_encoding` (valid prompt tokens at positions
  `[0, n_valid)`, learnable registers in the tail). So
  `locate_action_token_indices` returns indices that are valid for both
  the video and the audio context — we reuse it verbatim.
- Audio has its own latent timeline: shape
  `AudioLatentShape(batch, channels, frames, mel_bins)` with
  `token_count == frames` and `frames ≈ duration_s · 25` (LTX-2.3's
  `latents_per_second = sample_rate / hop_length / audio_latent_downsample_factor
  = 16000 / 160 / 4 = 25`). So a 241-frame @ 24 fps video (≈10.04 s)
  decodes to ~251 audio latent frames. The per-pixel-frame F window is
  re-pooled to per-audio-latent-frame weights via simple linear binning
  (not causal — there's no asymmetry on the audio side analogous to the
  VAE's first-frame causal handling).

Independent substring + window per modality:
- Video substring (e.g. `"wave hello"`) typically targets the visual
  action keyword; gated to the F window via the existing
  `--temporal-attention-mask-tokens` + `--temporal-attention-mask-window`
  knobs.
- Audio substring (e.g. `"'Hello!'"` or `"calling out"`) targets the
  spoken word literal; gated via new
  `--audio-temporal-attention-mask-tokens` + (optional)
  `--audio-temporal-attention-mask-window`. The audio window defaults
  to the video window if not given; a separate audio window lets the wave and
  speech be scheduled independently.
"""
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.video_vae import TilingConfig
from ltx_core.types import Audio, AudioLatentShape, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    modality_from_latent_state,
)
from ltx_pipelines.utils.types import ModalitySpec

# Video-mask primitives — reused verbatim. The siblings live in this same `lib/`
# directory (helloworld is self-contained); ensure that dir is importable.
import sys as _sys
_LIB = Path(__file__).resolve().parent
if str(_LIB) not in _sys.path:
    _sys.path.insert(0, str(_LIB))
from temporal_attention_mask import (  # noqa: E402
    TemporalAttentionMaskICLoraPipeline,
    TemporalAttentionMaskWindow,
    _count_reference_tokens,
    build_temporal_context_mask,
    locate_action_token_indices,
)
from mixed_velocity import build_latent_frame_weights  # noqa: E402


# ---------------------------------------------------------------------------
# Per-audio-latent-frame weights from a pixel F window.
# ---------------------------------------------------------------------------
def build_audio_latent_frame_weights(
    num_pixel_frames: int,
    num_audio_latent_frames: int,
    window: TemporalAttentionMaskWindow,
) -> torch.Tensor:
    """Pixel F-window → per-audio-latent-frame weight tensor of shape
    ``(num_audio_latent_frames,)``.

    Reuses :func:`build_latent_frame_weights`'s pixel-side weight construction
    (zero outside, `max_weight` inside, optional linear ramp at the edges),
    then re-pools to the audio latent timeline by **linear binning**: audio
    latent frame `l` averages the pixel-frame block
    ``[round(l · F_pix / F_lat), round((l+1) · F_pix / F_lat))``. (No causal
    first-frame handling — that's specific to the video VAE's encoder.)
    """
    # First build the pixel-side weight by calling build_latent_frame_weights
    # with num_latent_frames=num_pixel_frames (trivial pooling = identity).
    pixel_w = build_latent_frame_weights(
        num_pixel_frames=num_pixel_frames,
        num_latent_frames=num_pixel_frames,
        window=window,
    ).numpy()  # (num_pixel_frames,) float32 in [0, max_weight]

    if num_audio_latent_frames <= 0:
        raise ValueError(f"num_audio_latent_frames must be > 0, got {num_audio_latent_frames}")

    audio_w = np.zeros(num_audio_latent_frames, dtype=np.float32)
    for l in range(num_audio_latent_frames):
        lo = int(round(l * num_pixel_frames / num_audio_latent_frames))
        hi = int(round((l + 1) * num_pixel_frames / num_audio_latent_frames))
        hi = max(hi, lo + 1)  # never empty bin
        hi = min(hi, num_pixel_frames)
        audio_w[l] = pixel_w[lo:hi].mean()
    return torch.from_numpy(audio_w)


# ---------------------------------------------------------------------------
# Denoiser: the video mask + an audio mask on the audio modality.
# ---------------------------------------------------------------------------
class TemporalAttentionMaskAVDenoiser:
    """Same shape as :class:`TemporalAttentionMaskDenoiser` (single batched
    forward per step) but also injects a precomputed bias into
    ``audio.context_mask`` so the audio text-cross-attention is gated the
    same way the video one is.

    ``audio_context_mask`` may be ``None`` — in that case audio is passed
    through verbatim (no audio gating).
    """

    def __init__(
        self,
        v_context: torch.Tensor,
        a_context: torch.Tensor | None,
        video_context_mask: torch.Tensor,
        audio_context_mask: torch.Tensor | None,
    ) -> None:
        if video_context_mask.ndim != 4:
            raise ValueError(
                f"video_context_mask must be 4-D (1, 1, Nq, T_text); got {tuple(video_context_mask.shape)}"
            )
        if video_context_mask.shape[-1] != v_context.shape[1]:
            raise ValueError(
                f"video_context_mask T_text={video_context_mask.shape[-1]} does not match "
                f"v_context T_text={v_context.shape[1]}"
            )
        if audio_context_mask is not None:
            if a_context is None:
                raise ValueError("audio_context_mask given but a_context is None.")
            if audio_context_mask.ndim != 4:
                raise ValueError(
                    f"audio_context_mask must be 4-D (1, 1, Na, T_text); got "
                    f"{tuple(audio_context_mask.shape)}"
                )
            if audio_context_mask.shape[-1] != a_context.shape[1]:
                raise ValueError(
                    f"audio_context_mask T_text={audio_context_mask.shape[-1]} does not match "
                    f"a_context T_text={a_context.shape[1]}"
                )
        self.v_context = v_context
        self.a_context = a_context
        self.video_context_mask = video_context_mask
        self.audio_context_mask = audio_context_mask

    def __call__(
        self,
        transformer,
        video_state,
        audio_state,
        sigmas: torch.Tensor,
        step_index: int,
    ):
        sigma = sigmas[step_index]
        v_mod = None
        if video_state is not None:
            v_mod = modality_from_latent_state(video_state, self.v_context, sigma)
            num_total_v = int(video_state.latent.shape[1])
            v_mask = self.video_context_mask
            if v_mask.shape[2] != num_total_v:
                raise RuntimeError(
                    f"video_context_mask Nq={v_mask.shape[2]} does not match latent token "
                    f"count {num_total_v}; rebuild for the current stage."
                )
            if v_mask.device != v_mod.latent.device or v_mask.dtype != v_mod.latent.dtype:
                v_mask = v_mask.to(device=v_mod.latent.device, dtype=v_mod.latent.dtype)
                self.video_context_mask = v_mask
            v_mod = dataclasses.replace(v_mod, context_mask=v_mask)

        a_mod = None
        if audio_state is not None and self.a_context is not None:
            a_mod = modality_from_latent_state(audio_state, self.a_context, sigma)
            if self.audio_context_mask is not None:
                num_total_a = int(audio_state.latent.shape[1])
                a_mask = self.audio_context_mask
                if a_mask.shape[2] != num_total_a:
                    raise RuntimeError(
                        f"audio_context_mask Na={a_mask.shape[2]} does not match audio "
                        f"latent token count {num_total_a}; rebuild for the current stage."
                    )
                if a_mask.device != a_mod.latent.device or a_mask.dtype != a_mod.latent.dtype:
                    a_mask = a_mask.to(device=a_mod.latent.device, dtype=a_mod.latent.dtype)
                    self.audio_context_mask = a_mask
                a_mod = dataclasses.replace(a_mod, context_mask=a_mask)

        return transformer(video=v_mod, audio=a_mod, perturbations=None)


# ---------------------------------------------------------------------------
# Pipeline subclass.
# ---------------------------------------------------------------------------
class TemporalAttentionMaskAVICLoraPipeline(TemporalAttentionMaskICLoraPipeline):
    """The video temporal-mask pipeline + an optional audio-side cross-attention mask.

    When ``audio_action_substring`` is given, an
    ``(1, 1, num_audio_tokens, T_text)`` additive bias is built from the
    audio F window and injected into ``audio.context_mask`` in both stages.
    Audio token count is taken from
    :meth:`AudioLatentShape.from_video_pixel_shape` (LTX-2.3 default rates:
    `sample_rate=16000, hop_length=160, audio_latent_downsample_factor=4` →
    25 latents/s).

    Falls back to video-only masking when
    ``audio_action_substring is None``.
    """

    def __call__(  # noqa: PLR0913, PLR0915
        self,
        prompt: str,
        action_substring: str,
        frame_weights: torch.Tensor,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images,
        video_conditioning,
        enhance_prompt: bool = False,
        tiling_config: TilingConfig | None = None,
        conditioning_attention_strength: float = 1.0,
        skip_stage_2: bool = False,
        conditioning_attention_mask: torch.Tensor | None = None,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        bias_magnitude: float | None = None,
        audio_action_substring: str | None = None,
        audio_window: TemporalAttentionMaskWindow | None = None,
        audio_bias_magnitude: float | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        """Video temporal-mask signature + audio-mask args:

        Args:
            audio_action_substring: substring of ``prompt`` whose tokens get
                masked outside ``audio_window`` in the AUDIO cross-attention.
                If ``None``, no audio mask is applied (video-only).
            audio_window: pixel-frame F window for the audio mask. If
                ``None`` (and ``audio_action_substring`` is set), reuses the
                video F window built from ``frame_weights`` — but `frame_weights`
                lives in *video* latent time, so we re-derive the pixel window
                from the supplied ``audio_window`` argument when given. For
                "audio window == video window", pass the same ``MixedVelocityWindow``
                that was passed to ``build_latent_frame_weights``.
            audio_bias_magnitude: ``-bias_magnitude`` is the additive bias
                outside the audio window. Default ``finfo(bf16).max``.
        """
        assert_resolution(height=height, width=width, is_two_stage=True)
        if not (0.0 <= conditioning_attention_strength <= 1.0):
            raise ValueError(
                f"conditioning_attention_strength must be in [0.0, 1.0], got {conditioning_attention_strength}"
            )
        if enhance_prompt:
            logging.warning(
                "[temporal-attn-mask-av] enhance_prompt is ignored — enhancing rewrites the "
                "prompt so the substring lookup for action tokens would be stale."
            )

        # ---- Action-token localisation (video + audio share the indices) ----
        video_action_indices = locate_action_token_indices(
            gemma_root=self._gemma_root,
            prompt=prompt,
            substring=action_substring,
        )
        logging.info(
            f"[temporal-attn-mask-av] video: {len(video_action_indices)} token(s) "
            f"(substring={action_substring!r}, indices={video_action_indices[:8]}"
            f"{'...' if len(video_action_indices) > 8 else ''})"
        )

        audio_action_indices: list[int] = []
        if audio_action_substring is not None:
            audio_action_indices = locate_action_token_indices(
                gemma_root=self._gemma_root,
                prompt=prompt,
                substring=audio_action_substring,
            )
            logging.info(
                f"[temporal-attn-mask-av] audio: {len(audio_action_indices)} token(s) "
                f"(substring={audio_action_substring!r}, indices={audio_action_indices[:8]}"
                f"{'...' if len(audio_action_indices) > 8 else ''})"
            )

        # ---- Prompt encoding + audio latent shape ----
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=False,
            enhance_prompt_image=None,
            enhance_prompt_seed=seed,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding
        if audio_context is None and audio_action_substring is not None:
            raise RuntimeError(
                "Audio modality is disabled on this build of LTX (audio_encoding is None) "
                "but --audio-temporal-attention-mask-tokens was given. Aborting."
            )
        num_text_tokens = int(video_context.shape[1])

        stage_1_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=frame_rate,
        )
        # Both stages use the same audio latent timeline.
        audio_latent_shape = AudioLatentShape.from_video_pixel_shape(stage_1_shape)
        num_audio_tokens = audio_latent_shape.token_count()
        logging.info(
            f"[temporal-attn-mask-av] audio latent: frames={audio_latent_shape.frames} "
            f"(token_count={num_audio_tokens})  text_tokens={num_text_tokens}"
        )

        audio_frame_weights: torch.Tensor | None = None
        if audio_action_substring is not None:
            if audio_window is None:
                raise ValueError(
                    "audio_action_substring given but audio_window is None; pass the same "
                    "MixedVelocityWindow used for the video frame_weights to mirror the "
                    "video F window onto audio."
                )
            audio_frame_weights = build_audio_latent_frame_weights(
                num_pixel_frames=num_frames,
                num_audio_latent_frames=num_audio_tokens,
                window=audio_window,
            )
            logging.info(
                f"[temporal-attn-mask-av] audio frame weights: "
                f"active={(audio_frame_weights > 0).sum().item()}/{num_audio_tokens}, "
                f"min={audio_frame_weights.min().item():.2f}, "
                f"max={audio_frame_weights.max().item():.2f}"
            )

        # ---- Stage 1 -----------------------------------------------------
        stage_1_latent_shape = VideoLatentShape.from_pixel_shape(stage_1_shape)
        f_lat = stage_1_latent_shape.frames
        if frame_weights.shape != (f_lat,):
            raise ValueError(
                f"frame_weights shape {tuple(frame_weights.shape)} does not match stage-1 "
                f"F_lat={f_lat}. Pass `build_latent_frame_weights(num_pixel_frames=num_frames, "
                f"num_latent_frames=F_lat, ...)`."
            )
        stage_1_target_tokens = stage_1_latent_shape.token_count()

        stage_1_conditionings = self.image_conditioner(
            lambda enc: self._create_conditionings(
                images=images,
                video_conditioning=video_conditioning,
                height=stage_1_shape.height,
                width=stage_1_shape.width,
                video_encoder=enc,
                num_frames=num_frames,
                conditioning_attention_strength=conditioning_attention_strength,
                conditioning_attention_mask=conditioning_attention_mask,
            )
        )
        stage_1_num_reference = _count_reference_tokens(stage_1_conditionings)
        stage_1_num_total = stage_1_target_tokens + stage_1_num_reference

        stage_1_v_mask = build_temporal_context_mask(
            frame_weights=frame_weights,
            action_token_indices=video_action_indices,
            num_text_tokens=num_text_tokens,
            num_target_tokens=stage_1_target_tokens,
            num_total_tokens=stage_1_num_total,
            dtype=self.dtype,
            device=self.device,
            bias_magnitude=bias_magnitude,
        )
        stage_1_a_mask = None
        if audio_action_indices:
            stage_1_a_mask = build_temporal_context_mask(
                frame_weights=audio_frame_weights,
                action_token_indices=audio_action_indices,
                num_text_tokens=num_text_tokens,
                num_target_tokens=num_audio_tokens,
                num_total_tokens=num_audio_tokens,  # audio has no reference-token concept here
                dtype=self.dtype,
                device=self.device,
                bias_magnitude=audio_bias_magnitude,
            )

        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        stage_1_denoiser = TemporalAttentionMaskAVDenoiser(
            v_context=video_context,
            a_context=audio_context,
            video_context_mask=stage_1_v_mask,
            audio_context_mask=stage_1_a_mask,
        )
        video_state, audio_state = self.stage_1(
            denoiser=stage_1_denoiser,
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=stage_1_shape.width,
            height=stage_1_shape.height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
            audio=ModalitySpec(context=audio_context),
        )

        if skip_stage_2:
            logging.info("[temporal-attn-mask-av] Skipping Stage 2 (--skip-stage-2 enabled)")
            decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
            decoded_audio = self.audio_decoder(audio_state.latent)
            return decoded_video, decoded_audio

        # ---- Stage 2 -----------------------------------------------------
        upscaled_video_latent = self.upsampler(video_state.latent[:1])
        stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_latent_shape = VideoLatentShape.from_pixel_shape(stage_2_shape)
        if stage_2_latent_shape.frames != f_lat:
            raise RuntimeError(
                f"Stage-2 F_lat ({stage_2_latent_shape.frames}) differs from stage-1 ({f_lat})."
            )
        stage_2_target_tokens = stage_2_latent_shape.token_count()
        stage_2_num_total = stage_2_target_tokens  # no reference video in stage 2

        stage_2_v_mask = build_temporal_context_mask(
            frame_weights=frame_weights,
            action_token_indices=video_action_indices,
            num_text_tokens=num_text_tokens,
            num_target_tokens=stage_2_target_tokens,
            num_total_tokens=stage_2_num_total,
            dtype=self.dtype,
            device=self.device,
            bias_magnitude=bias_magnitude,
        )
        # Stage 2 audio mask = stage 1 (same audio latent shape).
        stage_2_a_mask = stage_1_a_mask

        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_2_shape.height,
                width=stage_2_shape.width,
                video_encoder=enc,
                dtype=self.dtype,
                device=self.device,
            )
        )
        stage_2_denoiser = TemporalAttentionMaskAVDenoiser(
            v_context=video_context,
            a_context=audio_context,
            video_context_mask=stage_2_v_mask,
            audio_context_mask=stage_2_a_mask,
        )
        video_state, audio_state = self.stage_2(
            denoiser=stage_2_denoiser,
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=audio_context,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio


__all__ = [
    "TemporalAttentionMaskAVDenoiser",
    "TemporalAttentionMaskAVICLoraPipeline",
    "build_audio_latent_frame_weights",
]
