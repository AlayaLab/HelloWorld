"""Per-frame two-prompt x0 blend + shared F-window utilities.

This module provides the F-window primitives used across the temporal-control
paths — :class:`MixedVelocityWindow` (the pixel-frame action window dataclass)
and :func:`build_latent_frame_weights` (pixel F-window → per-latent-frame
weight) — which the cross-attention temporal-mask pipeline imports.

It also implements a training-free two-prompt blend: encode a "scene-only" and
a "scene + action" prompt, run both through the transformer at each denoising
step in a single batched forward pass, then blend the two predicted-x0 tensors
per latent-frame by the F-window weight (F=0 → scene only, F=1 → action). This
costs ~2× the per-step transformer compute. The unified inference driver uses
the cross-attention temporal mask instead, but the classes here are kept for
completeness:
- :class:`MixedVelocityDenoiser` — drop-in replacement for `SimpleDenoiser`.
- :class:`MixedVelocityICLoraPipeline` — `ICLoraPipeline` subclass using the
  mixed-velocity denoiser in both stages.
- :func:`build_latent_frame_weights` — pixel F-window → per-latent-frame weight.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationConfig
from ltx_core.model.video_vae import TilingConfig
from ltx_core.types import Audio, VideoLatentShape, VideoPixelShape
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import _repeat_state  # internal but stable
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    modality_from_latent_state,
)
from ltx_pipelines.utils.types import ModalitySpec


# -----------------------------------------------------------------------------
# F-window -> per-latent-frame weight
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MixedVelocityWindow:
    """Pixel-frame F window. start ≤ end; both in [0, num_pixel_frames]."""
    start: int
    end: int
    ramp: int = 0  # pixel-frame count for the linear edge ramp (each side)
    max_weight: float = 1.0  # weight inside the window (1.0 = full action prompt)


def build_latent_frame_weights(
    num_pixel_frames: int,
    num_latent_frames: int,
    window: MixedVelocityWindow,
) -> torch.Tensor:
    """Pixel F-window → per-latent-frame weight tensor of shape ``(F_lat,)``.

    The pixel-side weight is `max_weight` inside ``[start, end)``, 0 outside,
    with an optional linear ramp of `ramp` pixel-frames at each edge. The
    latent-side weight is then computed by **causal temporal pooling** matching
    the LTX VAE: latent frame 0 takes pixel frame 0; latent frame `l>=1`
    averages the pixel-frame block ``[1 + (l-1)·t, 1 + l·t)`` where
    ``t = (F_pix - 1) / (F_lat - 1)``.

    Returns a float32 CPU tensor; caller is expected to cast/move it.
    """
    if not (0 <= window.start <= window.end <= num_pixel_frames):
        raise ValueError(
            f"window=[{window.start}, {window.end}) out of bounds [0, {num_pixel_frames}]"
        )
    if not (0.0 <= window.max_weight <= 1.0):
        raise ValueError(f"max_weight must be in [0, 1], got {window.max_weight}")
    if window.ramp < 0:
        raise ValueError(f"ramp must be ≥ 0, got {window.ramp}")

    # Pixel-frame weights.
    pixel_w = np.zeros(num_pixel_frames, dtype=np.float32)
    pixel_w[window.start : window.end] = window.max_weight

    if window.ramp > 0:
        # Left ramp: pixels [start-ramp, start) rise from 0 to max_weight.
        left_lo = max(0, window.start - window.ramp)
        for i in range(left_lo, window.start):
            t = (i - (window.start - window.ramp)) / window.ramp  # 0 → 1
            pixel_w[i] = max(pixel_w[i], t * window.max_weight)
        # Right ramp: pixels [end, end+ramp) fall from max_weight to 0.
        right_hi = min(num_pixel_frames, window.end + window.ramp)
        for i in range(window.end, right_hi):
            t = 1.0 - (i - window.end) / window.ramp  # 1 → 0
            pixel_w[i] = max(pixel_w[i], t * window.max_weight)

    # Causal temporal pooling to latent frames.
    if num_pixel_frames == num_latent_frames:
        latent_w = pixel_w.copy()
    else:
        if num_latent_frames < 1:
            raise ValueError(f"num_latent_frames must be ≥ 1, got {num_latent_frames}")
        if (num_pixel_frames - 1) % (num_latent_frames - 1) != 0:
            raise ValueError(
                f"VAE temporal-compat broken: (F_pix-1)={num_pixel_frames - 1} not "
                f"divisible by (F_lat-1)={num_latent_frames - 1}"
            )
        t_factor = (num_pixel_frames - 1) // (num_latent_frames - 1)
        latent_w = np.zeros(num_latent_frames, dtype=np.float32)
        latent_w[0] = pixel_w[0]
        for l in range(1, num_latent_frames):
            lo = 1 + (l - 1) * t_factor
            hi = 1 + l * t_factor
            latent_w[l] = pixel_w[lo:hi].mean()

    return torch.from_numpy(latent_w)


def expand_to_target_tokens(
    frame_weights: torch.Tensor,
    num_latent_frames: int,
    num_target_tokens: int,
) -> torch.Tensor:
    """Expand ``(F_lat,)`` weights to ``(num_target_tokens,)`` assuming
    patchifier token order ``(f, h, w)``: each frame occupies
    ``num_target_tokens // F_lat`` consecutive token positions.
    """
    if frame_weights.shape != (num_latent_frames,):
        raise ValueError(f"frame_weights shape {frame_weights.shape} != ({num_latent_frames},)")
    if num_target_tokens % num_latent_frames != 0:
        raise ValueError(
            f"num_target_tokens={num_target_tokens} not divisible by F_lat={num_latent_frames}"
        )
    tokens_per_frame = num_target_tokens // num_latent_frames
    return frame_weights.repeat_interleave(tokens_per_frame)


# -----------------------------------------------------------------------------
# Denoiser
# -----------------------------------------------------------------------------
class MixedVelocityDenoiser:
    """Single batched transformer call with two prompt contexts, then a per-token
    x0 blend `(1 - w) · v_scene + w · v_action`.

    Matches the `SimpleDenoiser` protocol so it drops into `DiffusionStage.run`
    without other changes.
    """

    def __init__(
        self,
        v_context_scene: torch.Tensor,
        v_context_action: torch.Tensor,
        a_context: torch.Tensor | None,
        target_token_weights: torch.Tensor,
    ) -> None:
        """
        Args:
            v_context_scene: Video text-context for prompt A (no action), shape
                ``(1, T_text, D)``.
            v_context_action: Video text-context for prompt B (with action), same shape.
            a_context: Audio text-context (shared by both passes), or ``None``.
            target_token_weights: ``(num_target_tokens,)`` tensor in [0, 1].
                weight=0 ⇒ pure scene prediction; weight=1 ⇒ pure action prediction.
                Length must equal the number of target tokens (i.e. F_lat·H_lat·W_lat
                for the corresponding diffusion stage).
        """
        if v_context_scene.shape != v_context_action.shape:
            raise ValueError(
                f"prompt context shape mismatch: scene={v_context_scene.shape} vs "
                f"action={v_context_action.shape}. Both prompts must share the same "
                "padded token-count (PromptEncoder pads to a fixed length)."
            )
        self.v_context_scene = v_context_scene
        self.v_context_action = v_context_action
        self.a_context = a_context
        # Keep weights on CPU until the first forward pass; we'll move/cast then.
        self.target_token_weights = target_token_weights
        self._token_weight_cache: torch.Tensor | None = None

    def _get_token_weight(self, ref_tensor: torch.Tensor, num_total_tokens: int) -> torch.Tensor:
        """Return a ``(1, num_total_tokens, 1)`` weight tensor on `ref_tensor`'s
        device/dtype. Target tokens get `self.target_token_weights`; reference
        tokens (appended after target by `VideoConditionByReferenceLatent`) get
        weight 1.0 — their predicted x0 is ignored by the sampler (denoise_mask=0),
        so the value does not matter, but we use 1.0 to keep the math obvious.
        """
        if (
            self._token_weight_cache is not None
            and self._token_weight_cache.shape[1] == num_total_tokens
            and self._token_weight_cache.device == ref_tensor.device
            and self._token_weight_cache.dtype == ref_tensor.dtype
        ):
            return self._token_weight_cache

        num_target = int(self.target_token_weights.shape[0])
        if num_target > num_total_tokens:
            raise RuntimeError(
                f"target_token_weights ({num_target}) longer than total tokens "
                f"({num_total_tokens}) — check stage/resolution wiring."
            )
        w_target = self.target_token_weights.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        if num_target < num_total_tokens:
            w_ref = torch.ones(num_total_tokens - num_target, device=ref_tensor.device, dtype=ref_tensor.dtype)
            w_full = torch.cat([w_target, w_ref], dim=0)
        else:
            w_full = w_target
        self._token_weight_cache = w_full.view(1, num_total_tokens, 1)
        return self._token_weight_cache

    def __call__(
        self,
        transformer,
        video_state,
        audio_state,
        sigmas: torch.Tensor,
        step_index: int,
    ):
        if video_state is None:
            # No video to denoise — fall back to a single audio pass with action context.
            sigma = sigmas[step_index]
            a_mod = (
                modality_from_latent_state(audio_state, self.a_context, sigma)
                if audio_state is not None
                else None
            )
            _, denoised_a = transformer(video=None, audio=a_mod, perturbations=None)
            return None, denoised_a

        sigma = sigmas[step_index]
        n = 2  # scene + action

        # Repeat state along batch dim (matches `_repeat_state` semantics: pass-0 = scene, pass-1 = action).
        v_state_rep = _repeat_state(video_state, n)
        v_ctx_cat = torch.cat([self.v_context_scene, self.v_context_action], dim=0)
        v_sigma = sigma.expand(video_state.latent.shape[0] * n)
        v_mod = modality_from_latent_state(v_state_rep, v_ctx_cat, v_sigma)

        a_mod = None
        if audio_state is not None and self.a_context is not None:
            a_state_rep = _repeat_state(audio_state, n)
            a_ctx_cat = torch.cat([self.a_context, self.a_context], dim=0)
            a_sigma = sigma.expand(audio_state.latent.shape[0] * n)
            a_mod = modality_from_latent_state(a_state_rep, a_ctx_cat, a_sigma)

        # BatchSplitAdapter (wrapping the transformer with max_batch_size=1) splits
        # the batch and slices the per-batch perturbations along with it; passing
        # `perturbations=None` here would crash inside `_split_perturbations`.
        perturbations = BatchedPerturbationConfig([PerturbationConfig.empty()] * n)
        all_v, all_a = transformer(video=v_mod, audio=a_mod, perturbations=perturbations)

        v_scene, v_action = all_v.chunk(n, dim=0)
        w = self._get_token_weight(v_scene, num_total_tokens=v_scene.shape[1])
        denoised_v = (1.0 - w) * v_scene + w * v_action

        denoised_a = None
        if all_a is not None:
            # Audio context is identical across passes → outputs are the same; just take pass 0.
            denoised_a = all_a.chunk(n, dim=0)[0]

        return denoised_v, denoised_a


# -----------------------------------------------------------------------------
# Pipeline subclass
# -----------------------------------------------------------------------------
class MixedVelocityICLoraPipeline(ICLoraPipeline):
    """`ICLoraPipeline` with a per-frame two-prompt x0 blend in both stages.

    Stage-1 and stage-2 share the same latent-frame count (the 2× spatial
    upsampler does not touch the temporal axis), so the same per-latent-frame
    weight schedule applies to both — we just re-tile it to each stage's
    target-token count via :func:`expand_to_target_tokens`.
    """

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        scene_prompt: str,
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
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        """Same signature as `ICLoraPipeline.__call__` plus:

        Args:
            prompt: the "scene + action" prompt (prompt B). This is the one that
                fully describes the desired behaviour including the action verb.
            scene_prompt: the "scene-only" prompt (prompt A). Should describe
                the scene + camera + lighting *without* mentioning the action,
                so its predicted velocity carries no action signal.
            frame_weights: ``(F_lat,)`` tensor in [0, 1] — the per-latent-frame
                weight of the action prompt. `0` = use scene-only velocity at
                this frame; `1` = use action velocity; intermediate values
                linearly blend.
        """
        assert_resolution(height=height, width=width, is_two_stage=True)
        if not (0.0 <= conditioning_attention_strength <= 1.0):
            raise ValueError(
                f"conditioning_attention_strength must be in [0.0, 1.0], got {conditioning_attention_strength}"
            )
        if enhance_prompt:
            # Enhancing only one of the two prompts would mismatch them; disabling
            # for both is safer than silently enhancing one.
            logging.warning(
                "[mixed-velocity] enhance_prompt is ignored — the two-prompt blend "
                "needs scene/action prompts to share narrative scope verbatim."
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        # Encode both prompts in one PromptEncoder call → identical padding length.
        ctx_scene, ctx_action = self.prompt_encoder(
            [scene_prompt, prompt],
            enhance_first_prompt=False,
            enhance_prompt_image=None,
            enhance_prompt_seed=seed,
        )
        v_ctx_scene = ctx_scene.video_encoding
        v_ctx_action = ctx_action.video_encoding
        # Audio context comes from the action prompt (the "primary" one).
        a_context = ctx_action.audio_encoding

        # Stage 1 ---------------------------------------------------------------
        stage_1_shape = VideoPixelShape(batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=frame_rate)
        stage_1_latent_shape = VideoLatentShape.from_pixel_shape(stage_1_shape)
        f_lat = stage_1_latent_shape.frames
        if frame_weights.shape != (f_lat,):
            raise ValueError(
                f"frame_weights shape {tuple(frame_weights.shape)} does not match stage-1 F_lat={f_lat}. "
                "Pass `build_latent_frame_weights(num_pixel_frames=num_frames, num_latent_frames=F_lat, ...)`."
            )
        stage_1_target_tokens = stage_1_latent_shape.token_count()
        stage_1_token_w = expand_to_target_tokens(frame_weights, f_lat, stage_1_target_tokens)

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

        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        stage_1_denoiser = MixedVelocityDenoiser(
            v_context_scene=v_ctx_scene,
            v_context_action=v_ctx_action,
            a_context=a_context,
            target_token_weights=stage_1_token_w,
        )
        video_state, audio_state = self.stage_1(
            denoiser=stage_1_denoiser,
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=stage_1_shape.width,
            height=stage_1_shape.height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=v_ctx_action, conditionings=stage_1_conditionings),
            audio=ModalitySpec(context=a_context),
        )

        if skip_stage_2:
            logging.info("[mixed-velocity] Skipping Stage 2 (--skip-stage-2 enabled)")
            decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
            decoded_audio = self.audio_decoder(audio_state.latent)
            return decoded_video, decoded_audio

        # Stage 2 ---------------------------------------------------------------
        upscaled_video_latent = self.upsampler(video_state.latent[:1])
        stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_latent_shape = VideoLatentShape.from_pixel_shape(stage_2_shape)
        if stage_2_latent_shape.frames != f_lat:
            raise RuntimeError(
                f"Stage-2 F_lat ({stage_2_latent_shape.frames}) differs from stage-1 ({f_lat}); "
                "the upsampler should keep the temporal axis intact. Unexpected."
            )
        stage_2_target_tokens = stage_2_latent_shape.token_count()
        stage_2_token_w = expand_to_target_tokens(frame_weights, f_lat, stage_2_target_tokens)

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
        stage_2_denoiser = MixedVelocityDenoiser(
            v_context_scene=v_ctx_scene,
            v_context_action=v_ctx_action,
            a_context=a_context,
            target_token_weights=stage_2_token_w,
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
                context=v_ctx_action,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=a_context,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio
