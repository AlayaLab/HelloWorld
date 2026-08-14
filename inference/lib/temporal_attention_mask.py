"""Temporal cross-attention mask on action tokens.

Training-free temporal control: add a per-``(latent_frame, action_text_token)``
additive bias on LTX's text cross-attention (``attn2``, the second cross-attn
inside each transformer block) so that action-related text tokens are visible
to the latent only inside a caller-specified pixel-frame F window. Action
vocabulary is selected as a substring of ``--prompt`` (e.g. ``"wave hello"``);
the rest of the prompt's tokens are unaffected at every frame.

A single prompt encoding and a single batched transformer forward pass per step
→ ~1× the per-step compute. The mask is built once before the denoising loop
and reused for every step.

This module exposes:

- :class:`TemporalAttentionMaskDenoiser` — :class:`SimpleDenoiser`-compatible
  callable that injects a precomputed ``context_mask`` into the video
  :class:`Modality` on every step.
- :class:`TemporalAttentionMaskICLoraPipeline` — :class:`ICLoraPipeline`
  subclass that builds the cross-attention mask from the prompt + window +
  action substring and runs both stages through the temporal-mask denoiser.
- :func:`locate_action_token_indices` — Gemma tokenizer → indices of
  ``substring``'s tokens inside the final ``video_encoding`` sequence.
- :func:`build_temporal_context_mask` — frame-weights + action indices →
  additive bias tensor of shape ``(1, 1, Nq, Nk)`` ready for
  ``Modality.context_mask``.

The frame-weight schedule reuses :func:`build_latent_frame_weights` so the F
window semantics match across the temporal-control paths.
"""
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning import (
    ConditioningItemAttentionStrengthWrapper,
    VideoConditionByReferenceLatent,
)
from ltx_core.model.video_vae import TilingConfig
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
from ltx_core.types import Audio, VideoLatentShape, VideoPixelShape
from ltx_core.utils import find_matching_file
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    modality_from_latent_state,
)
from ltx_pipelines.utils.types import ModalitySpec

# Window dataclass and frame-weight builder (causal-pooled to latent frames,
# optional linear ramp at edges) shared across the temporal-control paths.
from mixed_velocity import MixedVelocityWindow, build_latent_frame_weights

TemporalAttentionMaskWindow = MixedVelocityWindow


# ---------------------------------------------------------------------------
# Tokeniser helper: find action-token positions in the final video_encoding.
# ---------------------------------------------------------------------------
def locate_action_token_indices(
    gemma_root: str | Path,
    prompt: str,
    substring: str,
    max_length: int = 1024,
) -> list[int]:
    """Return final-sequence indices of ``substring``'s tokens.

    Process:

    1. Tokenise ``prompt.strip()`` with the same Gemma tokeniser
       :class:`LTXVGemmaTokenizer` uses (``padding="max_length"``,
       ``max_length=1024``, ``truncation=True``, left-padding).
    2. Use the fast tokeniser's ``offset_mapping`` to find tokens whose
       character span overlaps **any** occurrence of ``substring`` in the
       stripped prompt.
    3. Convert padded indices to final-sequence indices: the
       :class:`Embeddings1DConnector` replaces left-padding with learnable
       registers and reorders so valid prompt tokens occupy positions
       ``[0, n_valid)`` in tokenisation order. Hence
       ``final_pos = padded_pos − pad_count`` for each valid token.

    Raises:
        ValueError: substring not found in prompt, or its tokens were
            truncated past ``max_length=1024``.
        RuntimeError: tokeniser is not a fast tokeniser (no offset_mapping).
    """
    tokenizer_root = str(find_matching_file(str(gemma_root), "tokenizer.model").parent)
    # We only need the underlying HF tokenizer; LTXVGemmaTokenizer loads it cheaply
    # (no LLM weights), but we go through it so the path matches `PromptEncoder`
    # exactly (same model_max_length, same padding side, same pad token fallback).
    ltxv_tok = LTXVGemmaTokenizer(tokenizer_root, max_length=max_length)
    hf_tok = ltxv_tok.tokenizer

    stripped_prompt = prompt.strip()
    if substring not in stripped_prompt:
        raise ValueError(
            f"--temporal-attention-mask-tokens={substring!r} not found in --prompt "
            f"(after .strip()). Verify the substring is verbatim, including case "
            "and punctuation."
        )

    # Find ALL character spans of the substring; we mask each occurrence.
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        hit = stripped_prompt.find(substring, cursor)
        if hit < 0:
            break
        spans.append((hit, hit + len(substring)))
        cursor = hit + 1  # allow overlapping matches

    encoded = hf_tok(
        stripped_prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
    )
    if "offset_mapping" not in encoded:
        raise RuntimeError(
            "Gemma tokenizer did not return offset_mapping — a fast tokenizer is "
            f"required. Loaded from {tokenizer_root!r}."
        )
    offsets = encoded["offset_mapping"]
    attn = encoded["attention_mask"]
    pad_count = sum(1 for m in attn if m == 0)
    n_valid = max_length - pad_count

    padded_idx: list[int] = []
    for i, ((cs, ce), m) in enumerate(zip(offsets, attn, strict=True)):
        if m == 0 or cs == ce:
            # Padding tokens and zero-width special tokens (BOS/EOS) never overlap.
            continue
        for s_start, s_end in spans:
            if min(ce, s_end) > max(cs, s_start):
                padded_idx.append(i)
                break

    if not padded_idx:
        raise ValueError(
            f"--temporal-attention-mask-tokens={substring!r} matched character "
            f"span(s) {spans} in --prompt but no tokens overlap. Likely cause: "
            f"truncated past max_length={max_length} (prompt has {n_valid} valid tokens)."
        )

    return [i - pad_count for i in padded_idx]


# ---------------------------------------------------------------------------
# Cross-attention mask builder.
# ---------------------------------------------------------------------------
def build_temporal_context_mask(
    frame_weights: torch.Tensor,
    action_token_indices: Sequence[int],
    num_text_tokens: int,
    num_target_tokens: int,
    num_total_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    bias_magnitude: float | None = None,
) -> torch.Tensor:
    """Construct an additive bias on ``attn2``'s cross-attention logits.

    Shape: ``(1, 1, num_total_tokens, num_text_tokens)``. Broadcasts across
    attention heads and (if the surrounding batch is >1) along the batch dim.
    Passed through ``_prepare_attention_mask`` unchanged because float-valued
    masks are treated as additive biases verbatim.

    For a query latent token at frame ``t`` and a key text-token at index ``k``:

    - ``k ∈ action_token_indices`` → bias = ``(frame_weights[t] - 1) * bias_magnitude``.
      So ``frame_weights[t] = 1`` ⇒ bias = 0 (full attention); ``= 0`` ⇒
      bias ≈ ``-bias_magnitude`` (softmax → 0 ⇒ no attention).
    - Otherwise → bias = 0 (untouched).
    - Reference tokens (rows ``[num_target_tokens, num_total_tokens)``) get
      bias = 0 everywhere — they're frozen across denoising and their
      cross-attention output is dropped by ``post_process_latent``, so the
      bias would be inert. We zero those rows so the mask is also valid for
      diagnostic inspection.

    Args:
        frame_weights: shape ``(F_lat,)`` in [0, 1].
        action_token_indices: indices into the final ``video_encoding`` text
            axis. Empty list returns an all-zero bias (no-op).
        num_text_tokens: K — typically 1024 for LTX/Gemma-3.
        num_target_tokens: F_lat × H_lat × W_lat (patch_size=1) for the
            current stage.
        num_total_tokens: ``num_target_tokens + num_reference_tokens``.
        dtype, device: target placement; should match ``Modality.latent``.
        bias_magnitude: "no-attention" bias magnitude. Defaults to
            ``torch.finfo(dtype).max`` — matches ``convert_to_additive_mask``.
    """
    if frame_weights.ndim != 1:
        raise ValueError(f"frame_weights must be 1-D (F_lat,), got {tuple(frame_weights.shape)}")
    f_lat = int(frame_weights.shape[0])
    if num_target_tokens % f_lat != 0:
        raise ValueError(
            f"num_target_tokens={num_target_tokens} not divisible by F_lat={f_lat}"
        )
    if num_total_tokens < num_target_tokens:
        raise ValueError(
            f"num_total_tokens={num_total_tokens} < num_target_tokens={num_target_tokens}"
        )

    mask = torch.zeros((1, 1, num_total_tokens, num_text_tokens), dtype=dtype, device=device)
    if not action_token_indices:
        return mask
    if bias_magnitude is None:
        bias_magnitude = float(torch.finfo(dtype).max)

    tokens_per_frame = num_target_tokens // f_lat
    w_target = (
        frame_weights.to(device=device, dtype=dtype).repeat_interleave(tokens_per_frame)
    )  # (num_target,)
    target_bias = (w_target - 1.0) * bias_magnitude  # (num_target,) — 0 in-window, -mag out.

    col_idx = torch.tensor(list(action_token_indices), device=device, dtype=torch.long)
    bias_block = target_bias.unsqueeze(-1).expand(-1, col_idx.shape[0])  # (num_target, K_act)
    mask[0, 0, :num_target_tokens].index_copy_(dim=1, index=col_idx, source=bias_block)
    return mask


# ---------------------------------------------------------------------------
# Reference-token counter — peek at the conditionings to size the mask.
# ---------------------------------------------------------------------------
def _count_reference_tokens(conditionings: list) -> int:
    """Sum patchified token counts of any reference-video conditionings in the list.

    LTX-2.3's video patchifier uses patch_size=1, so the patchified token count
    equals ``F · H · W`` of the encoded reference latent.
    """
    total = 0
    for cond in conditionings:
        inner = cond.conditioning if isinstance(cond, ConditioningItemAttentionStrengthWrapper) else cond
        if isinstance(inner, VideoConditionByReferenceLatent):
            f, h, w = inner.latent.shape[-3], inner.latent.shape[-2], inner.latent.shape[-1]
            total += f * h * w
    return total


# ---------------------------------------------------------------------------
# Denoiser.
# ---------------------------------------------------------------------------
class TemporalAttentionMaskDenoiser:
    """Single batched transformer call per step, with a precomputed temporal
    cross-attention bias injected into the video :class:`Modality`.

    Matches the ``Denoiser`` protocol used by :class:`DiffusionStage.run` so it
    drops into the existing pipeline plumbing.
    """

    def __init__(
        self,
        v_context: torch.Tensor,
        a_context: torch.Tensor | None,
        video_context_mask: torch.Tensor,
    ) -> None:
        """
        Args:
            v_context: Video text-context, shape ``(1, T_text, D)``.
            a_context: Audio text-context (passed through verbatim — no
                temporal mask applied on the audio cross-attention path).
            video_context_mask: Additive bias of shape ``(1, 1, Nq, T_text)``
                — see :func:`build_temporal_context_mask`.
        """
        if video_context_mask.ndim != 4:
            raise ValueError(
                f"video_context_mask must be 4-D (1, 1, Nq, T_text); got {tuple(video_context_mask.shape)}"
            )
        if video_context_mask.shape[-1] != v_context.shape[1]:
            raise ValueError(
                f"video_context_mask T_text={video_context_mask.shape[-1]} does not match "
                f"v_context T_text={v_context.shape[1]}"
            )
        self.v_context = v_context
        self.a_context = a_context
        self.video_context_mask = video_context_mask

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
            num_total = int(video_state.latent.shape[1])
            mask = self.video_context_mask
            if mask.shape[2] != num_total:
                raise RuntimeError(
                    f"video_context_mask Nq={mask.shape[2]} does not match latent token "
                    f"count {num_total}; rebuild for the current stage."
                )
            if mask.device != v_mod.latent.device or mask.dtype != v_mod.latent.dtype:
                mask = mask.to(device=v_mod.latent.device, dtype=v_mod.latent.dtype)
                self.video_context_mask = mask
            v_mod = dataclasses.replace(v_mod, context_mask=mask)
        a_mod = (
            modality_from_latent_state(audio_state, self.a_context, sigma)
            if audio_state is not None and self.a_context is not None
            else None
        )
        return transformer(video=v_mod, audio=a_mod, perturbations=None)


# ---------------------------------------------------------------------------
# Pipeline subclass.
# ---------------------------------------------------------------------------
class TemporalAttentionMaskICLoraPipeline(ICLoraPipeline):
    """:class:`ICLoraPipeline` with a per-(latent_frame, action_text_token)
    additive bias on ``attn2`` applied in both stages.

    Stage-1 and stage-2 share the latent-frame count (the 2× spatial upsampler
    keeps the temporal axis intact), so the same per-latent-frame schedule is
    re-tiled to each stage's target-token count. Reference tokens appear only
    in stage-1 (stage-2 uses ``combined_image_conditionings`` with no reference
    video), so the per-stage Nq differs and the mask is rebuilt per stage.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # `ICLoraPipeline.__init__` stores everything it needs but doesn't
        # keep the gemma path; we need it to load the tokenizer for action-
        # token localisation. The driver passes gemma_root via kwargs.
        gemma_root = kwargs.get("gemma_root")
        if gemma_root is None:
            raise ValueError(
                "TemporalAttentionMaskICLoraPipeline requires `gemma_root` so the Gemma "
                "tokeniser can locate action-token indices."
            )
        self._gemma_root = str(gemma_root)

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
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        """Same signature as :meth:`ICLoraPipeline.__call__` plus:

        Args:
            action_substring: substring of ``prompt`` whose tokens get masked
                outside the F window. Forwarded to
                :func:`locate_action_token_indices`. Matches verbatim (case
                and punctuation sensitive) and applies to **all** occurrences.
            frame_weights: ``(F_lat,)`` tensor in [0, 1]. ``0`` ⇒ action tokens
                fully masked at that latent frame; ``1`` ⇒ unmasked.
            bias_magnitude: see :func:`build_temporal_context_mask`.
        """
        assert_resolution(height=height, width=width, is_two_stage=True)
        if not (0.0 <= conditioning_attention_strength <= 1.0):
            raise ValueError(
                f"conditioning_attention_strength must be in [0.0, 1.0], got {conditioning_attention_strength}"
            )
        if enhance_prompt:
            logging.warning(
                "[temporal-attn-mask] enhance_prompt is ignored — enhancing rewrites the "
                "prompt so the substring lookup for action tokens would be stale."
            )

        action_indices = locate_action_token_indices(
            gemma_root=self._gemma_root,
            prompt=prompt,
            substring=action_substring,
        )
        logging.info(
            f"[temporal-attn-mask] {len(action_indices)} action token(s) localised "
            f"(substring={action_substring!r}, final indices={action_indices[:8]}"
            f"{'...' if len(action_indices) > 8 else ''})"
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=False,
            enhance_prompt_image=None,
            enhance_prompt_seed=seed,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding
        num_text_tokens = int(video_context.shape[1])

        # ---- Stage 1 -------------------------------------------------------
        stage_1_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=frame_rate,
        )
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

        stage_1_mask = build_temporal_context_mask(
            frame_weights=frame_weights,
            action_token_indices=action_indices,
            num_text_tokens=num_text_tokens,
            num_target_tokens=stage_1_target_tokens,
            num_total_tokens=stage_1_num_total,
            dtype=self.dtype,
            device=self.device,
            bias_magnitude=bias_magnitude,
        )
        logging.info(
            f"[temporal-attn-mask] stage-1: F_lat={f_lat}, num_target={stage_1_target_tokens}, "
            f"num_reference={stage_1_num_reference}, T_text={num_text_tokens}, "
            f"active_frames={(frame_weights > 0).sum().item()}/{f_lat}"
        )

        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        stage_1_denoiser = TemporalAttentionMaskDenoiser(
            v_context=video_context,
            a_context=audio_context,
            video_context_mask=stage_1_mask,
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
            logging.info("[temporal-attn-mask] Skipping Stage 2 (--skip-stage-2 enabled)")
            decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
            decoded_audio = self.audio_decoder(audio_state.latent)
            return decoded_video, decoded_audio

        # ---- Stage 2 -------------------------------------------------------
        upscaled_video_latent = self.upsampler(video_state.latent[:1])
        stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_latent_shape = VideoLatentShape.from_pixel_shape(stage_2_shape)
        if stage_2_latent_shape.frames != f_lat:
            raise RuntimeError(
                f"Stage-2 F_lat ({stage_2_latent_shape.frames}) differs from stage-1 ({f_lat})."
            )
        stage_2_target_tokens = stage_2_latent_shape.token_count()
        # Stage 2 uses combined_image_conditionings only — no reference video tokens.
        stage_2_num_total = stage_2_target_tokens
        stage_2_mask = build_temporal_context_mask(
            frame_weights=frame_weights,
            action_token_indices=action_indices,
            num_text_tokens=num_text_tokens,
            num_target_tokens=stage_2_target_tokens,
            num_total_tokens=stage_2_num_total,
            dtype=self.dtype,
            device=self.device,
            bias_magnitude=bias_magnitude,
        )
        logging.info(
            f"[temporal-attn-mask] stage-2: F_lat={f_lat}, num_target={stage_2_target_tokens}"
        )

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
        stage_2_denoiser = TemporalAttentionMaskDenoiser(
            v_context=video_context,
            a_context=audio_context,
            video_context_mask=stage_2_mask,
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
    "TemporalAttentionMaskWindow",
    "TemporalAttentionMaskDenoiser",
    "TemporalAttentionMaskICLoraPipeline",
    "locate_action_token_indices",
    "build_temporal_context_mask",
    "build_latent_frame_weights",  # re-exported for driver-side ergonomics.
]
