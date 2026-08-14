#!/usr/bin/env bash
# =====================================================================
# run_helloworld.sh — single-clip HelloWorld inference.
#
# Pipeline (4 stages, 2 conda envs):
#   1. camera trajectory   -> camera_poses.npz (+ phases.json)  (warp env)
#   2. Pi3X warp render    -> warp.mp4 + mask.npz               (warp env)
#   3. LTX-2.3 + our LoRA  -> gen.mp4                           (ltx  env)
#   4. keyboard UI overlay -> gen_ui.mp4                        (ui   env)
#
# Inputs you control (env-overridable, see USER CONFIG):
#   IMAGE          first-frame image (the scene + character)
#   TEXT_PROMPT    the full text prompt (scene + character + action + speech)
#   TRAJECTORY     the camera trajectory: an .npz holding `camera_poses`,
#                  one 4x4 camera-to-world matrix per frame. Give this OR POSE.
#   POSE           convenience alternative to TRAJECTORY: a WASD/arrow string
#                  compiled into a trajectory for you. Movements:
#                  w/s (fwd/back) a/d (left/right strafe); rotations:
#                  left/right (yaw) up/down (pitch). "<action>-<N>[, ...]".
#                  N is a strict frame count: each unit = 8 frames, frame 0 is
#                  a static lead-in, so the N's must sum to 30 (= 241 frames).
#                  e.g. POSE='w-3, left-6, right-12, left-6, w-3'  (24f / 48f / 96f / 48f / 24f)
#                  Per-frame SPEED/amplitude is set by YAW_RATE/FWD_RATE/... (optional).
#   PHASES         optional with TRAJECTORY: key-press timeline JSON used only
#                  for the keyboard-overlay video (auto-derived from POSE;
#                  without it the overlay stage is skipped).
#   INTERACTION_TIME   "start,end" in SECONDS — the F-key high window.
#   INTERACTION_PROMPT video action, a substring of TEXT_PROMPT (e.g. "wave hello").
#   INTERACTION_SPEECH spoken phrase, a substring of TEXT_PROMPT (e.g. "'Hello!'").
#
# Ablation toggles (true/false):
#   ENABLE_AUDIO              keep the co-generated speech track
#   ENABLE_VIDEOTEMPORALMASK  cross-attn temporal control of the video action
#   ENABLE_AUDIOTEMPORALMASK  cross-attn temporal control of the speech
#   WITH_UI                   also emit the keyboard-overlay video
#
# Output (all in $OUTPUT_DIR):
#   params.txt      every input parameter for this run
#   gen.mp4         the generated video (always)
#   gen_ui.mp4      gen.mp4 + WASD/arrow/F keyboard overlay (WITH_UI, default on)
#   warp/warp.mp4   the camera-warp condition video (WITH_WARP, default on)
#   trajectory/     camera_poses.npz (+ phases.json when built from POSE)
#
# Run:                bash run_helloworld.sh
# Override any field:  OUTPUT_DIR=... IMAGE=... POSE='w-30' bash run_helloworld.sh
# Detached:
#   setsid nohup bash run_helloworld.sh > "$OUTPUT_DIR/run.out" 2>&1 < /dev/null & disown
# =====================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# =====================================================================
# USER CONFIG — edit freely, or override via env vars
# =====================================================================
: "${OUTPUT_DIR:=${HERE}/outputs/demo}"
: "${IMAGE:=${HERE}/../assets/examples/lakeside_hikers/input.jpg}"

: "${TEXT_PROMPT:=A man in a dark business suit standing on a wide empty Los Angeles avenue lined with palm trees. Eye-level long take. Midway through the shot, the man turns his face toward the viewer and warmly gives a wave hello with a bright friendly smile, brightly calling out 'Hi!' in a cheerful clear voice. No other people appear in the scene. Photoreal, natural lighting.}"

# Camera trajectory: set TRAJECTORY to your own camera_poses.npz (one 4x4
# camera-to-world per frame), or leave it empty and give POSE — a WASD/arrow
# shorthand compiled into a trajectory (see build_pose_trajectory.py).
: "${TRAJECTORY:=}"
: "${PHASES:=}"        # optional with TRAJECTORY: key-press timeline for the UI overlay
: "${POSE:=w-3, left-6, right-12, left-6, w-3}"

# F-key social-action window, in SECONDS, as "start,end".
: "${INTERACTION_TIME:=4,7}"
# What the social action is. Both must be substrings of TEXT_PROMPT.
: "${INTERACTION_PROMPT:=wave hello}"
: "${INTERACTION_SPEECH:='Hi!'}"

# Ablation toggles (true/false). Defaults = the full method (audio + both temporal masks).
: "${ENABLE_AUDIO:=true}"
: "${ENABLE_VIDEOTEMPORALMASK:=true}"
: "${ENABLE_AUDIOTEMPORALMASK:=true}"
# Extra outputs (true/false): gen.mp4 is always written; these control the other two.
: "${WITH_UI:=true}"     # also write gen_ui.mp4 (keyboard overlay)
: "${WITH_WARP:=true}"   # keep warp/warp.mp4 (the camera-warp condition) in the output

: "${SEED:=1234}"
: "${GPU:=0}"

# =====================================================================
# Locked configuration — change only if you know why
# =====================================================================
# LTX render geometry / operating point.
: "${NUM_FRAMES:=241}"          # 10.04 s @ 24 fps (LoRA training length; do not exceed -> fade-to-black)
: "${FRAME_RATE:=24}"
: "${WIDTH:=1280}"
: "${HEIGHT:=704}"
: "${COND_S:=0.3}"              # cond-attn-strength operating point
: "${VIS_THRESHOLD:=0.1}"
: "${RAMP:=12}"                 # F-window edge ramp (pixel frames)
# Pi3X warp calibration: maps pose-translation units to scene-metric motion at
# the scale the warp/LoRA was rendered with. LOCKED constant — NOT a speed knob;
# use FWD_RATE / STRAFE_RATE to change dolly/strafe magnitude.
TRANSLATION_SCALE=0.1

# The released HelloWorld LoRA (large file kept outside this folder — see README).
# It is trained with the "No other people appear in the scene." clause, so ending
# TEXT_PROMPT with that clause actually suppresses hallucinated extra characters
# (the base model is CFG-less; the constraint works because training included it).
: "${LORA_CKPT:=${HERE}/../checkpoints/helloworld_lora_v1.safetensors}"
: "${LORA_STRENGTH:=1.0}"

# Model checkpoints (large; kept in the shared HF cache — see README).
: "${HF_ROOT:=${HOME}/.cache/huggingface/hub}"
: "${DISTILLED_CKPT:=${HF_ROOT}/Lightricks/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors}"
: "${UPSCALER_CKPT:=${HF_ROOT}/Lightricks/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors}"
: "${GEMMA_ROOT:=${HF_ROOT}/google/gemma-3-12b-it-qat-q4_0-unquantized}"

# Conda env pythons (kept outside this folder — see README).
: "${WARP_PY:=python}"  # Pi3X warp + trajectory
: "${LTX_PY:=python}"                     # LTX-2.3 inference
: "${UI_PY:=${LTX_PY}}"           # imageio + PIL for the overlay (reuses the ltx env)
: "${FFMPEG:=ffmpeg}"

# Stage-1 (warp) resolution = full / 2.
WARP_WIDTH=$(( WIDTH / 2 ))
WARP_HEIGHT=$(( HEIGHT / 2 ))

# =====================================================================
# Derive the F-window in frames from INTERACTION_TIME (seconds).
# =====================================================================
F_START_S="${INTERACTION_TIME%,*}"
F_END_S="${INTERACTION_TIME#*,}"
F_START=$(awk "BEGIN{printf \"%d\", ${F_START_S} * ${FRAME_RATE} + 0.5}")
F_END=$(awk "BEGIN{printf \"%d\", ${F_END_S} * ${FRAME_RATE} + 0.5}")
(( F_END > NUM_FRAMES )) && F_END=$NUM_FRAMES

# =====================================================================
# Validate substrings up front (the temporal mask localises by verbatim match).
# =====================================================================
[[ -f "$IMAGE" ]]      || { echo "ERROR: missing IMAGE: $IMAGE" >&2; exit 1; }
[[ -f "$LORA_CKPT" ]]  || { echo "ERROR: missing LORA_CKPT: $LORA_CKPT" >&2; exit 1; }
[[ "$TEXT_PROMPT" == *"$INTERACTION_PROMPT"* ]] || {
  echo "ERROR: INTERACTION_PROMPT '$INTERACTION_PROMPT' is not a substring of TEXT_PROMPT." >&2; exit 1; }
if [[ "$ENABLE_AUDIO" == "true" && "$ENABLE_AUDIOTEMPORALMASK" == "true" ]]; then
  [[ "$TEXT_PROMPT" == *"$INTERACTION_SPEECH"* ]] || {
    echo "ERROR: INTERACTION_SPEECH '$INTERACTION_SPEECH' is not a substring of TEXT_PROMPT." >&2; exit 1; }
fi

# =====================================================================
# Layout
# =====================================================================
TRAJ_DIR="${OUTPUT_DIR}/trajectory"
WARP_DIR="${OUTPUT_DIR}/warp"
WARP_MP4="${WARP_DIR}/warp.mp4"   # the warp render location
GEN_MP4="${OUTPUT_DIR}/gen.mp4"
GEN_UI_MP4="${OUTPUT_DIR}/gen_ui.mp4"
PARAMS="${OUTPUT_DIR}/params.txt"
mkdir -p "$OUTPUT_DIR" "$TRAJ_DIR" "$WARP_DIR"

echo "[$(date '+%F %T')] helloworld run -> $OUTPUT_DIR"
echo "  image            : $IMAGE"
echo "  pose             : $POSE"
echo "  interaction_time : ${INTERACTION_TIME}s  -> frames [${F_START}, ${F_END}) of ${NUM_FRAMES}"
echo "  interaction      : video='${INTERACTION_PROMPT}'  speech='${INTERACTION_SPEECH}'"
echo "  toggles          : audio=$ENABLE_AUDIO  vmask=$ENABLE_VIDEOTEMPORALMASK  amask=$ENABLE_AUDIOTEMPORALMASK  ui=$WITH_UI"
echo "  seed=$SEED  gpu=$GPU  lora=$(basename "$LORA_CKPT")"

# =====================================================================
# 1. POSE -> trajectory
# =====================================================================
if [[ -n "$TRAJECTORY" ]]; then
  echo "[$(date '+%F %T')] [1/4] use supplied camera trajectory: $TRAJECTORY"
  [[ -f "$TRAJECTORY" ]] || { echo "ERROR: missing TRAJECTORY: $TRAJECTORY" >&2; exit 1; }
  mkdir -p "$TRAJ_DIR"
  cp "$TRAJECTORY" "${TRAJ_DIR}/camera_poses.npz"
  [[ -n "$PHASES" ]] && cp "$PHASES" "${TRAJ_DIR}/phases.json"
else
  echo "[$(date '+%F %T')] [1/4] build trajectory from POSE"
  TRAJ_RATE_ARGS=()
  [[ -n "${YAW_RATE:-}" ]]    && TRAJ_RATE_ARGS+=(--yaw_rate "$YAW_RATE")
  [[ -n "${PITCH_RATE:-}" ]]  && TRAJ_RATE_ARGS+=(--pitch_rate "$PITCH_RATE")
  [[ -n "${FWD_RATE:-}" ]]    && TRAJ_RATE_ARGS+=(--fwd_rate "$FWD_RATE")
  [[ -n "${STRAFE_RATE:-}" ]] && TRAJ_RATE_ARGS+=(--strafe_rate "$STRAFE_RATE")
  "$WARP_PY" "${HERE}/build_pose_trajectory.py" \
    --pose "$POSE" --total "$NUM_FRAMES" --out_dir "$TRAJ_DIR" "${TRAJ_RATE_ARGS[@]}"
fi

# =====================================================================
# 2. Pi3X warp render
# =====================================================================
echo "[$(date '+%F %T')] [2/4] render Pi3X warp"
CUDA_VISIBLE_DEVICES="$GPU" "$WARP_PY" "${HERE}/render_warp.py" \
  --image "$IMAGE" --poses "${TRAJ_DIR}/camera_poses.npz" \
  --height "$WARP_HEIGHT" --width "$WARP_WIDTH" \
  --num_frames "$NUM_FRAMES" --fps "$FRAME_RATE" \
  --translation_scale "$TRANSLATION_SCALE" \
  --out_dir "$WARP_DIR"

# =====================================================================
# 3. LTX-2.3 inference
# =====================================================================
echo "[$(date '+%F %T')] [3/4] LTX-2.3 inference"
INFER_TOGGLES=()
[[ "$ENABLE_AUDIO" == "true" ]]             && INFER_TOGGLES+=(--enable_audio)
[[ "$ENABLE_VIDEOTEMPORALMASK" == "true" ]] && INFER_TOGGLES+=(--enable_videotemporalmask)
[[ "$ENABLE_AUDIOTEMPORALMASK" == "true" ]] && INFER_TOGGLES+=(--enable_audiotemporalmask)

CUDA_VISIBLE_DEVICES="$GPU" "$LTX_PY" "${HERE}/infer.py" \
  --distilled-checkpoint-path "$DISTILLED_CKPT" \
  --spatial-upsampler-path "$UPSCALER_CKPT" \
  --gemma-root "$GEMMA_ROOT" \
  --prompt "$TEXT_PROMPT" \
  --image "$IMAGE" 0 1.0 \
  --width "$WIDTH" --height "$HEIGHT" \
  --num-frames "$NUM_FRAMES" --frame-rate "$FRAME_RATE" \
  --seed "$SEED" \
  --warp-mp4 "$WARP_MP4" --warp-mask "${WARP_DIR}/mask.npz" \
  --cond-attn-strength "$COND_S" --vis-threshold "$VIS_THRESHOLD" \
  --lora "$LORA_CKPT" "$LORA_STRENGTH" \
  --interaction-prompt "$INTERACTION_PROMPT" \
  --interaction-speech "$INTERACTION_SPEECH" \
  --interaction-window "$F_START" "$F_END" \
  --interaction-ramp "$RAMP" \
  "${INFER_TOGGLES[@]}" \
  --output-path "$GEN_MP4"

# =====================================================================
# 4. Keyboard UI overlay
# =====================================================================
if [[ "$WITH_UI" == "true" && ! -f "${TRAJ_DIR}/phases.json" ]]; then
  echo "[$(date '+%F %T')] [4/4] UI overlay skipped (no phases.json — supply PHASES with TRAJECTORY to get the overlay)"
  WITH_UI=false
fi
if [[ "$WITH_UI" == "true" ]]; then
  echo "[$(date '+%F %T')] [4/4] keyboard UI overlay"
  UI_AUDIO_ARG=()
  [[ "$ENABLE_AUDIO" == "true" ]] && UI_AUDIO_ARG+=(--with-audio)
  "$UI_PY" "${HERE}/make_ui.py" \
    --in-mp4 "$GEN_MP4" --out-mp4 "$GEN_UI_MP4" \
    --phases "${TRAJ_DIR}/phases.json" \
    --f-start "$F_START" --f-end "$F_END" --num-frames "$NUM_FRAMES" \
    --ffmpeg "$FFMPEG" "${UI_AUDIO_ARG[@]}"
else
  echo "[$(date '+%F %T')] [4/4] UI overlay skipped (WITH_UI=$WITH_UI)"
fi

# =====================================================================
# params.txt
# =====================================================================
{
  echo "# helloworld run parameters"
  echo "timestamp           = $(date '+%F %T')"
  echo "image               = $IMAGE"
  echo "text_prompt         = $TEXT_PROMPT"
  if [[ -n "$TRAJECTORY" ]]; then
    echo "trajectory          = $TRAJECTORY"
  else
    echo "pose                = $POSE"
  fi
  echo "interaction_time_s  = $INTERACTION_TIME"
  echo "interaction_window  = [${F_START}, ${F_END}) frames"
  echo "interaction_prompt  = $INTERACTION_PROMPT"
  echo "interaction_speech  = $INTERACTION_SPEECH"
  echo "enable_audio              = $ENABLE_AUDIO"
  echo "enable_videotemporalmask  = $ENABLE_VIDEOTEMPORALMASK"
  echo "enable_audiotemporalmask  = $ENABLE_AUDIOTEMPORALMASK"
  echo "with_ui                   = $WITH_UI"
  echo "with_warp                 = $WITH_WARP"
  echo "seed                = $SEED"
  echo "gpu                 = $GPU"
  echo "num_frames          = $NUM_FRAMES"
  echo "frame_rate          = $FRAME_RATE"
  echo "width x height      = ${WIDTH} x ${HEIGHT}"
  echo "cond_attn_strength  = $COND_S"
  echo "vis_threshold       = $VIS_THRESHOLD"
  echo "ramp                = $RAMP"
  echo "translation_scale   = $TRANSLATION_SCALE"
  echo "lora_ckpt           = $LORA_CKPT"
  echo "lora_strength       = $LORA_STRENGTH"
  echo "distilled_ckpt      = $DISTILLED_CKPT"
  echo "upscaler_ckpt       = $UPSCALER_CKPT"
  echo "gemma_root          = $GEMMA_ROOT"
  if [[ -f "${TRAJ_DIR}/trajectory.txt" ]]; then
    echo
    echo "# trajectory fingerprint"
    cat "${TRAJ_DIR}/trajectory.txt"
  fi
} > "$PARAMS"

# The warp is always rendered (it is the model's camera condition); WITH_WARP
# only controls whether it is kept as an output artefact.
if [[ "$WITH_WARP" != "true" ]]; then
  rm -rf "$WARP_DIR"
fi

echo "[$(date '+%F %T')] DONE -> $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.mp4 "$PARAMS" 2>/dev/null || true
