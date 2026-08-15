#!/usr/bin/env bash
# =====================================================================
# run_train.sh — reproduce the HelloWorld social-interaction LoRA.
#
# One command:      bash run_train.sh
# Custom scenes:    edit prompts.tsv (or point config.json at your own TSV)
# Everything else:  config.json (filters, LoRA hyperparameters, retries, GPUs)
#
# Five stages, all self-contained inside this train/ folder (data under
# train/data/, the trained LoRA under train/runs/lora/). The training data is
# FULLY SYNTHETIC — LTX-2.3 text-to-video — so there is no dataset to download.
#
#   1. PROMPTS  read the prompt TSV (default: prompts.tsv, the released set;
#               build_prompts.py is the generator that produced it)
#   2. GENERATE generate_data.py            -> data/raw/<scene>__<family>.mp4
#               per-clip seed retry + two quality gates (fade / cast-count)
#   3. WARP     prepare_warps.py            -> data/prepared/clip_*/{gt,warp,
#               mask,meta} + manifest.tsv   (Pi3X pose-estimate + camera warp)
#   4. FILTER   subject-motion gate + warp-visibility gate
#               -> data/filtered/manifest.tsv
#   5. TRAIN    train_lora.py               -> runs/lora/*.safetensors
#
# Config precedence: env var > config.json > built-in default. Example:
#   MAX_STEPS=3000 bash run_train.sh          # override one field ad hoc
#   CONFIG=my_config.json bash run_train.sh   # or use another config file
#
# Two conda envs (see ../env/):
#   LTX_PY  — stages 2, 4 (motion gate), 5: ltx_core / ltx_pipelines.
#   WARP_PY — stage 3: the Pi3X warp package (warp-as-history).
#
# Resumable: each stage skips work whose output already exists.
# =====================================================================
set -uo pipefail   # NOT -e: keep going if a single clip fails a gate
HERE="$(cd "$(dirname "$0")" && pwd)"

# =====================================================================
# Load config.json (any field can still be overridden by an env var)
# =====================================================================
: "${CONFIG:=${HERE}/config.json}"
if [[ -f "$CONFIG" ]]; then
  eval "$(python3 - "$CONFIG" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
g, f, l = c.get("generation", {}), c.get("filters", {}), c.get("lora", {})
def emit(k, v):
    if isinstance(v, bool): v = "true" if v else "false"
    print(f'CFG_{k}="{v}"')
emit("PROMPTS_TSV", c.get("prompts_tsv", "prompts.tsv"))
emit("MAX_SEEDS", g.get("retries_per_prompt", 4))
emit("GPUS", g.get("gpus", "0"))
emit("OFFLOAD", g.get("offload", "none"))
emit("FILTER_FADE", f.get("fade_gate", True))
emit("FILTER_CAST", f.get("cast_gate", True))
emit("FILTER_VIS", f.get("warp_visibility", True))
emit("FILTER_MOTION", f.get("subject_motion", True))
emit("FADE_THRESHOLD", f.get("fade_threshold", 0.7))
emit("VIS_THRESHOLD", f.get("visibility_threshold", 0.65))
emit("LORA_RANK", l.get("rank", 32))
emit("MAX_STEPS", l.get("max_steps", 2000))
emit("LR", l.get("learning_rate", 1e-4))
emit("SAVE_EVERY", l.get("save_every", 500))
emit("LAMBDA_MAX", l.get("warp_lambda_max", 0.5))
PY
)"
else
  echo "WARN: no config file at $CONFIG — using built-in defaults" >&2
fi

# =====================================================================
# Effective settings (env var > config.json > default)
# =====================================================================
: "${OUT_ROOT:=${HERE}/data}"             # all intermediate data lives here
: "${RUN_DIR:=${HERE}/runs/lora}"         # trained LoRA output

: "${PROMPTS_TSV:=${CFG_PROMPTS_TSV:-prompts.tsv}}"
[[ "$PROMPTS_TSV" = /* ]] || PROMPTS_TSV="${HERE}/${PROMPTS_TSV}"

: "${GPUS:=${CFG_GPUS:-0}}"
: "${MAX_SEEDS:=${CFG_MAX_SEEDS:-4}}"     # seed-retry budget per prompt row
: "${OFFLOAD:=${CFG_OFFLOAD:-none}}"      # none (~78 GB VRAM) | cpu | disk

: "${FILTER_FADE:=${CFG_FILTER_FADE:-true}}"
: "${FILTER_CAST:=${CFG_FILTER_CAST:-true}}"
: "${FILTER_VIS:=${CFG_FILTER_VIS:-true}}"
: "${FILTER_MOTION:=${CFG_FILTER_MOTION:-true}}"
: "${FADE_THRESHOLD:=${CFG_FADE_THRESHOLD:-0.7}}"
: "${VIS_THRESHOLD:=${CFG_VIS_THRESHOLD:-0.65}}"

: "${LORA_RANK:=${CFG_LORA_RANK:-32}}"
: "${MAX_STEPS:=${CFG_MAX_STEPS:-2000}}"
: "${LR:=${CFG_LR:-1e-4}}"
: "${SAVE_EVERY:=${CFG_SAVE_EVERY:-500}}"
: "${LAMBDA_MAX:=${CFG_LAMBDA_MAX:-0.5}}" # visible-anchor strength; MUST match inference
: "${LOG_EVERY:=10}"

# Locked render geometry (must match inference).
: "${WIDTH:=1280}"
: "${HEIGHT:=704}"
: "${GEN_FRAMES:=361}"          # LTX generation length (15 s); tail absorbs LTX's fade
: "${FRAME_NUM:=241}"           # saved clip length (10.04 s; must be 8k+1)
: "${FRAME_RATE:=24}"
: "${PERSON_SCORE:=0.9}"        # detector confidence for the cast gate
: "${PERSON_MIN_AREA:=0.015}"   # min box area fraction to count as "present"
# Positive, gentle anti-fade nudge (gen-prompt only; never reaches train_prompt).
: "${POSITIVE_NUDGE:= Soft, even, natural lighting that stays consistent through to the final frame.}"

# Model weights (same variables as inference — see ../inference/README.md).
: "${HF_ROOT:=${HOME}/.cache/huggingface/hub}"
: "${DISTILLED_CKPT:=${HF_ROOT}/Lightricks/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors}"
: "${UPSCALER_CKPT:=${HF_ROOT}/Lightricks/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors}"
: "${GEMMA_ROOT:=${HF_ROOT}/google/gemma-3-12b-it-qat-q4_0-unquantized}"

# Pi3X warp package root (used by prepare_warps.py).
: "${WARP_REPO_ROOT:=${HERE}/../third_party/warp-as-history}"
export WARP_REPO_ROOT

# Interpreters (see ../env/): LTX env for generate/filter/train, warp env for Pi3X.
: "${LTX_PY:=python}"
: "${WARP_PY:=python}"

# =====================================================================
# Derived
# =====================================================================
RAW_DIR="${OUT_ROOT}/raw"
PREP_DIR="${OUT_ROOT}/prepared"
FILTERED_DIR="${OUT_ROOT}/filtered"
WARP_WIDTH=$(( WIDTH / 2 )); WARP_HEIGHT=$(( HEIGHT / 2 ))
IFS=',' read -r GPU_A GPU_B <<< "$GPUS"
GPU_B="${GPU_B:-$GPU_A}"   # single-GPU fallback if only one given
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "$OUT_ROOT" "$RAW_DIR" "$PREP_DIR" "$FILTERED_DIR" "$RUN_DIR" "$LOG_DIR" "${LOG_DIR}/.tmp"

echo "[$(date '+%F %T')] train pipeline -> $OUT_ROOT  (GPUs ${GPU_A},${GPU_B})"
echo "  prompts=$PROMPTS_TSV  retries=$MAX_SEEDS  filters: fade=$FILTER_FADE cast=$FILTER_CAST vis=$FILTER_VIS motion=$FILTER_MOTION"
echo "  lora: rank=$LORA_RANK steps=$MAX_STEPS lr=$LR lambda=$LAMBDA_MAX"

# =====================================================================
# Stage 1 — prompts
# =====================================================================
[[ -s "$PROMPTS_TSV" ]] || { echo "ERROR: prompt TSV not found: $PROMPTS_TSV" >&2; exit 1; }
NROWS=$(awk 'END{print NR}' "$PROMPTS_TSV")
NDATA=$(( NROWS - 1 ))
(( NDATA >= 1 )) || { echo "ERROR: no data rows in $PROMPTS_TSV" >&2; exit 1; }
echo "[$(date '+%F %T')] [1/5] prompts: $NDATA rows from $PROMPTS_TSV"

# =====================================================================
# Stage 2 — generate T2V clips (one long-lived process per GPU)
# =====================================================================
echo "[$(date '+%F %T')] [2/5] generate T2V clips"
GATE_ARGS=()
[[ "$FILTER_FADE" != "true" ]] && GATE_ARGS+=(--no-fade-gate)
[[ "$FILTER_CAST" != "true" ]] && GATE_ARGS+=(--no-cast-gate)

gen_worker() {
  local gpu="$1" row_start="$2" row_end="$3"
  local worker_log="${LOG_DIR}/gen_gpu${gpu}.log"
  local shard="${LOG_DIR}/manifest.gen_gpu${gpu}.tsv"
  : > "$shard"
  echo "[$(date '+%F %T')] GPU $gpu gen driver starting (rows ${row_start}..${row_end})" | tee -a "$worker_log"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="$gpu" "$LTX_PY" "${HERE}/generate_data.py" \
    --prompts-tsv "$PROMPTS_TSV" --row-start "$row_start" --row-end "$row_end" \
    --out-dir "$RAW_DIR" --tmp-dir "${LOG_DIR}/.tmp/gpu${gpu}" --manifest-shard "$shard" \
    --gpu-label "$gpu" \
    --distilled "$DISTILLED_CKPT" --upscaler "$UPSCALER_CKPT" --gemma "$GEMMA_ROOT" \
    --width "$WIDTH" --height "$HEIGHT" \
    --gen-frames "$GEN_FRAMES" --frame-num "$FRAME_NUM" --frame-rate "$FRAME_RATE" \
    --max-seeds "$MAX_SEEDS" --person-score "$PERSON_SCORE" --person-min-area "$PERSON_MIN_AREA" \
    --fade-threshold "$FADE_THRESHOLD" --positive-nudge "$POSITIVE_NUDGE" \
    --offload-mode "$OFFLOAD" "${GATE_ARGS[@]}" \
    >> "$worker_log" 2>&1
  echo "[$(date '+%F %T')] GPU $gpu gen driver done (rc=$?)." | tee -a "$worker_log"
}

MANIFEST="${RAW_DIR}/manifest.tsv"
[[ -f "$MANIFEST" ]] || printf 'ts_end\tgpu\tscene_id\tfamily\tsource\tclip_path\tseed\tgen_seconds\tnum_frames\tsize\tprompt\tstatus\tfade_ratio\n' > "$MANIFEST"
MID=$(( 2 + (NDATA / 2) - 1 ))
if [[ "$GPU_A" == "$GPU_B" ]]; then
  gen_worker "$GPU_A" 2 "$NROWS"
else
  gen_worker "$GPU_A" 2 "$MID" & GA=$!
  gen_worker "$GPU_B" $(( MID+1 )) "$NROWS" & GB=$!
  wait $GA; wait $GB
fi
cat "${LOG_DIR}"/manifest.gen_gpu*.tsv >> "$MANIFEST" 2>/dev/null
N_OK=$(find "$RAW_DIR" -maxdepth 1 -name '*.mp4' | wc -l)
echo "  kept $N_OK generated clips in $RAW_DIR"
(( N_OK > 0 )) || { echo "ERROR: no clips passed the gates in $RAW_DIR" >&2; exit 1; }

# =====================================================================
# Stage 3 — warp-prep (Pi3X), dual-GPU slices
# =====================================================================
echo "[$(date '+%F %T')] [3/5] Pi3X warp-prep"
prep_slice() {
  local gpu="$1" slice_num="$2" slice_total="$3"
  local log="${LOG_DIR}/prep_gpu${gpu}_slice${slice_num}.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$WARP_PY" "${HERE}/prepare_warps.py" \
    --raw-dir "$RAW_DIR" --prompts-tsv "$PROMPTS_TSV" --out-root "$PREP_DIR" \
    --height "$WARP_HEIGHT" --width "$WARP_WIDTH" \
    --clip-frames "$FRAME_NUM" --fps "$FRAME_RATE" \
    --slice-num "$slice_num" --slice-total "$slice_total" \
    >> "$log" 2>&1
  echo "[$(date '+%F %T')] prep slice ${slice_num}/${slice_total} on GPU $gpu done (rc=$?)"
}
MERGED="${PREP_DIR}/manifest.tsv"
if [[ "$GPU_A" == "$GPU_B" ]]; then
  prep_slice "$GPU_A" 0 1
else
  prep_slice "$GPU_A" 0 2 & PA=$!
  prep_slice "$GPU_B" 1 2 & PB=$!
  wait $PA; wait $PB
  S0="${PREP_DIR}/manifest.slice0of2.tsv"; S1="${PREP_DIR}/manifest.slice1of2.tsv"
  [[ -f "$S0" && -f "$S1" ]] || { echo "ERROR: missing slice manifest(s)" >&2; exit 1; }
  head -1 "$S0" > "$MERGED"; tail -n +2 "$S0" >> "$MERGED"; tail -n +2 "$S1" >> "$MERGED"
fi
[[ -f "$MERGED" ]] || { echo "ERROR: no prepared manifest at $MERGED" >&2; exit 1; }
echo "  prepared $(( $(wc -l < "$MERGED") - 1 )) clips -> $MERGED"

# =====================================================================
# Stage 4 — filters: subject-motion gate + warp-visibility gate
# =====================================================================
echo "[$(date '+%F %T')] [4/5] filter (motion=$FILTER_MOTION, visibility=$FILTER_VIS >= $VIS_THRESHOLD)"
MOTION_JSONL="${PREP_DIR}/subject_motion.jsonl"
if [[ "$FILTER_MOTION" == "true" && ! -s "$MOTION_JSONL" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_A" "$LTX_PY" "${HERE}/subject_motion_filter.py" \
    --prep-dir "$PREP_DIR" --prompts-tsv "$PROMPTS_TSV" --out "$MOTION_JSONL" \
    --max-centroid 0.10 --max-area-ratio 1.6 \
    >> "${LOG_DIR}/motion.log" 2>&1 || echo "WARN: subject-motion filter failed; see ${LOG_DIR}/motion.log" >&2
fi
"$LTX_PY" - "$PREP_DIR" "$FILTERED_DIR" "$FILTER_VIS" "$VIS_THRESHOLD" "$FILTER_MOTION" <<'PY'
import json, glob, os, sys
PREP, DEST, USE_VIS, THRESH, USE_MOTION = sys.argv[1], sys.argv[2], sys.argv[3] == "true", float(sys.argv[4]), sys.argv[5] == "true"
vis = {}
for m in glob.glob(f"{PREP}/clip_*/meta.json"):
    j = json.load(open(m)); vis[int(j["clip_id"])] = float(j["mask_mean_visible"])
moved = set()
mp = f"{PREP}/subject_motion.jsonl"
if USE_MOTION and os.path.isfile(mp):
    for line in open(mp):
        r = json.loads(line)
        if not r.get("passed"):
            moved.add(int(r["clip_id"]))
lines = open(f"{PREP}/manifest.tsv").read().splitlines()
header = lines[0].replace("family", "level")   # the trainer reads 'level'
kept, drop_vis, drop_mov = [], 0, 0
for r in lines[1:]:
    cid = int(r.split("\t")[0])
    if USE_VIS and vis.get(cid, 0.0) < THRESH:
        drop_vis += 1; continue
    if cid in moved:
        drop_mov += 1; continue
    kept.append(r)
os.makedirs(DEST, exist_ok=True)
open(f"{DEST}/manifest.tsv", "w").write("\n".join([header] + kept) + "\n")
print(f"  kept {len(kept)}/{len(lines)-1} clips (dropped: {drop_vis} low-visibility, {drop_mov} subject-movers)")
PY
N_TRAIN=$(( $(wc -l < "${FILTERED_DIR}/manifest.tsv") - 1 ))
(( N_TRAIN > 0 )) || { echo "ERROR: no clips left after filtering" >&2; exit 1; }

# =====================================================================
# Stage 5 — train the LoRA
# =====================================================================
echo "[$(date '+%F %T')] [5/5] train LoRA (rank $LORA_RANK, lambda $LAMBDA_MAX, $MAX_STEPS steps on $N_TRAIN clips, GPU $GPU_A)"
CUDA_VISIBLE_DEVICES="$GPU_A" "$LTX_PY" "${HERE}/train_lora.py" \
  --distilled-checkpoint "$DISTILLED_CKPT" \
  --gemma-root "$GEMMA_ROOT" \
  --data-root "$FILTERED_DIR" \
  --output-dir "$RUN_DIR" \
  --warp-lambda-max "$LAMBDA_MAX" \
  --max-steps "$MAX_STEPS" \
  --save-every "$SAVE_EVERY" \
  --log-every "$LOG_EVERY" \
  --lr "$LR" \
  --lora-rank "$LORA_RANK" \
  2>&1 | tee "${RUN_DIR}/train.log"

echo "[$(date '+%F %T')] DONE -> $RUN_DIR"
ls -lh "$RUN_DIR"/*.safetensors 2>/dev/null || true
