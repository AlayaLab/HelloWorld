#!/usr/bin/env bash
# run_batch.sh — convenience wrapper around batch_infer.py.
#
# Loads the model ONCE on one GPU and runs every job in an input JSON, writing
# a results JSON (mainly the generated-video paths). All config/defaults live
# in the input JSON (see example_batch.json); this wrapper just picks the ltx
# python and forwards --input / --output / --gpu.
#
# Usage:
#   bash run_batch.sh                                  # example_batch.json on its config.gpu
#   INPUT=my_jobs.json OUTPUT=my_results.json GPU=3 bash run_batch.sh
#
# Detached:
#   setsid nohup bash run_batch.sh > batch.out 2>&1 < /dev/null & disown
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

: "${INPUT:=${HERE}/example_batch.json}"
: "${OUTPUT:=}"                  # empty -> batch_infer.py defaults to <output_root>/results.json
: "${GPU:=}"                     # empty -> use config.gpu from the JSON
: "${LTX_PY:=python}"

[[ -f "$INPUT" ]] || { echo "ERROR: input JSON not found: $INPUT" >&2; exit 1; }

ARGS=(--input "$INPUT")
[[ -n "$OUTPUT" ]] && ARGS+=(--output "$OUTPUT")
[[ -n "$GPU" ]]    && ARGS+=(--gpu "$GPU")

echo "[run_batch] input=$INPUT  gpu=${GPU:-<from json>}"
exec "$LTX_PY" "${HERE}/batch_infer.py" "${ARGS[@]}"
