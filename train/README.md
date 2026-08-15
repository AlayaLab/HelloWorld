# HelloWorld — training

Reproduce the released `helloworld_lora_v1` from scratch, or retrain it on your
own scenes/characters by editing one file. The training data is **fully
synthetic** — generated on the fly with LTX-2.3 text-to-video — so there is no
dataset to download.

```bash
bash run_train.sh        # one-command reproduction (reads config.json)
```

## The pipeline (5 stages)

| stage | script | what | output |
|---|---|---|---|
| 1. prompts  | (`prompts.tsv`) | the training prompt set — one T2V clip per row | — |
| 2. generate | `generate_data.py` | LTX-2.3 T2V generation with per-clip seed retry + two quality gates (fade / cast-count) | `data/raw/*.mp4` |
| 3. warp     | `prepare_warps.py` | Pi3X pose-estimate + camera warp per clip | `data/prepared/clip_*/` |
| 4. filter   | `subject_motion_filter.py` + inline | drop subject-movers and low-visibility warps | `data/filtered/manifest.tsv` |
| 5. train    | `train_lora.py` | warp-conditioned LoRA (visible-anchor loss) | `runs/lora/helloworld_lora_final.safetensors` |

Every knob lives in **`config.json`**; any field can also be overridden per-run
with an env var (`MAX_STEPS=3000 bash run_train.sh`). Each stage skips work
whose output already exists, so an interrupted run resumes where it stopped.

## config.json

```jsonc
{
  "prompts_tsv": "prompts.tsv",       // THE file to edit for custom scenes/characters
  "generation": {
    "retries_per_prompt": 4,          // seed-retry budget per prompt row
    "gpus": "0",                      // "0" = single GPU; "0,1" halves stages 2-3
    "offload": "none"                 // none | cpu (low-VRAM generation, slower)
  },
  "filters": {                        // all on by default
    "fade_gate": true,                // drop clips that fade in/out (checked during generation)
    "cast_gate": true,                // drop clips with the wrong subject/people count (during generation)
    "warp_visibility": true,          // drop clips whose Pi3X warp is mostly disoccluded
    "subject_motion": true,           // drop clips where the SUBJECT (not the camera) moves
    "fade_threshold": 0.7,
    "visibility_threshold": 0.65
  },
  "lora": {
    "rank": 32,
    "max_steps": 2000,
    "learning_rate": 1e-4,
    "save_every": 500,
    "warp_lambda_max": 0.5            // visible-anchor strength; must match inference
  }
}
```

## Training on your own scenes / characters

Point `prompts_tsv` at your own TSV (or edit `prompts.tsv` in place). One row =
one training clip. Columns:

| column | what |
|---|---|
| `scene_id` | unique scene name (one scene appears once per camera family) |
| `family` | camera family: `static`, `pan_left`, `pan_right`, `orbit_left`, `orbit_right`, `dolly` |
| `gen_prompt` | full prompt used to GENERATE the clip — scene + character(s) + interaction + **camera clause** |
| `train_prompt` | the SAME prompt with the camera clause removed — used at TRAIN time |
| `expected_count` | how many people/subjects the clip should contain (for the cast gate) |
| `subject_type` | `""` (human), `animal`, `toy`, or `robot` |
| `gate` | cast-gate mode: `person`, a COCO animal class (`cat`, `dog`, …), `noperson`, or `none` |

The `gen_prompt` / `train_prompt` split is the core trick: camera words appear
only in the generation prompt, so at training time the LoRA is forced to read
camera motion from the warp video rather than from text. Keep everything else
(scene, characters, action) identical between the two. Write each scene once
per camera family so the model sees the same content under different camera
paths. `prompts.tsv` (the released set: 216 rows = 36 scenes × 6 families,
~75% human / ~14% animal / ~11% toy-robot) is the reference for the phrasing
that works; `build_prompts.py` is the generator that produced it and documents
the prompt-design rules (bounded camera vocabulary, positive-only constraints,
appearance diversity).

## Requirements

Same two environments and model weights as [inference](../inference/README.md):
the **ltx** env (stages 2, 4, 5 — set `LTX_PY`) and the **warp** env (stage 3 —
set `WARP_PY`), plus the LTX-2.3 distilled + upscaler checkpoints and the
Gemma-3 text encoder (`HF_ROOT` / `DISTILLED_CKPT` / `UPSCALER_CKPT` /
`GEMMA_ROOT`).

## Runtime & GPU memory

Measured on 140 GB-class GPUs with the default config (216 prompt rows,
retries=4, 2000 steps); times scale linearly with the number of prompt rows.

| stage | GPU memory | time (1 GPU) | time (2 GPUs) |
|---|---|---|---|
| 2. generate | ~81 GB with all weights resident (`offload: "cpu"` drops it to ~5 GB VRAM + ~36 GB RAM, ~5× slower) | ~4 h | ~2 h |
| 3. warp | ~11 GB | ~9 h | ~4.5 h |
| 4. filter | ~4 GB (detector) | ~5 min | — |
| 5. train | ~61 GB (trains on 640×352 latents, gradient checkpointing on) | ~50 min | — |

Rough totals: **~14 h on one GPU, ~7.5 h on two** (stages 2–3 split across
GPUs; training itself is single-GPU). Disk: ~6 GB of intermediate data under
`train/data/`.

**On a single 80 GB GPU (e.g. A100):** training (~61 GB) and warp (~11 GB) fit
as-is; only the generation stage exceeds 80 GB with weights resident — set
`"offload": "cpu"` in `config.json` and the whole pipeline runs on one 80 GB
card (generation slows to roughly ~20 h for the default 216 rows; everything
else is unchanged). Shortening the clips does not help here: the footprint is
dominated by the resident weights, not the sequence length.
