# HelloWorld — inference

Give it a scene image, a text prompt, a camera trajectory, and an interaction
time window → a 10 s clip where the camera follows the trajectory and the
character turns to the viewer and performs a social action (e.g. wave + spoken
greeting) in that window.

**The inputs, per clip:**

| input | what |
|---|---|
| image | first-frame scene image (the scene + character) |
| text prompt | scene + character + social action + spoken phrase |
| camera trajectory | `camera_poses.npz`: one 4×4 camera-to-world matrix per frame |
| interaction window | `[start, end]` seconds — when the action/speech happens |
| seed | random seed |

**The outputs, per clip** (`gen.mp4` is always written; the other two are on by
default and each has an off switch):

| output | what | switch |
|---|---|---|
| `gen.mp4` | the generated video | — |
| `gen_ui.mp4` | `gen.mp4` + WASD/arrow/F keyboard overlay | `with_ui` |
| `warp/warp.mp4` | the camera-warp condition video (the model's camera input) | `with_warp` |

---

## Setup

One CUDA GPU (base model is LTX-2.3-22B distilled — use a ~40 GB-class GPU).

### 1. Python environments

Two environments (pinned requirements + full reference freezes in [`../env/`](../env)):

| env | used for | setup |
|---|---|---|
| **ltx**  | LTX-2.3 inference + keyboard-overlay video | `pip install -r ../env/requirements_ltx.txt`, then install `ltx-core` / `ltx-pipelines` from the official [LTX-2 repo](https://github.com/Lightricks/LTX-2) |
| **warp** | camera trajectory + Pi3X warp | `pip install -r ../env/requirements_warp.txt`, then install the Pi3X warp package (`pi3`) from the [Warp-as-History](https://github.com/yyfz/warp-as-history) repo (clone it to `../third_party/warp-as-history`, or set `WARP_REPO_ROOT`) |

The overlay stage (`make_ui.py`) just needs `imageio` + `Pillow` + `ffmpeg`, all
already in the ltx env — `UI_PY` defaults to the same interpreter as `LTX_PY`.

### 2. Model weights

| weight | obtain from | variable |
|---|---|---|
| LTX-2.3 distilled (base) | HF `Lightricks/LTX-2.3` → `ltx-2.3-22b-distilled-1.1.safetensors` | `DISTILLED_CKPT` |
| LTX-2.3 spatial upscaler | HF `Lightricks/LTX-2.3` → `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | `UPSCALER_CKPT` |
| Gemma-3 text encoder | HF `google/gemma-3-12b-it-qat-q4_0-unquantized` | `GEMMA_ROOT` |
| HelloWorld social-interaction LoRA | HF [`oyly/HelloWorld_V1`](https://huggingface.co/oyly/HelloWorld_V1) → put at `../checkpoints/helloworld_lora_v1.safetensors` | `LORA_CKPT` |

### 3. Point the scripts at everything

Edit the `USER CONFIG` block at the top of `run_helloworld.sh`, or pass each value
as an env var. The interpreter for each stage is picked with `WARP_PY` / `LTX_PY` /
`UI_PY` / `FFMPEG` (each defaults to `python` / `ffmpeg` on PATH).

---

## Quick start

### Reproduce the bundled examples (batch; loads the model once)
```bash
INPUT=examples_batch.json GPU=0 bash run_batch.sh
```
This regenerates all 7 examples in [`../assets/examples/`](../assets/examples)
from their recipes (each folder: input image, `camera_poses.npz`, `phases.json`,
`params.json`).

### Single clip
```bash
# With your own camera trajectory:
TRAJECTORY=/path/to/camera_poses.npz \
IMAGE=/path/to/scene.png \
INTERACTION_TIME='4,7' \
INTERACTION_PROMPT='wave hello' \
INTERACTION_SPEECH="'Hi!'" \
TEXT_PROMPT="... wave hello ... calling out 'Hi!' ..." \
bash run_helloworld.sh

# Or let the WASD/arrow shorthand build the trajectory for you:
POSE='w-3, left-6, right-12, left-6, w-3' IMAGE=... TEXT_PROMPT=... bash run_helloworld.sh

# Skip the extra outputs:
WITH_UI=false WITH_WARP=false ... bash run_helloworld.sh
```

### Outputs (one folder per run / per batch item)

| file | what |
|---|---|
| `gen.mp4`     | the generated video |
| `gen_ui.mp4`  | keyboard-overlay version (`with_ui`, default on; needs a key-press timeline — auto-derived from `POSE`, or pass `PHASES`/`phases` with a supplied trajectory) |
| `warp/warp.mp4` | the camera-warp condition video (`with_warp`, default on) |
| `trajectory/` | the camera trajectory actually used |
| `params.txt`  | every input parameter of the run |

Batch also writes `<output_root>/results.json` (per-item status + video paths).

---

## The interface

### Camera trajectory
The camera input is a `camera_poses.npz` holding `camera_poses`, an array of
shape `(N, 4, 4)` — one camera-to-world matrix per frame (`N` = `NUM_FRAMES`,
default 241 ≈ 10 s @ 24 fps). Frame 0 should be identity (the first frame is
the input image); keep net yaw/pitch within ~±30° of it. Translations are in
scene-relative units: 1.0 ≈ 0.1 × the median scene depth of the first frame.

You can author the npz however you like, or compile one from the WASD/arrow
shorthand:

```bash
python build_pose_trajectory.py --pose 'w-3, left-6, right-12, left-6, w-3' \
    --total 241 --out_dir mytraj/
```

**`POSE` grammar:** `"<action>[+<action>...]-<N>[, ...]"`.

| action | key | motion |
|---|---|---|
| `w` / `s` | W / S | dolly forward / back |
| `a` / `d` | A / D | strafe left / right |
| `left` / `right` | ◀ / ▶ | yaw left / right |
| `up` / `down`    | ▲ / ▼ | pitch up / down |
| `hold` / `static` | — | hold still (no motion) |

Join actions with `+` to hold them together in one phase:

```bash
POSE='w-3, left-6, right-12, left-6, w-3'   # forward nudge, left/right scan
POSE='hold-30'                               # static camera
POSE='w+d+left-30'                           # orbit-right (forward + strafe + counter-yaw)
POSE='w+a+right-15, w+d+left-15'             # orbit out then back
```

**N → frames.** 1 unit = 8 frames; frame 0 is a static lead-in. For the default
241-frame clip the N's must **sum to 30** (each unit = 1/3 s). Motion is
cumulative across phases. Per-frame speed/amplitude knobs: `YAW_RATE` /
`PITCH_RATE` (deg/frame, default 0.125) and `FWD_RATE` / `STRAFE_RATE`
(trajectory units/frame, default 0.004167); a phase's amplitude =
`rate × (N × 8)`. Building from `POSE` also emits `phases.json`, the key-press
timeline that drives the keyboard overlay.

### Text — `TEXT_PROMPT`
Full prompt: scene + character + social action + spoken phrase.
`INTERACTION_PROMPT` and `INTERACTION_SPEECH` must each appear **verbatim** as a
substring of it. End `TEXT_PROMPT` with `No other people appear in the scene.`
(use `No other figures appear in the scene.` for object/figure scenes) to
suppress extra characters.

### Interaction window (the F-key)
- `INTERACTION_TIME="start,end"` — seconds; when the action/speech occurs.
- `INTERACTION_PROMPT` — video action substring, e.g. `"wave hello"`.
- `INTERACTION_SPEECH` — spoken phrase substring, e.g. `"'Hi!'"`.

### Toggles (true/false)
| toggle | effect |
|---|---|
| `ENABLE_AUDIO` | keep the co-generated speech track (else silent) |
| `ENABLE_VIDEOTEMPORALMASK` | temporal control of the video action |
| `ENABLE_AUDIOTEMPORALMASK` | temporal control of the speech (needs `ENABLE_AUDIO`) |
| `WITH_UI` | also emit the keyboard-overlay video |
| `WITH_WARP` | keep the camera-warp condition video in the output |

Defaults: all `true`. Other fields: `SEED`, `GPU`, `OUTPUT_DIR`. Locked render
config (top of `run_helloworld.sh`): `NUM_FRAMES=241`, `FRAME_RATE=24`, `WIDTH=1280`,
`HEIGHT=704`, `COND_S=0.3`, `VIS_THRESHOLD=0.1`, `RAMP=12`.

In the batch JSON the same switches are per-item (or `defaults`) fields:
`with_ui`, `with_warp`, `enable_audio`, `enable_videotemporalmask`,
`enable_audiotemporalmask`, plus `camera_poses` / `phases` (or `pose`) for the
camera. See `examples_batch.json`.

---

## Files

| file | role |
|---|---|
| `run_helloworld.sh`        | single-clip driver |
| `run_batch.sh`             | batch wrapper around `batch_infer.py` |
| `batch_infer.py`           | batch driver (loads the model once) |
| `examples_batch.json`      | batch input reproducing the bundled examples |
| `build_pose_trajectory.py` | `POSE` string → `camera_poses.npz` + `phases.json` |
| `render_warp.py`           | Pi3X camera-warp render (single) |
| `render_warp_batch.py`     | Pi3X warp render for many pairs (used by batch) |
| `infer.py`                 | inference driver |
| `make_ui.py`               | keyboard-overlay video |
| `lib/`                     | pipeline modules (`mixed_velocity.py`, `temporal_attention_mask*.py`) |
