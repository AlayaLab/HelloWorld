# Examples

Seven ready-to-reproduce examples. Each folder holds exactly the inputs the
pipeline needs — nothing else:

| file | what |
|---|---|
| `input.jpg` / `input.png` | first-frame scene image |
| `camera_poses.npz` | the camera trajectory (one 4×4 camera-to-world per frame) |
| `phases.json` | key-press timeline (only used for the keyboard-overlay video) |
| `params.json` | the full recipe: text prompt, interaction window, audio switches, seed |

Animated previews (camera-warp condition on the left, generation with keyboard
overlay on the right) are in [`../gifs/`](../gifs).

| example | subject | camera | interaction (F-window) | audio | preview |
|---|---|---|---|---|---|
| `garden_pond` | two people on benches | dolly-in + orbit-right, pitch-down | woman turns her whole body around and waves, "Hi" (6–10 s) | ✔ | [gif](../gifs/garden_pond.gif) |
| `park_yoga` | woman doing yoga | left/right yaw scan | palms together + bow, "Namaste" (0–4 s) | ✔ | [gif](../gifs/park_yoga.gif) |
| `anime_figures` | two anime figurines | orbit-left + push-in | each raises one hand and waves, "Hello!" (4–7 s) | ✔ | [gif](../gifs/anime_figures.gif) |
| `moor_crows` | crows on a fence | orbit-left | nearest crow flaps wings and caws (0–4 s) | ✔ | [gif](../gifs/moor_crows.gif) |
| `bear_mascot` | bear mascot on steps | right/left yaw scan | forms a heart with its arms (0–4 s) | ✖ | [gif](../gifs/bear_mascot.gif) |
| `shore_skeleton` | skeleton prop on rocks | orbit-left | turns skull and waves (4–7 s) | ✖ | [gif](../gifs/shore_skeleton.gif) |
| `lakeside_hikers` | two hikers at a lake | 30° orbit | both raise a thumbs-up, "Good!" (4–7 s) | ✔ | [gif](../gifs/lakeside_hikers.gif) |

Reproduce all seven in one command (loads the model once); each run writes
`gen.mp4`, plus `gen_ui.mp4` and `warp/warp.mp4` unless switched off with
`with_ui` / `with_warp`:

```bash
cd ../../inference
INPUT=examples_batch.json GPU=0 bash run_batch.sh
```
