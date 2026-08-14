#!/usr/bin/env python
"""Overlay the WASD + arrow + F-key UI onto a generated clip.

Self-contained: the overlay primitives are inlined here so this folder has no
cross-directory dependency.

The per-frame keyboard timeline is derived directly from the SAME
`phases.json` that build_pose_trajectory.py wrote for the warp, plus the
F-window — so the on-screen keys are guaranteed to match the camera path and
the social-action timing of the generated video, with no hand-authoring.

Key mapping (the arrow keys denote the camera's turn direction):
    w -> W (forward)      s -> S (back)
    a -> A (left strafe)  d -> D (right strafe)
    left  -> arrow_left   right -> arrow_right   (yaw)
    up    -> arrow_up     down  -> arrow_down    (pitch)
    F-window -> F (orange)

Usage:
    python make_ui.py --in-mp4 gen.mp4 --out-mp4 gen_ui.mp4 \\
        --phases <warp_dir>/phases.json --f-start 96 --f-end 168 \\
        --num-frames 241 [--with-audio --ffmpeg /path/to/ffmpeg]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Overlay primitives (inlined from make_concept_ui_video.py)
# ---------------------------------------------------------------------------
KEY_SIZE_BASE = 70
KEY_SPACING_BASE = 6
CORNER_RADIUS_BASE = 14
MARGIN_BASE = 40
FONT_SIZE_BASE = 28
REF_HEIGHT = 704

BG_NORMAL = (0, 0, 0, 128)
BG_ACTIVE_CAMERA = (30, 120, 255, 220)
BG_ACTIVE_INTERACT = (255, 140, 0, 230)
TEXT_COLOR = (255, 255, 255, 255)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _scaled(frame_height: int):
    s = max(0.4, frame_height / REF_HEIGHT)
    return dict(
        key_size=max(20, int(round(KEY_SIZE_BASE * s))),
        spacing=max(2, int(round(KEY_SPACING_BASE * s))),
        radius=max(4, int(round(CORNER_RADIUS_BASE * s))),
        margin=max(10, int(round(MARGIN_BASE * s))),
        font_size=max(10, int(round(FONT_SIZE_BASE * s))),
    )


def _load_font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except (IOError, OSError):
        return ImageFont.load_default()


def draw_rounded_rectangle(draw, xy, radius, fill):
    x1, y1, x2, y2 = xy
    d = radius * 2
    draw.ellipse([x1, y1, x1 + d, y1 + d], fill=fill)
    draw.ellipse([x2 - d, y1, x2, y1 + d], fill=fill)
    draw.ellipse([x1, y2 - d, x1 + d, y2], fill=fill)
    draw.ellipse([x2 - d, y2 - d, x2, y2], fill=fill)
    draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
    draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)


def _draw_key(draw, x, y, label, is_active, font, sz, active_color=BG_ACTIVE_CAMERA):
    bg = active_color if is_active else BG_NORMAL
    draw_rounded_rectangle(draw, [x, y, x + sz["key_size"], y + sz["key_size"]], sz["radius"], bg)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x + (sz["key_size"] - tw) // 2, y + (sz["key_size"] - th) // 2),
              label, fill=TEXT_COLOR, font=font)


def create_wasd_keyboard(actions, sz):
    cols, rows = 3, 2
    kb_w = cols * sz["key_size"] + (cols - 1) * sz["spacing"]
    kb_h = rows * sz["key_size"] + (rows - 1) * sz["spacing"]
    img = Image.new("RGBA", (kb_w, kb_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(sz["font_size"])
    fwd, lft = actions.get("forward", 0), actions.get("left", 0)
    keys = [("W", 1, 0, fwd > 0), ("A", 0, 1, lft > 0), ("S", 1, 1, fwd < 0), ("D", 2, 1, lft < 0)]
    for label, col, row, is_active in keys:
        x = col * (sz["key_size"] + sz["spacing"])
        y = row * (sz["key_size"] + sz["spacing"])
        _draw_key(draw, x, y, label, is_active, font, sz, BG_ACTIVE_CAMERA)
    return img


def create_interact_key(actions, sz):
    img = Image.new("RGBA", (sz["key_size"], sz["key_size"]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(sz["font_size"])
    _draw_key(draw, 0, 0, "F", actions.get("interact", 0) > 0, font, sz, BG_ACTIVE_INTERACT)
    return img


def _draw_triangle_key(draw, x, y, direction, is_active, sz):
    bg = BG_ACTIVE_CAMERA if is_active else BG_NORMAL
    draw_rounded_rectangle(draw, [x, y, x + sz["key_size"], y + sz["key_size"]], sz["radius"], bg)
    cx, cy = x + sz["key_size"] // 2, y + sz["key_size"] // 2
    s = sz["key_size"] // 8
    if direction == "up":
        pts = [(cx, cy - s), (cx - s, cy + s // 2), (cx + s, cy + s // 2)]
    elif direction == "down":
        pts = [(cx, cy + s), (cx - s, cy - s // 2), (cx + s, cy - s // 2)]
    elif direction == "left":
        pts = [(cx - s, cy), (cx + s // 2, cy - s), (cx + s // 2, cy + s)]
    else:
        pts = [(cx + s, cy), (cx - s // 2, cy - s), (cx - s // 2, cy + s)]
    draw.polygon(pts, fill=TEXT_COLOR)


def create_arrow_keyboard(actions, sz):
    cols = 3
    kb_w = cols * sz["key_size"] + (cols - 1) * sz["spacing"]
    kb_h = 2 * sz["key_size"] + sz["spacing"]
    img = Image.new("RGBA", (kb_w, kb_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    yaw, pitch = actions.get("yaw", 0), actions.get("pitch", 0)
    keys = [("up", 1, 0, pitch > 0), ("left", 0, 1, yaw < 0), ("down", 1, 1, pitch < 0), ("right", 2, 1, yaw > 0)]
    for direction, col, row, is_active in keys:
        x = col * (sz["key_size"] + sz["spacing"])
        y = row * (sz["key_size"] + sz["spacing"])
        _draw_triangle_key(draw, x, y, direction, is_active, sz)
    return img


def blend_overlay(base_frame, overlay_img, position):
    x, y = position
    arr = np.asarray(overlay_img)
    oh, ow = arr.shape[:2]
    bh, bw = base_frame.shape[:2]
    if x + ow > bw:
        ow = bw - x; arr = arr[:, :ow]
    if y + oh > bh:
        oh = bh - y; arr = arr[:oh, :]
    if ow <= 0 or oh <= 0:
        return base_frame
    rgb = arr[:, :, :3].astype(np.float32)
    a = arr[:, :, 3:4].astype(np.float32) / 255.0
    region = base_frame[y:y + oh, x:x + ow].astype(np.float32)
    blended = (rgb * a + region * (1 - a)).astype(np.uint8)
    out = base_frame.copy()
    out[y:y + oh, x:x + ow] = blended
    return out


def overlay_video(in_path, out_path, action_timeline):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    reader = iio.get_reader(in_path)
    meta = reader.get_meta_data()
    fps = meta.get("fps", 24)
    first = reader.get_data(0)
    h, w = first.shape[:2]
    sz = _scaled(h)

    block_h = 2 * sz["key_size"] + sz["spacing"]
    wasd_w = 3 * sz["key_size"] + 2 * sz["spacing"]
    arrow_w = 3 * sz["key_size"] + 2 * sz["spacing"]
    wasd_pos = (sz["margin"], h - sz["margin"] - block_h)
    arrow_pos = (w - sz["margin"] - arrow_w, h - sz["margin"] - block_h)
    f_pos = ((w - sz["key_size"]) // 2, h - sz["margin"] - sz["key_size"])

    wasd_cache, arrow_cache, f_cache = {}, {}, {}

    def sign(x):
        return 1 if x > 0 else (-1 if x < 0 else 0)

    writer = iio.get_writer(
        out_path, fps=fps, codec="libx264", quality=None,
        macro_block_size=1, ffmpeg_params=["-crf", "12", "-pix_fmt", "yuv420p"],
    )
    try:
        for i, frame in enumerate(reader):
            a = action_timeline[i] if i < len(action_timeline) else action_timeline[-1]
            wkey = (sign(a.get("forward", 0)), sign(a.get("left", 0)))
            akey = (sign(a.get("yaw", 0)), sign(a.get("pitch", 0)))
            fkey = sign(a.get("interact", 0))
            if wkey not in wasd_cache:
                wasd_cache[wkey] = create_wasd_keyboard(a, sz)
            if akey not in arrow_cache:
                arrow_cache[akey] = create_arrow_keyboard(a, sz)
            if fkey not in f_cache:
                f_cache[fkey] = create_interact_key(a, sz)
            frame = blend_overlay(frame, wasd_cache[wkey], wasd_pos)
            frame = blend_overlay(frame, arrow_cache[akey], arrow_pos)
            frame = blend_overlay(frame, f_cache[fkey], f_pos)
            writer.append_data(frame)
    finally:
        writer.close()
        reader.close()


# ---------------------------------------------------------------------------
# Timeline from phases.json  (single source of truth shared with the warp)
# ---------------------------------------------------------------------------
def build_timeline_from_phases(phases: list[dict], num_frames: int,
                               f_start: int, f_end: int) -> list[dict]:
    """phases.json entries {action, n, start, end} -> per-frame action dicts."""
    timeline = [{"forward": 0, "left": 0, "yaw": 0, "pitch": 0, "interact": 0}
                for _ in range(num_frames)]
    for ph in phases:
        # action may be a combo ("w+d+left") or single ("w"); "hold"/"static" -> no keys lit.
        sub_actions = str(ph["action"]).split("+")
        for i in range(ph["start"], min(ph["end"], num_frames)):
            a = timeline[i]
            for action in sub_actions:
                if action == "w":
                    a["forward"] = 1
                elif action == "s":
                    a["forward"] = -1
                elif action == "a":
                    a["left"] = 1
                elif action == "d":
                    a["left"] = -1
                elif action == "left":
                    a["yaw"] = -1
                elif action == "right":
                    a["yaw"] = 1
                elif action == "up":
                    a["pitch"] = 1
                elif action == "down":
                    a["pitch"] = -1
                # "hold" / "static": no key lit (camera idle)
    for i in range(max(0, f_start), min(f_end, num_frames)):
        timeline[i]["interact"] = 1
    return timeline


def remux_audio(video_only_mp4: Path, audio_source_mp4: Path, out_mp4: Path, ffmpeg: str) -> bool:
    """Copy video stream from `video_only_mp4` + audio from `audio_source_mp4`.
    Returns True on success; False if the source has no audio track (caller
    should then keep the video-only overlay)."""
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_only_mp4), "-i", str(audio_source_mp4),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest", str(out_mp4),
    ]
    try:
        subprocess.run(cmd, check=True)
        return out_mp4.exists()
    except subprocess.CalledProcessError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-mp4", required=True, type=Path)
    ap.add_argument("--out-mp4", required=True, type=Path)
    ap.add_argument("--phases", required=True, type=Path, help="phases.json from build_pose_trajectory.py")
    ap.add_argument("--f-start", type=int, required=True)
    ap.add_argument("--f-end", type=int, required=True)
    ap.add_argument("--num-frames", type=int, default=241)
    ap.add_argument("--with-audio", action="store_true",
                    help="Remux the source clip's audio track into the UI video.")
    ap.add_argument("--alpha-scale", type=float, default=1.0,
                    help="Scale the key overlay opacity (1.0 = original; <1 = more transparent). "
                         "Key text fades at half the rate to stay readable.")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    args = ap.parse_args()

    if args.alpha_scale != 1.0:
        global BG_NORMAL, BG_ACTIVE_CAMERA, BG_ACTIVE_INTERACT, TEXT_COLOR
        s = max(0.0, min(1.0, args.alpha_scale))
        _bg = lambda c: (*c[:3], int(round(c[3] * s)))
        BG_NORMAL = _bg(BG_NORMAL)
        BG_ACTIVE_CAMERA = _bg(BG_ACTIVE_CAMERA)
        BG_ACTIVE_INTERACT = _bg(BG_ACTIVE_INTERACT)
        TEXT_COLOR = (*TEXT_COLOR[:3], int(round(TEXT_COLOR[3] * (0.5 + 0.5 * s))))

    phases = json.loads(args.phases.read_text())
    timeline = build_timeline_from_phases(phases, args.num_frames, args.f_start, args.f_end)
    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)

    if not args.with_audio:
        overlay_video(str(args.in_mp4), str(args.out_mp4), timeline)
        print(f"[make_ui] wrote {args.out_mp4} (video-only)")
        return

    # Overlay to a temp video-only file, then remux the source audio back in.
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=args.out_mp4.parent, delete=False) as tf:
        tmp = Path(tf.name)
    try:
        overlay_video(str(args.in_mp4), str(tmp), timeline)
        if remux_audio(tmp, args.in_mp4, args.out_mp4, args.ffmpeg):
            print(f"[make_ui] wrote {args.out_mp4} (with audio)")
        else:
            tmp.replace(args.out_mp4)
            print(f"[make_ui] wrote {args.out_mp4} (no audio track in source; video-only)")
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
