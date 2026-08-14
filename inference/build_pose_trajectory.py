#!/usr/bin/env python
"""POSE string -> camera_poses.npz  (+ phases.json sidecar)

Compiles a game-style camera-trajectory string into the [T, 4, 4] OpenCV
c2w pose array that `render_warp.py` (Pi3X) consumes: it lets a caller
describe a 10 s camera path with a compact, human-readable string instead of
hand-authoring a per-frame pose array.

POSE grammar
------------
    "<action>-<N>[, <action>-<N> ...]"

    Translations (WASD):   w (forward)  s (back)   a (left)   d (right)
    Rotations  (arrows):   left / right (yaw)       up / down (pitch)
    Hold:                  hold / static            (camera holds, no motion)

    SIMULTANEOUS keys (combo): join actions with '+' to hold them together in
    one phase, e.g. `w+d+left-30` = forward + strafe-right + yaw-left held for
    30 units. This is how to express an ORBIT / arc (and any diagonal). A single
    `hold-30` (or `static-30`) holds the camera still; it cannot be combined
    with motion keys. Single-action tokens (`left-6`) are unchanged.

    <N> is a positive integer that maps STRICTLY to frames: each unit is
    `frames_per_unit` frames (default 8). Frame 0 is a static lead-in (the
    source image, identity pose), so the total is

        total == sum(N) * frames_per_unit + 1.

    For the default 10 s clip (241 frames @ 24 fps, 8 frames/unit) the N's
    must sum to 30. So N is purely a duration: `left-6` = 6*8 = 48 frames =
    2.0 s of left-yaw. Motion is CUMULATIVE across phases (each phase
    continues from wherever the previous one left the camera), like holding a
    key in a game.

    Example (sum(N) = 30 -> 241 frames):
        POSE='w-3, left-6, right-12, left-6, w-3'
        -> push in (24 f), pan left (48 f), pan right twice as long (96 f),
           pan part-way back (48 f), push in (24 f). A left/right scan with
           two forward nudges.

Rates = speed / amplitude (optional)
------------------------------------
Per-frame increments, DECOUPLED from N now that N fixes the frame count. The
amplitude of a phase = rate * (N * frames_per_unit). Defaults are fixed and
Pi3X-safe (30 deg of yaw/pitch, or 1.0 c2w of dolly/strafe, over a full
240-motion-frame clip). To move further in the same number of frames, raise the
relevant rate — e.g. a stronger forward push without a longer `w` phase: bump
fwd_rate. (This is the speed knob.)

Pi3X note: a single image only has a ~+-30 deg angular envelope before
visibility collapses. Keep the *net* yaw/pitch within that envelope or the warp
disoccludes the subject.

Sign conventions — the arrow keys denote the CAMERA's turn direction (turning the camera left makes the scene pan right):
    left  -> camera turns left   (_ry(-theta); gaze toward world -X; scene pans right)
    right -> camera turns right  (_ry(+theta); gaze toward world +X; scene pans left)
    up    -> camera tilts up      (_rx(+theta); scene pans down)
    down  -> camera tilts down    (_rx(-theta))
    w     -> +z (dolly in),       s -> -z (dolly out)
    d     -> +x (strafe right),   a -> -x (strafe left)
We name rotations by the camera's turn direction so they match the on-screen
arrow keys (turning the camera left makes the scene content pan right).

Output
------
    <out_dir>/camera_poses.npz   key 'camera_poses', float32 [T, 4, 4]
    <out_dir>/phases.json        [{action, n, start, end}]  (frame ranges;
                                 read by make_ui.py so the on-screen keyboard
                                 timeline is always identical to the warp)
    <out_dir>/trajectory.txt     human-readable fingerprint
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

TRANSLATIONS = {"w", "s", "a", "d"}
ROTATIONS = {"left", "right", "up", "down"}
NOOP = {"hold", "static"}                 # camera holds its current pose (no motion)
VALID_ACTIONS = TRANSLATIONS | ROTATIONS | NOOP

# An action token may be a SINGLE action ("w", "left") or several pressed
# SIMULTANEOUSLY, joined by '+' ("w+d+left" = forward + strafe-right + yaw-left,
# the orbit / arc combo). So the action part is one-or-more [a-z] names + '+'.
_TOKEN_RE = re.compile(r"^([a-zA-Z+]+)-([0-9]+)$")


# ---------------------------------------------------------------------------
# Rotation matrices (OpenCV camera-to-world, right-handed)
# ---------------------------------------------------------------------------
def _ry(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array(
        [[c, 0.0, s],
         [0.0, 1.0, 0.0],
         [-s, 0.0, c]],
        dtype=np.float32,
    )


def _rx(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array(
        [[1.0, 0.0, 0.0],
         [0.0, c, -s],
         [0.0, s, c]],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# POSE parsing + frame allocation
# ---------------------------------------------------------------------------
def parse_pose(pose: str) -> list[tuple[tuple[str, ...], int]]:
    """Parse a POSE string into [(actions, N), ...].

    Each phase's actions is a TUPLE of one or more simultaneously-held actions:
        'w-3, left-14'   -> [(('w',), 3), (('left',), 14)]          # single keys (legacy)
        'w+d+left-30'    -> [(('w', 'd', 'left'), 30)]              # combo: orbit / arc
        'hold-30'        -> [(('hold',), 30)]                       # static (no motion)
    """
    phases: list[tuple[tuple[str, ...], int]] = []
    for raw in pose.split(","):
        tok = raw.strip()
        if not tok:
            continue
        m = _TOKEN_RE.match(tok)
        if not m:
            raise ValueError(
                f"cannot parse POSE token {tok!r}; expected '<action>[+<action>...]-<N>' "
                f"with N a positive integer (e.g. 'left-6', 'w+d+left-30', 'hold-30')."
            )
        actions = tuple(a.lower() for a in m.group(1).split("+") if a)
        n = int(m.group(2))
        for a in actions:
            if a not in VALID_ACTIONS:
                raise ValueError(
                    f"unknown action {a!r} in token {tok!r}; valid: {sorted(VALID_ACTIONS)}"
                )
        if (NOOP & set(actions)) and len(actions) > 1:
            raise ValueError(f"'hold'/'static' cannot be combined with motion keys in {tok!r}")
        if n <= 0:
            raise ValueError(f"N must be > 0 in token {tok!r}")
        phases.append((actions, n))
    if not phases:
        raise ValueError(f"POSE {pose!r} parsed to zero phases")
    return phases


def validate_pose_length(phases: list[tuple[str, int]], total: int, frames_per_unit: int) -> int:
    """Strict N↔frame mapping: each N unit is exactly `frames_per_unit` frames,
    and frame 0 is a static lead-in (identity), so

        total == sum(N) * frames_per_unit + 1.

    Returns sum(N). Raises with the required sum if the POSE doesn't fit."""
    sum_n = sum(n for _, n in phases)
    if sum_n * frames_per_unit + 1 != total:
        if (total - 1) % frames_per_unit == 0:
            req = (total - 1) // frames_per_unit
            hint = f"For total={total} at {frames_per_unit} frames/unit, the N's must sum to {req} (you gave {sum_n})."
        else:
            hint = (f"total-1={total - 1} is not divisible by frames_per_unit={frames_per_unit}; "
                    f"choose a total of k*{frames_per_unit}+1.")
        raise ValueError(
            f"POSE sum(N)={sum_n} does not fit total frames {total}: need "
            f"sum(N)*{frames_per_unit}+1 == {total}. {hint}"
        )
    return sum_n


# ---------------------------------------------------------------------------
# Trajectory construction
# ---------------------------------------------------------------------------
def build_trajectory(
    phases: list[tuple[tuple[str, ...], int]],
    total: int,
    frames_per_unit: int,
    yaw_rate: float,
    pitch_rate: float,
    fwd_rate: float,
    strafe_rate: float,
) -> tuple[np.ndarray, list[dict]]:
    """Return ([T,4,4] c2w poses, phase-range list).

    Strict mapping: frame 0 is the static initial frame (identity pose), and
    each POSE phase occupies exactly N * frames_per_unit subsequent frames.
    Motion is cumulative (each phase continues from the previous camera state),
    accumulated at a constant per-frame rate — so amplitude = rate * frames =
    rate * N * frames_per_unit, and the rates are the (optional) speed knob."""
    validate_pose_length(phases, total, frames_per_unit)

    poses = np.zeros((total, 4, 4), dtype=np.float32)
    poses[:, 3, 3] = 1.0
    poses[0, :3, :3] = np.eye(3, dtype=np.float32)  # frame 0: identity / source image

    # Cumulative camera state.
    yaw = pitch = 0.0          # degrees
    x = y = z = 0.0            # c2w translation (c2w units; render scales by translation_scale)

    # per-frame increment for one action (no-op for hold/static)
    incr = {
        "w": ("z", +fwd_rate), "s": ("z", -fwd_rate),
        "d": ("x", +strafe_rate), "a": ("x", -strafe_rate),
        "right": ("yaw", +yaw_rate),  # _ry(+): gaze toward world +X -> camera turns RIGHT
        "left": ("yaw", -yaw_rate),   # _ry(-): gaze toward world -X -> camera turns LEFT
        "up": ("pitch", +pitch_rate), "down": ("pitch", -pitch_rate),
    }
    ranges: list[dict] = []
    frame = 1                  # frame 0 is the static lead-in
    for actions, n in phases:
        start = frame
        # sum the simultaneous actions' deltas (combo = orbit/arc; hold = none)
        d = {"x": 0.0, "yaw": 0.0, "pitch": 0.0, "z": 0.0}
        for a in actions:
            if a in incr:
                axis, val = incr[a]
                d[axis] += val
        for _ in range(n * frames_per_unit):
            x += d["x"]; z += d["z"]; yaw += d["yaw"]; pitch += d["pitch"]
            R = _ry(np.radians(yaw)) @ _rx(np.radians(pitch))
            poses[frame, :3, :3] = R
            poses[frame, :3, 3] = np.array([x, y, z], dtype=np.float32)
            frame += 1
        ranges.append({"action": "+".join(actions), "n": n, "start": start, "end": frame})
    assert frame == total, (frame, total)
    return poses, ranges


def build_and_write(
    pose: str,
    total: int,
    out_dir: Path,
    frames_per_unit: int = 8,
    yaw_rate: float = 30.0 / 240.0,
    pitch_rate: float = 30.0 / 240.0,
    fwd_rate: float = 1.0 / 240.0,
    strafe_rate: float = 1.0 / 240.0,
    verbose: bool = True,
) -> list[dict]:
    """Parse a POSE string, build the trajectory, and write camera_poses.npz,
    phases.json, trajectory.txt into `out_dir`. Returns the phase-range list
    (same content as phases.json). Reused by both the CLI and batch_infer.py."""
    out_dir = Path(out_dir)
    phases = parse_pose(pose)
    poses, ranges = build_trajectory(
        phases, total, frames_per_unit,
        yaw_rate=yaw_rate, pitch_rate=pitch_rate, fwd_rate=fwd_rate, strafe_rate=strafe_rate,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "camera_poses.npz", camera_poses=poses)
    (out_dir / "phases.json").write_text(json.dumps(ranges, indent=2) + "\n")

    final_yaw = float(np.degrees(np.arctan2(poses[-1, 0, 2], poses[-1, 2, 2])))
    fp = [
        f"pose={pose}",
        f"total={total}",
        f"frames_per_unit={frames_per_unit}",
        f"sum_n={sum(n for _, n in phases)}",
        f"yaw_rate={yaw_rate}",
        f"pitch_rate={pitch_rate}",
        f"fwd_rate={fwd_rate}",
        f"strafe_rate={strafe_rate}",
        "phases=" + " | ".join(
            f"{'+'.join(a)}-{n}[{r['start']}:{r['end']}]" for (a, n), r in zip(phases, ranges)
        ),
        f"final_yaw_deg={final_yaw:.2f}",
        f"final_pos=({poses[-1, 0, 3]:.3f}, {poses[-1, 1, 3]:.3f}, {poses[-1, 2, 3]:.3f})",
    ]
    (out_dir / "trajectory.txt").write_text("\n".join(fp) + "\n")

    if verbose:
        print(f"[build_pose_trajectory] {pose!r} -> {poses.shape} poses")
        for (a, n), r in zip(phases, ranges):
            print(f"    {'+'.join(a):>10}-{n:<3} frames [{r['start']:>3}, {r['end']:>3})  ({r['end'] - r['start']} f)")
        print(f"    final yaw={final_yaw:.2f} deg  "
              f"pos=({poses[-1,0,3]:.3f},{poses[-1,1,3]:.3f},{poses[-1,2,3]:.3f})")
        print(f"    wrote {out_dir}/camera_poses.npz, phases.json, trajectory.txt")
    return ranges


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", required=True, type=str,
                    help="Pose string. With the defaults, the N's must sum to 30 "
                         "(8 frames/unit, 241 frames). e.g. 'w-3, left-6, right-12, left-6, w-3'")
    ap.add_argument("--total", type=int, default=241,
                    help="Total frames (default 241 = 10 s @ 24 fps). Must equal sum(N)*frames_per_unit+1.")
    ap.add_argument("--frames_per_unit", type=int, default=8,
                    help="Frames per N unit (strict mapping; default 8 -> sum(N)=30 for 241 frames).")
    ap.add_argument("--out_dir", required=True, type=Path)
    # Per-frame rates = the (optional) speed/amplitude knob. Defaults are fixed
    # and Pi3X-safe; raise a rate to move faster per frame without changing N.
    ap.add_argument("--yaw_rate", type=float, default=30.0 / 240.0,
                    help="deg/frame for left/right (default = 30deg over a full 240-motion-frame clip).")
    ap.add_argument("--pitch_rate", type=float, default=30.0 / 240.0,
                    help="deg/frame for up/down.")
    ap.add_argument("--fwd_rate", type=float, default=1.0 / 240.0,
                    help="c2w units/frame for w/s (default = 1.0 over a full 240-motion-frame clip).")
    ap.add_argument("--strafe_rate", type=float, default=1.0 / 240.0,
                    help="c2w units/frame for a/d.")
    args = ap.parse_args()
    build_and_write(
        args.pose, args.total, args.out_dir, frames_per_unit=args.frames_per_unit,
        yaw_rate=args.yaw_rate, pitch_rate=args.pitch_rate,
        fwd_rate=args.fwd_rate, strafe_rate=args.strafe_rate,
    )


if __name__ == "__main__":
    main()
