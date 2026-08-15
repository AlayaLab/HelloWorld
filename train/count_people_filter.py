#!/usr/bin/env python
"""People-count gate — the no-extras analog of crop_and_check_fade.py.

Under pure T2V there's no first-frame image to pin the cast, and LTX-2.3
distilled is CFG-less (no negative prompt), so prompt-only suppression of
background people has a hard ceiling (confirmed empirically: kitchen/ramen
clips still spawn mid-ground bystanders). So we filter the OUTPUT instead of
fighting generation — exactly the philosophy of the fade filter.

Check: detect *main* people (COCO person boxes above a size threshold, so
distant specks on e.g. an adjacent basketball court don't count) in the
FIRST / MIDDLE / LAST frame. Pass rule (see verdict()): the FIRST frame must
equal the expected count (full cast at the open, no extras), and no later
frame may EXCEED it. A transient under-count mid-clip (occlusion / subject
drifts out of frame) is tolerated; an extra person anywhere is not.

Three frames (not all) on purpose: cheap. It's a best-effort cleanliness gate,
not absolute — a setting can have a plausible stray bystander; pair it with
seed retry in the generator.

Box-size threshold is the strictness knob: --min-area-frac is the fraction of
the frame a person box must cover to count as "present". 0.015 (1.5%) keeps
sizable mid-ground intruders in, lets faraway background figures slide.

Exit: 0 = keep (all sampled frames == expected), 1 = drop, 2 = error.
Prints a JSON metrics line to stdout (keys: counts, expected, pass, ...).

Usage:
    count_people_filter.py --video clip.mp4 --expected 3 \\
        --score 0.9 --min-area-frac 0.015
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as iio
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

COCO_PERSON = 1

# COCO (91-class, torchvision FasterRCNN_ResNet50_FPN_Weights.COCO_V1) label ids
# for the animal classes used by the non-human character scenes. The detector
# is reliable on these photoreal animals; species with no COCO class (snake,
# fox, ...) use the "noperson" gate instead (see verdict_subject / build_prompts).
COCO_ANIMALS = {
    "bird": 16, "cat": 17, "dog": 18, "horse": 19, "sheep": 20,
    "cow": 21, "elephant": 22, "bear": 23, "zebra": 24, "giraffe": 25,
}

_MODEL = None


def coco_id(name):
    """COCO label id for a gate class name ('person' or a COCO animal)."""
    if name == "person":
        return COCO_PERSON
    if name in COCO_ANIMALS:
        return COCO_ANIMALS[name]
    raise ValueError(f"no COCO class for gate {name!r}")


def _model(device):
    global _MODEL
    if _MODEL is None:
        _MODEL = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        ).eval().to(device)
    return _MODEL


def verdict(counts, expected):
    """Pass/fail rule shared by the CLI and the batched driver.

    `counts` is ordered [first, ...middle(s)..., last]. Rule:
      - FIRST frame must == expected: the opening must show the full cast and
        no extras (the first frame anchors the scene, like an I2V first frame).
      - later frames must NOT exceed expected: a transient under-count from
        occlusion or the subject drifting out of frame is acceptable, but an
        extra person appearing anywhere is not.
    """
    if not counts:
        return False
    if counts[0] != expected:
        return False
    return all(c <= expected for c in counts[1:])


def count_class(frame, device, score_thr, min_area_frac, label=COCO_PERSON):
    """Count *main* boxes of one COCO `label` (default person) in a frame —
    detections above `score_thr` whose box covers >= `min_area_frac` of the
    frame (so distant specks don't count)."""
    H, W = frame.shape[:2]
    t = torch.from_numpy(frame).permute(2, 0, 1).float().div(255).to(device)
    with torch.no_grad():
        out = _model(device)([t])[0]
    keep = (out["labels"] == label) & (out["scores"] > score_thr)
    b = out["boxes"][keep]
    if b.numel() == 0:
        return 0
    areas = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])) / (H * W)
    return int((areas >= min_area_frac).sum())


def count_main_people(frame, device, score_thr, min_area_frac):
    """Backward-compatible person counter (== count_class with COCO_PERSON)."""
    return count_class(frame, device, score_thr, min_area_frac, COCO_PERSON)


def verdict_subject(person_counts, target_counts, expected, gate):
    """Cast-cleanliness verdict, generalized over subject type (see
    build_prompts.py for the gate vocabulary). `person_counts` and (for animal
    gates) `target_counts` are ordered [first, ...middle..., last].

      - person   : human scene -> the original verdict() rule on people.
      - none      : human-shaped non-human (robot/skeleton) -> the person
                    detector fires on the SUBJECT, so we can't use it; pass and
                    rely on the fade gate + seed retry.
      - <else>    : non-human -> NO human may appear in any sampled frame
                    (the one robust check across all non-human subjects).
          - noperson : that's the whole gate (snake/fox/teddy/droid).
          - <animal> : additionally the COCO animal must be present at the open
                       (first frame >= 1) and never exceed `expected` (no
                       spawned extra animals); transient mid-clip under-count
                       (occlusion / detector miss) is tolerated.
    """
    if gate == "person":
        return verdict(person_counts, expected)
    if gate == "none":
        return True
    if not person_counts or any(p > 0 for p in person_counts):
        return False
    if gate == "noperson":
        return True
    # detectable-animal gate
    if not target_counts or target_counts[0] < 1:
        return False
    return all(c <= expected for c in target_counts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--expected", required=True, type=int)
    ap.add_argument("--score", type=float, default=0.9)
    ap.add_argument("--min-area-frac", type=float, default=0.015)
    ap.add_argument("--gate", default="person",
                    help="person | <coco-animal> | noperson | none "
                         "(see build_prompts.py).")
    ap.add_argument("--no-filter", action="store_true",
                    help="Report counts but always pass (exit 0).")
    args = ap.parse_args()

    if not args.video.exists():
        print(json.dumps({"error": f"missing {args.video}"}))
        return 2

    reader = iio.get_reader(str(args.video))
    frames = [np.asarray(f, dtype=np.uint8) for f in reader]
    reader.close()
    n = len(frames)
    if n == 0:
        print(json.dumps({"error": "no frames"}))
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    idxs = sorted({0, n // 2, n - 1})  # first / middle / last

    if args.gate == "none":
        person_counts, target_counts = None, None
    else:
        person_counts = [count_class(frames[i], device, args.score,
                                     args.min_area_frac, COCO_PERSON)
                         for i in idxs]
        if args.gate in ("person", "noperson"):
            target_counts = person_counts if args.gate == "person" else None
        else:
            tid = coco_id(args.gate)
            target_counts = [count_class(frames[i], device, args.score,
                                         args.min_area_frac, tid) for i in idxs]

    ok = verdict_subject(person_counts, target_counts, args.expected, args.gate)
    print(json.dumps({
        "video": str(args.video),
        "expected": args.expected,
        "gate": args.gate,
        "frame_idxs": idxs,
        "person_counts": person_counts,
        "target_counts": target_counts,
        "score": args.score,
        "min_area_frac": args.min_area_frac,
        "pass": bool(ok),
    }))
    return 0 if (ok or args.no_filter) else 1


if __name__ == "__main__":
    sys.exit(main())
