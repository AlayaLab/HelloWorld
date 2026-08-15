#!/usr/bin/env python
"""Subject-displacement gate — drop training clips whose SUBJECT moved (not the
camera), because that corrupts the Pi3X static-scene warp and degrades the
LoRA's camera-following (CamFollow).

Why GT-vs-WARP (not raw bbox motion): under camera pan/orbit/dolly a *static*
subject's bbox also translates & scales in the frame — so raw image-space motion
can't tell subject-motion from camera-motion, and would wrongly drop exactly the
high-camera-motion clips we want. The warp reconstructs the scene under the SAME
camera path, so comparing the subject's box in GT vs WARP is camera-invariant:
  * centroid distance  -> lateral subject motion (e.g. an elephant strolling)
  * area ratio         -> axial subject motion   (e.g. a dog trotting toward cam)
A clip fails if either exceeds its threshold on any sampled frame.

Only COCO-detectable subjects are gated (person + the animal classes); for
non-detectable subjects (snake/toy/droid via gate noperson) the detector can't
localize them, so we rely on the lying/resting prompt + the visibility filter.

    python subject_motion_filter.py --prep-dir data/prepared \\
        --prompts-tsv data/prompts.tsv --out data/prepared/subject_motion.jsonl \\
        [--max-centroid 0.10] [--max-area-ratio 1.6]
"""
from __future__ import annotations
import argparse, json, glob, os, sys
from pathlib import Path
import numpy as np
import imageio.v2 as iio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import count_people_filter as cp   # _model, COCO_PERSON, COCO_ANIMALS, coco_id


def gate_to_label(gate: str):
    """COCO label to track for a scene's gate, or None if not RELIABLY detectable.

    Gate only where detection is trustworthy: `person` (human) and the COCO
    animal classes. Skip `none` (human-shaped robot/skeleton — the detector fires
    on the subject but their low-texture warps yield garbage boxes -> false drops)
    and `noperson` (snake/fox/teddy/droid — no COCO class). Those rely on the
    lying/resting prompt + the visibility filter instead.
    """
    if gate == "person":
        return cp.COCO_PERSON
    if gate in cp.COCO_ANIMALS:
        return cp.COCO_ANIMALS[gate]
    return None


def largest_box(frame, device, score_thr, label, min_area_frac=0.01):
    """Largest box of `label` as (cx, cy, area_frac), normalized. None if the
    biggest detection is below min_area_frac (spurious speck -> unreliable)."""
    H, W = frame.shape[:2]
    t = torch.from_numpy(frame).permute(2, 0, 1).float().div(255).to(device)
    with torch.no_grad():
        o = cp._model(device)([t])[0]
    keep = (o["labels"] == label) & (o["scores"] > score_thr)
    b = o["boxes"][keep]
    if b.numel() == 0:
        return None
    ar = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    i = int(ar.argmax())
    area = float(ar[i]) / (H * W)
    if area < min_area_frac:
        return None
    x1, y1, x2, y2 = b[i].tolist()
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, area)


def clip_motion(clip_dir, label, device, score_thr, n_sample):
    """Return (max_centroid_dist, max_area_ratio, n_measured) over sampled frames,
    comparing GT vs WARP subject boxes. None if not measurable (no GT detection)."""
    gt = [np.asarray(f, np.uint8) for f in iio.get_reader(f"{clip_dir}/gt.mp4")]
    wp = [np.asarray(f, np.uint8) for f in iio.get_reader(f"{clip_dir}/warp.mp4")]
    n = min(len(gt), len(wp))
    if n == 0:
        return None
    idxs = sorted(set(int(round(x)) for x in np.linspace(0, n - 1, n_sample)))
    cmax, amax, meas = 0.0, 1.0, 0
    for i in idxs:
        g = largest_box(gt[i], device, score_thr, label)
        w = largest_box(wp[i], device, score_thr, label)
        if g is None or w is None:
            continue
        meas += 1
        cmax = max(cmax, ((g[0] - w[0]) ** 2 + (g[1] - w[1]) ** 2) ** 0.5)
        if g[2] > 0 and w[2] > 0:
            amax = max(amax, max(g[2] / w[2], w[2] / g[2]))
    return (cmax, amax, meas) if meas else None


def load_gates(prompts_tsv):
    """(scene_id) -> gate, from the 7-col prompts TSV."""
    out = {}
    for i, line in enumerate(open(prompts_tsv)):
        c = line.rstrip("\n").split("\t")
        if i == 0 or len(c) < 7:
            continue
        out[c[0]] = c[6]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", required=True, type=Path)
    ap.add_argument("--prompts-tsv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--n-sample", type=int, default=8)
    ap.add_argument("--max-centroid", type=float, default=0.10,
                    help="Drop if GT-vs-warp subject centroid dist exceeds this (frac of frame).")
    ap.add_argument("--max-area-ratio", type=float, default=1.6,
                    help="Drop if GT-vs-warp subject area ratio exceeds this.")
    args = ap.parse_args()

    gates = load_gates(args.prompts_tsv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cp._model(device)
    rows = []
    for cd in sorted(glob.glob(f"{args.prep_dir}/clip_*")):
        meta = json.load(open(f"{cd}/meta.json"))
        sid = meta["image_id"]
        label = gate_to_label(gates.get(sid, "person"))
        rec = {"clip_id": meta["clip_id"], "clip_dir": cd, "image_id": sid,
               "gate": gates.get(sid), "label": label}
        if label is None:
            rec.update(passed=True, reason="not-detectable (skip)", centroid=None, area_ratio=None)
        else:
            m = clip_motion(cd, label, device, args.score, args.n_sample)
            if m is None:
                rec.update(passed=True, reason="no detection (skip)", centroid=None, area_ratio=None)
            else:
                c, a, meas = m
                ok = (c <= args.max_centroid) and (a <= args.max_area_ratio)
                rec.update(passed=bool(ok), reason="ok" if ok else "subject-moved",
                           centroid=round(c, 4), area_ratio=round(a, 3), n_measured=meas)
        rows.append(rec)
        print(f"  {'PASS' if rec['passed'] else 'DROP'} {sid:26} "
              f"c={rec.get('centroid')} a={rec.get('area_ratio')} ({rec['reason']})", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_drop = sum(1 for r in rows if not r["passed"])
    print(f"[subject_motion] {len(rows)} clips; {n_drop} would be dropped -> {args.out}")


if __name__ == "__main__":
    main()
