#!/usr/bin/env python
"""Diverse text-to-video (T2V) training-set prompts.

Design principles:

1. T2V, not I2V. No first-frame image — the prompt is the sole source of
   scene content. So the training set is not tied to any photo set; we design
   a *diverse* scene set directly in text.

2. Two-prompt asymmetry is camera-ONLY. The gen_prompt / train_prompt split
   exists for one reason: the warp controls the camera, so camera words must
   be ABSENT from train_prompt to force the LoRA to read motion from the warp
   (not from text). There is no warp-like signal for appearance — nothing else
   controls it — so scene + appearance + action stay SYMMETRIC in both prompts
   (and at inference). The ONLY difference between gen_prompt and train_prompt
   is the per-family camera clause.

       gen_prompt   = people + setting + CAMERA_CLAUSE + action + constraints
       train_prompt = people + setting +    (none)     + action + constraints

3. Diversity is the point. Vary settings, people COUNT (1-3, the paper's
   <=3-character scope), APPEARANCE (age / ethnicity / build / hair / dress),
   and the greeting ACTION (wave / nod / thumbs-up / salute / tip-hat). The
   LoRA should learn camera-follow + viewpoint-directed interaction
   independent of any particular look, so a broad distribution helps it
   generalize. (Note: actively varied appearance != dropping appearance —
   dropping it just cedes to LTX's narrow default prior, which is why the
   I2V kitchen clip spawned an unprompted head-mounted display.)

Preserved tricks (unchanged):
  - Bounded camera vocabulary: never the noun "camera" (LTX renders a camera
    object); "side arc" not "orbital tracking shot" (the latter -> unbounded
    360°); no "slow" (LTX reads it as "barely move"). Clauses in FAMILIES.
  - Extra-character suppression: positive-only "these are the only people"
    in BOTH prompts (LTX hallucinates people when pan/orbit reveals new
    regions; must be in the training text to take effect — distilled mode is
    CFG-less).
  - Fade-tail crop + positive nudge + last-1s fade filter live in
    generate_data.py / crop_and_check_fade.py.
  - Pi3X visibility filter: downstream, unchanged.

Output: 7-column TSV
  (scene_id, family, gen_prompt, train_prompt, expected_count, subject_type, gate).
  N scenes x 6 families. Run:  python build_prompts.py > prompts.tsv

Non-human characters. To probe the LoRA's GENERALIZATION beyond
people, `--include-nonhuman` appends a small slice of non-human "character"
scenes — animals (cat/dog/snake/elephant/fox) and toys/robots
(teddy/skeleton/humanoid-robot/droid). They follow exactly the same camera-only
two-prompt asymmetry and positive-only rules as the human scenes; the only
differences are subject-appropriate wording and the per-clip quality `gate`
(see below). With the 27 human scenes this adds 5 animal + 4 toy/robot scenes
=> ~75% human / ~14% animal / ~11% toy-robot — the target mix
(roughly the requested 75/15/10; clips track scene ratios after the gates).

The `gate` column tells the generator how to clean each clip's cast (the
people-count gate is COCO-person-only and would wrongly drop every non-human
clip — and would even false-drop human-shaped robots/skeletons detected as
people). Values:
  - person    : human scene; first frame == expected, later frames <= expected.
  - <animal>  : COCO-detectable animal (cat/dog/bird/elephant/...); zero humans
                may appear, the animal must be present at the open and never
                exceed `expected`.
  - noperson  : non-human, not human-shaped (snake/fox/teddy/droid); zero humans
                may appear (the only robust check), no subject count.
  - none      : human-shaped non-human (humanoid robot/skeleton); the person
                detector fires on the SUBJECT, so skip the cast check entirely
                and rely on the fade gate + seed-retry.
"""
from __future__ import annotations

import argparse
import sys

# 6 motion families spanning static / pan / dolly / orbit with L/R variants for
# direction diversity. camera_clause=None -> static.
# Wording is bounded on purpose (see module docstring).
FAMILIES = [
    ("static",      None),
    ("pan_left",    "a panning sweep from right to left"),
    ("pan_right",   "a panning sweep from left to right"),
    ("dolly",       "a forward dolly, advancing through the scene toward the subject"),
    ("orbit_left",  "a side arc moving to the left around the subject"),
    ("orbit_right", "a side arc moving to the right around the subject"),
]

# Each scene: a diverse (count, setting, appearance, action) combination.
#   people  — count + appearance, woven (subject of the opening clause)
#   setting — where/what they're doing (ends the opening sentence)
#   pronoun — how the action clause refers back to them
#   action  — full viewpoint-directed social-action predicate (correct
#             conjugation baked in). One action per scene drawn from a broad
#             social-action vocabulary (greeting/approval/disapproval/deictic/
#             reaction/affection), so the training set spans the full
#             vocabulary instead of only "wave hello".
#             Always directed TOWARD the viewer; an in-place gesture (no walking).
# Counts: 1-3 only (paper scope). Photoreal throughout (stylized = later slice).
SCENES = [
    # ---- trios (3) ----
    dict(id="kitchen_cards_trio", people="Three friends in their thirties — a Black woman with long box braids, a slim South Asian man in glasses, and a stocky white man with a beard",
         setting="seated around a wooden table in a sunlit kitchen, playing cards spread between them",
         pronoun="the three friends", action="turn their faces toward the viewer and warmly wave hello with bright smiles"),  # wave
    dict(id="street_skaters_trio", people="Three young skateboarders — a freckled boy, a Latino boy in a beanie, and a girl with green-dyed hair",
         setting="sitting close together on a low curb on a quiet, empty graffiti-painted street corner, skateboards across their laps",
         pronoun="the three skaters", action="turn to the viewer and flash a peace sign with grins"),  # peace-sign
    dict(id="office_breakroom_trio", people="Three coworkers — a tall Black man in a dress shirt, a petite East Asian woman with a bob, and an older white man in suspenders",
         setting="gathered around the counter of a bright, empty office breakroom holding coffee mugs",
         pronoun="the three coworkers", action="turn to the viewer and give an enthusiastic thumbs-up"),  # thumbs-up
    dict(id="basketball_court_trio", people="Three teenagers in basketball jerseys, one tall and Black, one short and Filipino, one redheaded and freckled",
         setting="standing on an empty outdoor basketball court holding a ball",
         pronoun="the three players", action="turn to the viewer and pump their fists in the air, cheering"),  # cheer
    dict(id="backyard_bbq_trio", people="Three neighbors — a burly bald man holding a spatula, a curvy Latina woman with hoop earrings, and a slim older Indian man",
         setting="standing around a smoking grill in a suburban backyard",
         pronoun="the three neighbors", action="turn to the viewer and clap warmly in greeting"),  # clap
    dict(id="university_quad_trio", people="Three students — a Black man in glasses, a young woman in a hijab, and a punk with a mohawk",
         setting="lounging on the grass of a near-empty university quad with backpacks beside them",
         pronoun="the three students", action="turn to the viewer and shrug with wry smiles"),  # shrug
    dict(id="rooftop_garden_trio", people="Three urban gardeners — a tall Scandinavian woman, a short Latino man, and a bearded hipster",
         setting="tending potted plants on a green rooftop garden, the city skyline behind them",
         pronoun="the three gardeners", action="turn to the viewer and form heart shapes with their hands"),  # heart-hands
    dict(id="riverbank_picnic_trio", people="Three friends in their twenties — a Black woman in a headwrap, a tattooed white man, and an Asian woman in a sunhat",
         setting="sitting on a checkered blanket on a grassy riverbank",
         pronoun="the three friends", action="turn to the viewer and burst into warm laughter"),  # laugh
    # ---- pairs (2) ----
    dict(id="cafe_patio_elderly_pair", people="An elderly couple, a silver-haired woman in a floral blouse and a balding man in a cardigan",
         setting="seated across from each other at a small round table on a quiet, empty cobbled cafe patio",
         pronoun="the couple", action="smile and nod warmly in greeting toward the viewer"),  # nod
    dict(id="park_bench_teens_pair", people="Two teenage girls, one with curly red hair and freckles, the other with a high ponytail and a denim jacket",
         setting="sitting on a park bench under a leafy tree in an empty park",
         pronoun="the two girls", action="turn to the viewer and give playful winks"),  # wink
    dict(id="beach_boardwalk_pair", people="Two athletic men in their twenties, one Black with a shaved head, one tanned with a man-bun, in board shorts",
         setting="standing on an empty wooden beach boardwalk with the ocean behind them",
         pronoun="the two men", action="give the viewer a big relaxed wave and grin"),  # wave
    dict(id="library_students_pair", people="Two university students, a young woman in a hijab and a lanky boy with headphones around his neck",
         setting="seated at a long wooden table in a deserted, quiet library, books stacked around them",
         pronoun="the two students", action="look up at the viewer and raise a finger to their lips in a shush"),  # shush
    dict(id="gym_floor_pair", people="Two athletic women in workout gear, one Black with cornrows, one with a blonde ponytail",
         setting="standing by the weight racks on an empty gym floor",
         pronoun="the two women", action="turn to the viewer and make an OK sign"),  # OK-sign
    dict(id="train_station_elderly_pair", people="An older couple in matching travel coats, the man leaning on a cane",
         setting="seated on a bench beneath a grand train-station clock on a deserted concourse",
         pronoun="the couple", action="turn to the viewer and blow a kiss in farewell"),  # blow-kiss
    dict(id="ramen_shop_pair", people="Two friends, a heavyset Japanese man and a freckled Western woman",
         setting="seated at the counter of an otherwise empty ramen shop with bowls in front of them",
         pronoun="the two friends", action="turn to the viewer and beckon them to come over"),  # beckon-come
    dict(id="cabin_porch_pair", people="A cozy older couple in knit sweaters cradling mugs",
         setting="sitting on a snowy log-cabin porch wrapped in blankets",
         pronoun="the couple", action="turn to the viewer and smile warmly"),  # smile
    dict(id="night_market_pair", people="Two young tourists with cameras around their necks, a tall Black man and a Thai woman",
         setting="standing beside a glowing food stall at a near-empty night market",
         pronoun="the two tourists", action="turn to the viewer and point excitedly toward the lens"),  # point-at-cam
    dict(id="vineyard_row_pair", people="A sun-tanned vintner in a wide-brimmed hat and a young apprentice with a ponytail",
         setting="standing between rows of grapevines in a sunlit vineyard",
         pronoun="the two", action="turn to the viewer and raise a hand in a friendly salute"),  # salute
    dict(id="plaza_fountain_pair", people="Two fashionable women, one tall with an afro, one petite with a pixie cut",
         setting="sitting on the edge of a stone fountain in an empty city plaza",
         pronoun="the two women", action="turn to the viewer with a surprised, delighted gasp"),  # gasp
    # ---- solos (1) ----
    dict(id="rooftop_dusk_solo", people="A young Latina woman with wavy dark hair in a yellow sundress",
         setting="leaning on a rooftop railing at dusk with city lights behind her",
         pronoun="the woman", action="turns toward the viewer and gives a graceful bow"),  # bow
    dict(id="forest_trail_solo", people="A bearded middle-aged hiker in a red flannel shirt with a backpack",
         setting="pausing on a leaf-strewn forest trail",
         pronoun="the hiker", action="turns to the viewer with a playful eye-roll"),  # eye-roll
    dict(id="subway_platform_solo", people="A stylish young Korean man in a long black coat with dyed-blond hair",
         setting="waiting alone on a deserted tiled subway platform",
         pronoun="the man", action="turns to the viewer and crosses his arms"),  # cross-arms
    dict(id="mountain_overlook_solo", people="A weathered older man with a grey beard in a wool sweater",
         setting="standing at a mountain overlook with peaks behind him",
         pronoun="the man", action="turns to the viewer and raises a palm in a stop gesture"),  # palm-stop
    dict(id="parking_night_solo", people="A young man in a hoodie with a buzz cut",
         setting="leaning against a concrete pillar in a dim parking garage at night",
         pronoun="the man", action="turns toward the viewer and shakes his head"),  # shake-head
    dict(id="bookstore_aisle_solo", people="A bespectacled middle-aged woman with a grey bun in a cozy sweater",
         setting="browsing a tall shelf in a deserted, warm wooden bookstore aisle",
         pronoun="the woman", action="turns to the viewer with a facepalm"),  # facepalm
    dict(id="desert_highway_solo", people="A lone biker in a leather jacket with long grey hair",
         setting="standing beside a motorcycle on an empty desert highway",
         pronoun="the biker", action="turns to the viewer and gives a thumbs-down"),  # thumbs-down
    dict(id="fishing_pier_solo", people="An old fisherman with a weathered face in a yellow rain hat",
         setting="sitting on a wooden fishing pier with a rod",
         pronoun="the fisherman", action="turns to the viewer, smiles, and tips his hat in greeting"),  # tip-hat
]

# POSITIVE-ONLY phrasing. A "strengthened" negative version ("no bystanders, no
# background figures, no one at other tables, no one in the distance")
# BACKFIRED: LTX-2.3 distilled is CFG-less, so naming the unwanted nouns
# (tables / bystanders / distance) made the model attend to and RENDER them —
# background tables-with-people appeared MORE than with the old short clause.
# Same trap as "no walking". So assert only what IS present and rely on the
# people-count gate to drop any extras that still slip through.
NO_EXTRAS_CLAUSE = "These are the only people in the scene; the surroundings are open and calm."

# STATIONARY-SUBJECT constraint. Under T2V, LTX sometimes has the subjects
# translate WITH the camera (e.g. walking backward as the framing dollies back).
# That breaks the Pi3X warp: the warp assumes a static scene + moving camera, so
# subject translation corrupts the recovered geometry.
#
# Phrase POSITIVELY, never as negation. LTX-2.3 distilled is CFG-less, so a
# negative clause ("no walking, no stepping") just makes the model attend to
# "walking"/"stepping" and can INDUCE the very motion we want gone — same lesson
# as the no-extras clause. So describe the desired STILL state explicitly
# (remain still / static / fixed pose / only hands+face move) with zero
# "no/not/without/walk/step/drift" tokens. Subject-behavior content (not a
# camera clause) -> stays SYMMETRIC in gen+train; avoids the noun "camera".
STATIONARY_CLAUSE = ("They hold a relaxed photo pose, bodies settled and steady, "
                     "gesturing with their hands.")
TAIL = f"{NO_EXTRAS_CLAUSE} {STATIONARY_CLAUSE} Photoreal, natural lighting."


# ======================================================================
# NON-HUMAN CHARACTERS (appended only with --include-nonhuman)
# ======================================================================
# Same design rules as the human scenes, just non-human subjects:
#   * camera-ONLY two-prompt asymmetry (the per-scene tail is symmetric);
#   * POSITIVE-only phrasing (no "no/not/without" tokens) — describe the
#     desired still state, never forbid motion (CFG-less LTX renders forbidden
#     tokens);
#   * IN-PLACE expressive actions only. The viewer-directed "interaction" is a
#     head/limb/face gesture (tongue-flick, ear-perk, paw-raise, wing-flutter,
#     mechanical wave) — NEVER locomotion (trot/waddle/approach), which would
#     translate the subject and corrupt the Pi3X static-scene warp, exactly as
#     for the human scenes.
# Each scene carries an explicit count `n` and a `gate` (see module docstring).
# `still`  = subject-specific stationary clause (positive, in-place).
# `noun`   = how the no-extras clause names the subject.
NONHUMAN_SCENES = [
    # ---- animals (5): ~14% of the 27+9 scene set (target ~15%) ----
    # LYING / RESTING poses: standing/sitting animals
    # WALK under T2V (dog approached, elephant strolled), translating the subject
    # and corrupting the Pi3X static-scene warp -> the LoRA's camera-following
    # (CamFollow) regressed. Lying-down poses keep the subject planted; the
    # viewer-directed interaction is a head/trunk/ear gesture from rest. Same
    # lesson as the human STATIONARY_CLAUSE, escalated to an explicit resting pose.
    # (Backstopped by the GT-vs-warp subject-displacement filter downstream.)
    dict(id="windowsill_cat_solo", subject_type="animal", gate="cat", n=1, noun="creature",
         people="A fluffy orange tabby cat lying curled up",
         setting="resting on a sunlit wooden windowsill",
         pronoun="the cat",
         action="turns its head toward the viewer, blinks slowly, and gives a lazy flick of one ear",
         still="It stays lying in place, its body resting and still, moving only its head."),
    dict(id="yard_dog_solo", subject_type="animal", gate="dog", n=1, noun="creature",
         people="A golden retriever lying down resting",
         setting="on the grass of a sunlit backyard",
         pronoun="the dog",
         action="lifts and turns its head toward the viewer, ears perking, giving a soft open-mouthed pant",
         still="It stays lying down in place, its body resting on the grass, moving only its head and ears."),
    dict(id="log_snake_solo", subject_type="animal", gate="noperson", n=1, noun="creature",
         people="A slender green tree snake lying coiled and still",
         setting="on a mossy fallen log in a forest clearing",
         pronoun="the snake",
         action="raises its head toward the viewer and flicks its forked tongue out again and again",
         still="It stays lying coiled in place, its body settled on the log, moving only its head and tongue."),
    dict(id="savanna_elephant_solo", subject_type="animal", gate="elephant", n=1, noun="creature",
         people="A young elephant calf lying down resting",
         setting="in tall golden savanna grass",
         pronoun="the elephant",
         action="lifts its head and curls its trunk up toward the viewer, ears flapping",
         still="It stays lying down in place, its body resting in the grass, moving only its head, trunk, and ears."),
    dict(id="snow_fox_solo", subject_type="animal", gate="noperson", n=1, noun="creature",
         people="A red fox lying curled up with its bushy tail wrapped around it",
         setting="resting in a snowy woodland clearing",
         pronoun="the fox",
         action="lifts its head toward the viewer, ears perking, tilting its head with a curious look",
         still="It stays lying curled in place, its body resting in the snow, moving only its head and ears."),
    # ---- toys / figures / robots (4): ~11% of the scene set (target ~10%) ----
    dict(id="shelf_teddy_solo", subject_type="toy", gate="noperson", n=1, noun="figure",
         people="A small brown plush teddy bear",
         setting="sitting on a wooden shelf in a cozy room",
         pronoun="the teddy bear",
         action="comes alive, turns toward the viewer, and lifts a paw in a friendly wave",
         still="It stays seated in place, its body settled on the shelf, moving only its head and paw."),
    dict(id="porch_skeleton_solo", subject_type="toy", gate="none", n=1, noun="figure",
         people="A plastic posable skeleton figure",
         setting="standing on a wooden porch",
         pronoun="the skeleton",
         action="comes alive, turns its skull toward the viewer, and raises a bony hand in a wave",
         still="It stays standing in place, its feet planted, moving only its skull and arm."),
    dict(id="lab_humanoid_robot_solo", subject_type="robot", gate="none", n=1, noun="robot",
         people="A sleek white humanoid robot",
         setting="standing in a bright minimalist laboratory",
         pronoun="the robot",
         action="turns toward the viewer, tilts its head, and raises one arm in a smooth mechanical wave",
         still="It stays standing in place, its feet planted, moving only its head and arm."),
    dict(id="desert_droid_solo", subject_type="robot", gate="noperson", n=1, noun="robot",
         people="A small domed astromech droid on three legs",
         setting="parked on an open desert plain",
         pronoun="the droid",
         action="swivels its domed head toward the viewer, beeps, and bobs in a little nod",
         still="It stays parked in place, its legs planted, moving only its dome and head."),
]


def _nonhuman_tail(sc):
    """Per-scene tail for a non-human character: positive no-extras clause +
    subject-specific stationary clause. Symmetric across gen/train (like the
    human TAIL) — only the camera clause differs between the two prompts."""
    extras = f"It is the only {sc['noun']} in the scene; the surroundings are open and calm."
    return f"{extras} {sc['still']} Photoreal, natural lighting."


def _compose(sc, camera_clause, tail):
    cam = "Eye-level long take." if camera_clause is None else f"Eye-level long take with {camera_clause}."
    return f"{sc['people']} {sc['setting']}. {cam} Midway through the shot, {sc['pronoun']} {sc['action']}. {tail}"


def build_gen_prompt(sc, camera_clause, tail):
    """Generation prompt — carries the family camera clause for motion."""
    return _compose(sc, camera_clause, tail)


def build_train_prompt(sc, tail):
    """Training/eval prompt — identical to gen MINUS the camera clause
    (static framing wording). Appearance + scene + action stay symmetric."""
    return _compose(sc, None, tail)


# Expected on-screen people count, derived from the id suffix. Used by the
# count_people_filter.py post-check (analogous to the fade filter): drop a
# generated clip whose first/mid/last frame doesn't show this many *main*
# (box-size-thresholded) people. Not absolute — a setting can plausibly have
# stray bystanders (a ramen shop with other diners) — so it's a best-effort
# cleanliness gate paired with seed retry, not a guarantee.
SUFFIX_COUNT = {"solo": 1, "pair": 2, "trio": 3}


def expected_count(scene_id):
    return SUFFIX_COUNT[scene_id.rsplit("_", 1)[1]]


def _iter_scenes(include_nonhuman, only_nonhuman):
    """Yield (scene, expected_count, subject_type, gate, tail) tuples.

    Human scenes are subject_type=human / gate=person / shared TAIL.
    Non-human scenes carry explicit n / gate / per-scene tail."""
    if not only_nonhuman:
        for sc in SCENES:
            yield sc, expected_count(sc["id"]), "human", "person", TAIL
    if include_nonhuman or only_nonhuman:
        for sc in NONHUMAN_SCENES:
            yield sc, sc["n"], sc["subject_type"], sc["gate"], _nonhuman_tail(sc)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-nonhuman", action="store_true",
                    help="Append the non-human character scenes "
                         "(6 animal + 4 toy/robot) after the human scenes.")
    ap.add_argument("--only-nonhuman", action="store_true",
                    help="Emit ONLY the non-human scenes (smoke / inspection).")
    args = ap.parse_args()

    out = sys.stdout
    out.write("scene_id\tfamily\tgen_prompt\ttrain_prompt\texpected_count\t"
              "subject_type\tgate\n")
    ids = set()
    for sc, n, subject_type, gate, tail in _iter_scenes(args.include_nonhuman,
                                                         args.only_nonhuman):
        assert sc["id"] not in ids, f"dup id {sc['id']}"
        ids.add(sc["id"])
        train = build_train_prompt(sc, tail)
        for fam_name, camera_clause in FAMILIES:
            gen = build_gen_prompt(sc, camera_clause, tail)
            for p in (gen, train):
                assert "\t" not in p and "\n" not in p
            out.write(f"{sc['id']}\t{fam_name}\t{gen}\t{train}\t{n}\t"
                      f"{subject_type}\t{gate}\n")


if __name__ == "__main__":
    main()
