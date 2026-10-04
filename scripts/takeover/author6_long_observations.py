#!/usr/bin/env python3
"""Independently authored TRAINING-ONLY draft; no model inference or corpus inputs.

Run with the project Python. Outputs are new-only. The only runtime input bytes
are the explicitly named tokenizer, detector config, hf_encoder.py (only its
window_jobs function is executed), and this source for its provenance hash.
This is deliberately artificial longitudinal-observation stress material, not
naturally sampled notes. Gold is recorded while composing literal pieces.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import unicodedata

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
SEED = 61004117
POLICY = "clinical-phi-v2"
TOKENIZER = ROOT / "models/takeover-mbbase-mix3-001/step-003081/hf/tokenizer.json"
CONFIG = ROOT / "models/takeover-mbbase-mix3-001/step-003081/config.json"
RUNTIME = ROOT / "src/clinical_deid/hf_encoder.py"
CORPUS = ROOT / "data/tk6_train_long_observations.jsonl"
REPORT = ROOT / "reports/takeover/mix3_001/author6_long_observations/author_report.json"
LABELS = "NAME DATE ADDRESS PHONE EMAIL MRN SSN ACCOUNT ID URL IP AGE LOCATION FACILITY".split()
LAYOUTS = ("connected_prose", "bullet_rounds", "observation_grid", "handoff_sections", "numbered_checks")

# Every context and clause below was written for this bank. No corpus, model
# output, external template, clinical record, or development example was read.
# A tuple contains a context key, opening, focal purpose, and eight observations.
FAMILIES = [
    ("rehab_balance", "Balance practice resumed after a seated rest.", "compare balance before and after the supported walk", [
        "feet remained planted during the chair rise", "turning required a brief pause but no loss of balance",
        "the walker stayed within comfortable reach", "weight shifted evenly with light rail support",
        "the return to sitting was slow and controlled", "a visual target helped steady the stance",
        "the next step followed the verbal cue", "ankle movement was comfortable while seated"]),
    ("bedside_shift", "The bedside shift began with a comfort check.", "track comfort through routine bedside care", [
        "the call button was within reach", "the blanket was adjusted without interrupting conversation",
        "the head of the bed stayed at the requested angle", "the water cup could be reached without strain",
        "a position change relieved pressure at the shoulder", "the room remained quiet during rest",
        "the bedside light was lowered after the check", "the next care step was explained before contact"]),
    ("transport_handoff", "Transport staff completed a seated readiness check.", "observe tolerance of the transfer sequence", [
        "the footrests supported both feet", "the brake check preceded the chair transfer",
        "the transfer belt stayed flat over clothing", "the turn into the corridor was tolerated",
        "conversation continued during the short pause", "the receiving surface was aligned before movement",
        "the return transfer used the same verbal count", "the destination team repeated the positioning plan"]),
    ("home_visit", "The home visit focused on the usual morning routine.", "review safe movement between rest and activity", [
        "the walking route was clear of loose items", "the chair arms provided support for rising",
        "the usual slippers fit without rubbing", "the rest interval was taken before fatigue increased",
        "the cup was carried with both hands", "the sitting surface allowed both feet to touch the floor",
        "the exercise sheet remained visible from the chair", "the familiar route was completed at a comfortable pace"]),
    ("respiratory_rounds", "Breathing comfort was reviewed during quiet activity.", "compare speech and breathing with gentle movement", [
        "speech continued in complete phrases", "the shoulders stayed relaxed during exhalation",
        "a slow breath followed the demonstration", "the cough cleared a small amount of thin mucus",
        "the breathing pattern settled with seated rest", "the mouthpiece was held without a leak",
        "the chest movement appeared even", "the next attempt began after a comfortable pause"]),
    ("swallow_observation", "A supervised meal observation began with upright positioning.", "follow pacing during the observed meal", [
        "the small spoonful was taken at the planned pace", "the next bite waited until the mouth was clear",
        "the cup was set down between sips", "the voice remained clear after the observed sip",
        "a pause allowed the bite to be completed", "the tray stayed at a comfortable height",
        "the upright posture was maintained through the check", "the utensil returned to the tray between bites"]),
    ("wound_dressing", "A dressing review was completed in a comfortable position.", "monitor tolerance of the dressing routine", [
        "the outer wrap remained dry", "the surrounding skin was comfortable to light touch",
        "the limb was supported during the check", "the tape edge lay flat without pulling",
        "the dressing margin remained in place", "the position change caused no new discomfort",
        "the clean cover stayed clear of the joint crease", "the skin check ended before the limb was lowered"]),
    ("dialysis_recovery", "The recovery chair was checked after the scheduled treatment.", "observe comfort before standing from the recovery chair", [
        "the access cover remained dry", "the supported arm stayed relaxed",
        "the first sitting pause was tolerated", "the hand remained warm during the check",
        "the chair rise was delayed until readiness was confirmed", "the small drink was taken without nausea",
        "the standing pause allowed a comfort reassessment", "the final seated check was similar to the first"]),
    ("infusion_comfort", "Comfort checks continued around a routine infusion visit.", "follow the arm position and general comfort", [
        "the supported arm was still during the check", "the sleeve stayed clear of the line connection",
        "the site dressing lay smooth against the skin", "the tubing rested without tension",
        "the hand could open and close comfortably", "the chair angle was adjusted after discussion",
        "the next check began with a brief explanation", "the line remained visible from the chair side"]),
    ("post_procedure_rest", "Recovery observations followed a short procedure.", "follow wakefulness and comfort during recovery", [
        "the eyes opened readily to a spoken cue", "the breathing remained even during rest",
        "the request for a warmer blanket was clear", "the first supported sit was tolerated",
        "the small sip stayed down without nausea", "the room noise was reduced for the next rest interval",
        "the next instruction was repeated correctly", "the position was changed slowly at the stated preference"]),
    ("sleep_observation", "Overnight observations were recorded during routine rounds.", "describe rest and response to gentle checks", [
        "the pillow supported the neck comfortably", "the breathing pattern stayed regular while resting",
        "the door was closed softly after the check", "the bedside noise remained low",
        "the response to a quiet question was appropriate", "the covers stayed clear of the face",
        "the room light was lowered after repositioning", "the return to rest occurred without a further request"]),
    ("neuro_observation", "A focused bedside observation checked simple responses.", "compare attention across short bedside checks", [
        "both hands followed the same simple instruction", "the gaze shifted toward the speaker",
        "the reply matched the question asked", "the familiar sequence was repeated in order",
        "the seated posture remained supported", "the next step was recalled after a brief pause",
        "the object was passed between hands smoothly", "the response remained similar on the repeat check"]),
    ("mobility_circuit", "A short activity circuit alternated walking and seated tasks.", "compare effort across the activity circuit", [
        "the first lap ended with a planned rest", "the turn around the marker was controlled",
        "the chair was reached without rushing", "the handrail was used only during the step task",
        "the pace slowed slightly near the final marker", "the seated task allowed the breathing to settle",
        "the next lap started after a verbal readiness check", "the activity ended with an easy return to sitting"]),
    ("occupational_task", "An everyday-task session used familiar household objects.", "follow hand use during the practical task", [
        "the lid turned with the aid of a textured grip", "the object was moved within easy reach",
        "the supporting hand steadied the container", "the seated position allowed the elbows to rest",
        "the task was divided into comfortable small steps", "the item was set down before fatigue increased",
        "the fingers released the handle without catching", "the final attempt used the same safe sequence"]),
    ("speech_practice", "Communication practice used short everyday exchanges.", "observe clarity and pacing of spoken replies", [
        "the first response was easy to hear", "the sentence was repeated at a slower pace",
        "the pause gave enough time to finish the thought", "the listener confirmed the intended request",
        "the voice remained steady through the short exchange", "the cue was understood without a second explanation",
        "the next phrase began after an easy breath", "the final reply preserved the meaning of the earlier one"]),
    ("pediatric_play", "A play-based observation followed a familiar activity sequence.", "observe comfort during quiet play and movement", [
        "the toy was reached for with both hands", "the floor activity alternated with a seated rest",
        "the next game began after a simple explanation", "the small step was approached with support nearby",
        "the favorite object held attention during the check", "the turn-taking sequence was followed with a cue",
        "the toy was placed back in the basket", "the activity ended without a new complaint"]),
    ("maternity_comfort", "A comfort round reviewed feeding posture and easy movement.", "follow comfort through position changes and rest", [
        "the supported posture reduced shoulder tension", "the water cup stayed within easy reach",
        "the rise from the chair was slow and comfortable", "the pillow position was adjusted before the next rest",
        "the short walk ended with a seated pause", "the request for help was made before standing",
        "the bed edge offered steady support while sitting", "the next comfort check used the preferred position"]),
    ("behavioral_observation", "Supportive observations continued during a quiet group interval.", "describe engagement during the observation period", [
        "the group direction was followed without repeated prompting", "the reply was brief and relevant",
        "the chosen seat allowed comfortable personal space", "the short break was requested calmly",
        "the planned activity resumed after the pause", "the voice stayed at a conversational volume",
        "the available choices were discussed in turn", "the closing plan was repeated in the same order"]),
    ("low_vision_training", "A practical session used high-contrast household materials.", "follow orientation during the visual task", [
        "the item was found after the lighting was adjusted", "the hand traced the edge of the tray",
        "the large-print line was followed at a comfortable distance", "the object stayed against the contrasting mat",
        "the path was described before the transfer began", "the next target was located after a short pause",
        "the seated position reduced glare from the window", "the final object was placed in its usual location"]),
    ("hearing_support", "A listening check was completed in a quiet room.", "review comprehension during supported conversation", [
        "the speaker faced the listener before beginning", "the first instruction was repeated accurately",
        "the device rested comfortably behind the ear", "the room noise was reduced before the second exchange",
        "the short sentence was understood at a normal pace", "the listener asked for one word to be repeated",
        "the next reply matched the question", "the final direction was restated without a cue"]),
    ("exercise_recovery", "Low-intensity exercise alternated with planned recovery pauses.", "compare effort before and after the recovery pause", [
        "the warm-up ended before the pace increased", "the supported stretch stayed within comfort",
        "the movement became easier after the demonstration", "the next set began after breathing settled",
        "the feet stayed aligned during the slow bend", "the recovery pause lasted two minutes",
        "the final repetition used a shorter comfortable range", "the cool-down ended with a seated check"]),
    ("continence_routine", "A routine-care review focused on comfort and easy access.", "observe the sequence around a planned bathroom visit", [
        "the path to the bathroom remained clear", "the clothing adjustment was completed while seated",
        "the grab bar was reached before the turn", "the call for assistance preceded standing",
        "the return walk used the same steady pace", "the chair was positioned before the transfer",
        "the skin remained comfortable after routine care", "the next planned check was explained during rest"]),
    ("cast_comfort", "A limb-support review checked fit and comfortable movement.", "follow comfort around the supported limb", [
        "the exposed fingers moved easily", "the support stayed clear of the skin crease",
        "the pillow kept the limb at the comfortable height", "the edge padding remained smooth",
        "the hand stayed warm during the check", "the supported position reduced the pulling sensation",
        "the next position change was slow and assisted", "the final finger movement matched the earlier check"]),
    ("oral_care", "Routine oral care was observed with the chair upright.", "review pacing and comfort during the care sequence", [
        "the brush was held with a comfortable grip", "the rinse was taken in a small amount",
        "the cup returned to the tray between steps", "the mouth opened comfortably for the brief check",
        "the head stayed supported during the pause", "the next step followed a clear verbal cue",
        "the lips remained comfortable after the rinse", "the final wipe was tolerated without pulling away"]),
    ("nutrition_round", "A meal-support round reviewed appetite and comfortable access.", "follow appetite and pacing during the meal", [
        "the tray position allowed an easy reach", "the first portion was taken without rushing",
        "the cup was lifted with a steady grip", "the rest pause came before the next serving",
        "the requested food was placed within reach", "the upright position stayed comfortable",
        "the remaining portion was discussed without pressure", "the meal ended at the stated level of comfort"]),
]

STATUS = [
    "alert and able to follow the short instruction", "calm with an even conversational tone",
    "awake and comfortable in the supported position", "quiet during the pause and responsive when addressed",
    "stable in the chair with no new complaint", "well supported during the position change",
    "still while the brief check was completed", "clear in speech and comfortable at rest",
    "ready for the next step after a short pause", "settled after the position adjustment",
    "comfortable with the current pace", "engaged in the brief explanation",
    "responsive to the first spoken cue", "rested after the planned pause",
    "attentive while the next step was demonstrated", "steady during the supported movement",
    "relaxed after the room noise was reduced", "oriented to the immediate task",
    "cooperative with the agreed sequence", "at ease after the comfort adjustment",
    "drowsy but readily responsive during the rest interval", "tired after activity and willing to rest",
    "careful during the turn toward the chair", "comfortable enough to finish the short exchange",
]
ENDS = [
    "A short pause preceded the next activity.", "The next step was explained in plain language.",
    "The request was heard and repeated back.", "No additional assistance was requested at that check.",
    "The same comfortable position was retained.", "The observation was discussed before moving on.",
    "A seated rest was offered between tasks.", "The pace was adjusted to the stated preference.",
    "The response was similar on the next brief check.", "The planned sequence continued after agreement.",
    "Attention returned to the current task.", "The next round began after the room was quiet.",
]
FIRSTS = "Amara|Milo|Leona|Tarek|Nadine|Ivo|Selma|Rafi|Oona|Kiran|Elara|Ronan|Mirela|Dario|Safiya|Lucan|Anika|Soren|Ines|Tomas|Yara|Bastien|Nola|Dev|Priya|Ezra|Alina|Samir|Noemi|Jules|Talia|Idris|Maren|Pavel|Zora|Mika|Lina|Arun|Soraya|Emil|Nuria|Mateo|Dalia|Ari|Liora|Felix|Eleni|Niko|Samira|Adrian|Maëlle|Raúl|Sébastien|İlker|Łucja|Élodie|Renée|Aïcha|Cătălin|Noëlle".split("|")
LASTS = "Vennar|Calderin|Orvale|Densley|Mavren|Larkwell|Teren|Morlace|Sennett|Valdor|Fenwicke|Havren|Solmere|Brinlow|Talwick|Virelli|Nordell|Relland|Dalmere|Kelvara|Orsen|Falloway|Verado|Braskel|Lemori|Tavren|Belwyn|Sarnell|Rosven|Halvern|Corvel|Nemerin|Vardell|Tenwick|Avelin|Merrowe|Dalven|Pelmar|Zevran|Olsarin|Dovran|Sablett|Fennari|Lunwick|Rostell|Norvale|Mireven|Alvero|Vesperin|Braylen|D’Arven|O’Nellar|Venn-Solare|de la Vela|van Merel|St. Varden|Béravel|Nuñelar|Sølvren|Al-Kevan".split("|")
UNUSUAL = ("Alert", "Calm", "Awake", "Quiet", "Stable", "Well", "Still", "Clear", "Care", "He")
STATUS_WORDS = "alert calm awake quiet stable well still clear care he ready settled comfortable responsive attentive steady relaxed oriented tired rested".split()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity_digest(name):
    return digest(unicodedata.normalize("NFC", name).casefold().encode("utf-8"))


class Note:
    def __init__(self):
        self.parts = []
        self.spans = []
        self.length = 0

    def add(self, value, label=None, identity=None):
        start = self.length
        self.parts.append(value)
        self.length += len(value)
        if label:
            span = {"start": start, "end": self.length, "label": label, "text": value}
            if identity:
                span["identity_group"] = identity_digest(identity)
            self.spans.append(span)
        return self

    @property
    def text(self):
        return "".join(self.parts)


class Author:
    def __init__(self):
        self.rng = random.Random(SEED)
        self.tok = Tokenizer.from_file(str(TOKENIZER))
        self.tok.no_truncation()
        self.tok.no_padding()
        self.tokenizer_sha = digest(TOKENIZER.read_bytes())
        self.config = json.loads(CONFIG.read_text())
        source = ast.parse(RUNTIME.read_text())
        fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "window_jobs")
        scope = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(RUNTIME), "exec"), scope)
        self.windows = scope["window_jobs"]
        probe = "clinical note"
        inner = self.tok.encode(probe, add_special_tokens=False).ids
        outer = self.tok.encode(probe, add_special_tokens=True).ids
        self.special_count = len(outer) - len(inner)
        assert outer[1:-1] == inner and self.special_count == 2
        self.content_window = self.config["max_length"] - self.special_count
        self.used = set()
        self.rejected = 0

    def identity(self):
        while True:
            full = self.rng.choice(FIRSTS) + " " + self.rng.choice(LASTS)
            # This extra middle name expands the pool without a suffix scheme.
            if self.rng.random() < .27:
                full = full.split(" ", 1)[0] + " " + self.rng.choice(FIRSTS) + " " + full.split(" ", 1)[1]
            key = identity_digest(full)
            if int(key[:8], 16) % 10 < 8 and key not in self.used:
                self.used.add(key)
                return full
            self.rejected += 1

    def count(self, text):
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    def ordinary(self, family, index, rng, case_mode):
        obs = rng.choice(family[3])
        status = rng.choice(STATUS)
        end = rng.choice(ENDS)
        style = index % 8
        if style == 0:
            value = f"The patient was {status}; {obs}. {end}"
        elif style == 1:
            value = f"On review, {obs}. The patient remained {status}. {end}"
        elif style == 2:
            value = f"Care continued at the agreed pace. The patient appeared {status}, and {obs}. {end}"
        elif style == 3:
            value = f"Observed: {obs}. General response: {status}. {end}"
        elif style == 4:
            value = f"After the rest interval, {obs}. He answered when asked about comfort. He was {status}."
        elif style == 5:
            value = f"The next check found the patient {status}. In the practical task, {obs}. {end}"
        elif style == 6:
            value = f"Rest and activity alternated. During the check, {obs}; the patient was {status}. {end}"
        else:
            value = f"A brief observation confirmed that {obs}. The patient stayed {status}. {end}"
        if case_mode == "lowercase":
            value = value.lower()
        elif case_mode == "uppercase":
            value = value.upper()
        elif case_mode == "mixed_headings" and index % 4 == 0:
            value = "RECHECK — " + value
        return value

    def header(self, family, layout, age):
        headers = {
            "connected_prose": f"{family[1]} The aim was to {family[2]}. Age at review: {age} years.\n\n",
            "bullet_rounds": f"{family[1]}\nCare aim: {family[2]}.\nAge: {age} years.\n\nRounds\n",
            "observation_grid": f"{family[1]}\nReview focus | {family[2]}\nAge | {age} years\nPass | Observation\n",
            "handoff_sections": f"REASON FOR OBSERVATION\n{family[1]}\nCURRENT PLAN\nThe aim is to {family[2]}. Age: {age} years.\nCOURSE\n",
            "numbered_checks": f"{family[1]}\nPlan: {family[2]}; age {age} years.\nChecks follow in order.\n",
        }
        return headers[layout]

    def block(self, content, layout, index):
        if layout == "connected_prose":
            return content + ("\n\n" if index % 3 == 2 else " ")
        if layout == "bullet_rounds":
            return f"- {content}\n"
        if layout == "observation_grid":
            return f"{index + 1} | {content}\n"
        if layout == "handoff_sections":
            return ("INTERVAL REVIEW\n" if index % 4 == 0 else "") + content + "\n"
        return f"{index + 1}) {content}\n"

    def identity_block(self, note, slot, phi, people, alias, q, case_mode):
        patient, clinician, relative = people
        def name(value, who):
            if case_mode == "uppercase":
                value = value.upper()
            elif case_mode == "lowercase":
                value = value.lower()
            note.add(value, "NAME", who)
        if slot == 0:
            note.add("\n")
            if phi:
                note.add("Patient for this observation: ")
                name(patient, patient)
                note.add("; preferred spoken name: ")
                name(alias, patient)
                note.add(". ")
            else:
                note.add("The patient confirmed the preferred form of address. ")
            note.add("Care was discussed while the patient was alert, calm, and well supported.\n")
        elif slot == 1:
            note.add("\n")
            if phi:
                note.add("The patient using the preferred name ")
                name(alias, patient)
                note.add(" asked for a short rest. The reviewing clinician, Dr. ")
                name(clinician, clinician)
                note.add(", MD, confirmed the comfort plan. ")
            else:
                note.add("The patient asked for a short rest. The reviewing clinician confirmed the comfort plan. ")
            note.add("The room was quiet. The patient stayed awake, still, and comfortable; speech was clear.\n")
        else:
            note.add("\nThe care discussion included ")
            if phi:
                name(patient, patient)
                note.add(" and the support person, ")
                name(relative, relative)
            else:
                note.add("the patient and the support person")
            note.add(". He repeated the next step. The seated position was stable and the care plan was understood.\n")

    def extra(self, note, phi, q):
        # Each invented field is self-contained. No copied IDs or live networks.
        extra = q % 8
        note.add("\n")
        if extra == 0:
            note.add("The next review is ")
            if phi:
                note.add(f"November {3 + q % 23}, 2031", "DATE")
            else:
                note.add("after the current sequence")
            note.add(". The earlier trial of the supported position lasted 12 minutes.\n")
        elif extra == 1:
            note.add("The receiving team is based ")
            if phi:
                note.add("at ").add(f"{['Lanternfern','Mossrill','Sablecove','Dovewater'][(q // 8) % 4]} Recovery Rooms", "FACILITY")
                note.add(" in ").add(f"{['North','Upper','East','West'][(q // 8) % 4]} Velmara, CO", "LOCATION")
            else:
                note.add("in the receiving unit")
            note.add(". The transfer plan was repeated in a clear voice.\n")
        elif extra == 2:
            note.add("The callback route was confirmed")
            if phi:
                note.add(": ").add(f"+1 (202) 555-{100 + q % 100:04d} ext. {20 + q % 60}", "PHONE")
                note.add("; ").add(f"visit{q:04d}@notes.example", "EMAIL")
            note.add(". The patient was comfortable while listening.\n")
        elif extra == 3:
            note.add("The home visit destination was reviewed")
            if phi:
                note.add(": ").add(f"{110 + q} Imaginary Fern Walk, Velmara, CO 80000", "ADDRESS")
            note.add(". Rest was offered before the departure discussion.\n")
        elif extra == 4:
            note.add("The paper chart reference was checked")
            if phi:
                note.add(": MRN ").add(f"SYN-A6-{q:06d}", "MRN")
                note.add("; visit account ").add(f"TRAIN6-{q:06d}", "ACCOUNT")
            note.add(". The next movement was explained before it began.\n")
        elif extra == 5:
            note.add("The device route for the remote check was reviewed")
            if phi:
                note.add(": ").add(f"https://rounds.test/session/{q:04d}", "URL")
                note.add("; endpoint ").add(f"198.51.100.{1 + q % 254}", "IP")
            note.add(". The voice remained steady during the check.\n")
        elif extra == 6:
            note.add("The transport reference was read back")
            if phi:
                note.add(": ").add(f"A6-TRANSPORT-{q:05d}", "ID")
            note.add(". The supported position was comfortable.\n")
        else:
            note.add("The earlier visit interval was ")
            if phi:
                # Both interval endpoints belong to one patient-specific phrase.
                note.add(f"April {2 + q % 12}–{5 + q % 12}, 2030", "DATE")
            else:
                note.add("three days")
            note.add(". The care sequence remains familiar.\n")

    def pair(self, fidx, local, q):
        family = FAMILIES[fidx]
        long = local < 12
        rng = random.Random(SEED + q * 1009)
        layout = LAYOUTS[q % 5]
        case_mode = ("sentence", "sentence", "mixed_headings", "lowercase", "uppercase")[q % 5 if q % 3 == 0 else (q // 7) % 5]
        people = [self.identity() for _ in range(3)]
        unusual = local in (fidx % 12, 12 + fidx % 8)
        alias = UNUSUAL[(fidx + (7 if not long else 0)) % len(UNUSUAL)] if unusual else people[0].split()[0]
        age = (9 + q % 7) if fidx == 15 else (24 + q % 62)
        target = ([820, 980, 1210, 1480, 1830, 2190, 2640, 3110, 3690, 4310, 5070, 5960][(local + fidx) % 12]
                  if long else [60, 100, 150, 205, 265, 335, 415, 490][local - 12])
        header = self.header(family, layout, age)
        lines = []
        estimate = self.count(header)
        while len(lines) < 3 or estimate < target:
            k = len(lines)
            line = self.block(self.ordinary(family, k, rng, case_mode), layout, k)
            lines.append(line)
            estimate += self.count(line)
        # Confirm against the whole exact string; component tokenization is
        # used only to avoid repeatedly tokenizing a growing prefix.
        while self.count(header + "".join(lines)) < target:
            k = len(lines)
            lines.append(self.block(self.ordinary(family, k, rng, case_mode), layout, k))
        # Rotate introductions across the document, with alias-before-full
        # examples supported by the explicit preferred-name declaration.
        n = len(lines)
        placements = [0, max(1, n // 2), n]
        rotation = q % 3
        slots = {(placements[i]): (i + rotation) % 3 for i in range(3)}
        pair_id = f"a6-longobs-{q:04d}"
        out = []
        for phi in (False, True):
            note = Note().add(header)
            for i in range(n + 1):
                if i in slots:
                    self.identity_block(note, slots[i], phi, people, alias, q, case_mode)
                if i == n // 3:
                    self.extra(note, phi, q)
                if i < n:
                    note.add(lines[i])
            note.add("\nEnd of the observed sequence. The next check will revisit comfort and readiness.\n")
            enc = self.tok.encode(note.text, add_special_tokens=False)
            windows = list(self.windows(len(enc.ids), self.content_window))
            ownership_edges = sorted({w[2] for w in windows if w[2] > 0} | {w[3] for w in windows if w[3] < len(enc.ids)})
            placements_meta = []
            for s in note.spans:
                if s["label"] != "NAME":
                    continue
                covered = [i for i, (a, b) in enumerate(enc.offsets) if b > s["start"] and a < s["end"]]
                assert covered
                a, b = min(covered), max(covered) + 1
                nearest = min(ownership_edges, key=lambda edge: min(abs(a-edge), abs(b-edge))) if ownership_edges else None
                placements_meta.append({"span_start": s["start"], "span_end": s["end"], "token_start": a, "token_end": b,
                    "relative_char_position": round(s["start"] / len(note.text), 6),
                    "nearest_ownership_boundary": nearest,
                    "distance_to_boundary_tokens": min(abs(a-nearest), abs(b-nearest)) if nearest is not None else None,
                    "crosses_ownership_boundary": any(a < edge < b for edge in ownership_edges)})
            phenomena = ["clinical_status_negative", "name_observation_proximity", "ordinary_pronouns", "ordinary_age",
                         "paired_contextual_contrast", "independent_synthetic_authoring", f"layout_{layout}", f"case_{case_mode}"]
            if long:
                phenomena += ["multiwindow_observation_sequence", "artificial_repeated_progress_stress"]
            if unusual:
                phenomena += ["explicit_personal_alias", "status_word_as_established_alias"]
            if rotation == 1:
                phenomena += ["alias_before_full_identity"]
            if any(any(ord(c) > 127 for c in p) for p in people):
                phenomena += ["unicode_identity"]
            row = {"id": f"{pair_id}-{'phi' if phi else 'control'}", "text": note.text, "spans": note.spans,
                "split": "train", "annotation_policy": POLICY, "full_identity": people[0], "full_identities": people,
                "identity_group": identity_digest(people[0]), "identity_groups": [identity_digest(p) for p in people],
                "pair_id": pair_id, "stable_pair_id": pair_id, "template_family": f"author6_{family[0]}",
                "phenomena": sorted(set(phenomena)), "cohort": "unusual_name" if unusual else ("long_document" if long else "realistic"),
                "length_stratum": "multiwindow" if long else "short_medium", "layout": layout, "case_mode": case_mode,
                "negative_only": not note.spans, "paired_variant": "phi_bearing" if phi else "clinical_only",
                "synthetic": True, "draft": True, "role": "training", "artificial_stress": long,
                "authoring_provenance": "independent_author6; literal pieces with spans by construction; no predictions used",
                "identity_aliases": [
                    {"full_identity": people[0], "identity_group": identity_digest(people[0]), "role": "patient", "aliases": [alias]},
                    {"full_identity": people[1], "identity_group": identity_digest(people[1]), "role": "clinician", "aliases": []},
                    {"full_identity": people[2], "identity_group": identity_digest(people[2]), "role": "support_person", "aliases": []}],
                "pair_identity_inheritance": not phi,
                "tokenization": {"tokenizer_sha256": self.tokenizer_sha, "content_tokens": len(enc.ids),
                    "runtime_max_length": self.config["max_length"], "special_tokens": self.special_count,
                    "content_window": self.content_window, "window_count": len(windows),
                    "windows": [{"start": a, "stop": b, "owned_left": c, "owned_right": d} for a,b,c,d in windows],
                    "name_placements": placements_meta}}
            assert not long or len(windows) >= 3
            out.append(row)
        return out


def validate(rows):
    ids, texts = set(), set()
    for row in rows:
        assert row["id"] not in ids and row["text"] not in texts
        ids.add(row["id"]); texts.add(row["text"])
        groups = [identity_digest(v) for v in row["full_identities"]]
        assert groups == row["identity_groups"] and len(set(groups)) == len(groups)
        assert row["identity_group"] == groups[0] and row["full_identity"] == row["full_identities"][0]
        assert all(int(g[:8],16) % 10 < 8 for g in groups)
        previous = 0
        for span in row["spans"]:
            a, b = span["start"], span["end"]
            assert 0 <= a < b <= len(row["text"]) and a >= previous
            assert row["text"][a:b] == span["text"] == span["text"].strip()
            assert span["label"] in LABELS
            if span["label"] == "NAME":
                assert span["identity_group"] in groups
                aliases = [p["full_identity"] for p in row["identity_aliases"] if p["identity_group"] == span["identity_group"]]
                aliases += [a for p in row["identity_aliases"] if p["identity_group"] == span["identity_group"] for a in p["aliases"]]
                assert unicodedata.normalize("NFC", span["text"]).casefold() in [unicodedata.normalize("NFC",a).casefold() for a in aliases]
            previous = b
        assert row["negative_only"] == (not row["spans"])
    assert len(rows) == 1000


def distribution(values):
    values = sorted(values)
    return {"min": values[0], "p10": values[int((len(values)-1)*.1)], "median": statistics.median(values),
            "p90": values[int((len(values)-1)*.9)], "max": values[-1], "mean": round(statistics.mean(values),2)}


def report_for(rows, author, corpus_sha, source_sha):
    strata = defaultdict(list)
    for r in rows:
        strata[f"{r['length_stratum']}/{r['layout']}/{r['paired_variant']}"] .append(r)
    words = {}
    for word in STATUS_WORDS:
        occurrences = Counter()
        docs = defaultdict(set)
        for r in rows:
            for m in re.finditer(r"(?<!\w)" + word + r"(?!\w)", r["text"], re.I):
                overlapping = [s for s in r["spans"] if s["start"] < m.end() and s["end"] > m.start()]
                label = "name" if any(s["label"] == "NAME" for s in overlapping) else ("other_phi" if overlapping else "ordinary")
                occurrences[label] += 1
                docs[label].add(r["id"])
        words[word] = {"occurrences": dict(occurrences), "documents": {k:len(v) for k,v in docs.items()}}
    name_positions = [p for r in rows for p in r["tokenization"]["name_placements"]]
    boundary_dist = [p["distance_to_boundary_tokens"] for p in name_positions if p["distance_to_boundary_tokens"] is not None]
    return {
        "schema_version": 1, "status": "draft_pending_independent_qa_normalization_and_overlap_audits", "split": "train",
        "annotation_policy": POLICY, "training_run_unchanged": True, "not_selected_for_training": True,
        "corpus": str(CORPUS.relative_to(ROOT)), "sha256": corpus_sha,
        "generator": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": source_sha, "seed": SEED},
        "counts": {"documents": len(rows), "pairs": len({r['pair_id'] for r in rows}),
            "phi_bearing": sum(bool(r["spans"]) for r in rows), "phi_free": sum(not r["spans"] for r in rows),
            "unique_full_person_identities": len(author.used), "identity_rejection_attempts": author.rejected,
            "length_strata": dict(Counter(r["length_stratum"] for r in rows)),
            "actual_window_counts": dict(sorted(Counter(r["tokenization"]["window_count"] for r in rows).items())),
            "cohorts": dict(Counter(r["cohort"] for r in rows)), "families": dict(Counter(r["template_family"] for r in rows)),
            "layouts": dict(Counter(r["layout"] for r in rows)), "case_modes": dict(Counter(r["case_mode"] for r in rows)),
            "label_support": {label:sum(s["label"] == label for r in rows for s in r["spans"]) for label in LABELS}},
        "strata": {k:{"documents":len(v), "content_tokens":distribution([r["tokenization"]["content_tokens"] for r in v]),
                       "windows":distribution([r["tokenization"]["window_count"] for r in v])} for k,v in sorted(strata.items())},
        "tokenizer": {"path": str(TOKENIZER.relative_to(ROOT)), "sha256": digest(TOKENIZER.read_bytes()),
            "config_path": str(CONFIG.relative_to(ROOT)), "config_sha256":digest(CONFIG.read_bytes()),
            "runtime_source":str(RUNTIME.relative_to(ROOT)), "runtime_source_sha256":digest(RUNTIME.read_bytes()),
            "max_length":author.config["max_length"], "special_token_count":author.special_count,
            "content_window":author.content_window, "effective_overlap_margin":min(128,author.content_window//4),
            "stride":author.content_window - 2*min(128,author.content_window//4),
            "window_counter": "window_jobs AST extracted from runtime source; tokenizer encode with add_special_tokens=False",
            "config_overlap_note": "config declares 128; executable runtime computes min(128,510//4)=127"},
        "name_position_coverage": {"relative_document_quarters":dict(Counter(min(3,int(p["relative_char_position"]*4)) for p in name_positions)),
            "nearest_boundary_distance_tokens":distribution(boundary_dist),
            "within_8_tokens_of_boundary":sum(v<=8 for v in boundary_dist),
            "crosses_ownership_boundary":sum(p["crosses_ownership_boundary"] for p in name_positions),
            "placement_policy":"opening, middle, closing blocks rotate; exact token positions and ownership boundaries recorded per row"},
        "status_word_positive_and_negative_coverage": words,
        "provenance": {
            "author":"independent_author6", "method":"New context prose and multiword observation clauses composed in this source; deterministic variation and literal-span construction.",
            "allowed_reference_reads":["docs/annotation-policy-v2.md","src/clinical_deid/annotation_policy.py",
                "scripts/takeover/prepare_training_mixture.py","scripts/takeover/normalize_conventions.py",
                "scripts/audit_corpus_quality.py","src/clinical_deid/hf_encoder.py",str(TOKENIZER.relative_to(ROOT)),str(CONFIG.relative_to(ROOT))],
            "dataset_or_prediction_reads":[], "model_weights_loaded":False,"inference_performed":False,"network_used":False,
            "reserved_identifier_schemes":{"PHONE":"+1 (202) 555-0100 through 555-0199; invented extensions",
                "EMAIL":"notes.example reserved domain", "URL":"rounds.test reserved domain", "IP":"198.51.100.0/24 TEST-NET-2",
                "MRN":"SYN-A6- plus synthetic pair number", "ACCOUNT":"TRAIN6- plus synthetic pair number", "ID":"A6-TRANSPORT- plus synthetic pair number",
                "ADDRESS":"invented Imaginary Fern Walk / Velmara locality / 80000 placeholder ZIP", "SSN":"not used"},
            "identity_partition":"NFC then casefold SHA256; rejection sampled until int(first8hex,16)%10<8; all people included; controls inherit pair identity metadata",
            "gold_source":"literal pieces and spans by construction, never model predictions"},
        "author_checks":{"structural_checks_passed":True,"unique_document_texts":len({r['text'] for r in rows}),
            "all_primary_and_secondary_identities_train":True, "long_notes_at_least_three_runtime_windows":True,
            "independent_qa":"pending", "normalization":"pending independent check", "overlap_audits":"not performed by author"},
        "manual_review":{"status":"pending_author_sample_review", "sample_ids":[]},
        "limitations":[
            "Long notes deliberately repeat varied independently authored observation clauses; they are artificial stress data, not naturally sampled longitudinal records.",
            "Pairs share much of their prose and inherit identities; paired counts and repeated clauses do not represent independent clinical evidence.",
            "Unusual status-word aliases are an explicitly separate cohort, with ordinary occurrences retained elsewhere in the same notes; frequency is not representative.",
            "No performance claim, sealed overlap result, external-corpus independence claim, qualification claim, or clinical validity claim is made.",
            "This author checked only new output structure and tokenizer windows; independent semantic QA, normalization and sanctioned overlap audits remain required.",
            "AGE and SSN positives are absent; ordinary ages occur throughout. The bank targets contextual NAME behavior rather than complete PHI label coverage.",
            "Token boundary placement is measured and disclosed, not optimized using inference; no frozen-run inputs, configuration, weights, or selection are changed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Optional new directory for reproducibility replay; defaults to the requested paths")
    args = parser.parse_args()
    corpus = CORPUS if args.output_dir is None else args.output_dir / CORPUS.name
    manifest = Path(str(corpus) + ".manifest.json")
    report_path = REPORT if args.output_dir is None else args.output_dir / "author_report.json"
    for path in (corpus, manifest, report_path):
        if path.exists() or path.is_symlink():
            raise SystemExit(f"refusing to overwrite {path}")
    author = Author()
    rows = [row for fidx in range(len(FAMILIES)) for local in range(20) for row in author.pair(fidx,local,fidx*20+local)]
    validate(rows)
    payload = "".join(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n" for r in rows).encode("utf-8")
    sha = digest(payload)
    report = report_for(rows, author, sha, digest(Path(__file__).read_bytes()))
    counts = report["counts"]
    sidecar = {"schema_version":1,"corpus":str(CORPUS.relative_to(ROOT)),"sha256":sha,"split":"train","role":"training",
        "annotation_policy":POLICY,"draft":True,"status":report["status"],"synthetic":True,"rows":len(rows),"documents":len(rows),
        "label_support":counts["label_support"],"entity_support":counts["label_support"],"label_counts":counts["label_support"],
        "cohort_counts":counts["cohorts"],"phi_free_rows":counts["phi_free"],"entity_count":sum(counts["label_support"].values()),
        "generator":report["generator"],"provenance":report["provenance"],"limitations":report["limitations"],
        "author_report":str(REPORT.relative_to(ROOT)),"not_selected_for_training":True,"current_frozen_run_unchanged":True}
    for path, data in ((corpus,payload),(manifest,(json.dumps(sidecar,ensure_ascii=False,indent=2,sort_keys=True)+"\n").encode()),
                       (report_path,(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n").encode())):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
    print(json.dumps({"corpus":str(corpus),"manifest":str(manifest),"report":str(report_path),"sha256":sha,"counts":counts},ensure_ascii=False))


if __name__ == "__main__":
    main()
