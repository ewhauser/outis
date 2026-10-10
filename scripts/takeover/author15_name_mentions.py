#!/usr/bin/env python3
"""Author the tk15 training bank: first-name and short name mentions.

Deterministic. See docs/name-mentions-bank-brief.md and
docs/annotation-policy-v2.md. Five families: staff and family members named by
first name only, aliases after a full name, `LAST, FIRST M` forms, and
name-like ordinary words (negatives only). Every name is invented from broad
uncommon pools; every row is checked against the hash-only blind exclusion
list and re-drawn on a match. Only counts are recorded, never the strings.

Usage:
  PYTHONPATH=src python scripts/takeover/author15_name_mentions.py \
      --out data/tk15_train_name_mentions.jsonl \
      --report-dir reports/takeover/author15_name_mentions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from outis.data.annotation_policy import (  # noqa: E402
    POLICY_VERSION,
    annotation_errors,
    canonical_identity_group,
    identity_split,
)
from outis.data.generators.places import (  # noqa: E402
    excluded_sequences,
    excluded_value,
)
from outis.types import ENTITY_TYPES  # noqa: E402

SEED = 20261006
CAP = 8  # rows per name value; 0.5% of 1,800 rows is 9
SUITE = "tk15_name_mentions"
CASES = ("upper", "lower", "normal")
CASE_SHARE = {"upper": 0.4, "lower": 0.2, "normal": 0.4}

# family -> (pair units, positive singles, solo negatives)
PLAN = {
    "staff_first_name": (120, 260, 0),
    "family_first_name": (120, 260, 0),
    "alias_after_full_name": (80, 140, 0),
    "last_first_forms": (40, 180, 40),
    "name_like_words": (0, 0, 200),
}

MARK = re.compile(r"\[\[([A-Z]+)\|([^\[\]]+)\]\]")


def span(label, value):
    return f"[[{label}|{value}]]"


def N(value):
    return span("NAME", value)


# --------------------------------------------------------------------------- name pools

GIVEN_CURATED = """Adaeze Aloysia Anouk Aurelio Bahar Baptiste Berit Bronislav Calliope Casimir Cosmin Dagny Desislava
Dilnoza Eamon Eira Emeka Eskil Fenna Folasade Fridtjof Gerasim Gwenllian Halvard Hesper Hiroko Ibrahima Ignatia Ioana
Jaromir Jovita Kalani Kazimir Kiona Leocadia Lorcan Lubomir Maelis Malachy Mirela Mstislav Naveen Nuriel Oluwaseun
Orsolya Pernille Petronela Quillon Radmila Reidar Sidonie Soledad Stellan Tamsin Teodor Thandeka Ulrika Vasilis Vesna
Wiktor Wilhelmina Xiomara Yevgenia Zdenka Zsofia Anselm Beatrix Cormac Dorothea Eulalia Fiachra Gisela Hartwig
Ingeborg Jocasta Kasimira Ludovica Marguerite Nerys Octavia Perpetua Rosalind Sunniva Thaddeus Ursina Valdemar
Winifred Ysolde Zephyrine Abiola Bashir Chidi Dalia Esi Farrukh Gulnara Hamid Ife Jamshid Kofi Lwazi Mbali Nkechi
Olamide Piotr Quynh Rashida Sibusiso Tendai Umaima Viktoriya Wanjiru Xolani Yaroslav Bao Chaya Dmitar Elif Farida
Haruka Isidro Katalin Lucjan Nasrin Ottilie Alaric Brigid Cezary Dessa Evander Florentin Genoveva Hrafn Idunn Jorunn
Kleopatra Leif Marit Nikodem Oriel Pelagia Ragnhild Saga Torquil Una Vigdis Wladyslaw Yrsa Zenobia Amadou Bisi Cyprian
Dumisani Efua Gbenga Hadiza Ikenna Jabu Kagiso Lindiwe Makena Ngozi Obinna Pili Rutendo Sizwe Temitope Uchenna Zola
Anahit Bakhyt Chinara Davit Eldar Fatima Gor Hayk Isolde Jiri Karlis Laima Marek Neda Olgierd Pavle Ruzica Stanko
Tihana Vilmos Alinta Bidu Corazon Dalisay Elpidio Gumercindo Imelda Jacinto Lourdes Marisela Nilda Osvaldo Placida
Rufina Sixto Telmo Ubaldo Yolotl Anselma Bertil Carola Dietmar Elfriede Gudrun Heiko Irmtraud Joachim Kunigunde Lothar
Mechthild Norbert Ottokar Reinhild Siegfried Traudl Ulf Wolfram""".split()

GIVEN_SYL_A = ["Al", "Be", "Cor", "Da", "El", "Fen", "Gal", "Hal", "Il", "Jor", "Kes", "Lu", "Mar", "Nes", "Or", "Pel",
               "Quin", "Ros", "Sel", "Tor", "Ul", "Vel", "Wen", "Yar", "Zel", "Bryn", "Cass", "Dov", "Eri", "Fyn", "Gwy",
               "Hesk", "Isk", "Jun", "Kel", "Lor", "Mel", "Nor", "Osk", "Pry", "Rhy", "Sav", "Tam", "Vor", "Wil"]
GIVEN_SYL_B = ["", "", "a", "i", "en", "or", "an", "el", "ar", "in"]
GIVEN_SYL_C = ["a", "en", "is", "wyn", "et", "ine", "ik", "o", "ys", "ard", "el", "ora", "ina", "ek", "ud", "ila", "ow",
                "ane", "us", "ith"]

SUR_CURATED = """Adeyemi Bakkevig Chukwuma Dzhaparov Eliasson Fairweather Gudmundsson Haldane Iwasaki Janowicz Kvasnicka
Lindqvist Mbeki-Ross Nwosu Okonjo Pellegrini Quarrie Radcliffe Szymanski Teixeira Ulvestad Vandermeer Wojciechowski
Xanthos Yamazaki Zielinski Abernethy Bergstrom Cavanagh Drozdov Eberhardt Fitzgibbon Grzeskowiak Hakimi Ilyushin
Jovanovic Karagiannis Lemaitre Moravec Nakagawa Ostrowski Papadakis Quigley Rasmussen Soderberg Tomaszewski
Underhill Vasconcelos Wasilewski Yilmaz Zabokrzycki Ayodele Bondarenko Czarnecki Dragomir Engstrom Fedorov Gallagher
Hollingsworth Ivanisevic Jankowski Kaczmarek Larkspur Mazurek Novakovic Oyelaran Petrovic Quintanilla Rosenthal
Stefanidis Trzcinski Urquhart Vukovic Winterbourne Yakubu Zawadzki Achterberg Brzezinski Cieslak Dunmore Eskelinen
Forsythe Gorecki Havlicek Ingerslev Jablonski Kowalska Lukacs Mikkelsen Nygaard Orlowski Pasztor Rutkowski Sandoval
Tkachenko Uzun Vlachos Wisniewski Yoshimura Zhelyazkov Ashcombe Blakiston Culpepper Dunsany Ellacott Fotheringay
Gaskarth Heathcote Ingleby Jessamy Kilbride Lanyon Moorcroft Nethercott Ormerod Pendleton Quennell Rosslyn Sallow
Thwaite Umpleby Verrall Whitlock Yarnold""".split()

SUR_PRE = ["Brann", "Wick", "Tarl", "Holm", "Vess", "Ashf", "Corr", "Dunl", "Penr", "Ostr", "Kell", "Marl", "Quern",
           "Thorn", "Rav", "Sark", "Lund", "Gren", "Fisk", "Harl", "Jarr", "Morv", "Nesk", "Orm", "Prath", "Rusk",
           "Skell", "Tev", "Ulm", "Varn", "Weth", "Yeld", "Zorn", "Bask", "Crav", "Dren", "Elm", "Fenn", "Gask"]
SUR_END = ["ock", "wick", "ford", "mere", "ston", "by", "den", "well", "ham", "ridge", "worth", "gate", "hurst", "ard",
           "ow", "ley", "ner", "som", "vik", "dal", "holt", "man", "ing", "eth"]

NICKS = ["Pip", "Bix", "Teddo", "Lumi", "Nonie", "Kit", "Juno", "Bram", "Tilda", "Dex", "Fitz", "Zeke", "Lottie",
         "Bunty", "Ossie", "Tibby", "Cass", "Wyn", "Mimsy", "Dolly", "Nel", "Quill", "Rafe", "Sunny", "Tavi", "Ivo",
         "Posy", "Gus", "Lark", "Moll", "Fen", "Hobie", "Jory", "Kiki", "Lenny", "Minka", "Obie", "Pico", "Remy"]


def _build_pools():
    rng = random.Random(1515)
    given = list(GIVEN_CURATED)
    seen = {g.casefold() for g in given}
    while len(given) < 1100:
        g = rng.choice(GIVEN_SYL_A) + rng.choice(GIVEN_SYL_B) + rng.choice(GIVEN_SYL_C)
        if len(g) < 5 or g.casefold() in seen:
            continue
        seen.add(g.casefold())
        given.append(g)
    for _ in range(70):  # hyphenated given names
        a, b = rng.choice(GIVEN_CURATED), rng.choice(GIVEN_CURATED)
        if a != b and len(a) <= 7 and len(b) <= 7:
            given.append(f"{a}-{b}")
    sur = list(SUR_CURATED)
    seen = {s.casefold() for s in sur}
    while len(sur) < 900:
        s = rng.choice(SUR_PRE) + rng.choice(SUR_END)
        r = rng.random()
        if r < 0.06:
            s = "O'" + s
        elif r < 0.10:
            s = "de " + s
        elif r < 0.13:
            s = "van " + s
        elif r < 0.17:
            s = s + "-" + rng.choice(SUR_CURATED)
        if s.casefold() in seen:
            continue
        seen.add(s.casefold())
        sur.append(s)
    return given, sur


GIVEN, SURNAMES = _build_pools()
PLACE_PRE = ["Vossm", "Ombr", "Lark", "Tarn", "Wend", "Brack", "Fenn", "Ell", "Hollin", "Marl", "Sorr", "Quill",
             "Thistl", "Corr", "Aldr", "Beck", "Dunr", "Esk", "Garr", "Halv", "Jessl", "Kirr", "Lorn", "Mell"]
PLACE_END = ["ere", "ey", "ford", "wick", "mere", "hollow", "field", "crest", "brook", "vale", "haven", "ton", "side"]
FAC_TYPES = ["Hospice", "Home Health", "VNA", "Health Center", "Medical Center", "Hospital", "Home Care",
             "Palliative Care", "Adult Day Program", "Rehabilitation Center"]


def facility(rng):
    return rng.choice(PLACE_PRE) + rng.choice(PLACE_END) + " " + rng.choice(FAC_TYPES)


# --------------------------------------------------------------------------- templates (positive, matched negative)

STAFF_GROUPS = {
    "care": (
        ["aide", "HHA", "CNA", "home health aide", "sitter", "volunteer"],
        [
            ("{SR} {N} for bathing and linen change", "{SR} for bathing and linen change"),
            ("{SR} {N} assisted with morning care; skin intact", "the {SR} assisted with morning care; skin intact"),
            ("Bath done by {SR} {N}, tolerated well", "Bath done by the {SR}, tolerated well"),
            ("{SR} ({N}) reports pt ate half of breakfast", "{SR} reports pt ate half of breakfast"),
            ("Handoff from {N}, {SR}, at 0700: repositioned every 2 hours", "Handoff from the {SR} at 0700: repositioned every 2 hours"),
            ("{N} (the {SR}) will return Thursday for bathing", "the {SR} will return Thursday for bathing"),
            ("per {SR} {N}, bed rail was left down overnight", "per the {SR}, bed rail was left down overnight"),
        ],
    ),
    "therapy": (
        ["PT", "OT", "SLP", "respiratory therapist", "physical therapist", "speech therapist"],
        [
            ("{SR} {N} worked on transfers and gait with walker", "{SR} worked on transfers and gait with walker"),
            ("{SR} {N} eval completed; goals reviewed with pt", "{SR} eval completed; goals reviewed with pt"),
            ("Seen by {SR} {N} for balance training", "Seen by {SR} for balance training"),
            ("{N}, {SR}, recommends 3x/week for 4 weeks", "the {SR} recommends 3x/week for 4 weeks"),
            ("{SR}: {N} saw pt in the afternoon, fatigued after 10 minutes", "{SR}: saw pt in the afternoon, fatigued after 10 minutes"),
            ("pt cooperative with {N} (the {SR}) today", "pt cooperative with the {SR} today"),
        ],
    ),
    "nursing": (
        ["RN", "LPN", "hospice nurse", "night nurse", "agency nurse", "wound nurse", "float nurse", "visiting nurse"],
        [
            ("{SR} {N} gave report at shift change", "{SR} gave report at shift change"),
            ("Pain reassessed by {N}, {SR}; pt rates 3/10", "Pain reassessed by the {SR}; pt rates 3/10"),
            ("Dressing changed by {SR} {N}, wound bed clean", "Dressing changed by the {SR}, wound bed clean"),
            ("Called {SR} {N} re: new swelling in left ankle", "Called the {SR} re: new swelling in left ankle"),
            ("{N} ({SR}) will visit again Monday for med set-up", "{SR} will visit again Monday for med set-up"),
            ("Spoke with {N}, {SR}, about the PRN order", "Spoke with the {SR} about the PRN order"),
        ],
    ),
    "psychosocial": (
        ["chaplain", "social worker", "MSW", "bereavement counselor", "music therapist", "case manager", "volunteer"],
        [
            ("{SR} {N} visited; pt appreciated the company", "the {SR} visited; pt appreciated the company"),
            ("{SR} {N} will call family regarding the care meeting", "the {SR} will call family regarding the care meeting"),
            ("{N} ({SR}) completed the psychosocial assessment", "{SR} completed the psychosocial assessment"),
            ("Referral to {SR} {N} placed", "Referral to {SR} placed"),
            ("{SR}: {N} sat with pt during the afternoon; calm", "{SR}: sat with pt during the afternoon; calm"),
            ("{N}, the {SR}, offered a visit on Saturday", "the {SR} offered a visit on Saturday"),
        ],
    ),
    "other": (
        ["pharmacist", "dietitian"],
        [
            ("{SR} {N} reviewed the medication list with family", "the {SR} reviewed the medication list with family"),
            ("Diet consult: {SR} {N} advised thickened liquids", "Diet consult: the {SR} advised thickened liquids"),
            ("Dr. {N} made the visit and adjusted comfort orders", "the hospice physician made the visit and adjusted comfort orders"),
            ("Dr. {N} (hospice physician) will call after rounds", "the hospice physician will call after rounds"),
        ],
    ),
}
STAFF_FOLLOW = [
    ("{N} will return tomorrow at the same time", "the {SR} will return tomorrow at the same time"),
    ("Thank you to {N} for the extra visit", "Thank you to the {SR} for the extra visit"),
    ("pt asked for {N} again", "pt asked for the {SR} again"),
    ("{N} to follow up by phone", "the {SR} to follow up by phone"),
]
RELATIONS = ["daughter", "son", "wife", "husband", "spouse", "sister", "brother", "niece", "nephew", "grandson",
             "granddaughter", "neighbor", "friend", "caregiver", "daughter-in-law", "son-in-law", "partner", "cousin",
             "roommate", "church friend"]
FAMILY_TEMPLATES = [
    ("{REL} {N} at bedside", "{REL} at bedside"),
    ("{N} ({REL}) helps with meals", "{REL} helps with meals"),
    ("social work: {N} is exhausted", "social work: family is exhausted"),
    ("per {N}, pt slept poorly last night", "per {REL}, pt slept poorly last night"),
    ("{REL} {N} will stay overnight", "{REL} will stay overnight"),
    ("Spoke with {N}, pt's {REL}, who requests a call after 5 pm", "Spoke with pt's {REL}, who requests a call after 5 pm"),
    ("{N}, pt's {REL}, is the primary caregiver and lives with pt", "pt's {REL} is the primary caregiver and lives with pt"),
    ("{N}'s phone is the best number to reach family", "the {REL}'s phone is the best number to reach family"),
    ("{REL} {N} checks in each morning", "{REL} checks in each morning"),
    ("family meeting attended by {N} and pt", "family meeting attended by {REL} and pt"),
    ("{N} brought pt's glasses from home", "{REL} brought pt's glasses from home"),
    ("{N} asked about hospice eligibility; questions answered", "{REL} asked about hospice eligibility; questions answered"),
    ("pt wants {N} to be called first if status changes", "pt wants the {REL} to be called first if status changes"),
    ("{REL} {N} declined respite for now", "{REL} declined respite for now"),
    ("{HON} {N} ({REL}) will be present for the visit", "the {REL} will be present for the visit"),
    ("{REL} {N} and {REL2} {N2} at bedside", "{REL} and {REL2} at bedside"),
    ("{N} and {N2} share the overnight shifts", "{REL} and {REL2} share the overnight shifts"),
    ("worried about {N}; says {N}'s sleep is poor", "worried about the {REL}; says the {REL}'s sleep is poor"),
]
FAMILY_FOLLOW = [
    ("{N} will call back after 6", "the {REL} will call back after 6"),
    ("{N} asked to be updated daily", "the {REL} asked to be updated daily"),
    ("pt calmer when {N} is present", "pt calmer when the {REL} is present"),
    ("left message for {N}", "left message for the {REL}"),
]

ALIAS_ROLES = ["patient", "daughter", "son", "aide", "chaplain", "volunteer", "neighbor", "nurse", "wife", "husband",
               "caregiver", "sister", "brother"]
ALIAS_EST = [
    ("{FN} is known to the team as {AL}", "the {R} is known to the team as {W}"),
    ("Preferred name: {AL}; legal name {FN}", "Preferred name: none recorded; legal name on file"),
    ("{FN} goes by {AL}, so please use that at visits", "the {R} is {W}, so please plan visits for mornings"),
    ('{FN} ("{AL}") is the {R}', "the {R} is {W}, per family"),
    ("{AL} ({FN}, {R}) arrived for the visit", "the {R} arrived for the visit"),
    ("{R}: {FN}, called {AL} by family", "{R}: no preferred name given"),
    ("family calls the {R} {AL}; full name {FN}", "family describes the {R} as {W}"),
    ("{HON} {FN} prefers {AL}", "the {R} prefers a quiet room"),
    ("{FN}, who answers to {AL}, was seen today", "the {R}, who is {W}, was seen today"),
    ("a.k.a. {AL}: {FN} is the {R}", "the {R} is {W}, per family"),
]
ALIAS_LATER = [
    ("{AL} refused lunch today, accepted fluids", "the {R} refused lunch today, accepted fluids"),
    ("{AL} was resting when seen; no distress", "the {R} was resting when seen; no distress"),
    ("will call {AL} tomorrow re: schedule", "will call the {R} tomorrow re: schedule"),
    ("{AL} asked that the window stay open", "the {R} asked that the window stay open"),
    ("{AL} left at 1600", "the {R} left at 1600"),
    ("per {AL}, appetite is improving", "per the {R}, appetite is improving"),
]
ORDINARY_NICK = ["a night owl", "a morning person", "a creature of habit", "a tough cookie", "a good sport",
                 "an early riser", "a real homebody", "a good listener"]

LASTFIRST = [
    ("PATIENT NAME: {NM}", "PATIENT NAME: see registration"),
    ("Patient: {NM}", "Patient: see registration"),
    ("POA: {NM}", "POA: none on file"),
    ("POA / HCP: {NM} ({REL})", "POA / HCP: none on file"),
    ("Healthcare agent: {NM}", "Healthcare agent: not designated"),
    ("Emergency contact: {NM} - {REL} - {PH}", "Emergency contact: declined to list"),
    ("EMERGENCY CONTACT: {NM} ({REL}) {PH}", "EMERGENCY CONTACT: none provided"),
    ("NOK: {NM}", "NOK: unknown"),
    ("Authorized representative: {NM}", "Authorized representative: none"),
    ("Guarantor: {NM}", "Guarantor: self"),
    ("Attending of record: {NM} MD", "Attending of record: pending"),
    ("Primary RN: {NM} RN", "Primary RN: not yet assigned"),
    ("Consent signed by {NM} ({REL})", "Consent signed by the {REL}"),
    ("Per chart, POA is {NM}; copy of papers requested", "Per chart, no POA is designated; copy of papers requested"),
    ("Contact on file: {NM}; best number {PH}", "Contact on file: none; best number unavailable"),
    ("Spoke with {NM} ({REL}) about the visit schedule", "Spoke with the {REL} about the visit schedule"),
    ("Release of information to {NM}", "Release of information on hold"),
]
LASTFIRST_SOLO_NEG = [
    "see sections A, B and C of the consent",
    "Hepatitis A, B screening negative",
    "Medicare Part A, B active",
    "plan options reviewed: A, B or C",
    "vitamin D, K and B12 continued",
    "reviewed forms A, B before signing",
    "items A, B pending; C complete",
    "Plan A, B discussed with team",
]

NAME_LIKE = {
    "faith": ["Faith is important to pt and family; chaplain visit welcomed", "pt draws strength from {W} and attends services weekly", "{W} community brings meals on Sundays"],
    "hope": ["Hope for improvement was expressed by family", "family expressed {W} that pt could return home", "pt says there is little {W} of getting stronger; palliative goals discussed"],
    "grace": ["Grace period on the copay ends this month", "family said {W} before the meal", "walks with more {W} after therapy"],
    "will": ["Will reassess pain after the next dose", "living {W} on file, copy requested", "pt states {W} stay home as long as possible"],
    "joy": ["Joy in daily activities has returned per family", "pt takes {W} in the garden view", "pt finds {W} in music sessions"],
    "mark": ["Mark on left shin is a healed bruise", "old pressure {W} on sacrum, no open area", "dark {W} noted near the incision; skin otherwise intact"],
    "rose": ["bp {W} to 150 systolic after transfers, settled with rest", "temp {W} slightly overnight, now normal", "pt keeps a dried {W} on the table"],
    "bill": ["Bill for the equipment rental was sent to family", "family has questions about the {W} for the oxygen concentrator", "hospital {W} reviewed by the billing office"],
    "pat": ["Pat skin dry after bathing to prevent breakdown", "advised to {W} the skin dry, not rub", "{W} dry between toes after washing"],
    "art": ["Art therapy session offered Tuesday", "pt enjoyed the {W} cart on the unit", "{W} supplies requested by volunteer"],
    "max": ["Max assist of two for transfers", "pt is at {W} dose for the stool softener", "reached {W} reach with the left arm"],
    "dawn": ["Dawn and dusk are hardest for pt; lights kept on", "pt restless from {W} until mid-morning", "up at {W} for pain pills"],
    "summer": ["Summer heat makes pt short of breath", "pt prefers the porch in {W}", "fan placed for the {W} months"],
    "autumn": ["Autumn visit schedule shared with family", "pt's cough is worse each {W}", "flu shot offered in {W}"],
    "frank": ["Frank blood noted in the stool, MD notified", "no {W} bleeding at the site", "{W} swelling of the right leg noted"],
    "rich": ["Rich foods upset stomach per family", "pt avoids {W} gravy", "{W} desserts declined"],
    "ray": ["Ray of sunlight on the bed seemed to calm pt", "pt turns toward each {W} of light from the window", "a {W} of hope noted in mood today"],
    "penny": ["Penny-sized sore on the left heel, dressed", "pt keeps a {W} in the bedside drawer", "sore about the size of a {W}"],
    "amber": ["Amber urine noted, hydration encouraged", "{W} urine on morning check; fluids encouraged", "{W} light left on in the hallway"],
    "violet": ["Violet discoloration over the left heel", "{W} bruising on right forearm, no swelling", "{W} blanket requested from home"],
    "ivy": ["Ivy covers the porch rail; pt likes the view", "poison {W} rash on both forearms", "{W} cuttings brought in by family"],
    "olive": ["Olive oil used for dry skin per family", "pt prefers {W} oil on salads", "{W} drab chair moved near the window"],
    "iris": ["Iris reactive to light, pupils equal", "{W} exam normal per last visit", "pt's {W} appears clear on exam"],
    "crystal": ["Crystal arthropathy noted in old records", "history of {W} arthropathy in right knee", "{W} clear fluid in the drain"],
    "willow": ["Willow bark tea declined; reviewed supplement list", "pt asked about {W} bark for pain", "{W} branch outside the window noted by pt"],
    "jasmine": ["Jasmine tea requested at bedtime", "pt likes {W} rice with meals", "{W} scent calms pt per family"],
    "ginger": ["Ginger tea helps nausea per pt", "{W} chews tried for nausea", "pt takes {W} ale for upset stomach"],
    "hazel": ["Hazel eyes noted on ID band check", "pt's eyes are {W}, sclera clear", "{W} nut allergy reviewed"],
    "chip": ["Chip of paint on bedrail removed", "ice {W} given for dry mouth", "pt asked for a {W} of ice"],
    "sunny": ["Sunny room preferred; blinds kept open", "pt in a {W} mood after the visit", "{W} window seat is pt's favorite spot"],
    "pearl": ["Pearl necklace kept at bedside; do not remove", "pt keeps a {W} earring in the dish", "{W} gray skin tone noted at the lips"],
    "chase": ["Chase the refill with the pharmacy before Friday", "need to {W} the lab result by phone", "{W} down the missing order"],
    "patience": ["Patience needed with slow speech; allow extra time", "family asking for {W} as pt adjusts", "pt shows great {W} during dressing changes"],
    "charity": ["Charity care application started for the bill", "{W} care form signed by family", "pt receives {W} meals from the church"],
    "honor": ["Honor pt's wish for no feeding tube", "family wants to {W} pt's wishes", "{W} guard attended the service"],
    "robin": ["Robin at the feeder outside the window cheered pt", "pt watches a {W} on the sill", "{W} egg blue blanket requested"],
    "sky": ["Sky was clear and pt asked to sit outside", "pt watches the {W} from the window", "{W} blue pajamas requested by family"],
    "river": ["River view from the window is pt's favorite", "pt asked to see the {W} again", "walk along the {W} path discussed as a goal"],
    "lily": ["Lily on the table was moved away due to scent", "pt keeps a {W} by the bed", "{W} pad removed after bathing"],
    "sage": ["Sage tea requested with dinner", "pt wants {W} advice from the chaplain", "dried {W} bundle at bedside, removed per policy"],
    "heath": ["Heath views were pt's favorite walk, per family", "pt remembers walks on the {W}", "{W} photographs on the wall comfort pt"],
    "summit": ["Summit of the stairs is where pt tires", "pt stops at the {W} of the ramp to rest", "{W} meeting of the team scheduled"],
    "bishop": ["Bishop pillow wedge ordered for positioning", "pt asked to see the {W} from the parish", "{W} visit arranged by family"],
    "hunter": ["Hunter green robe requested from home", "pt was a {W} in younger years and talks about it", "{W} orange vest on the closet hook"],
}


# --------------------------------------------------------------------------- layouts

FILLER = [
    "Pain controlled on current regimen", "Skin warm and dry; no new redness", "Appetite fair, took about half of lunch",
    "Oxygen at 2 L per nasal cannula; comfortable", "Bowel movement yesterday", "Bed alarm in place",
    "Rested most of the afternoon", "Mouth care done every 4 hours", "Fluids encouraged", "Lungs with scattered crackles at the bases",
    "Edema in both feet unchanged", "Turned and repositioned as scheduled", "Respirations even and unlabored",
    "Sleeping more during the day", "No falls this week", "Medication box refilled and checked", "Lights kept low in the evening",
    "Constipation managed with scheduled senna", "Vital signs stable", "Pt oriented to person and place",
    "Walker within reach", "Reviewed pain scale with pt", "Weight unchanged since last visit", "Tolerated soft diet",
    "Cough present at night, improved with elevation", "Call light within reach", "Hydration adequate",
    "Dressing dry and intact", "Heel protectors in place", "Reviewed signs of worsening breathlessness",
    "Supplies restocked at bedside", "Appointment reminder reviewed", "Mood calm this morning", "Slept through most of the night",
    "Eating small frequent meals", "Mild confusion in the evening", "Comfort measures continued",
    "Home oxygen tubing checked", "Safety rails up at night", "Reviewed after-hours phone number with family",
]

# name: (title, header field formats, body style, footer)
LAYOUTS = {
    "hospice_visit_note": ("HOSPICE VISIT NOTE", ["{F}", "Visit date: {D}", "MRN: {M}", "Age: {A}"], "dash", ""),
    "home_health_visit": ("Home health visit", ["Agency: {F}", "Date of service: {D}", "MRN {M}", "Narrative:"], "para", ""),
    "idt_plan": ("INTERDISCIPLINARY PLAN OF CARE", ["Agency: {F}", "Review date: {D}", "Discipline updates"], "numbered", ""),
    "nursing_flowsheet": ("NURSING FLOWSHEET  {D}", ["Unit MRN {M}", "Time | Entry"], "time", ""),
    "social_work_note": ("Social work note", ["Date: {D}", "MRN: {M}", "Assessment:"], "para", ""),
    "facesheet": ("FACESHEET", ["Facility: {F}", "Admit date: {D}", "MRN {M}", "Phone {P}", "Age: {A}", "Comments"], "lines", ""),
    "contact_list": ("CONTACTS (updated {D})", ["Office line {P}", "Agency: {F}"], "dash", ""),
    "fax_cover": ("FAX COVER SHEET", ["From: {F}", "Fax: {P}", "Date: {D}", "Pages: 2", "Message:"], "lines", "Confidential: if received in error, call the sender."),
    "discharge_summary": ("DISCHARGE SUMMARY", ["{F}", "Discharge date: {D}", "MRN: {M}", "Course and plan:"], "para", ""),
    "aide_care_plan": ("AIDE CARE PLAN", ["Agency: {F}", "Effective {D}"], "checks", ""),
    "shift_handoff": ("SHIFT HANDOFF - {D}", ["{F}"], "star", ""),
    "phone_triage": ("TELEPHONE ENCOUNTER", ["Date: {D}", "Caller line: {P}", "MRN: {M}", "Summary:"], "para", ""),
    "care_conference": ("CARE CONFERENCE NOTES", ["Held {D}, {F}"], "numbered", ""),
    "admission_assessment": ("ADMISSION ASSESSMENT", ["Admitted {D}  Age: {A}", "{F}", "MRN {M}"], "lines", ""),
    "wound_care_note": ("Wound care visit - {D}", ["{F}", "MRN {M}"], "dash", ""),
    "bereavement_followup": ("Bereavement follow-up log", ["Service: {F}", "Contact date: {D}"], "para", ""),
    "therapy_daily_note": ("DAILY THERAPY NOTE", ["{D}", "{F}", "Age: {A}"], "soap", ""),
    "med_rec_call": ("Medication reconciliation call", ["Date {D}; MRN {M}"], "lines", ""),
    "respite_referral": ("RESPITE REFERRAL", ["Referring agency: {F}", "Date: {D}", "Callback: {P}", "Notes:"], "lines", ""),
    "home_safety_eval": ("HOME SAFETY EVALUATION", ["Visit {D}", "{F}", "Findings:"], "numbered", ""),
    "spiritual_care_log": ("Spiritual care log", ["Date {D} | {F}"], "gt", ""),
    "visit_log_table": ("VISIT LOG", ["DATE: {D}", "MRN: {M}"], "time", ""),
    "intake_checklist": ("INTAKE CHECKLIST", ["Referral received {D} by {F}", "Intake line {P}"], "checks", ""),
    "progress_soap": ("PROGRESS NOTE ({D})", ["Age: {A}", "MRN: {M}"], "soap", ""),
    "transfer_form": ("TRANSFER FORM", ["Sending: {F}", "Date: {D}", "Phone: {P}"], "lines", ""),
}
LAYOUT_NAMES = list(LAYOUTS)


def sent(x):
    if x.endswith(".") or re.search(r"\.\]\]$", x):
        return x
    return x + "."


def cap_first(x):
    return x[:1].upper() + x[1:] if x and x[0].isalpha() else x


def fmt_body(style, items):
    if style == "para":
        return " ".join(sent(x) for x in items)
    if style == "lines":
        return "\n".join(items)
    if style == "numbered":
        return "\n".join(f"{i}) {x}" for i, x in enumerate(items, 1))
    if style == "time":
        return "\n".join(f"{700 + 100 * i:04d} | {x}" for i, x in enumerate(items))
    if style == "soap":
        labels = ["S", "O", "A", "P"]
        return "\n".join(f"{labels[i]}: {x}" if i < 4 else f"   {x}" for i, x in enumerate(items))
    prefix = {"dash": "- ", "checks": "( ) ", "star": "* ", "gt": "> "}[style]
    return "\n".join(prefix + x for x in items)


def render(ctx, lines):
    title, fields, style, foot = LAYOUTS[ctx["layout"]]
    head = []
    for spec in [title] + fields:
        keys = re.findall(r"\{([A-Z])\}", spec)
        if any(not ctx.get(k) for k in keys):
            continue
        head.append(spec.format(**{k: ctx[k] for k in keys}) if keys else spec)
    items = [cap_first(x) for x in ctx["pre"] + lines + ctx["post"]]
    out = "\n".join(head) + "\n" + fmt_body(style, items)
    return out + ("\n" + foot if foot else "")


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
AREA = [206, 303, 414, 612, 719, 801]


def make_ctx(rng, layout):
    c = {"layout": layout}
    c["F"] = span("FACILITY", facility(rng)) if rng.random() < 0.7 else ""
    if rng.random() < 0.85:
        y, m, d = rng.randint(2023, 2026), rng.randint(1, 12), rng.randint(1, 28)
        shapes = [f"{m:02d}/{d:02d}/{y}", f"{y}-{m:02d}-{d:02d}", f"{m}/{d}/{y}", f"{d} {MONTHS[m - 1]} {y}"]
        c["D"] = span("DATE", rng.choice(shapes))
    else:
        c["D"] = ""
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    c["M"] = span("MRN", f"{rng.choice(letters)}{rng.choice(letters)}-{rng.randint(0, 9999999):07d}") if rng.random() < 0.55 else ""
    c["P"] = span("PHONE", phone(rng)) if rng.random() < 0.4 else ""
    c["A"] = span("AGE", str(rng.randint(90, 103))) if rng.random() < 0.6 else ""
    pool = rng.sample(FILLER, 5)
    k = rng.randint(1, 2)
    c["pre"], c["post"] = pool[:k], pool[k:k + rng.randint(0, 2)]
    return c


def phone(rng):
    a, n = rng.choice(AREA), rng.randint(0, 99)
    return rng.choice([f"({a}) 555-01{n:02d}", f"{a}-555-01{n:02d}"])


# --------------------------------------------------------------------------- draws

class State:
    def __init__(self):
        self.counts = Counter()
        self.texts = set()


class Attempt:
    """Names drawn in one attempt; counted only if the row is accepted."""

    def __init__(self, rng, st):
        self.rng, self.st = rng, st
        self.taken, self.names, self.persons = set(), [], []

    def usable(self, v):
        k = v.casefold()
        return k not in self.taken and self.st.counts[k] < CAP

    def take(self, v):
        self.taken.add(v.casefold())
        self.names.append(v.casefold())

    def pick(self, pool):
        for _ in range(5000):
            v = self.rng.choice(pool)
            if self.usable(v):
                self.take(v)
                return v
        raise RuntimeError("name pool exhausted")

    def person(self, count_surname=False):
        """A given name, plus a surname so the canonical identity is in the train partition."""
        g = self.pick(GIVEN)
        for _ in range(500):
            s = self.rng.choice(SURNAMES)
            if count_surname and not self.usable(s):
                continue
            if identity_split(canonical_identity_group(f"{g} {s}")) == "train":
                break
        else:
            raise RuntimeError("no train-partition identity")
        if count_surname:
            self.take(s)
        self.persons.append(f"{g} {s}")
        return g, s

    def hidden(self):
        for _ in range(500):
            h = f"{self.rng.choice(GIVEN)} {self.rng.choice(SURNAMES)}"
            if identity_split(canonical_identity_group(h)) == "train":
                return h
        raise RuntimeError("no train-partition identity")


def build_staff(att):
    rng = att.rng
    roles, tmpls = STAFF_GROUPS[rng.choice(list(STAFF_GROUPS))]
    role = rng.choice(roles)
    g, _ = att.person()
    pos_t, neg_t = rng.choice(tmpls)
    kw = {"SR": role, "N": N(g)}
    pos, neg = [pos_t.format(**kw)], [neg_t.format(**kw)]
    if rng.random() < 0.25:
        fp, fn = rng.choice(STAFF_FOLLOW)
        pos.append(fp.format(**kw))
        neg.append(fn.format(**kw))
    return pos, neg, ["staff_role", "first_name_only"] + (["honorific_outside"] if "Dr." in pos_t else [])


def build_family(att):
    rng = att.rng
    rel, rel2 = rng.sample(RELATIONS, 2)
    pos_t, neg_t = rng.choice(FAMILY_TEMPLATES)
    g, _ = att.person()
    kw = {"REL": rel, "REL2": rel2, "N": N(g), "HON": rng.choice(["Mr.", "Mrs.", "Ms."])}
    if "{N2}" in pos_t:
        g2, _ = att.person()
        kw["N2"] = N(g2)
    pos, neg = [pos_t.format(**kw)], [neg_t.format(**kw)]
    if rng.random() < 0.25 and "{N2}" not in pos_t:
        fp, fn = rng.choice(FAMILY_FOLLOW)
        pos.append(fp.format(**kw))
        neg.append(fn.format(**kw))
    return pos, neg, ["family_or_caregiver", "first_name_only"] + (["honorific_outside"] if "{HON}" in pos_t else [])


def derive_nick(rng, given):
    if rng.random() < 0.5:
        return rng.choice(NICKS)
    base = given.split("-")[0]
    stem = rng.choice([base[:3], base[:4], base[-3:], base[-4:]])
    return stem.capitalize() + rng.choice(["y", "ie", "o", "a", "s", "i"])


def build_alias(att):
    rng = att.rng
    g, s = att.person(count_surname=True)
    for _ in range(100):
        al = derive_nick(rng, g)
        if len(al) >= 3 and al.casefold() != g.casefold() and att.usable(al):
            break
    else:
        raise RuntimeError("no alias")
    att.take(al)
    mid = f" {rng.choice('ABCDEGHJKLMNPRSTW')}." if rng.random() < 0.2 else ""
    suffix = f" {rng.choice(['Jr.', 'Sr.', 'II', 'III'])}" if rng.random() < 0.12 else ""
    fn = f"{g}{mid} {s}{suffix}"
    role = rng.choice(ALIAS_ROLES)
    kw = {"FN": N(fn), "AL": N(al), "R": role, "W": rng.choice(ORDINARY_NICK), "HON": rng.choice(["Ms.", "Mr.", "Mrs."])}
    ep, en = rng.choice(ALIAS_EST)
    pos, neg = [ep.format(**kw)], [en.format(**kw)]
    for lp, ln in rng.sample(ALIAS_LATER, rng.randint(1, 2)):
        pos.append(lp.format(**kw))
        neg.append(ln.format(**kw))
    phen = ["alias_after_full_name", "later_alias_mention"]
    if "{HON}" in ep:
        phen.append("honorific_outside")
    if mid or suffix:
        phen.append("initial_or_suffix_inside")
    if '("{AL}")' in ep:
        phen.append("quoted_nickname")
    return pos, neg, phen


def lastfirst_value(rng, att):
    g, s = att.person(count_surname=True)
    suffix = rng.choice(["Jr.", "Sr.", "II", "III", "IV"]) if rng.random() < 0.15 else ""
    init = rng.choice("ABCDEFGHJKLMNPRSTVW") if rng.random() < 0.6 else ""
    return f"{s}, {g}" + (f" {suffix}" if suffix else "") + (f" {init}" if init else ""), bool(suffix), bool(init)


def build_lastfirst(att):
    rng = att.rng
    pos, neg, phen = [], [], ["last_first_form"]
    for pos_t, neg_t in rng.sample(LASTFIRST, rng.randint(1, 3)):
        nm, suf, ini = lastfirst_value(rng, att)
        kw = {"NM": N(nm), "REL": rng.choice(RELATIONS), "PH": span("PHONE", phone(rng))}
        pos.append(pos_t.format(**kw))
        neg.append(neg_t.format(**kw))
        if suf:
            phen.append("suffix_inside")
        if ini:
            phen.append("initial_inside")
        if pos_t.endswith((" MD", " RN")):
            phen.append("credential_outside")
    return pos, neg, list(dict.fromkeys(phen))


def build_lastfirst_solo(att):
    rng = att.rng
    lines = rng.sample(LASTFIRST_SOLO_NEG, 2 if rng.random() < 0.4 else 1)
    return None, lines, ["single_letter_list_marker"]


def build_namelike(att):
    rng = att.rng
    for _ in range(2000):
        w = rng.choice(list(NAME_LIKE))
        if att.usable(w):
            break
    else:
        raise RuntimeError("no word")
    att.take(w)
    return None, [rng.choice(NAME_LIKE[w]).format(W=w)], ["name_like_ordinary_word"]


def build_solo_dispatch(family):
    return build_lastfirst_solo if family == "last_first_forms" else build_namelike


BUILDERS = {
    "staff_first_name": build_staff,
    "family_first_name": build_family,
    "alias_after_full_name": build_alias,
    "last_first_forms": build_lastfirst,
}


def apply_case(marked, mode):
    if mode == "normal":
        return marked
    f = str.upper if mode == "upper" else str.lower
    out, prev = [], 0
    for m in MARK.finditer(marked):
        out.append(f(marked[prev:m.start()]))
        out.append(f"[[{m[1]}|{f(m[2])}]]")
        prev = m.end()
    out.append(f(marked[prev:]))
    return "".join(out)


def parse_marked(marked):
    pieces, spans, pos, prev = [], [], 0, 0
    for m in MARK.finditer(marked):
        lit = marked[prev:m.start()]
        pieces.append(lit)
        pos += len(lit)
        pieces.append(m[2])
        spans.append({"start": pos, "end": pos + len(m[2]), "label": m[1], "text": m[2]})
        pos += len(m[2])
        prev = m.end()
    pieces.append(marked[prev:])
    return "".join(pieces), spans


def assign_cases(units, rng):
    """Greedy quota fill so the row-level case mix is exact to within one unit."""
    total = sum(u["size"] for u in units)
    remaining = {c: CASE_SHARE[c] * total for c in CASES}
    order = list(range(len(units)))
    rng.shuffle(order)
    for i in order:
        best = max(CASES, key=lambda c: remaining[c] + rng.random() * 0.01)
        units[i]["case"] = best
        remaining[best] -= units[i]["size"]


def exclusion_check(texts, spans_list, counters):
    """True when clean. Counts only; never records which string matched."""
    for text, spans in zip(texts, spans_list):
        if any(excluded_value(s["text"]) for s in spans):
            counters["rejected_inserted_value"] += 1
            return False
        if excluded_sequences(text):
            counters["rejected_text_sequence"] += 1
            return False
    return True


def build_unit(family, unit, st, counters):
    builder = build_solo_dispatch(family) if unit["kind"] == "solo" else BUILDERS[family]
    for attempt in range(400):
        rng = random.Random(f"{SEED}:{family}:{unit['index']}:{attempt}")
        att = Attempt(rng, st)
        pos, neg, phen = builder(att)
        ctx = make_ctx(rng, unit["layout"])
        outputs = []
        if unit["kind"] in ("pair", "single"):
            outputs.append(("positive", pos))
        if unit["kind"] in ("pair", "solo"):
            outputs.append(("negative", neg))
        built = []
        for role, lines in outputs:
            text, spans = parse_marked(apply_case(render(ctx, lines), unit["case"]))
            built.append((role, text, spans))
        if not exclusion_check([b[1] for b in built], [b[2] for b in built], counters):
            continue
        if any(b[1] in st.texts for b in built):
            counters["rejected_duplicate_text"] += 1
            continue
        for key in set(att.names):
            st.counts[key] += 1
        for b in built:
            st.texts.add(b[1])
        groups = list(dict.fromkeys([canonical_identity_group(att.hidden())] + [canonical_identity_group(p) for p in att.persons]))
        counters["redraws"] += attempt
        return built, phen, groups
    raise RuntimeError(f"unit {family} {unit['index']} could not be drawn")


def build_bank():
    rng = random.Random(SEED)
    st, counters = State(), Counter()
    rows = []
    deck = []
    for family, (pairs, singles, solos) in PLAN.items():
        units = ([{"kind": "pair", "size": 2} for _ in range(pairs)]
                 + [{"kind": "single", "size": 1} for _ in range(singles)]
                 + [{"kind": "solo", "size": 1} for _ in range(solos)])
        for i, u in enumerate(units):
            u["index"] = i
        assign_cases(units, rng)
        for u in units:
            if not deck:
                deck = LAYOUT_NAMES[:]
                rng.shuffle(deck)
            u["layout"] = deck.pop()
        for u in units:
            built, phen, groups = build_unit(family, u, st, counters)
            pid = f"tk15-{family}-{u['index']:04d}"
            for role, text, spans in built:
                rows.append({
                    "id": f"{pid}-{role[:3]}",
                    "text": text,
                    "spans": spans,
                    "split": "train",
                    "suite": SUITE,
                    "annotation_policy": POLICY_VERSION,
                    "cohort": "realistic",
                    "family": f"tk15_{family}",
                    "template_family": f"tk15_{family}",
                    "layout": u["layout"],
                    "phenomena": phen + [f"case_{u['case']}", "matched_negative" if role == "negative" else "positive"],
                    "identity_group": groups[0],
                    "identity_groups": groups,
                    "pair_id": pid if u["kind"] == "pair" else f"{pid}-unpaired",
                    "case": u["case"],
                    "role": role,
                })
    rng.shuffle(rows)
    for n, row in enumerate(rows):
        row["id"] = f"tk15-{n:04d}-" + row["id"].split("-", 1)[1]
    return rows, counters, st


def stats_for(rows):
    out = {"rows": len(rows)}
    out["by_family"] = dict(Counter(r["family"] for r in rows))
    out["by_family_role"] = {f: dict(Counter(r["role"] for r in rows if r["family"] == f)) for f in out["by_family"]}
    out["case_mix"] = {c: sum(1 for r in rows if r["case"] == c) for c in CASES}
    out["layouts"] = dict(Counter(r["layout"] for r in rows))
    out["negatives"] = sum(1 for r in rows if r["role"] == "negative")
    out["label_support"] = {lab: sum(1 for r in rows for s in r["spans"] if s["label"] == lab) for lab in ENTITY_TYPES}
    return out


def name_value_rows(rows):
    """Rows per casefolded name span value, and per name-like ordinary word."""
    counts = Counter()
    for r in rows:
        seen = {s["text"].casefold() for s in r["spans"] if s["label"] == "NAME"}
        if r["family"] == "tk15_name_like_words":
            seen |= {w for w in re.findall(r"[^\W\d_]+", r["text"].casefold()) if w in NAME_LIKE}
        for v in seen:
            counts[v] += 1
    return counts


def write_spot_check(rows, path):
    rng = random.Random(SEED + 1)
    by_family = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r)
    chosen = []
    for items in by_family.values():
        chosen += rng.sample(items, 20)
    lines = ["# Spot-check sample (100 rows)", "",
             "Twenty rows per family, drawn with a fixed seed. Gold spans are listed under each row as `LABEL [text] start-end`.", ""]
    for r in chosen:
        lines += [f"## {r['id']} ({r['family']}, {r['case']}, {r['role']}, layout {r['layout']})", "", "```text", r["text"], "```", ""]
        lines += ["Gold: " + ("; ".join(f"{s['label']} [{s['text']}] {s['start']}-{s['end']}" for s in r["spans"]) or "none"), ""]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tk15_train_name_mentions.jsonl")
    ap.add_argument("--report-dir", default="reports/takeover/author15_name_mentions")
    args = ap.parse_args()
    rows, counters, st = build_bank()
    for r in rows:
        errs = annotation_errors(r["text"], r["spans"])
        if errs:
            raise SystemExit(f"{r['id']}: {errs}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    out.write_bytes(payload)
    stats = stats_for(rows)
    nv = name_value_rows(rows)
    final_matches = sum(1 for r in rows if excluded_sequences(r["text"]) or any(excluded_value(s["text"]) for s in r["spans"]))
    support = stats["label_support"]
    manifest = {
        "corpus": out.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(rows),
        "documents": len(rows),
        "split": "train",
        "annotation_policy": POLICY_VERSION,
        "independent_final": False,
        "suite": SUITE,
        "generator": "scripts/takeover/author15_name_mentions.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": SEED,
        "label_support": support,
        "entity_support": support,
        "label_counts": support,
        "families": stats["by_family"],
        "families_by_role": stats["by_family_role"],
        "case_mix": stats["case_mix"],
        "negative_rows": stats["negatives"],
        "layouts": stats["layouts"],
        "name_value_row_cap": CAP,
        "max_rows_per_name_value": max(nv.values()),
        "distinct_name_values": len(nv),
        "exclusion_filter": {
            "list": "data/places_exclusions.sha256.json",
            "rejected_draws": dict(sorted((k, v) for k, v in counters.items() if k.startswith("rejected"))),
            "total_redraws": counters["redraws"],
            "final_rows_with_any_match": final_matches,
        },
        "brief": "docs/name-mentions-bank-brief.md",
    }
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    rd = ROOT / args.report_dir
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "stats.json").write_text(json.dumps({"stats": stats, "counters": dict(counters)}, indent=1) + "\n")
    write_spot_check(rows, rd / "spot_check_sample.md")
    print(json.dumps({"rows": len(rows), "sha256": manifest["sha256"], "case_mix": stats["case_mix"],
                      "negatives": stats["negatives"], "max_rows_per_name_value": max(nv.values()),
                      "exclusion": manifest["exclusion_filter"]}, indent=1))


if __name__ == "__main__":
    main()
