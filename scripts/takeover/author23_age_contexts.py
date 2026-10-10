"""Author the tk23 training bank: ages in context, with matched decoys.

Commissioned by docs/age-contexts-bank-brief.md. Annotation contract:
docs/annotation-policy-v2.md (clinical-phi-v2), including the 2026-10-06
all-ages rule and the 2026-10-07 decisions D1 (a hypothetical, example or
template person's age is not AGE) and D2 (a decade of any age is AGE on the
decade word; "in her", "early" and "late" stay outside). Fixtures:
src/outis/data/annotation_policy_ages.py.

Every person, date, identifier, facility and locality is invented. No real
place names are used, so the places addendum's value pools do not apply.
The blind-bank exclusion filter (hash list only) is applied to every name and
place value and to every 1-8 word sequence of every row.

Deterministic for its seed:
  PYTHONPATH=src python scripts/takeover/author23_age_contexts.py [--seed 2323]
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
from outis.data.generators.places import excluded_sequences, excluded_value  # noqa: E402
from outis.types import ENTITY_TYPES  # noqa: E402

SEED = 2323
SUITE = "tk23_age_contexts"
OUTPUT = ROOT / "data" / "tk23_train_age_contexts.jsonl"
MRN_SCHEME = "AC- prefix and seven invented random digits"
PHONE_SCHEME = "US 555-0100..0199 fictional lines with invented area codes"

# family: (positive rows, decoy rows)
FAMILIES = {
    "uncued_patient_ages": (255, 45),
    "relatives_household": (215, 35),
    "spoken_ages": (170, 30),
    "infant_child_ages": (130, 20),
    "decades": (130, 20),
    "hypothetical_guideline_ages": (0, 450),
}
FAMILY_LAYOUTS = {
    "uncued_patient_ages": [
        "hp_note", "progress_note", "intake_form", "geriatric_assessment", "ed_triage", "nursing_note",
        "social_work_note", "home_health_visit", "handoff", "discharge_summary", "referral_letter",
        "portal_message", "preop_assessment", "pharmacy_consult", "care_conference_note"],
    "relatives_household": [
        "family_history", "hp_note", "social_work_note", "intake_form", "geriatric_assessment",
        "home_health_visit", "portal_message", "care_conference_note", "discharge_summary", "referral_letter"],
    "spoken_ages": ["telehealth_transcript", "phone_triage_transcript", "bedside_transcript", "dictation"],
    "infant_child_ages": [
        "pediatric_note", "well_child_visit", "nicu_note", "portal_message", "phone_triage_transcript", "ed_triage"],
    "decades": [
        "hp_note", "geriatric_assessment", "social_work_note", "portal_message", "family_history",
        "telehealth_transcript", "referral_letter", "care_conference_note"],
    "hypothetical_guideline_ages": [
        "education_excerpt", "guideline_excerpt", "progress_note", "pharmacy_consult", "preop_assessment",
        "discharge_summary", "hp_note", "well_child_visit", "geriatric_assessment", "telehealth_transcript",
        "intake_form", "nursing_note", "referral_letter", "care_conference_note", "dictation"],
}
TRANSCRIPT_LAYOUTS = {"telehealth_transcript", "phone_triage_transcript", "bedside_transcript"}

MARK = re.compile(r"\[\[([A-Z]+)\|([^\[\]|]+)\]\]")

# ---------------------------------------------------------------- markup


def parse_marked(marked: str) -> tuple[str, list[dict]]:
    """Turn ``[[LABEL|value]]`` markup into text and exact character-offset spans."""
    pieces, spans, offset, previous = [], [], 0, 0
    for match in MARK.finditer(marked):
        literal = marked[previous:match.start()]
        pieces.extend((literal, match[2]))
        offset += len(literal)
        spans.append({"start": offset, "end": offset + len(match[2]), "label": match[1], "text": match[2]})
        offset += len(match[2])
        previous = match.end()
    pieces.append(marked[previous:])
    text = "".join(pieces)
    if "[[" in text or "]]" in text:
        raise ValueError("Malformed markup")
    for span in spans:
        if span["label"] not in ENTITY_TYPES:
            raise ValueError(f"Unknown label {span['label']}")
    return text, spans


def tag(label: str, value: str) -> str:
    return f"[[{label}|{value}]]"


def A(value) -> str:
    return tag("AGE", str(value))


def N(value: str) -> str:
    return tag("NAME", value)


def D(value: str) -> str:
    return tag("DATE", value)


ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen").split()
TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}
DECADE_WORDS = {2: "twenties", 3: "thirties", 4: "forties", 5: "fifties", 6: "sixties", 7: "seventies",
                8: "eighties", 9: "nineties"}


def an(value) -> str:
    """Indefinite article for an age said aloud: 'an 81-year-old', 'an eleven-year-old'."""
    spoken = str(value).lower()
    return "an" if spoken.startswith(("8", "eight", "eleven")) or spoken in ("11", "18") else "a"


def words(n: int) -> str:
    if not 0 <= n <= 99:
        raise ValueError("spelled ages cover 0-99")
    if n < 20:
        return ONES[n]
    tens, ones = divmod(n, 10)
    return TENS[tens] + (f"-{ONES[ones]}" if ones else "")


WORD_VALUES = {words(n): n for n in range(100)}
DECADE_VALUES = {**{w: k * 10 for k, w in DECADE_WORDS.items()}, **{f"{k}0s": k * 10 for k in DECADE_WORDS}}


def decade_phrase(possessive: str, decade_word: str, modifier: str = "") -> str:
    """D2: AGE on the decade word only; 'in her', 'early' and 'late' stay outside."""
    lead = f"in {possessive} " + (f"{modifier} " if modifier else "")
    return lead + A(decade_word)


# ---------------------------------------------------------------- invented pools

FEMALE = ("Ada Bettina Calla Delphine Edie Faye Greta Hollis Ines Juna Kerstin Lorna Maribel Nell Odette Pia "
          "Quinta Rosalind Sabine Thea Una Vivian Wren Ysolde Zinnia Amara Bridget Coral Dagny Elspeth Flora "
          "Gwen Hester Imogen Jessamy Kaia Leonie Mabry Noor Oona Petra Rhea Saoirse Tamsin Verity Willa "
          "Yara Zelda Alma Beryl Cleo Dorit Esme Fern Gilda Halle Ilse Joanna Kit Lark").split()
MALE = ("Abel Bram Cyrus Dex Emory Felix Gideon Hugo Ivo Jasper Kees Lionel Milo Nils Otis Pascal Quill "
        "Ronan Silas Tobin Ulric Viggo Wendell Xavi Yusuf Zeke Alban Basil Corin Dashiell Elias Fitz Gage "
        "Hale Idris Jory Kasimir Leif Matthias Nico Orrin Perrin Rafe Soren Teodor Vance Warrick Yannick "
        "Anselm Bertil Caspian Dov Egon Florian Garrick Henrik Ignatius Jules Konrad Lorcan").split()
SUR_A = ("Al Bran Cor Dal Ell Fen Gar Hal Iver Jor Kel Lind Mar Nor Os Pell Quen Rav Sel Tor Ul Wen Yar "
         "Zell Ash Cald Dun Hart Mor Thorn Wick Bel Crane Drum Elm Fal Grim Holl Kest Lang Mell Ness Orm "
         "Pren Rook Stam Tull Vell Wex").split()
SUR_B = ("ford wick by ton stead more well ridge sley brook dale worth hollow ham croft mere son land "
         "gate field cott thwaite bury shaw").split()
PLACE_A = ("Quill Ost Bel Tarn Fenni Harrow Marl Corri Vess Ombr Lyn Brack Kell Selb Drem Fall Gorse Hath "
           "Lorn Pyre Ravel Stell Thist Umber Wyn Yarl").split()
PLACE_B = ("mere haven brook ford wick vale stead fell holt mouth crest field moor ridge").split()
STATES = ("UT NM OR WA ID MT WY CO NV AZ NE KS IA MN WI MI OH PA VT ME").split()
FACILITY_TYPES = ("Medical Center", "Family Medicine", "Community Hospital", "Health Clinic", "Pediatrics",
                  "Senior Care Clinic", "Regional Hospital", "Internal Medicine", "Children's Clinic",
                  "Health Center")

PEDIATRIC_TYPES = ("Pediatrics", "Children's Clinic")
COMPLAINTS = ("cough", "chest pain", "dizziness", "a fall at home", "shortness of breath", "back pain",
              "abdominal pain", "fatigue", "leg swelling", "a rash", "headache", "palpitations", "urinary frequency",
              "knee pain", "confusion", "poor appetite")
HISTORY = ("hypertension", "type 2 diabetes", "COPD", "atrial fibrillation", "CKD stage 3", "osteoarthritis",
           "heart failure", "hypothyroidism", "asthma", "depression", "Parkinson disease", "osteoporosis",
           "gout", "glaucoma", "migraine")
DRUGS = ("lisinopril", "metformin", "amlodipine", "atorvastatin", "levothyroxine", "sertraline",
         "furosemide", "apixaban", "gabapentin", "tamsulosin")
JOBS = ("teacher", "machinist", "nurse", "bus driver", "bookkeeper", "carpenter", "farmer", "postal carrier",
        "librarian", "electrician", "cook", "welder")
CAUSES = ("heart failure", "a stroke", "colon cancer", "pneumonia", "kidney failure", "an MI", "dementia",
          "lung cancer")


class Pools:
    """Invented value pools, filtered against the blind-bank exclusion hashes."""

    def __init__(self):
        self.dropped = Counter()
        self.female = self._keep("first_name", FEMALE)
        self.male = self._keep("first_name", MALE)
        self.surnames = self._keep("surname", sorted({a + b for a in SUR_A for b in SUR_B}))
        self.places = self._keep("place", sorted({a + b for a in PLACE_A for b in PLACE_B}))
        facilities = [f"{p} {t}" for p in self.places for t in FACILITY_TYPES]
        self.facilities = self._keep("facility", facilities)
        self.considered = {
            "first_name": len(FEMALE) + len(MALE), "surname": len(SUR_A) * len(SUR_B),
            "place": len(PLACE_A) * len(PLACE_B), "facility": len(facilities),
        }

    def _keep(self, kind, values):
        kept = []
        for value in values:
            if excluded_value(value):
                self.dropped[kind] += 1
            else:
                kept.append(value)
        return kept


# ---------------------------------------------------------------- per-row context


class Ctx:
    def __init__(self, r: random.Random, pools: Pools, kind: str = "adult"):
        self.r = r
        self.female = r.random() < 0.55
        self.identities = []
        self.surfaces = {}
        self.first, self.last, self.name = self.person(self.female)
        self.identity_group = canonical_identity_group(self.name)
        self.she, self.her, self.hers = ("she", "her", "her") if self.female else ("he", "his", "him")
        self.She, self.Her = self.she.capitalize(), self.her.capitalize()
        self.noun = "woman" if self.female else "man"
        self.sx = "F" if self.female else "M"
        self.child = "girl" if self.female else "boy"
        self.spouse = r.choice(["husband", "partner"]) if self.female else r.choice(["wife", "partner"])
        if kind == "adult":
            x = r.random()
            self.age = r.randint(100, 110) if x < 0.04 else r.randint(90, 99) if x < 0.15 else r.randint(18, 89)
        elif kind == "child":
            self.age = r.randint(1, 17)
        else:
            self.age = None
        relatives = ["daughter", "son", "niece", "sister", "brother"]
        if self.age is not None and self.age >= 50:
            relatives += ["grandson", "granddaughter"]
        if self.age is not None and self.age < 40:
            relatives = ["mother", "father", "sister", "brother", "roommate"]
        self.rel = r.choice(relatives)
        cf, self.clin_last, self.clin = self.person(r.random() < 0.5)
        _, self.clin2_last, self.clin2 = self.person(r.random() < 0.5)
        self.pools = pools
        place = r.choice(pools.places)
        self.loc = f"{place}, {r.choice(STATES)}" if r.random() < 0.6 else place
        self.fac = r.choice([f for f in pools.facilities if not f.endswith(PEDIATRIC_TYPES)])
        self.fac_peds = r.choice([f for f in pools.facilities if f.endswith(PEDIATRIC_TYPES)])
        self.mrn = f"AC-{r.randint(1000000, 9999999)}"
        self.phone = f"({r.randint(201, 989)}) 555-01{r.randint(0, 99):02d}"
        self.year = r.choice([2025, 2026])
        self.date, self.date2 = self.make_date_pair()
        self.kind = kind
        self.dob = self.make_dob()
        self.vitals()

    def person(self, female):
        pools = getattr(self, "pools", None)
        for _ in range(200):
            first = self.r.choice(PERSON_POOLS.female if female else PERSON_POOLS.male)
            last = self.r.choice(PERSON_POOLS.surnames)
            name = f"{first} {last}"
            group = canonical_identity_group(name)
            if identity_split(group) == "train" and group not in self.identities:
                self.identities.append(group)
                self.surfaces[group] = (name, last)
                return first, last, name
        raise RuntimeError(f"no train-partition name found {pools}")

    def relative(self, female=None):
        first, last, name = self.person(self.r.random() < 0.5 if female is None else female)
        return first

    def make_date(self):
        m, d = self.r.randint(1, 12), self.r.randint(1, 28)
        months = ("January February March April May June July August September October November "
                  "December").split()
        style = self.r.randrange(4)
        if style == 0:
            return f"{m:02d}/{d:02d}/{self.year}"
        if style == 1:
            return f"{months[m - 1]} {d}, {self.year}"
        if style == 2:
            return f"{self.year}-{m:02d}-{d:02d}"
        return f"{months[m - 1][:3]} {d}, {self.year}"

    def make_date_pair(self):
        months = ("January February March April May June July August September October November "
                  "December").split()
        m, d = self.r.randint(1, 11), self.r.randint(1, 20)
        m2, d2 = (m, d + self.r.randint(2, 8)) if self.r.random() < 0.5 else (m + 1, self.r.randint(1, 28))
        style = self.r.randrange(4)

        def fmt(mm, dd):
            if style == 0:
                return f"{mm:02d}/{dd:02d}/{self.year}"
            if style == 1:
                return f"{months[mm - 1]} {dd}, {self.year}"
            if style == 2:
                return f"{self.year}-{mm:02d}-{dd:02d}"
            return f"{months[mm - 1][:3]} {dd}, {self.year}"
        return fmt(m, d), fmt(m2, d2)

    def make_dob(self):
        age = self.age if self.age is not None else self.r.randint(0, 1)
        y = self.year - age - (1 if self.r.random() < 0.5 else 0)
        return f"{self.r.randint(1, 12):02d}/{self.r.randint(1, 28):02d}/{y}"

    def vitals(self):
        r, age = self.r, self.age or 0
        self.hr = age if 50 <= age <= 110 and r.random() < 0.5 else r.randint(56, 104)
        self.kg = age if 45 <= age <= 110 and r.random() < 0.5 else r.randint(48, 118)
        self.room = age if age >= 10 and r.random() < 0.5 else r.randint(101, 640)
        self.sbp, self.dbp = r.randint(102, 168), r.randint(58, 96)
        self.phq, self.gad, self.moca = r.randint(0, 27), r.randint(0, 21), r.randint(14, 30)
        self.gluc, self.lb = r.randint(84, 290), r.randint(5, 40)
        self.mg = r.choice([5, 10, 20, 25, 40, 50, 100])
        self.yrs, self.grade = r.randint(2, 45), r.randint(6, 11)
        self.cigs, self.steps = r.choice([5, 10, 15, 20, 30, 40]), r.choice([800, 1500, 2400, 3000, 4200])
        self.bmi, self.a1c = r.randint(19, 41), round(r.uniform(6.1, 11.8), 1)
        self.barthel, self.feet = r.choice([60, 70, 80, 85, 90, 95, 100]), r.choice([50, 100, 150, 200, 300])
        self.spo2, self.pain = r.randint(88, 99), r.randint(0, 10)
        self.cc, self.pmh = r.choice(COMPLAINTS), r.choice(HISTORY)
        self.drug, self.job = r.choice(DRUGS), r.choice(JOBS)


PERSON_POOLS: Pools  # set in build()

# ---------------------------------------------------------------- family 1: uncued patient ages


def uncued_templates(c: Ctx):
    a = A(c.age)
    return [
        (f"{c.She} is {a} and lives alone.", f"{c.Her} HR is {c.hr} and regular; {c.she} lives alone."),
        (f"Pt {a} presenting with {c.cc}.", f"Pt presenting with {c.cc}; PHQ-9 is {c.phq}."),
        (f"{a} y.o. {c.noun} with {c.pmh}.", f"{c.noun.capitalize()} with {c.pmh}, on {c.drug} {c.mg} mg daily."),
        (f"Turning {a} next month and wants to keep gardening.",
         f"Glucose {c.gluc} this morning; wants to keep gardening."),
        (f"{c.first}, {a}, arrived by car with {c.her} {c.rel}.",
         f"{c.first}, room {c.room}, arrived by car with {c.her} {c.rel}."),
        (f"At {a}, {c.she} still bakes for the church sale.",
         f"At {c.kg} kg, {c.she} still bakes for the church sale."),
        (f"{c.She} is now {a} and uses a cane outdoors.", f"{c.She} is now {c.kg} kg and uses a cane outdoors."),
        (f"{a}{c.sx} with {c.cc}.", f"{c.sx} pt with {c.cc}; HR {c.hr}, BP {c.sbp}/{c.dbp}."),
        (f"{c.She} left school at {a} to work on the family farm.",
         f"{c.She} left school after grade {c.grade} to work on the family farm."),
        (f"{c.She} quit smoking at {a}.", f"{c.She} smoked {c.cigs} cigarettes a day before quitting."),
        (f"{c.She} has used insulin since age {a}.", f"{c.She} has used insulin for {c.yrs} years."),
        (f"{c.She}'s {a} and sharp as ever.", f"{c.Her} MoCA is {c.moca}/30 and {c.she} is sharp as ever."),
        (f"Now {a}, {c.she} needs help with stairs.", f"Now down {c.lb} lb, {c.she} needs help with stairs."),
        (f"{N(c.name)} is {an(c.age)} {a}-year-old {c.noun} seen for {c.cc}.",
         f"{N(c.name)} is seen for {c.cc}; weight {c.kg} kg today."),
        (f"Pt is {a} yo, lives with {c.her} {c.rel}.", f"Pt is {c.kg} kg, lives with {c.her} {c.rel}."),
        (f"{an(c.age).capitalize()} {a} y/o {c.noun} who presents for follow-up of {c.pmh}.",
         f"A {c.noun} who presents for follow-up of {c.pmh}, BMI {c.bmi}."),
        (f"{c.She} was {a} when the {c.pmh} was diagnosed.",
         f"The {c.pmh} was diagnosed when {c.her} A1c reached {c.a1c}."),
        (f"{c.She} turned {a} last week and celebrated with family.",
         f"{c.She} walked {c.steps} steps last week and celebrated with family."),
        (f"Retired at {a} after a career as a {c.job}.", f"Retired after {c.yrs} years as a {c.job}."),
        (f"{c.She} is {a} years old and independent in ADLs.",
         f"Barthel {c.barthel}/100; {c.she} is independent in ADLs."),
        (f"Pt ({a}{c.sx}) ambulating in hall with walker.", f"Pt ambulating {c.feet} ft in hall with walker."),
        (f"{c.first} is {a} and still drives to church.", f"{c.first} is on {c.drug} {c.mg} mg and still drives to church."),
        (f"{a} year old {c.noun}, independent at baseline.", f"SpO2 {c.spo2}% on room air, independent at baseline."),
        (f"Pt states {c.she} is {a} and has never been hospitalized.",
         f"Pt states pain is {c.pain}/10 and {c.she} has never been hospitalized."),
    ]


UNCUED_DECOYS = (
    "HR {hr}, BP {sbp}/{dbp}.", "PHQ-9 is {phq}.", "Weight {kg} kg.", "GAD-7 score {gad}.", "Pain {pain}/10.",
    "Room {room}.", "SpO2 {spo2}% on room air.", "Glucose {gluc}.", "MoCA {moca}/30.", "BMI {bmi}.",
    "Walked {feet} ft with PT.", "{drug} {mg} mg daily.",
)

# ---------------------------------------------------------------- family 2: relatives and household


def relative_templates(c: Ctx):
    r = c.r
    old, mid, kid = r.randint(60, 108), r.randint(25, 70), r.randint(1, 17)
    died = r.randint(40, 104)
    n_small = r.randint(2, 9)
    rel_name = c.relative()
    return [
        (f"{c.Her} {c.spouse} is {A(old)} and still drives.",
         f"{c.Her} {c.spouse} has driven for {c.yrs} years and still drives."),
        (f"Grandson, {A(kid)}, lives with {c.hers}.", f"{n_small} grandchildren; one lives with {c.hers}."),
        (f"Mother passed at {A(died)} of {r.choice(CAUSES)}.",
         f"Mother passed after {c.yrs} years with {r.choice(CAUSES)}."),
        (f"Father died at age {A(died)} of {r.choice(CAUSES)}.", f"Father had {n_small} admissions for {r.choice(CAUSES)}."),
        (f"Sister ({A(mid)}) with breast cancer.", f"{n_small} sisters, one with breast cancer."),
        (f"Son is {A(mid)} and helps with medications.", f"Son visits {n_small} times a week and helps with medications."),
        (f"Lives with {c.her} daughter, {A(mid)}, and son-in-law.", f"Lives with {c.her} daughter and {n_small} cats."),
        (f"Brother, {A(mid)}, had a stroke last year.", f"Brother had a stroke {n_small} years ago."),
        (f"{c.spouse.capitalize()} ({A(old)}) is the primary caregiver.",
         f"{c.spouse.capitalize()} of {c.yrs} years is the primary caregiver."),
        (f"Mother: deceased at {A(died)}, CHF.", f"Mother: CHF, on {n_small} medications."),
        (f"Partner, {A(mid)}, works nights.", f"Partner works {r.choice([10, 12])}-hour night shifts."),
        (f"{c.Her} mother is {A(old)} and lives in a nursing home.",
         f"{c.Her} mother has lived in a nursing home for {n_small} years."),
        (f"Roommate, {A(mid)}, checks on {c.hers} daily.", f"Roommate checks on {c.hers} {n_small} times daily."),
        (f"Child at home ({A(kid)} y/o).", f"{n_small} children at home."),
        (f"Daughter {N(rel_name)} ({A(mid)}) is health care proxy.",
         f"Daughter {N(rel_name)} is health care proxy; married {c.yrs} years."),
        (f"Father is {A(old)}, legally blind, lives with the family.",
         f"Father, legally blind for {n_small} years, lives with the family."),
        (f"Granddaughter, age {A(r.randint(6, 17))}, stays after school.", f"Granddaughter stays after school {n_small} days a week."),
        (f"Married {c.yrs} years; {c.spouse} is {A(old)}.", f"Married {c.yrs} years; {c.spouse} had {n_small} falls this year."),
        (f"Maternal aunt diagnosed with ovarian cancer at {A(mid)}.",
         f"Maternal aunt treated for ovarian cancer for {n_small} years."),
        (f"Paternal grandfather lived to {A(r.randint(85, 106))}.",
         f"Paternal grandfather had {n_small} heart attacks."),
        (f"Nephew ({A(kid)}) moved in after school ended.", f"Nephew moved in {n_small} weeks ago."),
    ]


RELATIVE_DECOYS = (
    "Married {yrs} years.", "Has {lb} grandchildren nearby.", "Father smoked for {yrs} years.",
    "Mother had 2 strokes.", "Lives with {lb} family members.", "Son calls every {grade} days.",
)

# ---------------------------------------------------------------- family 3: spoken ages (turns)


def spoken_templates(c: Ctx):
    r = c.r
    age = min(c.age, 99)
    aw = words(age)
    other = r.randint(20, 99)
    ow = words(other)
    num = words(r.randint(2, 99))
    small = words(r.randint(2, 10))
    corr = r.randint(40, 98)
    spouse_age = r.randint(55, 99)
    return [
        ([("clin", "and how old are you?"), ("pt", f"i'm {A(aw)}.")],
         [("clin", "and what room are you in?"), ("pt", f"room {ow}.")]),
        ([("clin", "how old is your mother now?"), ("fam", f"she'll be {A(words(spouse_age))} next month.")],
         [("clin", "how long has she had the cough?"), ("fam", f"about {small} days now.")]),
        ([("clin", "how old are you, sir?" if not c.female else "how old are you, ma'am?"), ("pt", f"{A(aw)}."),
          ("clin", f"{A(aw)}, okay. and you live alone?"), ("pt", "yes.")],
         [("clin", "what was your sugar this morning?"), ("pt", f"{ow}."),
          ("clin", f"{ow}, okay. and you live alone?"), ("pt", "yes.")]),
        ([("pt", f"i'm {A(words(corr))}, no, {A(words(corr + 1))}, i always forget.")],
         [("pt", f"i take {small}, no, {words(min(10, r.randint(3, 11)))} pills a day, i always forget.")]),
        ([("pt", f"i just turned {A(aw)}.")], [("pt", f"i just took {small} of the tylenol.")]),
        ([("caller", f"my {c.spouse} is {A(words(spouse_age))}, and he fell this morning."
           if c.spouse != "wife" else f"my wife is {A(words(spouse_age))}, and she fell this morning.")],
         [("caller", f"my {c.spouse} fell about {num} minutes ago.")]),
        ([("pt", f"i was {A(words(r.randint(17, 40)))} when i had my first baby.")],
         [("pt", f"i was in labor for {small} hours with my first baby.")]),
        ([("clin", "how old are you now?"), ("pt", f"{A(aw)} years young."), ("clin", "good for you.")],
         [("clin", "how far can you walk?"), ("pt", f"maybe {num} yards."), ("clin", "good for you.")]),
        ([("interp", f"she says she is {A(aw)} and lives with her son."
           if c.female else f"he says he is {A(aw)} and lives with his son.")],
         [("interp", f"she says the pain is {an(small)} {small} and she lives with her son."
           if c.female else f"he says the pain is {an(small)} {small} and he lives with his son.")]),
        ([("pt", f"my son's {A(words(r.randint(20, 70)))}, he lives with me.")],
         [("pt", f"my son's been with me {small} years now.")]),
        ([("clin", "can you confirm your age for me?"), ("pt", f"{A(age)}.")],
         [("clin", "can you confirm your room number for me?"), ("pt", f"{r.randint(101, 640)}.")]),
        ([("pt", f"i'm {A(aw)} and i still mow my own lawn.")],
         [("pt", f"i weigh about one {words(r.randint(10, 90))} and i still mow my own lawn.")]),
        ([("fam", f"mom's {A(words(spouse_age))} now, she forgets things.")],
         [("fam", f"mom's on {small} different pills now, she forgets things.")]),
        ([("clin", "and your age?"), ("pt", f"{A(aw)}... no wait, {A(words(min(99, age + 1)))} since last week.")],
         [("clin", "and your pharmacy?"), ("pt", "the one on main street, i think.")]),
    ]


def dictation_templates(c: Ctx):
    r = c.r
    aw = words(min(c.age, 99))
    num = words(r.randint(56, 99))
    sex = "female" if c.female else "male"
    return [
        (f"this is {an(aw)} {A(aw)}-year-old {sex} seen today for {c.cc}.",
         f"this is a {sex} seen today for {c.cc}, pulse {num}."),
        (f"patient is {A(aw)} years of age and lives at home.", f"patient's heart rate is {num} and {c.she} lives at home."),
        (f"{c.she} is {A(c.age)} and independent.", f"{c.she} weighs {c.kg} kilograms and is independent."),
        (f"the patient, {A(aw)}, reports {c.cc}.", f"the patient, in room {words(r.randint(10, 99))}, reports {c.cc}."),
        (f"{c.she} turned {A(aw)} this past winter.", f"{c.she} walked {words(r.randint(20, 90))} minutes this past winter."),
    ]


SPOKEN_DECOYS = (
    ("clin", "you're in room {room} for now."), ("pt", "it's been about {smallw} minutes."),
    ("pt", "the pain's about {painart} {painw}."), ("clin", "we'll recheck in {smallw} hours."),
    ("pt", "i took {smallw} of them."),
)

# ---------------------------------------------------------------- family 4: infant and child ages


def infant_templates(c: Ctx):
    r = c.r
    wk, mo, dy, yr = r.randint(1, 12), r.randint(1, 23), r.randint(1, 14), r.randint(2, 17)
    ga = r.randint(28, 37)
    return [
        (f"{A(wk)} weeks old, here for weight check.", f"Born at {ga} weeks gestation, here for weight check."),
        (f"{A(mo)}-month-old {c.child} with fever.", f"{c.child.capitalize()} with fever, day {r.randint(2, 5)} of amoxicillin."),
        (f"Infant is {A(dy)} days old and breastfeeding well.",
         f"Infant feeds every {r.randint(2, 4)} hours and is breastfeeding well."),
        (f"{c.first} is {A(mo)} mo, rolling over.", f"{c.first} takes {r.choice([60, 90, 120, 150])} mL per feed, rolling over."),
        (f"{A(wk)}-week-old seen for jaundice.", f"Seen for jaundice; bilirubin {round(r.uniform(8, 19), 1)}."),
        (f"Now {A(max(mo, 7))} months; started solids at {A(r.randint(4, 6))} months.",
         f"Started solids; weight {round(r.uniform(6, 11), 1)} kg."),
        (f"{A(yr)} yo {c.child} with asthma.", f"{c.child.capitalize()} with asthma, albuterol {r.randint(2, 4)} puffs q4h prn."),
        (f"Sibling, {A(yr)}, at home with a cold.", f"{r.randint(2, 5)} siblings at home with colds."),
        (f"{A(dy)}-day-old born at {ga} weeks gestation.", f"Born at {ga} weeks gestation; Apgars 8 and 9."),
        (f"Mom reports the baby is {A(wk)} weeks today.", f"Mom reports {r.randint(2, 6)} days of loose stools."),
        (f"{A(r.randint(4, 6))}-year-old here for kindergarten physical.", "Here for kindergarten physical; vaccines due today."),
        (f"{A(mo)} month old {c.child}, {r.randint(10, 90)}th percentile for weight.",
         f"{c.child.capitalize()} at the {r.randint(10, 90)}th percentile for weight."),
        (f"Patient is {an(yr)} {A(yr)} y/o {c.child} with ear pain.", f"Patient is a {c.child} with ear pain x {r.randint(2, 5)} days."),
        (f"Twins, {A(mo)} months, both with cough.", f"Twins born at {ga} weeks, both with cough."),
        (f"{c.first}, {A(yr)}, broke {c.her} wrist at recess.", f"{c.first} broke {c.her} wrist at recess; {r.randint(4, 6)}-week cast."),
    ]


def infant_spoken_templates(c: Ctx):
    r = c.r
    wk, mo = r.randint(1, 12), r.randint(1, 23)
    return [
        ([("clin", "how old is the baby?"), ("caller", f"{c.she}'s {A(words(wk))} weeks old.")],
         [("clin", "was the baby early?"), ("caller", f"{c.she} was born at {words(r.randint(30, 37))} weeks.")]),
        ([("clin", "and how old is he?" if not c.female else "and how old is she?"), ("caller", f"{A(words(mo))} months.")],
         [("clin", "how long has the fever been going?"), ("caller", f"about {words(r.randint(2, 4))} days.")]),
        ([("caller", f"my son is {A(r.randint(2, 12))} and he's wheezing.")],
         [("caller", f"my son had {words(r.randint(2, 4))} puffs and he's still wheezing.")]),
        ([("caller", f"the baby's {A(words(r.randint(2, 14)))} days old and won't eat.")],
         [("caller", f"the baby hasn't eaten in {words(r.randint(3, 6))} hours.")]),
    ]


INFANT_DECOYS = (
    "Born at {ga} weeks gestation.", "Amoxicillin x {d} days.", "Follow up every 6 months.",
    "Weight {wt} kg.", "Feeds every {h} hours.", "Apgars 8 and 9.",
)

# ---------------------------------------------------------------- family 5: decades (D2)


def decade_templates(c: Ctx):
    r = c.r
    k = 9 if c.age is not None and c.age >= 90 and c.age < 100 else r.randint(2, 9)
    digits = r.random() < 0.4
    dw = f"{k}0s" if digits else DECADE_WORDS[k]
    k2 = r.randint(4, 9)
    dw2 = f"{k2}0s" if r.random() < 0.4 else DECADE_WORDS[k2]
    kp = r.randint(6, 9)
    dwp = f"{kp}0s" if r.random() < 0.4 else DECADE_WORDS[kp]
    mod = r.choice(["early", "late", ""])
    hr_dec = r.choice(["60s", "70s", "80s", "90s"])
    era = r.choice(["1950s", "1960s", "1970s", "1980s"])
    return [
        (f"{c.noun.capitalize()} {decade_phrase(c.her, dw)} with {c.pmh}.",
         f"{c.noun.capitalize()} with {c.pmh}; systolics in the {r.choice(['90s', '100s'])}."),
        (f"{c.She} is {decade_phrase(c.her, dw, mod)} and lives with {c.her} {c.rel}.",
         f"{c.Her} HR stays in the {hr_dec} and {c.she} lives with {c.her} {c.rel}."),
        (f"{c.Her} father, {decade_phrase('his', dwp, 'late')}, has dementia.",
         f"{c.Her} father has dementia and listens to music from the {era}."),
        (f"{c.She} has had {c.pmh} since {c.her} {A(DECADE_WORDS[r.randint(2, 6)])}.",
         f"{c.She} has had {c.pmh}; sats in the {r.choice(['80s', '90s'])} on arrival."),
        (f"{c.Her} mother, now {decade_phrase('her', dwp)}, lives nearby.",
         f"{c.Her} mother lives nearby and likes films from the {era}."),
        (f"A {c.noun} {decade_phrase(c.her, f'{k}0s')} presenting with {c.cc}.",
         f"A {c.noun} presenting with {c.cc}; glucose in the {r.choice(['200s', '300s'])}."),
        (f"{c.She} says {c.she} is {decade_phrase(c.her, dw)} and slowing down.",
         f"{c.She} says the {era} were hard and {c.she} is slowing down."),
        (f"Both parents are {decade_phrase('their', dwp)}.",
         "Screening discussed for relatives ages 40 to 60."),
        (f"{c.Her} {c.spouse}, {decade_phrase(SPOUSE_POSSESSIVE[c.spouse], dw2, mod)}, helps at home.",
         f"{c.Her} {c.spouse} helps at home; temps in the {r.choice(['80s', '90s'])} this week."),
        (f"Started running {decade_phrase(c.her, DECADE_WORDS[r.randint(3, 6)])}.",
         f"Started running; resting HR now in the {r.choice(['50s', '60s'])}."),
    ]


def decade_spoken_templates(c: Ctx):
    r = c.r
    k = r.randint(5, 9)
    return [
        ([("pt", f"i'm {decade_phrase('my', DECADE_WORDS[k])}, you know, things wear out.")],
         [("pt", f"my sugars run in the {r.choice(['two hundreds', 'one fifties'])}, you know.")]),
        ([("pt", f"my mother's {decade_phrase('her', DECADE_WORDS[r.randint(7, 9)])} and still sharp.")],
         [("pt", f"my mother loves the music from the {r.choice(['sixties', 'seventies'])}, still sharp."
           )]),
        ([("clin", "how old was your father when that started?"),
          ("pt", f"oh, {decade_phrase('his', DECADE_WORDS[r.randint(4, 7)], r.choice(['early', 'late']))}.")],
         [("clin", "what was your pressure at home?"), ("pt", f"oh, in the {r.choice(['one forties', 'one thirties'])}.")]),
    ]


SPOUSE_POSSESSIVE = {"husband": "his", "wife": "her", "partner": "their"}

DECADE_DECOYS = (
    "BP in the 90s systolic overnight.", "Sats in the 80s on arrival.", "The 1970s were hard.",
    "Guideline covers ages 40 to 60.", "HR in the 60s overnight.",
)

# ---------------------------------------------------------------- family 6: hypothetical and guideline ages (D1)


def d1_sentences(r: random.Random):
    n = r.randint(18, 99)
    n2 = r.choice([45, 50, 60, 65, 70, 75])
    pct = r.randint(20, 95)
    hypothetical = [
        f"A comparison exercise described a person aged {n}.",
        f"In the comparison exercise, a person aged {n} chose the lower dose.",
        f"For example, {an(n)} {n}-year-old would qualify for the program.",
        f"If a patient were {n}, the dose would change.",
        f"The teaching case describes {an(n)} {n}-year-old man with chest pain.",
        f"Consider a hypothetical {n}-year-old woman with the same risk factors.",
        f"The calculator's sample entry uses {an(n)} {n}-year-old nonsmoker.",
        f"Example: {an(n)} {n} yo with an eGFR of 40 would need a lower dose.",
        f"Suppose a person aged {n} started this medicine; the handout lists what to watch for.",
        f"The decision aid compares 100 people aged {n} with and without treatment.",
        f"The illustration shows {an(n)} {n}-year-old grandmother walking daily.",
        f"A sample dosing table starts with {an(n)} {n}-year-old weighing 70 kg.",
        f"If someone were {n} and had the same history, the risk would be higher.",
        f"In this practice scenario, the person is {n} and lives alone.",
        "The template reads: a __-year-old presents with __.",
        f"For instance, {an(n)} {n}-year-old with diabetes would start at the lower dose.",
        f"A worked example in the handout uses a man aged {n}.",
        f"In the video, {an(n)} {n}-year-old describes her first colonoscopy.",
        f"The quiz asks what {an(n)} {n}-year-old should do after a missed dose.",
        f"A person aged {n} in this exercise would score in the moderate range.",
    ]
    guideline = [
        f"Screen from age {r.choice([40, 45, 50])}.",
        "Per guideline, adults over 65 should get the pneumococcal vaccine.",
        "Colon screening is advised for adults aged 45 to 75.",
        "Shingles vaccine is recommended for adults 50 and older.",
        "Lung cancer screening applies to adults 50 to 80 with a 20 pack-year history.",
        "AAA screening is recommended once for men 65 to 75 who ever smoked.",
        "Bone density screening is advised for women 65 and older.",
        "Ibuprofen is not for infants under 6 months.",
        "Not recommended for children under 2.",
        "The RSV vaccine is offered to adults 60 and older.",
        "Annual wellness visit is covered for members 65 and older.",
        "Statin benefit is greatest for ages 40 to 75.",
        f"Adults over {n2} need a lower starting dose.",
        "Children older than 6 months may get the flu vaccine.",
        "This product is not for use under age 12.",
    ]
    duration = [
        f"5-year survival is about {pct}%.",
        f"{r.randint(2, 30)}-year history of asthma.",
        f"10-year ASCVD risk is {round(r.uniform(3, 30), 1)}%.",
        "30-day readmission risk is moderate.",
        "Completed a 2-week course of antibiotics.",
        "A 3-month trial of PT is planned.",
        "Follow up in 6 months.",
        f"{r.randint(5, 50)}-year history of smoking.",
        f"The device is {r.randint(5, 15)} years old and due for replacement.",
        f"The bruise appears about {r.randint(2, 7)} days old.",
        f"The house is {r.randint(40, 120)} years old and has steep stairs.",
        f"The scar is {r.randint(5, 40)} years old and well healed.",
        "A 72-hour hold was not needed.",
    ]
    return hypothetical, guideline, duration


def d1_core(r: random.Random):
    hyp, guide, dur = d1_sentences(r)
    picks = [r.choice(hyp)] if r.random() < 0.7 else []
    picks.append(r.choice(guide))
    if r.random() < 0.6:
        picks.append(r.choice(dur))
    if r.random() < 0.3:
        picks.append(r.choice([s for s in hyp if s not in picks]))
    r.shuffle(picks)
    return picks


def d1_spoken(r: random.Random):
    n = r.randint(30, 95)
    choices = [
        [("clin", f"the handout uses a person aged {words(n)} as the example."), ("pt", "okay.")],
        [("clin", f"so for example, {an(n)} {words(n)}-year-old would take the lower dose."), ("pt", "got it.")],
        [("clin", "screening starts at forty-five for most people."), ("pt", "okay.")],
        [("clin", "the vaccine is for adults over sixty-five."), ("pt", "i'll ask about it.")],
        [("clin", f"if a patient were {words(n)}, we'd change the dose."), ("pt", "makes sense.")],
        [("clin", "five-year survival is quite good with this."), ("pt", "that's a relief.")],
        [("clin", f"imagine a person aged {words(n)} with your same numbers; the tool shows their risk."),
         ("pt", "okay.")],
    ]
    picks = r.sample(choices, 2)
    return [turn for pick in picks for turn in pick]


# ---------------------------------------------------------------- layouts

PROSE_FILLER = (
    "No acute distress.", "Lungs clear bilaterally.", "Tolerating diet.", "Will follow up with primary care.",
    "Medications reconciled.", "Fall precautions reviewed.", "Denies fever or chills.", "Skin intact.",
    "Questions answered.", "Ambulates with steady gait.", "Plan discussed and agreed.", "Alert and oriented.",
)


def filler(r, k=2):
    return r.sample(PROSE_FILLER, k)


def prose_layout(layout: str, c: Ctx, core: list[str]) -> str:
    r = c.r
    body = " ".join(core)
    nm, clin, clin2 = N(c.name), N(c.clin), N(c.clin2)
    fac, loc = tag("FACILITY", c.fac), tag("LOCATION", c.loc)
    if layout in ("pediatric_note", "well_child_visit", "nicu_note"):
        fac = tag("FACILITY", c.fac_peds)
    mrn, date, date2 = tag("MRN", c.mrn), D(c.date), D(c.date2)
    phone, dob = tag("PHONE", c.phone), D(c.dob)
    f1, f2 = filler(r)
    if layout == "hp_note":
        head = r.choice([f"HISTORY AND PHYSICAL\nPatient: {nm}  MRN: {mrn}\nDate of service: {date}\nAttending: {clin}, MD",
                         f"H&P - {fac}\n{nm} | {mrn} | DOB {dob}\nSeen {date} by Dr. {clin}"])
        return f"{head}\n\nHPI: {body} {f1}\nExam: {f2}\nAssessment and plan: continue current regimen."
    if layout == "progress_note":
        return (f"Progress note {date}\n{nm} ({mrn})\nS: {body}\nO: {f1} {f2}\n"
                f"A/P: stable; follow up with Dr. {clin} as scheduled.")
    if layout == "intake_form":
        return (f"NEW PATIENT INTAKE - {fac}\nName: {nm}\nDOB: {dob}\nPhone: {phone}\nCity: {loc}\n"
                f"Reason for visit: {c.cc}\nNotes from intake: {body}\nCompleted by: {clin}")
    if layout == "geriatric_assessment":
        return (f"Comprehensive geriatric assessment\nPatient {nm}, MRN {mrn}, seen {date} at {fac}.\n"
                f"Social and functional: {body}\nCognition: {f1}\nMobility: {f2}\nGeriatrician: {clin}")
    if layout == "ed_triage":
        return (f"ED TRIAGE {date}\nPt: {nm}  MRN {mrn}\nChief complaint: {c.cc}\nTriage note: {body} {f1}\n"
                f"Triage RN: {clin}")
    if layout == "nursing_note":
        return f"Nursing note, {date}, {fac}\n{nm}: {body} {f1} {f2}\nRN {clin}"
    if layout == "social_work_note":
        return (f"Social work assessment {date}\nClient: {nm} ({mrn})\nLiving situation: {body}\n"
                f"Resources: home safety review offered. {f1}\nSW: {clin}")
    if layout == "home_health_visit":
        return (f"Home health visit {date}\nPatient {nm} at home in {loc}.\n{body} {f1}\n"
                f"Visit by {clin}, RN; next visit {date2}.")
    if layout == "handoff":
        return f"Handoff {date}\nBed: {nm} ({mrn})\nSituation: {body}\nTo do: {f1}\nFrom {clin} to {clin2}"
    if layout == "discharge_summary":
        return (f"DISCHARGE SUMMARY\n{nm}  MRN {mrn}\nAdmitted {date}; discharged {date2} from {fac}.\n"
                f"Hospital course: {body} {f1}\nDischarging physician: {clin}, MD")
    if layout == "referral_letter":
        return (f"{date}\nDear Dr. {clin2},\nI am referring {nm} (DOB {dob}) for evaluation. {body} {f1}\n"
                f"Please call {phone} with questions.\nSincerely,\n{clin}, MD\n{fac}")
    if layout == "portal_message":
        return (f"Portal message from {nm} to Dr. {clin}, sent {date}\n"
                f"Hi Dr. {N(c.clin_last)}, quick update. {body} Thanks, {N(c.first)}")
    if layout == "preop_assessment":
        return (f"Pre-anesthesia assessment {date}\n{nm}, MRN {mrn}\nProcedure planned at {fac}.\n"
                f"History: {body}\nAirway: {f1}\nAnesthesiologist: {clin}")
    if layout == "pharmacy_consult":
        return (f"Pharmacy consult {date}\nPatient: {nm} ({mrn})\nRequested by: {clin}\n"
                f"Assessment: {body} {f1}\nPharmacist: {clin2}, PharmD")
    if layout == "care_conference_note":
        return (f"Interdisciplinary care conference {date}\nResident: {nm}, {fac}\nAttendees: {clin}, {clin2}\n"
                f"Discussion: {body} {f1}\nNext review {date2}.")
    if layout == "family_history":
        return (f"FAMILY HISTORY - {nm} ({mrn}), updated {date}\n{body}\n"
                f"Reviewed with patient by {clin}.")
    if layout == "pediatric_note":
        return (f"Pediatric clinic note {date}, {fac}\nPatient: {nm}  MRN {mrn}\n"
                f"HPI: {body} {f1}\nPediatrician: {clin}")
    if layout == "well_child_visit":
        return (f"WELL CHILD VISIT {date}\n{nm}, MRN {mrn}\nAccompanied by parent.\n{body}\n"
                f"Development: on track. {f2}\nProvider: {clin}")
    if layout == "nicu_note":
        return (f"NICU daily note {date}\nInfant {nm}, MRN {mrn}\n{body} Temp stable in open crib.\n"
                f"Neonatologist: {clin}")
    if layout == "education_excerpt":
        return (f"Patient education provided {date} to {nm} ({mrn}).\nHandout excerpt: {body}\n"
                f"Teach-back completed. Educator: {clin}")
    if layout == "guideline_excerpt":
        return (f"Clinical decision support note, {date}\nPatient: {nm}  MRN {mrn}\nGuideline summary: {body}\n"
                f"Reviewed by {clin}; discussed at {fac}.")
    if layout == "dictation":
        return (f"dictated by dr {N(c.clin_last.lower())} on {date}, patient {nm}, medical record {mrn}. "
                f"{body} end of dictation.")
    raise KeyError(layout)


SPEAKERS = {
    "clin": ("Dr. {last}", "dr {last}", "CLINICIAN ({last})", "Clinician"),
    "pt": ("Patient", "Pt", "PATIENT"),
    "fam": ("Daughter", "Son", "Family member"),
    "caller": ("Caller", "Parent", "Mother"),
    "interp": ("Interpreter",),
}


def transcript_layout(layout: str, c: Ctx, turns: list[tuple[str, str]]) -> str:
    r = c.r
    styles = {role: r.choice(opts) for role, opts in SPEAKERS.items()}
    if styles["clin"] in ("Clinician",) and r.random() < 0.5:
        styles["clin"] = "Dr. {last}"
    stamps = r.random() < 0.5
    seconds = r.randint(5, 40)

    def label(role):
        style = styles[role]
        return style.replace("{last}", N(c.clin_last)) if "{last}" in style else style

    lines = []
    if layout == "telehealth_transcript":
        lines.append(f"Telehealth visit transcript, {D(c.date)}. Patient: {N(c.name)}.")
        opening = [("clin", "can you hear me okay?"), ("pt", "yes, i can hear you.")]
    elif layout == "phone_triage_transcript":
        lines.append(f"Nurse triage call {D(c.date)}, callback {tag('PHONE', c.phone)}, re: {N(c.name)}")
        opening = [("clin", "triage line, how can i help?")]
    else:
        lines.append(f"Bedside conversation recorded {D(c.date)}, {tag('FACILITY', c.fac)}, patient {N(c.name)}")
        opening = [("clin", "good morning, i'm the doctor on today.")]
    closing = [("clin", r.choice(["okay, let's go over the plan.", "thanks, that helps.", "all right, we'll take care of that."]))]
    for role, utterance in opening + turns + closing:
        prefix = f"[00:{seconds // 60:02d}:{seconds % 60:02d}] " if stamps else ""
        lines.append(f"{prefix}{label(role)}: {utterance}")
        seconds += r.randint(3, 25)
    return "\n".join(lines)


# ---------------------------------------------------------------- row assembly


def fill_decoy(fmt: str, c: Ctx) -> str:
    r = c.r
    values = dict(
        hr=c.hr, sbp=c.sbp, dbp=c.dbp, phq=c.phq, kg=c.kg, gad=c.gad, pain=c.pain, room=c.room, spo2=c.spo2,
        gluc=c.gluc, moca=c.moca, bmi=c.bmi, feet=c.feet, drug=c.drug.capitalize(), mg=c.mg, yrs=c.yrs,
        lb=r.randint(2, 9), grade=r.randint(2, 7), ga=r.randint(28, 37), d=r.randint(5, 10),
        wt=round(r.uniform(3, 12), 1), h=r.randint(2, 4), smallw=words(r.randint(2, 40)), painw=words(r.randint(2, 9)),
    )
    values["painart"] = an(values["painw"])
    return fmt.format(**values)


def make_core(family: str, layout: str, c: Ctx, template_index: int, decoy: bool, extras: dict):
    """Return (kind, core) where kind is 'prose' or 'turns'."""
    r = c.r
    pick = 1 if decoy else 0
    if family == "hypothetical_guideline_ages":
        if layout in TRANSCRIPT_LAYOUTS:
            return "turns", d1_spoken(r), "d1"
        if layout == "dictation":
            core = [s.lower() for s in d1_core(r)]
            return "prose", core, "d1"
        return "prose", d1_core(r), "d1"
    if layout in TRANSCRIPT_LAYOUTS:
        if family == "spoken_ages":
            options = spoken_templates(c)
        elif family == "infant_child_ages":
            options = infant_spoken_templates(c)
        else:
            options = decade_spoken_templates(c)
        idx = template_index % len(options)
        turns = list(options[idx][pick])
        add, (role, fmt), where = r.random() < 0.5, r.choice(SPOKEN_DECOYS), r.random()
        extra = (role, fill_decoy(fmt, c))
        if add and not decoy:
            turns.insert(0 if where < 0.5 else len(turns), extra)
            extras["in_row_decoy"] = True
        return "turns", turns, idx
    if family == "spoken_ages":  # dictation
        options = dictation_templates(c)
        idx = template_index % len(options)
        core = [options[idx][pick]]
        add = r.random() < 0.5
        extra = f"pulse {words(r.randint(56, 99))}, pressure one {words(r.randint(10, 60))} over {words(r.randint(60, 90))}."
        if add and not decoy:
            core.append(extra)
            extras["in_row_decoy"] = True
        return "prose", core, idx
    table = {"uncued_patient_ages": (uncued_templates, UNCUED_DECOYS),
             "relatives_household": (relative_templates, RELATIVE_DECOYS),
             "infant_child_ages": (infant_templates, INFANT_DECOYS),
             "decades": (decade_templates, DECADE_DECOYS)}
    builder, decoy_pool = table[family]
    options = builder(c)
    idx = template_index % len(options)
    core = [options[idx][pick]]
    second = r.random() < 0.35
    other = r.choice([o for i, o in enumerate(options) if i != idx])
    if second and family != "infant_child_ages":  # a second, different template of the same polarity
        core.append(other[pick])
    add_decoy, where = r.random() < 0.6, r.random()
    extra = fill_decoy(r.choice(decoy_pool), c)
    if add_decoy and not decoy:
        core.insert(int(where * (len(core) + 1)), extra)
        extras["in_row_decoy"] = True
    add_hyp = r.random() < 0.2
    hyp, guide, _ = d1_sentences(r)
    extra_hyp = r.choice(hyp + guide)
    if add_hyp and not decoy:
        core.append(extra_hyp)
        extras["in_row_hypothetical"] = True
    return "prose", core, idx


PHENOMENA = {
    "uncued_patient_ages": ("uncued_patient_age", "vital_score_decoy"),
    "relatives_household": ("relative_household_age", "count_duration_decoy"),
    "spoken_ages": ("spoken_age", "spoken_number_decoy"),
    "infant_child_ages": ("infant_child_age", "gestation_dosing_decoy"),
    "decades": ("decade_age", "era_range_decoy"),
    "hypothetical_guideline_ages": (None, "hypothetical_guideline_age"),
}


def case_style(r):
    x = r.random()
    return "normal" if x < 0.6 else ("upper" if x < 0.8 else "lower")


def apply_case(text, spans, style):
    if style == "normal":
        return text, spans
    new = text.upper() if style == "upper" else text.lower()
    if len(new) != len(text):
        raise ValueError("case change altered length")
    return new, [{**s, "text": new[s["start"]:s["end"]]} for s in spans]


def build_row(family, layout, row_id, pair_id, seed_key, template_index, decoy, style, pools):
    r = random.Random(seed_key)
    kind = "child" if family == "infant_child_ages" else "adult"
    c = Ctx(r, pools, kind)
    extras = {}
    core_kind, core, idx = make_core(family, layout, c, template_index, decoy, extras)
    if core_kind == "turns":
        marked = transcript_layout(layout, c, core)
    else:
        marked = prose_layout(layout, c, core)
    text, spans = parse_marked(marked)
    text, spans = apply_case(text, spans, style)
    positive_tag, decoy_tag = PHENOMENA[family]
    phenomena = [positive_tag, "all_ages"] if not decoy else [decoy_tag]
    if family == "hypothetical_guideline_ages":
        phenomena = ["hypothetical_age", "public_age", "durations"]
    if extras.get("in_row_decoy"):
        phenomena.append(decoy_tag)
    if extras.get("in_row_hypothetical"):
        phenomena.append("hypothetical_age")
    if style != "normal":
        phenomena.append("case_variation")
    if decoy and any(s["label"] == "AGE" for s in spans):
        raise AssertionError(f"decoy row {row_id} carries AGE")
    if not decoy and not any(s["label"] == "AGE" for s in spans):
        raise AssertionError(f"positive row {row_id} lacks AGE")
    names = {s["text"].casefold() for s in spans if s["label"] == "NAME"}
    groups = [g for g in c.identities
              if g == c.identity_group or any(v.casefold() in names for v in c.surfaces[g])] if spans else []
    return {
        "id": row_id, "text": text, "spans": spans, "split": "train", "suite": SUITE,
        "annotation_policy": POLICY_VERSION, "cohort": "realistic",
        "family": f"tk23_{family}", "template_family": f"tk23_{family}", "layout": layout,
        "phenomena": list(dict.fromkeys(phenomena)),
        "identity_group": c.identity_group if spans else f"tk23-phi-free-{pair_id}",
        "identity_groups": groups, "pair_id": pair_id,
        "contrast": "decoy" if decoy else "positive", "case_style": style,
        "template_index": idx,
    }


def build(seed: int = SEED):
    global PERSON_POOLS
    pools = Pools()
    PERSON_POOLS = pools
    master = random.Random(seed)
    rows, seen, stats = [], set(), Counter()
    for family, (n_pos, n_dec) in FAMILIES.items():
        layouts = FAMILY_LAYOUTS[family]
        if n_pos:
            with_decoy = set(master.sample(range(n_pos), n_dec))
            plan = [(i, (False, True) if i in with_decoy else (False,)) for i in range(n_pos)]
        else:
            plan = [(i, (True,)) for i in range(n_dec)]
        for i, members in plan:
            layout, style = layouts[i % len(layouts)], case_style(master)
            template, attempt = master.randrange(1000), 0
            pair_id = f"tk23-{family}-{i:04d}"
            while True:
                key = f"{seed}:{family}:{i}:{attempt}"
                pair = [build_row(family, layout, f"{pair_id}-{'d' if decoy else 'p'}", pair_id, key, template,
                                  decoy, style, pools) for decoy in members]
                hits = [excluded_sequences(row["text"]) for row in pair]
                if any(hits):
                    stats["rows_rejected_exclusion"] += sum(1 for h in hits if h)
                    stats["exclusion_sequence_hits"] += sum(hits)
                elif any(row["text"] in seen for row in pair) or len({row["text"] for row in pair}) < len(pair):
                    stats["rows_rejected_duplicate"] += 1
                else:
                    break
                attempt += 1
                template = master.randrange(1000)
            for row in pair:
                seen.add(row["text"])
                rows.append(row)
    return rows, pools, stats


# ---------------------------------------------------------------- manifest and report helpers


def age_span_value(text: str, span: dict):
    """(value, unit) for an AGE span: unit in years, months, weeks, days or decade."""
    value = span["text"].lower()
    after = text[span["end"]:span["end"] + 12].lower()
    if value in DECADE_VALUES:
        return DECADE_VALUES[value], "decade"
    number = int(value) if value.isdigit() else WORD_VALUES.get(value)
    if number is None:
        return None, None
    unit_match = re.match(r"[\s-]*(weeks?|wk|months?|mo\b|month|days?)", after)
    unit = "years"
    if unit_match:
        word = unit_match[1]
        unit = "weeks" if word.startswith("w") else "months" if word.startswith("mo") else "days"
    return number, unit


def support(rows):
    counts = Counter(s["label"] for row in rows for s in row["spans"])
    return {label: counts[label] for label in ENTITY_TYPES}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bank(rows, pools, stats, seed, output: Path = OUTPUT):
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    label_support = support(rows)
    policy_files = [
        "docs/annotation-policy-v2.md", "docs/age-contexts-bank-brief.md",
        "src/outis/data/annotation_policy.py", "src/outis/data/annotation_policy_ages.py",
        "src/outis/types.py", "src/outis/data/generators/places.py",
        "data/places_exclusions.sha256.json",
    ]
    manifest = {
        "corpus": f"data/{output.name}",
        "suite": SUITE,
        "annotation_policy": POLICY_VERSION,
        "annotation_policy_decisions": ["all_ages_2026-10-06", "D1_hypothetical_age_not_AGE_2026-10-07",
                                         "D2_decade_word_is_AGE_2026-10-07"],
        "split": "train",
        "role": "train",
        "independent_final": False,
        "rows": len(rows),
        "documents": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "label_support": label_support,
        "entity_support": label_support,
        "label_counts": label_support,
        "cohort_counts": dict(Counter(row["cohort"] for row in rows)),
        "family_counts": dict(Counter(row["family"] for row in rows)),
        "layout_counts": dict(sorted(Counter(row["layout"] for row in rows).items())),
        "contrast_counts": dict(Counter(row["contrast"] for row in rows)),
        "case_counts": dict(Counter(row["case_style"] for row in rows)),
        "phi_free_rows": sum(1 for row in rows if not row["spans"]),
        "real_place_values": 0,
        "exclusion_filter": {
            "list": "data/places_exclusions.sha256.json (hash-only, via places.exclusion_hash/excluded_sequences/excluded_value)",
            "pool_values_considered": pools.considered,
            "pool_values_dropped": dict(pools.dropped),
            "rows_rejected_exclusion": stats.get("rows_rejected_exclusion", 0),
            "exclusion_sequence_hits_in_rejected_rows": stats.get("exclusion_sequence_hits", 0),
            "rows_rejected_duplicate": stats.get("rows_rejected_duplicate", 0),
            "final_rows_with_excluded_sequences": sum(1 for row in rows if excluded_sequences(row["text"])),
        },
        "synthetic_schemes": {"MRN": MRN_SCHEME, "PHONE": PHONE_SCHEME,
                              "names_places_facilities": "invented from syllable and first-name pools"},
        "author_provenance": {
            "author": "isolated synthetic-data author agent (tk23)",
            "brief": "docs/age-contexts-bank-brief.md",
            "generator": "scripts/takeover/author23_age_contexts.py",
            "generator_sha256": sha256_file(Path(__file__).resolve()),
            "seed": seed,
            "policy_sources_sha256": {p: sha256_file(ROOT / p) for p in policy_files if (ROOT / p).exists()},
            "inputs_read": "brief, annotation policy and places addendum, types.py, annotation_policy fixture modules, "
                           "policy.age_value, validate_bank.py, AGENTS.md, CLEANUP.md, places exclusion helpers; "
                           "no corpus, bank, report, model or prediction was read",
        },
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    rows, pools, stats = build(args.seed)
    for row in rows:
        errors = annotation_errors(row["text"], row["spans"])
        if errors:
            raise SystemExit(f"{row['id']}: {errors}")
    manifest = write_bank(rows, pools, stats, args.seed, Path(args.output))
    print(json.dumps({k: manifest[k] for k in ("rows", "sha256", "label_support", "contrast_counts",
                                                 "exclusion_filter")}, indent=1))


if __name__ == "__main__":
    main()
