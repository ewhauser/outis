"""Original clinical name contrasts: Apache-2.0 code and CC0-1.0 data.

No patient records, external datasets, retired generator, or qualification corpus
is used. Matched notes place the same authored value under person-identification
proof or under a clinical/public interpretation. Development has independently
authored frames and disjoint private value grammars.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import re
import unicodedata

import tiktoken

from . import generalization_synthetic as shared
from .types import ENTITY_TYPES

VERSION = "1.0"
SEED = 930174
COLLISIONS = {
    "train": (
        "We",
        "He",
        "For",
        "An",
        "The",
        "No",
        "Na",
        "I",
        "A",
        "To",
        "On",
        "Do",
        "In",
        "At",
        "Be",
        "Will",
        "May",
    ),
    "validation": (
        "You",
        "Me",
        "Us",
        "By",
        "If",
        "Or",
        "And",
        "But",
        "Not",
        "So",
        "As",
        "Of",
        "From",
        "Is",
        "It",
        "Go",
    ),
}
CONSONANTS = {
    "train": ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w"),
    "validation": ("c", "h", "j", "q", "x", "y", "z", "sh", "ch", "th", "zh"),
}
VOWELS = ("a", "e", "i", "o", "u", "ai", "ie", "ia")
LEADING_VOWELS = (
    "A",
    "E",
    "I",
    "O",
    "U",
    "Á",
    "É",
    "Í",
    "Ó",
    "Ú",
    "Å",
    "Ö",
    "Æ",
    "Ø",
    "Ü",
    "Ê",
    "Ē",
    "Ā",
    "Ī",
    "Œ",
)
ACCENTED_VOWELS = ("Á", "É", "Í", "Ó", "Ú", "Å", "Ö", "Ü", "Ê", "Ē", "Ā", "Ī")
ROLES = ("patient", "clinician", "nurse", "caregiver")
PREDICATES = {
    "patient": (
        "reported easier movement after rest",
        "described the course of symptoms",
        "answered the review questions",
        "demonstrated the seated exercise",
        "confirmed the regular medicine schedule",
        "explained how symptoms changed during the week",
    ),
    "clinician": (
        "reviewed the examination findings",
        "explained the monitoring plan",
        "checked the treatment list",
        "documented the clinical assessment",
        "discussed the follow-up arrangements",
        "confirmed the proposed management",
    ),
    "nurse": (
        "reviewed the daily observations",
        "explained safe transfer technique",
        "checked treatment handling",
        "completed the nursing assessment",
        "discussed the home-care plan",
        "confirmed the observation schedule",
    ),
    "caregiver": (
        "described the usual support arrangements",
        "confirmed the home medicine schedule",
        "explained the recent symptom pattern",
        "answered questions about daily function",
        "demonstrated the agreed support technique",
        "described changes noticed during recovery",
    ),
}
TITLE = {
    "patient": "Patient",
    "clinician": "Dr.",
    "nurse": "Nurse",
    "caregiver": "Caregiver",
}
LAYOUTS = ("prose", "sections", "bullets", "numbered", "worksheet", "compact")
HEADERS = {
    "train": (
        "Care discussion concerning {topic}",
        "Interval clinical observations: {topic}",
        "Treatment review for {topic}",
        "Recorded progress regarding {topic}",
        "Symptom reassessment — {topic}",
        "Routine examination of {topic}",
        "Recovery monitoring: {topic}",
        "Current clinical findings on {topic}",
    ),
    "validation": (
        "Review record about {topic}",
        "Clinical follow-up centred on {topic}",
        "Assessment notes — {topic}",
        "Observed changes in {topic}",
        "Continuity discussion of {topic}",
        "Clinical progress account: {topic}",
    ),
}
TOPICS = (
    "walking comfort",
    "breathing effort",
    "treatment tolerance",
    "daytime function",
    "symptom progression",
    "self-care",
    "hydration",
    "activity pacing",
    "joint movement",
    "support arrangements",
    "rest and recovery",
    "medicine handling",
)
CAPTIONS = {
    "train": (
        "Verified full name of the {role}: {v}.",
        "Name stated aloud by the {role} = {v}.",
        "Personal identity, {role} confirmation: {v}.",
        "The {role}'s given and family name is {v}.",
        "Human identity on the {role}'s signed form: {v}.",
        "Confirmed personal name for the {role} | {v}.",
        "The {role} supplied the identification name {v}.",
        "NAME OF {role_upper}, CHECKED AT INTERVIEW: {v}.",
        "The stated name attached to the {role}'s identity is {v}.",
        "Private name entry, verified with the {role}: '{v}'.",
        "Identity verification / {role} / full name: {v}.",
        "The {role}'s documented naming entry reads {v}.",
        "The name written by the {role} on the identity form was {v}.",
        "Full-name confirmation from the {role}: [{v}].",
        "Signed personal-name field for the {role} >> {v}.",
        "For the {role}'s identity check, the supplied name was {v}.",
    ),
    "validation": (
        "The {role} verified the personal name as {v}.",
        "Recorded identity name belonging to the {role}: {v}.",
        "The interviewer's naming check with the {role} gives {v}.",
        "Under personal identification, the {role} entered '{v}'.",
        "The {role} provided {v} when asked for their name.",
        "NAME CONFIRMED BY {role_upper} — {v}.",
        "Personal name, checked against the {role}'s consent form: {v}.",
        "Identification of the {role}: full name = {v}.",
        "The {role}'s signed identity statement names {v}.",
        "Naming details read back to the {role}: [{v}].",
        "The {role}'s personal-name box contains {v}.",
        "Identity attestation completed by the {role}; name: {v}.",
    ),
}
HUMAN_FRAMES = {
    "train": (
        "The {role}, named {v}, {verb}.",
        "{title} {v} {verb}.",
        "The {role}'s completed personal signature reads {v}.",
        "'{v}' was the name stated when the {role} verified their identity.",
        "For identity verification, the {role} gave {v} as their full name.",
        "{v}'s identity was confirmed against the signed form; they are the {role} in this record.",
        "The person named {v}, acting as the {role}, {verb}.",
        "After introducing themself as {v}, the {role} {verb}.",
        "During care planning, {v}, the {role}, {verb}.",
        "The {role} signing this record identified themself as {v}.",
        "The named {role} is {v}; they {verb}.",
        "This history was checked with {v}, whose role was explicitly recorded as {role}.",
        "Following an identity check, {v} {verb} in their role as {role}.",
        "The interview was completed with {v}, the identified {role}.",
        "{v}, introduced as the {role}, {verb}.",
        "The {role} wrote their personal name, {v}, before the discussion.",
        "When asked who was speaking, the {role} answered with the name {v}.",
        "The {role}'s identity statement gives {v} as the person who {verb}.",
        "The person acting as {role} was introduced by the name {v}.",
        "Under the personal signature, the {role} has written {v}.",
        "The {role} confirmed: 'My name is {v}'.",
        "The {role}'s verified name, {v}, was read back before the assessment.",
        "{title} {v} personally confirmed the discussion of care.",
        "As the identified {role}, {v} {verb}.",
        "The discussion notes identify {v} as the {role} giving this account.",
        "A personal introduction established {v} as the {role}.",
        "The {role} bearing the name {v} {verb}.",
        "Personal identity was checked with the {role}, who gave the name {v}.",
        "The statement signed by {v} was supplied by the identified {role}.",
        "The {role}'s name for this record is {v}; the identity was confirmed.",
        "The {role} introduced themself: 'I am {v}'.",
        "The person recorded as {role}, {v}, {verb}.",
    ),
    "validation": (
        "The identity discussion established that the {role} is named {v}.",
        "{title} {v} was the person who {verb} during this discussion.",
        "The {role} authenticated the signature bearing the personal name {v}.",
        "In their capacity as {role}, the person introduced themself as {v}.",
        "{v}, identified personally as the {role}, {verb}.",
        "A naming check confirmed {v} as the {role} involved in the discussion.",
        "The {role} read their personal signature aloud as '{v}'.",
        "The person who {verb} was {v}, acting as the {role}.",
        "Identification was completed for {v}, whose stated role is {role}.",
        "The {role} gave their personal introduction using the name {v}.",
        "The {role}'s verified identity was recorded under {v}.",
        "'{v}' identifies the {role} who personally confirmed the account.",
        "The {role} added the name {v} to their signed identity statement.",
        "For the naming check, the {role} stated 'I am called {v}'.",
        "{v}'s personal identity was verified while they acted as the {role}.",
        "The discussion was acknowledged personally by {title} {v}.",
        "The {role} who {verb} supplied their name as {v}.",
        "The identity attestation belongs to {v}, the person acting as {role}.",
        "The personal-name entry for the {role} was confirmed directly with {v}.",
        "Speaking as the {role}, the named individual {v} {verb}.",
        "The {role}'s personal attestation was signed '{v}'.",
        "The identity check identifies the {role} as {v}.",
        "The person introduced as {v} confirmed their role as {role}.",
        "Personal identification for {title} {v} was confirmed before discussion.",
    ),
}
CLINICAL_FUNCTIONS = {
    "a": "{v} gentle breathing exercise was demonstrated by the therapist.",
    "an": "{v} additional seated check was performed before mobilisation.",
    "the": "{v} monitoring plan remains consistent with the current examination.",
    "for": "{v} symptom monitoring, a daily diary was suggested.",
    "he": "{v} reported easier breathing after a short period of rest.",
    "we": "{v} reviewed treatment handling and clarified the plan for follow-up.",
    "no": "{v} acute chest discomfort was reported during the examination.",
    "na": "{v} 139 mmol/L and K 4.1 mmol/L were within the reference range.",
    "i": "{v} discussed the findings with the care team.",
    "it": "{v} was agreed that gradual mobilisation would be appropriate.",
    "you": "'{v} can repeat the seated movement if it remains comfortable' was the instruction.",
    "me": "The request was phrased as 'help {v} practise the transfer again'.",
    "us": "The caregiver asked staff to 'show {v} the exercise again'.",
    "to": "{v} avoid fatigue, the task was divided into short periods.",
    "be": "'{v} cautious with abrupt standing while dizziness persists' was the advice.",
    "do": "'{v} repeat the inhaler demonstration after the explanation' was the instruction.",
    "by": "{v} the end of the session, the transfer was more comfortable.",
    "in": "{v} a seated position, breathing effort remained even.",
    "on": "{v} examination, hand power was symmetrical.",
    "at": "{v} rest, respiratory effort showed no extra work.",
    "as": "{v} recovery progressed, walking distance was increased gradually.",
    "if": "{v} symptoms recur, pause activity and review hydration.",
    "or": "Either use a short seated task {v} allow a further rest break.",
    "and": "The plan covers hydration {v} gradual activity.",
    "but": "Comfort improved {v} movement still needs supervision.",
    "not": "The current findings are {v} consistent with a new fever.",
    "so": "The task was tolerated, {v} the agreed exercise may continue.",
    "may": "A short seated rest {v} help before the next mobility task.",
    "will": "The symptom diary {v} support review of changes.",
    "is": "The current regimen {v} tolerated without new nausea.",
    "from": "Symptoms improved after rest away {v} prolonged standing.",
    "of": "The review concerns the timing {v} regular medication.",
    "go": "'Only {v} as far as remains comfortable' was the activity advice.",
}
PUBLIC_FRAMES = (
    "The public product card uses '{v}' as a catalogue display label for an airway spacer.",
    "An equipment catalogue prints {v} as the name of a demonstration device series.",
    "The public medicine inventory contains the product label {v}; this entry describes a formulation.",
    "The teaching worksheet calls the demonstration test '{v}' and records no personal identity.",
    "The reference handout has the generic exercise heading {v}.",
    "For the language task, the printed card contains the practice words '{v}'.",
    "The manufacturer sheet lists {v} under public model labels, rather than individual assignments.",
    "A public clinical glossary uses {v} as an example label in its terminology table.",
    "The nonpersonal device catalogue has a product-name entry reading '{v}'.",
    "A generic screening worksheet labels its demonstration sequence {v}.",
    "The teaching software's public menu displays {v} as a procedure category.",
    "The printed medicine catalogue gives the formulation heading as '{v}'.",
    "The exercise diagram bears the public demonstration title {v}.",
    "Under reference terminology, the public study handout includes the phrase '{v}'.",
    "The sample rehabilitation chart uses {v} as an instructional example heading.",
    "The public equipment specification calls the sample controller series {v}.",
)
TREATMENTS = (
    (
        "blood-pressure monitoring",
        "amlodipine 5 mg daily",
        "Seated pressure checks and treatment tolerance were discussed.",
    ),
    (
        "stable airway symptoms",
        "inhaled budesonide 180 mcg twice daily",
        "Device handling was reviewed and breathing remained comfortable at rest.",
    ),
    (
        "joint discomfort",
        "paracetamol 500 mg as needed",
        "Movement remains possible and the skin has no new breakdown.",
    ),
    (
        "reflux symptoms",
        "omeprazole 20 mg daily",
        "Meal timing was reviewed and there is no swallowing difficulty.",
    ),
    (
        "glucose monitoring",
        "metformin 500 mg twice daily",
        "Food intake and treatment tolerance remain satisfactory.",
    ),
    (
        "thyroid replacement",
        "levothyroxine 75 mcg daily",
        "Appetite is stable and no new tremor was observed.",
    ),
)
PUBLIC_FACTS = (
    "The teaching booklet's edition year is {year}; this is bibliographic metadata.",
    "A public practice guideline published in {year} was consulted for background terminology.",
    "The equipment manual has revision date {year}-05-12, which belongs to the public manual.",
    "The public medicine information sheet was revised in June {year}.",
    "A printed reference chart issued during {year} illustrates ordinary clinical measurements.",
    "The archive entry identifies {year} as the publication year of the public examination guide.",
    "The public teaching slide carries a copyright year of {year}.",
    "The manufacturer catalogue edition is dated September {year}, independent of this patient's events.",
    "Research references include public trial NCT{digits} and publication PMID {digits}.",
    "A public device catalogue uses product ref MODEL-{short} and demonstration lot REF-{short}.",
    "The pharmacy product table lists NDC 53120-442-08 as a formulation code, without personal linkage.",
    "The public study protocol identifier is STUDY-{digits}, not an assigned patient record number.",
    "Routine teaching thresholds show sodium 135-145 mmol/L and potassium 3.5-5.0 mmol/L.",
    "Reference intervals on the public handout include temperature 36.0-37.5 C and pulse 60-100/min.",
    "A medicine demonstration uses a 500 mg tablet and a 20 mg capsule to explain dose units.",
    "Observation for 100 days and a 95-minute practice interval describe duration, not chronological age.",
    "The teaching activity lasts one hundred and eleven minutes; those words describe elapsed time.",
    "Score examples use pain 3/10, power 5/5 and screening 28/30 without identifying an individual.",
    "The public handout's reference year, {year}, is printed beside its edition number.",
    "Guideline revision history lists {year} as the year of the published update.",
    "An illustrative device calibration sheet was published on {year}-03-09.",
    "The public literature index assigns {year} to this teaching article's publication.",
    "The supplied background material describes a workshop manual released in December {year}.",
    "A public screening reference dated April {year} explains how measurement scales are reported.",
)


def _word(rng, split, *, vowel_first=True, accented=False):
    if vowel_first:
        value = rng.choice(ACCENTED_VOWELS if accented else LEADING_VOWELS)
    else:
        value = rng.choice(CONSONANTS[split]).capitalize() + rng.choice(VOWELS)
    for _ in range(rng.choice((2, 3))):
        value += rng.choice(CONSONANTS[split]) + rng.choice(VOWELS)
    return value + rng.choice(("", "n", "r", "s", "l"))


def name_value(rng, split, pair):
    style = pair % 12
    first, surname = _word(rng, split), _word(rng, split, vowel_first=False)
    if style < 4:
        value = COLLISIONS[split][(pair // 12 + style * 3) % len(COLLISIONS[split])]
        case = ("title", "lower", "upper", "lower")[style]
        value = (
            value.lower()
            if case == "lower"
            else value.upper()
            if case == "upper"
            else value
        )
        return value, f"collision-{case}", case
    if style == 4:
        return f"{first} {surname}".lower(), "lowercase-full", "lower"
    if style == 5:
        return f"{first} {surname}", "vowel-leading-full", "title"
    if style == 6:
        first = unicodedata.normalize("NFD", _word(rng, split, accented=True))
        return f"{first} {surname}", "nfd-leading-full", "title"
    if style == 7:
        initial = unicodedata.normalize("NFD", rng.choice(ACCENTED_VOWELS))
        return f"{first} {initial}. {surname}", "unicode-middle-initial", "title"
    if style == 8:
        initial = rng.choice("AEIOUBDFKLMNPRSTVW")
        return (
            f"{first} {initial}. {surname}".lower(),
            "lowercase-middle-initial",
            "lower",
        )
    if style == 9:
        return (
            f"{first} {surname} {rng.choice(('Jr.', 'Sr.', 'II', 'III'))}",
            "suffix-full",
            "title",
        )
    if style == 10:
        return (
            f"{first} {surname} {rng.choice(('Jr.', 'Sr.', 'II', 'III'))}".lower(),
            "lowercase-suffix",
            "lower",
        )
    first = unicodedata.normalize("NFD", _word(rng, split, accented=True))
    return f"{surname}, {first}", "reversed-unicode", "title"


def clinical_body(rng, split, pair):
    condition, medicine, finding = TREATMENTS[pair % len(TREATMENTS)]
    facts = {
        "condition": condition,
        "medicine": medicine,
        "finding": finding,
        "age": rng.randrange(42, 85),
        "sbp": rng.randrange(112, 147),
        "dbp": rng.randrange(64, 87),
        "pulse": rng.randrange(60, 91),
        "oxygen": rng.randrange(95, 100),
        "sodium": rng.randrange(136, 143),
        "glucose": rng.randrange(87, 165),
        "duration": rng.randrange(90, 120),
        "pain": rng.randrange(0, 5),
        "score": rng.randrange(25, 31),
    }
    frames = {
        "train": (
            "Clinical review concerns {condition} in an adult aged {age} years. The current regimen is {medicine}. BP {sbp}/{dbp} mmHg; pulse {pulse}/min; oxygen saturation {oxygen}%. Sodium {sodium} mmol/L and glucose {glucose} mg/dL were reviewed. {finding} Pain {pain}/10, power 5/5 and screening {score}/30 remain clinical findings. Monitoring extends over {duration} days.",
            "Assessment: {condition}. Present treatment — {medicine}. The adult is {age} years old. Seated BP was {sbp}/{dbp} mmHg with pulse {pulse}/min and saturation {oxygen}%. Routine results include Na {sodium} mmol/L and glucose {glucose} mg/dL. {finding} Duration of observation is {duration} days; pain score {pain}/10, motor power 5/5 and cognition {score}/30 were discussed.",
        ),
        "validation": (
            "The clinical issue is {condition}; regular medication is {medicine}. Age {age} years remains relevant clinical information. Current measurements show BP {sbp}/{dbp} mmHg, pulse {pulse}/min and SpO2 {oxygen}%. Laboratory review found sodium {sodium} mmol/L and glucose {glucose} mg/dL. {finding} Function was recorded as power 5/5, pain {pain}/10 and screening {score}/30. The planned observation interval lasts {duration} days.",
            "Review findings concern {condition} and tolerance of {medicine}. The assessed adult is {age} years old. Vitals: {sbp}/{dbp} mmHg, pulse {pulse}/min, saturation {oxygen}%. Sodium is {sodium} mmol/L; glucose is {glucose} mg/dL. {finding} Clinical scoring remains {pain}/10 for pain, 5/5 for strength and {score}/30 for screening. Follow-up observation covers {duration} days.",
        ),
    }
    return rng.choice(frames[split]).format(**facts)


def _join(note, layout, ordinal):
    separators = {
        "prose": " ",
        "sections": "\nVerification:\n",
        "bullets": "\n- ",
        "numbered": f"\n{ordinal}. ",
        "worksheet": "\n    ",
        "compact": " | ",
    }
    note.plain(separators[layout])


def generate_pair(pair, *, split="train", seed=SEED, attempt=0):
    if split not in COLLISIONS:
        raise ValueError("Split must be train or validation")
    if not isinstance(pair, int) or isinstance(pair, bool) or pair < 0:
        raise ValueError("Pair must be a nonnegative integer")
    rng = random.Random(
        (seed << 32) + pair * 90149 + (0 if split == "train" else 77171) + attempt
    )
    value, style, case = name_value(rng, split, pair)
    role = rng.choice(ROLES)
    arguments = {
        "role": role,
        "role_upper": role.upper(),
        "title": TITLE[role],
        "verb": rng.choice(PREDICATES[role]),
    }
    layout = rng.choice(LAYOUTS)
    header = rng.choice(HEADERS[split]).format(topic=rng.choice(TOPICS))
    if rng.random() < 0.2:
        header = header.upper()
    body = clinical_body(rng, split, pair)
    episode = date(2012, 1, 1) + timedelta(days=rng.randrange(4700))
    # Month/day publication metadata must also precede an early-year episode.
    publication = rng.randrange(1998, min(2021, episode.year - 1) + 1)
    fact = rng.randrange(len(PUBLIC_FACTS))
    reference = PUBLIC_FACTS[fact].format(
        year=publication,
        digits=rng.randrange(10000000, 99999999),
        short=rng.randrange(10000, 99999),
    )
    reference += " Teaching examples discuss Addison disease, Romberg testing and Graves disease as clinical terminology."
    positive, negative = shared.Note(), shared.Note()
    for note in (positive, negative):
        note.plain(header + "\n" + body)
    contexts = []
    if pair % 5:
        caption = rng.randrange(len(CAPTIONS[split]))
        _join(positive, layout, 1)
        positive.clause(CAPTIONS[split][caption], "NAME", value, **arguments)
        contexts.append(f"caption-{caption}")
    else:
        frame = rng.randrange(len(HUMAN_FRAMES[split]))
        _join(positive, layout, 1)
        positive.clause(HUMAN_FRAMES[split][frame], "NAME", value, **arguments)
        contexts.append(f"human-{frame}")
    for ordinal in (2, 3):
        frame = rng.randrange(len(HUMAN_FRAMES[split]))
        _join(positive, layout, ordinal)
        if style in {"suffix-full", "lowercase-suffix"} and ordinal == 3:
            positive.clause(
                "{v}'s personal identity was verified; this individual is the {role} who {verb}.",
                "NAME",
                value,
                **arguments,
            )
            contexts.append("suffix-possessive")
        else:
            positive.clause(HUMAN_FRAMES[split][frame], "NAME", value, **arguments)
            contexts.append(f"human-{frame}")
    public = rng.randrange(len(PUBLIC_FRAMES))
    _join(negative, layout, 1)
    if style.startswith("collision-"):
        if pair % 3 == 0:
            # A care-team section heading is not a same-line personal title.
            negative.plain(rng.choice(("Nurse", "Clinician", "Caregiver")) + "\n")
        negative.plain(CLINICAL_FUNCTIONS[value.casefold()].format(v=value))
    else:
        negative.plain(PUBLIC_FRAMES[public].format(v=value))
    _join(negative, layout, 2)
    negative.plain(PUBLIC_FRAMES[(public + 5) % len(PUBLIC_FRAMES)].format(v=value))
    # Public value mentions also appear beside genuine person mentions, but
    # only in clearly nonpersonal catalogue/teaching contexts.
    if pair % 4 == 0:
        _join(positive, layout, 4)
        positive.plain(PUBLIC_FRAMES[(public + 9) % len(PUBLIC_FRAMES)].format(v=value))
    if pair % 4 == 1:
        positive.plain("\n")
        positive.clause(
            "This patient's assessment took place on {v}.", "DATE", episode.isoformat()
        )
        positive.plain(" ")
        positive.clause(
            "The next clinical review is arranged for {v}.",
            "DATE",
            (episode + timedelta(days=28)).isoformat(),
        )
    if pair % 7 == 2:
        positive.plain("\n")
        locality = _word(rng, split, accented=True) + " Vale"
        positive.clause(
            "The {role}'s usual home town is {v}.", "LOCATION", locality, **arguments
        )
    for note in (positive, negative):
        note.plain("\n" + reference)
    rows = []
    for kind, note in (("person", positive), ("clinical", negative)):
        row = {
            "id": f"name-contrastive-{split}-{seed}-{pair:05d}-{kind}",
            "pair_id": f"name-contrastive-{split}-{seed}-{pair:05d}",
            "text": note.text,
            "spans": note.spans,
            "split": split,
            "suite": "authored_name_contrastive_training"
            if split == "train"
            else "authored_name_contrastive_development",
            "family": f"name_contrastive.{split}.{layout}.{kind}",
            "negative": kind == "clinical",
            "authored_name_style": style,
            "authored_case_style": case,
            "person_role": role,
            "authored_contexts": contexts
            if kind == "person"
            else [f"clinical-{public}"],
            "clinical_episode_year": episode.year,
            "publication_year": publication,
            "public_fact_context": fact,
        }
        validate_row(row)
        rows.append(row)
    return rows


def validate_row(row):
    if row.get("split") not in {"train", "validation"}:
        raise ValueError("Rows must be authored training or development")
    expected = (
        "authored_name_contrastive_training"
        if row["split"] == "train"
        else "authored_name_contrastive_development"
    )
    if row.get("suite") != expected:
        raise ValueError("Row suite does not match its declared role")
    end = 0
    for span in row["spans"]:
        if (
            not 0 <= span["start"] < span["end"] <= len(row["text"])
            or span["start"] < end
            or span["label"] not in ENTITY_TYPES
        ):
            raise ValueError("Gold offsets must be ordered, disjoint and valid")
        end = span["end"]
    if row["negative"] != (not row["spans"]):
        raise ValueError("Contrast metadata and annotations disagree")
    if row["publication_year"] >= row["clinical_episode_year"]:
        raise ValueError("Reference publication cannot follow the clinical episode")


def _canonical(value):
    return unicodedata.normalize("NFC", value).casefold()


def write_dataset(
    output="data",
    train_count=4000,
    val_count=1000,
    seed=SEED,
    *,
    compare_corpora=(),
    max_tokens=512,
):
    for count in (train_count, val_count):
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 2
            or count % 2
        ):
            raise ValueError(
                "Each split requires a positive even note count of at least two"
            )
    if not isinstance(max_tokens, int) or max_tokens < 100:
        raise ValueError("Token ceiling must be an integer at least 100")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = {
        split: output
        / f"name_contrastive_{'train' if split == 'train' else 'validation'}.jsonl"
        for split in ("train", "validation")
    }
    for path in names.values():
        if path.exists() or path.with_suffix(".jsonl.manifest.json").exists():
            raise FileExistsError(
                "Authored corpora are immutable; choose a new directory"
            )
    comparison = shared.comparison_hashes(compare_corpora)
    forbidden = set().union(*(record["text_hashes"] for record in comparison))
    forbidden_ids = set()
    for record in comparison:
        # The shared helper preflights every declaration before payload access.
        # Bind the ID audit to precisely those approved, already hashed bytes.
        payload = Path(record["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError("Approved comparison corpus changed during ID audit")
        forbidden_ids.update(
            json.loads(line)["id"]
            for line in payload.decode("utf-8").splitlines()
            if line.strip()
        )
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(".cache/tiktoken").resolve()))
    encoding = tiktoken.get_encoding("o200k_base")
    seen_text, seen_ids, train_private = set(), set(), set()
    manifests = {}
    temporary = []
    try:
        for split, count in (("train", train_count), ("validation", val_count)):
            path = names[split]
            draft = path.with_suffix(".jsonl.tmp")
            temporary.append(draft)
            labels, tags, styles, cases, roles, contexts, name_traits = (
                Counter() for _ in range(7)
            )
            public_facts = Counter()
            lengths, private, retries = [], set(), 0
            with draft.open("x", encoding="utf-8") as handle:
                for pair in range(count // 2):
                    for attempt in range(40):
                        rows = generate_pair(
                            pair, split=split, seed=seed, attempt=attempt
                        )
                        stats = [shared.token_audit(row, encoding) for row in rows]
                        row_private = {
                            _canonical(row["text"][s["start"] : s["end"]])
                            for row in rows
                            for s in row["spans"]
                            if s["label"] in {"NAME", "LOCATION"}
                        }
                        if max(n for n, _ in stats) <= max_tokens and not (
                            split == "validation" and row_private & train_private
                        ):
                            break
                    else:
                        raise ValueError(
                            "Unable to author a short, private-value-disjoint contrast pair"
                        )
                    retries += attempt
                    private.update(row_private)
                    for row, (length, row_tags) in zip(rows, stats):
                        digest = hashlib.sha256(row["text"].encode()).hexdigest()
                        if (
                            digest in seen_text
                            or digest in forbidden
                            or row["id"] in seen_ids
                            or row["id"] in forbidden_ids
                        ):
                            raise ValueError(
                                "Authored text/ID overlaps this split or a declared corpus"
                            )
                        seen_text.add(digest)
                        seen_ids.add(row["id"])
                        labels.update(s["label"] for s in row["spans"])
                        tags.update(row_tags)
                        lengths.append(length)
                        styles[row["authored_name_style"]] += 1
                        cases[row["authored_case_style"]] += 1
                        roles[row["person_role"]] += 1
                        public_facts[str(row["public_fact_context"])] += 1
                        if not row["negative"]:
                            contexts.update(row["authored_contexts"])
                        for span in row["spans"]:
                            if span["label"] != "NAME":
                                continue
                            value = row["text"][span["start"] : span["end"]]
                            name_traits["total"] += 1
                            name_traits["fully_lowercase"] += value.islower()
                            name_traits["all_uppercase"] += value.isupper()
                            name_traits["non_ascii"] += not value.isascii()
                            name_traits["nfd"] += value != unicodedata.normalize(
                                "NFC", value
                            )
                            name_traits["leading_vowel"] += (
                                unicodedata.normalize("NFD", value)[0].casefold()
                                in "aeiou"
                            )
                            name_traits["leading_combining_grapheme"] += len(
                                value
                            ) > 1 and unicodedata.category(value[1]).startswith("M")
                            name_traits["middle_initial"] += bool(
                                re.search(r"\S+\s+[^\s]+\.\s+\S+", value)
                            )
                            name_traits["generational_suffix"] += bool(
                                re.search(r"\b(?:jr\.?|sr\.?|ii|iii)$", value, re.I)
                            )
                            name_traits["possessive_following"] += row["text"][
                                span["end"] : span["end"] + 2
                            ] in {"'s", "’s"}
                            name_traits["collision_value"] += _canonical(value) in {
                                _canonical(v) for v in COLLISIONS[split]
                            }
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if split == "train":
                train_private = private
            order = sorted(lengths)
            manifest = {
                "schema_version": 1,
                "generator": "clinical_deid.name_contrastive_synthetic",
                "generator_version": VERSION,
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "shared_annotation_audit_helper_sha256": hashlib.sha256(
                    Path(shared.__file__).read_bytes()
                ).hexdigest(),
                "seed": seed,
                "rows": count,
                "sha256": hashlib.sha256(draft.read_bytes()).hexdigest(),
                "split": split,
                "suite": "authored_name_contrastive_training"
                if split == "train"
                else "authored_name_contrastive_development",
                "license": "CC0-1.0",
                "code_license": "Apache-2.0",
                "external_datasets": [],
                "source": "Original authored clinical/person/public grammar and productive fictional identities; no retired sentences, identities, generator source, patient records or external dataset.",
                "independent_final": False,
                "used_for_training": split == "train",
                "used_for_checkpoint_selection": split == "validation",
                "development_scope": "Newly authored development only; prior cohorts remain unchanged and must be reported separately.",
                "contrast_pairs": count // 2,
                "phi_free_notes": count // 2,
                "label_support": {label: labels[label] for label in ENTITY_TYPES},
                "bioes_tag_support": dict(sorted(tags.items())),
                "name_style_note_support": dict(sorted(styles.items())),
                "case_style_note_support": dict(sorted(cases.items())),
                "person_role_note_support": dict(sorted(roles.items())),
                "person_context_span_support": dict(sorted(contexts.items())),
                "public_fact_note_support": dict(sorted(public_facts.items())),
                "name_span_traits": dict(sorted(name_traits.items())),
                "tokenizer": "o200k_base / OPF original Unicode character projection",
                "token_length": {
                    "min": min(lengths),
                    "mean": sum(lengths) / len(lengths),
                    "p95": order[int(0.95 * (len(order) - 1))],
                    "max": max(lengths),
                    "ceiling": max_tokens,
                },
                "annotation_audit": {
                    "rows_validated": count,
                    "token_aligned_rows": count,
                    "gold_overlap": 0,
                    "shared_phi_tokens": 0,
                    "duplicate_texts": 0,
                    "length_or_private_value_retries": retries,
                },
                "private_value_policy": "Train/development NAME and LOCATION values are disjoint after NFC/casefold normalization, including collision words. Clinical vocabulary and public reference dates are shared deliberately.",
                "private_values": {
                    "distinct_normalized_values": len(private),
                    "train_development_overlap": 0,
                },
                "cross_split_audit": {
                    "id_overlap": 0,
                    "exact_text_overlap": 0,
                    "name_location_value_overlap": 0,
                },
                "clinical_coherence": {
                    "repeated_person_identity_is_consistent": True,
                    "titles_and_predicates_match_person_role": True,
                    "followup_is_28_days_after_assessment": True,
                    "publication_precedes_assessment_year": True,
                    "normal_age_doses_labs_scores_durations_public_dates_are_unlabelled": True,
                },
                "comparison_audit": [
                    {k: v for k, v in item.items() if k != "text_hashes"}
                    | {"exact_text_overlap": 0, "exact_id_overlap": 0}
                    for item in comparison
                ],
            }
            manifests[split] = manifest
        for split, path in names.items():
            draft = path.with_suffix(".jsonl.tmp")
            # Atomic exclusive publication: even a concurrently created file
            # cannot be replaced by this immutable corpus writer.
            os.link(draft, path)
            draft.unlink()
            with path.with_suffix(".jsonl.manifest.json").open(
                "x", encoding="utf-8"
            ) as handle:
                json.dump(manifests[split], handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        return manifests
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--compare-corpus", action="append", default=[])
    args = parser.parse_args()
    manifests = write_dataset(
        args.output,
        args.train_count,
        args.val_count,
        args.seed,
        compare_corpora=args.compare_corpus,
        max_tokens=args.max_tokens,
    )
    print(
        json.dumps(
            {
                split: {
                    k: m[k]
                    for k in (
                        "rows",
                        "sha256",
                        "phi_free_notes",
                        "token_length",
                        "name_span_traits",
                    )
                }
                for split, m in manifests.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
