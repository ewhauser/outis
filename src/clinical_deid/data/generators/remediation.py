"""Original policy-v2 replacement training corpus; no legacy corpus is read.

Generated prose/annotations: CC0-1.0. Source: Apache-2.0. Identities are invented
combinations, never records of real people. This finite template generator is
development supervision, not an independent qualification benchmark. Long notes
exercise measured tokenizer joins but cannot represent institutional diversity.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import unicodedata

import tiktoken

from clinical_deid.types import ENTITY_TYPES
from clinical_deid.paths import PROJECT_ROOT


VERSION = "1.0"
ANNOTATION_POLICY = "clinical-phi-v2"
SEED = 11031
TOKENIZER = "o200k_base"
INFERENCE_WINDOW = 512
INFERENCE_STRIDE = 256
FIRST_INFERENCE_JOIN = 384
TRAINING_WINDOW = 512

# These are plausible spelling elements, not person-level data or dictionaries
# imported from another generator. Every spelling pool participates in every
# role: only a hash of the complete identity assigns train/development roles.
NAME_POOLS = (
    (
        ("Amara", "Adaeze", "Chidi", "Emeka", "Ifeoma", "Kelechi", "Nneka", "Obinna"),
        ("Okafor", "Eze", "Nwosu", "Umeh", "Obi", "Okeke", "Nnamani", "Ibe"),
    ),
    (
        ("Aisha", "Farah", "Hassan", "Leila", "Mariam", "Nadia", "Omar", "Samir"),
        (
            "Haddad",
            "Mansour",
            "Saleh",
            "Nasser",
            "Rahman",
            "Karim",
            "Darwish",
            "Khalil",
        ),
    ),
    (
        ("Anjali", "Arun", "Deepa", "Kavita", "Meera", "Nikhil", "Priya", "Rohan"),
        ("Iyer", "Rao", "Shah", "Menon", "Kapoor", "Desai", "Patel", "Basu"),
    ),
    (
        ("Aiko", "Akira", "Emi", "Haruka", "Keiko", "Mika", "Ren", "Yuki"),
        ("Fujita", "Ishikawa", "Mori", "Nakano", "Ogawa", "Sakai", "Tanaka", "Yamada"),
    ),
    (
        ("Bo", "Jia", "Jing", "Lin", "Mei", "Min", "Wei", "Yan"),
        ("Chen", "Deng", "Gao", "Huang", "Liang", "Sun", "Wang", "Zhou"),
    ),
    (
        ("Bảo", "Dung", "Hà", "Hạnh", "Linh", "Minh", "Quỳnh", "Thảo"),
        ("Bùi", "Đặng", "Đỗ", "Hoàng", "Lê", "Nguyễn", "Phạm", "Trần"),
    ),
    (
        ("Carmen", "Diego", "Elena", "Inés", "Javier", "Lucía", "Mateo", "Sofía"),
        ("Álvarez", "Cárdenas", "Díaz", "García", "Muñoz", "Núñez", "Ortega", "Ríos"),
    ),
    (
        ("Ana", "Beatriz", "Caio", "João", "Lívia", "Luís", "Marina", "Tiago"),
        ("Almeida", "Barros", "Costa", "Duarte", "Ferreira", "Lopes", "Melo", "Silva"),
    ),
    (
        ("Aoife", "Cian", "Eoin", "Fionn", "Maeve", "Niamh", "Orla", "Siobhán"),
        ("Byrne", "Daly", "Doyle", "Dunne", "Kelly", "Murphy", "O'Neill", "Walsh"),
    ),
    (
        ("Alina", "Daria", "Ilya", "Marta", "Nadia", "Oksana", "Pavel", "Sofiya"),
        (
            "Bondar",
            "Koval",
            "Kravchenko",
            "Melnyk",
            "Moroz",
            "Petrenko",
            "Shevchenko",
            "Tkachenko",
        ),
    ),
    (
        (
            "Aminata",
            "Awa",
            "Fatou",
            "Ibrahima",
            "Mariama",
            "Moussa",
            "Ndeye",
            "Ousmane",
        ),
        ("Ba", "Diop", "Fall", "Gueye", "Ndiaye", "Sarr", "Sow", "Sy"),
    ),
    (
        ("Anna", "Daniel", "Emily", "James", "Julia", "Noah", "Sarah", "Thomas"),
        (
            "Andersen",
            "Bennett",
            "Clarke",
            "Cooper",
            "Fischer",
            "Martin",
            "Nguyen",
            "Reed",
        ),
    ),
)
COMMON_GIVEN = ("Rose", "May", "Will", "Grace", "Hope", "Mark", "Joy", "Summer")
EPONYMS = (
    ("Wilson", "Wilson disease"),
    ("Parkinson", "Parkinson disease"),
    ("Addison", "Addison disease"),
    ("Baker", "Baker cyst"),
    ("Hodgkin", "Hodgkin lymphoma"),
    ("Huntington", "Huntington disease"),
)
UNUSUAL_NAMES = ("He", "She", "We", "They", "Care", "Unknown", "None", "I", "You", "It")
GENRES = (
    "respiratory_followup",
    "mobility_review",
    "medication_reconciliation",
    "telephone_triage",
    "home_nursing",
    "nutrition_visit",
    "sleep_review",
    "discharge_handover",
    "wound_check",
    "neurology_visit",
    "renal_followup",
    "therapy_progress",
    "psychiatry_review",
    "prenatal_review",
    "dental_review",
    "genetics_counseling",
)
LAYOUTS = ("narrative", "soap", "bullets", "handover", "table", "compact")
SYMPTOMS = (
    "exertional breathlessness",
    "morning stiffness",
    "interrupted sleep",
    "reduced appetite",
    "intermittent dizziness",
    "ankle swelling",
    "incisional discomfort",
    "dry cough",
    "fatigue after walking",
    "tingling in the fingertips",
    "reflux after meals",
    "difficulty with stairs",
)
OBSERVATIONS = (
    "Lungs are clear and breathing is unlabored.",
    "There is no new focal weakness; grip is symmetric.",
    "The dressing is dry with no surrounding erythema.",
    "Transfers are independent with use of the handrail.",
    "The abdomen is soft and bowel sounds are present.",
    "There is mild dependent edema without calf tenderness.",
    "Speech is fluent and attention is sustained throughout the visit.",
    "Oral intake is improving and swallowing is comfortable.",
    "The skin is warm, intact, and without a new rash.",
    "Gait is steady on level ground with the usual walking aid.",
    "The cough is nonproductive and does not interrupt conversation.",
    "The patient completes the exercise sequence with verbal prompts.",
)
PLANS = (
    "Continue the current plan and reassess tolerance at the next review.",
    "Check sitting and standing symptoms before increasing activity.",
    "Review the medication organizer and repeat teach-back of the schedule.",
    "Use shorter exercise sessions with a rest period between tasks.",
    "Keep a symptom diary and call if symptoms become persistent.",
    "Maintain oral hydration and record intake during the next shift.",
    "Practice transfers with supervision until confidence improves.",
    "Discuss the response to treatment during the next team meeting.",
    "Use the written action plan if breathlessness limits routine activity.",
    "Monitor the wound for drainage, spreading redness, or increasing pain.",
)
PRONOUNS = (
    "He asked about the exercise schedule. We reviewed it together.",
    "She reports that rest helps. I checked understanding with teach-back.",
    "They brought the medication list. You can use the same organizer at home.",
    "We discussed pacing. It is easier to complete tasks after a short rest.",
)
ORDINARY_LANGUAGE = (
    "Care was coordinated with the therapy team.",
    "The hand rose comfortably above shoulder height.",
    "Symptoms may improve with gradual activity.",
    "The team will review the response after treatment.",
    "There is hope that sleep will improve as discomfort settles.",
    "Mark the symptom score before the evening dose.",
)
STREETS = ("Ash", "Cedar", "Linden", "Maple", "Orchard", "Sycamore", "Willow", "Rowan")
TOWNS = (
    "Briar Crossing",
    "Cedar Hollow",
    "Fern Harbour",
    "Linden Vale",
    "Rowan Bay",
    "Willow Reach",
)
FACILITY_NAMES = (
    "Birchway",
    "Cedarbridge",
    "Fernridge",
    "Lindenford",
    "Rowancrest",
    "Willowbank",
)


@dataclass
class TextBuilder:
    text: str = ""
    spans: list[dict] = field(default_factory=list)

    def add(self, text: str, label: str | None = None) -> None:
        start = len(self.text)
        self.text += text
        if label is not None:
            self.spans.append({"start": start, "end": len(self.text), "label": label})

    def extend(self, other: TextBuilder) -> None:
        offset = len(self.text)
        self.text += other.text
        self.spans.extend(
            {**span, "start": span["start"] + offset, "end": span["end"] + offset}
            for span in other.spans
        )


def identity_group(full_name: str) -> str:
    """Shared contract; aliases inherit this full-owner group unchanged."""
    canonical = unicodedata.normalize("NFC", full_name).casefold()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def identity_partition(full_name: str) -> str:
    return "train" if int(identity_group(full_name)[:8], 16) % 10 < 8 else "validation"


def _identity(
    rng: random.Random, *, unusual: bool = False, collision: bool = False
) -> dict:
    for _ in range(10000):
        given_pool, surname_pool = rng.choice(NAME_POOLS)
        given, surname = rng.choice(given_pool), rng.choice(surname_pool)
        if collision:
            given = rng.choice(COMMON_GIVEN)
            surname = rng.choice(EPONYMS)[0]
        if unusual:
            given = rng.choice(UNUSUAL_NAMES)
        middle = (
            f" {rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}."
            if rng.random() < 0.65
            else ""
        )
        full = f"{given}{middle} {surname}"
        if identity_partition(full) == "train":
            return {
                "given": given,
                "surname": surname,
                "full": full,
                "group": identity_group(full),
                "unusual": unusual,
            }
    raise RuntimeError("Could not assign a full identity to the training partition")


def get_encoding():
    os.environ.setdefault(
        "TIKTOKEN_CACHE_DIR",
        str(PROJECT_ROOT / ".cache/tiktoken"),
    )
    return tiktoken.get_encoding(TOKENIZER)


def _age_words(age: int) -> str:
    units = ("", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    return (
        ("ninety" + (f"-{units[age - 90]}" if age > 90 else ""))
        if age < 100
        else "one hundred"
    )


def _clinical(
    rng: random.Random, *, long_block: int | None = None, genre: str | None = None
) -> str:
    """Same clinical prose distribution in positive and negative notes."""
    symptom, finding, plan = (
        rng.choice(SYMPTOMS),
        rng.choice(OBSERVATIONS),
        rng.choice(PLANS),
    )
    duration = rng.choice((2, 3, 7, 14, 21, 42, 90, 100, 120))
    public_date = rng.choice(("2018", "2020", "March 2021", "2023", "June 2024"))
    disease = rng.choice(EPONYMS)[1]
    bp, pulse = rng.randrange(108, 150, 2), rng.randrange(60, 94)
    specialty = {
        "psychiatry_review": [
            "Mood has improved with the current treatment, but sleep onset remains difficult.",
            "Thought process is organized; no perceptual disturbance is reported.",
            "There is no current suicidal intent or plan. The crisis contacts and safety plan were reviewed.",
            "Continue behavioral activation and review medication tolerance at the next visit.",
        ],
        "prenatal_review": [
            "Routine antenatal review at 24 weeks and 3 days; fetal movement is present.",
            "There is no vaginal bleeding, fluid loss, or regular contraction pattern.",
            "Fundal height is 24 cm and fetal heart rate is 146/min.",
            "Discussed the glucose screen, warning symptoms, and when to contact maternity triage.",
        ],
        "dental_review": [
            "Sensitivity with cold drinks is localized to the lower left molar.",
            "No facial swelling or trismus; the adjacent gingiva is mildly inflamed.",
            "The radiograph shows an occlusal lesion without periapical change.",
            "Discussed restoration, brushing technique, and fluoride toothpaste use.",
        ],
        "genetics_counseling": [
            "The visit reviewed an inherited cardiomyopathy panel and the limits of interpretation.",
            "A variant of uncertain significance does not establish a molecular diagnosis.",
            "No sequence change is recorded in this note; family screening was discussed in general terms.",
            "Arrange a cardiology review and revisit the classification if the laboratory issues an update.",
        ],
    }.get(genre)
    core = [
        f"The main concern is {symptom} over {duration} days.",
        finding,
        rng.choice(PRONOUNS),
        f"BP {bp}/{rng.randrange(62, 88, 2)} mmHg; pulse {pulse}/min; pain {rng.randrange(0, 6)}/10.",
        rng.choice(ORDINARY_LANGUAGE),
        plan,
    ]
    if rng.random() < 0.35:
        core.append(
            f"The educational review published in {public_date} discusses {disease}."
        )
    if rng.random() < 0.30:
        core.append("Screening recommendations for ages 50 to 75 were discussed.")
    if specialty:
        pronouns = PRONOUNS[1] if genre == "prenatal_review" else rng.choice(PRONOUNS)
        core = [
            *specialty,
            pronouns,
            f"The symptom diary covers {duration} days.",
            rng.choice(ORDINARY_LANGUAGE),
        ]
        if rng.random() < 0.35:
            core.append(
                f"The counseling leaflet published in {public_date} was reviewed."
            )
    if long_block is not None:
        # Block numbers are local clinical sequence numbers, never an identity.
        core.insert(0, f"Progress entry {long_block + 1}: response to the care plan.")
        core.append(
            f"Activity: {rng.randrange(20, 240, 5)} metres with {rng.randrange(1, 5)} seated rests; "
            f"fluid intake {rng.randrange(100, 400, 25)} mL during observation."
        )
        # Shuffle independent observations without breaking clinical chronology.
        middle = core[1:-1]
        rng.shuffle(middle)
        core[1:-1] = middle
    return " ".join(core)


def _name_context(
    identity: dict, style: int, *, collision: bool
) -> tuple[TextBuilder, list[str]]:
    b, phenomena = TextBuilder(), []
    full, given, surname = identity["full"], identity["given"], identity["surname"]
    if identity["unusual"]:
        b.add("Patient registration name: ")
        b.add(full, "NAME")
        b.add(". The patient requested the recorded name ")
        b.add(given, "NAME")
        b.add(" on the care plan.")
        phenomena.append("explicit_unusual_name")
    elif style == 0:
        b.add("Reviewed with ")
        b.add(full, "NAME")
        b.add(" at the bedside.")
    elif style == 1:
        nickname = {
            "Daniel": "Dan",
            "Thomas": "Tom",
            "Emily": "Em",
            "James": "Jamie",
        }.get(given, given)
        b.add("Preferred name: ")
        b.add(nickname, "NAME")
        b.add(". The registration lists ")
        b.add(full, "NAME")
        b.add(".")
        phenomena.extend(
            ("nickname" if nickname != given else "preferred_name", "alias_first")
        )
    elif style == 2:
        b.add("Message from patient ")
        b.add(f"{given[0]}. {surname}", "NAME")
        b.add("; verified against ")
        b.add(full, "NAME")
        b.add(" in the chart.")
        phenomena.append("initial")
    elif style == 3:
        b.add("patient: ")
        b.add(full.lower(), "NAME")
        b.add("; the recorded spelling was confirmed during intake.")
        phenomena.append("lowercase")
    elif style == 4:
        b.add("Patient ")
        b.add(surname, "NAME")
        b.add(" described the interval course. Patient identification later confirmed ")
        b.add(full, "NAME")
        b.add(".")
        phenomena.append("alias_first")
    else:
        damaged = full.replace(" ", "  ").replace("o", "0", 1)
        b.add("Scanned registration reads ")
        b.add(damaged, "NAME")
        b.add(". The patient confirmed the name ")
        b.add(full, "NAME")
        b.add(".")
        phenomena.append("ocr")
    if collision:
        disease = dict(EPONYMS).get(surname, "Wilson disease")
        b.add(f" The history includes {disease}. Care will review the response.")
        phenomena.extend(("eponym_collision", "word_name_collision"))
    return b, phenomena


def _values(rng: random.Random, identity: dict) -> dict[str, str]:
    town = rng.choice(TOWNS)
    day = date(2025, 1, 1) + timedelta(days=rng.randrange(500))
    serial = rng.randrange(100000, 999999)
    # .test/.example and TEST-NET ranges cannot identify routable real endpoints.
    domain = rng.choice(
        ("care-mail.test", "contact.example", "recovery.test", "clinic.example")
    )
    phone_digits = f"01{rng.randrange(100):02d}"
    area = rng.choice((202, 212, 303, 415, 617, 703))
    phones = (
        f"({area}) 555-{phone_digits}",
        f"+1 {area} 555 {phone_digits}",
        f"{area}.555.{phone_digits}",
        f"{area}-555-{phone_digits} ext. {rng.randrange(100, 999)}",
    )
    return {
        "NAME": identity["full"],
        "DATE": rng.choice(
            (
                day.isoformat(),
                day.strftime("%d %b %Y"),
                day.strftime("%B %Y"),
                day.strftime("%m/%d"),
            )
        ),
        "ADDRESS": f"{rng.randrange(10, 8000)} {rng.choice(STREETS)} {rng.choice(('Lane', 'Road', 'Avenue'))}, {town}, {rng.choice(('CO', 'Oregon', 'VA', 'Maine', 'WI', 'Utah'))} {rng.randrange(10000, 99999)}",
        "PHONE": rng.choice(phones),
        "EMAIL": f"{rng.choice(('messages', 'reply', 'contact', 'person'))}.{serial}@{domain}",
        "MRN": f"{rng.choice(('MR', 'C', 'HX'))}-{serial}-{rng.randrange(10)}",
        "SSN": f"000-{rng.randrange(10, 99)}-{rng.randrange(1000, 9999)}",
        "ACCOUNT": f"{rng.choice(('AC', 'BILL', 'FIN'))}/{serial}/{rng.randrange(100, 999)}",
        "ID": f"{rng.choice(('MEM', 'VISIT', 'ENC'))}-{serial}-{rng.choice('ABCDEFGH')}",
        "URL": f"https://{rng.choice(('portal', 'records', 'appointments'))}.{domain}/{rng.choice(('visit', 'person', 'link'))}/{serial}",
        "IP": f"{rng.choice(('192.0.2', '198.51.100', '203.0.113'))}.{rng.randrange(1, 255)}",
        "AGE": str(rng.randrange(90, 101)),
        "LOCATION": town,
        "FACILITY": f"{rng.choice(FACILITY_NAMES)} {rng.choice(('Health Centre', 'Community Clinic', 'Rehabilitation Unit', 'Hospital'))}",
    }


FIELD_HEADINGS = {
    "DATE": ("Visit", "Appointment date", "Symptom onset"),
    "ADDRESS": ("Home address", "Mailing address", "Discharge address"),
    "PHONE": ("Callback", "Home phone", "Contact number"),
    "EMAIL": ("Contact email", "Patient email", "Reply address"),
    "MRN": ("Chart number", "MRN", "Clinical record"),
    "SSN": ("SSN", "Social security number", "Social-security identifier"),
    "ACCOUNT": ("Billing account", "Private payment account", "Patient account"),
    "ID": ("Member ID", "Encounter identifier", "Visit ID"),
    "URL": ("Patient portal link", "Private results link", "Appointment link"),
    "IP": ("Patient session IP", "Portal login IP", "Remote visit IP"),
    "LOCATION": ("Home community", "Residence locality", "Recovery location"),
    "FACILITY": ("Receiving facility", "Referring clinic", "Care provider site"),
}


def _field(
    builder: TextBuilder, label: str, value: str, rng: random.Random, layout: str
) -> None:
    heading = rng.choice(FIELD_HEADINGS[label])
    if layout == "table":
        builder.add(f"| {heading} | ")
        builder.add(value, label)
        builder.add(" |\n")
    elif layout == "compact":
        builder.add(f"{heading}=")
        builder.add(value, label)
        builder.add("; ")
    else:
        builder.add(f"{heading}: ")
        builder.add(value, label)
        builder.add(".\n")


def _row(
    builder: TextBuilder,
    *,
    index: int,
    seed: int,
    identity: dict,
    pair_id: str,
    genre: str,
    layout: str,
    cohort: str,
    phenomena: list[str],
) -> dict:
    row = {
        "id": f"remediation-v{VERSION}-s{seed}-{index:07d}",
        "text": builder.text,
        "spans": builder.spans,
        "split": "train",
        "suite": "training",
        "family": f"remediation_train.{genre}.{layout}",
        "annotation_policy": ANNOTATION_POLICY,
        "identity_group": identity["group"],
        "identity_groups": [identity["group"]],
        "full_identity": identity["full"],
        "pair_id": pair_id,
        "template_family": f"remediation_train.{genre}.{layout}",
        "phenomena": sorted(set(phenomena)),
        "cohort": cohort,
        "negative": not builder.spans,
    }
    validate_row(row)
    return row


def _short_row(
    index: int,
    group_index: int,
    *,
    seed: int,
    negative: bool,
    unusual: bool,
    paired: bool,
) -> dict:
    rng = random.Random(f"remediation:{seed}:short:{group_index}")
    collision = group_index % 3 == 0 and not unusual
    identity = _identity(rng, unusual=unusual, collision=collision)
    genre, layout = (
        GENRES[group_index % len(GENRES)],
        LAYOUTS[(group_index // len(GENRES)) % len(LAYOUTS)],
    )
    clinical = _clinical(rng, genre=genre)
    name, phenomena = _name_context(identity, group_index % 6, collision=collision)
    values = _values(rng, identity)
    age = int(values["AGE"])
    b = TextBuilder()
    b.add(genre.replace("_", " ").capitalize() + "\n")
    if layout == "soap":
        b.add("S / Interval history\n")
    elif layout == "handover":
        b.add("Situation / background / assessment / plan\n")
    elif layout == "bullets":
        b.add("- ")
    if not negative:
        b.extend(name)
        b.add("\n")
    elif collision:
        disease = dict(EPONYMS)[identity["surname"]]
        b.add(f"The history includes {disease}. Care will review the response.\n")
    # Age grammar and prose are shared across positive and PHI-free notes.
    if genre == "prenatal_review":
        age = rng.randrange(20, 45)
    elif genre == "dental_review":
        age = rng.randrange(8, 81)
    elif negative or group_index % 4 == 0:
        age = rng.randrange(18, 90)
    age_value = str(age)
    b.add("Patient: ")
    style = group_index % 5
    if style == 4 and age > 89:
        age_value = _age_words(age)
    b.add(age_value, "AGE" if age > 89 and not negative else None)
    sex = "M" if "He asked" in clinical else "F"
    b.add((sex, "yo", " y/o", "-year-old", " years old")[style] + ". ")
    b.add(clinical + "\n")
    phenomena.extend(
        (
            "ordinary_pronouns",
            "clinical_durations",
            "compact_age" if style < 3 else "age_value",
        )
    )
    if "published in" in clinical:
        phenomena.append("public_dates")
    if age <= 89 or "ages 50 to 75" in clinical:
        phenomena.append("younger_age_context")
    if not negative:
        candidates = [label for label in ENTITY_TYPES if label not in {"NAME", "AGE"}]
        # Rotate a modest density instead of putting every identifier in every note.
        width = 3 + group_index % 4
        selected = [
            candidates[(group_index * 5 + offset) % len(candidates)]
            for offset in range(width)
        ]
        rng.shuffle(selected)
        if layout == "table":
            b.add("| Field | Recorded value |\n| --- | --- |\n")
            phenomena.append("structured_table")
        for label in selected:
            _field(b, label, values[label], rng, layout)
        if "DATE" in selected:
            phenomena.append("patient_date")
            if len(values["DATE"]) == 5 or any(
                month in values["DATE"]
                for month in (
                    "January",
                    "February",
                    "March",
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                    "November",
                    "December",
                )
            ):
                phenomena.append("partial_date")
        if any(label in selected for label in ("PHONE", "EMAIL", "URL", "IP")):
            phenomena.append("reserved_identifier")
    else:
        # Negatives contain clinical language, not explanations of annotation rules.
        if layout == "table":
            b.add(
                "| Observation | Result |\n| --- | --- |\n| Strength | 5/5 |\n| Temperature | 36.8 C |\n"
            )
            phenomena.append("structured_table")
        # Positive alias phenomena must never be claimed on an unlabeled variant.
        phenomena = [
            p
            for p in phenomena
            if p
            not in {
                "explicit_unusual_name",
                "nickname",
                "preferred_name",
                "alias_first",
                "initial",
                "lowercase",
                "ocr",
                "eponym_collision",
                "word_name_collision",
            }
        ]
        if collision:
            phenomena.extend(("eponym_negative", "word_name_negative"))
    if paired:
        phenomena.append("context_pair")
    return _row(
        b,
        index=index,
        seed=seed,
        identity=identity,
        pair_id=f"remediation-train-{seed}-short-{group_index}",
        genre=genre,
        layout=layout,
        cohort="unusual_name" if unusual else "realistic",
        phenomena=phenomena,
    )


def token_metadata(row: dict, encoding=None) -> dict:
    """Measure actual OPF token ownership; never load a neural model."""
    from opf._core.spans import token_char_ranges_for_text

    encoding = encoding or get_encoding()
    ids = encoding.encode(row["text"], disallowed_special=())
    starts, ends = token_char_ranges_for_text(ids, encoding, row["text"])
    projected = []
    for span in row["spans"]:
        owned = [
            i
            for i, (start, end) in enumerate(zip(starts, ends))
            if start < span["end"] and end > span["start"]
        ]
        if not owned:
            raise ValueError("Gold span has no token ownership")
        projected.append(
            {"label": span["label"], "start": owned[0], "end": owned[-1] + 1}
        )
    joins = range(FIRST_INFERENCE_JOIN, len(ids), INFERENCE_STRIDE)
    training_seams = range(TRAINING_WINDOW, len(ids), TRAINING_WINDOW)

    def coverage(boundaries):
        return [
            {
                "token": seam,
                "crossing_spans": sum(
                    span["start"] < seam < span["end"] for span in projected
                ),
                "nearest_span_distance": min(
                    (
                        max(span["start"] - seam, seam - span["end"], 0)
                        for span in projected
                    ),
                    default=None,
                ),
            }
            for seam in boundaries
        ]

    return {
        "tokenizer": TOKENIZER,
        "token_count": len(ids),
        "inference_window": INFERENCE_WINDOW,
        "inference_stride": INFERENCE_STRIDE,
        "training_window": TRAINING_WINDOW,
        "inference_joins": coverage(joins),
        "training_seams": coverage(training_seams),
    }


def _pad_to(
    builder: TextBuilder, goal: int, rng: random.Random, encoding, block_index: int
) -> int:
    """Fill with complete clinical sentences; no token-truncated prose."""
    small = (
        "Stable.\n",
        "Alert.\n",
        "No pain.\n",
        "Resting comfortably.\n",
        "No new concerns.\n",
        "The plan is unchanged.\n",
    )
    while True:
        current = len(encoding.encode(builder.text, disallowed_special=()))
        if current >= goal - 1:
            return block_index
        # Each progress entry combines a new topic, finding, measurement and plan.
        # Short blocks fit the 128-token gaps between interleaved join families.
        block = (
            f"Progress entry {block_index + 1}: {rng.choice(SYMPTOMS)}. "
            f"{rng.choice(OBSERVATIONS)} "
            f"Activity {rng.randrange(20, 240, 5)} metres with {rng.randrange(1, 5)} seated rests; "
            f"oral intake {rng.randrange(100, 400, 25)} mL. {rng.choice(PLANS)}\n"
        )
        if len(encoding.encode(builder.text + block, disallowed_special=())) <= goal:
            builder.add(block)
            block_index += 1
            continue
        sentences = [rng.choice(OBSERVATIONS) + "\n", rng.choice(PLANS) + "\n", *small]
        usable = [
            (
                len(encoding.encode(builder.text + sentence, disallowed_special=())),
                sentence,
            )
            for sentence in sentences
        ]
        usable = [
            (count, sentence) for count, sentence in usable if current < count <= goal
        ]
        if not usable:
            return block_index
        _, sentence = max(usable, key=lambda item: item[0])
        builder.add(sentence)


def _narrative_long_row(
    index: int,
    seed: int,
    target_tokens: int,
    rng: random.Random,
    identity: dict,
    encoding,
) -> dict:
    b = TextBuilder("Multidisciplinary continuity record\n")
    b.add("Preferred name: ")
    b.add(identity["given"], "NAME")
    b.add(". Registration confirms ")
    b.add(identity["full"], "NAME")
    b.add(".\n")
    copied = TextBuilder("Handover carried forward: ")
    copied.add(identity["full"], "NAME")
    copied.add(
        " uses the medication organizer and asks for written exercise instructions.\n"
    )
    sections = (
        "Nursing observations",
        "Therapy review",
        "Medication review",
        "Functional assessment",
        "Nutrition and hydration",
        "Home care planning",
        "Interval symptoms",
        "Patient priorities",
    )
    count, block = 0, 0
    while True:
        heading = sections[block % len(sections)]
        observations = rng.sample(OBSERVATIONS, 2)
        measures = (
            f"Sitting pulse {rng.randrange(60, 95)}/min; pain {rng.randrange(0, 5)}/10. "
            f"Walking distance {rng.randrange(40, 250, 5)} metres. "
        )
        segment = (
            f"\n{heading}\n"
            + " ".join(observations)
            + " "
            + measures
            + rng.choice(PRONOUNS)
            + " "
            + rng.choice(PLANS)
            + "\n"
        )
        candidate_count = len(encoding.encode(b.text + segment, disallowed_special=()))
        if candidate_count > target_tokens - 140:
            break
        b.add(segment)
        count = candidate_count
        if block in (2, 6):
            b.extend(copied)
        block += 1
    # A final short review and tail identifiers exercise aliases and document ends
    # without placing anything at model window boundaries.
    b.add("\nFinal handover\nPatient ")
    b.add(identity["surname"], "NAME")
    b.add(" confirmed the follow-up arrangements and repeated the escalation plan.\n")
    values = _values(rng, identity)
    for label in ("DATE", "PHONE", "FACILITY"):
        _field(b, label, values[label], rng, "narrative")
    if count < 1500:
        _pad_to(b, 1500, rng, encoding, block)
        if len(encoding.encode(b.text, disallowed_special=())) < 1500:
            b.add("Stable.\n")
    row = _row(
        b,
        index=index,
        seed=seed,
        identity=identity,
        pair_id=f"remediation-train-{seed}-long-{index}",
        genre="continuity_record",
        layout="sectioned",
        cohort="long_document",
        phenomena=[
            "long_document",
            "alias_first",
            "copied_block",
            "tail_phi",
            "ordinary_pronouns",
            "clinical_measurements",
        ],
    )
    row["diagnostic_slice"] = "long_narrative"
    row["token_audit"] = token_metadata(row, encoding)
    if not 1500 <= row["token_audit"]["token_count"] <= 6000:
        raise ValueError("Narrative long note is outside the measured token range")
    return row


def generate_long_row(
    index: int,
    *,
    seed: int = SEED,
    target_tokens: int = 3000,
    mode: str = "seam_stress",
    encoding=None,
) -> dict:
    if not 1500 <= target_tokens <= 6000:
        raise ValueError("Long-document target must be between 1500 and 6000 tokens")
    encoding = encoding or get_encoding()
    rng = random.Random(f"remediation:{seed}:long:{index}")
    identity = _identity(rng)
    if mode == "long_narrative":
        return _narrative_long_row(index, seed, target_tokens, rng, identity, encoding)
    if mode != "seam_stress":
        raise ValueError("Long mode must be seam_stress or long_narrative")
    # A mult-token name lets the corpus exercise a true seam crossing; shorter
    # realistic names remain abundant in ordinary notes and the opening alias.
    while len(encoding.encode(" " + identity["full"], disallowed_special=())) < 4:
        identity = _identity(rng)
    b = TextBuilder("Integrated care record\n")
    # Alias comes before the full identity and reappears after several joins.
    b.add("The patient asked to be called ")
    b.add(identity["given"], "NAME")
    b.add(".\n")
    boundaries = sorted(
        set(range(FIRST_INFERENCE_JOIN, target_tokens - 150, INFERENCE_STRIDE))
        | set(range(TRAINING_WINDOW, target_tokens - 150, TRAINING_WINDOW))
    )
    # Each declared target is checked against actual whole-text token offsets.
    block_index = 0
    copied = TextBuilder()
    copied.add("Earlier handover: ")
    copied.add(identity["full"], "NAME")
    copied.add(" remains comfortable at rest; the medication schedule was reviewed.\n")
    for sequence, seam in enumerate(boundaries):
        prefix = rng.choice(("Patient: ", "Reviewed with ", "Name on handover: "))
        # Prefixes tokenize differently when attached to names. Verify each
        # candidate using the full text instead of trusting prefix token counts.
        prefix_tokens = len(encoding.encode(prefix, disallowed_special=()))
        block_index = _pad_to(b, seam - prefix_tokens - 3, rng, encoding, block_index)
        from opf._core.spans import token_char_ranges_for_text

        for extra in (
            "",
            "Alert.\n",
            "No pain.\n",
            "No new concerns.\n",
            "The plan is unchanged.\n",
        ):
            start = len(b.text + extra + prefix)
            end = start + len(identity["full"])
            candidate = b.text + extra + prefix + identity["full"] + ".\n"
            ids = encoding.encode(candidate, disallowed_special=())
            starts, ends = token_char_ranges_for_text(ids, encoding, candidate)
            owned = [
                i
                for i, (left, right) in enumerate(zip(starts, ends))
                if left < end and right > start
            ]
            if owned[0] < seam < owned[-1] + 1:
                b.add(extra + prefix)
                b.add(identity["full"], "NAME")
                b.add(".\n")
                break
        else:
            raise ValueError(
                f"Could not place a measured name crossing at token {seam}"
            )
        if sequence in (1, len(boundaries) - 2):
            b.extend(copied)
    values = _values(rng, identity)
    for label in ("DATE", "MRN", "PHONE", "FACILITY"):
        _field(b, label, values[label], rng, "table")
    _pad_to(b, target_tokens - 1, rng, encoding, block_index)
    row = _row(
        b,
        index=index,
        seed=seed,
        identity=identity,
        pair_id=f"remediation-train-{seed}-long-{index}",
        genre="integrated_care_record",
        layout="longitudinal",
        cohort="long_document",
        phenomena=[
            "long_document",
            "alias_first",
            "copied_block",
            "inference_join",
            "training_seam",
            "ordinary_pronouns",
            "public_dates",
            "clinical_durations",
            "structured_table",
        ],
    )
    row["token_audit"] = token_metadata(row, encoding)
    count = row["token_audit"]["token_count"]
    # Padding can stop one token early. The minimum requested length still holds.
    if count < 1500:
        b.add("Stable.\n")
        row["text"] = b.text
        row["token_audit"] = token_metadata(row, encoding)
        count = row["token_audit"]["token_count"]
    if not 1500 <= count <= 6000:
        raise ValueError("Generated long note is outside the measured token range")
    for key in ("inference_joins", "training_seams"):
        targeted = [
            item for item in row["token_audit"][key] if item["token"] in boundaries
        ]
        if not targeted or not all(item["crossing_spans"] for item in targeted):
            raise ValueError(f"Generated long note lacks a declared crossing at {key}")
    row["token_audit"]["targeted_boundaries"] = boundaries
    row["diagnostic_slice"] = "seam_stress"
    return row


def build_corpus(
    *,
    short_count: int = 5000,
    long_count: int = 100,
    seed: int = SEED,
    negative_fraction: float = 0.18,
    unusual_fraction: float = 0.05,
    natural_long_fraction: float = 0.25,
) -> list[dict]:
    if short_count < 1 or long_count < 0:
        raise ValueError("Require positive short_count and nonnegative long_count")
    if not 0.15 <= negative_fraction <= 0.20 or not 0 <= unusual_fraction <= 0.10:
        raise ValueError(
            "Negative fraction must be 0.15..0.20; unusual fraction must be 0..0.10"
        )
    if not 0 <= natural_long_fraction <= 1:
        raise ValueError("Natural long fraction must be between zero and one")
    negative_count = round(short_count * negative_fraction)
    unusual_count = round(short_count * unusual_fraction)
    rows = []
    for pair in range(negative_count):
        for negative in (False, True):
            rows.append(
                _short_row(
                    len(rows),
                    pair,
                    seed=seed,
                    negative=negative,
                    unusual=False,
                    paired=True,
                )
            )
    while len(rows) < short_count:
        index = len(rows)
        rows.append(
            _short_row(
                index,
                index - negative_count,
                seed=seed,
                negative=False,
                unusual=index >= short_count - unusual_count,
                paired=False,
            )
        )
    if long_count:
        encoding = get_encoding()
        for offset in range(long_count):
            target = 1500 + (4500 * offset // max(long_count - 1, 1))
            # Spread the natural prose notes across the complete token range.
            narrative = int((offset + 1) * natural_long_fraction) > int(
                offset * natural_long_fraction
            )
            rows.append(
                generate_long_row(
                    short_count + offset,
                    seed=seed,
                    target_tokens=target,
                    mode="long_narrative" if narrative else "seam_stress",
                    encoding=encoding,
                )
            )
    if len({row["text"] for row in rows}) != len(rows):
        raise ValueError(
            "Duplicate note text; choose another seed or increase template diversity"
        )
    return rows


def validate_row(row: dict) -> None:
    text, previous_end = row["text"], 0
    if not text.strip() or row["split"] != "train" or row["suite"] != "training":
        raise ValueError("Replacement corpus rows are nonempty training text only")
    if row["annotation_policy"] != ANNOTATION_POLICY:
        raise ValueError("Unexpected annotation policy")
    if (
        row["identity_group"] != identity_group(row["full_identity"])
        or identity_partition(row["full_identity"]) != "train"
    ):
        raise ValueError("Incorrect full-identity partition")
    for span in row["spans"]:
        start, end = span["start"], span["end"]
        if span["label"] not in ENTITY_TYPES or not previous_end <= start < end <= len(
            text
        ):
            raise ValueError("Invalid, overlapping, or unsorted annotation")
        if text[start:end] != text[start:end].strip():
            raise ValueError("Gold spans cannot include boundary whitespace")
        previous_end = end


def write_corpus(
    path: str | Path, rows: list[dict], *, config: dict | None = None
) -> dict:
    """Exclusively create immutable bytes and manifest; never replace a file."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    if path.exists() or sidecar.exists():
        raise FileExistsError(
            "Corpus artifacts are immutable; choose a new output path"
        )
    if (
        not rows
        or len({row["id"] for row in rows}) != len(rows)
        or len({row["text"] for row in rows}) != len(rows)
    ):
        raise ValueError("Require nonempty corpus with unique row IDs and texts")
    for row in rows:
        validate_row(row)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    labels = Counter(span["label"] for row in rows for span in row["spans"])
    negative = sum(not row["spans"] for row in rows)
    encoding = get_encoding()
    lengths = [len(encoding.encode(row["text"], disallowed_special=())) for row in rows]
    manifest = {
        "schema_version": 1,
        "generator": "clinical_deid.data.generators.remediation",
        "generator_version": VERSION,
        "annotation_policy": ANNOTATION_POLICY,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(rows),
        "label_support": {label: labels[label] for label in ENTITY_TYPES},
        "split": "train",
        "suite": "training",
        "used_for_training": True,
        "used_for_checkpoint_selection": False,
        "independent_final": False,
        "replaces_legacy_training_inputs": True,
        "config": config or {},
        "license": "CC0-1.0",
        "code_license": "Apache-2.0",
        "external_datasets": [],
        "source": "Original synthetic clinical prose and invented identities; no legacy, development or qualification corpus read.",
        "phi_free_notes": negative,
        "phi_free_fraction": negative / len(rows),
        "cohort_support": dict(Counter(row["cohort"] for row in rows)),
        "families": dict(Counter(row["family"] for row in rows)),
        "phenomenon_support": dict(
            Counter(p for row in rows for p in row["phenomena"])
        ),
        "identity_partition": "sha256(NFC(full_identity).casefold().encode('utf-8')); int(hex[:8],16)%10 < 8 assigns train; 8..9 assigns development",
        "tokenizer": TOKENIZER,
        "token_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
        "limitations": [
            "Finite authored templates are correlated; no claim of clinical representativeness or independent qualification.",
            "Long notes contain synthetic progress entries and two intentional copied handovers; token-join coverage does not establish long-document accuracy.",
            "Synthetic cultural spelling pools are illustrative, not demographic coverage guarantees.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    owned = []
    try:
        # Both opens are exclusive. Clean up only artifacts created by this call.
        with path.open("xb") as handle:
            owned.append(path)
            handle.write(payload)
        with sidecar.open("x", encoding="utf-8") as handle:
            owned.append(sidecar)
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except BaseException:
        for artifact in owned:
            artifact.unlink(missing_ok=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--short-count", type=int, default=5000)
    parser.add_argument("--long-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--negative-fraction", type=float, default=0.18)
    parser.add_argument("--unusual-fraction", type=float, default=0.05)
    parser.add_argument("--natural-long-fraction", type=float, default=0.25)
    args = parser.parse_args()
    config = {key: value for key, value in vars(args).items() if key != "output"}
    if (
        args.output.exists()
        or args.output.with_suffix(args.output.suffix + ".manifest.json").exists()
    ):
        raise FileExistsError(
            "Corpus artifacts are immutable; choose a new output path"
        )
    rows = build_corpus(**config)
    manifest = write_corpus(args.output, rows, config=config)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": manifest["rows"],
                "sha256": manifest["sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
