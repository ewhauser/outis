"""Independently authored synthetic *development* diagnostics for clinical-phi-v2.

This module never imports training generators, detector code, prediction files,
or qualification material. A different seed is still development data. All
identities and clinical narratives are invented; no real patient data is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import unicodedata
from typing import Any

import tiktoken

from outis.types import ENTITY_TYPES
from outis.paths import PROJECT_ROOT


POLICY = "clinical-phi-v2"
DEFAULT_SEED = 20261003
TOKENIZER = "o200k_base"
ALLOWED_CONTEXT = [
    "Task instructions defining the development generator and clinical-phi-v2 rules",
    "reports/evaluation_protocol.md (protocol only; no referenced banks or reports opened)",
    "src/outis/types.py (entity labels and character-offset schema)",
    "src/outis/evaluation/evaluate.py (manifest and evaluation schema only)",
    "docs/annotation-policy-v2.md (annotation policy)",
    "Canonical full-name SHA-256 split allocation supplied by the coordinating agent",
    "Advisory policy edge-case review based solely on the task's annotation rules",
    "Source-review feedback on this generator's annotations, metadata and diagnostic scope; no training templates or rows supplied",
    "Local Python executable and tiktoken version; this generator's own tests and output",
]

# Ordinary spelling components are intentionally not assigned to separate
# alphabets. Only a complete canonical identity determines split allocation.
GIVEN = (
    "Adriana", "Bennett", "Camila", "Dorian", "Eliana", "Farid", "Greta", "Hamish",
    "Imani", "Jasper", "Keiko", "Lucian", "Marisol", "Nikhil", "Odette", "Pavel",
    "Rania", "Soren", "Thalia", "Umar", "Valeria", "Wesley", "Yasmin", "Zahir",
    "Anika", "Brendan", "Celine", "Dalia", "Emil", "Freya", "Gideon", "Hana",
    "Ismael", "Joelle", "Karim", "Leona", "Mateo", "Nadia", "Omar", "Priya",
    "Rafael", "Sabina", "Tomas", "Vera", "Yuna", "Zara", "Marta", "Noel",
    "Inés", "Sébastien", "Zoë", "Márton",
)
SURNAMES = (
    "Avelar", "Bellamy", "Carrillo", "Dunmore", "Estevez", "Falkner", "Garrido",
    "Haldane", "Ishikawa", "Jensen", "Khatri", "Lindholm", "Mendoza", "Navarro",
    "Okafor", "Petrov", "Quintero", "Rahmani", "Sato", "Tavares", "Varela",
    "Whitcomb", "Yilmaz", "Zamani", "Aranda", "Berglund", "Chandra", "Delaney",
    "Engstrom", "Ferreira", "Gowda", "Hassan", "Ivanova", "Kovacs", "Laurent",
    "Mori", "Nasseri", "Ocampo", "Pereira", "Ravindran", "Salcedo", "Tran",
    "Velasco", "Westcott", "Zuniga", "O'Rourke", "Kim-Ramos", "de la Cruz",
    "Núñez", "Lefèvre", "Jørgensen", "Gutiérrez",
)
WORD_NAMES = ("Hope", "May", "Rose", "Will", "Care", "He", "She", "They")
LOCALITIES = (
    "Larkspur Hollow, Oregon", "Juniper Cove, Maine", "Copperfield, Nevada",
    "Cedar Quay, Wisconsin", "Alder Reach, Vermont", "Tamarack Glen, Colorado",
    "Willow Crossing, New Mexico", "Heron Vale, Washington",
)
FACILITIES = (
    "Alder Quay Community Hospital", "Juniper Lantern Rehabilitation Center",
    "Cedar Reach Children's Hospital", "Copperleaf Diagnostic Pavilion",
    "Willow Arc Regional Medical Center", "Heron Vale Family Practice",
    "Tamarack Harbor Sleep Institute", "Larkspur Bridge Infusion Center",
)
EPONYMS = (
    ("Wilson", "Wilson disease"), ("Addison", "Addison disease"),
    ("Turner", "Turner syndrome"), ("Graves", "Graves disease"),
    ("Parkinson", "Parkinson disease"), ("Hodgkin", "Hodgkin lymphoma"),
)


class _Text:
    """Append authored text and annotations together; never search for entities."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.spans: list[dict[str, Any]] = []
        self.length = 0

    @property
    def text(self) -> str:
        return "".join(self.parts)

    def add(self, value: str, label: str | None = None) -> "_Text":
        if label is not None:
            if label not in ENTITY_TYPES or not value:
                raise ValueError("Invalid authored entity")
            self.spans.append({"start": self.length, "end": self.length + len(value), "label": label})
        self.parts.append(value)
        self.length += len(value)
        return self


def identity_group(full_name: str) -> str:
    return hashlib.sha256(unicodedata.normalize("NFC", full_name).casefold().encode("utf-8")).hexdigest()


def _is_development(full_name: str) -> bool:
    return int(identity_group(full_name)[:8], 16) % 10 in {8, 9}


def _identities(count: int, seed: int) -> list[str]:
    candidates = [f"{first} {last}" for first in GIVEN for last in SURNAMES]
    random.Random(seed).shuffle(candidates)
    accepted = [name for name in candidates if _is_development(name)]
    if count > len(accepted):
        # Additional normal three-part names allow larger requested corpora.
        candidates = [f"{first} {middle} {last}" for first in GIVEN for middle in GIVEN
                      if first != middle for last in SURNAMES]
        random.Random(seed + 1).shuffle(candidates)
        accepted.extend(name for name in candidates if _is_development(name))
    if count > len(accepted):
        raise ValueError("Requested corpus exceeds the independent identity vocabulary")
    return accepted[:count]


def _clinician(number: int) -> tuple[str, str]:
    surname, condition = EPONYMS[number % len(EPONYMS)]
    candidates = [f"{first} {surname}" for first in GIVEN if _is_development(f"{first} {surname}")]
    return candidates[(number // len(EPONYMS)) % len(candidates)], condition


def _relative(number: int) -> str:
    first = GIVEN[(number * 7 + 3) % len(GIVEN)]
    last = SURNAMES[(number * 13 + 5) % len(SURNAMES)]
    for offset in range(len(GIVEN)):
        middle = GIVEN[(number * 11 + 13 + offset) % len(GIVEN)]
        candidate = f"{first} {middle} {last}"
        if middle != first and _is_development(candidate):
            return candidate
    raise AssertionError("No development-allocated invented relative identity")


def _values(full_name: str, number: int) -> dict[str, str]:
    street = ("Lantern Walk", "Alder Terrace", "Copperleaf Lane", "Heron Mews")[number % 4]
    dates = ("November 19, 2038", "04/27/2036", "2037-08-06", "February 2039", "2041", "11/06")
    ages = ("90", "92", "94", "97", "101", "ninety-three", "one hundred and two", "91")
    area, line = (202, 303, 415, 503)[number % 4], 100 + number % 100
    phones = (f"({area}) 555-{line:04d}", f"+1 {area} 555 {line:04d}", f"{area}-555-{line:04d}")
    emails = (f"care.{number:05d}@patient.example.org", f"followup+{number:05d}@portal.example.net",
              f"member-{number:05d}@mail.example", f"visit_{number:05d}@care.test")
    urls = (f"https://followup.example.net/visit/{920000 + number}",
            f"https://portal.example.org/record?visit={920000 + number}",
            f"https://care.test/patient/{920000 + number}",
            f"https://handoff.example/note/{920000 + number}")
    postal_base = (97000, 4000, 89000, 53000, 5000, 80000, 87000, 98000)[number % 8]
    return {
        "NAME": full_name,
        "DATE": dates[number % len(dates)],
        "ADDRESS": f"{218 + number % 790} {street}, Unit {2 + number % 31}, {LOCALITIES[number % 8]} {postal_base + number % 99:05d}",
        "PHONE": phones[number % len(phones)],
        "EMAIL": emails[number % len(emails)],
        "MRN": f"MRC-{730000 + number}",
        "SSN": f"900-{10 + number % 80:02d}-{1000 + number % 8999:04d}",
        "ACCOUNT": f"AC{830000000 + number}",
        "ID": f"DEV-{440000 + number}-X",
        "URL": urls[number % len(urls)],
        "IP": f"{('192.0.2', '198.51.100', '203.0.113')[number % 3]}.{1 + number % 254}",
        "AGE": ages[number % len(ages)],
        "LOCATION": LOCALITIES[(number + 3) % len(LOCALITIES)],
        "FACILITY": FACILITIES[number % len(FACILITIES)],
    }


# Each clinical family has its own reason for carrying the selected identifiers.
# Layout changes in a pair represent copied/reformatted same-person material.
FAMILIES = (
    ("home_oxygen", "home-care handoff", "Breathlessness improved after supervised walking. The nasal cannula fit was checked; no skin injury was seen. The home-monitor connection was verified.", ("ADDRESS", "PHONE", "DATE", "AGE", "LOCATION", "FACILITY", "IP")),
    ("infusion_clearance", "infusion checklist", "The infusion was paused for line assessment and resumed after patency was confirmed. No flushing or new rash developed. Coverage clearance was recorded with the visit.", ("MRN", "DATE", "AGE", "FACILITY", "ACCOUNT", "ID", "SSN")),
    ("portal_recovery", "patient-support record", "A caregiver helped restore access to the visit summary. The patient accurately repeated the medication schedule.", ("EMAIL", "PHONE", "URL", "IP", "MRN", "DATE")),
    ("benefit_reconciliation", "coverage reconciliation", "Coverage was reconciled for the ordered mobility aid. The signed request was returned to the referral desk.", ("SSN", "ACCOUNT", "ID", "ADDRESS", "PHONE", "DATE")),
    ("sleep_equipment", "device follow-up", "Mask fit was comfortable while seated. The overnight download showed fewer interruptions and no sustained desaturation.", ("ID", "IP", "URL", "EMAIL", "AGE", "FACILITY")),
    ("imaging_transfer", "radiology transfer note", "The prior images arrived with the referral. The comparison view was reviewed before the scan protocol was chosen.", ("MRN", "DATE", "FACILITY", "LOCATION", "ID", "URL")),
    ("rehab_transport", "rehabilitation transport plan", "The patient managed a supervised pivot using the rail. Transport requires space for the folded walker.", ("ADDRESS", "LOCATION", "PHONE", "AGE", "ACCOUNT", "FACILITY")),
    ("pharmacy_bridge", "medication bridge note", "A short refill was arranged while the medication list was reconciled. Teach-back confirmed the intended morning dose.", ("MRN", "EMAIL", "PHONE", "DATE", "ID", "ACCOUNT")),
    ("records_release", "records routing slip", "The requested therapy summaries were collated. An obsolete duplicate cover page was removed from the packet.", ("SSN", "ADDRESS", "EMAIL", "URL", "LOCATION", "MRN")),
    ("remote_wound", "remote wound follow-up", "The image showed a dry dressing edge with no spreading redness. The caregiver described steady improvement in comfort.", ("IP", "URL", "PHONE", "AGE", "DATE", "LOCATION")),
    ("social_support", "support-service intake", "The patient requested meal delivery during recovery. The intake worker confirmed access to a working refrigerator.", ("SSN", "ADDRESS", "ACCOUNT", "AGE", "FACILITY", "EMAIL")),
    ("referral_directory", "referral verification", "The receiving service confirmed that the referral was complete. The preferred route for follow-up was recorded.", ("ID", "MRN", "FACILITY", "LOCATION", "PHONE", "DATE")),
    ("billing_correction", "administrative correction", "A duplicate transport charge was identified. The adjustment was linked to the existing visit without changing the clinical plan.", ("SSN", "ACCOUNT", "MRN", "EMAIL", "DATE", "ADDRESS")),
    ("telemetry_setup", "home-monitor setup", "The sensor transmitted a clean resting trace. Instructions for replacing the adhesive patch were reviewed aloud.", ("IP", "ID", "URL", "ACCOUNT", "AGE", "EMAIL")),
    ("psychiatric_review", "psychiatric video follow-up", "The 42yo patient reported fewer early awakenings and a more regular daily routine. Speech was organized and the patient remained engaged throughout the video visit. The agreed sleep diary was reviewed together.", ("PHONE", "EMAIL", "URL", "IP", "MRN", "DATE")),
    ("obstetric_followup", "postpartum examination", "The 30yo patient described improving comfort while feeding. The incision was dry and well approximated, without surrounding redness. The nurse reviewed positioning and arranged a follow-up weight check for the infant.", ("DATE", "MRN", "PHONE", "ADDRESS", "LOCATION", "FACILITY")),
    ("dental_assessment", "dental examination and benefits record", "The patient described tenderness while chewing on the right. Examination found localized gingival irritation without facial swelling. A soft brush technique was demonstrated and the restorative treatment plan was reviewed alongside the benefits record.", ("AGE", "DATE", "MRN", "ID", "ACCOUNT", "SSN")),
    ("genetics_counseling", "genetics consultation", "A three-generation pedigree was reviewed during counseling about inherited cardiomyopathy. The patient asked how an uncertain result could affect relatives. The counselor explained the laboratory report and confirmed the agreed return visit.", ("DATE", "AGE", "ID", "MRN", "SSN", "FACILITY")),
)
FIELD_NAMES = {
    "NAME": "Patient", "DATE": "Patient encounter date", "ADDRESS": "Delivery address",
    "PHONE": "Return telephone", "EMAIL": "Patient email", "MRN": "Chart number",
    "SSN": "Social Security number", "ACCOUNT": "Patient account", "ID": "Assigned identifier",
    "URL": "Private visit link", "IP": "Patient gateway address", "AGE": "Patient age",
    "LOCATION": "Home locality", "FACILITY": "Receiving institution",
}

NEGATIVE_NOTES = (
    ("airway_review", "Breathing was comfortable at rest. Respiratory rate 16 and pulse 92. No wheeze was heard. The inhaler technique was repeated twice, and the patient demonstrated a slow inspiration."),
    ("joint_followup", "Knee discomfort improved with a warm pack. They walked to the doorway without buckling. Strength was 4/5 on both sides; a further 3 weeks of home exercise were discussed."),
    ("thyroid_teaching", "The discussion compared Graves disease with painless thyroiditis. A 2021 review and a 2018 consensus statement were used as public references. The uptake pattern in the published figure was reviewed together."),
    ("interpreter_note", "The patient used a Spanish interpreter and repeated the plan in their own words. She requested large-print instructions. The clinic supplied a clear copy and checked understanding."),
    ("wound_round", "The wound bed was pink with scant brown drainage. A pale rose-colored mark was unchanged. The dressing remained dry after gentle repositioning. No odor was present."),
    ("neurology_review", "Testing for Parkinson disease was discussed alongside the differential for essential tremor. Rapid alternating movements were smooth. The patient held a cup steadily with either hand."),
    ("flowsheet_review", "Room 92 | pulse 89 | oxygen 97% | temperature 36.7 C. The 89F patient was comfortable. Two cups of water were tolerated without coughing; observation continued for 20 minutes."),
    ("medication_education", "May continue the existing schedule. The patient said they will ask for help if dizzy. There was hope that sleep would improve. A dose of 5 mg was checked against the current package."),
    ("specimen_note", "Specimen rack 92F passed the control check. Sodium was 139 mmol/L and glucose was 104 mg/dL. The reference interval and instrument flag were reviewed by the laboratory team."),
    ("exercise_review", "After 6 weeks of walking practice the patient could climb a flight of stairs. They paused briefly at the top and recovered within 2 minutes. No chest tightness occurred."),
    ("endocrine_review", "The educational summary covered Addison disease, Turner syndrome and Wilson disease. The published diagram compared young and old control samples. White cells were assessed in the accompanying laboratory worksheet."),
    ("routine_handoff", "The hospital team checked the infusion pump and the rehabilitation clinic received the care plan. She was alert, her speech was clear, and they agreed to continue the same precautions."),
)


def _row(text: _Text, *, identifier: str, name: str, family: str, template: str,
         pair_id: str, genre: str, layout: str, phenomena: list[str], cohort: str = "realistic",
         diagnostic_slice: str = "ordinary", canonical_people: tuple[str, ...] = ()) -> dict[str, Any]:
    groups = {identity_group(person) for person in canonical_people}
    if any(span["label"] == "NAME" for span in text.spans):
        groups.add(identity_group(name))
    return {
        "id": identifier, "text": text.text, "spans": text.spans, "split": "validation",
        "suite": "development", "family": family, "annotation_policy": POLICY,
        "identity_group": identity_group(name), "identity_groups": sorted(groups), "pair_id": pair_id,
        "template_family": f"remediation_dev.{template}", "phenomena": sorted(set(phenomena)),
        "cohort": cohort, "genre": genre, "layout": layout, "diagnostic_slice": diagnostic_slice,
    }


def _short_positive(name: str, pair: int, variant: int, seed: int) -> dict[str, Any]:
    family, genre, clinical, labels = FAMILIES[pair % len(FAMILIES)]
    values = _values(name, pair)
    specific_facilities = {
        "psychiatric_review": "Juniper Lantern Behavioral Health Center",
        "obstetric_followup": "Willow Arc Maternity Pavilion",
        "dental_assessment": "Copperleaf Community Dental Clinic",
        "genetics_counseling": "Alder Quay Genomic Medicine Clinic",
    }
    if family in specific_facilities:
        values["FACILITY"] = specific_facilities[family]
    if "AGE" in labels and pair % 5 == 3:
        values["AGE"] = ("34", "57", "72", "89")[(pair // 5) % 4]
    b = _Text()
    phenomena = ["paired_copy", "ordinary_pronouns", "clinical_hard_negatives"]
    if family in {"psychiatric_review", "obstetric_followup"}:
        phenomena += ["ordinary_age"]
    layout = ("prose", "pipe_table", "field_sheet", "message_thread")[(pair + variant) % 4]
    word_name = pair % 12 == 0
    alias = WORD_NAMES[(pair // 12) % len(WORD_NAMES)] if word_name else name.split()[0]
    unusual = word_name and alias in {"Care", "He", "She", "They"}
    ocr = pair % 17 == 8 and not word_name
    cohort = "unusual_name" if unusual else "realistic"
    diagnostic_slice = "unusual_name" if unusual else ("ordinary_name_contrast" if word_name else ("ocr_stress" if ocr else "ordinary"))
    if not unusual and not ocr:
        phenomena += ["normal_spelling"]
    if word_name:
        phenomena += ["explicit_wordlike_name", "alias_before_full_name", "same_note_homograph"]
        if unusual:
            phenomena += ["unusual_nominal_name"]
        b.add('Preferred personal name: "').add(alias, "NAME").add('". The chart owner is ')
        ordinary_context = {
            "Hope": "There is hope that symptoms will settle.", "May": "The plan may change after review.",
            "Rose": "A rose-colored area remained unchanged.", "Will": "They will request help if symptoms change.",
            "Care": "Care instructions were reviewed.", "He": "He requested a printed copy.",
            "She": "She requested a printed copy.", "They": "They requested a printed copy.",
        }
        b.add(name, "NAME").add(". " + ordinary_context[alias] + "\n")
    elif pair % 9 == 1:
        phenomena += ["alias_before_full_name"]
        b.add("The patient asked to be called ").add(alias, "NAME").add("; the full chart name is ")
        b.add(name, "NAME").add(".\n")
    elif pair % 9 == 2:
        phenomena += ["lowercase_name"]
        b.add("Patient name as entered on the referral: ").add(name.lower(), "NAME").add(".\n")
    elif pair % 9 == 3:
        phenomena += ["initial_name"]
        b.add("The chart identifies ").add(name, "NAME").add("; the patient initialed the slip as ")
        b.add(f"{name[0]}. {name.split(' ', 1)[1]}", "NAME").add(".\n")
    else:
        b.add("Patient: ").add(name, "NAME").add(".\n")
    if ocr:
        phenomena += ["ocr_stress", "ocr_name"]
        # This explicitly flagged OCR transcription is outside the ordinary slice.
        first, rest = name.split(" ", 1)
        b.add("Scanned patient-name field: ").add(first[:2] + " " + first[2:] + " " + rest, "NAME").add(".\n")
    b.add(clinical + " ").add("They described the plan correctly and requested a printed copy.\n")
    clinician, condition = _clinician(pair)
    canonical_people = [clinician]
    b.add("Dr. ").add(clinician, "NAME").add(", MD, reviewed the note. ")
    b.add(f"The education sheet discussed {condition} and its presenting symptoms.\n")
    phenomena += ["honorific_credential_exclusion", "eponym_surname_contrast"]
    if pair % 5 == 2:
        relative = _relative(pair)
        canonical_people.append(relative)
        b.add("The patient's daughter, ").add(relative, "NAME").add(", requested an extra copy of the plan.\n")
        phenomena += ["relative_name"]
    if pair % 4 == 3:
        b.add("Dr. ").add(clinician.rsplit(" ", 1)[1], "NAME").add(" confirmed the follow-up arrangements.\n")
        phenomena += ["surname_only_context"]
    if layout == "pipe_table":
        b.add("Verified field | Entered value\n--- | ---\n")
    elif layout == "message_thread":
        b.add("Care coordination message — verified details follow.\n")
    for label in labels:
        prefix = FIELD_NAMES[label] + (" | " if layout == "pipe_table" else ": ")
        b.add(prefix)
        value = values[label]
        entity_label = None if label == "AGE" and value.isdigit() and int(value) <= 89 else label
        if label == "AGE" and value.isdigit() and pair % 3 != 0:
            b.add(value, entity_label).add("F" if pair % 2 else "yo")
            phenomena += ["compact_age"]
        elif label == "ADDRESS" and pair % 5 == 2:
            b.add(value.replace(", Unit", "\nUnit").replace(", " + LOCALITIES[pair % 8], "\n" + LOCALITIES[pair % 8]), label)
            phenomena += ["multiline_address"]
        else:
            b.add(value, entity_label)
        b.add("; " if layout == "prose" else "\n")
        if label == "DATE" and pair % 6 in {3, 4, 5}:
            phenomena += ["partial_date"]
        if label == "AGE" and not value.isdigit():
            phenomena += ["written_age"]
        if label == "AGE" and entity_label is None:
            phenomena += ["ordinary_age"]
    if variant:
        b.add("\nCopy-forward check: " + ("the patient named " if word_name else ""))
        b.add(alias if word_name else name, "NAME")
        b.add(" confirmed the same instructions. The earlier summary remains in the chart.\n")
        phenomena += ["repeated_name", "copy_forward"]
    if pair % 7 == 4:
        b.add("Patient hospitalized in ").add("2021", "DATE")
        b.add("; the public review cited 2021 as its publication year. Symptoms lasted 3 years.\n")
        phenomena += ["patient_year_public_year_contrast", "duration_negative"]
    return _row(b, identifier=f"remediation-dev-{seed}-short-{pair:05d}-{variant}", name=name,
                family=family, template=f"{family}.{layout}", pair_id=f"dev-{seed}-pair-{pair:05d}",
                genre=genre, layout=layout, phenomena=phenomena, cohort=cohort,
                diagnostic_slice=diagnostic_slice, canonical_people=tuple(canonical_people))


def _negative(index: int, seed: int) -> dict[str, Any]:
    family, prose = NEGATIVE_NOTES[index % len(NEGATIVE_NOTES)]
    rng = random.Random(seed * 1000003 + index)
    pulse, breaths, pain = rng.randrange(58, 99), rng.randrange(12, 21), rng.randrange(0, 7)
    saturation, temperature = rng.randrange(95, 101), rng.randrange(361, 375) / 10
    action = rng.choice(("A warm pack was offered for comfort.", "The chair was adjusted for a supported position.",
                         "Fluids were placed within reach.", "The patient practiced a slow supported transfer.",
                         "The patient completed a short seated exercise sequence.", "The dressing edges were checked again."))
    response = rng.choice(("The patient tolerated the activity comfortably.", "They requested a short rest afterward.",
                           "The patient described the next step in their own words.", "They agreed to continue the current plan."))
    layout = ("clinical_prose", "soap_excerpt", "observation_table", "nursing_bullets")[index % 4]
    vitals = f"Pulse {pulse}; respiratory rate {breaths}; oxygen {saturation}%; temperature {temperature:.1f} C; pain {pain}/10."
    if layout == "soap_excerpt":
        body = f"Assessment: {prose}\nObservation: {vitals}\nPlan: {action} {response}"
    elif layout == "observation_table":
        body = f"Narrative | {prose}\nPulse | {pulse}\nRespirations | {breaths}\nOxygen | {saturation}%\nTemperature | {temperature:.1f} C\nPain | {pain}/10\nCare | {action} {response}"
    elif layout == "nursing_bullets":
        body = f"- {prose}\n- {vitals}\n- {action}\n- {response}"
    else:
        body = f"{prose}\n{vitals} {action} {response}"
    b = _Text().add(body)
    phenomena = ["negative_only", "clinical_hard_negatives"]
    if re.search(r"\b(?:he|she|they|him|her|them|his|their|theirs|hers)\b", body, re.IGNORECASE):
        phenomena.append("ordinary_pronouns")
    record_key = f"non-person development negative {seed} {index}"
    while not _is_development(record_key):
        record_key += "."
    return _row(b, identifier=f"remediation-dev-{seed}-negative-{index:05d}",
                name=record_key, family=f"negative_{family}",
                template=f"negative.{family}", pair_id=f"dev-{seed}-negative-{index:05d}",
                genre="clinical note without identifiers", layout=layout,
                phenomena=phenomena)


LONG_PROSE = (
    "The bedside assessment was repeated after a period of rest. Breathing remained easy, the dressing stayed dry, and the patient could explain when to request assistance. The care team reviewed the same precautions during handoff.\n",
    "Mobility practice began from a seated position. The patient stood with the rail, paused to establish balance, and walked at a comfortable pace. No dizziness or new discomfort occurred. The return transfer required verbal guidance only.\n",
    "Medication reconciliation continued with a read-back of the current schedule. The patient described the purpose of the morning dose and identified the package used at home. The nurse clarified the difference between scheduled and as-needed use.\n",
    "Nutrition review found that small portions were better tolerated. Fluids were offered between tasks, and the patient took several comfortable sips without coughing. The caregiver was encouraged to keep the next meal within easy reach.\n",
    "The copied progress section was reviewed against the current examination. Skin was warm, speech was clear, and attention remained steady throughout the conversation. The assessment and precautions were unchanged after the repeat observation.\n",
    "The therapy checklist recorded smooth hand movements, a secure grip on the rail, and an unhurried transfer. Exercises were demonstrated in a comfortable range. The patient repeated each movement and requested a rest between sets.\n",
)


def _crossing(b: _Text, target: int, value: str, label: str, encoding: Any,
              prose_index: int) -> dict[str, Any]:
    """Place a complete authored entity across a nominal global-token join.

    A compact serial-observation flowsheet absorbs at most a few dozen tokens.
    Candidate placement depends on tokenizer offsets only, never model output.
    """
    while True:
        paragraph = LONG_PROSE[prose_index % len(LONG_PROSE)]
        if len(encoding.encode(b.text + paragraph)) > target - 45:
            break
        b.add(paragraph)
        prose_index += 1
    base = b.text
    statuses = ("awake", "alert", "calm", "comfortable", "resting", "stable")
    for observations in range(100):
        strip = "".join(" " + statuses[(prose_index + i) % len(statuses)] for i in range(observations))
        prefix = "Flow strip (serial status samples):" + strip + ".\nVerified entry: "
        candidate = base + prefix + value + ".\n"
        tokens = encoding.encode(candidate)
        if len(tokens) <= target:
            continue
        _, offsets = encoding.decode_with_offsets(tokens)
        start = len(base) + len(prefix)
        if start < offsets[target] < start + len(value):
            b.add(prefix).add(value, label).add(".\n")
            return {"token_boundary": target, "span_start": start, "span_end": start + len(value),
                    "label": label, "boundary_character": offsets[target]}
        if start > offsets[target]:
            break
    raise AssertionError(f"Cannot place authored {label} across token {target}")


def _long(name: str, index: int, seed: int, encoding: Any) -> dict[str, Any]:
    values = _values(name, 10000 + index)
    b = _Text().add("INTERDISCIPLINARY CONTINUITY RECORD\nPatient: ").add(name, "NAME")
    b.add("\nThe following sections combine bedside assessments and copied care-coordination entries. They could explain the precautions in their own words.\n")
    # Every long record exercises all labels and begins with a realistic routing block.
    for label in ENTITY_TYPES:
        if label != "NAME":
            b.add(FIELD_NAMES[label] + ": ").add(values[label], label).add("\n")
    target_length = (1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 5888)[index % 10]
    narrative = index % 4 == 0
    boundaries = [] if narrative else sorted({512, *range(384, target_length - 80, 256)})
    checks = []
    for ordinal, boundary in enumerate(boundaries):
        label = ("NAME", "ADDRESS", "DATE", "FACILITY")[ordinal % 4]
        value = values[label]
        if label == "DATE":
            value = "November 19, 2038"  # A multi-token clinical date can straddle a join.
        checks.append(_crossing(b, boundary, value, label, encoding, index + ordinal))
        if ordinal % 3 == 1:
            b.add("Copy-forward contact verified again for ").add(name, "NAME").add(".\n")
    cycle = 0
    while len(encoding.encode(b.text)) < target_length - 55:
        if narrative:
            section = ("Nursing course", "Therapy reassessment", "Medication review", "Care coordination",
                       "Discharge planning", "Copied interval history")[cycle % 6]
            b.add(f"\n{section}\n")
        b.add(LONG_PROSE[(index + cycle) % len(LONG_PROSE)])
        if narrative:
            b.add(f"Observation: pulse {64 + (index * 3 + cycle) % 27}; respiratory rate {12 + cycle % 7}; pain {cycle % 4}/10. ")
            b.add(("The bedside chair was adjusted for comfort.\n", "A warm drink was offered after the exercise.\n",
                   "The call button remained within reach.\n")[(index + cycle) % 3])
            if cycle % 3 == 0:
                b.add("These instructions were reviewed again with ").add(name, "NAME").add(".\n")
            if cycle % 5 == 2:
                b.add("The copied review from ").add(values["DATE"], "DATE")
                b.add(" was checked against the current assessment.\n")
        cycle += 1
    b.add("FINAL ROUTING CHECK\nThe completed packet for ").add(name, "NAME")
    b.add(" should go to ").add(values["ADDRESS"], "ADDRESS")
    b.add(". Return call: ").add(values["PHONE"], "PHONE").add(".\n")
    result = _row(b, identifier=f"remediation-dev-{seed}-long-{index:04d}", name=name,
                  family=("long_rehabilitation", "long_home_care", "long_transition", "long_outpatient")[index % 4],
                  template=f"long.{'narrative' if narrative else 'seam_stress'}_{index % 4}", pair_id=f"dev-{seed}-long-{index:04d}",
                  genre="longitudinal clinical record", layout="clinical_sections" if narrative else "mixed_sections_and_flowsheet",
                  phenomena=["long_document", "copy_forward", "repeated_name", "phi_in_tail", "ordinary_pronouns"] +
                            (["long_narrative"] if narrative else ["seam_stress", "nominal_join_crossing", "window_512_crossing"]),
                  cohort="long_document", diagnostic_slice="long_narrative" if narrative else "seam_stress")
    tokens = encoding.encode(result["text"])
    _, final_offsets = encoding.decode_with_offsets(tokens)
    for check in checks:
        check["boundary_character"] = final_offsets[check["token_boundary"]]
        if not check["span_start"] < check["boundary_character"] < check["span_end"]:
            raise AssertionError("Later text changed a deliberate token crossing")
    if not 1500 <= len(tokens) <= 6000:
        raise AssertionError(f"Long document has unexpected token length {len(tokens)}")
    result.update(tokenizer=TOKENIZER, token_count=len(tokens), boundary_checks=checks,
                  intended_token_length=target_length)
    if not narrative and {check["token_boundary"] for check in checks} != {512, *range(384, len(tokens), 256)}:
        raise AssertionError("A nominal join lacks a deliberate PHI crossing")
    return result


def build_rows(short_count: int = 600, long_count: int = 40, seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    """Build independent diagnostic rows without writing any corpus or model data."""
    if isinstance(short_count, bool) or isinstance(long_count, bool) or not isinstance(short_count, int) or not isinstance(long_count, int):
        raise ValueError("Counts must be nonnegative integers")
    if short_count < 0 or long_count < 0 or short_count + long_count == 0:
        raise ValueError("Request at least one document and nonnegative counts")
    negative_count = short_count // 5
    positive_count = short_count - negative_count
    pair_count = (positive_count + 1) // 2
    names = _identities(pair_count + long_count, seed)
    positives = [_short_positive(names[i // 2], i // 2, i % 2, seed) for i in range(positive_count)]
    negatives = [_negative(i, seed) for i in range(negative_count)]
    rows = []
    for i in range(short_count):
        rows.append(negatives.pop(0) if i % 5 == 0 and negatives else positives.pop(0))
    if long_count:
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "tiktoken"))
        encoding = tiktoken.get_encoding(TOKENIZER)
        rows.extend(_long(names[pair_count + i], i, seed, encoding) for i in range(long_count))
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    pair_groups: dict[str, str] = {}
    for row in rows:
        if row["id"] in ids:
            raise AssertionError("Duplicate row identifier")
        ids.add(row["id"])
        previous_end = 0
        for span in row["spans"]:
            if not 0 <= span["start"] < span["end"] <= len(row["text"]) or span["label"] not in ENTITY_TYPES:
                raise AssertionError("Invalid authored annotation")
            if span["start"] < previous_end:
                raise AssertionError("Overlapping or unordered annotations")
            previous_end = span["end"]
        group = pair_groups.setdefault(row["pair_id"], row["identity_group"])
        if group != row["identity_group"]:
            raise AssertionError("An identity pair spans groups")


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2 if pretty else None) + "\n").encode("utf-8")


def generate_corpus(output_path: str | Path, short_count: int = 600, long_count: int = 40,
                    seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Write immutable JSONL, manifest, and independent-author declaration.

    Existing identical bytes are accepted. Any differing output or sidecar is
    rejected before writing; exclusive creation also protects against races.
    """
    rows = build_rows(short_count=short_count, long_count=long_count, seed=seed)
    payload = b"".join(_json_bytes(row) for row in rows)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    declaration = {
        "schema_version": 1, "author_role": "independent synthetic development author",
        "allowed_context": ALLOWED_CONTEXT, "training_generator_read": False,
        "training_rows_read": False, "prior_development_rows_read": False,
        "model_implementation_read": False, "predictions_or_failures_read": False,
        "sealed_qualification_material_read": False,
        "synthetic_only": True, "real_patient_data": False,
        "used_for_training": False, "used_for_checkpoint_selection": True,
        "independent_final": False, "annotation_policy": POLICY,
        "scope": "Development diagnostics and checkpoint selection only; no independent qualification claim.",
        "generator_sha256": source_sha,
        "synthetic_identifier_schemes": {
            "MRN": "MRC- followed by 730000 plus the invented identity ordinal",
            "ACCOUNT": "AC followed by 830000000 plus the invented identity ordinal",
            "ID": "DEV- followed by 440000 plus the invented identity ordinal and -X",
            "SSN": "Invented 900-XX-XXXX values; not copied from any person or record",
            "PHONE": "US 555-0100 through 555-0199 in parenthesized, hyphenated and +1 formats",
            "EMAIL_URL": "Reserved example.org, example.net, .example and .test domains only",
            "IP": "TEST-NET 192.0.2.0/24, 198.51.100.0/24 and 203.0.113.0/24",
        },
    }
    declaration_payload = _json_bytes(declaration, pretty=True)
    labels = Counter(span["label"] for row in rows for span in row["spans"])
    manifest = {
        "schema_version": 1, "sha256": hashlib.sha256(payload).hexdigest(), "rows": len(rows),
        "label_support": {label: labels[label] for label in ENTITY_TYPES},
        "annotation_policy": POLICY, "suite": "development", "split": "validation",
        "generator": "outis.data.generators.remediation_development", "generator_sha256": source_sha,
        "seed": seed, "short_count": short_count, "long_count": long_count,
        "negative_documents": sum(not row["spans"] for row in rows),
        "cohort_counts": dict(sorted(Counter(row["cohort"] for row in rows).items())),
        "diagnostic_slice_counts": dict(sorted(Counter(row["diagnostic_slice"] for row in rows).items())),
        "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        "phenomenon_counts": dict(sorted(Counter(p for row in rows for p in row["phenomena"]).items())),
        "used_for_training": False, "used_for_checkpoint_selection": True, "independent_final": False,
        "identity_group_rule": "sha256(NFC(full_name).casefold().encode('utf-8')); first 8 hex modulo 10 in {8,9}",
        "identity_pair_policy": "Copies, aliases and layout variants retain the canonical full-name group.",
        "tokenizer": TOKENIZER,
        "tiktoken_version": tiktoken.__version__,
        "long_document_scope": "One quarter independently authored sectioned narratives; remaining rows deliberately padded serial-observation seam stress. Both are synthetic, not naturally sampled clinical records; report their slices separately.",
        "long_document_token_range": [min((r["token_count"] for r in rows if "token_count" in r), default=0),
                                      max((r["token_count"] for r in rows if "token_count" in r), default=0)],
        "author_declaration_sha256": hashlib.sha256(declaration_payload).hexdigest(),
        "scope": "Invented development diagnostics; synthetic results do not establish clinical readiness.",
    }
    output = Path(output_path)
    artifacts = [(output, payload),
                 (output.with_suffix(output.suffix + ".manifest.json"), _json_bytes(manifest, pretty=True)),
                 (output.with_suffix(output.suffix + ".author-declaration.json"), declaration_payload)]
    for path, content in artifacts:
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite frozen development artifact: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, content in artifacts:
        try:
            with path.open("xb") as handle:
                handle.write(content)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(f"Concurrent differing development artifact: {path}") from None
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--short-count", type=int, default=600)
    parser.add_argument("--long-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    print(json.dumps(generate_corpus(args.output, args.short_count, args.long_count, args.seed), indent=2))


if __name__ == "__main__":
    main()
