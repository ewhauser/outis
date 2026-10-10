"""Versioned, original contextual PHI training data (CC0-1.0 prose).

Only training code and policy are reused; no corpus is loaded by this module.
The v1 generator and its frozen artifacts are retained unchanged. This is a
finite synthetic curriculum, not a population sample or qualification bank.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import random
import re
import time

from outis.data.annotation_policy import POLICY_VERSION, annotation_errors, canonical_identity_group, identity_split
from outis.data.generators.remediation import NAME_POOLS, COMMON_GIVEN, EPONYMS, UNUSUAL_NAMES, TextBuilder, get_encoding, token_metadata
from outis.types import ENTITY_TYPES
from outis.data.v12.contexts import CLINICAL_CONTEXTS, NEGATIVE_CONTEXTS, PUBLIC_CONTEXT_TEMPLATES
from outis.paths import PACKAGE_ROOT


VERSION = "12.0"
SEED = 12037
YEARS = tuple(range(1980, 2027))
EXTRA_NAME_POOLS = (
    (("Elżbieta", "Łukasz", "Michał", "Zofia"), ("Wójcik", "Zielińska", "Dąbrowski", "Kowalczyk")),
    (("İpek", "Çağla", "Ömer", "Gül"), ("Yılmaz", "Öztürk", "Şahin", "Aydın")),
    (("Søren", "Åsa", "Bjørn", "Ingrid"), ("Lindström", "Sørensen", "Berg", "Nyström")),
    (("Αλέξανδρος", "Ελένη", "Νίκος", "Μαρία"), ("Παπαδάκη", "Γεωργίου", "Νικολάου", "Δημητρίου")),
    (("Олена", "Ірина", "Андрій", "Тарас"), ("Коваль", "Мельник", "Бондар", "Ткаченко")),
    (("李", "陈", "张", "林"), ("明华", "文静", "志远", "晓梅")),
    (("आशा", "नीलम", "अरुण", "दीपा"), ("शर्मा", "वर्मा", "कपूर", "मेहता")),
)
SURNAME_PHRASES = ("de la Vega", "dos Santos", "van der Meer", "al-Hakim", "O’Connor", "García-López")
TOWNS = ("Alder Sound", "Birch Meadow", "Foxglove Reach", "Hazel Bend", "Mallow Ridge", "Orchard Basin", "Pinewater", "Quince Vale", "Sorrel Point", "Yarrow Brook", "Érable Heights", "Arroyo Claro")
STATES = ("ME", "Vermont", "AZ", "Iowa", "CA", "Minnesota", "NC", "Kansas", "RI", "Colorado")
STREETS = ("Foxglove Terrace", "Hazel Crescent", "Cypress Spur", "Tamarack Way", "Sorrel Avenue", "Quince Row", "Juniper Vista", "Laurel Lane", "Érable Road", "Arroyo Court")
FACILITY_BASES = ("Hazel Compass", "Sorrel House", "Foxglove Shore", "Alder Beacon", "Mallow Vista", "Quince Beacon", "Pinewater Commons", "Yarrow Crescent", "Érable Crest", "Cypress Lantern")
FACILITY_TYPES = ("Regional Hospital", "Medical Pavilion", "Community Health Center", "Imaging Institute", "Recovery Clinic", "Surgical Centre", "Nursing Residence", "Family Practice", "Day Treatment Unit", "Dialysis Center")
LAYOUTS = ("prose", "soap", "letter", "handoff", "fields", "table", "bullet", "compact")
CLINICAL_HEADINGS = ("Assessment", "Interval symptoms", "Care plan", "Examination", "Medication review", "Follow-up", "History", "Impression", "Treatment response", "Discharge planning")
PRONOUN_CONTEXTS = (
    "They explained the next step and agreed to continue.",
    "She asked whether the same precautions apply at home.",
    "He repeated the instructions in his own words.",
    "We checked understanding before ending the discussion.",
    "I reviewed the written instructions with the patient.",
    "You may stop the activity if symptoms recur, as discussed.",
)
FIELD_LABELS = {
    "DATE": ("Encounter date", "Date of assessment", "Last visit", "Reviewed on"),
    "ADDRESS": ("Postal address", "Address for correspondence", "Current address", "Aftercare address"),
    "PHONE": ("Telephone", "Telephone for follow-up", "Preferred callback", "Call number"),
    "EMAIL": ("Email", "Email for replies", "Correspondence email", "Contact by email"),
    "MRN": ("MRN", "Patient chart number", "Record identifier", "Medical record number"),
    "SSN": ("SSN", "Social security number", "Social-security number"),
    "ACCOUNT": ("Billing account", "Account reference", "Patient account number"),
    "ID": ("Encounter ID", "Insurance member identifier", "Case ID", "Authorization ID"),
    "URL": ("Personal portal URL", "Link to visit record", "Private chart link"),
    "IP": ("Remote-session IP", "Patient device IP", "Telehealth client address"),
    "LOCATION": ("Home town", "Current locality", "Town of residence"),
    "FACILITY": ("Facility", "Clinic of referral", "Receiving hospital", "Outpatient site"),
}
WORD_NEGATIVES = {
    "Rose": "The pale rose tint faded after the dressing was changed.",
    "May": "Activity may increase once the discomfort subsides.",
    "Will": "The team will review the dose after the next blood draw.",
    "Grace": "There is a brief grace period for rescheduling the therapy visit.",
    "Hope": "There is hope that the swallowing symptoms will settle.",
    "Mark": "Mark the symptom severity before each scheduled dose.",
    "Joy": "The patient described joy after returning to the usual routine.",
    "Summer": "Dry skin was worse during the summer months.",
}


def identity(rng: random.Random, *, kind: str = "ordinary") -> dict:
    """Invent a full identity before assigning its fixed shared partition."""
    for _ in range(10000):
        pools = EXTRA_NAME_POOLS if rng.random() < .13 else NAME_POOLS
        firsts, lasts = rng.choice(pools)
        first, last = rng.choice(firsts), rng.choice(lasts)
        if kind == "collision":
            first, last = rng.choice(COMMON_GIVEN), rng.choice(EPONYMS)[0]
        elif kind == "unusual":
            first = rng.choice(UNUSUAL_NAMES)
        elif pools is NAME_POOLS and rng.random() < .10:
            last = rng.choice(SURNAME_PHRASES)
        middle = f" {rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}." if pools is NAME_POOLS and rng.random() < .16 else ""
        full = f"{first}{middle} {last}"
        group = canonical_identity_group(full)
        if identity_split(group) == "train":
            return {"given": first, "surname": last, "full": full, "group": group, "kind": kind}
    raise RuntimeError("Could not generate a training identity")


def values(rng: random.Random) -> dict:
    year = rng.choice(YEARS)
    day = date(year, rng.randrange(1, 13), rng.randrange(1, 29))
    form = rng.randrange(7)
    calendar = (day.isoformat(), day.strftime("%B %d, %Y"), day.strftime("%d %b %Y"), day.strftime("%m/%d/%Y"), day.strftime("%B %Y"), str(year), day.strftime("%m/%d"))[form]
    town, state = rng.choice(TOWNS), rng.choice(STATES)
    locality = (town, f"{town}, {state}", f"North {town}", f"{town}, {state}")[rng.randrange(4)]
    house = rng.randrange(10, 9900)
    street = rng.choice(STREETS)
    unit = rng.choice(("", f", Apt {rng.randrange(1, 90)}", f", Unit {rng.randrange(1, 45)}B"))
    address = f"{house} {street}{unit}, {town}, {state} {rng.randrange(10000, 99999)}"
    if rng.random() < .18:
        address = address.replace(", " + town, "\n" + town)
    domain = rng.choice(("notes.test", "aftercare.example", "example.org", "messages.example.net", "visits.example.com"))
    serial = rng.randrange(10000000, 99999999)
    phone = f"{rng.choice((202, 212, 303, 415, 617, 703))}55501{rng.randrange(100):02d}"
    phones = (f"({phone[:3]}) {phone[3:6]}-{phone[6:]}", f"+1 {phone[:3]} {phone[3:6]} {phone[6:]}", f"{phone[:3]}.{phone[3:6]}.{phone[6:]}", f"{phone[:3]}-{phone[3:6]}-{phone[6:]} ext {rng.randrange(10, 990)}")
    return {
        "DATE": calendar, "year": year, "date_form": form,
        "ADDRESS": address, "LOCATION": locality,
        "FACILITY": f"{rng.choice(FACILITY_BASES)} {rng.choice(FACILITY_TYPES)}",
        "PHONE": rng.choice(phones), "EMAIL": f"care-{serial}@{domain}",
        "MRN": f"{rng.choice(('MRN', 'CH', 'REC'))}-{serial}",
        "SSN": f"000-{rng.randrange(10, 99)}-{rng.randrange(1000, 9999)}",
        "ACCOUNT": f"{rng.choice(('AC', 'BILL', 'PAY'))}/{serial}/{rng.randrange(10, 99)}",
        "ID": f"{rng.choice(('VIS', 'AUTH', 'MEM'))}-{serial}-{rng.choice('ABCDEFGH')}",
        "URL": f"https://{domain}/{rng.choice(('record', 'consultation', 'patient'))}/{serial}",
        "IP": f"{rng.choice(('192.0.2', '198.51.100', '203.0.113'))}.{rng.randrange(1, 255)}",
    }


def _record_evidence(row_evidence: dict, tag: str, b: TextBuilder, begin: int) -> None:
    row_evidence.setdefault(tag, []).append({"start": begin, "end": len(b.text)})


def _name(b: TextBuilder, owner: dict, style: int, evidence: dict) -> None:
    full, first, surname = owner["full"], owner["given"], owner["surname"]
    begin = len(b.text)
    if owner["kind"] == "unusual":
        b.add("The registered personal name is "); b.add(full, "NAME")
        b.add(". The requested spoken name is "); b.add(first, "NAME"); b.add(". ")
        _record_evidence(evidence, "explicit_unusual_name", b, begin)
    elif style == 0:
        b.add(full, "NAME"); b.add(" attended the appointment with a list of concerns. ")
    elif style == 1:
        b.add("After discussing the options, "); b.add(full, "NAME"); b.add(" preferred to continue the current plan. ")
    elif style == 2:
        b.add("The person scheduled for this assessment, "); b.add(full, "NAME"); b.add(", arrived with the completed questionnaire. ")
    elif style == 3:
        b.add("At the start of the call, "); b.add(full, "NAME"); b.add(" described the change in symptoms. ")
    elif style == 4:
        b.add(first, "NAME"); b.add(" is the familiar name used by "); b.add(full, "NAME"); b.add(" during visits. ")
        _record_evidence(evidence, "alias_before_full_name", b, begin)
    elif style == 5:
        b.add("The patient, "); b.add(full, "NAME"); b.add(", prefers "); b.add(first, "NAME"); b.add(" in conversation. ")
        _record_evidence(evidence, "preferred_name", b, begin)
    elif style == 6:
        b.add("The message signed "); b.add(f"{first[0]}. {surname}", "NAME"); b.add(" was verified with "); b.add(full, "NAME"); b.add(". ")
        _record_evidence(evidence, "initial_name", b, begin)
    elif style == 7:
        b.add("Name supplied during intake: "); b.add(full.lower(), "NAME"); b.add(". ")
        _record_evidence(evidence, "lowercase_name", b, begin)
    elif style == 8:
        b.add("Patient "); b.add(surname, "NAME"); b.add(" was called from the waiting area; the booked name was "); b.add(full, "NAME"); b.add(". ")
        _record_evidence(evidence, "surname_only_context", b, begin)
    elif style == 9:
        b.add("The discharge discussion with "); b.add(full, "NAME"); b.add(" covered transport and medication access. ")
    elif style == 10:
        b.add("Consent was reviewed with "); b.add(full, "NAME"); b.add(" before the assessment. ")
    elif style == 11:
        b.add("No new concern was raised by "); b.add(full, "NAME"); b.add(" during the final review. ")
        _record_evidence(evidence, "negated_person_context", b, begin)
    elif style == 12:
        b.add("The response from "); b.add(full, "NAME"); b.add(" was entered in the telephone note. ")
    elif style == 13:
        b.add("I met "); b.add(full, "NAME"); b.add(" after the examination to discuss the written plan. ")
    elif style == 14:
        b.add("Discussion participant — "); b.add(full, "NAME"); b.add(" — requested a copy of the instructions. ")
    elif style == 15:
        b.add("The card carried the name "); b.add(full, "NAME"); b.add("; the patient confirmed those details. ")
    elif style == 16:
        b.add("Concern reported by "); b.add(full, "NAME"); b.add(": the earlier advice was difficult to follow at home. ")
    else:
        b.add("The follow-up arrangements were agreed with "); b.add(full, "NAME"); b.add(". ")
    if owner["kind"] == "collision":
        start = len(b.text)
        disease = dict(EPONYMS)[surname]
        b.add(f"The information leaflet addressed {disease}. {WORD_NEGATIVES[first]} ")
        _record_evidence(evidence, "eponym_name_contrast", b, begin)
        _record_evidence(evidence, "word_name_contrast", b, start)


def _secondary(b: TextBuilder, rng: random.Random, owners: list[dict], evidence: dict, *, relative: bool = False) -> None:
    other = identity(rng)
    while other["group"] in {p["group"] for p in owners}:
        other = identity(rng)
    owners.append(other)
    start = len(b.text)
    if relative:
        b.add(rng.choice(("The patient's sibling, ", "Their designated contact, ", "A family member named ")))
        b.add(other["full"], "NAME"); b.add(", joined the discussion with the patient's consent. ")
        tag = "relative_name"
    else:
        b.add(rng.choice(("Discussed with Dr. ", "Reviewed by Dr. ", "The note was signed by Dr. ")))
        b.add(other["full"], "NAME"); b.add(rng.choice((", MD. ", ", DO. ", ". ")))
        if rng.random() < .35:
            b.add("Dr. "); b.add(other["surname"], "NAME"); b.add(" recommended continuing the plan. ")
        tag = "clinician_name"
    _record_evidence(evidence, tag, b, start)


def _field(b: TextBuilder, label: str, value: str, rng: random.Random, layout: str, evidence: dict) -> None:
    start = len(b.text)
    if layout == "prose" or (layout in {"letter", "soap", "handoff"} and rng.random() < .7):
        shells = {
            "DATE": (("The assessment took place on ", ". "), ("The patient recalled the earlier visit in ", ". ")),
            "ADDRESS": (("Correspondence should be sent to ", ". "), ("After discharge the patient will stay at ", ". ")),
            "LOCATION": (("The patient lives in ", ". "), ("Recovery will continue near family in ", ". "), ("Travel from ", " makes regular appointments difficult. ")),
            "FACILITY": (("The referral came from ", ". "), ("The appointment will be arranged at ", ". "), ("No transfer to ", " is needed today. ")),
            "PHONE": (("For scheduling, call ", ". "),),
            "EMAIL": (("The patient requested a reply at ", ". "),),
            "URL": (("The personal visit summary is available at ", ". "),),
            "IP": (("The patient's remote connection used ", ". "),),
        }
        if label in shells:
            left, right = rng.choice(shells[label]); b.add(left); b.add(value, label); b.add(right)
        else:
            b.add(rng.choice(FIELD_LABELS[label]) + " "); b.add(value, label); b.add(". ")
    elif layout == "table":
        b.add("| " + rng.choice(FIELD_LABELS[label]) + " | "); b.add(value, label); b.add(" |\n")
    else:
        b.add(("- " if layout == "bullet" else "") + rng.choice(FIELD_LABELS[label]) + ("=" if layout == "compact" else ": "))
        b.add(value, label); b.add("; " if layout == "compact" else "\n")
    tag = {"DATE": "patient_date", "ADDRESS": "address_boundary", "LOCATION": "location_boundary", "FACILITY": "facility_boundary"}.get(label, "reserved_identifier" if label in {"PHONE", "EMAIL", "URL", "IP"} else "structured_identifier")
    _record_evidence(evidence, tag, b, start)


def _age(b: TextBuilder, rng: random.Random, age: int, evidence: dict, *, sex: str = "F") -> None:
    elderly = age > 89
    style = rng.randrange(6)
    start = len(b.text)
    b.add("Age at this visit: ")
    value = str(age)
    if elderly and style == 5:
        value = "one hundred" if age == 100 else "ninety" + ("-" + ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine")[age - 91] if age > 90 else "")
    b.add(value, "AGE" if elderly else None)
    b.add((" years; ", "-year-old; ", "yo; ", " y/o; ", sex + "; ", " years old; ")[style])
    _record_evidence(evidence, "age_over_89" if elderly else "ordinary_age", b, start)
    if style in {2, 3, 4}:
        _record_evidence(evidence, "compact_age", b, start)


def _clinical(b: TextBuilder, context: dict, rng: random.Random, layout: str, evidence: dict) -> None:
    begin = len(b.text)
    def prose(value: str) -> None:
        previous = 0
        for match in re.finditer(r"\b(9\d|100)-year-old\b", value):
            b.add(value[previous:match.start(1)]); b.add(match[1], "AGE")
            previous = match.end(1)
        b.add(value[previous:])
    if layout == "soap":
        prose("Subjective: " + context["opening"] + "\nObjective: " + context["findings"] + "\nPlan: " + context["plan"] + "\n")
    elif layout == "handoff":
        prose("Reason for handoff — " + context["opening"] + "\nCurrent assessment — " + context["findings"] + "\nNext actions — " + context["plan"] + "\n")
    else:
        prose(context["opening"] + " " + context["findings"] + " " + context["plan"] + "\n")
    _record_evidence(evidence, "clinical_context", b, begin)
    if re.search(r"\b(?:9\d|100)-year-old\b", context["opening"]):
        _record_evidence(evidence, "age_over_89", b, begin)
    elif re.search(r"\b\d+-year-old\b|Age: \d+ years", context["opening"]):
        _record_evidence(evidence, "ordinary_age", b, begin)
    pronouns = [p for p in PRONOUN_CONTEXTS if not (context.get("sex") == "M" and p.startswith("She ")) and not (context.get("sex") == "F" and p.startswith("He "))]
    start = len(b.text); b.add(rng.choice(pronouns) + " "); _record_evidence(evidence, "ordinary_pronouns", b, start)
    if rng.random() < .62:
        start = len(b.text); b.add(f"The symptom log covers {rng.choice((2, 3, 5, 7, 14, 21, 28, 42, 60, 90))} days. "); _record_evidence(evidence, "clinical_durations", b, start)
    if rng.random() < .52:
        start = len(b.text); b.add(f"At rest, pulse was {rng.randrange(60, 95)}/min and oxygen saturation was {rng.randrange(95, 101)}%. "); _record_evidence(evidence, "clinical_measurements", b, start)


def _context(rng: random.Random, *, negative: bool = False) -> tuple[dict, int | None]:
    if negative and rng.random() < .36:
        i = rng.randrange(len(NEGATIVE_CONTEXTS))
        return {"genre": f"clinical_negative_{i:02d}", "opening": NEGATIVE_CONTEXTS[i], "findings": "", "plan": ""}, None
    context = dict(rng.choice(CLINICAL_CONTEXTS))
    context["sex"] = "M" if context["genre"] == "bph_symptom_management" else "F" if context["genre"] in {"pregnancy_nausea_review", "endometriosis_pain_review", "iron_deficiency_workup"} else rng.choice(("F", "M"))
    age_match = re.search(r"\b(\d+)-year-old", context["opening"])
    age = int(age_match[1]) if age_match else None
    if age is not None:
        # Only older-adult scenarios acquire an elderly value. Reproductive and
        # young-adult disease scenarios retain their authored age distribution.
        age = rng.randrange(90, 101) if not negative and age >= 60 and rng.random() < .43 else max(20, min(89, age + rng.randrange(-4, 5)))
        context["opening"] = context["opening"][:age_match.start(1)] + str(age) + context["opening"][age_match.end(1):]
        context["opening"] = re.sub(r"\bAn? (\d+)-year-old", lambda m: ("An " if m[1].startswith("8") else "A ") + m[1] + "-year-old", context["opening"])
    return context, age


def _public(b: TextBuilder, year: int, rng: random.Random, evidence: dict) -> None:
    start = len(b.text); b.add(rng.choice(PUBLIC_CONTEXT_TEMPLATES).format(year=year) + " ")
    _record_evidence(evidence, "public_dates", b, start)


def _row(b: TextBuilder, index: int, seed: int, owners: list[dict], context: dict, layout: str, evidence: dict, *, paired: int | None = None, cohort: str = "realistic", diagnostic_slice: str = "ordinary") -> dict:
    row = {"id": f"v12-train-{seed}-{index:06d}", "text": b.text, "spans": b.spans,
           "split": "train", "suite": "training", "annotation_policy": POLICY_VERSION,
           "identity_group": owners[0]["group"], "identity_groups": sorted({p["group"] for p in owners}),
           "full_identity": owners[0]["full"], "full_identities": sorted({p["full"] for p in owners}),
           "pair_id": f"v12-train-{seed}-pair-{paired}" if paired is not None else f"v12-train-{seed}-single-{index}",
           "template_family": f"v12_training.{context['genre']}.{layout}", "family": context["genre"], "layout": layout,
           "phenomena": sorted(evidence), "phenomenon_evidence": evidence,
           "cohort": cohort, "diagnostic_slice": diagnostic_slice, "negative": not b.spans}
    validate_row(row)
    return row


def short_row(index: int, *, seed: int = SEED, negative: bool = False, unusual: bool = False, pair: int | None = None, variation: int = 0) -> dict:
    rng = random.Random(f"v12:{seed}:{pair if pair is not None else index}:short:{variation}")
    identity_rng = random.Random(f"v12:{seed}:{pair if pair is not None else index}:identity")
    kind = "unusual" if unusual else "collision" if identity_rng.random() < .19 else "ordinary"
    owner = identity(identity_rng, kind=kind)
    # Paired notes share a clinical scenario and ordinary age; identifiers are
    # then added only to the positive member. Unpaired positives also cover >89.
    context, patient_age = _context(rng, negative=negative if pair is None else True)
    layout = rng.choice(LAYOUTS)
    owners = [owner]
    vals = values(rng)
    evidence: dict = {}
    b = TextBuilder()
    start = len(b.text); b.add(rng.choice(CLINICAL_HEADINGS) + "\n"); _record_evidence(evidence, "clinical_heading", b, start)
    position = rng.randrange(3)
    if not negative and position == 0:
        _name(b, owner, rng.randrange(18), evidence)
    _clinical(b, context, rng, layout, evidence)
    if not negative and position == 1:
        _name(b, owner, rng.randrange(18), evidence)
    if patient_age is not None and rng.random() < .4:
        _age(b, rng, patient_age, evidence, sex=context.get("sex", "F"))
    if not negative:
        if rng.random() < .34:
            _secondary(b, rng, owners, evidence)
        if rng.random() < .19:
            _secondary(b, rng, owners, evidence, relative=True)
        if position == 2:
            _name(b, owner, rng.randrange(18), evidence)
        labels = [label for label in ENTITY_TYPES if label not in {"NAME", "AGE"}]
        selected = rng.sample(labels, rng.choice((2, 3, 4, 5, 6)))
        # DATE is also taught next to a public reference containing the same year.
        contrast = rng.random() < .34
        if contrast and "DATE" not in selected:
            selected.append("DATE")
        if contrast:
            vals["DATE"] = str(vals["year"]) if rng.random() < .38 else f"March {vals['year']}"
            vals["date_form"] = 5
        if layout == "table":
            start = len(b.text); b.add("| Entry | Value |\n| --- | --- |\n"); _record_evidence(evidence, "structured_table", b, start)
        joined_site = "LOCATION" in selected and "FACILITY" in selected and rng.random() < .65
        if joined_site:
            start = len(b.text)
            b.add(rng.choice(("The arranged follow-up is at ", "The referral was sent to ", "Care continues at ")))
            b.add(vals["FACILITY"], "FACILITY"); b.add(" in "); b.add(vals["LOCATION"], "LOCATION"); b.add(". ")
            _record_evidence(evidence, "facility_boundary", b, start)
            _record_evidence(evidence, "location_boundary", b, start)
            _record_evidence(evidence, "adjacent_facility_location", b, start)
        for label in selected:
            if not joined_site or label not in {"FACILITY", "LOCATION"}:
                _field(b, label, vals[label], rng, layout, evidence)
        if "DATE" in selected and vals["date_form"] in {4, 5, 6}:
            evidence["partial_date"] = list(evidence["patient_date"])
        if contrast:
            start = len(b.text); _public(b, vals["year"], rng, evidence); _record_evidence(evidence, "same_year_patient_public", b, start)
        elif rng.random() < .4:
            _public(b, rng.choice(YEARS), rng, evidence)
    elif rng.random() < .62:
        _public(b, rng.choice(YEARS), rng, evidence)
    if pair is not None:
        evidence["context_pair"] = [{"start": 0, "end": len(b.text)}]
    row = _row(b, index, seed, owners, context, layout, evidence, paired=pair, cohort="unusual_name" if unusual else "realistic")
    row["resampled_variations"] = variation
    return row


def long_row(index: int, *, seed: int = SEED, target_tokens: int = 2500) -> dict:
    """Sectioned, repeated-course training notes; no claim of natural prevalence."""
    rng = random.Random(f"v12:{seed}:{index}:long")
    context, _ = _context(rng)
    owners = [identity(rng)]
    evidence: dict = {}
    b = TextBuilder("Course summary and continuity plan\n")
    _name(b, owners[0], 5, evidence)
    vals = values(rng)
    for label in ("DATE", "ADDRESS", "PHONE", "EMAIL", "LOCATION", "FACILITY"):
        _field(b, label, vals[label], rng, "fields", evidence)
    _public(b, vals["year"], rng, evidence)
    encoding = get_encoding()
    section = 0
    while len(encoding.encode(b.text, disallowed_special=())) < target_tokens - 100:
        b.add(f"\nInterval review {section + 1}\n")
        _clinical(b, context, rng, "prose", evidence)
        if section % 3 == 0:
            _name(b, owners[0], rng.randrange(18), evidence)
        if section % 4 == 1:
            _secondary(b, rng, owners, evidence)
        if section % 4 == 2:
            _field(b, "DATE", vals["DATE"], rng, "prose", evidence)
        section += 1
    b.add("\nEnd-of-course handoff\n")
    _name(b, owners[0], 17, evidence)
    _field(b, "ADDRESS", vals["ADDRESS"], rng, "prose", evidence)
    evidence["long_document"] = [{"start": 0, "end": len(b.text)}]
    evidence["tail_phi"] = [{"start": b.spans[-1]["start"], "end": b.spans[-1]["end"]}]
    row = _row(b, index, seed, owners, context, "continuity_summary", evidence, cohort="long_document", diagnostic_slice="repeated_course_narrative")
    row["token_audit"] = token_metadata(row, encoding)
    row["repeated_context_sections"] = section
    return row


def validate_row(row: dict) -> None:
    errors = annotation_errors(row["text"], row["spans"])
    if errors:
        raise ValueError(errors)
    if row["split"] != "train" or row["annotation_policy"] != POLICY_VERSION:
        raise ValueError("Only policy-v2 training rows are supported")
    expected = sorted({canonical_identity_group(name) for name in row["full_identities"]})
    if expected != row["identity_groups"] or canonical_identity_group(row["full_identity"]) != row["identity_group"]:
        raise ValueError("Identity metadata does not match its authored owners")
    if any(identity_split(group) != "train" for group in expected):
        raise ValueError("Every owner must belong to the training partition")
    if sorted(row["phenomenon_evidence"]) != row["phenomena"]:
        raise ValueError("Every phenomenon must have authored evidence")
    for tag, occurrences in row["phenomenon_evidence"].items():
        if not occurrences or any(not 0 <= item["start"] < item["end"] <= len(row["text"]) for item in occurrences):
            raise ValueError(f"Invalid evidence for {tag}")
    if "same_year_patient_public" in row["phenomena"]:
        patient = {m[0] for span in row["spans"] if span["label"] == "DATE" for m in re.finditer(r"\b(?:19|20)\d{2}\b", row["text"][span["start"]:span["end"]])}
        public = {m[0] for loc in row["phenomenon_evidence"].get("public_dates", []) for m in re.finditer(r"\b(?:19|20)\d{2}\b", row["text"][loc["start"]:loc["end"]])}
        if not patient & public:
            raise ValueError("Claimed same-year contrast is absent")


def build_corpus(*, short_count: int = 18000, long_count: int = 240, seed: int = SEED, negative_fraction: float = .20, unusual_fraction: float = .025) -> list[dict]:
    if short_count < 1 or long_count < 0 or not 0 <= negative_fraction < .45 or not 0 <= unusual_fraction < .10:
        raise ValueError("Invalid corpus configuration")
    negative_count = round(short_count * negative_fraction)
    unusual_count = round(short_count * unusual_fraction)
    paired_count = negative_count // 2
    rows = []
    seen = set()
    def add_short(**kwargs):
        for variation in range(100):
            row = short_row(len(rows), seed=seed, variation=variation, **kwargs)
            if row["text"] not in seen:
                seen.add(row["text"]); rows.append(row)
                return
        raise ValueError("Clinical combinations exhausted; add distinct authored contexts")
    for pair in range(paired_count):
        for variation in range(100):
            positive = short_row(len(rows), seed=seed, pair=pair, variation=variation)
            negative = short_row(len(rows) + 1, seed=seed, pair=pair, negative=True, variation=variation)
            if positive["text"] not in seen and negative["text"] not in seen:
                rows.extend((positive, negative)); seen.update((positive["text"], negative["text"]))
                break
        else:
            raise ValueError("Paired context combinations exhausted")
    for _ in range(negative_count - paired_count):
        add_short(negative=True)
    while len(rows) < short_count:
        add_short(unusual=len(rows) >= short_count - unusual_count)
    for i in range(long_count):
        rows.append(long_row(len(rows), seed=seed, target_tokens=1200 + (i % 12) * 320))
    if len({row["text"] for row in rows}) != len(rows):
        raise ValueError("Duplicate text detected; expand authored context rather than duplicate notes")
    return rows


def write_corpus(path: Path | str, rows: list[dict], *, config: dict | None = None) -> dict:
    path = Path(path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if path.exists() or manifest_path.exists():
        raise FileExistsError("Frozen corpus bytes must not be replaced; choose a new version")
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Require nonempty, uniquely identified rows")
    for row in rows:
        validate_row(row)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()
    encoding = get_encoding()
    tokens = [len(encoding.encode(row["text"], disallowed_special=())) for row in rows]
    name_shapes, date_years, public_years = Counter(), Counter(), Counter()
    for row in rows:
        for span in row["spans"]:
            value = row["text"][span["start"]:span["end"]]
            if span["label"] == "NAME":
                name_shapes["ascii" if value.isascii() else "non_ascii"] += 1
                name_shapes["punctuated" if re.search(r"[.\-'’]", value) else "unpunctuated"] += 1
            if span["label"] == "DATE":
                date_years.update(re.findall(r"\b(?:19|20)\d{2}\b", value))
        for loc in row["phenomenon_evidence"].get("public_dates", []):
            public_years.update(re.findall(r"\b(?:19|20)\d{2}\b", row["text"][loc["start"]:loc["end"]]))
    source_files = tuple(PACKAGE_ROOT / name for name in ("data/v12/synthetic.py", "data/v12/contexts.py", "data/generators/remediation.py", "data/annotation_policy.py", "types.py"))
    manifest = {
        "schema_version": 1, "generator": "outis.data.v12.synthetic", "generator_version": VERSION,
        "sha256": hashlib.sha256(payload).hexdigest(), "annotation_policy": POLICY_VERSION,
        "split": "train", "suite": "training", "rows": len(rows), "config": config or {},
        "used_for_training": True, "used_for_checkpoint_selection": False, "independent_final": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "license": "CC0-1.0", "code_license": "Apache-2.0", "external_datasets": [],
        "provenance": "Original synthetic contexts and invented identifiers. Reuses v1 training spelling pools and offset/token utilities; no corpus file is loaded by the generator. No development or qualification contexts are imported.",
        "authoring_scope": "Training author previously reviewed v1 development samples; v12 clinical contexts were authored by a separate clean-context author. This is training material, not independent evaluation.",
        "identity_partition": "sha256(NFC(full identity).casefold()); first eight hex digits modulo ten below eight; secondary owners obey same rule",
        "cohort_support": dict(Counter(row["cohort"] for row in rows)),
        "diagnostic_slice_support": dict(Counter(row["diagnostic_slice"] for row in rows)),
        "label_support": dict(Counter(s["label"] for row in rows for s in row["spans"])),
        "phenomenon_support": dict(Counter(p for row in rows for p in row["phenomena"])),
        "families": dict(Counter(row["family"] for row in rows)),
        "phi_free_notes": sum(not row["spans"] for row in rows),
        "duplicate_combinations_resampled": sum(row.get("resampled_variations", 0) for row in rows),
        "unique_identity_groups": len({g for row in rows for g in row["identity_groups"]}),
        "name_shapes": dict(name_shapes), "patient_year_support": dict(sorted(date_years.items())),
        "public_year_support": dict(sorted(public_years.items())),
        "shared_patient_public_years": sorted(set(date_years) & set(public_years)),
        "tokenizer": "o200k_base", "token_length": {"min": min(tokens), "max": max(tokens), "mean": sum(tokens) / len(tokens), "total": sum(tokens)},
        "synthetic_identifier_schemes": {"PHONE": "US 555-0100..0199", "EMAIL_URL": "Reserved .test/.example and example.com/.net/.org domains", "IP": "TEST-NET IPv4 ranges", "SSN": "Invented 000-XX-XXXX", "MRN_ACCOUNT_ID": "Invented prefixes and random numeric serials; no external record source"},
        "limitations": ["Finite authored clinical scenarios and phrase combinations are correlated; row count is not independent clinical coverage.", "Long documents repeat the same clinical course with new observations and identities; report the repeated_course_narrative slice separately.", "Spelling pools are illustrative, not guarantees of cultural or linguistic coverage.", "All metadata claims are backed by character ranges, but structural validation does not replace semantic annotation review."],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with path.open("xb") as handle:
            created.append(path); handle.write(payload)
        with manifest_path.open("x", encoding="utf-8") as handle:
            created.append(manifest_path); json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False); handle.write("\n")
    except BaseException:
        for owned in created:
            owned.unlink(missing_ok=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--short-count", type=int, default=18000)
    parser.add_argument("--long-count", type=int, default=240)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".manifest.json").exists():
        raise FileExistsError("Choose a new immutable output path")
    started = time.monotonic()
    config = {k: v for k, v in vars(args).items() if k != "output"}
    rows = build_corpus(**config)
    manifest = write_corpus(args.output, rows, config=config)
    print(json.dumps({"rows": len(rows), "sha256": manifest["sha256"], "seconds": round(time.monotonic() - started, 2), "token_length": manifest["token_length"]}))


if __name__ == "__main__":
    main()
