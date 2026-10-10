"""Independent, synthetic clinical challenge generation.

This module deliberately does not import the training-data generator.  Synthetic
identities and templates were authored independently; no external patient corpus
is used.  Final-test files must be frozen before a detector is evaluated.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any

from outis.types import ENTITY_TYPES

PROTOCOL_VERSION = "independent-synthetic-v1"
TAG = re.compile(r"\[\[([A-Z]+)\|([^\]]+)\]\]")

# Fictional names; arbitrary combinations do not identify real patients.
FIRST = ("Marisol", "Theo", "Imani", "Rogelio", "Aisha", "Priya", "Nikolai", "Mei",
         "Saoirse", "Lucía", "Étienne", "Jürgen", "Chloë", "Søren", "Amélie", "Ji-woo",
         "André", "Dóra", "Renée", "Álvaro", "Zoë", "Hélène", "Márton", "Mika")
LAST = ("Velasco", "Okafor", "Iversen", "Haddad", "Banerjee", "Navarro", "Kowalski",
        "Mori", "O'Connell", "D'Amato", "Núñez", "Björk", "García-López", "Santos",
        "de Vries", "van Dijk", "Dubois", "Al-Karim", "Zieliński", "Černý", "Yılmaz")
STREETS = ("Cobalt Terrace", "Juniper Hollow", "Orchid Crescent", "Aster Lane", "Mica Road",
           "Thistle Court", "Hawthorn Place", "Rue des Saules", "Birkenstraße", "Calle Bruma")
TOWNS = ("Asterwick", "Cobalt Glen", "Thistledale", "Mossford", "Larchhaven", "Brumavale",
         "Willowcrest", "Pinebriar", "Orchid Bay", "Mica Falls", "Fernborough", "Alderfield")
FACILITIES = ("Cobalt Glen Medical Center", "Orchid Bay Rehabilitation Hospital",
              "Asterwick Family Clinic", "Larchhaven Infusion Centre", "Mossford Dialysis Unit",
              "Willowcrest Hospice", "Thistledale Community Hospital", "Pinebriar Children's Clinic",
              "Fernborough Surgical Centre", "Alderfield Regional Hospital")
HARD_NEGATIVES = (
    "Temperature 37.1 C; glucose 109 mg/dL; creatinine 1.2 mg/dL; oxygen saturation 94%.",
    "Take 25 mg twice daily for 14 days; return after six weeks if symptoms persist.",
    "Stage II disease is stable. The 2018 public trial supports continued therapy.",
    "Parkinson disease and Graves disease were discussed; Murphy sign was negative.",
    "The 54-year-old participant walks 2 km daily. Family history includes Alzheimer disease.",
    "An 89-year-old adult received 100 mL saline at 08:30; respiratory rate was 18/min.",
    "Tanner stage 3; Glasgow coma scale 15; Mini-Mental State Examination score 28/30.",
    "Dose reduced from 10 to 5 mg over 3 months; follow-up interval is 90 days.",
    "Brown syndrome, Wilson disease, and Baker cyst are diagnostic considerations.",
    "Study PMID 31234567 was published in 2020. No patient-specific date appears in this sentence.",
)

SCENARIOS = {
    "development": (
        ("medication_reconciliation", "Medication reconciliation: edema has improved after the diuretic change."),
        ("pharmacy_callback", "Pharmacy callback: nausea is transient and there is no rash or wheeze."),
        ("therapy_handoff", "Therapy handoff: gait endurance improved; continue assisted stair practice."),
        ("outpatient_discharge", "Discharge review: incision is dry, pain is controlled, and oral intake is adequate."),
        ("referral_intake", "Referral intake: intermittent palpitations occur during exercise without syncope."),
        ("transfusion_review", "Transfusion review: mild fatigue persists; no bleeding or transfusion reaction."),
        ("home_nursing", "Home nursing: the dressing is intact and distal pulses are palpable."),
        ("social_work", "Social work note: transport and medication access were arranged for recovery at home."),
    ),
    "final": (
        ("palliative_telemedicine", "Palliative telemedicine review: dyspnea responds to pacing and symptom control."),
        ("travel_infection", "Infectious disease consultation: fever resolved, cultures are negative, and hydration improved."),
        ("obstetric_aftercare", "Postpartum review: feeding is established; there is no heavy bleeding or headache."),
        ("dialysis_routing", "Dialysis transfer note: access thrill is present and dry weight remains appropriate."),
        ("memory_care", "Memory clinic assessment: orientation is preserved and sleep fragmentation has improved."),
        ("dermatology_report", "Dermatology review: erythema has faded and there is no mucosal involvement."),
        ("asthma_action", "Pulmonary follow-up: peak flow is stable; rescue inhaler use has decreased."),
        ("cardiac_transfer", "Cardiology transfer: exercise tolerance improved after rate-control adjustment."),
        ("anesthesia_call", "Pre-anesthesia review: airway examination is reassuring; fasting instructions were discussed."),
        ("vestibular_rehab", "Vestibular rehabilitation: positional dizziness is improving with home exercises."),
        ("hospice_handoff", "Hospice handoff: constipation is controlled and the caregiver understands breakthrough dosing."),
        ("nutrition_review", "Nutrition assessment: oral intake is improving and swallowing remains comfortable."),
        ("genetic_counseling", "Genetic counseling: inheritance patterns and the limitations of testing were explained."),
        ("burn_dressing", "Burn clinic review: epithelialization is progressing; there is no purulent drainage."),
        ("anticoagulation", "Anticoagulation review: INR is within the target range and there is no melena."),
        ("wound_followup", "Wound follow-up: granulation tissue is healthy and surrounding cellulitis has resolved."),
    ),
    "adversarial": (
        ("dense_identifiers", "Triage handover: contact details were dictated rapidly during an otherwise stable admission."),
        ("unicode_and_narrative", "Narrative case note: clinical decisions were made after discussion with the family."),
        ("mixed_date_formats", "Referral chronology: preserve intervals between investigations and treatment."),
        ("negative_only", "Clinical teaching note with no patient-specific identifiers."),
    ),
}


def marked(label: str, value: str) -> str:
    return f"[[{label}|{value}]]"


def unmark(template: str) -> tuple[str, list[dict[str, Any]]]:
    """Convert explicit annotation markup to Unicode character offsets."""
    parts: list[str] = []
    spans: list[dict[str, Any]] = []
    cursor = 0
    length = 0
    for match in TAG.finditer(template):
        prefix = template[cursor:match.start()]
        parts.append(prefix)
        length += len(prefix)
        label, value = match.groups()
        if label not in ENTITY_TYPES:
            raise ValueError(f"Unknown label {label}")
        parts.append(value)
        spans.append({"start": length, "end": length + len(value), "label": label})
        length += len(value)
        cursor = match.end()
    parts.append(template[cursor:])
    return "".join(parts), spans


def _values(rng: random.Random, index: int) -> dict[str, str]:
    day = date(1992, 1, 1) + timedelta(days=rng.randrange(11000))
    date_formats = ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%b %d, %Y", "%d %B %Y", "%B %d, %Y", "%d-%b-%Y")
    town = rng.choice(TOWNS)
    first, last = rng.choice(FIRST), rng.choice(LAST)
    email_name = re.sub(r"[^a-z]", "", (first + last).lower()) or "person"
    # Reserved domains, TEST-NET addresses, and reserved fictional phone blocks.
    return {
        "NAME": f"{first} {last}",
        "DATE": day.strftime(rng.choice(date_formats)),
        "ADDRESS": f"{rng.randrange(10, 9999)} {rng.choice(STREETS)}, {town}, ZZ {rng.randrange(10000, 99999)}",
        "PHONE": rng.choice((f"(202) 555-01{index % 100:02d}", f"+1-202-555-01{index % 100:02d}", f"202.555.01{index % 100:02d}")),
        "EMAIL": f"{email_name}.{index}@example.org",
        "MRN": rng.choice((f"MR{rng.randrange(100000, 999999)}", f"{rng.randrange(10000000, 99999999)}", f"ZX-{rng.randrange(1000, 9999)}-{index % 100:02d}")),
        "SSN": f"000-{rng.randrange(10, 99)}-{rng.randrange(1000, 9999)}",
        "ACCOUNT": f"AC-{rng.randrange(10000, 99999)}-{index % 1000:03d}",
        "ID": rng.choice((f"PX{rng.randrange(1000000, 9999999)}", f"DL-{rng.randrange(100000, 999999)}", f"ST{rng.randrange(100000, 999999)}")),
        "URL": f"https://portal.example.org/patient/{rng.randrange(100000, 999999)}/{index}",
        "IP": f"192.0.2.{rng.randrange(1, 255)}",
        "AGE": str(rng.randrange(90, 109)),
        "LOCATION": town,
        "FACILITY": rng.choice(FACILITIES),
    }


# Explicitly distinct wording and clinical scenario families across splits.
FRAGMENTS = {
    "development": {
        "NAME": ("Patient: {v}.", "The patient, {v}, reports improved sleep.", "Dr. {v} reviewed the results."),
        "DATE": ("Encounter date: {v}.", "Symptoms began on {v}.", "The procedure took place {v}."),
        "ADDRESS": ("Home address: {v}.", "Send wound supplies to {v}."),
        "PHONE": ("Telephone: {v}.", "Call {v} for follow-up."),
        "EMAIL": ("Email: {v}.", "Instructions were requested at {v}."),
        "MRN": ("MRN: {v}.", "Chart number {v} was confirmed."),
        "SSN": ("SSN: {v}.", "Social security number: {v}."),
        "ACCOUNT": ("Account: {v}.", "Billing account number {v}."),
        "ID": ("Passport: {v}.", "Driver license number: {v}.", "Insurance member ID: {v}."),
        "URL": ("Patient portal: {v}.", "Upload the report to {v}."),
        "IP": ("Remote session IP: {v}.", "Device IP address: {v}."),
        "AGE": ("Age: {v} years.", "A {v}-year-old patient was assessed."),
        "LOCATION": ("Lives in {v}.", "The family is staying in {v}."),
        "FACILITY": ("Seen at {v}.", "Transfer arranged to {v}."),
    },
    "final": {
        "NAME": ("{v} feels less short of breath today.", "Accompanied by spouse {v}.", "Discussed care with consultant {v}.", "This is {v}'s return visit.", "Dear {v}, your recovery plan is below."),
        "DATE": ("Review completed {v}; findings are stable.", "Since {v}, pain has gradually decreased.", "Next in-person assessment is scheduled for {v}.", "DOB {v}; identity checked verbally."),
        "ADDRESS": ("Domicile recorded as {v}.", "Courier destination is {v}.", "Residence for the home visit: {v}."),
        "PHONE": ("Reach the caregiver on {v}.", "Callback line {v} is preferred.", "Voicemail left at {v}."),
        "EMAIL": ("Use {v} to deliver the care summary.", "Preferred electronic contact is {v}.", "Reply destination: {v}."),
        "MRN": ("Hospital record no. {v} accompanies this handover.", "Medical record identifier {v} matches the wristband.", "Patient chart ref: {v}."),
        "SSN": ("Social Security # {v} supplied during registration.", "Registration SSN = {v}."),
        "ACCOUNT": ("Financial account reference {v} accompanies the claim.", "Invoice acct # {v}.", "Hospital billing ref: {v}."),
        "ID": ("Identity document no. {v} was used for check-in.", "Policy member ref {v} was verified.", "Travel document identifier {v}."),
        "URL": ("The private report link is {v}.", "Use the individual's portal URL {v}."),
        "IP": ("Televisit originated from network address {v}.", "Connection source: {v}."),
        "AGE": ("Now {v} years old, the patient remains independent.", "An adult aged {v} completed the assessment.", "The {v}-year-old reports no falls."),
        "LOCATION": ("Home community is {v}.", "Following discharge, recuperation will occur in {v}.", "Local transport within {v} was arranged."),
        "FACILITY": ("Clinical handover originated at {v}.", "Referred through {v} for specialist care.", "Admission location was {v}."),
    },
    "adversarial": {
        "NAME": ("Care recipient {v} describes transient dizziness.", "Relatives identified {v} as the family contact."),
        "DATE": ("Recorded timeline entry: {v}.", "Last evaluation was documented on {v}."),
        "ADDRESS": ("Delivery address in the chart — {v}.",),
        "PHONE": ("After-hours contact number — {v}.",),
        "EMAIL": ("Electronic correspondence destination — {v}.",),
        "MRN": ("Medical record number — {v}.",),
        "SSN": ("Social security identifier — {v}.",),
        "ACCOUNT": ("Billing account — {v}.",),
        "ID": ("Insurance ID — {v}.",),
        "URL": ("Private clinical portal address — {v}.",),
        "IP": ("Source IP — {v}.",),
        "AGE": ("Assessment of a patient aged {v} years.",),
        "LOCATION": ("Resident community — {v}.",),
        "FACILITY": ("Treating hospital — {v}.",),
    },
}


def generate(count: int = 1200, seed: int = 8675309, suite: str = "final") -> list[dict[str, Any]]:
    if count < 1 or suite not in SCENARIOS:
        raise ValueError("count must be positive and suite must be development, final, or adversarial")
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        family, intro = SCENARIOS[suite][index % len(SCENARIOS[suite])]
        values = _values(rng, index)
        pieces = [intro, rng.choice(HARD_NEGATIVES)]
        # Explicitly include negative-only documents to measure false positives.
        negative_only = index % 17 == 0 or (suite == "adversarial" and family == "negative_only")
        if not negative_only:
            labels = ["NAME", "DATE", "FACILITY"]
            candidates = [label for label in ENTITY_TYPES if label not in labels]
            labels += rng.sample(candidates, rng.randrange(7, 12))
            rng.shuffle(labels)
            wording = FRAGMENTS[suite]
            for label in labels:
                value = marked(label, values[label])
                pieces.append(rng.choice(wording[label]).format(v=value))
            if index % 5 == 0:
                later = date(2024, 1, 1) + timedelta(days=rng.randrange(700))
                pieces.append(f"Treatment started {marked('DATE', later.strftime('%Y-%m-%d'))}; review occurred {marked('DATE', (later + timedelta(days=14)).strftime('%Y-%m-%d'))}.")
            if index % 7 == 0:
                pieces.append(f"Nurse {marked('NAME', rng.choice(FIRST) + ' ' + rng.choice(LAST))} confirmed the care plan.")
            if suite == "adversarial" and family == "dense_identifiers":
                pieces.append("Contact " + ";".join(marked(label, values[label]) for label in ("PHONE", "EMAIL", "URL", "IP")) + ".")
            if suite == "adversarial" and family == "mixed_date_formats":
                pieces.append("Timeline: " + " → ".join(marked("DATE", value) for value in ("03-Oct-2024", "October 17, 2024", "2024-11-02")) + ".")
        pieces.append(rng.choice(HARD_NEGATIVES))
        separator = rng.choice((" ", "\n", "\n\n"))
        text, spans = unmark(separator.join(pieces))
        rows.append({"id": f"challenge-{suite}-{seed}-{index:05d}", "text": text, "spans": spans,
                     "family": family, "split": "test" if suite == "final" else "validation", "suite": suite})
    return rows


def write_corpus(path: str | Path, rows: list[dict[str, Any]], seed: int, suite: str) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently replacing a previously frozen holdout.
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    if path.exists() and path.read_bytes() != payload:
        raise FileExistsError(f"Refusing to overwrite frozen corpus: {path}")
    if not path.exists():
        with path.open("xb") as handle:
            handle.write(payload)
    counts = {label: 0 for label in ENTITY_TYPES}
    for row in rows:
        for span in row["spans"]:
            counts[span["label"]] += 1
    manifest = {"protocol_version": PROTOCOL_VERSION, "sha256": hashlib.sha256(payload).hexdigest(),
                "rows": len(rows), "spans": sum(counts.values()), "label_support": counts,
                "seed": seed, "suite": suite, "source": "independently authored synthetic clinical text",
                "external_datasets": [], "patient_data": False,
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "interpretation": "Synthetic challenge performance does not establish production clinical accuracy."}
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text() != manifest_payload:
        raise FileExistsError(f"Refusing to overwrite corpus manifest: {manifest_path}")
    if not manifest_path.exists():
        manifest_path.write_text(manifest_payload)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=8675309)
    parser.add_argument("--suite", choices=tuple(SCENARIOS), default="final")
    args = parser.parse_args()
    manifest = write_corpus(args.output, generate(args.count, args.seed, args.suite), args.seed, args.suite)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
