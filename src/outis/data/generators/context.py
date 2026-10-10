"""Additional authored supervision for clinical captions and tight PHI boundaries.

All text is synthetic.  Concept-level feedback from the explicitly designated
development/adversarial suites motivated this generator.  The independent final
evaluation corpus and its generator are not used.  Train and validation use
separate identity vocabularies, invented geographic roots, and clinical frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from outis.data.generators.augment import CLAUSES, IDENTITIES, _negative_sentences, _values
from outis.data.generators.base import MarkedValue, render_marked, validate_row
from outis.types import ENTITY_TYPES

GEOGRAPHY_ROOTS = {
    "train": (
        "Saffron", "Calder", "Mallow", "Osprey", "Marshfern", "Dapple", "Rill",
        "Harrier", "Sorrel", "Fennel", "Honeysuckle", "Flaxen", "Plover", "Tansy",
        "Hemlock", "Wheatgrass", "Lupine", "Tallow", "Mistral", "Skylark",
    ),
    "validation": (
        "Celandine", "Petrel", "Myrtle", "Sundew", "Pennyroyal", "Cormorant", "Tern",
        "Lazuli", "Ocher", "Kittiwake", "Nuthatch", "Auburn", "Thrasher", "Sumac",
        "Dogwood", "Sedge", "Aconite", "Puffin", "Heron", "Vireo",
    ),
}
GEOGRAPHY_SUFFIXES = (
    "", "ford", "mere", "wick", "dale", " Bay", " Sound", " Run", " Cove", " Reach",
    " Hollow", " Grove", " Falls", " Bend", " Park", " County", " Harbor", " Valley",
)

LOCATION_FRAMES = {
    "train": (
        "Resident locality: {location}.", "Residential community — {location}.",
        "Community of residence: {location}.", "Resident community recorded as {location}.",
        "The residential district is {location}.", "Patient geography / community: {location}.",
        "The home municipality is {location}.", "Local community for home services: {location}.",
        "Origin of travel: {location}; transport time was 45 minutes.",
        "Transfer region: {location}; ambulance support is unnecessary.",
        "The patient is domiciled in {location}.", "Home region on the referral: {location}.",
        "The family identifies {location} as their home town.",
        "The patient's usual residence is in {location}.",
        "Patient's city: {location}; home therapy will be arranged.",
        "Discharge community = {location}.", "Home area [{location}] was verified.",
        "Residential location ({location}) was supplied by the family.",
        "The patient comes from {location} and stayed with a relative overnight.",
        "A journey from {location} preceded the clinic visit.",
        "The patient and family live near {location}.",
        "The destination town for discharge is {location}.",
        "Current residential city — {location}.", "County of residence: {location}.",
        "Geographic area for community care: {location}.",
        "Locality given by the caregiver was {location}.",
        "Place of habitual residence: {location}.",
        "HOME COMMUNITY | {location} | outpatient transport approved.",
    ),
    "validation": (
        "Residence community field contains {location}.",
        "The town listed for home follow-up is {location}.",
        "Usual patient locality / {location}.",
        "Geographical origin on the assessment is {location}.",
        "The receiving community for discharge is {location}.",
        "Home district submitted by the guardian: {location}.",
        "Municipal area of the patient's residence: {location}.",
        "The patient resides around {location}.",
        "Home-care locality ({location}) has a visiting nurse service.",
        "Registered town = {location}; mobility needs were reviewed.",
        "The patient travelled toward {location} following discharge.",
        "Recorded community for patient transport [{location}].",
    ),
}

AGE_FRAMES = {
    "train": (
        "The patient is aged {age} years.", "Patient aged {age} was assessed for balance problems.",
        "Assessment: a person aged {age} years with preserved appetite.",
        "A resident aged {age} reports stable symptoms.", "Chronological age: {age} years.",
        "The patient's recorded age is {age}.", "Age at assessment — {age} years.",
        "The care recipient is now {age} years old.", "Geriatric age in years = {age}.",
        "At {age} years of age, the patient can walk with assistance.",
        "The patient, {age} years old, remains able to feed themself.",
        "This person is {age} yr old and receives home support.",
        "Recorded age [{age}] years; no acute cognitive change.",
        "The patient is {age} y/o with no recent fall.",
        "A {age}-year-old patient is independent with most basic activities.",
        "The age on the intake form was {age} years.",
        "AGE | {age} | review of medication safety completed.",
        "Age: {age}; blood pressure is 126/74 mmHg.",
        "Care assessment for an adult aged {age} yrs.",
        "The patient's stated age in years is {age}.",
    ),
    "validation": (
        "The assessed individual's age is {age} years.",
        "Aged {age} years at the time of this clinical review.",
        "The subject of the care plan is aged {age}.",
        "Age documented on today's assessment: {age} yr.",
        "This {age}-year-old person reports satisfactory sleep.",
        "The age provided by the family was {age} years.",
        "Clinical age field [{age}]; swallowing is unaffected.",
        "The individual undergoing examination is {age} years of age.",
        "A {age} y/o adult attended the review.",
    ),
}

EMAIL_FRAMES = {
    "train": (
        "Electronic destination — {email}.", "Correspondence destination is {email}.",
        "The contact's electronic address is {email}.", "Patient email <{email}> was verified.",
        "Electronic mail destination [{email}] has the patient's permission.",
        "The family supplied e-mail address {email}.", "Patient reply-to: {email}.",
        "Electronic correspondence address: {email}.", "Contact mailbox = {email}.",
        "Preferred email destination ({email}); use the secure workflow.",
        "E-MAIL | {email} | instructions requested.",
        "Use the patient's mailbox '{email}' for approved instructions.",
    ),
    "validation": (
        "The electronic destination approved today is {email}.",
        "Requested patient mailbox [{email}] is on file.",
        "The email field on the referral contains <{email}>.",
        "For electronic replies use ({email}).",
        "Approved e-mail delivery location / {email}.",
        "Contact's mailbox address: '{email}'.",
    ),
}

CONTACT_BUNDLES = {
    "train": (
        "Contact channels: {phone};{email};{url};{ip}.",
        "Patient communication / {email};{phone};{url};{ip}.",
        "Remote contact [{phone}|{email}|{url}|{ip}] verified by registration.",
        "CONTACT={phone},{email},{url},{ip}.",
        "Clinical communication ({phone};{email};{url};{ip}) is authorized.",
        "Telephone {phone};mailbox {email};portal {url};session {ip}.",
        "Approved channels: <{email}> / {phone} / <{url}> / {ip}.",
        "Contact table\n{phone}\t{email}\t{url}\t{ip}",
        "Follow-up contact information: {phone}; {email}; {url}; {ip}.",
    ),
    "validation": (
        "Confirmed communication ({email}|{phone}|{url}|{ip}).",
        "The patient designated these channels: {phone};{email};{url};{ip}.",
        "CONTACT LIST [{email},{phone},{url},{ip}].",
        "Telecare details\n{phone}\t{email}\t{url}\t{ip}\nEnd of contact fields.",
        "Contact routing: <{email}>;<{url}>;{phone};{ip}.",
    ),
}


def _location(rng, split):
    root = rng.choice(GEOGRAPHY_ROOTS[split])
    suffix = rng.choice(GEOGRAPHY_SUFFIXES)
    location = root + suffix
    if rng.random() < 0.15:
        location += ", " + rng.choice(IDENTITIES[split]["state"])
    return location


def _parts(rng, split, index):
    values = _values(rng, split, index)
    values["LOCATION"] = {"location": MarkedValue("LOCATION", _location(rng, split))}
    age_frame = rng.choice(AGE_FRAMES[split])
    location_frame = rng.choice(LOCATION_FRAMES[split])
    email_frame = rng.choice(EMAIL_FRAMES[split])
    parts = [render_marked(age_frame, values["AGE"]),
             render_marked(location_frame, values["LOCATION"]),
             render_marked(email_frame, values["EMAIL"])]
    # Include a nearby personal identity, a named care facility, and an address
    # to teach taxonomy in realistic contexts rather than rely on a city list.
    for label in ("NAME", "FACILITY", "ADDRESS"):
        parts.append(render_marked(rng.choice(CLAUSES[split][label]), values[label]))
    contact_values = {**values["PHONE"], **values["EMAIL"], **values["URL"], **values["IP"]}
    parts.append(render_marked(rng.choice(CONTACT_BUNDLES[split]), contact_values))
    if index % 4 == 0:
        for label in ENTITY_TYPES:
            if label not in {"NAME", "FACILITY", "ADDRESS", "AGE", "LOCATION", "EMAIL", "PHONE", "URL", "IP"}:
                clause = rng.choice(CLAUSES[split][label])
                clause_values = values["BIRTH_DATE"] if label == "DATE" and "birth" in clause.casefold() else values[label]
                parts.append(render_marked(clause, clause_values))
    parts += [(sentence, []) for sentence in _negative_sentences(rng, split)]
    return parts


def generate_rows(count=3000, *, split="train", seed=790131):
    if count < 0 or split not in IDENTITIES:
        raise ValueError("Counts must be nonnegative; split must be train or validation")
    rng = random.Random(int.from_bytes(hashlib.sha256(f"context:{seed}:{split}".encode()).digest()[:8], "big"))
    for index in range(count):
        negative = index % 10 == 9
        parts = [(sentence, []) for sentence in _negative_sentences(rng, split)] if negative else _parts(rng, split, index)
        if negative:
            # These ages are clinical data. Reuse the complete grammatical frame
            # to teach the threshold rather than a superficial caption shortcut.
            age = str(rng.randint(1, 89))
            frame = rng.choice(AGE_FRAMES[split])
            text, _ = render_marked(frame, {"age": age})
            parts.append((text, []))
        rng.shuffle(parts)
        prefix = "Patient care and contact review\n" if split == "train" else "Care planning and communications\n"
        separator = rng.choice((" ", "\n", "\n\n", "\r\n", "\n• "))
        pieces = [prefix]
        spans = []
        offset = len(prefix)
        for part_index, (text, gold_spans) in enumerate(parts):
            if part_index:
                pieces.append(separator)
                offset += len(separator)
            pieces.append(text)
            spans += [{"start": span["start"] + offset, "end": span["end"] + offset,
                       "label": span["label"]} for span in gold_spans]
            offset += len(text)
        row = {"id": f"context-{split}-{seed}-{index:06d}", "text": "".join(pieces), "spans": spans,
               "family": f"context_{split}_{'negative' if negative else 'dense' if index % 4 == 0 else 'focused'}",
               "split": split, "suite": "authored_context_training" if split == "train" else "authored_context_validation"}
        validate_row(row)
        yield row


def write_dataset(output="data", train_count=3000, val_count=600, seed=790131):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for split, count in (("train", train_count), ("validation", val_count)):
        path = output / f"context_{split}.jsonl"
        digest = hashlib.sha256()
        entities = Counter()
        families = Counter()
        with path.open("w", encoding="utf-8", newline="") as handle:
            for row in generate_rows(count, split=split, seed=seed):
                line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                handle.write(line)
                digest.update(line.encode())
                entities.update(span["label"] for span in row["spans"])
                families.update([row["family"]])
        datasets[split] = {"file": path.name, "count": count, "sha256": digest.hexdigest(),
                           "entity_counts": dict(sorted(entities.items())), "family_counts": dict(families)}
    manifest = {"schema_version": 1, "generator": "outis.data.generators.context", "seed": seed,
                "license": "CC0-1.0", "generator_code_license": "Apache-2.0", "offset_unit": "unicode_code_points",
                "provenance": {"source": "Self-authored clinical context frames and invented geography, combined with project-authored identity vocabularies.",
                               "external_datasets": False, "external_patient_records": False,
                               "independent_final_evaluation_used": False,
                               "development_feedback": "Administrative geography, grammatical age variants, and punctuation-attached contact boundaries."},
                "split_audit": {"identity_vocabulary_intersection": [], "geography_roots_disjoint": True,
                                "context_frames_disjoint": True, "clinical_vocabulary_shared": True},
                "datasets": datasets}
    (output / "context_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--train-count", type=int, default=3000)
    parser.add_argument("--val-count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=790131)
    args = parser.parse_args()
    if args.train_count < 0 or args.val_count < 0:
        parser.error("Counts must be nonnegative")
    print(json.dumps(write_dataset(args.output, args.train_count, args.val_count, args.seed), indent=2))


if __name__ == "__main__":
    main()
