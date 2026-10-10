"""Self-authored clinical role, identifier-taxonomy, and OCR NER supervision.

This supplemental corpus uses invented identities and new clinical frames.  It
does not import any patient corpus or independent final-evaluation generator.
Development diagnostics informed coverage categories, not sentence copying.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from outis.data.generators.augment import CLINICAL_FACTS, EPONYMS, EXAM_NAMES
from outis.data.generators.base import MarkedValue, render_marked, validate_row

VOCAB = {
    "train": {
        "first": ("Adele", "Bertram", "Clarissa", "Denzel", "Eleni", "Fergus", "Gilda", "Hermione",
                  "Irina", "Josiah", "Khadija", "Liesel", "Maurice", "Niamh", "Octavia", "Percival",
                  "Rosalind", "Simeon", "Tamsin", "Vaughn", "Wendell", "Yasmin", "Zelda", "Cyril",
                  "Adélaïde", "Bérénice", "Clément", "Emília", "Fátima", "Håkon", "Łucja", "Mårten"),
        "last": ("Avelmont", "Boskirk", "Cindrel", "Darnholm", "Esterfield", "Firthcombe", "Gresow",
                 "Hatherden", "Ilversham", "Kendalby", "Loxmere", "Malmwick", "Norvane", "Osterwell",
                 "Pellcombe", "Rothfen", "Sarnholm", "Tirren", "Ullmont", "Veldham", "Wickerson",
                 "Zellford", "d'Évermont", "de Malbrun", "von Gresow", "Al-Tirren", "O'Rothfen",
                 "São-Veldham", "Hågensen", "Błażek", "Chávez-Ren", "Kršovan"),
        "place": ("Avel", "Bosk", "Cindrel", "Darn", "Ester", "Firth", "Gres", "Hather", "Ilver",
                  "Kendal", "Lox", "Malm", "Norvan", "Oster", "Pell", "Roth", "Sarn", "Tirren",
                  "Ull", "Veld", "Wick", "Zell"),
        "street": ("Avelmont", "Bosk", "Cindrel", "Darn", "Ester", "Firth", "Gres", "Hather",
                   "Via Salice", "Rua da Amora", "Rue du Noyer", "Vogelstraße"),
        "state": ("KS", "NE", "OK", "TX", "MO", "ZZ"),
    },
    "validation": {
        "first": ("Althea", "Barnaby", "Conrad", "Dorothea", "Edmund", "Florence", "Gwendolyn", "Horace",
                  "Ilaria", "Judith", "Kristof", "Linnea", "Millicent", "Neville", "Ottilie", "Prudence",
                  "Ramona", "Severin", "Tatiana", "Viktor", "Wilhelmina", "Yvonne", "Zachariah", "Clemens",
                  "Álida", "Béatrice", "Céline", "Efrén", "Fernão", "Hjördis", "Łukasz", "Mélisande"),
        "last": ("Arnbrook", "Beldren", "Crevel", "Durnwick", "Eldholm", "Falkmere", "Gendahl",
                 "Hurstwick", "Ivelcombe", "Karsen", "Lindren", "Mernholt", "Norsham", "Orvel",
                 "Perrick", "Ruskholm", "Seldahl", "Tervane", "Uldwick", "Varncombe", "Weymont",
                 "Zedholm", "d'Orvel", "de Varnen", "von Karsen", "Al-Mernholt", "O'Ruskholm",
                 "São-Tervane", "Jørgensdal", "Żuraw", "Muñoz-Crevel", "Štrban"),
        "place": ("Arn", "Beld", "Crevel", "Durn", "Eld", "Falk", "Gendal", "Hurst", "Ivel",
                  "Kars", "Lindren", "Mern", "Nors", "Orvel", "Perr", "Rusk", "Seld", "Terv",
                  "Uld", "Varn", "Wey", "Zed"),
        "street": ("Arnbrook", "Beld", "Crevel", "Durn", "Eld", "Falk", "Gendal", "Hurst",
                   "Via Ginestra", "Rua do Freixo", "Rue des Aulnes", "Fasanenstraße"),
        "state": ("IA", "IL", "IN", "OH", "WI", "YY"),
    },
}

PLACE_ENDINGS = ("mere", "wick", "port", "fold", "bridge", " Crossing", " Meadow", " Quay", " Heath", " Vale")
CARE_SUFFIXES = ("Infirmary", "Medical Pavilion", "Cancer Institute", "Pediatrics", "Home Health",
                 "Children's Clinic", "Women's Health", "Infusion Centre", "Dialysis Unit", "Hospice",
                 "Hospital", "Sleep Clinic", "Memorial Hospital", "Rehabilitation Unit", "Nursing Home",
                 "Medical Associates", "Health Centre", "Outpatient Department", "Diagnostic Centre",
                 "Surgery Practice", "Urgent Treatment Centre", "Orthopedic Clinic", "Wellness Clinic")

FRAMES = {
    "train": {
        "person": (
            "The appointment belongs to {name}; the care plan was discussed in person.",
            "Patient identity supplied at check-in: {name}.",
            "A history was obtained directly from {name}.",
            "The patient's full name on the consent is {name}.",
            "{name} described improved appetite and no new weakness.",
            "Patient / {name} / symptoms unchanged.",
            "NAME :: {name}", "N A M E — {name}", "PATlENT NAME = {name}",
        ),
        "clinician": (
            "The examining professional was {clinician}, who reviewed the imaging.",
            "The team discussed management with Dr. {clinician}.",
            "The follow-up clinician is {clinician}.",
            "The examination was completed by {clinician}.",
            "Clinical signer {clinician}, RN, documented stable observations.",
            "PROVIDER :: {clinician}", "D0CT0R = {clinician}",
            "Clinician / {clinician} / medication list reviewed.",
        ),
        "facility": (
            "The ongoing care service is {facility}.",
            "The visit was arranged through {facility}.",
            "The clinical team at {facility} completed the examination.",
            "The patient attended {facility} for monitoring.",
            "The inpatient care location is {facility}.",
            "SITE OF CARE :: {facility}", "TREATMENT CENTrE = {facility}",
            "Care institution / {facility} / next appointment pending.",
        ),
        "location": (
            "The household has settled within {location}.",
            "The home-care team will visit the family's accommodation in {location}.",
            "The patient's relatives have a temporary home near {location}.",
            "The patient returned to the home town of {location} after the appointment.",
            "A journey out of {location} was needed for specialist care.",
            "The patient normally stays in {location}.",
            "The place where the family lives is {location}.",
            "The registered residential area is {location}.",
            "HOME TOWN :: {location}", "RESIDENCE AREA = {location}",
            "Geographic home area / {location} / transport arranged.",
        ),
        "age": (
            "The person being reviewed is {age} years old.",
            "The patient reports being {age} years of age.",
            "This examination concerns an individual aged {age} years.",
            "A resident aged {age} yrs has good appetite.",
            "The attending patient is {age} y/o.",
            "A {age}-year-old attended for medication review.",
            "The patient's age at this review is {age}.",
            "AGE IN YEARS :: {age}", "A G E = {age}",
            "Clinical age / {age} / mobility unchanged.",
        ),
        "mrn": (
            "Medical-file number {mrn} was verified against prior clinical notes.",
            "Patient chart reference {mrn} retrieves the longitudinal record.",
            "The verified medical record no. is {mrn}.",
            "The clinical chart key is {mrn}.",
            "MR N :: {mrn}", "MEDlCAL RECORD = {mrn}",
            "Chart reference / {mrn} / identity confirmed.",
        ),
        "account": (
            "The invoice will be posted to billing number {account}.",
            "The payment ledger carries account reference {account}.",
            "Financial billing reference {account} was checked by registration.",
            "This billing statement belongs to account {account}.",
            "BILLING ACCT :: {account}", "FINANCIAL ACC0UNT = {account}",
        ),
        "identifier": (
            "The encounter-specific number is {identifier}.",
            "The laboratory accession key is {identifier}.",
            "The identity card has patient number {identifier}.",
            "The insurance plan identifier is {identifier}.",
            "The implanted device carries serial number {identifier}.",
            "The passport used for identity verification has number {identifier}.",
            "The prescriber licence is recorded as {identifier}.",
            "The health-plan policy reference is {identifier}.",
            "SPECIMEN ACCESSION :: {identifier}", "ENCOUNTER N0 = {identifier}",
            "DEVICE SERIAL / {identifier} / interrogation normal.",
        ),
    },
    "validation": {
        "person": (
            "The request names {name} as the patient receiving care.",
            "History source / {name} / no new symptoms reported.",
            "The patient introduced themself using the name {name}.",
            "FULL lDENTITY >> {name}", "PATIENT'S NAME | {name}",
        ),
        "clinician": (
            "The responsible treating professional is {clinician}.",
            "The assessment bears the signature of {clinician}, MD.",
            "Dr. {clinician} accepted the referral after reviewing the tests.",
            "EXAMINER >> {clinician}", "D0CT0R'S NAME | {clinician}",
        ),
        "facility": (
            "The continuing care provider is {facility}.",
            "The specialist service was delivered by {facility}.",
            "The healthcare site documented for this visit is {facility}.",
            "CARE ESTABLISHMENT >> {facility}", "INSTITUTI0N | {facility}",
        ),
        "location": (
            "The family household is located within {location}.",
            "Temporary housing for the caregiver is near {location}.",
            "The patient went back to {location} after leaving hospital.",
            "The residential municipality provided at registration is {location}.",
            "HOME GE0GRAPHY >> {location}", "DOMICILE | {location}",
        ),
        "age": (
            "Age supplied for the care recipient is {age} years.",
            "The subject of today's review is a {age} year old.",
            "The person's age was confirmed as {age} yrs.",
            "AGE AT EXAM >> {age}", "PATIENT AGE/YRS | {age}",
        ),
        "mrn": (
            "The medical record key supplied with this referral is {mrn}.",
            "Longitudinal hospital chart reference {mrn} was retrieved.",
            "CHART NUMBER >> {mrn}", "PATIENT MEDlCAL FILE | {mrn}",
        ),
        "account": (
            "The financial statement lists billing reference {account}.",
            "Billing payment account {account} was provided for the charges.",
            "BILLING REFERENCE >> {account}", "PATIENT FINANCE ACCT | {account}",
        ),
        "identifier": (
            "The visit transaction uses encounter key {identifier}.",
            "The sample was registered with accession reference {identifier}.",
            "The subscriber's plan number is {identifier}.",
            "The equipment identification serial is {identifier}.",
            "Passport identifier presented today: {identifier}.",
            "ACCESSION REFERENCE >> {identifier}", "SERIAL IDENTlFIER | {identifier}",
        ),
    },
}

ROLE_RELATIONSHIPS = {
    "train": (
        "Review {date} at {facility} with {clinician}; patient {name} travels from {location}.",
        "Follow-up {date} @ {facility}, clinician {clinician}; the home district is {location}.",
        "At {facility}, {clinician} examined {name} on {date}; transport was arranged from {location}.",
        "The resident of {location}, {name}, is scheduled at {facility} on {date} with {clinician}.",
    ),
    "validation": (
        "For {name}, arrange {date} at {facility}; examiner {clinician}; home locality {location}.",
        "Clinical examination {date}, institution {facility}, performed by {clinician}; referred from {location}.",
        "{clinician} plans to assess {name} at {facility} on {date}. The family is based near {location}.",
    ),
}


def _name(rng, split, style):
    vocabulary = VOCAB[split]
    first, last = rng.choice(vocabulary["first"]), rng.choice(vocabulary["last"])
    if style == 0:
        return f"{first} {last}"
    if style == 1:
        return f"{first} {last}".upper()
    if style == 2:
        return f"{last}, {first}"
    return f"{first[0]}. {last}"


def _number(rng):
    # Same shapes can occur in MRNs, accounts, and generic identifiers; context
    # defines their taxonomy instead of a train-only prefix shortcut.
    number = rng.randint(1_000_000, 99_999_999)
    return rng.choice((str(number), f"{number:09d}", f"{number // 10000:04d}-{number % 10000:04d}",
                       f"{rng.choice('ABDFKMPV')}{number}", f"{rng.choice('ABDFKMPV')}-{number}",
                       f"{number // 10000:04d}/{number % 10000:04d}", f"RX-{number % 100000:05d}-{rng.randint(1,99):02d}"))


def _date(rng, value):
    result = value.strftime(rng.choice(("%Y-%m-%d", "%d %B %Y", "%b %d, %Y", "%m/%d/%Y",
                                       "%d/%m/%Y", "%d.%m.%Y", "%d-%b-%Y", "%m-%d-%Y")))
    return re.sub(r"(?<!\d)0([1-9])(?!\d)", r"\1", result) if rng.random() < 0.2 else result


def _context(rng, split, index):
    vocabulary = VOCAB[split]
    name = _name(rng, split, index % 4)
    clinician = _name(rng, split, (index + 2) % 4)
    location = rng.choice(vocabulary["place"]) + rng.choice(PLACE_ENDINGS)
    state = rng.choice(vocabulary["state"])
    facility = location + " " + rng.choice(CARE_SUFFIXES)
    number = rng.randint(10, 9999)
    street = rng.choice(vocabulary["street"])
    if not street.startswith(("Via", "Rua", "Rue")) and not street.endswith("straße"):
        street += " " + rng.choice(("Street", "Rd", "Way", "Avenue", "Terrace", "Lane", "Boulevard", "Place"))
    unit = rng.choice(("", "", f", Apt {rng.randint(1, 90)}", f", Suite {rng.randint(100, 899)}"))
    postal = rng.randint(60001, 69999) if split == "train" else rng.randint(30001, 39999)
    address = f"{number} {street}{unit}, {location}, {state} {postal}"
    area = rng.choice((214, 316, 402, 405, 417) if split == "train" else (219, 319, 513, 608, 715))
    phone = rng.choice((f"({area}) 555-01{index % 100:02d}", f"+1 {area} 555 01{index % 100:02d}",
                        f"{area}-555-01{index % 100:02d}", f"{area}.555.01{index % 100:02d}",
                        f"{area}55501{index % 100:02d}"))
    email_user = re.sub(r"[^a-z]", "", name.casefold())
    visit = date(2011 if split == "train" else 2021, 1, 1) + timedelta(days=rng.randrange(365 * 5))
    age = rng.randint(90, 109) if index % 5 else rng.choice((7, 14, 37, 55, 78, 88, 89))
    m = MarkedValue
    return {
        "name": m("NAME", name), "clinician": m("NAME", clinician), "date": m("DATE", _date(rng, visit)),
        "facility": m("FACILITY", facility), "location": m("LOCATION", location),
        "age": m("AGE", str(age)) if age > 89 else str(age),
        "mrn": m("MRN", _number(rng)), "account": m("ACCOUNT", _number(rng)),
        "identifier": m("ID", _number(rng)), "address": m("ADDRESS", address),
        "phone": m("PHONE", phone), "email": m("EMAIL", f"{email_user}.{index}@example.net"),
        "ssn": m("SSN", f"000-{rng.randint(1,99):02d}-{rng.randint(1,9999):04d}"),
        "url": m("URL", f"https://clinical.example.org/record/{index}/{number}"),
        "ip": m("IP", f"192.0.2.{rng.randint(1,126)}" if split == "train" else f"203.0.113.{rng.randint(128,254)}"),
    }


def _negative(rng, split):
    disease, second = rng.sample(EPONYMS, 2)
    exam = rng.choice(EXAM_NAMES)
    publication = rng.randint(1998, 2024)
    pmid = rng.randint(21_000_000, 38_999_999)
    if split == "train":
        return [
            f"There is no evidence of {disease}; {second} remains a diagnostic possibility. {exam} was normal.",
            "Pain 3/10, strength 5/5, Glasgow Coma Scale 15/15, and cognitive score 27/30 are unchanged.",
            f"The cited public review (PMID {pmid}, published {publication}) describes the evidence rather than this patient's dates.",
            rng.choice(CLINICAL_FACTS),
        ]
    return [
        f"The team excluded {disease} and discussed {second}. Testing of {exam} showed no abnormality.",
        "Clinical measurements: pain 2/10, strength 4/5, coma scale 15/15, and cognitive testing 26/30.",
        f"Bibliographic reference: public article PMID {pmid}, research publication year {publication}.",
        rng.choice(CLINICAL_FACTS),
    ]


def generate_rows(count=3000, *, split="train", seed=980321):
    if count < 0 or split not in VOCAB:
        raise ValueError("Count must be nonnegative; split must be train or validation")
    rng = random.Random(int.from_bytes(hashlib.sha256(f"coverage:{split}:{seed}".encode()).digest()[:8], "big"))
    for index in range(count):
        clean = index % 12 == 11
        values = _context(rng, split, index)
        parts = [(sentence, []) for sentence in _negative(rng, split)]
        if clean:
            young = str(rng.choice((9, 28, 48, 77, 88, 89)))
            parts.append(render_marked(rng.choice(FRAMES[split]["age"]), {"age": young}))
        else:
            keys = ["person", "clinician", "facility", "location", "age", "mrn", "account", "identifier"]
            for key in keys:
                parts.append(render_marked(rng.choice(FRAMES[split][key]), values))
            # Repeat role-bearing entities in a short joint clause to make the
            # semantic role visible when there are no standalone field captions.
            parts.append(render_marked(rng.choice(ROLE_RELATIONSHIPS[split]), values))
            if split == "train":
                contacts = rng.choice((
                    "Patient contacts / {phone};{email};{url};{ip}.",
                    "Home telephone {phone}; electronic mailbox {email}. Personal link {url}, remote IP {ip}.",
                    "CONTACTS :: {phone} | {email} | {url} | {ip}",
                ))
                address = rng.choice(("The full postal home address is {address}.", "HOME ADDR :: {address}", "Visit the patient's home at {address}."))
                ssn = rng.choice(("Protected registration SSN {ssn}.", "SOCIAL SECURITY :: {ssn}"))
            else:
                contacts = rng.choice((
                    "Approved contact record [{phone};{email};{url};{ip}].",
                    "Telephone callback {phone}, email reply {email}, patient portal {url}, network IP {ip}.",
                    "CONTACT FIELD >> {phone} | {email} | {url} | {ip}",
                ))
                address = rng.choice(("Delivery destination is the complete address {address}.", "POSTAL HOME >> {address}", "The recorded household address is {address}."))
                ssn = rng.choice(("Social-security identifier recorded for registration: {ssn}.", "SSN FIELD >> {ssn}"))
            parts += [render_marked(contacts, values), render_marked(address, values), render_marked(ssn, values)]
        rng.shuffle(parts)
        separator = rng.choice((" ", "\n", "\n\n", "\r\n", "\n• "))
        header = "Clinical assessment and registration\n" if split == "train" else "Care record and administrative review\n"
        text_parts = [header]
        spans = []
        offset = len(header)
        for position, (text, gold_spans) in enumerate(parts):
            if position:
                text_parts.append(separator)
                offset += len(separator)
            text_parts.append(text)
            spans += [{"start": span["start"] + offset, "end": span["end"] + offset,
                       "label": span["label"]} for span in gold_spans]
            offset += len(text)
        row = {"id": f"coverage-{split}-{seed}-{index:06d}", "text": "".join(text_parts), "spans": spans,
               "family": f"coverage_{split}_{'negative' if clean else 'roles_and_fields'}",
               "split": split, "suite": "authored_coverage_training" if split == "train" else "authored_coverage_validation"}
        validate_row(row)
        yield row


def write_dataset(output="data", train_count=3000, val_count=600, seed=980321):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for key in VOCAB["train"]:
        if set(VOCAB["train"][key]) & set(VOCAB["validation"][key]):
            raise ValueError("Coverage identity vocabularies overlap")
    datasets = {}
    for split, count in (("train", train_count), ("validation", val_count)):
        path = output / f"coverage_{split}.jsonl"
        digest = hashlib.sha256()
        labels = Counter()
        families = Counter()
        with path.open("w", encoding="utf-8", newline="") as handle:
            for row in generate_rows(count, split=split, seed=seed):
                line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                handle.write(line)
                digest.update(line.encode())
                labels.update(span["label"] for span in row["spans"])
                families.update([row["family"]])
        datasets[split] = {"file": path.name, "count": count, "sha256": digest.hexdigest(),
                           "entity_counts": dict(sorted(labels.items())), "family_counts": dict(families)}
    manifest = {"schema_version": 1, "generator": "outis.data.generators.coverage", "seed": seed,
                "license": "CC0-1.0", "generator_code_license": "Apache-2.0", "offset_unit": "unicode_code_points",
                "provenance": {"source": "Locally authored clinical role and field frames with invented identities.",
                               "external_datasets": False, "external_patient_records": False,
                               "independent_final_evaluation_used": False,
                               "development_feedback": "Clinical roles, organizational care suffixes, narrative geography, age grammar, identifier taxonomy, and OCR captions."},
                "split_audit": {"identity_vocabularies_disjoint": True, "context_frames_disjoint": True,
                                "clinical_negative_vocabulary_shared": True}, "datasets": datasets}
    (output / "coverage_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--train-count", type=int, default=3000)
    parser.add_argument("--val-count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=980321)
    args = parser.parse_args()
    if args.train_count < 0 or args.val_count < 0:
        parser.error("Counts must be nonnegative")
    print(json.dumps(write_dataset(args.output, args.train_count, args.val_count, args.seed), indent=2))


if __name__ == "__main__":
    main()
