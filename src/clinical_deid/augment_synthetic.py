"""Author additional synthetic clinical NER examples with shuffled discourse.

This corpus is assembled from locally authored clauses, not clinical records or
an external dataset.  It extends the initial generator with identifier taxonomy
contrasts, names versus medical eponyms, public bibliography numbers, bare
geography, multilingual street names, and clinical numeric distractors.  Held-out
validation uses separate PHI identity vocabularies and clause families.
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

from .synthetic import MarkedValue, render_marked, validate_row
from .types import ENTITY_TYPES

IDENTITIES = {
    "train": {
        "first": (
            "Amina", "Bruno", "Cedric", "Danica", "Eira", "Fabian", "Ginevra", "Hector",
            "Ilana", "Jerome", "Kaia", "Ludovic", "Meryem", "Nestor", "Oriana", "Pia",
            "Raffael", "Soraya", "Tristan", "Ursula", "Valentin", "Winona", "Yvette", "Zoran",
            "Aurélien", "Cécile", "Damián", "Esmée", "François", "Leïla", "Máximo", "Sébastien",
            "Anaïs", "Björn", "Cătălin", "Désirée", "Elżbieta", "Göran", "Hélène", "João",
        ),
        "last": (
            "Ardenhall", "Briarcombe", "Cresswell", "Dewhurst", "Eastfen", "Frostholt",
            "Gorsefield", "Hearthmere", "Isenbrook", "Jadewell", "Kelmscott", "Lockwood",
            "Mistbourne", "Netherwick", "Oxbury", "Primrose", "Rushcroft", "Shoreham",
            "Tarnwood", "Upwell", "Velden", "Windermark", "Yonderhill", "Zephyrwood",
            "D'Ambray", "O'Mistwell", "de Rochel", "van Elsen", "al Saffron", "Saint-Clair",
            "Dufrêne", "Bärlund", "Caamaño", "López-Riva", "Dvořák", "Kovačević",
        ),
        "place": (
            "Briarfold", "Cressport", "Dewfen", "Eastmere", "Frostbay", "Gorsehaven",
            "Hearthwick", "Isenvale", "Jadebridge", "Kelmsbury", "Lockfen", "Mistport",
            "Netherbay", "Oxhaven", "Primrose Bend", "Rush Crossing", "Shore Hollow",
            "Tarnbridge", "Upmere", "Velden Bay", "Windermarch", "Yonder Falls",
        ),
        "street": (
            "Asterleaf", "Briarpatch", "Cress", "Dewdrop", "Eastwood", "Frost", "Gorse",
            "Hearth", "Jade", "Kelm", "Lock", "Mist", "Primrose", "Rush", "Shore",
            "Kastanienstraße", "Rue du Hibou", "Calle Lirio", "Allée des Tilleuls",
        ),
        "state": ("ZZ", "CO", "OR", "WY", "NV", "UT"),
    },
    "validation": {
        "first": (
            "Ansel", "Bianca", "Corinne", "Dominic", "Evelina", "Fintan", "Georgina", "Hamish",
            "Ivana", "Jacinta", "Kerstin", "Lorcan", "Marisol", "Nerissa", "Oswin", "Philippa",
            "Rasmus", "Saskia", "Thalia", "Ulysses", "Violeta", "Wilfred", "Yolanda", "Zinnia",
            "Agustín", "Benoît", "Élodie", "Fiorella", "Grégoire", "Iñaki", "Mélanie", "Sándor",
            "Aïcha", "Błażej", "Dóra", "Eugénie", "Françoise", "Günther", "Íris", "Jürgen",
        ),
        "last": (
            "Arrowcombe", "Baythorn", "Cragwell", "Downmere", "Ebbhurst", "Flintcombe",
            "Gallmere", "Hollowbrook", "Inkwood", "Loomfield", "Moorstead", "Newcombe",
            "Orcham", "Plumridge", "Reedsworth", "Strathwick", "Tidecroft", "Uppercombe",
            "Verdant", "Wavewell", "Yarnhurst", "Zenithvale", "D'Aubec", "O'Waveford",
            "de Merel", "van Orven", "el Roshan", "Saint-Fleur", "Deschênes", "Müllerwald",
            "Espiñeira", "Martín-Sol", "Šimek", "Vučević", "Ahlström", "Kırbaş",
        ),
        "place": (
            "Arrowfen", "Baymere", "Cragport", "Downhaven", "Ebbford", "Flintbay",
            "Gallbridge", "Hollowfen", "Inkvale", "Loomhaven", "Moor Crossing", "Newmere",
            "Orcham Bay", "Plum Hollow", "Reedhaven", "Strathmere", "Tidefold",
            "Upperfen", "Verdant Hollow", "Wave Crossing", "Yarnbridge", "Zenith Bay",
        ),
        "street": (
            "Arrowroot", "Bayleaf", "Crag", "Downland", "Ebb", "Flintstone", "Gall",
            "Hollow", "Ink", "Loom", "Moor", "Plum", "Reedbank", "Strath", "Tide",
            "Lindenstraße", "Rue des Grives", "Calle Magnolia", "Allée des Cèdres",
        ),
        "state": ("YY", "MN", "MI", "NY", "NJ", "PA"),
    },
}

EPONYMS = (
    "Wilson disease", "Baker cyst", "Parkinson disease", "Graves disease",
    "Crohn disease", "Hodgkin lymphoma", "Raynaud phenomenon", "Addison disease",
    "Cushing syndrome", "Hashimoto thyroiditis", "Guillain-Barré syndrome", "Bell palsy",
    "Friedreich ataxia", "Marfan syndrome", "Ehlers-Danlos syndrome", "Alzheimer disease",
    "Paget disease", "Behçet disease", "Huntington disease", "Wegener granulomatosis",
)
EXAM_NAMES = (
    "Murphy sign", "Romberg test", "Tinel sign", "Phalen maneuver", "Babinski response",
    "McBurney point tenderness", "Homan sign", "Brudzinski sign", "Kernig sign",
    "Weber test", "Rinne test", "Trendelenburg sign",
)
CLINICAL_FACTS = (
    "Serum sodium is 137 mmol/L; potassium 4.2 mmol/L; creatinine 1.1 mg/dL. No medication change is needed.",
    "Metformin was increased from 500 mg to 1000 mg twice daily. Home glucose is 105–165 mg/dL.",
    "The echocardiogram shows EF 58%. Blood pressure is 124/72 mmHg and heart rate is 67/min.",
    "Left ankle dorsiflexion is 12 degrees, gait distance 45 meters, and strength 4/5. Continue therapy for 6 weeks.",
    "Hemoglobin is 11.8 g/dL, platelets 205 ×10⁹/L, and white count 6.9 ×10⁹/L. Repeat CBC in 2 months.",
    "Apixaban 5 mg twice daily is tolerated without bleeding. There is no new edema or dyspnea.",
    "Temperature is 36.7°C, oxygen saturation 97% on room air, and respiratory rate 16/min.",
    "The lesion measures 1.8 × 0.9 cm. ICD-10 L98.9 and CPT 11102 describe this procedure.",
    "Levothyroxine 75 mcg is taken before breakfast. TSH 2.6 mIU/L and free T4 1.2 ng/dL are stable.",
    "A 10 mg prednisone dose was tapered by 2.5 mg every 7 days. Follow up after 90 days.",
    "MRI shows a small L3/L4 disc bulge and C5/C6 osteophytes. Pain is 4/10 with activity.",
    "The patient drinks 1.5 L per day, walks for 20 minutes, and sleeps 7 hours nightly.",
)

# Each label appears with multiple independently authored clinical/admin cues.
# Validation contains different paraphrases rather than shuffled training cues.
CLAUSES = {
    "train": {
        "NAME": (
            "Identity was checked with {name} before discussing the care plan.",
            "The patient identified themself as {name} during triage.",
            "Dr. {name} examined the patient and reconciled the medicine list.",
            "Nurse {name} confirmed the contact details during the intake call.",
            "The accompanying relative is {name}; consent was documented.",
            "Results were reviewed with {name}, who understood the instructions.",
            "Patient name entered in the chart: {name}.",
            "Signed electronically by {name}, MD.",
        ),
        "DATE": (
            "The infusion was administered on {date} without a reaction.",
            "On {date}, the wound was inspected and was dry.",
            "A specimen was obtained {date} before the first antibiotic dose.",
            "The chart lists date of birth as {date}.",
            "The clinician requested return on {date} for interval review.",
            "Visit date recorded by the scheduler: {date}.",
            "Discharge occurred {date} after stable observations.",
            "Procedure performed: {date}; blood loss was 12 mL.",
        ),
        "ADDRESS": (
            "The courier will deliver the home monitor to {address}.",
            "Correspondence should be mailed to {address}.",
            "Residence address verified during intake: {address}.",
            "Home nursing will attend at {address}.",
            "The home oxygen order gives the service address {address}.",
            "The patient moved to {address} before starting this treatment.",
            "Postal address of the family: {address}.",
            "For discharge transport, the destination is {address}.",
        ),
        "PHONE": (
            "The contact number for the care discussion is {phone}.",
            "A callback can be made to {phone} after the results are ready.",
            "The pharmacy requested a telephone reply at {phone}.",
            "Reach the patient on {phone}; no voicemail details should be left.",
            "Tel. {phone}; morning calls are preferred.",
            "The accompanying relative supplied {phone} as a backup number.",
            "Fax the signed plan using {phone}.",
            "Home telephone was reconfirmed: {phone}.",
        ),
        "EMAIL": (
            "The approved electronic contact is {email}.",
            "Education materials can be sent to {email} with consent.",
            "Correspondence email: {email}; use the secure workflow.",
            "The patient requested an electronic reply at {email}.",
            "Email address in the chart is {email}.",
        ),
        "MRN": (
            "The medical chart is indexed by {mrn}.",
            "Clinical record number {mrn} matched the wristband.",
            "The patient's medical record number is {mrn}.",
            "MRN {mrn} is the verified record for this visit.",
            "Chart no. {mrn} was used to retrieve prior laboratory results.",
            "Hospital record identifier: {mrn}.",
            "M.R.N. {mrn} appears on the specimen label.",
            "The chart number supplied by registration was {mrn}.",
        ),
        "SSN": (
            "Registration recorded social security number {ssn}.",
            "SSN supplied for identity verification: {ssn}.",
            "The guarantor's Social Security no. is {ssn}.",
            "The protected registration field contains SSN {ssn}.",
        ),
        "ACCOUNT": (
            "The billing ledger uses account {account} for these charges.",
            "Financial account number {account} was confirmed by the guarantor.",
            "Billing account: {account}; copayment was $25.",
            "Hospital billing acct. {account} is attached to the invoice.",
            "The payment account reference is {account}.",
            "Statement account number {account} will receive the charge.",
            "Guarantor billing account {account} remains active.",
        ),
        "ID": (
            "The insurance subscriber identifier is {identifier}.",
            "Policy number {identifier} was verified before authorization.",
            "Insurance member number: {identifier}.",
            "Driving licence number {identifier} was checked at registration.",
            "The identity check used passport {identifier}.",
            "Specimen accession number {identifier} is linked to the sample.",
            "Encounter identifier {identifier} refers to this clinical visit.",
            "Device serial no. {identifier} appears on the implant card.",
            "Professional license identifier for the signer: {identifier}.",
            "Authorization ID {identifier} was issued by the insurer.",
            "The patient's membership ID is {identifier}.",
            "Health plan member ID: {identifier}.",
        ),
        "URL": (
            "The patient's secure upload location is {url}.",
            "Appointments are managed through the personal link {url}.",
            "Use the patient portal at {url} for the signed instructions.",
            "The personal clinical record is accessible via {url}.",
        ),
        "IP": (
            "The video connection originated from IP {ip}.",
            "Network address {ip} was logged for the clinical session.",
            "The remote care device reported its IP address as {ip}.",
            "Session IP: {ip}; the consultation completed normally.",
        ),
        "AGE": (
            "The patient is {age} years of age and uses a walker.",
            "At age {age}, the patient remains independent with dressing.",
            "The {age}-year-old has stable appetite and weight.",
            "Age in years: {age}; medication safety was reviewed.",
        ),
        "LOCATION": (
            "The patient currently resides in {location}.",
            "The family relocated from {location} before this admission.",
            "A community nurse is being arranged in {location}.",
            "The patient travelled from {location} for this appointment.",
            "Home town: {location}.",
            "The patient will recover with family in {location}.",
            "Residence municipality recorded as {location}.",
            "Transport from {location} was arranged by the social worker.",
            "Current city of residence: {location}.",
        ),
        "FACILITY": (
            "The inpatient assessment was completed at {facility}.",
            "The referral destination is {facility}.",
            "The patient receives dialysis at {facility}.",
            "Admission to {facility} was arranged after assessment.",
            "Care facility: {facility}; transport has been confirmed.",
            "Laboratory testing is performed by {facility}.",
            "The surgical team at {facility} reviewed the scan.",
            "The discharge summary came from {facility}.",
        ),
    },
    "validation": {
        "NAME": (
            "The person attending today's review is {name}.",
            "Ms. {name} agreed to the proposed monitoring plan.",
            "Treating doctor {name} reported a normal examination.",
            "A family contact named {name} participated in teaching.",
            "Full patient identity on the request is {name}.",
        ),
        "DATE": (
            "The clinical appointment took place {date}.",
            "Review of the dressing is booked for {date}.",
            "The patient's birthdate is documented as {date}.",
            "The first inpatient examination was dated {date}.",
        ),
        "ADDRESS": (
            "The destination for the printed care plan is {address}.",
            "The home equipment installation is booked at {address}.",
            "Contact mailing location: {address}.",
            "The discharge destination provided by the patient is {address}.",
        ),
        "PHONE": (
            "Please telephone {phone} to confirm collection.",
            "The number designated for scheduling is {phone}.",
            "The callback telephone field contains {phone}.",
            "Patient telephone contact supplied today: {phone}.",
        ),
        "EMAIL": (
            "The email destination requested for instructions is {email}.",
            "Patient electronic contact address supplied today: {email}.",
            "Send the approved message using {email}.",
        ),
        "MRN": (
            "The medical records department located chart {mrn}.",
            "The longitudinal clinical chart number is {mrn}.",
            "Medical-record identifier verified today: {mrn}.",
            "Prior patient records were retrieved under MRN {mrn}.",
        ),
        "SSN": (
            "The social security registration field was verified as {ssn}.",
            "Protected SSN for the subscriber is {ssn}.",
        ),
        "ACCOUNT": (
            "Invoice billing account reference: {account}.",
            "The patient payment statement is associated with account {account}.",
            "The finance department confirmed billing account {account}.",
        ),
        "ID": (
            "The insurance coverage card gives member identifier {identifier}.",
            "The health plan's policy reference is {identifier}.",
            "The patient's passport number was documented as {identifier}.",
            "This visit has encounter number {identifier}.",
            "The accession key for the collected specimen is {identifier}.",
            "The patient's driver licence identifier is {identifier}.",
            "Unique device serial identifier: {identifier}.",
        ),
        "URL": (
            "The link associated with this patient's record is {url}.",
            "Personal portal address recorded for follow-up: {url}.",
        ),
        "IP": (
            "The patient's remote network IP was {ip}.",
            "The consultation connection recorded address {ip}.",
        ),
        "AGE": (
            "The person assessed is {age} years old.",
            "A {age}-year-old resident reports no new complaints.",
        ),
        "LOCATION": (
            "The patient's residence is within {location}.",
            "The family's temporary accommodation is in {location}.",
            "Discharge transportation is planned toward {location}.",
            "Municipality where the patient lives: {location}.",
        ),
        "FACILITY": (
            "The healthcare institution accepting the referral is {facility}.",
            "The follow-up care site is {facility}.",
            "The originating clinical institution was {facility}.",
        ),
    },
}


def _name(rng, vocabulary):
    first, last = rng.choice(vocabulary["first"]), rng.choice(vocabulary["last"])
    form = rng.randrange(12)
    if form == 0:
        return first[0] + ". " + last
    if form == 1:
        return last + ", " + first
    if form == 2:
        return first + " " + rng.choice("BCDFGHJKLMNPRSTV") + ". " + last
    if form == 3:
        return (first + " " + last).upper()
    return first + " " + last


def _date(rng, split, value=None):
    if value is None:
        value = date(2009 if split == "train" else 2021, 1, 1) + timedelta(days=rng.randrange(365 * 5))
    fmt = rng.choice(("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%d.%m.%Y",
                      "%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%d-%b-%Y",
                      "%m-%d-%Y", "%m/%d/%y", "%Y/%m/%d"))
    rendered = value.strftime(fmt)
    return re.sub(r"(?<!\d)0([1-9])(?!\d)", r"\1", rendered) if rng.random() < 0.3 else rendered


def _values(rng, split, index):
    vocabulary = IDENTITIES[split]
    name = _name(rng, vocabulary)
    place = rng.choice(vocabulary["place"])
    state = rng.choice(vocabulary["state"])
    street = rng.choice(vocabulary["street"])
    suffix = "" if street.startswith(("Rue", "Calle", "Allée")) or street.endswith("straße") else " " + rng.choice(("Road", "Street", "Avenue", "Lane", "Drive", "Court", "Rd", "St", "Ave"))
    unit = rng.choice(("", "", f", Apt {rng.randint(1, 99)}", f", Unit {rng.randint(1, 99)}", f", Suite {rng.randint(100, 699)}"))
    postal = rng.randint(20001, 29999) if split == "train" else rng.randint(50001, 59999)
    address = f"{rng.randint(12, 9999)} {street}{suffix}{unit}, {place}, {state} {postal}"
    facility = place + " " + rng.choice(("Hospital", "Medical Center", "Health Centre", "Family Clinic", "Regional Hospital", "Community Health", "Rehabilitation Unit", "Outpatient Pavilion", "Surgical Centre", "Nursing Home"))
    location = place if rng.random() < 0.7 else place + ", " + state
    number = rng.randint(10_000_000, 99_999_999)
    identifier = rng.choice((f"PX-{number % 1_000_000:06d}", f"QF{number}", f"{number}", f"U-{number % 100000:05d}-{index % 100:02d}", f"PL-{number % 100000:05d}/{index % 100:02d}"))
    mrn = rng.choice((str(number), f"HC-{number % 100000:05d}-{index % 100:02d}", f"M{number}", f"{number // 10000:04d}-{number % 10000:04d}"))
    account = rng.choice((f"BA-{number % 100000:05d}-{index % 1000:03d}", f"{number}{index % 100:02d}", f"F{number}", f"INV-{number % 1000000:06d}"))
    area = rng.choice((201, 204, 250, 604, 778) if split == "train" else (206, 212, 218, 612, 651))
    phone = rng.choice((f"{area}.555.01{index % 100:02d}", f"({area}) 555-01{index % 100:02d}",
                        f"{area}-555-01{index % 100:02d}", f"+1 {area} 555 01{index % 100:02d}",
                        f"1 ({area}) 555-01{index % 100:02d}", f"{area}55501{index % 100:02d}"))
    if rng.random() < 0.15:
        phone += rng.choice((" ext. ", " x", " extension ")) + str(rng.randint(10, 999))
    user = re.sub(r"[^a-z]", "", name.casefold())
    email = f"{user}.{index}@example.org"
    ssn = f"000-{rng.randint(1, 99):02d}-{rng.randint(1, 9999):04d}"
    ip = f"203.0.113.{rng.randint(1, 126)}" if split == "train" else f"198.51.100.{rng.randint(128, 254)}"
    age = rng.randint(90, 107)
    visit = date(2009 if split == "train" else 2021, 1, 1) + timedelta(days=rng.randrange(365 * 5))
    birth = date(visit.year - age, visit.month, min(visit.day, 28))
    fields = {
        "NAME": ("name", name), "DATE": ("date", _date(rng, split, visit)),
        "ADDRESS": ("address", address), "PHONE": ("phone", phone),
        "EMAIL": ("email", email), "MRN": ("mrn", mrn), "SSN": ("ssn", ssn),
        "ACCOUNT": ("account", account), "ID": ("identifier", identifier),
        "URL": ("url", f"https://followup.example.net/patient/{index}/{number}"),
        "IP": ("ip", ip), "AGE": ("age", str(age)),
        "LOCATION": ("location", location), "FACILITY": ("facility", facility),
    }
    marked = {label: {key: MarkedValue(label, value)} for label, (key, value) in fields.items()}
    marked["BIRTH_DATE"] = {"date": MarkedValue("DATE", _date(rng, split, birth))}
    return marked


def _negative_sentences(rng, split):
    eponym, second = rng.sample(EPONYMS, 2)
    exam = rng.choice(EXAM_NAMES)
    score = rng.randint(21, 29)
    pmid = rng.randint(20_000_000, 39_999_999)
    year = rng.randint(1997, 2024)
    age = rng.randint(18, 89)
    if split == "train":
        return [
            f"The differential includes {eponym}; {second} was considered less likely.",
            f"Examination includes {exam}; the result was negative.",
            f"Montreal Cognitive Assessment score was {score}/30; Glasgow Coma Scale was 15/15.",
            f"A public systematic review (PMID {pmid}, publication year {year}) supports the treatment choice.",
            f"An accompanying {age}-year-old adult took 5 mg at 08:00, with repeat dosing at 20:00; symptoms lasted 3 days.",
            rng.choice(CLINICAL_FACTS),
        ]
    return [
        f"The assessment considered {eponym} and found no evidence for {second}.",
        f"The documented {exam} was normal on examination.",
        f"Cognitive screening yielded {score}/30. Coma score remains 15/15 and muscle strength is 5/5.",
        f"The publicly available article indexed as PubMed PMID {pmid} appeared in {year}; this is a research citation.",
        f"Clinical example: age {age} years, dose 25 mcg, infusion 100 mL over 2 hours, temperature 37.0°C.",
        rng.choice(CLINICAL_FACTS),
    ]


def generate_rows(count=4000, *, split="train", seed=480217):
    """Generate dense, sparse, and clean examples with exact PHI boundaries."""
    if count < 0 or split not in IDENTITIES:
        raise ValueError("count must be nonnegative and split must be train or validation")
    rng = random.Random(int.from_bytes(hashlib.sha256(f"augmentation:{seed}:{split}".encode()).digest()[:8], "big"))
    labels = list(ENTITY_TYPES)
    for index in range(count):
        mode = "negative" if index % 10 == 9 else "sparse" if index % 10 == 8 else "dense"
        selected = [] if mode == "negative" else rng.sample(labels, rng.randint(3, 8)) if mode == "sparse" else list(labels)
        # Extra identifiers contrast clinical chart, billing account, and generic
        # encounter/policy/accession identifiers in the same immediate context.
        if mode == "dense" and index % 3 == 0:
            selected += ["ID", "MRN", "ACCOUNT"]
        if mode == "dense" and index % 4 == 0:
            selected += ["NAME", "DATE", "LOCATION"]
        parts = []
        values = _values(rng, split, index)
        for label in selected:
            clause = rng.choice(CLAUSES[split][label])
            clause_values = values[label]
            if label == "DATE" and "birth" in clause.casefold():
                clause_values = values["BIRTH_DATE"]
            if label == "NAME" and any(cue in clause for cue in ("Dr.", "Nurse ", "relative", "Signed", "doctor", "family contact")):
                clause_values = {"name": MarkedValue("NAME", _name(rng, IDENTITIES[split]))}
            rendered = render_marked(clause, clause_values)
            parts.append(rendered)
        clinical = _negative_sentences(rng, split)
        # Vary negative density without losing the most important distinctions.
        if mode == "sparse":
            clinical += rng.sample(list(CLINICAL_FACTS), 2)
        parts += [(sentence, []) for sentence in clinical]
        rng.shuffle(parts)
        separator = rng.choice((" ", "\n", "\n\n", "\r\n", "\n• "))
        prefix = rng.choice(("Clinical review\n", "Care coordination note\n", "History and monitoring\n", "")) if split == "train" else rng.choice(("Interval care record\n", "Multidisciplinary summary\n", ""))
        text_parts = [prefix]
        spans = []
        offset = len(prefix)
        for part_index, (text, gold_spans) in enumerate(parts):
            if part_index:
                text_parts.append(separator)
                offset += len(separator)
            text_parts.append(text)
            spans.extend({"start": span["start"] + offset, "end": span["end"] + offset, "label": span["label"]} for span in gold_spans)
            offset += len(text)
        row = {"id": f"augmentation-{split}-{seed}-{index:06d}", "text": "".join(text_parts),
               "spans": spans, "family": f"augmentation_{split}_{mode}_clauses", "split": split,
               "suite": "augmented_training" if split == "train" else "augmented_validation"}
        validate_row(row)
        yield row


def write_dataset(output="data", train_count=4000, val_count=1000, seed=480217):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for category in IDENTITIES["train"]:
        if set(IDENTITIES["train"][category]) & set(IDENTITIES["validation"][category]):
            raise ValueError("Identity vocabulary overlap between augmentation splits")
    datasets = {}
    for split, count in (("train", train_count), ("validation", val_count)):
        path = output / f"augmentation_{split}.jsonl"
        digest = hashlib.sha256()
        entities = Counter()
        families = Counter()
        total_characters = 0
        with path.open("w", encoding="utf-8", newline="") as handle:
            for row in generate_rows(count, split=split, seed=seed):
                line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                handle.write(line)
                digest.update(line.encode())
                entities.update(span["label"] for span in row["spans"])
                families.update([row["family"]])
                total_characters += len(row["text"])
        datasets[split] = {"file": path.name, "count": count, "sha256": digest.hexdigest(),
                           "entity_counts": dict(sorted(entities.items())), "family_counts": dict(families),
                           "total_characters": total_characters}
    manifest = {
        "schema_version": 1, "generator": "clinical_deid.augment_synthetic", "seed": seed,
        "license": "CC0-1.0", "generator_code_license": "Apache-2.0", "offset_unit": "unicode_code_points",
        "provenance": {"source": "Project-authored shuffled clinical clauses and invented identity vocabularies.",
                       "external_datasets": False, "external_patient_records": False,
                       "development_feedback": "Taxonomy, clinical negative distractors, and boundary error concepts informed augmentation; final evaluation source was not used."},
        "split_audit": {"identity_vocabulary_intersection": [], "clause_families_disjoint": True,
                        "clinical_vocabulary_shared": True, "numeric_values_may_overlap": True},
        "datasets": datasets,
    }
    (output / "augmentation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--output", type=Path, default=Path("data"))
    generate.add_argument("--train-count", type=int, default=4000)
    generate.add_argument("--val-count", type=int, default=1000)
    generate.add_argument("--seed", type=int, default=480217)
    args = parser.parse_args()
    if args.train_count < 0 or args.val_count < 0:
        parser.error("Counts must be nonnegative")
    print(json.dumps(write_dataset(args.output, args.train_count, args.val_count, args.seed), indent=2))


if __name__ == "__main__":
    main()
