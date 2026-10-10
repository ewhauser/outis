"""Self-authored narrative PHI supervision with procedural identity generation.

The corpus is CC0-1.0. Names and places are assembled from a letter grammar,
not imported from an identity dictionary or patient dataset. Local tokenizer
filtering deliberately supplies single-token NAME and LOCATION supervision.
Training and development use separate identity partitions and sentence frames.
No independent qualification corpus or generator is read by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import re
from collections import Counter
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import tiktoken

from outis.data.generators.base import MarkedValue, render_marked, validate_row

CONSONANTS = "bcdfghjklmnprstvwyz"
VOWELS = "aeiou"
CARE_SUFFIXES = (
    "Community Hospital", "Family Practice", "Medical Centre", "Rehabilitation Clinic",
    "Regional Infirmary", "Outpatient Pavilion", "Home Nursing Service", "Care Residence",
    "Diagnostic Centre", "Infusion Clinic", "Children's Hospital", "Urgent Care",
)

# These are grammatical supervision frames, never an identity lookup table.
FRAMES = {
    "train": {
        "person": (
            "During review, {name} reported that the cough had resolved.",
            "{name} attended with a caregiver and described improved walking.",
            "The person giving this history was {name}, who denied new pain.",
            "Care recipient {name} explained that appetite remains good.",
            "The patient prefers to be addressed as {name} during consultations.",
            "History was obtained from the patient, {name}, without an interpreter.",
            "Patient called {name}; medication reconciliation completed.",
            "PATIENT / {name} / normal respiratory effort.",
        ),
        "clinician": (
            "Dr. {clinician} reviewed the examination findings with the family.",
            "The treating nurse, {clinician}, checked the home care plan.",
            "The referral was accepted by {clinician}, MD, after reviewing the images.",
            "Clinical reviewer {clinician} documented stable function.",
            "Seen by {clinician}; no change to the current medication dose.",
            "SIGNER / {clinician}, RN / observations reviewed.",
        ),
        "location": (
            "After discharge the family will return to {location} for home support.",
            "Travel from {location} took one hour without symptom worsening.",
            "A visiting nurse near {location} can provide wound care.",
            "The patient lives just outside {location} and has reliable transport.",
            "Their current home community is {location}, where family can assist.",
            "The patient relocated to {location} before this care episode.",
            "Home geography: {location}; access to stairs was discussed.",
            "RESIDENTIAL TOWN / {location} / transport already arranged.",
        ),
        "facility": (
            "An appointment at {facility} will assess the healing wound.",
            "The team from {facility} provided physiotherapy advice.",
            "The previous admission was managed at {facility}.",
            "ONGOING CARE SITE / {facility} / mobility review requested.",
        ),
        "date": (
            "The assessment occurred on {date}; follow-up is booked for {future_date}.",
            "Symptoms began on {past_date}, before the examination on {date}.",
            "Visit date {date}; the next review remains {future_date}.",
            "CARE EPISODE DATE / {date} / symptoms improving.",
        ),
        "age": (
            "This {age}-year-old patient remains independent with eating.",
            "The individual is aged {age} years and uses a walking frame.",
            "At {age} years of age, the care recipient reports restful sleep.",
            "Clinical age in years: {age}; no acute confusion.",
        ),
        "mrn": (
            "The medical record number {mrn} links this note to earlier admissions.",
            "Longitudinal chart key {mrn} was verified before prescribing.",
            "The patient's chart is retrieved using medical-file identifier {mrn}.",
            "MEDICAL RECORD REF / {mrn} / identity verified.",
        ),
        "account": (
            "The charge for this visit belongs to billing account {account}.",
            "Financial ledger reference {account} was checked by registration.",
            "The payment statement carries account number {account}.",
            "BILLING LEDGER REF / {account} / charge review pending.",
        ),
        "identifier": (
            "The blood sample's accession number is {identifier}.",
            "Encounter identifier {identifier} is printed on the appointment slip.",
            "The implanted device serial is {identifier}; interrogation was normal.",
            "The health-plan membership key supplied today is {identifier}.",
            "The patient's passport identifier is {identifier}.",
            "PERSONAL IDENTIFIER / {identifier} / registration complete.",
        ),
        "address": (
            "The visiting team should attend the home address {address}.",
            "Full postal residence was confirmed as {address}.",
            "HOME DELIVERY ADDRESS / {address} / supplies requested.",
        ),
        "phone": (
            "Call the patient using {phone} to confirm the appointment.",
            "Telephone for the caregiver: {phone}.",
            "CALLBACK NUMBER / {phone} / patient consent recorded.",
        ),
        "email": (
            "The agreed electronic reply address is {email}.",
            "Patient correspondence is directed to <{email}>.",
            "REPLY MAILBOX / {email} / instructions requested.",
        ),
        "ssn": (
            "Social-security number {ssn} was presented for identity verification.",
            "SOCIAL SECURITY IDENTIFIER / {ssn} / registration only.",
        ),
        "url": (
            "The patient's personal record link is {url}.",
            "PERSONAL PORTAL LINK / {url} / access assistance requested.",
        ),
        "ip": (
            "The home telecare connection used network address {ip}.",
            "PATIENT SESSION IP / {ip} / remote connection successful.",
        ),
    },
    "validation": {
        "person": (
            "In conversation, {name} described gradual improvement in stamina.",
            "The assessed patient introduced themself as {name}.",
            "For this history, {name} was able to answer without assistance.",
            "The name used by the care recipient is {name}.",
            "PATIENT IDENTITY >> {name} >> care goals confirmed.",
        ),
        "clinician": (
            "The clinician responsible for the visit is {clinician}, MD.",
            "Management advice came from Dr. {clinician} following review of the tests.",
            "The care assessment carries the signature of {clinician}, RN.",
            "EXAMINING PROFESSIONAL >> {clinician} >> findings reviewed.",
        ),
        "location": (
            "The household is based near {location} and has access to a pharmacy.",
            "The journey back to {location} can be made by private transport.",
            "Community support will be provided in {location} after this visit.",
            "Accommodation for the family is situated within {location}.",
            "The patient recently moved from {location} while retaining family support.",
            "PATIENT LOCALITY >> {location} >> home care discussed.",
        ),
        "facility": (
            "The specialist care provider for this episode is {facility}.",
            "Follow-up services will be delivered through {facility}.",
            "INSTITUTION OF CARE >> {facility} >> appointment requested.",
        ),
        "date": (
            "Clinical review took place {date}; the subsequent appointment is {future_date}.",
            "Onset was recorded as {past_date}, with assessment completed {date}.",
            "APPOINTMENT DAY >> {date} >> recovery reviewed.",
        ),
        "age": (
            "The patient's stated age is {age} years; oral intake is satisfactory.",
            "An adult aged {age} years underwent this functional assessment.",
            "Age at the consultation was {age} yr, with no reported fall.",
            "AGE OF CARE RECIPIENT >> {age} >> sleep pattern unchanged.",
        ),
        "mrn": (
            "This referral identifies the hospital medical record as {mrn}.",
            "The clinical chart number supplied for record retrieval is {mrn}.",
            "HOSPITAL FILE KEY >> {mrn} >> prior notes available.",
        ),
        "account": (
            "The financial statement for care uses billing reference {account}.",
            "Payment processing will use the patient's account key {account}.",
            "FINANCE ACCOUNT KEY >> {account} >> statement requested.",
        ),
        "identifier": (
            "The subscriber's insurance identifier was given as {identifier}.",
            "Sample accession reference {identifier} accompanies the laboratory request.",
            "This consultation has encounter reference {identifier}.",
            "Personal device identification number {identifier} was read during review.",
            "INDIVIDUAL REGISTRATION KEY >> {identifier} >> consent checked.",
        ),
        "address": (
            "The recorded delivery destination is {address}.",
            "Home visits will use the complete residence address {address}.",
            "POSTAL RESIDENCE >> {address} >> delivery approved.",
        ),
        "phone": (
            "The requested telephone callback destination is {phone}.",
            "Use the family contact number {phone} for the clinical update.",
            "TELEPHONE CONTACT >> {phone} >> callback planned.",
        ),
        "email": (
            "Electronic correspondence should go to ({email}).",
            "The mailbox selected by the patient is {email}.",
            "EMAIL DESTINATION >> {email} >> reply permitted.",
        ),
        "ssn": (
            "Registration recorded the social-security identifier as {ssn}.",
            "SOCIAL SECURITY KEY >> {ssn} >> identity check only.",
        ),
        "url": (
            "The individualized portal destination supplied today is {url}.",
            "RECORD ACCESS URL >> {url} >> support requested.",
        ),
        "ip": (
            "Remote consultation logged the patient's network IP as {ip}.",
            "REMOTE NETWORK ADDRESS >> {ip} >> connection checked.",
        ),
    },
}

NARRATIVE_FRAMES = {
    "train": (
        "{name} travelled from {location} to see {clinician} at {facility} on {date}.",
        "At {facility}, {clinician} assessed {name}; the family lives near {location}.",
        "Home support in {location} was discussed with {name} by {clinician} after the visit to {facility}.",
    ),
    "validation": (
        "For {name}, {clinician} will arrange care through {facility}; the household is in {location}.",
        "The visit to {facility} on {date} brought {name} from {location} for assessment by {clinician}.",
        "The family of {name} is based in {location}; {clinician} reviewed the care plan from {facility}.",
    ),
}

NEGATIVE_FRAMES = {
    "train": (
        "Parkinson disease and Cushing syndrome were considered; the Babinski response was absent.",
        "Addison disease is unlikely; Romberg testing and the Tinel sign were normal.",
        "A Turner phenotype was discussed, with no evidence of Sjögren syndrome.",
        "Pain 3/10, strength 5/5, Glasgow Coma Scale 15/15, MoCA score 27/30.",
        "Sodium 139 mmol/L, potassium 4.2 mmol/L, HbA1c 6.4%, and creatinine 0.9 mg/dL.",
        "Amlodipine 5 mg daily and metformin 500 mg twice daily were continued for 14 days.",
        "Published evidence (PMID {pmid}; publication year {year}) supports the clinical approach.",
        "The reference journal article was published in {year}; bibliographic PMID {pmid} is public.",
    ),
    "validation": (
        "Graves disease and Hashimoto thyroiditis remain alternatives; Hoffmann testing was negative.",
        "Crohn disease is unlikely, and no finding suggests Hodgkin lymphoma.",
        "Clinical discussion included Behçet disease; the Phalen maneuver caused no symptoms.",
        "Clinical scores: pain 2/10, power 4/5, coma scale 15/15, cognitive testing 26/30.",
        "Chemistry: sodium 140 mmol/L, potassium 4.1 mmol/L, HbA1c 6.1%, creatinine 1.0 mg/dL.",
        "Continue lisinopril 10 mg daily and paracetamol 500 mg as needed over 7 days.",
        "Public research citation PMID {pmid}, published during {year}, informed the review.",
        "Publication year {year} refers to the cited paper, whose public identifier is PMID {pmid}.",
    ),
}


def _partition(word: str) -> str:
    digest = hashlib.sha256(word.casefold().encode()).digest()
    return "validation" if int.from_bytes(digest[:4], "big") % 5 == 0 else "train"


@lru_cache(maxsize=1)
def _encoding():
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path.cwd() / ".cache" / "tiktoken"))
    return tiktoken.get_encoding("o200k_base")


@lru_cache(maxsize=2)
def single_token_words(split: str) -> tuple[str, ...]:
    """Enumerate fictitious word candidates rather than consult a name list."""
    if split not in FRAMES:
        raise ValueError("split must be train or validation")
    encoding = _encoding()
    words = set()
    for length in (2, 3, 4):
        alphabets = [CONSONANTS if position % 2 == 0 else VOWELS for position in range(length)]
        for letters in itertools.product(*alphabets):
            word = "".join(letters).capitalize()
            if _partition(word) == split and len(encoding.encode(word)) == 1 and len(encoding.encode(" " + word)) == 1:
                words.add(word)
    if len(words) < 50:
        raise ValueError("Tokenizer did not supply enough single-token procedural identities")
    return tuple(sorted(words))


def _invent_word(rng: random.Random, split: str, *, unicode: bool = False) -> str:
    while True:
        length = rng.choice((5, 6, 7, 8))
        word = "".join(rng.choice(CONSONANTS if index % 2 == 0 else VOWELS) for index in range(length)).capitalize()
        if unicode:
            vowels = [index for index, letter in enumerate(word) if letter.casefold() in VOWELS]
            index = rng.choice(vowels)
            accented = dict(zip(VOWELS, "áéíóú"))[word[index].casefold()]
            word = word[:index] + accented + word[index + 1:]
        if _partition(word) == split:
            return word


def _person(rng, split, style):
    first = rng.choice(single_token_words(split))
    if style < 5:
        return first
    last = _invent_word(rng, split, unicode=style == 9)
    if style == 6:
        return f"{first} {last}".upper()
    if style == 7:
        return f"{last}, {first}"
    if style == 8:
        return f"{first[0]}. {last}"
    if style == 9:
        return f"{_invent_word(rng, split, unicode=True)} {last}"
    return f"{first} {last}"


def _number(rng, split):
    # All three classes share every shape. The surrounding clinical role is the
    # only class cue; split ranges keep the actual protected values disjoint.
    number = rng.randrange(10_000_000, 50_000_000) if split == "train" else rng.randrange(50_000_000, 90_000_000)
    return rng.choice((str(number), f"{number:09d}", f"{number // 10000:04d}-{number % 10000:04d}",
                       f"{number // 10000:04d}/{number % 10000:04d}", f"{rng.choice('BCDPRV')}-{number}"))


def _format_date(value, index):
    return value.strftime(("%Y-%m-%d", "%d %b %Y", "%b %d, %Y", "%m/%d/%Y")[index % 4])


def _context(rng, split, index):
    name = _person(rng, split, index % 10)
    clinician = _person(rng, split, (index + 3) % 10)
    location = rng.choice(single_token_words(split))
    if index % 5 == 3:
        location = _invent_word(rng, split) + rng.choice((" Cove", " District", " Bay", " County"))
    elif index % 5 == 4:
        location = _invent_word(rng, split, unicode=True) + rng.choice((" Heights", " Valley", " Cross"))
    facility = _invent_word(rng, split) + " " + rng.choice(CARE_SUFFIXES)
    street = _invent_word(rng, split, unicode=index % 7 == 0)
    address = f"{rng.randint(10, 9999)} {street} {rng.choice(('Road', 'Lane', 'Way', 'Avenue'))}, {location}, ZZ {rng.randint(10000,99999)}"
    visit = date(2009 if split == "train" else 2019, 1, 1) + timedelta(days=rng.randrange(365 * 4))
    age = rng.randint(90, 110) if index % 5 != 3 else rng.randint(1, 89)
    namespace = "training" if split == "train" else "development"
    serial = index + 1
    telephone = f"+1 {'507' if split == 'train' else '629'} 555 01{index % 100:02d}"
    user = re.sub(r"[^a-z]", "", name.casefold()) or "person"
    m = MarkedValue
    return {
        "name": m("NAME", name), "clinician": m("NAME", clinician),
        "location": m("LOCATION", location), "facility": m("FACILITY", facility),
        "date": m("DATE", _format_date(visit, index)),
        "past_date": m("DATE", _format_date(visit - timedelta(days=rng.randint(7, 28)), index)),
        "future_date": m("DATE", _format_date(visit + timedelta(days=rng.randint(14, 42)), index)),
        "age": m("AGE", str(age)) if age > 89 else str(age),
        "mrn": m("MRN", _number(rng, split)), "account": m("ACCOUNT", _number(rng, split)),
        "identifier": m("ID", _number(rng, split)), "address": m("ADDRESS", address),
        "phone": m("PHONE", telephone), "email": m("EMAIL", f"{user}.{namespace}.{serial}@example.net"),
        "ssn": m("SSN", f"000-{'11' if split == 'train' else '22'}-{index % 9999 + 1:04d}"),
        "url": m("URL", f"https://records.example.org/{namespace}/person/{serial}"),
        "ip": m("IP", f"192.0.2.{rng.randint(1,254)}" if split == "train" else f"198.51.100.{rng.randint(1,254)}"),
    }


def _negatives(rng, split, index):
    frames = NEGATIVE_FRAMES[split]
    values = {"pmid": str(rng.randint(21_000_000, 38_999_999)), "year": str(rng.randint(1998, 2024))}
    # Rotate across an eponym, a quantitative or drug negative, and a public
    # citation. Values here are clinical or bibliographic, never protected PHI.
    return [render_marked(frames[index % 3], values),
            render_marked(frames[3 + index % 3], values),
            render_marked(frames[6 + index % 2], values)]


def _join(parts, header, separator):
    text_parts, spans = [header], []
    offset = len(header)
    for index, (text, gold_spans) in enumerate(parts):
        if index:
            text_parts.append(separator)
            offset += len(separator)
        text_parts.append(text)
        spans.extend({"start": span["start"] + offset, "end": span["end"] + offset,
                      "label": span["label"]} for span in gold_spans)
        offset += len(text)
    return "".join(text_parts), spans


def generate_rows(count=4000, *, split="train", seed=630429):
    if count < 0 or split not in FRAMES:
        raise ValueError("Count must be nonnegative; split must be train or validation")
    rng = random.Random(int.from_bytes(hashlib.sha256(f"narrative:{split}:{seed}".encode()).digest()[:8], "big"))
    keys = tuple(FRAMES[split])
    for index in range(count):
        style = index % 10
        kind = "clean" if style == 9 else "dense" if style < 6 else "focused"
        parts = _negatives(rng, split, index)
        values = _context(rng, split, index)
        if kind == "clean":
            parts.append(render_marked(rng.choice(FRAMES[split]["age"]), {"age": str(rng.randint(1, 89))}))
        else:
            chosen = keys if kind == "dense" else rng.sample(keys, rng.randint(4, 7))
            for key in chosen:
                parts.append(render_marked(rng.choice(FRAMES[split][key]), values))
            parts.append(render_marked(rng.choice(NARRATIVE_FRAMES[split]), values))
        rng.shuffle(parts)
        header = "Care history and follow-up discussion\n" if split == "train" else "Clinical history and continuity of care\n"
        text, spans = _join(parts, header, rng.choice((" ", "\n", "\n\n", "\r\n")))
        row = {"id": f"narrative-{split}-{seed}-{index:06d}", "text": text, "spans": spans,
               "family": f"narrative_{split}_{kind}", "split": split,
               "suite": "authored_narrative_training" if split == "train" else "authored_narrative_validation"}
        validate_row(row)
        if len(_encoding().encode(text)) > 500:
            raise ValueError(f"Narrative note exceeds the declared 500-token limit: {row['id']}")
        yield row


def _token_offsets(text, encoding):
    """o200k_base token ids and character ranges, for corpus statistics only."""
    ids = encoding.encode(text, disallowed_special=())
    _, starts = encoding.decode_with_offsets(ids)
    ends = starts[1:] + [len(text)]
    return ids, list(zip(starts, ends))


def _token_statistics(row, encoding):
    """Token count and BIOES tag counts under o200k_base, for the manifest."""
    ids, offsets = _token_offsets(row["text"], encoding)
    tags = Counter()
    for span in row["spans"]:
        covered = [
            (a, b)
            for a, b in offsets
            if a < span["end"]
            and b > span["start"]
            and row["text"][max(a, span["start"]) : min(b, span["end"])].strip()
        ]
        for k in range(len(covered)):
            boundary = (
                "S"
                if len(covered) == 1
                else "B"
                if k == 0
                else "E"
                if k == len(covered) - 1
                else "I"
            )
            tags[f"{boundary}-{span['label']}"] += 1
    tags["O"] = len(ids) - sum(tags.values())
    return len(ids), tags


def write_dataset(output="data", train_count=4000, val_count=1000, seed=630429):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    datasets, identity_surfaces, family_sets = {}, {}, {}
    for split, count in (("train", train_count), ("validation", val_count)):
        path = output / f"narrative_{split}.jsonl"
        digest, labels, families, tags = hashlib.sha256(), Counter(), Counter(), Counter()
        lengths, identities = [], set()
        with path.open("w", encoding="utf-8", newline="") as handle:
            for row in generate_rows(count, split=split, seed=seed):
                line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                handle.write(line)
                digest.update(line.encode())
                labels.update(span["label"] for span in row["spans"])
                families.update([row["family"]])
                length, note_tags = _token_statistics(row, _encoding())
                lengths.append(length)
                tags.update(note_tags)
                identities.update(row["text"][span["start"]:span["end"]].casefold()
                                  for span in row["spans"]
                                  if span["label"] in {"NAME", "LOCATION", "FACILITY", "ADDRESS", "EMAIL"})
        identity_surfaces[split], family_sets[split] = identities, set(families)
        datasets[split] = {
            "file": path.name, "count": count, "sha256": digest.hexdigest(),
            "entity_counts": dict(sorted(labels.items())), "family_counts": dict(sorted(families.items())),
            "token_counts": dict(sorted(tags.items())),
            "length_tokens": {"min": min(lengths, default=0), "max": max(lengths, default=0),
                              "mean": sum(lengths) / len(lengths) if lengths else 0},
            "unique_identity_surfaces": len(identities),
        }
    overlaps = identity_surfaces["train"] & identity_surfaces["validation"]
    if overlaps or family_sets["train"] & family_sets["validation"]:
        raise ValueError("Training and development narrative identities or families overlap")
    manifest = {
        "schema_version": 1, "generator": "outis.data.generators.narrative", "seed": seed,
        "license": "CC0-1.0", "generator_code_license": "Apache-2.0", "offset_unit": "unicode_code_points",
        "provenance": {"source": "Locally authored clinical narratives and procedural fictitious identities.",
                       "external_datasets": False, "external_patient_records": False,
                       "identity_lookup_dictionaries": False, "independent_qualification_used": False},
        "split_audit": {"identity_surfaces_disjoint_casefolded": True, "identity_overlap_count": len(overlaps),
                        "procedural_single_token_pools_disjoint": set(single_token_words("train")).isdisjoint(single_token_words("validation")),
                        "context_frames_disjoint": all(set(FRAMES["train"][key]).isdisjoint(FRAMES["validation"][key]) for key in FRAMES["train"]),
                        "narrative_frames_disjoint": set(NARRATIVE_FRAMES["train"]).isdisjoint(NARRATIVE_FRAMES["validation"]),
                        "clinical_negative_frames_disjoint": set(NEGATIVE_FRAMES["train"]).isdisjoint(NEGATIVE_FRAMES["validation"]),
                        "clinical_age_threshold_and_measurement_vocabulary_shared": True},
        "datasets": datasets,
    }
    (output / "narrative_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split, details in datasets.items():
        adjacent = {key: value for key, value in manifest.items() if key != "datasets"}
        adjacent.update(details)
        # Evaluation's frozen-corpus reader uses these canonical names; retain
        # the generator's count/entity_counts fields as descriptive aliases.
        adjacent["rows"] = details["count"]
        adjacent["label_support"] = details["entity_counts"]
        adjacent["split"] = split
        (output / f"narrative_{split}.jsonl.manifest.json").write_text(json.dumps(adjacent, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=630429)
    args = parser.parse_args()
    if args.train_count < 0 or args.val_count < 0:
        parser.error("Counts must be nonnegative")
    print(json.dumps(write_dataset(args.output, args.train_count, args.val_count, args.seed), indent=2))


if __name__ == "__main__":
    main()
