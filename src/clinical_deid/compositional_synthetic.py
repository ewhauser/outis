"""CC0 clinical supervision with productive role and field grammars.

This replaces the first narrative training bank, preserving its per-note PHI
values, labels, name styles, and negative-note schedule exactly. Independent
grammar randomness changes only the wording, role placement, and layout. The
development banks and every qualification corpus remain untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from . import narrative_synthetic as baseline
from .synthetic import render_marked, validate_row

PREFIXES = (
    "",
    "During care review, ",
    "At registration, ",
    "Before discharge, ",
    "For continuity of care, ",
    "At the clinical visit, ",
)
OPERATORS = (": ", " = ", " — ", " is ", " was checked as ", " has value ", " → ")
SEPARATORS = ("; ", " / ", " | ", " — ", ". ")
METADATA = (
    "verified",
    "record checked",
    "details confirmed",
    "registration complete",
    "entry reviewed",
    "unchanged",
)
CLINICAL_ENDINGS = (
    "symptoms remain stable",
    "oral intake is adequate",
    "mobility has improved",
    "no acute pain was reported",
    "the medication list was reviewed",
    "support was arranged",
)
FIELD_NOUNS = ("number", "key", "reference", "identifier", "code", "index", "no.", "ID")
IDENTIFIER_SCOPES = {
    "mrn": (
        "medical record",
        "hospital medical chart",
        "longitudinal clinical file",
        "electronic medical record",
        "patient care record",
        "inpatient medical file",
        "outpatient clinical chart",
        "hospital chart",
    ),
    "account": (
        "billing account",
        "financial ledger",
        "patient payment account",
        "invoice account",
        "charge account",
        "billing statement account",
        "care payment ledger",
        "financial account",
    ),
    "identifier": (
        "encounter",
        "laboratory specimen accession",
        "insurance subscriber",
        "health-plan member",
        "passport",
        "driver's licence",
        "implant serial",
        "referral transaction",
    ),
}
PERSON_ROLES = (
    "patient",
    "care recipient",
    "person attending care",
    "individual receiving treatment",
)
CLINICIAN_ROLES = (
    "treating clinician",
    "responsible doctor",
    "examining nurse",
    "clinical reviewer",
    "attending physician",
    "care practitioner",
)
PERSON_ACTIONS = (
    "gave the history",
    "discussed the symptoms",
    "confirmed the care plan",
    "reviewed the medication list",
)
CLINICIAN_ACTIONS = (
    "completed the assessment",
    "reviewed the investigations",
    "arranged ongoing care",
    "documented the examination",
)
HOME_ROLES = (
    "home community",
    "residential area",
    "household locality",
    "home municipality",
    "usual place of residence",
)
HOME_ACTIONS = (
    "lives in",
    "resides near",
    "has a home in",
    "returns home to",
    "receives home support in",
)
CARE_ACTIONS = (
    "delivered the treatment",
    "provided the clinical service",
    "arranged outpatient follow-up",
    "completed the care review",
)
DATE_ROLES = {
    "date": (
        "visit day",
        "clinical assessment date",
        "day of examination",
        "appointment date",
    ),
    "past_date": (
        "symptom-onset day",
        "date the symptoms started",
        "recorded onset date",
    ),
    "future_date": (
        "next review date",
        "planned follow-up day",
        "scheduled return date",
    ),
}


def _rng(seed, namespace):
    digest = hashlib.sha256(f"{namespace}:{seed}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _prefaced(body, rng):
    # Keep most clinical sentences short; occasional workflow preambles still
    # vary the distance between a semantic cue and its protected value.
    prefix = rng.choice(PREFIXES) if rng.randrange(3) == 0 else ""
    if prefix and body.partition(" ")[0] in {
        "The",
        "A",
        "Use",
        "At",
        "From",
        "Home",
        "Name",
        "Named",
        "Clinical",
        "Identity",
    }:
        body = body[0].lower() + body[1:]
    return prefix + body


def _field(scope, noun, placeholder, rng):
    """Identifiers share shape and operators; clinical scope supplies the type."""
    value = "{" + placeholder + "}"
    metadata, separator = rng.choice(METADATA), rng.choice(SEPARATORS)
    form = rng.randrange(5)
    if form == 0:
        caption = f"{scope} {noun}"
        if rng.randrange(3) == 0:
            caption = caption.upper()
        return caption + rng.choice(OPERATORS) + value + separator + metadata + "."
    if form == 1:
        return (
            f"The {scope} {noun}"
            + rng.choice(OPERATORS)
            + value
            + separator
            + metadata
            + "."
        )
    if form == 2:
        return f"{value} is the {noun} recorded for the {scope}{separator}{metadata}."
    if form == 3:
        return (
            f"{scope} ({noun})"
            + rng.choice(OPERATORS)
            + value
            + separator
            + metadata
            + "."
        )
    return f"Use {value} as the {scope} {noun}{separator}{metadata}."


def _person_unit(key, rng):
    value = "{" + ("name" if key == "person" else "clinician") + "}"
    roles, actions = (
        (PERSON_ROLES, PERSON_ACTIONS)
        if key == "person"
        else (CLINICIAN_ROLES, CLINICIAN_ACTIONS)
    )
    role, action, ending = (
        rng.choice(roles),
        rng.choice(actions),
        rng.choice(CLINICAL_ENDINGS),
    )
    form = rng.randrange(6)
    if form == 0:
        return f"The {role} named {value} {action}; {ending}."
    if form == 1:
        return f"{value} {action} as the {role}; {ending}."
    if form == 2:
        return (
            f"Identity of the {role}"
            + rng.choice(OPERATORS)
            + value
            + rng.choice(SEPARATORS)
            + ending
            + "."
        )
    if form == 3:
        return f"The name supplied for the {role} is {value}; {ending}."
    if form == 4:
        if key == "clinician":
            title = (
                "Dr."
                if role in {"responsible doctor", "attending physician"}
                else "Nurse"
            )
            return f"{title} {value}, the {role}, {action}; {ending}."
        return f"The {role}, identified as {value}, {action}; {ending}."
    caption = f"{role} name".upper() if rng.randrange(2) else f"Name of {role}"
    credential = rng.choice((", MD", ", RN", "")) if key == "clinician" else ""
    return (
        caption
        + rng.choice(OPERATORS)
        + value
        + credential
        + rng.choice(SEPARATORS)
        + ending
        + "."
    )


def _location_unit(rng):
    role, ending = rng.choice(HOME_ROLES), rng.choice(CLINICAL_ENDINGS)
    form = rng.randrange(6)
    if form == 0:
        return f"The household {rng.choice(HOME_ACTIONS)} {{location}}; {ending}."
    if form == 1:
        return f"From {{location}}, the patient can reach the clinic; {ending}."
    if form == 2:
        return (
            f"The family's {role}"
            + rng.choice(OPERATORS)
            + "{location}"
            + rng.choice(SEPARATORS)
            + ending
            + "."
        )
    if form == 3:
        return f"{{location}} is the registered {role}; {ending}."
    if form == 4:
        return f"Home transport to {{location}} was coordinated; {ending}."
    caption = role.upper() if rng.randrange(2) else role.capitalize()
    return (
        caption
        + rng.choice(OPERATORS)
        + "{location}"
        + rng.choice(SEPARATORS)
        + ending
        + "."
    )


def _facility_unit(rng):
    action, ending = rng.choice(CARE_ACTIONS), rng.choice(CLINICAL_ENDINGS)
    choices = (
        f"The healthcare organisation {{facility}} {action}; {ending}.",
        f"{{facility}} {action} as the care provider; {ending}.",
        "Named treatment establishment"
        + rng.choice(OPERATORS)
        + "{facility}"
        + rng.choice(SEPARATORS)
        + ending
        + ".",
        f"Clinical services were received through {{facility}}; {ending}.",
        f"The documented care institution is {{facility}}; {ending}.",
    )
    return rng.choice(choices)


def _date_unit(key, rng):
    role, ending = rng.choice(DATE_ROLES[key]), rng.choice(METADATA)
    value = "{" + key + "}"
    return rng.choice(
        (
            f"The {role}" + rng.choice(OPERATORS) + value + "; " + ending + ".",
            f"{value} was recorded as the {role}; {ending}.",
            f"{role.upper()}" + rng.choice(OPERATORS) + value + " / " + ending + ".",
        )
    )


def _frame(kind, rng, fields=None):
    """Compose a grammatical training frame without consulting dev sentences."""
    if kind in {"person", "clinician"}:
        body = _person_unit(kind, rng)
    elif kind == "location":
        body = _location_unit(rng)
    elif kind == "facility":
        body = _facility_unit(rng)
    elif kind in IDENTIFIER_SCOPES:
        body = _field(
            rng.choice(IDENTIFIER_SCOPES[kind]), rng.choice(FIELD_NOUNS), kind, rng
        )
    elif kind == "date":
        body = " ".join(_date_unit(key, rng) for key in (fields or ("date",)))
    elif kind == "age":
        ending = rng.choice(CLINICAL_ENDINGS)
        body = rng.choice(
            (
                f"The care recipient is {{age}} years old; {ending}.",
                "Age in completed years"
                + rng.choice(OPERATORS)
                + "{age}; "
                + ending
                + ".",
                f"The patient, aged {{age}} years, attended the review; {ending}.",
                f"A {{age}} year old person attended care; {ending}.",
                f"At the current age of {{age}} yr, the person reports stable function; {ending}.",
            )
        )
    else:
        scopes = {
            "address": (
                "full home postal address",
                "home-visit address",
                "postal residence",
            ),
            "phone": (
                "patient callback telephone",
                "caregiver contact phone",
                "home telephone",
            ),
            "email": (
                "patient correspondence email",
                "electronic reply mailbox",
                "personal email address",
            ),
            "ssn": ("social-security identity", "social-security registration", "SSN"),
            "url": (
                "patient-specific portal URL",
                "personal clinical record link",
                "individual portal destination",
            ),
            "ip": (
                "home telecare IP address",
                "patient remote-session IP",
                "home network IP",
            ),
        }
        if kind not in scopes:
            raise ValueError(f"Unknown compositional frame kind: {kind}")
        body = _field(
            rng.choice(scopes[kind]),
            rng.choice(("value", "entry", "reference")),
            kind,
            rng,
        )
    return _prefaced(body, rng)


def _joint_frame(rng, fields):
    # Full role-bearing units prevent an adjacent patient/clinician NAME pair
    # from becoming an ambiguous comma-delimited list, including reversed names.
    patient = rng.choice(
        (
            "Patient: {name}.",
            "The patient is {name}.",
            "{name} attended as the patient.",
            "Care recipient name: {name}.",
        )
    )
    clinician = rng.choice(
        (
            "Treating clinician: {clinician}.",
            "The examining professional is {clinician}.",
            "{clinician} is the care practitioner.",
            "Clinical reviewer name: {clinician}.",
        )
    )
    home = rng.choice(
        (
            "Home area: {location}.",
            "The household lives in {location}.",
            "{location} is the home community.",
            "Transport home serves {location}.",
        )
    )
    facility = rng.choice(
        (
            "Care provider: {facility}.",
            "Treatment is provided at {facility}.",
            "{facility} is the healthcare service.",
            "Clinical institution: {facility}.",
        )
    )
    units = [patient, clinician, home, facility]
    if "date" in fields:
        units.append(_date_unit("date", rng))
    rng.shuffle(units)
    return _prefaced(rng.choice((" ", "\n", "\n\n")).join(units), rng)


def generate_rows(count=4000, *, seed=630429):
    """Preserve baseline RNG choices; an independent RNG changes prose only."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("Count must be a nonnegative integer")
    rng = _rng(seed, "narrative:train")
    grammar_rng = _rng(seed, "compositional-grammar")
    keys = tuple(baseline.FRAMES["train"])
    for index in range(count):
        kind = "clean" if index % 10 == 9 else "dense" if index % 10 < 6 else "focused"
        parts = baseline._negatives(rng, "train", index)
        values = baseline._context(rng, "train", index)
        if kind == "clean":
            rng.choice(baseline.FRAMES["train"]["age"])
            age = str(rng.randint(1, 89))
            parts.append(render_marked(_frame("age", grammar_rng), {"age": age}))
        else:
            chosen = keys if kind == "dense" else rng.sample(keys, rng.randint(4, 7))
            for key in chosen:
                original = rng.choice(baseline.FRAMES["train"][key])
                fields = re.findall(r"\{([^}]+)\}", original)
                parts.append(render_marked(_frame(key, grammar_rng, fields), values))
            original_joint = rng.choice(baseline.NARRATIVE_FRAMES["train"])
            parts.append(
                render_marked(
                    _joint_frame(
                        grammar_rng, re.findall(r"\{([^}]+)\}", original_joint)
                    ),
                    values,
                )
            )
        rng.shuffle(parts)
        separator = rng.choice((" ", "\n", "\n\n", "\r\n"))
        text, spans = baseline._join(
            parts, "Clinical coordination and record verification\n", separator
        )
        row = {
            "id": f"narrative-compositional-train-{seed}-{index:06d}",
            "text": text,
            "spans": spans,
            "family": f"narrative_compositional_train_{kind}",
            "split": "train",
            "suite": "authored_compositional_training",
        }
        validate_row(row)
        if len(baseline._encoding().encode(text)) > 500:
            raise ValueError(f"Compositional note exceeds 500 tokens: {row['id']}")
        yield row


def _protected_values(row):
    return Counter(
        (span["label"], row["text"][span["start"] : span["end"]])
        for span in row["spans"]
    )


def write_dataset(output="data", train_count=4000, seed=630429):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "narrative_compositional_train.jsonl"
    labels, families, tags, lengths = Counter(), Counter(), Counter(), []
    digest = hashlib.sha256()
    preserved = 0
    # Baseline generation supplies a proof that this intervention changes only
    # prose/layout: every row has exactly the same protected values and labels.
    originals = baseline.generate_rows(train_count, split="train", seed=seed)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row, original in zip(generate_rows(train_count, seed=seed), originals):
            if _protected_values(row) != _protected_values(original):
                raise ValueError(
                    "Compositional grammar changed baseline protected values or label counts"
                )
            preserved += 1
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode())
            labels.update(span["label"] for span in row["spans"])
            families.update([row["family"]])
            length, row_tags = baseline._token_statistics(row, baseline._encoding())
            lengths.append(length)
            tags.update(row_tags)
    manifest = {
        "schema_version": 1,
        "generator": "clinical_deid.compositional_synthetic",
        "seed": seed,
        "file": path.name,
        "rows": train_count,
        "count": train_count,
        "sha256": digest.hexdigest(),
        "split": "train",
        "license": "CC0-1.0",
        "generator_code_license": "Apache-2.0",
        "offset_unit": "unicode_code_points",
        "label_support": dict(sorted(labels.items())),
        "family_counts": dict(sorted(families.items())),
        "token_counts": dict(sorted(tags.items())),
        "length_tokens": {
            "min": min(lengths, default=0),
            "max": max(lengths, default=0),
            "mean": sum(lengths) / len(lengths) if lengths else 0,
        },
        "provenance": {
            "source": "Locally authored productive clinical role and field grammars, using existing procedural fictitious training values.",
            "external_datasets": False,
            "external_patient_records": False,
            "identity_lookup_dictionaries": False,
            "independent_qualification_used": False,
            "development_sentences_copied": False,
        },
        "experiment": {
            "baseline": "clinical_deid.narrative_synthetic train split",
            "per_note_protected_values_and_labels_unchanged": True,
            "rows_verified_against_baseline": preserved,
            "baseline_style_schedule_and_clean_ratio_unchanged": True,
            "changed_factors": [
                "Clinical grammatical combinations",
                "Role cue position",
                "Field operator and metadata",
                "In-note layout and punctuation",
            ],
            "development_corpora_modified": False,
        },
    }
    (output / "narrative_compositional_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=630429)
    args = parser.parse_args()
    if args.train_count < 0:
        parser.error("Counts must be nonnegative")
    print(json.dumps(write_dataset(args.output, args.train_count, args.seed), indent=2))


if __name__ == "__main__":
    main()
