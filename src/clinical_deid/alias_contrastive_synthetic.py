"""Authored full-name/alias clinical contrasts; Apache-2.0 code, CC0 data.

No patient records, retired note/generator, external dataset or qualification
payload is used. A full human identity grounds explicitly related first-name
and surname mentions; matched notes use those strings as public/nonpersonal
clinical language. Common English alias words may cross splits, whereas full
private identities use split-disjoint productive grammars.
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

from . import name_contrastive_synthetic as components
from .types import ENTITY_TYPES

VERSION = "1.0"
SEED = 1930741
COMMON_FIRST = (
    "Have",
    "For",
    "We",
    "He",
    "An",
    "The",
    "No",
    "Na",
    "I",
    "A",
    "To",
    "Do",
    "In",
    "Be",
    "Will",
    "May",
    "Not",
    "But",
    "Me",
    "Us",
    "You",
    "As",
    "It",
    "Or",
)
LAYOUTS = ("paragraphs", "bullets", "numbered", "handoff", "columns", "interview")
CAPTIONS = {
    "train": (
        "The {role}'s identity was checked; the full personal name is {v}.",
        "Private identification for the {role}: complete name = {v}.",
        "Identity statement signed by the {role}, full name: {v}.",
        "The {role} confirmed that their full name is {v}.",
        "Personal naming details supplied by the {role}: {v}.",
        "The {role}'s verified given name and surname appear as {v}.",
        "The personal signature on the {role}'s consent form reads {v}.",
        "Human identification / {role} / complete personal name: {v}.",
        "At the naming check, the {role} stated the full name {v}.",
        "The {role} entered the full identifying name '{v}' on the signed form.",
        "PERSONAL NAME OF {role_upper}, VERIFIED IN PERSON: {v}.",
        "Personal identity was established for the {role}, whose full name is {v}.",
        "The name personally read back by the {role} was {v}.",
        "The {role}'s identity declaration records the full name [{v}].",
        "The personal-name entry belonging to the {role} contains {v}.",
        "Full identity supplied directly by the {role} >> {v}.",
        "The {role} identified themself using the complete name {v}.",
        "Naming verification completed with the {role}; full name '{v}'.",
    ),
    "validation": (
        "The verified full identity name for this {role} is {v}.",
        "The {role} authenticated the complete personal name {v}.",
        "Under personal naming details, the {role} wrote [{v}].",
        "Identity was read aloud by the {role} as '{v}'.",
        "The {role} acknowledged the full personal-name field: {v}.",
        "Signed identification by the {role} gives the name {v}.",
        "The complete name associated with the {role}'s personal signature is {v}.",
        "The naming attestation identifies the {role} personally as {v}.",
        "FULL NAME, CONFIRMED PERSONALLY WITH {role_upper} — {v}.",
        "The {role} confirmed the full name written on their identity form: {v}.",
        "Personal naming check for the {role}, complete entry = {v}.",
        "The {role}'s identity check established the full name '{v}'.",
    ),
}
ALIAS_FRAMES = {
    "train": (
        "During the history discussion, {v} answered questions about usual activity.",
        "The account from {v} clarified when the symptoms became noticeable.",
        "When invited to explain the change, {v} described gradual recovery.",
        "After hearing the options, {v} agreed to arrange the next appointment.",
        "{v} will organise transport for the planned care review.",
        "With the care plan explained, {v} confirmed the appointment arrangements.",
        "Relatives of {v} described a home with level access.",
        "The family of {v} confirmed that support is available near their residence.",
        "{v}'s family explained how help is shared in the household.",
        "Care arrangements were discussed directly with {v}.",
        "The exercise was demonstrated by {v} after the explanation.",
        "The history supplied by {v} was checked against the symptom diary.",
        "For the review of their own history, {v} described changes in stamina.",
        "While discussing care, {v} asked how to keep the activity diary.",
        "The chart was shown to {v}, who confirmed the reported treatment use.",
        "On reviewing the history, {v} was able to explain the medicine routine.",
        "The {role}, known here by the given name {v}, answered the history questions.",
        "The {role} called {v} described the usual support arrangements.",
        "Using the personal short name {v}, the {role} confirmed the care discussion.",
        "The {role} identified by the surname {v} reviewed the appointment options.",
        "The named {role}, {v}, explained the recent symptom course.",
        "The {role}'s personal first-name entry is {v}; they answered the questions.",
        "The {role} bearing the personal name {v} agreed to arrange follow-up.",
        "The {role}, personally identified as {v}, demonstrated the seated exercise.",
        "Family members of the {role} named {v} described the residence as accessible.",
        "The person called {v}, acting as the {role}, confirmed the care arrangements.",
        "The {role}'s reduced personal-name entry reads {v} on this history form.",
        "As the named {role}, {v} took part in the discussion of monitoring.",
        "The {role} whose personal name is {v} offered the history directly.",
        "The {role}'s personal surname for this discussion is {v}.",
        "For this naming check, the {role} introduced themself as {v}.",
        "The care conversation continued with the {role} called {v}.",
    ),
    "validation": (
        "In recounting the symptom pattern, {v} provided further detail.",
        "Questions about daily function were answered by {v}.",
        "Following the explanation, {v} described the support already available.",
        "{v} agreed to coordinate the transport needed for care.",
        "The appointment details will be checked personally by {v}.",
        "After the plan was reviewed, {v} offered to make the care arrangements.",
        "Members of {v}'s family confirmed that the home has an accessible entrance.",
        "The relatives of {v} explained their usual household support.",
        "During discussion of the residence, {v} described the help provided by family.",
        "The review of past symptoms was completed with {v}.",
        "A demonstration of safe movement was supplied by {v}.",
        "For the personal history review, {v} supplied details of the daily routine.",
        "The {role}, using the personal given name {v}, explained the history.",
        "The {role} who introduced themself as {v} confirmed the arrangements.",
        "A direct naming check with the {role} established the short name {v}.",
        "The {role}, addressed personally by the surname {v}, discussed support.",
        "The person identified as {v} was the {role} explaining the symptoms.",
        "The {role}'s personal-name detail reads {v}; this individual answered the review.",
        "With identity confirmed, the {role} named {v} described the care needs.",
        "Family support was reviewed with the {role} bearing the personal name {v}.",
        "The {role} stated that {v} is the personal name used for this conversation.",
        "The {role} bearing the surname {v} gave the household history.",
        "The personal naming detail for the {role} is {v}, as confirmed by that person.",
        "During the identity check, the {role} supplied the personal name {v}.",
    ),
}
# Function words have stronger identity/role proof than ordinary nonce aliases.
SAFE_COMMON = {"train": tuple(range(16, 32)), "validation": tuple(range(12, 24))}
CLINICAL_STYLES = (
    "The head and neck examination was comfortable; swallowing remains unchanged.",
    "Oral inspection found moist mucosa and no new ulceration.",
    "Skin inspection showed intact pressure areas; regular repositioning was discussed.",
    "Cough is infrequent and airway clearance remains effective.",
    "Fatigue decreases after a seated rest and activity pacing was reviewed.",
    "Monofilament testing was explained as a clinical sensory assessment.",
    "The inhaler controller and spacer were demonstrated without distress.",
    "Otoscopy was tolerated and there was no new discharge.",
    "Presyncope was discussed in relation to abrupt standing and hydration.",
    "Disequilibrium has eased during controlled transitions from sitting.",
    "Inflammation is settling and joint movement remains possible.",
    "Medication adjustment was explained together with the observation schedule.",
    "S: comfort is improving. O: movement is steady. A/P: continue monitored activity.",
    "Teach-back confirmed understanding of the instruction to pause if dizziness occurs.",
    "The clinical team reviewed adherence and device handling during the session.",
    "Respiratory effort remains even and air entry was adequate at rest.",
    "Saturation was stable during the timed walk and seated recovery.",
    "Lower-limb power is sufficient for a supported transfer.",
    "Shoulder movement improved following a brief period of rest.",
    "Morning stiffness was discussed as a symptom rather than an identifying detail.",
    "The neurological screen included balance, sensation and controlled movement.",
    "Evidence review concerned the published monitoring guidance and ordinary reference ranges.",
    "Handoff observations covered treatment tolerance, hydration and safe pacing.",
    "The observation checklist records breathing, skin integrity and everyday function.",
)
PUBLIC_REFERENCES = (
    "The public monitoring guideline has publication year {year}.",
    "A public teaching manual revised on {year}-03-08 explains the examination scales.",
    "The public medicine reference was released in September {year}.",
    "The equipment information sheet has copyright year {year}.",
    "Public study NCT{digits} and bibliographic PMID {digits} provide background references.",
    "Public device model DEMO-{short} and product lot PUBLIC-{short} are catalogue information.",
    "Clinical score examples show strength 5/5, pain 3/10 and screening 28/30.",
    "The duration examples are 105 days and one hundred and twelve minutes.",
)
PUBLIC_HAVE = (
    "Staff explained that symptoms may {v} several ordinary contributing factors."
)


def _canonical(value):
    return unicodedata.normalize("NFC", value).casefold()


def identity(rng, split, pair, *, secondary=False):
    mode = pair % 10
    first = components._word(rng, split)
    surname = components._word(rng, split, vowel_first=False)
    common = not secondary and mode < 4
    if common:
        first = COMMON_FIRST[(pair // 10 + mode * 7) % len(COMMON_FIRST)]
    elif not secondary and mode == 4:
        first = rng.choice(components.CONSONANTS[split]).capitalize() + rng.choice(
            "aeiou"
        )
    elif not secondary and mode == 6:
        first = unicodedata.normalize(
            "NFD", components._word(rng, split, accented=True)
        )
    # Productive short names can coincidentally spell words such as He or No;
    # they require the same explicit human-role proof as designated collisions.
    common = first.casefold() in {value.casefold() for value in COMMON_FIRST}
    particle = rng.choice(("de", "van", "al", "von")) if mode == 8 else ""
    surname_part = f"{particle} {surname}" if particle else surname
    initial = (
        unicodedata.normalize("NFD", rng.choice(components.ACCENTED_VOWELS)) + "."
        if mode == 7
        else ""
    )
    suffix = rng.choice(("Jr.", "Sr.", "II", "III")) if mode == 9 else ""
    comma = pair % 4 == 2
    if mode == 5 or (common and pair % 7 == 0):
        first, surname_part = first.lower(), surname_part.lower()
        initial, suffix = initial.lower(), suffix.lower()
        case = "lower"
    elif pair % 17 == 0:
        first, surname_part = first.upper(), surname_part.upper()
        initial, suffix = initial.upper(), suffix.upper()
        case = "upper"
    else:
        case = "title"
    if comma:
        full = (
            " ".join(v for v in (surname_part, suffix) if v)
            + ", "
            + " ".join(v for v in (first, initial) if v)
        )
    else:
        full = " ".join(v for v in (first, initial, surname_part, suffix) if v)
    return {
        "full": full,
        "first": first,
        "surname": surname_part,
        "common_first": common,
        "case": case,
        "mode": mode,
        "comma_order": comma,
        "suffix": suffix,
    }


def generate_pair(pair, *, split="train", seed=SEED, attempt=0):
    if (
        split not in CAPTIONS
        or not isinstance(pair, int)
        or isinstance(pair, bool)
        or pair < 0
    ):
        raise ValueError(
            "A nonnegative pair index and train/validation split are required"
        )
    rng = random.Random(
        (seed << 32) + pair * 110503 + (0 if split == "train" else 570071) + attempt
    )
    person = identity(rng, split, pair)
    other = identity(rng, split, pair + 117, secondary=True)
    role = "caregiver" if pair % 4 == 3 else "patient"
    layout = rng.choice(LAYOUTS)
    profile = pair % len(CLINICAL_STYLES)
    header = f"Clinical {rng.choice(('discussion', 'progress review', 'observations', 'follow-up'))}: {rng.choice(components.TOPICS)}"
    body = components.clinical_body(rng, split, pair) + " " + CLINICAL_STYLES[profile]
    episode = date(2014, 1, 1) + timedelta(days=rng.randrange(3900))
    year = rng.randrange(1995, min(2020, episode.year - 1) + 1)
    public = pair % len(PUBLIC_REFERENCES)
    reference = PUBLIC_REFERENCES[public].format(
        year=year,
        digits=rng.randrange(10000000, 99999999),
        short=rng.randrange(10000, 99999),
    )
    reference += " Teaching terminology includes Romberg testing, Addison disease and Graves disease."
    notes = [components.shared.Note(), components.shared.Note()]
    for note in notes:
        note.plain(header + "\n" + body)
    positive, negative = notes
    separator = {
        "paragraphs": "\n",
        "bullets": "\n- ",
        "numbered": "\n1. ",
        "handoff": "\nIdentity and history: ",
        "columns": " | ",
        "interview": "\nDiscussion / ",
    }[layout]
    caption = rng.randrange(len(CAPTIONS[split]))
    positive.plain(separator)
    positive.clause(
        CAPTIONS[split][caption],
        "NAME",
        person["full"],
        role=role,
        role_upper=role.upper(),
    )
    mentions = [{"person": "primary", "kind": "full"}]
    contexts = [f"caption-{caption}"]
    for ordinal, kind in enumerate(
        ("first", "surname", rng.choice(("first", "surname"))), start=2
    ):
        value = person[kind]
        # Surname-only captions and first-name-only captions retain their meaning.
        choices = (
            list(SAFE_COMMON[split])
            if kind == "first" and person["common_first"]
            else list(range(len(ALIAS_FRAMES[split])))
        )
        excluded = (
            {19, 29}
            if kind == "first" and split == "train"
            else {15, 21}
            if kind == "first"
            else {16, 21}
            if split == "train"
            else {12}
        )
        choices = [frame for frame in choices if frame not in excluded]
        frame = rng.choice(choices)
        positive.plain(separator if layout != "numbered" else f"\n{ordinal}. ")
        positive.clause(ALIAS_FRAMES[split][frame], "NAME", value, role=role)
        contexts.append(f"alias-{frame}")
        mentions.append({"person": "primary", "kind": kind})
    if pair % 3 == 0:
        positive.plain("\n")
        positive.clause(
            "The consulting clinician personally confirmed the full name {v}.",
            "NAME",
            other["full"],
        )
        mentions.append({"person": "clinician", "kind": "full"})
        positive.plain(" ")
        positive.clause(
            "Dr. {v} explained the observation plan to the identified {role}.",
            "NAME",
            other["surname"],
            role=role,
        )
        mentions.append({"person": "clinician", "kind": "surname"})
        positive.plain("\nFor ")
        positive.phi(person["full"], "NAME")
        mentions.append({"person": "primary", "kind": "full"})
        positive.plain(", the consulting clinician, Dr. ")
        positive.phi(other["surname"], "NAME")
        mentions.append({"person": "clinician", "kind": "surname"})
        positive.plain(", confirmed the clinical arrangements.")
    negative.plain(separator)
    if person["common_first"]:
        value = person["first"]
        function = (
            PUBLIC_HAVE
            if value.casefold() == "have"
            else components.CLINICAL_FUNCTIONS[value.casefold()]
        )
        if pair % 2 == 0:
            negative.plain(rng.choice(("Nurse", "Clinician", "Caregiver")) + "\n")
        negative.plain(function.format(v=value))
    else:
        negative.plain(
            components.PUBLIC_FRAMES[
                rng.randrange(len(components.PUBLIC_FRAMES))
            ].format(v=person["first"])
        )
    negative.plain(separator)
    negative.plain(
        components.PUBLIC_FRAMES[(pair + 5) % len(components.PUBLIC_FRAMES)].format(
            v=person["surname"]
        )
    )
    negative.plain(separator)
    negative.plain(
        components.PUBLIC_FRAMES[(pair + 9) % len(components.PUBLIC_FRAMES)].format(
            v=person["full"]
        )
    )
    if pair % 5 == 0:
        positive.plain("\n")
        positive.clause(
            "The private clinical assessment occurred on {v}.",
            "DATE",
            episode.isoformat(),
        )
        positive.plain(" ")
        positive.clause(
            "The individual's next review was scheduled for {v}.",
            "DATE",
            (episode + timedelta(days=21)).isoformat(),
        )
    if pair % 4 == 1:
        positive.plain(
            "\n"
            + components.PUBLIC_FRAMES[
                (pair + 3) % len(components.PUBLIC_FRAMES)
            ].format(v=person["first"])
        )
    for note in notes:
        note.plain("\n" + reference)
    rows = []
    for kind, note in (("person", positive), ("clinical", negative)):
        row = {
            "id": f"alias-contrastive-{split}-{seed}-{pair:05d}-{kind}",
            "pair_id": f"alias-contrastive-{split}-{seed}-{pair:05d}",
            "text": note.text,
            "spans": note.spans,
            "split": split,
            "suite": f"authored_alias_contrastive_{'training' if split == 'train' else 'development'}",
            "family": f"alias_contrastive.{split}.{layout}.{kind}",
            "negative": kind == "clinical",
            "person_identity": person,
            "other_identity": other if pair % 3 == 0 else None,
            "person_role": role,
            "name_mentions": mentions if kind == "person" else [],
            "authored_contexts": contexts if kind == "person" else [],
            "clinical_profile": profile,
            "public_reference_context": public,
            "clinical_episode_year": episode.year,
            "publication_year": year,
        }
        validate_row(row)
        rows.append(row)
    return rows


def validate_row(row):
    if (
        row["split"] not in CAPTIONS
        or row["suite"]
        != f"authored_alias_contrastive_{'training' if row['split'] == 'train' else 'development'}"
    ):
        raise ValueError(
            "Only explicitly authored training/development rows are allowed"
        )
    end = 0
    names = []
    for span in row["spans"]:
        if (
            not 0 <= span["start"] < span["end"] <= len(row["text"])
            or span["start"] < end
            or span["label"] not in ENTITY_TYPES
        ):
            raise ValueError("Gold offsets must be ordered, disjoint and valid")
        end = span["end"]
        if span["label"] == "NAME":
            names.append(row["text"][span["start"] : span["end"]])
    if row["negative"] != (not row["spans"]) or len(names) != len(row["name_mentions"]):
        raise ValueError("Annotation role and identity-mention ownership disagree")
    for name, mention in zip(names, row["name_mentions"]):
        person = (
            row["person_identity"]
            if mention["person"] == "primary"
            else row["other_identity"]
        )
        if person is None or name != person[mention["kind"]]:
            raise ValueError(
                "Every alias must belong to its explicitly identified person"
            )
    if row["publication_year"] >= row["clinical_episode_year"]:
        raise ValueError("Public references must predate the private assessment")


def write_dataset(
    output="data",
    train_count=2000,
    val_count=500,
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
            raise ValueError("Each split requires a positive even note count")
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens < 100
    ):
        raise ValueError("Token ceiling must be an integer at least 100")
    comparison = components.shared.comparison_hashes(compare_corpora)
    forbidden_text = set().union(*(record["text_hashes"] for record in comparison))
    forbidden_ids = set()
    for record in comparison:
        payload = Path(record["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError("Approved comparison corpus changed during ID audit")
        forbidden_ids.update(
            json.loads(line)["id"]
            for line in payload.decode().splitlines()
            if line.strip()
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        split: output
        / f"alias_contrastive_{'train' if split == 'train' else 'validation'}.jsonl"
        for split in CAPTIONS
    }
    if any(
        path.exists() or path.with_suffix(".jsonl.manifest.json").exists()
        for path in paths.values()
    ):
        raise FileExistsError("Authored corpora are immutable; choose a new directory")
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(".cache/tiktoken").resolve()))
    encoding = tiktoken.get_encoding("o200k_base")
    seen_ids, seen_texts, train_full, aliases, manifests, drafts = (
        set(),
        set(),
        set(),
        {},
        {},
        [],
    )
    try:
        for split, count in (("train", train_count), ("validation", val_count)):
            path = paths[split]
            draft = path.with_suffix(".jsonl.tmp")
            drafts.append(draft)
            labels, tags, traits, contexts, profiles, references = (
                Counter() for _ in range(6)
            )
            full_names, alias_names, lengths, retries = set(), set(), [], 0
            with draft.open("x", encoding="utf-8") as handle:
                for pair in range(count // 2):
                    for attempt in range(40):
                        rows = generate_pair(
                            pair, split=split, seed=seed, attempt=attempt
                        )
                        stats = [
                            components.shared.token_audit(row, encoding) for row in rows
                        ]
                        identities = [
                            rows[0]["person_identity"],
                            rows[0]["other_identity"],
                        ]
                        private_full = {
                            _canonical(person["full"])
                            for person in identities
                            if person is not None
                        }
                        if max(n for n, _ in stats) <= max_tokens and not (
                            split == "validation" and private_full & train_full
                        ):
                            break
                    else:
                        raise ValueError(
                            "Could not author a short full-identity-disjoint pair"
                        )
                    retries += attempt
                    full_names.update(private_full)
                    for row, (length, row_tags) in zip(rows, stats):
                        digest = hashlib.sha256(row["text"].encode()).hexdigest()
                        if (
                            digest in seen_texts
                            or digest in forbidden_text
                            or row["id"] in seen_ids
                            or row["id"] in forbidden_ids
                        ):
                            raise ValueError(
                                "Authored text/ID overlaps a split or declared corpus"
                            )
                        seen_texts.add(digest)
                        seen_ids.add(row["id"])
                        lengths.append(length)
                        labels.update(s["label"] for s in row["spans"])
                        tags.update(row_tags)
                        profiles[str(row["clinical_profile"])] += 1
                        references[str(row["public_reference_context"])] += 1
                        if not row["negative"]:
                            contexts.update(row["authored_contexts"])
                            name_spans = [
                                span for span in row["spans"] if span["label"] == "NAME"
                            ]
                            for span, mention in zip(name_spans, row["name_mentions"]):
                                value = row["text"][span["start"] : span["end"]]
                                traits["name_spans"] += 1
                                traits[f"{mention['person']}_{mention['kind']}"] += 1
                                traits["fully_lowercase"] += value.islower()
                                traits["non_ascii"] += not value.isascii()
                                traits["nfd"] += value != unicodedata.normalize(
                                    "NFC", value
                                )
                                traits["possessive_following"] += (
                                    row["text"][span["end"] : span["end"] + 2] == "'s"
                                )
                                if mention["kind"] == "full":
                                    traits["comma_order_full"] += ", " in value
                                    traits["full_middle_initial"] += bool(
                                        re.search(r"\s[^\s,]+\.\s", value)
                                    )
                                    traits["full_generational_suffix"] += bool(
                                        re.search(
                                            r"\b(?:jr\.?|sr\.?|ii|iii)(?:,|$)",
                                            value,
                                            re.I,
                                        )
                                    )
                                else:
                                    alias_names.add(_canonical(value))
                                    traits["common_word_alias"] += _canonical(
                                        value
                                    ) in {v.casefold() for v in COMMON_FIRST}
                            traits["common_first_full_identity"] += row[
                                "person_identity"
                            ]["common_first"]
                            traits["short_nonce_first_full_identity"] += (
                                row["person_identity"]["mode"] == 4
                            )
                            traits["particle_full_identity"] += (
                                row["person_identity"]["mode"] == 8
                            )
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if split == "train":
                train_full = full_names
            aliases[split] = alias_names
            order = sorted(lengths)
            manifests[split] = {
                "schema_version": 1,
                "generator": "clinical_deid.alias_contrastive_synthetic",
                "generator_version": VERSION,
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "component_source_sha256": hashlib.sha256(
                    Path(components.__file__).read_bytes()
                ).hexdigest(),
                "shared_annotation_audit_helper_sha256": hashlib.sha256(
                    Path(components.shared.__file__).read_bytes()
                ).hexdigest(),
                "seed": seed,
                "rows": count,
                "sha256": hashlib.sha256(draft.read_bytes()).hexdigest(),
                "split": split,
                "suite": f"authored_alias_contrastive_{'training' if split == 'train' else 'development'}",
                "license": "CC0-1.0",
                "code_license": "Apache-2.0",
                "external_datasets": [],
                "source": "Original authored full-identity/reduced-alias clinical grammar, productive fictional identities and reused owned clinical/audit components; no copied retired sentences/identities/source or qualification input.",
                "used_for_training": split == "train",
                "used_for_checkpoint_selection": split == "validation",
                "independent_final": False,
                "development_scope": "New authored alias-style coverage, reported separately from previous development cohorts; no independent qualification claim.",
                "contrast_pairs": count // 2,
                "phi_free_notes": count // 2,
                "label_support": {label: labels[label] for label in ENTITY_TYPES},
                "bioes_tag_support": dict(sorted(tags.items())),
                "name_span_traits": dict(sorted(traits.items())),
                "person_context_span_support": dict(sorted(contexts.items())),
                "clinical_profile_note_support": dict(sorted(profiles.items())),
                "public_reference_note_support": dict(sorted(references.items())),
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
                    "length_or_identity_retries": retries,
                },
                "private_value_policy": "Full private identities are NFC/casefold-disjoint across splits. Common English first-name aliases, clinical vocabulary and public dates deliberately may overlap.",
                "private_values": {
                    "distinct_full_identities": len(full_names),
                    "full_identity_train_development_overlap": 0,
                    "distinct_reduced_aliases": len(alias_names),
                },
                "cross_split_audit": {
                    "id_overlap": 0,
                    "exact_text_overlap": 0,
                    "full_identity_overlap": 0,
                },
                "clinical_coherence": {
                    "aliases_belong_to_explicit_full_identity": True,
                    "distinct_clinician_identity_is_preserved": True,
                    "followup_is_21_days_after_assessment": True,
                    "publication_precedes_assessment_year": True,
                    "normal_age_doses_labs_scores_durations_public_dates_are_unlabelled": True,
                },
                "comparison_audit": [
                    {k: v for k, v in item.items() if k != "text_hashes"}
                    | {"exact_text_overlap": 0, "exact_id_overlap": 0}
                    for item in comparison
                ],
            }
        for manifest in manifests.values():
            manifest["cross_split_audit"]["shared_reduced_alias_values"] = len(
                aliases["train"] & aliases["validation"]
            )
        for split, path in paths.items():
            draft = path.with_suffix(".jsonl.tmp")
            os.link(draft, path)
            draft.unlink()
            with path.with_suffix(".jsonl.manifest.json").open(
                "x", encoding="utf-8"
            ) as handle:
                json.dump(manifests[split], handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        return manifests
    finally:
        for draft in drafts:
            draft.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--val-count", type=int, default=500)
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
                    key: manifest[key]
                    for key in (
                        "rows",
                        "sha256",
                        "phi_free_notes",
                        "token_length",
                        "name_span_traits",
                        "cross_split_audit",
                    )
                }
                for split, manifest in manifests.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
