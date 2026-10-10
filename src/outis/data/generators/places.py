"""Focused real-place curriculum under the clinical-phi-v2-places addendum.

Generated prose and annotations: CC0-1.0; source: Apache-2.0. Implements
decisions P1-P6 and N1-N5 of docs/annotation-policy-v2-places.md across four
layouts (terse query/chat, clinician shorthand, narrative sentence, structured
field/list line). Every positive unit is followed, three times in four, by a
matched negative in the same layout and split that reuses the positive's real
name where an N1-N5 role exists for it. Mixed rows put one institution in a
care role and a citation role ("care beats citation").

Real surface values come only from pinned US federal public-domain files and a
committed curated list (``outis.data.generators.place_sources``). Every real value, and
every abbreviation's or short name's expansion, must be in the ``train`` pool of
``place_value_pool``; the ``heldout`` pool is never emitted. Gold labels come
from each slot's authored role, never from a dictionary lookup. People are
invented from the shared remediation spelling pools and split by the v2
identity hash; dates, MRNs and 555-01xx phones are invented under the schemes
recorded in the manifest. Train and validation use disjoint template families,
and Census/CMS values are further partitioned so validation rows use place and
CMS facility names unseen in train (curated and public-reference lists are
shared). Names listed (as hashes only) in the pinned blind-bank exclusion file
are never used as values, and any row containing a 1-8 word sequence whose hash
is listed is rejected and resampled.

Regenerate (requires ``python -m outis.data.generators.place_sources fetch`` once)::

    PYTHONPATH=src .venv/bin/python -m outis.data.generators.places generate
    PYTHONPATH=src .venv/bin/python -m outis.data.generators.places check
    PYTHONPATH=src:scripts .venv/bin/python scripts/audit_corpus_quality.py \\
      --corpus train=data/places_train.jsonl --corpus validation=data/places_validation.jsonl

This is development training supervision from finite authored templates; it is
not an independent evaluation bank and no detector was run to build it.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
import functools
import hashlib
import json
from pathlib import Path
import random
import re
import string
from string import Formatter
import unicodedata

from outis.data.annotation_policy import POLICY_VERSION, annotation_errors, canonical_identity_group, identity_split
from outis.data.annotation_policy_places import ADDENDUM, place_value_pool
from outis.data.generators import place_sources
from outis.data.generators.places_contexts import (
    AMBIGUOUS_STATES, COHORT_STUDIES, COUNTRIES, DEVICES, DISEASES, DRUGS, FINDINGS, LABELLED_SLOTS, LAYOUTS,
    MATCHED_NEGATIVES, PLACE_GIVEN, PLACE_SURNAMES, PLACE_WORDS, REGIONS, STATES, SYMPTOMS, TEMPLATES,
)
from outis.data.generators.remediation import NAME_POOLS
from outis.types import ENTITY_TYPES
from outis.paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
VERSION = "1.0"
SEED = 26104
TRAIN_COUNT, VALIDATION_COUNT = 3000, 600
SPLITS = ("train", "validation")
SUITES = {"train": "training", "validation": "places_development"}
POSITIVES = ("p1", "p2", "p3", "p4", "p5", "p6", "mixed")
# One 27-slot cycle; layouts rotate once per cycle.
SCHEDULE = ("p1",) * 8 + ("p2",) * 4 + ("p3",) * 4 + ("p4",) * 3 + ("p5",) * 3 + ("p6",) * 3 + ("mixed",) * 2
PHENOMENON_TAGS = {
    "p1": "p1_patient_locality", "p2": "p2_care_site_abbreviation", "p3": "p3_care_site_without_type",
    "p4": "p4_full_care_site", "p5": "p5_care_site_then_locality", "p6": "p6_clinician_affiliation",
    "mixed": "mixed_roles", "n1": "n1_named_instrument", "n2": "n2_place_disease",
    "n3": "n3_institution_source", "n4": "n4_state_or_country", "n5": "n5_place_word_other_role",
}
FORM_TAGS = {"abbr": "p2_care_site_abbreviation", "short": "p3_care_site_without_type", "full": "p4_full_care_site"}
NEGATIVE_GROUPS = {
    "n1": (("n1_finding", .7), ("n1_device", .2), ("n1_study", .1)),
    "n2": (("n2", 1.),),
    "n3": (("n3", .78), ("n3_clinician", .22)),
    "n4": (("n4_state", .6), ("n4_country", .25), ("n4_region", .15)),
    "n5": (("n5_word", .55), ("n5_person", .45)),
}
POSITIVE_GROUPS = {"p1": "p1", "p2": "care_site", "p3": "care_site", "p4": "care_site", "p5": "p5", "p6": "p6", "mixed": "mixed"}
LEADING_ABBREVIATION = re.compile(r"^(?:St|Ste|Mt|Ft)\. ")
CARE_TYPE_WORDS = {"Hospital", "Hospitals", "Medical", "Center", "Centers", "Clinic", "Clinics", "Health", "Healthcare",
                   "Infirmary", "Institute", "Sanitarium", "Campus", "System"}
GENERIC_SHORT = {"Memorial", "Community", "General", "Regional", "Mercy", "Baptist", "Methodist", "Presbyterian",
                 "Providence", "Valley", "County", "University", "Children's", "Childrens", "Behavioral", "Psychiatric",
                 "Specialty", "Surgical", "Heart", "VA", "District", "Area", "City", "State", "Women's", "Rehabilitation",
                 "Mental", "Doctors", "Doctors'", "Lutheran", "Adventist", "Catholic", "Good", "Samaritan", "Holy",
                 "Cross", "Family", "Indian", "Public", "Veterans", "Kindred", "Select", "Oceans", "North", "South",
                 "East", "West", "Northern", "Southern", "Eastern", "Western", "Central", "Northeast", "Northwest",
                 "Southeast", "Southwest", "Saint", "St.", "Mt.", "Ft.", "Ste.", "Trinity", "Unity", "Union",
                 "Summit", "Grace", "Hope", "Faith", "Christian", "Advent", "AdventHealth", "Ascension", "Banner",
                 "Encompass", "Kaiser", "Sutter", "Sanford", "Avera", "Aspirus", "Christus", "Corewell"}
_ACRONYMS = {"VA", "UPMC", "HCA", "SSM", "CHI", "UNC", "UAB", "MUSC", "OSF", "IU", "UC", "UCSF", "UCLA", "NYU", "LDS",
             "SMP", "HSHS", "USA", "UCI", "UCSD", "UNM", "UT", "UVA", "OU", "KU", "LSU", "UAMS", "SLU", "WVU", "OHSU",
             "UW", "AMG", "USC", "UH", "TMC", "BSA", "JFK", "NEA", "MD", "ECU", "UMC", "II", "III", "HSC", "SCL", "AHN",
             "PIH", "UK", "UF", "UI", "UTMB", "VUMC", "CHS", "OSU", "MSU", "PHS", "UCHEALTH"}
_CAMEL = {"ADVENTHEALTH": "AdventHealth", "MYMICHIGAN": "MyMichigan", "CAROMONT": "CaroMont", "PROMEDICA": "ProMedica",
          "WELLSPAN": "WellSpan", "OHIOHEALTH": "OhioHealth", "MERCYONE": "MercyOne", "CHRISTIANACARE": "ChristianaCare",
          "UNITYPOINT": "UnityPoint", "MEDSTAR": "MedStar", "PEACEHEALTH": "PeaceHealth", "SAGEWEST": "SageWest",
          "CENTRACARE": "CentraCare", "HEALTHPARTNERS": "HealthPartners", "MAINEHEALTH": "MaineHealth",
          "DUPAGE": "DuPage", "UCHEALTH": "UCHealth", "TRUSTPOINT": "TrustPoint", "HEALTHONE": "HealthOne",
          "HONORHEALTH": "HonorHealth", "WAKEMED": "WakeMed", "ECHEALTH": "ECHealth", "MCLAREN": "McLaren"}
_LOWER = {"OF", "THE", "AND", "AT", "FOR", "IN", "ON", "DE", "DEL", "LA"}
_SAINTS = {"ST": "St.", "ST.": "St.", "MT": "Mt.", "MT.": "Mt.", "FT": "Ft.", "FT.": "Ft.", "STE": "Ste.", "STE.": "Ste."}
_CMS_REJECT = re.compile(
    r"[/(),#@\"]|\b(?:LLC|INC|INC\.|LP|L\.P\.|CTR|CTRS|CNTR|HOSP|MED|CAH|DBA|SVCS|REHAB|CORP|CO|LTD|PC|SYS|REG|MEM|GEN|"
    r"AUTH|DIST|MC|HLTH|HC|CTY|UNIV|SVC|MGMT|ACUTE|LTAC|LTACH|LLP|PLLC)\b| -|- |\s(?:AT|IN)\s|\b(?:OF|AND|AT|THE|&)$")
# Blind places bank v1: hashes of names that must never appear in training data.
EXCLUSIONS = ROOT / "data" / "places_exclusions.sha256.json"
EXCLUSIONS_SHA256 = "cfa5c5025bed2911eb06663ae7abdd58370e424d422f725acf45337e7b07f053"
# Measured on the v1.0 corpora (commit 8069787) before the exclusion filter existed.
CONTAMINATION_BEFORE_FILTER = {
    "corpus_sha256": {"train": "25a7cde543239ffd47b7a8d7ee0c41826b1e88dde588ddbaf5921ad9ae46168b",
                      "validation": "75d248b6fb327bb3a46d23149ddd55ef3868bf3aa637da79ade4da721f139e47"},
    "train": {"rows": 3000, "distinct_real_values": 1581, "value_level_matches": 5, "rows_with_ngram_match": 82},
    "validation": {"rows": 600, "distinct_real_values": 506, "value_level_matches": 2, "rows_with_ngram_match": 9},
}
_STRIP = string.punctuation + "\u2018\u2019\u201c\u201d"
MRN_SCHEME = "PL- prefix and eight invented random digits"
PHONE_SCHEME = "US 555-0100..0199 fictional lines with invented area codes"


def _norm(value: str) -> str:
    return unicodedata.normalize("NFC", value.strip()).casefold()


def exclusion_hash(value: str) -> str:
    """The blind bank's normalisation: sha256(NFC(value.strip()).casefold())."""
    return hashlib.sha256(unicodedata.normalize("NFC", value.strip()).casefold().encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=1)
def exclusion_hashes() -> frozenset:
    """Load the pinned hash-only exclusion list after verifying its SHA-256."""
    payload = EXCLUSIONS.read_bytes()
    if hashlib.sha256(payload).hexdigest() != EXCLUSIONS_SHA256:
        raise ValueError(f"{EXCLUSIONS} does not match its pinned SHA-256")
    hashes = json.loads(payload)["sha256"]
    if not all(re.fullmatch(r"[0-9a-f]{64}", item) for item in hashes):
        raise ValueError("Exclusion list must contain SHA-256 hashes only")
    return frozenset(hashes)


def word_sequences(text: str, longest: int = 8):
    """Every contiguous 1-8 whitespace-word sequence, punctuation-trimmed, with and without a trailing 's."""
    words = text.split()
    for start in range(len(words)):
        for size in range(1, min(longest, len(words) - start) + 1):
            sequence = " ".join(words[start:start + size]).strip(_STRIP)
            if not sequence:
                continue
            yield sequence
            for suffix in ("'s", "\u2019s"):
                if sequence.endswith(suffix):
                    yield sequence[:-len(suffix)].strip(_STRIP)


def excluded_sequences(text: str) -> int:
    """Number of 1-8 word sequences in ``text`` whose hash is on the exclusion list."""
    hashes = exclusion_hashes()
    return sum(exclusion_hash(sequence) in hashes for sequence in word_sequences(text) if sequence)


def excluded_value(value: str) -> bool:
    return exclusion_hash(value) in exclusion_hashes() or excluded_sequences(value) > 0


class ExcludedNameError(ValueError):
    """A generated row contains a name on the blind-bank exclusion list."""


def _city_key(value: str) -> str:
    words = _norm(value).replace(".", "").split()
    return " ".join({"saint": "st", "mount": "mt", "fort": "ft"}.get(word, word) for word in words)


def value_partition(value: str) -> str:
    """Second, independent hash that keeps Census/CMS names out of both splits at once."""
    digest = hashlib.sha256(("clinical-phi-v2-places/value-split:" + _norm(value)).encode("utf-8")).hexdigest()
    return "validation" if int(digest[:8], 16) % 5 == 0 else "train"


def train_pool(*values: str, expansion: str | None = None) -> bool:
    """True only if every value (and the shared expansion, if any) is in the train pool."""
    keys = [value for value in values if value]
    if expansion is not None:
        keys.append(expansion)
    return all(place_value_pool(key) == "train" for key in keys)


def title_case_cms(raw: str) -> str | None:
    """Readable surface form of an all-caps CMS facility name, or None if unsuitable."""
    name = " ".join(raw.upper().replace("&", " & ").split())
    if _CMS_REJECT.search(name) or len(name.split()) > 6:
        return None

    def word(token: str, first: bool) -> str:
        if token in _SAINTS:
            return _SAINTS[token]
        if token in _CAMEL:
            return _CAMEL[token]
        if token in _ACRONYMS:
            return token
        if token in _LOWER and not first:
            return token.lower()
        if "-" in token:
            return "-".join(word(part, True) for part in token.split("-"))
        if len(token) > 1 and token.isalpha() and not re.search(r"[AEIOUY]", token):
            return token
        out = token[:1] + token[1:].lower()
        if out.startswith("Mc") and len(out) > 2:
            out = "Mc" + out[2].upper() + out[3:]
        if out.startswith("O'") and len(out) > 2:
            out = "O'" + out[2].upper() + out[3:]
        return out

    words = [word(token, index == 0) for index, token in enumerate(name.split())]
    if words[0] == "The":
        words = words[1:]
    return " ".join(words) if words else None


def short_name(full: str) -> str | None:
    """Drop trailing care-type words ("Houston Methodist Hospital" -> "Houston Methodist")."""
    words = full.split()
    while words and words[-1] in CARE_TYPE_WORDS | {"&", "and", "of", "the"}:
        words.pop()
    if not words or len(words) == len(full.split()) or len(words) > 4:
        return None
    if all(word in GENERIC_SHORT or word in {"&", "and", "of", "the"} for word in words):
        return None
    if len(words) == 1 and (len(words[0]) < 5 or "-" not in words[0] and words[0] in GENERIC_SHORT):
        return None
    return " ".join(words)


@dataclass(frozen=True)
class Real:
    """A real surface value and the keys that decided its pool."""
    key: str
    kind: str
    source: str
    expansion: str | None = None
    partition: str = "shared"


@dataclass
class Bank:
    cities: dict = field(default_factory=dict)       # partition -> [city]
    well_known: dict = field(default_factory=dict)   # partition -> [city]
    counties: dict = field(default_factory=dict)
    cms_full: dict = field(default_factory=dict)
    cms_short: dict = field(default_factory=dict)
    curated: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    devices: list = field(default_factory=list)
    studies: list = field(default_factory=list)
    diseases: list = field(default_factory=list)
    states: list = field(default_factory=list)
    countries: list = field(default_factory=list)
    regions: list = field(default_factory=list)
    words: list = field(default_factory=list)
    given: list = field(default_factory=list)
    surnames: list = field(default_factory=list)
    place_names: frozenset = frozenset()
    heldout_pattern: re.Pattern | None = None
    state_ok: dict = field(default_factory=dict)
    excluded: Counter = field(default_factory=Counter)


@functools.lru_cache(maxsize=1)
def value_bank() -> Bank:
    """Load pinned sources and keep only train-pool values (deterministic)."""
    bank = Bank()
    places, counties, hospitals = place_sources.gazetteer_places(), place_sources.gazetteer_counties(), place_sources.cms_hospitals()
    curated = place_sources.curated_care_sites()
    bank.place_names = frozenset(_norm(place["name"]) for place in places)
    state_ok = {abbr: train_pool(name) for abbr, name in STATES.items()}
    reserved = {_norm(name) for name in STATES.values()} | {_norm(name) for name in COUNTRIES}
    gazetteer_by_key = {}
    for place in places:
        gazetteer_by_key.setdefault((_city_key(place["name"]), place["state"]), place["name"])
    hospital_count = Counter((_city_key(row["City/Town"]), row["State"]) for row in hospitals)
    known_keys = {key for key, count in hospital_count.items() if count >= 2 and key in gazetteer_by_key}
    # Text-level guard: well-known heldout names must never appear anywhere.
    heldout = set()
    for (key, state) in known_keys:
        name = gazetteer_by_key[key, state]
        if place_value_pool(name) == "heldout":
            heldout.add(name)
    heldout |= {name for name in STATES.values() if place_value_pool(name) == "heldout"}
    heldout |= {name for name in COUNTRIES + REGIONS if place_value_pool(name.removeprefix("the ")) == "heldout"}
    for entry in curated["entries"]:
        surfaces = entry["abbreviations"] + entry["short_names"] + entry["full_names"] + [entry["expansion"]]
        if place_value_pool(entry["expansion"]) == "heldout":
            heldout |= set(surfaces)
        heldout |= {surface for surface in surfaces if place_value_pool(surface) == "heldout"}
    bank.heldout_pattern = re.compile(
        r"(?<![\w-])(?:" + "|".join(re.escape(name) for name in sorted(heldout, key=len, reverse=True)) + r")(?![\w-])",
        re.IGNORECASE)

    def clean(value: str) -> bool:
        if bank.heldout_pattern.search(value):
            return False
        if excluded_value(value):
            bank.excluded["blind_exclusion_list"] += 1
            return False
        return True

    # e.g. "West Virginia" is refused when "Virginia" is heldout.
    state_ok = {abbr: ok and clean(STATES[abbr]) for abbr, ok in state_ok.items()}
    bank.state_ok = state_ok

    for partition in SPLITS:
        for key in ("cities", "well_known", "counties", "cms_full", "cms_short"):
            getattr(bank, key)[partition] = []
    seen = set()
    for place in places:
        name, state = place["name"], place["state"]
        if (name, state) in seen:
            continue
        seen.add((name, state))
        if name.startswith("The "):
            # A leading article would blur the "article outside the span" rule.
            bank.excluded["gazetteer_place_leading_article"] += 1
            continue
        bare_ok = _norm(name) not in reserved
        if not (train_pool(name) and clean(name) and (bare_ok or state_ok[state])):
            bank.excluded["gazetteer_place"] += 1
            continue
        city = {"name": name, "state": state, "lsad": place["lsad"], "bare_ok": bare_ok,
                "partition": value_partition(name)}
        bank.cities[city["partition"]].append(city)
        if (_city_key(name), state) in known_keys:
            bank.well_known[city["partition"]].append(city)
    for county in counties:
        if train_pool(county["name"]) and clean(county["name"]):
            bank.counties[value_partition(county["name"])].append(county)
        else:
            bank.excluded["gazetteer_county"] += 1
    full_seen, short_seen = set(), set()
    for row in hospitals:
        full = title_case_cms(row["Facility Name"])
        city = gazetteer_by_key.get((_city_key(row["City/Town"]), row["State"]))
        if full is None or not clean(full):
            bank.excluded["cms_unsuitable_name"] += 1
            continue
        if not train_pool(full):
            bank.excluded["cms_heldout"] += 1
            continue
        usable_city = city and train_pool(city) and clean(city) and not city.startswith("The ") \
            and (_norm(city) not in reserved or state_ok[row["State"]])
        site = {"name": full, "city": city if usable_city else None, "state": row["State"],
                "source": "cms", "facility_id": row["Facility ID"], "partition": value_partition(full)}
        has_type = any(word in CARE_TYPE_WORDS for word in full.split())
        if has_type and full not in full_seen:
            full_seen.add(full)
            bank.cms_full[site["partition"]].append(site)
        short = short_name(full) if has_type else (full if len(full.split()) <= 4 else None)
        if short and short not in short_seen and _norm(short) not in bank.place_names | reserved \
                and train_pool(short) and clean(short):
            short_seen.add(short)
            bank.cms_short[site["partition"]].append({**site, "name": short, "full_name": full})
    for entry in curated["entries"]:
        if place_value_pool(entry["expansion"]) != "train":
            bank.excluded["curated_heldout_expansion"] += 1
            continue
        if not clean(entry["expansion"]):
            bank.excluded["curated_excluded_expansion"] += 1
            continue
        keep = {form: [s for s in entry[key] if train_pool(s, expansion=entry["expansion"]) and clean(s)]
                for form, key in (("abbr", "abbreviations"), ("short", "short_names"), ("full", "full_names"))}
        cite = keep["abbr"] + keep["short"] + ([entry["expansion"]] if not re.search(r"^The |Hospital", entry["expansion"]) else [])
        cite += [s for s in keep["full"] if "Clinic" in s and s not in cite]
        cite = [s for s in cite if clean(s)]
        city_ok = train_pool(entry["city"]) and clean(entry["city"])
        bank.curated.append({**entry, "forms": keep, "cite_names": cite if entry["citation_ok"] else [],
                             "city": entry["city"] if city_ok else None, "source": "curated"})

    def keep_key(key, raw):
        if train_pool(key) and clean(key) and clean(raw):
            return True
        bank.excluded["public_reference"] += 1
        return False

    bank.findings = [item for item in FINDINGS if keep_key(item[0], item[1]) and all(clean(phrase) for phrase in item[2])]
    bank.devices = [item for item in DEVICES if keep_key(*item)]
    bank.studies = [item for item in COHORT_STUDIES if keep_key(*item)]
    bank.diseases = [item for item in DISEASES if keep_key(*item)]
    bank.states = [name for name in STATES.values() if name not in AMBIGUOUS_STATES and keep_key(name, name)]
    bank.countries = [name for name in COUNTRIES if keep_key(name, name)]
    bank.regions = [name for name in REGIONS if keep_key(name.removeprefix("the "), name)]
    bank.words = [item for item in PLACE_WORDS if _norm(item[0]) in bank.place_names and keep_key(item[0], item[0])
                  and all(clean(clause) for clause in item[1])]
    bank.given = [name for name in PLACE_GIVEN if keep_key(name, name)]
    bank.surnames = [name for name in PLACE_SURNAMES if keep_key(name, name)]
    return bank


# --------------------------------------------------------------------------
# Invented people, dates and identifiers


def identity(rng: random.Random, split: str | None = None, *, given: str | None = None,
             surname: str | None = None, bank: Bank | None = None) -> dict:
    """Invent a full identity; with ``split`` it is redrawn until its hash partition matches."""
    for _ in range(10000):
        firsts, lasts = rng.choice(NAME_POOLS)
        first, last = given or rng.choice(firsts), surname or rng.choice(lasts)
        middle = f" {rng.choice('ABCDEFGHJKLMNPRSTVW')}." if rng.random() < .2 else ""
        full = f"{first}{middle} {last}"
        group = canonical_identity_group(full)
        if bank is not None and (bank.heldout_pattern.search(full) or excluded_value(full)):
            continue
        if split is None or identity_split(group) == split:
            return {"given": first, "surname": last, "full": full, "group": group}
    raise RuntimeError("Could not invent an identity in the requested partition")


def _date(rng: random.Random, start: int = 2012, end: int = 2025) -> str:
    day = date(start, 1, 1) + timedelta(days=rng.randrange((date(end, 12, 31) - date(start, 1, 1)).days))
    return rng.choice((day.strftime("%m/%d/%Y"), day.isoformat(), day.strftime("%B %-d, %Y"), day.strftime("%-m/%-d/%y")))


# --------------------------------------------------------------------------
# Rendering


class Builder:
    def __init__(self, lowercase: bool = False):
        self.text, self.spans, self.occurrences, self.lowercase = "", [], [], lowercase

    def add(self, literal: str):
        self.text += literal.lower() if self.lowercase else literal

    def value(self, slot: str, value: str, label: str | None, role: str, real: list[Real] = (), inner: str | None = None):
        surface = value.lower() if self.lowercase else value
        start = len(self.text)
        self.text += surface
        end = len(self.text)
        if label is not None:
            self.spans.append({"start": start, "end": end, "label": label, "text": surface})
        occurrence = {"slot": slot, "role": role, "label": label, "start": start, "end": end, "value": surface,
                      "real_values": [asdict(r) for r in real]}
        if inner is not None:
            offset = surface.index(inner.lower() if self.lowercase else inner)
            occurrence["reference"] = {"start": start + offset, "end": start + offset + len(inner)}
        self.occurrences.append(occurrence)


@dataclass
class Context:
    """Entities shared by the slots of one row (and reused by its matched negative)."""
    split: str
    rng: random.Random
    bank: Bank
    owner: dict
    form: str | None = None
    site: dict | None = None
    site_surface: str | None = None
    loc: dict | None = None
    people: dict = field(default_factory=dict)
    tags: set = field(default_factory=set)
    real: list = field(default_factory=list)


def _weighted(rng, options):
    pick, total = rng.random(), 0.
    for option, weight in options:
        total += weight
        if pick < total:
            return option
    return options[-1][0]


def _choose_city(ctx: Context, *, state_alone_ok: bool = False) -> dict:
    """About half well-known (2+ CMS hospitals) and half any Gazetteer place."""
    pool = ctx.bank.well_known[ctx.split] if ctx.rng.random() < .55 else ctx.bank.cities[ctx.split]
    for _ in range(1000):
        city = ctx.rng.choice(pool)
        if not state_alone_ok or STATES[city["state"]] in ctx.bank.states:
            return city
    raise RuntimeError("No suitable city")


def _locality(ctx: Context, name: str, state: str, kind: str, source: str) -> tuple[str, list[Real]]:
    """Render a locality expression; a state inside it stays inside the LOCATION span."""
    form = ctx.rng.random()
    partition = value_partition(name) if source != "curated" else "shared"
    real = [Real(name, kind, source, partition=partition)]
    bare_ok = kind == "county" or _norm(name) not in {_norm(s) for s in STATES.values()} | {_norm(c) for c in COUNTRIES}
    if not ctx.bank.state_ok[state] or (form < .45 and bare_ok):
        if not bare_ok:
            raise ValueError(f"{name!r} needs a state suffix but its state is not usable")
        return name, real
    real.append(Real(STATES[state], "state_in_locality", "fixed_list"))
    ctx.tags.add("state_inside_locality")
    if form < .85:
        return f"{name}, {state}", real
    return f"{name}, {STATES[state]}", real


def _site(ctx: Context, form: str, *, citable: bool = False) -> None:
    rng, bank = ctx.rng, ctx.bank
    curated = [e for e in bank.curated if e["forms"][form] and (not citable or e["cite_names"])]
    if form == "abbr" or citable:
        source = "curated"
    elif form == "short":
        source = "curated" if rng.random() < .55 else "cms"
    else:
        source = "curated" if rng.random() < .3 else "cms"
    if source == "curated":
        entry = rng.choice(curated)
        surface = rng.choice(entry["forms"][form])
        ctx.site = entry
        real = [Real(surface, f"care_site_{form}", "curated", expansion=entry["expansion"])]
    else:
        pool = bank.cms_full[ctx.split] if form == "full" else bank.cms_short[ctx.split]
        if form == "full" and rng.random() < .25:
            pool = [site for site in pool if LEADING_ABBREVIATION.match(site["name"])]
        site = rng.choice(pool)
        surface, ctx.site = site["name"], site
        real = [Real(surface, f"care_site_{form}", "cms", partition=site["partition"])]
        if form == "short":
            real.append(Real(site["full_name"], "care_site_full", "cms", partition=site["partition"]))
    ctx.form, ctx.site_surface = form, surface
    ctx.real.extend(real)
    ctx.tags.add(FORM_TAGS[form])
    if LEADING_ABBREVIATION.match(surface):
        ctx.tags.add("leading_abbreviation")
    if _norm(surface) in bank.place_names:
        ctx.tags.add("place_name_collision")


def _person(ctx: Context, key: str, **kwargs) -> dict:
    if key not in ctx.people:
        ctx.people[key] = identity(ctx.rng, ctx.split, bank=ctx.bank, **kwargs)
    return ctx.people[key]


def _render(template: str, ctx: Context, lowercase: bool, negative_hint: dict | None = None) -> Builder:
    rng, bank = ctx.rng, ctx.bank
    hint = negative_hint or {}
    builder = Builder(lowercase)
    cache = {}
    for literal, slot, spec, conversion in Formatter().parse(template):
        builder.add(literal)
        if slot is None:
            continue
        assert not spec and not conversion, template
        label = LABELLED_SLOTS.get(slot)
        if slot == "loc":
            if "loc" not in cache:
                if ctx.loc is None:
                    if rng.random() < .1 and bank.counties[ctx.split]:
                        county = rng.choice(bank.counties[ctx.split])
                        ctx.loc = {"name": county["name"], "state": county["state"], "kind": "county"}
                        ctx.tags.add("county_locality")
                    else:
                        city = _choose_city(ctx)
                        ctx.loc = {"name": city["name"], "state": city["state"], "kind": "city"}
                cache["loc"] = _locality(ctx, ctx.loc["name"], ctx.loc["state"], ctx.loc["kind"], "census")
            builder.value(slot, cache["loc"][0], label, "patient_locality", cache["loc"][1])
        elif slot == "loc_state":
            name = STATES[ctx.loc["state"]]
            builder.value(slot, name, None, "state_alone", [Real(name, "state", "fixed_list")])
            ctx.tags.add("n4_state_or_country")
        elif slot == "site":
            builder.value(slot, ctx.site_surface, label, "care_site", ctx.real)
        elif slot == "site_loc":
            if "site_loc" not in cache:
                cache["site_loc"] = _locality(ctx, ctx.site["city"], ctx.site["state"], "city",
                                              "curated" if ctx.site["source"] == "curated" else "census")
            builder.value(slot, cache["site_loc"][0], label, "care_site_locality", cache["site_loc"][1])
        elif slot == "cite":
            entry = hint.get("cite_entry") or (ctx.site if ctx.site and ctx.site.get("cite_names") else None)
            if entry is None:
                entry = rng.choice([e for e in bank.curated if e["cite_names"]])
            if "cite" not in cache:
                same = ctx.site_surface if ctx.site_surface in entry["cite_names"] else None
                cache["cite"] = same if same and rng.random() < .65 else rng.choice(entry["cite_names"])
            name = cache["cite"]
            builder.value(slot, name, None, "institution_as_source", [Real(name, "institution_citation", "curated", expansion=entry["expansion"])])
            ctx.tags.add("n3_institution_source")
        elif slot in {"clin", "clin_last"}:
            person = _person(ctx, "clin")
            builder.value(slot, person["full"] if slot == "clin" else person["surname"], label, "clinician")
            ctx.tags.add("name_boundaries")
        elif slot == "pt":
            builder.value(slot, ctx.owner["full"], label, "patient")
            ctx.tags.add("name_boundaries")
        elif slot in {"person", "person_first"}:
            if "person" not in ctx.people:
                if slot == "person_first" or rng.random() < .6:
                    _person(ctx, "person", given=rng.choice(bank.given))
                else:
                    _person(ctx, "person", surname=rng.choice(bank.surnames))
            person = ctx.people["person"]
            place_word = person["given"] if person["given"] in bank.given else person["surname"]
            builder.value(slot, person["full"] if slot == "person" else person["given"], label, "person_named_like_place",
                          [Real(place_word, "place_word_as_name", "fixed_list")])
            ctx.tags.update({"n5_place_word_other_role", "name_boundaries"})
        elif slot == "date":
            builder.value(slot, _date(rng), label, "patient_date")
            ctx.tags.add("patient_date")
        elif slot == "year":
            builder.value(slot, str(rng.randrange(2008, 2025)), label, "patient_year")
            ctx.tags.add("patient_date")
        elif slot == "mrn":
            builder.value(slot, f"PL-{rng.randrange(10**7, 10**8)}", label, "invented_mrn")
            ctx.tags.add("reserved_identifier")
        elif slot == "phone":
            builder.value(slot, f"({rng.choice((206, 312, 415, 512, 617, 704, 801, 919))}) 555-01{rng.randrange(100):02d}", label, "fictional_phone")
            ctx.tags.add("reserved_identifier")
        elif slot == "finding":
            key, surface, phrases = hint.get("finding") or rng.choice(bank.findings)
            phrase = rng.choice(phrases).replace("{n}", str(rng.randrange(1, 9)))
            builder.value(slot, phrase, None, "named_instrument", [Real(key, "instrument_place", "fixed_list")],
                          inner=surface if surface in phrase else key)
            ctx.tags.add("n1_named_instrument")
        elif slot in {"device", "cohort_study"}:
            key, surface = hint.get(slot) or rng.choice(bank.devices if slot == "device" else bank.studies)
            builder.value(slot, surface, None, "named_" + slot, [Real(key, "instrument_place", "fixed_list")])
            ctx.tags.update({"n1_named_instrument", "n1_named_device" if slot == "device" else "n1_named_study"})
        elif slot == "dz":
            key, surface = hint.get("dz") or rng.choice(bank.diseases)
            builder.value(slot, surface, None, "place_named_disease", [Real(key, "disease_place", "fixed_list")])
            ctx.tags.add("n2_place_disease")
        elif slot in {"state", "country", "region"}:
            if slot == "state":
                name = hint.get("state") or rng.choice(bank.states)
            else:
                name = rng.choice(bank.countries if slot == "country" else bank.regions)
            builder.value(slot, name, None, slot + "_alone", [Real(name.removeprefix("the "), slot, "fixed_list")])
            ctx.tags.add("n4_state_or_country")
        elif slot == "word":
            key, clauses = hint.get("word") or rng.choice(bank.words)
            clause = rng.choice(clauses).replace("{n}", str(rng.randrange(2, 9)))
            builder.value(slot, clause, None, "place_word_as_ordinary_word", [Real(key, "place_word", "fixed_list")])
            ctx.tags.add("n5_place_word_other_role")
        elif slot == "age":
            builder.add(str(rng.randrange(19, 89)))
        elif slot == "drug":
            builder.add(rng.choice(DRUGS))
        elif slot == "symptom":
            builder.add(rng.choice(SYMPTOMS))
        elif slot == "n":
            builder.add(str(rng.randrange(2, 9)))
        elif slot == "days":
            builder.add(str(rng.randrange(2, 15)))
        else:
            raise KeyError(f"Unknown slot {slot!r} in {template!r}")
    return builder


def template_key(split: str, group: str, layout: str, index: int) -> str:
    return f"places_{split}.{group}.{layout}.t{index:02d}"


def _pick_template(rng, split, group, layout):
    options = TEMPLATES[split][group][layout]
    index = rng.randrange(len(options))
    return template_key(split, group, layout, index), options[index]


def _negative_hint(ctx: Context, family: str) -> dict:
    """Prefer an N-role use of the positive's own real name when one exists."""
    bank, names = ctx.bank, set()
    states = set()
    if ctx.loc:
        names.add(ctx.loc["name"])
        states.add(STATES[ctx.loc["state"]])
    if ctx.site:
        names |= {ctx.site.get("city") or "", ctx.site_surface or ""}
        states.add(STATES.get(ctx.site.get("state"), ""))
    hint = {}
    findings = [item for item in bank.findings if item[0] in names | states]
    if findings:
        hint["finding"] = ctx.rng.choice(findings)
    for slot, items in (("device", bank.devices), ("cohort_study", bank.studies), ("dz", bank.diseases)):
        matches = [item for item in items if item[0] in names | states]
        if matches:
            hint[slot] = ctx.rng.choice(matches)
    words = [item for item in bank.words if item[0] in names]
    if words:
        hint["word"] = words[0]
    usable = [state for state in states if state in bank.states]
    if usable:
        hint["state"] = sorted(usable)[0]
    if ctx.site and ctx.site.get("cite_names"):
        hint["cite_entry"] = ctx.site
    return hint


def _shared_value(positive: dict, negative: dict) -> bool:
    def keys(row):
        return {_norm(r[name]) for o in row["occurrences"] for r in o["real_values"]
                for name in ("key", "expansion") if r[name]}
    return bool(keys(positive) & keys(negative))


def boundary_errors(text: str, spans: list[dict]) -> list[str]:
    """Addendum span-boundary invariants (exercised against PLACE_FIXTURES too)."""
    errors = []
    states = {_norm(name) for name in STATES.values()} | {_norm(name) for name in COUNTRIES}
    for span in spans:
        start, end, label = span["start"], span["end"], span["label"]
        surface = text[start:end]
        if label in {"LOCATION", "FACILITY"}:
            if re.match(r"(?i)(?:in|at|from|to|of|the|near)\s", surface):
                errors.append(f"{label} {surface!r} starts with a preposition or article")
            if re.search(r"(?:St|Ste|Mt|Ft)\.\s$", text[max(0, start - 5):start]):
                errors.append(f"{label} {surface!r} leaves a leading St./Mt./Ft. outside the span")
            if surface[-1] in ".,;:!?" and not re.search(r"\b(?:St|Ste|Mt|Ft|Jr|Inc)\.$", surface):
                errors.append(f"{label} {surface!r} ends with punctuation")
            if _norm(surface) in states:
                errors.append(f"{label} {surface!r} is a state or country alone")
        if label == "FACILITY" and re.search(r"\s(?:in|at)\s+[A-Z]", surface):
            errors.append(f"FACILITY {surface!r} contains an adjacent locality")
        if label == "FACILITY" and re.search(r",\s*[A-Z]{2}$", surface):
            errors.append(f"FACILITY {surface!r} absorbs a state")
        if label == "NAME":
            if re.match(r"(?i)(?:dr|mr|mrs|ms|prof)\.?\s", surface) or re.search(r",?\s(?:MD|DO|NP|RN|PA)$", surface):
                errors.append(f"NAME {surface!r} includes a title or credential")
    return errors


def _finish_row(builder: Builder, ctx: Context, *, number: int, seed: int, family: str, layout: str,
                pair_id: str, role: str, phenomenon: str) -> dict:
    spans = builder.spans
    people = [ctx.owner] + [ctx.people[key] for key in sorted(ctx.people)]
    groups = list(dict.fromkeys(person["group"] for person in people))
    tags = set(ctx.tags) | {PHENOMENON_TAGS[phenomenon]}
    if builder.lowercase:
        tags.add("lowercase_text")
    row = {
        "id": f"places-v{VERSION}-s{seed}-{ctx.split}-{number:05d}",
        "text": builder.text,
        "spans": spans,
        "split": ctx.split,
        "suite": SUITES[ctx.split],
        "annotation_policy": POLICY_VERSION,
        "annotation_addendum": ADDENDUM,
        "identity_group": ctx.owner["group"],
        "identity_groups": groups,
        "full_identity": ctx.owner["full"],
        "full_identities": list(dict.fromkeys(person["full"] for person in people)),
        "pair_id": pair_id,
        "template_family": family,
        "family": family,
        "cohort": "realistic",
        "layout": layout,
        "role": role,
        "positive_phenomenon": phenomenon if role != "negative" else None,
        "negative_family": phenomenon if role == "negative" else None,
        "phenomena": sorted(tags),
        "negative": not spans,
        "place_negative": not any(s["label"] in {"LOCATION", "FACILITY", "ADDRESS"} for s in spans),
        "occurrences": builder.occurrences,
        "dataset_status": "TRAIN_ONLY" if ctx.split == "train" else "DEVELOPMENT_ONLY",
    }
    return row


def validate_row(row: dict, bank: Bank | None = None) -> None:
    text, spans = row["text"], row["spans"]
    problems = annotation_errors(text, spans) + boundary_errors(text, spans)
    if problems:
        raise ValueError(f"{row['id']}: {problems}")
    if row["annotation_policy"] != POLICY_VERSION or row["annotation_addendum"] != ADDENDUM:
        raise ValueError("Unexpected policy")
    if any(identity_split(group) != row["split"] for group in row["identity_groups"]):
        raise ValueError("Identity partition mismatch")
    if {canonical_identity_group(name) for name in row["full_identities"]} != set(row["identity_groups"]):
        raise ValueError("Identity groups do not match identities")
    gold = {(s["start"], s["end"], s["label"]) for s in spans}
    for o in row["occurrences"]:
        if text[o["start"]:o["end"]] != o["value"]:
            raise ValueError("Occurrence offsets drifted")
        if o["label"] is not None and (o["start"], o["end"], o["label"]) not in gold:
            raise ValueError("Labelled occurrence lacks its gold span")
        if o["label"] is None and any(s["start"] < o["end"] and o["start"] < s["end"] for s in spans):
            raise ValueError("Public reference overlaps a gold span")
        for real in o["real_values"]:
            if not train_pool(real["key"], expansion=real["expansion"]):
                raise ValueError(f"Heldout-pool value emitted: {real['key']!r}")
            if real["partition"] not in {"shared", row["split"]}:
                raise ValueError(f"Value partition {real['partition']} used in {row['split']}")
    if bank is not None and bank.heldout_pattern.search(text):
        raise ValueError(f"Heldout-pool name in text: {bank.heldout_pattern.search(text)[0]!r}")
    if excluded_sequences(text) or any(excluded_value(value) for value in row_values(row)):
        raise ExcludedNameError(f"{row['id']}: text contains a name on the blind-bank exclusion list")


def row_values(row: dict) -> set:
    """Every real or public-reference value a row inserted (keys, expansions and rendered surfaces)."""
    values = set()
    for o in row["occurrences"]:
        for real in o["real_values"]:
            values.add(real["key"])
            if real["expansion"]:
                values.add(real["expansion"])
        if o["real_values"] or o["label"] in {"LOCATION", "FACILITY"}:
            values.add(o["value"])
        if "reference" in o:
            values.add(row["text"][o["reference"]["start"]:o["reference"]["end"]])
    return {value.strip() for value in values if value.strip()}


def contamination(rows: list[dict]) -> dict:
    """Value-level and 1-8 word sequence matches against the exclusion list."""
    hashes = exclusion_hashes()
    used = {unicodedata.normalize("NFC", value).casefold() for row in rows for value in row_values(row)}
    return {"rows": len(rows), "distinct_real_values": len(used),
            "value_level_matches": sum(exclusion_hash(value) in hashes for value in used),
            "rows_with_ngram_match": sum(excluded_sequences(row["text"]) > 0 for row in rows)}


def _make(template_info, ctx, *, number, seed, layout, pair_id, role, phenomenon, used, lowercase, hint=None):
    family, template = template_info
    builder = _render(template, ctx, lowercase, hint)
    row = _finish_row(builder, ctx, number=number, seed=seed, family=family, layout=layout,
                      pair_id=pair_id, role=role, phenomenon=phenomenon)
    try:
        validate_row(row, ctx.bank)
    except ExcludedNameError:
        return None
    if row["text"] in used:
        return None
    return row


def _contrast_city(ctx: Context, planned: str) -> dict | None:
    """A city whose own name also has the planned N-role (e.g. Rochester / Rochester criteria)."""
    bank = ctx.bank
    if planned == "n1":
        keys = {item[0] for item in bank.findings + bank.devices + bank.studies}
    elif planned == "n2":
        keys = {item[0] for item in bank.diseases}
    elif planned == "n5":
        keys = {item[0] for item in bank.words}
    else:
        return None
    options = [city for city in bank.cities[ctx.split] if city["name"] in keys and (city["bare_ok"] or bank.state_ok[city["state"]])]
    return ctx.rng.choice(options) if options else None


def _positive(rng, split, owner, phenomenon, layout, bank, planned=None):
    ctx = Context(split, rng, bank, owner)
    # Steer toward a value that the planned matched negative can reuse in a public role.
    contrast = planned is not None and rng.random() < .6
    if phenomenon in {"p2", "p3", "p4"}:
        _site(ctx, {"p2": "abbr", "p3": "short", "p4": "full"}[phenomenon], citable=contrast and planned == "n3")
    elif phenomenon == "p5":
        while True:
            _site(ctx, _weighted(rng, (("full", .5), ("short", .25), ("abbr", .25))))
            city = ctx.site.get("city")
            if city and (ctx.site["source"] == "curated" or value_partition(city) == split):
                break
            ctx.tags.clear()
            ctx.real.clear()
    elif phenomenon == "p6":
        _site(ctx, _weighted(rng, (("abbr", .35), ("short", .35), ("full", .3))), citable=contrast and planned == "n3")
    elif phenomenon == "mixed":
        _site(ctx, _weighted(rng, (("abbr", .35), ("short", .4), ("full", .25))), citable=True)
    info = _pick_template(rng, split, POSITIVE_GROUPS[phenomenon], layout)
    if phenomenon == "p1" and "{loc_state}" in info[1]:
        # The state named alone must be a usable N4 value (train pool, unambiguous).
        city = _choose_city(ctx, state_alone_ok=True)
        ctx.loc = {"name": city["name"], "state": city["state"], "kind": "city"}
    elif phenomenon == "p1" and contrast:
        city = _contrast_city(ctx, planned)
        if city is not None:
            ctx.loc = {"name": city["name"], "state": city["state"], "kind": "city"}
    lowercase = layout == "query" and rng.random() < .35
    return ctx, info, lowercase


def generate(*, seed: int = SEED, train_count: int = TRAIN_COUNT, validation_count: int = VALIDATION_COUNT) -> dict:
    """Return ``{"train": rows, "validation": rows}`` deterministically."""
    if train_count < 100 or validation_count < 100:
        raise ValueError("Require at least 100 rows per split for layout/phenomenon coverage")
    bank = value_bank()
    targets = {"train": train_count, "validation": validation_count}
    rows = {split: [] for split in SPLITS}
    units = Counter()
    used = set()
    index = 0
    while any(len(rows[split]) < targets[split] for split in SPLITS):
        rng = random.Random(f"places:{seed}:{index}")
        index += 1
        owner = identity(rng, bank=bank)
        split = identity_split(owner["group"])
        if len(rows[split]) >= targets[split]:
            continue
        k = units[split]
        units[split] += 1
        cycle = k // len(SCHEDULE)
        phenomenon = SCHEDULE[k % len(SCHEDULE)]
        layout = LAYOUTS[cycle % len(LAYOUTS)]
        pair_id = f"places-v{VERSION}-s{seed}-{split}-unit-{k:05d}"
        families = MATCHED_NEGATIVES[phenomenon]
        family = families[(cycle + cycle // len(LAYOUTS)) % len(families)]
        paired = k % 4 != 3
        planned = family if paired else None
        positive = None
        for _ in range(50):
            ctx, info, lowercase = _positive(rng, split, owner, phenomenon, layout, bank, planned)
            positive = _make(info, ctx, number=len(rows[split]), seed=seed, layout=layout, pair_id=pair_id,
                             role="mixed" if phenomenon == "mixed" else "positive", phenomenon=phenomenon,
                             used=used, lowercase=lowercase)
            if positive is not None:
                break
        if positive is None:
            raise RuntimeError("Could not author a unique positive row")
        rows[split].append(positive)
        used.add(positive["text"])
        if not paired or len(rows[split]) >= targets[split]:
            continue
        negative = None
        for attempt in range(80):
            group = _weighted(rng, NEGATIVE_GROUPS[family])
            nctx = Context(split, rng, bank, owner, form=ctx.form, site=ctx.site, site_surface=ctx.site_surface, loc=ctx.loc)
            nctx.people = {}
            hint = _negative_hint(nctx, family) if attempt < 10 else {}
            # A negative never reuses the positive's care-site or locality role.
            nctx.site = nctx.site if family == "n3" else None
            nctx.site_surface = None
            negative = _make(_pick_template(rng, split, group, layout), nctx, number=len(rows[split]), seed=seed,
                             layout=layout, pair_id=pair_id, role="negative", phenomenon=family, used=used,
                             lowercase=layout == "query" and rng.random() < .35, hint=hint)
            if negative is not None:
                break
        if negative is None:
            raise RuntimeError(f"Could not author a unique matched negative: {split} {family} {layout} {phenomenon} k={k}")
        negative["matched_positive"] = positive["id"]
        negative["matched_value_shared"] = _shared_value(positive, negative)
        rows[split].append(negative)
        used.add(negative["text"])
    for split in SPLITS:
        if len({row["id"] for row in rows[split]}) != len(rows[split]):
            raise RuntimeError("Duplicate row identifiers")
    return rows


# --------------------------------------------------------------------------
# Reporting and packaging


def support(rows: list[dict]) -> dict:
    """Per-phenomenon, per-negative-family and per-layout support for one split."""
    labels = Counter(span["label"] for row in rows for span in row["spans"])
    by_role = Counter(row["role"] for row in rows)
    return {
        "rows": len(rows),
        "roles": dict(sorted(by_role.items())),
        "place_negative_rows": sum(row["place_negative"] for row in rows),
        "place_negative_fraction": round(sum(row["place_negative"] for row in rows) / len(rows), 4),
        "phi_free_rows": sum(row["negative"] for row in rows),
        "positive_phenomenon": dict(sorted(Counter(row["positive_phenomenon"] for row in rows if row["positive_phenomenon"]).items())),
        "negative_family": dict(sorted(Counter(row["negative_family"] for row in rows if row["negative_family"]).items())),
        "layout": dict(sorted(Counter(row["layout"] for row in rows).items())),
        "layout_by_primary": {key: dict(sorted(value.items())) for key, value in sorted(_cross(rows).items())},
        "phenomena_tags": dict(sorted(Counter(tag for row in rows for tag in row["phenomena"]).items())),
        "label_support": {label: labels[label] for label in ENTITY_TYPES if labels[label]},
        "matched_negatives_sharing_positive_value": sum(bool(row.get("matched_value_shared")) for row in rows),
        "template_families": len({row["template_family"] for row in rows}),
    }


def _cross(rows):
    table = defaultdict(Counter)
    for row in rows:
        table[row["positive_phenomenon"] or row["negative_family"]][row["layout"]] += 1
    return table


def _payload(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def _json(data) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


SOURCE_FILES = ("data/generators/places.py", "data/generators/places_contexts.py", "data/generators/place_sources.py",
                "data/annotation_policy.py", "data/annotation_policy_places.py", "data/generators/remediation.py", "types.py")
LIMITATIONS = [
    "Finite authored templates from one author; validation families are disjoint in wording but not independently authored, so validation is development evidence, not an independent bank.",
    "Real place and care-site names are surface values only; no row reproduces a real patient record, and invented people, dates and identifiers are unrelated to the named places.",
    "Census Gazetteer places are sampled about half from cities with at least two CMS hospitals and half uniformly; this proxy for well-known places is not a prevalence estimate.",
    "CMS facility names are mechanically title-cased and shortened; a few surface forms may differ from local usage.",
    "Curated institutions and public-reference lists (instruments, diseases, states, countries, place words) are shared across splits; Census and CMS values are hash-partitioned so validation names are unseen in train.",
    "LOCATION/FACILITY/NAME/DATE are the only gold labels exercised besides a small MRN/PHONE retention; other labels have no support.",
    "Device names containing a place (for example a cervical collar or brace) are treated as N1 named instruments; the addendum lists instruments, criteria, scores, rules and trials.",
]


def build_artifacts(*, seed: int = SEED, train_count: int = TRAIN_COUNT, validation_count: int = VALIDATION_COUNT) -> dict:
    """Return ``{relative path: bytes}`` for corpora, manifests, lineage and declaration."""
    rows = generate(seed=seed, train_count=train_count, validation_count=validation_count)
    from outis.data.generators.remediation import get_encoding
    encoding = get_encoding()
    config = {"seed": seed, "train_count": train_count, "validation_count": validation_count}
    source_hashes = {name: _sha((ROOT / "src/outis" / name).read_bytes()) for name in SOURCE_FILES}
    pins = place_sources.pins()
    bank = value_bank()
    exclusion = {
        "path": str(EXCLUSIONS.relative_to(ROOT)), "sha256": EXCLUSIONS_SHA256, "hashes": len(exclusion_hashes()),
        "normalisation": "sha256(unicodedata.normalize('NFC', value.strip()).casefold().encode('utf-8'))",
        "filter": "values whose hash (or any of whose 1-8 word sequences' hash) is listed are never used; generated rows containing a listed 1-8 word sequence (punctuation-trimmed, also without a trailing 's) are rejected and resampled",
        "contamination_before_filter": CONTAMINATION_BEFORE_FILTER,
        "contamination_after_filter": {split: contamination(rows[split]) for split in SPLITS},
    }
    if any(exclusion["contamination_after_filter"][split][key] for split in SPLITS
           for key in ("value_level_matches", "rows_with_ngram_match")):
        raise RuntimeError("Generated rows still match the blind-bank exclusion list")
    artifacts, corpora = {}, {}
    for split in SPLITS:
        payload = _payload(rows[split])
        corpora[split] = payload
        artifacts[f"data/places_{split}.jsonl"] = payload
    lineage_hashes = {}
    for split in SPLITS:
        lineage = {
            "schema_version": 1, "generator": "outis.data.generators.places", "generator_version": VERSION,
            "corpus": f"data/places_{split}.jsonl", "corpus_sha256": _sha(corpora[split]), "config": config,
            "source_hashes": source_hashes, "value_sources": pins, "blind_bank_exclusions": exclusion,
            "clinical_text_or_annotations_modified": False, "derived_from_existing_corpus": False,
            "rows": [{"line_1based": number, "id": row["id"], "row_sha256": _sha(line),
                      "text_sha256": _sha(row["text"].encode("utf-8")),
                      "annotations_sha256": _sha(json.dumps(row["spans"], sort_keys=True).encode("utf-8")),
                      "template_family": row["template_family"], "pair_id": row["pair_id"]}
                     for number, (row, line) in enumerate(zip(rows[split], corpora[split].splitlines(keepends=True)), 1)],
        }
        raw = _json(lineage)
        lineage_hashes[split] = _sha(raw)
        artifacts[f"data/places_{split}.lineage.json"] = raw
    declaration = {
        "schema_version": 1, "annotation_policy": POLICY_VERSION, "annotation_addendum": ADDENDUM,
        "generator": "outis.data.generators.places", "generator_version": VERSION, "config": config,
        "corpora": {split: {"path": f"data/places_{split}.jsonl", "sha256": _sha(corpora[split]),
                            "lineage": f"data/places_{split}.lineage.json", "lineage_sha256": lineage_hashes[split]}
                    for split in SPLITS},
        "source_hashes": source_hashes,
        "prose_author": {
            "agent": "places_addendum_generator_author",
            "inputs_read": ["docs/annotation-policy-v2-places.md", "docs/annotation-policy-v2.md",
                            "src/outis/annotation_policy*.py", "existing generator conventions (code style, metadata, manifests)",
                            "data/places_exclusions.sha256.json (hashes only)"],
            "existing_train_or_dev_rows_read": False, "model_predictions_read": False,
            "experiments_or_external_benchmarks_read": False, "blind_bank_rows_or_names_read": False,
            "blind_bank_access": "Only the published hash-only exclusion list, read on coordinator instruction after verifying its SHA-256; matching names were counted, never printed or inspected.",
            "detector_run": False,
        },
        "real_value_sources": pins,
        "blind_bank_exclusions": exclusion,
        "value_pool_rule": "place_value_pool: int(sha256(NFC(casefold(value)))[:8],16)%10<8 is train; every emitted real value, every embedded instrument/disease/place-word key and every abbreviation/short-name expansion must be train; well-known heldout names are additionally refused anywhere in text",
        "commercially_licensed_data_used": False, "external_datasets": [], "synthetic_only": True,
        "independent_evaluation_claim": False, "training_performed": False,
        "license": "CC0-1.0", "code_license": "Apache-2.0", "limitations": LIMITATIONS,
    }
    declaration_raw = _json(declaration)
    artifacts["data/places.author_declaration.json"] = declaration_raw
    for split in SPLITS:
        split_rows = rows[split]
        lengths = [len(encoding.encode(row["text"], disallowed_special=())) for row in split_rows]
        manifest = {
            "schema_version": 1, "generator": "outis.data.generators.places", "generator_version": VERSION,
            "annotation_policy": POLICY_VERSION, "annotation_addendum": ADDENDUM, "addenda": [ADDENDUM],
            "sha256": _sha(corpora[split]), "rows": len(split_rows), "split": split, "suite": SUITES[split],
            "purpose": split, "used_for_training": split == "train", "used_for_checkpoint_selection": split == "validation",
            "independent_final": False, "training_performed": False, "config": config,
            "source_hashes": source_hashes, "value_sources": pins, "blind_bank_exclusions": exclusion,
            "value_pools": {"rule": declaration["value_pool_rule"], "pool_used": "train", "heldout_emitted": 0,
                            "split_value_partition": "Census/CMS values: sha256('clinical-phi-v2-places/value-split:'+NFC(casefold(value)))[:8] % 5 == 0 -> validation; curated and public-reference lists shared",
                            "excluded_values": dict(sorted(bank.excluded.items()))},
            "identity_partition": "sha256(NFC(full identity).casefold()); first eight hex digits modulo ten below eight is train; every listed person on a row shares that partition",
            "synthetic_identifier_schemes": {"PHONE": PHONE_SCHEME, "MRN": MRN_SCHEME,
                                             "NAME": "Invented combinations from remediation_synthetic.NAME_POOLS plus place-word given names/surnames for N5",
                                             "DATE": "Invented 2008-2025 dates and years"},
            "author_declaration": "data/places.author_declaration.json", "author_declaration_sha256": _sha(declaration_raw),
            "lineage": f"data/places_{split}.lineage.json", "lineage_sha256": lineage_hashes[split],
            "support": support(split_rows),
            "cohort_support": dict(Counter(row["cohort"] for row in split_rows)),
            # Project-wide corpus fields read by load_verified_corpus and the
            # training-mixture checks: every entity type, zeros included.
            "documents": len(split_rows),
            "label_support": {label: sum(span["label"] == label for row in split_rows for span in row["spans"])
                              for label in ENTITY_TYPES},
            "spans": sum(len(row["spans"]) for row in split_rows),
            "phi_free_rows": sum(not row["spans"] for row in split_rows),
            "cohort_counts": dict(sorted(Counter(row["cohort"] for row in split_rows).items())),
            "tokenizer": "o200k_base",
            "tokens": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths), "total": sum(lengths)},
            "license": "CC0-1.0", "code_license": "Apache-2.0", "external_datasets": [],
            "authoring_note": "Original authored templates; real public place and care-site names from pinned US federal public-domain files and a curated list (see value_sources); no existing corpus, benchmark or blind bank read.",
            "dataset_status": "TRAIN_ONLY" if split == "train" else "DEVELOPMENT_ONLY",
            "limitations": LIMITATIONS,
        }
        artifacts[f"data/places_{split}.jsonl.manifest.json"] = _json(manifest)
    return artifacts


def write_artifacts(artifacts: dict, root: Path = ROOT) -> dict:
    """Create missing artifacts; identical files are left alone and differing files are refused."""
    conflicts = [path for path, raw in artifacts.items() if (root / path).exists() and (root / path).read_bytes() != raw]
    if conflicts:
        raise FileExistsError(f"Artifacts are immutable and differ from this build; remove or rename: {conflicts}")
    written = []
    for path, raw in artifacts.items():
        target = root / path
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(raw)
        written.append(path)
    return {"written": written, "unchanged": sorted(set(artifacts) - set(written))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("generate", "check", "report"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-count", type=int, default=TRAIN_COUNT)
    parser.add_argument("--validation-count", type=int, default=VALIDATION_COUNT)
    args = parser.parse_args(argv)
    config = {"seed": args.seed, "train_count": args.train_count, "validation_count": args.validation_count}
    if args.command == "report":
        rows = generate(**config)
        print(json.dumps({split: support(rows[split]) for split in SPLITS}, indent=2))
        return 0
    artifacts = build_artifacts(**config)
    if args.command == "check":
        mismatched = [path for path, raw in artifacts.items() if not (ROOT / path).exists() or (ROOT / path).read_bytes() != raw]
        if mismatched:
            raise SystemExit(f"Reproduction mismatch: {mismatched}")
        result = {"reproduced": sorted(artifacts)}
    else:
        result = write_artifacts(artifacts)
    result["corpora"] = {path: {"sha256": _sha(raw), "rows": raw.count(b"\n")}
                         for path, raw in artifacts.items() if path.endswith(".jsonl")}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
