"""Deterministic, byte-preserving V13 replay from pinned V12 TRAIN only.

Selection preserves complete pair units and all canonical TRAIN owner groups.
The 12 retained long rows are original repeated-course narrative stress.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/v12_train.jsonl"
SOURCE_MANIFEST = ROOT / "data/v12_train.jsonl.manifest.json"
SOURCE_SHA = "b6487cbefd57e912680002e67c86526b7e4b3f4dfa97e897917107d2e54d64e0"
SEED = "v13-focused-replay-prototype-20261003-v1"
SUPERSEDED_REPLAY_SHA = "32334d7cd5d69187d3b7c381c94a09c64d07d3184dd1186add456e80dc84d867"
BASE_REPLAY_SHA = "211a663ab7a3c09fdce8b05ded7c5e5a3e6eb45ab5e24312bba8a7abad6cfade"
LONG_FORM_QUOTAS = {"iso_full": 2, "slash_full": 2, "day_named_month_year": 2,
                    "named_month_day_year": 2, "year_only": 4}
MARCH_MONTH_YEAR_CAPS = {"context_pair": 8, "phi_free_singleton": 0,
                       "realistic_positive_singleton": 50, "unusual_name_singleton": 2}
LABELS = sorted("ACCOUNT ADDRESS AGE DATE EMAIL FACILITY ID IP LOCATION MRN NAME PHONE SSN URL".split())
MONTHS = {name.casefold(): i for i, name in enumerate(
    "January February March April May June July August September October November December".split(), 1)}
MONTHS.update({name[:3]: i for name, i in list(MONTHS.items())})
BUCKET_QUOTAS = {"context_pair": 300, "phi_free_singleton": 375,
                 "realistic_positive_singleton": 1950, "unusual_name_singleton": 75}


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def rank(value: str) -> str:
    return sha((SEED + "\0" + value).encode("utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def identity_hash(value: str) -> str:
    return sha(unicodedata.normalize("NFC", value).casefold().encode("utf-8"))


@dataclass
class Unit:
    pair_id: str
    indices: list[int]
    bucket: str
    family: str
    layout: str


def load_source() -> tuple[bytes, list[bytes], list[dict], dict, bytes]:
    # Hash the whole raw source before parsing or accessing any corpus row.
    raw = SOURCE.read_bytes()
    require(sha(raw) == SOURCE_SHA, "V12 TRAIN bytes differ from the authorized pinned SHA")
    manifest_raw = SOURCE_MANIFEST.read_bytes()
    manifest = json.loads(manifest_raw)
    require(manifest["sha256"] == SOURCE_SHA, "Source manifest SHA does not match pinned corpus")
    require(manifest["split"] == "train" and manifest["suite"] == "training", "Wrong source split")
    lines = raw.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    require(len(rows) == manifest["rows"], "Source row count differs from manifest")
    seen_ids = set()
    for row in rows:
        require(row["id"] not in seen_ids, "Duplicate source row id")
        seen_ids.add(row["id"])
        require(row["split"] == "train" and row["suite"] == "training", "Non-TRAIN source row")
        expected = {identity_hash(name) for name in row["full_identities"]}
        require(expected == set(row["identity_groups"]), "Owner groups are not canonical")
        require(identity_hash(row["full_identity"]) == row["identity_group"], "Primary owner is not canonical")
        require(row["full_identity"] in row["full_identities"], "Primary owner missing from owner list")
        require(all(int(group[:8], 16) % 10 < 8 for group in expected), "Owner outside TRAIN partition")
        require(row["negative"] == (not row["spans"]), "PHI-free flag disagrees with spans")
        require(all(s["label"] in LABELS and 0 <= s["start"] < s["end"] <= len(row["text"])
                    for s in row["spans"]), "Invalid source span")
    return raw, lines, rows, manifest, manifest_raw


def make_units(rows: list[dict]) -> list[Unit]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        require(bool(row["pair_id"]), "Missing pair unit identifier")
        grouped[row["pair_id"]].append(i)
    units = []
    for pair_id, indices in grouped.items():
        members = [rows[i] for i in indices]
        families = {row["family"] for row in members}
        layouts = {row["layout"] for row in members}
        require(len(families) == len(layouts) == 1, "Pair crosses family/layout strata")
        if any(row["cohort"] == "long_document" or row["diagnostic_slice"] == "repeated_course_narrative"
               for row in members):
            bucket = "excluded_repeated_course_long"
        elif len(members) == 2:
            require(sum(not row["spans"] for row in members) == 1, "Unexpected context-pair composition")
            require(all(row["cohort"] == "realistic" for row in members), "Unexpected context-pair cohort")
            require(len({row["identity_group"] for row in members}) == 1, "Context-pair primary owner differs")
            bucket = "context_pair"
        else:
            require(len(members) == 1, "Unsupported pair-unit size")
            row = members[0]
            bucket = ("phi_free_singleton" if not row["spans"] else
                      "unusual_name_singleton" if row["cohort"] == "unusual_name" else
                      "realistic_positive_singleton")
            require(row["cohort"] in {"unusual_name", "realistic"}, "Unexpected source cohort")
        units.append(Unit(pair_id, indices, bucket, next(iter(families)), next(iter(layouts))))
    return units


def allocate(capacities: dict[str, int], quota: int, context: str) -> dict[str, int]:
    """Capacity-safe largest-remainder quotas, reserving one/stratum if possible."""
    require(0 <= quota <= sum(capacities.values()), "Quota exceeds source capacity")
    result = {key: int(quota >= len(capacities)) for key in capacities}
    remaining = quota - sum(result.values())
    available = {key: capacity - result[key] for key, capacity in capacities.items()}
    total = sum(available.values())
    if remaining and total:
        floors = {key: remaining * capacity // total for key, capacity in available.items()}
        for key, value in floors.items():
            result[key] += value
        residual = remaining - sum(floors.values())
        ordered = sorted(available, key=lambda key: (
            -(remaining * available[key] % total), rank(context + "\0" + key)))
        for key in ordered[:residual]:
            result[key] += 1
    require(sum(result.values()) == quota and all(result[k] <= capacities[k] for k in result),
            "Invalid stratified allocation")
    return result


def date_combos(row: dict) -> Counter:
    return Counter("|".join(date_parts(row["text"][span["start"]:span["end"]]))
                   for span in row["spans"] if span["label"] == "DATE")


def select(units: list[Unit], rows: list[dict]) -> tuple[list[Unit], dict]:
    selected = []
    allocation_audit = {}
    unit_march_counts = {}
    for unit in units:
        if unit.bucket == "excluded_repeated_course_long":
            continue
        unit_march_counts[unit.pair_id] = sum(
            count for i in unit.indices for combo, count in date_combos(rows[i]).items()
            if combo.startswith("month_year|03|"))

    for bucket, quota in BUCKET_QUOTAS.items():
        candidates = [unit for unit in units if unit.bucket == bucket]
        march_capacities = Counter()
        for unit in candidates:
            if unit_march_counts[unit.pair_id]:
                march_capacities[unit.family] += unit_march_counts[unit.pair_id]
        march_family_caps = allocate(dict(march_capacities),
            min(MARCH_MONTH_YEAR_CAPS[bucket], sum(march_capacities.values())), bucket + "\0march-cap")
        selected_march_by_family = Counter()

        def admissible(unit: Unit) -> bool:
            return (selected_march_by_family[unit.family] + unit_march_counts[unit.pair_id]
                    <= march_family_caps.get(unit.family, 0))

        eligible = [unit for unit in candidates if admissible(unit)]
        by_family: dict[str, list[Unit]] = defaultdict(list)
        for unit in eligible:
            by_family[unit.family].append(unit)
        family_quota = allocate({key: len(value) for key, value in by_family.items()}, quota, bucket)
        allocation_audit[bucket] = {"source_units": len(candidates), "selected_units": quota,
                                    "admissible_at_bucket_start": len(eligible),
                                    "march_month_year_span_cap": MARCH_MONTH_YEAR_CAPS[bucket],
                                    "march_month_year_family_caps": dict(sorted(march_family_caps.items())),
                                    "family_unit_quotas": dict(sorted(family_quota.items()))}
        for family in sorted(by_family):
            by_layout: dict[str, list[Unit]] = defaultdict(list)
            for unit in by_family[family]:
                by_layout[unit.layout].append(unit)
            layout_quota = allocate({key: len(value) for key, value in by_layout.items()},
                                    family_quota[family], bucket + "\0" + family)
            for layout in sorted(by_layout):
                ordered = sorted(by_layout[layout], key=lambda unit: rank(unit.pair_id))
                taken = 0
                for unit in ordered:
                    if taken == layout_quota[layout]:
                        break
                    if admissible(unit):
                        selected.append(unit)
                        selected_march_by_family[unit.family] += unit_march_counts[unit.pair_id]
                        taken += 1
                require(taken == layout_quota[layout], "March DATE cap makes a stratum quota infeasible")
        allocation_audit[bucket]["selected_march_month_year_spans"] = sum(selected_march_by_family.values())
    return selected, allocation_audit


def date_parts(surface: str) -> tuple[str, str, str]:
    """Classify the surface; never infer an unexpressed month or year."""
    month = year = "not_expressed"
    if re.fullmatch(r"\d{4}", surface):
        return "year_only", month, surface
    if match := re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", surface):
        return "iso_full", match[2], match[1]
    if match := re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", surface):
        return "slash_full", match[1], match[3]
    if match := re.fullmatch(r"(\d{2})/(\d{2})", surface):
        return "slash_month_day", match[1], year
    if match := re.fullmatch(r"([A-Za-z]+) (\d{4})", surface):
        return "month_year", f"{MONTHS[match[1].casefold()]:02d}", match[2]
    if match := re.fullmatch(r"(\d{2}) ([A-Za-z]+) (\d{4})", surface):
        return "day_named_month_year", f"{MONTHS[match[2].casefold()]:02d}", match[3]
    if match := re.fullmatch(r"([A-Za-z]+) (\d{2}), (\d{4})", surface):
        return "named_month_day_year", f"{MONTHS[match[1].casefold()]:02d}", match[3]
    return "unclassified", month, year


def add_repeated_course_longs(selected: list[Unit], units: list[Unit], rows: list[dict]) -> tuple[list[Unit], dict]:
    """Replace exactly 12 positive singletons; preserve the prior draft on disk."""
    removable = [unit for unit in selected if unit.bucket == "realistic_positive_singleton"
                 and not any(span["label"] == "DATE" for i in unit.indices for span in rows[i]["spans"])]
    removed = sorted(removable, key=lambda unit: rank("v2-remove\0" + unit.pair_id))[:12]
    require(len(removed) == 12, "Insufficient positive singleton rows for long-note replacements")
    candidates = [unit for unit in units if unit.bucket == "excluded_repeated_course_long"]
    candidates_by_form = defaultdict(list)
    for unit in candidates:
        require(len(unit.indices) == 1, "Long candidate is not a complete singleton unit")
        row = rows[unit.indices[0]]
        require(row["spans"] and not row["negative"], "Long candidate is PHI-free")
        combos = [date_parts(row["text"][span["start"]:span["end"]])
                  for span in row["spans"] if span["label"] == "DATE"]
        forms = {form for form, _, _ in combos}
        if len(forms) == 1 and next(iter(forms)) in LONG_FORM_QUOTAS:
            candidates_by_form[next(iter(forms))].append(unit)
    added = []
    used_families = set()
    used_years = set()
    for form, quota in LONG_FORM_QUOTAS.items():
        for slot in range(quota):
            target_tokens = (1500, 2500, 3500, 4500)[slot] if quota == 4 else (1500, 3500)[slot]
            available = [unit for unit in candidates_by_form[form] if unit.family not in used_families]
            require(bool(available), f"No distinct-family long candidate for {form}")

            def priority(unit: Unit) -> tuple:
                row = rows[unit.indices[0]]
                years = {year for _, _, year in (date_parts(row["text"][s["start"]:s["end"]])
                         for s in row["spans"] if s["label"] == "DATE") if year != "not_expressed"}
                return (bool(years & used_years), abs(row["token_audit"]["token_count"] - target_tokens),
                        rank("v2-long\0" + unit.pair_id))

            unit = min(available, key=priority)
            added.append(unit)
            used_families.add(unit.family)
            row = rows[unit.indices[0]]
            used_years.update(year for _, _, year in (date_parts(row["text"][s["start"]:s["end"]])
                              for s in row["spans"] if s["label"] == "DATE") if year != "not_expressed")
    removed_ids = {unit.pair_id for unit in removed}
    result = [unit for unit in selected if unit.pair_id not in removed_ids] + added
    return result, {"base_replay_sha256": BASE_REPLAY_SHA, "preserved_base_artifact": "reports/v13_focused_context_prototype/v12_replay_draft.jsonl",
                    "removed_realistic_positive_singleton_ids": [rows[u.indices[0]]["id"] for u in removed],
                    "added_repeated_course_long_ids": [rows[u.indices[0]]["id"] for u in added],
                    "long_note_form_quotas": LONG_FORM_QUOTAS, "long_families": sorted(used_families),
                    "long_years": sorted(used_years),
                    "long_selection": "Original singleton TRAIN units, 12 distinct families, full-date/year-only forms, preferring distinct years and varied source-declared token lengths. Exact o200k lengths are independently recomputed below."}



REPLAY_SHA = '94811f4d0e577d9ba483d9c78a08db91ae0de3108678fc8958eba07947ac844d'


def build_replay():
    """Return approved original row bytes and source-line hash lineage."""
    raw, lines, rows, manifest, manifest_raw = load_source()
    units = make_units(rows)
    selected, _ = select(units, rows)
    base_indices = sorted(i for unit in selected for i in unit.indices)
    require(sha(b''.join(lines[i] for i in base_indices)) == BASE_REPLAY_SHA, 'Original replay allocation changed')
    selected, _ = add_repeated_course_longs(selected, units, rows)
    indices = sorted(i for unit in selected for i in unit.indices)
    require(len(indices) == len(set(indices)) == 3000, 'Expected exactly 3000 unique TRAIN source lines')
    selected_set = set(indices)
    require(all(not (selected_set & set(unit.indices)) or set(unit.indices) <= selected_set for unit in units), 'A pair was split')
    payload = b''.join(lines[i] for i in indices)
    require(sha(payload) == REPLAY_SHA, 'Reviewed replay bytes changed')
    lineage = [{'source_line_1based': i + 1, 'exact_source_line_sha256': sha(lines[i]),
                'id_sha256': sha(rows[i]['id'].encode())} for i in indices]
    return payload, lineage
