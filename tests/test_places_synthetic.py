from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import pytest

from clinical_deid import place_sources
from clinical_deid.annotation_policy import POLICY_VERSION, annotation_errors, canonical_identity_group, identity_split
from clinical_deid.annotation_policy_places import ADDENDUM, PLACE_FIXTURES, place_value_pool
from clinical_deid.places_contexts import LABELLED_SLOTS, LAYOUTS, TEMPLATES
from clinical_deid.places_synthetic import (
    PHENOMENON_TAGS, POSITIVES, SPLITS, boundary_errors, short_name, title_case_cms, train_pool, validate_row,
    value_partition,
)


ROOT = Path(__file__).resolve().parents[1]
needs_sources = pytest.mark.skipif(
    not place_sources.sources_available(),
    reason="pinned place sources not cached; run `python -m clinical_deid.place_sources fetch`")
DECISIONS = {PHENOMENON_TAGS[key] for key in ("p1", "p2", "p3", "p4", "p5", "p6", "n1", "n2", "n3", "n4", "n5", "mixed")}
DECISIONS |= {"state_inside_locality", "leading_abbreviation"}


@pytest.fixture(scope="module")
def corpus():
    from clinical_deid.places_synthetic import generate
    return generate()


@pytest.fixture(scope="module")
def bank():
    from clinical_deid.places_synthetic import value_bank
    return value_bank()


def gold(row):
    return [(row["text"][s["start"]:s["end"]], s["label"]) for s in row["spans"]]


# ---------------------------------------------------------------- pure checks


def test_place_fixtures_satisfy_the_generator_boundary_rules():
    for row in PLACE_FIXTURES:
        assert boundary_errors(row["text"], row["spans"]) == [], row["id"]


@pytest.mark.parametrize("text,span,label,message", [
    ("Seen at St. Francis Hospital.", "Francis Hospital", "FACILITY", "leading St."),
    ("Treated in Rochester, MN.", "in Rochester, MN", "LOCATION", "preposition"),
    ("Treated at Mayo Clinic in Rochester.", "Mayo Clinic in Rochester", "FACILITY", "adjacent locality"),
    ("Lives in Texas.", "Texas", "LOCATION", "state or country alone"),
    ("Dr. Priya Testier reviewed.", "Dr. Priya Testier", "NAME", "title"),
    ("Pt from Chicago.", "Chicago.", "LOCATION", "punctuation"),
])
def test_boundary_rules_reject_addendum_violations(text, span, label, message):
    start = text.index(span)
    errors = boundary_errors(text, [{"start": start, "end": start + len(span), "label": label}])
    assert any(message in error for error in errors)


def test_cms_title_case_and_short_names():
    assert title_case_cms("ST FRANCIS HOSPITAL") == "St. Francis Hospital"
    assert title_case_cms("MCPHERSON HOSPITAL") == "McPherson Hospital"
    assert title_case_cms("The Medical Center at Russellville") is None
    assert title_case_cms("CORTERRA OF WICHITA LLC") is None
    assert title_case_cms("PHOENIX VA MEDICAL CENTER") == "Phoenix VA Medical Center"
    assert short_name("Houston Methodist Hospital") == "Houston Methodist"
    assert short_name("St. Francis Medical Center") == "St. Francis"
    assert short_name("Memorial Hospital") is None
    assert short_name("Southeast Health Medical Center") is None


def test_train_pool_requires_value_and_expansion_in_train():
    heldout = next(f"Testford {i}" for i in range(100) if place_value_pool(f"Testford {i}") == "heldout")
    train = next(f"Testford {i}" for i in range(100) if place_value_pool(f"Testford {i}") == "train")
    assert train_pool(train) and not train_pool(heldout)
    assert not train_pool(train, expansion=heldout)
    assert value_partition(" Chicago ") == value_partition("chicago")


def test_templates_are_split_disjoint_and_use_known_slots():
    seen = defaultdict(set)
    known = set(LABELLED_SLOTS) | {"cite", "finding", "device", "cohort_study", "dz", "state", "country", "region",
                                   "loc_state", "word", "age", "drug", "symptom", "n", "days"}
    for split, groups in TEMPLATES.items():
        for group, layouts in groups.items():
            assert set(layouts) == set(LAYOUTS), (split, group)
            for texts in layouts.values():
                for text in texts:
                    seen[split].add(text)
                    assert set(re.findall(r"\{(\w+)\}", text)) <= known, text
    assert not seen["train"] & seen["validation"]


def test_curated_care_site_list_records_a_public_source_per_entry():
    data = json.loads(place_sources.CURATED_CARE_SITES.read_text())
    surfaces = Counter()
    for entry in data["entries"]:
        assert entry["source"].startswith("https://") and entry["expansion"].strip()
        assert entry["abbreviations"] or entry["short_names"] or entry["full_names"]
        assert all(re.fullmatch(r"[A-Z]{2,7}", a) for a in entry["abbreviations"])
        assert not any(s.casefold().startswith("the ") for s in entry["short_names"] + entry["full_names"])
        surfaces.update(entry["abbreviations"] + entry["short_names"])
    assert not [s for s, n in surfaces.items() if n > 1]


def test_sources_are_pinned_by_url_and_sha256():
    for source in place_sources.SOURCES.values():
        assert source["url"].startswith(("https://www2.census.gov/", "https://data.cms.gov/"))
        assert re.fullmatch(r"[0-9a-f]{64}", source["sha256"])


# ---------------------------------------------------------------- generated corpus


@needs_sources
def test_generation_is_deterministic_and_seeded():
    from clinical_deid.places_synthetic import generate
    first = generate(seed=7, train_count=300, validation_count=120)
    assert first == generate(seed=7, train_count=300, validation_count=120)
    assert first != generate(seed=8, train_count=300, validation_count=120)


@needs_sources
def test_rows_are_valid_v2_rows_with_exact_offsets(corpus, bank):
    for split in SPLITS:
        assert len({row["text"] for row in corpus[split]}) == len(corpus[split])
        for row in corpus[split]:
            assert row["split"] == split and row["annotation_policy"] == POLICY_VERSION
            assert row["annotation_addendum"] == ADDENDUM and row["cohort"] == "realistic"
            assert annotation_errors(row["text"], row["spans"]) == []
            assert boundary_errors(row["text"], row["spans"]) == []
            validate_row(row, bank)
            for o in row["occurrences"]:
                assert row["text"][o["start"]:o["end"]] == o["value"]


@needs_sources
def test_every_addendum_decision_is_exercised_in_both_splits(corpus):
    for split in SPLITS:
        tags = Counter(tag for row in corpus[split] for tag in row["phenomena"])
        assert DECISIONS <= set(tags), DECISIONS - set(tags)
        assert {row["positive_phenomenon"] for row in corpus[split]} - {None} == set(POSITIVES)
        assert {row["negative_family"] for row in corpus[split]} - {None} == {"n1", "n2", "n3", "n4", "n5"}


@needs_sources
def test_only_train_pool_real_values_are_emitted(corpus, bank):
    emitted = 0
    for split in SPLITS:
        for row in corpus[split]:
            assert not bank.heldout_pattern.search(row["text"])
            for o in row["occurrences"]:
                for real in o["real_values"]:
                    emitted += 1
                    assert place_value_pool(real["key"]) == "train", real
                    if real["expansion"]:
                        assert place_value_pool(real["expansion"]) == "train", real
    assert emitted > 3000


@needs_sources
def test_heldout_pool_guard_rejects_a_heldout_value(corpus, bank):
    curated = json.loads(place_sources.CURATED_CARE_SITES.read_text())["entries"]
    heldout = next(e["expansion"] for e in curated if place_value_pool(e["expansion"]) == "heldout")
    row = json.loads(json.dumps(next(r for r in corpus["train"] if r["positive_phenomenon"] == "p2")))
    site = next(o for o in row["occurrences"] if o["slot"] == "site")
    site["real_values"][0]["expansion"] = heldout
    with pytest.raises(ValueError, match="Heldout"):
        validate_row(row)
    assert bank.heldout_pattern.search(f"seen at {heldout} last year")


@needs_sources
def test_gold_comes_from_authored_roles(corpus):
    for split in SPLITS:
        for row in corpus[split]:
            labels = {s["label"] for s in row["spans"]}
            for o in row["occurrences"]:
                assert o["label"] == LABELLED_SLOTS.get(o["slot"]), o
            if row["role"] == "negative":
                assert not labels & {"LOCATION", "FACILITY", "ADDRESS"}
                assert row["place_negative"]
            else:
                assert labels & {"LOCATION", "FACILITY"}


@needs_sources
def test_mixed_rows_annotate_care_and_not_citation_of_one_institution(corpus):
    for split in SPLITS:
        mixed = [row for row in corpus[split] if row["role"] == "mixed"]
        assert mixed
        for row in mixed:
            site = next(o for o in row["occurrences"] if o["slot"] == "site")
            cite = next(o for o in row["occurrences"] if o["slot"] == "cite")
            assert site["label"] == "FACILITY" and cite["label"] is None
            assert site["real_values"][0]["expansion"] == cite["real_values"][0]["expansion"]
            assert {"mixed_roles", "n3_institution_source"} <= set(row["phenomena"])


@needs_sources
def test_matched_negatives_share_unit_layout_identity_and_cover_every_pattern(corpus):
    for split in SPLITS:
        by_id = {row["id"]: row for row in corpus[split]}
        covered, families = set(), set()
        for row in corpus[split]:
            if row["role"] != "negative":
                continue
            positive = by_id[row["matched_positive"]]
            assert positive["pair_id"] == row["pair_id"] and positive["layout"] == row["layout"]
            assert positive["identity_group"] == row["identity_group"]
            covered.add((positive["positive_phenomenon"], positive["layout"]))
            families.add((row["negative_family"], row["layout"]))
        assert covered == {(p, layout) for p in POSITIVES for layout in LAYOUTS}
        assert families == {(n, layout) for n in ("n1", "n2", "n3", "n4", "n5") for layout in LAYOUTS}
        negatives = [row for row in corpus[split] if row["role"] == "negative"]
        assert len(negatives) / len(corpus[split]) >= .35
        assert sum(row["matched_value_shared"] for row in negatives) >= .15 * len(negatives)


@needs_sources
def test_layouts_are_balanced(corpus):
    for split in SPLITS:
        counts = Counter(row["layout"] for row in corpus[split])
        assert set(counts) == set(LAYOUTS)
        assert all(.2 <= count / len(corpus[split]) <= .31 for count in counts.values()), counts


@needs_sources
def test_identities_and_template_families_respect_the_split(corpus):
    families = {}
    for split in SPLITS:
        for row in corpus[split]:
            assert all(identity_split(group) == split for group in row["identity_groups"])
            assert {canonical_identity_group(name) for name in row["full_identities"]} == set(row["identity_groups"])
            assert row["template_family"].startswith(f"places_{split}.")
            families.setdefault(row["template_family"], split)
            assert families[row["template_family"]] == split
            for o in row["occurrences"]:
                for real in o["real_values"]:
                    assert real["partition"] in {"shared", split}


@needs_sources
def test_no_reserved_scheme_violations(corpus):
    for split in SPLITS:
        for row in corpus[split]:
            for text, label in gold(row):
                assert label in {"NAME", "DATE", "LOCATION", "FACILITY", "MRN", "PHONE"}
                if label == "PHONE":
                    assert re.fullmatch(r"\(\d{3}\) 555-01\d\d", text)
                if label == "MRN":
                    assert re.fullmatch(r"PL-\d{8}", text)
                if label == "DATE":
                    assert re.search(r"20(?:0[89]|1\d|2[0-5])|/(?:0[89]|1\d|2[0-5])$", text), text


@needs_sources
def test_span_boundaries_follow_the_fixtures(corpus):
    seen = Counter()
    for split in SPLITS:
        for row in corpus[split]:
            for o in row["occurrences"]:
                if o["slot"] == "site" and re.match(r"(?i)(?:st|mt|ft|ste)\. ", o["value"]):
                    seen["leading_abbreviation_inside"] += 1
                if o["slot"] in {"loc", "site_loc"} and "," in o["value"]:
                    seen["state_inside_locality"] += 1
                if o["slot"] in {"loc_state", "state", "country", "region"}:
                    assert o["label"] is None
                    seen["state_alone_unlabelled"] += 1
            if row["positive_phenomenon"] == "p5":
                site = next(o for o in row["occurrences"] if o["slot"] == "site")
                loc = next(o for o in row["occurrences"] if o["slot"] == "site_loc")
                between = row["text"][site["end"]:loc["start"]]
                assert site["end"] < loc["start"] and re.fullmatch(r"(?: in |, | \(| \| | / )", between), between
    assert min(seen.values()) > 20 and len(seen) == 3


@needs_sources
def test_packaged_artifacts_reproduce_and_pass_strict_audit():
    from clinical_deid.corpus_quality import audit_corpora
    from clinical_deid.places_synthetic import SOURCE_FILES, build_artifacts
    if not (ROOT / "data/places_train.jsonl").exists():
        pytest.skip("generated corpora are gitignored; run `python -m clinical_deid.places_synthetic generate`")
    artifacts = build_artifacts()
    for path, raw in artifacts.items():
        assert (ROOT / path).read_bytes() == raw, path
    for split, count in (("train", 3000), ("validation", 600)):
        manifest = json.loads((ROOT / f"data/places_{split}.jsonl.manifest.json").read_bytes())
        payload = (ROOT / f"data/places_{split}.jsonl").read_bytes()
        assert manifest["sha256"] == hashlib.sha256(payload).hexdigest() and manifest["rows"] == count
        assert manifest["annotation_policy"] == POLICY_VERSION and manifest["annotation_addendum"] == ADDENDUM
        assert manifest["independent_final"] is False and manifest["used_for_training"] is (split == "train")
        for name in SOURCE_FILES:
            assert hashlib.sha256((ROOT / "src/clinical_deid" / name).read_bytes()).hexdigest() == manifest["source_hashes"][name]
        assert manifest["support"]["place_negative_fraction"] >= .35
    result = audit_corpora({"train": [ROOT / "data/places_train.jsonl"],
                            "validation": [ROOT / "data/places_validation.jsonl"]}, strict_policy=True)
    assert result["passed"], result["errors"][:5]
    assert {w["code"] for w in result["warnings"]} <= {"limited_label_support", "token_lengths_unavailable"}
