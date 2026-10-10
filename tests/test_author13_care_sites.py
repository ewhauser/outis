"""Tests for the tk13 care-site and locality generator."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from outis.data.annotation_policy import annotation_errors
from outis.data.annotation_policy_care_sites import (
    CARE_SITE_FIXTURES,
    CARE_SITE_FOLLOWUP_FIXTURES,
)
from outis.data.annotation_policy_places import PLACE_FIXTURES, place_value_pool

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "author13", ROOT / "scripts" / "takeover" / "author13_care_sites.py"
)
author13 = importlib.util.module_from_spec(SPEC)
sys.modules["author13"] = author13
SPEC.loader.exec_module(author13)

lab = author13.lab
build_row = author13.build_row

GENERIC_HEADS = (
    "clinic",
    "office",
    "site",
    "unit",
    "branch",
    "location",
    "center",
    "hospital",
    "campus",
    "neighborhood",
)
SERVICE_WORDS = re.compile(
    r"\b(vna|dme|adult day|hospice|post-acute|skilled nursing|reference laboratory|specialty pharmacy|"
    r"mail-order|home health|home infusion|home medical)\b"
)


@pytest.fixture(scope="module")
def bank():
    rows, _world = author13.generate(author13.SEED)
    return rows


def gold(example):
    return [
        (s["label"], example["text"][s["start"] : s["end"]]) for s in example["spans"]
    ]


def test_generator_is_deterministic_for_its_seed(bank):
    again, _ = author13.generate(author13.SEED)
    assert [(r["id"], r["text"], r["spans"], r["pool"]) for r in bank] == [
        (r["id"], r["text"], r["spans"], r["pool"]) for r in again
    ]
    other, _ = author13.generate(author13.SEED + 1)
    assert [r["text"] for r in other] != [r["text"] for r in bank]


def test_rows_have_valid_offsets_and_metadata(bank):
    assert len(bank) == 2400
    ids = set()
    for row in bank:
        assert not annotation_errors(row["text"], row["spans"])
        assert row["split"] == "train" and row["annotation_policy"] == "clinical-phi-v2"
        assert row["family"] == row["template_family"]
        assert row["id"] not in ids
        ids.add(row["id"])
        for span in row["spans"]:
            assert row["text"][span["start"] : span["end"]] == span["text"]
        for real in row["pool"]:
            assert (
                real["pool"] == "train" and place_value_pool(real["value"]) == "train"
            )
    layouts = {r["layout"] for r in bank}
    assert len(layouts) >= 25
    counts = {name: sum(r["layout"] == name for r in bank) for name in layouts}
    assert max(counts.values()) <= 0.08 * len(bank)


def test_conventions_in_a_sample_of_rows(bank):
    p7 = the_name = service = article = 0
    for row in bank[::3]:
        text, spans = row["text"], row["spans"]
        for span in spans:
            before = text[: span["start"]].lower()
            after = text[span["end"] :].lower()
            value = span["text"]
            if span["label"] == "LOCATION":
                # P7: a generic head word stays outside the locality.
                assert not any(
                    re.search(rf"\b{h}\b", value.lower()) for h in GENERIC_HEADS
                )
                if any(after.startswith(f" {h}") for h in GENERIC_HEADS):
                    p7 += 1
            if span["label"] == "FACILITY":
                if value.lower().startswith("the "):
                    # D3: inside only for authored names, which are normal-case rows.
                    assert row["case_mode"] == "normal" and value.startswith("The ")
                    the_name += 1
                if before.endswith((" the ", "\nthe ")) or before == "the ":
                    # D3: an article before a name stays outside.
                    assert not value.lower().startswith("the ")
                    article += 1
                if SERVICE_WORDS.search(value.lower()):
                    # D2: the type words are inside the span, not left outside it.
                    assert not re.match(
                        r"\s*(vna|dme|program|center|care|health|agency|association|supply|pharmacy)\b",
                        after,
                    )
                    service += 1
            if span["label"] == "MRN":
                assert not value.lower().startswith(("mrn", "#", ":"))
            if span["label"] == "NAME":
                assert not re.match(r"(dr|mr|mrs|ms)\b", value.lower())
                assert not re.search(r"\b(rn|lpn|msw|pt|np|md)$", value.lower())
            if span["label"] == "AGE":
                assert (
                    re.fullmatch(r"ninety(-\w+)?|nineties", value.lower())
                    or int(value) > 89
                )
    assert p7 and the_name and service and article


def test_negative_rows_have_no_place_or_facility_spans(bank):
    negatives = [r for r in bank if r["polarity"] == "negative"]
    assert 0.3 <= len(negatives) / len(bank) <= 0.37
    for row in negatives:
        assert not any(s["label"] in ("LOCATION", "FACILITY") for s in row["spans"])


def test_case_modes_keep_offsets_exact():
    for example in (*CARE_SITE_FIXTURES, *PLACE_FIXTURES):
        marked = author13.marked_from_example(example)
        for case in ("normal", "upper", "lower"):
            text, spans = build_row(marked, case)
            assert not annotation_errors(text, spans)
            assert [s["label"] for s in spans] == [s["label"] for s in example["spans"]]


def test_span_construction_reproduces_fixture_gold():
    for example in (*CARE_SITE_FIXTURES, *PLACE_FIXTURES):
        text, spans = build_row(author13.marked_from_example(example))
        assert text == example["text"]
        assert [(s["start"], s["end"], s["label"]) for s in spans] == [
            (s["start"], s["end"], s["label"]) for s in example["spans"]
        ]


def test_helpers_compose_the_care_site_fixtures():
    by_id = {e["id"]: e for e in CARE_SITE_FIXTURES}

    def loc(v):
        return lab("LOCATION", v)

    def fac(v):
        return lab("FACILITY", v)

    built = {
        "p7_city_modifies_unnamed_site": f"Seen at our {loc('Vellmont')} clinic; labs drawn at the {loc('Corriston')} office.",
        "p7_named_site_is_still_facility": f"Seen at {fac('Vellmont Clinic')}; our {loc('Corriston')} site sent records.",
        "service_organisations_are_facilities": (
            f"Referred to {fac('Vossmere VNA')}; bed delivered by {fac('Tindle Brook DME')}; attends "
            f"{fac('Saint Odile Adult Day Program')}; labs from {fac('Yarrowick Laboratory Services')}."
        ),
        "unnamed_service_is_not_phi": "Home health referral sent; DME delivered a hospital bed; the hospice team will visit.",
        "employer_without_care_role": (
            f"Works nights at a warehouse for Pellham Logistics; admitted to {fac('Hospice of Marrowdale')}."
        ),
        "the_part_of_name": f"Lives at {fac('The Marlowe at Fennick Ridge')}; moved from {fac('The Quillon at Ashby Mill')}.",
        "the_as_article": (
            f"The {fac('Pellham Drug & Gift')} pharmacist called; she was seen at the {fac('Ashgrove Family Health')} office."
        ),
    }
    assert set(built) == set(by_id)
    for key, marked in built.items():
        text, spans = build_row(marked)
        assert text == by_id[key]["text"], key
        assert gold({"text": text, "spans": spans}) == gold(by_id[key]), key


def test_helpers_compose_the_place_fixtures():
    by_id = {e["id"]: e for e in PLACE_FIXTURES}

    def loc(v):
        return lab("LOCATION", v)

    def fac(v):
        return lab("FACILITY", v)

    def name(v):
        return lab("NAME", v)

    built = {
        "p1_patient_locality": f"Pt from {loc('Chicago')}; daughter lives in {loc('Tacoma, WA')}.",
        "p5_care_site_then_locality": f"Treated at {fac('Mayo Clinic')} in {loc('Rochester, MN')}.",
        "p6_clinician_affiliation": f"Dr. {name('Priya Testier')} from {fac('UCSF')} reviewed the scan.",
        "n5_place_word_as_person_or_word": f"{name('Florence Testier')} needs reading glasses and a mobile phone charger.",
        "state_inside_locality": f"Lives in {loc('Tacoma, WA')}, not elsewhere in Washington.",
    }
    for key, marked in built.items():
        text, spans = build_row(marked)
        assert text == by_id[key]["text"], key
        assert gold({"text": text, "spans": spans}) == gold(by_id[key]), key


def test_no_reused_example_names_or_heldout_values(bank):
    low = "\n".join(r["text"].casefold() for r in bank)
    for fragment in author13.BANNED_FRAGMENTS:
        assert fragment not in low
    assert not [x for r in bank for x in r["pool"] if x["pool"] != "train"]


# --------------------------------------------------------------------------- v2


@pytest.fixture(scope="module")
def bank2():
    rows, _world = author13.generate(author13.SEED_V2, "v2")
    return rows


def tagged(rows, tag):
    # long notes mix many cases in one document, so they are checked by the generic tests only
    return [r for r in rows if tag in r["phenomena"] and r["cohort"] != "long_document"]


@pytest.mark.parametrize(
    ("version", "seed", "name"),
    [
        ("v1", author13.SEED, "tk13_train_care_sites_localities.jsonl"),
        ("v2", author13.SEED_V2, "tk13_train_care_sites_localities__v2.jsonl"),
        ("v3", author13.SEED_V3, "tk13_train_care_sites_localities__v3.jsonl"),
        ("v4", author13.SEED_V4, "tk13_train_care_sites_localities__v4.jsonl"),
    ],
)
def test_every_version_reproduces_byte_for_byte(version, seed, name):
    import hashlib
    import json

    rows, _ = author13.generate(seed, version)
    payload = (
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    ).encode()
    manifest = json.loads((ROOT / "data" / f"{name}.manifest.json").read_text())
    assert hashlib.sha256(payload).hexdigest() == manifest["sha256"]
    if not (ROOT / "data" / name).exists():
        pytest.skip("bank not present (corpora live in the private data repository)")
    assert (ROOT / "data" / name).read_bytes() == payload


def test_v2_is_deterministic_and_sized(bank2):
    again, _ = author13.generate(author13.SEED_V2, "v2")
    assert [(r["id"], r["text"], r["spans"]) for r in bank2] == [
        (r["id"], r["text"], r["spans"]) for r in again
    ]
    assert 2400 <= len(bank2) <= 2800
    ids = {r["id"] for r in bank2}
    assert len(ids) == len(bank2)
    assert all(i.startswith("tk13_care_sites_localities_v2-") for i in ids)
    for row in bank2:
        assert not annotation_errors(row["text"], row["spans"])
        assert row["split"] == "train" and row["family"] == row["template_family"]
        assert all(
            x["pool"] == "train" and place_value_pool(x["value"]) == "train"
            for x in row["pool"]
        )
    layouts = [r["layout"] for r in bank2]
    assert len(set(layouts)) >= 25
    assert max(layouts.count(x) for x in set(layouts)) <= 0.08 * len(bank2)
    case = [r["case_mode"] for r in bank2]
    assert 0.58 <= case.count("normal") / len(case) <= 0.62


def test_v2_repetition_and_banned_names(bank2):
    shares = author13.value_shares(bank2)
    assert max(shares.values()) <= 0.005 * len(bank2)
    low = "\n".join(r["text"].casefold() for r in bank2)
    for fragment in author13.BANNED_FRAGMENTS:
        assert fragment not in low


def test_v2_counties_are_location_including_county(bank2):
    rows = tagged(bank2, "county_name")
    assert rows
    for row in rows:
        spans = [s for s in row["spans"] if s["label"] == "LOCATION"]
        assert spans
        for span in spans:
            assert re.fullmatch(r"(?i)[\w.' -]+ county(, [a-z]{2})?", span["text"])
    for row in tagged(bank2, "county_word_generic"):
        assert not any(s["label"] == "LOCATION" for s in row["spans"])


def test_v2_named_sites_and_p7_in_all_cases(bank2):
    cap = tagged(bank2, "named_site_capitalised")
    assert cap
    assert all(r["case_mode"] == "normal" for r in cap)
    for row in cap:
        assert any(s["label"] == "FACILITY" for s in row["spans"])
    hidden = tagged(bank2, "named_site_case_hidden")
    assert hidden
    assert all(r["case_mode"] != "normal" for r in hidden)
    for row in hidden:
        for s in row["spans"]:
            if s["label"] == "FACILITY":
                before = row["text"][: s["start"]].lower().rstrip()
                assert not before.endswith(("our", " the"))
    assert tagged(bank2, "p7_case_hidden")
    assert tagged(bank2, "the_in_name_case_hidden")
    assert all(r["case_mode"] != "normal" for r in tagged(bank2, "p7_case_hidden"))


def test_v2_care_team_pharmacy_the_and_employer_roles(bank2):
    assert tagged(bank2, "p7_care_team")
    for row in tagged(bank2, "p7_care_team"):
        for s in row["spans"]:
            if s["label"] == "LOCATION":
                assert not re.search(r"team|hospice|nurse|clinic", s["text"].lower())
    assert tagged(bank2, "named_pharmacy_case_hidden")
    for row in tagged(bank2, "the_service_org_in_name"):
        assert any(
            s["label"] == "FACILITY" and s["text"].lower().startswith("the ")
            for s in row["spans"]
        )
    for row in tagged(bank2, "the_service_org_article"):
        for s in row["spans"]:
            if s["label"] == "FACILITY" and row["text"][: s["start"]].lower().endswith(
                "the "
            ):
                assert not s["text"].lower().startswith("the ")
    both = tagged(bank2, "employer_and_care_roles")
    assert both
    for row in both:
        facility = [s for s in row["spans"] if s["label"] == "FACILITY"]
        assert len(facility) == 1
        name = facility[0]["text"]
        assert row["text"].count(name) >= 2
        assert row["text"].index(name) < facility[0]["start"]
    for tag in (
        "employer_name",
        "ordinary_place_word_case_hidden",
        "invented_place_decoy",
    ):
        rows = tagged(bank2, tag)
        assert rows
        for row in rows:
            assert not any(s["label"] in ("FACILITY", "LOCATION") for s in row["spans"])


def test_followup_fixtures_are_rebuilt_exactly():
    by_id = {e["id"]: e for e in CARE_SITE_FOLLOWUP_FIXTURES}
    for example in CARE_SITE_FOLLOWUP_FIXTURES:
        text, spans = build_row(author13.marked_from_example(example))
        assert text == example["text"]
        assert [(s["start"], s["end"], s["label"]) for s in spans] == [
            (s["start"], s["end"], s["label"]) for s in example["spans"]
        ]

    def loc(v):
        return lab("LOCATION", v)

    def fac(v):
        return lab("FACILITY", v)

    built = {
        "county_name_is_location": f"Lives in {loc('Kestrine County')}; moved from {loc('Tollund County, OR')}.",
        "county_modifier_and_bare_county": (
            f"Records from the {loc('Kestrine County')} office; she grew up in {loc('Kestrine')}."
        ),
        "capitalised_site_word_names_site": f"Seen at our {fac('Ombrey Clinic')}; labs at our {loc('Gorsefield')} clinic.",
        "case_hidden_role_decides": (
            f"SEEN AT {fac('OMBREY CLINIC')}. LIVES AT {fac('THE HEARTH AT OMBREY')}. "
            f"labs at our {loc('gorsefield')} clinic."
        ),
        "the_service_organisation": (
            f"Enrolled with {fac('The Hearth at Ombrey')} hospice program; "
            f"The {fac('Saint Brevin VNA')} nurse called."
        ),
        "p7_care_team_and_lowercase_pharmacy": (
            f"the {loc('Gorsefield')} hospice team visited; refills at {fac('harlow vane drug')}."
        ),
        "ordinary_place_word_any_case": "NEEDS READING GLASSES AND A MOBILE PHONE CHARGER.",
        "employer_and_care_roles": (
            "Works as a nurse at Ombrey General Hospital; admitted to "
            f"{fac('Ombrey General Hospital')} overnight."
        ),
    }
    assert set(built) == set(by_id)
    for key, marked in built.items():
        text, spans = build_row(marked)
        assert text == by_id[key]["text"], key
        assert gold({"text": text, "spans": spans}) == gold(by_id[key]), key


@pytest.fixture(scope="module")
def bank3():
    rows, _world = author13.generate(author13.SEED_V3, "v3")
    return rows


def test_v3_size_ids_pools_and_caps(bank3):
    assert 2700 <= len(bank3) <= 2900
    assert all(r["id"].startswith("tk13_care_sites_localities_v3-") for r in bank3)
    assert all(r["suite"] == author13.SUITE_V3 for r in bank3)
    for row in bank3:
        assert not annotation_errors(row["text"], row["spans"])
        assert all(
            x["pool"] == "train" and place_value_pool(x["value"]) == "train"
            for x in row["pool"]
        )
    assert max(author13.value_shares(bank3).values()) <= 0.005 * len(bank3)
    layouts = [r["layout"] for r in bank3]
    assert max(layouts.count(x) for x in set(layouts)) <= 0.08 * len(bank3)
    case = [r["case_mode"] for r in bank3]
    assert 0.58 <= case.count("normal") / len(case) <= 0.62


def test_v3_county_modifier_and_bare_county(bank3):
    mod = tagged(bank3, "county_modifier_p7")
    bare = tagged(bank3, "bare_county_locality")
    neg = tagged(bank3, "county_site_generic")
    assert mod and bare and neg
    for case in ("normal", "upper", "lower"):
        assert any(r["case_mode"] == case for r in mod)
        assert any(r["case_mode"] == case for r in bare)
    for row in mod:
        county = [
            s
            for s in row["spans"]
            if s["label"] == "LOCATION" and s["text"].lower().endswith(" county")
        ]
        assert county
        after = row["text"][county[0]["end"] :].lower()
        assert re.match(r" (office|clinic|site|branch|location)\b", after)
    for row in bare:
        locs = [s for s in row["spans"] if s["label"] == "LOCATION"]
        assert locs and not any("county" in s["text"].lower() for s in locs)
    for row in neg:
        assert not any(s["label"] in ("LOCATION", "FACILITY") for s in row["spans"])
    assert all(r["polarity"] == "negative" for r in neg)


@pytest.fixture(scope="module")
def bank4():
    rows, world = author13.generate(author13.SEED_V4, "v4")
    return rows, world


def test_v4_size_ids_pools_and_caps(bank4):
    bank, _world = bank4
    assert 2700 <= len(bank) <= 2900
    assert all(r["id"].startswith("tk13_care_sites_localities_v4-") for r in bank)
    assert all(r["suite"] == author13.SUITE_V4 for r in bank)
    for row in bank:
        assert not annotation_errors(row["text"], row["spans"])
        assert all(
            x["pool"] == "train" and place_value_pool(x["value"]) == "train"
            for x in row["pool"]
        )
    assert max(author13.value_shares(bank).values()) <= 0.005 * len(bank)
    layouts = [r["layout"] for r in bank]
    assert len(set(layouts)) >= 25
    assert max(layouts.count(x) for x in set(layouts)) <= 0.08 * len(bank)
    case = [r["case_mode"] for r in bank]
    assert 0.58 <= case.count("normal") / len(case) <= 0.62
    negatives = [r for r in bank if r["polarity"] == "negative"]
    assert 0.3 <= len(negatives) / len(bank) <= 0.37


def test_v4_has_zero_exclusion_matches(bank4):
    from outis.data.generators.places import excluded_sequences, excluded_value

    bank, world = bank4
    # In memory and in the written file; only counts are asserted, never the strings.
    assert author13.exclusion_matches(bank) == 0
    path = ROOT / "data" / "tk13_train_care_sites_localities__v4.jsonl"
    if not path.exists():
        pytest.skip("bank not present (corpora live in the private data repository)")
    import json

    on_disk = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(on_disk) == len(bank)
    assert sum(excluded_sequences(r["text"]) > 0 for r in on_disk) == 0
    assert (
        sum(excluded_value(s["text"]) or False for r in on_disk for s in r["spans"])
        == 0
    )
    assert sum(excluded_value(x["value"]) for r in on_disk for x in r["pool"]) == 0
    manifest = json.loads(
        (
            ROOT / "data" / "tk13_train_care_sites_localities__v4.jsonl.manifest.json"
        ).read_text()
    )
    assert manifest["exclusion_filter"]["rows_with_a_match"] == 0
    assert manifest["exclusion_filter"]["rejected_draws"] == dict(
        sorted(world.rejected.items())
    )
    assert world.rejected  # the filter drew and rejected candidates


def test_v4_aged_and_snf_names_in_all_cases(bank4):
    bank, _ = bank4
    for tag, ending in (
        ("home_for_aged_name", "home for the aged"),
        ("snf_name", "skilled nursing facility"),
    ):
        rows = tagged(bank, tag)
        assert rows
        for case in ("normal", "upper", "lower"):
            assert any(r["case_mode"] == case for r in rows), (tag, case)
        for row in rows:
            facility = [s for s in row["spans"] if s["label"] == "FACILITY"]
            assert any(s["text"].lower().endswith(ending) for s in facility)
            for s in facility:
                # the type words are inside the span, and an article stays outside
                if s["text"].lower().endswith(ending):
                    assert not s["text"].lower().startswith("the ")
                    assert row["text"][s["end"] :].lower()[:1] in (".", " ", ";", ",")
    unnamed = tagged(bank4[0], "aged_snf_unnamed")
    assert unnamed
    for row in unnamed:
        assert row["polarity"] == "negative"
        assert not any(s["label"] in ("LOCATION", "FACILITY") for s in row["spans"])
