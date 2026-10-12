import importlib.util
import ipaddress
import json
import random
import re
import string
from pathlib import Path

import pytest
from pydantic import ValidationError

from outis.replacement import realistic_data as data
from outis.replacement.primitives import canonical_form
from outis.replacement.surrogates import replace
from outis.replacement.synthetic import synthetic_value
from outis.replacement.transform import ReplacementOptions, render
from outis.service import Deidentifier
from outis.types import Span

SECRET = bytes(range(32))
PINNED_SECRET = b"regression-secret-" + b"x" * 32
PATIENT = "patient-a"
ALL_LABELS = (
    "MRN ACCOUNT ID SSN PHONE EMAIL URL IP LOCATION ADDRESS FACILITY NAME AGE".split()
)


def spans_for(text, values):
    spans, cursor = [], 0
    for value, label in values:
        start = text.index(value, cursor)
        spans.append(Span(start, start + len(value), label, 0.9, "test"))
        cursor = start + len(value)
    return spans


def realistic(value, label, *, patient=PATIENT, keep_region=True):
    return synthetic_value(
        value,
        label,
        SECRET,
        patient,
        {},
        style="realistic",
        keep_region=keep_region,
    )


NOTE = (
    "Mara Calder (MRN MR-004521) seen 2024-02-28 at Lakeshore Pediatric Clinic, "
    "Brackenford, NM. Call (617) 555-0193 or mara.calder@example.org."
)
NOTE_SPANS = [
    ("Mara Calder", "NAME"),
    ("MR-004521", "MRN"),
    ("2024-02-28", "DATE"),
    ("Lakeshore Pediatric Clinic", "FACILITY"),
    ("Brackenford, NM", "LOCATION"),
    ("(617) 555-0193", "PHONE"),
    ("mara.calder@example.org", "EMAIL"),
]


# Outputs recorded before the style option existed. The default must not move.
PINNED_OBVIOUS = [
    ("MRN", "MR-004521", "SYN-MRN-393B59E5AB0A"),
    ("ACCOUNT", "ACCT-88213", "SYN-ACCOUNT-FD61214982B6"),
    ("ID", "EMP123", "SYN-ID-773DA572E793"),
    ("SSN", "123-45-6789", "000-57-9817"),
    ("PHONE", "(617) 555-0193", "(202) 555-0169"),
    ("EMAIL", "mara.calder@example.org", "person.f9801b98f908@example.invalid"),
    (
        "URL",
        "https://portal.example.com/records/42",
        "https://synthetic-68c29cafc849.invalid",
    ),
    ("IP", "10.1.2.3", "192.0.2.75"),
    ("LOCATION", "Brackenford, NM", "Example City 1589A8"),
    (
        "ADDRESS",
        "12 Tessford Lane, Apt 4B, Brackenford, NM 87101",
        "850 Synthetic Way, Example City, ZZ 00000",
    ),
    ("FACILITY", "Lakeshore Pediatric Clinic", "Synthetic Pediatric Clinic 34E95E"),
    ("NAME", "Mara Calder", "Avery Haven"),
    ("AGE", "93", "90+"),
]
PINNED_NOTE = (
    "Harper Juniper (MRN SYN-MRN-E4FDDAFDCAE1) seen 2024-09-24 at Synthetic "
    "Pediatric Clinic 4ED5C5, Example City 469484. Call (202) 555-0106 or "
    "person.bbb88c0c0af0@example.invalid."
)


@pytest.mark.parametrize("label,value,expected", PINNED_OBVIOUS)
def test_default_style_is_byte_identical_to_the_pinned_obvious_output(
    label, value, expected
):
    default = synthetic_value(value, label, PINNED_SECRET, PATIENT, {})
    explicit = synthetic_value(
        value, label, PINNED_SECRET, PATIENT, {}, style="obvious", keep_region=False
    )
    assert default == explicit == expected


def test_default_note_rendering_is_unchanged_on_every_entry_point():
    spans = spans_for(NOTE, NOTE_SPANS)
    legacy = replace(NOTE, spans, mode="synthetic", patient_id=PATIENT, secret=SECRET)
    configured = render(
        NOTE,
        spans,
        options={"type": "synthetic"},
        mode="synthetic",
        patient_id=PATIENT,
        secret=SECRET,
    )
    assert legacy["text"] == configured["text"] == PINNED_NOTE
    assert ReplacementOptions().style == "obvious"
    assert ReplacementOptions().keep_region is True


def test_style_options_are_strict():
    for bad in ({"style": "plausible"}, {"style": None}, {"keep_region": "yes"}):
        with pytest.raises(ValidationError):
            ReplacementOptions.model_validate(bad)


def _shape(value):
    return [
        "9"
        if char in string.digits
        else "A"
        if char in string.ascii_uppercase
        else "a"
        if char in string.ascii_lowercase
        else char
        for char in value
    ]


@pytest.mark.parametrize("label", ["MRN", "ACCOUNT", "ID"])
def test_identifiers_keep_length_character_classes_and_punctuation(label):
    rng = random.Random(label)
    shapes = ["999", "9999999", "AA-999999", "aa9-99/A", "A9A 9A9", "000123", "Z"]
    shapes += ["".join(rng.choices("9Aa-./ ", k=rng.randint(4, 30))) for _ in range(40)]
    for shape in shapes:
        value = "".join(
            rng.choice(string.digits)
            if c == "9"
            else rng.choice(string.ascii_uppercase)
            if c == "A"
            else rng.choice(string.ascii_lowercase)
            if c == "a"
            else c
            for c in shape
        )
        if not any(c.isalnum() for c in value):
            continue
        surrogate = synthetic_value(
            value, label, SECRET, PATIENT, {}, style="realistic"
        )
        assert len(surrogate) == len(value)
        assert _shape(surrogate) == _shape(value)
        assert canonical_form(surrogate) != canonical_form(value)
        assert surrogate == realistic(value, label)


def test_identifiers_are_injective_over_a_whole_short_domain():
    for values in (
        [f"{n:03d}" for n in range(1000)],
        [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase],
    ):
        surrogates = [realistic(value, "MRN") for value in values]
        # A keyed derangement: every value moves and none share a surrogate.
        assert sorted(surrogates) == sorted(values)
        assert all(s != v for s, v in zip(surrogates, values))


def test_identifiers_depend_on_key_patient_and_label():
    value = "MR-004521"
    base = realistic(value, "MRN")
    assert realistic(value, "MRN", patient="patient-b") != base
    assert realistic(value, "ACCOUNT") != base
    other = synthetic_value(value, "MRN", PINNED_SECRET, PATIENT, {}, style="realistic")
    assert other != base


def test_non_ascii_identifier_characters_become_ascii_of_the_same_class():
    surrogate = realistic("ÄB-١٢٣", "ID")
    assert re.fullmatch(r"[A-Z]{2}-\d{3}", surrogate)


def _ssn_digits(value):
    return "".join(c for c in value if c.isdigit())


def test_ssns_keep_layout_and_are_never_issuable():
    rng = random.Random(7)
    itin_groups = (
        set(range(50, 66)) | set(range(70, 89)) | {90, 91, 92} | set(range(94, 100))
    )
    layouts = ["{}-{}-{}", "{}{}{}", "{} {} {}"]
    for index in range(2000):
        layout = layouts[index % 3]
        value = layout.format(
            f"{rng.randint(1, 899):03d}",
            f"{rng.randint(1, 99):02d}",
            f"{rng.randint(1, 9999):04d}",
        )
        surrogate = realistic(value, "SSN")
        assert _shape(surrogate) == _shape(value)
        digits = _ssn_digits(surrogate)
        area, group = digits[:3], int(digits[3:5])
        assert area in {"000", "666"} or area.startswith("9")
        if area.startswith("9"):
            assert group not in itin_groups
    assert realistic("XXX-XX-6789", "SSN").startswith("XXX-XX-")


def test_phones_keep_layout_and_use_only_the_fictional_555_01_range():
    cases = [
        "(617) 555-0193",
        "617-555-0193",
        "617.555.0193",
        "6175550193",
        "+1 617 555 0193",
        "1-617-555-0193",
        "(612) 555-0193, extension 2276",
        "617-555-0193 x12",
        "555-0193",
        "Tel: 617-555-0193",
    ]
    for value in cases:
        surrogate = realistic(value, "PHONE")
        assert _shape(surrogate) == _shape(value), value
        digits = _ssn_digits(surrogate)
        start = 1 if digits[0] == "1" and len(digits) >= 11 else 0
        if len(digits) - start >= 10:
            assert digits[start : start + 3] in data.AREA_CODES
            start += 3
        assert digits[start : start + 5] == "55501"


def test_unrecognized_phone_shapes_fall_back_to_a_fictional_nanp_number():
    for value in ("+44 20 7946 0958", "0123 456", "1-800-FLOWERS"):
        surrogate = realistic(value, "PHONE")
        assert re.fullmatch(r"\(\d{3}\) 555-01\d\d", surrogate)
        assert surrogate[1:4] in data.AREA_CODES


RESERVED_HOST = re.compile(r"(?:[a-z0-9-]+\.)*example\.(?:com|org|net)", re.I)


def test_emails_use_realistic_local_parts_on_reserved_domains():
    for value in (
        "mara.calder@example.org",
        "MCalder88@stmarys.hospital.edu",
        "m_calder-2@mail.northwood.net",
        "calder@clinic.com",
        "not-an-email",
    ):
        surrogate = realistic(value, "EMAIL")
        local, _, host = surrogate.rpartition("@")
        assert RESERVED_HOST.fullmatch(host), surrogate
        assert "calder" not in surrogate.casefold()
        assert local and not local.startswith("person.")
    surrogate = realistic("mara.calder88@example.org", "EMAIL")
    assert re.fullmatch(r"[a-z]+\.[a-z]+\d\d@example\.org", surrogate)


def test_urls_keep_structure_on_reserved_hosts():
    value = (
        "https://portal.northwood-health.org/patients/4417/summary.pdf?mrn=88213#top"
    )
    surrogate = realistic(value, "URL")
    match = re.fullmatch(
        r"https://portal\.[a-z]+\.example\.org/[a-z]+/\d{4}/[a-z]+\.pdf\?[a-z]+=\d{5}#[a-z]+",
        surrogate,
    )
    assert match, surrogate
    for value in ("www.northwood.com", "http://northwood.co.uk:8080/x", "northwood"):
        surrogate = realistic(value, "URL")
        host = re.sub(r"^[a-z]+://", "", surrogate).split("/")[0].split(":")[0]
        assert RESERVED_HOST.fullmatch(host), surrogate
        assert "northwood" not in surrogate


DOCUMENTATION_NETWORKS = [
    ipaddress.ip_network(n)
    for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24", "2001:db8::/32")
]


def test_ips_stay_inside_documentation_ranges():
    rng = random.Random(3)
    values = [".".join(str(rng.randint(0, 255)) for _ in range(4)) for _ in range(300)]
    values += [
        "10.0.0.5/24",
        "fe80::1ff:fe23:4567:890a",
        "[2600:1f18::1]:443",
        "garbage",
    ]
    values += ["2001:0db8:85a3:0000:0000:8a2e:0370:7334"]
    for value in values:
        surrogate = realistic(value, "IP")
        core = re.sub(r"^\[|\](:\d+)?$|/\d+$", "", surrogate)
        address = ipaddress.ip_address(core)
        assert any(address in network for network in DOCUMENTATION_NETWORKS), value
    assert realistic("10.0.0.5/24", "IP").endswith("/24")
    exploded = realistic("2001:0db8:85a3:0000:0000:8a2e:0370:7334", "IP")
    assert len(exploded) == 39


def test_locations_keep_state_and_country_by_default():
    assert realistic("Brackenford, NM", "LOCATION").endswith(", NM")
    kept = realistic("Brackenford, New Mexico, USA", "LOCATION")
    assert kept.endswith(", New Mexico, USA") and "Brackenford" not in kept
    assert realistic("BRACKENFORD", "LOCATION").isupper()
    county = realistic("Tessford County, NM 87101", "LOCATION")
    town, zip_code = re.fullmatch(r"(.+) County, NM (\d{5})", county).groups()
    assert "Tessford" not in town
    assert 870 <= int(zip_code[:3]) <= 884


def test_location_region_can_be_replaced_with_a_consistent_zip():
    replaced = realistic("Brackenford, NM 87101", "LOCATION", keep_region=False)
    town, state, zip_code = re.fullmatch(r"(.+), ([A-Z]{2}) (\d{5})", replaced).groups()
    assert state != "NM" and state in data.US_STATES
    assert any(
        low <= int(zip_code[:3]) <= high for low, high in data.US_STATES[state][1]
    )
    country = realistic("Ashworth, Canada", "LOCATION", keep_region=False)
    assert not country.endswith("Canada")


def test_region_only_spans_are_replaced_rather_than_echoed():
    for value, pool in (("Ohio", "states"), ("NM", "codes"), ("Canada", "countries")):
        surrogate = realistic(value, "LOCATION")
        assert canonical_form(surrogate) != canonical_form(value)
        if pool == "codes":
            assert surrogate in data.US_STATES


def test_addresses_keep_layout_and_match_location_surrogates():
    city = realistic("Brackenford, NM", "LOCATION").removesuffix(", NM")
    value = "4021 N. Calder St.\nUnit 7B\nBrackenford, NM 87101-2231"
    surrogate = realistic(value, "ADDRESS")
    match = re.fullmatch(
        r"(\d{4}) N\. ([A-Z][a-z]+) ([A-Z][a-z]+)\.\nUnit (\d[A-Z])\n(.+), NM (\d{5})-(\d{4})",
        surrogate,
    )
    assert match, surrogate
    assert match[2] != "Calder" and match[5] == city
    assert 870 <= int(match[6][:3]) <= 884
    assert (match[3], match[3]) != ("St", "Street")
    assert match[3] in {abbreviation for _, abbreviation in data.STREET_SUFFIXES}

    box = realistic("PO Box 12345, Brackenford, NM 87101", "ADDRESS")
    zip_code = re.fullmatch(rf"PO Box \d{{5}}, {city}, NM (\d{{5}})", box)[1]
    assert 870 <= int(zip_code[:3]) <= 884
    apartment = realistic("12 Tessford Lane, Apt 4B, Brackenford", "ADDRESS")
    assert re.fullmatch(
        rf"\d\d [A-Z][a-z]+ [A-Z][a-z]+, Apt \d[A-Z], {city}", apartment
    ), apartment
    no_state = realistic("12 Oak St, Brackenford 87101", "ADDRESS")
    assert re.search(r" 00[1-4]\d\d$", no_state), no_state


def test_facilities_keep_whitelisted_setting_terms_with_fictional_names():
    for value, setting in (
        ("Lakeshore Pediatric Clinic", "Pediatric Clinic"),
        ("St. Marwick Hospital", "Hospital"),
        ("Northwood Dialysis Unit", "Dialysis Unit"),
        ("Marwick", "Medical Center"),
    ):
        surrogate = realistic(value, "FACILITY")
        assert surrogate.endswith(" " + setting), surrogate
        assert not set(value.split()) - set(setting.split()) & set(surrogate.split())
        assert "Synthetic" not in surrogate


def test_names_dates_and_ages_are_the_same_in_both_styles():
    for value, label in (("Mara Calder", "NAME"), ("93", "AGE")):
        assert realistic(value, label) == synthetic_value(
            value, label, SECRET, PATIENT, {}
        )


def test_realistic_rendering_copies_text_outside_spans_and_hides_secrets():
    spans = spans_for(NOTE, NOTE_SPANS)
    result = render(
        NOTE,
        spans,
        options={"type": "synthetic", "style": "realistic"},
        mode="synthetic",
        patient_id=PATIENT,
        secret=SECRET,
    )
    again = replace(
        NOTE,
        spans,
        mode="synthetic",
        patient_id=PATIENT,
        secret=SECRET,
        style="realistic",
    )
    assert result["text"] == again["text"]
    text, cursor = result["text"], 0
    for entity, span in zip(result["entities"], spans):
        assert (
            text[entity["processed_start"] : entity["processed_end"]]
            == entity["replacement"]
        )
        assert NOTE[cursor : span.start] in text
        cursor = span.end
    assert result["text"].startswith("Harper Juniper (MRN ")
    assert ", NM. Call (" in result["text"]
    encoded = json.dumps(result)
    for value, _ in NOTE_SPANS:
        assert value not in encoded
    assert SECRET.hex() not in encoded and PATIENT not in encoded


def _collisions():
    spec = importlib.util.spec_from_file_location(
        "surrogate_collisions",
        Path(__file__).parents[1] / "scripts" / "surrogate_collisions.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Upper bounds on the share of 2,000 distinct study-scope values whose
# realistic surrogate is shared; docs/replacement.md reports the measurements.
REALISTIC_COLLISION_BOUNDS = {
    "MRN (3 digits)": 0.0,
    "MRN (6 digits)": 0.0,
    "MRN (AA-999999)": 0.0,
    "ACCOUNT": 0.0,
    "ID": 0.0,
    "SSN": 0.005,
    "ADDRESS": 0.005,
    "EMAIL": 0.02,
    "URL": 0.02,
    "PHONE": 0.05,
    "FACILITY": 0.05,
    "LOCATION": 0.08,
}


def test_measured_collisions_stay_within_documented_bounds():
    collisions = _collisions()
    for label, bound in REALISTIC_COLLISION_BOUNDS.items():
        rate = collisions.collision_share(
            label, collisions.distinct_values(label, 2_000), style="realistic"
        )
        assert rate <= bound, (label, rate)


def test_realistic_style_composes_with_study_scope():
    study = {"type": "synthetic", "style": "realistic", "scope": "study"}
    study["study_id"] = "styles-test"
    text = "MRN MR-004521 lives in Brackenford, NM."
    spans = spans_for(text, [("MR-004521", "MRN"), ("Brackenford, NM", "LOCATION")])

    def rendered(patient, options):
        return render(
            text,
            spans,
            options=options,
            mode="synthetic",
            patient_id=patient,
            secret=SECRET,
        )["text"]

    first, second = rendered("patient-a", study), rendered("patient-b", study)
    assert first == second and re.fullmatch(
        r"MRN [A-Z]{2}-\d{6} lives in .+, NM\.", first
    )
    patient = {"type": "synthetic", "style": "realistic"}
    assert rendered("patient-a", patient) != rendered("patient-b", patient)
    assert rendered("patient-a", patient) != first


def test_service_and_cli_accept_the_style_option(monkeypatch, capsys):
    import io
    import sys

    import outis.service
    from outis import cli

    class Detector:
        def detect_batch(self, texts):
            return [
                [Span(m.start(), m.end(), "MRN") for m in re.finditer(r"MR-\d+", t)]
                for t in texts
            ]

    deid = Deidentifier(detector=Detector(), secret=SECRET)
    result = deid.deidentify(
        "MRN MR-004521.",
        mode="synthetic",
        patient_id=PATIENT,
        replacement={"type": "synthetic", "style": "realistic"},
    )
    assert re.fullmatch(r"MRN [A-Z]{2}-\d{6}\.", result["text"])

    monkeypatch.setenv("OUTIS_SECRET", SECRET.hex())
    monkeypatch.setattr(
        outis.service,
        "Deidentifier",
        lambda model_dir, *, secret=None, device="auto": Deidentifier(
            detector=Detector(), secret=secret
        ),
    )
    args = ["outis", "text", "--mode", "synthetic", "--patient-id", PATIENT]
    monkeypatch.setattr(sys, "argv", [*args, "--style", "realistic"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("MRN MR-004521."))
    cli.main()
    assert re.fullmatch(
        r"MRN [A-Z]{2}-\d{6}\.", json.loads(capsys.readouterr().out)["text"]
    )
    monkeypatch.setattr(sys, "argv", ["outis", "text", "--style", "realistic"])
    with pytest.raises(SystemExit):
        cli.main()
