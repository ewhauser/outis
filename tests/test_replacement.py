import json
import re
from datetime import date
from types import SimpleNamespace

import pytest

from clinical_deid.replacement import replace
from clinical_deid.types import Span


SECRET = bytes(range(32))


def spans_for(text, values):
    spans = []
    cursor = 0
    for value, label in values:
        start = text.index(value, cursor)
        spans.append(Span(start, start + len(value), label, 0.97, "test"))
        cursor = start + len(value)
    return spans


def synthetic(text, spans, **kwargs):
    return replace(text, spans, mode="synthetic", patient_id="patient-a", secret=SECRET, **kwargs)


def test_redaction_preserves_clinical_data_and_returns_only_input_offsets():
    text = "🩺 Jane Doe has HbA1c 9.2%, BP 122/80. Call 801-555-8999."
    spans = spans_for(text, [("Jane Doe", "NAME"), ("801-555-8999", "PHONE")])
    result = replace(text, list(reversed(spans)))
    assert result["text"] == "🩺 [NAME] has HbA1c 9.2%, BP 122/80. Call [PHONE]."
    assert result["entities"][1] == {
        "start": spans[1].start, "end": spans[1].end, "label": "PHONE",
        "replacement": "[PHONE]", "score": 0.97, "source": "test",
    }
    encoded = json.dumps(result)
    assert "Jane Doe" not in encoded
    assert "801-555-8999" not in encoded
    assert "original" not in result["entities"][0]
    assert result["preservation"]["clinical_text_outside_spans_preserved"] is True


def test_dates_preserve_interval_across_leap_day_and_year_boundary():
    text = "Admitted 2024-02-28, discharged 2024-03-01. Prior 2023-12-31."
    spans = spans_for(text, [("2024-02-28", "DATE"), ("2024-03-01", "DATE"), ("2023-12-31", "DATE")])
    result = synthetic(text, spans, date_shift_days=2)
    assert result["text"] == "Admitted 2024-03-01, discharged 2024-03-03. Prior 2024-01-02."
    before = [date.fromisoformat(text[s.start:s.end]) for s in spans]
    after = [date.fromisoformat(entity["replacement"]) for entity in result["entities"]]
    assert [(d - before[0]).days for d in before] == [(d - after[0]).days for d in after]
    assert result["preservation"]["date_intervals_preserved"] is True
    assert result["preservation"]["exact_dates_shifted"] == 3
    assert all("shift_days" not in field for field in result["preservation"])


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("02/29/2024", "03/01/2024"),
        ("2-29-2024", "3-1-2024"),
        ("2024/02/29", "2024/03/01"),
        ("2024.02.29", "2024.03.01"),
        ("13.09.2024", "14.09.2024"),
        ("9.13.2024", "9.14.2024"),
        ("02/9/2024", "02/10/2024"),
        ("2/09/2024", "2/10/2024"),
        ("February 29, 2024", "March 1, 2024"),
        ("Feb. 29 2024", "Mar. 1 2024"),
        ("29 February 2024", "1 March 2024"),
        ("FEB 29, 2024", "MAR 1, 2024"),
        ("feb 29, 2024", "mar 1, 2024"),
        ("February 29th, 2024", "March 1st, 2024"),
        ("February 09, 2024", "February 10, 2024"),
        (" 2024-02-29 ", " 2024-03-01 "),
    ],
)
def test_date_format_is_preserved(original, expected):
    result = synthetic(original, [Span(0, len(original), "DATE")], date_shift_days=1)
    assert result["text"] == expected


def test_mixed_numeric_widths_preserve_explicit_source_padding():
    text = "03/9/2024 and 3/09/2024"
    spans = spans_for(text, [("03/9/2024", "DATE"), ("3/09/2024", "DATE")])
    result = synthetic(text, spans, date_shift_days=-1)
    assert result["text"] == "03/8/2024 and 3/08/2024"
    assert result["preservation"]["date_intervals_preserved"]


def test_dotted_dates_share_one_shift_and_preserve_intervals():
    text = "Admitted 13.09.2024; discharged 15.09.2024."
    spans = spans_for(text, [("13.09.2024", "DATE"), ("15.09.2024", "DATE")])
    result = synthetic(text, spans, date_shift_days=31)
    assert result["text"] == "Admitted 14.10.2024; discharged 16.10.2024."
    assert result["preservation"]["exact_dates_shifted"] == 2
    assert result["preservation"]["date_intervals_preserved"]


@pytest.mark.parametrize("value", ["2024", "February 2024", "02/29", "2024-02", "2024-02-30", "02/29/2023", "yesterday"])
def test_incomplete_unsupported_or_invalid_dates_fail_closed(value):
    result = synthetic(value, [Span(0, len(value), "DATE")], date_shift_days=10)
    assert result["text"] == "[DATE]"
    assert result["warnings"]
    assert all(value not in warning for warning in result["warnings"])
    assert result["preservation"]["unshiftable_dates_redacted"] == 1
    assert result["preservation"]["date_intervals_preserved"] is False


def test_unrepresentable_shifted_date_fails_closed():
    result = synthetic("9999-12-31", [Span(0, 10, "DATE")], date_shift_days=1)
    assert result["text"] == "[DATE]"


def test_default_shift_is_nonzero_patient_consistent_and_bounded():
    original = "2024-03-04"
    spans = [Span(0, 10, "DATE")]
    first = synthetic(original, spans)
    assert first == synthetic(original, spans)
    delta = (date.fromisoformat(first["text"]) - date.fromisoformat(original)).days
    assert 1 <= abs(delta) <= 365
    changes = {
        replace(original, spans, mode="synthetic", secret=SECRET, patient_id=f"patient-{i}")["text"]
        for i in range(8)
    }
    assert len(changes) > 1


def test_names_repeated_casefolded_title_and_partial_mentions():
    text = "Dr. Jane Smith saw JANE   SMITH. Jane called Smith. Smith, Jane returned."
    spans = spans_for(text, [("Dr. Jane Smith", "NAME"), ("JANE   SMITH", "NAME"), ("Jane", "NAME"), ("Smith", "NAME"), ("Smith, Jane", "NAME")])
    result = synthetic(text, spans)
    values = [entity["replacement"] for entity in result["entities"]]
    assert values[0] == "Dr. " + values[1]
    assert values[1] == values[4]
    assert values[2] == values[1].split()[0]
    assert values[3] == values[1].split()[-1]
    assert values[2] != "Jane"
    assert values[3] != "Smith"


def test_ambiguous_partial_names_do_not_link_two_people():
    text = "Jane Smith met Jane Brown. Jane called."
    spans = spans_for(text, [("Jane Smith", "NAME"), ("Jane Brown", "NAME"), ("Jane", "NAME")])
    result = synthetic(text, spans)
    assert len(result["entities"][2]["replacement"].split()) == 2


def test_patient_and_secret_scope_identity():
    text = "Jane Smith"
    spans = [Span(0, len(text), "NAME")]
    same = synthetic(text, spans)["text"]
    assert same == synthetic(text, spans)["text"]
    outputs = {
        replace(text, spans, mode="synthetic", patient_id=f"patient-{i}", secret=SECRET)["text"]
        for i in range(12)
    }
    assert len(outputs) > 1
    keyed = {
        replace(text, spans, mode="synthetic", patient_id="patient-a", secret=bytes([i]) * 32)["text"]
        for i in range(12)
    }
    assert len(keyed) > 1


@pytest.mark.parametrize("age", ["90", "95 years old", "102-year-old", "90+"])
def test_older_ages_are_generalized(age):
    spans = [Span(0, len(age), "AGE")]
    assert replace(age, spans)["text"] == "[AGE_90_PLUS]"
    assert synthetic(age, spans)["text"] == "90+"


def test_normal_ages_can_retain_clinical_value_in_synthetic_mode():
    assert synthetic("52 years old", [Span(0, 12, "AGE")])["text"] == "52 years old"


def test_contact_and_identifier_replacements_are_fictional():
    values = [
        ("801-555-8999", "PHONE"), ("jane@example.org", "EMAIL"),
        ("123 Main St", "ADDRESS"), ("123-45-6789", "SSN"),
        ("1234567", "MRN"), ("ACCT123", "ACCOUNT"), ("EMP123", "ID"),
        ("https://patient.example.com", "URL"), ("172.16.1.1", "IP"),
        ("Salt Lake City", "LOCATION"), ("Regional Hospital", "FACILITY"),
    ]
    text = " | ".join(original for original, _ in values)
    result = synthetic(text, spans_for(text, values))
    replacements = {entity["label"]: entity["replacement"] for entity in result["entities"]}
    assert re.fullmatch(r"\(202\) 555-01\d{2}", replacements["PHONE"])
    assert replacements["EMAIL"].endswith("@example.invalid")
    assert replacements["SSN"].startswith("000-")
    assert "Synthetic Way" in replacements["ADDRESS"]
    assert replacements["URL"].endswith(".invalid")
    assert replacements["IP"].startswith("192.0.2.")
    for original, label in values:
        assert replacements[label] != original
        assert original not in json.dumps(result)
    assert "patient-a" not in json.dumps(result)
    assert SECRET.hex() not in json.dumps(result)


def test_unknown_labels_fall_back_to_safe_redaction():
    span = SimpleNamespace(start=0, end=6, label="DEVICE", score=1.0, source="test")
    assert synthetic("serial", [span])["text"] == "[DEVICE]"
    unsafe = SimpleNamespace(start=0, end=6, label="John Doe", score=1.0, source="test")
    result = synthetic("serial", [unsafe])
    assert result["text"] == "[PHI]"
    assert result["entities"][0]["label"] == "PHI"


@pytest.mark.parametrize("secret", [None, b"", b"short", "short", 7])
def test_synthetic_mode_requires_strong_application_secret(secret):
    with pytest.raises(ValueError, match="secret"):
        replace("", [], mode="synthetic", patient_id="patient-a", secret=secret)


@pytest.mark.parametrize("patient", [None, "", "  ", 1])
def test_synthetic_mode_requires_patient_anchor(patient):
    with pytest.raises(ValueError, match="patient_id"):
        replace("", [], mode="synthetic", patient_id=patient, secret=SECRET)


@pytest.mark.parametrize("offset", [0, 3651, -3651, True, 1.5, "1"])
def test_explicit_offset_must_be_nonzero_bounded_integer(offset):
    with pytest.raises(ValueError, match="date_shift_days"):
        synthetic("", [], date_shift_days=offset)


def test_invalid_overlapping_duplicate_and_out_of_range_spans_are_rejected():
    for spans in ([Span(0, 2, "NAME"), Span(1, 3, "NAME")], [Span(0, 3, "NAME"), Span(0, 3, "NAME")], [Span(0, 4, "NAME")]):
        with pytest.raises(ValueError, match="nonoverlapping"):
            replace("abc", spans)


def test_adjacent_spans_and_empty_inputs_are_safe():
    assert replace("Jane123", [Span(0, 4, "NAME"), Span(4, 7, "MRN")])["text"] == "[NAME][MRN]"
    assert replace("No PHI: glucose 123 mg/dL.", [])["text"] == "No PHI: glucose 123 mg/dL."
    assert replace("", [])["text"] == ""


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        replace("", [], mode="randomize")


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("14/07/2007", "15/07/2007"),
        ("31-12-2023", "01-01-2024"),
        ("29/2/2024", "1/3/2024"),
        ("13-May-2002", "14-May-2002"),
        ("May-13-2002", "May-14-2002"),
        ("29-Feb-2024", "01-Mar-2024"),
        ("FEB-29-2024", "MAR-01-2024"),
        ("May 13,2002", "May 14,2002"),
        ("2024-02-29T23:45:00Z", "2024-03-01T23:45:00Z"),
        ("2024-02-29T23:45:00.123456+05:30", "2024-03-01T23:45:00.123456+05:30"),
        ("2024-02-29 08:30", "2024-03-01 08:30"),
        ("2024-02-29t08:30:00z", "2024-03-01t08:30:00z"),
        ("2024-02-29T08:30:00-0700", "2024-03-01T08:30:00-0700"),
    ],
)
def test_additional_common_date_formats_preserve_format_and_time(original, expected):
    result = synthetic(original, [Span(0, len(original), "DATE")], date_shift_days=1)
    assert result["text"] == expected
    assert result["preservation"]["date_intervals_preserved"] is True


@pytest.mark.parametrize("value", ["2024-02-29T25:30:00Z", "2024-02-29T08:61:00Z", "2024-02-29T08:30:61Z", "2024-02-29T08:30:00+25:00", "2024-02-29T08:30:00+05:61"])
def test_invalid_time_or_timezone_fails_closed_without_phi_in_warning(value):
    result = synthetic(value, [Span(0, len(value), "DATE")], date_shift_days=1)
    assert result["text"] == "[DATE]"
    assert result["warnings"]
    assert value not in " ".join(result["warnings"])


def test_ambiguous_numeric_date_remains_us_convention_with_warning():
    result = synthetic("04/07/2007", [Span(0, 10, "DATE")], date_shift_days=1)
    assert result["text"] == "04/08/2007"
    assert "month/day/year" in result["warnings"][0]
    assert "04/07/2007" not in result["warnings"][0]


def test_equivalent_mixed_formats_share_one_patient_offset():
    values = [
        "2002-05-13", "13/05/2002", "13-May-2002", "May-13-2002",
        "May 13, 2002", "2002-05-13T08:30:00+05:30",
    ]
    text = " | ".join(values)
    spans = spans_for(text, [(value, "DATE") for value in values])
    result = synthetic(text, spans, date_shift_days=20)
    assert [entity["replacement"] for entity in result["entities"]] == [
        "2002-06-02", "02/06/2002", "02-Jun-2002", "Jun-02-2002",
        "June 2, 2002", "2002-06-02T08:30:00+05:30",
    ]
    assert result["preservation"]["exact_dates_shifted"] == 6
    assert result["preservation"]["date_intervals_preserved"] is True


def test_mixed_date_datetime_intervals_across_leap_day_are_preserved():
    text = "Admit 2024-02-28T23:15:30-07:00; follow up 14/03/2024; resolved 15-Mar-2024."
    values = ["2024-02-28T23:15:30-07:00", "14/03/2024", "15-Mar-2024"]
    result = synthetic(text, spans_for(text, [(value, "DATE") for value in values]), date_shift_days=2)
    assert [entity["replacement"] for entity in result["entities"]] == [
        "2024-03-01T23:15:30-07:00", "16/03/2024", "17-Mar-2024",
    ]
    assert (date(2024, 3, 16) - date(2024, 3, 1)).days == (date(2024, 3, 14) - date(2024, 2, 28)).days
    assert (date(2024, 3, 17) - date(2024, 3, 16)).days == 1


def test_patient_anchor_is_exact_case_sensitive_and_preserves_whitespace():
    text = "MR12345678"
    spans = [Span(0, len(text), "MRN")]
    anchors = ["AB12", "ab12", " AB12", "AB12 ", "A  B12", "A B12"]
    results = [
        replace(text, spans, mode="synthetic", patient_id=anchor, secret=SECRET)["text"]
        for anchor in anchors
    ]
    assert len(set(results)) == len(anchors)
    assert results == [
        replace(text, spans, mode="synthetic", patient_id=anchor, secret=SECRET)["text"]
        for anchor in anchors
    ]


@pytest.mark.parametrize(
    ("original", "setting"),
    [
        ("Pinecrest Hospice", "Hospice"),
        ("Alder Rehabilitation Hospital", "Rehabilitation Hospital"),
        ("Westbrook Children's Clinic", "Pediatric Clinic"),
        ("Westbrook Children’s Hospital", "Pediatric Hospital"),
        ("Rockwell Infusion Centre", "Infusion Center"),
        ("Brookhaven Dialysis Unit", "Dialysis Unit"),
        ("Elmwood Cardiology Practice", "Cardiology Practice"),
        ("Oakwood Family Clinic", "Family Clinic"),
        ("Willow Psychiatric Hospital", "Psychiatric Hospital"),
        ("Northcove Oncology Medical Center", "Oncology Medical Center"),
        ("Rosewater Regional Hospital", "Hospital"),
        ("Unknown Identifying Words", "Medical Center"),
    ],
)
def test_synthetic_facility_preserves_only_whitelisted_generic_setting(original, setting):
    spans = [Span(0, len(original), "FACILITY")]
    result = synthetic(original, spans)
    assert re.fullmatch(rf"Synthetic {re.escape(setting)} [A-F0-9]{{6}}", result["text"])
    assert result == synthetic(original, spans)
    assert original not in json.dumps(result)
    assert original.split()[0] not in result["text"]


def test_facility_whitelist_does_not_copy_unknown_embedded_specialty_names():
    original = "Infusionville Rehabilitationford Clinic"
    result = synthetic(original, [Span(0, len(original), "FACILITY")])
    assert re.fullmatch(r"Synthetic Clinic [A-F0-9]{6}", result["text"])
    assert "Infusionville" not in result["text"]
    assert "Rehabilitationford" not in result["text"]
