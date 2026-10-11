import json
import re
import unicodedata
from datetime import date
from types import SimpleNamespace

import pytest

from outis.replacement.surrogates import replace
from outis.types import Span

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
    return replace(
        text, spans, mode="synthetic", patient_id="patient-a", secret=SECRET, **kwargs
    )


@pytest.mark.parametrize(
    "value,kind",
    [
        ("Exampleford Hospice House", "Hospice House"),
        ("Exampleford Care Lodge", "Care Residence"),
        ("Exampleford Nursing Home", "Nursing Home"),
        ("Exampleford Infirmary", "Infirmary"),
        ("Exampleford Infusion Pavilion", "Infusion Pavilion"),
    ],
)
def test_compound_facility_replacement_preserves_public_care_kind(value, kind):
    result = synthetic(value, spans_for(value, [(value, "FACILITY")]))
    assert result["text"].startswith("Synthetic " + kind + " ")
    assert "Exampleford" not in result["text"]


def test_redaction_preserves_clinical_data_and_returns_only_input_offsets():
    text = "🩺 Jane Doe has HbA1c 9.2%, BP 122/80. Call 801-555-8999."
    spans = spans_for(text, [("Jane Doe", "NAME"), ("801-555-8999", "PHONE")])
    result = replace(text, list(reversed(spans)))
    assert result["text"] == "🩺 [NAME] has HbA1c 9.2%, BP 122/80. Call [PHONE]."
    assert result["entities"][1] == {
        "start": spans[1].start,
        "end": spans[1].end,
        "label": "PHONE",
        "replacement": "[PHONE]",
        "score": 0.97,
        "source": "test",
    }
    encoded = json.dumps(result)
    assert "Jane Doe" not in encoded
    assert "801-555-8999" not in encoded
    assert "original" not in result["entities"][0]
    assert result["preservation"]["clinical_text_outside_spans_preserved"] is True


def test_dates_preserve_interval_across_leap_day_and_year_boundary():
    text = "Admitted 2024-02-28, discharged 2024-03-01. Prior 2023-12-31."
    spans = spans_for(
        text, [("2024-02-28", "DATE"), ("2024-03-01", "DATE"), ("2023-12-31", "DATE")]
    )
    result = synthetic(text, spans, date_shift_days=2)
    assert (
        result["text"]
        == "Admitted 2024-03-01, discharged 2024-03-03. Prior 2024-01-02."
    )
    before = [date.fromisoformat(text[s.start : s.end]) for s in spans]
    after = [date.fromisoformat(entity["replacement"]) for entity in result["entities"]]
    assert [(d - before[0]).days for d in before] == [
        (d - after[0]).days for d in after
    ]
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


@pytest.mark.parametrize("separator", ["/", ".", "-"])
def test_shifted_day_first_dates_remain_unambiguous_and_preserve_intervals(separator):
    values = [
        separator.join(parts)
        for parts in [("13", "09", "2024"), ("14", "09", "2024"), ("2024", "09", "15")]
    ]
    text = (
        "Admit "
        + values[0]
        + "; review "
        + values[1]
        + "; discharge "
        + values[2]
        + "."
    )
    result = synthetic(
        text, spans_for(text, [(value, "DATE") for value in values]), date_shift_days=20
    )
    replacements = [entity["replacement"] for entity in result["entities"]]
    assert replacements == [
        separator.join(("2024", "10", day)) for day in ("03", "04", "05")
    ]
    # Read the emitted year-first dates independently of the replacement parser.
    dates = [
        date(*(int(part) for part in value.split(separator))) for value in replacements
    ]
    assert [(value - dates[0]).days for value in dates] == [0, 1, 2]
    assert (
        result["text"]
        == "Admit "
        + replacements[0]
        + "; review "
        + replacements[1]
        + "; discharge "
        + replacements[2]
        + "."
    )
    assert result["preservation"]["date_intervals_preserved"] is True
    assert result["preservation"]["exact_dates_shifted"] == 3
    assert len(result["warnings"]) == 2
    assert all("year/month/day" in warning for warning in result["warnings"])
    assert not any(value in " ".join(result["warnings"]) for value in values)


def test_day_first_backward_shift_changes_order_only_when_meaning_would_be_ambiguous():
    values = ["13/09/2024", "14/09/2024", "10/13/2024"]
    text = " | ".join(values)
    result = synthetic(
        text, spans_for(text, [(value, "DATE") for value in values]), date_shift_days=-1
    )
    assert result["text"] == "2024/09/12 | 13/09/2024 | 10/12/2024"
    assert len(result["warnings"]) == 1
    assert "year/month/day" in result["warnings"][0]
    assert result["preservation"]["date_intervals_preserved"] is True


@pytest.mark.parametrize(
    "value",
    [
        "2024",
        "February 2024",
        "02/29",
        "2024-02",
        "2024-02-30",
        "02/29/2023",
        "yesterday",
    ],
)
def test_incomplete_unsupported_or_invalid_dates_fail_closed(value):
    result = synthetic(value, [Span(0, len(value), "DATE")], date_shift_days=10)
    assert result["text"] == "[DATE]"
    assert result["warnings"]
    assert all(value not in warning for warning in result["warnings"])
    assert result["preservation"]["unshiftable_dates_redacted"] == 1
    assert result["preservation"]["date_intervals_preserved"] is False


@pytest.mark.parametrize(
    ("original", "shift", "expected"),
    [
        ("03/02/2024 to 03/09/2024", 1, "03/03/2024 to 03/10/2024"),
        ("03/02/2024-03/09/2024", 1, "03/03/2024-03/10/2024"),
        ("2024-03-02 - 2024-03-09", 1, "2024-03-03 - 2024-03-10"),
        ("Jan 5, 2024 until Feb 1, 2024", 1, "Jan 6, 2024 until Feb 2, 2024"),
        ("March 2-9, 2024", 1, "March 3-10, 2024"),
        ("Mar 2 – 9 2024", 1, "Mar 3 – 10 2024"),
        ("March 2nd through 9th, 2024", 1, "March 3rd through 10th, 2024"),
        ("2-9 March 2024", 1, "3-10 March 2024"),
        ("March 28 - April 4, 2024", 1, "March 29 - April 5, 2024"),
        ("28 March - 4 April 2024", 1, "29 March - 5 April 2024"),
        # A compressed range stays compressed when both shifted endpoints share
        # a month; otherwise each endpoint is written out so it stays unambiguous.
        ("March 25-30, 2024", 7, "April 1-6, 2024"),
        ("March 25-30, 2024", 4, "March 29-April 3, 2024"),
        ("25-30 March 2024", 4, "29 March-3 April 2024"),
        ("Dec 20-27, 2024", 7, "Dec 27, 2024-Jan 3, 2025"),
    ],
)
def test_date_range_span_shifts_both_endpoints(original, shift, expected):
    result = synthetic(
        original, [Span(0, len(original), "DATE")], date_shift_days=shift
    )
    assert result["text"] == expected
    assert result["preservation"]["exact_dates_shifted"] == 1
    assert result["preservation"]["date_intervals_preserved"] is True


def test_date_range_and_single_dates_share_one_patient_shift():
    text = "Admitted 03/02/2024; rehab March 4-9, 2024; seen 03/20/2024 to 03/22/2024."
    spans = spans_for(
        text,
        [
            ("03/02/2024", "DATE"),
            ("March 4-9, 2024", "DATE"),
            ("03/20/2024 to 03/22/2024", "DATE"),
        ],
    )
    result = synthetic(text, spans, date_shift_days=-30)
    assert result["text"] == (
        "Admitted 02/01/2024; rehab February 3-8, 2024; seen 02/19/2024 to 02/21/2024."
    )
    assert result["preservation"]["exact_dates_shifted"] == 3
    assert result["preservation"]["date_intervals_preserved"] is True


def test_date_range_uses_the_patient_offset_without_an_explicit_shift():
    single, ranged = "03/02/2024", "03/02/2024 to 03/09/2024"
    alone = synthetic(single, [Span(0, len(single), "DATE")])["text"]
    both = synthetic(ranged, [Span(0, len(ranged), "DATE")])["text"]
    start, end = both.split(" to ")
    assert start == alone
    parse = [
        date(int(y), int(m), int(d)) for m, d, y in (v.split("/") for v in (start, end))
    ]
    assert (parse[1] - parse[0]).days == 7


@pytest.mark.parametrize(
    "value",
    [
        "December 28 - January 4, 2024",
        "March 30-February 31, 2024",
        "03/02/2024 to soon",
        "2019-2021",
    ],
)
def test_unshiftable_date_ranges_fail_closed(value):
    result = synthetic(value, [Span(0, len(value), "DATE")], date_shift_days=10)
    assert result["text"] == "[DATE]"
    assert result["preservation"]["unshiftable_dates_redacted"] == 1


@pytest.mark.parametrize(
    ("original", "shift", "expected"),
    [
        ("3/20/26", 7, "3/27/26"),
        ("03/20/26", 7, "03/27/26"),
        ("1-5-24", 10, "1-15-24"),
        ("12/30/99", 5, "01/04/00"),
        ("02/27/00", 1, "02/28/00"),
        ("2/29/96", 1, "3/1/96"),
        ("25/03/26", 3, "28/03/26"),
        ("3/20/26 to 3/27/26", -30, "2/18/26 to 2/25/26"),
    ],
)
def test_two_digit_year_dates_shift_and_keep_two_digits(original, shift, expected):
    result = synthetic(
        original, [Span(0, len(original), "DATE")], date_shift_days=shift
    )
    assert result["text"] == expected
    assert result["preservation"]["date_intervals_preserved"] is True


@pytest.mark.parametrize(
    ("value", "shift"),
    [
        # 1900 is not a leap year and 2000 is, so the century changes the result.
        ("02/28/00", 1),
        # Year-first output would be unreadable with a two-digit year.
        ("25/03/26", -20),
    ],
)
def test_ambiguous_two_digit_year_shifts_fail_closed(value, shift):
    result = synthetic(value, [Span(0, len(value), "DATE")], date_shift_days=shift)
    assert result["text"] == "[DATE]"
    assert result["preservation"]["unshiftable_dates_redacted"] == 1


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
        replace(
            original, spans, mode="synthetic", secret=SECRET, patient_id=f"patient-{i}"
        )["text"]
        for i in range(8)
    }
    assert len(changes) > 1


def test_names_repeated_casefolded_title_and_partial_mentions():
    text = "Dr. Jane Smith saw JANE   SMITH. Jane called Smith. Smith, Jane returned."
    spans = spans_for(
        text,
        [
            ("Dr. Jane Smith", "NAME"),
            ("JANE   SMITH", "NAME"),
            ("Jane", "NAME"),
            ("Smith", "NAME"),
            ("Smith, Jane", "NAME"),
        ],
    )
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
    spans = spans_for(
        text, [("Jane Smith", "NAME"), ("Jane Brown", "NAME"), ("Jane", "NAME")]
    )
    result = synthetic(text, spans)
    assert len(result["entities"][2]["replacement"].split()) == 2


def test_unicode_given_and_particle_family_aliases_share_one_full_identity():
    values = ["Åsa de Valen", "Åsa", "de Valen", "Dr. Åsa", "de Valen, Åsa"]
    text = " | ".join(values)
    result = synthetic(text, spans_for(text, [(value, "NAME") for value in values]))
    full, first, family, titled, reversed_full = [
        entity["replacement"] for entity in result["entities"]
    ]
    assert first == full.split()[0]
    assert family == full.split()[-1]
    assert titled == "Dr. " + first
    assert reversed_full == full
    assert all(value not in json.dumps(result, ensure_ascii=False) for value in values)


@pytest.mark.parametrize(
    "full,first,family",
    [
        ("Éva É. O’Neri-Rios Jr.", "Éva", "O’Neri-Rios"),
        ("Åsa de Valen", "Åsa", "de Valen"),
        ("médico van pélier", "médico", "van pélier"),
        ("Σοφία Παππά", "Σοφία", "Παππά"),
    ],
)
def test_nfc_nfd_equivalence_uses_original_offsets_for_full_and_reduced_names(
    full, first, family
):
    nfd_first = unicodedata.normalize("NFD", first)
    nfd_family = unicodedata.normalize("NFD", family)
    values = [full, nfd_first, nfd_family]
    text = "🙂 Clinical note: " + " | ".join(values) + " | Na 139 mmol/L; pain 3/10."
    spans = spans_for(text, [(value, "NAME") for value in values])
    result = synthetic(text, spans)
    replacements = [entity["replacement"] for entity in result["entities"]]
    assert replacements[1] == replacements[0].split()[0]
    assert replacements[2] == replacements[0].split()[-1]
    assert (
        result["text"]
        == "🙂 Clinical note: "
        + " | ".join(replacements)
        + " | Na 139 mmol/L; pain 3/10."
    )
    assert [(entity["start"], entity["end"]) for entity in result["entities"]] == [
        (span.start, span.end) for span in spans
    ]
    assert (
        synthetic(full, [Span(0, len(full), "NAME")])["text"]
        == synthetic(
            unicodedata.normalize("NFD", full),
            [Span(0, len(unicodedata.normalize("NFD", full)), "NAME")],
        )["text"]
    )
    assert full not in json.dumps(result, ensure_ascii=False)
    assert nfd_first not in json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize("suffix", ["Jr.", "Sr.", "II", "III"])
def test_generational_suffix_full_name_orders_link_without_suffix_alias(suffix):
    values = [
        f"Elian Vale {suffix}",
        f"Elian Vale, {suffix}",
        f"Vale {suffix}, Elian",
        f"Vale, Elian {suffix}",
        "Elian",
        "Vale",
        "Dr. Elian",
    ]
    text = " | ".join(values)
    result = synthetic(text, spans_for(text, [(value, "NAME") for value in values]))
    replacements = [entity["replacement"] for entity in result["entities"]]
    assert len(set(replacements[:4])) == 1
    assert replacements[4] == replacements[0].split()[0]
    assert replacements[5] == replacements[0].split()[-1]
    assert replacements[6] == "Dr. " + replacements[4]
    assert all(value not in json.dumps(result) for value in values)


def test_generationally_distinct_people_do_not_share_reduced_alias_mapping():
    values = ["Elian Vale Jr.", "Elian Vale Sr.", "Elian", "Vale"]
    text = " | ".join(values)
    result = synthetic(text, spans_for(text, [(value, "NAME") for value in values]))
    replacements = [entity["replacement"] for entity in result["entities"]]
    assert replacements[0] != replacements[1]
    assert len(replacements[2].split()) == len(replacements[3].split()) == 2


@pytest.mark.parametrize("suffix", ["Jr", "Sr", "II", "III", "IV", "V", "VI"])
def test_optional_suffix_period_does_not_create_competing_alias_owners(suffix):
    values = [
        f"Elian Vale {suffix}",
        f"Elian Vale {suffix}.",
        f"Elian Vale, {suffix}",
        f"Vale {suffix}., Elian",
        f"Vale, Elian {suffix}.",
        "Elian",
        "Vale",
    ]
    text = " | ".join(values)
    result = synthetic(text, spans_for(text, [(value, "NAME") for value in values]))
    replacements = [entity["replacement"] for entity in result["entities"]]
    assert len(set(replacements[:5])) == 1
    assert replacements[5] == replacements[0].split()[0]
    assert replacements[6] == replacements[0].split()[-1]
    # Full identities also remain consistent without a note-local alias map.
    assert (
        len(
            {
                synthetic(value, [Span(0, len(value), "NAME")])["text"]
                for value in values[:5]
            }
        )
        == 1
    )


def test_optional_period_normalization_keeps_distinct_generations_ambiguous():
    values = [
        "Elian Vale Jr",
        "Elian Vale Jr.",
        "Elian Vale Sr",
        "Elian Vale Sr.",
        "Vale",
    ]
    text = " | ".join(values)
    result = synthetic(text, spans_for(text, [(value, "NAME") for value in values]))
    replacements = [entity["replacement"] for entity in result["entities"]]
    assert replacements[0] == replacements[1]
    assert replacements[2] == replacements[3]
    assert replacements[0] != replacements[2]
    assert len(replacements[4].split()) == 2


@pytest.mark.parametrize("apostrophe", ["'", "’"])
@pytest.mark.parametrize("hyphen", ["-", "‐", "‑"])
def test_unicode_name_punctuation_variants_share_full_and_reduced_identity(
    apostrophe, hyphen
):
    variant = f"O{apostrophe}Neri{hyphen}Rios"
    values = [
        "Åsa O'Neri-Rios",
        f"Åsa {variant}",
        f"{variant}, Åsa",
        unicodedata.normalize("NFD", f"Åsa {variant}"),
        variant,
        "O'Neri-Rios",
        "Dr. Åsa",
    ]
    text = "🙂 Review: " + " | ".join(values) + " | K 4.1 mmol/L; pain 3/10."
    spans = spans_for(text, [(value, "NAME") for value in values])
    result = synthetic(text, spans)
    replacements = [entity["replacement"] for entity in result["entities"]]
    assert len(set(replacements[:4])) == 1
    assert replacements[4] == replacements[5] == replacements[0].split()[-1]
    assert replacements[6] == "Dr. " + replacements[0].split()[0]
    assert (
        result["text"]
        == "🙂 Review: " + " | ".join(replacements) + " | K 4.1 mmol/L; pain 3/10."
    )
    assert [(entity["start"], entity["end"]) for entity in result["entities"]] == [
        (span.start, span.end) for span in spans
    ]
    assert (
        len(
            {
                synthetic(value, [Span(0, len(value), "NAME")])["text"]
                for value in values[:4]
            }
        )
        == 1
    )
    assert all(value not in json.dumps(result, ensure_ascii=False) for value in values)


@pytest.mark.parametrize(
    "original,expected",
    [
        ("Elian Vale Jr.", "Parker Orchard"),
        ("Elian Vale Sr.", "Emery Meadow"),
        ("Elian Vale III", "Drew Meadow"),
        ("Asa O'Neri-Rios", "Quinn Juniper"),
    ],
)
def test_conventional_ascii_suffix_and_punctuation_hmac_outputs_stay_exact(
    original, expected
):
    # Pinned before spelling normalization, using the existing test secret.
    assert synthetic(original, [Span(0, len(original), "NAME")])["text"] == expected


def test_name_key_normalization_does_not_change_identifier_or_patient_anchor(
    monkeypatch,
):
    import outis.replacement.synthetic as module

    calls = []
    original_digest = module.keyed_digest
    patient = "Åsa O’Neri‐Rios Jr"
    values = ["O’Neri‐Rios Jr", "O'Neri-Rios Jr."]

    def recording(secret, patient, label, value, salt=0):
        calls.append((patient, label, value))
        return original_digest(secret, patient, label, value, salt)

    monkeypatch.setattr(module, "keyed_digest", recording)
    text = " | ".join(values)
    result = replace(
        text,
        spans_for(text, [(value, "ID") for value in values]),
        mode="synthetic",
        patient_id=patient,
        secret=SECRET,
    )
    assert {value for _, label, value in calls if label == "ID"} == set(values)
    assert all(anchor == patient for anchor, _, _ in calls)
    assert values[0] not in json.dumps(result, ensure_ascii=False)
    assert values[1] not in json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize(
    "full,alias",
    [
        ("Ren Jules Alder", "Ren"),
        ("Ren Jules Alder", "Alder"),
        ("Ren Jules Alder, Jr.", "Alder"),
        ("R. Alder", "Alder"),
        ("Ren, Jules, Alder", "Ren"),
        ("Ren Ren", "Ren"),
    ],
)
def test_ambiguous_multipart_initial_and_shared_component_aliases_are_not_guessed(
    full, alias
):
    values = [full, alias]
    text = " | ".join(values)
    result = synthetic(text, spans_for(text, [(value, "NAME") for value in values]))
    assert len(result["entities"][1]["replacement"].split()) == 2


def test_particle_family_tail_is_not_guessed_from_a_partial_surname():
    values = ["Åsa de Valen", "Valen"]
    text = " | ".join(values)
    result = synthetic(text, spans_for(text, [(value, "NAME") for value in values]))
    assert len(result["entities"][1]["replacement"].split()) == 2


def test_reduced_alias_linking_is_note_local_not_a_cross_note_identity_claim():
    full = "Åsa de Valen"
    alias = "Åsa"
    text = full + " | " + alias
    together = synthetic(text, spans_for(text, [(full, "NAME"), (alias, "NAME")]))
    assert len(together["entities"][1]["replacement"].split()) == 1
    separately = synthetic(alias, [Span(0, len(alias), "NAME")])
    assert len(separately["text"].split()) == 2


def test_existing_ascii_hmac_identity_title_order_case_and_alias_outputs_stay_exact():
    # Pinned before the Unicode/alias repair using the original implementation.
    values = ["Jane Smith", "Smith, Jane", "Dr. Jane Smith", "Jane", "Smith"]
    text = " | ".join(values)
    spans = spans_for(text, [(value, "NAME") for value in values])
    result = replace(
        text,
        spans,
        mode="synthetic",
        patient_id="patient-a",
        secret=b"0123456789abcdef0123456789abcdef",
    )
    assert [entity["replacement"] for entity in result["entities"]] == [
        "Quinn Grove",
        "Quinn Grove",
        "Dr. Quinn Grove",
        "Quinn",
        "Grove",
    ]
    assert (
        synthetic("JANE   SMITH", [Span(0, len("JANE   SMITH"), "NAME")])["text"]
        == synthetic("Jane Smith", [Span(0, len("Jane Smith"), "NAME")])["text"]
    )


def test_alias_component_cannot_remain_private_surname_unchanged_after_suffix(
    monkeypatch,
):
    import outis.replacement.synthetic as module

    monkeypatch.setattr(module, "_FIRST_NAMES", ("Fictional",))
    monkeypatch.setattr(module, "_LAST_NAMES", ("Vale", "FictionalSurname"))
    digest = bytes(32)
    original = module.keyed_digest

    def controlled(secret, patient, label, value, salt=0):
        if label == "NAME":
            return digest if salt == 0 else bytes(4) + b"\x00\x00\x00\x01" + bytes(24)
        return original(secret, patient, label, value, salt)

    monkeypatch.setattr(module, "keyed_digest", controlled)
    values = ["Elian Vale Jr.", "Vale"]
    text = " | ".join(values)
    result = synthetic(text, spans_for(text, [(value, "NAME") for value in values]))
    assert [entity["replacement"] for entity in result["entities"]] == [
        "Fictional FictionalSurname",
        "FictionalSurname",
    ]
    assert "Vale" not in json.dumps(result)


@pytest.mark.parametrize(
    "original,setting",
    [
        ("Zephrel Wound Pavilion", "Wound Pavilion"),
        ("Zephrel Wound Care Pavilion", "Wound Care Pavilion"),
        ("Zephrel Wound Care Clinic", "Wound Care Clinic"),
        ("Woundfield Carepoint Pavilion", "Medical Center"),
    ],
)
def test_wound_setting_replacement_uses_only_fixed_public_care_words(original, setting):
    result = synthetic(original, [Span(0, len(original), "FACILITY")])
    assert re.fullmatch(
        rf"Synthetic {re.escape(setting)} [A-F0-9]{{6}}", result["text"]
    )
    assert original not in json.dumps(result)
    assert original.split()[0] not in result["text"]


def test_patient_and_secret_scope_identity():
    text = "Jane Smith"
    spans = [Span(0, len(text), "NAME")]
    same = synthetic(text, spans)["text"]
    assert same == synthetic(text, spans)["text"]
    outputs = {
        replace(
            text, spans, mode="synthetic", patient_id=f"patient-{i}", secret=SECRET
        )["text"]
        for i in range(12)
    }
    assert len(outputs) > 1
    keyed = {
        replace(
            text,
            spans,
            mode="synthetic",
            patient_id="patient-a",
            secret=bytes([i]) * 32,
        )["text"]
        for i in range(12)
    }
    assert len(keyed) > 1


@pytest.mark.parametrize("age", ["90", "95 years old", "102-year-old", "90+"])
def test_older_ages_are_generalized(age):
    spans = [Span(0, len(age), "AGE")]
    assert replace(age, spans)["text"] == "[AGE_90_PLUS]"
    assert synthetic(age, spans)["text"] == "90+"


def test_ages_89_and_under_that_reach_replacement_are_removed():
    # Ages of 89 and under are kept by the request policy's default (ages="over_89")
    # before replacement; one that reaches replacement was selected for removal.
    assert synthetic("52 years old", [Span(0, 12, "AGE")])["text"] == "[AGE]"
    assert replace("52", [Span(0, 2, "AGE")])["text"] == "[AGE]"


def test_contact_and_identifier_replacements_are_fictional():
    values = [
        ("801-555-8999", "PHONE"),
        ("jane@example.org", "EMAIL"),
        ("123 Main St", "ADDRESS"),
        ("123-45-6789", "SSN"),
        ("1234567", "MRN"),
        ("ACCT123", "ACCOUNT"),
        ("EMP123", "ID"),
        ("https://patient.example.com", "URL"),
        ("172.16.1.1", "IP"),
        ("Salt Lake City", "LOCATION"),
        ("Regional Hospital", "FACILITY"),
    ]
    text = " | ".join(original for original, _ in values)
    result = synthetic(text, spans_for(text, values))
    replacements = {
        entity["label"]: entity["replacement"] for entity in result["entities"]
    }
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


@pytest.mark.parametrize(
    "label,values",
    [
        ("MRN", ["AB12", "ab12", "AB12 ", "AB 12", "AB  12", "ＡＢ12"]),
        ("ACCOUNT", ["AB12", "ab12", "AB12 ", "AB 12", "AB  12", "ＡＢ12"]),
        ("ID", ["AB12", "ab12", "AB12 ", "AB 12", "AB  12", "ＡＢ12"]),
        ("EMAIL", ["A@example.org", "a@example.org", "a@example.org "]),
        (
            "URL",
            [
                "https://example.org/A",
                "https://example.org/a",
                "https://example.org/a ",
            ],
        ),
    ],
)
def test_case_sensitive_identifier_replacements_preserve_exact_identity(label, values):
    text = " | ".join(values)
    spans = spans_for(text, [(value, label) for value in values])
    result = synthetic(text, spans)
    replacements = [entity["replacement"] for entity in result["entities"]]
    assert len(set(replacements)) == len(values)
    assert result["preservation"]["patient_consistent_replacements"] is True
    for value, replacement in zip(values, replacements):
        assert replacement == synthetic(value, [Span(0, len(value), label)])["text"]
    redacted = replace(text, spans)
    assert redacted["text"] == " | ".join(f"[{label}]" for _ in values)


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
    for spans in (
        [Span(0, 2, "NAME"), Span(1, 3, "NAME")],
        [Span(0, 3, "NAME"), Span(0, 3, "NAME")],
        [Span(0, 4, "NAME")],
    ):
        with pytest.raises(ValueError, match="nonoverlapping"):
            replace("abc", spans)


def test_adjacent_spans_and_empty_inputs_are_safe():
    assert (
        replace("Jane123", [Span(0, 4, "NAME"), Span(4, 7, "MRN")])["text"]
        == "[NAME][MRN]"
    )
    assert (
        replace("No PHI: glucose 123 mg/dL.", [])["text"]
        == "No PHI: glucose 123 mg/dL."
    )
    assert replace("", [])["text"] == ""


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        replace("", [], mode="randomize")


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("14/07/2007", "15/07/2007"),
        ("31-12-2023", "01-01-2024"),
        ("29/2/2024", "2024/3/1"),
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


@pytest.mark.parametrize(
    "value",
    [
        "2024-02-29T25:30:00Z",
        "2024-02-29T08:61:00Z",
        "2024-02-29T08:30:61Z",
        "2024-02-29T08:30:00+25:00",
        "2024-02-29T08:30:00+05:61",
    ],
)
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
        "2002-05-13",
        "13/05/2002",
        "13-May-2002",
        "May-13-2002",
        "May 13, 2002",
        "2002-05-13T08:30:00+05:30",
    ]
    text = " | ".join(values)
    spans = spans_for(text, [(value, "DATE") for value in values])
    result = synthetic(text, spans, date_shift_days=20)
    assert [entity["replacement"] for entity in result["entities"]] == [
        "2002-06-02",
        "2002/06/02",
        "02-Jun-2002",
        "Jun-02-2002",
        "June 2, 2002",
        "2002-06-02T08:30:00+05:30",
    ]
    assert result["preservation"]["exact_dates_shifted"] == 6
    assert result["preservation"]["date_intervals_preserved"] is True


def test_mixed_date_datetime_intervals_across_leap_day_are_preserved():
    text = (
        "Admit 2024-02-28T23:15:30-07:00; follow up 14/03/2024; resolved 15-Mar-2024."
    )
    values = ["2024-02-28T23:15:30-07:00", "14/03/2024", "15-Mar-2024"]
    result = synthetic(
        text, spans_for(text, [(value, "DATE") for value in values]), date_shift_days=2
    )
    assert [entity["replacement"] for entity in result["entities"]] == [
        "2024-03-01T23:15:30-07:00",
        "16/03/2024",
        "17-Mar-2024",
    ]
    assert (date(2024, 3, 16) - date(2024, 3, 1)).days == (
        date(2024, 3, 14) - date(2024, 2, 28)
    ).days
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
def test_synthetic_facility_preserves_only_whitelisted_generic_setting(
    original, setting
):
    spans = [Span(0, len(original), "FACILITY")]
    result = synthetic(original, spans)
    assert re.fullmatch(
        rf"Synthetic {re.escape(setting)} [A-F0-9]{{6}}", result["text"]
    )
    assert result == synthetic(original, spans)
    assert original not in json.dumps(result)
    assert original.split()[0] not in result["text"]


def test_facility_whitelist_does_not_copy_unknown_embedded_specialty_names():
    original = "Infusionville Rehabilitationford Clinic"
    result = synthetic(original, [Span(0, len(original), "FACILITY")])
    assert re.fullmatch(r"Synthetic Clinic [A-F0-9]{6}", result["text"])
    assert "Infusionville" not in result["text"]
    assert "Rehabilitationford" not in result["text"]
