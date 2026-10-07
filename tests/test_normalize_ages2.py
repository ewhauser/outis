import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "normalize_ages2", ROOT / "scripts/takeover/normalize_ages2.py"
)
ages2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ages2)


def found(text, spans=()):
    return [text[a:b] for a, b in ages2.find_ages(text, list(spans))]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Sentence-final cued ages v1 skipped.
        (
            "The patient recalls needing fewer rest breaks at age 68. The concern",
            ["68"],
        ),
        ("Her mother died at 72.", ["72"]),
        ("He was diagnosed at the age of 40.", ["40"]),
        # Uncued personal ages, written and spoken.
        ("She is 72 and lives alone.", ["72"]),
        ("My father was 81 when he fell.", ["81"]),
        ("i'm eighty-four, doc", ["eighty-four"]),
        # Family-history forms.
        ("Mother lived to 56. Father died at 68.", ["56", "68"]),
        ("Sister: MI at 66; died at 70.", ["66", "70"]),
        ("breast cancer, age at diagnosis 36.", ["36"]),
        ("Cancer at 52 in a cousin", ["52"]),
    ],
)
def test_adds_ages_v1_missed(text, expected):
    assert found(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Her pulse was 72 and regular.",
        "She is 72 kg today.",
        "Patient is 32 weeks pregnant.",
        "Pain is 7/10.",
        "They were 2 hours late.",
        "He was 1 of 3 siblings.",
        "COPD at 2 L nasal cannula",
        "Stroke at 5 pm yesterday.",
        "Weight 68.5 kg at age 68.5",
        # Public, guideline and hypothetical ages stay unannotated (v1 rule).
        "Screening starts at age 45.",
        "A comparison exercise described a person aged 46.",
        # Over 89 is already annotated by the original banks.
        "She is 95.",
    ],
)
def test_leaves_measurements_counts_and_public_ages_alone(text):
    assert found(text) == []


def test_skips_values_inside_existing_spans():
    text = "Her mother died at 72."
    start = text.index("72")
    assert found(text, [{"start": start, "end": start + 2, "label": "AGE"}]) == []


def test_refuses_sources_that_are_not_v1_derivatives(tmp_path):
    source = tmp_path / "bank.jsonl"
    source.write_text("")
    with pytest.raises(ValueError, match="__allages1"):
        ages2.derive(source)
    blind = tmp_path / "blind_bank__allages1.jsonl"
    blind.write_text("")
    with pytest.raises(ValueError, match="protected"):
        ages2.derive(blind)
