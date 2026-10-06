from clinical_deid.data.annotation_policy import POLICY_VERSION, annotation_errors
from clinical_deid.data.annotation_policy_transcripts import TRANSCRIPT_FIXTURES


def gold(key):
    row = {r["id"]: r for r in TRANSCRIPT_FIXTURES}[key]
    return [(s["label"], row["text"][s["start"] : s["end"]]) for s in row["spans"]]


def test_transcript_fixtures_are_valid():
    for row in TRANSCRIPT_FIXTURES:
        assert row["annotation_policy"] == POLICY_VERSION
        assert not annotation_errors(row["text"], row["spans"])


def test_transcript_decisions():
    assert gold("speaker_label_names") == [("NAME", "Marwick"), ("NAME", "Brannock")]
    assert gold("timestamp_with_date") == [
        ("DATE", "2025-08-24"),
        ("DATE", "Mon, 18 Dec 2023"),
    ]
    assert gold("session_and_access_codes") == [
        ("ID", "VX-4471"),
        ("URL", "https://visit.example.test/r/8812"),
        ("ID", "K7TQ-2RMB"),
    ]
