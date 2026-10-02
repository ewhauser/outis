import json

import pytest

from clinical_deid.challenge import FRAGMENTS, SCENARIOS, generate, unmark, write_corpus
from clinical_deid.evaluate import evaluate, load_verified_corpus, wilson_lower
from clinical_deid.types import ENTITY_TYPES, Span


def row(text="Zoë has fever", start=0, end=3, label="NAME"):
    return {"id": "1", "text": text, "spans": [{"start": start, "end": end, "label": label}], "family": "test"}


def test_exact_typed_span_and_character_metrics_distinguish_partial_leaks():
    result = evaluate([row()], lambda text: [Span(0, 2, "NAME")], minimum_label_support=1)
    assert result["typed_span"]["recall"] == 0
    assert result["typed_span"]["precision"] == 0
    assert result["character"]["recall"] == pytest.approx(2 / 3)
    assert result["document_leak_free"]["rate"] == 0
    assert not result["goal"]["passed"]


def test_type_error_redacts_characters_but_is_not_exact_typed_match():
    result = evaluate([row()], lambda text: [Span(0, 3, "LOCATION")])
    assert result["typed_span"]["f1"] == 0
    assert result["character"]["recall"] == 1
    assert result["typed_character"]["recall"] == 0
    assert result["document_leak_free"]["rate"] == 1


def test_negative_documents_and_duplicate_predictions():
    rows = [{"id": "negative", "text": "5 mg for 14 days", "spans": [], "family": "negative"}, row()]
    result = evaluate(rows, lambda text: [Span(0, 3, "NAME"), Span(0, 3, "NAME")])
    assert result["typed_span"]["true_positives"] == 1
    assert result["typed_span"]["false_positives"] == 1
    assert result["negative_documents"]["clean_rate"] == 0
    assert result["duplicate_predictions_removed"] == 2


def test_unicode_annotation_offsets_are_character_offsets():
    text, spans = unmark("Hello [[NAME|Zoë Núñez]], DOB [[DATE|2024-10-03]].")
    assert text[spans[0]["start"]:spans[0]["end"]] == "Zoë Núñez"
    assert text[spans[1]["start"]:spans[1]["end"]] == "2024-10-03"


def test_synthetic_challenge_is_reproducible_and_has_all_labels():
    rows = generate(1000, 42)
    assert rows == generate(1000, 42)
    assert rows != generate(1000, 43)
    assert {s["label"] for r in rows for s in r["spans"]} == set(ENTITY_TYPES)
    assert sum(not r["spans"] for r in rows) > 0
    assert all(r["split"] == "test" for r in rows)
    for document in rows:
        for span in document["spans"]:
            assert 0 <= span["start"] < span["end"] <= len(document["text"])


def test_final_context_templates_are_disjoint_from_tuning_suites():
    final = {phrase for variants in FRAGMENTS["final"].values() for phrase in variants}
    for suite in ("development", "adversarial"):
        tuning = {phrase for variants in FRAGMENTS[suite].values() for phrase in variants}
        assert not final & tuning
        assert not {item[0] for item in SCENARIOS["final"]} & {item[0] for item in SCENARIOS[suite]}


def test_frozen_manifest_rejects_tampering_and_overwrite(tmp_path):
    path = tmp_path / "challenge.jsonl"
    rows = generate(5, 2)
    manifest = write_corpus(path, rows, 2, "final")
    loaded, actual = load_verified_corpus(path)
    assert loaded == rows
    assert actual == manifest
    with pytest.raises(FileExistsError):
        write_corpus(path, generate(5, 3), 3, "final")
    with path.open("a") as handle:
        handle.write(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_verified_corpus(path)


def test_wilson_interval_and_empty_evaluation():
    assert wilson_lower(1000, 1000) == pytest.approx(.9961732415)
    assert wilson_lower(0, 0) == 0
    with pytest.raises(ValueError, match="empty"):
        evaluate([], lambda text: [])
    with pytest.raises(ValueError, match="Invalid span"):
        evaluate([row()], lambda text: [{"start": 0, "end": 100, "label": "NAME"}])


def test_model_and_hybrid_reports_are_explicitly_separate():
    class Detector:
        def detect(self, text):
            return [Span(0, 3, "NAME")]
    assert evaluate([row()], Detector(), backend="model")["backend"] == "model"
    assert evaluate([row()], Detector(), backend="hybrid")["backend"] == "hybrid"


def test_batch_predictions_use_fast_path_and_validate_batch_size():
    class Detector:
        def detect(self, text):
            raise AssertionError("Scalar detect should not be used")

        def detect_batch(self, texts):
            return [[Span(0, 3, "NAME")] for text in texts]
    result = evaluate([row()], Detector(), backend="model")
    assert result["typed_span"]["f1"] == 1
    with pytest.raises(ValueError, match="batch_size"):
        evaluate([row()], Detector(), batch_size=0)


def test_duplicate_corpus_identifiers_are_rejected():
    with pytest.raises(ValueError, match="Duplicate corpus"):
        evaluate([row(), row()], lambda text: [])
def test_detector_source_manifest_includes_changed_transitive_helpers(tmp_path, monkeypatch):
    import clinical_deid.evaluate as evaluation

    for name, text in {
        "model.py": "from .types import Span\n",
        "rules.py": "from .helper import refine\n",
        "helper.py": "from .types import Span\ndef refine(): pass\n",
        "types.py": "class Span: pass\n",
    }.items():
        (tmp_path / name).write_text(text)
    monkeypatch.setattr(evaluation, "__file__", str(tmp_path / "evaluate.py"))
    before = evaluation.detector_source_manifest()
    assert set(before) == {"model.py", "rules.py", "helper.py", "types.py"}
    (tmp_path / "helper.py").write_text("def refine(): return 1\n")
    after = evaluation.detector_source_manifest()
    assert before["helper.py"] != after["helper.py"]
