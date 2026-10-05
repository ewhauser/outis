import json

import pytest

from clinical_deid.data.challenge import (
    FRAGMENTS,
    SCENARIOS,
    generate,
    unmark,
    write_corpus,
)
from clinical_deid.evaluation.evaluate import (
    evaluate,
    load_verified_corpus,
    wilson_lower,
)
from clinical_deid.types import ENTITY_TYPES, Span


def row(text="Zoë has fever", start=0, end=3, label="NAME"):
    return {
        "id": "1",
        "text": text,
        "spans": [{"start": start, "end": end, "label": label}],
        "family": "test",
    }


def test_exact_typed_span_and_character_metrics_distinguish_partial_leaks():
    result = evaluate(
        [row()], lambda text: [Span(0, 2, "NAME")], minimum_label_support=1
    )
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
    rows = [
        {
            "id": "negative",
            "text": "5 mg for 14 days",
            "spans": [],
            "family": "negative",
        },
        row(),
    ]
    result = evaluate(rows, lambda text: [Span(0, 3, "NAME"), Span(0, 3, "NAME")])
    assert result["typed_span"]["true_positives"] == 1
    assert result["typed_span"]["false_positives"] == 1
    assert result["negative_documents"]["clean_rate"] == 0
    assert result["duplicate_predictions_removed"] == 2


def test_unicode_annotation_offsets_are_character_offsets():
    text, spans = unmark("Hello [[NAME|Zoë Núñez]], DOB [[DATE|2024-10-03]].")
    assert text[spans[0]["start"] : spans[0]["end"]] == "Zoë Núñez"
    assert text[spans[1]["start"] : spans[1]["end"]] == "2024-10-03"


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
        tuning = {
            phrase for variants in FRAGMENTS[suite].values() for phrase in variants
        }
        assert not final & tuning
        assert not {item[0] for item in SCENARIOS["final"]} & {
            item[0] for item in SCENARIOS[suite]
        }


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
    assert wilson_lower(1000, 1000) == pytest.approx(0.9961732415)
    assert wilson_lower(0, 0) == 0
    with pytest.raises(ValueError, match="empty"):
        evaluate([], lambda text: [])
    with pytest.raises(ValueError, match="Invalid span"):
        evaluate([row()], lambda text: [{"start": 0, "end": 100, "label": "NAME"}])


def test_reports_record_the_model_only_detector():
    class Detector:
        def detect(self, text):
            return [Span(0, 3, "NAME")]

    assert evaluate([row()], Detector())["backend"] == "model"
    with pytest.raises(TypeError):
        evaluate([row()], Detector(), backend="hybrid")


def test_batch_predictions_use_fast_path_and_validate_batch_size():
    class Detector:
        def detect(self, text):
            raise AssertionError("Scalar detect should not be used")

        def detect_batch(self, texts):
            return [[Span(0, 3, "NAME")] for text in texts]

    result = evaluate([row()], Detector())
    assert result["typed_span"]["f1"] == 1
    with pytest.raises(ValueError, match="batch_size"):
        evaluate([row()], Detector(), batch_size=0)


def test_duplicate_corpus_identifiers_are_rejected():
    with pytest.raises(ValueError, match="Duplicate corpus"):
        evaluate([row(), row()], lambda text: [])


def test_detector_source_manifest_includes_changed_transitive_helpers(
    tmp_path, monkeypatch
):
    import clinical_deid.evaluation.evaluate as evaluation
    import clinical_deid.paths as paths

    for name, text in {
        "detection/hf_encoder.py": "from clinical_deid.detection.helper import refine\n",
        "detection/helper.py": "from clinical_deid.types import Span\ndef refine(): pass\n",
        "types.py": "class Span: pass\n",
    }.items():
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text(text)
    monkeypatch.setattr(paths, "PACKAGE_ROOT", tmp_path)
    before = evaluation.detector_source_manifest()
    assert set(before) == {
        "detection/hf_encoder.py",
        "detection/helper.py",
        "types.py",
    }
    (tmp_path / "detection/helper.py").write_text("def refine(): return 1\n")
    after = evaluation.detector_source_manifest()
    assert before["detection/helper.py"] != after["detection/helper.py"]


def test_phi_bearing_document_metric_excludes_automatically_safe_negatives():
    rows = [row(), {"id": "negative", "text": "Vitals normal", "spans": []}]
    result = evaluate(rows, lambda text: [])
    assert result["document_leak_free"]["rate"] == 0.5
    assert result["phi_bearing_documents"] == {
        "count": 1,
        "leak_free_count": 0,
        "leak_free_rate": 0.0,
        "wilson_lower_95": 0.0,
    }
    assert result["per_label_character_exposure"]["NAME"]["exposed_characters"] == 3


def test_character_exposure_tracks_gold_label_even_with_wrong_prediction_type():
    result = evaluate([row()], lambda text: [Span(0, 2, "LOCATION")])
    name = result["per_label_character_exposure"]["NAME"]
    assert name["support"] == 3
    assert name["covered_characters"] == 2
    assert name["exposed_characters"] == 1
    assert name["recall"] == pytest.approx(2 / 3)
    assert result["per_label_character_exposure"]["LOCATION"]["recall"] is None
    assert result["per_label"]["NAME"]["recall"] == 0


def test_empty_phi_support_is_not_reported_as_perfect_diagnostic_coverage():
    result = evaluate(
        [{"id": "negative", "text": "Vitals normal", "spans": []}], lambda text: []
    )
    assert result["phi_bearing_documents"] == {
        "count": 0,
        "leak_free_count": 0,
        "leak_free_rate": None,
        "wilson_lower_95": None,
    }
    assert all(
        stats["recall"] is None
        for stats in result["per_label_character_exposure"].values()
    )
    # Existing report semantics and admission gates remain backward compatible.
    assert result["negative_documents"]["clean_rate"] == 1
    assert result["schema_version"] == 1
    assert not result["goal"]["passed"]


def test_minimum_support_for_possible_99_percent_wilson_bound_is_381():
    assert wilson_lower(380, 380) < 0.99
    assert wilson_lower(381, 381) >= 0.99


def test_privacy_block_separates_masking_from_exact_typing():
    text = "Ann Lee MRN 12345 seen 2024-01-02 at Elm Clinic. Call 555-0100."
    rows = [
        {
            "id": "p1",
            "text": text,
            "spans": [
                {"start": 0, "end": 7, "label": "NAME"},  # exact hit
                {"start": 12, "end": 17, "label": "MRN"},  # right span, wrong type
                {"start": 23, "end": 33, "label": "DATE"},  # covered by a wider span
                {"start": 37, "end": 47, "label": "FACILITY"},  # only partly covered
                {"start": 54, "end": 62, "label": "PHONE"},  # not detected
            ],
        }
    ]
    predictions = [
        Span(0, 7, "NAME"),
        Span(12, 17, "ID"),
        Span(18, 33, "DATE"),
        Span(37, 40, "FACILITY"),
        Span(8, 11, "ID"),
    ]
    privacy = evaluate(rows, lambda _: predictions)["privacy"]
    assert privacy["direct"]["support"] == 3 and privacy["direct"]["masked"] == 2
    assert privacy["indirect"]["support"] == 2 and privacy["indirect"]["masked"] == 1
    assert privacy["per_label"]["MRN"]["exact_misses"]["wrong_type"] == 1
    assert privacy["per_label"]["DATE"]["exact_misses"]["boundary_differs"] == 1
    assert privacy["per_label"]["FACILITY"]["exact_misses"]["partly_covered"] == 1
    assert privacy["per_label"]["PHONE"]["exact_misses"]["not_detected"] == 1
    assert privacy["per_label"]["NAME"]["exact_misses"] == {
        "wrong_type": 0,
        "boundary_differs": 0,
        "partly_covered": 0,
        "not_detected": 0,
    }
    assert privacy["any_label_precision"] == {
        "predicted_spans": 5,
        "overlapping_gold_phi": 4,
        "precision": 0.8,
    }
    assert privacy["name_recall_cluster_lower_95"] == 1.0
    assert privacy["passed"] is False


def test_cluster_lower_resamples_documents_and_is_deterministic():
    from clinical_deid.evaluation.evaluate import cluster_lower

    documents = [(10, 10)] * 9 + [(0, 10)]
    lower = cluster_lower(documents)
    assert lower == cluster_lower(documents)
    assert 0.6 <= lower < 0.9
    assert cluster_lower([(0, 0)]) is None
