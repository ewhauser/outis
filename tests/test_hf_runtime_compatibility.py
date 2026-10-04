"""Local tiny CPU fixtures exercise HF runtime freezing and paired evaluation."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from clinical_deid.evaluation.evaluate import model_manifest
from clinical_deid.types import Span
from test_hf_encoder import tiny_detector


@pytest.fixture
def hf_candidate(tmp_path):
    detector = tiny_detector()
    directory = tmp_path / "candidate"
    detector.model.save_pretrained(directory / "hf")
    detector.tokenizer.save_pretrained(directory / "hf")
    config = {
        **detector.config,
        "base_model": "local-tiny-fixture",
        "model_labels": ["NAME"],
    }
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "training_summary.json").write_text(
        json.dumps({"corpus_audit": {"test_used_for_training": False}})
    )
    return directory, detector, config


@pytest.fixture
def cache_module(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "hf_runtime_cache_fixture", scripts / "prediction_cache.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "prediction_cache", module)
    return module


def test_runtime_pins_hf_code_versions_and_actual_tokenizer(hf_candidate, cache_module):
    directory, detector, _ = hf_candidate
    frozen = cache_module.neural_runtime_manifest(model_dir=directory)
    assert frozen == cache_module.neural_runtime_manifest(
        detector.tokenizer, model_dir=directory
    )
    assert {
        "hf_encoder.py",
        "hf_model_implementation",
        "hf_tokenizer_implementation",
    } <= frozen["neural_sources"].keys()
    assert frozen["encoder"]["transformers_version"]
    assert frozen["tokenizer"]["tokenizers_version"]
    detector.tokenizer.add_tokens(["a_new_fixture_token"])
    assert (
        cache_module.neural_runtime_manifest(detector.tokenizer, model_dir=directory)
        != frozen
    )


def test_hf_detector_uses_saved_local_files(hf_candidate):
    from clinical_deid.detection.hf_encoder import load_detector

    directory, detector, _ = hf_candidate
    actual = load_detector(directory, backend="model", device="cpu", batch_size=8)
    text = "patient seen at Harrow Clinic"

    def canonical(spans):
        return [(span.start, span.end, span.label) for span in spans]

    assert canonical(actual.detect(text)) == canonical(detector.detect(text))
    assert actual.window == detector.max_length


def test_paired_hf_policy_and_cache_match_the_served_detector(
    hf_candidate, cache_module, monkeypatch, tmp_path
):
    directory, detector, config = hf_candidate
    scripts = Path(__file__).parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location(
        "hf_paired_fixture", scripts / "evaluate_pair.py"
    )
    paired = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paired)
    rows = [
        {
            "id": "one",
            "text": "Zoë has fever.",
            "spans": [{"start": 0, "end": 3, "label": "NAME"}],
            "split": "validation",
        }
    ]
    data = tmp_path / "development.jsonl"
    data.write_text(json.dumps(rows[0]) + "\n")
    from clinical_deid.types import ENTITY_TYPES

    manifest = {
        "rows": 1,
        "sha256": cache_module.sha256_file(data),
        "split": "validation",
        "independent_final": False,
        "label_support": {label: int(label == "NAME") for label in ENTITY_TYPES},
    }
    data.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest))
    predictions = [[Span(0, 3, "NAME", 0.9, "encoder")]]
    detector.config = config
    monkeypatch.setattr(
        detector, "detect_batch", lambda texts: predictions[: len(texts)]
    )
    monkeypatch.setattr(paired, "load_detector", lambda *args, **kwargs: detector)
    from clinical_deid.detection import rules

    monkeypatch.setattr(
        rules, "augment", lambda text, spans: [Span(0, 7, "NAME", 0.9, "rules")]
    )
    prefix, output = tmp_path / "paired", tmp_path / "predictions.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_pair",
            "--data",
            str(data),
            "--model-dir",
            str(directory),
            "--prefix",
            str(prefix),
            "--device",
            "cpu",
            "--batch-size",
            "8",
            "--predictions-output",
            str(output),
        ],
    )
    paired.main()
    hybrid = json.loads(Path(str(prefix) + "_hybrid.json").read_text())
    assert hybrid["typed_span"]["f1"] == 1
    assert hybrid["model_labels"] == ["NAME"]
    assert hybrid["base_encoder"]["model"] == "local-tiny-fixture"
    assert hybrid["base_encoder"]["kind"] == "hf_token_classifier"
    assert cache_module.load_prediction_cache(output, data)[2] == predictions
    assert hybrid["model_artifacts"] == model_manifest(directory)
