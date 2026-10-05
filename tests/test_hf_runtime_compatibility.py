"""Local tiny CPU fixtures exercise HF runtime freezing and local loading."""

import json

import pytest
from test_hf_encoder import tiny_detector

from clinical_deid.evaluation.runtime import neural_runtime_manifest


@pytest.fixture
def hf_candidate(tmp_path):
    detector = tiny_detector()
    directory = tmp_path / "candidate"
    detector.model.save_pretrained(directory / "hf")
    detector.tokenizer.save_pretrained(directory / "hf")
    config = {**detector.config, "base_model": "local-tiny-fixture"}
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "training_summary.json").write_text(
        json.dumps({"corpus_audit": {"test_used_for_training": False}})
    )
    return directory, detector, config


def test_runtime_pins_hf_code_versions_and_actual_tokenizer(hf_candidate):
    directory, detector, _ = hf_candidate
    frozen = neural_runtime_manifest(directory)
    assert frozen == neural_runtime_manifest(directory, detector.tokenizer)
    assert {
        "decoding.py",
        "hf_encoder.py",
        "hf_model_implementation",
        "hf_tokenizer_implementation",
    } <= frozen["neural_sources"].keys()
    assert frozen["encoder"]["transformers_version"]
    assert frozen["tokenizer"]["tokenizers_version"]
    detector.tokenizer.add_tokens(["a_new_fixture_token"])
    assert neural_runtime_manifest(directory, detector.tokenizer) != frozen


def test_hf_detector_uses_saved_local_files(hf_candidate):
    from clinical_deid.detection.hf_encoder import load_detector

    directory, detector, _ = hf_candidate
    actual = load_detector(directory, device="cpu", batch_size=8)
    text = "patient seen at Harrow Clinic"

    def canonical(spans):
        return [(span.start, span.end, span.label) for span in spans]

    assert canonical(actual.detect(text)) == canonical(detector.detect(text))
    assert actual.window == detector.max_length
