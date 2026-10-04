"""Service detector selection and versioning using local fake model bundles."""

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clinical_deid.api import create_app
from clinical_deid.detection import adapter as model
from clinical_deid.detection import hf_encoder
from clinical_deid.service import Deidentifier
from clinical_deid.types import Span


class FakeDetector:
    def __init__(self, model_dir, **options):
        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "config.json").read_text())
        self.options = options

    def detect_batch(self, texts):
        return [
            [Span(0, 11, "NAME")] if text.startswith("Mara Calder") else []
            for text in texts
        ]


@pytest.fixture
def hf_bundle(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "encoder": hf_encoder.ENCODER_KIND,
                "max_length": 512,
                "model_labels": ["NAME"],
            }
        )
    )
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text('{"model_type": "modernbert"}')
    (hf / "tokenizer.json").write_text('{"version": "1.0"}')
    (hf / "model.safetensors").write_bytes(b"fake-model-weights")
    return tmp_path


def test_explicit_hf_service_uses_factory_and_preserves_options(hf_bundle, monkeypatch):
    monkeypatch.setattr(hf_encoder, "HFTokenDetector", FakeDetector)
    service = Deidentifier(hf_bundle, backend="model", device="cpu")
    assert isinstance(service.detector, FakeDetector)
    assert service.detector.options == {
        "backend": "model",
        "device": "cpu",
        "batch_size": 8,
    }
    assert len(service.version) == 16 and service.version != "injected-detector"
    assert service.deidentify("Mara Calder arrived.")["text"] == "[NAME] arrived."


def test_legacy_default_directory_options_and_adapter_version_are_unchanged(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "models" / "clinical-adapter"
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"adapter": {}}')
    (root / "adapter.safetensors").write_bytes(b"legacy-adapter")
    monkeypatch.setattr(model, "Detector", FakeDetector)
    service = Deidentifier(device="cpu", base_dir="local-base")
    assert service.detector.model_dir == Path("models/clinical-adapter")
    assert service.detector.options == {
        "backend": "hybrid",
        "device": "cpu",
        "base_dir": "local-base",
        "batch_size": 8,
    }
    assert service.version == hashlib.sha256(b"legacy-adapter").hexdigest()[:16]


@pytest.mark.parametrize(
    "artifact",
    [
        "config.json",
        "hf/config.json",
        "hf/tokenizer.json",
        "hf/model.safetensors",
    ],
)
def test_hf_version_changes_with_each_inference_artifact(
    hf_bundle, monkeypatch, artifact
):
    monkeypatch.setattr(hf_encoder, "HFTokenDetector", FakeDetector)
    original = Deidentifier(hf_bundle, device="cpu").version
    assert Deidentifier(hf_bundle, device="cpu").version == original
    path = hf_bundle / artifact
    if artifact == "config.json":
        config = json.loads(path.read_text())
        config["model_labels"] = ["FACILITY"]
        path.write_text(json.dumps(config))
    else:
        path.write_bytes(path.read_bytes() + b" ")
    assert Deidentifier(hf_bundle, device="cpu").version != original


def test_hf_version_includes_shards_and_is_independent_of_directory(
    hf_bundle, tmp_path, monkeypatch
):
    import shutil

    monkeypatch.setattr(hf_encoder, "HFTokenDetector", FakeDetector)
    original = Deidentifier(hf_bundle, device="cpu").version
    (hf_bundle / "hf" / "model-00002-of-00002.safetensors").write_bytes(b"shard-2")
    sharded = Deidentifier(hf_bundle, device="cpu").version
    assert sharded != original
    copy = tmp_path / "relocated"
    shutil.copytree(hf_bundle, copy, ignore=shutil.ignore_patterns("relocated"))
    assert Deidentifier(copy, device="cpu").version == sharded


def test_http_model_environment_selects_hf_and_exposes_bundle_version(
    hf_bundle, monkeypatch
):
    monkeypatch.setattr(hf_encoder, "HFTokenDetector", FakeDetector)
    monkeypatch.setenv("DEID_MODEL_DIR", str(hf_bundle))
    monkeypatch.setenv("DEID_DEVICE", "cpu")
    monkeypatch.delenv("DEID_API_KEY", raising=False)
    monkeypatch.delenv("DEID_BASE_DIR", raising=False)
    monkeypatch.delenv("DEID_SECRET", raising=False)
    with TestClient(create_app()) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200
        version = ready.json()["model_version"]
        assert version != "injected-detector"
        result = client.post("/v1/deidentify", json={"text": "Mara Calder arrived."})
        assert result.status_code == 200
        assert result.json()["text"] == "[NAME] arrived."
        assert result.json()["model_version"] == version


def test_injected_detector_without_model_artifacts_keeps_existing_version():
    class InjectedDetector:
        def detect_batch(self, texts):
            return [[] for _ in texts]

    service = Deidentifier(detector=InjectedDetector())
    assert service.version == "injected-detector"
