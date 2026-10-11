"""Service detector selection and versioning using local fake model bundles."""

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from outis.api import create_app
from outis.detection import hf_encoder
from outis.service import Deidentifier
from outis.types import Span


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
    service = Deidentifier(hf_bundle, device="cpu")
    assert isinstance(service.detector, FakeDetector)
    assert service.detector.options == {
        "device": "cpu",
        "batch_size": 8,
        "precision": "auto",
    }
    assert len(service.version) == 16 and service.version != "injected-detector"
    assert service.deidentify("Mara Calder arrived.")["text"] == "[NAME] arrived."


def test_default_model_dir_is_loaded_when_it_holds_weights(tmp_path, monkeypatch):
    from outis.paths import DEFAULT_MODEL_DIR

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hf_encoder, "HFTokenDetector", FakeDetector)
    root = tmp_path / DEFAULT_MODEL_DIR
    (root / "hf").mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"encoder": hf_encoder.ENCODER_KIND}))
    (root / "hf" / "model.safetensors").write_bytes(b"weights")
    service = Deidentifier(device="cpu")
    assert service.detector.model_dir == Path(DEFAULT_MODEL_DIR)
    assert len(service.version) == 16 and service.version != "injected-detector"


# The promoted checkpoint (takeover-deberta-mix9-001, step 9288) as published
# at DEFAULT_MODEL_REVISION. Git blob ids of the committed bundle files equal
# the Hugging Face file ids at that revision; the weights are pinned by SHA-256.
# Promoting a different checkpoint must update these deliberately.
SHIPPED_BUNDLE_BLOBS = {
    "config.json": "9545692b3b36327e8a1a9da8bfcf6ff36c65ccc0",
    "hf/config.json": "510655da490ecd0c3a551703bdd64dbe1ae2282f",
    "hf/tokenizer.json": "6192ab6e0d74b9d02b121ab94c21f9c075bf6525",
    "hf/tokenizer_config.json": "40488bf559d36702944b92078bb72bceb7c98a61",
}
SHIPPED_WEIGHTS_SHA256 = (
    "2443b72d83eb95a9163fcea0895f474631aaa4a839d27a902a87473ead696ae4"
)
SHIPPED_REVISION = "11019035c00e3354fffd9631d78ce8e8efadaa09"


def git_blob_id(path):
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def test_default_model_is_the_promoted_shipped_bundle():
    from outis.detection.decoding import TAGS
    from outis.paths import (
        DEFAULT_MODEL_DIR,
        DEFAULT_MODEL_REPO,
        DEFAULT_MODEL_REVISION,
        PROJECT_ROOT,
    )

    assert DEFAULT_MODEL_REPO == "ewhauser/outis-deberta-v3-base"
    assert DEFAULT_MODEL_REVISION == SHIPPED_REVISION
    root = PROJECT_ROOT / DEFAULT_MODEL_DIR
    assert {
        name: git_blob_id(root / name) for name in SHIPPED_BUNDLE_BLOBS
    } == SHIPPED_BUNDLE_BLOBS
    config = json.loads((root / "config.json").read_text())
    assert config["encoder"] == hf_encoder.ENCODER_KIND
    assert config["base_model"] == "microsoft/deberta-v3-base"
    assert config["tags"] == TAGS
    hf_config = json.loads((root / "hf" / "config.json").read_text())
    assert hf_config["architectures"] == ["DebertaV2ForTokenClassification"]
    assert [hf_config["id2label"][str(i)] for i in range(len(TAGS))] == TAGS
    summary = json.loads((root / "training_summary.json").read_text())
    assert summary["checkpoint"] == {
        "step": 9288,
        "path": "models/takeover-deberta-mix9-001/step-009288",
    }
    weights = root / "hf" / "model.safetensors"
    if not weights.is_file():
        pytest.skip("shipped weights are not present in this checkout")
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    assert digest.hexdigest() == SHIPPED_WEIGHTS_SHA256


def fake_snapshot_download(bundle, calls):
    def download(repo_id, **options):
        calls.append({"repo_id": repo_id, **options})
        return str(bundle)

    return download


def test_default_downloads_the_pinned_published_model(hf_bundle, tmp_path, monkeypatch):
    import huggingface_hub

    from outis import paths

    calls = []
    monkeypatch.chdir(tmp_path / "hf")
    monkeypatch.setattr(hf_encoder, "HFTokenDetector", FakeDetector)
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", fake_snapshot_download(hf_bundle, calls)
    )
    service = Deidentifier(device="cpu")
    assert calls == [
        {
            "repo_id": paths.DEFAULT_MODEL_REPO,
            "revision": paths.DEFAULT_MODEL_REVISION,
            "allow_patterns": ["config.json", "hf/*"],
        }
    ]
    assert service.detector.model_dir == hf_bundle
    assert service.version == Deidentifier(hf_bundle, device="cpu").version


def test_checkout_without_weights_downloads_the_published_model(
    hf_bundle, tmp_path, monkeypatch
):
    import huggingface_hub

    from outis.paths import DEFAULT_MODEL_DIR

    calls = []
    checkout = tmp_path / "checkout"
    root = checkout / DEFAULT_MODEL_DIR
    (root / "hf").mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"encoder": hf_encoder.ENCODER_KIND}))
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(hf_encoder, "HFTokenDetector", FakeDetector)
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", fake_snapshot_download(hf_bundle, calls)
    )
    assert Deidentifier(device="cpu").detector.model_dir == hf_bundle
    assert len(calls) == 1


def test_explicit_model_dir_never_downloads(hf_bundle, monkeypatch):
    import huggingface_hub

    def refuse(*args, **kwargs):
        raise AssertionError("an explicit model directory must not download")

    monkeypatch.setattr(hf_encoder, "HFTokenDetector", FakeDetector)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse)
    assert Deidentifier(hf_bundle, device="cpu").detector.model_dir == hf_bundle
    monkeypatch.setenv("OUTIS_MODEL_DIR", str(hf_bundle))
    monkeypatch.setenv("OUTIS_DEVICE", "cpu")
    monkeypatch.delenv("OUTIS_API_KEY", raising=False)
    with TestClient(create_app()) as client:
        assert client.get("/ready").status_code == 200


def test_unsupported_checkpoints_are_rejected(tmp_path):
    (tmp_path / "config.json").write_text('{"adapter": {}}')
    with pytest.raises(ValueError, match="only fine-tuned hf_token_classifier"):
        Deidentifier(tmp_path, device="cpu")


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
    monkeypatch.setenv("OUTIS_MODEL_DIR", str(hf_bundle))
    monkeypatch.setenv("OUTIS_DEVICE", "cpu")
    monkeypatch.delenv("OUTIS_API_KEY", raising=False)
    monkeypatch.delenv("OUTIS_SECRET", raising=False)
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
