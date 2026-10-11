"""Release check: the shipped tokenizer and DeBERTa loader through the HTTP API.

The fixture is a two-layer DeBERTa-v2 built from the shipped ``hf/config.json``
and saved beside the shipped tokenizer files, byte for byte. Its weights are set
by hand so that every token's tag depends only on its id: each layer's
attention and feed-forward outputs are zero, so the hidden state stays the
normalized one-hot embedding of the tag, and the classifier reads it back. The
expected output is therefore known without a trained model, and the test runs
the real loader, special-token layout, overlapping windows, constrained
decoding, boundary refinement, policy, replacement and API on CPU.
"""

import json
import shutil

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from fastapi.testclient import TestClient  # noqa: E402

from outis.api import create_app  # noqa: E402
from outis.detection.decoding import TAGS  # noqa: E402
from outis.paths import PROJECT_ROOT  # noqa: E402
from outis.service import Deidentifier  # noqa: E402

SHIPPED = PROJECT_ROOT / "models" / "deid-deberta-v3-base"
SECRET = b"test-only-secret-never-use-in-production-12345"
NOTE = (
    "Patient Zoë Núñez was seen on 2024-03-05 at Mercy Clinic by Dr. Lee. "
    "MRN 448812. Call tomorrow."
)
# The shipped tokenizer's segmentation of NOTE. A tokenizer or normalization
# change shows up here first, before it moves any span.
NOTE_TOKENS = [
    "▁Patient", "▁Zoë", "▁N", "ú", "ñez", "▁was", "▁seen", "▁on", "▁2024",
    "-", "03", "-", "05", "▁at", "▁Mercy", "▁Clinic", "▁by", "▁Dr", ".",
    "▁Lee", ".", "▁M", "RN", "▁448", "812", ".", "▁Call", "▁tomorrow", ".",
]  # fmt: skip
# Each listed token always receives this tag; every other token is O. The
# entities straddle the window seams at max_length 16 (see the layout test).
TAG_BY_TOKEN = {
    "▁Zoë": "B-NAME",
    "▁N": "I-NAME",
    "ú": "I-NAME",
    "ñez": "E-NAME",
    "▁2024": "B-DATE",
    "-": "I-DATE",
    "03": "I-DATE",
    "05": "E-DATE",
    "▁Mercy": "B-FACILITY",
    "▁Clinic": "E-FACILITY",
    "▁Lee": "S-NAME",
    "▁448": "B-MRN",
    "812": "E-MRN",
    "▁Jane": "B-NAME",
    "▁Doe": "E-NAME",
}
MAX_LENGTH = 16


def deterministic_deberta(tokenizer):
    config = transformers.DebertaV2Config.from_pretrained(
        SHIPPED / "hf",
        hidden_size=64,
        intermediate_size=64,
        num_attention_heads=2,
        num_hidden_layers=2,
        pooler_hidden_size=64,
    )
    assert config.model_type == "deberta-v2" and config.num_labels == len(TAGS)
    model = transformers.DebertaV2ForTokenClassification(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.weight.fill_(1.0)
        embeddings = model.deberta.embeddings.word_embeddings.weight
        embeddings[:, TAGS.index("O")] = 1.0
        for token, tag in TAG_BY_TOKEN.items():
            index = tokenizer.convert_tokens_to_ids(token)
            assert index != tokenizer.unk_token_id, token
            embeddings[index].zero_()
            embeddings[index, TAGS.index(tag)] = 1.0
        model.classifier.weight[:, : len(TAGS)] = 8.0 * torch.eye(len(TAGS))
    return model


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("shipped-stack")
    (directory / "hf").mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(SHIPPED / "hf" / name, directory / "hf" / name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        directory / "hf", local_files_only=True
    )
    deterministic_deberta(tokenizer).save_pretrained(directory / "hf")
    config = json.loads((SHIPPED / "config.json").read_text())
    (directory / "config.json").write_text(
        json.dumps({**config, "max_length": MAX_LENGTH})
    )
    return directory


@pytest.fixture(scope="module")
def service(bundle):
    return Deidentifier(bundle, secret=SECRET, device="cpu")


def test_shipped_tokenizer_segmentation_is_pinned(service):
    tokenizer = service.detector.tokenizer
    encoded = tokenizer(NOTE, add_special_tokens=False)
    assert tokenizer.convert_ids_to_tokens(encoded["input_ids"]) == NOTE_TOKENS
    assert (service.detector.prefix, service.detector.suffix) == (
        [tokenizer.cls_token_id],
        [tokenizer.sep_token_id],
    )


def test_loader_uses_the_shipped_bundle(service, bundle):
    detector = service.detector
    assert type(detector.model).__name__ == "DebertaV2ForTokenClassification"
    assert detector.config["tags"] == TAGS
    assert detector.content_window == MAX_LENGTH - 2
    assert len(service.version) == 16
    for name in ("tokenizer.json", "tokenizer_config.json"):
        assert (bundle / "hf" / name).read_bytes() == (
            SHIPPED / "hf" / name
        ).read_bytes()


def test_entities_cross_window_seams():
    from outis.detection.hf_encoder import window_jobs

    # 14 content tokens: windows own [0, 11), [11, 19), [19, 29). The date
    # (tokens 8-12) crosses the first seam, the MRN (23-24) sits past the second.
    assert [(left, right) for *_, left, right in window_jobs(29, 14)] == [
        (0, 11),
        (11, 19),
        (19, 29),
    ]


def test_detected_spans_match_the_golden_offsets(service):
    spans = service.detector.detect_batch([NOTE])[0]
    assert [(NOTE[s.start : s.end], s.label) for s in spans] == [
        ("Zoë Núñez", "NAME"),
        ("2024-03-05", "DATE"),
        ("Mercy Clinic", "FACILITY"),
        ("Lee", "NAME"),
        ("448812", "MRN"),
    ]
    assert all(span.score > 0.99 for span in spans)


def test_http_releases_the_golden_redaction(service):
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/v1/batch",
            json={"items": [{"text": NOTE}, {"text": "Seen by Jane Doe."}]},
        )
    assert response.status_code == 200
    first, second = response.json()
    assert first["text"] == (
        "Patient [NAME] was seen on [DATE] at [FACILITY] by Dr. [NAME]. "
        "MRN [MRN]. Call tomorrow."
    )
    assert [entity["label"] for entity in first["entities"]] == [
        "NAME",
        "DATE",
        "FACILITY",
        "NAME",
        "MRN",
    ]
    assert first["model_version"] == service.version
    assert second["text"] == "Seen by [NAME]."
