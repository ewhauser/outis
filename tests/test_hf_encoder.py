import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from clinical_deid.detection.decoding import TAGS  # noqa: E402
from clinical_deid.detection.hf_encoder import (  # noqa: E402
    ENCODER_KIND,
    HFTokenDetector,
    refine_boundaries,
    special_layout,
    window_jobs,
)
from clinical_deid.types import Span  # noqa: E402


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    words = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "clinical",
        "note",
        "patient",
        "seen",
        "at",
        "Harrow",
        "Clinic",
        "on",
        "March",
        "4",
        ",",
        "2025",
        ".",
        "tel",
        "=",
        "(",
        ")",
        "555",
        "-",
        "0171",
    ]
    vocab = {word: index for index, word in enumerate(words)}
    core = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
    )
    core.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=core,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def tiny_detector(max_length=16, batch_size=4):
    tokenizer = tiny_tokenizer()
    config = transformers.ModernBertConfig(
        vocab_size=len(tokenizer.get_vocab()),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        num_labels=len(TAGS),
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        global_attn_every_n_layers=1,
        local_attention=16,
    )
    torch.manual_seed(0)
    model = transformers.ModernBertForTokenClassification(config)
    return HFTokenDetector(
        None,
        device="cpu",
        batch_size=batch_size,
        model=model,
        tokenizer=tokenizer,
        config={"encoder": ENCODER_KIND, "max_length": max_length},
    )


@pytest.mark.parametrize(
    "n_tokens,window",
    [(0, 510), (1, 510), (510, 510), (511, 510), (2000, 510), (37, 14)],
)
def test_windows_own_every_token_exactly_once(n_tokens, window):
    owned = []
    for start, stop, left, right in window_jobs(n_tokens, window):
        assert start <= left <= right <= stop and stop - start <= window
        owned.extend(range(left, right))
    assert owned == list(range(n_tokens))


def test_special_layout_finds_cls_and_sep():
    tokenizer = tiny_tokenizer()
    prefix, suffix = special_layout(tokenizer)
    assert prefix == [tokenizer.cls_token_id] and suffix == [tokenizer.sep_token_id]


def test_refine_boundaries_trims_merged_separators_only():
    text = "tel=(617) 555-0171; “Comfort”"
    spans = [Span(3, 18, "PHONE"), Span(20, 29, "NAME")]
    refined = refine_boundaries(spans, text)
    assert [(s.start, s.end, s.label) for s in refined] == [
        (4, 18, "PHONE"),
        (21, 28, "NAME"),
    ]
    assert text[4:18] == "(617) 555-0171" and text[21:28] == "Comfort"


def test_detector_offsets_are_valid_and_batch_invariant():
    detector = tiny_detector()
    texts = [
        "patient seen at Harrow Clinic on March 4 , 2025 .",
        " ".join(["clinical note"] * 12) + " tel = ( 555 ) 555 - 0171 .",
        "",
    ]
    batched = detector.raw_batch(texts)
    single = [detector.raw_batch([text])[0] for text in texts]
    assert [[(s.start, s.end, s.label) for s in doc] for doc in batched] == [
        [(s.start, s.end, s.label) for s in doc] for doc in single
    ]
    for doc_a, doc_b in zip(batched, single):
        assert all(abs(a.score - b.score) < 1e-4 for a, b in zip(doc_a, doc_b))
    for text, spans in zip(texts, batched):
        for span in spans:
            assert 0 <= span.start < span.end <= len(text)
            assert text[span.start : span.end] == text[span.start : span.end].strip()


def test_emissions_cover_long_documents_across_windows():
    detector = tiny_detector(max_length=12)
    text = " ".join(["clinical note patient seen"] * 10)
    ((logits, offsets),) = detector.emissions([text])
    assert logits.shape == (len(offsets), len(TAGS))
    assert torch.isfinite(logits).all()


def test_trim_merged_punctuation_balances_parentheses_and_periods():
    from clinical_deid.detection.hf_encoder import trim_merged_punctuation

    text = (
        "call ((717) 555-0185) or (+1 509 555 0126) and Samira O.. Next (617) 555-0101"
    )
    a = text.index("((")
    assert text[slice(*trim_merged_punctuation(text, a, a + 15))] == "(717) 555-0185"
    b = text.index("(+1")
    assert text[slice(*trim_merged_punctuation(text, b, b + 16))] == "+1 509 555 0126"
    c = text.index("Samira")
    assert text[slice(*trim_merged_punctuation(text, c, c + 10))] == "Samira O."
    d = text.index("(617)")
    assert text[slice(*trim_merged_punctuation(text, d, d + 14))] == "(617) 555-0101"


def test_detect_batch_is_model_only():
    detector = tiny_detector()
    texts = ["Seen at UCSF on 03/02/2024 by Dr. Ana Ruiz."]
    assert detector.detect_batch(texts) == detector.raw_batch(texts)


def test_only_encoder_checkpoints_load(tmp_path):
    import json

    from clinical_deid.detection.hf_encoder import load_detector

    (tmp_path / "config.json").write_text(json.dumps({"adapter": {"hidden_size": 640}}))
    with pytest.raises(ValueError, match="only fine-tuned hf_token_classifier"):
        load_detector(tmp_path)


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_window_rejected_without_looping(size):
    with pytest.raises(ValueError, match="window size"):
        list(window_jobs(10, size))


@pytest.mark.parametrize("batch_size", [0, -1, True])
def test_invalid_batch_size_rejected(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        HFTokenDetector(None, batch_size=batch_size)


def test_model_root_errors_are_clear_and_publication_fails_closed(tmp_path):
    import json

    from clinical_deid.detection.hf_encoder import load_detector

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "modernbert"}))
    with pytest.raises(ValueError, match="nested Hugging Face"):
        load_detector(tmp_path)
    (tmp_path / "publication_in_progress.json").write_text("{}")
    with pytest.raises(ValueError, match="publication is in progress"):
        load_detector(tmp_path)


@pytest.mark.parametrize("batch_size", [1, 8, 16, 32])
def test_cpu_window_batch_sizes_and_mixed_companions_preserve_spans(batch_size):
    detector = tiny_detector(batch_size=1)
    text = "patient seen at Harrow Clinic on March 4 , 2025 ."
    long_text = " ".join([text] * 4)
    cases = [text, long_text, "", "tel = ( 555 ) 555 - 0171 ."]
    reference = [detector.raw_batch([item])[0] for item in cases]
    detector.batch_size = batch_size
    mixed = detector.raw_batch(list(reversed(cases)))

    def canonical(docs):
        return [[(s.start, s.end, s.label) for s in spans] for spans in docs]

    assert canonical(mixed) == list(reversed(canonical(reference)))
