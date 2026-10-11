import re
import types

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from outis.detection.decoding import TAGS  # noqa: E402
from outis.detection.hf_encoder import (  # noqa: E402
    ENCODER_KIND,
    HFTokenDetector,
    refine_boundaries,
    special_layout,
    window_jobs,
)
from outis.types import Span  # noqa: E402


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


# Scripted per-token tags for the tiny vocabulary; every other token is O.
SCRIPT = {
    "Harrow": "B-FACILITY",
    "Clinic": "E-FACILITY",
    "March": "B-DATE",
    "4": "I-DATE",
    ",": "I-DATE",
    "2025": "E-DATE",
    "(": "B-PHONE",
    ")": "I-PHONE",
    "555": "I-PHONE",
    "-": "I-PHONE",
    "0171": "E-PHONE",
}
SENTENCE = "patient seen at Harrow Clinic on March 4 , 2025 ."
PHONE = "tel = ( 555 ) 555 - 0171 ."


def scripted_logits(table, previous, ids, following):
    """Tag scores for each token, marked in the O column with its row neighbours.

    The marker makes a token's logits depend on the tokens beside it in the
    window the model saw, so logits taken from the wrong position, or from a
    window edge the token does not own, differ from the expected values.
    """
    logits = table[ids].clone()
    logits[..., 0] += previous.float() / 100 + following.float() / 10_000
    return logits


class ScriptedEncoder(torch.nn.Module):
    """Token classifier whose logits are a fixed function of the input rows."""

    def __init__(self, tokenizer):
        super().__init__()
        vocab = tokenizer.get_vocab()
        self.pad_id = tokenizer.pad_token_id
        self.table = torch.zeros((len(vocab), len(TAGS)))
        self.table[:, 0] = 10.0
        for word, tag in SCRIPT.items():
            self.table[vocab[word], 0] = 0.0
            self.table[vocab[word], TAGS.index(tag)] = 10.0
        self.config = types.SimpleNamespace(num_labels=len(TAGS))
        self.rows = []

    def forward(self, input_ids, attention_mask):
        self.rows.extend(
            row[mask.bool()].tolist() for row, mask in zip(input_ids, attention_mask)
        )
        pad = torch.full((len(input_ids), 1), self.pad_id)
        previous = torch.cat([pad, input_ids[:, :-1]], dim=1)
        following = torch.cat([input_ids[:, 1:], pad], dim=1)
        return types.SimpleNamespace(
            logits=scripted_logits(self.table, previous, input_ids, following)
        )


def scripted_detector(max_length=12, batch_size=4):
    tokenizer = tiny_tokenizer()
    return HFTokenDetector(
        None,
        device="cpu",
        batch_size=batch_size,
        model=ScriptedEncoder(tokenizer),
        tokenizer=tokenizer,
        config={"encoder": ENCODER_KIND, "max_length": max_length},
    )


def expected_logits(detector, text):
    """Logits of each token as the middle of its full-document neighbourhood."""
    ids = detector.tokenizer(text, add_special_tokens=False)["input_ids"]
    previous = detector.prefix[-1:] + ids[:-1]
    following = ids[1:] + detector.suffix[:1]
    return scripted_logits(
        detector.model.table,
        torch.tensor(previous, dtype=torch.long),
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(following, dtype=torch.long),
    )


def spans_of(text, *entities):
    found = [
        (match.start(), match.end(), label)
        for value, label in entities
        for match in re.finditer(re.escape(value), text)
    ]
    return sorted(found)


def canonical(spans):
    return [(s.start, s.end, s.label) for s in spans]


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


def test_trim_merged_punctuation_balances_parentheses_and_periods():
    from outis.detection.hf_encoder import trim_merged_punctuation

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


def test_only_encoder_checkpoints_load(tmp_path):
    import json

    from outis.detection.hf_encoder import load_detector

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

    from outis.detection.hf_encoder import load_detector

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


@pytest.mark.parametrize("batch_size", [1, 3, 64])
def test_emissions_are_the_model_logits_from_each_owning_window(batch_size):
    detector = scripted_detector(max_length=12, batch_size=batch_size)
    long_text = " ".join([SENTENCE] * 4)
    texts = [SENTENCE, long_text, "", PHONE]
    emitted = detector.emissions(texts)
    for text, (logits, offsets) in zip(texts, emitted):
        encoded = detector.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        assert offsets == [tuple(pair) for pair in encoded["offset_mapping"]]
        assert torch.equal(logits, expected_logits(detector, text))
    # The long document needs several overlapping windows of 10 content tokens.
    assert len(detector.model.rows) > len(texts)
    assert max(len(row) for row in detector.model.rows) == 12


def test_entities_crossing_window_boundaries_decode_whole():
    detector = scripted_detector(max_length=12)
    text = " ".join([SENTENCE] * 4)
    ids = detector.tokenizer(text, add_special_tokens=False)["input_ids"]
    windows = list(window_jobs(len(ids), detector.content_window))
    owned_boundaries = {left for _, _, left, _ in windows[1:]}
    window_edges = {stop for _, stop, _, _ in windows[:-1]}
    date_tokens = [
        range(i, i + 4)
        for i in range(len(ids))
        if ids[i] == detector.tokenizer.convert_tokens_to_ids("March")
    ]
    # Fixture check: dates straddle both ownership boundaries and window edges.
    assert any(t.start < b <= t.stop - 1 for t in date_tokens for b in owned_boundaries)
    assert any(t.start < e <= t.stop - 1 for t in date_tokens for e in window_edges)

    expected = spans_of(text, ("Harrow Clinic", "FACILITY"), ("March 4 , 2025", "DATE"))
    assert len(expected) == 8
    assert canonical(detector.raw_batch([text])[0]) == expected
    assert canonical(detector.raw_batch([PHONE])[0]) == spans_of(
        PHONE, ("( 555 ) 555 - 0171", "PHONE")
    )


def test_detect_batch_releases_exactly_the_model_spans():
    detector = scripted_detector(max_length=16)
    texts = [
        # Out-of-vocabulary text the model tags O throughout: no rule may add
        # the obvious date or name.
        "Seen at UCSF on 03/02/2024 by Dr. Ana Ruiz.",
        "patient seen at Harrow Clinic on 03/02/2024 by Dr. Ana Ruiz .",
    ]
    assert [canonical(spans) for spans in detector.detect_batch(texts)] == [
        [],
        spans_of(texts[1], ("Harrow Clinic", "FACILITY")),
    ]


def test_invalid_precision_rejected():
    with pytest.raises(ValueError, match="precision"):
        HFTokenDetector(None, precision="fp16")


def test_bf16_precision_autocasts_and_returns_float32_logits():
    detector = tiny_detector()
    text = "patient seen at Harrow Clinic on March 4 , 2025 ."
    reference = detector.emissions([text])[0][0]
    detector.precision = "bf16"
    seen = []
    hook = detector.model.register_forward_hook(
        lambda module, args, output: seen.append(output.logits.dtype)
    )
    try:
        logits = detector.emissions([text])[0][0]
    finally:
        hook.remove()
    assert seen == [torch.bfloat16]
    assert logits.dtype == torch.float32
    assert torch.allclose(logits, reference, atol=0.1)
