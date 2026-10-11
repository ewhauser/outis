"""Constrained BIOES decoding and window stitching against independent expectations.

The decoder is checked against a brute-force search over every tag sequence
whose legality is defined here, not by the decoder's transition table. Window
stitching is checked with a fake encoder whose logits record which window and
row position produced them, so each token's owner is asserted exactly.
"""

import itertools
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from outis.detection.decoding import TAGS, decode_bioes  # noqa: E402
from outis.detection.hf_encoder import (  # noqa: E402
    ENCODER_KIND,
    HFTokenDetector,
    window_jobs,
)

LOW = -40.0
HIGH = 40.0


def words(count, width=4):
    """Text of ``count`` digit words separated by spaces, with token offsets."""
    tokens = [str(index).rjust(width, "0") for index in range(count)]
    offsets, position = [], 0
    for _token in tokens:
        offsets.append((position, position + width))
        position += width + 1
    return " ".join(tokens), offsets


def emissions(tags):
    logits = torch.full((len(tags), len(TAGS)), LOW)
    for index, tag in enumerate(tags):
        logits[index, TAGS.index(tag)] = HIGH
    return logits


def decoded(logits, offsets, text):
    return [
        (span.start, span.end, span.label)
        for span in decode_bioes(logits, offsets, text)
    ]


def legal(sequence):
    """BIOES grammar, written independently of the decoder."""
    inside = None
    for tag in sequence:
        boundary, label = (tag, None) if tag == "O" else tag.split("-", 1)
        if inside is None:
            if boundary in {"I", "E"}:
                return False
            if boundary == "B":
                inside = label
        else:
            if boundary not in {"I", "E"} or label != inside:
                return False
            if boundary == "E":
                inside = None
    return inside is None


def spans_of(sequence, offsets):
    spans, start = [], None
    for index, tag in enumerate(sequence):
        if tag[0] in "BS":
            start = index
        if tag[0] in "ES":
            spans.append((offsets[start][0], offsets[index][1], tag[2:]))
    return spans


@pytest.mark.parametrize(
    "tags,expected",
    [
        (["B-DATE", "I-DATE", "I-DATE", "E-DATE"], [(0, 19, "DATE")]),
        (["O", "B-MRN", "E-MRN", "O"], [(5, 14, "MRN")]),
        (
            ["S-DATE", "B-MRN", "E-MRN", "S-ID"],
            [
                (0, 4, "DATE"),
                (5, 14, "MRN"),
                (15, 19, "ID"),
            ],
        ),
        (["B-DATE", "E-DATE", "B-DATE", "E-DATE"], [(0, 9, "DATE"), (10, 19, "DATE")]),
        (["O", "O", "O", "O"], []),
    ],
)
def test_legal_multi_token_sequences_decode_to_exact_spans(tags, expected):
    text, offsets = words(len(tags))
    assert decoded(emissions(tags), offsets, text) == expected


@pytest.mark.parametrize(
    "favored,forbidden",
    [
        (["I-DATE", "E-DATE"], "cannot start inside an entity"),
        (["B-DATE", "B-DATE"], "cannot leave an entity open"),
        (["B-DATE", "E-MRN"], "cannot change label inside an entity"),
        (["B-DATE", "O", "E-DATE"], "cannot leave an entity for O"),
        (["S-DATE", "I-DATE"], "cannot continue after S"),
    ],
)
def test_illegal_argmax_sequences_are_never_emitted(favored, forbidden):
    # Each favored sequence is illegal; ties are broken by small, explicit
    # second choices so the legal optimum is unique.
    text, offsets = words(len(favored))
    logits = torch.full((len(favored), len(TAGS)), LOW)
    for index, tag in enumerate(favored):
        logits[index, TAGS.index(tag)] = HIGH
        logits[index, TAGS.index("O")] = HIGH - 1 - index * 0.1
    best = max(
        (
            sequence
            for sequence in itertools.product(
                sorted({*favored, "O", "S-DATE", "B-DATE", "E-DATE", "I-DATE"}),
                repeat=len(favored),
            )
            if legal(sequence)
        ),
        key=lambda sequence: sum(
            float(logits[i, TAGS.index(tag)]) for i, tag in enumerate(sequence)
        ),
    )
    assert not legal(favored), forbidden
    assert decoded(logits, offsets, text) == spans_of(best, offsets)


def test_first_and_last_tokens_obey_start_and_end_constraints():
    text, offsets = words(1)
    logits = torch.full((1, len(TAGS)), LOW)
    # A lone token cannot be I, E (no start) or B (no end); S is the best legal tag.
    logits[0, TAGS.index("I-DATE")] = 30
    logits[0, TAGS.index("E-DATE")] = 29
    logits[0, TAGS.index("B-DATE")] = 28
    logits[0, TAGS.index("S-DATE")] = 10
    assert decoded(logits, offsets, text) == [(0, 4, "DATE")]
    logits[0, TAGS.index("S-DATE")] = LOW
    assert decoded(logits, offsets, text) == []


# Small subsets keep exhaustive search fast while still mixing two labels.
SUBSET = [
    "O",
    "B-DATE",
    "I-DATE",
    "E-DATE",
    "S-DATE",
    "B-MRN",
    "I-MRN",
    "E-MRN",
    "S-MRN",
]


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("length", [1, 2, 3, 5])
def test_viterbi_matches_exhaustive_search_over_legal_sequences(seed, length):
    generator = torch.Generator().manual_seed(seed * 10 + length)
    text, offsets = words(length)
    logits = torch.full((length, len(TAGS)), -1e4)
    columns = [TAGS.index(tag) for tag in SUBSET]
    logits[:, columns] = torch.randn((length, len(SUBSET)), generator=generator) * 3

    def score(sequence):
        return sum(float(logits[i, TAGS.index(t)]) for i, t in enumerate(sequence))

    ranked = sorted(
        (s for s in itertools.product(SUBSET, repeat=length) if legal(s)),
        key=score,
        reverse=True,
    )
    if len(ranked) > 1 and math.isclose(score(ranked[0]), score(ranked[1])):
        pytest.skip("tied optimum")
    assert decoded(logits, offsets, text) == spans_of(ranked[0], offsets)


def test_span_confidence_is_mean_probability_of_the_decoded_tags():
    text, offsets = words(3)
    logits = torch.zeros((3, len(TAGS)))
    logits[0, TAGS.index("B-DATE")] = 4.0
    logits[1, TAGS.index("I-DATE")] = 3.0
    logits[2, TAGS.index("E-DATE")] = 2.0
    (span,) = decode_bioes(logits, offsets, text)
    others = len(TAGS) - 1
    expected = sum(math.exp(x) / (math.exp(x) + others) for x in (4, 3, 2)) / 3
    assert span.score == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("row", [0, 2])
def test_non_finite_emissions_are_rejected(bad, row):
    text, offsets = words(3)
    logits = emissions(["B-DATE", "I-DATE", "E-DATE"])
    logits[row, TAGS.index("O")] = bad
    with pytest.raises(RuntimeError, match="Invalid model logits"):
        decode_bioes(logits, offsets, text)


def confident(tags, unsure):
    """Emissions where the token at ``unsure`` keeps its tag with ~50% probability."""
    logits = emissions(tags)
    logits[unsure, TAGS.index("O")] = HIGH - 0.1
    return logits


@pytest.mark.parametrize("unsure,label", [(0, "MRN"), (1, "ID")])
def test_tokens_sharing_one_character_merge_into_one_span(unsure, label):
    # Byte-level BPE gives every byte of a multi-byte character the same offsets.
    text = "😀 0001"
    offsets = [(0, 1), (0, 1), (2, 6)]
    logits = confident(["S-ID", "S-MRN", "S-DATE"], unsure)
    assert decoded(logits, offsets, text) == [(0, 1, label), (2, 6, "DATE")]


@pytest.mark.parametrize("unsure,label", [(0, "MRN"), (1, "DATE")])
def test_overlapping_token_offsets_merge_with_higher_confidence_label(unsure, label):
    text = "ABCDEFG"
    offsets = [(0, 4), (2, 7)]
    (span,) = decode_bioes(confident(["S-DATE", "S-MRN"], unsure), offsets, text)
    assert (span.start, span.end, span.label) == (0, 7, label)
    assert span.score == pytest.approx(1.0)


# Window stitching ---------------------------------------------------------

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "clinical", "note"] + [
    f"w{index}" for index in range(40)
]


def word_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {word: index for index, word in enumerate(VOCAB)}
    core = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
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


class RecordingEncoder(torch.nn.Module):
    """Emits (token id, row position, per-id tag score) and records each row."""

    def __init__(self, tags_by_id=None, provenance=True):
        super().__init__()
        self.provenance = provenance
        self.config = SimpleNamespace(num_labels=len(TAGS), model_type="recording")
        self.tags_by_id = tags_by_id or {}
        self.rows = []

    def forward(self, input_ids, attention_mask):
        self.rows.extend(
            ids[mask.bool()].tolist() for ids, mask in zip(input_ids, attention_mask)
        )
        batch, length = input_ids.shape
        logits = torch.zeros((batch, length, len(TAGS)))
        logits[..., TAGS.index("O")] = 1.0
        for (row, position), token in zip(
            itertools.product(range(batch), range(length)), input_ids.flatten()
        ):
            tag = self.tags_by_id.get(int(token))
            if tag is not None:
                logits[row, position, TAGS.index(tag)] = 10.0
        if not self.provenance:
            return SimpleNamespace(logits=logits)
        # Two tag columns carry the provenance of each row (not decodable).
        logits[..., -1] = input_ids.float()
        logits[..., -2] = torch.arange(length).float()
        return SimpleNamespace(logits=logits)


def recording_detector(model, max_length=10, batch_size=2):
    return HFTokenDetector(
        None,
        device="cpu",
        batch_size=batch_size,
        model=model,
        tokenizer=word_tokenizer(),
        config={"encoder": ENCODER_KIND, "max_length": max_length},
    )


def test_window_layout_for_a_known_document():
    # 8 content tokens: overlap 2, stride 4; inner seams sit 2 tokens inside.
    assert list(window_jobs(14, 8)) == [
        (0, 8, 0, 6),
        (4, 12, 6, 10),
        (8, 14, 10, 14),
    ]
    assert list(window_jobs(8, 8)) == [(0, 8, 0, 8)]
    assert list(window_jobs(9, 8)) == [(0, 8, 0, 6), (4, 9, 6, 9)]


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_each_token_takes_emissions_from_its_owning_window_position(batch_size):
    model = RecordingEncoder()
    detector = recording_detector(model, batch_size=batch_size)
    text = " ".join(f"w{index}" for index in range(14))
    ((logits, offsets),) = detector.emissions([text])
    assert len(offsets) == 14
    # Token -> first content token of the owning window, written out by hand.
    owner = [0] * 6 + [4] * 4 + [8] * 4
    for token in range(14):
        assert logits[token, -1] == VOCAB.index(f"w{token}")
        # Row position = [CLS] + offset of the token inside its window.
        assert logits[token, -2] == 1 + token - owner[token]
    assert model.rows == [
        [VOCAB.index("[CLS]")]
        + [VOCAB.index(f"w{i}") for i in range(start, stop)]
        + [VOCAB.index("[SEP]")]
        for start, stop in [(0, 8), (4, 12), (8, 14)]
    ]


def test_entity_crossing_window_seams_decodes_once():
    tags = {VOCAB.index(f"w{i}"): "I-DATE" for i in range(4, 11)}
    tags[VOCAB.index("w3")] = "B-DATE"
    tags[VOCAB.index("w11")] = "E-DATE"
    detector = recording_detector(RecordingEncoder(tags, provenance=False))
    text = " ".join(f"w{index}" for index in range(14))
    (spans,) = detector.raw_batch([text])
    start, end = text.index("w3"), text.index("w11") + 3
    assert [(s.start, s.end, s.label) for s in spans] == [(start, end, "DATE")]


def test_documents_in_one_batch_do_not_share_windows():
    model = RecordingEncoder()
    detector = recording_detector(model, batch_size=4)
    texts = [" ".join(f"w{i}" for i in range(9)), "w20 w21", ""]
    results = detector.emissions(texts)
    assert [len(offsets) for _, offsets in results] == [9, 2, 0]
    first, second = results[0][0], results[1][0]
    assert first[:, -1].tolist() == [VOCAB.index(f"w{i}") for i in range(9)]
    assert first[:, -2].tolist() == [1, 2, 3, 4, 5, 6, 3, 4, 5]
    assert second[:, -1].tolist() == [VOCAB.index("w20"), VOCAB.index("w21")]
    assert model.rows[1] == [
        VOCAB.index(t) for t in ["[CLS]", "w4", "w5", "w6", "w7", "w8", "[SEP]"]
    ]
    assert len(model.rows) == 3  # the empty document is never encoded
