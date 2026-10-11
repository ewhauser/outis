import pytest

torch = pytest.importorskip("torch")

from outis.detection import decoding  # noqa: E402
from outis.detection.decoding import TAGS, decode_bioes  # noqa: E402


def logits_for(tags):
    scores = torch.full((len(tags), len(TAGS)), -10.0)
    for i, tag in enumerate(tags):
        scores[i, TAGS.index(tag)] = 10.0
    return scores


def test_cached_constraints_survive_repeated_decodes():
    snapshot = [
        t.clone() for t in (decoding._LEGAL, decoding._INITIAL, decoding._FINAL)
    ]
    text = "Ann Lee"
    offsets = [(0, 3), (4, 7)]
    first = decode_bioes(logits_for(["B-NAME", "E-NAME"]), offsets, text)
    second = decode_bioes(logits_for(["B-NAME", "E-NAME"]), offsets, text)
    assert first == second
    assert [(s.start, s.end, s.label) for s in first] == [(0, 7, "NAME")]
    for before, after in zip(
        snapshot, (decoding._LEGAL, decoding._INITIAL, decoding._FINAL)
    ):
        assert torch.equal(before, after)


def test_cached_constraints_encode_bioes_legality():
    legal, initial, final = decoding._LEGAL, decoding._INITIAL, decoding._FINAL
    tag = TAGS.index
    assert legal.shape == (len(TAGS), len(TAGS))
    assert legal[tag("O"), tag("B-NAME")] == 0
    assert legal[tag("B-NAME"), tag("I-NAME")] == 0
    assert legal[tag("I-NAME"), tag("E-NAME")] == 0
    assert legal[tag("S-NAME"), tag("O")] == 0
    assert legal[tag("O"), tag("I-NAME")] == float("-inf")
    assert legal[tag("B-NAME"), tag("E-DATE")] == float("-inf")
    assert legal[tag("B-NAME"), tag("O")] == float("-inf")
    assert initial[tag("I-NAME")] == float("-inf")
    assert initial[tag("S-NAME")] == 0
    assert final[tag("B-NAME")] == float("-inf")
    assert final[tag("E-NAME")] == 0


def test_wrong_tag_count_fails_closed():
    with pytest.raises(RuntimeError, match="tag logits"):
        decode_bioes(torch.zeros(1, len(TAGS) - 1), [(0, 1)], "a")
