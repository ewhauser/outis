"""Synthetic regressions for punctuation merged into NAME token boundaries."""

import pytest

torch = pytest.importorskip("torch")

from clinical_deid.model import TAGS, decode_bioes  # noqa: E402


def _decode(text, offsets, labels=None):
    labels = labels or ["S-NAME"] * len(offsets)
    logits = torch.full((len(offsets), len(TAGS)), -40.0)
    for index, tag in enumerate(labels):
        logits[index, TAGS.index(tag)] = 40.0
    return [(span.start, span.end, span.label)
            for span in decode_bioes(logits, offsets, text)]


@pytest.mark.parametrize("name", ["R. T.", "é.", "E\u0301.", "Ω. K.", "古.", "Morgan Jr.", "Morgan Sr."])
@pytest.mark.parametrize("suffix", ["'s", "’s", "'S", "’S"])
@pytest.mark.parametrize("extra", [0, 1, 2])
def test_initial_or_suffix_possessive_keeps_the_name_period(name, suffix, extra):
    text = "For " + name + suffix + " appointment."
    start, end = 4, 4 + len(name)
    assert _decode(text, [(start, end + extra)]) == [(start, end, "NAME")]


@pytest.mark.parametrize("name", ["R.", "E\u0301.", "Morgan Jr."])
@pytest.mark.parametrize("continuation", ["sSurname", "s-Jones", "s’Connor", "s\u2010Venn", "s\u2011Venn", "s0", "s_", "s\u0301"])
@pytest.mark.parametrize("extra", [1, 2])
def test_initial_possessive_shape_inside_a_word_keeps_owned_characters(name, continuation, extra):
    text = name + "'" + continuation
    predicted_end = len(name) + extra
    actual = _decode(text, [(0, predicted_end)])
    assert len(actual) == 1
    assert actual[0][0] == 0
    assert actual[0][1] >= predicted_end


@pytest.mark.parametrize("opening,closing", [("'", "'"), ("‘", "’")])
@pytest.mark.parametrize("name", ["R. T.", "É. Ω.", "Morgan Jr.", "O'Sullivan", "D’Arcy"])
@pytest.mark.parametrize("include_open,include_close", [(False, True), (True, False), (True, True)])
def test_paired_quotes_adjacent_to_name_edges_are_excluded(opening, closing, name, include_open, include_close):
    text = "Label: " + opening + name + closing + "; verified."
    start = len("Label: ") + 1
    end = start + len(name)
    assert _decode(text, [(start - include_open, end + include_close)]) == [(start, end, "NAME")]


@pytest.mark.parametrize("opening,closing", [("'", "'"), ("‘", "’")])
@pytest.mark.parametrize("ending", [",", ".", "!", "?"])
def test_initial_name_in_punctuated_quoted_utterance_excludes_merged_opener(opening, closing, ending):
    text = "She said " + opening + "Sable is my name, and I don't use the chart spelling" + ending + closing
    start = len("She said ")
    assert _decode(text, [(start, start + len("'Sable"))]) == [(start + 1, start + len("'Sable"), "NAME")]


@pytest.mark.parametrize("name", ["'t Hooft", "'s-Gravenhage", "'s\u2011Gravenhage", "’t Hooft", "'Ali", "'Aïsha"])
def test_unmatched_leading_lexical_apostrophes_are_preserved(name):
    text = "Name: " + name + "; next appointment."
    assert _decode(text, [(6, 6 + len(name))]) == [(6, 6 + len(name), "NAME")]


@pytest.mark.parametrize("name", ["'t Hooft", "'s-Gravenhage", "'s\u2011Gravenhage"])
def test_one_letter_lexical_elision_is_preserved_despite_a_later_quote(name):
    text = "Name: " + name + "; comment ended.'"
    assert _decode(text, [(6, 6 + len(name))]) == [(6, 6 + len(name), "NAME")]


def test_distant_plural_possessive_is_not_evidence_for_leading_quote():
    text = "Name: 'Ali; see James' chart."
    assert _decode(text, [(6, 10)]) == [(6, 10, "NAME")]


def test_competing_opener_does_not_strip_a_lexical_apostrophe():
    text = "Name: 'Ali; she said 'the appointment is soon.'"
    assert _decode(text, [(6, 10)]) == [(6, 10, "NAME")]


@pytest.mark.parametrize("boundary", ["\r", "\n", "\r\n", "\u0085", "\u2028", "\u2029"])
def test_quote_pairing_does_not_cross_a_line_or_paragraph(boundary):
    text = "Name: 'Ali" + boundary + "The unrelated sentence ended.'"
    assert _decode(text, [(6, 10)]) == [(6, 10, "NAME")]


@pytest.mark.parametrize("label", ["EMAIL", "FACILITY", "LOCATION", "ID"])
def test_name_quote_repair_does_not_rewrite_other_entity_labels(label):
    text = "'O'Sullivan'"
    assert _decode(text, [(0, len(text))], ["S-" + label]) == [(0, len(text), label)]


def test_adjacent_detected_entity_keeps_its_own_offsets():
    text = "'Arden' 42"
    assert _decode(text, [(0, 7), (8, 10)], ["S-NAME", "S-MRN"]) == [
        (1, 6, "NAME"), (8, 10, "MRN")
    ]


@pytest.mark.parametrize("hyphen", ["-", "\u2010", "\u2011"])
def test_ambiguous_quoted_prefix_of_a_compound_keeps_owned_punctuation(hyphen):
    text = "'Arden'" + hyphen + "Venn"
    assert _decode(text, [(0, 7)]) == [(0, 7, "NAME")]


def test_abutting_detected_entity_is_not_reinterpreted_as_quote_punctuation():
    text = "'Arden'42"
    assert _decode(text, [(0, 7), (7, 9)], ["S-NAME", "S-MRN"]) == [
        (0, 7, "NAME"), (7, 9, "MRN")
    ]
