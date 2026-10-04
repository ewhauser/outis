"""Independent, self-authored semantic controls; no clinical corpus was read."""
import unicodedata

import pytest
from clinical_deid.types import Span

from clinical_deid.detection.context import nickname as nick


def span(text, value, occurrence=0, label='NAME'):
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(value, start + 1)
    return Span(start, start + len(value), label, .85, 'encoder')


def added(text, seeds):
    return {(s.start, s.end, s.label) for s in nick.contextual_person_names(text, seeds)}


def expected(text, values):
    return {(s.start, s.end, s.label) for value, occurrence in values for s in [span(text, value, occurrence)]}


@pytest.mark.parametrize('prefix', [
    'It is untrue that ', 'The patient denies that ', 'The sentence ',
    'The grammar exercise contains the string ', 'Do not assume that ',
    'There is no evidence that ', 'We dispute the assertion that ',
    'Public teaching example:\n', 'Device script:\n',
    'A quoted account: "\n', 'A quoted account: "The exercise follows. ',
    'An invented example:\n', 'Medication label:\n',
    'Published character biography:\n',
])
def test_nonliteral_declaration_cannot_seed_recovery(prefix):
    text = prefix + 'May is the preferred personal name of Dara Wrenford. The nurse asked May to begin.'
    assert added(text, [span(text, 'Dara Wrenford')]) == set()


@pytest.mark.parametrize('prefix', [
    'Public teaching example:\n', 'Device script:\n',
    'A quoted account: "\n', 'A quoted account: "The exercise follows. ',
    'An invented example:\n', 'Medication label:\n',
    'Published character biography:\n',
])
def test_nonhuman_source_does_not_inherit_declared_alias(prefix):
    text = 'May is the preferred personal name of Dara Wrenford. ' + prefix + 'The nurse asked May to begin.'
    assert added(text, [span(text, 'Dara Wrenford')]) == expected(text, [('May', 0)])


def test_month_range_is_not_a_human_infinitive_predicate():
    text = ('May is the preferred personal name of Dara Wrenford. '
            'The nurse helped May to June transitions by revising the discharge schedule.')
    assert added(text, [span(text, 'Dara Wrenford')]) == expected(text, [('May', 0)])


@pytest.mark.parametrize('alias', ['May', 'Autumn', 'Robin', 'Sol', 'Élan', 'İpek', 'Straße', 'O’Neal', 'An‑Li'])
@pytest.mark.parametrize('action', ['asked', 'invited', 'encouraged', 'reminded', 'helped'])
def test_literal_human_object_retains_original_unicode_boundaries(alias, action):
    subsequent = unicodedata.normalize('NFD', alias)
    text = f'{alias} is the personal name of Dara Wrenford. The nurse {action} {subsequent} to demonstrate the exercise.'
    result = added(text, [span(text, 'Dara Wrenford')])
    first = text.index(alias)
    later = text.index(subsequent, first + len(alias))
    assert result == {(first, first + len(alias), 'NAME'), (later, later + len(subsequent), 'NAME')}


@pytest.mark.parametrize('suffix', [' Jones', '-Jones', '.local', '/index', '\u200dJones', '\u0301'])
def test_owner_must_be_full_identity_not_known_prefix(suffix):
    text = f'May is the personal name of Dara Wrenford{suffix}. The nurse asked May to demonstrate the exercise.'
    assert added(text, [span(text, 'Dara Wrenford')]) == set()


@pytest.mark.parametrize('label', ['ACCOUNT', 'MRN', 'ID', 'EMAIL', 'PHONE', 'URL', 'ADDRESS'])
def test_structured_owner_remains_authoritative(label):
    text = 'May is the personal name of Dara Wrenford. The nurse asked May to demonstrate the exercise.'
    entire = Span(0, len(text), label, .95, 'encoder')
    assert added(text, [span(text, 'Dara Wrenford'), entire]) == set()


def test_competing_declarations_do_not_resolve_later_object():
    text = ('May is the personal name of Dara Wrenford. '
            'May is the personal name of Fern Halwick. '
            'The nurse asked May to demonstrate the exercise.')
    assert added(text, [span(text, 'Dara Wrenford'), span(text, 'Fern Halwick')]) == expected(text, [('May', 0), ('May', 1)])


@pytest.mark.parametrize('prefix', [
    'False claim: ', 'Negated assertion: ', 'Fabricated assertion: ',
    'Incorrect statement: ', 'Unverified claim: ', 'Account value: ',
    'URL parameter: ', 'JSON payload: ',
    "A quoted account: '\n", "A quoted account: 'The exercise follows. ",
    'Public teaching example follows. ',
])
def test_arbitrary_heading_or_outer_quote_is_not_a_literal_identity_scope(prefix):
    text = prefix + 'May is the personal name of Dara Wrenford. The nurse asked May to demonstrate the exercise.'
    assert added(text, [span(text, 'Dara Wrenford')]) == set()


@pytest.mark.parametrize('prefix', [
    "A quoted account: '\n", "A quoted account: 'The exercise follows. ",
    'Public teaching example follows. ',
])
def test_later_source_sentence_or_single_quote_cannot_inherit_person_identity(prefix):
    text = 'May is the personal name of Dara Wrenford. ' + prefix + 'The nurse asked May to demonstrate the exercise.'
    assert added(text, [span(text, 'Dara Wrenford')]) == expected(text, [('May', 0)])


@pytest.mark.parametrize('prefix', [
    'False claim:\n', 'Negated assertion:\n', 'Fabricated assertion:\n',
    'Incorrect statement:\n', 'Unverified claim:\n', 'Account value:\n',
    'URL parameter:\n', 'JSON payload:\n',
])
def test_line_wrapping_does_not_change_nonliteral_or_structured_scope(prefix):
    text = prefix + 'May is the personal name of Dara Wrenford. The nurse asked May to demonstrate the exercise.'
    assert added(text, [span(text, 'Dara Wrenford')]) == set()


def test_line_wrapped_declaration_tail_is_not_complete_at_owner_linebreak():
    text = 'May is the personal name of Dara Wrenford\nin the teaching exercise. The nurse asked May to demonstrate the exercise.'
    assert added(text, [span(text, 'Dara Wrenford')]) == set()


@pytest.mark.parametrize('alias,ordinary', [
    ('Sit', 'The therapist helped Sit to Stand class participants practise safely.'),
    ('sit', 'The therapist helped sit to stand class participants practise safely.'),
    ('Return', 'The nurse helped Return to Practice participants complete the exercises.'),
    ('return', 'The nurse helped return to practice participants complete the exercises.'),
])
def test_clinical_noun_phrase_is_not_person_plus_infinitive(alias, ordinary):
    text = f'{alias} is the personal name of Dara Wrenford. ' + ordinary
    assert added(text, [span(text, 'Dara Wrenford')]) == expected(text, [(alias, 0)])


@pytest.mark.parametrize('distance', [170, 1900, 2000, 2001, 8192])
def test_alias_object_recovery_distance_is_measured_on_original_span_end(distance):
    prefix = 'May is the personal name of Dara Wrenford. '
    tail = 'The nurse asked May to demonstrate the exercise.'
    object_end_in_tail = tail.index('May') + len('May')
    padding = distance + len('May') - len(prefix) - object_end_in_tail
    text = prefix + ' ' * padding + tail
    values = [('May', 0), ('May', 1)] if distance <= 2000 else [('May', 0)]
    assert added(text, [span(text, 'Dara Wrenford')]) == expected(text, values)


@pytest.mark.parametrize('later_alias', ['May', 'MAY', 'may'])
def test_case_variant_competing_owner_is_still_ambiguous(later_alias):
    text = (f'May is the personal name of Dara Wrenford. '
            f'{later_alias} is the personal name of Fern Halwick. '
            f'The nurse asked May to demonstrate the exercise.')
    out = added(text, [span(text, 'Dara Wrenford'), span(text, 'Fern Halwick')])
    second_start = text.index(later_alias, len('May'))
    assert out == {(0,3,'NAME'),(second_start,second_start+len(later_alias),'NAME')}


@pytest.mark.parametrize('clause', [
    'The patient may return to light exercise.',
    'Return to baseline was slow.',
    'Sit to stand transfer was performed safely.',
    'Summer exercise plans were reviewed.',
    'The nurse described May to June as a period of improvement.',
    'The nurse asked for May to June appointment counts.',
])
def test_declared_words_in_ordinary_clinical_grammar_stay_unmodified(clause):
    text = ('May is the personal name of Dara Wrenford. '
            'Return is the personal name of Fern Halwick. '
            'Sit is the personal name of Rowan Velwick. '
            'Summer is the personal name of Nia Bracken. ' + clause)
    out = added(text,[span(text,n) for n in ['Dara Wrenford','Fern Halwick','Rowan Velwick','Nia Bracken']])
    assert out == expected(text,[(name,0) for name in ['May','Return','Sit','Summer']])


def test_repeated_same_owner_declarations_emit_one_evidence_record_per_output_span():
    text = ('May is the personal name of Dara Wrenford. '
            'May is the preferred name of Dara Wrenford. '
            'The nurse asked May to demonstrate the exercise.')
    evidence = []
    seeds = [span(text, 'Dara Wrenford'), span(text, 'Dara Wrenford', 1)]
    out = nick.contextual_person_names(text, seeds, evidence=evidence)
    assert {(s.start,s.end,s.label) for s in out} == expected(text, [('May',0),('May',1),('May',2)])
    assert len(evidence) == len(out)
    assert len({(e['start'],e['end']) for e in evidence}) == len(evidence)
