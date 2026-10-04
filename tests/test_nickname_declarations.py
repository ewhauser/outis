import unicodedata

import pytest

from clinical_deid.detection.context import nickname as nick
from clinical_deid.types import Span

def span(text, value, label='NAME', occurrence=0):
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(value, start + 1)
    return Span(start, start + len(value), label, .85, 'encoder')

def coordinates(values):
    return {(s.start, s.end, s.label) for s in values}

@pytest.mark.parametrize('alias', ['Summer', 'Level', 'Wait', 'Ready', 'Nearer', 'Asa', 'Élan', 'Aoife', 'Ó', 'A'])
@pytest.mark.parametrize('relation', ['is the preferred personal name of', 'was the chosen name of', 'is a name of', 'is the personal nickname used by'])
def test_declaration_and_nominal_object_require_literal_relation(alias, relation):
    text = f'{alias} {relation} Dara Wrenford; confirmed. The nurse asked {alias} to demonstrate the exercise. {alias} breathing is the topic.'
    result = coordinates(nick.contextual_person_names(text, [span(text, 'Dara Wrenford')]))
    expected = set() if relation == 'is a name of' else coordinates([span(text, alias), span(text, alias, occurrence=1)])
    assert result == expected

@pytest.mark.parametrize('prefix', ['Public teaching example: ', 'Device model: ', 'The specimen name: ', 'Not ', 'Never ', 'A quoted account: "', 'Medication identity: ', 'https://host.invalid/'])
def test_nonliteral_declaration_does_not_establish_alias(prefix):
    text = prefix + 'Summer is the preferred personal name of Dara Wrenford; discussed. The nurse asked Summer to demonstrate the exercise.'
    assert nick.contextual_person_names(text, [span(text, 'Dara Wrenford')]) == []

@pytest.mark.parametrize('tail', ['Summer breathing remains easy.', 'Summer is the preferred season for gardening.', 'The nurse asked Summer tablets to be counted.', 'The device asked Summer to begin.', 'The nurse asked Summers to begin.', 'The nurse asked Summer.opaque to begin.', 'The nurse asked Summer-B to begin.', 'The nurse asked a Summer to begin.', 'The nurse did not ask Summer to begin.', 'Public teaching example: The nurse asked Summer to begin.'])
def test_alias_is_not_unrestricted_word_propagation(tail):
    text = 'Summer is the preferred personal name of Dara Wrenford; confirmed. ' + tail
    assert coordinates(nick.contextual_person_names(text, [span(text, 'Dara Wrenford')])) == coordinates([span(text, 'Summer')])

@pytest.mark.parametrize('alias', ['He', 'Her', 'They', 'Someone', 'I'])
def test_pronominal_collision_needs_literal_proof_at_each_occurrence(alias):
    text = f'{alias} is the preferred personal name of Dara Wrenford; confirmed. The nurse asked {alias} to begin.'
    assert coordinates(nick.contextual_person_names(text, [span(text, 'Dara Wrenford')])) == coordinates([span(text, alias)])


def test_multiple_alias_owners_preserve_declarations_but_do_not_resolve_object():
    text = 'Summer is the preferred personal name of Dara Wrenford; confirmed. Summer is the personal name of Olwen Heath; confirmed. The nurse asked Summer to begin.'
    assert coordinates(nick.contextual_person_names(text, [span(text, 'Dara Wrenford'), span(text, 'Olwen Heath')])) == coordinates([span(text, 'Summer'), span(text, 'Summer', occurrence=1)])


def test_unicode_normalization_keeps_full_original_graphemes():
    alias = unicodedata.normalize('NFD', 'Élan')
    text = f'Élan is the personal name of Dara Wrenford; confirmed. The nurse asked {alias} to begin.'
    assert coordinates(nick.contextual_person_names(text, [span(text, 'Dara Wrenford')])) == coordinates([span(text, 'Élan'), span(text, alias)])


def test_declared_alias_inside_structured_owner_cannot_be_retagged():
    text = 'Summer is the preferred personal name of Dara Wrenford; confirmed. The nurse asked Summer to begin.'
    value = Span(0, text.index(';'), 'ID')
    assert nick.contextual_person_names(text, [span(text, 'Dara Wrenford'), value]) == []

