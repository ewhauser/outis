"""Original adversarial controls for bounded context truncation."""
import pytest
from clinical_deid import nickname_context as nick
from clinical_deid.types import Span


@pytest.mark.parametrize('prefix', [
    'The patient denies that ', 'It is untrue that ', 'The sentence ',
    'The false assertion is ',
])
def test_padding_cannot_erase_unasserted_declaration_context(prefix):
    text = prefix + ' ' * 220 + 'May is the personal name of Dara Wrenford. The nurse asked May to demonstrate the exercise.'
    start = text.index('Dara Wrenford')
    assert nick.contextual_person_names(text, [Span(start,start+len('Dara Wrenford'),'NAME')]) == []


@pytest.mark.parametrize('heading', [
    'False' + ' ' * 220 + 'claim',
    'Arbitrary assessment metadata ' * 8,
])
def test_long_heading_requires_same_explicit_authority_as_short_heading(heading):
    text = heading + ':\nMay is the personal name of Dara Wrenford. The nurse asked May to demonstrate the exercise.'
    start = text.index('Dara Wrenford')
    assert nick.contextual_person_names(text, [Span(start,start+len('Dara Wrenford'),'NAME')]) == []
