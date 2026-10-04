"""Synthetic fixtures for occurrence ownership; no encoder or corpus access."""
from dataclasses import replace as with_fields
from datetime import date
import re

import pytest

from clinical_deid.entities import Entity
from clinical_deid.replacement.surrogates import replace
from clinical_deid.replacement.nicknames import nickname_bindings
from clinical_deid.service import Deidentifier
from clinical_deid.replacement.transform import link_entities, prepare_context, reidentify, render

SECRET = b"candidate-nickname-regression-secret-at-least-32-bytes"
PATIENT = "invented-patient"


def marked(value):
    text, spans, cursor = "", [], 0
    for match in re.finditer(r"\{\{(NAME|DATE)\|([^}]+)\}\}", value):
        text += value[cursor:match.start()]
        start = len(text)
        text += match[2]
        spans.append(Entity(start, len(text), match[1], .6, "learned-recovery"))
        cursor = match.end()
    return text + value[cursor:], spans


DECLARATION = "{{NAME|Pip}} is the preferred nickname of {{NAME|Mira Voss}}."


def synthetic(text, spans, **kwargs):
    return render(text, spans, options={"type": "synthetic"}, secret=SECRET,
                  patient_id=PATIENT, **kwargs)


@pytest.mark.parametrize("alias,owner", [("Pip", "Mira Voss"), ("Žu", "Elin Marlow"),
                                         ("Lumi", "Oren Vale")])
@pytest.mark.parametrize("keyed", [False, True])
def test_declaration_uses_owner_key_and_given_name(alias, owner, keyed):
    text, spans = marked(f"{{{{NAME|{alias}}}}} is the preferred nickname of {{{{NAME|{owner}}}}}.")
    result = render(text, spans, options={"type": "synthetic", "entity_ids": "keyed" if keyed else "numbered"},
                    secret=SECRET, patient_id=PATIENT)
    a, b = result["entities"]
    assert a["entity_id"] == b["entity_id"]
    assert a["replacement"] == b["replacement"].split()[0]
    legacy = replace(text, spans, mode="synthetic", secret=SECRET, patient_id=PATIENT)
    assert legacy["text"] == result["text"]


def test_each_supported_followup_needs_its_own_evidence():
    text, spans = marked(DECLARATION + " The nurse asked {{NAME|Pip}} to choose the next activity."
                         " The {{NAME|Pip}} software is offline. Pip is printed on the screen.")
    result = synthetic(text, spans)
    a, owner, later, software = result["entities"]
    assert a["entity_id"] == owner["entity_id"] == later["entity_id"]
    assert a["replacement"] == later["replacement"] == owner["replacement"].split()[0]
    assert software["entity_id"] != owner["entity_id"]
    assert result["text"].endswith("Pip is printed on the screen.")


def test_repeated_individually_proved_aliases_keep_same_owner():
    text, spans = marked(DECLARATION + " The nurse asked {{NAME|Pip}} to choose the next activity."
                         " The therapist invited {{NAME|Pip}} to walk safely.")
    a, owner, later, again = synthetic(text, spans)["entities"]
    assert len({e["entity_id"] for e in (a, owner, later, again)}) == 1
    assert a["replacement"] == later["replacement"] == again["replacement"] == owner["replacement"].split()[0]


def test_two_owners_sharing_alias_never_merge_and_unproved_reuse_stays_unresolved():
    text, spans = marked(DECLARATION + " {{NAME|Pip}} is the preferred nickname of {{NAME|Nola Reed}}."
                         " The nurse asked {{NAME|Pip}} to choose the next activity.")
    a, owner_a, b, owner_b, unresolved = synthetic(text, spans)["entities"]
    assert a["entity_id"] == owner_a["entity_id"]
    assert b["entity_id"] == owner_b["entity_id"]
    assert a["entity_id"] != b["entity_id"]
    assert a["replacement"] == owner_a["replacement"].split()[0]
    assert b["replacement"] == owner_b["replacement"].split()[0]
    assert unresolved["entity_id"] not in {a["entity_id"], b["entity_id"]}


@pytest.mark.parametrize("note", [
    "Software registry:\n" + DECLARATION,
    "Fictional story: " + DECLARATION,
    "It is false that " + DECLARATION,
    "The manual states: “" + DECLARATION + "”",
    DECLARATION + " That nickname declaration was retracted as incorrect.",
    DECLARATION + " That statement was retracted.",
    DECLARATION + " The preceding statement is false.",
    "Retracted identity statement:\n" + DECLARATION,
    "{{NAME|Pip}} is not the preferred nickname of {{NAME|Mira Voss}}.",
    "{{NAME|Pip}} is the preferred nickname of the software package {{NAME|Mira Voss}}.",
])
def test_invalid_witnesses_cannot_establish_ownership_but_all_spans_are_replaced(note):
    text, spans = marked(note)
    result = synthetic(text, spans)
    assert not nickname_bindings(text, spans)
    assert len({e["entity_id"] for e in result["entities"]}) == 2
    assert len(result["entities"]) == len(spans)


def test_former_usage_alone_does_not_retract_identity():
    text, spans = marked(DECLARATION.replace(" is ", " was ") + " The previous nickname is no longer used.")
    a, b = synthetic(text, spans)["entities"]
    assert a["entity_id"] == b["entity_id"]


@pytest.mark.parametrize("caption", ["Other patient record: Neri Cole.", "Patient: Neri Cole.",
                                      "Another person's chart: Neri Cole."])
def test_followup_proof_does_not_cross_patient_record_transition(caption):
    text, spans = marked(DECLARATION + "\n" + caption.replace("Neri Cole", "{{NAME|Neri Cole}}")
                         + "\nThe therapist invited {{NAME|Pip}} to walk safely.")
    alias, owner, other, later = synthetic(text, spans)["entities"]
    assert alias["entity_id"] == owner["entity_id"]
    assert later["entity_id"] not in {owner["entity_id"], other["entity_id"]}


def test_no_retained_owner_or_no_retained_alias_means_no_new_span():
    text, spans = marked(DECLARATION)
    assert not nickname_bindings(text, spans[:1])
    assert not nickname_bindings(text, spans[1:])
    result = synthetic(text, spans[1:])
    assert len(result["entities"]) == 1
    assert result["text"].startswith("Pip is")


def test_source_tags_scores_and_supplied_ids_are_not_ownership_authority():
    text, spans = marked("{{NAME|Mira Voss}} met {{NAME|Pip}}.")
    spans = [with_fields(s, score=1.0, source="owner:mira voss", entity_id="same") for s in spans]
    result = synthetic(text, spans)
    assert not nickname_bindings(text, spans)
    assert result["entities"][0]["replacement"].split()[0] != result["entities"][1]["replacement"]


def test_conflicting_supplied_ids_are_preserved_and_proof_abstains():
    text, spans = marked(DECLARATION)
    spans = [with_fields(s, entity_id="provided" + str(i)) for i, s in enumerate(spans)]
    result = synthetic(text, spans)
    assert not nickname_bindings(text, spans)
    assert [e["entity_id"] for e in result["entities"]] == [s.entity_id for s in spans]


def test_supplied_alias_id_bound_to_other_full_person_cannot_transfer_to_owner():
    text, spans = marked(DECLARATION + " {{NAME|Neri Cole}} attended separately.")
    spans = [with_fields(s, entity_id="other-person" if i in {0, 2} else None)
             for i, s in enumerate(spans)]
    assert not nickname_bindings(text, spans)
    alias, owner, other = synthetic(text, spans)["entities"]
    assert alias["entity_id"] == other["entity_id"] == "other-person"
    assert owner["entity_id"] != "other-person"


def test_two_alias_proofs_with_same_supplied_id_cannot_merge_distinct_owners():
    text, spans = marked(DECLARATION + " {{NAME|Lumi}} is the preferred nickname of {{NAME|Oren Vale}}.")
    spans = [with_fields(s, entity_id="shared" if i in {0, 2} else None) for i, s in enumerate(spans)]
    assert not nickname_bindings(text, spans)
    result = synthetic(text, spans)["entities"]
    assert result[1]["entity_id"] != result[3]["entity_id"]


def test_prepared_bindings_are_immutable_and_shared_mapping_requires_exact_offsets():
    first, a = marked(DECLARATION)
    second, b = marked("The nurse asked {{NAME|Pip}} to choose the next activity.")
    offset = len(first) + 1
    context = first + "\n" + second
    spans = a + [with_fields(s, start=s.start + offset, end=s.end + offset) for s in b]
    prepared = prepare_context(context, spans)
    with pytest.raises(TypeError):
        prepared.bindings[(1, 2)] = None
    result = synthetic(second, b, prepared_context=prepared, context_offset=offset)
    owner = synthetic(first, a)["entities"][1]
    assert result["entities"][0]["replacement"] == owner["replacement"].split()[0]
    with pytest.raises(ValueError, match="exact text segment"):
        synthetic(second, b, prepared_context=prepared, context_offset=0)


class Injected:
    def __init__(self, notes):
        self.table = dict(notes)
        offset, spans = 0, []
        for text, local in notes:
            spans.extend(with_fields(s, start=s.start + offset, end=s.end + offset) for s in local)
            offset += len(text) + 1
        self.table["\n".join(text for text, _ in notes)] = spans

    def detect_batch(self, texts):
        return [self.table[text] for text in texts]


@pytest.mark.parametrize("sharing", [{}, {"shared_entities": True}, {"link_batch": True}])
def test_service_sharing_uses_character_offsets_and_only_explicit_context(sharing):
    notes = [marked(DECLARATION), marked("The nurse asked {{NAME|Pip}} to choose the next activity.")]
    service = Deidentifier(detector=Injected(notes), secret=SECRET)
    requests = [{"text": text, "mode": "synthetic", "patient_id": PATIENT,
                 "replacement": {"type": "synthetic", "entity_ids": "keyed"}} for text, _ in notes]
    first, second = service.deidentify_batch(requests, **sharing)
    a, owner = first["entities"]
    later = second["entities"][0]
    assert a["entity_id"] == owner["entity_id"]
    assert (later["entity_id"] == owner["entity_id"]) == bool(sharing)
    assert (later["replacement"] == owner["replacement"].split()[0]) == bool(sharing)


def test_dates_and_clinical_text_are_unchanged_by_identity_link():
    text, spans = marked(DECLARATION + " HbA1c 7.4%; review {{DATE|2024-02-20}} to {{DATE|2024-03-03}}.")
    result = synthetic(text, spans, date_shift_days=17)
    left, right = result["entities"][-2:]
    assert left["replacement"] == "2024-03-08"
    assert right["replacement"] == "2024-03-20"
    assert (date.fromisoformat(right["replacement"]) - date.fromisoformat(left["replacement"])).days == 12
    assert "HbA1c 7.4%; review " in result["text"]
    assert result["preservation"]["date_intervals_preserved"]


def test_markers_masks_and_reversible_index_contract_remain_intact():
    text, spans = marked(DECLARATION + " The nurse asked {{NAME|Pip}} to choose the next activity.")
    result = render(text, spans, options={"pattern": "[{label}_{id}_{index}]", "include_mapping": True})
    assert reidentify(result["text"], result["mappings"], restore_sensitive=True)["text"] == text
    mask = render(text, spans, options={"type": "mask"})
    assert mask["entities"][0]["replacement"] == "***"
    with pytest.raises(ValueError, match="Ambiguous restoration"):
        render(text, spans, options={"include_mapping": True})


def test_literal_first_and_family_aliases_preserve_existing_keys():
    text, spans = marked("{{NAME|Mira Voss}} / {{NAME|Mira}} / {{NAME|Voss}}")
    linked = link_entities(text, spans)
    assert {s.entity_id for s in linked} == {"e1"}
    full, first, last = synthetic(text, spans)["entities"]
    assert full["replacement"] == first["replacement"] + " " + last["replacement"]


@pytest.mark.parametrize("extended", [False, True])
def test_nickname_equal_to_owner_synthetic_first_uses_safe_family_name(extended):
    text, spans = marked(DECLARATION.replace("Pip", "Reese"))
    result = (synthetic(text, spans) if extended else
              replace(text, spans, mode="synthetic", secret=SECRET, patient_id=PATIENT))
    alias, owner = result["entities"]
    assert owner["replacement"] == "Reese Fern"
    assert alias["replacement"] == "Fern"
    assert "reese" not in alias["replacement"].casefold().split()
    assert not result["warnings"]
    if extended:
        assert alias["entity_id"] == owner["entity_id"]


@pytest.mark.parametrize("extended", [False, True])
@pytest.mark.parametrize("first,last", [("Reese", "Reese"), ("Robin Reese", "Reese Fern"),
                                        ("Reese", "")])
def test_pathological_given_and_family_collision_redacts_with_warning(monkeypatch, extended, first, last):
    import clinical_deid.replacement.surrogates as replacement_module

    monkeypatch.setattr(replacement_module, "_fictional_name", lambda *_: (first, last))
    text, spans = marked(DECLARATION.replace("Pip", "Reese"))
    result = (synthetic(text, spans) if extended else
              replace(text, spans, mode="synthetic", secret=SECRET, patient_id=PATIENT))
    assert result["entities"][0]["replacement"] == "[NAME]"
    assert result["warnings"] == [
        "Entity 0 (NAME): Synthetic owner name contains the original nickname; NAME redacted."
    ]


@pytest.mark.parametrize("nickname", ["Reese", "REESE", "Ｒｅｅｓｅ"])
@pytest.mark.parametrize("extended", [False, True])
def test_contained_normalized_nickname_token_uses_family_alias(monkeypatch, nickname, extended):
    import clinical_deid.replacement.surrogates as replacement_module

    monkeypatch.setattr(replacement_module, "_fictional_name", lambda *_: ("Robin Reese", "Fern"))
    text, spans = marked(DECLARATION.replace("Pip", nickname))
    result = (synthetic(text, spans) if extended else
              replace(text, spans, mode="synthetic", secret=SECRET, patient_id=PATIENT))
    alias, owner = result["entities"]
    assert owner["replacement"] == "Robin Reese Fern"
    assert alias["replacement"] == "Fern"
    assert "reese" not in replacement_module._name_words(replacement_module._name_identity(alias["replacement"])[1])
    if extended:
        assert alias["entity_id"] == owner["entity_id"]


def test_http_synthetic_response_uses_same_proof_for_ids_and_rendering():
    from fastapi.testclient import TestClient
    from clinical_deid.api import create_app

    note = marked(DECLARATION)
    service = Deidentifier(detector=Injected([note]), secret=SECRET)
    with TestClient(create_app(service, api_key="fixture-key")) as client:
        response = client.post("/v1/deidentify", headers={"X-API-Key": "fixture-key"},
                               json={"text": note[0], "mode": "synthetic", "patient_id": PATIENT,
                                     "replacement": {"type": "synthetic"}})
    assert response.status_code == 200
    a, b = response.json()["entities"]
    assert a["entity_id"] == b["entity_id"]
    assert a["replacement"] == b["replacement"].split()[0]


def test_structured_shared_fields_preserve_occurrence_offsets_and_nulls():
    from clinical_deid.formats.structured import process_structured

    notes = [marked(DECLARATION), marked("The nurse asked {{NAME|Pip}} to choose the next activity.")]
    service = Deidentifier(detector=Injected([("first", []), notes[0], ("second", []), notes[1],
                                             ("empty", [])]), secret=SECRET)
    result = process_structured(service, {"first": notes[0][0], "second": notes[1][0], "empty": None},
                                options={"mode": "synthetic", "patient_id": PATIENT,
                                         "replacement": {"type": "synthetic"}})
    alias, owner, later = result["entities"]
    assert alias["entity_id"] == owner["entity_id"] == later["entity_id"]
    assert alias["replacement"] == later["replacement"] == owner["replacement"].split()[0]
    assert result["data"]["empty"] is None
