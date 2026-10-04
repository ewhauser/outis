from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from clinical_deid.entities import Entity
from clinical_deid.replacement.surrogates import replace
from clinical_deid.replacement.transform import (
    ReidentifyRequest,
    ReplacementOptions,
    link_entities,
    reidentify,
    render,
)

SECRET = b"0123456789abcdef0123456789abcdef"


def spans_for(text, values, label="NAME"):
    spans = []
    cursor = 0
    for value in values:
        start = text.index(value, cursor)
        spans.append(Entity(start, start + len(value), label, source="test"))
        cursor = start + len(value)
    return spans


def test_unicode_output_offsets_and_default_metadata_do_not_copy_originals():
    text = "🙂 Åsa Vale met Jo. A\u030asa returned."
    spans = spans_for(text, ["Åsa Vale", "Jo", "A\u030asa"])
    result = render(text, spans, options=ReplacementOptions())
    assert result["text"] == "🙂 [NAME_e1] met [NAME_e2]. [NAME_e1] returned."
    for entity in result["entities"]:
        assert (
            result["text"][entity["processed_start"] : entity["processed_end"]]
            == entity["replacement"]
        )
        assert "original" not in entity
    assert "mappings" not in result


def test_alias_ownership_uses_all_names_and_preserves_ambiguity():
    values = ["Jane", "Jane Smith", "Jane Brown", "Jane", "Smith"]
    text = " | ".join(values)
    linked = link_entities(text, spans_for(text, values))
    ids = [span.entity_id for span in linked]
    assert ids[0] not in ids[1:]
    assert ids[3] not in ids[:3]
    assert ids[1] == ids[4]
    assert ids[1] != ids[2]


@pytest.mark.parametrize(
    "full,alias",
    [
        ("Jane Mary Smith", "Smith"),
        ("J. Smith", "Smith"),
        ("Ren Ren", "Ren"),
        ("Åsa de Valen", "Valen"),
    ],
)
def test_uncertain_name_components_are_never_linked(full, alias):
    text = full + " | " + alias
    entities = link_entities(text, spans_for(text, [full, alias]))
    assert entities[0].entity_id != entities[1].entity_id


def test_unicode_spelling_and_full_name_order_link_to_one_identity():
    values = ["Åsa O’Neri‐Rios", "O'Neri-Rios, A\u030asa", "Dr. Åsa", "O'Neri-Rios"]
    text = " | ".join(values)
    linked = link_entities(text, spans_for(text, values))
    assert len({span.entity_id for span in linked}) == 1


def test_opaque_ids_preserve_case_and_whitespace():
    text = "AB12 ab12 AB12 "
    spans = [Entity(0, 4, "ID"), Entity(5, 9, "ID"), Entity(10, 15, "ID")]
    assert len({span.entity_id for span in link_entities(text, spans)}) == 3


def test_keyed_ids_are_patient_scoped_and_stable_across_positions():
    kwargs = {"secret": SECRET, "patient_id": "patient-a"}
    original = link_entities("Jane Smith", [Entity(0, 10, "NAME")], **kwargs)[
        0
    ].entity_id
    assert original.startswith("k")
    assert (
        original
        == link_entities("🙂 Jane Smith", [Entity(2, 12, "NAME")], **kwargs)[
            0
        ].entity_id
    )
    assert (
        original
        != link_entities(
            "Jane Smith", [Entity(0, 10, "NAME")], secret=SECRET, patient_id="patient-b"
        )[0].entity_id
    )
    assert (
        original
        != link_entities(
            "Jane Smith",
            [Entity(0, 10, "NAME")],
            secret=b"x" * 32,
            patient_id="patient-a",
        )[0].entity_id
    )


def test_assigned_ids_are_preserved_and_new_numbered_ids_avoid_collision():
    spans = [Entity(0, 4, "NAME", entity_id="e1"), Entity(5, 8, "NAME")]
    linked = link_entities("Jane Bob", spans)
    assert [span.entity_id for span in linked] == ["e1", "e2"]
    keyed = link_entities("Jane Bob", spans, secret=SECRET, patient_id="a")
    assert keyed[0].entity_id == "e1"
    assert keyed[1].entity_id.startswith("k")


@pytest.mark.parametrize(
    "options",
    [
        {"type": "synthetic"},
        {"entity_ids": "keyed"},
        {"per_type": {"NAME": "synthetic"}},
    ],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"secret": b"short", "patient_id": "a"},
        {"secret": SECRET},
        {"secret": SECRET, "patient_id": " "},
    ],
)
def test_synthetic_and_keyed_options_require_key_and_patient_even_for_empty_text(
    options, kwargs
):
    with pytest.raises(ValueError):
        render("", [], options=options, **kwargs)


@pytest.mark.parametrize(
    "options",
    [
        {"unexpected": True},
        {"type": "keep"},
        {"mask_character": ""},
        {"mask_character": "ab"},
        {"mask_character": "\n"},
        {"per_type": {"Name": "mask"}},
        {"per_type": {"NAME": "delete"}},
        {"pattern": "{label.__class__}"},
        {"pattern": "{original}"},
        {"pattern": "{id:10000000}"},
        {"pattern": "{id!r}"},
        {"pattern": "{"},
        {"pattern": " "},
        {"include_mapping": "true"},
    ],
)
def test_replacement_options_reject_unknown_fields_and_unsafe_templates(options):
    with pytest.raises(ValidationError):
        ReplacementOptions.model_validate(options)


@pytest.mark.parametrize(
    "metadata",
    [
        {"score": float("nan")},
        {"score": float("inf")},
        {"score": True},
        {"score": 2},
        {"source": 1},
        {"label": "bad label"},
        {"entity_id": "bad id"},
        {"start": True},
        {"end": 5},
    ],
)
def test_malformed_span_metadata_fails_closed(metadata):
    values = {
        "start": 0,
        "end": 4,
        "label": "NAME",
        "score": 1.0,
        "source": "test",
        "entity_id": None,
    }
    values.update(metadata)
    span = SimpleNamespace(**values)
    with pytest.raises(ValueError):
        render("Jane", [span], options=ReplacementOptions())


def test_per_type_mask_keep_and_custom_marker():
    text = "Jane MR001 STUDY-9"
    spans = [Entity(0, 4, "NAME"), Entity(5, 10, "MRN"), Entity(11, 18, "STUDY_ID")]
    result = render(
        text,
        spans,
        options={
            "pattern": "<{label}:{id}>",
            "mask_character": "█",
            "per_type": {"NAME": "mask", "MRN": "keep"},
        },
    )
    assert result["text"] == "████ MR001 <STUDY_ID:e3>"


def test_synthetic_semantics_and_dates_match_existing_replacement():
    values = ["Jane Smith", "Jane", "2024-02-29", "95 years old", "MR001"]
    labels = ["NAME", "NAME", "DATE", "AGE", "MRN"]
    text = " | ".join(values)
    spans = []
    cursor = 0
    for value, label in zip(values, labels):
        start = text.index(value, cursor)
        spans.append(Entity(start, start + len(value), label))
        cursor = start + len(value)
    kwargs = {"patient_id": "patient-a", "secret": SECRET, "date_shift_days": 2}
    result = render(text, spans, options={"type": "synthetic"}, **kwargs)
    legacy = replace(text, spans, mode="synthetic", **kwargs)
    assert result["text"] == legacy["text"]
    assert result["preservation"] == legacy["preservation"]
    assert result["entities"][2]["replacement"] == "2024-03-02"


def test_unshiftable_and_overridden_dates_do_not_claim_interval_preservation():
    text = "March 2024"
    spans = [Entity(0, len(text), "DATE")]
    result = render(
        text, spans, options={"type": "synthetic"}, secret=SECRET, patient_id="a"
    )
    assert result["preservation"]["date_intervals_preserved"] is False
    assert "March 2024" not in str(result["warnings"])
    masked = render(
        text,
        spans,
        options={"type": "synthetic", "per_type": {"DATE": "mask"}},
        secret=SECRET,
        patient_id="a",
    )
    assert masked["preservation"]["date_intervals_preserved"] is False


def test_linked_context_alias_uses_full_identity_for_synthetic_value():
    context = "Jane Smith\nJane"
    context_spans = link_entities(
        context, [Entity(0, 10, "NAME"), Entity(11, 15, "NAME")]
    )
    result = render(
        "Jane",
        [Entity(0, 4, "NAME")],
        options={"type": "synthetic"},
        secret=SECRET,
        patient_id="a",
        context_text=context,
        context_spans=context_spans,
    )
    full = render(
        "Jane Smith",
        [Entity(0, 10, "NAME")],
        options={"type": "synthetic"},
        secret=SECRET,
        patient_id="a",
    )
    assert result["text"] == full["text"].split()[0]
    assert result["entities"][0]["entity_id"] == context_spans[0].entity_id


def test_custom_synthetic_types_and_cross_boundary_fragments_use_safe_markers():
    spans = [
        Entity(0, 5, "STUDY_ID"),
        Entity(6, 10, "NAME", source="context_fragment", entity_id="e7"),
    ]
    result = render(
        "ABCDE Jane",
        spans,
        options={"type": "synthetic"},
        secret=SECRET,
        patient_id="a",
    )
    assert result["text"] == "[STUDY_ID_e1] [NAME_e7]"


def test_mapping_opt_in_and_restoration_requires_explicit_sensitive_permission():
    result = render(
        "🙂 Jane met Bob.",
        [Entity(2, 6, "NAME"), Entity(11, 14, "NAME")],
        options={"include_mapping": True},
    )
    assert result["mappings"] == [
        {"marker": "[NAME_e1]", "original": "Jane", "label": "NAME"},
        {"marker": "[NAME_e2]", "original": "Bob", "label": "NAME"},
    ]
    assert reidentify(result["text"], result["mappings"]) == {
        "text": result["text"],
        "restored_count": 0,
    }
    assert reidentify(result["text"], result["mappings"], restore_sensitive=True) == {
        "text": "🙂 Jane met Bob.",
        "restored_count": 2,
    }
    assert (
        reidentify(
            result["text"],
            result["mappings"],
            restore_sensitive=True,
            allowed_types=["MRN"],
        )["restored_count"]
        == 0
    )


def test_medical_concepts_need_an_explicit_allowlist_to_restore_without_sensitive_opt_in():
    mapping = [{"marker": "[CONDITION]", "original": "asthma", "label": "CONDITION"}]
    assert reidentify("[CONDITION]", mapping)["restored_count"] == 0
    assert (
        reidentify("[CONDITION]", mapping, allowed_types=["CONDITION"])["text"]
        == "asthma"
    )


def test_reidentification_is_one_pass_and_does_not_expand_inserted_markers():
    mappings = [
        {"marker": "[A]", "original": "[B]", "label": "NAME"},
        {"marker": "[B]", "original": "Jane", "label": "NAME"},
    ]
    assert reidentify("[A] and [B]", mappings, restore_sensitive=True) == {
        "text": "[B] and Jane",
        "restored_count": 2,
    }


@pytest.mark.parametrize(
    "markers,text",
    [
        (["[A]", "[A]"], "[A]"),
        (["[A]", "A"], "[A]"),
        (["ABC", "BCD"], "ABCD"),
        ([""], "any"),
    ],
)
def test_duplicate_nested_overlapping_and_empty_markers_are_rejected(markers, text):
    with pytest.raises(ValueError):
        reidentify(
            text,
            [
                {"marker": marker, "original": "Jane", "label": "NAME"}
                for marker in markers
            ],
            restore_sensitive=True,
        )


def test_repeated_original_value_reuses_one_mapping_and_restores_every_occurrence():
    text = "Jane and Jane"
    result = render(
        text, spans_for(text, ["Jane", "Jane"]), options={"include_mapping": True}
    )
    assert len(result["mappings"]) == 1
    assert reidentify(result["text"], result["mappings"], restore_sensitive=True) == {
        "text": text,
        "restored_count": 2,
    }


def test_ambiguous_alias_mapping_requires_distinct_mention_markers():
    text = "Jane Smith | Jane"
    spans = spans_for(text, ["Jane Smith", "Jane"])
    with pytest.raises(ValueError, match="Ambiguous"):
        render(text, spans, options={"include_mapping": True})
    result = render(
        text,
        spans,
        options={"include_mapping": True, "pattern": "[{label}_{id}_{index}]"},
    )
    assert result["entities"][0]["entity_id"] == result["entities"][1]["entity_id"]
    assert (
        reidentify(result["text"], result["mappings"], restore_sensitive=True)["text"]
        == text
    )


def test_generated_mapping_cannot_collide_with_existing_source_literal():
    text = "Jane also typed [NAME_e1]."
    with pytest.raises(ValueError, match="collides"):
        render(text, [Entity(0, 4, "NAME")], options={"include_mapping": True})


def test_mapping_validation_does_not_accept_extra_metadata_or_coerce_booleans():
    with pytest.raises(ValidationError):
        ReidentifyRequest(text="", mappings=[], restore_sensitive="true")
    with pytest.raises(ValidationError):
        reidentify(
            "[A]",
            [{"marker": "[A]", "original": "Jane", "label": "NAME", "unexpected": 1}],
        )


def test_restored_output_expansion_is_bounded():
    with pytest.raises(ValueError, match="size limit"):
        reidentify(
            "[A]" * 11,
            [{"marker": "[A]", "original": "a" * 100_000, "label": "NAME"}],
            restore_sensitive=True,
        )


def test_large_literal_marker_set_does_not_require_regex_evaluation():
    mappings = [
        {"marker": f"[NAME_{index}]", "original": f"person{index}", "label": "NAME"}
        for index in range(2000)
    ]
    assert (
        reidentify("[NAME_1] met [NAME_1999]", mappings, restore_sensitive=True)["text"]
        == "person1 met person1999"
    )


def test_preservation_counts_only_unshiftable_attempted_dates_and_is_conservative():
    text = "2024-02-29"
    for action in ("mask", "marker", "keep"):
        result = render(
            text,
            [Entity(0, 10, "DATE")],
            options={"type": "synthetic", "per_type": {"DATE": action}},
            secret=SECRET,
            patient_id="a",
        )
        assert result["preservation"]["unshiftable_dates_redacted"] == 0
        assert result["preservation"]["date_intervals_preserved"] is False
        assert result["preservation"]["patient_consistent_replacements"] is False
    mixed = render(
        "Jane STUDY9",
        [Entity(0, 4, "NAME"), Entity(5, 11, "STUDY_ID")],
        options={"type": "synthetic"},
        secret=SECRET,
        patient_id="a",
    )
    assert mixed["preservation"]["patient_consistent_replacements"] is False


def test_linked_message_marker_indexes_use_the_global_mention_offset():
    result = render(
        "Jane",
        [Entity(0, 4, "NAME", entity_id="e1")],
        options={"pattern": "[{label}_{id}_{index}]", "include_mapping": True},
        index_offset=4,
    )
    assert result["text"] == "[NAME_e1_5]"
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match="index_offset"):
            render("", [], options={}, index_offset=bad)


def test_preassigned_ids_cannot_capture_another_generated_keyed_identity():
    bob_id = link_entities(
        "Bob", [Entity(0, 3, "NAME")], secret=SECRET, patient_id="a"
    )[0].entity_id
    with pytest.raises(ValueError, match="conflicts"):
        link_entities(
            "Jane Bob",
            [Entity(0, 4, "NAME", entity_id=bob_id), Entity(5, 8, "NAME")],
            secret=SECRET,
            patient_id="a",
        )


def test_ambiguous_context_requires_assigned_local_ids_instead_of_restarting_numbering():
    context = "Jane Smith\nJane Brown\nJane Jane"
    context_spans = link_entities(
        context, spans_for(context, ["Jane Smith", "Jane Brown", "Jane", "Jane"])
    )
    with pytest.raises(ValueError, match="unique identity"):
        render(
            "Jane Jane",
            [Entity(0, 4, "NAME"), Entity(5, 9, "NAME")],
            options={},
            context_text=context,
            context_spans=context_spans,
        )
    local = [
        Entity(0, 4, "NAME", entity_id=context_spans[2].entity_id),
        Entity(5, 9, "NAME", entity_id=context_spans[3].entity_id),
    ]
    result = render(
        "Jane Jane",
        local,
        options={},
        context_text=context,
        context_spans=context_spans,
    )
    assert result["text"] == "[NAME_e3] [NAME_e4]"


def test_prepared_context_builds_global_indexes_once_for_many_fields(monkeypatch):
    import clinical_deid.replacement.transform as module

    values = ["Jane Smith"] * 512
    context = "\n".join(values)
    spans = spans_for(context, values)
    indexed_context_sizes = []
    context_alias_calls = []
    original_keys = module._identity_keys
    original_aliases = module._name_aliases

    def counted_keys(text, spans):
        if text == context:
            indexed_context_sizes.append(len(spans))
        return original_keys(text, spans)

    def counted_aliases(text, spans):
        if text == context:
            context_alias_calls.append(len(spans))
        return original_aliases(text, spans)

    monkeypatch.setattr(module, "_identity_keys", counted_keys)
    monkeypatch.setattr(module, "_name_aliases", counted_aliases)
    prepared = module.prepare_context(context, spans)
    setup_key_calls = indexed_context_sizes.copy()
    setup_alias_calls = context_alias_calls.copy()
    for index, (value, span) in enumerate(zip(values, prepared.spans)):
        local = Entity(0, len(value), "NAME", entity_id=span.entity_id)
        result = render(
            value, [local], options={}, prepared_context=prepared, index_offset=index
        )
        assert result["text"] == "[NAME_e1]"
    assert indexed_context_sizes == setup_key_calls == [512, 512]
    assert context_alias_calls == setup_alias_calls == [512, 512, 512]


def test_prepared_context_is_immutable_and_keeps_identifiers_out_of_repr():
    from dataclasses import FrozenInstanceError

    from clinical_deid.replacement.transform import prepare_context

    prepared = prepare_context("Jane Smith", [Entity(0, 10, "NAME")])
    assert "Jane" not in repr(prepared)
    with pytest.raises(FrozenInstanceError):
        prepared.text = "other"
    with pytest.raises(TypeError):
        prepared.aliases["jane"] = ("other", "first")
    with pytest.raises(TypeError):
        prepared.identities[("NAME", "jane smith")] = frozenset({"e9"})
    assert isinstance(prepared.spans, tuple)
    assert all(isinstance(ids, frozenset) for ids in prepared.identities.values())


def test_prepared_and_raw_context_render_identically_including_synthetic_aliases():
    from clinical_deid.replacement.transform import prepare_context

    text = "Jane Smith\nJane"
    spans = [Entity(0, 10, "NAME"), Entity(11, 15, "NAME")]
    prepared = prepare_context(text, spans, secret=SECRET, patient_id="a")
    kwargs = {
        "options": {"type": "synthetic", "entity_ids": "keyed"},
        "secret": SECRET,
        "patient_id": "a",
    }
    assert render(
        "Jane", [Entity(0, 4, "NAME")], prepared_context=prepared, **kwargs
    ) == render(
        "Jane", [Entity(0, 4, "NAME")], context_text=text, context_spans=spans, **kwargs
    )
    with pytest.raises(ValueError, match="either"):
        render(
            "",
            [],
            options={},
            prepared_context=prepared,
            context_text=text,
            context_spans=spans,
        )
    with pytest.raises(ValueError, match="prepared"):
        render("", [], options={}, prepared_context={})


def test_mapping_creation_does_not_walk_global_context_in_python_per_field(monkeypatch):
    import clinical_deid.replacement.transform as module

    context = "Jane Smith\nJane"
    prepared = module.prepare_context(
        context, [Entity(0, 10, "NAME"), Entity(11, 15, "NAME")]
    )
    original_matches = module._MarkerMatcher.matches

    def reject_global_walk(matcher, text):
        assert text != context
        return original_matches(matcher, text)

    monkeypatch.setattr(module._MarkerMatcher, "matches", reject_global_walk)
    result = render(
        "Jane",
        [Entity(0, 4, "NAME", entity_id="e1")],
        options={"include_mapping": True},
        prepared_context=prepared,
    )
    assert result["mappings"] == [
        {"marker": "[NAME_e1]", "original": "Jane", "label": "NAME"}
    ]
    colliding_context = module.prepare_context(
        "Jane\n[NAME_e1]", [Entity(0, 4, "NAME")]
    )
    with pytest.raises(ValueError, match="collides"):
        render(
            "Jane",
            [Entity(0, 4, "NAME", entity_id="e1")],
            options={"include_mapping": True},
            prepared_context=colliding_context,
        )


def test_structured_service_prepares_shared_context_exactly_once(monkeypatch):
    import clinical_deid.replacement.transform as module
    from clinical_deid.formats.structured import process_structured
    from clinical_deid.service import Deidentifier

    class Detector:
        def detect_batch(self, texts):
            return [[Entity(0, len(text), "NAME")] for text in texts]

    prepared_sizes = []
    original_prepare = module.prepare_context

    def counted_prepare(text, spans, **kwargs):
        prepared_sizes.append(len(spans))
        return original_prepare(text, spans, **kwargs)

    monkeypatch.setattr(module, "prepare_context", counted_prepare)
    result = process_structured(Deidentifier(detector=Detector()), ["Jane Smith"] * 512)
    assert result["entity_count"] == 512
    assert prepared_sizes == [512]
