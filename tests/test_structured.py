"""Structured reconstruction and failure boundaries using synthetic identifiers."""

import csv
import io
import json
import re
from xml.etree import ElementTree as ET

import pytest

from clinical_deid.formats.structured import process_structured


class StubService:
    def __init__(self):
        self.calls = []

    def deidentify_batch(self, requests):
        self.calls.append(requests)
        results = []
        for request in requests:
            entities = []
            chunks = []
            end = 0
            for match in re.finditer(r"Alice Smith|Robert Jones|alice@example\.test", request["text"]):
                label = "EMAIL" if "@" in match[0] else "NAME"
                chunks.extend((request["text"][end:match.start()], f"[{label}]"))
                entities.append({"start": match.start(), "end": match.end(), "label": label,
                                 "score": 1.0, "source": "test"})
                end = match.end()
            chunks.append(request["text"][end:])
            results.append({"text": "".join(chunks), "entities": entities, "model_version": "fixture"})
        return results


def test_json_keys_values_and_scalar_types_preserved_without_input_mutation():
    data = {"Alice Smith": {"name": "Alice Smith", "value": 42, "active": True, "missing": None},
            "items": ["alice@example.test", 1.5]}
    result = process_structured(StubService(), data)
    assert "Alice Smith" in data
    assert result["data"] == {"[NAME]": {"name": "[NAME]", "value": 42, "active": True, "missing": None},
                              "items": ["[EMAIL]", 1.5]}
    assert result["entity_count"] == 3
    assert result["model_version"] == "fixture"
    assert "Alice Smith" not in json.dumps(result)
    assert result["entities"][0]["location"] == {"format": "json", "path": [0], "part": "key"}
    assert result["warnings"]


def test_redacted_json_keys_colliding_fails_without_partial_result():
    with pytest.raises(ValueError, match="keys collide"):
        process_structured(StubService(), {"Alice Smith": 1, "Robert Jones": 2})
    with pytest.raises(ValueError, match="keys collide"):
        process_structured(StubService(), {"Alice Smith": 1, "[NAME]": 2})


def test_json_selection_unescapes_pointers_and_supports_wildcards():
    data = {"a/b": {"~field": "Alice Smith"}, "records": [{"name": "Alice Smith"}, {"name": "Robert Jones"}],
            "excluded": "Alice Smith"}
    result = process_structured(StubService(), data, paths=["/a~1b/~0field", "/records/*/name"])
    assert result["data"]["a/b"]["~field"] == "[NAME]"
    assert result["data"]["records"] == [{"name": "[NAME]"}, {"name": "[NAME]"}]
    assert result["data"]["excluded"] == "Alice Smith"
    assert result["entity_count"] == 3


def test_json_root_pointer_selects_entire_document_and_subtree_selects_keys():
    data = {"selected": {"Alice Smith": "Robert Jones"}, "excluded": "Alice Smith"}
    all_result = process_structured(StubService(), data, paths=[""])
    assert "Alice Smith" not in json.dumps(all_result)
    selected = process_structured(StubService(), data, paths=["/selected"])
    assert selected["data"]["selected"] == {"[NAME]": "[NAME]"}
    assert selected["data"]["excluded"] == "Alice Smith"


@pytest.mark.parametrize("paths", [[], ["missing"], ["/bad~2"], ["/absent"], [None]])
def test_invalid_or_unmatched_paths_fail_closed(paths):
    with pytest.raises(ValueError):
        process_structured(StubService(), {"name": "Alice Smith"}, paths=paths)


def test_csv_headers_multiline_and_embedded_delimiters_are_processed_and_quoted():
    source = '"Alice Smith",note\r\n"Alice Smith","First line\nRobert Jones, seen"\r\n'
    result = process_structured(StubService(), source, format="csv")
    assert list(csv.reader(io.StringIO(result["data"]))) == [
        ["[NAME]", "note"], ["[NAME]", "First line\n[NAME], seen"]]
    assert result["entity_count"] == 3
    assert result["entities"][0]["location"] == {"format": "csv", "row": 0, "column": 0}
    assert "Alice Smith" not in json.dumps(result)


def test_csv_selection_uses_numeric_cells_including_header_row():
    result = process_structured(StubService(), "name,note\nAlice Smith,Robert Jones\n", format="csv", paths=["/1/0"])
    assert list(csv.reader(io.StringIO(result["data"]))) == [["name", "note"], ["[NAME]", "Robert Jones"]]


def test_csv_malformed_quote_rejected():
    with pytest.raises(ValueError, match="Invalid CSV"):
        process_structured(StubService(), 'name\n"Alice Smith', format="csv")


def test_xml_text_tails_attributes_comments_and_processing_instructions():
    source = '<?xml version="1.0"?><root name="Alice Smith">Robert Jones<child>alice@example.test</child>Alice Smith<!--Alice Smith--><?identity Alice Smith?></root>'
    result = process_structured(StubService(), source, format="xml")
    root = ET.fromstring(result["data"])
    assert root.attrib["name"] == "[NAME]"
    assert root.text == "[NAME]"
    assert root[0].text == "[EMAIL]"
    assert root[0].tail == "[NAME]"
    assert result["entity_count"] == 4
    assert "Alice Smith" not in json.dumps(result)
    assert all("name" not in entity["location"] for entity in result["entities"])


def test_xml_selection_uses_numeric_element_and_attribute_coordinates():
    result = process_structured(StubService(), '<root name="Alice Smith"><child>Alice Smith</child>Robert Jones</root>',
                                format="xml", paths=["/0/attributes/0", "/1/text"])
    root = ET.fromstring(result["data"])
    assert root.attrib["name"] == "[NAME]"
    assert root[0].text == "[NAME]"
    assert root[0].tail == "Robert Jones"


@pytest.mark.parametrize("source", [
    '<!DOCTYPE root [<!ENTITY sensitive SYSTEM "file:///etc/passwd">]><root>&sensitive;</root>',
    '<!DOCTYPE root [<!ENTITY a "Alice Smith">]><root>&a;</root>',
    '<!DOCTYPE root SYSTEM "https://example.test/private"><root>safe</root>',
    '<root><unclosed></root>',
])
def test_xml_external_entities_dtd_and_malformed_input_rejected(source):
    with pytest.raises(ValueError, match="Invalid or unsafe XML"):
        process_structured(StubService(), source, format="xml")


@pytest.mark.parametrize("source", ["<a>" * 66 + "Alice Smith" + "</a>" * 66,
                                     "<root>" + "<a/>" * 50_000 + "</root>"])
def test_xml_depth_and_node_limits_stop_before_inference(source):
    service = StubService()
    with pytest.raises(ValueError, match="Invalid or unsafe XML"):
        process_structured(service, source, format="xml")
    assert service.calls == []


def test_shared_segment_scope_and_linked_detection_options_are_forwarded():
    class SegmentedService(StubService):
        def process_segments(self, requests, **kwargs):
            self.segment_options = kwargs
            return self.deidentify_batch(requests)

    service = SegmentedService()
    process_structured(service, ["Alice Smith", "Robert Jones"], options={"mode": "redact", "project_id": "study"}, link_records=True)
    assert service.segment_options == {"link_batch": True, "shared_entities": True}
    assert service.calls[0] == [{"text": "Alice Smith", "mode": "redact", "project_id": "study"},
                                {"text": "Robert Jones", "mode": "redact", "project_id": "study"}]


def test_structured_fields_can_exceed_http_batch_limit_in_one_shared_scope():
    class SegmentedService(StubService):
        def process_segments(self, requests, **kwargs):
            assert len(requests) == 32
            return self.deidentify_batch(requests)

    result = process_structured(SegmentedService(), ["Alice Smith"] * 32)
    assert result["data"] == ["[NAME]"] * 32
    assert result["entity_count"] == 32


@pytest.mark.parametrize("data", [float("nan"), {1: "value"}, ("value",), ["x"] * 4097, "x" * 100001])
def test_invalid_json_and_size_limits_rejected(data):
    with pytest.raises(ValueError):
        process_structured(StubService(), data)


def test_cyclic_and_deep_json_rejected_without_echoing_content():
    cyclic = ["Alice Smith"]
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="Cyclic"):
        process_structured(StubService(), cyclic)
    deep = "Alice Smith"
    for _ in range(66):
        deep = [deep]
    with pytest.raises(ValueError, match="nesting"):
        process_structured(StubService(), deep)


@pytest.mark.parametrize("kwargs", [{"format": "yaml"}, {"options": {"raw_text": True}}, {"link_records": "yes"}])
def test_unsupported_configuration_rejected_even_when_no_text(kwargs):
    with pytest.raises(ValueError):
        process_structured(StubService(), None, **kwargs)


def test_invalid_processor_output_rejected():
    class BrokenService:
        def deidentify_batch(self, requests):
            return []

    with pytest.raises(RuntimeError, match="invalid structured output"):
        process_structured(BrokenService(), ["Alice Smith"])


def test_processor_warnings_and_explicit_restoration_mappings_are_preserved():
    class MappingService(StubService):
        def deidentify_batch(self, requests):
            results = super().deidentify_batch(requests)
            for request, result in zip(requests, results):
                result["mappings"] = [{"marker": "[NAME]", "original": request["text"], "label": "NAME"}]
                result["warnings"] = ["Some dates were redacted."]
            return results

    result = process_structured(MappingService(), ["Alice Smith", "Alice Smith"])
    assert result["mappings"] == [{"marker": "[NAME]", "original": "Alice Smith", "label": "NAME"}]
    assert result["warnings"] == ["Some dates were redacted."]
    with pytest.raises(ValueError, match="Conflicting restoration"):
        process_structured(MappingService(), ["Alice Smith", "Robert Jones"])


def test_real_service_shares_markers_across_more_than_sixteen_fields():
    from clinical_deid.service import Deidentifier
    from clinical_deid.types import Span

    class Detector:
        def detect_batch(self, texts):
            return [[Span(match.start(), match.end(), "NAME")
                     for match in re.finditer("Alice Smith|Robert Jones", text)] for text in texts]

    data = ["Alice Smith", "Robert Jones"] * 16
    result = process_structured(Deidentifier(detector=Detector()), data,
                                options={"replacement": {"include_mapping": True}})
    assert result["data"] == ["[NAME_e1]", "[NAME_e2]"] * 16
    assert result["entity_count"] == 32
    assert len(result["mappings"]) == 2
    assert result["entities"][0]["entity_id"] == result["entities"][2]["entity_id"]
    assert result["entities"][0]["entity_id"] != result["entities"][1]["entity_id"]


def test_linked_detection_cross_field_entities_are_redacted_in_every_fragment():
    from clinical_deid.service import Deidentifier
    from clinical_deid.types import Span

    class Detector:
        def detect_batch(self, texts):
            return [[Span(match.start(), match.end(), "NAME")
                     for match in re.finditer(r"Alice\s+Smith", text)] for text in texts]

    service = Deidentifier(detector=Detector())
    independent = process_structured(service, ["Alice", "Smith"])
    assert independent["data"] == ["Alice", "Smith"]
    linked = process_structured(service, ["Alice", "Smith"], link_records=True,
                                options={"replacement": {}})
    assert linked["data"] == ["[NAME_e1]", "[NAME_e1]"]
    assert linked["entity_count"] == 2
    assert linked["entities"][0]["entity_id"] == linked["entities"][1]["entity_id"]
    assert all(entity["source"] == "context_fragment" for entity in linked["entities"])
