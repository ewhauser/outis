"""Bounded structured text processing without exposing field names in metadata.

JSON pointers select a value or subtree, and ``*`` matches one path component.
CSV is addressed as rows/cells (including header row zero). XML is addressed as
preorder element index followed by ``text``, ``tail``, or ``attributes/index``.
Absent selection processes every string, including JSON keys and CSV headers.
Numbers and booleans retain their original JSON types and are not analyzed.
"""

from __future__ import annotations

import csv
import io
import math
import re
from typing import Any
from xml.etree import ElementTree as ET

MAX_SEGMENTS = 4096
MAX_CHARACTERS = 100_000
MAX_INPUT_BYTES = 10 * 1024 * 1024
MAX_DEPTH = 64
MAX_NODES = 50_000
_OPTIONS = {
    "mode",
    "patient_id",
    "date_shift_days",
    "policy",
    "replacement",
    "project_id",
}


def _process_texts(service, texts, options=None, link_records=False):
    """Apply one shared replacement scope to a bounded collection of fields."""
    if not isinstance(link_records, bool):
        raise ValueError("link_records must be a boolean")
    if options is not None and (
        not isinstance(options, dict) or set(options) - _OPTIONS
    ):
        raise ValueError("Unsupported processing options")
    if len(texts) > MAX_SEGMENTS or any(not isinstance(text, str) for text in texts):
        raise ValueError("A structured document may contain at most 4096 text fields")
    if sum(len(text) for text in texts) > MAX_CHARACTERS:
        raise ValueError(
            "A structured document may contain at most 100000 text characters"
        )
    if not texts:
        return []
    requests = [{**(options or {}), "text": text} for text in texts]
    if hasattr(service, "process_segments"):
        results = service.process_segments(
            requests, link_batch=link_records, shared_entities=True
        )
    else:
        # Lightweight injected services can implement just this public method.
        # The production service always takes the shared-context path above.
        results = service.deidentify_batch(requests)
    if not isinstance(results, (list, tuple)) or len(results) != len(texts):
        raise RuntimeError("Processor returned invalid structured output")
    if any(
        not isinstance(result, dict)
        or not isinstance(result.get("text"), str)
        or not isinstance(result.get("entities", []), list)
        for result in results
    ):
        raise RuntimeError("Processor returned invalid structured output")
    return results


class _Selection:
    def __init__(self, paths):
        self.patterns = None
        self.matched = set()
        if paths is None:
            return
        if not isinstance(paths, (list, tuple)) or not 1 <= len(paths) <= 256:
            raise ValueError("paths must contain between 1 and 256 JSON pointers")
        self.patterns = []
        for path in paths:
            if (
                not isinstance(path, str)
                or len(path) > 4096
                or (path and not path.startswith("/"))
            ):
                raise ValueError("Invalid selection pointer")
            parts = [] if path == "" else path[1:].split("/")
            if any(re.search(r"~(?![01])", part) for part in parts):
                raise ValueError("Invalid selection pointer escape")
            self.patterns.append(
                tuple(part.replace("~1", "/").replace("~0", "~") for part in parts)
            )

    def includes(self, path):
        if self.patterns is None:
            return True
        selected = False
        for index, pattern in enumerate(self.patterns):
            if len(pattern) <= len(path) and all(
                want == "*" or want == actual for want, actual in zip(pattern, path)
            ):
                self.matched.add(index)
                selected = True
        return selected

    def validate(self):
        if self.patterns is not None and len(self.matched) != len(self.patterns):
            raise ValueError("A selection pointer did not match any field")


def _entity_result(results, locations):
    entities = []
    mappings = {}
    has_mapping = False
    warnings = []
    for result, location in zip(results, locations):
        for entity in result.get("entities", []):
            if not isinstance(entity, dict):
                raise RuntimeError("Processor returned invalid entity metadata")
            entities.append({**entity, "location": location})
        for warning in result.get("warnings", []):
            if warning not in warnings:
                warnings.append(warning)
        if "mappings" in result:
            has_mapping = True
        for mapping in result.get("mappings", []):
            if not isinstance(mapping, dict) or set(mapping) != {
                "marker",
                "original",
                "label",
            }:
                raise RuntimeError("Processor returned invalid restoration mapping")
            marker = mapping["marker"]
            if marker in mappings and mappings[marker] != mapping:
                raise ValueError(
                    "Conflicting restoration markers across document fields"
                )
            mappings[marker] = mapping
    out = {"entities": entities, "entity_count": len(entities), "warnings": warnings}
    if has_mapping:
        out["mappings"] = list(mappings.values())
    versions = {
        result["model_version"] for result in results if "model_version" in result
    }
    if len(versions) == 1:
        out["model_version"] = versions.pop()
    return out


def _input_text(data):
    if not isinstance(data, str):
        raise ValueError("CSV and XML data must be strings")
    try:
        length = len(data.encode("utf-8"))
    except UnicodeError:
        raise ValueError("Input must contain valid UTF-8 text") from None
    if length > MAX_INPUT_BYTES:
        raise ValueError("Structured input exceeds the 10 MiB limit")
    return data


def _json_document(service, data, selection, options, link_records):
    texts, locations = [], []
    active = set()
    nodes = 0
    size = 0
    has_numbers = False

    def add(text, path, kind):
        index = len(texts)
        texts.append(text)
        locations.append({"format": "json", "path": list(path), "part": kind})
        return index

    def visit(value, path=(), numeric_path=(), depth=0):
        nonlocal nodes, size, has_numbers
        nodes += 1
        if nodes > MAX_NODES or depth > MAX_DEPTH:
            raise ValueError("Structured input exceeds nesting or node limits")
        selected = selection.includes(path)
        if isinstance(value, str):
            try:
                size += len(value.encode("utf-8"))
            except UnicodeError:
                raise ValueError("Input must contain valid UTF-8 text") from None
            if size > MAX_INPUT_BYTES:
                raise ValueError("Structured input exceeds the 10 MiB limit")
            return (
                ("field", add(value, numeric_path, "value"))
                if selected
                else ("scalar", value)
            )
        if value is None or isinstance(value, bool):
            return ("scalar", value)
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("JSON numbers must be finite")
            has_numbers = True
            return ("scalar", value)
        if not isinstance(value, (dict, list)):
            raise ValueError("data must contain only JSON values")
        if id(value) in active:
            raise ValueError("Cyclic structured input is not supported")
        active.add(id(value))
        try:
            if isinstance(value, list):
                return (
                    "list",
                    [
                        visit(
                            item,
                            path + (str(index),),
                            numeric_path + (index,),
                            depth + 1,
                        )
                        for index, item in enumerate(value)
                    ],
                )
            items = []
            for index, (key, item) in enumerate(value.items()):
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")
                try:
                    size += len(key.encode("utf-8"))
                except UnicodeError:
                    raise ValueError("Input must contain valid UTF-8 text") from None
                if size > MAX_INPUT_BYTES:
                    raise ValueError("Structured input exceeds the 10 MiB limit")
                child_path = path + (key,)
                child_numeric = numeric_path + (index,)
                key_node = (
                    ("field", add(key, child_numeric, "key"))
                    if selection.includes(child_path)
                    else ("scalar", key)
                )
                items.append(
                    (key_node, visit(item, child_path, child_numeric, depth + 1))
                )
            return ("dict", items)
        finally:
            active.remove(id(value))

    tree = visit(data)
    selection.validate()
    results = _process_texts(service, texts, options, link_records)

    def materialize(node):
        kind, value = node
        if kind == "field":
            return results[value]["text"]
        if kind == "scalar":
            return value
        if kind == "list":
            return [materialize(item) for item in value]
        output = {}
        for key_node, item in value:
            key = materialize(key_node)
            if key in output:
                raise ValueError(
                    "Redacted JSON keys collide; use unique replacement markers"
                )
            output[key] = materialize(item)
        return output

    metadata = _entity_result(results, locations)
    if has_numbers:
        metadata["warnings"].append(
            "JSON numbers and booleans are preserved and are not analyzed."
        )
    return {"data": materialize(tree), "format": "json", **metadata}


def _csv_document(service, data, selection, options, link_records):
    source = _input_text(data)
    rows = []
    budget = MAX_NODES
    try:
        for row in csv.reader(io.StringIO(source, newline=""), strict=True):
            # Enforce the budget while parsing; a blank row still costs a node.
            budget -= max(1, len(row))
            if budget < 0:
                raise ValueError("CSV input exceeds row or cell limits")
            rows.append(row)
    except (csv.Error, OverflowError):
        raise ValueError("Invalid CSV input") from None
    texts, locations, fields = [], [], []
    selection.includes(())
    for row_index, row in enumerate(rows):
        selection.includes((str(row_index),))
        for column_index, text in enumerate(row):
            if selection.includes((str(row_index), str(column_index))):
                texts.append(text)
                locations.append(
                    {"format": "csv", "row": row_index, "column": column_index}
                )
                fields.append((row_index, column_index))
    selection.validate()
    results = _process_texts(service, texts, options, link_records)
    for (row, column), result in zip(fields, results):
        rows[row][column] = result["text"]
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\r\n").writerows(rows)
    return {
        "data": stream.getvalue(),
        "format": "csv",
        **_entity_result(results, locations),
    }


def _xml_document(service, data, selection, options, link_records):
    from defusedxml import ElementTree as SafeET
    from defusedxml.common import DefusedXmlException

    source = _input_text(data)
    try:
        parser = SafeET.iterparse(
            io.StringIO(source),
            events=("start", "end"),
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        )
        depth = 0
        count = 0
        for event, _element in parser:
            if event == "start":
                count += 1
                depth += 1
                if count > MAX_NODES or depth > MAX_DEPTH + 1:
                    raise ValueError("XML limits exceeded")
            else:
                depth -= 1
        root = parser.root
    except (ET.ParseError, DefusedXmlException, ValueError):
        raise ValueError("Invalid or unsafe XML input") from None
    texts, locations, fields = [], [], []
    selection.includes(())
    stack = [(root, 0)]
    index = 0
    while stack:
        element, depth = stack.pop()
        if index >= MAX_NODES or depth > MAX_DEPTH:
            raise ValueError("XML input exceeds nesting or element limits")
        prefix = (str(index),)
        selection.includes(prefix)
        for kind in ("text", "tail"):
            text = getattr(element, kind)
            if text is not None and selection.includes(prefix + (kind,)):
                texts.append(text)
                locations.append({"format": "xml", "element": index, "part": kind})
                fields.append((element, kind, None))
        selection.includes(prefix + ("attributes",))
        for attribute_index, (name, value) in enumerate(element.attrib.items()):
            if selection.includes(prefix + ("attributes", str(attribute_index))):
                texts.append(value)
                locations.append(
                    {
                        "format": "xml",
                        "element": index,
                        "part": "attribute",
                        "attribute": attribute_index,
                    }
                )
                fields.append((element, "attribute", name))
        stack.extend((child, depth + 1) for child in reversed(element))
        index += 1
    selection.validate()
    results = _process_texts(service, texts, options, link_records)
    for (element, kind, name), result in zip(fields, results):
        if kind == "attribute":
            element.set(name, result["text"])
        else:
            setattr(element, kind, result["text"])
    metadata = _entity_result(results, locations)
    metadata["warnings"].extend(
        [
            "XML comments and processing instructions are removed; namespace formatting may change.",
            "XML element and attribute names are treated as schema identifiers and are not analyzed.",
        ]
    )
    return {"data": ET.tostring(root, encoding="unicode"), "format": "xml", **metadata}


def process_structured(
    service, data: Any, *, format="json", paths=None, options=None, link_records=False
):
    """Process selected strings while keeping JSON types and document structure.

    Locations contain numeric coordinates only: original JSON keys, CSV headers,
    and XML attribute names are never copied into response entity metadata.
    Explicit paths leave unselected fields unchanged. Shared replacements apply
    across selected fields; linked detection context is separately opt-in.
    """
    if format not in {"json", "csv", "xml"}:
        raise ValueError("format must be json, csv, or xml")
    selection = _Selection(paths)
    function = {"json": _json_document, "csv": _csv_document, "xml": _xml_document}[
        format
    ]
    return function(service, data, selection, options, link_records)
