"""Executable examples for FHIR decisions F1 to F4 (2026-10-06).

See docs/annotation-policy-v2.md and docs/fhir-bank-brief.md. Kept in its own
module so source hashes recorded by earlier generators stay reproducible.
Every name, number and URL is invented.
"""

from __future__ import annotations

from outis.data.annotation_policy import _example

FHIR_FIXTURES = (
    # F1 and F2: patient-linked logical ids are ID, a record fullUrl is URL; system URIs and OIDs are not annotated.
    _example(
        "fhir_ids_and_systems",
        '{"fullUrl": "[[URL|https://fhir.example.test/Patient/pt-77]]", "resource": {"resourceType": "Patient", '
        '"id": "[[ID|pt-77]]", "meta": {"versionId": "3"}, "identifier": [{"system": "urn:oid:2.16.840.1.113883.19.5", '
        '"value": "[[MRN|MRN-7720931]]"}]}}',
        ["fhir_id", "fhir_system"],
    ),
    _example(
        "fhir_reference",
        '{"subject": {"reference": "Patient/[[ID|pt-77]]"}, "code": {"coding": [{"system": "http://loinc.org", "code": "2345-7"}]}}',
        ["fhir_reference", "fhir_code"],
    ),
    # F3: a FHIR date-time has its date part as DATE; time and timezone outside.
    _example(
        "fhir_datetime",
        '{"effectiveDateTime": "[[DATE|2026-03-14]]T07:42:00-05:00", "birthDate": "[[DATE|1941-03-03]]"}',
        ["fhir_datetime"],
    ),
    # F4: each name and address part is its own span; state and country alone are not annotated; keys and quotes outside.
    _example(
        "fhir_name_address_parts",
        '{"name": [{"family": "[[NAME|Brannock]]", "given": ["[[NAME|Odalys]]", "[[NAME|Q]]"]}], '
        '"address": [{"line": ["[[ADDRESS|412 Wenlow Lane]]", "[[ADDRESS|Apt 2B]]"], "city": "[[LOCATION|Corriston]]", '
        '"state": "WI", "postalCode": "[[LOCATION|53728]]", "country": "US"}], "gender": "female"}',
        ["fhir_name_parts", "fhir_address_parts"],
    ),
    # Narrative XHTML: tags outside spans.
    _example(
        "fhir_narrative",
        '{"text": {"status": "generated", "div": "<div xmlns=\\"http://www.w3.org/1999/xhtml\\"><p>[[NAME|Odalys Brannock]], '
        'born [[DATE|3 March 1941]]</p></div>"}}',
        ["fhir_narrative"],
    ),
)
