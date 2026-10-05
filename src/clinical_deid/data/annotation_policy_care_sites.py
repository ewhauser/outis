"""Executable examples for the care-site and locality decisions of 2026-10-05.

See docs/annotation-policy-v2.md (census table) and
docs/annotation-policy-v2-places.md (P7). Kept in its own module so source
hashes recorded by earlier generators for ``annotation_policy`` and
``annotation_policy_places`` stay reproducible. These are authoring and
evaluation examples, not a detector dictionary. Every name is invented.
"""

from __future__ import annotations

from clinical_deid.data.annotation_policy import _example

CARE_SITE_FIXTURES = (
    # D1 / P7: a city that modifies an unnamed care site is LOCATION; the generic word stays outside.
    _example(
        "p7_city_modifies_unnamed_site",
        "Seen at our [[LOCATION|Vellmont]] clinic; labs drawn at the [[LOCATION|Corriston]] office.",
        ["p7_city_modifier"],
    ),
    _example(
        "p7_named_site_is_still_facility",
        "Seen at [[FACILITY|Vellmont Clinic]]; our [[LOCATION|Corriston]] site sent records.",
        ["p7_city_modifier", "facility_boundary"],
    ),
    # D2: a named organisation that provides care, services, supplies or housing to the patient is FACILITY.
    _example(
        "service_organisations_are_facilities",
        "Referred to [[FACILITY|Vossmere VNA]]; bed delivered by [[FACILITY|Tindle Brook DME]]; "
        "attends [[FACILITY|Saint Odile Adult Day Program]]; labs from [[FACILITY|Yarrowick Laboratory Services]].",
        ["service_organisation", "facility_boundary"],
    ),
    _example(
        "unnamed_service_is_not_phi",
        "Home health referral sent; DME delivered a hospital bed; the hospice team will visit.",
        ["service_organisation_unnamed"],
    ),
    _example(
        "employer_without_care_role",
        "Works nights at a warehouse for Pellham Logistics; admitted to [[FACILITY|Hospice of Marrowdale]].",
        ["employer_name", "service_organisation"],
    ),
    # D3: "The" inside when it is part of the proper name; outside when it is an article.
    _example(
        "the_part_of_name",
        "Lives at [[FACILITY|The Marlowe at Fennick Ridge]]; moved from [[FACILITY|The Quillon at Ashby Mill]].",
        ["the_in_name", "facility_boundary"],
    ),
    _example(
        "the_as_article",
        "The [[FACILITY|Pellham Drug & Gift]] pharmacist called; she was seen at the [[FACILITY|Ashgrove Family Health]] office.",
        ["the_article", "facility_boundary"],
    ),
)
