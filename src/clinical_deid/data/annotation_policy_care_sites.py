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

# Follow-up decisions of 2026-10-05 (cases left open by the tk13 author).
CARE_SITE_FOLLOWUP_FIXTURES = (
    # 1. A county name is LOCATION including "County".
    _example(
        "county_name_is_location",
        "Lives in [[LOCATION|Kestrine County]]; moved from [[LOCATION|Tollund County, OR]].",
        ["county_name", "location_boundary"],
    ),
    # 2. A capitalised site word in normal-case text names the site: FACILITY on the whole.
    _example(
        "capitalised_site_word_names_site",
        "Seen at our [[FACILITY|Ombrey Clinic]]; labs at our [[LOCATION|Gorsefield]] clinic.",
        ["p7_city_modifier", "facility_boundary"],
    ),
    # 2/3. In all-caps or lowercase text the authored role decides, not capitals.
    _example(
        "case_hidden_role_decides",
        "SEEN AT [[FACILITY|OMBREY CLINIC]]. LIVES AT [[FACILITY|THE HEARTH AT OMBREY]]. "
        "labs at our [[LOCATION|gorsefield]] clinic.",
        ["case_variation", "the_in_name", "p7_city_modifier"],
    ),
    # 4. "The" inside a service organisation's name only when it is part of the name.
    _example(
        "the_service_organisation",
        "Enrolled with [[FACILITY|The Hearth at Ombrey]] hospice program; "
        "The [[FACILITY|Saint Brevin VNA]] nurse called.",
        ["the_in_name", "the_article", "service_organisation"],
    ),
    # 5. P7 covers generic care-team phrases; a named pharmacy is FACILITY in any case.
    _example(
        "p7_care_team_and_lowercase_pharmacy",
        "the [[LOCATION|Gorsefield]] hospice team visited; refills at [[FACILITY|harlow vane drug]].",
        ["p7_city_modifier", "service_organisation", "case_variation"],
    ),
    # 6. A place-word used as an ordinary word stays unannotated in any case.
    _example(
        "ordinary_place_word_any_case",
        "NEEDS READING GLASSES AND A MOBILE PHONE CHARGER.",
        ["n5_place_word_other_role", "case_variation"],
    ),
    # 7. Per occurrence: employer role not annotated; care role is FACILITY.
    _example(
        "employer_and_care_roles",
        "Works as a nurse at Ombrey General Hospital; admitted to [[FACILITY|Ombrey General Hospital]] overnight.",
        ["employer_name", "same_spelling_different_context"],
    ),
)
