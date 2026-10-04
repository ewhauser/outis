"""Original, independently authored synthetic clinical context families.

This file contains no patient records, borrowed clinical templates, corpus rows,
model output, or generated identities. The parent composer supplies all values.

SPAN CONTRACT
-------------
A Mention selects the exact characters inserted for its placeholder, indexed
from zero among occurrences of that placeholder in template order. Punctuation,
articles, prepositions, and surrounding prose are excluded from that span.
Mention ownership is local to that occurrence; it is never inferred from whether
the same value appeared elsewhere. `target_label` specifies DATE or FACILITY for
this focused prototype. Other entity kinds retain ownership metadata so the
composer can map them to its own full annotation scheme. A null target_label
therefore does not mean that a personal name or place should remain visible.

Personal dates describe this person's care, history, or appointment. Public dates
describe a public document, public event, or general teaching example. Personal
facilities provided, will provide, or are being asked to provide this person's
care. Render these as specifically named physical care sites, not merely a
network, a service category, or an organization without a physical care role. A facility merely named as a public author, example, or source is public.
Generic service names listed in non_entity_phrases are ordinary clinical text,
even after a service or referral caption. They do not acquire FACILITY spans.
Ownership metadata takes precedence over placeholder spelling: one alias family
uses public_site for a SECOND PERSONAL name of the same care facility. Its
same_entity_groups require distinct rendered aliases of a single physical site.
These groups differ from same_value_groups, which require identical strings.
Intervals describe elapsed time and have no DATE target span.

DATE CONTRACT
-------------
DateOrder constraints apply to latent event dates before rendering. The composer
must preserve their visible sense when rendering partial dates (for example,
month/year, a month name, or a year); it must not render a strict order as an
apparently reversed sequence. No family requires complete numeric date strings.
DateGap additionally equates elapsed days to interval_days. same_value_groups
require identical rendered strings for the named placeholders, deliberately
allowing one value to have both personal and public ownership. No constraints
connect unrelated public dates to the personal timeline unless explicitly stated.

The module neither selects identities, assigns dataset splits, nor emits rows.
"""

from dataclasses import dataclass
from string import Formatter
from typing import Literal

Ownership = Literal["personal", "public", "context"]
EntityKind = Literal["person", "facility", "date", "place", "duration"]


@dataclass(frozen=True)
class Mention:
    placeholder: str
    occurrence: int
    ownership: Ownership
    entity_kind: EntityKind
    semantic_role: str
    target_label: Literal["DATE", "FACILITY"] | None = None


@dataclass(frozen=True)
class DateOrder:
    before: str
    after: str
    minimum_days: int = 1


@dataclass(frozen=True)
class DateGap:
    start: str
    end: str
    days_placeholder: str = "interval_days"


@dataclass(frozen=True)
class Family:
    family_id: str
    focus: Literal["DATE", "FACILITY", "DATE_AND_FACILITY"]
    note_form: str
    template: str
    mentions: tuple[Mention, ...]
    date_order: tuple[DateOrder, ...] = ()
    date_gaps: tuple[DateGap, ...] = ()
    same_value_groups: tuple[tuple[str, ...], ...] = ()
    narrative_note: str = ""
    non_entity_phrases: tuple[str, ...] = ()
    same_entity_groups: tuple[tuple[str, ...], ...] = ()


def m(key, owner, kind, role, label=None, occurrence=0):
    """Concise authoring helper; ownership and role are always explicit."""
    return Mention(key, occurrence, owner, kind, role, label)


def personal_date(key, role, occurrence=0):
    return m(key, "personal", "date", role, "DATE", occurrence)


def public_date(key="public_date", role="public document date", occurrence=0):
    return m(key, "public", "date", role, occurrence=occurrence)


def care(key="care_site", role="facility delivering personal care", occurrence=0):
    return m(key, "personal", "facility", role, "FACILITY", occurrence)


def public_site(key="public_site", role="public institutional source", occurrence=0):
    return m(key, "public", "facility", role, occurrence=occurrence)


def patient(occurrence=0):
    return m("patient", "personal", "person", "patient", occurrence=occurrence)


def clinician(occurrence=0):
    return m("clinician", "personal", "person", "treating clinician", occurrence=occurrence)


def personal_place(role="patient residence", occurrence=0):
    return m("place", "personal", "place", role, occurrence=occurrence)


def duration(occurrence=0):
    return m("interval_days", "context", "duration", "elapsed days between personal events", occurrence=occurrence)


FAMILIES = (
    Family(
        "date_review_after_leaflet", "DATE", "care review",
        "{patient} brought the sleep leaflet issued in {public_date}. At the review in {date_b}, "
        "{clinician} compared the diary with the first visit in {date_a}; waking during the night was less frequent.",
        (patient(), public_date(role="leaflet issue date"), personal_date("date_b", "current care review"),
         clinician(), personal_date("date_a", "earlier clinical visit")),
        date_order=(DateOrder("date_a", "date_b"),),
        narrative_note="A public publication date opens a patient-bearing review; personal visits are narrated in reverse order.",
    ),
    Family(
        "date_therapy_milestone_then_manual", "DATE", "therapy progress note",
        "The first stair session was in {date_a}. By {date_b}, {patient} could pause on the landing and continue "
        "with the rail. The therapist used a pacing illustration from the manual published in {public_date}.",
        (personal_date("date_a", "first personal stair therapy"), personal_date("date_b", "personal functional milestone"),
         patient(), public_date(role="manual publication date")),
        date_order=(DateOrder("date_a", "date_b"),),
        narrative_note="A dated functional milestone remains personal even without an appointment keyword in its clause.",
    ),
    Family(
        "date_handover_two_meanings", "DATE", "handover addendum",
        'The handover records a wound review in {date_a} for {patient}. An attached consensus statement was '
        'released in {public_date}. A later examination in {date_b} found the dressing intact.',
        (personal_date("date_a", "personal wound review referred to by handover"), patient(),
         public_date(role="consensus release date"), personal_date("date_b", "later personal examination")),
        date_order=(DateOrder("date_a", "date_b"),),
        same_value_groups=(("date_a", "public_date"),),
        narrative_note="Identical date text is personal in the handover and public on the consensus statement.",
    ),
    Family(
        "date_public_case_before_visit", "DATE", "review of symptoms",
        'A teaching case dated {public_date} prompted a discussion of breathlessness. {patient} described '
        'the walk assessment in {date_a}. At the follow-up in {date_b}, the patient reported taking the same '
        'route with fewer stops.',
        (public_date(role="public teaching case date"), patient(), personal_date("date_a", "personal walk assessment"),
         personal_date("date_b", "personal follow-up")),
        date_order=(DateOrder("date_a", "date_b"),),
        narrative_note="A public teaching date is embedded in a note whose later clauses describe personal care.",
    ),
    Family(
        "date_medication_history_correction", "DATE", "medication review",
        '{patient} clarified that the earlier medication check took place in {date_a}. They brought a '
        'journal extract published in {public_date}. {clinician} updated the medication history at the '
        'review in {date_b}.',
        (patient(), personal_date("date_a", "corrected personal medication check"),
         public_date(role="journal extract date"), clinician(), personal_date("date_b", "review recording correction")),
        date_order=(DateOrder("date_a", "date_b"),),
        narrative_note="Ownership depends on the date referent, including an explicit correction of a source-date ambiguity.",
    ),
    Family(
        "date_public_manual_matches_milestone", "DATE", "occupational therapy review",
        'The kitchen task was completed without a rest in {date_b}. {patient} had needed two rests at the '
        'baseline session in {date_a}. The therapist used an illustration from the teaching manual published '
        'in {public_date}.',
        (personal_date("date_b", "personal kitchen-task milestone"), patient(),
         personal_date("date_a", "personal baseline therapy"), public_date(role="teaching manual edition date")),
        date_order=(DateOrder("date_a", "date_b"),),
        same_value_groups=(("date_b", "public_date"),),
        narrative_note="The later personal milestone and public edition intentionally share their rendered date.",
    ),
    Family(
        "date_historical_result_review", "DATE", "results review",
        'At the consultation in {date_b}, {clinician} went back to the test performed in {date_a}. {patient} '
        'recalled feeling well that morning. A methods paper from {public_date} helped explain the test '
        'scale.',
        (personal_date("date_b", "personal consultation"), clinician(), personal_date("date_a", "historical personal test"),
         patient(), public_date(role="methods paper publication date")),
        date_order=(DateOrder("date_a", "date_b"),),
        narrative_note="The historical test is the date referent, while the scale explanation cites public literature.",
    ),
    Family(
        "date_actual_gap_from_public_calendar", "DATE", "telephone follow-up",
        "The public webinar listed for {public_date} was discussed briefly. {patient} then confirmed that the "
        "dressing was changed in {date_a} and checked again in {date_b}, {interval_days} days later. "
        "No further change had been needed between those visits.",
        (public_date(role="public webinar date"), patient(), personal_date("date_a", "personal dressing change"),
         personal_date("date_b", "personal dressing recheck"), duration()),
        date_order=(DateOrder("date_a", "date_b"),), date_gaps=(DateGap("date_a", "date_b"),),
        narrative_note="The interval is contextual duration; both endpoint dates belong to personal care.",
    ),
    Family(
        "facility_paper_source_after_treatment", "FACILITY", "discharge follow-up",
        '{patient} described the practice sessions received at {care_site} and demonstrated the same '
        'transfer at home. The handout included an illustration from the service report published by '
        '{public_site}.',
        (patient(), care(role="site of completed personal practice sessions"), public_site(role="institution named as public report source")),
        narrative_note="A care facility and a report-producing institution have separate semantic ownership.",
    ),
    Family(
        "facility_same_site_two_roles", "FACILITY", "referral discussion",
        'The audit from {public_site} was used to explain how referral queues are measured. {clinician} sent '
        'a referral for {patient} to {care_site} for a face-to-face mobility assessment.',
        (public_site(role="public audit author"), clinician(), patient(), care(role="actual destination of personal referral")),
        same_value_groups=(("care_site", "public_site"),),
        narrative_note="One institution string is public audit authorship first and a personal referral destination later.",
    ),
    Family(
        "facility_transfer_public_comparator", "FACILITY", "transfer handover",
        'A bed at {care_site} was accepted for {patient}, and the transport request is on the chart. The '
        'staffing comparison attached to the handover includes published figures from {public_site}.',
        (care(role="accepted destination for personal transfer"), patient(), public_site(role="institution in public staffing comparison")),
        narrative_note="A concrete transfer destination is distinct from an institution mentioned in background material.",
    ),
    Family(
        "facility_source_first_home_review", "FACILITY", "home visit note",
        "A public booklet prepared by {public_site} shows a raised-chair setup. During the home visit in {place}, "
        "{patient} said the team at {care_site} had already practised that transfer with them.",
        (public_site(role="public booklet author"), personal_place(role="location of personal home visit"), patient(),
         care(role="site of earlier personal transfer training")),
        narrative_note="The public source comes first; the actual care relationship appears in reported history.",
    ),
    Family(
        "facility_case_example_no_referral", "FACILITY", "education note",
        "{clinician} used the public case example from {public_site} to explain appointment preparation. "
        "{patient} is scheduled at {care_site} and will take the symptom diary there.",
        (clinician(), public_site(role="institution supplying public case example"), patient(), care(role="site of scheduled personal appointment")),
        narrative_note="A future care site is personal even though the visit has not yet occurred.",
    ),
    Family(
        "facility_same_string_care_then_report", "FACILITY", "care history summary",
        '{patient} received the initial hand exercises at {care_site} and still keeps that exercise sheet. '
        'The discussion also drew on the annual report published by {public_site}, which describes how '
        'exercise groups are organised.',
        (patient(), care(role="site of personal initial hand therapy"), public_site(role="public annual-report author")),
        same_value_groups=(("care_site", "public_site"),),
        narrative_note="Identical institution text changes ownership between treatment history and publication attribution.",
    ),
    Family(
        "facility_pending_personal_intake", "FACILITY", "care coordination call",
        "The family asked whether the public intake guide from {public_site} applied locally. "
        "For {patient}, the coordinator has asked {care_site} to arrange an intake review and is awaiting an appointment time.",
        (public_site(role="public intake-guide publisher"), patient(), care(role="requested provider of personal intake review")),
        narrative_note="A personal care request establishes ownership even while appointment timing is unresolved.",
    ),
    Family(
        "facility_background_not_care_team", "FACILITY", "rehabilitation review",
        "The review for {patient} was completed with the rehabilitation team at {care_site}. "
        "A diagram credited to {public_site} was included in the discussion of step height. "
        "The patient preferred to practise on the bottom step before trying the whole flight.",
        (patient(), care(role="site delivering personal rehabilitation review"), public_site(role="public diagram credit")),
        narrative_note="An institution in a diagram credit remains public inside a detailed patient note.",
    ),
    Family(
        "joint_discharge_public_summary", "DATE_AND_FACILITY", "community handover",
        "The public summary issued by {public_site} in {public_date} was attached for background. "
        "{patient} left {care_site} in {date_a}; at the community review in {date_b}, the home exercise plan was still manageable.",
        (public_site(role="public summary issuer"), public_date(role="public summary issue date"), patient(),
         care(role="facility discharging this patient"), personal_date("date_a", "personal discharge"),
         personal_date("date_b", "subsequent community review")),
        date_order=(DateOrder("date_a", "date_b"),),
        narrative_note="Public institution-date pair precedes a personal discharge and follow-up sequence.",
    ),
    Family(
        "joint_same_pair_therapy_publication", "DATE_AND_FACILITY", "therapy history review",
        "{patient} began balance sessions at {care_site} in {date_a}. A public report by {public_site}, dated "
        "{public_date}, was cited in the explanation of the exercise. At the check in {date_b}, the patient described "
        "using the kitchen counter for support only when turning.",
        (patient(), care(role="site of personal balance therapy"), personal_date("date_a", "start of personal balance sessions"),
         public_site(role="public exercise-report author"), public_date(role="public exercise-report date"),
         personal_date("date_b", "later personal balance check")),
        date_order=(DateOrder("date_a", "date_b"),),
        same_value_groups=(("care_site", "public_site"), ("date_a", "public_date")),
        narrative_note="Both institution and date values recur with public ownership between two personal-care clauses.",
    ),
    Family(
        "joint_source_later_than_old_care", "DATE_AND_FACILITY", "historical care note",
        'At the review in {date_b}, {patient} recalled attending {care_site} in {date_a} after difficulty '
        'using the stairs. The accompanying page from {public_site}, published in {public_date}, describes '
        'ways to arrange support on a staircase.',
        (personal_date("date_b", "current personal review"), patient(), care(role="historical site of personal care"),
         personal_date("date_a", "historical personal attendance"), public_site(role="institution publishing supporting reading"),
         public_date(role="supporting-reading publication date")),
        date_order=(DateOrder("date_a", "date_b"),),
        narrative_note="The latest narrated event comes first; a public page date must not replace the historical care date.",
    ),
    Family(
        "joint_actual_destination_after_teaching", "DATE_AND_FACILITY", "referral plan",
        'During teaching, {clinician} referred to the case series from {public_site} published in '
        '{public_date}. The consultation with {patient} took place in {date_a}. An assessment is booked at '
        '{care_site} in {date_b} to review walking outdoors.',
        (clinician(), public_site(role="public case-series institution"), public_date(role="case-series publication date"),
         patient(), personal_date("date_a", "personal consultation"), care(role="booked personal assessment site"),
         personal_date("date_b", "booked personal assessment date")),
        date_order=(DateOrder("date_a", "date_b"),),
        narrative_note="A public teaching citation and a real later appointment coexist in the same clinician note.",
    ),
    Family(
        "joint_receiving_team_source_attribution", "DATE_AND_FACILITY", "receiving-team handover",
        "{patient} reached {care_site} in {date_b} after the assessment recorded in {date_a}. "
        "The receiving team used the checklist credited to {public_site} in the {public_date} training bulletin "
        "and confirmed that the patient could explain the call-bell routine.",
        (patient(), care(role="receiving facility for personal care"), personal_date("date_b", "personal arrival at receiving facility"),
         personal_date("date_a", "pre-transfer personal assessment"), public_site(role="public checklist source"),
         public_date(role="training bulletin date")),
        date_order=(DateOrder("date_a", "date_b"),),
        narrative_note="Transfer chronology is narrated backwards before the public checklist attribution.",
    ),
    Family(
        "joint_plain_language_return_home", "DATE_AND_FACILITY", "plain-language care note",
        "{patient} said getting home to {place} had gone better than expected. The team at {care_site} had practised "
        "the doorstep with them in {date_a}. When {clinician} called in {date_b}, they had managed it again with a neighbour nearby. "
        "They also talked about a public leaflet from {public_site}, printed in {public_date}, which showed a similar step.",
        (patient(), personal_place(role="patient home location"), care(role="site providing personal doorstep practice"),
         personal_date("date_a", "personal doorstep therapy"), clinician(), personal_date("date_b", "personal follow-up call"),
         public_site(role="public leaflet source"), public_date(role="public leaflet printing date")),
        date_order=(DateOrder("date_a", "date_b"),),
        narrative_note="Original plain-language narrative with a coherent return-home story and public reading inside it.",
    ),
    Family(
        "joint_one_date_two_documents", "DATE_AND_FACILITY", "record reconciliation",
        'The teaching sheet from {public_site} is dated {public_date}. It was brought along with an '
        'attendance record showing that {patient} was seen at {care_site} in {date_a}. At the review in '
        '{date_b}, {clinician} went through both pages with the family.',
        (public_date(role="date stated as date on a public teaching sheet"), public_site(role="public teaching-sheet source"),
         patient(), care(role="site confirmed by personal attendance record"), personal_date("date_a", "confirmed personal attendance"),
         personal_date("date_b", "personal record-reconciliation review"), clinician()),
        date_order=(DateOrder("date_a", "date_b"),),
        same_value_groups=(("date_a", "public_date"),),
        narrative_note="Shared date is repeated with an explicit personal attendance referent on its second occurrence.",
    ),
    Family(
        "joint_site_twice_public_then_private", "DATE_AND_FACILITY", "post-visit telephone note",
        "The {public_date} public report from {public_site} describes a general approach to follow-up. "
        "For {patient}, the visit to {care_site} was in {date_a}, and this call in {date_b} was arranged to check "
        "whether the exercises fitted around the morning routine.",
        (public_date(role="public follow-up report date"), public_site(role="public follow-up report source"), patient(),
         care(role="site of actual personal visit"), personal_date("date_a", "personal visit before call"),
         personal_date("date_b", "personal follow-up call")),
        date_order=(DateOrder("date_a", "date_b"),),
        same_value_groups=(("care_site", "public_site"),),
        narrative_note="The public source and personal visit share an institution value while preserving separate occurrence labels.",
    ),
    Family(
        "date_repeated_personal_reference", "DATE", "therapy timeline clarification",
        "The referral for {patient} says {date_a}. The therapist confirmed that {date_a} was the first assessment, "
        "while {date_b} marked the return to outdoor practice. The copied review article was published in {public_date}.",
        (personal_date("date_a", "date recorded in personal referral"), patient(),
         personal_date("date_a", "same personal first assessment clarified", occurrence=1),
         personal_date("date_b", "personal return to outdoor practice"), public_date(role="review-article publication date")),
        date_order=(DateOrder("date_a", "date_b"),),
        narrative_note="A personal partial date is repeated and must have both exact occurrences annotated.",
    ),
    Family(
        "date_public_heading_private_margin", "DATE", "document-assisted history",
        "{patient} brought a printout headed {public_date}, the date of a public lecture. "
        "In the margin they had written {date_a} beside a note about their own hand review. "
        "At the appointment in {date_b}, the clinician used the handwritten entry to complete the care history.",
        (patient(), public_date(role="public lecture date on printout"), personal_date("date_a", "personal hand review in handwritten history"),
         personal_date("date_b", "personal history-completion appointment")),
        date_order=(DateOrder("date_a", "date_b"),),
        same_value_groups=(("date_a", "public_date"),),
        narrative_note="An identical visible date can describe a public lecture heading and a private handwritten care event.",
    ),
    Family(
        "facility_repeated_receiving_site", "FACILITY", "transport coordination",
        '{care_site} confirmed that the assessment room was available for {patient}. The transport team was '
        'told to use the main entrance at {care_site}. A discussion of clear signage drew on an access '
        'diagram in the report published by {public_site}.',
        (care(role="confirmed personal assessment facility"), patient(),
         care(role="same facility as personal transport destination", occurrence=1),
         public_site(role="institution credited in public access report")),
        narrative_note="Both personal occurrences of a care facility are annotated; an unrelated publication credit is public.",
    ),
    Family(
        "facility_public_source_repeated", "FACILITY", "exercise education record",
        '{patient} practised the seated reach at {care_site}. The educator showed an illustration from '
        '{public_site} and a second diagram from the same booklet published by {public_site}. The patient '
        'then demonstrated the movement with their own chair.',
        (patient(), care(role="site delivering personal seated-reach practice"),
         public_site(role="public illustration source"), public_site(role="second public diagram from the same booklet", occurrence=1)),
        narrative_note="Repeated public facility text stays public even beside a specific personal exercise demonstration.",
    ),
    Family(
        "joint_elapsed_time_with_source", "DATE_AND_FACILITY", "therapy attendance summary",
        "{patient} attended {care_site} in {date_a} and returned in {date_b}, an interval of {interval_days} days. "
        "The second session covered standing from a low chair. The public teaching note from {public_site}, dated "
        "{public_date}, supplied the illustration used during that session.",
        (patient(), care(role="site of both personal therapy sessions"), personal_date("date_a", "first personal therapy attendance"),
         personal_date("date_b", "second personal therapy attendance"), duration(),
         public_site(role="public teaching-note institution"), public_date(role="public teaching-note date")),
        date_order=(DateOrder("date_a", "date_b"),), date_gaps=(DateGap("date_a", "date_b"),),
        narrative_note="A measured personal interval is constrained by both endpoints and kept distinct from source publication time.",
    ),
    Family(
        "joint_public_pair_at_end_same_values", "DATE_AND_FACILITY", "outpatient progress note",
        'At {care_site} in {date_a}, {patient} needed help arranging the walking aid before standing. The '
        'later review in {date_b} recorded that they could set it up independently. The exercise discussion '
        'drew on the review published by {public_site} in {public_date}.',
        (care(role="facility providing personal initial mobility review"), personal_date("date_a", "personal initial mobility review"),
         patient(), personal_date("date_b", "later personal independence review"),
         public_site(role="public review issuer"), public_date(role="public review issue date")),
        date_order=(DateOrder("date_a", "date_b"),),
        same_value_groups=(("care_site", "public_site"), ("date_a", "public_date")),
        narrative_note="Private institution-date pair opens the note; an identical public pair closes it.",
    ),
    Family(
        "joint_personal_consultation_between_sources", "DATE_AND_FACILITY", "education and care summary",
        'The exercise explanation drew on a workshop summary from {public_site}. {clinician} saw {patient} '
        'at {care_site} in {date_a}, then reviewed the home practice in {date_b}. The summary was published '
        'in {public_date} and includes drawings of seated exercises.',
        (public_site(role="public workshop-summary source"), clinician(), patient(),
         care(role="site of personal consultation"), personal_date("date_a", "personal exercise consultation"),
         personal_date("date_b", "later personal home-practice review"), public_date(role="public workshop-summary edition date")),
        date_order=(DateOrder("date_a", "date_b"),),
        narrative_note="Public source institution and publication date bracket the personal care account.",
    ),
    Family(
        "facility_weak_record_header_generic_service", "FACILITY", "review of correspondence",
        "The earlier letter for {patient} is filed under {care_site}. Its heading reads 'Service: occupational therapy', "
        "and the body records practice with a cup and a dressing aid. A public leaflet credited to {public_site} "
        "was enclosed to illustrate the equipment.",
        (patient(), care(role="physical care site identified by personal clinical correspondence"),
         public_site(role="institution credited on enclosed public leaflet")),
        narrative_note="A weak record header identifies a named personal care site; the generic service caption is ordinary text.",
        non_entity_phrases=("occupational therapy",),
    ),
    Family(
        "facility_correspondence_alias", "FACILITY", "appointment letter reconciliation",
        '{patient} brought an appointment letter headed {care_site}, with the shorter site name '
        "{public_site} in the return address. Under 'Referral: community rehabilitation', the letter "
        'confirms that the first assessment will take place there.',
        (patient(), care(role="full name of physical site in personal appointment letter"),
         care(key="public_site", role="shorter alias of same physical personal care site")),
        narrative_note="Both distinct names identify the same physical site through personal correspondence; the service stays generic.",
        non_entity_phrases=("community rehabilitation",),
        same_entity_groups=(("care_site", "public_site"),),
    ),
    Family(
        "facility_same_site_referral_caption", "FACILITY", "referral record review",
        'The published access survey includes information from {public_site}. In the record for {patient}, '
        "the caption 'Referral requested: falls clinic' is followed by an appointment letter from "
        '{care_site}. The letter includes directions to the assessment room.',
        (public_site(role="institution listed in public access survey"), patient(),
         care(role="physical care site identified in personal appointment correspondence")),
        same_value_groups=(("care_site", "public_site"),),
        narrative_note="Same institution text is public in a survey and personal in correspondence; the generic specialty caption is not a facility.",
        non_entity_phrases=("falls clinic",),
    ),
    Family(
        "joint_report_date_first_private_date_last", "DATE_AND_FACILITY", "care-plan review",
        'The {public_date} service report from {public_site} was mentioned while discussing appointment '
        'access. {patient} had already completed an assessment at {care_site} in {date_a}. The plan agreed '
        'with {clinician} was checked again at the review in {date_b}.',
        (public_date(role="public service-report date"), public_site(role="public service-report source"), patient(),
         care(role="site of completed personal assessment"), personal_date("date_a", "completed personal assessment"),
         clinician(), personal_date("date_b", "personal care-plan review")),
        date_order=(DateOrder("date_a", "date_b"),),
        same_value_groups=(("date_b", "public_date"),),
        narrative_note="Identical date text appears first for public reporting and last for a personal review.",
    ),
)


NEGATIVE_FAMILIES = (
    Family(
        "negative_public_seminar_notice", "DATE_AND_FACILITY", "education discussion",
        "The team used a public seminar notice dated {public_date} to introduce the discussion. "
        "The notice names {public_site} as the host institution. The patient asked about the difference "
        "between an open teaching event and an individual therapy appointment.",
        (public_date(role="public seminar notice date"), public_site(role="public seminar host")),
        narrative_note="Entire passage is PHI-free; a generic patient mention supplies clinical context only.",
        non_entity_phrases=("individual therapy",),
    ),
    Family(
        "negative_generic_service_caption", "DATE_AND_FACILITY", "referral explanation",
        "Referral requested: speech and language therapy. The patient asked what an initial conversation "
        "with that service might involve. A public leaflet issued by {public_site} in {public_date} "
        "was used to explain the general process.",
        (public_site(role="public service-information leaflet issuer"), public_date(role="public leaflet issue date")),
        narrative_note="The referral caption contains a generic service, with no named personal care facility or appointment date.",
        non_entity_phrases=("speech and language therapy",),
    ),
    Family(
        "negative_disease_clinic_question", "DATE_AND_FACILITY", "patient education",
        'The patient asked how a memory clinic differs from a general outpatient review. The explanation '
        'used a guide from {public_site}, published in {public_date}. The patient found the section '
        'describing a first appointment helpful.',
        (public_site(role="public clinic-information guide source"), public_date(role="public guide edition date")),
        narrative_note="The generic disease clinic is ordinary clinical language; public source details remain unannotated.",
        non_entity_phrases=("memory clinic", "general outpatient review"),
    ),
    Family(
        "negative_public_report_endnote", "DATE_AND_FACILITY", "equipment discussion",
        'The patient compared the chair heights shown in a general information sheet and asked why a '
        'footrest might help. The discussion covered how the feet can be supported while sitting. The '
        'illustrations came from the equipment report published by {public_site} in {public_date}.',
        (public_site(role="public equipment report source"), public_date(role="public equipment report publication date")),
        narrative_note="Public date and institution occur at the end of a patient-bearing but PHI-free narrative.",
    ),
    Family(
        "negative_service_header_only", "FACILITY", "service orientation",
        "Service: respiratory clinic. The discussion covered what information a patient can bring to a "
        "first consultation, including a current medication list and questions about daily activities. "
        "A public checklist from {public_site} was shown as an example of how to prepare.",
        (public_site(role="public consultation-preparation checklist source"),),
        narrative_note="A service caption does not make a generic specialty phrase a named physical care facility.",
        non_entity_phrases=("respiratory clinic",),
    ),
    Family(
        "negative_public_date_mid_narrative", "DATE", "group education note",
        "The patient asked why two diagrams used different words for the same movement. The facilitator "
        "explained that the version published in {public_date} used plainer language. The discussion then "
        "returned to choosing words that were comfortable for the patient.",
        (public_date(role="public diagram version publication date"),),
        narrative_note="A lone date in the middle of clinical prose refers only to public teaching material.",
    ),
    Family(
        "negative_public_institution_repetition", "FACILITY", "information-source discussion",
        'The annual report from {public_site} was open during the discussion. The patient asked about a '
        'diagram credited to {public_site} that showed the layout of an exercise group. The clinician '
        'explained how the chairs were arranged to leave room for movement.',
        (public_site(role="public annual report source"), public_site(role="repeated public illustration attribution in annual report", occurrence=1)),
        narrative_note="Repeated institution text remains public; generic references to patient and clinician are not names.",
    ),
    Family(
        "negative_public_teaching_date_caption", "DATE_AND_FACILITY", "general rehabilitation information",
        'The open teaching session listed for {public_date} includes examples from {public_site}. Under the '
        "heading 'Service discussed: community physiotherapy', the patient wrote a question about how home "
        'exercises are explained. The group talked about useful ways to ask for a demonstration.',
        (public_date(role="public teaching session date"), public_site(role="institution used in public teaching examples")),
        narrative_note="Public teaching facts coexist with an unbranded generic service caption and no personal identifiers.",
        non_entity_phrases=("community physiotherapy",),
    ),
)


# Optional positioning material authored here. These paragraphs contain no names,
# dates, locations, named facilities, or latent timeline events. They can precede
# or follow a family where their conversational context fits. The composer should
# vary placement rather than concatenate all paragraphs or use a fixed preamble.
CLINICAL_CONTEXT_PARAGRAPHS = (
    "The discussion used everyday language, with pauses for questions. Unfamiliar terms were explained before moving on.",
    "The patient was invited to describe which parts of the routine were straightforward and which needed more explanation. The team allowed time for that account.",
    "The explanation linked the written material to ordinary tasks at home. The patient could ask for a different example if the first one was unclear.",
    "The conversation covered both the practical task and how it might fit around a usual day. No assumptions were made about which part mattered most to the patient.",
    "The team checked that the purpose of the discussion was clear. Questions about the wording of the notes were addressed as they arose.",
    "The patient was encouraged to describe the task in their own words. The explanation was adjusted when a phrase was unfamiliar.",
    "The written summary used short sentences and left space for questions. The patient could point to a sentence that needed further explanation.",
    "The discussion stayed focused on the immediate practical concern. Additional topics could be raised once that concern had been understood.",
    "The team worked through the handout a section at a time. Examples were used to make the wording easier to follow.",
    "The patient and clinician checked that they were using the same words for the activity being discussed. This helped make the account easier to follow.",
)


# Independent retention snippets for optional sparse use by the parent composer.
# They do not change the DATE/FACILITY family mix. Suggested ceiling: append one
# snippet to at most one eighth of fresh notes, only when the surrounding context
# is compatible. These snippets contain no names or dates.
#
# SPAN / TYPE CONTRACT: each listed placeholder occurs exactly once. Annotate
# exactly its replacement text, excluding caption punctuation. MRN, ID, and
# ACCOUNT use the parent's canonical taxonomy. ID covers the private device
# serials and the recipient routing credential. An ID or ACCOUNT value is a
# negative MRN example, not a PHI-free value.
# `owner` is personal for patient-linked identifiers, recipient for a receiving
# clinician's routing account, and public for published reference numbers.
# Public references have label None and are not annotated.
#
# VALUE CONTRACT: within each paragraph, assign distinct values with the SAME
# superficial format (for example, all eight-digit numbers, or all two-letter
# prefixes plus six digits). Do not reserve MRN/DEV/ACC prefixes or separate
# number lengths for individual labels. Context establishes type; the word
# "Record" itself has no fixed annotation meaning. No values are supplied here.
RETENTION_CONTEXTS = (
    {
        "id": "retention_chart_and_pump_log",
        "template": (
            "The admission chart was retrieved using the medical record number in patient administration. "
            "Record: {medical_record_id}. The equipment handover lists the serial number of the infusion pump "
            "assigned to the patient. Record: {device_id}."
        ),
        "mentions": [
            {"placeholder": "medical_record_id", "label": "MRN", "owner": "personal"},
            {"placeholder": "device_id", "label": "ID", "owner": "personal"},
        ],
    },
    {
        "id": "retention_portal_then_chart",
        "template": (
            "The help desk reopened the patient portal account after a failed sign-in. Account record: {account_id}. "
            "To attach the support call to the clinical chart, the clerk used the medical record number "
            "from the registration screen. Record: {medical_record_id}."
        ),
        "mentions": [
            {"placeholder": "account_id", "label": "ACCOUNT", "owner": "personal"},
            {"placeholder": "medical_record_id", "label": "MRN", "owner": "personal"},
        ],
    },
    {
        "id": "retention_article_reference_and_chart",
        "template": (
            "The attached journal article is indexed in an open publication catalogue. Catalogue record: "
            "{public_reference_id}. The follow-up request was filed under the medical record number shown "
            "on the patient chart. Record: {medical_record_id}."
        ),
        "mentions": [
            {"placeholder": "public_reference_id", "label": None, "owner": "public"},
            {"placeholder": "medical_record_id", "label": "MRN", "owner": "personal"},
        ],
    },
    {
        "id": "retention_chart_and_home_monitor",
        "template": (
            "The visit summary was saved to the chart identified by the patient's medical record number. "
            "Record: {medical_record_id}. The home monitoring kit was checked against its asset serial number "
            "before the patient took it home. Equipment record: {device_id}."
        ),
        "mentions": [
            {"placeholder": "medical_record_id", "label": "MRN", "owner": "personal"},
            {"placeholder": "device_id", "label": "ID", "owner": "personal"},
        ],
    },
    {
        "id": "retention_recipient_routing_and_patient_chart",
        "template": (
            "The receiving clinician supplied the routing identifier for the secure referral inbox. "
            "Recipient record: {account_id}. Before transmission, the sender matched the referral against "
            "the patient's medical record number on the chart cover. Record: {medical_record_id}."
        ),
        "mentions": [
            {"placeholder": "account_id", "label": "ID", "owner": "recipient"},
            {"placeholder": "medical_record_id", "label": "MRN", "owner": "personal"},
        ],
    },
    {
        "id": "retention_registry_brochure_after_chart",
        "template": (
            "The chart copy request carries the medical record number used for this patient's clinical file. "
            "Record: {medical_record_id}. A brochure discussed during the visit is listed in the public "
            "health-education catalogue. Publication record: {public_reference_id}."
        ),
        "mentions": [
            {"placeholder": "medical_record_id", "label": "MRN", "owner": "personal"},
            {"placeholder": "public_reference_id", "label": None, "owner": "public"},
        ],
    },
)


ALLOWED_PLACEHOLDERS = frozenset({
    "patient", "clinician", "care_site", "public_site", "place",
    "date_a", "date_b", "public_date", "interval_days",
})


def validate_families():
    """Check this authored module's internal contract without reading other data."""
    assert 24 <= len(FAMILIES) <= 36
    all_families = FAMILIES + NEGATIVE_FAMILIES
    assert len({family.family_id for family in all_families}) == len(all_families)
    assert len(NEGATIVE_FAMILIES) >= 6
    assert all(mention.ownership != "personal" and mention.target_label is None
               for family in NEGATIVE_FAMILIES for mention in family.mentions)
    for paragraph in CLINICAL_CONTEXT_PARAGRAPHS:
        assert "{" not in paragraph and "}" not in paragraph
    assert 4 <= len(RETENTION_CONTEXTS) <= 6
    assert len({item["id"] for item in RETENTION_CONTEXTS}) == len(RETENTION_CONTEXTS)
    for item in RETENTION_CONTEXTS:
        keys = [field for _, field, _, _ in Formatter().parse(item["template"]) if field is not None]
        assert set(keys) <= {"medical_record_id", "device_id", "account_id", "public_reference_id"}, item["id"]
        assert len(keys) == len(set(keys)), item["id"]
        assert sorted(keys) == sorted(mention["placeholder"] for mention in item["mentions"]), item["id"]
        assert sum(mention["label"] == "MRN" for mention in item["mentions"]) == 1, item["id"]
        assert any(mention["label"] != "MRN" for mention in item["mentions"]), item["id"]
        for mention in item["mentions"]:
            expected = {"medical_record_id": "MRN", "device_id": "ID",
                        "account_id": "ID" if mention["owner"] == "recipient" else "ACCOUNT",
                        "public_reference_id": None}[mention["placeholder"]]
            assert mention["label"] == expected, item["id"]
            assert (mention["owner"] == "public") == (mention["label"] is None), item["id"]
    for family in all_families:
        keys = [field for _, field, _, _ in Formatter().parse(family.template) if field is not None]
        assert set(keys) <= ALLOWED_PLACEHOLDERS, family.family_id
        observed = []
        counts = {}
        for key in keys:
            observed.append((key, counts.get(key, 0)))
            counts[key] = counts.get(key, 0) + 1
        declared = [(mention.placeholder, mention.occurrence) for mention in family.mentions]
        assert len(declared) == len(set(declared)), family.family_id
        assert set(observed) == set(declared), family.family_id
        for mention in family.mentions:
            expected = None
            if mention.ownership == "personal" and mention.entity_kind in {"date", "facility"}:
                expected = mention.entity_kind.upper()
            assert mention.target_label == expected, (family.family_id, mention)
        for order in family.date_order:
            assert order.before in keys and order.after in keys and order.minimum_days > 0, family.family_id
        for gap in family.date_gaps:
            assert {gap.start, gap.end, gap.days_placeholder} <= set(keys), family.family_id
        for phrase in family.non_entity_phrases:
            assert phrase in family.template, family.family_id
        for group in family.same_entity_groups:
            assert len(group) >= 2 and set(group) <= set(keys), family.family_id
            assert all(mention.entity_kind == "facility" and mention.ownership == "personal"
                       for mention in family.mentions if mention.placeholder in group), family.family_id
        for group in family.same_value_groups:
            assert len(group) >= 2 and set(group) <= set(keys), family.family_id
            # Equality must never collapse the personal timeline's strict order.
            assert not any(order.before in group and order.after in group for order in family.date_order), family.family_id


if __name__ == "__main__":
    validate_families()
    print(f"Validated {len(FAMILIES)} original focus families, "
          f"{len(NEGATIVE_FAMILIES)} PHI-free families, and "
          f"{len(CLINICAL_CONTEXT_PARAGRAPHS)} optional context paragraphs, and "
          f"{len(RETENTION_CONTEXTS)} identifier retention snippets.")
