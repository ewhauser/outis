"""Broad, self-authored clinical PHI supervision with no external dataset.

Code: Apache-2.0. Generated text and annotations: CC0-1.0.
The identity grammar, clinical prose, and supervision grammar are authored here.
No development generator, patient record, or identity dictionary is imported.
Development corpora may be read only for explicitly requested text-hash overlap
audits; qualification declarations are refused before any corpus payload read.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import re
import unicodedata

import tiktoken

from .types import ENTITY_TYPES


VERSION = "1.0"
SEED = 871043
CONSONANTS = (
    "b",
    "d",
    "f",
    "g",
    "h",
    "k",
    "l",
    "m",
    "n",
    "p",
    "r",
    "s",
    "t",
    "v",
    "z",
    "br",
    "dr",
    "ph",
    "sh",
    "tr",
)
VOWELS = ("a", "e", "i", "o", "u", "ae", "ai", "eo", "ia", "ou")
CODAS = ("", "l", "m", "n", "r", "s", "t", "v", "nd", "rk", "st")
NONVALUES = {
    "unknown",
    "none",
    "nil",
    "pending",
    "unavailable",
    "declined",
    "not available",
    "not recorded",
}
HEADINGS = (
    "Outpatient interval assessment",
    "Respiratory symptom check",
    "Neurologic follow-up summary",
    "Sleep and fatigue review",
    "Medication reconciliation",
    "Pain control assessment",
    "Nutrition progress record",
    "Functional capacity screening",
    "Cardiovascular follow-up",
    "Endocrine symptom assessment",
    "Morning care review",
    "Postprocedure recovery note",
    "Home safety assessment",
    "Joint motion examination",
    "General medicine update",
    "Infection monitoring note",
    "Clinical trajectory review",
    "Rehabilitation progress summary",
    "Sensory symptom assessment",
    "Hydration status record",
    "Clinical communications review",
    "Gait and balance assessment",
    "Renal function follow-up",
    "Treatment tolerance summary",
)
HEADER_TOPICS = (
    "exercise tolerance",
    "standing pressure response",
    "airway symptoms",
    "swallowing comfort",
    "bowel function",
    "fluid balance",
    "skin integrity",
    "night-time rest",
    "joint movement",
    "muscle endurance",
    "appetite changes",
    "sensory function",
    "walking stability",
    "treatment response",
    "wound healing",
    "cough progression",
    "medication tolerance",
    "daily activity",
    "pain pattern",
    "nutritional intake",
    "bladder symptoms",
    "visual comfort",
    "oral hydration",
    "recovery progress",
    "home safety",
    "support needs",
    "clinical observations",
    "symptom trajectory",
    "self-care capacity",
    "functional status",
)
HEADER_GRAMMAR = (
    "Interval account of {topic}",
    "Assessment concerning {topic}",
    "Current clinical record: {topic}",
    "Review of {topic} during recovery",
    "Ongoing monitoring — {topic}",
    "Follow-up observations on {topic}",
    "Care discussion: {topic}",
    "Clinical summary for {topic}",
    "Progress notes concerning {topic}",
    "Reassessment of {topic}",
    "Recorded observations: {topic}",
    "Routine assessment — {topic}",
    "Continuity review for {topic}",
    "Current findings regarding {topic}",
    "Patient-reported change in {topic}",
    "Functional review: {topic}",
)
SECTION_TITLES = (
    "History",
    "Reported symptoms",
    "Current treatment",
    "Examination",
    "Clinical measurements",
    "Care planning",
    "Continuity information",
    "Verification",
    "Supplementary record",
    "Assessment",
)
LAYOUTS = (
    "paragraph",
    "sectioned",
    "bullets",
    "numbered",
    "compact_fields",
    "worksheet",
    "indented",
    "handover",
)
CARE_SUFFIXES = (
    "Hospital",
    "Health Centre",
    "Primary Care Practice",
    "Recovery Clinic",
    "Nursing Service",
    "Infusion Centre",
    "Diagnostic Unit",
    "Surgical Practice",
    "Home Care Organisation",
    "Community Health Service",
    "Rehabilitation Facility",
    "Day Treatment Centre",
    "Specialty Clinic",
    "Medical Pavilion",
    "Outpatient Service",
    "Care Home",
)

# All frames are original grammatical supervision. They are neither lookup
# dictionaries nor transcriptions of a development note. Field grammar below
# adds noun/metadata/operator/position combinations on top of these frames.
FRAMES = {
    "NAME": (
        "After identity verification, {v} gave the account of symptoms.",
        "The account supplied by {v}'s family describes improved daily function.",
        "{v} is the individual being assessed in this note.",
        "At interview the patient identified themself as {v}.",
        "The accompanying relative is {v}, who confirmed the medication list.",
        "{v}, the examining clinician, checked gait and balance.",
        "Review of the patient by {v} did not change the management plan.",
        "Written observations were endorsed by {v}, RN.",
        "The booking carries {v} as the recipient's full name.",
        "The signer recorded their name as {v}.",
        "With consent from the patient, {v}, the referral was prepared.",
        "The individual is known locally as {v}.",
        "Collateral history was supplied by {v}, the patient's partner.",
        "The receiving practitioner, {v}, acknowledged the handover.",
        "Today's patient identification names {v} as the person receiving care.",
        "The nurse completing this assessment was {v}.",
    ),
    "LOCATION": (
        "The household remained in {v} during convalescence.",
        "Family support is available around {v}, where the patient currently stays.",
        "Home nursing was arranged in the locality of {v}.",
        "Residence district {op} {v}; transport options were assessed.",
        "During recovery the patient travelled from {v} to this review.",
        "The recent move into {v} was unrelated to the new cough.",
        "{v} is the town entered for the patient's living arrangements.",
        "The discharge planner discussed services serving the residential district of {v}.",
        "Local support near the community of {v} can assist with dressing changes.",
        "The individual considers {v} their usual home community.",
        "According to the family, accommodation was secured within {v}.",
        "Where the person resides was recorded as {v}.",
        "The locality chosen for home visits was {v}.",
        "A return journey towards {v} was planned after the assessment.",
        "Residential geography entered after verification {op} {v}.",
        "The patient's home is situated on the outskirts of {v}.",
    ),
    "FACILITY": (
        "The clinical handover went to {v} for continuity of care.",
        "Prior assessment records arrived from {v}.",
        "Appointments at {v} will monitor recovery.",
        "{v} is coordinating the visiting service.",
        "Day surgery was delivered through {v}.",
        "Receiving institution {op} {v}.",
        "Provider organisation entered on the referral {op} {v}.",
        "After-hours review took place inside {v}.",
        "The team affiliated with {v} will manage the wound.",
        "Discharge summaries were exchanged with {v}.",
        "The care organisation chosen by the family was {v}.",
        "The follow-up site was listed under {v}.",
        "The referral identifies {v} as the organisation accepting care.",
        "The previous inpatient service belonged to {v}.",
        "Correspondence bearing the name {v} confirmed an available appointment.",
        "{v} supplied the rehabilitation plan carried by the patient.",
    ),
    "DATE": (
        "The symptom diary places onset on {v}.",
        "An examination was completed on {v}.",
        "Consent for this episode was recorded on {v}.",
        "The next clinical review remains scheduled for {v}.",
        "The last personal home visit took place on {v}.",
        "Registration supplied {v} as the individual's appointment date.",
        "The operation is dated {v} in the person's clinical record.",
        "Attendance on {v} was confirmed by telephone.",
        "Clinical event calendar entry {op} {v}.",
        "First symptomatic night {op} {v}.",
        "The community team received this patient's referral on {v}.",
        "The date assigned to the individual's assessment is {v}.",
        "The record of treatment carries {v} as its clinical date.",
        "Symptoms were last reviewed on {v} rather than at the earlier visit.",
        "Personal follow-up date after discharge {op} {v}.",
        "Home monitoring began on {v} for this recipient.",
    ),
    "AGE": (
        "Chronological age at this visit {op} {v} years.",
        "The assessed individual is {v} years old.",
        "The recipient has reached {v} years and eats independently.",
        "For this assessment, the age supplied was {v} years.",
        "The history describes a {v}-year-old adult.",
        "Current patient age, measured in years {op} {v}.",
        "Age entered after collateral discussion {op} {v} years.",
        "The older person's age is recorded as {v} years.",
        "Years lived according to the patient {op} {v}.",
        "This assessment concerns an adult aged {v} years.",
        "Family members verified an age of {v} years for the care recipient.",
        "The demographic section gives {v} years as the patient's present age.",
        "Age confirmed during the home-care interview was {v} years.",
        "The recipient is currently {v} years of age.",
        "The person undergoing this evaluation has lived for {v} years.",
        "The reported chronological age is {v} years for this clinical encounter.",
    ),
    "ADDRESS": (
        "Visiting staff were given {v} as the patient's postal residence.",
        "Home delivery should be routed to {v} for this recipient.",
        "The individual's residential address has been verified as {v}.",
        "Registration retained the full home address {v}.",
        "The family asked that care supplies be sent to {v}.",
        "The location entered for the home assessment is {v}.",
        "Postal residence, including the delivery code {op} {v}.",
        "The patient confirmed living at {v}.",
        "The patient's dwelling is recorded in full as {v}.",
        "For home visits use the residential entry {v}.",
        "The mailing destination supplied for this person's care is {v}.",
        "The care coordinator verified {v} as the home address.",
        "The household's postal address appears as {v} on the referral.",
        "Deliver the patient's dressing supplies to {v}.",
        "The individual's address after relocation was given as {v}.",
        "Confirmed residential postal details {op} {v}.",
    ),
    "PHONE": (
        "The patient's callback line is {v}.",
        "A return call can reach the caregiver at {v}.",
        "Telephone contact supplied for appointment messages {op} {v}.",
        "The number to dial for this recipient was recorded as {v}.",
        "For discussion of symptoms, telephone the patient on {v}.",
        "The individual requested voice contact using {v}.",
        "An evening reply can be made to the personal telephone {v}.",
        "The registration team confirmed the patient's phone number as {v}.",
        "The caregiver's preferred call number is {v}.",
        "Messages about the clinical plan should be phoned to {v}.",
        "The reachable mobile number furnished by the patient was {v}.",
        "The personal contact record specifies {v} for telephone replies.",
        "The recipient can be contacted by calling {v} after lunch.",
        "A private telephone line for this consultation {op} {v}.",
        "The home-care form lists a patient contact number of {v}.",
        "The patient's telephone reply destination was checked as {v}.",
    ),
    "EMAIL": (
        "The electronic correspondence address belongs to the patient: {v}.",
        "The patient requested written instructions at {v}.",
        "The caregiver's reply mailbox is {v}.",
        "E-mail used for the individual's appointment messages {op} {v}.",
        "The registration email furnished by the patient was {v}.",
        "Consent permits clinical correspondence to be emailed to {v}.",
        "The patient's preferred electronic contact is <{v}>.",
        "The mailbox attached to the private patient record is {v}.",
        "For personal replies, the care recipient supplied {v}.",
        "The electronic address recorded after verification was {v}.",
        "The patient reads instructions sent to the personal email {v}.",
        "The household uses {v} for this person's clinical messages.",
        "An email reply should be directed to {v} for this recipient.",
        "The contact form specifies [{v}] as the patient's email.",
        "Verified e-mail for discharge correspondence {op} {v}.",
        "Personal email address attached to the assessment {op} {v}.",
    ),
    "URL": (
        "The individual's record can be accessed through {v}.",
        "A personal follow-up page was supplied as {v}.",
        "The private appointment link is {v}.",
        "The recipient's secure upload address was recorded as {v}.",
        "Personal portal URL after verification {op} {v}.",
        "The family was given <{v}> for this patient's care record.",
        "The patient's browser bookmark points to {v}.",
        "The web address attached to this individual's referral is {v}.",
        "The record-specific patient link appears as [{v}] on the instructions.",
        "The person's statement is available at the private URL {v}.",
        "Online access for this care recipient uses {v}.",
        "The individual supplied {v} as their clinical portal location.",
        "The personal portal path retained in the referral is {v}.",
        "The recipient's dedicated review page is located at {v}.",
        "The web link associated with the patient was verified as {v}.",
        "The handover includes {v} as a link to the private patient record.",
    ),
    "IP": (
        "The home telecare session originated from IP address {v}.",
        "The network address recorded for the private consultation was {v}.",
        "The individual's remote connection used {v} as its client IP.",
        "Patient session network host {op} {v}.",
        "The household's consultation device reported IP {v}.",
        "The patient's private portal login originated at {v}.",
        "The remote care record associates the person with network address {v}.",
        "Verified IP for the patient's home connection {op} {v}.",
        "The network client address attached to this clinical session is {v}.",
        "The connection log gives {v} for the individual's remote device.",
        "The recipient's home-care network endpoint was recorded as {v}.",
        "The patient's live consultation used the IP {v}.",
        "Host address for the personal telecare connection {op} {v}.",
        "The family connection supporting this patient's visit had IP {v}.",
        "The record-specific remote session was traced to {v}.",
        "The consultation's private patient network address is {v}.",
    ),
}

IDENTIFIER_FIELDS = {
    "MRN": (
        "medical record number",
        "longitudinal chart identifier",
        "hospital chart number",
        "patient's medical file key",
        "clinical record reference",
        "care-history file number",
        "patient chart key",
        "unique medical-record ID",
        "MRN",
        "outpatient clinical chart reference",
        "record number for the clinical history",
        "hospital record index",
        "care-chart number",
        "medical-chart locator",
        "patient-record reference",
        "clinical file identifier",
    ),
    "ACCOUNT": (
        "patient billing account",
        "visit finance account",
        "payment ledger number",
        "charge-account key",
        "financial account reference",
        "reimbursement account",
        "billing statement account",
        "patient's receivables account",
        "revenue-ledger account",
        "account number used for charges",
        "payment-account identifier",
        "finance account ID",
        "bill-account reference",
        "ledger account for this visit",
        "clinical billing account number",
        "personal charge ledger ID",
    ),
    "ID": (
        "insurance member number",
        "individual encounter number",
        "specimen accession",
        "patient's passport number",
        "driving licence reference",
        "health-plan policy identifier",
        "patient wristband ID",
        "home-care enrollment identifier",
        "implanted device serial number",
        "personal transport reservation key",
        "referral-tracking identifier",
        "patient registration key",
        "insurance policy key",
        "personal certificate number",
        "patient's service enrollment ID",
        "care-recipient encounter reference",
    ),
    "SSN": (
        "social security number",
        "person's social-security identifier",
        "national social security number",
        "private social-security registration number",
        "social-security verification number",
        "SSN",
        "social-security reference",
        "social-security identity key",
        "recipient's social security identifier",
        "private social security field",
        "social security registration reference",
        "patient's social-security number",
        "national social-security identifier",
        "social-security identity reference",
        "individual social security number",
        "recorded social-security ID",
    ),
}
IDENTIFIER_FRAMES = (
    "The {field} is {v} for this individual.",
    "Verification of the {field} produced {v} before this visit.",
    "Registration entered {v} under the person's {field}.",
    "The patient's {field}, checked during intake, reads {v}.",
    "The supplied {field} was retained as {v}.",
    "The recipient presented {v} as their {field}.",
    "The {field} attached to this patient's record {op} {v}.",
    "For this recipient the {field} was confirmed to be {v}.",
    "The recorded value {v} belongs to the individual's {field}.",
    "The field for the {field} was populated with {v} after verification.",
    "Under {field}, the form for this person lists {v}.",
    "The care coordinator verified {v} against the recipient's {field}.",
    "The {field} supplied by the person is documented as {v}.",
    "Intake associates the patient with {v}, their {field}.",
    "The {field} for the individual's care episode was entered as {v}.",
    "The private {field} is held in the record with value {v}.",
)
FIELD_NAMES = {
    "NAME": (
        "patient's stated name",
        "recipient identity",
        "full name of the individual",
        "name on the personal referral",
        "identification name",
        "name verified at interview",
    ),
    "LOCATION": (
        "home district",
        "residential locality",
        "town for home-care routing",
        "community of residence",
        "home geography",
        "locality supplied for this household",
    ),
    "FACILITY": (
        "receiving care organisation",
        "referring institution",
        "clinical establishment",
        "named treating organisation",
        "follow-up care site",
        "provider organisation",
    ),
    "DATE": (
        "date of personal clinical attendance",
        "individual follow-up date",
        "patient's treatment date",
        "clinical event date",
        "date in the symptom diary",
        "personal assessment date",
    ),
    "AGE": (
        "patient's age in years",
        "chronological age of the individual",
        "years of age at this assessment",
        "recipient age in years",
        "current age of the person",
        "verified age in years",
    ),
    "ADDRESS": (
        "patient's full postal address",
        "private residential mailing address",
        "home delivery address for care supplies",
        "postal dwelling address",
        "household street address",
        "residential postal entry",
    ),
    "PHONE": (
        "patient's telephone number",
        "individual callback phone",
        "caregiver telephone line",
        "personal contact phone",
        "telephone for care messages",
        "preferred voice-call number",
    ),
    "EMAIL": (
        "patient's email address",
        "personal electronic mailbox",
        "caregiver email",
        "e-mail for private care messages",
        "recipient's electronic reply address",
        "private correspondence email",
    ),
    "URL": (
        "patient's personal portal URL",
        "private patient-record link",
        "individual clinical web address",
        "personal follow-up URL",
        "recipient's secure record link",
        "private referral URL",
    ),
    "IP": (
        "patient telecare-session IP",
        "home connection network address",
        "private session client IP",
        "recipient's device IP address",
        "patient's network host address",
        "private consultation endpoint IP",
    ),
}
OPERATORS = (
    "is",
    "was verified as",
    "has the value",
    "was entered as",
    "is recorded as",
    "=",
    ":",
    "|",
    ">>",
    "—",
    "was supplied as",
    "now reads",
)
FIELD_METADATA = (
    "after intake verification",
    "as supplied during interview",
    "checked by registration",
    "recorded on the current form",
    "entered for this individual",
    "confirmed before handover",
    "retained for continuity",
    "reviewed with the patient",
    "following a record check",
    "for this care recipient",
    "at the start of the visit",
    "during the private identity check",
)

CLINICAL_PROFILES = (
    (
        "blood-pressure follow-up",
        "losartan 25 mg each morning",
        "No chest discomfort or new breathlessness was reported.",
        "Continue symptom observation and seated-to-standing pressure checks.",
    ),
    (
        "stable obstructive airway symptoms",
        "tiotropium 18 mcg inhaled daily",
        "Air entry is equal and no accessory-muscle use was observed.",
        "Inhaler technique and activity tolerance were discussed.",
    ),
    (
        "recovery from a minor lower-limb injury",
        "paracetamol 500 mg as needed for discomfort",
        "Weight bearing is improving and the skin remains intact.",
        "Gradual mobilisation and skin inspection were advised.",
    ),
    (
        "thyroid replacement review",
        "levothyroxine 50 mcg before breakfast",
        "There is no tremor, and appetite has remained stable.",
        "The current replacement dose will be reviewed with the next blood test.",
    ),
    (
        "diet-controlled glucose monitoring",
        "no regular glucose-lowering medication",
        "No symptomatic hypoglycaemia or unexpected weight loss was described.",
        "Dietary intake and home monitoring were reviewed.",
    ),
    (
        "reflux symptom follow-up",
        "pantoprazole 20 mg before the evening meal",
        "The abdomen is soft and there is no swallowing difficulty.",
        "Meal timing and treatment tolerance were discussed.",
    ),
    (
        "iron-deficiency follow-up",
        "ferrous fumarate 210 mg every other day",
        "There is no visible bleeding and exertional symptoms are improving.",
        "Tolerance of supplementation and repeat laboratory testing were reviewed.",
    ),
    (
        "sleep and mood assessment",
        "sertraline 25 mg daily",
        "Sleep is more regular and no acute distress was observed.",
        "Sleep routine, appetite, and family support were discussed.",
    ),
    (
        "osteoarthritis symptom review",
        "topical diclofenac used on the painful joint",
        "Movement causes mild discomfort but distal strength is preserved.",
        "Joint protection and safe activity were discussed.",
    ),
    (
        "postinfectious recovery",
        "no continuing antimicrobial treatment",
        "The cough is resolving and there is no current fever.",
        "Hydration, rest, and warning symptoms were reviewed.",
    ),
    (
        "fluid-status monitoring",
        "torsemide 5 mg each morning",
        "There is no resting breathlessness and only mild dependent swelling.",
        "Weight trends, oral intake, and treatment response were reviewed.",
    ),
    (
        "sensory symptom assessment",
        "no new medication was prescribed",
        "Sensation is intact on screening and no focal motor deficit was identified.",
        "Function, fall risk, and symptom progression were discussed.",
    ),
)
PUBLIC_CONTROLS = (
    "The literature note cites a guideline published in {year}; this is a publication date, not a patient event.",
    "Research indexing includes PubMed PMID {digits} and study accession EVID-{code}.",
    "A public trial listing uses trial identifier NCT{digits}; no person is linked to that catalogue entry.",
    "The medicine inventory uses NDC {ndc} and product lot MED-{code} as manufacturer references.",
    "Manufacturer documentation gives device catalogue ref DEV-{code} and model number SYS-{short}.",
    "Procedure code CPT 99214 and diagnosis code ICD-10 I10 are clinical classification entries.",
    "A public equipment manual revised in {month} {year} lists serial format CAT-{code} for demonstration.",
    "Research protocol PRO-{code} and registry index {digits} describe a public study, not this person's identifiers.",
    "The pharmacy's product card records batch DRUG-{short} and medicine catalogue number {digits}.",
    "The teaching table describes Wilson disease, Parkinson disease and Baker cyst as clinical conditions.",
    "Discussion of Graves disease, Addison disease and Raynaud phenomenon concerns terminology only.",
    "The screening explanation mentions Romberg testing, Babinski signs and McBurney point tenderness.",
    "Generic outpatient services, the emergency department and the rehabilitation unit were discussed without a named organisation.",
    "Reference measurements include Na 137 mmol/L, K 4.2 mmol/L and Cl 102 mmol/L.",
    "A published reference table from {year} lists sodium 135-145 mmol/L and potassium 3.5-5.0 mmol/L.",
    "Teaching scores include pain 3/10, motor power 5/5 and cognitive screening 28/30.",
)
CLINICAL_LANGUAGE = (
    "We checked treatment handling together and agreed to continue the prescribed regimen.",
    "I reviewed the examination findings with the team and explained the monitoring plan.",
    "Our review focused on appetite, daytime stamina and the ability to manage washing independently.",
    "You demonstrated the exercise once more while the therapist checked comfort and technique.",
    "Head movement was smooth; oral mucosa was moist and the skin showed no new breakdown.",
    "Cough strength improved after coached breathing; fatigue settled following a brief seated rest.",
    "Monofilament responses were present and otoscopy showed no new canal inflammation.",
    "Presyncope was absent during standing, and disequilibrium did not recur with slow turning.",
    "Inflammation diminished with local care; medication adjustment was deferred pending reassessment.",
    "Airflow through the spacer appeared adequate; inspiratory effort stayed even during a timed practice task.",
    "Oxygen saturation remained stable through the transfer, with no additional work of breathing.",
    "Controller handling and adherence were checked using a demonstration, followed by teach-back.",
    "Handoff: the team discussed symptom monitoring, treatment tolerance and the plan for escalation.",
    "Teach-back: the recipient restated the medicine schedule and signs that warrant urgent clinical review.",
    "Evidence review: symptom observations support gradual progression of activity as tolerated.",
    "Observation: walking pace remained even and upper-limb power was maintained during the task.",
    "S: symptom change was discussed. O: observations were reviewed. A: recovery is gradual. P: maintain the agreed monitoring.",
    "A/P: continue routine support and review clinical warning symptoms.",
    "Procedure discussion: simple wound care and pressure-relief positioning were demonstrated.",
    "Adherence check: the timing of regular treatment was restated accurately during the conversation.",
    "Progress assessment: the recipient completed a timed reach-and-stand task without new dizziness.",
    "Therapy monitoring: comfort and physical effort were checked throughout the practice period.",
    "Support planning: staff described how to request help if symptoms change between reviews.",
    "Examination notes: the mouth was moist, the neck moved freely and limb power was symmetrical.",
)


class Note:
    """Concatenation owns offsets; values are never recovered with substring search."""

    def __init__(self):
        self.parts, self.spans, self.contexts = [], [], []
        self.length = 0

    def plain(self, value):
        self.parts.append(value)
        self.length += len(value)

    def phi(self, value, label):
        self.spans.append(
            {"start": self.length, "end": self.length + len(value), "label": label}
        )
        self.plain(value)

    def clause(self, frame, label, value, **arguments):
        if frame.count("{v}") != 1:
            raise ValueError("A supervision clause must contain one marked value")
        before, after = frame.split("{v}")
        self.plain(before.format(**arguments))
        self.phi(value, label)
        # A surname suffix such as "Jr." can also close its sentence; keep
        # its lexical period inside gold without producing a doubled stop.
        if value.endswith(".") and after.startswith("."):
            after = after[1:]
        self.plain(after.format(**arguments))

    @property
    def text(self):
        return "".join(self.parts)


def identity_word(rng, *, short=False):
    """A fresh productive letter grammar, with optional Unicode variants."""
    while True:
        syllables = 1 if short else rng.choice((2, 2, 3))
        value = "".join(
            rng.choice(CONSONANTS) + rng.choice(VOWELS) for _ in range(syllables)
        )
        if not short or rng.random() < 0.7:
            value += rng.choice(CODAS)
        value = value.capitalize()
        if value.casefold() in NONVALUES:
            continue
        if rng.random() < 0.18:
            choices = [i for i, character in enumerate(value) if character in "aeiou"]
            if choices:
                position = rng.choice(choices)
                accents = {"a": "á", "e": "ë", "i": "í", "o": "ö", "u": "ü"}
                value = (
                    value[:position] + accents[value[position]] + value[position + 1 :]
                )
                if rng.random() < 0.35:
                    value = unicodedata.normalize("NFD", value)
        return value


def person(rng):
    first = identity_word(rng, short=rng.random() < 0.30)
    surname = identity_word(rng)
    style = rng.randrange(12)
    if style < 3:
        return first
    if style == 3:
        initial = first[0]
        for character in first[1:]:
            if not unicodedata.category(character).startswith("M"):
                break
            initial += character
        return f"{initial}. {surname}"
    if style == 4:
        return f"{surname}, {first}"
    if style == 5:
        return f"{first} {rng.choice(('de', 'van', 'al', 'von'))} {surname}"
    if style == 6:
        return f"{first} O’{surname}"
    value = f"{first} {surname}"
    if style in {10, 11}:
        return value + " " + rng.choice(("Jr.", "Sr.", "II", "III"))
    return value.upper() if style == 9 else value


def code_value(rng):
    """Shared morphology forces MRN/ACCOUNT/ID typing to depend on context."""
    width = rng.choice((8, 9))
    digits = str(rng.randrange(10 ** (width - 1), 10**width))
    prefix = "".join(rng.choice("BCDFGHJKLMNPRSTVZ") for _ in range(2))
    return rng.choice(
        (
            digits,
            f"{prefix}-{digits[:4]}-{digits[4:]}",
            f"{prefix}{digits}",
            f"{digits[:3]} {digits[3:5]} {digits[5:]}",
        )
    )


def age_words(age):
    if not isinstance(age, int) or isinstance(age, bool) or not 90 <= age <= 119:
        raise ValueError("Authored older-adult age must be an integer from 90 to 119")
    units = (
        "",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    )
    if age < 100:
        return "ninety" + ("-" + units[age - 90] if age > 90 else "")
    return "one hundred" + (" and " + units[age - 100] if age > 100 else "")


def date_value(rng, value):
    return rng.choice(
        (
            value.isoformat(),
            value.strftime("%m/%d/%Y"),
            value.strftime("%d %b %Y"),
            value.strftime("%B %d, %Y"),
            value.strftime("%b %Y"),
            value.strftime("%B %Y"),
            str(value.year),
        )
    )


def _ascii(value):
    return (
        re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", value).casefold())
        or "reply"
    )


def identity_values(rng):
    patient, clinician = person(rng), person(rng)
    relative = person(rng)
    locality = identity_word(rng, short=rng.random() < 0.30)
    if rng.random() < 0.35:
        locality += " " + rng.choice(
            ("Meadow", "Crossing", "Reach", "Vale", "District", "Shore")
        )
    # A note's home-community and current residential address describe the
    # same locality. Mailing labels may still use the same town coherently.
    street, postal_town = identity_word(rng), locality
    house, postal, apartment = (
        rng.randrange(12, 980),
        rng.randrange(10000, 99999),
        rng.randrange(1, 90),
    )
    state = rng.choice(("ID", "OR", "IN", "MD", "ME", "TX", "VT", "CO", "NM"))
    address = rng.choice(
        (
            f"{house} {street} {rng.choice(('Road', 'Street', 'Lane', 'Avenue'))}, {postal_town}, {state} {postal}",
            f"{house} {street} Road; {postal_town} {state} {postal}",
            f"Flat {apartment}, {house} {street} Lane, {postal_town}, AB{postal % 20 + 1} {postal % 10}XY",
            f"{house} rue {street}, {postal} {postal_town}, France",
            f"{street}straße {house}, {postal} {postal_town}, Germany",
            f"{house} avenue {street}, Appartement {apartment}; {postal_town} {postal}",
            f"{street}väg {house}, {postal} {postal_town}, Sweden",
            f"PO Box {house}, {postal_town}, {state} {postal}",
        )
    )
    domain = _ascii(identity_word(rng)) + ".example"
    local = (
        _ascii(identity_word(rng))
        + rng.choice(("", ".care", "+reply", "_home", ".family"))
        + str(rng.randrange(10, 99))
    )
    telephone = rng.choice(
        (
            f"+1 ({rng.randrange(300, 850)}) 555-01{rng.randrange(10, 99)}",
            f"{rng.randrange(300, 850)}.555.01{rng.randrange(10, 99)}",
            f"+44 20 7946 01{rng.randrange(10, 99)}",
            f"+1 {rng.randrange(300, 850)} 555 01{rng.randrange(10, 99)} ext. {rng.randrange(10, 70)}",
            f"({rng.randrange(300, 850)}) 555-01{rng.randrange(10, 99)} x{rng.randrange(10, 70)}",
        )
    )
    ssn_digits = str(rng.randrange(100000000, 899999999))
    social = rng.choice(
        (
            ssn_digits,
            f"{ssn_digits[:3]}-{ssn_digits[3:5]}-{ssn_digits[5:]}",
            f"{ssn_digits[:3]} {ssn_digits[3:5]} {ssn_digits[5:]}",
        )
    )
    episode = date(2010, 1, 1) + timedelta(days=rng.randrange(5700))
    age = rng.randrange(90, 120)
    return {
        "NAME": patient,
        "clinician": clinician,
        "relative": relative,
        "LOCATION": locality,
        "FACILITY": identity_word(rng) + " " + rng.choice(CARE_SUFFIXES),
        "DATE": date_value(rng, episode),
        "onset": date_value(rng, episode - timedelta(days=rng.randrange(2, 15))),
        "followup": date_value(rng, episode + timedelta(days=rng.randrange(14, 43))),
        "episode_year": episode.year,
        "AGE": age_words(age) if rng.random() < 0.25 else str(age),
        "ADDRESS": address,
        "PHONE": telephone,
        "EMAIL": f"{local}@{domain}",
        "MRN": code_value(rng),
        "SSN": social,
        "ACCOUNT": code_value(rng),
        "ID": code_value(rng),
        "URL": f"https://{domain}/personal/{rng.randrange(1000, 9999)}?case={rng.randrange(10000, 99999)}&review=1",
        "IP": rng.choice(
            (
                f"192.0.2.{rng.randrange(1, 255)}",
                f"198.51.100.{rng.randrange(1, 255)}",
                f"2001:db8:{rng.randrange(1, 65535):x}::{rng.randrange(1, 65535):x}",
            )
        ),
    }


def public_control(rng, index, *, max_year=2022):
    return PUBLIC_CONTROLS[index % len(PUBLIC_CONTROLS)].format(
        year=rng.randrange(1998, min(2022, max_year) + 1),
        month=rng.choice(("January", "April", "July", "October")),
        digits=f"{rng.randrange(10000000, 99999999)}",
        code=f"{rng.randrange(100000, 999999)}",
        short=rng.randrange(1000, 9999),
        ndc=f"{rng.randrange(10000, 99999)}-{rng.randrange(100, 999)}-{rng.randrange(10, 99)}",
    )


def clinical_body(rng, profile, *, negative, index, short=False):
    condition, treatment, finding, plan = CLINICAL_PROFILES[profile]
    systolic, diastolic = rng.randrange(108, 149), rng.randrange(60, 88)
    pulse, respirations = rng.randrange(58, 96), rng.randrange(12, 21)
    glucose, creatinine = rng.randrange(84, 192), rng.randrange(7, 15) / 10
    text = (
        f"Review concerns {condition}. Current treatment: {treatment}. "
        f"BP {systolic}/{diastolic} mmHg; pulse {pulse}/min; respirations {respirations}/min; "
        f"temperature {rng.randrange(362, 374) / 10:.1f} C; oxygen saturation {rng.randrange(94, 100)}%. "
        f"{finding} Glucose {glucose} mg/dL and creatinine {creatinine:.1f} mg/dL were reviewed. "
        f"{plan}"
    )
    if not short:
        text += f" Pain was {rng.randrange(0, 6)}/10, power 5/5, and cognitive screening {rng.randrange(24, 31)}/30."
        duration = rng.randrange(90, 120)
        if rng.random() < 0.25:
            text += f" Monitoring over {age_words(duration)} days concerns duration rather than age."
        else:
            text += f" A {duration}-day monitoring interval concerns duration, not age."
    if negative:
        text += f" The patient is {rng.randrange(18, 90)} years old. Age below 90 remains clinical information."
        if index % 3 == 0:
            text += (
                " Patient name: Unknown; the note contains no identified individual."
            )
    if negative or not short:
        # The same language appears beside PHI and in fully clean notes. That
        # prevents absence of identity fields from becoming the negative cue.
        count = 2 if negative else 1
        selected = rng.sample(range(len(CLINICAL_LANGUAGE)), count)
        text += " " + " ".join(CLINICAL_LANGUAGE[item] for item in selected)
    return text


def _separator(note, layout, ordinal, rng):
    if ordinal == 0:
        return
    if layout == "paragraph":
        note.plain(" ")
    elif layout == "bullets":
        note.plain("\n- ")
    elif layout == "numbered":
        note.plain(f"\n{ordinal}. ")
    elif layout == "compact_fields":
        note.plain(" | ")
    elif layout == "worksheet":
        note.plain("\n  ")
    elif layout == "indented":
        note.plain("\n    ")
    elif layout == "sectioned":
        note.plain(f"\n{rng.choice(SECTION_TITLES)}:\n")
    else:
        note.plain("\n")


def clinical_layout(body, layout):
    """Clinical paragraphs and headings receive the same layout diversity as PHI."""
    if layout in {"paragraph", "compact_fields", "handover"}:
        return body
    sentences = body.split(". ")
    sentences = [
        sentence + ("." if i < len(sentences) - 1 else "")
        for i, sentence in enumerate(sentences)
    ]
    if layout == "sectioned":
        return "\n".join(
            f"{SECTION_TITLES[i % len(SECTION_TITLES)]}: {sentence}"
            for i, sentence in enumerate(sentences)
        )
    if layout == "numbered":
        return "\n".join(f"{i + 1}. {sentence}" for i, sentence in enumerate(sentences))
    prefix = "- " if layout == "bullets" else "    " if layout == "indented" else "  "
    return "\n".join(prefix + sentence for sentence in sentences)


def _field_frame(rng, label):
    fields = IDENTIFIER_FIELDS.get(label, FIELD_NAMES.get(label))
    field = rng.choice(fields)
    metadata = rng.choice(FIELD_METADATA)
    arrangement = rng.randrange(8)
    frames = (
        "{field}, {metadata}, {op} {v}.",
        "For this individual, {field} {op} {v}; {metadata}.",
        "{v} was retained as the {field}, {metadata}.",
        "{metadata}: {field} {op} {v}.",
        "The patient's form gives {v} for the {field}; {metadata}.",
        "{field} [{metadata}] {op} {v}.",
        "The value for {field} {op} {v}, {metadata}.",
        "{field} {op} {v} | {metadata}.",
    )
    if arrangement in {0, 3, 5, 7} and rng.random() < 0.45:
        field = field.upper()
    return (
        frames[arrangement],
        {"field": field, "metadata": metadata},
        f"field-{arrangement}",
    )


def generate_row(index, *, seed=SEED, attempt=0):
    rng = random.Random((seed << 32) + index * 65537 + attempt)
    negative = index % 25 < 4
    layout = LAYOUTS[(index // 25 + index) % len(LAYOUTS)]
    profile = (index // len(LAYOUTS)) % len(CLINICAL_PROFILES)
    note = Note()
    heading = HEADINGS[index % len(HEADINGS)]
    if index % 4:
        heading = rng.choice(HEADER_GRAMMAR).format(topic=rng.choice(HEADER_TOPICS))
    heading_style = rng.randrange(5)
    if heading_style == 0:
        heading = heading.upper()
    elif heading_style == 1:
        heading = heading.title()
    elif heading_style == 2:
        heading = f"[{heading}]"
    elif heading_style == 3:
        heading = "Clinical note — " + heading
    note.plain(heading + "\n")
    dense = not negative and index % 5 == 4
    body = clinical_body(rng, profile, negative=negative, index=index, short=dense)
    body = clinical_layout(body, layout)
    if negative:
        note.plain(
            body
            + "\n"
            + public_control(rng, index)
            + "\n"
            + public_control(rng, index + 7)
        )
    else:
        values = identity_values(rng)
        if dense:
            labels = list(ENTITY_TYPES)
        else:
            labels = rng.sample(list(ENTITY_TYPES), rng.randrange(4, 11))
            if "NAME" not in labels and index % 3 == 0:
                labels.append("NAME")
        rng.shuffle(labels)
        body_first = index % 3 != 0
        if body_first:
            note.plain(body + "\n")
        for ordinal, label in enumerate(labels, 1):
            _separator(note, layout, ordinal, rng)
            # Independent grammar choices avoid coupling note position or layout
            # to a particular type-specific phrase.
            frame_index = rng.randrange(16)
            if index % 4 == 0 or (ordinal + index) % 7 == 0:
                frame, arguments, context = _field_frame(rng, label)
            elif label in IDENTIFIER_FIELDS:
                frame = IDENTIFIER_FRAMES[frame_index]
                field = rng.choice(IDENTIFIER_FIELDS[label])
                # Possessive subjects already establish whose identifier it is.
                # Avoid nesting "patient's patient's" in productive combinations.
                if frame_index in {2, 3, 8, 11}:
                    field = re.sub(r"^(?:patient|person|recipient)'s ", "", field)
                arguments = {"field": field}
                context = f"narrative-{frame_index}"
            else:
                frame, arguments = FRAMES[label][frame_index], {}
                context = f"narrative-{frame_index}"
            arguments["op"] = rng.choice(OPERATORS)
            value = values[label]
            if (
                label == "DATE"
                and context.startswith("narrative-")
                and frame_index in {0, 9}
            ):
                value = values["onset"]
            elif label == "DATE" and (
                (context.startswith("narrative-") and frame_index in {3, 14})
                or "follow-up" in arguments.get("field", "").casefold()
            ):
                value = values["followup"]
            elif label == "NAME" and context.startswith("narrative-"):
                if frame_index in {5, 6, 7, 9, 13, 15}:
                    value = values["clinician"]
                elif frame_index in {4, 12}:
                    value = values["relative"]
            note.clause(frame, label, value, **arguments)
            note.contexts.append({"label": label, "frame": context})
        # Repeat values in independently grammatical person or temporal roles;
        # some records have no name-field caption at all.
        if "NAME" in labels and index % 3 == 0:
            note.plain("\n")
            note.clause(
                "The person giving the history, {v}, confirmed that symptoms improved.",
                "NAME",
                values["NAME"],
            )
            note.contexts.append({"label": "NAME", "frame": "relative-role-repeat"})
        if "NAME" in labels and index % 6 == 0:
            note.plain(" ")
            note.clause(
                "The assessment was checked by nurse {v} before handover.",
                "NAME",
                values["clinician"],
            )
            note.contexts.append(
                {"label": "NAME", "frame": "passive-professional-repeat"}
            )
        if not body_first:
            note.plain("\n" + body)
        note.plain("\n" + public_control(rng, index, max_year=values["episode_year"]))
    row = {
        "id": f"generalization-train-{seed}-{index:05d}",
        "text": note.text,
        "spans": note.spans,
        "split": "train",
        "suite": "authored_generalization_training",
        "family": f"generalization.{layout}.{'phi_free' if negative else 'dense' if dense else 'mixed'}",
        "clinical_profile": profile,
        "authored_contexts": note.contexts,
        "negative": negative,
    }
    validate_row(row)
    return row


def validate_row(row):
    if (
        row.get("split") != "train"
        or row.get("suite") != "authored_generalization_training"
    ):
        raise ValueError("This generator is exclusively authored training data")
    previous = 0
    for span in row["spans"]:
        start, end = span["start"], span["end"]
        if (
            not 0 <= start < end <= len(row["text"])
            or start < previous
            or span["label"] not in ENTITY_TYPES
        ):
            raise ValueError(
                "Gold annotations must be valid ordered disjoint character ranges"
            )
        if not row["text"][start:end].strip():
            raise ValueError("PHI annotations must contain a value")
        previous = end
    if row["negative"] != (not row["spans"]):
        raise ValueError("PHI-free annotation policy and note metadata disagree")


def token_audit(row, encoding):
    """Match the production tokenizer projection without loading any model."""
    from opf._core.spans import token_char_ranges_for_text

    ids = encoding.encode(row["text"], disallowed_special=())
    starts, ends = token_char_ranges_for_text(ids, encoding, row["text"])
    tags, occupied = Counter(), set()
    for span in row["spans"]:
        indices = [
            token
            for token, (start, end) in enumerate(zip(starts, ends))
            if start < span["end"]
            and end > span["start"]
            and row["text"][max(start, span["start"]) : min(end, span["end"])].strip()
        ]
        if not indices or occupied.intersection(indices):
            raise ValueError(
                "Gold PHI values must have distinct nonempty tokenizer-token ownership"
            )
        occupied.update(indices)
        if len(indices) == 1:
            tags[f"S-{span['label']}"] += 1
        else:
            tags[f"B-{span['label']}"] += 1
            tags[f"E-{span['label']}"] += 1
            tags[f"I-{span['label']}"] += len(indices) - 2
    tags["O"] += len(ids) - len(occupied)
    return len(ids), tags


def comparison_hashes(paths):
    """Exact-note contamination audit; never inspect qualification payloads."""
    declarations = []
    for path in paths:
        path = Path(path)
        sidecar = path.with_suffix(path.suffix + ".manifest.json")
        if sidecar.exists():
            declaration_payload = sidecar.read_bytes()
            manifest = json.loads(declaration_payload)
        else:
            # The first two authored corpora predate adjacent sidecars. Their
            # exact filename/hash entries remain pinned in aggregate manifests.
            manifest = None
            for candidate in ("manifest.json", "augmentation_manifest.json"):
                aggregate_path = path.parent / candidate
                if not aggregate_path.exists():
                    continue
                declaration_payload = aggregate_path.read_bytes()
                aggregate = json.loads(declaration_payload)
                for split, entry in aggregate.get("datasets", {}).items():
                    if entry.get("file") == path.name:
                        manifest = {
                            "sha256": entry["sha256"],
                            "rows": entry.get("count", entry.get("rows")),
                            "suite": aggregate.get("suite"),
                            "split": split,
                            "independent_final": aggregate.get("independent_final"),
                        }
                        sidecar = aggregate_path
                        break
                if manifest is not None:
                    break
            if manifest is None:
                raise ValueError("Comparison corpus needs a pinned declaration")
        if (
            manifest.get("suite") in {"final", "test"}
            or manifest.get("split") == "test"
            or manifest.get("independent_final") is True
        ):
            raise ValueError("Qualification corpora cannot be opened by this generator")
        declarations.append(
            (path, manifest, sidecar, hashlib.sha256(declaration_payload).hexdigest())
        )
    records = []
    for path, manifest, sidecar, sidecar_hash in declarations:
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != manifest.get("sha256"):
            raise ValueError("Comparison corpus does not match its declaration")
        rows = [
            json.loads(line)
            for line in payload.decode("utf-8").splitlines()
            if line.strip()
        ]
        if manifest.get("rows") is not None and len(rows) != manifest["rows"]:
            raise ValueError(
                "Comparison corpus row count does not match its declaration"
            )
        if any(
            row.get("split") == "test" or row.get("suite") in {"final", "test"}
            for row in rows
        ):
            raise ValueError("Comparison rows must be declared training or development")
        records.append(
            {
                "path": str(path.resolve()),
                "sha256": digest,
                "rows": len(rows),
                "declaration": str(sidecar.resolve()),
                "declaration_sha256": sidecar_hash,
                "text_hashes": {
                    hashlib.sha256(row["text"].encode()).hexdigest() for row in rows
                },
            }
        )
    return records


def write_dataset(
    output="data", train_count=10000, seed=SEED, *, compare_corpora=(), max_tokens=512
):
    if train_count < 1 or max_tokens < 100:
        raise ValueError("Count must be positive and token ceiling at least 100")
    comparison = comparison_hashes(compare_corpora)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "generalization_train.jsonl"
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    if path.exists() or sidecar.exists():
        raise FileExistsError("Authored corpora are immutable; choose a new directory")
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(".cache/tiktoken").resolve()))
    encoding = tiktoken.get_encoding("o200k_base")
    labels, tags, families, layouts = Counter(), Counter(), Counter(), Counter()
    contexts = defaultdict(Counter)
    lengths, hashes, negative, retries = [], set(), 0, 0
    ages, age_styles = Counter(), Counter()
    word_ages = {age_words(age): age for age in range(90, 120)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for index in range(train_count):
                for attempt in range(30):
                    row = generate_row(index, seed=seed, attempt=attempt)
                    length, row_tags = token_audit(row, encoding)
                    if length <= max_tokens:
                        break
                else:
                    raise ValueError("Could not author a note within the token ceiling")
                retries += attempt
                digest = hashlib.sha256(row["text"].encode()).hexdigest()
                if digest in hashes or any(
                    digest in corpus["text_hashes"] for corpus in comparison
                ):
                    raise ValueError(
                        "Authored note overlaps another note or declared corpus"
                    )
                hashes.add(digest)
                labels.update(span["label"] for span in row["spans"])
                for span in row["spans"]:
                    if span["label"] == "AGE":
                        value = row["text"][span["start"] : span["end"]]
                        ages[int(value) if value.isdigit() else word_ages[value]] += 1
                        age_styles["numeric" if value.isdigit() else "written"] += 1
                tags.update(row_tags)
                families[row["family"]] += 1
                layouts[row["family"].split(".")[1]] += 1
                negative += row["negative"]
                lengths.append(length)
                for context in row["authored_contexts"]:
                    contexts[context["label"]][context["frame"]] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        payload_hash = hashlib.sha256(temporary.read_bytes()).hexdigest()
        sorted_lengths = sorted(lengths)
        manifest = {
            "schema_version": 1,
            "generator": "clinical_deid.generalization_synthetic",
            "generator_version": VERSION,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "seed": seed,
            "rows": train_count,
            "sha256": payload_hash,
            "split": "train",
            "suite": "authored_generalization_training",
            "license": "CC0-1.0",
            "code_license": "Apache-2.0",
            "external_datasets": [],
            "source": "Entirely self-authored clinical prose and productive identity/semantic grammar; no patient records, identity dictionaries, development-generator imports, or copied development text.",
            "independent_final": False,
            "used_for_training": True,
            "used_for_checkpoint_selection": False,
            "phi_free_notes": negative,
            "phi_free_fraction": negative / train_count,
            "spans": sum(labels.values()),
            "label_support": {label: labels[label] for label in ENTITY_TYPES},
            "families": dict(sorted(families.items())),
            "layout_support": dict(sorted(layouts.items())),
            "semantic_context_support": {
                label: dict(sorted(contexts[label].items())) for label in ENTITY_TYPES
            },
            "distinct_contexts_per_label": {
                label: len(contexts[label]) for label in ENTITY_TYPES
            },
            "tokenizer": "o200k_base / OPF original Unicode character projection",
            "token_length": {
                "min": min(lengths),
                "mean": sum(lengths) / len(lengths),
                "p95": sorted_lengths[int(0.95 * (len(lengths) - 1))],
                "max": max(lengths),
                "ceiling": max_tokens,
            },
            "bioes_tag_support": dict(sorted(tags.items())),
            "annotation_audit": {
                "rows_validated": train_count,
                "token_aligned_rows": train_count,
                "overlapping_gold_spans": 0,
                "shared_phi_tokens": 0,
                "duplicate_note_texts": 0,
                "generation_length_retries": retries,
            },
            "age_policy": "Only patient age greater than 89 years is PHI; numeric and spelled-out age values are labeled, while younger ages, doses, scores and temporal durations remain clinical.",
            "authored_age_range": [90, 119],
            "age_value_support": {
                str(age): count for age, count in sorted(ages.items())
            },
            "age_style_support": dict(sorted(age_styles.items())),
            "date_policy": "Patient clinical dates are PHI, including partial month/year precision; explicit publication/reference dates are public clinical negatives.",
            "clinical_coherence": {
                "home_and_residential_postal_locality_are_shared": True,
                "onset_precedes_episode_followup_follows_episode": True,
                "reference_publication_year_not_after_episode_year": True,
                "patient_clinician_relative_names_match_declared_roles": True,
            },
            "identifier_policy": "MRN means clinical chart/file; ACCOUNT means private billing/finance account; ID means personal encounter/accession/member/licence/device-assignment identifier; SSN means social-security number. Public study/manufacturer/drug/classification references are negatives.",
            "comparison_audit": [
                {key: value for key, value in corpus.items() if key != "text_hashes"}
                | {"exact_text_overlap": 0}
                for corpus in comparison
            ],
        }
        if train_count >= 8000 and (
            negative < 1000 or any(len(contexts[label]) <= 8 for label in ENTITY_TYPES)
        ):
            raise ValueError(
                "Large authored corpus must retain 1000 negatives and more than eight contexts per PHI type"
            )
        temporary.rename(path)
        with sidecar.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return manifest
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data", type=Path)
    parser.add_argument("--train-count", default=10000, type=int)
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--max-tokens", default=512, type=int)
    parser.add_argument(
        "--compare-corpus",
        action="append",
        default=[],
        help="Declared existing train/development corpus; hash-audit only, never qualification",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            write_dataset(
                args.output,
                args.train_count,
                args.seed,
                compare_corpora=args.compare_corpus,
                max_tokens=args.max_tokens,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
