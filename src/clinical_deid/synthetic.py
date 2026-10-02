"""Self-authored synthetic clinical notes with exact PHI span annotations.

No patient records, public clinical corpora, or licensed datasets are used.  The
clinical facts, vocabulary, and templates in this file were authored for this
project.  Training and validation use separate note families and separate PHI
identity vocabularies.  Clinical vocabulary is deliberately shared: diagnoses,
medication doses, and test results must remain available to the encoder.

Offsets are Python Unicode character offsets into the original text.  Values are
inserted as marked fields, rather than subsequently located with text matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Literal

from .types import ENTITY_TYPES

Split = Literal["train", "validation"]
SCHEMA_VERSION = 1
GENERATOR_VERSION = "1.0"

# These invented identities do not originate in a person-level dataset.  Split
# vocabularies are intentionally disjoint, including places and organizations.
VOCABULARY: dict[str, dict[str, tuple[str, ...]]] = {
    "train": {
        "first": (
            "Adela", "Bastian", "Celia", "Dorian", "Elara", "Finley", "Greta", "Hugo",
            "Iona", "Jasper", "Kira", "Leander", "Maren", "Nolan", "Orla", "Pascal",
            "Quinn", "Rhea", "Soren", "Talia", "Uma", "Vera", "Wren", "Xenia",
            "Yara", "Zev", "Amara", "Brenna", "Cato", "Dahlia", "Emrys", "Faye",
            "Galen", "Hana", "Isolde", "Jules", "Keira", "Luca", "Mila", "Nadia",
            "Owen", "Priya", "Remy", "Selma", "Tomas", "Viola", "Anika", "Lucía",
        ),
        "last": (
            "Alderwick", "Brambleton", "Cedarvale", "Dunhaven", "Eversley", "Foxmere",
            "Glenford", "Hartwell", "Iverson", "Juniper", "Kestrel", "Larkspur",
            "Mossfield", "Northcott", "Oakridge", "Pinecroft", "Quillan", "Ravenwood",
            "Stoneleigh", "Thornby", "Underhill", "Valecrest", "Westbrook", "Yarrow",
            "Ashbourne", "Bracken", "Cloudwell", "Dalesworth", "Elmhurst", "Fernwick",
            "Goldleaf", "Hawthorne", "Ironwood", "Longmere", "Mapleby", "O'Fenwick",
            "Pritchford", "Riverdale", "Sableton", "Wintermere", "Bélanger", "Muñiz",
        ),
        "street": (
            "Alder", "Bracken", "Cedar", "Dove", "Elm", "Fern", "Grove", "Heather",
            "Ivy", "Juniper", "Kestrel", "Laurel", "Maple", "Nettle", "Orchard",
            "Pine", "Quartz", "Redwood", "Sparrow", "Thistle", "Willow", "Aspen",
        ),
        "city": (
            "Alderhaven", "Birchmere", "Cedarford", "Doveshire", "Elmbridge", "Fernvale",
            "Glenhaven", "Heatherby", "Ivybrook", "Juniper Bay", "Kestrel Falls",
            "Laurel Crossing", "Maple Hollow", "Northmere", "Oakford", "Pinehaven",
        ),
        "facility": (
            "Alderhaven Medical Center", "Birchmere General Hospital", "Cedarford Clinic",
            "Doveshire Rehabilitation", "Elmbridge Children's Hospital", "Fernvale Health",
            "Glenhaven Family Practice", "Heatherby Memorial Hospital", "Ivybrook Hospice",
            "Juniper Bay Urgent Care", "Kestrel Falls Imaging", "Laurel Crossing Dialysis",
            "Maple Hollow Pediatrics", "Northmere Community Hospital", "Oakford Surgery",
            "Pinehaven Women's Center",
        ),
        "state": ("CO", "UT", "ID", "OR", "WA", "AZ", "NM", "MT"),
    },
    "validation": {
        "first": (
            "Alina", "Benedict", "Cosima", "Desmond", "Esme", "Florian", "Gianna",
            "Hadrian", "Ines", "Jocelyn", "Kellan", "Livia", "Matteo", "Noemi",
            "Odette", "Petra", "Ronan", "Sabine", "Teagan", "Ulric", "Vivian",
            "Willa", "Yael", "Zara", "Arvid", "Beatrix", "Cassian", "Devika",
            "Estelle", "Felix", "Giselle", "Henrik", "Imogen", "Jonah", "Katya",
            "Leon", "Mireille", "Nico", "Opal", "Pavel", "Renata", "Sylvie",
            "Tobias", "Valeria", "Azélie", "Frédéric", "Chloë", "Maëlle",
        ),
        "last": (
            "Amberhurst", "Bluecombe", "Copperwell", "Driftwood", "Evergrove", "Faircliff",
            "Greenmantle", "Hillcombe", "Indigoford", "Kingsmere", "Lightfoot", "Marlowe",
            "Nightingale", "Oriel", "Pebbleworth", "Rainford", "Silverflint", "Turnstone",
            "Uptonvale", "Vesper", "Whitemere", "Yewstead", "Bellflower", "Brookstone",
            "Cloverby", "Duskhaven", "Fallowmere", "Greystone", "Hazelcroft", "Lindwood",
            "Meadowbridge", "O'Silver", "Rosefield", "Seabright", "Woldham", "Lavallée",
            "Delacôte", "Núñez", "Schönfeld", "Tremblay", "Usherton", "Volterra",
        ),
        "street": (
            "Amber", "Bluebell", "Copper", "Drift", "Evergreen", "Flax", "Goldenrod",
            "Hazel", "Indigo", "Linden", "Meadow", "Olive", "Pebble", "Rose", "Silver",
            "Turnstone", "Violet", "Wheat", "Yew", "Clover", "Sunflower", "Sycamore",
        ),
        "city": (
            "Amberwick", "Bluehaven", "Copperbridge", "Driftmere", "Everfield",
            "Fairwater", "Golden Hollow", "Hazel Bay", "Indigovale", "Linden Falls",
            "Meadowgate", "Oliveshire", "Pebbleford", "Rosehaven", "Silvergrove", "Yewmere",
        ),
        "facility": (
            "Amberwick Regional Hospital", "Bluehaven Medical Pavilion", "Copperbridge Infirmary",
            "Driftmere Wellness Clinic", "Everfield Cancer Institute", "Fairwater Day Surgery",
            "Golden Hollow Home Health", "Hazel Bay Medical Associates", "Indigovale Hospital",
            "Linden Falls Women's Clinic", "Meadowgate Family Health", "Oliveshire Pediatrics",
            "Pebbleford Rehabilitation Center", "Rosehaven Sleep Center", "Silvergrove Hospice",
            "Yewmere Community Clinic",
        ),
        "state": ("VA", "ME", "VT", "NH", "MA", "RI", "CT", "DE"),
    },
}

DIAGNOSES = (
    "type 2 diabetes mellitus", "community-acquired pneumonia", "chronic obstructive pulmonary disease",
    "acute kidney injury", "heart failure with preserved ejection fraction", "migraine without aura",
    "rheumatoid arthritis", "iron-deficiency anemia", "major depressive disorder",
    "osteoarthritis of the left knee", "hypothyroidism", "obstructive sleep apnea",
    "recurrent urinary tract infection", "atrial fibrillation", "hypertension",
    "lumbar radiculopathy", "viral gastroenteritis", "stage 3 chronic kidney disease",
)
MEDICATIONS = (
    "metformin 500 mg twice daily", "lisinopril 10 mg daily", "levothyroxine 75 mcg each morning",
    "albuterol 90 mcg two puffs every 4 hours as needed", "amoxicillin 875 mg twice daily",
    "apixaban 5 mg twice daily", "sertraline 50 mg daily", "ferrous sulfate 325 mg every other day",
    "furosemide 20 mg daily", "acetaminophen 650 mg every 6 hours as needed",
    "atorvastatin 40 mg nightly", "insulin glargine 18 units at bedtime",
    "prednisone 10 mg daily", "omeprazole 20 mg before breakfast", "gabapentin 300 mg nightly",
)
LABS = (
    "Na 138 mmol/L, K 4.1 mmol/L, creatinine 1.2 mg/dL, glucose 146 mg/dL",
    "WBC 7.4 ×10⁹/L, hemoglobin 12.1 g/dL, platelets 218 ×10⁹/L",
    "HbA1c 7.2%, LDL 102 mg/dL, HDL 48 mg/dL, triglycerides 166 mg/dL",
    "TSH 3.8 mIU/L, free T4 1.1 ng/dL; CRP 8.2 mg/L",
    "troponin <0.01 ng/mL, BNP 320 pg/mL, INR 1.1",
    "eGFR 54 mL/min/1.73 m²; urine albumin/creatinine ratio 42 mg/g",
    "sodium 140, potassium 3.9, chloride 103, bicarbonate 25; anion gap 12",
)
VITALS = (
    "BP 128/76 mmHg, pulse 72/min, RR 16/min, SpO2 98% on room air, temperature 36.8°C",
    "BP 146/82, HR 88, RR 18, oxygen saturation 96%, temperature 37.1 °C",
    "blood pressure 112/68 mmHg; pulse 64; weight 73.2 kg; BMI 24.6",
    "BP 132/80 mmHg, pulse 79, respirations 17, SpO₂ 97%, pain 3/10",
)
FINDINGS = (
    "Lungs are clear. Heart rhythm is regular. No edema. Strength is 5/5 in all extremities.",
    "Mild bibasilar crackles with trace ankle edema. Abdomen is soft and nontender.",
    "Left knee flexion is limited to 95 degrees; distal sensation and pulses are intact.",
    "Alert and oriented ×3. Cranial nerves II–XII are intact. There is no focal weakness.",
    "Chest radiograph shows no acute infiltrate. ECG has sinus rhythm without ischemic changes.",
    "Incision is clean, dry, and intact. There is no erythema, drainage, or fluctuance.",
)
PLANS = (
    "Continue the current regimen. Repeat metabolic panel in 2 weeks and follow up in 6 weeks.",
    "Encourage oral fluids and activity as tolerated. Return for fever, chest pain, or worsening dyspnea.",
    "Discussed diet, adherence, and home blood pressure monitoring. Recheck HbA1c in 3 months.",
    "Arrange physical therapy twice weekly for 4 weeks. Weight bearing as tolerated.",
    "Complete a 7-day antibiotic course. Review culture results and adjust therapy if indicated.",
    "Repeat ultrasound in 6 months. No urgent procedure is indicated at this time.",
)
NEGATIVES = (
    "The 47-year-old has type 2 diabetes. Glucose 182 mg/dL; HbA1c 8.1%. Increase metformin to 1000 mg twice daily.",
    "A 63 year old presents with chest pain for 2 days. Troponin <0.01, ECG unchanged, BP 130/80. Follow up in 1 week.",
    "Patient is 89 years old. EF 55–60%, creatinine 1.4 mg/dL, eGFR 48. Continue apixaban 2.5 mg twice daily.",
    "The lesion measures 1.2 × 0.8 × 0.6 cm. ICD-10 E11.9, CPT 99213. No new focal deficit.",
    "Urine culture: 10^5 CFU/mL. Take one tablet at 08:00 and one at 20:00 for 5 days; review in 48 hours.",
    "GCS 15/15. Muscle strength 4/5 on the left and 5/5 on the right. Sodium 136; potassium 4.4.",
    "A 7-year-old requires 7.5 mL of amoxicillin twice daily. Weight 24.7 kg; temperature 38.2°C.",
    "Pain 6/10 began after lifting. MRI: L4/L5 disc protrusion and L5/S1 foraminal narrowing. Rest for 2 days.",
    "A1c 6.8%; vitamin B12 320 pg/mL. Estimated 10-year ASCVD risk 12.4%. Discuss statin therapy.",
    "Oxygen at 2 L/min, FiO2 0.28, pH 7.39, PaCO2 41 mmHg. Wean to room air as tolerated.",
)


@dataclass(frozen=True)
class MarkedValue:
    label: str
    text: str

    def __post_init__(self) -> None:
        if self.label not in ENTITY_TYPES or not self.text:
            raise ValueError("Marked values require a nonempty supported PHI label")


@dataclass(frozen=True)
class Template:
    family: str
    text: str


# A family is an independently authored rendering, not a shuffled copy of a
# held-out note.  This separation makes template memorization visible at eval.
TRAIN_TEMPLATES = (
    Template("train_intake", "PATIENT REGISTRATION\nName: {name}\nDOB: {dob}\nMRN: {mrn}\nAddress: {address}\nHome telephone: {phone}\nEmail: {email}\nInsurance account: {account}\nChief complaint: {diagnosis}.\n{vitals}.\n{labs}.\n{plan}"),
    Template("train_progress", "{name} was seen on {date} at {facility} for {diagnosis}. The patient takes {medication}. {vitals}. {findings} {labs}. {plan} Dictated by {clinician}."),
    Template("train_hpi", "HPI: {name}, age {adult_age}, reports 3 days of increasing fatigue. Her home is in {location}. Symptoms started after a 2-hour walk. {diagnosis} is stable on {medication}. Prior evaluation was {past_date}; return is scheduled for {future_date}. Contact {phone} if symptoms worsen."),
    Template("train_discharge", "DISCHARGE SUMMARY — {name}\nMedical record #{mrn}; admission {past_date}, discharge {date}.\nHospital: {facility}. Principal diagnosis: {diagnosis}.\n{findings} {labs}. Discharge medication: {medication}.\nFollow-up with {clinician} on {future_date}. Send discharge instructions to {address}."),
    Template("train_referral", "Dear Dr. {clinician}:\nI am referring {name} (DOB {dob}) to {facility}. The patient has {diagnosis}. {findings} Current treatment is {medication}. Please fax questions to {fax} or email {email}. Sincerely, {sender}."),
    Template("train_telephone", "Telephone encounter {date}, 14:35. Caller {name} verified name and date of birth ({dob}). Return number {phone}. {diagnosis}; currently using {medication}. Reviewed warning signs. Emergency contact {contact}, {contact_phone}. {plan}"),
    Template("train_portal", "Portal message from {name} <{email}>\nSent: {date}\nSubject: glucose follow-up\nHome glucose has ranged from 120 to 180 mg/dL. Taking {medication}. Account {account}; record {mrn}. Reply securely via {url}. Client IP {ip}. {plan}"),
    Template("train_billing", "Encounter billing details\nPatient: {name}\nVisit date {date}; guarantor {contact}.\nAcct # {account}; member ID {identifier}; SSN {ssn}.\nBilling address {address}.\nDiagnosis {diagnosis}; procedure CPT 99214. Total charges $184.20. Clinical assessment: {findings}"),
    Template("train_geriatric", "{name} is a {old_age}-year-old seen at {facility}. DOB {dob}; MRN {mrn}. Lives with {contact} in {location}. Has {diagnosis}, treated with {medication}. {vitals}. {findings} {plan}"),
    Template("train_pediatrics", "Pediatric follow-up: {name}, aged {child_age} years, accompanied by parent {contact}. Born {child_dob}. Family address is {address}. Clinic {facility}. {vitals}. Receiving {medication}. {findings} Call parent at {contact_phone}."),
    Template("train_surgery", "OPERATIVE NOTE\n{facility}\nPatient {name} / MRN {mrn}\nProcedure date: {date}; surgeon {clinician}.\nPre-op diagnosis: {diagnosis}. Estimated blood loss 15 mL. No complications.\n{findings} Follow-up on {future_date}. Device serial {identifier}."),
    Template("train_lab_order", "Order received from {clinician}, {facility}, on {date}. Specimen collected for {name}, date of birth {dob}, medical record {mrn}. {labs}. Diagnosis: {diagnosis}. Results routed to {email}; phone {phone}."),
    Template("train_radiology", "RADIOLOGY: patient {name} ({mrn}); DOB {dob}. Study {date}. Referrer {clinician}. {findings} Comparison study {past_date}. Please send report to {facility}, {address}. {plan}"),
    Template("train_therapy", "Therapy visit on {date}: {name}. Treating therapist {clinician} at {facility}. Knee ROM 0–95 degrees, gait 25 meters with walker. {findings} {plan} Next appointment {future_date}. Home: {address}; telephone {phone}."),
    Template("train_homecare", "Home health assessment\nClient {name}; start of care {date}.\nResidence {address}; lives in {location}.\nCaregiver {contact} may be reached at {contact_phone}. {vitals}. {diagnosis}.\nMedication reconciliation: {medication}. Agency {facility}. {plan}"),
    Template("train_emergency", "ED {date}: {name}, {adult_age} y/o, arrived by private vehicle. MRN {mrn}; DOB {dob}. {vitals}. {findings} Impression: {diagnosis}. Discharged with {medication}. Follow-up {future_date} at {facility}."),
    Template("train_transport", "Transfer authorized {date}. Patient {name}, record {mrn}, transferred from {facility} to {other_facility}. Origin city {location}. Family notified: {contact} at {phone}. {diagnosis}. On room air; {vitals}. Ambulance authorization {identifier}."),
    Template("train_social_work", "Social work spoke with {name} on {date}. Household: {contact}; current residence {address}. The patient requests assistance traveling from {location} to {facility}. Contact phone {phone}, email {email}. {diagnosis}; {plan}"),
    Template("train_clinic_header", "{facility}\n{address}\nP: {phone}  F: {fax}\n{date}\nRE: {name}, DOB {dob}, MR #{mrn}\nAssessment: {diagnosis}. {labs}. Current medications include {medication}. Signed: {clinician}."),
    Template("train_csv", "patient_name,dob,mrn,appointment,provider,clinic,diagnosis\n\"{name}\",{dob},{mrn},{date},\"{clinician}\",\"{facility}\",\"{diagnosis}\"\nVitals: {vitals}. {plan}"),
    Template("train_jsonish", "Clinical export: {\"patient\":\"{name}\",\"birth_date\":\"{dob}\",\"record\":\"{mrn}\",\"email\":\"{email}\",\"portal\":\"{url}\",\"ip\":\"{ip}\"}. Assessment: {diagnosis}. {labs}. {plan}"),
    Template("train_ocr", "PATlENT: {name}\nD0B: {dob}   M.R.N.: {mrn}\nD A T E: {date}\nADD RESS: {address}\nTEL: {phone}\n{facility}\nAssessment—{diagnosis}. {findings} Meds: {medication}."),
    Template("train_unicode", "患者 / Patient: {name}\n生年月日 / DOB: {dob}\nRecord № {mrn} • seen {date} • clinician {clinician}\nResidence — {address}\n☎ {phone} · ✉ {email}\n{vitals}. Diagnosis: {diagnosis}. {labs}. {plan}"),
    Template("train_plain_demographics", "Demographics confirmed: {name}; born on {dob}; home {address}; phone {phone}; email {email}; SSN {ssn}; insurance policy {identifier}; billing account {account}. Care at {facility}. {diagnosis}. {medication}."),
    Template("train_security", "Security audit linked to clinical encounter: user {name}, record {mrn}, account {account}, session IP {ip}, patient portal URL {url}. Event occurred {date}. Authorized contact {email}. Clinical note: {diagnosis}; {labs}. {plan}"),
    Template("train_vaccine", "Immunization review for {name}, born {dob}. Seen by {clinician} at {facility} on {date}. Prior dose administered {past_date}. No fever; temperature 36.9°C. Lot TX-2147; CPT 90471. Patient phone {phone}. Next check {future_date}."),
    Template("train_ecg", "ECG acquired {date} at {facility}. {name}, DOB {dob}, MRN {mrn}. Ventricular rate 72, PR 160 ms, QRS 92 ms, QTc 428 ms. No acute ST changes. Interpreted by {clinician}. Diagnosis: {diagnosis}."),
    Template("train_pathology", "Pathology report {date}. Patient {name}; born {dob}; case identifier {identifier}; MRN {mrn}. Ordering doctor {clinician}, {facility}. Sample is 1.4 × 0.8 × 0.5 cm. Benign fibrous tissue, no malignancy. {plan}"),
    Template("train_letter", "To {name}\n{address}\n{date}\nYour test results show: {labs}. Continue {medication}. Please attend {facility} on {future_date}. Call {phone} or visit {url} to reschedule. Yours, {clinician}."),
    Template("train_longitudinal", "{name} developed {diagnosis} after the visit on {past_date}. The appointment today, {date}, is at {facility}; repeat review on {future_date}. {medication}. {labs}. {findings} Address remains {address}, telephone {phone}, record {mrn}."),
    Template("train_consent", "Consent signed by {name} on {date}; witnessed by {clinician}. Guardian {contact}. DOB {dob}, record {mrn}. Patient identity document {identifier}; last reported SSN {ssn}. Procedure explained at {facility}. {diagnosis}. {findings}"),
    Template("train_badges", "Prescriber: {clinician}\nProfessional license: {identifier}\nPatient: {name}, birth date {dob}\nFacility: {facility}; appointment: {date}\n{medication}; indication {diagnosis}. Questions to {email} or {phone}."),
    Template("train_fax_cover", "Fax from {facility} to {other_facility}\nTo {clinician}; from {sender}\nFax number {fax}, callback {phone}\nDate {date}; regarding {name}, DOB {dob}, MRN {mrn}\nPages 3. Clinical details: {diagnosis}. {vitals}. {labs}."),
    Template("train_insurance", "Insurance verification for {name}. Policy no. {identifier}; member account {account}; SSN {ssn}. DOB {dob}. Subscriber {contact}. Mailing address {address}. Appointment with {clinician} {date} for {diagnosis}. {plan}"),
    Template("train_short_phone", "Please reach {name} on {phone} regarding {diagnosis}. {medication}."),
    Template("train_short_email", "Send the {diagnosis} education sheet to {email} for {name}. {plan}"),
    Template("train_short_date", "Reviewed {name}'s {labs} from {past_date}. Repeat testing booked for {future_date}. {plan}"),
    Template("train_short_ids", "Patient {name}: MRN {mrn}, encounter account {account}, member ID {identifier}. {diagnosis}. {vitals}."),
    Template("train_short_location", "{name} lives in {location} and receives care at {facility}. {diagnosis}. {medication}."),
    Template("train_short_age", "Seen {date}: {name} is {old_age} years old. {diagnosis}. {vitals}. {findings}"),
)

VALIDATION_TEMPLATES = (
    Template("validation_medication_reconciliation", "Medication review • {date}\n{clinician} reconciled treatment for {name} at {facility}. Verified birth date {dob} and record number {mrn}. Taking {medication} for {diagnosis}. {labs}. {findings} Home contact: {phone}; residence: {address}."),
    Template("validation_observation", "OBSERVATION COURSE\nFrom {past_date} through {date}, {name} was monitored for {diagnosis}. {vitals}. {findings} Next evaluation is {future_date} with {clinician} at {facility}. Family member {contact} was informed by phone {contact_phone}."),
    Template("validation_after_visit", "After-visit instructions for {name}\nWe met {date} at {facility}. Your tests: {labs}. Use {medication}. {plan}\nThe next visit is {future_date}. Questions: telephone {phone}, electronic mail {email}, or patient portal {url}. Chart key {mrn}."),
    Template("validation_handoff", "Handoff to {clinician}: {name}, born {dob}, chart {mrn}, is being sent from {facility} to {other_facility}. Departure {date}. {diagnosis}. {vitals}. {findings} Destination city {location}. Reach relative {contact} using {contact_phone}."),
    Template("validation_financial_review", "Benefit inquiry linked to {name}, DOB {dob}. Subscriber {contact}; social security number {ssn}; billing reference {account}; plan identifier {identifier}. Address of record: {address}. Service {date} at {facility} concerned {diagnosis}. CPT 99213, amount $92.40."),
    Template("validation_telemedicine", "Video visit completed on {date} for {name}. Identity verified by DOB {dob}, medical record {mrn}. Connected from {location} via {url}; network address {ip}. Email for documents {email}. {vitals}. Assessment {diagnosis}; treatment {medication}."),
    Template("validation_eldercare", "Residential review: {name}, {old_age} years of age. Care directed by {clinician}; referral origin {facility}. The resident's home is {address}. Birth date {dob}. {diagnosis}. {labs}. {findings} Next check {future_date}; caregiver {contact}, {contact_phone}."),
    Template("validation_urgent_message", "Urgent callback request — {name} [{mrn}]\nOriginally evaluated {past_date}; worsening today {date}. {diagnosis}. {vitals}. Please call {phone}. Primary practice {facility}. Backup recipient {contact}, address {email}. {plan}"),
    Template("validation_structured_extract", "NAME|{name}\nBIRTH|{dob}\nCHART|{mrn}\nVISIT|{date}\nCONTACT|{phone}\nMAIL|{email}\nHOME|{address}\nCARE SITE|{facility}\nDX|{diagnosis}\nRX|{medication}\nRESULTS|{labs}"),
    Template("validation_scanned_worksheet", "Scanned worksheet\nPATIENT . . {name}\nBIRTHDATE . . {dob}\nREC0RD . . {mrn}\nSEEN . . {date}\nD0CTOR . . {clinician}\nH0ME . . {address}\nCALL . . {phone}\n{vitals}. Impression: {diagnosis}. {findings}"),
    Template("validation_device_registration", "Implant follow-up: {name}; record {mrn}; serial number {identifier}. Device interrogation {date} at {facility} by {clinician}. Battery voltage 2.91 V, atrial pacing 18%, ventricular pacing 2%. {diagnosis}. {medication}."),
    Template("validation_access_log", "Chart access summary for {name}: session IP address {ip}; portal location {url}; account reference {account}; clinical record {mrn}; notification email {email}. Session {date}. Summary: {diagnosis}. {labs}. {plan}"),
    Template("validation_school_health", "School health letter: {name}, aged {child_age}, DOB {child_dob}, was examined {date}. Guardian is {contact}, reachable at {phone}. Send the letter to {address}. {findings} Treating professional {clinician} from {facility}."),
    Template("validation_combined_review", "Record review {date}: {name} (birth {dob}); medical chart {mrn}; patient number {identifier}; account {account}; SSN {ssn}. From {location}, mailing address {address}, telephone {phone}, email {email}. Seen at {facility} by {clinician}. Portal {url}; IP {ip}. {diagnosis}. {labs}."),
    Template("validation_unicode_summary", "Résumé clinique / clinical summary\n{date} — {name} — âge {old_age}\nDate de naissance: {dob} • dossier: {mrn}\nLieu de soins: {facility}\nAdresse: {address}; téléphone: {phone}; courriel: {email}\n{diagnosis}. {vitals}. {findings} {plan}"),
    Template("validation_abbreviated", "Pt {name}, DOB {dob}, rec {mrn}; eval {date}. Dx {diagnosis}. Rx {medication}. {labs}. F/u {future_date} @ {facility} w/ {clinician}. Tel {phone}."),
)

FIELD_PATTERN = re.compile(r"\{([a-z_]+)\}")


def render_marked(template: str, values: dict[str, str | MarkedValue]) -> tuple[str, list[dict]]:
    """Assemble original text and exact annotations, including repeated fields."""
    pieces: list[str] = []
    spans: list[dict] = []
    cursor = 0
    offset = 0
    for match in FIELD_PATTERN.finditer(template):
        prefix = template[cursor:match.start()]
        pieces.append(prefix)
        offset += len(prefix)
        key = match.group(1)
        if key not in values:
            raise ValueError(f"Unknown template field: {key}")
        value = values[key]
        if isinstance(value, MarkedValue):
            pieces.append(value.text)
            spans.append({"start": offset, "end": offset + len(value.text), "label": value.label})
            offset += len(value.text)
        else:
            pieces.append(value)
            offset += len(value)
        cursor = match.end()
    pieces.append(template[cursor:])
    return "".join(pieces), spans


def _date_value(rng: random.Random, value: date) -> str:
    formats = (
        "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d.%m.%Y",
        "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%Y/%m/%d",
        "%m/%d/%y", "%d-%b-%Y",
    )
    result = value.strftime(rng.choice(formats))
    # Both zero-padded and naturally written dates occur in clinical records.
    if rng.random() < 0.25:
        result = re.sub(r"(?<!\d)0([1-9])(?!\d)", r"\1", result)
    return result


def _name(rng: random.Random, vocab: dict[str, tuple[str, ...]]) -> str:
    first, last = rng.choice(vocab["first"]), rng.choice(vocab["last"])
    style = rng.randrange(12)
    if style == 0:
        return f"{last}, {first}"
    if style == 1:
        return f"{first} {rng.choice('ABCDEFGHJKLMNPRSTVW')}. {last}"
    if style == 2:
        return f"{first[0]}. {last}"
    if style == 3:
        return f"{first} {last}-{rng.choice(vocab['last'])}"
    if style == 4:
        return f"{first} {last}".upper()
    return f"{first} {last}"


def _phone(rng: random.Random, split: Split) -> str:
    # 555-0100..0199 is the reserved fictional North American subscriber range.
    area = rng.choice((202, 303, 406, 503, 720) if split == "train" else (207, 302, 401, 617, 802))
    subscriber = rng.randint(100, 199)
    fmt = rng.randrange(8)
    if fmt == 0:
        return f"({area}) 555-{subscriber:04d}"
    if fmt == 1:
        return f"{area}-555-{subscriber:04d}"
    if fmt == 2:
        return f"{area}.555.{subscriber:04d}"
    if fmt == 3:
        return f"+1 {area} 555 {subscriber:04d}"
    if fmt == 4:
        return f"1-{area}-555-{subscriber:04d}"
    if fmt == 5:
        return f"{area}555{subscriber:04d}"
    if fmt == 6:
        return f"({area})555-{subscriber:04d} ext. {rng.randint(10, 99)}"
    return f"+1 ({area}) 555-{subscriber:04d}"


def _context(rng: random.Random, split: Split, index: int, template: str = "") -> dict[str, str | MarkedValue]:
    vocab = VOCABULARY[split]
    m = MarkedValue
    name, clinician, contact, sender = (_name(rng, vocab) for _ in range(4))
    city = rng.choice(vocab["city"])
    state = rng.choice(vocab["state"])
    street = rng.choice(vocab["street"])
    suffix = rng.choice(("Street", "St", "Avenue", "Ave", "Road", "Rd", "Lane", "Ln", "Drive", "Dr", "Court", "Way"))
    unit = rng.choice(("", "", "", f", Apt {rng.randint(1, 99)}", f", Suite {rng.randint(100, 900)}", f" #{rng.randint(1, 99)}"))
    postcode = rng.randint(80001, 84999) if split == "train" else rng.randint(10001, 14999)
    address = f"{rng.randint(10, 9999)} {street} {suffix}{unit}, {city}, {state} {postcode}"
    if rng.random() < 0.08:
        address = f"PO Box {rng.randint(1, 999)}, {city}, {state} {postcode}"
    visit = date(2008 if split == "train" else 2021, 1, 1) + timedelta(days=rng.randrange(365 * 7))
    # Explicit dates preserve a coherent sequence within every note.
    past = visit - timedelta(days=rng.randint(3, 90))
    future = visit + timedelta(days=rng.choice((7, 14, 28, 42, 90, 180)))
    old_age = rng.randint(90, 109)
    adult_age = rng.randint(18, 89)
    child_age = rng.randint(3, 17)
    # Preserve stated ages as well as event order. Leap days are avoided here
    # because these notes do not need a special birthday convention.
    reported_age = old_age if "{old_age}" in template else adult_age
    dob = date(visit.year - reported_age, visit.month, min(visit.day, 28))
    child_dob = date(visit.year - child_age, visit.month, min(visit.day, 28))
    number = rng.randint(1_000_000, 9_999_999)
    base_mrn = f"{number}"
    mrn_style = rng.randrange(8)
    if mrn_style == 0:
        mrn = f"MR{number}"
    elif mrn_style == 1:
        mrn = f"{number:08d}"
    elif mrn_style == 2:
        mrn = f"{number // 10000:03d}-{number % 10000:04d}"
    elif mrn_style == 3:
        mrn = f"{rng.choice(('A', 'B', 'C') if split == 'train' else ('D', 'E', 'F'))}-{number}"
    else:
        mrn = base_mrn
    account = rng.choice((f"AC-{number}-{index % 97:02d}", f"{number}{index % 1000:03d}", f"BILL{number}", f"{number // 10000:03d}/{number % 10000:04d}"))
    identifier = rng.choice((f"LIC-{rng.choice('QRT' if split == 'train' else 'VWX')}{number}", f"DEV{number:08d}", f"{rng.choice('ABCE' if split == 'train' else 'KLMN')}{number}", f"POL-{number}-{rng.randint(10, 99)}"))
    ssn_prefix = rng.randint(900, 944) if split == "train" else rng.randint(955, 999)
    ssn = f"{ssn_prefix}-{rng.randint(1, 99):02d}-{rng.randint(1, 9999):04d}"
    if rng.random() < 0.12:
        ssn = ssn.replace("-", "")
    email_user = re.sub(r"[^a-z]", "", name.lower())[:28] or "patient"
    domain = rng.choice(("care.example.com", "patient.example.org", "clinic.example.net"))
    email = f"{email_user}{rng.randint(10, 999)}@{domain}"
    portal_host = rng.choice(("care.example.com", "portal.example.org", "records.example.net"))
    url = f"https://{portal_host}/patient/{email_user}/{number}"
    if rng.random() < 0.15:
        url += f"?account={number}"
    ip = f"192.0.2.{rng.randint(1, 126)}" if split == "train" else f"198.51.100.{rng.randint(128, 254)}"
    values: dict[str, str | MarkedValue] = {
        "name": m("NAME", name), "clinician": m("NAME", clinician),
        "contact": m("NAME", contact), "sender": m("NAME", sender),
        "date": m("DATE", _date_value(rng, visit)),
        "past_date": m("DATE", _date_value(rng, past)),
        "future_date": m("DATE", _date_value(rng, future)),
        "dob": m("DATE", _date_value(rng, dob)),
        "child_dob": m("DATE", _date_value(rng, child_dob)),
        "address": m("ADDRESS", address), "location": m("LOCATION", f"{city}, {state}"),
        "facility": m("FACILITY", rng.choice(vocab["facility"])),
        "other_facility": m("FACILITY", rng.choice(vocab["facility"])),
        "phone": m("PHONE", _phone(rng, split)), "fax": m("PHONE", _phone(rng, split)),
        "contact_phone": m("PHONE", _phone(rng, split)), "email": m("EMAIL", email),
        "mrn": m("MRN", mrn), "account": m("ACCOUNT", account),
        "identifier": m("ID", identifier), "ssn": m("SSN", ssn),
        "url": m("URL", url), "ip": m("IP", ip), "old_age": m("AGE", str(old_age)),
        "adult_age": str(adult_age), "child_age": str(child_age),
        "diagnosis": rng.choice(DIAGNOSES), "medication": rng.choice(MEDICATIONS),
        "labs": rng.choice(LABS), "vitals": rng.choice(VITALS),
        "findings": rng.choice(FINDINGS), "plan": rng.choice(PLANS),
    }
    return values


def validate_row(row: dict) -> None:
    """Reject corrupt, overlapping, empty, or unsupported gold annotations."""
    if row.get("split") not in ("train", "validation") or not isinstance(row.get("text"), str):
        raise ValueError("Invalid synthetic row schema")
    previous_end = 0
    for span in row["spans"]:
        start, end, label = span["start"], span["end"], span["label"]
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("Offsets must be Unicode character integers")
        if not (previous_end <= start < end <= len(row["text"])) or label not in ENTITY_TYPES:
            raise ValueError(f"Invalid span in {row['id']}: {span}")
        if not row["text"][start:end].strip():
            raise ValueError("A PHI span cannot contain only whitespace")
        previous_end = end


def generate_rows(count: int, split: Split = "train", seed: int = 1729) -> Iterable[dict]:
    """Yield reproducible self-authored notes with family-disjoint held-out notes.

    Approximately 8% of examples contain no PHI.  Every template is selected in a
    seeded shuffled cycle, so smaller smoke-test corpora have broad coverage.
    The split is mixed into the PRNG seed so changing one split's size cannot
    silently modify the other split.
    """
    if count < 0:
        raise ValueError("count cannot be negative")
    if split not in VOCABULARY:
        raise ValueError("split must be 'train' or 'validation'")
    seed_bytes = hashlib.sha256(f"clinical-deid:{seed}:{split}".encode()).digest()
    rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    templates = list(TRAIN_TEMPLATES if split == "train" else VALIDATION_TEMPLATES)
    cycle = list(templates)
    rng.shuffle(cycle)
    template_index = 0
    for index in range(count):
        if index % 13 == 12:
            negative = rng.choice(NEGATIVES)
            # Disjoint negative families have different narrative wrappers.
            text = f"Clinical assessment: {negative}" if split == "train" else f"Unidentified case summary\n{negative}\nPlan reviewed with teach-back."
            row = {"id": f"{split}-{index:07d}", "text": text, "spans": [], "family": f"{split}_clinical_negative", "split": split}
        else:
            if template_index and template_index % len(cycle) == 0:
                rng.shuffle(cycle)
            template = cycle[template_index % len(cycle)]
            template_index += 1
            text, spans = render_marked(template.text, _context(rng, split, index, template.text))
            # Preserve gold offsets under varying line endings / indentation by
            # rendering those separators before values are inserted.
            if rng.random() < 0.08 and "\n" in template.text:
                text, spans = render_marked(template.text.replace("\n", "\r\n"), _context(rng, split, index, template.text))
            row = {"id": f"{split}-{index:07d}", "text": text, "spans": spans, "family": template.family, "split": split}
        validate_row(row)
        yield row


def _vocabulary_audit() -> dict:
    intersections = {}
    for key in VOCABULARY["train"]:
        common = sorted(set(VOCABULARY["train"][key]) & set(VOCABULARY["validation"][key]))
        if common:
            intersections[key] = common
    train_families = {t.family for t in TRAIN_TEMPLATES}
    val_families = {t.family for t in VALIDATION_TEMPLATES}
    if intersections or train_families & val_families:
        raise ValueError("Training/validation identity vocabulary or family leakage")
    return {
        "identity_vocabulary_intersections": intersections,
        "template_family_intersection": [],
        "clinical_vocabulary_shared": True,
        "numeric_values_may_overlap": True,
        "note": "Identity vocabularies and template families are disjoint; generic medical words, date numerals, age numerals, and reserved internet domains may recur.",
    }


def write_dataset(output: str | Path, train_count: int = 12000, val_count: int = 1000, seed: int = 1729) -> dict:
    """Write annotated JSONL splits and a inspectable provenance manifest."""
    audit = _vocabulary_audit()
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for split, count in (("train", train_count), ("validation", val_count)):
        path = directory / f"{split}.jsonl"
        label_counts: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        total_chars = 0
        clean = 0
        digest = hashlib.sha256()
        with path.open("w", encoding="utf-8", newline="") as handle:
            for row in generate_rows(count, split=split, seed=seed):
                serialized = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                handle.write(serialized)
                digest.update(serialized.encode("utf-8"))
                label_counts.update(span["label"] for span in row["spans"])
                family_counts.update([row["family"]])
                total_chars += len(row["text"])
                clean += int(not row["spans"])
        datasets[split] = {
            "file": path.name, "count": count, "sha256": digest.hexdigest(),
            "entity_counts": dict(sorted(label_counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
            "negative_notes": clean, "total_characters": total_chars,
        }
    manifest = {
        "schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION,
        "seed": seed, "entity_types": list(ENTITY_TYPES),
        "offset_unit": "unicode_code_points", "license": "CC0-1.0",
        "generator_code_license": "Apache-2.0",
        "provenance": {
            "source": "Entirely generated from project-authored clinical templates and identity vocabularies.",
            "external_patient_records": False, "external_datasets": False,
            "commercial_dataset_licenses": False,
            "internet_values": "Reserved example.com/example.org/example.net domains and documentation IP ranges.",
            "ssn_values": "Synthetic invalid SSNs with area numbers 900 through 999.",
        },
        "split_audit": audit, "datasets": datasets,
        "evaluation_scope": "A held-out synthetic validation score measures this generator's examples; it is not clinical deployment qualification or a HIPAA certification.",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--train-count", type=int, default=12000)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.train_count < 0 or args.val_count < 0:
        parser.error("Example counts must be nonnegative")
    manifest = write_dataset(args.output, args.train_count, args.val_count, args.seed)
    print(json.dumps({"output": str(args.output.resolve()), "datasets": manifest["datasets"]}, indent=2))


if __name__ == "__main__":
    main()
