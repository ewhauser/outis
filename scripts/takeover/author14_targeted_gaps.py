#!/usr/bin/env python3
"""Deterministic generator for the tk14 targeted-gaps training bank.

Brief: docs/targeted-gaps-bank-brief.md. Policy: docs/annotation-policy-v2.md and
its places addendum. Four families:

* history_years   patient-specific years in history and screening lists (DATE)
* pharmacy_zip    pharmacy name (FACILITY) followed by a ZIP code (LOCATION)
* phone_extension a phone number with a written extension (one PHONE span)
* funeral_homes   a named funeral home handling the arrangements (FACILITY)

Every positive has matched negatives in the same layouts. Every name, facility,
town, ZIP, phone (fictional 555-0100 to 555-0199 lines), domain and identifier is
invented. Rows are built from parts, so offsets are exact by construction, and
the case transform (normal / all-caps / lowercase) is applied to whole rows with
gold following the authored role.

Usage:
  PYTHONPATH=src python scripts/takeover/author14_targeted_gaps.py \
      [--out data/tk14_train_targeted_gaps.jsonl] \
      [--report-dir reports/takeover/author14_targeted_gaps]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from clinical_deid.data.annotation_policy import (  # noqa: E402
    POLICY_VERSION,
    annotation_errors,
    canonical_identity_group,
    identity_split,
)
from clinical_deid.data.generators.places import (  # noqa: E402
    excluded_sequences,
    excluded_value,
)
from clinical_deid.types import ENTITY_TYPES  # noqa: E402

SEED = 20261014
SUITE = "tk14_targeted_gaps"
SEED2 = 20261015  # tk14-v2
SUITE2 = "tk14_targeted_gaps_v2"
SEED3 = 20261016  # tk14-v3
SUITE3 = "tk14_targeted_gaps_v3"
YEAR_CAP3 = 85  # v3 history-year cap (under 2.8% of roughly 3,100 spans)
EXCLUSIONS_PATH = ROOT / "data" / "places_exclusions.sha256.json"
YEAR_CAP2 = 68  # v2 history-year cap (under 2.7% of roughly 2,500 spans)
FUTURE_CAP = 40
CAP = (
    9  # at most 9 rows per pharmacy, funeral home, ZIP, town or facility value (0.45%)
)
YEAR_CAP = 60  # per-year cap on history-year spans (about 2.5% of roughly 2,400)

FAMILY_ROWS = {
    "history_years": 900,
    "pharmacy_zip": 450,
    "phone_extension": 450,
    "funeral_homes": 200,
}
LAYOUTS = {
    "history_years": [
        "awv_note",
        "soap_note",
        "screening_schedule",
        "surgical_history_list",
        "problem_list",
        "immunization_record",
        "telegraphic_oneliner",
        "nursing_intake_history",
        "portal_message_history",
        "referral_letter",
    ],
    "pharmacy_zip": [
        "intake_form",
        "registration_form",
        "refill_request",
        "portal_message_pharmacy",
        "discharge_med_list",
        "pharmacy_oneliner",
    ],
    "phone_extension": [
        "contact_block",
        "fax_cover",
        "callback_note",
        "shift_handoff",
        "nursing_note_family",
        "followup_instructions",
    ],
    "funeral_homes": [
        "death_note",
        "hospice_discharge",
        "social_work_bereavement",
        "nursing_expiration",
    ],
}
CASE_COUNTS = (("normal", 0.6), ("upper", 0.2), ("lower", 0.2))

# v2 settled cases of 2026-10-05: (positive rows, negative rows, family, layouts)
HIST_NEW_LAYOUTS = [
    "awv_note",
    "soap_note",
    "telegraphic_oneliner",
    "nursing_intake_history",
    "portal_message_history",
    "referral_letter",
    "problem_list",
]
PHARM_NEW_LAYOUTS = list(LAYOUTS["pharmacy_zip"])
PHONE_NEW_LAYOUTS = list(LAYOUTS["phone_extension"])
NEW_CASES = {
    "onset_year": (30, 15, "history_years", HIST_NEW_LAYOUTS),
    "year_range": (37, 18, "history_years", HIST_NEW_LAYOUTS),
    "future_year": (37, 18, "history_years", HIST_NEW_LAYOUTS),
    "zip_plus_four": (33, 17, "pharmacy_zip", PHARM_NEW_LAYOUTS),
    "rx_number": (40, 20, "pharmacy_zip", PHARM_NEW_LAYOUTS),
    "pager": (30, 15, "phone_extension", PHONE_NEW_LAYOUTS),
    "bare_extension": (0, 40, "phone_extension", PHONE_NEW_LAYOUTS),
}
HIST_NEW = ("onset_year", "year_range", "future_year")

# v3 additions (2026-10-06): (family, layouts with row counts, case plan)
V3_HIST_LAYOUTS_UPPER = [("awv_note", 80), ("soap_note", 80)]  # all-caps
V3_HIST_LAYOUTS_OTHER = [
    ("screening_schedule", 18),
    ("surgical_history_list", 18),
    ("immunization_record", 18),
    ("problem_list", 18),
    ("nursing_intake_history", 17),
    ("telegraphic_oneliner", 17),
    ("portal_message_history", 17),
    ("referral_letter", 17),
]
V3_SIG_LAYOUTS = [
    "sig_progress_note",
    "sig_attestation",
    "sig_telephone_encounter",
    "sig_discharge_summary",
    "sig_nursing_note",
    "sig_addendum",
]
V3_RARE_LAYOUTS = [
    "device_record_note",
    "telehealth_visit_note",
    "call_center_log",
    "portal_activation_message",
    "device_registration_form",
    "after_visit_summary",
]
V3_CASES = {
    "hist_v3": "history_years",
    "sig_weekday": "signature_dates",
    "rare_id": "rare_identifiers",
}
RARE_TAGS = ("serial", "session", "call", "activation")

# ---------------------------------------------------------------------------
# Value pools (all invented)
# ---------------------------------------------------------------------------
FIRSTS = """Aaliyah Abel Adaeze Adela Alasdair Amara Anneliese Anselm Beatriz Benedikt Bronwen Calla Caspian
Cecilia Chidi Cormac Dalia Dariusz Delphine Dmitri Edda Eilidh Elio Emeka Esperanza Fenella Florian Gideon
Greta Gwendolyn Hamza Henrietta Ignatius Ilse Ines Isidore Jasper Jovana Kasimir Katarina Kofi Lakshmi
Leocadia Lorcan Lucinda Magnus Marisol Mateus Mirela Nadezhda Nils Noor Odalys Oluwaseun Orla Osvaldo
Pasquale Perpetua Priyanka Quentin Rasmus Renata Rosalind Sabela Saoirse Selma Siobhan Tamsin Tavita
Thaddeus Ulrike Valentina Vikram Wilhelmina Xiomara Yusuf Zbigniew Zora Linda Robert Maria James Patricia
Ahmed Wei Mei Hiroshi Fatima Carlos Denise Harold Doris Walter Gladys Lloyd Bernice Clarence Mavis Rosa
Luis Ngozi Yolanda Josefina Renee Soren Zoe Anika Bao Ravi Mikhail Lucille Otis Hattie Everett Winifred
Cornelius Marcella Bashir Takumi Ingrid Leopold Corinne Desmond Palmira Eugenie Raymundo Tomasz Birgit""".split()
LASTS = """Abernathy Adeyemi Alvarado Bellweather Brandvold Castellanos Chaudhry Czerwinski Delacroix Dunmore
Eastergard Fairbairn Fontenot Garrido Gudmundsson Haverlock Hollingsworth Ibarra Jankowski Kowalczyk
Lindqvist Machado Nakamura Okafor Pemberton Quimby Rasmussen Sandoval Tavares Underhill Valdivia
Whitcombe Yamamoto Zielinski Ainsworth Balogun Cavanaugh Dimitrov Ellsworth Farouk Galloway Hartigan
Iyer Jablonski Kaplan Lombardi Mbeki Novak Oyelaran Pettibone Quintero Radcliffe Szymanski Thibodeaux
Ulloa Vasquez Wojcik Xavier Yilmaz Zamora Aguilar Bergstrom Coughlin Draper Espinoza Fitzgerald Grimaldi
Hendricks Ivanov Jorgensen Keller Lacroix Mendoza Nwosu Olmstead Prochaska Raghavan Sorensen Tillman
Uchenna Vandenberg Whitlock Yoder Zeller Armitage Boateng Carrasco Dlamini Edmondson Figueroa Gallardo
Holloway Ishikawa Jameson Kuznetsov Lundgren Moreau Nikolaidis Ostrowski Prescott Reyes Stanczyk Teague
Valenzuela Wasserman Amoah Brzezinski Calloway Dufresne Engstrom Ferreira Goldfarb Hutchins Kowalski
O'Halloran D'Amato Nguyen-Reyes Smythe-Okoye MacAllister""".split()
LASTS += ["van der Linden", "de la Fuente", "del Rio", "St. Clair"]
FIRSTS_V2 = """Abelard Adelmar Aelfric Agrippina Alaric Albertina Aldwin Alphonsine Amadeus Ambrosine Anatole
Angharad Anscom Apolline Archibald Ariadne Arnaud Arvid Athalia Augustin Aurelian Avelina Balthazar Barnaby
Beatrijs Benedetta Berengar Bertille Blandina Bram Brannoch Brigida Bryony Cadmus Caelan Caldwyn Camilo
Casimir Cassiopeia Cenwulf Cesare Clarimond Clothilde Corbin Cosima Crispin Cuthbert Cyprian Damaris Dagny
Delphina Demetrio Desideria Dietrich Dorothea Drusilla Dunstan Eadric Ebenezer Edmonda Eirian Elbert Eleazar
Elfreda Elodie Emerentia Engelbert Ephraim Erasmus Esmond Eudora Eulalia Evander Evangeline Fabrizio
Faustina Felicitas Ferdinand Fidelma Fiorella Florentin Fridolin Gaspard Gawain Genevra Gervase Gisela
Godfrey Griselda Gudrun Gustavo Hadrian Halvard Hermione Hilarion Honoria Hortense Hubertus Humbert Ianthe
Idony Ignazio Ingeborg Isaura Isolde Jedidiah Jocasta Jorund Juliane Justinian Kaspar Ladislas Laurentia
Leandro Leofric Lettice Linnea Lisandro Ludovic Lysander Magdalena Marcellin Mathilde Maximilian Melisande
Merewyn Mirabel Modesta Montague Nerissa Nicodemus Norbert Octavia Odilia Olwen Oriel Osmund Ottoline
Pascale Pelagia Percival Philippa Placida Polycarp Quincy Radegund Raimund Reinhold Rosamund Rufus Rupert
Sabina Salome Sebastiane Seraphina Sigrid Silvanus Sixtus Tancred Thaddea Theodora Thessaly Tiberius Ulysses
Ursula Valdemar Vespasian Wolfram Xanthe Zenobia""".split()
LASTS_V2 = """Abbotsford Aldersey Ashcombe Attwater Barraclough Bellingham Birtwistle Bonnycastle Brackenridge
Brimblecombe Broadhurst Burrowes Cadwallader Calthorpe Carruthers Cheesman Chillingworth Clutterbuck
Coppinger Cravenhurst Crouchback Dalrymple Delahunty Dinsdale Dowdeswell Dunstable Eddowes Fairweather
Farquharson Featherstone Fenwicke Fetherston Fothergill Gallowglass Garnsey Gawthrop Gilfillan Goodenough
Gorringe Grisewood Haddington Halliwell Hargreave Hazeldine Heatherington Hepplewhite Hollinshead Hornblower
Ilchester Inglethorpe Jessop Kilbride Lanchester Larkworthy Lavington Leadbetter Lightfoot Littlejohn
Lockyer Longbottom Lumsden Mainwaring Maudsley Merriweather Micklethwaite Mortlock Mountjoy Netherwood
Nuttall Oddie Orrell Overbury Pargeter Pennyfeather Pettigrew Plumtree Prendergast Quarrell Quenby Radclyffe
Ravenscroft Rowntree Rudkin Scrimshaw Selwyn Shackleford Sidebottom Skelton Slingsby Spofforth Stancliffe
Stourton Swinburne Tattersall Thistlewood Throckmorton Tolhurst Trelawney Tunstall Twigg Umfreville Upjohn
Vavasour Wakelin Wedgwood Wigglesworth Winterbottom Woolgar Wyndham Yarborough Yelverton Zouche Ardagh
Bannerman Cheverell Dymoke Eversleigh Fanshawe Greenhalgh Hapgood Ingoldsby Jenkinson-Pryce Kenrick
Lisle-Tarrant Mellersh Northcott Okell Pinsent Quennell Rotherham Shuttleworth Tetlow Uttley Vesey Wardlaw""".split()
UNUSUAL_FIRSTS = [
    "He",
    "Care",
    "Hope",
    "Will",
    "Faith",
    "Joy",
    "Mark",
    "Bill",
    "Rich",
    "May",
    "June",
    "Sunday",
]

STEMS = """Alvern Bexmoor Calmere Dornick Eskwith Fallowmere Garrowick Hestavale Ivermoor Jessamoor Kirkwen
Lindenmoor Marrenby Nethermoor Orrinvale Pendrake Quenmoor Rosswick Sandrik Tolliver Ulmenby Vessmere
Wexcombe Yarnwick Zelmoor Ambrewick Brennoch Caldrith Dunvale Eldermoor Farnwick Glenmarrow Harrowmere
Ilkesby Jarvenmill Kelborne Lunnock Merrowby Norrisvale Oldencroft Pyrmoor Quillhaven Rimmerby Saltwhistle
Thornvale Underby Verdemoor Wenlow Yewdale Zinnock Aldrecott Breckenmoor Cobbleby Dellwick Embermoor
Foxcombe Galbraith-Vale Hobsmere Inglemoor Jadwick Knollby Larchmoor Mistlewick Needham-Vale Ormerod
Pippinby Quarlby Ravensmoor Sorrelby Tansmoor Umberley Vinhaven Whinmoor Yarrowby Zephmoor Ashlowe Burnmere
Corvale Dinmoor Elswick Fernlow Gilmoor Heathwick Irnby Junmoor Kestrelby Lowmoor Maplecroft""".replace(
    "-Vale", "vale"
).split()
STREET_STEMS = """Wren Alder Fennel Harrow Juniper Kestrel Linden Marten Nettle Orchard Plover Quarry Rowan Sedge
Tamarack Umber Vesper Willow Yarrow Hollis Bramble Cinder Dapple Elder Finch Gable Heron Ivy""".split()
STREET_TYPES = [
    "Lane",
    "Court",
    "Street",
    "Avenue",
    "Road",
    "Way",
    "Drive",
    "Place",
    "Terrace",
]
STATES = "UT OR NM MN WA OH CO AZ ID VT IA KS NE MT WI PA NC GA TN KY".split()

CLINIC_TYPES = [
    "Family Medicine",
    "Internal Medicine",
    "Medical Center",
    "Health Center",
    "Clinic",
    "Community Hospital",
    "Primary Care",
    "Family Health",
    "Regional Medical Center",
    "Medical Associates",
    "Health Partners",
    "Senior Care Clinic",
    "Cardiology Associates",
    "Urgent Care",
    "Women's Health",
    "Specialty Clinic",
]
PHARM_TYPES = [
    "Drug",
    "Pharmacy",
    "Apothecary",
    "Rx",
    "Drugs",
    "Compounding Pharmacy",
    "Family Pharmacy",
    "Pharmacy & Gift",
    "Prescription Shop",
    "Discount Drug",
    "Community Pharmacy",
    "Corner Drug",
    "Health Pharmacy",
    "Medicine Cabinet",
    "Rx Express",
]
FUNERAL_TYPES = [
    "Funeral Home",
    "Funeral Chapel",
    "Mortuary",
    "Funeral Service",
    "Memorial Chapel",
    "Cremation Society",
    "Funeral Directors",
    "Family Funeral Home",
    "Memorial Funeral Home",
    "Cremation & Burial Service",
]
FUNERAL_SURNAME_TYPES = [
    "& Sons Funeral Home",
    "& Sons Mortuary",
    "& Daughters Funeral Home",
    "& Son Mortuary",
]
HOSPICE_FORMS = [
    "Hospice of {s}",
    "{s} Hospice",
    "{s} Hospice & Palliative Care",
    "{s} Comfort Care Hospice",
]
MON3 = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
MONFULL = "January February March April May June July August September October November December".split()
WEEKDAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
TENS = {9: "ninety"}
ONES = "zero one two three four five six seven eight nine".split()
AREA_CODES = [
    201,
    212,
    303,
    312,
    404,
    415,
    502,
    513,
    520,
    602,
    612,
    615,
    651,
    702,
    719,
    801,
    802,
    907,
    913,
    971,
]
RELATIONS = [
    "daughter",
    "son",
    "spouse",
    "sibling",
    "child",
    "grandchild",
    "niece or nephew",
    "friend",
    "neighbor",
    "partner",
    "caregiver",
    "relative",
]

TESTS = [
    ("bone density scan", 1990),
    ("DEXA", 1990),
    ("colonoscopy", 1995),
    ("screening mammogram", 1980),
    ("mammogram", 1975),
    ("Pap smear", 1970),
    ("low-dose chest CT", 2013),
    ("PSA", 1992),
    ("lipid panel", 1985),
    ("hemoglobin A1c", 1995),
    ("dilated eye exam", 1975),
    ("echocardiogram", 1985),
    ("stress test", 1980),
    ("nuclear stress test", 1990),
    ("EGD", 1985),
    ("upper endoscopy", 1985),
    ("flexible sigmoidoscopy", 1985),
    ("stool DNA test", 2015),
    ("FIT", 2005),
    ("carotid duplex", 1995),
    ("AAA ultrasound", 2005),
    ("hepatitis C antibody", 1995),
    ("audiogram", 1970),
    ("spirometry", 1980),
    ("sleep study", 1995),
    ("thyroid ultrasound", 1995),
    ("MRI brain", 1990),
    ("lumbar spine films", 1960),
    ("nerve conduction study", 1985),
    ("TB skin test", 1960),
    ("abdominal CT", 1985),
    ("chest X-ray", 1950),
    ("colposcopy", 1980),
    ("skin check with dermatology", 1990),
    ("whole-body bone scan", 1980),
    ("ankle-brachial index", 1995),
    ("Holter monitor", 1980),
    ("cardiac cath", 1980),
    ("coronary calcium score", 2000),
    ("hearing screen", 1990),
    ("PET scan", 2000),
    ("liver ultrasound", 1990),
    ("renal ultrasound", 1985),
    ("cystoscopy", 1980),
    ("bronchoscopy", 1985),
    ("capsule endoscopy", 2003),
    ("urine microalbumin", 1995),
    ("fasting glucose", 1985),
]
RESULTS = [
    "normal",
    "negative",
    "benign",
    "unremarkable",
    "osteopenia",
    "T-score -1.8",
    "T-score -2.6",
    "mild LVH",
    "EF 55%",
    "EF 60-65%",
    "no polyps",
    "two small adenomas",
    "one tubular adenoma",
    "BI-RADS 2",
    "BI-RADS 1",
    "stable nodule",
    "no change",
    "negative for malignancy",
    "within normal limits",
    "borderline",
    "A1c 6.1%",
    "mild sensory loss",
    "moderate OSA",
    "mild OSA",
    "repeat due",
    "no ischemia",
    "mild stenosis",
    "trace MR",
    "grade 1 diastolic dysfunction",
    "LDL 98",
    "TSH 2.1",
    "PSA 1.4",
    "mild fatty liver",
    "no hydronephrosis",
    "negative HPV",
    "two benign nevi",
    "no acute findings",
]
VACCINES = [
    ("influenza", 1990),
    ("Tdap", 2006),
    ("Td booster", 1970),
    ("Shingrix", 2018),
    ("zoster vaccine", 2007),
    ("PCV13", 2014),
    ("PCV20", 2022),
    ("PPSV23", 1990),
    ("hepatitis B series", 1992),
    ("hepatitis A series", 1996),
    ("MMR", 1960),
    ("COVID-19 vaccine", 2021),
    ("COVID booster", 2021),
    ("RSV vaccine", 2023),
    ("HPV series", 2007),
    ("tetanus shot", 1960),
    ("varicella vaccine", 1996),
    ("meningococcal vaccine", 2005),
    ("pneumonia vaccine", 1995),
    ("flu shot", 1990),
]
PROCEDURES = [
    "appendectomy",
    "cholecystectomy",
    "total knee replacement, right",
    "total knee replacement, left",
    "total hip replacement",
    "hysterectomy",
    "tonsillectomy",
    "cataract surgery, left eye",
    "cataract surgery, right eye",
    "CABG x3",
    "PCI with stent to LAD",
    "inguinal hernia repair",
    "lumpectomy",
    "thyroidectomy",
    "partial colectomy",
    "TURP",
    "lumbar fusion",
    "pacemaker placement",
    "carpal tunnel release",
    "rotator cuff repair",
    "bunionectomy",
    "cesarean section",
    "vasectomy",
    "septoplasty",
    "ACL reconstruction",
    "prostatectomy",
    "mastectomy",
    "angioplasty",
    "parathyroidectomy",
    "hemorrhoidectomy",
    "sleeve gastrectomy",
    "cervical discectomy",
    "ventral hernia repair",
    "knee arthroscopy",
    "oophorectomy",
    "ablation for AFib",
    "aortic valve replacement",
    "carotid endarterectomy",
    "lithotripsy",
]
PROC_MIN = {
    "pacemaker placement": 1975,
    "PCI with stent to LAD": 1995,
    "ablation for AFib": 2002,
    "sleeve gastrectomy": 2005,
    "aortic valve replacement": 1980,
    "CABG x3": 1975,
}
PROC_DETAIL = [
    "uncomplicated",
    "no complications",
    "outside hospital",
    "recovered well",
    "mesh used",
    "per patient report",
    "records not available",
    "good result",
    "healed well",
    "done as outpatient",
]
DIAGNOSES = [
    "hypertension",
    "type 2 diabetes",
    "hyperlipidemia",
    "atrial fibrillation",
    "breast cancer",
    "prostate cancer",
    "myocardial infarction",
    "DVT",
    "TIA",
    "kidney stones",
    "GERD",
    "major depression",
    "hypothyroidism",
    "asthma",
    "wrist fracture",
    "gout",
    "COPD",
    "osteoarthritis of the knees",
    "CKD",
    "anemia",
    "melanoma in situ",
]
GUIDE_TOPICS = [
    "colorectal screening",
    "lung cancer screening",
    "osteoporosis",
    "hypertension",
    "diabetes care",
    "cervical screening",
    "immunization",
    "lipid management",
    "breast screening",
    "prostate screening",
]
MEDS = [
    "lisinopril 20 mg daily",
    "metformin 500 mg BID",
    "atorvastatin 40 mg nightly",
    "amlodipine 5 mg daily",
    "levothyroxine 75 mcg daily",
    "omeprazole 20 mg daily",
    "sertraline 50 mg daily",
    "metoprolol succinate 25 mg daily",
    "losartan 50 mg daily",
    "gabapentin 300 mg TID",
    "apixaban 5 mg BID",
    "furosemide 20 mg daily",
    "tamsulosin 0.4 mg nightly",
    "alendronate 70 mg weekly",
    "montelukast 10 mg nightly",
    "glipizide 5 mg daily",
    "carvedilol 6.25 mg BID",
    "pantoprazole 40 mg daily",
    "allopurinol 100 mg daily",
    "donepezil 5 mg nightly",
]
FILLER = [
    "BP 128/76, HR 72, weight stable.",
    "Denies chest pain, shortness of breath or dizziness.",
    "Appetite is good and sleep is adequate.",
    "Walks about 20 minutes most days.",
    "No falls reported since the last visit.",
    "Mood is stable; PHQ-2 negative.",
    "Medications reviewed and reconciled with the patient.",
    "Lungs clear, heart regular, abdomen soft.",
    "Discussed diet, exercise and sodium restriction.",
    "Home blood pressure readings average 130/78.",
    "Lives with family and manages medications with a pill organizer.",
    "Vision and hearing adequate for daily tasks.",
    "Labs reviewed with the patient and tolerated well.",
    "Advance directive discussion offered.",
    "Fall-risk screen negative today.",
    "Cognition grossly intact on brief screen.",
    "Follow-up in clinic in three months or sooner if symptoms change.",
    "Patient verbalizes understanding of the plan.",
    "Exam unremarkable apart from mild bilateral ankle edema.",
    "No new complaints; reports good adherence.",
    "Encouraged adequate hydration and daily walking.",
    "Reviewed warning signs that need prompt evaluation.",
    "Diet is mostly home cooked with limited sweets.",
    "Pain is controlled on the current regimen.",
]

DEATH_FILLER = [
    "Family at the bedside and supported by staff.",
    "Chaplain visited and offered support.",
    "Personal belongings returned to the family.",
    "Hospice team notified.",
    "Death certificate paperwork started.",
    "Family given time with the patient.",
    "Comfort measures continued until the end.",
    "No distress noted at the time of death.",
    "Pastoral care called at the family's request.",
    "Family expressed gratitude for the care received.",
    "Bereavement resources reviewed with the family.",
    "Attending physician informed.",
]
PH_LABELS = [
    "Unit clerk",
    "Billing office",
    "Callback line",
    "Direct line",
    "Nurse line",
    "Front desk",
    "Case manager",
    "Home health coordinator",
    "Radiology scheduling",
    "Social work",
    "Discharge planner",
    "Referral coordinator",
    "Clinic office",
    "Charge nurse",
    "On-call service",
    "Transfer center",
    "Patient access",
    "Scheduling",
    "Care coordinator",
    "Medical records",
    "Lab results line",
    "Hospice intake",
    "Infusion center",
    "Triage nurse",
    "Physical therapy",
    "PCP office",
    "Cardiology office",
    "Answering service",
    "Utilization review",
]
EXT_FORMS = [
    ", extension {n}",
    " extension {n}",
    " EXT. {n}",
    " ext. {n}",
    " ext {n}",
    " x{n}",
    ", x{n}",
    " Ext {n}",
    ", ext. {n}",
    " EXT {n}",
    " extension #{n}",
    " ext: {n}",
    " x {n}",
    " ext #{n}",
]
PH_NEG = [
    "knee extension lacks 5 degrees",
    "full extension at the elbow",
    "ext. rotation 45 degrees bilaterally",
    "extension of the right great toe intact",
    "great toe extension 5/5",
    "back extension painful at end range",
    "extension tubing exchanged",
    "request for extension of stay pending",
    "hip extension 0-20 degrees",
    "wrist extension 4/5",
    "neck extension limited",
    "tumor extension into the pelvis not seen",
    "extended-release formulation continued",
    "ext. auditory canal clear",
    "EXT: no edema, pulses intact",
    "knee extension lag 5 degrees",
    "extension to 180 degrees",
    "ext. cord replaced at the bedside",
    "extension of the incision noted",
    "elbow extension and flexion full",
]
ROOMS = [
    "Room 4120",
    "Rm 318",
    "bed 7B",
    "room 2204-B",
    "transferred to room 214",
    "bed 12 West",
    "Room 5120",
    "Rm 4B",
]


def spelled_age(n: int) -> str:
    ones = n % 10
    return "ninety" if ones == 0 else f"ninety-{ONES[ones]}"


# ---------------------------------------------------------------------------
# Row builder: parts are plain strings or (LABEL, text[, tag]) tuples
# ---------------------------------------------------------------------------
def flat(*items):
    out = []
    for item in items:
        if isinstance(item, list):
            out.extend(flat(*item))
        elif item is not None:
            out.append(item)
    return out


CASE_FUNCS = {"normal": lambda s: s, "upper": str.upper, "lower": str.lower}


class RowBuilder:
    def __init__(self):
        self.parts: list = []

    def add(self, *items):
        self.parts.extend(flat(*items))
        return self

    def nl(self, count: int = 1):
        self.parts.append("\n" * count)
        return self

    def build(self, case: str = "normal"):
        fold = CASE_FUNCS[case]
        pieces, spans, tags, pos = [], [], [], 0
        for part in self.parts:
            if isinstance(part, tuple):
                label, value = part[0], fold(part[1])
                if not value or value != value.strip():
                    raise ValueError(f"bad span value {value!r}")
                spans.append(
                    {
                        "start": pos,
                        "end": pos + len(value),
                        "label": label,
                        "text": value,
                    }
                )
                if len(part) > 2:
                    tags.append((part[2], value))
            else:
                value = fold(part)
            pieces.append(value)
            pos += len(value)
        text = "".join(pieces)
        lead = len(text) - len(text.lstrip())
        text = text.strip()
        for span in spans:
            span["start"] -= lead
            span["end"] -= lead
        return text, spans, tags


class Registry:
    """Run-level caps on repeated values (rows per value)."""

    def __init__(self):
        self.uses: Counter = Counter()
        self.years: Counter = Counter()
        self.future: Counter = Counter()

    def snapshot(self):
        return Counter(self.uses), Counter(self.years), Counter(self.future)

    def restore(self, snapshot):
        self.uses, self.years, self.future = (Counter(part) for part in snapshot)

    def pick(self, rng, make, kind, cap=CAP):
        for _ in range(400):
            value = make(rng)
            key = (kind, value.casefold())
            if self.uses[key] < cap:
                self.uses[key] += 1
                return value
        raise RuntimeError(f"pool exhausted for {kind}")


class Ctx:
    def __init__(
        self,
        rng,
        reg,
        family,
        cohort,
        positive,
        case,
        year_cap=YEAR_CAP,
        pools=None,
        newcase=None,
    ):
        self.rng, self.reg, self.family, self.cohort = rng, reg, family, cohort
        self.positive, self.case = positive, case
        self.year_cap = year_cap
        self.firsts, self.lasts = pools if pools else (FIRSTS, LASTS)
        self.newcase = newcase
        self.hit = False
        self.neghit = False
        self.rb = RowBuilder()
        self.unusual = cohort == "unusual_name"
        self.long = cohort == "long_document"
        self.note_year = rng.choices([2022, 2023, 2024, 2025, 2026], [1, 2, 3, 4, 3])[0]
        old_share = 0.22 if family == "funeral_homes" else 0.09
        if rng.random() < old_share:
            self.age = rng.randint(90, 99)
        else:
            self.age = min(89, max(40, int(rng.gauss(68, 12))))
        self.birth_year = self.note_year - self.age
        self.people: list[str] = []
        self.patient = self.person(patient=True)
        self.mrn = rng.choice(["QX-", "", "MR", "K-", "HC", "LP-"]) + "".join(
            rng.choice("0123456789") for _ in range(rng.randint(6, 8))
        )
        self.bullet = rng.choice(["- ", "* ", "", "- ", "# "])
        self.clin = None

    # --- people -----------------------------------------------------------
    def person(self, patient=False):
        rng = self.rng
        for _ in range(1000):
            first = (
                rng.choice(UNUSUAL_FIRSTS)
                if (patient and self.unusual)
                else rng.choice(self.firsts)
            )
            last = rng.choice(self.lasts)
            if rng.random() < 0.1 and not (patient and self.unusual):
                first = f"{first} {rng.choice('ABCDEFGHJKLMNPRSTW')}."
            name = f"{first} {last}"
            if name in self.people:
                continue
            if identity_split(canonical_identity_group(name)) == "train":
                self.people.append(name)
                return name
        raise RuntimeError("no train-partition name found")

    def clinician(self):
        name = self.person()
        form = self.rng.choice(["dr", "dr", "md", "np", "rn", "pa", "do"])
        if form == "dr":
            return ["Dr. ", ("NAME", name)]
        suffix = {
            "md": ", MD",
            "np": ", NP-C",
            "rn": ", RN",
            "pa": ", PA-C",
            "do": ", DO",
        }[form]
        return [("NAME", name), suffix]

    def relative(self):
        name = self.person()
        return [self.rng.choice(RELATIONS), " ", ("NAME", name)]

    def name_ref(self):
        """The patient's name inside prose; unusual-name rows name the patient only in the header."""
        return [("NAME", self.patient)]

    # --- sizes and filler --------------------------------------------------
    def n(self, low, high):
        count = self.rng.randint(low, high)
        return count * 3 if self.long else count

    def fill(self, k=1):
        pool = DEATH_FILLER if self.family == "funeral_homes" else FILLER
        return " ".join(self.rng.sample(pool, min(k, len(pool))))

    def pad(self):
        """Extra PHI-free paragraphs for long_document rows."""
        if not self.long:
            return []
        return ["\n\n", self.fill(self.rng.randint(4, 7)), "\n"]

    # --- values -------------------------------------------------------------
    def facility(self):
        return (
            "FACILITY",
            self.reg.pick(
                self.rng,
                lambda r: f"{r.choice(STEMS)} {r.choice(CLINIC_TYPES)}",
                "facility",
            ),
            "facility",
        )

    def town(self):
        return self.reg.pick(self.rng, lambda r: r.choice(STEMS), "town")

    def town_line(self):
        r = self.rng
        town = self.town()
        if r.random() < 0.5:
            town = f"{town}, {r.choice(STATES)}"
        return [
            r.choice(
                [
                    "Lives in ",
                    "Family lives in ",
                    "Pt from ",
                    "Daughter lives in ",
                    "Resides in ",
                    "Returned from ",
                ]
            ),
            ("LOCATION", town, "town"),
            ". ",
        ]

    def zip_code(self):
        return self.reg.pick(self.rng, lambda r: f"{r.randint(10000, 99998)}", "zip")

    def phone(self, ext=False, fax=False):
        rng = self.rng
        ac, line = rng.choice(AREA_CODES), 100 + rng.randint(0, 99)
        number = rng.choice(
            [
                f"({ac}) 555-0{line}",
                f"{ac}-555-0{line}",
                f"{ac}.555.0{line}",
                f"+1 {ac} 555 0{line}",
                f"+1 ({ac}) 555-0{line}",
                f"{ac} 555 0{line}",
                f"({ac}) 555-0{line}",
                f"{ac}-555-0{line}",
            ]
        )
        if ext:
            n = "".join(
                rng.choice("0123456789")
                for _ in range(rng.choice([2, 3, 3, 4, 4, 5, 6]))
            )
            if n[0] == "0":
                n = "3" + n[1:]
            number += rng.choice(EXT_FORMS).format(n=n)
        return ("PHONE", number, "ext" if ext else "phone")

    def email(self):
        first, last = self.patient.split()[0], self.patient.split()[-1]
        handle = re.sub(r"[^a-z]", "", f"{first[0]}{last}".lower()) or "patient"
        return (
            "EMAIL",
            f"{handle}{self.rng.randint(1, 99)}@{self.rng.choice(['mail.example', 'example.org', 'example.net', 'post.test', 'inbox.example.com'])}",
        )

    def url(self):
        dom = self.rng.choice(
            ["clinic.test", "example.org", "health.example", "careportal.example.net"]
        )
        return ("URL", f"https://portal.{dom}/visit/{self.rng.randint(1000, 99999)}")

    def address(self):
        rng = self.rng
        town = self.town()
        z = self.zip_code()
        street = f"{rng.randint(10, 9899)} {rng.choice(STREET_STEMS)} {rng.choice(STREET_TYPES)}"
        unit = rng.choice(
            ["", "", "", f", Apt {rng.randint(1, 40)}", f", Unit {rng.choice('ABCDE')}"]
        )
        return ("ADDRESS", f"{street}{unit}, {town}, {rng.choice(STATES)} {z}")

    def insurance_id(self):
        rng = self.rng
        value = "".join(
            rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(3)
        ) + "".join(rng.choice("0123456789") for _ in range(9))
        return ("ID", value)

    def account(self):
        return (
            "ACCOUNT",
            "".join(
                self.rng.choice("0123456789") for _ in range(self.rng.randint(7, 9))
            ),
        )

    def ssn(self):
        r = self.rng
        return (
            "SSN",
            f"{r.randint(900, 999)}-{r.randint(10, 99)}-{r.randint(1000, 9999)}",
        )

    # --- dates ---------------------------------------------------------------
    def full_date(self, year):
        rng = self.rng
        month, day = rng.randint(1, 12), rng.randint(1, 28)
        if year >= 2026:
            month = rng.randint(1, 9)
        wd = WEEKDAYS[date(year, month, day).weekday()]
        form = rng.randrange(8)
        text = [
            f"{month:02d}/{day:02d}/{year}",
            f"{MONFULL[month - 1]} {day}, {year}",
            f"{day} {MON3[month - 1]} {year}",
            f"{year}-{month:02d}-{day:02d}",
            f"{MON3[month - 1]} {day}, {year}",
            f"{month}/{day}/{year % 100:02d}",
            f"{wd}, {MONFULL[month - 1]} {day}, {year}",
            f"{month}/{day}/{year}",
        ][form]
        clock = rng.choice(["", "", " 09:30", " 14:05", " 0830"])
        return [("DATE", text), clock]

    def year(self, miny=1950, floor=None):
        lo = max(1950, miny, self.birth_year + 10, floor or 0)
        hi = self.note_year
        cands = [y for y in range(lo, hi + 1) if self.reg.years[y] < self.year_cap]
        if not cands:
            return None
        year = self.rng.choices(cands, [1 + (y - 1950) / 40 for y in cands])[0]
        self.reg.years[year] += 1
        return year

    def year_parts(self, year):
        rng = self.rng
        if rng.random() < 0.74:
            return [("DATE", str(year), "year")]
        mi = rng.randrange(9 if year >= 2026 else 12)
        form = rng.randrange(4)
        text = [
            f"{MON3[mi]} {year}",
            f"{MONFULL[mi]} {year}",
            f"{mi + 1:02d}/{year}",
            f"{mi + 1}/{year}",
        ][form]
        return [("DATE", text, "year")]

    def dp(self, miny=1950, floor=None):
        year = self.year(miny, floor)
        return None if year is None else (self.year_parts(year), year)

    # --- age -------------------------------------------------------------------
    def age_parts(self, style=None):
        rng, a = self.rng, self.age
        if a <= 89:
            return [
                rng.choice(
                    [
                        f"{a}-year-old",
                        f"{a} y/o",
                        f"{a} year old",
                        f"{a}yo",
                        f"{a} years old",
                    ]
                )
            ]
        form = style if style is not None else rng.choice([0, 1, 1, 2, 3, 4, 5, 6])
        if form == 0:
            return [("AGE", str(a)), "-year-old"]
        if form == 1:
            return [("AGE", str(a)), " y/o"]
        if form == 2:
            return [("AGE", str(a)), rng.choice(["F", "M"])]
        if form == 3:
            return [("AGE", str(a)), " years old"]
        if form == 4:
            return [("AGE", spelled_age(a)), " years old"]
        if form == 5:
            return [("AGE", str(a)), "yo"]
        return [("AGE", str(a)), " year old"]

    def age_plain(self):
        return [("AGE", str(self.age))] if self.age > 89 else [str(self.age)]

    def age_years(self):
        if self.age <= 89:
            return [f"{self.age} years old"]
        return self.age_parts(self.rng.choice([3, 4]))

    # --- header ---------------------------------------------------------------------
    def hdr(self, keys, sep=None):
        rng = self.rng
        sep = sep or rng.choice(["\n", " | ", "   ", "; ", "\n"])
        items = []
        for key in keys:
            if key == "name":
                label = (
                    "Patient name"
                    if self.unusual
                    else rng.choice(
                        [
                            "Patient",
                            "Name",
                            "Pt",
                            "Patient name",
                            "PATIENT",
                            "Patient Name",
                        ]
                    )
                )
                items.append([label + ": ", ("NAME", self.patient)])
            elif key == "mrn":
                items.append(
                    [
                        rng.choice(
                            [
                                "MRN: ",
                                "MRN# ",
                                "MRN ",
                                "Medical record number: ",
                                "MR#: ",
                                "MRN - ",
                            ]
                        ),
                        ("MRN", self.mrn),
                    ]
                )
            elif key == "dob":
                items.append(
                    [
                        rng.choice(
                            ["DOB: ", "Date of birth: ", "Birth date: ", "D.O.B. "]
                        ),
                        self.full_date(self.birth_year)[:1],
                    ]
                )
            elif key == "dos":
                items.append(
                    [
                        rng.choice(
                            [
                                "Date of service: ",
                                "DOS: ",
                                "Visit date: ",
                                "Seen: ",
                                "Encounter date: ",
                                "Date: ",
                            ]
                        ),
                        self.full_date(self.note_year),
                    ]
                )
            elif key == "prov":
                items.append(
                    [
                        rng.choice(
                            [
                                "Provider: ",
                                "Clinician: ",
                                "PCP: ",
                                "Seen by: ",
                                "Attending: ",
                                "Author: ",
                            ]
                        ),
                        self.clinician(),
                    ]
                )
            elif key == "fac":
                items.append(
                    [
                        rng.choice(["Clinic: ", "Facility: ", "Site: ", ""]),
                        self.facility(),
                    ]
                )
            elif key == "age":
                items.append(
                    ["Age: ", self.age_parts(0 if self.age > 89 else None)[:1]]
                )
        out = []
        for i, item in enumerate(items):
            if i:
                out.append(sep)
            out.append(item)
        return out

    def sign(self):
        """Closing lines naming the patient; skipped for unusual-name rows."""
        if self.unusual:
            return ["Thank you"]
        return [
            self.rng.choice(["Thanks, ", "Sincerely, ", "Thank you, ", "Regards, "]),
            ("NAME", self.patient),
        ]


# ---------------------------------------------------------------------------
# History-year items
# ---------------------------------------------------------------------------
def cap_first(rng, text):
    return text[0].upper() + text[1:] if rng.random() < 0.5 else text


TEST_LIST = [
    lambda T, d, R: [T, " ", d, ": ", R],
    lambda T, d, R: [T, " (", d, ") - ", R],
    lambda T, d, R: [T, ", ", d, ", ", R],
    lambda T, d, R: [T, " done in ", d, "; ", R],
    lambda T, d, R: [T, " - last ", d],
    lambda T, d, R: [d, " ", T, ": ", R],
    lambda T, d, R: [T, ": ", R, " (", d, ")"],
    lambda T, d, R: ["last ", T, " was ", d, ", ", R],
    lambda T, d, R: [T, " on file from ", d],
    lambda T, d, R: [T, " | ", d, " | ", R],
    lambda T, d, R: [T, "  ", d, "  ", R],
    lambda T, d, R: [T, " - ", d],
    lambda T, d, R: [T, " (", d, ")"],
    lambda T, d, R: [T, " completed ", d, ", ", R],
]
VAX_LIST = [
    lambda V, d, d2, k: [V, " ", d],
    lambda V, d, d2, k: [V, " booster ", d],
    lambda V, d, d2, k: [V, " (", d, ")"],
    lambda V, d, d2, k: [V, " - ", d, ", ", d2] if d2 else [V, " - ", d],
    lambda V, d, d2, k: [V, " series completed ", d],
    lambda V, d, d2, k: [V, ": received ", d],
    lambda V, d, d2, k: [V, " given ", d],
    lambda V, d, d2, k: [V, " dose ", str(k), " ", d],
    lambda V, d, d2, k: [V, " ", d, " - no reaction"],
    lambda V, d, d2, k: [V, " last ", d],
    lambda V, d, d2, k: [V, ", ", d],
]
PROC_LIST = [
    lambda P, d, X: [P, " (", d, ")"],
    lambda P, d, X: ["s/p ", P, " ", d],
    lambda P, d, X: [P, " in ", d],
    lambda P, d, X: [P, " ", d, " - ", X],
    lambda P, d, X: [d, " - ", P],
    lambda P, d, X: [P, ", ", d],
    lambda P, d, X: [P, " (", d, "), ", X],
    lambda P, d, X: [P, ": ", d, ", ", X],
    lambda P, d, X: [P, " performed ", d],
    lambda P, d, X: ["history of ", P, " in ", d],
    lambda P, d, X: [P, " - ", d],
]
DX_LIST = [
    lambda D, d: [D, " diagnosed ", d],
    lambda D, d: [D, " since ", d],
    lambda D, d: ["h/o ", D, " (", d, ")"],
    lambda D, d: [D, " dx ", d],
    lambda D, d: [D, ", onset ", d],
]
SENT_POS = [
    lambda T, d, R: ["Last ", T, " was in ", d, "; ", R, "."],
    lambda T, d, R: [T, " in ", d, " showed ", R, "."],
    lambda T, d, R: ["Had a ", T, " around ", d, " (", R, ")."],
    lambda T, d, R: ["Patient reports ", T, " ", d, " with ", R, "."],
    lambda T, d, R: ["Per patient, the most recent ", T, " was ", d, "."],
    lambda T, d, R: ["I had my ", T, " back in ", d, " and it was ", R, "."],
    lambda T, d, R: ["My last ", T, " was ", d, "."],
]
SENT_PROC = [
    lambda P, d, X: ["Underwent ", P, " in ", d, ", ", X, "."],
    lambda P, d, X: ["Had ", P, " ", d, "."],
    lambda P, d, X: ["Reports ", P, " (", d, "), ", X, "."],
    lambda P, d, X: ["I had ", P, " in ", d, "."],
    lambda P, d, X: ["Surgical history includes ", P, " ", d, "."],
]
SENT_VAX = [
    lambda V, d, d2, k: ["Received ", V, " in ", d, "."],
    lambda V, d, d2, k: [V, " was given ", d, "."],
    lambda V, d, d2, k: ["My ", V, " was ", d, "."],
    lambda V, d, d2, k: ["Last ", V, " documented ", d, "."],
]


def results_for(rng, test):
    t = test.lower()
    if "a1c" in t:
        return rng.choice(["A1c 6.1%", "A1c 7.2%", "A1c 6.8%", "controlled"])
    if "lipid" in t:
        return rng.choice(["LDL 98", "LDL 112", "LDL 74", "at goal"])
    if "psa" in t:
        return rng.choice(["PSA 1.4", "PSA 0.8", "stable"])
    if any(k in t for k in ("colonoscopy", "egd", "endoscopy", "sigmoidoscopy")):
        return rng.choice(
            [
                "no polyps",
                "two small adenomas",
                "one tubular adenoma",
                "benign",
                "normal",
                "no acute findings",
            ]
        )
    if "mammogram" in t:
        return rng.choice(["BI-RADS 1", "BI-RADS 2", "normal", "benign"])
    if "dexa" in t or "bone" in t:
        return rng.choice(["osteopenia", "T-score -1.8", "T-score -2.6", "normal"])
    if any(
        k in t
        for k in (
            "echo",
            "stress",
            "cath",
            "holter",
            "calcium",
            "ankle-brachial",
            "carotid",
        )
    ):
        return rng.choice(
            [
                "mild LVH",
                "EF 55%",
                "EF 60-65%",
                "no ischemia",
                "mild stenosis",
                "trace MR",
                "grade 1 diastolic dysfunction",
                "normal",
            ]
        )
    if "audiogram" in t or "hearing" in t:
        return rng.choice(["mild high-frequency loss", "normal", "moderate loss"])
    if "spirometry" in t:
        return rng.choice(["FEV1 78% predicted", "normal", "mild obstruction"])
    if "sleep" in t:
        return rng.choice(["mild OSA", "moderate OSA", "negative"])
    if "nerve" in t:
        return rng.choice(["mild sensory loss", "normal"])
    if "skin" in t:
        return rng.choice(["two benign nevi", "benign", "no suspicious lesions"])
    if "liver" in t:
        return rng.choice(["mild fatty liver", "normal", "stable"])
    if "renal" in t:
        return rng.choice(["no hydronephrosis", "normal", "stable cyst"])
    if "colposcopy" in t or "pap" in t:
        return rng.choice(["negative", "normal", "negative HPV"])
    if any(
        k in t for k in ("ct", "mri", "x-ray", "pet", "ultrasound", "films", "scan")
    ):
        return rng.choice(
            [
                "stable nodule",
                "no change",
                "unremarkable",
                "no acute findings",
                "negative for malignancy",
            ]
        )
    return rng.choice(
        [
            "normal",
            "negative",
            "unremarkable",
            "within normal limits",
            "borderline",
            "repeat due",
        ]
    )


def topic_for(test):
    t = test.lower()
    for keys, topic in (
        (("colon", "sigmoid", "fit", "stool"), "colorectal screening"),
        (("mammogram",), "breast screening"),
        (("pap", "colposcopy"), "cervical screening"),
        (("psa",), "prostate screening"),
        (("chest ct",), "lung cancer screening"),
        (("bone", "dexa"), "osteoporosis"),
        (("a1c", "glucose", "microalbumin", "eye"), "diabetes care"),
        (("lipid", "calcium"), "lipid management"),
    ):
        if any(k in t for k in keys):
            return topic
    return "preventive care"


def pub_year(rng):
    return str(rng.randint(1995, 2025))


ONSET_FMT = [
    lambda D, d: [D, " diagnosed in ", d],
    lambda D, d: [D, " since ", d],
    lambda D, d: ["diagnosed with ", D, " in ", d],
    lambda D, d: [D, ", since ", d],
    lambda D, d: ["carries a diagnosis of ", D, " since ", d],
    lambda D, d: [D, " (dx ", d, ")"],
    lambda D, d: [D, " - onset ", d],
    lambda D, d: ["Dx ", D, ", ", d],
    lambda D, d: [D, " first diagnosed ", d],
]
RANGE_CTX = [
    "smoked",
    "on tamoxifen",
    "on prednisone",
    "lived overseas",
    "chemotherapy",
    "physical therapy",
    "on dialysis",
    "used CPAP",
    "took warfarin",
    "treated with insulin",
    "cardiac rehab",
    "on a statin",
    "tobacco use",
    "heavy alcohol use",
    "inpatient psychiatric care",
]
RANGE_FMT = [
    lambda x, r: [x, " ", r],
    lambda x, r: [x, ": ", r],
    lambda x, r: [x, " (", r, ")"],
    lambda x, r: [x, " from ", r],
    lambda x, r: [x, ", ", r],
]
FUTURE_FMT = [
    lambda T, R, d, V: [T, ": ", R, "; next due ", d],
    lambda T, R, d, V: [T, " next due ", d],
    lambda T, R, d, V: ["Repeat ", T, " due ", d],
    lambda T, R, d, V: ["next ", T, ": ", d],
    lambda T, R, d, V: [T, ", recheck ", d],
    lambda T, R, d, V: [V, ": next due ", d],
    lambda T, R, d, V: [T, ", due again ", d],
    lambda T, R, d, V: ["Next screening ", d],
]
FUTURE_SENT = [
    lambda T, d, V: ["Next ", T, " is due ", d, "."],
    lambda T, d, V: ["Plan to repeat ", T, " in ", d, "."],
    lambda T, d, V: ["Due for ", T, " again ", d, "."],
    lambda T, d, V: [V, " is next due ", d, "."],
]


def sentence_case(item):
    item = flat(item)
    if item and isinstance(item[0], str) and item[0]:
        item[0] = item[0][0].upper() + item[0][1:]
    return item


def future_parts(c):
    rng = c.rng
    for _ in range(80):
        year = c.note_year + rng.randint(1, 12)
        if c.reg.future[year] < FUTURE_CAP:
            c.reg.future[year] += 1
            break
    else:
        raise RuntimeError("future years exhausted")
    if rng.random() < 0.72:
        return [("DATE", str(year), "future")]
    mi = rng.randrange(12)
    text = rng.choice(
        [f"{MON3[mi]} {year}", f"{MONFULL[mi]} {year}", f"{mi + 1:02d}/{year}"]
    )
    return [("DATE", text, "future")]


def hist_new_pos(c, sentence):
    rng, kind = c.rng, c.newcase
    for _ in range(60):
        if kind == "onset_year":
            D = rng.choice(DIAGNOSES)
            got = c.dp(1950)
            if got is None:
                continue
            item = rng.choice(ONSET_FMT)(D if not sentence else D, got[0])
        elif kind == "year_range":
            y1 = c.year(1950)
            if y1 is None:
                continue
            y2 = c.year(y1 + 1, floor=y1 + 1)
            if y2 is None:
                continue
            text = rng.choice(
                [
                    f"{y1}-{y2}",
                    f"{y1}-{y2}",
                    f"{y1}-{y2}",
                    f"{y1} - {y2}",
                    f"{y1} to {y2}",
                ]
            )
            item = rng.choice(RANGE_FMT)(
                rng.choice(RANGE_CTX), [("DATE", text, "year")]
            )
        else:
            T, _miny = rng.choice(TESTS)
            V = rng.choice(VACCINES)[0]
            d = future_parts(c)
            if sentence:
                item = rng.choice(FUTURE_SENT)(T, d, V)
            else:
                item = rng.choice(FUTURE_FMT)(T, results_for(rng, T), d, V)
        c.hit = True
        item = flat(item)
        if sentence:
            return (
                sentence_case(item) if item[-1] == "." else [*sentence_case(item), "."]
            )
        return [cap_first_part(rng, item)]
    raise RuntimeError("no feasible new-case history item")


def cap_first_part(rng, item):
    item = flat(item)
    if rng.random() < 0.5 and isinstance(item[0], str) and item[0]:
        item[0] = item[0][0].upper() + item[0][1:]
    return item


def hist_new_neg(c, sentence):
    rng, kind = c.rng, c.newcase
    D, T = rng.choice(DIAGNOSES), rng.choice(TESTS)[0]
    n, m, a = rng.choice([2, 3, 5, 7, 10]), rng.choice([2, 3, 4, 6]), min(c.age, 89)
    py = pub_year(rng)
    TG = topic_for(T)
    if kind == "onset_year":
        options = [
            f"{D} diagnosed at age {rng.randint(25, max(26, a - 3))}",
            f"{D} for {n} years",
            f"{D} diagnosed years ago",
            f"{D} since age {rng.randint(25, max(26, a - 3))}",
            f"{D}, stage {rng.choice(['1', '2', '3a', 'II'])}, dx per the {py} criteria",
            f"{D} of long standing",
        ]
    elif kind == "year_range":
        y = int(py)
        options = [
            f"per the {y}-{y + rng.randint(2, 5)} {TG} cohort study",
            f"ages {rng.randint(45, 55)}-{rng.randint(70, 80)} per the {py} {TG} guideline",
            rng.choice(
                [
                    "metformin 500-1000 mg",
                    "lisinopril 10-40 mg",
                    "atorvastatin 10-80 mg",
                    "amlodipine 2.5-10 mg",
                ]
            ),
            f"BP goal {rng.randint(110, 120)}-{rng.randint(125, 135)}/{rng.randint(65, 75)}-{rng.randint(78, 85)}",
            f"{D}, stage {rng.choice(['2', '3'])}-{rng.choice(['3', '4'])}",
            f"recovery over {m}-{m + 3} weeks",
            f"{D} for {n}-{n + 4} years",
            f"{y}-{y + 3} registry data reviewed",
        ]
    else:
        options = [
            f"{T}: next due in {n} years",
            f"repeat {T} in {n} years",
            f"next dose in {m} months",
            f"next visit in {m} months",
            f"{T} due per the {py} {TG} guideline",
            f"{T} to be repeated every {n} years",
            f"recheck in {m} weeks",
        ]
    text = rng.choice(options)
    if sentence:
        text = text[0].upper() + text[1:] + "."
    elif rng.random() < 0.5:
        text = text[0].upper() + text[1:]
    c.neghit = True
    return [text]


def hist_pos(c, sentence=False):
    if c.newcase == "hist_v3":
        return hist_v3_pos(c, sentence)
    if c.newcase in HIST_NEW and (not c.hit or c.rng.random() < 0.6):
        return hist_new_pos(c, sentence)
    rng = c.rng
    for _ in range(60):
        kind = rng.choices(["test", "vax", "proc", "dx"], [40, 22, 28, 10])[0]
        if kind == "test":
            T, miny = rng.choice(TESTS)
            got = c.dp(miny)
            if got is None:
                continue
            R = results_for(rng, T)
            T = T if sentence else cap_first(rng, T)
            return (rng.choice(SENT_POS) if sentence else rng.choice(TEST_LIST))(
                T, got[0], R
            )
        if kind == "vax":
            V, miny = rng.choice(VACCINES)
            got = c.dp(miny)
            if got is None:
                continue
            d2 = None
            if rng.random() < 0.3:
                second = c.dp(got[1] + 1, floor=got[1] + 1)
                d2 = second[0] if second else None
            V = V if sentence else cap_first(rng, V)
            return (rng.choice(SENT_VAX) if sentence else rng.choice(VAX_LIST))(
                V, got[0], d2, rng.randint(1, 3)
            )
        if kind == "proc":
            P = rng.choice(PROCEDURES)
            got = c.dp(PROC_MIN.get(P, 1950))
            if got is None:
                continue
            P = P if sentence else cap_first(rng, P)
            return (rng.choice(SENT_PROC) if sentence else rng.choice(PROC_LIST))(
                P, got[0], rng.choice(PROC_DETAIL)
            )
        D = rng.choice(DIAGNOSES)
        got = c.dp(1950)
        if got is None:
            continue
        item = rng.choice(DX_LIST)(D if sentence else cap_first(rng, D), got[0])
        return [*item, "."] if sentence else item
    raise RuntimeError("no feasible history year item")


def hist_neg(c, sentence=False):
    """A history-flavoured item with no patient-specific year (public years, durations, doses, stages, ages <= 89)."""
    if c.newcase in HIST_NEW and (not c.neghit or c.rng.random() < 0.6):
        return hist_new_neg(c, sentence)
    if c.newcase == "hist_v3":
        item = hist_v3_neg(c, sentence)
        if item is not None:
            return item
    rng = c.rng
    Tl = rng.choice(TESTS)[0]
    T = cap_first(rng, Tl)
    V = cap_first(rng, rng.choice(VACCINES)[0])
    P = cap_first(rng, rng.choice(PROCEDURES))
    D = cap_first(rng, rng.choice(DIAGNOSES))
    R, G, n, a = (
        results_for(rng, Tl),
        rng.choice(GUIDE_TOPICS),
        rng.choice([2, 3, 5, 7, 10, 12, 15, 20]),
        min(c.age, 89),
    )
    TG = topic_for(Tl)
    py = pub_year(rng)
    dose_total = rng.randint(2, 3)
    options = [
        f"{T}: {R}; repeat in {n} years",
        f"{T} due per the {py} {TG} guideline",
        f"{V}: dose {rng.randint(1, dose_total)} of {dose_total}",
        f"{P} at age {rng.randint(25, max(26, a - 3))}",
        f"{P}, about {n} years ago",
        f"{D} for {n} years",
        f"{D}, stage {rng.choice(['1', '2', '3a', '3b', 'II', 'III'])}",
        f"{V} offered; patient declined",
        f"{T}: {R}",
        f"{T} ordered; {R} expected",
        f"{T} every {n} years per the {py} {TG} recommendations",
        f"{V} 0.5 mL IM, lot HK{rng.randint(1000, 9999)}",
        f"{G} reviewed at age {rng.randint(min(45, a), a)}",
        f"{D}; smoker for {n} years, {n + rng.randint(2, 12)} pack-years",
        f"{T} not on file; ordered today",
        f"Results of the {py} {rng.choice(['cohort study', 'randomized trial', 'meta-analysis', 'consensus statement'])} on {G} discussed",
        f"{D}, A1c {rng.randint(55, 90) / 10:.1f}%, eGFR {rng.randint(35, 95)}",
        f"{V} schedule reviewed per the {py} immunization guideline",
        f"{T}: {R} (per the {py} {TG} guideline, next in {n} years)",
    ]
    text = rng.choice(options)
    if sentence:
        text = text[0].upper() + text[1:] + "."
    return [text]


# ---------------------------------------------------------------------------
# v3 history-year variety (phrasings written for v3; years stay DATE)
# ---------------------------------------------------------------------------
TEST_V3 = [
    lambda T, d, R: [T, " completed ", d],
    lambda T, d, R: [T, ": negative (", d, ")"],
    lambda T, d, R: [T, " ", d, " - ", R],
    lambda T, d, R: [T, " (", d, ") ", R],
    lambda T, d, R: [T, " last done ", d, "; ", R],
    lambda T, d, R: ["Last ", T, " (", d, "): ", R],
    lambda T, d, R: [T, ", result ", R, " (", d, ")"],
    lambda T, d, R: [d, " ", T, " - ", R],
    lambda T, d, R: [T, " report dated ", d, " on file"],
    lambda T, d, R: [T, " done ", d, ", ", R, ", no follow-up needed"],
    lambda T, d, R: [T, " -- ", d, " -- ", R],
    lambda T, d, R: [T, " (outside records, ", d, ")"],
    lambda T, d, R: [T, " ", d],
    lambda T, d, R: [T, " normal (", d, ")"],
]
SENT_TEST_V3 = [
    lambda T, d, R: [T, " completed ", d, ", ", R, "."],
    lambda T, d, R: ["Last ", T, " (", d, "): ", R, "."],
    lambda T, d, R: ["Records show ", T, " ", d, ", result ", R, "."],
    lambda T, d, R: ["Most recent ", T, " was ", d, "; no repeat needed yet."],
    lambda T, d, R: [T, ": negative (", d, ")."],
    lambda T, d, R: ["Screening history: ", T, " ", d, ", ", R, "."],
    lambda T, d, R: ["Pt had ", T, " in ", d, "; ", R, "."],
    lambda T, d, R: [T, " ", d, " - ", R, "."],
]
VAX_V3 = [
    lambda V, d, d2, k: [V, " ", d],
    lambda V, d, d2, k: [V, " completed ", d],
    lambda V, d, d2, k: [V, " (", d, "): tolerated well"],
    lambda V, d, d2, k: ["Immunized: ", V, " ", d],
    lambda V, d, d2, k: (
        [V, " given ", d, " and ", d2] if d2 else [V, " administered ", d]
    ),
    lambda V, d, d2, k: [V, " dose ", str(k), " - ", d],
    lambda V, d, d2, k: [V, ", last documented ", d],
    lambda V, d, d2, k: [V, " per registry ", d],
    lambda V, d, d2, k: [V, " ", d, "; next per schedule"],
]
SENT_VAX_V3 = [
    lambda V, d, d2, k: [V, " ", d, "."],
    lambda V, d, d2, k: ["Immunization history: ", V, " ", d, "."],
    lambda V, d, d2, k: [V, " completed ", d, ", no reaction."],
    lambda V, d, d2, k: ["Up to date on ", V, " (", d, ")."],
    lambda V, d, d2, k: [V, " administered ", d, " per registry."],
]
PROC_V3 = [
    lambda P, d, X: [P, " ", d, ", ", X],
    lambda P, d, X: ["S/p ", P, " (", d, ")"],
    lambda P, d, X: [P, " completed ", d],
    lambda P, d, X: [P, " - ", d, " - ", X],
    lambda P, d, X: [d, " ", P, ", ", X],
    lambda P, d, X: [P, " done ", d, "; ", X],
    lambda P, d, X: ["Remote ", P, " (", d, ")"],
    lambda P, d, X: [P, ", ", d, " (outside hospital)"],
]
SENT_PROC_V3 = [
    lambda P, d, X: ["Surgical history: ", P, " (", d, ")."],
    lambda P, d, X: [P, " completed ", d, "."],
    lambda P, d, X: ["Status post ", P, " ", d, ", ", X, "."],
    lambda P, d, X: ["Remote ", P, " in ", d, "."],
    lambda P, d, X: [P, " ", d, "; ", X, "."],
]


def hist_v3_pos(c, sentence):
    rng = c.rng
    for _ in range(60):
        kind = rng.choices(["test", "vax", "proc"], [46, 24, 30])[0]
        if kind == "test":
            T, miny = rng.choice(TESTS)
            got = c.dp(miny)
            if got is None:
                continue
            R = results_for(rng, T)
            T = T if sentence else cap_first(rng, T)
            item = (rng.choice(SENT_TEST_V3) if sentence else rng.choice(TEST_V3))(
                T, got[0], R
            )
        elif kind == "vax":
            V, miny = rng.choice(VACCINES)
            got = c.dp(miny)
            if got is None:
                continue
            d2 = None
            if rng.random() < 0.25:
                second = c.dp(got[1] + 1, floor=got[1] + 1)
                d2 = second[0] if second else None
            V = V if sentence else cap_first(rng, V)
            item = (rng.choice(SENT_VAX_V3) if sentence else rng.choice(VAX_V3))(
                V, got[0], d2, rng.randint(1, 3)
            )
        else:
            P = rng.choice(PROCEDURES)
            got = c.dp(PROC_MIN.get(P, 1950))
            if got is None:
                continue
            P = P if sentence else cap_first(rng, P)
            item = (rng.choice(SENT_PROC_V3) if sentence else rng.choice(PROC_V3))(
                P, got[0], rng.choice(PROC_DETAIL)
            )
        c.hit = True
        return sentence_case(item) if sentence else flat(item)
    raise RuntimeError("no feasible v3 history year item")


def hist_v3_neg(c, sentence):
    """A year-free v3 history item, or None to fall back to the general negatives."""
    rng = c.rng
    if rng.random() < 0.4:
        return None
    Tl = rng.choice(TESTS)[0]
    T = cap_first(rng, Tl)
    V = cap_first(rng, rng.choice(VACCINES)[0])
    P = cap_first(rng, rng.choice(PROCEDURES))
    a = min(c.age, 89)
    x = rng.randint(20, max(21, a - 3))
    n = rng.choice([2, 3, 5, 7, 10])
    py, TG, R = pub_year(rng), topic_for(Tl), results_for(rng, Tl)
    text = rng.choice(
        [
            f"{T}: negative; date not recorded",
            f"{T} completed, date unknown",
            f"{T}: {R} (per patient report, no date given)",
            f"{V} completed in childhood",
            f"{V}: series completed, dates not documented",
            f"{V} given at age {x}",
            f"{P} completed, date not recorded",
            f"{P} at age {x}, no complications",
            f"{T}: negative (per the {py} {TG} guideline, repeat in {n} years)",
            f"{T} due in {n} years",
            f"{V} due this season",
            f"{T} ordered; result pending",
        ]
    )
    c.neghit = True
    if sentence:
        text = text[0].upper() + text[1:] + "."
    return [text]


def hist_lines(c, count, mix=0.0, sentence=False, join="\n", prefix=None):
    """Positive rows: patient-specific year items, with an occasional negative line mixed in. Negative rows: only negatives."""
    out = []
    for i in range(count):
        use_neg = (not c.positive) or c.rng.random() < mix
        item = hist_neg(c, sentence) if use_neg else hist_pos(c, sentence)
        if i:
            out.append(join)
        out.append(prefix if prefix is not None else c.bullet)
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# History layouts
# ---------------------------------------------------------------------------
def pick(rng, options):
    return rng.choice(options)


def L_awv_note(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng,
            [
                "ANNUAL WELLNESS VISIT",
                "Annual Wellness Visit Note",
                "Medicare annual wellness visit",
                "AWV - preventive visit",
                "Preventive visit note",
            ],
        ),
        "\n",
    )
    rb.add(c.hdr(["name", "mrn", "dos", "prov"][: rng.randint(3, 4)]), "\n\n")
    rb.add(
        pick(
            rng,
            [
                "Annual preventive visit; age ",
                "Here for the yearly preventive visit, age ",
                "Presents for routine wellness care, age ",
            ],
        ),
        c.age_plain(),
        ". ",
        c.fill(2),
        " ",
        c.pad(),
        "\n",
    )
    rb.add(
        pick(
            rng,
            [
                "HEALTH MAINTENANCE:",
                "Health Maintenance",
                "PREVENTIVE CARE:",
                "Preventive care summary:",
                "HEALTH MAINTENANCE / SCREENING",
            ],
        ),
        "\n",
    )
    rb.add(hist_lines(c, c.n(3, 7), mix=0.25))
    rb.add(c.pad(), "\n\nASSESSMENT/PLAN: ", c.fill(2))
    if c.positive and rng.random() < 0.5:
        rb.add(" ", hist_neg(c, True))


def L_soap_note(c):
    rb, rng = c.rb, c.rng
    rb.add(
        "SOAP NOTE\n",
        c.hdr(["name", "mrn", "dos"], sep=pick(rng, ["\n", " | "])),
        "\n\n",
    )
    rb.add("S: Age ", c.age_plain(), ", seen for follow-up. ", c.fill(1), " ")
    for _i in range(c.n(2, 4)):
        item = (
            hist_neg(c, True)
            if (not c.positive or rng.random() < 0.2)
            else hist_pos(c, True)
        )
        rb.add(item, " ")
    rb.add(c.pad(), "\nO: ", c.fill(2), "\nA: ", c.fill(1), "\nP: ", c.fill(1))
    if c.positive:
        rb.add(" ", hist_neg(c, True))
    rb.add("\n", c.clinician())


def L_screening_schedule(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng,
            [
                "SCREENING SCHEDULE:",
                "Screening schedule",
                "SCREENING SCHEDULE / REMINDERS",
                "Screening tracker:",
            ],
        ),
        "\n",
    )
    rb.add(c.hdr(["name", "mrn"], sep=" | "), "\n")
    rb.add(
        pick(
            rng,
            [
                "Test | Last done | Interval",
                "Item   Last done   Repeat",
                "Screening - last - interval",
            ],
        ),
        "\n",
    )
    for _i in range(c.n(3, 6)):
        T = pick(rng, TESTS)
        interval = pick(
            rng,
            [
                "10 yr",
                "5 yr",
                "every 3 years",
                "annual",
                "q2y",
                "every 12 months",
                "q10y",
            ],
        )
        if c.positive:
            got = c.dp(T[1])
            if got is None:
                continue
            rb.add(
                cap_first(rng, T[0]),
                pick(rng, [" | ", "   ", " - ", ": "]),
                got[0],
                pick(rng, [" | ", "   ", " - ", ", repeat "]),
                interval,
                "\n",
            )
        else:
            last = pick(
                rng,
                [
                    "never done",
                    "not on file",
                    "declined",
                    "ordered",
                    "pending referral",
                    "unknown",
                    "due now",
                ],
            )
            rb.add(
                cap_first(rng, T[0]),
                pick(rng, [" | ", "   ", " - ", ": "]),
                last,
                pick(rng, [" | ", "   ", " - ", ", repeat "]),
                interval,
                "\n",
            )
    rb.add(c.pad(), hist_neg(c, True))


def L_surgical_history_list(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng,
            [
                "PAST SURGICAL HISTORY:",
                "Past surgical history",
                "PSH:",
                "SURGICAL HISTORY",
                "Surgical hx:",
            ],
        ),
        "\n",
    )
    for i in range(c.n(2, 6)):
        P = cap_first(rng, pick(rng, PROCEDURES))
        if c.positive:
            got = c.dp(PROC_MIN.get(P.lower(), 1950))
            if got is None:
                continue
            body = rng.choice(PROC_V3 if c.newcase == "hist_v3" else PROC_LIST)(
                P, got[0], rng.choice(PROC_DETAIL)
            )
        else:
            body = [
                pick(
                    rng,
                    [
                        f"{P} - remote",
                        f"{P}, about {rng.randint(3, 30)} years ago",
                        f"{P} at age {rng.randint(18, min(c.age, 89))}",
                        f"{P}, no complications",
                        f"{P}, date unknown",
                    ],
                )
            ]
        rb.add(f"{i + 1}. ", body, "\n")
    rb.add(
        c.pad(),
        "\n",
        c.hdr(["name", "mrn", "dos"], sep=" | "),
        "\nNo other operations. ",
        c.fill(1),
    )


def L_problem_list(c):
    rb, rng = c.rb, c.rng
    rb.add(c.hdr(["name", "dob" if rng.random() < 0.4 else "mrn"], sep=" | "), "\n\n")
    rb.add(
        pick(
            rng,
            [
                "PROBLEM LIST:",
                "Active problems",
                "PROBLEM LIST (reviewed)",
                "Problem list / history",
            ],
        ),
        "\n",
    )
    for i in range(c.n(3, 7)):
        body = hist_neg(c) if not c.positive or rng.random() < 0.25 else hist_pos(c)
        rb.add(f"{i + 1}) ", body, "\n")
    rb.add(c.pad(), "Last reviewed with patient and family. ", c.clinician())


def L_immunization_record(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng,
            [
                "IMMUNIZATIONS:",
                "Immunization record",
                "VACCINE HISTORY",
                "Immunizations on file",
            ],
        ),
        "\n",
    )
    rb.add(c.hdr(["name", "mrn"], sep="   "), "\n")
    for _i in range(c.n(3, 6)):
        V, miny = pick(rng, VACCINES)
        V = cap_first(rng, V)
        if c.positive:
            got = c.dp(miny)
            if got is None:
                continue
            d2 = None
            if rng.random() < 0.25:
                second = c.dp(got[1] + 1, floor=got[1] + 1)
                d2 = second[0] if second else None
            body = rng.choice(VAX_V3 if c.newcase == "hist_v3" else VAX_LIST)(
                V, got[0], d2, rng.randint(1, 3)
            )
        else:
            body = [
                pick(
                    rng,
                    [
                        f"{V}: dose {rng.randint(1, 3)} of {rng.randint(2, 3)}",
                        f"{V} offered; declined",
                        f"{V} due",
                        f"{V} 0.5 mL IM",
                        f"{V} not documented",
                        f"{V} every {rng.randint(1, 10)} years per the {pub_year(rng)} schedule",
                    ],
                )
            ]
        rb.add(c.bullet, body, "\n")
    rb.add(c.pad(), "Reviewed with patient. ", c.facility())


def L_telegraphic_oneliner(c):
    rb, rng = c.rb, c.rng
    rb.add(c.hdr(["name"]), "; ", c.age_parts(), pick(rng, [" w/ ", " with ", " hx "]))
    for i in range(c.n(2, 5)):
        if i:
            rb.add(pick(rng, ["; ", ", ", " / ", ". "]))
        item = hist_neg(c) if (not c.positive or rng.random() < 0.2) else hist_pos(c)
        rb.add(item)
    rb.add(".", c.pad() or [" "], c.fill(1))


def L_nursing_intake_history(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng,
            [
                "NURSING ADMISSION ASSESSMENT",
                "Nursing intake",
                "RN admission note",
                "Admission nursing history",
            ],
        ),
        "\n",
    )
    rb.add(c.hdr(["name", "mrn", "dos"], sep=pick(rng, ["\n", " | "])), "\n")
    rb.add(
        pick(
            rng,
            [
                "History per patient: ",
                "PMH/PSH per pt: ",
                "Past history obtained from patient and chart: ",
            ],
        )
    )
    for _i in range(c.n(2, 5)):
        item = (
            hist_neg(c, True)
            if (not c.positive or rng.random() < 0.2)
            else hist_pos(c, True)
        )
        rb.add(item, " ")
    rb.add(c.pad(), "\nAssessed by ", c.clinician(), ". ", c.fill(1))


def L_portal_message_history(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng,
            [
                "Portal message",
                "MyChart-style message",
                "Patient portal message",
                "Message to care team",
            ],
        ),
        "\n",
    )
    rb.add("To: ", c.clinician(), "\n")
    rb.add(pick(rng, ["Hello, ", "Hi, ", "Good morning, ", "To whom it may concern, "]))
    rb.add(
        "I am ", c.age_years(), " and trying to find out what screening I still need. "
    )
    for _i in range(c.n(2, 4)):
        item = (
            hist_neg(c, True)
            if (not c.positive or rng.random() < 0.2)
            else hist_pos(c, True)
        )
        rb.add(item, " ")
    rb.add(
        c.pad(),
        pick(
            rng,
            [
                "Can you tell me what is due? ",
                "Do I need anything else this year? ",
                "Please advise. ",
            ],
        ),
        "\n",
        c.sign(),
    )
    if rng.random() < 0.3:
        rb.add("\n", c.email())
    elif rng.random() < 0.3:
        rb.add("\nPortal: ", c.url())


def L_referral_letter(c):
    rb, rng = c.rb, c.rng
    rb.add("Dear ", c.clinician(), ",\n\n")
    if c.unusual:
        rb.add(
            "Thank you for seeing the patient, age ",
            c.age_plain(),
            ". ",
            c.hdr(["name"]),
            "\n",
        )
    else:
        rb.add(
            "Thank you for seeing ", ("NAME", c.patient), ", age ", c.age_plain(), ". "
        )
    rb.add(pick(rng, ["Pertinent history: ", "Relevant history: ", "Background: "]))
    for _i in range(c.n(2, 5)):
        item = (
            hist_neg(c, True)
            if (not c.positive or rng.random() < 0.2)
            else hist_pos(c, True)
        )
        rb.add(item, " ")
    rb.add(c.pad(), "\n\nSincerely,\n", c.clinician(), "\n", c.facility())


# ---------------------------------------------------------------------------
# Pharmacy and ZIP
# ---------------------------------------------------------------------------
PH_LABELS_RX = [
    "PHARMACY: ",
    "Preferred pharmacy: ",
    "Pharmacy on file: ",
    "Dispensing pharmacy - ",
    "Rx sent to: ",
    "Send refills to: ",
    "Pharm: ",
    "Pharmacy of choice: ",
    "Local pharmacy: ",
    "Default pharmacy: ",
    "Retail pharmacy: ",
    "Pharmacy for refills: ",
    "Prescriptions go to ",
    "Usual drugstore: ",
    "Fill at ",
    "Outpatient pharmacy: ",
    "Pt's pharmacy: ",
    "Meds filled at ",
    "e-prescribe to ",
    "Preferred drugstore = ",
    "Pharmacy (retail): ",
    "PHARMACY - ",
]
ZIP_JOINS = [
    ("{n}, {z}", ", "),
    ("{n} - {z}", " - "),
    ("{n} {z}", " "),
    ("{n}; {z}", "; "),
    ("{n} ({z})", None),
    ("{n}, ZIP {z}", None),
    ("{n}, zip code {z}", None),
    ("{n} / {z}", " / "),
    ("{n}, {z}", ", "),
    ("{n}, {z}", ", "),
]


def pharm_name(c):
    rng = c.rng
    if rng.random() < 0.12:
        return c.reg.pick(rng, lambda r: f"{r.choice(LASTS)}'s Pharmacy", "pharmacy")
    return c.reg.pick(
        rng, lambda r: f"{r.choice(STEMS)} {r.choice(PHARM_TYPES)}", "pharmacy"
    )


def pharm_tail(c):
    rng = c.rng
    roll = rng.random()
    if roll < 0.18:
        return [pick(rng, [", ph ", ", phone ", " tel "]), c.phone()]
    if roll < 0.26:
        return [", fax ", c.phone()]
    return []


def pharm_line(c, positive=None, label=None):
    rng = c.rng
    positive = c.positive if positive is None else positive
    label = label if label is not None else pick(rng, PH_LABELS_RX)
    name = ("FACILITY", pharm_name(c), "pharmacy")
    if positive:
        zval = c.zip_code()
        if c.newcase == "zip_plus_four":
            zval = f"{zval}-{rng.randint(1000, 9999)}"
            c.hit = True
        z = ("LOCATION", zval, "zip")
        form = pick(
            rng,
            [
                "comma",
                "comma",
                "comma",
                "dash",
                "space",
                "semi",
                "paren",
                "zipword",
                "zipcode",
                "slash",
            ],
        )
        join = {
            "comma": [", ", z],
            "dash": [" - ", z],
            "space": [" ", z],
            "semi": ["; ", z],
            "paren": [" (", z, ")"],
            "zipword": [", ZIP ", z],
            "zipcode": [", zip code ", z],
            "slash": [" / ", z],
        }[form]
        return [label, name, join, pharm_tail(c)]
    kind = rng.randrange(10)
    n5 = "".join(rng.choice("0123456789") for _ in range(5))
    if c.newcase == "zip_plus_four":
        c.neghit = True
        code = rng.choice(
            [
                f", lot {n5}-{rng.randint(1000, 9999)}",
                f", NDC {n5}-{rng.randint(1000, 9999)}",
                f", batch {n5}-{rng.randint(100, 999)}",
            ]
        )
        return [label, name, [code]]
    tail = {
        0: [],
        1: [],
        2: pharm_tail(c) or [", open until 9 pm"],
        3: [", ", c.address()],
        4: [f", NDC {n5}-{rng.randint(1000, 9999)}-{rng.randint(10, 99)}"],
        5: [f", lot {n5}"],
        6: [f", {rng.choice([30, 60, 90, 120])} tablets"],
        7: [f", #{rng.choice([30, 60, 90])}, {rng.randint(0, 5)} refills"],
        8: [f", {rng.choice([30, 90])} day supply, DAW {rng.randint(0, 2)}"],
        9: [f", dose count {rng.randint(10, 200)}, batch {n5}"],
    }[kind]
    return [label, name, tail]


def pharm_neg_nameless(c):
    rng = c.rng
    return [
        pick(
            rng,
            [
                "Pharmacy: none on file",
                "Pharmacy: not yet chosen",
                "Preferred pharmacy: declined to list one",
                "Rx sent to: patient will pick a pharmacy",
                "Pharmacy: mail order, details pending",
            ],
        )
    ]


def rx_line(c):
    rng = c.rng
    med = rng.choice(MEDS)
    if c.positive:
        n = "".join(rng.choice("0123456789") for _ in range(rng.randint(6, 8)))
        rxid = ("ID", n, "rx")
        alt = ("ID", "RX-" + "".join(rng.choice("0123456789") for _ in range(5)), "rx")
        c.hit = True
        return rng.choice(
            [
                ["Rx #", rxid, " refilled"],
                ["Rx# ", rxid, " - ", med],
                ["Rx no. ", rxid, " renewed"],
                ["Rx number: ", rxid],
                ["prescription number ", alt, " transferred"],
                ["Rx ID ", rxid, " ready for pickup"],
                ["Rx #", rxid, ", ", med],
                ["Refill of Rx #", rxid, " requested"],
                ["prescription number ", rxid, " approved"],
            ]
        )
    c.neghit = True
    return [
        rng.choice(
            [
                f"Rx for {med}",
                "Rx sent electronically",
                f"{med} #{rng.choice([30, 60, 90])}, {rng.randint(0, 5)} refills",
                f"Rx refills: {rng.randint(0, 5)}",
                "prescription renewed for 90 days",
                f"Rx quantity #{rng.choice([30, 60, 90])}",
                "Rx pending prior authorization",
                "new Rx written today",
            ]
        )
    ]


def mailing_zip_line(c):
    rng = c.rng
    label = rng.choice(["Mailing ZIP: ", "ZIP+4: ", "Billing ZIP: ", "Mail ZIP "])
    if c.positive:
        c.hit = True
        z = f"{c.zip_code()}-{rng.randint(1000, 9999)}"
        return [label, ("LOCATION", z, "zip")]
    return [label + rng.choice(["not provided", "same as home", "unknown"])]


def pharm_item(c, allow_nameless=True):
    if c.newcase == "rx_number":
        return [rx_line(c), "\n", pharm_line(c)]
    if c.newcase == "zip_plus_four" and c.rng.random() < 0.3:
        return [mailing_zip_line(c), "\n", pharm_line(c)]
    if (
        allow_nameless
        and c.newcase is None
        and not c.positive
        and c.rng.random() < 0.12
    ):
        return pharm_neg_nameless(c)
    return pharm_line(c)


def L_intake_form(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng,
            [
                "PATIENT INTAKE FORM",
                "New patient intake",
                "Intake questionnaire",
                "INTAKE - DEMOGRAPHICS AND PHARMACY",
            ],
        ),
        "\n",
    )
    fields = [
        ["Name: ", ("NAME", c.patient)],
        ["DOB: ", c.full_date(c.birth_year)[:1]],
        ["Address: ", c.address()],
        ["Phone: ", c.phone()],
        ["Email: ", c.email()],
        ["Insurance member ID: ", c.insurance_id()],
        [
            "Allergies: ",
            pick(rng, ["penicillin", "none known", "sulfa", "latex", "shellfish"]),
        ],
        ["Emergency contact: ", ("NAME", c.person()), ", ", c.phone()],
    ]
    if rng.random() < 0.08:
        fields.append(["Account #: ", c.account()])
    rng.shuffle(fields)
    chosen = fields[: rng.randint(3, 6)]
    if all(f[0] != "Name: " for f in chosen):
        chosen[0] = fields[[f[0] for f in fields].index("Name: ")]
    for field in chosen:
        rb.add(field, "\n")
    for k in range(c.n(1, 2)):
        rb.add(pharm_item(c, allow_nameless=(k == 0)), "\n")
    rb.add(c.pad(), c.fill(1))


def L_registration_form(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng,
            [
                "REGISTRATION",
                "Patient registration",
                "REGISTRATION FORM - FRONT DESK",
                "Registration / demographics",
            ],
        ),
        "\n",
    )
    rb.add(c.hdr(["name", "dob", "mrn"], sep="\n"), "\n")
    rb.add("Guarantor account: ", c.account(), "\n")
    if rng.random() < 0.1:
        rb.add("SSN: ", c.ssn(), "\n")
    rb.add("Contact: ", c.phone(), "\n")
    if rng.random() < 0.5:
        rb.add("Responsible party: ", c.relative(), "\n")
    rb.add(
        pick(rng, ["PHARMACY INFORMATION", "Pharmacy", "Preferred pharmacy section"]),
        "\n",
    )
    for k in range(c.n(1, 2)):
        rb.add(pharm_item(c, allow_nameless=(k == 0)), "\n")
    rb.add(c.pad(), "Registered at ", c.facility())


def L_refill_request(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng,
            [
                "TELEPHONE ENCOUNTER - refill request",
                "Refill request",
                "Rx refill note",
                "Medication refill call",
            ],
        ),
        "\n",
    )
    rb.add(c.hdr(["name", "dos"], sep=" | "), "\n")
    rb.add("Requested: ", ", ".join(rng.sample(MEDS, rng.randint(1, 3))), ".\n")
    for k in range(c.n(1, 2)):
        rb.add(pharm_item(c, allow_nameless=(k == 0)), "\n")
    rb.add(c.pad(), "Routed to ", c.clinician(), " for approval. ", c.fill(1))


def L_portal_message_pharmacy(c):
    rb, rng = c.rb, c.rng
    rb.add("Portal message\nTo: ", c.clinician(), "\n")
    rb.add(pick(rng, ["Hello, ", "Hi there, ", "Good afternoon, "]))
    rb.add(
        pick(
            rng,
            [
                "I would like to change where my prescriptions are sent. ",
                "Please send my refills to a different store. ",
                "Can you update my pharmacy? ",
            ],
        )
    )
    rb.add("I am ", c.age_years(), " and take ", rng.choice(MEDS), ". ")
    if c.newcase == "rx_number":
        rb.add(rx_line(c), ". ")
    if c.positive:
        rb.add(
            pick(
                rng,
                [
                    "Please use ",
                    "The new one is ",
                    "Send everything to ",
                    "My pharmacy is now ",
                ],
            )
        )
        rb.add(pharm_line(c, True, label=""), ". ")
    else:
        if rng.random() < 0.5:
            rb.add(
                pharm_line(
                    c,
                    False,
                    label=pick(
                        rng, ["Please use ", "The new one is ", "Send everything to "]
                    ),
                ),
                ". ",
            )
        else:
            rb.add(
                pick(
                    rng,
                    [
                        "I will call the pharmacy with the details. ",
                        "The pharmacy near my daughter's house is easier. ",
                        "I have not picked a new one yet. ",
                    ],
                )
            )
    if rng.random() < 0.3:
        rb.add("Link: ", c.url(), ". ")
    rb.add(c.pad(), "\n", c.sign())


def L_discharge_med_list(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng,
            [
                "DISCHARGE MEDICATIONS",
                "Discharge medication list",
                "Medications at discharge",
                "TRANSITION OF CARE - MEDS",
            ],
        ),
        "\n",
    )
    rb.add(c.hdr(["name", "mrn", "dos"], sep=" | "), "\n")
    for med in rng.sample(MEDS, rng.randint(3, 6)):
        rb.add(c.bullet, med, "\n")
    for k in range(c.n(1, 2)):
        rb.add(pharm_item(c, allow_nameless=(k == 0)), "\n")
    rb.add(c.pad(), c.fill(1), " ", c.clinician())


def L_pharmacy_oneliner(c):
    rb, rng = c.rb, c.rng
    rb.add(c.hdr(["name"]), ", ", c.age_parts(), "; meds reviewed; ")
    rb.add(pharm_item(c))
    rb.add(pick(rng, [". ", "; ", ".\n"]), rng.choice(MEDS), ". ", c.pad())


# ---------------------------------------------------------------------------
# Phone with written extension
# ---------------------------------------------------------------------------
def ph_pos_line(c):
    rng = c.rng
    label, phone = pick(rng, PH_LABELS), c.phone(ext=True)
    form = rng.randrange(12)
    if form == 0:
        return [label, ": ", phone]
    if form == 1:
        return [label, " - ", phone]
    if form == 2:
        return [label, " at ", phone]
    if form == 3:
        return [label, " (", phone, ")"]
    if form == 4:
        return ["call ", label.lower(), " ", phone]
    if form == 5:
        return [
            pick(
                rng, ["Ph: ", "Tel ", "Phone ", "Callback # ", "Reach at ", "Direct: "]
            ),
            phone,
        ]
    if form == 6:
        return ["Please call ", label.lower(), " at ", phone, " to confirm."]
    if form == 7:
        return ["Left voicemail; callback number ", phone, "."]
    if form == 8:
        return ["Family requests a call back at ", phone, "."]
    if form == 9:
        return ["Dial ", phone, " and ask for ", c.clinician(), "."]
    if form == 10:
        return ["Questions? ", phone]
    return [label, ", ", phone, pick(rng, [" (weekdays)", "", " after 8 am", ""])]


def ph_neg_line(c):
    rng = c.rng
    roll = rng.random()
    if roll < 0.45:
        return [
            rng.choice(PH_NEG).capitalize()
            if rng.random() < 0.5
            else rng.choice(PH_NEG)
        ]
    if roll < 0.65:
        return [pick(rng, ROOMS), "; ", rng.choice(PH_NEG)]
    if roll < 0.85:
        # a plain phone with no extension, followed by extension words that are not part of it
        return [
            pick(rng, PH_LABELS),
            ": ",
            c.phone(),
            ". ",
            rng.choice(PH_NEG).capitalize(),
        ]
    return [
        pick(rng, ["Call", "Ring", "Contact"]),
        " ",
        pick(rng, PH_LABELS).lower(),
        " at ",
        c.phone(),
        " for questions; ",
        pick(rng, ROOMS),
    ]


def pager_line(c):
    rng = c.rng
    label = rng.choice(["Pager", "pager", "Pager #", "Beeper", "On-call pager"])
    if c.positive:
        number = ("PHONE", c.phone()[1], "pager")
        c.hit = True
        return rng.choice(
            [
                [label, ": ", number],
                [label, " ", number],
                ["Page ", c.clinician(), " at pager ", number],
                ["paged via ", number, " with no reply"],
                ["Night pager ", number, ", carried until 7 am"],
                ["Resident pager: ", number],
                ["Unit pager (", number, ")"],
            ]
        )
    c.neghit = True
    return [
        rng.choice(
            [
                "Paged the covering physician; no answer yet",
                "Pager battery replaced at the desk",
                "Page the resident if BP drops below 90",
                "Pager not carried on nights",
                "Hold the pager until 7 am",
                "Paged overhead twice for the transport team",
            ]
        )
    ]


def bare_ext_line(c):
    rng = c.rng
    if rng.random() < 0.2:
        return ph_neg_line(c)
    n = "".join(rng.choice("0123456789") for _ in range(rng.choice([3, 4, 4, 5])))
    n = "5" + n[1:] if n[0] == "0" else n
    c.neghit = True
    return rng.choice(
        [
            [f"Ward phone ext. {n}"],
            [f"Charge desk x{n}"],
            [f"Dial extension {n} from any unit phone"],
            [f"Reach the clerk at ext {n}"],
            [f"Nurses' station ext. {n}"],
            [f"Pharmacy line EXT. {n}"],
            [f"ext #{n} for transport"],
            [f"Internal line extension {n}"],
        ]
    )


def ph_item(c):
    if c.newcase == "pager":
        return pager_line(c)
    if c.newcase == "bare_extension":
        return bare_ext_line(c)
    return ph_pos_line(c) if c.positive else ph_neg_line(c)


def ph_lines(c, count, join="\n", prefix=None):
    out = []
    for i in range(count):
        if i:
            out.append(join)
        out.append(prefix if prefix is not None else c.bullet)
        item = flat(ph_item(c))
        out.append(item)
        if join != "\n" and not (
            isinstance(item[-1], str) and item[-1].endswith((".", "?"))
        ):
            out.append(".")
    return out


def L_contact_block(c):
    rb, rng = c.rb, c.rng
    rb.add(c.hdr(["name", "mrn"], sep=" | "), "\n")
    rb.add(
        pick(
            rng,
            [
                "CONTACTS:",
                "Contact numbers",
                "KEY CONTACTS",
                "Phone directory for this stay",
            ],
        ),
        "\n",
    )
    rb.add(ph_lines(c, c.n(2, 5)))
    rb.add("\nFamily: ", c.relative(), " ", c.phone(), "\n", c.pad(), c.fill(1))


def L_fax_cover(c):
    rb, rng = c.rb, c.rng
    rb.add(pick(rng, ["FAX COVER SHEET", "Fax cover", "FAX TRANSMITTAL"]), "\n")
    rb.add("To: ", c.clinician(), pick(rng, ["  ", " | ", "\n"]), c.facility(), "\n")
    rb.add("From: ", c.clinician(), "\n")
    rb.add("Fax: ", c.phone(), "\n")
    rb.add("Re: ", c.hdr(["name", "mrn"], sep=" | "), "\n")
    rb.add(ph_lines(c, c.n(1, 3), prefix=""), "\n")
    rb.add(
        c.pad(),
        pick(rng, ["Pages: 3. ", "Please confirm receipt. ", "Records attached. "]),
        c.fill(1),
    )


def L_callback_note(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng, ["TELEPHONE ENCOUNTER", "Callback note", "Phone message", "Call log"]
        ),
        "\n",
    )
    rb.add(c.hdr(["name", "dos"], sep=" | "), "\n")
    rb.add(
        pick(rng, ["Caller: ", "Spoke with: ", "Message from: "]), c.relative(), ". "
    )
    rb.add(ph_lines(c, c.n(1, 3), join=" ", prefix=""), " ")
    rb.add(c.pad(), c.fill(1), " Routed to ", c.clinician(), ".")


def L_shift_handoff(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(rng, ["SHIFT HANDOFF", "SBAR handoff", "Handoff note", "Nursing handoff"]),
        "\n",
    )
    rb.add(c.hdr(["name"]), ", ", c.age_parts(), pick(rng, ["; ", "\n"]))
    rb.add(c.fill(2), "\n")
    rb.add(ph_lines(c, c.n(2, 4)), "\n")
    rb.add(
        c.town_line() if rng.random() < 0.25 else "",
        c.pad(),
        "Report given to ",
        c.clinician(),
    )


def L_nursing_note_family(c):
    rb, rng = c.rb, c.rng
    rb.add(pick(rng, ["Nursing note", "RN progress note", "Nursing narrative"]), "\n")
    rb.add(c.hdr(["name", "dos"], sep=" | "), "\n")
    rb.add(c.fill(2), " ")
    rb.add(pick(rng, ["Spoke with ", "Updated ", "Called "]), c.relative(), ". ")
    rb.add(
        ph_lines(c, c.n(1, 3), join=" ", prefix=""),
        " ",
        c.town_line() if rng.random() < 0.3 else "",
        c.pad(),
        c.fill(1),
    )


def L_followup_instructions(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng,
            [
                "FOLLOW-UP INSTRUCTIONS",
                "After-visit instructions",
                "Discharge follow-up",
                "NEXT STEPS",
            ],
        ),
        "\n",
    )
    rb.add(c.hdr(["name", "dos"], sep=" | "), "\n")
    rb.add(c.fill(2), " ", c.town_line() if rng.random() < 0.25 else "", "\n")
    rb.add(
        pick(
            rng,
            [
                "To schedule or change an appointment:",
                "Questions after you go home:",
                "Who to call:",
            ],
        ),
        "\n",
    )
    rb.add(ph_lines(c, c.n(2, 4)), "\n")
    rb.add(c.pad(), "Seen at ", c.facility(), ". ", c.clinician())


# ---------------------------------------------------------------------------
# Funeral homes
# ---------------------------------------------------------------------------
def funeral_name(c):
    rng = c.rng

    def make(r):
        if r.random() < 0.3:
            return f"{r.choice(LASTS)} {r.choice(FUNERAL_SURNAME_TYPES)}"
        return f"{r.choice(STEMS)} {r.choice(FUNERAL_TYPES)}"

    return ("FACILITY", c.reg.pick(rng, make, "funeral"), "funeral")


def hospice_name(c):
    return (
        "FACILITY",
        c.reg.pick(
            c.rng,
            lambda r: r.choice(HOSPICE_FORMS).format(s=r.choice(STEMS)),
            "hospice",
        ),
        "facility",
    )


def fh_pos_line(c):
    rng = c.rng
    f = funeral_name(c)
    form = rng.randrange(12)
    time = pick(rng, ["0930", "1415", "2210", "0645", "1130", "1845"])
    if form == 0:
        return [f, " notified at ", time, "."]
    if form == 1:
        return ["Family chose ", f, "."]
    if form == 2:
        return [f, " contacted for transport."]
    if form == 3:
        return ["Body released to ", f, " at ", time, "."]
    if form == 4:
        return ["Funeral arrangements are being handled by ", f, "."]
    if form == 5:
        return ["Family has contacted ", f, " regarding arrangements."]
    if form == 6:
        return ["Pickup by ", f, " completed at ", time, "."]
    if form == 7:
        return ["Per ", c.relative(), ", ", f, " will make the arrangements."]
    if form == 8:
        return ["Mortuary: ", f, "."]
    if form == 9:
        return ["Funeral home: ", f, "; family aware."]
    if form == 10:
        return [f, " staff arrived at ", time, " to receive the body."]
    return ["The family selected ", f, " and signed the release."]


def fh_neg_line(c):
    return [
        pick(
            c.rng,
            [
                "The funeral home will call the family.",
                "Funeral planning discussed with family.",
                "Family has not yet chosen a funeral home.",
                "Funeral home to be selected by the family.",
                "Mortuary transport arranged per family request.",
                "Family will contact a funeral home in the morning.",
                "Arrangements per family; no preference stated.",
                "Funeral home notified by the unit clerk.",
                "Cremation versus burial discussed; family undecided.",
                "The chosen funeral home was called.",
                "Social work provided a list of funeral services.",
                "Family asked about funeral costs and pre-planning.",
                "Funeral director to be contacted after family meeting.",
                "Body held pending release to the funeral home.",
            ],
        )
    ]


def fh_item(c):
    return fh_pos_line(c) if c.positive else fh_neg_line(c)


def fh_lines(c, count, join="\n"):
    out = []
    for i in range(count):
        if i:
            out.append(join)
        out.append(fh_item(c))
    return out


def death_facts(c, time_style=True):
    rng = c.rng
    return [
        "Patient pronounced dead at ",
        pick(rng, ["0412", "1530", "2250", "0905", "1740"]),
        " on ",
        c.full_date(c.note_year)[:1],
        " by ",
        c.clinician(),
        ". ",
    ]


def L_death_note(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(rng, ["DEATH NOTE", "Expiration note", "Death summary", "NOTE OF DEATH"]),
        "\n",
    )
    rb.add(c.hdr(["name", "mrn"], sep=" | "), "\n")
    rb.add(c.age_parts(), " patient with end-stage illness. ", death_facts(c), "\n")
    rb.add(
        "Next of kin: ",
        c.relative(),
        ", ",
        c.phone(),
        ". Autopsy ",
        pick(rng, ["declined", "not requested", "discussed and declined"]),
        ".\n",
    )
    rb.add(
        fh_lines(c, c.n(1, 2)),
        "\n",
        c.town_line() if rng.random() < 0.35 else "",
        c.pad(),
        c.fill(1),
    )


def L_hospice_discharge(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng,
            [
                "HOSPICE DISCHARGE SUMMARY",
                "Hospice discharge - death",
                "Hospice record of death",
                "HOSPICE NOTE",
            ],
        ),
        "\n",
    )
    rb.add(c.hdr(["name", "dos"], sep=" | "), "\n")
    rb.add(
        "Enrolled with ",
        hospice_name(c),
        ". ",
        c.age_parts(),
        " at time of death. ",
        death_facts(c),
        "\n",
    )
    rb.add("Bereavement packet given to ", c.relative(), ". ")
    rb.add(
        fh_lines(c, c.n(1, 2), join=" "),
        "\n",
        c.town_line() if rng.random() < 0.3 else "",
        c.pad(),
        c.fill(1),
    )


def L_social_work_bereavement(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng,
            [
                "SOCIAL WORK NOTE",
                "Social work - bereavement",
                "SW follow-up",
                "Chaplain and social work note",
            ],
        ),
        "\n",
    )
    rb.add(c.hdr(["name"]), "\n")
    rb.add(
        "Met with ",
        c.relative(),
        " and ",
        c.relative(),
        " at the bedside. ",
        c.fill(1),
        " ",
        c.town_line() if rng.random() < 0.6 else "",
    )
    rb.add(fh_lines(c, c.n(1, 3), join=" "), "\n", c.pad(), "Contact: ", c.phone())


def L_nursing_expiration(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng,
            [
                "Nursing note - expiration",
                "RN note",
                "Nursing narrative: death",
                "End-of-life nursing note",
            ],
        ),
        "\n",
    )
    rb.add(c.hdr(["name", "dos"], sep=" | "), "\n")
    rb.add(
        "Found without pulse or respirations; ",
        c.clinician(),
        " notified and pronounced. ",
        c.relative(),
        " at bedside. ",
    )
    rb.add(
        fh_lines(c, c.n(1, 2), join=" "),
        " ",
        c.pad(),
        c.fill(1),
        "\nUnit: ",
        c.facility(),
    )


# ---------------------------------------------------------------------------
# v3: weekday before a date in signature and attestation lines
# ---------------------------------------------------------------------------
CREDS = ["MD", "RN", "NP", "PA-C", "DO", "LPN", "PharmD", "RD", "LCSW", "PT", "OT"]


def sig_who(c):
    rng = c.rng
    name = c.person()
    if rng.random() < 0.25:
        return ["Dr. ", ("NAME", name)]
    return [("NAME", name), ", ", rng.choice(CREDS)]


def sig_clock(rng):
    h, m = rng.randint(0, 23), rng.randint(0, 59)
    h12 = h % 12 or 12
    suffix = "AM" if h < 12 else "PM"
    return rng.choice(
        [
            f"{h:02d}:{m:02d}",
            f"{h:02d}{m:02d}",
            f"{h12}:{m:02d} {suffix}",
            f"{h:02d}:{m:02d} hrs",
            f"{h12}:{m:02d}{suffix.lower()}",
            f"{h:02d}:{m:02d}:{rng.randint(0, 59):02d}",
        ]
    )


def sig_date(c):
    """A weekday and a date, one DATE span; the weekday matches the calendar."""
    rng = c.rng
    year = c.note_year
    month, day = rng.randint(1, 12), rng.randint(1, 28)
    if year >= 2026:
        month = rng.randint(1, 9)
    index = date(year, month, day).weekday()
    full, ab = WEEKDAYS[index], WEEKDAYS[index][:3]
    text = rng.choice(
        [
            f"{ab}, {day} {MON3[month - 1]} {year}",
            f"{full}, {MONFULL[month - 1]} {day}, {year}",
            f"{ab} {month:02d}/{day:02d}/{year}",
            f"{ab}, {month}/{day}/{year % 100:02d}",
            f"{full} {MON3[month - 1]} {day}, {year}",
            f"{ab}, {year}-{month:02d}-{day:02d}",
            f"{ab}. {month}/{day}/{year}",
            f"{full}, {day} {MONFULL[month - 1]} {year}",
            f"{ab}, {MON3[month - 1]} {day}, {year}",
        ]
    )
    return ("DATE", text, "sigdate")


def sig_line(c, positive):
    rng = c.rng
    who, t = sig_who(c), sig_clock(rng)
    if positive:
        d = sig_date(c)
        return rng.choice(
            [
                ["Note completed by ", who, ", ", d, " at ", t],
                ["Electronically signed by ", who, " on ", d, " at ", t],
                ["Attested by ", who, ", ", d, ", ", t],
                ["Signed: ", who, " | ", d, " ", t],
                ["Reviewed and cosigned by ", who, " ", d, " at ", t, "."],
                ["Documentation completed ", d, " ", t, " by ", who],
                ["Addendum entered by ", who, " on ", d, " at ", t],
                ["Dictated by ", who, ", ", d, " (", t, ")"],
                ["Authenticated ", d, " ", t, " - ", who],
                ["Chart closed by ", who, ", ", d, " at ", t],
            ]
        )
    elapsed = (
        f"{rng.randint(0, 1):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"
    )
    return rng.choice(
        [
            ["Note completed by ", who, " at ", t],
            ["Electronically signed by ", who, " at ", t],
            ["Attested by ", who, ", ", t],
            [f"[{elapsed}] Note completed by ", who],
            ["Signed: ", who, " | ", t],
            ["Pending signature: ", who],
            ["Addendum entered by ", who, " at ", t],
            ["Reviewed and cosigned by ", who, " (time ", t, ")"],
            ["Signature pending; ", who, " notified"],
            ["Dictated by ", who, " at ", t, " today"],
        ]
    )


def sig_lines(c, low, high):
    rng, out = c.rng, []
    for i in range(rng.randint(low, high)):
        if i:
            out.append("\n")
        out.append(sig_line(c, c.positive and (i == 0 or rng.random() < 0.6)))
    return out


def L_sig_progress_note(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng,
            [
                "PROGRESS NOTE",
                "Clinic progress note",
                "Follow-up visit note",
                "Office visit",
            ],
        ),
        "\n",
    )
    rb.add(
        c.hdr(
            ["name", "mrn", "dos"][: rng.randint(2, 3)], sep=pick(rng, ["\n", " | "])
        ),
        "\n\n",
    )
    rb.add(c.age_parts(), " patient seen for follow-up. ", c.fill(rng.randint(2, 3)))
    rb.add("\n\n", sig_lines(c, 1, 2))


def L_sig_attestation(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(rng, ["RESIDENT NOTE", "Teaching service note", "Fellow consult note"]),
        "\n",
    )
    rb.add(c.hdr(["name", "mrn"], sep=" | "), "\n")
    rb.add(c.fill(2), "\n\n")
    rb.add(
        pick(
            rng,
            [
                "ATTENDING ATTESTATION: I saw and examined the patient and agree with the plan.",
                "Attending attestation: patient seen and discussed with the resident.",
                "Attestation - I reviewed the findings and agree as documented.",
            ],
        ),
        "\n",
        sig_lines(c, 1, 2),
    )


def L_sig_telephone_encounter(c):
    rb, rng = c.rb, c.rng
    rb.add(pick(rng, ["TELEPHONE ENCOUNTER", "Phone call note", "Call note"]), "\n")
    rb.add(c.hdr(["name", "dos"], sep=" | "), "\n")
    rb.add(
        pick(rng, ["Spoke with ", "Returned call to ", "Message left for "]),
        c.relative(),
        ". ",
        c.fill(2),
        "\n",
        sig_lines(c, 1, 1),
    )


def L_sig_discharge_summary(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng, ["DISCHARGE SUMMARY", "Discharge summary", "Hospital course summary"]
        ),
        "\n",
    )
    rb.add(c.hdr(["name", "mrn", "dos"], sep="\n"), "\n\n")
    rb.add(c.age_parts(), " patient. ", c.fill(2), "\nDischarge medications:\n")
    for med in rng.sample(MEDS, rng.randint(2, 4)):
        rb.add(c.bullet, med, "\n")
    rb.add("\n", sig_lines(c, 1, 2))


def L_sig_nursing_note(c):
    rb, rng = c.rb, c.rng
    rb.add(pick(rng, ["Nursing note", "RN shift note", "Nursing narrative"]), "\n")
    rb.add(c.hdr(["name", "dos"], sep=" | "), "\n")
    rb.add(c.fill(2), " Report given to ", c.clinician(), ".\n", sig_lines(c, 1, 2))


def L_sig_addendum(c):
    rb, rng = c.rb, c.rng
    rb.add(pick(rng, ["ADDENDUM", "Late entry addendum", "Chart addendum"]), "\n")
    rb.add(c.hdr(["name", "mrn"], sep=" | "), "\n")
    rb.add(c.fill(1), "\n", sig_lines(c, 2, 3))


# ---------------------------------------------------------------------------
# v3: rare identifier formats (device serials, session/visit/call ids, access codes)
# ---------------------------------------------------------------------------
DEVICES = [
    "pacemaker",
    "insulin pump",
    "CPAP machine",
    "glucose meter",
    "hearing aid",
    "continuous glucose monitor",
    "nerve stimulator",
    "oxygen concentrator",
    "defibrillator",
    "infusion pump",
    "nebulizer",
    "cardiac event monitor",
    "home ventilator",
    "bone growth stimulator",
    "cochlear processor",
    "blood pressure monitor",
]
ID_LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"
ID_ALNUM = "ABCDEFGHJKLMNPRSTUVWXYZ23456789"


def rdigits(rng, n):
    return "".join(rng.choice("0123456789") for _ in range(n))


def rletters(rng, n, alphabet=ID_LETTERS):
    return "".join(rng.choice(alphabet) for _ in range(n))


def serial_value(rng):
    return rng.choice(
        [
            f"SN{rdigits(rng, 8)}",
            f"{rletters(rng, 2)}{rdigits(rng, 7)}",
            f"{rdigits(rng, 4)}-{rdigits(rng, 5)}",
            f"{rletters(rng, 3)}-{rdigits(rng, 6)}",
            f"{rdigits(rng, 2)}{rletters(rng, 2)}{rdigits(rng, 6)}",
            f"{rletters(rng, 1)}{rdigits(rng, 3)}-{rletters(rng, 2)}{rdigits(rng, 4)}",
            f"{rletters(rng, 4, ID_ALNUM)}{rdigits(rng, 6)}",
        ]
    )


def session_value(rng):
    return rng.choice(
        [
            f"{rletters(rng, 2)}-{rdigits(rng, rng.randint(4, 6))}",
            f"{rdigits(rng, 3)}-{rdigits(rng, 3)}-{rdigits(rng, 3)}",
            f"{rletters(rng, 1)}{rdigits(rng, 7)}",
            f"S-{rdigits(rng, 6)}",
            f"{rdigits(rng, 4)}-{rdigits(rng, 4)}",
            f"{rletters(rng, 3)}{rdigits(rng, 5)}",
        ]
    )


def call_value(rng):
    return rng.choice(
        [
            f"CALL-{rdigits(rng, 6)}",
            f"C{rdigits(rng, 8)}",
            f"{rdigits(rng, 5)}-{rdigits(rng, 3)}",
            f"NL{rdigits(rng, 7)}",
            f"{rletters(rng, 2)}{rdigits(rng, 6)}",
        ]
    )


def activation_value(rng):
    return rng.choice(
        [
            f"{rletters(rng, 4, ID_ALNUM)}-{rletters(rng, 4, ID_ALNUM)}",
            rdigits(rng, 6),
            f"ACT{rdigits(rng, 6)}",
            rletters(rng, 8, ID_ALNUM),
            f"{rletters(rng, 3, ID_ALNUM)}-{rdigits(rng, 3)}-{rletters(rng, 3, ID_ALNUM)}",
            rdigits(rng, 8),
        ]
    )


def meeting_link(rng):
    host = rng.choice(
        [
            "visit.example.test",
            "telehealth.example.org",
            "meet.example.net",
            "room.clinic.test",
            "video.example.com",
        ]
    )
    return (
        "URL",
        f"https://{host}/{rng.choice(['r', 'join', 'room', 'v'])}/{rletters(rng, 3, ID_ALNUM)}{rdigits(rng, rng.randint(3, 5))}",
    )


def serial_item(c):
    rng = c.rng
    dev = rng.choice(DEVICES)
    sid = ("ID", serial_value(rng), "serial")
    return rng.choice(
        [
            ["Device: ", dev, ", serial number ", sid],
            [dev.capitalize(), " (S/N ", sid, ") in use at home"],
            ["Serial #", sid, " - ", dev],
            ["Implant card: ", dev, ", SN ", sid],
            [dev.capitalize(), " serial: ", sid, "; settings reviewed"],
            ["Returned ", dev, " for service, serial no. ", sid, "."],
            ["Registered ", dev, " under serial ", sid, "."],
            ["Device serial ", sid, " (", dev, ")"],
        ]
    )


def session_item(c):
    rng = c.rng
    sid = ("ID", session_value(rng), "session")
    link = meeting_link(rng)
    lang = rng.choice(["Spanish", "Mandarin", "Somali", "Hmong", "Arabic"])
    return rng.choice(
        [
            ["Video session ", sid, " joined from ", link],
            ["Telehealth visit ID: ", sid],
            ["Session ", sid, " ended after ", str(rng.randint(12, 55)), " minutes"],
            ["Interpreter session ", sid, " (", lang, ")"],
            ["Visit reference ", sid],
            ["Meeting ID ", sid, "; link ", link],
            ["Virtual visit session ID ", sid, " recorded"],
            ["Joined ", link, " as session ", sid],
        ]
    )


def call_item(c):
    rng = c.rng
    cid = ("ID", call_value(rng), "call")
    return rng.choice(
        [
            ["Call ID ", cid, " logged by the nurse line"],
            ["Call reference ", cid],
            ["Inbound call ", cid, ", triaged by phone"],
            ["Call ID: ", cid],
            ["Recorded call ", cid, " reviewed by the supervisor"],
        ]
    )


def activation_item(c):
    rng = c.rng
    aid = ("ID", activation_value(rng), "activation")
    hours = str(rng.choice([24, 48, 72]))
    return rng.choice(
        [
            ["Portal activation code ", aid, " issued to the patient"],
            ["One-time access code: ", aid],
            ["Enrollment code ", aid, " expires in ", hours, " hours"],
            ["Activation code for the home monitoring app: ", aid],
            ["Pairing code ", aid, " entered at the kiosk"],
            ["Sign-up code ", aid, " given at checkout"],
            ["Access code ", aid, " sent by text"],
            ["Invite code: ", aid],
        ]
    )


RARE_POS = {
    "serial": serial_item,
    "session": session_item,
    "call": call_item,
    "activation": activation_item,
}


def rare_neg_item(c):
    rng = c.rng
    ver = f"{rng.randint(1, 9)}.{rng.randint(0, 29)}.{rng.randint(0, 9)}"
    dev = rng.choice(DEVICES)
    n = rng.randint(2, 9)
    m = n + rng.randint(1, 4)
    return rng.choice(
        [
            [f"App version {ver} installed on the tablet"],
            [f"Firmware v{ver} current on the {dev}"],
            [f"Software build {ver} running; no updates pending"],
            [f"Session {n} of {m} completed"],
            [f"Visit {n} of {m} scheduled"],
            [f"Call lasted {rng.randint(3, 40)} minutes"],
            [f"Session length {rng.randint(15, 60)} minutes"],
            [f"Serial BP checks every {rng.choice([2, 4, 6])} hours"],
            [f"Serial troponins x{rng.randint(2, 4)} negative"],
            ["Activation code not yet issued"],
            ["No activation code received by the patient"],
            [f"{dev.capitalize()} battery at {rng.randint(10, 95)}%"],
            ["Device serial number to follow from the manufacturer"],
            ["Visit ID not available at the time of this note"],
            ["Video session cancelled; patient will call back"],
            ["Join link: ", meeting_link(rng)],
            ["Telehealth link sent: ", meeting_link(rng)],
        ]
    )


def rare_item(c, kinds, positive=None):
    positive = c.positive if positive is None else positive
    if not positive:
        return rare_neg_item(c)
    return RARE_POS[c.rng.choice(kinds)](c)


def rare_lines(c, kinds, low, high):
    rng, out = c.rng, []
    for i in range(rng.randint(low, high)):
        if i:
            out.append("\n")
        out.append(rare_item(c, kinds, c.positive and (i == 0 or rng.random() < 0.7)))
    if rng.random() < 0.3:
        out.append(
            f"\nApp version {rng.randint(1, 9)}.{rng.randint(0, 29)}.{rng.randint(0, 9)}."
        )
    return out


def L_device_record_note(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng,
            ["DEVICE RECORD", "Medical device summary", "Home and implanted devices"],
        ),
        "\n",
    )
    rb.add(c.hdr(["name", "mrn", "dos"][: rng.randint(2, 3)], sep=" | "), "\n")
    rb.add(rare_lines(c, ["serial"], 2, 3), "\n", c.fill(1), " ", c.clinician())


def L_telehealth_visit_note(c):
    rb, rng = c.rb, c.rng
    rb.add(pick(rng, ["TELEHEALTH VISIT", "Virtual visit note", "Video visit"]), "\n")
    rb.add(c.hdr(["name", "dos"], sep=" | "), "\n")
    rb.add(
        rare_lines(c, ["session"], 1, 2), "\n", c.fill(2), "\nProvider: ", c.clinician()
    )


def L_call_center_log(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(rng, ["NURSE LINE CALL LOG", "Triage call log", "Call center record"]),
        "\n",
    )
    rb.add(c.hdr(["name"], sep=" | "), "\n")
    rb.add("Caller: ", c.relative(), ", ", c.phone(), "\n")
    rb.add(rare_lines(c, ["call"], 1, 2), "\n", c.fill(2))


def L_portal_activation_message(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng,
            [
                "Patient portal message",
                "Portal welcome message",
                "Account setup message",
            ],
        ),
        "\nTo: ",
        ("NAME", c.patient),
        "\n",
    )
    rb.add(pick(rng, ["Hello, ", "Welcome, ", "Hi, "]))
    rb.add(
        pick(
            rng,
            [
                "your portal account is ready to be set up. ",
                "use the code below to finish setting up your account. ",
                "here is what you need to sign in for the first time. ",
            ],
        ),
        "\n",
        rare_lines(c, ["activation"], 1, 2),
        "\n",
        pick(rng, ["Need help? ", "Sign-in help: ", "Questions? Call "]),
        c.phone(),
        ". ",
        c.fill(1),
    )


def L_device_registration_form(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng,
            [
                "DEVICE REGISTRATION",
                "Device enrollment form",
                "Remote monitoring enrollment",
            ],
        ),
        "\n",
    )
    rb.add("Name: ", ("NAME", c.patient), "\n")
    if rng.random() < 0.6:
        rb.add("DOB: ", c.full_date(c.birth_year)[:1], "\n")
    rb.add("Phone: ", c.phone(), "\n")
    rb.add(rare_lines(c, ["serial", "activation"], 2, 3), "\n", c.fill(1))


def L_after_visit_summary(c):
    rb, rng = c.rb, c.rng
    rb.add(
        pick(
            rng, ["AFTER-VISIT SUMMARY", "Visit summary for the patient", "NEXT STEPS"]
        ),
        "\n",
    )
    rb.add(c.hdr(["name", "dos"], sep=" | "), "\n")
    rb.add(
        c.fill(2),
        "\n",
        rare_lines(c, ["session", "call", "serial", "activation"], 2, 3),
    )
    rb.add("\n", c.clinician())


LAYOUT_FUNCS = {
    "awv_note": L_awv_note,
    "soap_note": L_soap_note,
    "screening_schedule": L_screening_schedule,
    "surgical_history_list": L_surgical_history_list,
    "problem_list": L_problem_list,
    "immunization_record": L_immunization_record,
    "telegraphic_oneliner": L_telegraphic_oneliner,
    "nursing_intake_history": L_nursing_intake_history,
    "portal_message_history": L_portal_message_history,
    "referral_letter": L_referral_letter,
    "intake_form": L_intake_form,
    "registration_form": L_registration_form,
    "refill_request": L_refill_request,
    "portal_message_pharmacy": L_portal_message_pharmacy,
    "discharge_med_list": L_discharge_med_list,
    "pharmacy_oneliner": L_pharmacy_oneliner,
    "contact_block": L_contact_block,
    "fax_cover": L_fax_cover,
    "callback_note": L_callback_note,
    "shift_handoff": L_shift_handoff,
    "nursing_note_family": L_nursing_note_family,
    "followup_instructions": L_followup_instructions,
    "death_note": L_death_note,
    "hospice_discharge": L_hospice_discharge,
    "social_work_bereavement": L_social_work_bereavement,
    "nursing_expiration": L_nursing_expiration,
    "sig_progress_note": L_sig_progress_note,
    "sig_attestation": L_sig_attestation,
    "sig_telephone_encounter": L_sig_telephone_encounter,
    "sig_discharge_summary": L_sig_discharge_summary,
    "sig_nursing_note": L_sig_nursing_note,
    "sig_addendum": L_sig_addendum,
    "device_record_note": L_device_record_note,
    "telehealth_visit_note": L_telehealth_visit_note,
    "call_center_log": L_call_center_log,
    "portal_activation_message": L_portal_activation_message,
    "device_registration_form": L_device_registration_form,
    "after_visit_summary": L_after_visit_summary,
}
FAMILY_PHENOMENA = {
    "history_years": ["history_year", "date_boundaries"],
    "pharmacy_zip": ["pharmacy_zip", "facility_boundary"],
    "phone_extension": ["phone_extension"],
    "funeral_homes": ["service_organisation", "funeral_home"],
    "signature_dates": ["weekday_date", "date_boundaries"],
    "rare_identifiers": ["device_serial", "session_identifier", "access_code"],
}


# ---------------------------------------------------------------------------
# Plan and generation
# ---------------------------------------------------------------------------
def build_plan(seed=SEED, with_new=False, with_v3=False):
    rng = random.Random(f"{seed}:plan")
    plan = []
    for family, total in FAMILY_ROWS.items():
        layouts = LAYOUTS[family]
        per = total // len(layouts)
        for layout in layouts:
            negatives = round(per / 3)
            for k in range(per):
                plan.append(
                    {"family": family, "layout": layout, "positive": k >= negatives}
                )
    rng.shuffle(plan)
    cases = []
    for name, share in CASE_COUNTS:
        cases += [name] * round(len(plan) * share)
    cases = (cases + ["normal"] * len(plan))[: len(plan)]
    rng.shuffle(cases)
    cohorts = ["realistic"] * len(plan)
    order = list(range(len(plan)))
    rng.shuffle(order)
    for index in order[: round(len(plan) * 0.05)]:
        cohorts[index] = "long_document"
    for index in order[round(len(plan) * 0.05) : round(len(plan) * 0.09)]:
        cohorts[index] = "unusual_name"
    for item, case, cohort in zip(plan, cases, cohorts):
        item["case"], item["cohort"] = case, cohort
    if with_new:
        mode_cycle = ["normal", "normal", "normal", "upper", "lower"]
        for name, (positives, negatives, family, layouts) in NEW_CASES.items():
            for k in range(positives + negatives):
                plan.append(
                    {
                        "family": family,
                        "layout": layouts[k % len(layouts)],
                        "positive": k < positives,
                        "case": mode_cycle[k % 5],
                        "cohort": "realistic",
                        "newcase": name,
                    }
                )
        rng.shuffle(plan)
    if with_v3:
        plan += v3_plan(rng)
        rng.shuffle(plan)
    return plan


def v3_plan(rng):
    """The v3 additions: history-year variety, weekday signature lines, rare identifiers."""

    def group(newcase, family, layouts, case_cycle):
        items = []
        for layout, count in layouts:
            negatives = round(count / 3)
            items += [
                {
                    "family": family,
                    "layout": layout,
                    "positive": k >= negatives,
                    "cohort": "realistic",
                    "newcase": newcase,
                }
                for k in range(count)
            ]
        rng.shuffle(items)
        for index, item in enumerate(items):
            item["case"] = case_cycle[index % len(case_cycle)]
        return items

    five = ["normal", "normal", "normal", "upper", "lower"]
    seven = ["normal"] * 5 + ["upper", "lower"]
    return [
        *group("hist_v3", "history_years", V3_HIST_LAYOUTS_UPPER, ["upper"]),
        *group("hist_v3", "history_years", V3_HIST_LAYOUTS_OTHER, seven),
        *group(
            "sig_weekday", "signature_dates", [(x, 25) for x in V3_SIG_LAYOUTS], five
        ),
        *group("rare_id", "rare_identifiers", [(x, 25) for x in V3_RARE_LAYOUTS], five),
    ]


VERSIONS = {
    "v1": {
        "seed": SEED,
        "suite": SUITE,
        "suffix": "",
        "year_cap": YEAR_CAP,
        "pools": None,
        "with_new": False,
    },
    "v2": {
        "seed": SEED2,
        "suite": SUITE2,
        "suffix": "_v2",
        "year_cap": YEAR_CAP2,
        "pools": (FIRSTS_V2, LASTS_V2),
        "with_new": True,
    },
    "v3": {
        "seed": SEED3,
        "suite": SUITE3,
        "suffix": "_v3",
        "version": "tk14-v3",
        "year_cap": YEAR_CAP3,
        "pools": (FIRSTS_V2, LASTS_V2),
        "with_new": True,
        "with_v3": True,
        "exclusion": True,
        "attempts": 200,
    },
}
REJECTED: Counter = (
    Counter()
)  # exclusion-filter rejections of the last generate() call, by case


def exclusion_rejects(text, spans):
    """True when the row text, or an inserted value, hashes onto the blind-bank exclusion list."""
    if excluded_sequences(text) > 0:
        return True
    return any(excluded_value(span["text"]) for span in spans)


def v3_case_ok(newcase, positive, tags):
    names = {tag for tag, _ in tags}
    has = {
        "hist_v3": "year" in names,
        "sig_weekday": "sigdate" in names,
        "rare_id": bool(names & set(RARE_TAGS)),
    }[newcase]
    return has == positive


def make_row(index, item, reg, seen_texts, cfg=None):
    cfg = cfg or VERSIONS["v1"]
    suffix = cfg["suffix"]
    newcase = item.get("newcase")
    for attempt in range(cfg.get("attempts", 60)):
        snapshot = reg.snapshot() if cfg.get("exclusion") else None
        rng = random.Random(f"{cfg['seed']}:{index}:{attempt}")
        c = Ctx(
            rng,
            reg,
            item["family"],
            item["cohort"],
            item["positive"],
            item["case"],
            year_cap=cfg["year_cap"],
            pools=cfg["pools"],
            newcase=newcase,
        )
        LAYOUT_FUNCS[item["layout"]](c)
        if newcase and newcase not in V3_CASES:
            if not (c.hit if item["positive"] else c.neghit):
                continue
        text, spans, tags = c.rb.build(item["case"])
        if newcase in V3_CASES and not v3_case_ok(newcase, item["positive"], tags):
            reg.restore(snapshot)
            continue
        if cfg.get("exclusion") and exclusion_rejects(text, spans):
            REJECTED[newcase or item["family"]] += 1
            reg.restore(snapshot)
            continue
        if text in seen_texts:
            continue
        spans.sort(key=lambda s: s["start"])
        errors = annotation_errors(text, spans)
        if errors:
            raise RuntimeError(f"{item['layout']} row {index}: {errors}")
        seen_texts.add(text)
        groups = [canonical_identity_group(name) for name in c.people]
        row = {
            "id": f"tk14_{index:05d}{suffix}",
            "text": text,
            "spans": spans,
            "split": "train",
            "suite": cfg["suite"],
            "annotation_policy": POLICY_VERSION,
            "cohort": item["cohort"],
            "family": f"tk14_{item['layout']}{suffix}",
            "template_family": f"tk14_{item['layout']}{suffix}",
            "layout": item["layout"],
            "target_family": item["family"],
            "polarity": "positive" if item["positive"] else "negative",
            "case_mode": item["case"],
            "phenomena": [
                *FAMILY_PHENOMENA[item["family"]],
                "matched_positive" if item["positive"] else "matched_negative",
                *([newcase] if newcase else []),
            ],
            "identity_group": groups[0],
            "identity_groups": list(dict.fromkeys(groups)),
            "pair_id": f"tk14_unpaired_{index:05d}{suffix}",
        }
        if suffix:
            row["new_case"] = newcase
            row["version"] = cfg.get("version", "tk14-v2")
        years = [
            int(y)
            for tag, value in tags
            if tag == "year"
            for y in re.findall(r"\d{4}", value)
        ]
        future = [
            int(re.search(r"\d{4}", value).group())
            for tag, value in tags
            if tag == "future"
        ]
        return row, {"tags": tags, "years": years, "future_years": future}
    raise RuntimeError(f"could not make a unique row {index}")


def generate(version="v1"):
    cfg = VERSIONS[version]
    reg, seen = Registry(), set()
    rows, metas = [], []
    REJECTED.clear()
    plan = build_plan(cfg["seed"], cfg["with_new"], cfg.get("with_v3", False))
    for index, item in enumerate(plan):
        row, meta = make_row(index, item, reg, seen, cfg)
        rows.append(row)
        metas.append(meta)
    return rows, metas


def serialise(rows) -> bytes:
    return (
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=False) for row in rows)
        + "\n"
    ).encode("utf-8")


def statistics(rows, metas):
    support = Counter(span["label"] for row in rows for span in row["spans"])
    years = Counter(year for meta in metas for year in meta["years"])
    per_family = Counter(row["target_family"] for row in rows)
    per_layout = Counter(row["layout"] for row in rows)
    row_values: dict = {}
    for row, meta in zip(rows, metas):
        for tag, value in {(t, v.casefold()) for t, v in meta["tags"]}:
            row_values.setdefault(tag, Counter())[value] += 1
        for label in ("FACILITY", "LOCATION", "ADDRESS"):
            for value in {
                s["text"].casefold() for s in row["spans"] if s["label"] == label
            }:
                row_values.setdefault(f"all_{label}", Counter())[value] += 1
    max_rep = {
        tag: counter.most_common(1)[0] for tag, counter in sorted(row_values.items())
    }
    history_spans = sum(years.values())
    return {
        "rows": len(rows),
        "per_family": dict(per_family),
        "per_layout": dict(sorted(per_layout.items())),
        "layout_max_share": max(per_layout.values()) / len(rows),
        "case_modes": dict(Counter(row["case_mode"] for row in rows)),
        "polarity": dict(Counter(row["polarity"] for row in rows)),
        "polarity_by_family": {
            f: dict(Counter(r["polarity"] for r in rows if r["target_family"] == f))
            for f in per_family
        },
        "cohorts": dict(Counter(row["cohort"] for row in rows)),
        "label_support": {label: support[label] for label in ENTITY_TYPES},
        "history_year_spans": history_spans,
        "history_year_min": min(years),
        "history_year_max": max(years),
        "distinct_years": len(years),
        "history_year_max_share": max(years.values()) / history_spans,
        "history_year_top": years.most_common(3),
        "year_decade_counts": dict(
            sorted(Counter((y // 10) * 10 for y in years.elements()).items())
        ),
        "max_value_repetition_rows": {
            tag: {"value": value, "rows": count, "share": count / len(rows)}
            for tag, (value, count) in max_rep.items()
        },
    }


UNSETTLED = """\
Cases the policy and brief do not settle. Each was left out of the bank rather than guessed.

1. **Year ranges** such as `smoked 1980-2005` or `2014 - 2016`. The policy settles written date ranges for full dates, not bare year pairs. Not generated.
2. **Future years** (`next due 2028`, `repeat colonoscopy in 2033`). A scheduled future date is patient-specific but no fixture covers it. Not generated; intervals are written as durations (`repeat in 10 years`).
3. **ZIP+4** (`53728-1042`) after a pharmacy. The policy covers a five-digit ZIP alone only. Not generated.
4. **Pharmacy name followed by town and ZIP** in one phrase (`Name, Town, ST 53728`). P5 settles a locality after a care site; whether the state and ZIP stay inside one LOCATION span is not stated for this shape. Not generated; the pharmacy rows use a ZIP only.
5. **Prescription (Rx) numbers and NPI-style codes** on dispensing lines. ID versus ACCOUNT versus unannotated is not settled for prescription numbers. Not generated; pharmacy negatives use NDC-style codes, lot numbers, quantities, day supplies and dose counts only.
6. **Internal extensions without a phone number** (`dial 4417 from any unit phone`) and **pager numbers**. The policy puts an extension inside the PHONE span only when it belongs to a phone. Not generated. Room and bed numbers are used as negatives.
7. **Funeral-home types without a person-style name** (`Cremation Society` with no proper name) and **a funeral home named only by a person's surname with no type word**. Not generated; every positive carries a proper name plus its type words.
8. **Diagnosis-onset years** (`hypertension since 2009`). These follow the general rule that a patient-specific year is DATE, and were included in a minority of history items, but no fixture covers a diagnosis year. Flagged for the owner to confirm.
9. **Real-place collision.** Town, stem and street words are invented spellings, but the author could not check them against any gazetteer (reading data files was not permitted), so a coincidental match with a real place cannot be ruled out.
10. **SSN rows.** A small number of registration forms carry SSNs from the never-issued 900-999 area range. The policy requires a documented invented scheme for SSNs; this is the scheme used.

Provenance note. The author read only the permitted files, plus a directory listing of `tests/` and the first 25 lines of one existing fixture-only test (`tests/test_annotation_policy_care_sites.py`) while looking for a test layout; no corpus, bank, report, model or other generator was opened.

"""


def write_reports(rows, metas, stats, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# tk14 targeted-gaps bank: author report",
        "",
        f"Generated by `scripts/takeover/author14_targeted_gaps.py` (seed {SEED}). Synthetic, invented values only. Training bank; not an evaluation bank.",
        "",
        "## Rows per family",
        "",
        "| Family | Rows | Positive | Negative |",
        "|---|---:|---:|---:|",
    ]
    for family, count in stats["per_family"].items():
        pol = stats["polarity_by_family"][family]
        lines.append(
            f"| {family} | {count} | {pol.get('positive', 0)} | {pol.get('negative', 0)} |"
        )
    lines += [
        f"| total | {stats['rows']} | {stats['polarity']['positive']} | {stats['polarity']['negative']} |",
        "",
        "Negatives are %.1f%% of rows."
        % (100 * stats["polarity"]["negative"] / stats["rows"]),
        "",
        "## Layouts",
        "",
        "| Layout | Rows | Share |",
        "|---|---:|---:|",
    ]
    for layout, count in stats["per_layout"].items():
        lines.append(f"| {layout} | {count} | {100 * count / stats['rows']:.1f}% |")
    lines += [
        "",
        f"{len(stats['per_layout'])} layouts; the largest holds {100 * stats['layout_max_share']:.1f}% of rows.",
        "",
        "## Case mode and cohort",
        "",
        "| Case mode | Rows |",
        "|---|---:|",
    ]
    lines += [f"| {k} | {v} |" for k, v in stats["case_modes"].items()]
    lines += ["", "| Cohort | Rows |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in stats["cohorts"].items()]
    lines += ["", "## Label support (spans)", "", "| Label | Spans |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in stats["label_support"].items()]
    lines += [
        "",
        "## History-year spread",
        "",
        f"{stats['history_year_spans']} history-year DATE spans (bare years and month-year forms in the history family), "
        f"{stats['distinct_years']} distinct years from {stats['history_year_min']} to {stats['history_year_max']}. "
        f"The most frequent year holds {100 * stats['history_year_max_share']:.2f}% of them (cap 3%); top three: {stats['history_year_top']}.",
        "",
        "| Decade | Spans |",
        "|---|---:|",
    ]
    lines += [f"| {k}s | {v} |" for k, v in stats["year_decade_counts"].items()]
    lines += [
        "",
        "## Maximum value repetition (rows containing the same value, case-folded)",
        "",
        "| Value type | Value | Rows | Share of rows |",
        "|---|---|---:|---:|",
    ]
    for tag, info in stats["max_value_repetition_rows"].items():
        if tag == "year":
            continue
        lines.append(
            f"| {tag} | {info['value']} | {info['rows']} | {100 * info['share']:.2f}% |"
        )
    lines += [
        "",
        "The cap is 0.5% of rows (10 rows).",
        "",
        "## Unsettled cases",
        "",
        UNSETTLED,
    ]
    (report_dir / "AUTHOR_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    sample_rng = random.Random(f"{SEED}:spot")
    sample = sorted(sample_rng.sample(range(len(rows)), 100))
    out = [
        "# Spot-check sample (100 rows, fixed seed)",
        "",
        "Spans are marked `[LABEL|text]`. Rows are shown with `\\n` for newlines.",
        "",
    ]
    for index in sample:
        row = rows[index]
        text, marked, pos = row["text"], [], 0
        for span in row["spans"]:
            marked.append(text[pos : span["start"]])
            marked.append(f"[{span['label']}|{text[span['start'] : span['end']]}]")
            pos = span["end"]
        marked.append(text[pos:])
        out.append(
            f"### {row['id']} ({row['target_family']} / {row['layout']} / {row['polarity']} / {row['case_mode']} / {row['cohort']})"
        )
        out.append("")
        out.append("```")
        out.append("".join(marked))
        out.append("```")
        out.append("")
    (report_dir / "spot_check_sample.md").write_text("\n".join(out), encoding="utf-8")


UNSETTLED_V2 = """\
Still out of scope by the owner's decision (not generated): a pharmacy followed by town, state and ZIP in one phrase, and funeral-home names with no type word.

Choices made inside the settled cases that the fixtures do not spell out. Each is flagged for the owner:

1. **Month-year future dates** (`next due Mar 2028`, `next due 03/2028`) are written as one DATE span, by analogy with partial dates such as `Mar 2015`. The fixture shows a bare future year only.
2. **Year ranges.** Only bare-year ranges are generated: `1984-2003`, `1984 - 2003` and `1984 to 2003`, each one DATE span, with a leading `from` outside the span (as for full-date ranges). Ranges of month-year dates are not generated.
3. **Future years** run from the year after the note year up to twelve years later, so they fall outside the 1950 to 2026 history-year window by design. They are counted separately and capped at 40 spans per year.
4. **Pager numbers** are generated only as full fictional phone numbers (555-0100 to 555-0199 lines). Short internal pager codes (`pager 4417`) are not generated, because the fixture and policy cover a pager that is a phone number.
5. **Prescription numbers** are generated only with an `Rx`, `Rx #`, `Rx no.`, `Rx number`, `Rx ID` or `prescription number` caption outside the span. The `RX-nnnnn` form includes its `RX-` prefix in the ID span, following the fixture. A bare number with no caption is not generated.
6. **ZIP+4** is generated after a pharmacy name and in `Mailing ZIP:`, `ZIP+4:`, `Billing ZIP:` and `Mail ZIP` fields. Hyphenated look-alike codes with no ZIP caption (`lot 53728-4416`, `NDC 53728-4416`, `batch 53728-441`) are the matched negatives and are not annotated. Whether a model should separate the two on caption alone is the owner's call.
7. **Bare extensions** (`Ward phone ext. 4412`, `Charge desk x5120`) are generated only as negatives and never directly after a phone number.
8. **Name pools.** v2 draws person names from rarer given-name and surname pools to reduce identity collisions. The author could not check them against the protected identity lists (not permitted), so collisions cannot be ruled out; the owner's overlap audit decides.
9. **Real-place collision** (unchanged from v1): invented town, stem and street spellings could not be checked against a gazetteer.
"""


def new_case_stats(rows, metas):
    out = {}
    for row, _meta in zip(rows, metas):
        case = row.get("new_case")
        if not case:
            continue
        entry = out.setdefault(
            case, {"positive": 0, "negative": 0, "normal": 0, "upper": 0, "lower": 0}
        )
        entry[row["polarity"]] += 1
        entry[row["case_mode"]] += 1
    future = Counter(y for m in metas for y in m["future_years"])
    return out, future


def write_reports_v2(rows, metas, stats, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)
    cases, future = new_case_stats(rows, metas)
    lines = [
        "## tk14 v2 (settled follow-up cases)",
        "",
        f"Generated by `scripts/takeover/author14_targeted_gaps.py --version v2` (seed {SEED2}, version `tk14-v2`) into `data/tk14_train_targeted_gaps__v2.jsonl`. v1 is unchanged and the generator still reproduces it byte for byte (tested). v2 repeats the 2,000-row v1 composition with a new seed and rarer name pools, and adds {sum(v['positive'] + v['negative'] for v in cases.values())} rows for the cases the owner settled on 2026-10-05. Synthetic, invented values only; training bank, not an evaluation bank.",
        "",
        f"Total {stats['rows']} rows; negatives are {100 * stats['polarity']['negative'] / stats['rows']:.1f}%.",
        "",
        "### Rows per family",
        "",
        "| Family | Rows | Positive | Negative |",
        "|---|---:|---:|---:|",
    ]
    for family, count in stats["per_family"].items():
        pol = stats["polarity_by_family"][family]
        lines.append(
            f"| {family} | {count} | {pol.get('positive', 0)} | {pol.get('negative', 0)} |"
        )
    lines += [
        "",
        "### Rows per new case",
        "",
        "| New case | Positive | Negative | Normal | All-caps | Lowercase |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in NEW_CASES:
        v = cases[name]
        lines.append(
            f"| {name} | {v['positive']} | {v['negative']} | {v['normal']} | {v['upper']} | {v['lower']} |"
        )
    lines += [
        "",
        "### Layouts, case and cohort",
        "",
        f"{len(stats['per_layout'])} layouts; the largest holds {100 * stats['layout_max_share']:.1f}% of rows. "
        f"Case modes: {stats['case_modes']}. Cohorts: {stats['cohorts']}.",
        "",
        "| Layout | Rows | Share |",
        "|---|---:|---:|",
    ]
    for layout, count in stats["per_layout"].items():
        lines.append(f"| {layout} | {count} | {100 * count / stats['rows']:.1f}% |")
    lines += ["", "### Label support (spans)", "", "| Label | Spans |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in stats["label_support"].items()]
    lines += [
        "",
        "### Year spread",
        "",
        f"{stats['history_year_spans']} history-year DATE endpoints (bare years, month-year forms, onset years and both ends of each range), "
        f"{stats['distinct_years']} distinct years from {stats['history_year_min']} to {stats['history_year_max']}. "
        f"The most frequent year holds {100 * stats['history_year_max_share']:.2f}% (cap 3%); top three: {stats['history_year_top']}.",
        "",
        f"Future patient-specific years: {sum(future.values())} spans over {len(future)} distinct years ({min(future)} to {max(future)}); the most frequent year holds {max(future.values())} spans. These sit outside the 1950 to 2026 window by design and are not in the figure above.",
        "",
        "### Maximum value repetition (rows containing the same value, case-folded)",
        "",
        "| Value type | Value | Rows | Share of rows |",
        "|---|---|---:|---:|",
    ]
    for tag, info in stats["max_value_repetition_rows"].items():
        if tag in ("year", "future"):
            continue
        lines.append(
            f"| {tag} | {info['value']} | {info['rows']} | {100 * info['share']:.2f}% |"
        )
    lines += [
        "",
        "The cap is 0.5% of rows (11 rows).",
        "",
        "### Unsettled and flagged",
        "",
        UNSETTLED_V2,
    ]
    report = report_dir / "AUTHOR_REPORT.md"
    existing = report.read_text(encoding="utf-8") if report.exists() else ""
    marker = "## tk14 v2 (settled follow-up cases)"
    if marker in existing:
        existing = existing[: existing.index(marker)]
    report.write_text(
        existing.rstrip("\n") + "\n\n" + "\n".join(lines), encoding="utf-8"
    )

    sample_rng = random.Random(f"{SEED2}:spot")
    by_case = {name: [] for name in NEW_CASES}
    base = []
    for index, row in enumerate(rows):
        (by_case[row["new_case"]] if row.get("new_case") else base).append(index)
    sample = []
    for indexes in by_case.values():
        sample += sample_rng.sample(indexes, 7)
    sample += sample_rng.sample(base, 100 - len(sample))
    sample.sort()
    out = [
        "# Spot-check sample, tk14 v2 (100 rows, fixed seed)",
        "",
        "49 rows on the new cases (7 per case) and 51 base rows. Spans are marked `[LABEL|text]`.",
        "",
    ]
    for index in sample:
        row = rows[index]
        text, marked, pos = row["text"], [], 0
        for span in row["spans"]:
            marked.append(text[pos : span["start"]])
            marked.append(f"[{span['label']}|{text[span['start'] : span['end']]}]")
            pos = span["end"]
        marked.append(text[pos:])
        out.append(
            f"### {row['id']} ({row['new_case'] or row['target_family']} / {row['layout']} / {row['polarity']} / {row['case_mode']} / {row['cohort']})"
        )
        out += ["", "```", "".join(marked), "```", ""]
    (report_dir / "spot_check_sample_v2.md").write_text(
        "\n".join(out), encoding="utf-8"
    )


def exclusion_summary(rows):
    """Counts only: the strings that matched are never recorded."""
    return {
        "list": "data/places_exclusions.sha256.json",
        "list_sha256": hashlib.sha256(EXCLUSIONS_PATH.read_bytes()).hexdigest(),
        "rule": "reject a row when its inserted values or any 1-8 word sequence of its text hash onto the list",
        "rows_checked": len(rows),
        "rejected_draws": dict(sorted(REJECTED.items())),
        "rejected_draws_total": sum(REJECTED.values()),
        "accepted_rows_with_a_match": sum(
            exclusion_rejects(row["text"], row["spans"]) for row in rows
        ),
    }


UNSETTLED_V3 = """\
Still out of scope by the owner's decision (not generated): a pharmacy followed by town, state and ZIP in one phrase, and funeral-home names with no type word.

Choices made inside the v3 additions that the policy and fixtures do not spell out. Each is flagged for the owner:

1. **Reading of "at least half in all-caps annual-wellness and SOAP layouts".** 160 of the 300 added history-year rows (53%) are all-caps and in the AWV and SOAP layouts (80 each, one-third negatives). The other 140 rows use eight other history layouts with a normal/all-caps/lowercase split of 100/20/20. This pushes the bank's case mix to roughly 57% normal, 24% all-caps and 19% lowercase, against the brief's 60/20/20.
2. **Weekday forms.** The weekday is inside the DATE span in every positive signature line (`Fri, 7 Feb 2025`, `Tuesday, March 4, 2025`, `Mon. 3/4/2025`), and always matches the calendar. A weekday written with no date (`signed on Tuesday`) is not generated: the policy settles a weekday before a date only.
3. **Signature-line negatives** carry a clock time with no date, an elapsed `[00:08:22]`-style stamp, or no stamp (`Pending signature: <name>`); the clinician name stays NAME. Header dates elsewhere in those rows are still annotated.
4. **Signature dates** use the note year and need not follow the header visit date, so a signature may precede it. Times include seconds and `hrs` forms; all are outside the span.
5. **Serial numbers** may begin with an `SN` prefix inside the value (`SN48201733`), by analogy with the `RX-` prefix in prescription numbers. The captions `serial number`, `S/N`, `SN` (when separate) and `serial #` stay outside.
6. **Access-code captions beyond the fixture.** `portal activation code` and `one-time access code` follow the policy text. `enrollment code`, `pairing code`, `sign-up code`, `access code` and `invite code` are extended to ID by analogy as patient-tied codes; the owner may want to confirm them.
7. **Meeting identifiers.** `Meeting ID 482-117-905` is ID by the policy's session, meeting and call row; the meeting link next to it is a separate URL span. A link with no identifier is URL only (negative rows).
8. **Not generated:** device model numbers, lot and catalogue numbers (left to the pharmacy and history negatives, where they are unannotated), and bare numeric codes with no caption. Software and firmware version strings appear in both positive and negative rows and are never annotated.
9. **Name pools** are the v2 rare pools. They could not be checked against the protected identity lists (not permitted); the owner's overlap audit decides.
10. **Exclusion filter.** Applied to every row of v3 (inserted values and every 1-8 word sequence of the text, case-folded). Rows that matched were redrawn; only counts are recorded, and which strings matched was never printed or inspected. The generator also hashes each span value on its own.
11. **Real-place collision** (unchanged): invented town, stem and street spellings could not be checked against a gazetteer other than the hash list.
"""


def write_reports_v3(rows, metas, stats, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)
    cases, future = new_case_stats(rows, metas)
    summary = exclusion_summary(rows)
    v3_cases = {k: v for k, v in cases.items() if k in V3_CASES}
    added = sum(v["positive"] + v["negative"] for v in v3_cases.values())
    lines = [
        "## tk14 v3 (blind-bank exclusion filter and additions)",
        "",
        f"Generated by `scripts/takeover/author14_targeted_gaps.py --version v3` (seed {SEED3}, version `tk14-v3`) into `data/tk14_train_targeted_gaps__v3.jsonl`. v1 and v2 are unchanged and the generator still reproduces both byte for byte (tested). v3 repeats the v2 composition (2,000 base rows plus the 350 settled follow-up rows) with a new seed, rarer name pools and the blind-bank exclusion filter on every row, and adds {added} rows: history-year variety, weekday-before-date signature lines and rare identifier formats. Synthetic, invented values only; training bank, not an evaluation bank.",
        "",
        f"Total {stats['rows']} rows; negatives are {100 * stats['polarity']['negative'] / stats['rows']:.1f}%.",
        "",
        "### Exclusion filter",
        "",
        f"Every row was checked against `data/places_exclusions.sha256.json` (SHA-256 {summary['list_sha256'][:16]}...): a row whose inserted values or any 1-8 word sequence of its text hash onto the list was rejected and redrawn (`exclusion_hash`, `excluded_sequences`, `excluded_value`). Counts only; the matching strings were never printed, logged or inspected.",
        "",
        f"- Rejected draws: {summary['rejected_draws_total']} in total, by case or family: {summary['rejected_draws']}.",
        f"- Rows in the file that still match: {summary['accepted_rows_with_a_match']} (of {summary['rows_checked']}).",
        "",
        "### Rows per family",
        "",
        "| Family | Rows | Positive | Negative |",
        "|---|---:|---:|---:|",
    ]
    for family, count in stats["per_family"].items():
        pol = stats["polarity_by_family"][family]
        lines.append(
            f"| {family} | {count} | {pol.get('positive', 0)} | {pol.get('negative', 0)} |"
        )
    lines += [
        "",
        "### Rows per addition and per v2 case",
        "",
        "| Case | Positive | Negative | Normal | All-caps | Lowercase |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, v in cases.items():
        lines.append(
            f"| {name} | {v['positive']} | {v['negative']} | {v['normal']} | {v['upper']} | {v['lower']} |"
        )
    hist_upper = [
        r
        for r in rows
        if r["new_case"] == "hist_v3"
        and r["case_mode"] == "upper"
        and r["layout"] in ("awv_note", "soap_note")
    ]
    kinds = Counter(
        tag for meta in metas for tag, _ in meta["tags"] if tag in RARE_TAGS
    )
    lines += [
        "",
        f"History-year additions in all-caps annual-wellness and SOAP layouts: {len(hist_upper)} of 300. Rare-identifier spans by kind: {dict(sorted(kinds.items()))}.",
        "",
        "### Layouts, case and cohort",
        "",
        f"{len(stats['per_layout'])} layouts; the largest holds {100 * stats['layout_max_share']:.1f}% of rows. "
        f"Case modes: {stats['case_modes']}. Cohorts: {stats['cohorts']}.",
        "",
        "| Layout | Rows | Share |",
        "|---|---:|---:|",
    ]
    for layout, count in stats["per_layout"].items():
        lines.append(f"| {layout} | {count} | {100 * count / stats['rows']:.1f}% |")
    lines += ["", "### Label support (spans)", "", "| Label | Spans |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in stats["label_support"].items()]
    lines += [
        "",
        "### Year spread",
        "",
        f"{stats['history_year_spans']} history-year DATE endpoints, {stats['distinct_years']} distinct years from {stats['history_year_min']} to {stats['history_year_max']}. "
        f"The most frequent year holds {100 * stats['history_year_max_share']:.2f}% (cap 3%); top three: {stats['history_year_top']}.",
        "",
        f"Future patient-specific years: {sum(future.values())} spans over {len(future)} distinct years ({min(future)} to {max(future)}); outside the 1950 to 2026 window by design and not in the figure above.",
        "",
        "### Maximum value repetition (rows containing the same value, case-folded)",
        "",
        "| Value type | Value | Rows | Share of rows |",
        "|---|---|---:|---:|",
    ]
    for tag, info in stats["max_value_repetition_rows"].items():
        if tag in ("year", "future"):
            continue
        lines.append(
            f"| {tag} | {info['value']} | {info['rows']} | {100 * info['share']:.2f}% |"
        )
    lines += [
        "",
        f"The cap is 0.5% of rows ({int(0.005 * stats['rows'])} rows).",
        "",
        "### Unsettled and flagged",
        "",
        UNSETTLED_V3,
    ]
    report = report_dir / "AUTHOR_REPORT.md"
    existing = report.read_text(encoding="utf-8") if report.exists() else ""
    marker = "## tk14 v3 (blind-bank exclusion filter and additions)"
    if marker in existing:
        existing = existing[: existing.index(marker)]
    report.write_text(
        existing.rstrip("\n") + "\n\n" + "\n".join(lines), encoding="utf-8"
    )

    sample_rng = random.Random(f"{SEED3}:spot")
    by_case = {name: [] for name in V3_CASES}
    base = []
    for index, row in enumerate(rows):
        if row.get("new_case") in V3_CASES:
            by_case[row["new_case"]].append(index)
        else:
            base.append(index)
    sample = []
    for indexes in by_case.values():
        sample += sample_rng.sample(indexes, 15)
    sample += sample_rng.sample(base, 100 - len(sample))
    sample.sort()
    out = [
        "# Spot-check sample, tk14 v3 (100 rows, fixed seed)",
        "",
        "45 rows on the v3 additions (15 each for history-year variety, weekday signature lines and rare identifiers) and 55 other rows. Spans are marked `[LABEL|text]`.",
        "",
    ]
    for index in sample:
        row = rows[index]
        text, marked, pos = row["text"], [], 0
        for span in row["spans"]:
            marked.append(text[pos : span["start"]])
            marked.append(f"[{span['label']}|{text[span['start'] : span['end']]}]")
            pos = span["end"]
        marked.append(text[pos:])
        out.append(
            f"### {row['id']} ({row['new_case'] or row['target_family']} / {row['layout']} / {row['polarity']} / {row['case_mode']} / {row['cohort']})"
        )
        out += ["", "```", "".join(marked), "```", ""]
    (report_dir / "spot_check_sample_v3.md").write_text(
        "\n".join(out), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=sorted(VERSIONS), default="v1")
    parser.add_argument("--out")
    parser.add_argument(
        "--report-dir",
        default=str(ROOT / "reports" / "takeover" / "author14_targeted_gaps"),
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing v1 bank"
    )
    args = parser.parse_args()
    version = args.version
    cfg = VERSIONS[version]
    default_name = (
        "tk14_train_targeted_gaps.jsonl"
        if version == "v1"
        else f"tk14_train_targeted_gaps__{version}.jsonl"
    )
    out = Path(args.out) if args.out else ROOT / "data" / default_name
    if version == "v1" and out.exists() and not args.force:
        raise SystemExit(f"{out} exists; v1 is frozen. Use --force to overwrite.")
    rows, metas = generate(version)
    payload = serialise(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    stats = statistics(rows, metas)
    support = stats["label_support"]
    manifest = {
        "name": out.name,
        "version": cfg.get("version", "tk14-v1" if version == "v1" else "tk14-v2"),
        "annotation_policy": POLICY_VERSION,
        "policy_addenda": [
            "clinical-phi-v2-places",
            "care-sites-2026-10-05",
            "targeted-gaps-2026-10-05",
            *(["targeted-followup-2026-10-05"] if version in ("v2", "v3") else []),
            *(["transcript-decisions-2026-10-06"] if version == "v3" else []),
        ],
        "split": "train",
        "role": "train",
        "independent_final": False,
        "rows": len(rows),
        "documents": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "label_support": support,
        "entity_support": support,
        "label_counts": support,
        "cohort_counts": stats["cohorts"],
        "family_counts": stats["per_family"],
        "layout_counts": stats["per_layout"],
        "case_mode_counts": stats["case_modes"],
        "polarity_counts": stats["polarity"],
        "history_year_spread": {
            k: stats[k]
            for k in (
                "history_year_spans",
                "history_year_min",
                "history_year_max",
                "distinct_years",
                "history_year_max_share",
            )
        },
        "max_value_repetition_rows": stats["max_value_repetition_rows"],
        "author_provenance": {
            "generator": "scripts/takeover/author14_targeted_gaps.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "seed": cfg["seed"],
            "brief": "docs/targeted-gaps-bank-brief.md",
            "synthetic_only": True,
            "inputs_read": "policy documents, fixtures and brief only; no corpus, evaluation or model file",
        },
    }
    if version in ("v2", "v3"):
        cases, future = new_case_stats(rows, metas)
        manifest["new_case_counts"] = cases
        manifest["future_year_spans"] = sum(future.values())
        manifest["supersedes_note"] = (
            "New file; tk14 v1 and v2 are unchanged. Names drawn from rarer pools."
        )
    if version == "v3":
        manifest["exclusion_filter"] = exclusion_summary(rows)
    out.with_suffix(out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if version == "v1":
        write_reports(rows, metas, stats, Path(args.report_dir))
    elif version == "v2":
        write_reports_v2(rows, metas, stats, Path(args.report_dir))
    else:
        write_reports_v3(rows, metas, stats, Path(args.report_dir))
    print(
        json.dumps(
            {
                k: stats[k]
                for k in (
                    "rows",
                    "per_family",
                    "case_modes",
                    "cohorts",
                    "label_support",
                    "history_year_max_share",
                    "layout_max_share",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
