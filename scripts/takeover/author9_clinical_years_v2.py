"""Prospectively author one training-only clinical-year corpus, without inference.

The prose below was newly written by this author for the frozen original.
This revision reads only that author's four pinned original artifacts and its
explicit policy/normalizer/runtime/tokenizer inputs. It does not discover data,
read another corpus, retrieve medical facts, or modify a training mixture.
Run from the repository with PYTHONPATH=src and the repository Python runtime.
"""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import stat
import unicodedata

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/tk9_train_clinical_years_v2.jsonl"
REPORT = ROOT / "reports/takeover/mix3_001/author9_clinical_years/revision2/author_report.json"
MODEL = ROOT / "models/takeover-mbbase-mix3-001/step-003081"
POLICY = "clinical-phi-v2"
LABELS = ("NAME", "DATE", "ADDRESS", "PHONE", "EMAIL", "MRN", "SSN", "ACCOUNT",
          "ID", "URL", "IP", "AGE", "LOCATION", "FACILITY")
LAYOUTS = ("narrative", "telephone_exchange", "sectioned_review", "ledger",
           "portal_message", "handoff", "question_answer", "review_checklist")

# Five independently written clinical formulations per family. Every {d} is
# an authored patient-specific date; {y} is reserved for public references.
FAMILIES = [
    ("immunization", "tetanus_wound_history", 1991, 2024, "Tetanus history", [
        "While discussing the wound, the patient recalled a tetanus booster in {d}.",
        "The injury intake form lists {d} as the year of the patient's last tetanus injection.",
        "Tetanus record: booster given in {d}; the patient is looking for the original card.",
        "An old urgent-care receipt places this patient's tetanus dose in {d}.",
        "Asked about tetanus, the patient answered, ‘{d}, after I cut my hand.’",
    ], "The wound history and the vaccination history were collected in separate parts of the interview."),
    ("immunization", "influenza_pharmacy_receipt", 2000, 2024, "Influenza record", [
        "A pharmacy receipt confirms an influenza vaccination for the patient in {d}.",
        "The patient remembers receiving the flu vaccine in {d} at a walk-in visit.",
        "Influenza vaccination appears under {d} on the patient's retained receipt.",
        "The patient brought a card showing a flu shot in {d}; the site field is unreadable.",
        "For this patient's flu-shot history, the only legible year on the card is {d}.",
    ], "The receipt was returned after a copy was added to the record."),
    ("immunization", "pneumococcal_registry_reconciliation", 1999, 2024, "Pneumococcal reconciliation", [
        "The registry places the patient's pneumococcal vaccination in {d}.",
        "Pneumococcal history supplied by the patient: one remembered injection in {d}.",
        "A transfer summary records a pneumococcal dose for this patient in {d}.",
        "The patient's wallet card has a pneumococcal entry beside {d}.",
        "During reconciliation, the patient identified {d} as the year of a pneumococcal injection.",
    ], "The product field will be checked against the original vaccination record."),
    ("immunization", "zoster_series_card", 2008, 2024, "Zoster card review", [
        "The patient's zoster vaccination card begins with an entry from {d}.",
        "A zoster dose is documented for the patient in {d}; the remaining card is missing.",
        "The patient places the first remembered shingles injection in {d}.",
        "Zoster vaccine history was transcribed from the patient's card, which gives {d}.",
        "The pharmacist's receipt in this chart dates the patient's shingles vaccination to {d}.",
    ], "The patient will bring the second page before the history is finalized."),
    ("immunization", "hepatitis_b_occupational_file", 1993, 2024, "Hepatitis B history", [
        "The patient's occupational file documents a hepatitis B dose in {d}.",
        "The patient recalls beginning hepatitis B vaccination in {d} before a clinical placement.",
        "Hepatitis B vaccination is listed in this patient's work-health record under {d}.",
        "A retained employment-health card links the patient's hepatitis B series to {d}.",
        "The patient supplied {d} as the year of the hepatitis B injections recorded by work health.",
    ], "Only the vaccination entries were requested from the employment-health office."),
    ("immunization", "mmr_school_record", 1985, 2024, "MMR record transfer", [
        "The patient's school record lists an MMR vaccination in {d}.",
        "An MMR entry dated {d} was found on the patient's childhood vaccine card.",
        "The patient remembers an MMR catch-up injection in {d} before enrollment.",
        "For this patient, the transferred MMR history contains a single year: {d}.",
        "The patient provided a school-health copy showing an MMR dose in {d}.",
    ], "The school-health copy was compared with the card brought to the visit."),
    ("immunization", "varicella_history_confirmation", 1999, 2024, "Varicella history", [
        "The patient reports a varicella vaccination in {d} and does not have the card.",
        "The family-held vaccine sheet records the patient's varicella injection in {d}.",
        "The patient's transferred immunization list associates varicella vaccine with {d}.",
        "Varicella history for the patient includes a dose in {d}, according to the scanned sheet.",
        "On review of the patient's card, the varicella row gives {d} as the administration year.",
    ], "A history of infection was discussed separately from the vaccination entries."),
    ("immunization", "hpv_adolescent_record", 2008, 2024, "HPV series history", [
        "The patient's adolescent record shows an HPV vaccine dose in {d}.",
        "An HPV injection for the patient is recorded in {d} on the transferred card.",
        "The patient recalls starting HPV vaccination in {d} while attending school.",
        "The HPV line in the patient's old clinic summary lists {d}.",
        "The patient brought a vaccine receipt from {d} that includes an HPV dose.",
    ], "The dose count remains a record-reconciliation question rather than an assumption from the year."),
    ("immunization", "covid_card_upload", 2021, 2024, "COVID vaccination upload", [
        "The uploaded card records the patient's COVID vaccination in {d}.",
        "The patient remembers a COVID vaccine appointment in {d}; the original card is at home.",
        "COVID vaccination for this patient appears in the registry under {d}.",
        "The patient's personal vaccine log lists a COVID injection in {d}.",
        "The year on the patient's saved COVID vaccination receipt is {d}.",
    ], "The patient agreed to send a clearer image of the product and dose fields."),
    ("immunization", "hepatitis_a_travel_card", 1998, 2024, "Travel vaccine history", [
        "Before an earlier trip, the patient received hepatitis A vaccine in {d}.",
        "The patient's travel-health card documents a hepatitis A injection in {d}.",
        "The patient places a prior hepatitis A vaccine visit in {d}.",
        "A hepatitis A dose appears beside {d} on the patient's retained travel card.",
        "The travel intake review found a patient-reported hepatitis A vaccination from {d}.",
    ], "The destination discussion was kept separate from the review of prior vaccine records."),
    ("screening_test", "colonoscopy_screening_report", 1991, 2024, "Colonoscopy history", [
        "The patient reports a screening colonoscopy in {d}.",
        "The patient's prior colonoscopy report is dated {d}; the images were not transferred.",
        "Colorectal screening history: the patient recalls a colonoscopy in {d}.",
        "The only colonoscopy year supplied by the patient was {d}.",
        "An outside summary records this patient's colonoscopy in {d}.",
    ], "The endoscopy report is needed before the screening history can be summarized."),
    ("screening_test", "fit_home_kit_history", 2002, 2024, "Home screening kit", [
        "The patient returned a home FIT kit in {d}, according to the result notice.",
        "The patient's last remembered stool screening test was in {d}.",
        "A FIT result letter in the patient's folder carries the year {d}.",
        "The patient reports completing the mailed colorectal screening kit in {d}.",
        "The transferred chart includes this patient's FIT screening from {d}.",
    ], "The patient was asked to bring the result letter rather than infer the result from receipt of the kit."),
    ("screening_test", "mammography_image_request", 1996, 2024, "Breast imaging history", [
        "The patient remembers a screening mammogram in {d}.",
        "The patient's mammography history lists {d}, with the report still outstanding.",
        "A prior mammogram for this patient is indexed under {d} in the transfer packet.",
        "The patient brought a screening mammography letter from {d}.",
        "The year given for the patient's earlier breast screening examination was {d}.",
    ], "The imaging request includes the prior report and the images, if available."),
    ("screening_test", "cervical_screening_recollection", 1994, 2024, "Cervical screening history", [
        "The patient recalls cervical screening in {d} but is unsure which test was used.",
        "The patient's transfer summary lists a cervical screening specimen from {d}.",
        "A cervical screening result letter supplied by the patient is dated {d}.",
        "The patient gave {d} as the year of the earlier cervical screening visit.",
        "The available chart places this patient's cervical screening in {d}.",
    ], "The original laboratory report was requested to clarify the test type."),
    ("screening_test", "bone_density_comparison", 1997, 2024, "Bone density history", [
        "The patient's earlier bone density study was performed in {d}.",
        "The patient recalls a DXA examination in {d} and has retained the printed graph.",
        "Bone density records for this patient include a study from {d}.",
        "The comparison examination in the patient's DXA report is identified by the year {d}.",
        "The patient supplied a bone density report whose examination year is {d}.",
    ], "The printed scores will be checked against the complete imaging report."),
    ("screening_test", "retinal_screening_photo", 2002, 2024, "Retinal screening record", [
        "The patient had retinal screening photographs taken in {d}.",
        "A retinal screening entry for the patient appears in {d} in the diabetes review.",
        "The patient remembers a retinal photograph visit in {d}.",
        "The transferred eye-care summary records this patient's retinal screening in {d}.",
        "The patient brought a retinal screening result notice from {d}.",
    ], "The screening photograph entry was distinguished from a separate comprehensive eye examination."),
    ("screening_test", "tuberculosis_clearance_test", 1995, 2024, "Tuberculosis test history", [
        "The patient's work-clearance record documents a tuberculosis test in {d}.",
        "The patient recalls tuberculosis screening in {d} before a placement.",
        "A tuberculosis test result for this patient is filed under {d}.",
        "The patient's copied clearance form gives {d} as the tuberculosis testing year.",
        "The only tuberculosis screening year the patient could recall was {d}.",
    ], "The original test record was requested because the copied form does not identify the method."),
    ("screening_test", "lung_screening_ct", 2011, 2024, "Lung screening record", [
        "The patient's earlier low-dose screening CT was performed in {d}.",
        "The patient reports attending a lung screening CT appointment in {d}.",
        "A lung screening report for this patient is indexed under {d}.",
        "The patient's retained screening letter refers to a chest CT from {d}.",
        "The transfer packet places the patient's prior lung screening examination in {d}.",
    ], "The images were requested so the reviewing clinician can see the original examination."),
    ("screening_test", "sleep_study_equipment_history", 1993, 2024, "Sleep study history", [
        "The patient underwent a sleep study in {d} before receiving the current equipment.",
        "The diagnostic study in this patient's sleep record was completed in {d}.",
        "The patient remembers an overnight study in {d} and has requested the report.",
        "The patient's sleep evaluation file lists a study from {d}.",
        "The year on the patient's original sleep study report is {d}.",
    ], "The diagnostic report and the equipment settings were requested as separate records."),
    ("screening_test", "audiology_baseline_test", 1990, 2024, "Hearing test history", [
        "The patient completed the baseline hearing test in {d}.",
        "An audiogram in this patient's folder dates from {d}.",
        "The patient recalls formal hearing testing in {d} before trying hearing aids.",
        "The outside audiology summary identifies the patient's earlier test by the year {d}.",
        "The patient brought an audiogram labeled with the examination year {d}.",
    ], "The complete audiogram is needed because the summary does not show the individual thresholds."),
    ("procedure", "appendectomy_history", 1978, 2024, "Appendectomy history", [
        "The patient reports an appendectomy in {d}.",
        "The patient's surgical history places the appendectomy in {d}.",
        "An operative summary confirms that the patient's appendix was removed in {d}.",
        "Asked about abdominal surgery, the patient recalled an appendectomy in {d}.",
        "The patient supplied {d} as the year of the appendix operation.",
    ], "The patient was unsure whether any other abdominal procedure was performed during that admission."),
    ("procedure", "gallbladder_operation", 1980, 2024, "Gallbladder surgery history", [
        "The patient had gallbladder surgery in {d}, according to the discharge summary.",
        "The patient's cholecystectomy is recorded under {d} in the old chart.",
        "The patient recalls gallbladder removal in {d}.",
        "The transferred operative history lists this patient's cholecystectomy in {d}.",
        "The year the patient gave for gallbladder surgery was {d}.",
    ], "The operative report was requested because the brief discharge summary does not describe the approach."),
    ("procedure", "knee_replacement_history", 1990, 2024, "Knee arthroplasty history", [
        "The patient underwent knee replacement in {d}.",
        "The implant card places the patient's knee arthroplasty in {d}.",
        "The patient's prior knee operation is recorded as a replacement performed in {d}.",
        "The patient recalls the knee replacement admission in {d}.",
        "The outside orthopedic summary lists {d} for the patient's knee arthroplasty.",
    ], "The side and implant details will be checked against the operative report."),
    ("procedure", "cataract_extraction_history", 1987, 2024, "Cataract procedure history", [
        "The patient's cataract extraction was performed in {d}.",
        "The patient remembers lens surgery for cataract in {d}.",
        "An eye-surgery card records this patient's cataract operation in {d}.",
        "The patient's cataract procedure appears under {d} in the transfer summary.",
        "The patient supplied a postoperative sheet from the cataract operation in {d}.",
    ], "Laterality will be reconciled with the original eye-surgery record."),
    ("procedure", "pacemaker_implant_record", 1986, 2024, "Device implantation history", [
        "The patient's first pacemaker was implanted in {d}.",
        "The patient's device card records the initial implantation year as {d}.",
        "The patient recalls the pacemaker implantation admission in {d}.",
        "The transferred cardiology history places the first pacemaker procedure in {d}.",
        "An implant summary in this patient's file gives {d} for the original pacemaker procedure.",
    ], "A subsequent generator change is mentioned elsewhere and will be reconciled separately."),
    ("care_timeline", "pneumonia_admission_recollection", 1989, 2024, "Prior admission", [
        "The patient was admitted with pneumonia in {d}.",
        "The patient's earlier pneumonia hospitalization is recorded in {d}.",
        "The patient recalls a hospital stay for pneumonia in {d}.",
        "The discharge summary supplied by the patient concerns a pneumonia admission in {d}.",
        "The patient identified {d} as the year of a previous inpatient stay for pneumonia.",
    ], "The patient does not recall the treatment details and prefers that the discharge record be obtained."),
    ("care_timeline", "rehabilitation_course_history", 1992, 2024, "Rehabilitation course", [
        "The patient attended an outpatient rehabilitation course in {d}.",
        "The patient's earlier therapy records begin in {d}.",
        "The patient places a prior supervised rehabilitation course in {d}.",
        "A rehabilitation discharge sheet in this patient's folder is from {d}.",
        "The patient's old therapy summary records completion of the course in {d}.",
    ], "The home-exercise sheet was retained for the therapist to compare with the current functional goals."),
    ("care_timeline", "antihypertensive_start_history", 1991, 2024, "Medication chronology", [
        "The patient recalls starting blood-pressure medication in {d}.",
        "The patient's medication history places the first antihypertensive prescription in {d}.",
        "The earliest blood-pressure prescription receipt retained by the patient is from {d}.",
        "The patient identified {d} as the year treatment for hypertension began.",
        "A prior clinic summary records initiation of this patient's blood-pressure medication in {d}.",
    ], "The old medication name is uncertain, so the pharmacy history was requested."),
    ("care_timeline", "delivery_history_review", 1985, 2024, "Obstetric history review", [
        "The patient gave birth in {d}, according to the obstetric history provided at intake.",
        "The patient's prior delivery is recorded in {d} in the transferred summary.",
        "The patient recalls the delivery admission in {d}.",
        "The year entered for the patient's previous delivery is {d}.",
        "A retained maternity discharge sheet places this patient's delivery in {d}.",
    ], "Details of the admission will be checked against the source record at the patient's request."),
    ("care_timeline", "infusion_course_record", 1998, 2024, "Infusion course history", [
        "The patient received a course of iron infusions in {d}.",
        "The patient's prior infusion-center record begins in {d}.",
        "The patient remembers attending iron infusion visits in {d}.",
        "The transferred treatment summary lists the patient's iron infusion course in {d}.",
        "A treatment card retained by the patient places the iron infusions in {d}.",
    ], "The original treatment summary was requested to verify the product and total number of visits."),
    ("public_control", "research_methods_teaching", 1980, 2024, "Methods teaching", [
        "The reading folder includes a {y} study selected for discussion of its methods.",
        "The seminar compares the design of a study published in {y} with the proposed classroom exercise.",
        "The journal-club citation list includes a paper from {y}; the group will discuss how its outcomes were defined.",
        "A {y} publication was placed in the teaching folder as an example of a methods section.",
        "For the literature exercise, the group will examine a study whose publication year is {y}.",
    ], "The facilitator asked learners to distinguish a paper's methods from the claims in its abstract."),
    ("public_control", "vaccine_information_versions", 2002, 2024, "Vaccine information library", [
        "The staff reference shelf contains a vaccine-information booklet published in {y}.",
        "An archived {y} vaccine education leaflet is retained for the document-history exercise.",
        "The library index lists {y} as the publication year of the vaccine-information handout.",
        "The teaching packet includes a vaccine-information sheet carrying a {y} revision year.",
        "A staff member found a {y} edition of the vaccine counseling booklet in the reference archive.",
    ], "The exercise concerns the handout's organization, and the team will consult the current source before using it for counseling."),
    ("public_control", "screening_guideline_archive", 1996, 2024, "Guideline archive exercise", [
        "The archive exercise uses a screening-guideline document published in {y}.",
        "A {y} screening guideline appears on the seminar reading list.",
        "The librarian cataloged a screening guidance booklet with a publication year of {y}.",
        "The education session will compare the headings in the {y} edition of a screening guidance document.",
        "The reading packet's guideline citation has the year {y} beside the title.",
    ], "The discussion will not assume that an archived recommendation still applies to current practice."),
    ("public_control", "generic_workflow_equipment", 1998, 2024, "Workflow teaching sheet", [
        "The generic equipment worksheet cites a maintenance manual published in {y}.",
        "The classroom workflow diagram was adapted for a fictional clinic in the {y} training workbook.",
        "The reference list for the equipment exercise includes a {y} manual.",
        "A {y} training handout illustrates the difference between a device model number and a serial number.",
        "The workshop binder labels its sample workflow diagram as a {y} teaching example.",
    ], "The sample device has model number 2014; the image is 2048 pixels wide and the display refresh interval is 2000 milliseconds."),
    ("public_control", "patient_education_editorial", 1995, 2024, "Patient education editing", [
        "The editorial workshop uses a {y} patient-education brochure to discuss readability.",
        "A generic follow-up handout from the {y} print archive is being revised for the writing exercise.",
        "The education team is reviewing a counseling booklet first published in {y}.",
        "The plain-language exercise quotes the headings of a {y} health-education pamphlet.",
        "The reference brochure in the writing workshop has a publication year of {y}.",
    ], "The sample audience description is an adult aged 68, and the workshop timetable begins at 14:30."),
]

# These are record-review sections, not copied progress notes. Each long note
# uses each selected paragraph at most once. They describe work on the same
# record rather than adding fictitious treatment recommendations.
REVIEW_SECTIONS = [
    ("Record inventory", "The patient brought a small folder containing a summary sheet, a folded instruction page, and a copy of an outside result. The pages were sorted by the kind of record they appeared to represent. Several sheets had printing on both sides, so both sides were scanned. The originals were placed back in the folder before the patient left. The summary was kept separate from the underlying report so that the two would not be mistaken for duplicate documents."),
    ("Interview approach", "The history was gathered in the patient's own words before the outside material was reviewed. When the patient could not recall a detail, the field was left open for reconciliation. The interview did not try to convert a general recollection into an exact event. The patient was comfortable with a brief summary being entered while the source material was requested. Questions that required the original record were collected together for the next review."),
    ("Source quality", "Some of the scanned lettering is faint near the edge of the page. The main narrative remains readable, but the small footer cannot be reliably transcribed. Staff will request a clearer copy instead of expanding an abbreviation that could have more than one meaning. The patient recognized the document as part of the folder previously received from the outside office. No conclusion about the missing content was drawn from the appearance of the scan."),
    ("Patient priorities", "The patient would like the chart to contain a usable summary rather than several disconnected fragments. The main concern is having to repeat the same history at future visits. The patient was asked which details seemed most important and which could be checked from records. The resulting questions were written in ordinary language. The patient reviewed the list and asked that the unresolved items remain visible until someone could examine the source documents."),
    ("Reconciliation process", "The short summary and the longer attachment were reviewed together. A repeated sentence was recognized as a copied portion of the same source, so it did not count as a second confirmation. The report title was retained when requesting the original material. Any differences between the patient's recollection and the attachment were left as open questions. This approach allows the next reviewer to see what was said during the interview and what came from a document."),
    ("Communication preferences", "The patient prefers a written explanation of any changes to the history after the remaining records arrive. A brief phone conversation is acceptable if a question cannot be handled in writing. Staff explained that a request being sent does not mean that the receiving office has already supplied the material. The patient understood the distinction and will keep the original folder available in case a clearer photograph or an additional page is needed."),
    ("Attachment review", "The uploaded file opens correctly, and the page order was checked against the paper copy. One page contains instructions while another contains the event summary. They were retained together because the instruction page helps explain the abbreviations used on the summary. A margin mark made by the patient was not transcribed as part of the outside report. The patient explained the mark during the interview, and that explanation was recorded separately in the review notes."),
    ("Unresolved details", "The patient could describe the general sequence of events but did not remember every administrative step. There was no reason to force a more precise account while source records were available to request. The chart review therefore distinguishes a remembered event from an item copied from a formal report. The remaining questions concern the documentation itself. They will be revisited when the full report can be read, rather than answered from a brief heading or a file name."),
    ("Coordination", "The record request was assigned to the staff member handling outside documents. The receiving clinician will see the summary after the attachments have been indexed. If the source office sends only part of the material, staff will identify the missing section before submitting another request. The patient understands that the office may need a signed release. The release process was explained without adding unrelated records to the request or treating a summary sheet as a complete chart."),
    ("Review with the patient", "Before closing the conversation, staff read back the main point of the history and the list of records still needed. The patient corrected the description of the paper folder but did not offer additional clinical details. The revised wording was kept concise so it would remain useful at the next visit. The patient was invited to send a clarification if another document in the folder answered an outstanding question. The originals remained with the patient."),
    ("Follow-through", "The open record request will remain attached to this history item until the response is reviewed. A document arriving in the inbox should not silently replace the patient's account; the reviewer can explain whether it adds detail or resolves a discrepancy. The patient asked for a copy of the final summary once reconciliation is complete. Staff noted that preference in the communication plan and confirmed that no additional recollection was needed to submit the initial request."),
    ("Chart organization", "The history item was filed in the section that best matches the event described by the patient. A link to the outside document was kept beside the summary so a future clinician can inspect the source. The brief display in the chart is intended to help locate the material, not reproduce every line of the attachment. The review remains open for source confirmation. The patient agreed that this was a reasonable way to preserve the history while avoiding repeated transcription."),
]
PUBLIC_SECTIONS = [
    ("Reading order", "The teaching packet is arranged so that readers first see the title page, then the example passage, and finally the discussion prompts. This order allows the facilitator to ask what the source actually says before the group proposes edits. Participants can move back and forth between the passage and the reference list. The exercise does not require access to any clinic chart, and the packet can be used in either a small seminar or a writing workshop."),
    ("Editorial purpose", "The group will focus on how the document communicates its purpose to a reader. A heading may promise an explanation that the paragraph never supplies, while a useful definition may be buried in a footnote. Participants will mark those editorial problems and suggest a clearer order. The workshop does not ask them to supply new treatment advice. Any substantive clinical question that arises will be noted for review against an appropriate current source after the session."),
    ("Citation exercise", "The reference list is part of the reading task. Participants will check whether a citation points to a complete source and whether the text distinguishes a quotation from a summary. A source should be described accurately even when the group disagrees with the way it was used. The facilitator will collect examples of ambiguous attribution for discussion. The exercise concerns the written argument and does not require the group to reconstruct an individual person's care."),
    ("Layout review", "The sample pages use a heading, a short explanatory paragraph, and a box containing questions. The group will assess whether that sequence helps a reader locate the main point. Some participants may prefer the questions before the explanation, so both arrangements can be discussed. The layout exercise uses the same words in each version. This makes it easier to distinguish the effect of presentation from the effect of adding or removing substantive content."),
    ("Terminology", "Abbreviations in the reading packet will be expanded on first use where the expansion is necessary for comprehension. A glossary can support the discussion, but it should not substitute for a clear explanation in the body of the document. Participants will identify terms that are ordinary to staff but unfamiliar to the intended audience. They can propose plainer wording and then check that the replacement preserves the meaning of the original sentence."),
    ("Group discussion", "The facilitator will ask each group to explain one proposed edit and the reason for it. Other participants can point to wording that supports a different interpretation. The aim is to make editorial choices inspectable rather than settle them by preference alone. When a disagreement depends on a clinical fact, the issue will be set aside for source checking. The remaining discussion can continue with questions about organization, emphasis, and clarity."),
    ("Accessible presentation", "The projected example uses generous spacing so that the audience can read it from the back of the room. The paper version preserves the same heading order and includes room for handwritten comments. Color is used sparingly because the packet should remain understandable when printed without color. The facilitator will read short quoted passages aloud before inviting discussion. Participants can review the printed page instead of relying entirely on the projected slide."),
    ("Source handling", "The archived example is retained in its original form alongside the exercise copy. Proposed editorial changes appear only in the exercise copy, allowing readers to compare the two versions. The reference entry remains attached to both. If a page is missing or illegible, the facilitator will obtain another copy rather than fill the gap from memory. The group can still discuss the available passages without attributing invented wording to the original source."),
    ("Writing practice", "Participants will draft a short opening paragraph for the intended audience and then exchange drafts. The reader will identify the main point before suggesting changes. If the main point is difficult to find, the writer can revise the order of the sentences. The exercise emphasizes a clear relationship between a question and its explanation. It does not reward longer prose simply because more technical vocabulary has been included."),
    ("Reference checking", "The final exercise copy will retain a reference trail for any factual statement carried over from the source. A clear source trail helps a later editor distinguish a stylistic revision from a change in meaning. The workshop notes will record unresolved questions without treating them as settled claims. A designated reviewer can then check those questions using the full source material. The packet remains a teaching aid rather than an instruction to change clinical practice."),
    ("Closing activity", "Each group will identify one feature worth retaining and one that should be revised. The facilitator will collect the suggestions without assuming that the most popular edit is automatically the clearest. Participants will be encouraged to explain what a reader could misunderstand and how the proposed wording addresses that problem. The final discussion returns to the intended audience and purpose of the handout so that the session ends with a concrete editorial task."),
    ("Materials inventory", "The workshop folder contains the source excerpt, an exercise copy, a reference sheet, and a blank response form. Spare copies are kept beside the room entrance. The response form asks about the clarity of the exercise and leaves space for an example revision. The facilitator will review that feedback before using the packet again. Loose pages will be returned to their labeled sections so the next group receives the complete set of materials."),
]

FIRST = ("Amira", "Belen", "Celia", "Dario", "Elena", "Farid", "Galen", "Hana", "Inez", "Jorin",
         "Kiran", "Livia", "Mateo", "Nadia", "Oren", "Paloma", "Quinn", "Raisa", "Soren", "Talia",
         "Uma", "Vera", "Willa", "Xavi", "Yara", "Zev", "Adela", "Bruno", "Cora", "Devon",
         "Éloi", "Fenna", "Greta", "Hugo", "Ilona", "Jules", "Kaia", "León", "Mina", "Nico",
         "Odette", "Petra", "Ruben", "Salma", "Tomas", "Viola", "Wren", "Zola")
LAST = ("Amberridge", "Bellacove", "Cedarvale", "Dunmere", "Elmstead", "Fernlake", "Glenwicket", "Hawkrill",
        "Ivydale", "Juniperford", "Kestrelbank", "Larkmere", "Mossbridge", "Northfen", "Oakbarrow", "Pineholt",
        "Quillhaven", "Reedwick", "Silvergrove", "Thornacre", "Umberfield", "Valebrook", "Willowcrest", "Yewcross",
        "Alderquay", "Brightfern", "Cloudmere", "Dovefield", "Evenbrook", "Fallowridge", "Goldwillow", "Hearthvale",
        "Ironglen", "Jasperwood", "Kindleford", "Leafstone", "Meadowfen", "Nightbrook", "Orchardwren", "Pebblecross",
        "Quietmere", "Rainhollow", "Stonefinch", "Tansybank", "Underleaf", "Velvetgrove", "Windquay", "Yarrowdell")
FACILITIES = ("Amber Kestrel Clinic", "Willow Lantern Health Center", "Quiet Cove Medical Office",
              "Silver Acorn Outpatient Center", "Moss Lantern Community Clinic", "Fern Quay Medical Center")
REFERENCE_LINES = (
    "The staff reading folder also contains a study published in {y}; its methods are being discussed at journal club.",
    "A {y} edition of a counseling booklet is cited in the staff reference list.",
    "The attached teaching bibliography includes a {y} publication about record-review methods.",
    "An archived guideline document from {y} is listed in the educational handout's references.",
)


def balanced_reference_order():
    """Choose order without using a year, lexical content, or personal identity.

    Each clinical family has fourteen mixed-role examples: six layouts occur
    twice and two occur once. Give each repeated layout both orders. Alternate
    the two singleton layouts across a shuffled, balanced set of families.
    This yields seven examples of each order per family and exact balance in
    every layout over the corpus; neither family nor layout predicts order.
    """
    rng = random.Random(9032902)
    singleton = [True] * 15 + [False] * 15
    rng.shuffle(singleton)
    assignments = {}
    for fi in range(30):
        for v in (0, 3, 6, 9, 12, 15):
            first = bool(rng.randrange(2))
            assignments[(fi, v)] = first
            assignments[(fi, v + 24)] = not first
        assignments[(fi, 18)] = singleton[fi]
        assignments[(fi, 21)] = not singleton[fi]
    return assignments


PUBLIC_FIRST = balanced_reference_order()
ORIGINAL_PINS = {
    "scripts/takeover/author9_clinical_years.py": "a7a500cb655bbc1d6c9449fe7c77c4f73cd4b93a03c6dd84b5dd13484ce4f058",
    "data/tk9_train_clinical_years.jsonl": "423cd75ce229488528114b2b729b30f234b97a3839a55dfaa633365753d517ba",
    "data/tk9_train_clinical_years.jsonl.manifest.json": "271908a0a5c52549839f4fde6823834db2aeb7666cc8003947056943f684c3bc",
    "reports/takeover/mix3_001/author9_clinical_years/author_report.json": "e0b6883ce4e9750e76f854db825bdd92dcf2022a435f136fe88a6ce1488748d9",
}


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def group(name):
    return sha(unicodedata.normalize("NFC", name).casefold().encode("utf-8"))


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def runtime_functions():
    # Execute just the actual runtime's three tokenizer/window functions; do
    # not import the detector, model module, torch, or any model weights.
    source = (ROOT / "src/outis/detection/hf_encoder.py").read_text()
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"window_jobs", "special_layout", "encode_text"}]
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "hf_encoder_tokenizer_functions", "exec"), namespace)
    return namespace


def marked(label, value):
    return f"[[{label}|{value}]]"


def unmark(value):
    chunks, spans, last, position = [], [], 0, 0
    for match in re.finditer(r"\[\[([A-Z]+)\|([^\[\]]+)\]\]", value):
        lead = value[last:match.start()]
        chunks.extend((lead, match[2]))
        position += len(lead)
        spans.append({"start": position, "end": position + len(match[2]), "label": match[1], "text": match[2]})
        position += len(match[2])
        last = match.end()
    chunks.append(value[last:])
    text = "".join(chunks)
    assert "[[" not in text and "]]" not in text
    return text, spans


def take_names():
    candidates = [f"{a} {b}" for a in FIRST for b in LAST]
    random.Random(90317).shuffle(candidates)
    # This is rejection sampling on complete identity hashes, never a letter
    # partition. The candidate pool and ordering are fixed for reproducibility.
    return iter(name for name in candidates if int(group(name)[:8], 16) % 10 < 8)


def layout(title, focus, context, variant, public):
    k = variant % len(LAYOUTS)
    if public:
        # Do not manufacture a patient conversation around a generic source.
        formats = (
            "{title}\n\n{focus} {context}",
            "Workshop briefing\nTopic: {title}\nDiscussion material: {focus}\nFacilitator's note: {context}",
            "RESOURCE REVIEW — {title}\nReading selection\n{focus}\nEditorial note\n{context}",
            "Reference worksheet | {title}\nSource entry | {focus}\nReview task | {context}",
            "Message to the education team\nRe: {title}\n\n{focus}\n\n{context}",
            "Teaching handoff: {title}\nMaterials: {focus}\nSession plan: {context}",
            "{title}\nWhat will the group read? {focus}\nWhat needs discussion? {context}",
            "{title}\n• Reading selection: {focus}\n• Discussion task: {context}",
        )
    else:
        formats = (
            "{title}\n\n{focus} {context}",
            "Telephone review — {title}\nNurse summary: {focus}\nFollow-up note: {context}",
            "HISTORY REVIEW\n{title}\n\nHistory obtained\n{focus}\n\nRecord plan\n{context}",
            "Chart reconciliation | {title}\nHistory entry | {focus}\nRecords needed | {context}",
            "Portal review\nSubject: {title}\n\nThe message and the attached material were reviewed. {focus}\n\n{context}",
            "Handoff: {title}\nHistory for the receiving clinician: {focus}\nOutstanding work: {context}",
            "{title}\nWhat history was obtained? {focus}\nWhat remains to be checked? {context}",
            "{title}\n• History reviewed: {focus}\n• Records follow-up: {context}",
        )
    return formats[k].format(title=title, focus=focus, context=context)


def build_rows(tokenizer, runtime, content_window):
    names = take_names()
    rows = []
    for fi, (category, family, low, high, title, formulations, context) in enumerate(FAMILIES):
        public = category == "public_control"
        for v in range(40):
            number = fi * 40 + v + 1
            year = low + ((v * 7 + fi * 3 + v // 5) % (high - low + 1))
            rid = f"tk9-clinical-years-{number:04d}"
            people = [] if public else [next(names)]
            formulation = formulations[(v + v // 5) % 5]
            partial = v % 8 == 0 and "year" not in formulation
            date = f"{('March', 'June', 'September', 'November')[v // 8 % 4]} {year}" if partial else str(year)
            focus = formulation.format(d=marked("DATE", date), y=year)
            primary_text, primary_spans = unmark(focus)
            primary_literal = str(year) if public else primary_spans[0]["text"]
            review_context = context
            if family == "screening_guideline_archive":
                # Removal of the unrelated answer-key sentence exposed twelve
                # duplicate pairs in this family. Vary the actual editorial
                # task, preserving the reference year and all clinical meaning.
                review_context = (
                    context,
                    "Participants will review the archived wording in its original context before consulting current screening guidance.",
                    "The group will identify passages that would need a fresh source check before being reused in a current handout.",
                    "The librarian will keep the archived citation beside the exercise copy so readers can distinguish the historical source from later revisions.",
                    "The facilitator asked the group to discuss the document's structure and leave questions about current practice for source review.",
                )[v // 8]
            body = layout(title, focus, review_context, v, public)
            phenomena = ["contextual_year", category, "public_reference_year" if public else "patient_history_year"]
            if not public:
                phenomena.append("partial_date" if partial else "standalone_four_digit_year")
                if v % 3 == 0:
                    reference = REFERENCE_LINES[(v // 3 + fi) % len(REFERENCE_LINES)].format(y=year)
                    body = reference + "\n\n" + body if PUBLIC_FIRST[(fi, v)] else body + "\n\n" + reference
                    phenomena.extend(("same_year_public_and_personal", "public_reference_year"))
                if v % 5 == 1:
                    body = f"Patient: {marked('NAME', people[0])}\n" + body
                    phenomena.append("personal_name")
                if v % 13 == 2:
                    provider = next(names)
                    people.append(provider)
                    body += f"\nReviewed by Dr. {marked('NAME', provider)}, MD."
                    phenomena.extend(("provider_name", "credential_boundary"))
                if v % 17 == 3:
                    caregiver = next(names)
                    people.append(caregiver)
                    caregiver_year = str(max(1980, year - 1))
                    body += (f"\nThe patient's caregiver, {marked('NAME', caregiver)}, said, "
                             f"‘I began helping with the record folder in {marked('DATE', caregiver_year)}.’")
                    phenomena.extend(("caregiver_name", "caregiver_timeline_year"))
                if v % 19 == 4:
                    provider = next(names)
                    people.append(provider)
                    body += (f"\nDr. {marked('NAME', provider)} recalled reviewing this patient's folder in "
                             f"{marked('DATE', str(min(2025, year + 1)))} and offered to check the archived correspondence.")
                    phenomena.extend(("provider_name", "provider_patient_care_year"))
                if v % 10 == 2:
                    body += f"\nRequested source office: {marked('FACILITY', FACILITIES[(fi + v) % len(FACILITIES)])}."
                    phenomena.append("facility_name")
                if v % 10 == 3:
                    # Invented patient-specific tracking identifiers can look
                    # like years; the caption establishes their identifier role.
                    body += f"\nThis patient's record-request tracking number: {marked('ID', str(2000 + (number % 25)))}."
                    phenomena.append("four_digit_personal_identifier")
                if v % 20 == 6:
                    body += f"\nMRN: {marked('MRN', f'SYN9-{number:07d}')}."
                    phenomena.append("invented_mrn")
                if v % 20 == 7:
                    day = 10 + number % 18
                    body += f"\nA record-review visit was booked for {marked('DATE', f'October {day}, 2025')}."
                    phenomena.append("complete_personal_date")
                if v % 11 == 5:
                    body += "\nThe uploaded image is 2048 pixels wide. The call to discuss the folder lasted 14 minutes."
                    phenomena.extend(("measurement_not_year", "duration_not_date"))
                if v % 20 == 9:
                    body += f"\nReturn number supplied by the patient: {marked('PHONE', f'(202) 555-{100 + number % 100:04d}')}."
                    phenomena.append("reserved_phone")
            else:
                phenomena.append("phi_free_reference")
                if family == "generic_workflow_equipment":
                    phenomena.extend(("generic_identifier", "duration_not_date", "device_model_not_personal_id", "measurement_not_year"))
                if family == "patient_education_editorial":
                    phenomena.extend(("ordinary_age", "public_clock_time"))
            is_long = v % 5 == 0
            if is_long:
                sections = PUBLIC_SECTIONS if public else REVIEW_SECTIONS
                rotation = (fi + v // 5) % len(sections)
                ordered = sections[rotation:] + sections[:rotation]
                # Nine to twelve substantive sections, selected without repeats.
                selected = [f"{heading}\n{paragraph}" for heading, paragraph in ordered[:9]]
                pos = (v // 5) % 4
                insertion = (0, 3, 6, len(selected))[pos]
                selected.insert(insertion, body)
                long_marked = "\n\n".join(selected)
                used = 9
                while len(tokenizer(unmark(long_marked)[0], add_special_tokens=False)["input_ids"]) < 1000:
                    heading, paragraph = ordered[used]
                    # Added material goes before the late target and after all
                    # other placements, preserving the planned target variation.
                    chunk = f"{heading}\n{paragraph}"
                    if pos == 3:
                        selected.insert(len(selected) - 1, chunk)
                    else:
                        selected.append(chunk)
                    used += 1
                    long_marked = "\n\n".join(selected)
                body = long_marked
                phenomena.extend(("multi_window_record_review", "varied_year_window_ownership"))
            text, spans = unmark(body)
            groups = [group(name) for name in people]
            row = {
                "id": rid, "text": text, "spans": spans, "split": "train", "suite": "train",
                "status": "training_draft", "independent_qa_complete": False,
                "annotation_policy": POLICY, "identity_group": groups[0] if groups else f"public-teaching-{rid}",
                "identity_groups": groups, "full_identities": people,
                "pair_id": f"tk9-unpaired-{number:04d}", "template_family": f"tk9-{family}",
                "family": f"tk9-{family}", "clinical_category": category, "layout": LAYOUTS[v % 8],
                "phenomena": list(dict.fromkeys(phenomena)), "cohort": "long_document" if is_long else "realistic",
                "negative_only": not spans, "synthetic": True, "author": "author9_clinical_years",
                "primary_year_literal": primary_literal, "primary_year_role": "public_reference" if public else "personal_date",
            }
            if people:
                row["full_identity"] = people[0]
            ids, offsets = runtime["encode_text"](tokenizer, text)
            jobs = list(runtime["window_jobs"](len(ids), content_window))
            date_positions = []
            for s in spans:
                if s["label"] != "DATE":
                    continue
                token_indexes = [i for i, (a, b) in enumerate(offsets) if b > s["start"] and a < s["end"]]
                owners = sorted({j for i in token_indexes for j, (_, _, left, right) in enumerate(jobs) if left <= i < right})
                date_positions.append({"start": s["start"], "end": s["end"], "text": s["text"],
                                       "token_start": min(token_indexes), "token_end": max(token_indexes) + 1,
                                       "owning_windows": owners})
            row["runtime_census"] = {"token_count": len(ids), "window_count": len(jobs),
                                     "content_window": content_window, "date_ownership": date_positions}
            assert not is_long or len(jobs) >= 3
            rows.append(row)
    return rows


def check_rows(rows, convention, intervals, mixture):
    assert len(rows) == 1400
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["text"] for r in rows}) == len(rows), "duplicate note text"
    assert len({r["template_family"] for r in rows}) == 35
    normalization = Counter()
    interval_changes = []
    for row in rows:
        mixture.validate_row(row, {"annotation_policy": POLICY}, training=True)
        assert row["split"] == row["suite"] == "train" and row["status"] == "training_draft"
        assert row["independent_qa_complete"] is False
        assert len(row["identity_groups"]) == len(set(row["identity_groups"]))
        assert row["identity_groups"] == [group(n) for n in row["full_identities"]]
        assert all(int(g[:8], 16) % 10 < 8 for g in row["identity_groups"])
        assert all(s["text"] == s["text"].strip() for s in row["spans"])
        assert len(row["phenomena"]) == len(set(row["phenomena"]))
        normalized = convention.normalize(deepcopy(row), normalization)
        assert normalized["spans"] == row["spans"], (row["id"], "convention change")
        normalized, changes, _ = intervals.normalize_row(deepcopy(row))
        interval_changes.extend(changes)
        assert normalized["spans"] == row["spans"], (row["id"], "interval change")
        for person in row["full_identities"]:
            for match in re.finditer(re.escape(person), row["text"]):
                assert any(s["start"] == match.start() and s["end"] == match.end() and s["label"] == "NAME"
                           for s in row["spans"]), (row["id"], "unlabeled full name")
        assert row["negative_only"] == (not row["spans"])
        if row["primary_year_role"] == "public_reference":
            assert not row["spans"]
        else:
            assert any(s["label"] == "DATE" and s["text"] == row["primary_year_literal"] for s in row["spans"])
    assert not normalization and not interval_changes
    assert sum(r["runtime_census"]["window_count"] >= 3 and bool(r["spans"]) for r in rows) >= len(rows) * .15
    return {"structural_rows_checked": len(rows), "duplicate_ids": 0, "duplicate_texts": 0,
            "normalization_changes": dict(normalization), "date_interval_changes": len(interval_changes),
            "identity_split_violations": 0, "all_train_role_guards_passed": True}


def revision_checks(rows):
    artifacts, blobs = [], {}
    for relative, pin in ORIGINAL_PINS.items():
        path = ROOT / relative
        info = path.stat()
        assert path.is_file() and not path.is_symlink() and info.st_nlink == 1
        assert stat.S_IMODE(info.st_mode) == 0o444
        blob = path.read_bytes()
        assert sha(blob) == pin, f"original changed: {relative}"
        blobs[relative] = blob
        artifacts.append({"path": relative, "sha256": pin, "mode": "0444"})
    original = [json.loads(line) for line in blobs["data/tk9_train_clinical_years.jsonl"].decode("utf-8").splitlines()]
    assert len(original) == len(rows)
    changed_text, changed_spans, changed_records = [], [], []
    unchanged_identities = ("identity_group", "identity_groups", "full_identities", "full_identity")
    for before, after in zip(original, rows):
        assert before["id"] == after["id"]
        assert all(before.get(key) == after.get(key) for key in unchanged_identities)
        assert [(s["label"], s["text"]) for s in before["spans"]] == [(s["label"], s["text"]) for s in after["spans"]]
        assert all(marker not in after["text"] for marker in ("The exercise answer key uses item number", "the workshop reserves 20 minutes"))
        if before["text"] != after["text"]:
            changed_text.append(after["id"])
        if before["spans"] != after["spans"]:
            changed_spans.append(after["id"])
        if before != after:
            changed_records.append(after["id"])
    assert len(changed_text) == 410 and len(changed_spans) == 210
    assert len(changed_records) == 410
    by_family, by_layout, by_year = {}, {}, {}
    order_rows = []
    for i, row in enumerate(rows):
        if "same_year_public_and_personal" not in row["phenomena"]:
            continue
        fi, v = divmod(i, 40)
        year = row["primary_year_literal"][-4:]
        reference = REFERENCE_LINES[(v // 3 + fi) % len(REFERENCE_LINES)].format(y=year)
        reference_start = row["text"].index(reference)
        reference_year_start = reference_start + reference.index(year)
        assert all(s["end"] <= reference_year_start or s["start"] >= reference_year_start + 4 for s in row["spans"])
        primary = next(s for s in row["spans"] if s["label"] == "DATE" and s["text"] == row["primary_year_literal"])
        order = "public_before_personal" if reference_start < primary["start"] else "personal_before_public"
        assert (order == "public_before_personal") == PUBLIC_FIRST[(fi, v)]
        for mapping, key in ((by_family, row["template_family"]), (by_layout, row["layout"]), (by_year, year)):
            mapping.setdefault(key, Counter())[order] += 1
        order_rows.append({"id": row["id"], "layout": row["layout"], "order": order})
    for counts in by_family.values():
        assert counts == {"public_before_personal": 7, "personal_before_public": 7}
    for counts in by_layout.values():
        assert counts["public_before_personal"] == counts["personal_before_public"]
    assert sum(x["public_before_personal"] for x in by_family.values()) == 210
    samples = []
    for order in ("public_before_personal", "personal_before_public"):
        for layout_name in LAYOUTS:
            samples.append(next(r["id"] for r in order_rows if r["order"] == order and r["layout"] == layout_name))
    # Two revised controls per family, one short and one long.
    samples += [rows[fi * 40 + v]["id"] for fi in range(30, 35) for v in (5, 8)]
    samples += [rows[32 * 40 + v]["id"] for v in (16, 24, 32)]
    assert len(samples) == len(set(samples)) == 29
    return {
        "revision": 2, "predecessor_artifacts": artifacts,
        "reason": "Prospective authoring-quality correction: remove the unrelated all-control marker and counterbalance public/personal reference order.",
        "original_artifacts_preserved": True,
        "changed_text_rows": len(changed_text), "changed_gold_offset_rows": len(changed_spans),
        "changed_record_rows": len(changed_records), "unchanged_record_rows": len(rows) - len(changed_records),
        "control_marker_removed_rows": 200, "mixed_role_order_reversed_rows": 210,
        "control_context_reworded_rows": 32,
        "control_context_rewording_reason": "Removing the unrelated control marker exposed twelve duplicate pairs in the guideline-reference family; varied source-review tasks replace the single repeated context sentence.",
        "gold_label_and_literal_changes": 0, "identity_changes": 0,
        "row_ids_preserved": True, "revision_replaces_original_for_future_admission": True,
        "changed_text_row_ids": changed_text,
        "order_support": {"public_before_personal": 210, "personal_before_public": 210},
        "order_by_family": {k: dict(v) for k, v in sorted(by_family.items())},
        "order_by_layout": {k: dict(v) for k, v in sorted(by_layout.items())},
        "order_by_year": {k: dict(v) for k, v in sorted(by_year.items())},
        "order_assignment": "Seeded choices independent of year and identity, balanced within every family and across every layout; repeated family/layout cells contain both orders.",
        "removed_marker_remaining_rows": 0,
        "remaining_limitations": ["Partial dates remain concentrated in the narrative layout; this focused revision does not alter that distribution.",
                                  "Clinical and public-reference families retain their natural differences in document purpose and vocabulary.",
                                  "Long notes remain disclosed synthetic compositions rather than naturally sampled clinical records."],
        "manual_sample_ids": samples,
    }


def main():
    manifest_path = OUT.with_suffix(OUT.suffix + ".manifest.json")
    for target in (OUT, manifest_path, REPORT):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing overwrite: {target}")
    convention = module("author9_conventions", "scripts/takeover/normalize_conventions.py")
    intervals = module("author9_intervals", "scripts/takeover/normalize_date_intervals.py")
    mixture = module("author9_mixture", "scripts/takeover/prepare_training_mixture.py")
    runtime = runtime_functions()
    tokenizer = AutoTokenizer.from_pretrained(MODEL / "hf", local_files_only=True)
    config = json.loads((MODEL / "config.json").read_text())
    prefix, suffix = runtime["special_layout"](tokenizer)
    content_window = config["max_length"] - len(prefix) - len(suffix)
    rows = build_rows(tokenizer, runtime, content_window)
    checks = check_rows(rows, convention, intervals, mixture)
    lineage = revision_checks(rows)
    support = Counter(s["label"] for r in rows for s in r["spans"])
    count = lambda key: dict(sorted(Counter(r[key] for r in rows).items()))
    token_counts = sorted(r["runtime_census"]["token_count"] for r in rows)
    long_rows = [r for r in rows if r["runtime_census"]["window_count"] >= 3]
    owners = Counter(w for r in long_rows for d in r["runtime_census"]["date_ownership"] for w in d["owning_windows"])
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode("utf-8")
    replayed_rows = build_rows(tokenizer, runtime, content_window)
    replayed_payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for r in replayed_rows).encode("utf-8")
    assert replayed_payload == payload
    checks["deterministic_corpus_replay"] = True
    corpus_sha = sha(payload)
    relative = lambda p: str(p.relative_to(ROOT))
    allowed_inputs = [ROOT / "docs/annotation-policy-v2.md", ROOT / "src/outis/data/annotation_policy.py",
                      ROOT / "scripts/takeover/normalize_conventions.py", ROOT / "scripts/takeover/normalize_date_intervals.py",
                      ROOT / "scripts/takeover/prepare_training_mixture.py", ROOT / "src/outis/detection/hf_encoder.py",
                      MODEL / "config.json", MODEL / "hf/config.json", MODEL / "hf/tokenizer.json", MODEL / "hf/tokenizer_config.json"]
    provenance = {relative(p): sha(p.read_bytes()) for p in allowed_inputs}
    common = {
        "schema_version": 1, "annotation_policy": POLICY, "split": "train", "suite": "train",
        "status": "training_draft", "independent_qa_complete": False, "synthetic": True,
        "rows": len(rows), "documents": len(rows), "sha256": corpus_sha,
        "label_support": {label: support[label] for label in LABELS},
        "entity_support": {label: support[label] for label in LABELS},
        "label_counts": {label: support[label] for label in LABELS},
        "cohort_counts": count("cohort"), "category_counts": count("clinical_category"),
        "family_counts": count("template_family"), "layout_counts": count("layout"),
        "phenomenon_support": dict(sorted(Counter(p for r in rows for p in r["phenomena"]).items())),
        "phi_free_rows": sum(not r["spans"] for r in rows),
        "purpose": "Prospective focused contextual clinical-year training preparation; separate QA is pending.",
        "generator": relative(Path(__file__)), "generator_sha256": sha(Path(__file__).read_bytes()),
        "source_policy": "Prospective authoring-quality revision of the same author's frozen original; no outside corpus, model prediction, or performance report used.",
        "allowed_input_sha256": provenance,
        "identity_scheme": "SHA256 of UTF8 NFC(full_name).casefold(); full-name candidate rejection sampling accepts int(digest[:8],16)%10<8.",
        "synthetic_identifiers": {"MRN": "SYN9- followed by seven invented digits", "ID": "Invented four-digit record-request tracking numbers", "PHONE": "Reserved 202-555-0100 through 202-555-0199", "FACILITY": "Newly invented office names"},
        "context_disclosure": "35 authored clinical/reference families, five formulations each, eight layouts. Long documents compose distinct record-review or teaching sections; they are synthetic expansions rather than naturally sampled records.",
        "runtime_census": {"tokenizer": relative(MODEL / "hf"), "model_weights_loaded": False,
                           "max_length": config["max_length"], "content_window": content_window,
                           "window_function": "Actual hf_encoder.py window_jobs extracted via AST",
                           "token_count_min": token_counts[0], "token_count_median": token_counts[len(rows)//2],
                           "token_count_max": token_counts[-1],
                           "window_count_support": {str(k): v for k, v in sorted(Counter(r["runtime_census"]["window_count"] for r in rows).items())},
                           "at_least_three_window_rows": len(long_rows),
                           "at_least_three_window_personal_year_rows": sum(bool(r["spans"]) for r in long_rows),
                           "long_date_owning_window_support": {str(k): v for k, v in sorted(owners.items())}},
        "checks": checks, "current_frozen_mixture_modified": False,
        "revision_lineage": lineage,
    }
    manifest = dict(common, corpus=relative(OUT), author="author9_clinical_years")
    report = dict(common, corpus=relative(OUT), manifest=relative(manifest_path),
                  author_manual_review={
                      "complete": True,
                      "method": "Author spot review of 29 generated note texts or long-note changed-context excerpts: both public/personal orders in all eight layouts, one short plus one long control from each public family, and all added guideline-review context variants.",
                      "sample_ids": lineage["manual_sample_ids"],
                      "review_scope": "Counterbalanced occurrence order, correct DATE exclusion for public years, exact offsets after reordering, and control coherence after marker removal.",
                      "independent_assessment": False,
                      "limitations": "A separate reader must review the frozen draft. Source composition and author checks do not establish clinical performance or qualification readiness.",
                  })
    mixture.check_metadata(manifest, training=True, require_split=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(payload)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"corpus": str(OUT), "sha256": corpus_sha, "rows": len(rows),
                      "label_support": dict(support), "runtime_census": common["runtime_census"],
                      "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
