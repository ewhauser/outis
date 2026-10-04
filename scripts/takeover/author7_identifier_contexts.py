#!/usr/bin/env python3
"""Independently authored TRAINING-ONLY identifier-context draft; never admits a bank.

Reads only the explicitly named local tokenizer/config files and this source.
No model weights, inference, network, historical corpus, or evaluation input.
Default writes three NEW artifacts; --verify compares the reproducible bytes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import re
import unicodedata

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/takeover-mbbase-mix3-001/step-003081"
CORPUS = ROOT / "data/tk7_train_identifier_contexts.jsonl"
MANIFEST = Path(str(CORPUS) + ".manifest.json")
REPORT = ROOT / "reports/takeover/mix3_001/author7_identifier_contexts/author_report.json"
LABELS = ("NAME", "DATE", "ADDRESS", "PHONE", "EMAIL", "MRN", "SSN", "ACCOUNT", "ID", "URL", "IP", "AGE", "LOCATION", "FACILITY")
TARGETS = ("MRN", "ACCOUNT", "ID")
SHAPES = ("digits_6", "zero_padded_digits_8", "digits_10", "zero_padded_digits_12", "letters2_digits5", "digits3_letters2_digits3", "letters2_hyphen_digits6", "digits3_hyphen3_hyphen3", "letters2_underscore_digits4", "digits4_slash_letters2_digits3")
LAYOUTS = ("narrative", "form", "bullets", "table", "handoff")
ORDERS = tuple(itertools.permutations(TARGETS))
SUBSETS = (("MRN",), ("ACCOUNT",), ("ID",), ("MRN", "ACCOUNT"), ("MRN", "ID"), ("ACCOUNT", "ID"))

# Each family was authored here independently. A field has an explicit semantic
# caption and a prose construction; the actual value never contains a label tag.
# title, clinical opening, clinical follow-up, MRN caption/prose, ACCOUNT caption/prose, ID caption/prose
FAMILIES = (
 ("referral_intake", "Specialty referral intake", "The referral describes intermittent hand tingling without new weakness. The patient can still button clothing and carry a light bag.", "The intake nurse reviewed symptom progression and the reasons to seek an earlier assessment.",
  ("Longitudinal patient chart number", "The referral was linked to the patient's longitudinal medical chart using {v}."), ("Patient billing account number", "The billing clerk assigned the referral charges to patient account {v}."), ("Referral request identifier", "The referring team's request carries identifier {v}.")),
 ("infusion_precheck", "Infusion readiness review", "The patient reports adequate fluid intake and no chills. The treatment team reviewed the recent symptom checklist before releasing the chair assignment.", "Observation instructions and the route for reporting a delayed reaction were reviewed aloud.",
  ("Patient medical record number", "The infusion nurse opened the patient's medical record under number {v}."), ("Infusion billing account", "The scheduled infusion is billed through account {v}."), ("Treatment authorization request ID", "The authorization request for this treatment is tracked as {v}.")),
 ("imaging_schedule", "Imaging preparation call", "The patient understands the positioning instructions and can remain supine with a cushion. Questions about clothing and removable metal items were addressed.", "The scheduler checked comprehension of the arrival and preparation instructions.",
  ("Medical chart lookup number", "To locate the patient's medical chart, the scheduler used lookup number {v}."), ("Imaging services account number", "Imaging charges will post to account number {v}."), ("Imaging order ID", "The ordering clinician's imaging order has ID {v}.")),
 ("discharge_transport", "Discharge transport coordination", "The patient completed a short supervised walk and is comfortable sitting upright. The team is arranging assistance for the transition from the unit.", "A seated rest break and slow position changes were included in the discharge teaching.",
  ("Hospital medical record number", "The transport coordinator located the patient's hospital chart with medical record number {v}."), ("Hospital billing account number", "The discharge ledger uses hospital billing account {v}."), ("Transport booking reference", "The wheelchair transport booking is filed under reference {v}.")),
 ("pharmacy_reconcile", "Medication reconciliation", "The patient describes taking the morning medicines with breakfast. The nurse compared the spoken routine with the active medication list and reviewed how to handle a missed dose.", "The patient used teach-back to describe the updated medication routine.",
  ("Patient chart identifier", "The medication review belongs to the patient's chart identified by {v}."), ("Dispensing account number", "Dispensing charges are attached to patient account {v}."), ("Medication review task ID", "The reconciliation task is identified in the work queue as {v}.")),
 ("sleep_equipment", "Sleep equipment teaching", "The patient practiced adjusting the mask while seated and described mild air leakage near the cheek. The fit improved after the headgear was loosened slightly.", "Cleaning and drying steps were demonstrated without changing the prescribed settings.",
  ("Sleep clinic medical record number", "The equipment educator retrieved the patient's sleep clinic chart using medical record number {v}."), ("Equipment rental account", "The rental agreement is billed to equipment account {v}."), ("Equipment service ticket ID", "The mask adjustment service ticket has ID {v}.")),
 ("rehab_goal_review", "Rehabilitation goal review", "The patient reports difficulty standing long enough to prepare a simple meal. Therapy focused on pacing, a stable work surface, and a brief seated rest before fatigue builds.", "Progress will be judged by safe task completion rather than speed alone.",
  ("Rehabilitation patient chart number", "The therapist saved the goal review in patient medical chart {v}."), ("Therapy billing account identifier", "The therapy ledger names billing account {v}."), ("Therapy plan approval ID", "The revised plan's approval request is identified as {v}.")),
 ("wound_supply", "Wound supply coordination", "The patient can describe the dressing sequence and reports no new drainage through the outer layer. The nurse reviewed gentle removal and protection of the surrounding skin.", "The care plan includes reassessment if discomfort or drainage increases.",
  ("Patient's medical record identifier", "The wound nurse matched the encounter to the patient's medical record identifier {v}."), ("Supply billing account", "The dressing shipment will be charged to supply account {v}."), ("Supply dispatch request number", "The dressing dispatch request has tracking identifier {v}.")),
 ("interpreter_booking", "Interpreter coordination", "The patient asked for extra time to discuss the home care plan. The team arranged language support and checked which teaching format would be easiest to follow.", "The patient was encouraged to pause the discussion whenever an instruction needed clarification.",
  ("Patient medical chart number", "The language service linked the visit to the patient's medical chart number {v}."), ("Patient service billing account", "The service entry is assigned to patient billing account {v}."), ("Interpreter booking ID", "The interpreter booking is identified by {v}.")),
 ("nutrition_followup", "Nutrition follow-up", "The patient describes early fullness and prefers small meals spaced through the day. The dietitian discussed adding a planned snack without making the meal schedule more complicated.", "The patient chose one manageable change to try before the next review.",
  ("Outpatient medical record number", "The dietitian accessed the patient's outpatient medical record with number {v}."), ("Nutrition service account number", "The nutrition visit is linked to billing account number {v}."), ("Nutrition consult request ID", "The consult request submitted to the dietitian has ID {v}.")),
 ("home_monitoring", "Home monitoring orientation", "The patient practiced placing the cuff with the arm supported and the feet resting on the floor. The nurse reviewed when to repeat a reading after quiet rest.", "The patient will keep a simple symptom log alongside the measurements.",
  ("Remote-care patient chart identifier", "The monitoring enrollment was attached to the patient's medical chart under identifier {v}."), ("Monitoring billing account", "The monitoring service bills patient account {v}."), ("Device enrollment request ID", "The enrollment request for the home device carries ID {v}.")),
 ("preprocedure_review", "Preprocedure telephone review", "The patient repeated the preparation sequence accurately and asked about managing a loose sleeve during the visit. The nurse reviewed comfort measures and the planned recovery observation.", "Any change in symptoms should be discussed with the procedure team before arrival.",
  ("Surgical patient medical record number", "The preparation review was entered in the patient's medical record numbered {v}."), ("Procedure billing account", "The planned procedure is associated with billing account {v}."), ("Procedure scheduling request ID", "The scheduling request is carried in the procedure queue as {v}.")),
 ("hearing_aid_review", "Hearing device follow-up", "The patient reports clearer speech in a quiet room but difficulty separating voices in a busy setting. The clinician practiced insertion and reviewed gradual adjustment to the device.", "The patient demonstrated the cleaning routine and confirmed the storage plan.",
  ("Audiology patient medical chart ID", "The audiologist located the patient's medical chart by chart identifier {v}."), ("Hearing device payment account", "The device payment arrangement is recorded under account {v}."), ("Device fitting work order ID", "The fitting work order uses identifier {v}.")),
 ("pulmonary_teaching", "Inhaler technique review", "The patient demonstrated a fast inhalation during the first attempt. After coaching, the patient used a slower breath and held the device steady through the maneuver.", "The clinician asked the patient to repeat the sequence once more without prompts.",
  ("Pulmonary patient chart number", "The respiratory educator documented the teaching in patient medical chart {v}."), ("Respiratory services billing account", "The respiratory service charges use billing account {v}."), ("Respiratory education referral ID", "The teaching referral is tracked with request identifier {v}.")),
 ("orthotic_fitting", "Orthotic fit check", "The patient reports rubbing after prolonged wear but no persistent redness after the device is removed. The therapist checked the contact points and practiced the fastening sequence.", "The wearing schedule will advance only if the skin remains comfortable.",
  ("Patient longitudinal medical record number", "The fit assessment was saved to the patient's longitudinal medical record numbered {v}."), ("Orthotic purchase account number", "The orthotic purchase is posted to account number {v}."), ("Orthotic adjustment job ID", "The adjustment job in the fabrication queue is identified by {v}.")),
 ("lab_collection", "Specimen collection planning", "The patient reviewed hydration and positioning preferences before collection. The phlebotomy team discussed a seated rest afterward because the patient sometimes feels briefly lightheaded.", "The collection team will reassess comfort before the patient stands.",
  ("Patient medical chart lookup key", "The collector retrieved the patient's medical chart with lookup key {v}."), ("Laboratory billing account", "The laboratory service will charge patient account {v}."), ("Specimen collection request ID", "The specimen collection request is identified by {v}.")),
 ("ambulatory_triage", "Ambulatory symptom review", "The patient describes brief positional dizziness without loss of consciousness. The nurse reviewed fluid intake, slow transitions, and the symptoms that should prompt urgent reassessment.", "The patient can explain the escalation plan in their own words.",
  ("Ambulatory medical record number", "The triage nurse opened the patient's ambulatory medical record under {v}."), ("Ambulatory patient billing account", "The telephone service is associated with billing account {v}."), ("Triage call event identifier", "The call event was entered in the communication log with identifier {v}.")),
 ("caregiver_teaching", "Home assistance teaching", "The patient practiced asking for assistance before rising from a low chair. The nurse demonstrated where a helper can stand while keeping the walking path clear.", "The patient and helper rehearsed the sequence with a pause between each step.",
  ("Patient chart index number", "The teaching record was placed in the patient's medical chart indexed as {v}."), ("Home care billing account", "The home care charges are routed to billing account {v}."), ("Teaching session booking ID", "The teaching session booking carries ID {v}.")),
 ("pain_group_intake", "Pain self-management intake", "The patient describes avoiding activity when discomfort increases. The clinician discussed pacing and a gradual return to a valued task, with rest planned before exhaustion.", "The patient selected a modest activity goal and a way to record progress.",
  ("Pain service medical record number", "The intake belongs to the patient's pain service medical record {v}."), ("Group program billing account", "The program fee is attached to billing account {v}."), ("Group enrollment request ID", "The group enrollment request is tracked as {v}.")),
 ("diabetes_education", "Glucose meter teaching", "The patient demonstrated hand washing and strip placement before checking the meter. The educator reviewed how symptoms, meals, and activity can help interpret a reading.", "The patient practiced the sequence again with the supplies arranged within reach.",
  ("Patient medical record lookup number", "The educator found the patient's medical record using lookup number {v}."), ("Education service account number", "The education visit is charged to service account {v}."), ("Education referral tracking ID", "The meter teaching referral has tracking ID {v}.")),
 ("ostomy_teaching", "Pouch change review", "The patient described difficulty seeing the lower edge of the appliance. The nurse suggested a mirror and a stable seated position, then reviewed gentle skin inspection.", "The patient chose a practical arrangement for supplies before the next change.",
  ("Patient's hospital chart number", "The pouch care review was entered in the patient's hospital medical chart {v}."), ("Ostomy supply account number", "Replacement supplies are billed to account number {v}."), ("Supply renewal request ID", "The renewal request for supplies is listed under identifier {v}.")),
 ("postoperative_call", "Recovery follow-up call", "The patient reports gradual improvement in activity tolerance and no new shortness of breath. The nurse reviewed comfortable positioning and the balance between movement and rest.", "The patient was able to repeat the reasons to contact the team sooner.",
  ("Patient surgical medical record identifier", "The recovery nurse used {v} to identify the patient's surgical medical record."), ("Surgical billing account number", "Recovery services are linked to surgical billing account {v}."), ("Follow-up call task identifier", "The follow-up task in the nurse work queue has identifier {v}.")),
 ("mobility_equipment", "Mobility equipment review", "The patient practiced turning with a walking aid while keeping the device close. The therapist checked grip comfort and reviewed how to approach a chair before sitting.", "A clear path and a deliberate pause before turning were included in the home plan.",
  ("Therapy patient medical record key", "The therapist retrieved the patient's medical record with key {v}."), ("Mobility equipment rental account", "The walking aid rental is assigned to account {v}."), ("Equipment fitting appointment ID", "The fitting appointment is tracked by event identifier {v}.")),
 ("vision_followup", "Vision care follow-up", "The patient reports eye fatigue after close work. The clinician reviewed lighting, brief breaks, and the safe use of the prescribed drops without touching the bottle tip.", "The patient demonstrated the drop sequence and described a reminder routine.",
  ("Ophthalmic medical record number", "The clinician opened the patient's ophthalmic medical record numbered {v}."), ("Vision service billing account", "The follow-up visit is billed through account {v}."), ("Eye clinic follow-up request ID", "The follow-up request uses identifier {v}.")),
 ("swallow_review", "Swallow strategy review", "The patient practiced a slower pace with a small sip between bites. The therapist reviewed upright positioning and the need to pause when attention or endurance decreases.", "The patient will use the agreed pacing strategy during the next supervised meal.",
  ("Speech therapy medical chart number", "The therapist linked the review to the patient's medical chart number {v}."), ("Speech therapy billing account", "The therapy billing entry belongs to account {v}."), ("Swallow assessment referral ID", "The assessment referral is identified in the queue as {v}.")),
 ("dental_clearance", "Dental clearance coordination", "The patient describes mild gum discomfort while brushing. The clinician reviewed gentle oral care and how to report increasing pain or swelling while the clearance review is pending.", "The coordination team checked that the patient understood the next contact step.",
  ("Patient's medical chart identifier", "The clearance note is attached to the patient's medical chart identified as {v}."), ("Dental services payment account", "The dental service payment is assigned to account {v}."), ("Clearance request reference", "The clearance request bears reference identifier {v}.")),
 ("continence_teaching", "Bladder routine teaching", "The patient discussed urgency during longer activities away from home. The nurse reviewed a planned toileting routine, comfortable access to fluids, and a simple symptom diary.", "The patient selected a routine that can be practiced without rushing.",
  ("Urology patient medical record number", "The teaching note was saved in the patient's urology medical record numbered {v}."), ("Urology visit billing account", "The teaching visit is posted to billing account {v}."), ("Nursing education task ID", "The education task has work queue identifier {v}.")),
 ("behavioral_intake", "Behavioral health intake", "The patient reports difficulty unwinding before sleep and would like a more predictable evening routine. The clinician discussed a short relaxation exercise and a way to track its usefulness.", "The patient chose one practice to try and described when it would fit into the evening.",
  ("Behavioral health medical record number", "The intake clinician retrieved the patient's medical record numbered {v}."), ("Counseling billing account identifier", "The counseling charges are assigned to account identifier {v}."), ("Intake appointment reference ID", "The intake appointment is registered under reference ID {v}.")),
 ("credentialed_consult", "Consultation readiness review", "The patient has several questions about maintaining daily activities during recovery. The consulting team reviewed the goals of the visit and the information needed for a shared plan.", "The patient prioritized the questions that would most affect daily routines.",
  ("Patient medical record identifier", "The consultant accessed the patient's medical record using identifier {v}."), ("Consultation billing account", "The consultation is charged to patient billing account {v}."), ("Reviewing clinician credential number", "The appointment file lists the reviewing clinician's credential number as {v}.")),
 ("records_transfer", "Care summary transfer", "The patient is preparing to continue routine care with another team. The nurse reviewed the current medication routine, active symptoms, and the questions to raise at the receiving visit.", "The patient received a clear explanation of what to discuss during the next care review.",
  ("Originating patient chart number", "The summary was drawn from the patient's medical chart numbered {v}."), ("Medical records fee account", "The copy fee is associated with billing account {v}."), ("Summary release request ID", "The release request for the care summary carries identifier {v}.")),
)

FIRST = ("Adela", "Bastian", "Celia", "Dario", "Elina", "Fabian", "Gisela", "Hugo", "Irena", "Jovan", "Kira", "Leandro", "Mirela", "Nolan", "Oriana", "Pavel", "Quinn", "Raisa", "Selma", "Tomas", "Una", "Vera", "Willa", "Xenia", "Yara", "Zoran", "Élodie", "Maëlle", "Lucía", "Søren", "Renée", "Aïcha", "Noémie", "Iñigo", "Zoila", "Ansel")
LAST = ("Arvendale", "Bellhaven", "Calderwood", "Dervain", "Ellermore", "Fenwicke", "Glenvar", "Haverlin", "Isenbrook", "Jarrowell", "Kestervale", "Lornwick", "Mardenwell", "Neswick", "Orlenby", "Perrivale", "Quenfield", "Ravelmont", "Selvinton", "Ternwell", "Umberlake", "Vallencroft", "Westerby", "Yarrowden", "Zellamont", "Duvallin", "O’Vellan", "Sánderin", "Bergström", "Ríos-Valen", "de Virel", "von Elsen", "Almerey", "Brenwick", "Cortevan", "Dalmorin")

# These generic expansions were independently written for multi-window stress.
# Their reuse across notes is deliberate, disclosed, and not clinical sampling.
EXPANSIONS = (
 "During the review, the patient described how the recommended routine would fit into an ordinary day. The nurse asked the patient to identify the step that felt least familiar and allowed time for an explanation. The discussion focused on a small number of achievable actions. The patient could state the immediate plan without reading from the written instructions, but requested another chance to practice before leaving.",
 "The clinician checked that the patient could comfortably see and handle the teaching materials. The materials were placed within reach, and the patient chose to review one topic at a time. Questions were answered in plain language. The patient was encouraged to ask for a different explanation whenever a term was unfamiliar. A brief pause was offered before moving from the demonstration to independent practice.",
 "Activity and rest were discussed in relation to the patient's usual routine. The patient described taking a short break when attention or endurance decreased. The team encouraged a gradual pace and a stable position during unfamiliar tasks. The patient was able to identify a practical place to keep needed items close at hand. No change to the treatment plan was made during this part of the discussion.",
 "The patient reviewed the sequence aloud before trying it again. The clinician observed without interrupting the first attempt, then addressed the one step that needed clarification. The second attempt was more deliberate. The patient explained why a pause could be useful when feeling rushed. The clinician confirmed that the instructions were understandable and invited the patient to describe any obstacle that might arise outside the clinic.",
 "The team discussed how to seek help if the current plan became difficult to follow. The patient was able to distinguish a routine question from a change that would require prompt reassessment. The clinician used an everyday example and asked the patient to explain the next action in their own words. The patient preferred to keep the written plan with the supplies used for the task.",
 "The patient remained engaged throughout the teaching and asked for time to compare the new instructions with the usual routine. The clinician emphasized completing each step safely rather than quickly. A comfortable position was maintained during the review. The patient identified one adjustment that would reduce unnecessary reaching and another that would make it easier to remember the sequence. The plan was then restated using the patient's own phrasing.",
 "The clinician reviewed the patient's understanding of symptom monitoring. The patient described paying attention to changes from the usual pattern rather than reacting to a single unfamiliar sensation in isolation. The discussion included how to describe a concern clearly when contacting the care team. The patient practiced giving a concise account of the concern, the action already taken, and the response to that action.",
 "A short rest was provided before the final teaching exercise. The patient repositioned comfortably and stated readiness to continue. The clinician asked whether written, spoken, or demonstrated instructions were most useful. The patient preferred a demonstration followed by an opportunity to repeat it. The clinician adjusted the pace accordingly and left time for the patient to review the steps that had required more concentration.",
 "The patient considered how the care routine might change on a busier day. The discussion focused on setting supplies out in advance and allowing enough time for a deliberate start. The patient described a simple reminder that would fit the existing routine. The clinician checked that the reminder would be easy to use and would not add a complicated new task. The patient agreed to reassess its usefulness after practice.",
 "The team reviewed the purpose of the follow-up discussion and the questions that would be useful to bring. The patient planned to note any difficulty with the routine and any change in comfort or function. The clinician emphasized describing what happened before and after a symptom rather than relying on a vague summary. The patient was able to give an example of an observation that would help guide the next review.",
 "The patient asked whether the routine had to be completed at the same pace every time. The clinician explained that a deliberate pause was reasonable when fatigue or distraction interfered with safe performance. The patient demonstrated returning to the sequence after a brief pause. The clinician checked that the patient knew which step had been completed and which came next. The patient then repeated the sequence with fewer prompts.",
 "Before the discussion ended, the clinician invited the patient to name any remaining concern. The patient wanted reassurance that questions could be raised after trying the plan outside the clinic. The clinician reviewed the usual route for routine questions and the symptoms that require more immediate attention. The patient repeated the contact plan accurately and chose a convenient place to keep the written instructions available for later review.",
 "The patient compared the planned routine with familiar daily activities and identified a time when distractions are usually lower. The clinician encouraged practicing in that setting while the steps were still new. The patient described how needed items could be arranged without blocking a walking path. The clinician checked the plan for comfort and practicality. The patient was then able to explain the arrangement and the intended sequence without assistance.",
 "The discussion returned to the patient's main goal for care. The clinician asked which improvement would make the greatest difference in ordinary activities. The patient chose a modest and observable goal rather than trying to change several routines at once. The team discussed how to recognize progress and how to report a setback. The patient expressed understanding that the plan could be adjusted during a later review if needed.",
 "The patient practiced summarizing the care instructions as if explaining them to a helper. The clinician listened for missed steps and corrected one point with another demonstration. The patient then restated the sequence clearly. The team discussed asking a helper to observe only when assistance would be useful. The patient preferred to remain involved in each part of the routine and to request support before becoming overly tired.",
 "The clinician reviewed comfort, attention, and readiness before asking for another demonstration. The patient reported that a slower pace made the instructions easier to follow. The demonstration was completed in short segments with time for questions between them. The patient identified the point at which a pause would be most helpful. The clinician confirmed that the plan was practical and that the patient could explain the reason for each step.",
)

INTERPRETATION = {
 "MRN": "An explicitly patient-specific identifier whose stated purpose is to find or identify that patient's medical chart or longitudinal medical record. Generic 'record', event, and document references do not qualify without this context.",
 "ACCOUNT": "An explicitly named patient billing, service payment, rental, or supply financial account identifier. An order, request, event, or authorization is not an account merely because money is discussed nearby.",
 "ID": "An explicitly identified request, order, booking, task, work order, release, event, service ticket, or clinician credential reference, distinct from the patient medical chart and financial account.",
 "scope": "Prospective authoring interpretation for this new draft; the v2 policy enumerates these types without a complete semantic definition. No existing gold, policy, or frozen mixture is revised. Ambiguous identifier roles were avoided.",
}

def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def identity_hash(name: str) -> str:
    return sha(unicodedata.normalize("NFC", name).casefold().encode("utf-8"))

def marked(label: str, value: str) -> str:
    return f"[[{label}|{value}]]"

def parse_marked(value: str):
    text, spans, cursor, end = [], [], 0, 0
    for match in re.finditer(r"\[\[([A-Z]+)\|([^\[\]]+)\]\]", value):
        prefix = value[end:match.start()]
        text.extend((prefix, match[2]))
        cursor += len(prefix)
        spans.append({"start": cursor, "end": cursor + len(match[2]), "label": match[1], "text": match[2]})
        cursor += len(match[2]); end = match.end()
    text.append(value[end:])
    clean = "".join(text)
    assert "[[" not in clean and "]]" not in clean
    return clean, spans

def make_value(shape: str, rng: random.Random) -> str:
    letters = "BCDFGHJKLNPQRTVWXYZ"
    pair = "".join(rng.choice(letters) for _ in range(2))
    digit = lambda n: str(rng.randrange(10 ** (n - 1), 10 ** n))
    return {
        "digits_6": lambda: digit(6),
        "zero_padded_digits_8": lambda: "00" + digit(6),
        "digits_10": lambda: digit(10),
        "zero_padded_digits_12": lambda: "000" + digit(9),
        "letters2_digits5": lambda: pair + digit(5),
        "digits3_letters2_digits3": lambda: digit(3) + pair + digit(3),
        "letters2_hyphen_digits6": lambda: pair + "-" + digit(6),
        "digits3_hyphen3_hyphen3": lambda: "-".join(digit(3) for _ in range(3)),
        "letters2_underscore_digits4": lambda: pair + "_" + digit(4),
        "digits4_slash_letters2_digits3": lambda: digit(4) + "/" + pair + digit(3),
    }[shape]()

def window_jobs(n, content):
    overlap = min(128, content // 4)
    stride = content - 2 * overlap
    start = 0
    while True:
        stop = min(n, start + content)
        left = 0 if start == 0 else start + overlap
        right = n if stop == n else stop - overlap
        yield (start, stop, left, right)
        if stop >= n:
            break
        start += stride

def clinical_observations(g):
    age = 34 + g % 48
    days = 2 + g % 11
    minutes = 4 + g % 13
    return [
        ("Clinical history", f"Age {age} years; symptoms have been discussed over {days} days", f"The patient is {age} years old and has tracked the current symptoms over {days} days."),
        ("Observed activity", f"Comfortable seated review for {minutes} minutes", f"The patient participated comfortably in a seated review lasting {minutes} minutes."),
        ("Respiratory observation", f"Respiratory rate {12 + g % 7}/min; unlabored at rest", f"Respirations were unlabored at rest, with a rate of {12 + g % 7}/min."),
    ]

def render_item(layout, caption, value, sentence):
    if layout == "narrative":
        return sentence
    if layout == "form":
        return f"{caption}: {value}"
    if layout == "bullets":
        return f"- {caption}: {value}."
    if layout == "table":
        return f"| {caption} | {value} |"
    return f"{caption} → {value}"

def make_rows(tokenizer, content):
    rng = random.Random(731094)
    candidates = [f"{first} {last}" for first in FIRST for last in LAST]
    rng.shuffle(candidates)
    pool = [name for name in candidates if int(identity_hash(name)[:8], 16) % 10 < 8]
    assert len(pool) >= 330
    used_values = set()
    repeat_counts = Counter()
    rows = []
    for fi, family in enumerate(FAMILIES):
        key, title, opening, followup, *fields = family
        for pi in range(10):
            g = fi * 10 + pi
            person = pool[g]
            clinician = pool[300 + fi] if g % 3 == 0 or key == "credentialed_consult" else None
            identities = [person] + ([clinician] if clinician else [])
            groups = [identity_hash(name) for name in identities]
            layout = LAYOUTS[pi % 5]
            long = pi in (fi % 5, 5 + (fi + 2) % 5)
            observations = clinical_observations(g)
            for variant in range(4):
                control = variant == 3
                selected = () if control else TARGETS if variant < 2 else SUBSETS[g % 6]
                order = tuple(t for t in ORDERS[(g + variant) % 6] if t in selected)
                shapes, values = {}, {}
                for label in selected:
                    shape = SHAPES[(g + TARGETS.index(label) * 3 + variant * 2) % len(SHAPES)]
                    value = make_value(shape, rng)
                    while value in used_values:
                        value = make_value(shape, rng)
                    used_values.add(value)
                    shapes[label] = shape; values[label] = value
                chunks = []
                for label in order:
                    caption, construction = fields[TARGETS.index(label)]
                    value = marked(label, values[label])
                    chunks.append(render_item(layout, caption, value, construction.format(v=value)))
                # All layouts occur in both positive and PHI-free notes.
                natural = [render_item(layout, a, b, c) for a, b, c in observations]
                items = natural if control else chunks + natural
                if not control and g % 4 == 0:
                    rotated = order[(g // 4 + variant) % len(order):] + order[:(g // 4 + variant) % len(order)]
                    label = min(rotated, key=lambda target: repeat_counts[target])
                    repeat_counts[label] += 1
                    caption = {"MRN": "Patient medical chart number confirmed", "ACCOUNT": "Billing account confirmed", "ID": fields[2][0] + " confirmed"}[label]
                    value = marked(label, values[label])
                    sentence = {"MRN": f"The staff member confirmed {value} as the patient's medical chart number.", "ACCOUNT": f"The staff member confirmed the billing account as {value}.", "ID": f"The staff member confirmed the {fields[2][0].lower()} as {value}."}[label]
                    items.append(render_item(layout, caption, value, sentence))
                headers = {"narrative": title + ".", "form": title.upper(), "bullets": title + "\nReview points", "table": title, "handoff": title + "\nCoordination handoff"}
                lead = opening
                extra = []
                aliases = {}
                if not control:
                    if g % 2 == 0:
                        if g % 6 == 0:
                            alias = person.split()[0]
                            extra.append(f"{marked('NAME', person)} asks to be addressed as {marked('NAME', alias)}. {marked('NAME', alias)} reviewed the plan with the nurse.")
                            aliases[person] = [alias]
                        else:
                            extra.append(f"The review concerns {marked('NAME', person)}.")
                    if clinician:
                        extra.append(f"Dr. {marked('NAME', clinician)}, MD, reviewed the plan.")
                    if g % 3 == 1:
                        date = f"{('April', 'June', 'September', 'November')[g % 4]} {1 + g % 27}, {2022 + g % 4}"
                        extra.append(f"The patient attended the preceding care review on {marked('DATE', date)}.")
                    if g % 10 == 2:
                        phone = f"(202) 555-{100 + g % 100:04d}"
                        extra.append(f"The patient requests follow-up at {marked('PHONE', phone)}.")
                    if g % 10 == 7:
                        email = f"followup{g:03d}@visit.test"
                        extra.append(f"Send the patient's routine contact material to {marked('EMAIL', email)}.")
                if extra and (g + variant) % 3 == 0:
                    lead = " ".join(extra) + " " + lead
                    extra = []
                parts = [headers[layout], lead]
                table_header = "| Review item | Detail |\n| --- | --- |"
                if long:
                    # Interleave ten distinct expansions with fields; later fields
                    # and optional repeat references cross ownership windows.
                    selection = [(fi + pi * 3 + i * 7) % len(EXPANSIONS) for i in range(10)]
                    if control:
                        placed = dict(zip((1, 4, 8), natural))
                    else:
                        placed = dict(zip((1, 4, 8), chunks))
                        placed.update(dict(zip((0, 3, 7), natural)))
                        if len(items) > len(chunks) + len(natural):
                            placed[9] = items[-1]
                    for i, exp_index in enumerate(selection):
                        parts.append(EXPANSIONS[exp_index])
                        if i in placed:
                            parts.append(table_header + "\n" + placed[i] if layout == "table" else placed[i])
                else:
                    if layout == "table":
                        parts.append(table_header)
                    parts.extend(items)
                parts.extend(extra)
                parts.append(followup)
                text, spans = parse_marked("\n\n".join(parts) if long or layout == "narrative" else "\n".join(parts))
                encoding = tokenizer.encode(text, add_special_tokens=False)
                jobs = list(window_jobs(len(encoding.ids), content))
                if long:
                    assert len(jobs) >= 3
                phenotype = ["identifier_semantics", "shared_identifier_shapes", "natural_clinical_context", "ordinary_age_duration_measurement", f"layout_{layout}"]
                phenotype += ["phi_free_control"] if control else ["three_identifier_types" if variant < 2 else "identifier_subset", "explicit_field_context"]
                if long:
                    phenotype += ["multi_window", "authored_expanded_documentation"]
                if aliases:
                    phenotype.append("linked_personal_alias")
                if clinician:
                    phenotype.append("secondary_identity_metadata")
                row = {
                    "id": f"tk7_identifier_contexts_{fi:02d}_{pi:02d}_{variant}",
                    "text": text, "spans": spans, "split": "train",
                    "annotation_policy": "clinical-phi-v2", "full_identity": person,
                    "full_identities": identities, "identity_group": groups[0], "identity_groups": groups,
                    "pair_id": f"tk7_identifier_contexts_pair_{fi:02d}_{pi:02d}",
                    "template_family": f"tk7_identifier_contexts_{key}",
                    "phenomena": phenotype, "cohort": "long_document" if long else "realistic",
                    "layout": layout, "negative_only": not spans,
                    "identifier_shapes": shapes, "identifier_first_order": list(order),
                    "person_aliases": aliases,
                    "authorship_role": "independent_training_draft",
                    "variant": ("triple_a", "triple_b", "subset", "control")[variant],
                    "runtime_token_count": len(encoding.ids), "runtime_window_count": len(jobs),
                }
                rows.append(row)
    return rows

def statistics(rows, tokenizer, content):
    errors = []
    ids, texts, shape_table, shape_documents = set(), set(), defaultdict(Counter), defaultdict(Counter)
    owner_table, boundary_distances = defaultdict(Counter), []
    identity_to_name = {}
    for row in rows:
        text = row["text"]
        if row["id"] in ids or text in texts:
            errors.append([row["id"], "duplicate id or exact text"])
        ids.add(row["id"]); texts.add(text)
        if row["identity_groups"] != [identity_hash(n) for n in row["full_identities"]]:
            errors.append([row["id"], "identity mismatch"])
        if row["full_identity"] != row["full_identities"][0] or row["identity_group"] != row["identity_groups"][0]:
            errors.append([row["id"], "singular/plural identity mismatch"])
        if len(set(row["identity_groups"])) != len(row["identity_groups"]):
            errors.append([row["id"], "duplicate canonical identity"])
        actual_order = list(dict.fromkeys(s["label"] for s in row["spans"] if s["label"] in TARGETS))
        if actual_order != row["identifier_first_order"]:
            errors.append([row["id"], "first occurrence order mismatch"])
        name_values = set(row["full_identities"]) | {alias for full, aliases in row["person_aliases"].items() for alias in aliases if full in row["full_identities"]}
        if any(s["label"] == "NAME" and s["text"] not in name_values for s in row["spans"]):
            errors.append([row["id"], "name lacks canonical identity linkage"])
        if re.search(r"\bPHI\b|\bgold label\b|\bnegative example\b|\bnot an identifier\b", text, re.I):
            errors.append([row["id"], "explanatory annotation cue"])
        for name, group in zip(row["full_identities"], row["identity_groups"]):
            identity_to_name[group] = name
            if int(group[:8], 16) % 10 >= 8:
                errors.append([row["id"], "non-training identity"])
        previous = -1
        for span in row["spans"]:
            a, b, label = span["start"], span["end"], span["label"]
            if not (0 <= a < b <= len(text) and a >= previous and label in LABELS and text[a:b] == span["text"] and text[a:b].strip() == text[a:b]):
                errors.append([row["id"], "span structure"])
            previous = b
            if label in TARGETS:
                shape_table[row["identifier_shapes"][label]][label] += 1
        target_values = {s["text"] for s in row["spans"] if s["label"] in TARGETS}
        for value in target_values:
            literal_occurrences = [(m.start(), m.end()) for m in re.finditer(re.escape(value), text)]
            annotated_occurrences = [(s["start"], s["end"]) for s in row["spans"] if s["text"] == value and s["label"] in TARGETS]
            if literal_occurrences != annotated_occurrences:
                errors.append([row["id"], "unaccounted identifier occurrence"])
        for label, shape in row["identifier_shapes"].items():
            shape_documents[shape][label] += 1
        if row["cohort"] == "long_document":
            enc = tokenizer.encode(text, add_special_tokens=False)
            jobs = list(window_jobs(len(enc.ids), content))
            boundaries = [j[3] for j in jobs[:-1]]
            for span in row["spans"]:
                if span["label"] in TARGETS:
                    tokens = [i for i, (a, b) in enumerate(enc.offsets) if b > span["start"] and a < span["end"]]
                    if not tokens:
                        errors.append([row["id"], "span without token coverage"])
                        continue
                    owner = next(i for i, (_, _, left, right) in enumerate(jobs) if left <= tokens[0] < right)
                    owner_table[str(owner)][span["label"]] += 1
                    boundary_distances.append(min(abs(tokens[0] - x) for x in boundaries))
    pair_rows = defaultdict(list)
    for row in rows:
        pair_rows[row["pair_id"]].append(row)
    for pair, quartet in pair_rows.items():
        assert len(quartet) == 4
        assert len({tuple(r["identity_groups"]) for r in quartet}) == 1
        assert len({r["layout"] for r in quartet}) == 1
        assert len({r["cohort"] for r in quartet}) == 1
    support = Counter(s["label"] for row in rows for s in row["spans"])
    cohorts = Counter(r["cohort"] for r in rows)
    counts = {
        "rows": len(rows), "documents": len(rows), "spans": sum(support.values()),
        "phi_free_rows": sum(not r["spans"] for r in rows),
        "label_support": {l: support[l] for l in LABELS},
        "cohort_counts": dict(sorted(cohorts.items())),
        "families": len(FAMILIES), "pair_groups": len(pair_rows), "canonical_identities": len(identity_to_name),
        "variant_counts": dict(sorted(Counter(r["variant"] for r in rows).items())),
        "layout_counts": dict(sorted(Counter(r["layout"] for r in rows).items())),
        "layout_by_variant": {l: dict(Counter(r["variant"] for r in rows if r["layout"] == l)) for l in LAYOUTS},
        "cohort_by_variant": {c: dict(Counter(r["variant"] for r in rows if r["cohort"] == c)) for c in cohorts},
        "family_support": dict(Counter(r["template_family"] for r in rows)),
        "target_set_counts": dict(Counter("+".join(sorted(r["identifier_shapes"])) or "none" for r in rows)),
        "triple_first_occurrence_orders": dict(sorted(Counter("->".join(r["identifier_first_order"]) for r in rows if len(r["identifier_shapes"]) == 3).items())),
        "all_first_occurrence_orders": dict(sorted(Counter("->".join(r["identifier_first_order"]) or "none" for r in rows).items())),
        "identifier_shape_by_label_spans": {shape: {l: shape_table[shape][l] for l in TARGETS} for shape in SHAPES},
        "identifier_shape_by_label_documents": {shape: {l: shape_documents[shape][l] for l in TARGETS} for shape in SHAPES},
        "shape_only_majority_accuracy_on_spans": sum(max(v.values()) for v in shape_table.values()) / sum(sum(v.values()) for v in shape_table.values()),
        "every_shape_has_all_three_labels": all(all(shape_table[s][l] > 0 for l in TARGETS) for s in SHAPES),
        "runtime_window_count_distribution": dict(sorted(Counter(str(r["runtime_window_count"]) for r in rows).items())),
        "long_target_spans_by_owning_window_zero_based": {k: dict(v) for k, v in sorted(owner_table.items())},
        "long_target_spans_starting_within_16_tokens_of_ownership_boundary": sum(d <= 16 for d in boundary_distances),
        "long_notes_with_at_least_three_runtime_windows": sum(r["cohort"] == "long_document" and r["runtime_window_count"] >= 3 for r in rows),
        "non_ascii_text_rows": sum(not r["text"].isascii() for r in rows),
        "alias_rows": sum(bool(r["person_aliases"]) for r in rows),
        "name_rows": sum(any(s["label"] == "NAME" for s in r["spans"]) for r in rows),
    }
    def summary(nums):
        nums = sorted(nums)
        return {"min": nums[0], "median": nums[len(nums)//2], "p90": nums[int(.9*(len(nums)-1))], "max": nums[-1]}
    counts["token_lengths_by_variant"] = {v: summary([r["runtime_token_count"] for r in rows if r["variant"] == v]) for v in ("triple_a", "triple_b", "subset", "control")}
    counts["token_lengths_by_cohort"] = {c: summary([r["runtime_token_count"] for r in rows if r["cohort"] == c]) for c in cohorts}
    counts["character_lengths_by_variant"] = {v: summary([len(r["text"]) for r in rows if r["variant"] == v]) for v in ("triple_a", "triple_b", "subset", "control")}
    counts["span_surface_case"] = {label: dict(Counter("uppercase" if s["text"].isupper() else "lowercase" if s["text"].islower() else "mixed_case" if any(c.isalpha() for c in s["text"]) else "uncased" for r in rows for s in r["spans"] if s["label"] == label)) for label in LABELS if support[label]}
    names = [s["text"] for r in rows for s in r["spans"] if s["label"] == "NAME"]
    counts["name_surface_features"] = {"non_ascii_occurrences": sum(not n.isascii() for n in names), "apostrophe_occurrences": sum("’" in n or "'" in n for n in names), "hyphen_occurrences": sum("-" in n for n in names), "space_occurrences": sum(" " in n for n in names), "script_scope": "Latin-script invented personal identities, including diacritics; no non-Latin-script coverage claimed"}
    cues = {"age": r"\bage\b|\byears old\b", "chart_record": r"\bchart\b|\bmedical record\b", "account": r"\baccount\b", "identifier_id": r"\bidentifier\b|\bID\b", "duration": r"\bminutes\b|\bdays\b", "contact": r"\bcontact\b", "patient": r"\bpatient\b"}
    counts["cue_rates"] = {cue: {stratum: sum(bool(re.search(pattern,r["text"],re.I)) for r in rows if (not r["spans"]) == negative) / sum((not r["spans"]) == negative for r in rows) for stratum, negative in (("positive", False),("control",True))} for cue, pattern in cues.items()}
    assert not errors, errors[:10]
    assert counts["rows"] == 1200 and counts["phi_free_rows"] == 300
    assert counts["families"] == 30 and counts["long_notes_with_at_least_three_runtime_windows"] == 240
    assert counts["every_shape_has_all_three_labels"]
    counts["structural_checks"] = {"passed": True, "errors": errors, "checks": ["unique ids and exact texts", "all marked occurrences emitted with Unicode exclusive offsets", "known labels and exact span text", "no span overlap", "all target identifier literal occurrences annotated", "first target occurrence order matches metadata", "every NAME surface links to canonical identity or declared alias", "no annotation explanation cues", "all person identities NFC/casefold hashed and in train partition", "quartets share pair/identity/layout/cohort", "every shape used with all three target labels", "all long notes span at least three actual runtime windows"]}
    return counts

def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Compare existing own artifacts to reproducibly generated bytes without writing")
    args = parser.parse_args()
    tokenizer_path = MODEL / "hf/tokenizer.json"
    runtime_path = MODEL / "config.json"
    hf_config_path = MODEL / "hf/config.json"
    runtime = json.loads(runtime_path.read_bytes())
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    special_count = len(tokenizer.encode("clinical note").ids) - len(tokenizer.encode("clinical note", add_special_tokens=False).ids)
    content = int(runtime["max_length"]) - special_count
    overlap = min(128, content // 4)
    tokenizer_info = {"tokenizer_path": str(tokenizer_path.relative_to(ROOT)), "tokenizer_sha256": sha(tokenizer_path.read_bytes()), "runtime_config_sha256": sha(runtime_path.read_bytes()), "hf_config_sha256": sha(hf_config_path.read_bytes()), "max_length": runtime["max_length"], "special_tokens": special_count, "content_window": content, "runtime_overlap_per_side": overlap, "stride": content - 2 * overlap, "method": "tokenizers.Tokenizer.from_file; content encoding with add_special_tokens=False; window_jobs formula from permitted hf_encoder.py", "config_window_overlap_field": runtime.get("window_overlap"), "window_overlap_note": "hf_encoder.window_jobs computes overlap as min(128, content_window//4), rather than using the checkpoint window_overlap field"}
    rows = make_rows(tokenizer, content)
    counts = statistics(rows, tokenizer, content)
    corpus = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode("utf-8")
    source_hash = sha(Path(__file__).read_bytes())
    provenance = {"author": "author7_identifier_contexts", "authorship": "Independently invented contexts, identities and wording in this script; no clinical records or previous templates used", "synthetic_identifier_scheme": "Locally seeded invented neutral identifiers; no medical-role prefixes inside values; neutral letters exclude vowels; no guarantee of global noncollision with real identifiers is possible", "synthetic_contact_scheme": "Only 202-555-0100 through 202-555-0199 and a .test email domain", "identity_partition": "NFC/casefold SHA256 canonical full identities rejection-sampled with int(first8hex,16)%10<8; no alphabet partition", "control_metadata": "Paired PHI-free controls inherit canonical full identities and identity groups of their three positive partners, as required by v2 paired-note policy; no personal names occur in control text", "permitted_reads": ["docs/annotation-policy-v2.md", "src/clinical_deid/data/annotation_policy.py", "scripts/takeover/prepare_training_mixture.py", "scripts/takeover/normalize_conventions.py", "scripts/audit_corpus_quality.py", "src/clinical_deid/detection/hf_encoder.py", "models/takeover-mbbase-mix3-001/step-003081/config.json", "models/takeover-mbbase-mix3-001/step-003081/hf/config.json", "models/takeover-mbbase-mix3-001/step-003081/hf/tokenizer.json"], "restricted_inputs_read": False, "evaluation_inputs_read": False, "existing_training_templates_read": False, "model_weights_read": False, "model_inference_run": False, "network_used": False}
    manifest = {"schema_version": 1, "corpus": str(CORPUS.relative_to(ROOT)), "sha256": sha(corpus), "split": "train", "dataset_role": "train", "status": "draft", "annotation_policy": "clinical-phi-v2", "rows": counts["rows"], "documents": counts["documents"], "spans": counts["spans"], "phi_free_rows": counts["phi_free_rows"], "label_support": counts["label_support"], "cohort_counts": counts["cohort_counts"], "generator": str(Path(__file__).relative_to(ROOT)), "generator_sha256": source_hash, "provenance": provenance, "semantic_interpretation": INTERPRETATION, "admitted_to_training": False, "requires_independent_qa": True}
    manifest_bytes = json_bytes(manifest)
    report = {"schema_version": 1, "status": "draft_pending_independent_qa", "annotation_policy": "clinical-phi-v2", "split": "train", "counts": counts, "semantic_interpretation": INTERPRETATION, "provenance": provenance, "runtime_tokenizer": tokenizer_info, "artifacts": {"corpus": {"path": str(CORPUS.relative_to(ROOT)), "sha256": sha(corpus)}, "manifest": {"path": str(MANIFEST.relative_to(ROOT)), "sha256": sha(manifest_bytes)}, "generator": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": source_hash}}, "shape_confounding": {"shared_shapes": list(SHAPES), "shape_alone_determines_label": False, "scope": "Aggregate support is diagnostic, not proof that contextual or format shortcuts are absent; field captions deliberately state real semantic roles and remain outside spans."}, "cue_limits": ["Explicit clinical chart, account, request, and credential wording supplies the intended semantics; corresponding word-frequency gaps against clinical controls are expected and published.", "All five layouts and all 30 families have positive and PHI-free variants; 20% of each variant is long, but identifiers and optional PHI make positive notes somewhat longer.", "Long notes interleave ten newly authored generic teaching paragraphs with context-specific fields; expansions are reused across notes, and this is disclosed synthetic documentation stress rather than naturally sampled clinical prose.", "The 1200 notes derive from 300 paired contexts and 30 families; row counts are not counts of independent clinical observations.", "No measured model failure, accuracy improvement, or production-readiness conclusion follows from this draft."], "manual_review": {"status": "author_varied_review_complete_independent_review_pending", "row_ids": [f"tk7_identifier_contexts_{f:02d}_{(f+1)%5:02d}_{f%4}" for f in range(30)] + [f"tk7_identifier_contexts_{f:02d}_{f:02d}_{(0,1,2,3,0)[f]}" for f in range(5)], "rows_reviewed": 35, "coverage": ["all 30 family contexts", "all five layouts", "all four variants", "five full multi-window notes", "Unicode full names and aliases", "secondary clinicians and credential ID context", "repeated target values", "nonidentifying clinical measurements and durations"], "observations": ["Identifier roles are stated outside spans and are intelligible in clinical or administrative context.", "Explicit medical wording was added to two chart field captions to reduce generic-record ambiguity.", "Long-note field placement was changed during author review to distribute target values across ownership windows; all final spans and counts are regenerated.", "Table headers were moved next to their row groups.", "This author review is a varied sample and does not substitute for independent semantic adjudication."]}, "pending_gates": ["independent semantic and metadata QA", "convention normalization review", "sanctioned overlap audit conducted by a separate authorized process", "any subsequent training admission decision"], "frozen_mix3_changed": False}
    outputs = ((CORPUS, corpus), (MANIFEST, manifest_bytes), (REPORT, json_bytes(report)))
    if args.verify:
        for path, expected in outputs:
            assert path.read_bytes() == expected, f"Reproducibility mismatch: {path}"
    else:
        for path, _ in outputs:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
        for path, payload in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(payload)
    print(json.dumps({"status": "reproducible" if args.verify else "draft_written", "rows": counts["rows"], "spans": counts["spans"], "phi_free_rows": counts["phi_free_rows"], "long_rows": counts["cohort_counts"]["long_document"], "shape_only_majority_accuracy_on_spans": counts["shape_only_majority_accuracy_on_spans"], "hashes": {str(p.relative_to(ROOT)): sha(b) for p,b in outputs}}, sort_keys=True))

if __name__ == "__main__":
    main()
