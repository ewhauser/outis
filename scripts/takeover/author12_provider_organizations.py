"""Independent synthetic TRAINING author: provider organizations, clinical-phi-v2.

No input corpora, prediction files, evaluation examples, model weights or network.
The only runtime source reads are the explicitly allowed policy, normalizer,
encoder window definitions and tokenizer/configuration files. Output is exclusive.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import unicodedata

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/tk12_train_provider_organizations.jsonl"
REPORT = ROOT / "reports/takeover/mix3_001/author12_provider_organizations"
MODEL = ROOT / "models/takeover-mbbase-mix3-001/step-003081"
LABELS = ("NAME", "DATE", "ADDRESS", "PHONE", "EMAIL", "MRN", "SSN", "ACCOUNT", "ID", "URL", "IP", "AGE", "LOCATION", "FACILITY")

# Each tuple is a separately composed clinical situation, not a renamed carrier.
# key, title, provider category, reason, observations, planned review, provider action
FAMILIES = [
 ("wound_drainage", "Incision and dressing review", "home_health", "a dressing that loosens before the next planned change", "The surrounding skin is dry without spreading redness. The patient can describe the dressing sequence but has difficulty reaching the lower edge. A small amount of clear drainage was seen on the removed pad.", "Review dressing technique and the supply count at the next contact. A new odor, fever, or expanding redness would prompt clinical reassessment.", "{org} accepted the skilled nursing visit for wound assessment and dressing teaching."),
 ("heart_failure_weights", "Weight record review", "home_health", "inconsistent home weights after a heart failure admission", "The scale had been moved between carpet and tile. The patient reports exertional fatigue without breathlessness at rest. The weight log mixes measurements before and after breakfast.", "Use a consistent weighing routine and review the symptom record with the clinical team. Medication decisions remain with the treating clinician.", "{org} will provide the home nursing assessment and review the weight log."),
 ("ostomy_teaching", "Pouch fit follow-up", "home_health", "difficulty maintaining a pouch seal while sitting", "The skin fold becomes more prominent in the seated position. The patient can empty the pouch but needs another demonstration of measuring the opening. No new abdominal pain was reported.", "Reassess the fit in the position used for daily activities. Continue teaching with return demonstration before reducing visit support.", "{org} arranged a visiting nurse to review pouch fit and peristomal skin care."),
 ("insulin_dexterity", "Injection technique review", "home_health", "hand stiffness interfering with use of the prescribed injection device", "The patient can identify the intended device but struggles to hold it steady. Written instructions are legible with the usual glasses. The observed difficulty concerns dexterity rather than a requested dose change.", "Ask the prescribing team to review a device the patient can handle reliably. Recheck technique using the approved supplies.", "{org} is the home-health provider assessing injection technique and self-management needs."),
 ("medication_reconciliation", "Medication container review", "home_health", "a mismatch between the discharge list and containers in the kitchen", "Two bottles contain the same medication strength. The patient has been setting aside uncertain tablets instead of taking both. The caregiver can explain which list came with the discharge paperwork.", "Clarify discrepancies with the prescriber before revising the organizer. Document the reconciled list where the patient normally prepares medication.", "{org} assigned a home nurse to reconcile the containers with the active medication list."),
 ("oxygen_tubing", "Oxygen equipment assessment", "equipment_supplier", "tubing that catches under the bedroom door", "The concentrator operates without an alarm during the visit. The tubing follows a walking route used at night. No change to the prescribed oxygen setting was requested or made.", "Review routing and the equipment setup with the supplier. The clinical team will assess any change in breathing symptoms.", "{org} supplies the prescribed oxygen equipment and accepted the tubing safety review."),
 ("pap_mask_leak", "PAP interface follow-up", "equipment_supplier", "mask leak interrupting use of PAP during sleep", "The cushion is visibly worn near one corner. The patient describes air reaching the eyes when turning onto the side. The pressure setting was left as prescribed.", "Have the equipment service review fit and replacement parts. Send persistent tolerance concerns to the sleep clinician.", "{org} is the medical-equipment supplier handling the PAP interface fitting."),
 ("walker_brake", "Mobility aid service review", "equipment_supplier", "a rollator brake that slips when the seat is used", "The left brake holds on a level surface while the right brake releases under light load. The patient can demonstrate the intended brake sequence. The equipment was not used for seated rest during this review.", "Arrange inspection before the rollator is relied on for seated support. Review an interim mobility plan with the therapy team.", "{org} accepted the repair request as the supplier of the mobility aid."),
 ("hospital_bed_transfer", "Bed setup assessment", "equipment_supplier", "a bed height that complicates assisted transfers", "The motor raises and lowers the frame without unusual noise. The patient prefers to transfer from the side nearest the bathroom. The caregiver reports difficulty reaching the hand control while assisting.", "Coordinate the equipment adjustment with the therapist's transfer assessment. Verify the completed setup during a return demonstration.", "{org} provides the home medical bed and will review its setup with the care team."),
 ("enteral_pump_alarm", "Feeding pump support", "equipment_supplier", "recurrent flow alarms during the established feeding routine", "The caregiver can reproduce an alarm when the tubing bends against the bag support. The connections appear intact on visual review. Nutrition orders and the prescribed rate were not changed during the equipment check.", "Ask the supply service to inspect the set and mounting arrangement. Persistent intolerance or abdominal symptoms require clinical review.", "{org} supplies the enteral pump and arranged technical support for the alarm concern."),
 ("hip_gait", "Postoperative walking review", "rehabilitation", "hesitation when turning with the usual walking aid", "The patient maintains the advised precautions during straight walking. Turns require several short steps and a verbal cue. Fatigue increases after repeated practice in the hallway.", "Continue supervised practice within the current plan. Review the aid and home route before advancing independence.", "{org} accepted the rehabilitation referral for gait and transfer training."),
 ("stroke_dressing", "Upper-body dressing practice", "rehabilitation", "difficulty positioning the affected arm during dressing", "The patient follows one-step cues and notices the sleeve caught beneath the forearm. Sitting balance is maintained during the observed task. The caregiver tends to complete the task before the patient can attempt it.", "Practice the agreed dressing sequence with adequate time for participation. Reassess caregiver cueing at the next therapy review.", "{org} provides occupational rehabilitation for dressing and daily activity practice."),
 ("vestibular_pacing", "Balance symptom review", "rehabilitation", "brief disequilibrium when moving from reading to walking", "The patient describes a short pause before standing as helpful. No fall occurred during observed mobility. Rapid head movement increases discomfort, and practice stopped when symptoms rose.", "The therapist will reassess the exercise plan and tolerance. New focal symptoms would require medical evaluation.", "{org} is the rehabilitation service following the balance and activity tolerance concerns."),
 ("hand_tendon", "Hand function follow-up", "rehabilitation", "difficulty fastening clothing within the current hand precautions", "The protective device fits without a new pressure mark. The patient can explain the movement restrictions. Fine tasks remain slow, especially when the fabric is stiff.", "Review task adaptations without advancing the prescribed movement limits. Confirm the therapy plan with the surgical instructions.", "{org} arranged hand rehabilitation focused on permitted daily activities."),
 ("swallowing_routine", "Mealtime strategy review", "rehabilitation", "inconsistent use of the established swallowing routine", "The patient takes another sip before completing the prompted pause. The caregiver can demonstrate the agreed cue. Food textures remain those already specified by the treating team.", "Reassess the routine in a usual meal setting. New coughing, fever, or reduced intake should be reported for clinical review.", "{org} is providing the speech rehabilitation visit for mealtime strategy practice."),
 ("dyspnea_comfort", "Comfort-focused breathing review", "hospice_service", "breathlessness during personal care under an established comfort plan", "The patient prefers a seated pause between care tasks. The caregiver is attentive to facial expression and the patient's request to stop. The team reviewed available support without changing medication orders in this conversation.", "Coordinate symptom review with the responsible clinician. Adjust the pace of personal care to the patient's stated preference.", "{org} is the hospice provider coordinating the comfort-focused nursing visits."),
 ("respite_planning", "Caregiver respite review", "hospice_service", "caregiver fatigue during continuous overnight supervision", "The caregiver describes fragmented sleep and limited time away from the home. The patient is comfortable with a short visit from another care worker. The current plan includes assistance with repositioning and hygiene.", "Confirm which respite arrangements fit the care plan and the patient's wishes. Reassess caregiver capacity after support begins.", "{org} reviewed available respite support as the patient's hospice care agency."),
 ("grief_support", "Bereavement service discussion", "hospice_service", "a family request for structured support after a recent death", "The family member describes variable sleep and difficulty returning to routine tasks. They prefer a brief telephone contact before joining a group. Immediate safety concerns were assessed during the conversation.", "Arrange the requested format of support and review whether additional clinical help is needed. Respect the person's preference about future contact.", "{org} provides the bereavement service linked to the hospice care team."),
 ("palliative_constipation", "Bowel symptom follow-up", "hospice_service", "constipation discussed during an established palliative care plan", "The patient reports reduced activity and discomfort when sitting for prolonged periods. Intake and bowel observations were reviewed with the caregiver. No medication change was made without the prescribing clinician.", "Send the symptom summary to the clinician responsible for the comfort plan. Reassess comfort and the ability to participate in preferred activities.", "{org} coordinates the palliative nursing service reviewing bowel symptoms."),
 ("bathing_assistance", "Personal care tolerance review", "hospice_service", "fatigue before completion of the usual bathing routine", "The patient prefers washing in stages and choosing the order of care. The aide observed that a rest before dressing improved participation. Privacy preferences were discussed with the caregiver present.", "Carry the patient's preferences into the next aide visit. Report a sustained decline in tolerance to the supervising nurse.", "{org} is the hospice service supervising the aide who assists with personal care."),
]

# Eight of these further situations adapt an auxiliary author's brainstorming
# without access to files, records, model behavior, or evaluation material.
ADDITIONAL_FAMILIES = [
 ("catheter_bag", "Urinary drainage bag review", "home_health", "difficulty switching between the established drainage arrangements", "The drainage tubing is not visibly kinked during the visit. The patient can identify the connections but needs help opening the fastener. The caregiver reports uncertainty about storage of the clean supplies.", "Review the routine in the existing nursing instructions. New pain, fever, or drainage concerns should be discussed with the clinical team.", "{org} arranged a visiting nurse to assess drainage-bag handling and teach the established routine."),
 ("orthostatic_symptoms", "Symptoms on standing", "home_health", "lightheadedness reported during the first transfer out of bed", "The patient describes a brief pause after sitting upright as helpful. No loss of consciousness was reported. The home record does not consistently distinguish seated from standing observations.", "Send the symptom history to the treating clinician. Review the observation method and the assistance available for transfers.", "{org} is the home-health agency providing the symptom assessment and transfer-safety review."),
 ("compression_garment", "Compression garment assistance", "home_health", "difficulty putting on the prescribed lower-leg garment", "The patient can remove the garment but cannot maintain a grip while applying it. The skin was inspected during the visit. The prescribed product and schedule were left unchanged.", "Review an application method compatible with the current plan. New pain, discoloration, or skin injury will be escalated for clinical assessment.", "{org} accepted the home nursing visit to assess garment use and skin condition."),
 ("anticoagulation_observation", "Bruising and medicine review", "home_health", "a request to review bruising while taking an established anticoagulant", "The patient describes a bruise after bumping the forearm against furniture. There is no reported active bleeding during the call. The medication list and symptom history were gathered for the treating clinician.", "The prescribing team will determine whether further assessment is needed. The home nurse will confirm the current instructions before the next visit.", "{org} provides the home nursing service communicating the symptom report to the prescriber."),
 ("hearing_instruction", "Care instructions with hearing loss", "home_health", "difficulty following spoken care instructions in a noisy room", "The patient responds more consistently when one person speaks at a time. The usual hearing aid is available. Written steps help the patient confirm the requested action.", "Use the patient's preferred communication format during teaching. Verify understanding through a practical demonstration of the existing plan.", "{org} coordinated home nursing education using accessible communication methods."),
 ("wheelchair_fit", "Wheelchair seating discomfort", "equipment_supplier", "pressure behind the knees during prolonged wheelchair use", "The foot supports appear uneven during inspection. The patient can describe the position that becomes uncomfortable. A seating assessment is needed before changing the equipment configuration.", "Coordinate any adjustment with the documented seating assessment. Review component condition and functional needs together.", "{org} arranged an equipment evaluation for wheelchair fit and component condition."),
 ("nebulizer_connector", "Nebulizer assembly review", "equipment_supplier", "a loose connection in the established nebulizer assembly", "The connector separates when the tubing is moved. The patient has retained the packaging for the supplied set. The prescribed medication regimen was not modified during the equipment discussion.", "Verify component compatibility with the supplier. Clinical concerns about treatment response will be directed to the treating team.", "{org} supplies the nebulizer equipment and accepted the component-compatibility review."),
 ("suction_canister", "Suction equipment service", "equipment_supplier", "a cracked collection canister noticed during routine cleaning", "The caregiver identified a crack near the lid attachment. The tubing and written equipment instructions are available for review. The team recorded the problem without attempting an improvised repair.", "Arrange an equipment inspection and confirm the replacement process. Clinical support needs will be reviewed with the home care team.", "{org} is the medical-equipment company handling service of the home suction unit."),
 ("mattress_pressure", "Pressure surface function review", "equipment_supplier", "loss of firmness in the prescribed pressure-relief surface", "The caregiver reports that the surface feels different after repositioning. The pump connection was visible and the display was reviewed. No new device setting was selected during this contact.", "Have the supplier inspect the surface and pump. The nursing team will separately review skin condition and positioning needs.", "{org} provides the pressure-relief equipment and scheduled its function check."),
 ("communication_device", "Communication aid charging problem", "equipment_supplier", "a communication aid that does not retain charge through daily use", "The patient can select familiar messages when the device is powered. The charger cable shows wear near its connector. A low-technology communication board remains available in the care plan.", "Coordinate the hardware review with the communication specialist. Check that the temporary method remains accessible to the patient.", "{org} supplies the communication equipment and accepted the charging-system service request."),
 ("stroke_kitchen", "Kitchen task performance", "rehabilitation", "reduced coordination affecting preparation of familiar meals", "The patient uses the unaffected hand for most observed tasks. Opening sealed containers requires additional time. The patient accurately describes the steps for a familiar cold meal.", "Occupational therapy will assess practical adaptations for selected kitchen tasks. Progress will be documented by assistance and task completion.", "{org} arranged occupational rehabilitation focused on daily kitchen activities."),
 ("prosthetic_adjustment", "Prosthetic comfort and mobility", "rehabilitation", "a sense of socket looseness during walking practice", "Discomfort limits the walking distance more than reported breathlessness. The patient uses the usual mobility aid during assessment. The fit concern was recorded before advancing the activity plan.", "Coordinate a prosthetic fit review with the treating team. Reassess mobility goals after the fit concerns are addressed.", "{org} provides the rehabilitation assessment and communication with the prosthetic service."),
 ("doorway_freezing", "Movement through confined spaces", "rehabilitation", "brief movement pauses during turns and doorway transitions", "The patient describes hesitation on entering the bathroom. Turning requires several small steps during observation. Movement is smoother along an unobstructed straight path.", "Assess movement strategies within the patient's usual environment. Track assistance needs during turns and transitions.", "{org} scheduled a physical rehabilitation assessment of turning and doorway access."),
 ("voice_endurance", "Conversational endurance", "rehabilitation", "voice fatigue limiting sustained conversation", "The voice becomes quieter during longer responses. Short spoken phrases remain understandable. The patient reports avoiding telephone conversations because speaking feels tiring.", "Assess functional communication and endurance during speech therapy. Findings needing medical evaluation will be shared with the referring clinician.", "{org} arranged speech rehabilitation focused on conversational communication."),
 ("cognitive_routine", "Household routine after brain injury", "rehabilitation", "difficulty completing multistep activities without losing the sequence", "The patient begins sorting but shifts to another task before finishing. A written checklist helps the patient return to the original task. The patient recognizes the interruption when it is discussed.", "Assess supports for household routines selected with the patient. Review progress through completion of meaningful everyday tasks.", "{org} coordinates the cognitive rehabilitation service for daily task completion."),
 ("comfort_positioning", "Comfort during repositioning", "hospice_service", "discomfort while caregivers assist with position changes", "The patient appears comfortable at rest in a preferred position. Facial tension increases during assisted movement. The caregiver wants to coordinate care activities with the established comfort plan.", "Reassess comfort during necessary care activities. The caregiver will review the established plan with the responsible hospice clinician.", "{org} arranged a hospice nursing visit to review comfort and positioning support."),
 ("communication_preferences", "Communication with progressive weakness", "hospice_service", "difficulty producing lengthy spoken responses", "The patient answers simple questions with brief words and gestures. The caregiver sometimes responds before the patient finishes. The patient consistently indicates a wish to remain involved in decisions.", "Document the patient's preferred communication methods. Allow enough time for the patient's own response during care discussions.", "{org} arranged a hospice care conference centered on the patient's communication preferences."),
 ("spiritual_support", "Spiritual care request", "hospice_service", "a request for a supportive conversation about meaning and family concerns", "The patient prefers an individual visit without a group activity. They identify familiar music as comforting. The caregiver supports the request and asks that the patient's preferences guide the conversation.", "Offer the requested visit within the established hospice plan. Confirm the patient's wishes before involving additional support people.", "{org} provides the hospice spiritual care service and accepted the request for an individual visit."),
 ("nighttime_restlessness", "Nighttime comfort review", "hospice_service", "restlessness interrupting sleep during an established comfort plan", "The caregiver describes repeated attempts to change position at night. The patient is resting during the conversation. The team gathered observations about discomfort, the room environment, and the existing support plan.", "The hospice clinician will review possible contributors to the change. Record the response to the agreed care plan at the next contact.", "{org} is the hospice provider coordinating the nighttime symptom review."),
 ("comfort_intake", "Comfort and preferred intake", "hospice_service", "reduced interest in meals during comfort-focused care", "The patient expresses a preference for small amounts of familiar food. The caregiver is concerned about the reduced intake. The discussion centers on the patient's comfort and the current clinical plan.", "Review concerns about swallowing or distress with the responsible clinician. Carry the patient's preferences into the next supportive care visit.", "{org} coordinates the hospice nursing and supportive-care visits reviewing oral comfort."),
]

BRANDS = ["Quenlark", "Velmora", "Tessavine", "Orvenlith", "Nerrowisp", "Dalsmere", "Veyrala", "Kelthren", "Pelmira", "Zorvelin", "Yevora", "Lornessa", "Fenquill", "Mervanta", "Wessaro", "Olvenix", "Talvessa", "Nimbrale", "Ardelune", "Covritha"]
SUFFIXES = {
 "home_health": [("Visiting Nurses of the Hearth", "VNH"), ("Home Health Partners", "HHP"), ("Nursing at Home Services", "NHS"), ("VNA Home Care", "VNA"), ("Home Health Cooperative", "HHC")],
 "equipment_supplier": [("Medical Equipment Supply", "MES"), ("DME Care Suppliers", "DCS"), ("Equipment for Home Health", "EHH"), ("Respiratory Equipment Services", "RES"), ("Home Medical Supply", "HMS")],
 "rehabilitation": [("Rehabilitation and Therapy Center", "RTC"), ("PT & OT Rehabilitation", "PTR"), ("Therapy for Daily Living", "TDL"), ("Restorative Care Clinic", "RCC"), ("Rehabilitation Associates", "RA")],
 "hospice_service": [("Hospice of the Lantern", "HL"), ("Palliative Care Partners", "PCP"), ("Hospice and Family Services", "HFS"), ("Comfort Care Agency", "CCA"), ("Hospice Care Collective", "HCC")],
}
CATEGORY_NOUN = {"home_health": "home-health agency", "equipment_supplier": "medical-equipment supplier", "rehabilitation": "rehabilitation provider", "hospice_service": "hospice and palliative service agency"}
GENERIC = {"home_health": "the home nursing agency", "equipment_supplier": "the medical-equipment supplier", "rehabilitation": "the rehabilitation service", "hospice_service": "the hospice support agency"}
NEGATIVE_TERMS = [
 "The discussion included ADL support and a shower chair. Nursing review is requested before the next visit.",
 "PT and OT will review use of a reacher and dressing stick during daily tasks.",
 "The equipment checklist includes DME, a transfer bench, and a bedside commode. Review requested: RN.",
 "The clinical summary retains COPD and CHF in the problem list. A pulse oximeter is included with the usual monitoring supplies.",
 "The review covered ROM, activity pacing, and safe use of a walker. Follow-up requested: physical therapist.",
 "The care discussion included PAP equipment, oxygen tubing, and basic hygiene supplies. There was no equipment order change in this review.",
 "An SLP consultation remains available under the rehabilitation plan. The caregiver asked about a built-up utensil and a nonslip mat.",
 "The support plan includes PRN assistance and a pressure-relief cushion. Visit support requested: home health aide.",
]

LONG_SECTIONS = [
 ("Daily routine", "The discussion began with how the patient moves through a usual day. Morning activity is completed slowly, with rest before a second task is attempted. The caregiver described which steps are already manageable and which ones still need another person. Preferences were recorded in practical terms so the next visit can begin with the same routine. The patient remained involved in deciding the order of care and asked that explanations be brief before each activity."),
 ("Environment", "The usual route between the resting area and the bathroom was reviewed verbally. Frequently used items should be reachable without an unnecessary turn or stretch. The patient described lighting and furniture arrangement rather than requesting a new level of care. The team recorded the environmental concerns that can be revisited during a home assessment. No conclusion about independent safety was made solely from this discussion; direct observation remains part of the planned review."),
 ("Understanding", "The patient was asked to describe the plan in their own words. The first explanation was broad, and a second discussion focused on the next concrete step. The caregiver used an example from the daily routine to check that both had the same understanding. Written material was placed with the existing care information. Questions about changes to treatment were kept for the responsible clinician rather than answered from an outdated instruction sheet."),
 ("Participation", "Participation varied with fatigue during the review. A brief rest allowed the patient to resume the discussion and identify the task that mattered most. The team distinguished what the patient had demonstrated from what was reported by the caregiver. Assistance was described by the action required, such as steadying an item or repeating a cue, so that the next clinician would not infer more independence than had been observed."),
 ("Caregiver review", "The caregiver explained when help is available and when the patient is usually alone. A potential gap in support was recorded for follow-up rather than assumed to be covered. The patient preferred to attempt the first part of the task before help was offered. Both agreed that the visit should include time to practice the routine. The team will revisit the arrangement if fatigue or the amount of assistance changes."),
 ("Equipment and records", "Available supplies were reviewed against the task being discussed. The patient could locate the written instructions but had not brought every item to the review. The team recorded the missing information for a later check. Equipment appearance alone was not taken as proof of correct function. Any repair, replacement, or change to the treatment plan will be confirmed through the appropriate clinical or equipment service before the routine is revised."),
 ("Response to practice", "The observed practice was brief and stayed within the existing care plan. The patient described which part felt easier after explanation and which part still required help. The caregiver was asked to wait for the patient's attempt before adding a cue. The team documented the response without treating a single successful attempt as proof that support could end. Ongoing review will consider comfort, tolerance, and consistency in the usual setting."),
 ("Escalation and continuity", "The closing discussion separated practical service questions from new clinical symptoms. The patient and caregiver could state how they would contact the care team if the routine became difficult to complete. Unresolved questions were left in the handoff with the action needed, not simply marked as complete. The next review will check whether the service response addressed the original concern and whether the patient's preference or support needs have changed."),
]

def mark(label, value):
    return f"[[{label}|{value}]]"

def unmark(value):
    parts, spans, last, offset = [], [], 0, 0
    for m in re.finditer(r"\[\[([A-Z]+)\|([^\[\]]+)\]\]", value):
        literal = value[last:m.start()]
        parts.extend((literal, m[2])); offset += len(literal)
        spans.append({"start": offset, "end": offset + len(m[2]), "label": m[1], "text": m[2]})
        offset += len(m[2]); last = m.end()
    parts.append(value[last:]); text = "".join(parts)
    assert "[[" not in text and "]]" not in text
    return text, spans

def group(name):
    return hashlib.sha256(unicodedata.normalize("NFC", name).casefold().encode()).hexdigest()

def person(index):
    given = ["Mara", "Oren", "Talia", "Emil", "Lina", "Soren", "Alina", "Niko", "Vera", "Ilan", "Amara", "Elio", "Zora", "Milan", "Nadia", "Luca", "Noemi", "Ari", "Mina", "Theo"]
    last = ["Fenliss", "Quenvar", "Orvellan", "Tessarik", "Nimbrath", "Velrissa", "Kelvorn", "Dalmire", "Yevrith", "Pelquinn", "Mervellis", "Zorveth", "Lorvane", "Wessalik", "Arvenquill", "Covrell", "Olvessan", "Talverin", "Veymora", "Nerrivan"]
    cursor = index
    while True:
        name = f"{given[cursor % 20]} {last[(cursor // 20) % 20]}"
        digest = group(name)
        if int(digest[:8], 16) % 10 < 8:
            return name, digest, cursor + 1
        cursor += 1

def provider(category, index):
    brand = BRANDS[index % len(BRANDS)]
    suffix, abbreviation = SUFFIXES[category][(index // len(BRANDS)) % 5]
    legal = ["", ", Inc.", ", LLC", "", ""][index % 5]
    full = f"{brand} {suffix}{legal}"
    # The acronym is explicitly established within every note using it.
    short = brand[0].upper() + abbreviation
    if index % 7 == 0:
        short = ".".join(short) + "."
    return full, short

def layout_note(layout, title, sections):
    if layout == 0:
        return title + "\n\n" + "\n\n".join(s for h, s in sections)
    if layout == 1:
        return title + "\n" + "\n".join(f"{h}: {s}" for h, s in sections)
    if layout == 2:
        return "Telephone care review — " + title + "\n" + "\n".join(f"- {h}\n  {s}" for h, s in sections)
    if layout == 3:
        return "Continuity handoff\n" + title + "\n\n" + "\n\n".join(f"{h.upper()}\n{s}" for h, s in sections)
    if layout == 4:
        return "Interdisciplinary review | " + title + "\n" + "\n".join(f"{i + 1}. {h} — {s}" for i, (h, s) in enumerate(sections))
    if layout == 5:
        return "Care coordination worksheet\n" + title + "\n\nTopic | Narrative\n" + "\n".join(f"{h} | {s}" for h, s in sections)
    if layout == 6:
        return title + "\n" + "\n\n".join(f"[{h}]\n{s}" for h, s in sections)
    return "Follow-up summary\n" + title + "\n\n" + "\n\n".join(f"{h}\n• {s}" for h, s in sections)

def build_row(family, family_number, local, serial, variant, identity):
    key, title, category, reason, observations, plan, action = family
    name, digest = identity
    control = variant == 3
    form = (serial + variant) % 3 if not control else serial % 3
    long = serial % 10 < 3
    layout = serial % 8
    primary, acronym = provider(category, serial + family_number * 13)
    other_category = {"home_health":"equipment_supplier", "equipment_supplier":"rehabilitation", "rehabilitation":"equipment_supplier", "hospice_service":"home_health"}[category]
    secondary, secondary_acronym = provider(other_category, serial + 57)
    casing = serial % 6
    if casing == 1:
        primary, secondary = primary.upper(), secondary.upper()
    elif casing == 2:
        base_brand = BRANDS[(serial + family_number * 13) % 20]
        midpoint = len(base_brand) // 2
        primary = primary.replace(base_brand, base_brand[:midpoint] + base_brand[midpoint:].capitalize())
    full = GENERIC[category] if control else mark("FACILITY", primary)
    short = GENERIC[category] if control else mark("FACILITY", acronym)
    ref = full if form == 0 else short
    second_ref = GENERIC[other_category] if control else mark("FACILITY", secondary)
    # A complete proper name includes its internal preposition/legal suffix.
    action_sentence = action.format(org=full)
    repeated_action = action.format(org=ref)
    if control:
        action_sentence = action_sentence[0].upper() + action_sentence[1:]
        repeated_action = repeated_action[0].upper() + repeated_action[1:]
    if form == 0:
        introduction = action_sentence
    elif form == 1:
        introduction = action_sentence
        introduction += (" The agency confirmed its role in the current care plan." if control else f" The provider is abbreviated {short} in the care plan.")
    else:
        introduction = (f"The care plan lists {short} as the {CATEGORY_NOUN[category]}. " +
            ("The service confirmed that it will handle this part of care. " if control else f"The agency's full name is {full}. ") + repeated_action)
    patient = mark("NAME", name) if variant == 0 else "the patient"
    age = 46 + serial % 39
    opening = f"Care review for {patient}, age {age}, concerned {reason}."
    # The clinical variation is identical across each positive/control quartet.
    context_variant = [
        "The patient prefers a brief explanation before an observed task.",
        "The caregiver attended part of the discussion with the patient's agreement.",
        "Written instructions were reviewed using the patient's usual glasses.",
        "The patient asked to begin with the activity that causes the most frustration.",
        "A rest break was taken before the practical part of the review.",
        "The patient described the home routine before the team reviewed the plan.",
        "The caregiver brought the current list of supplies for comparison.",
        "The patient repeated the main concern at the end to confirm the handoff.",
    ][local]
    sections = [("Reason for review", opening + " " + context_variant), ("Clinical observations", observations), ("Service involvement", introduction)]
    organizations = [(primary, category, "expanded")]
    if form:
        organizations.append((acronym, category, "acronym"))
    if serial % 3 == 0:
        sections.append(("Coordination", f"The team also contacted {second_ref}, a separate {CATEGORY_NOUN[other_category]}, to clarify the available support. The services were asked to describe their respective responsibilities before the plan was finalized."))
        organizations.append((secondary, other_category, "expanded"))
    if long:
        for offset in range(8):
            heading, paragraph = LONG_SECTIONS[(offset + serial) % len(LONG_SECTIONS)]
            # Every long section re-establishes service type locally, including
            # acronym mentions, so an owning window has actual local context.
            lead = [
                f"As the {CATEGORY_NOUN[category]} involved in care, {ref} was asked to review this concern. ",
                f"The care team asked {ref}, the {CATEGORY_NOUN[category]}, to include this issue in the next service review. ",
                f"The next visit from {ref}, a {CATEGORY_NOUN[category]}, will include this part of the care plan. ",
                f"A representative of {ref}, the {CATEGORY_NOUN[category]} providing support, reviewed the following issue with the team. ",
            ][(serial + offset) % 4]
            # Vary whether provider identity comes before or after prose.
            body = (lead + paragraph) if (serial + offset) % 2 else (paragraph + " " + lead.strip())
            sections.append((heading, body))
    sections.append(("Care resources", NEGATIVE_TERMS[serial % 8]))
    # Some positive rows include a locality/address distinct from the provider;
    # matched controls retain equivalent generic geographic context.
    if serial % 11 == 0:
        locality = "North Quenmere, UT"
        address = f"{1400 + serial} Fictional Lantern Way, Quenmere, UT 84000"
        where = "in the patient's area" if control else "in " + mark("LOCATION", locality)
        contact = "at its service office" if control else "at " + mark("ADDRESS", address)
        sections.append(("Service location", f"The service office for {full} is {where}. The equipment and care records can be reviewed {contact}; the care coordination desk will arrange access."))
    if serial % 5 == 0:
        ending = f"The {full if not control else CATEGORY_NOUN[category]} care coordination desk will confirm the next review. "
    else:
        ending = f"The team will share the follow-up summary with {ref}. "
    sections.append(("Plan", ending + plan))
    marked_note = layout_note(layout, title, sections)
    # A legal suffix or dotted acronym can already supply the terminal period.
    # Retain its authored period inside FACILITY and do not add another one.
    marked_note = re.sub(r"(\.\]\])\.", r"\1", marked_note)
    text, spans = unmark(marked_note)
    phenomena = ["provider_organization", category, "generic_service_contrast", "clinical_acronyms_not_facilities", "equipment_terms_not_facilities", "unsigned_job_roles", "matched_control" if control else "proper_provider_name", ["spelled_name", "expanded_before_acronym", "acronym_before_expanded"][form]]
    if long:
        phenomena.extend(["multiple_actual_windows", "local_provider_context"])
    if serial % 3 == 0:
        phenomena.append("multiple_provider_organizations")
    if serial % 11 == 0:
        phenomena.append("separate_locality_address")
    row = {"id": f"tk12_provider_{serial:04d}_{variant}", "text": text, "spans": spans, "split": "train", "annotation_policy": "clinical-phi-v2", "identity_group": digest, "identity_groups": [digest], "pair_id": f"tk12_provider_pair_{serial:04d}", "template_family": f"tk12_provider_{key}", "phenomena": phenomena, "cohort": "long_document" if long else "realistic", "negative_only": control, "layout": f"provider_layout_{layout}", "authoring_metadata": {"organization_category": category, "name_form": ["spelled", "expanded_first", "acronym_first"][form], "casing": ["title", "upper", "mixed", "title", "title", "title"][casing], "length_stratum": "long" if long else "short", "variant": variant}}
    return row, organizations if not control else []

def load_function_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def load_windows():
    # Compile only the pure window function; never import inference machinery.
    source = (ROOT / "src/clinical_deid/hf_encoder.py").read_text()
    parsed = ast.parse(source)
    functions = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "window_jobs"]
    assert len(functions) == 1
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(ROOT / "src/clinical_deid/hf_encoder.py"), "exec"), namespace)
    return namespace["window_jobs"]

def main():
    assert len(FAMILIES + ADDITIONAL_FAMILIES) == 40
    assert not OUT.exists() and not OUT.with_suffix(".jsonl.manifest.json").exists(), "Refusing to overwrite frozen artifacts"
    REPORT.mkdir(parents=True, exist_ok=True)
    rows, authored = [], {}
    serial = cursor = 0
    for family_number, family in enumerate(FAMILIES + ADDITIONAL_FAMILIES):
        for local in range(8 if family_number < 20 else 7):
            name, digest, cursor = person(cursor)
            for variant in range(4):
                row, organizations = build_row(family, family_number, local, serial, variant, (name, digest))
                rows.append(row); authored[row["id"]] = organizations
            serial += 1
    assert len(rows) == 1200 and len({r["id"] for r in rows}) == 1200
    assert len({r["text"] for r in rows}) == 1200
    conventions = load_function_module(ROOT / "scripts/takeover/normalize_conventions.py", "author12_conventions")
    intervals = load_function_module(ROOT / "scripts/takeover/normalize_date_intervals.py", "author12_intervals")
    tokenizer = Tokenizer.from_file(str(MODEL / "hf/tokenizer.json"))
    outer = json.loads((MODEL / "config.json").read_text())
    probe = "clinical note"
    special = len(tokenizer.encode(probe, add_special_tokens=True).ids) - len(tokenizer.encode(probe, add_special_tokens=False).ids)
    content_window = outer["max_length"] - special
    window_jobs = load_windows()
    support, organization_support, form_support, cohort_support, layout_support = Counter(), Counter(), Counter(), Counter(), Counter()
    length_records, ownership = [], []
    cues = defaultdict(Counter)
    normalization_changes = Counter()
    for row in rows:
        assert ".." not in row["text"] and "care.the" not in row["text"]
        assert row["spans"] == sorted(row["spans"], key=lambda s: s["start"])
        previous = 0
        for span in row["spans"]:
            assert span["label"] in LABELS and previous <= span["start"] < span["end"] <= len(row["text"])
            assert row["text"][span["start"]:span["end"]] == span["text"] == span["text"].strip()
            previous = span["end"]; support[span["label"]] += 1
        assert all(int(g[:8], 16) % 10 < 8 for g in row["identity_groups"])
        assert row["identity_group"] in row["identity_groups"]
        assert row["negative_only"] == (not row["spans"])
        assert conventions.normalize(deepcopy(row), normalization_changes) == row
        normalized, changes, exclusions = intervals.normalize_row(deepcopy(row))
        assert normalized == row and not changes
        # Each literal occurrence of each actually used authored institution is
        # checked. Clinical acronyms are never discovered through uppercase rules.
        for value, category, form in authored[row["id"]]:
            annotated = {(s["start"], s["end"]) for s in row["spans"] if s["label"] == "FACILITY" and s["text"] == value}
            occurrences = {(m.start(), m.end()) for m in re.finditer(re.escape(value), row["text"])}
            assert annotated == occurrences, (row["id"], value, annotated, occurrences)
            organization_support[category] += len(annotated)
            form_support[form] += len(annotated)
        cohort_support[(row["cohort"], "control" if row["negative_only"] else "positive")] += 1
        layout_support[(row["layout"], "control" if row["negative_only"] else "positive")] += 1
        encoded = tokenizer.encode(row["text"], add_special_tokens=False)
        jobs = list(window_jobs(len(encoded.ids), content_window))
        expected_long = row["cohort"] == "long_document"
        assert (len(jobs) > 1) == expected_long, (row["id"], len(encoded.ids), len(jobs))
        length_records.append({"id": row["id"], "tokens": len(encoded.ids), "windows": len(jobs), "control": row["negative_only"], "cohort": row["cohort"], "layout": row["layout"]})
        if expected_long and not row["negative_only"]:
            row_ownership = []
            for wi, (start, stop, left, right) in enumerate(jobs):
                window_text = row["text"][encoded.offsets[start][0]:encoded.offsets[stop-1][1]]
                assert CATEGORY_NOUN[row["authoring_metadata"]["organization_category"]] in window_text, (row["id"], wi, "missing provider type in window")
                owned_facilities = []
                for span in row["spans"]:
                    if span["label"] != "FACILITY": continue
                    owned = [i for i, (a,b) in enumerate(encoded.offsets) if a < span["end"] and b > span["start"] and left <= i < right]
                    if owned:
                        owned_facilities.append({"text": span["text"], "start": span["start"], "end": span["end"], "owned_tokens": len(owned)})
                assert owned_facilities, (row["id"], wi, "window has no local organization")
                row_ownership.append({"window": wi, "tokens": [start,stop], "owned": [left,right], "facilities": owned_facilities})
            ownership.append({"id": row["id"], "windows": row_ownership})
        stratum = "control" if row["negative_only"] else "positive"
        for cue in ["agency", "supplier", "provider", "nurse", "therapy", "equipment", "care", "plan", "patient", "ADL", "DME", "PT", "OT", "RN", "PAP", "COPD", "CHF"]:
            cues[cue][stratum] += bool(re.search(r"\b" + re.escape(cue) + r"\b", row["text"], re.I))
    assert not normalization_changes
    assert sum(r["negative_only"] for r in rows) == 300
    assert sum(r["cohort"] == "long_document" for r in rows) == 360
    assert len({r["template_family"] for r in rows}) == 40
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode()
    sha256 = hashlib.sha256(payload).hexdigest()
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    semantic_review = {
      "method": "Author reviewed all 40 context definitions and all 20 provider suffix/category combinations, all literal span construction branches, and stratified rendered notes; generated repetitions are mechanically checked occurrence by occurrence.",
      "provider_scope": "All brands refer explicitly to organizations providing home nursing, medical equipment, rehabilitation, or hospice/palliative support; no general employer cases.",
      "boundaries": "Full brands retain internal of/the/for/and/at, PT & OT or VNA/DME brand components and authored legal suffixes. Leading the, care coordination desk, distinct locality and full address are outside FACILITY.",
      "negative_semantics": "Generic service organizations/descriptions, equipment items, clinical acronyms and unsigned job roles are left unannotated in positives and controls.",
      "person_scope": "One invented canonical patient per quartet; positive variant 0 names that patient. Other variants and controls inherit the same identity metadata without naming the patient.",
      "limitations": ["Template-authored clinical scenarios, not sampled clinical records or clinically validated documentation.", "Shared long review paragraphs and quartet structure create dependence; 1200 rows are not 1200 independent clinical observations.", "Fictional institution names are locally invented and not checked against live registries; coincidental real-world collisions cannot be excluded.", "Controls preserve scenarios, layouts, clinical vocabulary and long/short assignments, but proper-name/acronym introductions necessarily differ from generic descriptions.", "Long notes explicitly repeat provider identity with local care context; this is an authored diagnostic construction.", "No external overlap audit, independent semantic QA, model inference, admission, or training is performed by this author."]
    }
    def summarize(records):
        lengths = sorted(r["tokens"] for r in records)
        return {"rows": len(records), "min_tokens": min(lengths), "median_tokens": statistics.median(lengths), "max_tokens": max(lengths), "multi_window": sum(r["windows"] > 1 for r in records), "window_counts": dict(Counter(r["windows"] for r in records))}
    report = {"corpus_sha256": sha256, "rows": len(rows), "controls": 300, "positive_rows": 900, "families": 40, "layouts": 8, "label_support": {label:support[label] for label in LABELS}, "organization_category_span_support": dict(organization_support), "organization_form_span_support": dict(form_support), "category_row_support": dict(Counter(r["authoring_metadata"]["organization_category"] for r in rows)), "form_row_support": dict(Counter(r["authoring_metadata"]["name_form"] for r in rows)), "cohort_support": {"|".join(k):v for k,v in cohort_support.items()}, "layout_support": {"|".join(k):v for k,v in layout_support.items()}, "tokenizer": {"file": str(MODEL / "hf/tokenizer.json"), "sha256": hashlib.sha256((MODEL / "hf/tokenizer.json").read_bytes()).hexdigest(), "max_length": outer["max_length"], "special_tokens": special, "content_window": content_window, "window_function": "AST-extracted exact hf_encoder.window_jobs"}, "lengths": {s:summarize([r for r in length_records if ("control" if r["control"] else "positive") == s]) for s in ("positive", "control")}, "length_cohorts": {f"{c}|{s}":summarize([r for r in length_records if r["cohort"] == c and ("control" if r["control"] else "positive") == s]) for c in ("long_document", "realistic") for s in ("positive", "control")}, "cue_rates": {cue:{"positive":count["positive"]/900,"control":count["control"]/300,"absolute_gap":abs(count["positive"]/900-count["control"]/300)} for cue,count in cues.items()}, "checks": {"unique_ids_texts": True, "full_offsets_and_authored_occurrences": True, "nonoverlap": True, "all_identity_hashes_train": True, "both_whole_row_normalizers_unchanged": True, "long_windows_have_local_organization_spans": True}, "semantic_review": semantic_review}
    manifest = {"role":"train", "split":"train", "annotation_policy":"clinical-phi-v2", "status":"training_candidate_authored", "sha256":sha256, "rows":len(rows), "documents":len(rows), "label_support":{label:support[label] for label in LABELS}, "provenance":{"kind":"independently_authored_synthetic_training", "author":"author12_provider_organizations", "source_script":str(Path(__file__).relative_to(ROOT)), "source_script_sha256":source_sha, "source_corpora":[], "model_inference":False, "external_material":False, "generation":"40 authored clinical context families, 8 layouts, 300 matched identity quartets, 3 positive variants and 1 control per quartet", "auxiliary_author":"Eight situations adapted from independently composed brainstorming without file access; remaining scenarios composed by main author"}, "synthetic_schemes":{"people":"Invented combinations of ordinary given names and constructed surnames, canonical SHA256 NFC/casefold, train hash partition only", "organizations":"Constructed non-geographic brand roots plus explicitly authored provider-service names and established acronym aliases; no real provider names intentionally used", "locations":"Invented Quenmere locality", "addresses":"Invented numbered Fictional Lantern Way in Quenmere, UT 84000; no actual address source", "other_identifiers":"None"}, "support_report":str(REPORT.relative_to(ROOT) / "self_check.json"), "limitations":semantic_review["limitations"]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("xb") as handle: handle.write(payload)
    manifest_path = OUT.with_suffix(".jsonl.manifest.json")
    with manifest_path.open("x") as handle: json.dump(manifest, handle, ensure_ascii=False, indent=2); handle.write("\n")
    for name, obj in [("self_check.json",report), ("token_lengths.json",length_records), ("window_ownership.json",ownership)]:
        with (REPORT / name).open("x") as handle: json.dump(obj,handle,ensure_ascii=False,indent=2); handle.write("\n")
    OUT.chmod(0o444); manifest_path.chmod(0o444)
    print(json.dumps({"path":str(OUT), "sha256":sha256, "manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "rows":len(rows), "controls":300,"label_support":dict(support),"organization_category_support":dict(organization_support), "organization_form_support":dict(form_support),"length_cohorts":report["length_cohorts"]},ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
