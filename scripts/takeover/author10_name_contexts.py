"""Independently authored, deterministic synthetic training bank. No inference.

Reads only the declared policy/runtime references and its own output artifacts.
Exclusive output creation and a version argument preserve any earlier corpus.
"""
from __future__ import annotations

import argparse
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

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
POLICY = "clinical-phi-v2"
LABELS = ("NAME", "DATE", "ADDRESS", "PHONE", "EMAIL", "MRN", "SSN", "ACCOUNT", "ID", "URL", "IP", "AGE", "LOCATION", "FACILITY")
MODEL = ROOT / "models/takeover-mbbase-mix3-001/step-003081"
REFERENCE_PATHS = [
    "docs/annotation-policy-v2.md", "src/clinical_deid/annotation_policy.py",
    "src/clinical_deid/types.py", "src/clinical_deid/hf_encoder.py",
    "scripts/takeover/normalize_conventions.py", "scripts/takeover/normalize_date_intervals.py",
    "models/takeover-mbbase-mix3-001/step-003081/hf/tokenizer.json",
    "models/takeover-mbbase-mix3-001/step-003081/hf/tokenizer_config.json",
    "models/takeover-mbbase-mix3-001/step-003081/hf/config.json",
    "models/takeover-mbbase-mix3-001/step-003081/config.json",
]

# Each clinical family was composed for this bank, independently of other data.
FAMILIES = [
    ("vestibular_rehab", "Balance and turning", "Brief unsteadiness occurs when turning toward a cupboard. There has been no fall, fainting, or new hearing change.", "Walking is stable in the straight corridor. A slow head turn reproduces mild motion sensitivity without loss of footing.", "The current limitation is confidence with turning rather than tolerance for a level walk.", "Practice will start beside a stable counter, with a pause between changes of direction. The patient can describe when to stop and ask for help."),
    ("voice_fatigue", "Voice review", "The voice becomes tired after long conversations. Rest improves the effort, and meals do not provoke coughing.", "Speech is understandable at a comfortable volume. Neck and shoulder tension increase when the patient tries to project across the room.", "The patient is compensating for vocal fatigue by pushing harder, which seems to increase discomfort.", "The discussion focused on shorter speaking periods, access to water, and reducing background noise. Further voice assessment remains part of the follow-up plan."),
    ("hand_strain", "Hand therapy", "Aching follows prolonged gripping during household tasks. Symptoms ease when the handle is released and do not wake the patient.", "The hand opens fully and light touch is reported throughout the fingers. Repeated squeezing brings on the familiar ache without visible swelling.", "Task duration and grip force appear more limiting than a single movement.", "The therapist demonstrated a lighter grip and a wider handle. The patient chose one short task on which to practice before increasing the workload."),
    ("tension_headache", "Headache follow-up", "A gradual tight sensation follows extended desk work. The pattern is unchanged, with no sudden onset, fever, or new weakness.", "The patient converses comfortably and moves the neck through the usual range. Shoulder elevation increases the familiar tightness.", "Long uninterrupted work periods may be contributing to the symptom pattern.", "The patient will record task duration and response to brief breaks. The reviewed escalation advice includes seeking prompt assessment for a distinctly different or sudden severe headache."),
    ("sleep_routine", "Sleep routine discussion", "Sleep timing drifts after evening naps. The patient feels more alert late at night and reports difficulty settling after going to bed early.", "The patient is attentive during the visit and describes the routine consistently. No breathing concern or unsafe sleepiness is reported.", "An irregular routine is a practical target for the next review.", "A feasible wake time and a quieter wind-down period were discussed. The patient selected a small change that fits ordinary responsibilities."),
    ("dermatitis", "Skin barrier review", "A dry itchy patch returns after repeated hand washing. There is no fever, drainage, or rapidly spreading redness.", "The affected area is dry with mild scaling. There are no open fissures or signs of a current spreading infection.", "Frequent wet work seems to aggravate the existing skin irritation.", "The patient reviewed the established skin-care plan and described how it fits around washing. Reassessment was advised if the area becomes painful, hot, or otherwise changes."),
    ("swallow_followup", "Mealtime follow-up", "The patient returns to discuss the previously assessed meal routine. Smaller bites feel easier, and no new choking episode is reported.", "The patient explains the agreed positioning and pacing sequence. No food trial was undertaken during this conversation.", "The main need is to carry the established recommendations into a busy meal setting.", "The written plan was reviewed without changing the recommended textures. The patient identified a quiet place to sit and a way to keep meals unhurried."),
    ("tinnitus", "Listening comfort review", "A steady background sound is most noticeable in a quiet room. The pattern is unchanged, without a sudden hearing loss or new imbalance.", "Conversation at an ordinary volume is comfortable. The patient can describe the difference between the background sound and external room noise.", "Quiet evening periods are the setting in which the sound is most intrusive.", "Previously discussed sound-enrichment options were reviewed at a comfortable level. The patient selected a simple way to compare comfort across activities."),
    ("knee_rehab", "Knee rehabilitation", "The knee aches after repeated stairs but settles with a break. There has been no locking, giving way, or new injury.", "The patient rises from a chair without assistance. A slow step-up produces the familiar discomfort without loss of balance.", "Repeated loading currently exceeds the comfortable activity dose.", "The exercise sequence was practiced with a lower step and a slower pace. The patient will use the agreed symptom response to judge whether to maintain or reduce repetitions."),
    ("inhaler_technique", "Inhaler teaching", "The patient is unsure about the sequence for the prescribed inhaler. There is no current breathlessness at rest or acute chest symptom.", "A demonstration device was used to observe hand position and timing. The main error occurred when the inhalation started before the device was ready.", "Technique needs reinforcement before changing conclusions about the existing regimen.", "The patient repeated the sequence with the demonstration device and checked the written instructions. Medication changes were left to the prescribing clinician."),
    ("dry_eye", "Eye comfort follow-up", "Grittiness occurs after sustained reading and improves with blinking. There is no sudden loss of vision, significant pain, or light sensitivity.", "The patient reads the instruction sheet comfortably under ordinary lighting. Frequent blinking is noted during a longer paragraph.", "Long reading sessions are the most consistent reported trigger for discomfort.", "The current eye-care instructions and planned review were revisited. The patient chose shorter reading segments and knows to seek assessment for a sudden change."),
    ("ankle_rehab", "Ankle mobility visit", "Stiffness follows a period of sitting after an earlier ankle strain. Walking becomes easier after a gentle warm-up, and no new injury is reported.", "The patient walks without an aid on the level surface. Ankle movement is mildly restricted but does not provoke a sharp pain.", "Start-up stiffness remains a barrier to comfortable daily activity.", "The patient practiced the established mobility sequence at an easy pace. The plan emphasizes a gradual return within the therapist's previous loading guidance."),
    ("reflux_review", "Digestive symptom review", "A burning sensation sometimes follows a large evening meal. The patient reports no painful swallowing, bleeding, or unintended weight loss.", "The patient appears comfortable and describes a stable pattern. This visit centers on history and review of the existing plan.", "Meal size and the interval before lying down are modifiable parts of the reported routine.", "The patient reviewed the current treatment instructions with the clinician. A short record of meals and symptoms was chosen to support the next conversation."),
    ("fatigue_pacing", "Activity pacing", "Energy drops after completing several chores without a break. A short rest is helpful, but the patient tends to resume all remaining tasks at once.", "The patient takes part in the discussion without distress. The activity account distinguishes tiredness from breathlessness or chest discomfort.", "A burst-and-rest pattern makes the daily workload difficult to predict.", "The patient divided one routine into smaller parts and planned a pause before marked fatigue. The plan can be revised if the underlying symptoms change."),
    ("constipation", "Bowel routine review", "The patient describes infrequent firm stools with a stable pattern. There is no severe abdominal pain, vomiting, or reported bleeding.", "The patient is comfortable during the discussion. The current medication list and previously provided bowel plan were reviewed together.", "The immediate task is understanding which parts of the established routine are difficult to follow.", "The patient chose a manageable way to track symptoms and intake. Questions about medication adjustment were directed to the treating clinician."),
    ("jaw_tension", "Jaw comfort review", "Jaw tightness is noticed during concentrated work. The patient does not describe locking, dental trauma, or a sudden change in the bite.", "Opening and closing are smooth at a comfortable range. The familiar tension appears when the patient clenches during a demonstration.", "Unnoticed clenching may be contributing to the reported discomfort.", "The patient practiced noticing resting jaw position and reviewed the existing dental follow-up plan. Forceful stretching was not part of the session."),
    ("low_back_rehab", "Back rehabilitation", "A familiar lower back ache follows prolonged sitting. There is no new weakness, altered bladder control, or saddle sensory change.", "The patient changes position independently and walks comfortably across the room. A supported sitting position reduces the familiar ache.", "Static posture is the most reproducible current activity trigger.", "The patient selected brief position changes during an ordinary work period. The agreed exercise plan was reviewed with attention to comfort and pacing."),
    ("footwear_review", "Foot protection visit", "The patient has reduced foot sensation and wants to check the fit of everyday shoes. No new wound or hot swollen area is reported.", "The shoes show pressure at the toe seam. The visible skin remains intact, and the patient cannot reliably judge tightness by sensation alone.", "Fit and visual skin checks are more useful here than waiting for discomfort.", "The clinician reviewed the established foot-care advice and options for fit assessment. The patient explained how to inspect the skin after removing the shoes."),
    ("wound_check", "Dressing review", "A small healing skin wound is being managed under the existing plan. The patient reports no fever, increasing pain, or new drainage.", "The surrounding skin is intact and the dressing is secure. No spreading redness or unexpected moisture is seen.", "The wound-care routine is currently manageable, but the patient wants clarification about supplies.", "The nurse reviewed the already prescribed dressing sequence and the signs that warrant reassessment. Supply questions were documented for the treating team."),
    ("home_pressure", "Home pressure technique", "Home readings vary when the patient measures immediately after household activity. There is no current headache, chest pain, or acute neurologic symptom.", "The cuff can be positioned correctly once the arm is supported. The patient had previously talked during some measurements.", "Differences in measurement conditions may contribute to the inconsistent log.", "The clinician reviewed a quiet seated technique using the current device. The patient will bring the log and device to the planned review rather than independently changing medication."),
    ("shoulder_stiffness", "Gradual shoulder stiffness", "The shoulder feels stiff when reaching toward a high shelf or putting on a jacket. There was no preceding injury.", "Active and passive shoulder motion are mildly limited. There is no swelling or focal weakness.", "The pattern suggests a gradual capsular restriction without an acute injury.", "Gentle range of motion exercises and routine reassessment were discussed."),
    ("lateral_hip_discomfort", "Lateral hip discomfort", "Aching develops along the outer hip after prolonged walking and when lying on that side. There is no groin pain or leg numbness.", "There is localized tenderness over the lateral hip with preserved joint motion.", "The symptoms are consistent with irritation of the tissues surrounding the greater trochanter.", "Activity adjustment and a gradual strengthening program were reviewed."),
    ("plantar_heel_pain", "Plantar heel discomfort", "The underside of the heel hurts during the first steps after rest. The discomfort eases with gentle movement.", "Tenderness is present near the medial plantar heel. There is no redness or swelling.", "The history and examination support plantar fascia irritation.", "Gentle stretching and gradual modification of aggravating activity were discussed."),
    ("seasonal_nasal_symptoms", "Seasonal nasal symptoms", "Sneezing and clear nasal drainage increase with seasonal changes. There is no fever or facial pain.", "The nasal lining appears pale and mildly swollen. Breathing is comfortable.", "The symptoms fit a mild allergic nasal pattern.", "Trigger awareness and routine symptom management options were reviewed."),
    ("facial_acne", "Mild facial acne", "Small facial bumps recur despite a simple cleansing routine. The lesions are occasionally tender but do not leave deep scars.", "Scattered closed comedones and superficial inflammatory papules are present.", "The findings are consistent with mild acne.", "A gentle skin care routine and gradual introduction of a topical treatment were discussed."),
    ("urinary_urgency", "Urinary urgency", "A sudden urge to urinate sometimes interrupts routine activities. There is no burning, visible blood, or difficulty emptying the bladder.", "The abdomen is soft and nontender. There is no flank tenderness.", "The symptoms suggest bladder urgency without current signs of an acute infection.", "A symptom diary and review of bladder irritants were discussed for routine follow-up."),
    ("vasomotor_symptoms", "Intermittent hot flashes", "Brief episodes of warmth and sweating occur during ordinary activities. There is no accompanying chest discomfort or faintness.", "The examination is unremarkable between episodes.", "The reported episodes are compatible with menopausal vasomotor symptoms.", "Comfort measures and individualized treatment options were reviewed for follow-up."),
    ("restless_legs", "Restless leg sensations", "An uncomfortable urge to move the legs develops while sitting quietly in the evening. Walking temporarily relieves the sensation.", "Strength, sensation, and gait are intact.", "The symptom pattern is suggestive of restless legs.", "Potential contributing factors and the role of routine laboratory review were discussed."),
    ("action_tremor", "Stable action tremor", "A mild hand tremor is noticeable while holding a cup or using utensils. The symptoms have remained stable and do not occur at rest.", "A fine symmetric tremor appears with sustained posture. Gait and muscle tone are normal.", "The findings suggest a mild action tremor.", "Practical adaptations and observation for changes in function were discussed."),
    ("dependent_leg_swelling", "Dependent leg swelling", "Both lower legs feel mildly swollen after prolonged standing. The swelling improves with overnight rest and is not painful.", "Mild symmetric ankle edema is present without warmth or skin breakdown.", "The pattern is compatible with dependent edema, with contributing factors still to be reviewed.", "Movement breaks, leg elevation, and routine evaluation of persistent swelling were discussed."),
    ("thickened_toenails", "Thickened toenails", "Several toenails have gradually become thicker and harder to trim. There is no surrounding pain or drainage.", "The affected nails are discolored with debris beneath the nail edges. The surrounding skin is intact.", "The nail changes may reflect a fungal process, although confirmation would guide treatment.", "Nail care and options for diagnostic sampling were reviewed."),
    ("diffuse_hair_shedding", "Diffuse hair shedding", "More hair is noticed during washing and brushing following a period of increased stress. There are no discrete bald patches.", "Hair thinning is diffuse without scalp redness, scarring, or broken hairs.", "The pattern is compatible with increased shedding, with nutritional and other contributors remaining possible.", "Gentle hair care and routine review of potential contributing factors were discussed."),
    ("meal_related_bloating", "Meal related bloating", "Abdominal fullness occurs after larger meals and settles without intervention. There is no persistent pain or change in appetite.", "The abdomen is soft and nondistended without focal tenderness.", "The symptoms suggest mild meal related bloating without concerning examination findings.", "A food and symptom diary and smaller meal portions were discussed."),
    ("hemorrhoidal_irritation", "Intermittent hemorrhoidal irritation", "Mild anal itching and irritation recur after prolonged sitting. There is no bleeding or significant pain.", "A small external hemorrhoidal tag is present without thrombosis or skin breakdown.", "The findings support uncomplicated local hemorrhoidal irritation.", "Gentle hygiene and measures to reduce local irritation were reviewed."),
    ("tooth_sensitivity", "Cold tooth sensitivity", "A brief sharp sensation occurs in a tooth when drinking something cold. There is no spontaneous pain or pain with chewing.", "No facial swelling or visible gum inflammation is present. Mild recession is noted near the sensitive area.", "The symptoms may relate to exposed dentin, although a dental assessment would clarify the cause.", "Gentle brushing and routine dental follow-up were discussed."),
    ("ear_wax_fullness", "Ear fullness from wax buildup", "An ear feels plugged with mildly muffled hearing. There is no pain, drainage, or dizziness.", "Cerumen partly obstructs the ear canal without visible canal inflammation.", "Wax accumulation may be contributing to the sensation of fullness.", "Routine wax management options and reassessment of hearing afterward were discussed."),
    ("motion_related_nausea", "Motion related nausea", "Queasiness develops during travel in a moving vehicle and resolves after the motion stops. There are no symptoms during ordinary daily activities.", "Balance and eye movements are normal during the visit.", "The history is consistent with motion sickness.", "Behavioral strategies and options for future travel were reviewed."),
    ("palmar_sweating", "Excessive palmar sweating", "The palms become noticeably sweaty during routine tasks and periods of anticipation. Sweating does not occur during sleep.", "Both palms are moist without rash or skin damage.", "The pattern suggests focal palmar hyperhidrosis.", "Skin care and available symptom control approaches were discussed."),
    ("vulvovaginal_dryness", "Persistent vulvovaginal dryness", "Dryness causes intermittent discomfort with daily activity and intimacy. There is no unusual discharge or bleeding.", "The vulvar tissues appear mildly dry without focal lesions or discharge.", "The symptoms may reflect reduced tissue moisture and require follow-up if persistent.", "Gentle care and vaginal moisturizer options were reviewed."),
    ("recurrent_minor_nosebleeds", "Recurrent minor nosebleeds", "Small nosebleeds occur intermittently after nose blowing in dry conditions. There is no easy bruising or bleeding elsewhere.", "The anterior nasal lining is dry with a small superficial crust. No active bleeding is present.", "The history and examination suggest irritation of the anterior nasal lining.", "Humidification and gentle nasal care were discussed, with routine reassessment if episodes continue."),
]

WORDS = [
    ("Balance", "Balance is easier to judge after a pause.", "The discussion returned to balance between activity and rest."),
    ("Pace", "Pace matters when a task is repeated.", "The patient chose a pace that allowed a comfortable conversation."),
    ("Rest", "Rest can be scheduled before marked fatigue.", "The patient described rest as a brief pause rather than the end of the activity."),
    ("Hope", "Hope for improvement was discussed without promising a particular result.", "The patient expressed hope that a smaller change would be easier to maintain."),
    ("Grace", "Grace was the word used for allowing an imperfect first attempt.", "The patient wanted room for grace when a planned task did not go smoothly."),
    ("Will", "Will a shorter practice period fit the routine?", "The patient will keep the next attempt brief."),
    ("Still", "Still seated, the patient reviewed the next step.", "The patient was still considering which change felt practical."),
    ("Clear", "Clear instructions were requested for the practice at home.", "The clinician kept the written sequence clear and brief."),
    ("Bright", "Bright overhead light was reduced for comfort.", "The patient preferred a bright but glare-free place to read the instructions."),
    ("Quiet", "Quiet surroundings made the discussion easier to follow.", "The patient chose a quiet setting for the next practice."),
    ("Chance", "Chance interruptions often shorten a planned break.", "The patient wanted a chance to rehearse before leaving."),
    ("Comfort", "Comfort was checked before the next demonstration.", "The position was adjusted for comfort during the discussion."),
    ("Trust", "Trust in the plan improved after the explanation.", "The patient said that trust depended on being able to ask questions."),
    ("Kind", "Kind language helped the patient describe a frustrating task.", "The patient preferred a kind reminder to a demanding instruction."),
    ("True", "True improvement would mean an easier daily routine for this patient.", "The patient asked whether the same advice remained true on a busy day."),
    ("Ready", "Ready access to the instruction sheet was arranged.", "The patient felt ready to try a modest change."),
    ("Ease", "Ease of use mattered more than adding another complicated step.", "The patient wanted to ease back into the routine gradually."),
    ("Calm", "Calm discussion helped organize the questions.", "The patient preferred a calm explanation with time to respond."),
    ("Resolve", "Resolve to start small was reinforced during the discussion.", "The patient wanted to resolve the uncertainty before trying the next step."),
    ("Work", "Work on the next step stopped briefly for a question.", "The patient wanted the plan to work on an ordinary busy day."),
    ("Rise", "Rise slowly was one of the patient's existing reminders.", "The patient noticed a rise in frustration when too many changes were proposed."),
    ("Steady", "Steady progress was preferred to a demanding first attempt.", "The patient chose a steady pace for reviewing the instructions."),
    ("Daily", "Daily routines provided a useful starting point for discussion.", "The patient wanted the change to fit into daily life."),
    ("Able", "Able to explain the sequence, the patient moved on to questions.", "The patient was able to summarize the plan without prompting."),
    ("Light", "Light pressure on the clipboard made writing easier.", "The patient found the instruction sheet light enough to carry with supplies."),
    ("Warm", "Warm hands were more comfortable during the discussion.", "The patient chose a warm layer before the seated review."),
    ("Well", "Well before the discussion ended, the patient asked to rehearse the sequence.", "The patient explained the sequence well after a brief pause."),
    ("Open", "Open questions encouraged a fuller account of the routine.", "The patient kept the booklet open at the relevant page."),
    ("Forward", "Forward planning included time for questions.", "The patient wanted to move forward with one practical change."),
    ("Present", "Present concerns were discussed before future goals.", "The patient preferred to focus on the present routine."),
    ("Content", "Content on the instruction sheet was reviewed together.", "The patient checked the content of the written plan against the discussion."),
    ("Support", "Support for the next step was discussed in practical terms.", "The patient asked what support would be available if the routine became difficult."),
    ("Listen", "Listen first was the patient's request for the conversation.", "The clinician paused to listen before suggesting a change."),
    ("Choice", "Choice of a starting task was left to the patient.", "The patient preferred a choice between two manageable approaches."),
    ("Repeat", "Repeat practice was offered after the explanation.", "The patient asked to repeat the sequence before leaving."),
    ("Pause", "Pause points were added to the written sequence.", "The patient used a pause to check the instructions."),
    ("Soft", "Soft chair padding was used for comfort.", "The patient preferred a soft cushion during the seated discussion."),
    ("Strong", "Strong preferences about pace were acknowledged.", "The patient had a strong preference for making one change at a time."),
    ("Gentle", "Gentle reminders were easier for the patient to accept.", "The patient preferred a gentle approach to restarting a difficult routine."),
    ("Sure", "Sure footing was confirmed before the patient left the chair.", "The patient wanted to be sure of the next step before leaving."),
]

FIRST = ["Irena", "Tomas", "Liora", "Marek", "Anika", "Dario", "Celina", "Nolan", "Elina", "Ravi", "Amara", "Felix", "Leona", "Ismael", "Nadia", "Soren", "Alina", "Mateo", "Mina", "Jonas", "Esme", "Tariq", "Lucia", "Oren", "Samira", "Pavel", "Emilia", "Kian", "Maren", "Idris", "Noemi", "Ronan", "Tessa", "Hugo", "Safiya", "Nico", "Yara", "Bram", "Ayla", "Dorian"]
SURNAMES = ["Merovel", "Raliven", "Taverin", "Selmora", "Norevan", "Calveri", "Valdren", "Belorin", "Demeril", "Farovan", "Lomeris", "Palvane", "Renovar", "Verlani", "Miravel", "Taloren", "Savelin", "Corvani", "Delvaren", "Rosmeri", "Nevoren", "Morlave", "Perovan", "Silvari", "Lavenor", "Toravel", "Melrani", "Dovaren", "Arvelli", "Kervano"]
CLINICIAN_SURNAMES = ["Joralven", "Veleran", "Navorel", "Rilavon", "Deromel", "Fanelor", "Gaverin", "Havoren", "Iverlan", "Kalvori", "Lavorel", "Nameril", "Orelvan", "Paroven", "Queralin", "Rovelan", "Sanvori", "Teravel", "Umerlan", "Valerom", "Weralin", "Xavorel", "Yemoril", "Zerovan"]

# Longer notes use distinct purposeful sections, not repeated length markers.
EXTENSIONS = [
    "The patient first walked through a usual day so the plan could be placed in context. Tasks tend to run together, and a break is often postponed until everything is finished. We selected a natural transition between activities as a reminder to check comfort. The patient preferred this to adding a separate alarm. The sequence was written in the order it would actually be used, with the most difficult step discussed before leaving.",
    "The first explanation included several steps at once. The patient asked to slow down and repeated the part that was already clear. We then separated the next step from the optional details. After a quiet interval, the patient described the whole sequence in their own words. A small uncertainty remained about what to do if the routine was interrupted, so that situation was rehearsed as well. The final version was acceptable to the patient.",
    "The patient described a previous attempt that was abandoned when ordinary responsibilities became busy. The difficulty was fitting the sequence into the day rather than disagreement with its purpose. We considered the simplest version that could still be useful and identified a place to keep the instructions. The patient did not want extra equipment. The plan therefore starts with items already available and leaves more demanding changes for review after the first attempt.",
    "The discussion distinguished a tolerable familiar symptom from an unexpected change. The patient was able to describe their usual pattern and the circumstances in which it settles. We revisited the previously supplied advice about seeking reassessment and checked that it was available in the written material. The patient preferred to ask the treating team when uncertain rather than improvise a change. No new symptom was disclosed during this part of the conversation.",
    "A practice attempt was interrupted by a question about the purpose of one step. The clinician stopped the sequence, answered the question, and asked the patient to choose whether to resume or begin again. The patient chose to restart at a slower pace. This made the order easier to remember. We kept the restart option in the plan so an interruption would not be treated as a reason to abandon the activity for the rest of the day.",
    "The patient wanted the written plan to reflect what had actually been discussed. We read through it together, removed an optional step that had not been chosen, and kept room for a short observation after practice. The patient preferred plain sentences to a dense checklist. The final sheet included the existing route for questions and the agreed review arrangement. The patient could locate both without having to read the whole page from the beginning.",
    "We considered what would happen on a day with less available time. The patient selected one essential part of the routine and explained how to keep it brief. A longer version could be used when circumstances allowed, but it was not required for a useful attempt. The clinician checked that the shortened version remained consistent with the existing plan. The patient appeared relieved that success did not depend on completing every optional activity at once.",
    "The patient identified one situation in which reminders from other people feel unhelpful. The conversation focused on what kind of support would be welcome and how the patient could ask for it. A brief prompt followed by time to respond was preferred. The clinician acknowledged that the patient could revise this preference. We practiced a simple request for more time, and the patient said that wording would be easier to use during an ordinary interaction.",
    "Before reviewing the next step, the patient summarized what had improved and what remained difficult. The clinician reflected those priorities back and checked that the order was right. The most troublesome task was discussed first. A less important concern was recorded for a later visit so the conversation did not become overloaded. The patient agreed with this order and could explain how the selected change related to the difficulty they wanted to address.",
    "The patient asked how to tell whether a modest change was worthwhile. We chose an observation tied to a familiar task rather than a complicated score. The patient will note whether the task feels easier, unchanged, or harder under similar circumstances. This is intended to support the next discussion, not to determine a medication change. The patient understood that a new or unexpected symptom should be considered separately from a routine practice result.",
    "The patient had brought several questions and wanted to check that none were lost. Each question was considered in turn, with the patient deciding which answer needed to be written down. One issue depended on the prescribing team's advice and was left for that team. The remaining questions concerned how to organize the existing routine. The patient reviewed the resulting notes and confirmed that they captured the practical points that would matter at home.",
    "At the end of the practice discussion, the clinician asked the patient to imagine returning to the usual setting. The patient described where the instructions would be placed, when there would be time to review them, and what might interrupt the first attempt. We adjusted the order to fit that account. The patient then gave a brief explanation without looking at the sheet. The explanation retained the main steps and the option to pause when uncertain.",
]

CONTEXT_HEADINGS = ["Daily routine", "Understanding the plan", "Practical barriers", "Symptom review", "Practice discussion", "Written instructions", "A shorter day", "Support preferences", "Priorities", "Review observations", "Questions raised", "Preparing to leave"]
LAYOUTS = ["narrative", "soap", "dialogue", "portal_exchange", "handover", "checklist", "consult_letter", "interview"]

CONTROL_OPENINGS = {
    "vestibular_rehab": "The patient is mainly concerned about feeling secure when turning in a confined space.",
    "voice_fatigue": "The patient wants speaking to remain comfortable during an ordinary conversation.",
    "hand_strain": "The patient wants to complete household tasks without having to maintain a firm grip for long periods.",
    "tension_headache": "The patient wants a way to interrupt desk work before the familiar tightness builds.",
    "sleep_routine": "The patient hopes to make sleep timing more predictable with a manageable change to the evening routine.",
    "dermatitis": "The patient wants necessary hand washing to be less aggravating to the skin.",
    "swallow_followup": "The patient wants to follow the previously agreed meal routine when other activities make the day busy.",
    "tinnitus": "The patient would like quiet activities to feel less dominated by the background sound.",
    "knee_rehab": "The patient would like stairs to be more manageable without pushing through an increasing ache.",
    "inhaler_technique": "The patient wants to check the device sequence and be sure the existing instructions are understood.",
    "dry_eye": "The patient wants to continue reading with fewer interruptions from eye discomfort.",
    "ankle_rehab": "The patient wants the first part of a walk to feel easier after sitting.",
    "reflux_review": "The patient wants to understand how the evening meal routine relates to the familiar burning sensation.",
    "fatigue_pacing": "The patient wants enough energy left for the later tasks in an ordinary day.",
    "constipation": "The patient wants to review which parts of the existing bowel routine are practical to maintain.",
    "jaw_tension": "The patient wants to notice jaw clenching before the familiar tightness becomes distracting.",
    "low_back_rehab": "The patient wants sitting to be more comfortable without abandoning the usual daily activities.",
    "footwear_review": "The patient wants to check shoe fit without relying on foot discomfort as a warning.",
    "wound_check": "The patient wants to clarify the dressing supplies needed for the established wound-care routine.",
    "home_pressure": "The patient wants to know whether the home measurement sequence is being carried out consistently.",
    "shoulder_stiffness": "Shoulder stiffness is making reaching and dressing less comfortable.",
    "lateral_hip_discomfort": "Outer hip aching is interfering with comfortable walking and side lying.",
    "plantar_heel_pain": "Heel discomfort is most bothersome when getting up after resting.",
    "seasonal_nasal_symptoms": "Recurrent sneezing and nasal drainage have become bothersome with seasonal changes.",
    "facial_acne": "Recurring facial bumps remain a concern despite regular cleansing.",
    "urinary_urgency": "Sudden urinary urges are interrupting everyday activities.",
    "vasomotor_symptoms": "Intermittent episodes of warmth and sweating are prompting a discussion of symptom relief.",
    "restless_legs": "Uncomfortable leg sensations make it difficult to sit quietly in the evening.",
    "action_tremor": "Hand shaking during eating and drinking is the main concern.",
    "dependent_leg_swelling": "Lower leg swelling after prolonged standing remains bothersome.",
    "thickened_toenails": "Increasing nail thickness is making routine trimming difficult.",
    "diffuse_hair_shedding": "Increased hair shedding during washing and brushing has raised concern.",
    "meal_related_bloating": "Fullness after larger meals is the main reason for the visit.",
    "hemorrhoidal_irritation": "Recurring anal irritation after prolonged sitting remains uncomfortable.",
    "tooth_sensitivity": "Brief tooth sensitivity with cold drinks is prompting a review.",
    "ear_wax_fullness": "A plugged sensation and muffled hearing are the main concerns.",
    "motion_related_nausea": "Queasiness during vehicle travel is prompting a discussion of prevention.",
    "palmar_sweating": "Sweaty palms during routine tasks have become bothersome.",
    "vulvovaginal_dryness": "Dryness is affecting comfort during daily activity and intimacy.",
    "recurrent_minor_nosebleeds": "Intermittent small nosebleeds in dry conditions are prompting a review of nasal care.",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(full):
    return sha(unicodedata.normalize("NFC", full).casefold().encode("utf-8"))


def name(value):
    return "[[NAME|" + value + "]]"


def extract(marked):
    out, spans, at, offset = [], [], 0, 0
    for match in re.finditer(r"\[\[NAME\|([^\[\]]+)\]\]", marked):
        literal, value = marked[at:match.start()], match[1]
        out += [literal, value]
        offset += len(literal)
        spans.append({"start": offset, "end": offset + len(value), "label": "NAME", "text": value})
        offset += len(value)
        at = match.end()
    out.append(marked[at:])
    text = "".join(out)
    assert "[[" not in text and "]]" not in text
    return text, spans


def make_person(counter, clinician=False):
    surnames = CLINICIAN_SURNAMES if clinician else SURNAMES
    while True:
        first = FIRST[counter % len(FIRST)]
        sur = surnames[(counter // len(FIRST)) % len(surnames)]
        # Distinct invented combinations, with realistic punctuation variations.
        if counter // (len(FIRST) * len(surnames)):
            sur += "-" + surnames[(counter // 17 + 5) % len(surnames)]
        suffix = "" if clinician or counter % 11 else " Jr."
        full = first + " " + sur + suffix
        counter += 1
        group = identity(full)
        if int(group[:8], 16) % 10 < 8:
            return {"full": full, "first": first, "surname": sur, "suffix": suffix,
                    "initials": first[0] + ". " + sur[0] + ".", "group": group}, counter


def alias_for(person, word, variant, serial):
    if variant == 0:
        return word if serial % 2 else word.lower(), "ordinary_word_alias"
    if variant == 1:
        return (word[:1].lower() + word[1:2].upper() + word[2:].lower()
                if serial % 3 == 0 else word.lower()), "mixed_case_alias" if serial % 3 == 0 else "lowercase_alias"
    return (person["initials"], "personal_initials") if serial % 2 else (person["first"], "ordinary_given_name")


def identity_intro(person, alias, positive, cue, family):
    full, a = name(person["full"]), name(alias)
    positives = [
        f"During introductions, {full} asked to be addressed as '{a}'. The patient repeated the preferred form of address before the discussion began.",
        f"'{a} is what I go by,' {full} said when the clinician asked how to address the patient. The preference was repeated back and accepted.",
        f"The registration entry uses {full}; the patient uses '{a}' in conversation. Both refer to the person attending this visit.",
        f"The patient introduced themself as {a}, then supplied {full} as the full name for the record. The conversation continued using the requested form of address.",
        f"Preferred form of address: {a}. The patient confirmed that the full name in the record, {full}, was correct.",
        f"The clinician asked whether to use the full name, {full}, during the conversation. The patient requested '{a}' and confirmed the pronunciation.",
        f"Before discussing symptoms, {full} wrote '{a}' as the name used in messages. The patient requested the same form of address in replies.",
        f"The person speaking as {a} in this interview is the patient, {full}. The patient chose that form of address at the start of the interview.",
        f"The patient responded to {a} and explained that this was the preferred name. The registration name, {full}, was confirmed during the same conversation.",
        f"When introductions resumed, the patient confirmed the full name {full} and asked that '{a}' be used in the discussion. The request was acknowledged.",
    ]
    result = positives[cue] if positive else CONTROL_OPENINGS[family]
    if positive and cue in (0, 2, 4, 5, 9):
        result += " A first name or chosen form of address was preferred to a formal title."
    return result


def local_mention(person, alias, positive, mode, action):
    a = name(alias)
    if not positive:
        options = [f"The patient said: '{action}'",
                   f"Reply from the patient: '{action}'",
                   f"Asked about the next step, the patient responded: '{action}'",
                   f"The patient paused and said, '{action}'",
                   f"Patient message: '{action}'",
                   f"The patient added: '{action}'"]
    else:
        options = [f"The patient who goes by {a} said: '{action}'",
                   f"Reply from {a}, the patient's chosen display name: '{action}'",
                   f"{a}'s response was recorded after confirming this was the patient's preferred name: '{action}'",
                   f"The patient, addressed as {a}, said, '{action}'",
                   f"Patient message from {a}: '{action}' The patient had chosen this name for the exchange.",
                   f"At this point {a}, the name the patient uses in conversation, added: '{action}'"]
    return options[mode % len(options)]


def attribution(clinician, positive, mode):
    full = name(clinician["full"])
    initial_name = name(clinician["first"][0] + ". " + clinician["surname"])
    surname = name(clinician["surname"])
    if positive:
        options = [f"Reviewed with Dr. {full}, MD. Dr. {surname} asked the patient to summarize the next step.",
                   f"The attending physician, {full}, DO, reviewed the discussion. The signature below uses the physician's first initial: {initial_name}, DO.",
                   f"Consulting clinician: Dr. {full}. The reply from Dr. {surname} confirmed the agreed follow-up discussion.",
                   f"{full}, MD, completed the review. The patient addressed the physician as Dr. {surname} when asking the final question.",
                   f"The plan was reviewed by Dr. {full}, whose abbreviated signature is {initial_name}, MD.",
                   f"Reply approved by {full}, DO. The physician answered a final question before closing the exchange."]
    else:
        options = ["The clinician asked the patient to summarize the next step and clarified the remaining question.",
                   "The questions raised during the discussion were reviewed before the plan was completed.",
                   "The follow-up discussion will include what was manageable and what remained difficult.",
                   "The patient explained the agreed next step in their own words before leaving.",
                   "The plan was reviewed with the patient, with time allowed for a final question.",
                   "The reply addressed the patient's remaining question about the agreed plan."]
    return options[mode % len(options)]


def layout_note(layout, title, intro, history, exam, assessment, plan, lexical, spoken, ending, extensions):
    core = [history, exam, assessment, plan]
    if layout == "narrative":
        blocks = [title + ". " + intro, history + " " + exam, lexical, assessment + " " + plan, spoken]
    elif layout == "soap":
        blocks = [title + "\n" + intro, "S: " + history, "O: " + exam, lexical, "A: " + assessment, "P: " + plan, spoken]
    elif layout == "dialogue":
        blocks = [title + " — discussion record\n" + intro, "Patient account: " + history, "Clinician observation: " + exam,
                  "Discussion: " + assessment + " " + lexical, spoken, "Agreed next step: " + plan]
    elif layout == "portal_exchange":
        blocks = ["Message thread: " + title, intro, "Opening message\n" + history,
                  "Review of the visit\n" + exam + " " + assessment, lexical, spoken, "Reply\n" + plan]
    elif layout == "handover":
        blocks = ["Handover — " + title, intro, "Concern / " + history, "Findings / " + exam,
                  "Interpretation / " + assessment, "Patient priorities / " + lexical, spoken, "Next steps / " + plan]
    elif layout == "checklist":
        blocks = [title + "\n" + intro, "• Account reviewed: " + history, "• Observation: " + exam,
                  "• Discussion: " + assessment + " " + lexical, "• Patient response: " + spoken, "• Agreed plan: " + plan]
    elif layout == "consult_letter":
        blocks = ["Re: " + title, "Thank you for continuing the review. " + intro, history, exam + " " + assessment, lexical,
                  "We discussed the following next step. " + plan, spoken]
    else:
        blocks = [title + "\nInterview opening: " + intro, "Account of the difficulty\n" + history,
                  "Observed during review\n" + exam, "What this means for the routine\n" + assessment + " " + lexical,
                  spoken, "What happens next\n" + plan]
    # Extensions are substantive, independently varied sections, with local name evidence.
    if extensions:
        blocks[3:3] = extensions[:3]
        blocks.extend(extensions[3:])
    blocks.append(ending)
    return "\n\n".join(blocks)


def load_allowed_module(relative, module_name):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_window_jobs():
    path = ROOT / "src/clinical_deid/hf_encoder.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "window_jobs")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["window_jobs"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="")
    args = parser.parse_args()
    if args.version and not re.fullmatch(r"v[0-9]+", args.version):
        raise ValueError("version must be v followed by digits")
    tail = "__" + args.version if args.version else ""
    output = ROOT / ("data/tk10_train_name_contexts" + tail + ".jsonl")
    manifest_path = Path(str(output) + ".manifest.json")
    report_dir = ROOT / "reports/takeover/mix3_001/author10_name_contexts" / (args.version or "v1")
    if output.exists() or manifest_path.exists() or report_dir.exists():
        raise FileExistsError("refusing to overwrite an authored version")
    if len(FAMILIES) != 40:
        raise ValueError("exactly 40 independently composed families required")
    tokenizer = Tokenizer.from_file(str(MODEL / "hf/tokenizer.json"))
    inner = tokenizer.encode("clinical note", add_special_tokens=False).ids
    outer = tokenizer.encode("clinical note", add_special_tokens=True).ids
    assert len(outer) - len(inner) == 2
    content = json.loads((MODEL / "config.json").read_text())["max_length"] - 2
    assert content == 510
    window_jobs = load_window_jobs()
    conventions = load_allowed_module("scripts/takeover/normalize_conventions.py", "author10_conventions")
    date_intervals = load_allowed_module("scripts/takeover/normalize_date_intervals.py", "author10_intervals")
    rows, detailed, people, lengths = [], [], [], []
    counter, clinician_counter = 0, 137
    normalization_changes = Counter()
    for fi, (slug, title, history, exam, assessment, plan) in enumerate(FAMILIES):
        for gi in range(10):
            serial = fi * 10 + gi
            person, counter = make_person(counter)
            clinician, clinician_counter = make_person(clinician_counter, clinician=True)
            word, capital_context, lower_context = WORDS[(fi * 7 + gi * 11) % len(WORDS)]
            is_long = gi in (1, 4, 7)
            layout = LAYOUTS[(fi + gi) % len(LAYOUTS)]
            pair_id = f"a10_name_pair_{fi:02d}_{gi:02d}"
            age = 45 + (serial * 3) % 18 if slug == "vasomotor_symptoms" else 22 + (serial * 13) % 64
            duration = 2 + (serial * 3) % 19
            identity_groups = [person["group"], clinician["group"]]
            people.append({"pair_id": pair_id, "patient": person, "clinician": clinician})
            for variant in range(4):
                positive = variant != 3
                alias, phenomenon = alias_for(person, word, min(variant, 2), serial)
                cue = (fi * 3 + gi) % 10
                intro = identity_intro(person, alias, positive, cue, slug)
                lexical = capital_context + " " + lower_context
                # All numeric material here is ordinary age, durations or counts.
                situation = f"The patient is {age} years old and has been trying the current routine for {duration} days. "
                history_variant = situation + history
                action = ["I can try the first step and pause if I need to.", "I would like to check the written sequence once more.",
                          "Please let me explain it back before we finish.", "I can see where this will fit into the day.",
                          "I want to start with the part we practiced together."][serial % 5]
                spoken = local_mention(person, alias, positive, serial + variant, action)
                ending = attribution(clinician, positive, fi + gi + variant)
                extensions = []
                if is_long:
                    indices = [(serial + k * 5) % len(EXTENSIONS) for k in range(7)]
                    for ei, index in enumerate(indices):
                        paragraph = CONTEXT_HEADINGS[index] + "\n" + EXTENSIONS[index]
                        if ei in (2, 5):
                            paragraph += " " + local_mention(person, alias, positive, serial + ei + variant,
                                "I would like to keep that part of the plan simple." if ei == 2 else "That answers the question I wanted to raise.")
                        extensions.append(paragraph)
                marked = layout_note(layout, title, intro, history_variant, exam, assessment, plan, lexical, spoken, ending, extensions)
                # A10-T01: drop only an outside sentence period after a name's
                # own final initial/suffix period; keep the NAME literal intact.
                marked = re.sub(r"(\[\[NAME\|[^\[\]]+\.\]\])\.", r"\1", marked)
                text, spans = extract(marked)
                phenomena = [phenomenon if positive else "matched_phi_free_control", "ordinary_word_context", "ordinary_pronouns",
                             "honorific_credential_boundary", "message_speaker_attribution", "local_identity_evidence", "ordinary_age_and_duration"]
                if cue in (1, 3, 4, 7, 8):
                    phenomena.append("alias_before_canonical_name")
                if positive and person["suffix"]:
                    phenomena.append("name_suffix_boundary")
                if positive and ((serial + variant) % 6 == 2 or is_long):
                    phenomena.append("possessive_or_quote_boundary")
                if is_long:
                    phenomena.append("multi_window_local_context")
                row = {"id": f"a10_name_contexts_{fi:02d}_{gi:02d}_{variant}{tail}", "text": text, "spans": spans,
                       "split": "train", "role": "train", "annotation_policy": POLICY,
                       "identity_group": person["group"], "identity_groups": identity_groups,
                       "pair_id": pair_id, "template_family": "a10_name_" + slug,
                       "phenomena": phenomena, "cohort": "long_document" if is_long else ("unusual_name" if positive and variant < 2 else "realistic"),
                       "layout": layout, "negative_only": not spans, "synthetic": True,
                       "authored_stratum": "long" if is_long else "short", "lexical_class": word.lower(),
                       "author_variant": variant, "identity_context": "local_repeated_alias_evidence"}
                # Entire-row normalizer comparisons; neither result replaces the gold.
                counts = Counter()
                normalized = conventions.normalize(deepcopy(row), counts)
                assert normalized == row, (row["id"], "convention normalization would change row", counts)
                interval_row, interval_changes, _ = date_intervals.normalize_row(deepcopy(row))
                assert interval_row == row and not interval_changes
                normalization_changes.update(counts)
                encoded = tokenizer.encode(text, add_special_tokens=False)
                offsets = encoded.offsets
                jobs = list(window_jobs(len(encoded.ids), content))
                assert (len(jobs) > 1) == is_long, (row["id"], len(encoded.ids))
                owners = []
                for span in spans:
                    tokens = [t for t, (a, b) in enumerate(offsets) if a < span["end"] and b > span["start"]]
                    assert tokens
                    owned_jobs = sorted({wi for wi, (_, _, l, r) in enumerate(jobs) for t in tokens if l <= t < r})
                    owners.append({"literal": span["text"], "span": [span["start"], span["end"]], "token_range": [tokens[0], tokens[-1] + 1], "owner_windows": owned_jobs})
                rows.append(row)
                lengths.append(len(encoded.ids))
                detailed.append({"id": row["id"], "tokens": len(encoded.ids), "windows": jobs, "span_ownership": owners})
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids) == 1600
    assert len({r["text"] for r in rows}) == len(rows)
    assert sum(r["negative_only"] for r in rows) == 400
    assert sum(r["authored_stratum"] == "long" for r in rows) == 480
    for row in rows:
        assert row["annotation_policy"] == POLICY and row["split"] == "train"
        assert row["identity_group"] in row["identity_groups"]
        assert len(row["identity_groups"]) == len(set(row["identity_groups"]))
        assert all(int(g[:8], 16) % 10 < 8 for g in row["identity_groups"])
        last = -1
        for s in row["spans"]:
            assert 0 <= s["start"] < s["end"] <= len(row["text"])
            assert s["start"] >= last and row["text"][s["start"]:s["end"]] == s["text"]
            assert s["text"] == s["text"].strip()
            assert not s["text"].startswith("Dr.") and not s["text"].endswith((", MD", ", DO", "'s"))
            last = s["end"]
        assert bool(row["spans"]) != row["negative_only"]
    support = Counter(s["label"] for r in rows for s in r["spans"])
    distribution = Counter((r["layout"], r["authored_stratum"], "control" if r["negative_only"] else "positive") for r in rows)
    for layout in LAYOUTS:
        for stratum in ("short", "long"):
            assert distribution[layout, stratum, "positive"] == 3 * distribution[layout, stratum, "control"] > 0
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
    references = [{"path": p, "sha256": sha((ROOT / p).read_bytes())} for p in REFERENCE_PATHS]
    manifest = {"schema_version": 1, "annotation_policy": POLICY, "split": "train", "role": "train", "dataset_role": "train",
                "status": "training_candidate_authored", "path": str(output.relative_to(ROOT)), "sha256": sha(payload), "rows": len(rows), "documents": len(rows),
                "label_support": {label: support[label] for label in LABELS}, "phi_free_rows": 400,
                "source_provenance": {"method": "new deterministic synthetic composition for training", "author": "author10_name_contexts",
                     "generator": str(Path(__file__).relative_to(ROOT)), "generator_sha256": sha(Path(__file__).read_bytes()),
                     "allowed_references": references, "clinical_context_families": 40,
                     "context_composition": "20 primary-author families and 20 separately composed fictional clinical families from a writing-only child agent; no corpus or model exposure",
                     "external_records_used": False, "model_weights_or_inference_used": False, "network_used": False},
                "synthetic_schemes": {"people": "Invented combinations of authored given names and invented surnames; no real person's details sourced. Coincidental real-name overlap is possible.",
                    "aliases": "Independently authored English-word preferred names, lowercase and deliberately chosen mixed-case spellings, ordinary given names, and personal initials.",
                    "other_identifiers": "No dates, facilities, locations, addresses, device or contact identifiers authored; numbers are ages 22–85 and ordinary durations.",
                    "partition": "Every full identity is SHA256 of NFC/casefold UTF-8; each digest first8 hexadecimal modulo10 is below8. Controls inherit paired identity metadata."},
                "tokenization": {"tokenizer": str((MODEL / "hf/tokenizer.json").relative_to(ROOT)), "runtime": "AST-isolated hf_encoder.window_jobs",
                    "model_max_length": 512, "content_window": 510, "special_tokens": 2, "multi_window_rows": 480,
                    "minimum_tokens": min(lengths), "median_tokens": statistics.median(lengths), "maximum_tokens": max(lengths)},
                "normalization": {"whole_row_comparison": True, "unreviewed_gold_mutations": False, "changes": dict(normalization_changes)},
                "limitations": ["Synthetic training only; not evidence of clinical performance.", "Templates and authored clinical process paragraphs are correlated within this bank.",
                    "English-word names and mixed-case aliases are intentionally enriched and not population-frequency estimates.",
                    "No inference, cross-bank overlap audit, or independent admission audit was performed by this author.",
                    "Repeated alias mentions carry local identity cues; pre-canonical aliases have nearby full identity evidence.",
                    "Only NAME is supported; other label families are intentionally outside this focused bank."]}
    report = {"rows": len(rows), "phi_positive": 1200, "phi_free": 400, "unique_texts": len({r["text"] for r in rows}),
              "families": len(FAMILIES), "identity_groups": len({g for r in rows for g in r["identity_groups"]}),
              "multi_window_rows": 480, "label_support": dict(support), "normalization_change_counts": dict(normalization_changes),
              "layout_strata": [{"layout": l, "stratum": s, "status": status, "rows": n} for (l, s, status), n in sorted(distribution.items())],
              "phenomena": dict(Counter(p for r in rows for p in r["phenomena"])),
              "cohorts": dict(Counter(r["cohort"] for r in rows)), "lexical_inventory": [w[0] for w in WORDS],
              "token_lengths": {"min": min(lengths), "median": statistics.median(lengths), "max": max(lengths)},
              "checks": {"offsets": "pass", "text_duplicates": "none", "identity_partition": "pass", "normalization_whole_rows": "unchanged", "matched_strata": "3:1 in every layout/length combination"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=False)
    with output.open("xb") as f:
        f.write(payload)
    with manifest_path.open("x") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    for filename, value in [("self_qa.json", report), ("identity_ledger.json", people), ("window_ownership.json", detailed)]:
        with (report_dir / filename).open("x") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
    for p in (output, manifest_path):
        p.chmod(0o444)
    print(json.dumps({"corpus": str(output), "sha256": sha(payload), "manifest": str(manifest_path),
                      "manifest_sha256": sha(manifest_path.read_bytes()), "report_dir": str(report_dir), **report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
