"""Independent, synthetic v12 DEVELOPMENT authoring; no corpus/model inputs.

This module intentionally contains all of its own prose. It reads only its
predeclaration, its source, the annotation contract, and a pinned local tokenizer
cache. It does not import another author's generator or discover data files.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import argparse
import base64
import hashlib
import json
from pathlib import Path
import random
import re

from .annotation_policy import POLICY_VERSION, annotation_errors, canonical_identity_group, identity_split
from .corpus_quality import audit_corpora
from .types import ENTITY_TYPES


TOTAL_DOCUMENTS = 1280
CACHE_FILE = "fb374d419588a4632f3f557e76b4b70aebbca790"
CACHE_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
SEED = 120021


def local_tokenizer(cache_dir: Path):
    """Construct o200k_base directly from verified local bytes; never download."""
    import tiktoken
    payload = (cache_dir / CACHE_FILE).read_bytes()
    if hashlib.sha256(payload).hexdigest() != CACHE_SHA256:
        raise ValueError("Local o200k_base cache hash mismatch")
    ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in payload.splitlines())}
    pattern = "|".join([
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}", r" ?[^\s\p{L}\p{N}]+[\r\n/]*", r"\s*[\r\n]+", r"\s+(?!\S)", r"\s+",
    ])
    return tiktoken.Encoding(name="o200k_base", pat_str=pattern, mergeable_ranks=ranks,
                             special_tokens={"<|endoftext|>": 199999, "<|endofprompt|>": 200018})


def mark(label: str, text: str) -> str:
    if label not in ENTITY_TYPES or not text or "[[" in text or "]]" in text:
        raise ValueError("Invalid authored value")
    return f"[[{label}|{text}]]"


def render(marked: str) -> tuple[str, list[dict]]:
    pieces, spans, offset, previous = [], [], 0, 0
    for match in re.finditer(r"\[\[([A-Z]+)\|([^\[\]]+)\]\]", marked):
        literal = marked[previous:match.start()]
        pieces.extend((literal, match[2]))
        offset += len(literal)
        spans.append({"start": offset, "end": offset + len(match[2]), "label": match[1]})
        offset += len(match[2])
        previous = match.end()
    pieces.append(marked[previous:])
    text = "".join(pieces)
    if "[[" in text or "]]" in text or annotation_errors(text, spans):
        raise ValueError("Malformed independently authored annotation")
    return text, spans


GIVEN = (
    "Adela", "Amir", "Béatrice", "Celia", "Dario", "Elif", "Farah", "Gita", "Hugo", "Inez",
    "Jalen", "Kavya", "Leonie", "Mateo", "Nadia", "Omar", "Priya", "Rafael", "Sora", "Talia",
    "Uma", "Vera", "Wesley", "Ximena", "Yara", "Zain", "Álvaro", "Élodie", "Íñigo", "Özlem",
    "Łucja", "Ji-yeon", "Minh", "Anika", "Nour", "Søren", "Zoë", "Renée", "Caoimhe", "Amani",
    "Désirée", "Noémie", "Seán", "Saoirse", "Rūta", "Māra", "Tenzin", "Haruto", "Eun-ji", "Ilona",
    "Mireya", "Nikhil", "Anouk", "Yusuf", "Zofia", "Léa", "Yūna", "Ana María", "Luz Elena", "Abdul Karim",
)
SURNAMES = (
    "Bellmere", "Caldovey", "Darsen", "Eldwick", "Fenlow", "Garriven", "Haldren", "Irven", "Jaswick", "Kelmere",
    "Larkin-Selvey", "Marden", "Novey", "Orswick", "Peldon", "Quinmere", "Ravelin", "Selwick", "Tarnell", "Uldren",
    "Varnes", "Wendall", "Xandrel", "Yarven", "Zeldin", "de la Vela", "van Aarden", "al-Nerim", "O’Rovain", "D’Avel",
    "Sáveira", "Bělanský", "Dębicka", "Güneşli", "Nørdahl", "Márquez-Lune", "Ben Sarel", "da Silveira", "dos Cevos", "Del Arven",
    "Ríos Vandel", "Koválen", "Müllerin", "Åsved", "Östrel", "Ișovei", "Səlimli", "Lévron", "Noël-Vern", "N’Dare",
)
OTHER_SCRIPT_IDENTITIES = (
    "Марина Велорина", "Антон Лесарев", "Лидия Орвелова", "Νεφέλη Καρένου", "Άρης Βελόπουλος",
    "سلمى نوران", "ليان مرادي", "ريم سلوان", "תמר אלוני", "נועה לבני", "दीपा नवलकर", "किरण वेलकर",
    "美咲 山澄", "遥 月森", "はるか 星野", "서윤 한별", "지우 달빛", "明月 洛川", "嘉宁 清禾",
)


def identity_pool(minimum: int = 1200) -> list[str]:
    result = []
    for name in OTHER_SCRIPT_IDENTITIES:
        if identity_split(canonical_identity_group(name)) == "validation":
            result.append(name)
    for middle in ("", " A.", " L.", " R.", " S."):
        for index, given in enumerate(GIVEN):
            for surname in SURNAMES[index % 7:] + SURNAMES[:index % 7]:
                name = f"{given}{middle} {surname}"
                if identity_split(canonical_identity_group(name)) == "validation":
                    result.append(name)
                if len(result) >= minimum:
                    return result
    raise ValueError("Insufficient independently authored validation identities")


@dataclass(frozen=True)
class Case:
    n: int
    name: str
    clinician: str
    relative: str

    @property
    def identities(self):
        return [canonical_identity_group(value) for value in (self.name, self.clinician, self.relative)]

    def value(self, label: str) -> str:
        n = self.n
        year = 2018 + n % 7
        month = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")[n % 12]
        locality = ("Larkhaven, UT", "North Aster Cove, OR", "Morrowfield, ME", "West Pellmere, CO", "South Vervain, WA", "Wrenford, VT", "East Dovewick, NM", "Brindle Reach, MN")[n % 8]
        values = {
            "NAME": self.name,
            "DATE": (f"{month} {n % 27 + 1}, {year}", f"{year}-{n % 12 + 1:02}-{n % 27 + 1:02}", f"{month} {year}", str(year), f"{n % 12 + 1}/{n % 27 + 1}/{year}")[n % 5],
            "AGE": ("90", "91", "ninety-two", "93", "94", "ninety-five", "96", "101")[n % 8],
            "LOCATION": locality,
            "FACILITY": ("Juniper Lantern Rehabilitation Institute", "Copper Wren Community Hospital", "Aster Sail Medical Pavilion", "Morrow Fen Care Center", "Willow Arc Outpatient Clinic", "Cobalt Finch Medical Center")[n % 6],
            "ADDRESS": f"{120 + n % 870} {('Wrenlight Lane', 'Mallow Bend', 'Kestrel Lantern Road', 'Aster Vale Court')[n % 4]}, {locality} {80000 + n % 9000:05}",
            "PHONE": (f"(202) 555-{100 + n % 100:04}", f"+1 303 555 {100 + n % 100:04}", f"646.555.{100 + n % 100:04}", f"206-555-{100 + n % 100:04}")[n % 4],
            "EMAIL": f"care.{n:04}@{('morrow.example', 'wren.test', 'example.org', 'example.net')[n % 4]}",
            "MRN": (f"VD-{n:07}", f"D{n:05}-CL", f"78{n:07}")[n % 3],
            "SSN": f"000-{10 + n % 90:02}-{1000 + n % 9000:04}",
            "ACCOUNT": (f"VDAC-{n:06}", f"V12/{n:06}/R", f"AC{n:08}")[n % 3],
            "ID": (f"VDS-{n:06}-Q", f"DEV-ID-{n:05}", f"QX{n:07}")[n % 3],
            "URL": f"https://{('portal.morrow.example', 'visits.wren.test', 'example.com', 'example.org')[n % 4]}/visits/vd-{n:04}",
            "IP": f"{('192.0.2', '198.51.100', '203.0.113')[n % 3]}.{1 + n % 253}",
        }
        return values[label]

    def m(self, label):
        return mark(label, self.value(label))


# Fifteen specialty scenarios were written specifically for this bank. These
# fragments are not shared with or obtained from the training author.
SPECIALTIES = (
    ("pulmonary", "Breathlessness follow-up", "a dry cough when climbing the last flight of stairs", "Breath sounds are even without resting wheeze", "The spacer seal improves after a slower demonstration", "Record exertional symptoms before the next inhaler review"),
    ("cardiology", "Volume assessment", "ankle swelling that is greater after a prolonged chair day", "The ankles have mild symmetric edema and the extremities are warm", "Standing slowly prevents the brief lightheaded sensation", "Compare morning weights and symptoms before changing diuretic timing"),
    ("neurology", "Headache review", "pressure across the forehead after interrupted sleep", "Speech is fluent and the upper limbs move symmetrically", "The headache diary separates nausea from visual discomfort", "Continue the symptom diary and review rescue medication frequency"),
    ("rheumatology", "Hand stiffness visit", "stiff fingers when opening the first jar of the morning", "Finger flexion is slow but complete, without new erythema", "Warm water makes the stretching sequence easier", "Pace hand exercises and reassess swelling after a short interval"),
    ("endocrine", "Glucose pattern discussion", "a lower glucose reading after an unexpectedly long walk", "Injection sites are intact without firm nodules", "Teach-back clarifies the difference between basal and mealtime dosing", "Bring the glucose log and usual meal schedule to the dose review"),
    ("nephrology", "Fluid and medication check", "a metallic taste that appears after the evening tablets", "Mucous membranes are moist and breathing is unlabored", "A written schedule separates the binder from unrelated medicines", "Recheck the laboratory trend before changing the treatment schedule"),
    ("gastroenterology", "Bowel symptom review", "loose stools after a recent change in meal timing", "The abdomen is soft with mild diffuse tenderness and no guarding", "Smaller meals reduce the sensation of fullness", "Track stool frequency, hydration and meal tolerance"),
    ("dermatology", "Skin response review", "itching under the edge of an elastic garment", "There is a dry patch without warmth, drainage or streaking", "A fragrance-free emollient is easier to apply than the prior ointment", "Photograph the evolving border for the next clinical review"),
    ("orthopedics", "Shoulder function review", "discomfort when reaching toward a high kitchen shelf", "Active elevation is limited by discomfort while the hand remains warm", "The home exercise is tolerable when the elbow stays supported", "Use the graded exercise plan and reassess functional reach"),
    ("ophthalmology", "Ocular surface visit", "grittiness after reading several pages under a fan", "The lids close fully and there is no visible discharge", "A pause during sustained reading reduces the irritation", "Review drop technique and continue the surface comfort log"),
    ("audiology", "Listening effort review", "difficulty following a soft voice in a room with clattering dishes", "The ear canals are clear on the limited inspection", "Facing the speaker helps more than increasing background volume", "Compare listening in a quiet room with the usual dining setting"),
    ("speech_swallow", "Mealtime observation", "a need to clear the throat after several quick sips", "The voice stays clear during paced single sips", "A smaller cup encourages a pause between swallows", "Keep upright for meals and use the agreed pacing cues"),
    ("behavioral", "Sleep and coping review", "worry at bedtime after a demanding caregiving day", "Thought is organized and responses remain relevant", "A brief written list reduces repeated checking of unfinished tasks", "Practice the evening routine and review daytime alertness"),
    ("rehabilitation", "Mobility goal review", "fatigue after moving laundry between two nearby rooms", "Transfers are controlled when the chair arms are used", "Two short activity periods are more manageable than one long period", "Practice the chosen household task with planned seated breaks"),
    ("wound_care", "Dressing tolerance review", "a pulling sensation during removal of the outer dressing", "The visible wound bed is moist without purulent drainage", "Moistening the edge reduces discomfort during removal", "Follow the dressing sequence and report spreading warmth or new drainage"),
)
OBSERVATIONS = (
    "The symptom settles after a seated pause; it does not interrupt a quiet conversation.",
    "The patient can describe the sequence clearly but is less certain about the total frequency.",
    "There has been no new difficulty taking the usual meals, and appetite is preserved.",
    "The first attempt is hesitant. A second attempt with a verbal cue is more comfortable.",
    "The change is most noticeable during the busiest part of the day and improves with pacing.",
    "The patient distinguishes the familiar symptom from the new sensation without prompting.",
    "A small notebook is easier to use than recalling events at the end of the week.",
    "The symptom has not prevented basic self-care, although tasks take longer to finish.",
    "Questions focus on how to judge a useful response rather than on changing every treatment at once.",
    "The patient remains engaged while the options are discussed and asks for a shorter written plan.",
    "The caregiver usually observes the activity but leaves room for the patient to make the first attempt.",
    "The practical concern is confidence during the task; examination alone does not reproduce the problem.",
)
PLANS = (
    "We rehearsed the first step together and asked for a repeat demonstration before adding a second step.",
    "They will write down what preceded the symptom and what made it settle, using a single line each time.",
    "She prefers a plan attached to the medication box so that the reminder appears during an existing routine.",
    "He will use the same chair and equipment during practice so that changes in effort are easier to compare.",
    "I asked the patient to explain when to stop the activity and when to seek an earlier clinical review.",
    "We agreed on a small functional goal and a check of tolerance before increasing the amount of practice.",
    "They requested large print; the key actions fit on one sheet and were read back in the room.",
    "She will bring the device and the handwritten observations so that the next review can reproduce the steps.",
)
SETTINGS = ("outpatient review", "bedside consultation", "telephone follow-up", "community visit", "therapy review", "handover discussion")


def clinical_body(specialty: tuple, index: int, *, positive: bool) -> str:
    key, title, symptom, examination, response, plan = specialty
    ordinary_age = 43 + (index * 7) % 47
    duration = 2 + index % 13
    year = 2018 + index % 7
    age_line = f"The patient is {ordinary_age} years old." if not positive or index % 3 == 2 else "The patient recalls needing fewer rest breaks at age 68."
    paragraphs = [
        f"{title} — {SETTINGS[index % len(SETTINGS)]}\n{age_line} The concern is {symptom}, present for {duration} days.",
        f"{examination}. {OBSERVATIONS[index % len(OBSERVATIONS)]}",
        f"{response}. {PLANS[(index // 3) % len(PLANS)]} {plan}.",
    ]
    if index % 2 == 0:
        paragraphs.append(f"A published {year} study came up during the discussion. We discussed how a group average can differ from the response to a practical change at home.")
    if index % 7 == 0:
        paragraphs.append("Medication teaching used the usual organizer. The patient completed the repeat demonstration without a second prompt.")
    return "\n\n".join(paragraphs)


def patient_context(case: Case, index: int) -> tuple[str, list[str]]:
    date, name = case.m("DATE"), case.m("NAME")
    patterns = (
        f"{name} was reviewed on {date}. The referral came from {case.m('FACILITY')} in {case.m('LOCATION')}; the medical ward sent the prior plan.",
        f"At {case.m('FACILITY')}, the follow-up for {name} concerns recovery since {date}. The patient lives in {case.m('LOCATION')} and uses the rehabilitation gym for practice.",
        f"Visit record | {name} | event date {date}\nResidence area: {case.m('LOCATION')}\nReferring service: {case.m('FACILITY')}\nCare area: general medical ward",
        f"From the step-down unit, Dr. {mark('NAME', case.clinician)}, MD, reviewed {name}'s course. The patient arrived from {case.m('LOCATION')} via {case.m('FACILITY')} on {date}.",
        f"The history was reviewed with {name} and the patient's sibling {mark('NAME', case.relative)}. Home is in {case.m('LOCATION')}; the encounter at {case.m('FACILITY')} followed the procedure in {date}.",
        f"{name}\nThe ward handover references {date}. {case.m('FACILITY')} arranged therapy after the return to {case.m('LOCATION')}. Occupational therapy will use the same checklist as the care team.",
    )
    text = patterns[index % len(patterns)]
    phenomena = ["patient_date", "public_patient_year_overlap", "location_facility_care_unit_contrast", "complete_location_boundary"]
    if index % 3 != 2:
        age = case.m("AGE")
        age_forms = [f"Age: {age} years old.", f"{age}-year-old patient.", f"The chart lists {age} y/o.", f"The patient is {age} years old."]
        if case.value("AGE").isdigit():
            age_forms += [f"Intake: {age}F, alert and conversant.", f"Handover: {age}yo, comfortable at rest."]
        text += " " + age_forms[index % len(age_forms)]
        phenomena.append("age_over_89_boundary")
    if index % 10 == 0:
        # Lowercase by explicit communication preference, not random annotation.
        text += f" The patient writes the name as {mark('NAME', case.name.lower())} in portal messages."
        phenomena.append("lowercase_person_name")
    if index % 11 == 0:
        initials = ". ".join(part[0] for part in case.name.split()[:2]) + "."
        text += f" {mark('NAME', initials)} is the patient's abbreviated signature on the exercise sheet."
        phenomena.append("initials_person_name")
    if index % 13 == 0:
        alias = ("rose", "dawn", "will", "may", "pat", "hope")[index % 6]
        ordinary = {"rose": "Pain rose during the last repetition.", "dawn": "The room is brighter at dawn.", "will": "We will keep the routine brief.", "may": "The patient may use a seated pause.", "pat": "Gently pat the skin dry.", "hope": "The patient describes hope for a less tiring routine."}[alias]
        text = f"{mark('NAME', alias)} asked for a short plan; {case.m('NAME')} uses that nickname with the care team. {ordinary}\n" + text
        phenomena.append("ordinary_word_alias_contrast")
    if index % 17 == 0:
        text += " The referral also asked whether Wilson disease should be considered; the question is deferred to the treating specialist."
        phenomena.append("eponym_nonpersonal")
    return text, phenomena


def administrative_context(case: Case, index: int) -> str:
    # Three layouts carry all identifier families in a coordination workflow.
    # This is intentionally overrepresented to measure each rare label.
    m = case.m
    if index % 3 == 0:
        return (f"Coordination record\nMailing address: {m('ADDRESS')}\nTelephone: {m('PHONE')}\nEmail: {m('EMAIL')}\n"
                f"MRN: {m('MRN')}\nBilling account: {m('ACCOUNT')}\nSSN supplied for the benefits form: {m('SSN')}\n"
                f"Equipment recipient ID: {m('ID')}\nSecure visit link: {m('URL')}\nHome monitor source IP: {m('IP')}")
    if index % 3 == 1:
        return (f"The delivery desk read back {m('ADDRESS')} and will call {m('PHONE')} before arriving. "
                f"A copy of the plan goes to {m('EMAIL')}. The equipment request is attached to record {m('MRN')}, "
                f"account {m('ACCOUNT')} and recipient credential {m('ID')}. The benefits worksheet records the SSN {m('SSN')}. "
                f"Remote support used {m('URL')} to check the monitor transmission from {m('IP')}.")
    return (f"Handover field | Confirmed value\nMedical record | {m('MRN')}\nCorrespondence | {m('ADDRESS')}\n"
            f"Call-back | {m('PHONE')}\nElectronic contact | {m('EMAIL')}\nEquipment ID | {m('ID')}\n"
            f"Account | {m('ACCOUNT')}\nSocial security number | {m('SSN')}\nPortal session | {m('URL')}\n"
            f"Transmission address | {m('IP')}\nAction | Match the equipment instructions to the patient's preferred format")


def selected_contacts(case: Case, index: int) -> str:
    if index % 3 == 0:
        return administrative_context(case, index // 3)
    if index % 3 == 1:
        return f"Scheduling confirmed {case.m('PHONE')} for a call-back and {case.m('EMAIL')} for the exercise sheet. The chart number is {case.m('MRN')}."
    return f"The equipment team will deliver to {case.m('ADDRESS')}. The account {case.m('ACCOUNT')} is linked to the patient card {case.m('ID')}."


def row_from_marked(identifier: str, marked: str, case: Case, *, pair_id: str, family: str,
                    phenomena: list[str], cohort="realistic", layout="narrative", subcohort=None) -> dict:
    text, spans = render(marked)
    row = {"id": identifier, "text": text, "spans": spans, "split": "validation",
           "annotation_policy": POLICY_VERSION, "identity_group": case.identities[0],
           "identity_groups": case.identities, "pair_id": pair_id, "template_family": family,
           "phenomena": list(dict.fromkeys(phenomena)), "cohort": cohort, "layout": layout,
           "synthetic": True, "authorship": "v12_independent_development_author"}
    if subcohort:
        row["subcohort"] = subcohort
    return row


UNUSUAL = ("He", "Care", "I", "May", "Will", "Hope", "Rose", "Dawn", "Summer", "Joy", "Bill", "Pat", "Light", "Still", "Young", "Well")
ORDINARY = {
    "He": "He listened, then repeated the instructions in his own words.",
    "Care": "Care is easier when the supplies are within reach.",
    "I": "I reviewed the exercise with the patient before the second attempt.",
    "May": "May is the month discussed in the seasonal allergy handout.",
    "Will": "Will the strap loosen with daily use? The patient asked to see the buckle again.",
    "Hope": "Hope for a steadier routine is the main theme of the discussion.",
    "Rose": "The respiratory rate rose briefly after walking and settled with rest.",
    "Dawn": "Dawn light makes the room bright enough to read without the bedside lamp.",
    "Summer": "Summer heat makes afternoon activity less comfortable.",
    "Joy": "Joy in shared meals remains an important reason to practice the routine.",
    "Bill": "Bill payment is managed with a checklist kept beside other household papers.",
    "Pat": "Pat the skin dry after rinsing; avoid vigorous rubbing around the dressing.",
    "Light": "Light pressure is enough to keep the pad in place.",
    "Still": "Still, the patient prefers to try a small adjustment before adding a new device.",
    "Young": "Young children visiting the home can make rest periods less predictable.",
    "Well": "Well supported sitting is more comfortable than leaning on the edge of the chair.",
}


def unusual_case(word: str, index: int, base: Case) -> Case:
    for surname in SURNAMES:
        canonical = f"{word} {surname}"
        if identity_split(canonical_identity_group(canonical)) == "validation":
            # The full identity is stated as well as the intentionally unusual alias.
            return Case(3000 + index, canonical, base.clinician, base.relative)
    raise ValueError("No validation identity for unusual-name example")


# Long narratives are multi-domain rehabilitation reviews. Every paragraph below
# is original prose. Within a natural long document a paragraph is used once;
# the separate seam slice explicitly repeats progress observations.
LONG_PASSAGES = (
    ("Reason for extended review", "The team brought the mobility, symptom and home routine discussions into a single review because the patient had been receiving several short lists of instructions. Each list was understandable on its own, but the order of tasks was becoming difficult to manage. The patient asked that the most tiring activities be spaced apart. We began with an ordinary day at home, then worked through the places where the routine slowed down or became uncomfortable."),
    ("Patient priorities", "The patient wants to prepare a simple breakfast, move between the main rooms and sit comfortably through a shared meal. These goals matter more to the patient than completing a long exercise list. We asked which task would make the largest difference if it became easier. The patient chose carrying a small meal to the table, explaining that this activity currently requires several stops and leaves little energy for conversation afterward."),
    ("History of the current pattern", "The difficulty developed gradually rather than after one clearly recalled event. At first the patient shortened only the most demanding household tasks. Later, pauses were added to activities that had previously felt routine. The patient can still complete the basic sequence but tends to hurry when another person is waiting. We discussed how this hurried start may hide the amount of effort required until the task is nearly finished."),
    ("Description of fatigue", "Fatigue is described as a loss of reserve rather than sleepiness. The patient remains alert and interested in the activity but finds it harder to maintain the same pace. Sitting quietly helps, especially when the next task can wait. The patient does not describe a sudden collapse or an abrupt loss of awareness. We used the patient's own description when deciding how to track effort, so that the diary does not confuse fatigue with pain."),
    ("Breathing during activity", "Breathing is comfortable during a quiet conversation. During the practice walk the patient pauses before the turn, allowing the breath to settle before starting again. The pause is more effective when taken early. The patient had previously waited until reaching the destination and then needed a longer recovery. We discussed using a familiar landmark as a reminder to slow down, without requiring a device alarm that might be difficult to hear."),
    ("Comfort at rest", "At rest the patient adjusts position several times before finding support for the lower back and both feet. A cushion helps when it does not slide forward. We compared the chair height with the height used during transfer practice. The lower chair feels comfortable initially but makes standing more demanding. The team will ask the patient to judge both parts of the task, including what happens when it is time to rise."),
    ("Transfers", "The patient positions both feet before standing and uses the chair arms when they are available. The first attempt is slow but controlled. When the chair is slightly farther from the table, the patient can bring the trunk forward without reaching around an obstacle. We practiced returning to the chair with the same deliberate sequence. The patient recognized that the final turn is easier when the feet stop moving before the hand reaches back."),
    ("Walking route", "The short route includes a doorway, a gentle turn and a return to the starting chair. The patient watches the floor near the doorway and briefly reduces pace. There is enough room for the usual aid, although a loose mat would make the turn less predictable. We discussed a clear route using the existing furniture. The patient prefers small changes that can be maintained consistently rather than moving every item in the room at once."),
    ("Balance confidence", "Confidence varies with the setting. On a clear, familiar surface the patient can focus on the destination. When carrying an object or responding to a question, the steps become shorter and the hand reaches for nearby support. We separated these demands during practice. The patient first walked with empty hands, then repeated the route while discussing the next task. This showed where a pause for conversation would be useful without abandoning the activity altogether."),
    ("Upper limb reach", "Reaching to a high shelf is uncomfortable near the end of the movement. The patient compensates by lifting the shoulder and leaning through the trunk. Moving the frequently used objects to a lower shelf reduces the need for this pattern. We practiced reaching with the elbow supported first, then with the arm moving freely within a comfortable range. The patient asked for a drawing of the starting position because a verbal description alone was difficult to remember."),
    ("Hand function", "The patient can grasp the usual cup but finds a tight lid more demanding, particularly before the hands have warmed. We compared a broad handle with a narrow one and discussed leaving the most frequently used container loosely closed when appropriate. The patient wants to continue handling the meal items independently. The chosen practice therefore uses the same objects found in the home routine, with rest between attempts rather than repeated forceful gripping."),
    ("Dressing sequence", "The patient dresses while seated for the lower-body steps and stands only after the clothing is gathered within reach. The most tiring part is recovering an item that has fallen to the floor. We arranged the trial items on a stable surface at the side of the chair. The patient completed the sequence with less leaning and said the arrangement felt practical. The plan includes setting out the items before beginning rather than searching midway through."),
    ("Personal care", "The sink routine is manageable when supplies are already opened and the towel is close to hand. The patient tends to remain standing while deciding what to do next, which adds effort without advancing the task. We practiced sitting for the preparation step and standing for the shorter active portion. The patient found this easier to follow than a general instruction to rest more, because the pause had a clear place in the sequence."),
    ("Meal preparation", "Preparing a small meal involves more carrying than the patient first described. A plate, cup and utensil are moved in separate trips, followed by a return for a napkin or condiment. We reviewed which items could be placed on the table before the main task begins. The patient preferred a light container with a secure lid for one part of the route. Carrying any object must remain compatible with safe use of the usual walking aid."),
    ("Eating and drinking", "The patient takes smaller sips when reminded to pause between them. Conversation sometimes speeds the sequence, especially near the end of a meal. During observation the voice remains clear and the patient can describe the difference between a comfortable sip and a rushed one. We asked the patient to show the preferred cup and explain how its size affects pacing. The plan uses the familiar cup rather than introducing several new utensils at once."),
    ("Appetite and portions", "Appetite is better for a modest portion served while the food is still appealing than for a large plate left out through several interruptions. The patient reports feeling full before the meal is complete on busier days. We discussed dividing the preparation into smaller steps and keeping the eating period separate from other household tasks. The patient will describe tolerance and enjoyment in ordinary words, with attention to changes that persist across more than one meal."),
    ("Bowel routine", "The bowel pattern is easier to describe when the patient uses a short written entry rather than trying to remember the whole week. The patient distinguishes stool consistency from the effort required and notes that these do not always change together. We reviewed the existing advice without adding a new restriction during this visit. The next review will use the record to judge the pattern and its effect on comfort, appetite and willingness to leave the home."),
    ("Hydration routine", "The patient often notices thirst only after finishing an activity. A drink placed near the resting chair is more likely to be used than one left in the kitchen. We discussed the current fluid plan and the need to keep any prescribed limits in view. The patient can explain which container is usually used but had not considered how often it is refilled. A simple tally is acceptable if it remains easy enough to continue."),
    ("Medication organization", "The organizer is filled in a quiet part of the room with the original packages nearby. The patient checks the compartment before taking a dose but sometimes starts another task before putting the organizer away. We discussed keeping the sequence together from opening the compartment to returning the box to its usual place. The patient will bring the actual organizer to the next review so that the team can examine the routine as it is performed."),
    ("Understanding medicines", "The patient recognizes several tablets by their purpose but is less confident when packages look similar. We asked the patient to explain what each medicine is expected to help, then used the written list to clarify the uncertain items. A change in appearance should prompt a check of the package and the supplied list. The patient prefers the purpose written beside the medicine rather than a separate paragraph that requires searching between two pages."),
    ("Symptom recording", "A useful symptom entry includes the activity, the sensation and what happened after a pause. The patient had previously recorded only whether the day felt good or bad, which made it difficult to connect a change with a particular task. We rehearsed one brief entry together. The patient chose a notebook with large spacing and will keep it by the resting chair. The record should support the conversation without becoming another demanding daily chore."),
    ("Sleep preparation", "The evening routine sometimes extends because the patient checks whether supplies are ready for the morning. Moving that check earlier allows the final part of the evening to be quieter. The patient does not want a complicated sleep schedule and prefers a small sequence that can be repeated. We discussed how a late rest period may feel different from nighttime sleep and asked the patient to describe both when reviewing daytime alertness."),
    ("Nighttime mobility", "The route from the resting area to the bathroom needs enough light to identify the doorway and the walking aid. The patient can reach the switch from the chair but sometimes leaves the aid farther away after the evening meal. We reviewed the position of the aid as part of preparing the room. A consistent place is easier to remember than a separate instruction delivered only when the patient is already tired."),
    ("Pain description", "The patient separates a dull ache during sustained posture from a sharper discomfort near the end of a reaching movement. These sensations have different practical consequences. The first may improve with a position change, while the second leads the patient to shorten the movement. We kept both descriptions in the review instead of collapsing them into one score. The patient will note whether the activity can resume comfortably after a brief adjustment."),
    ("Skin comfort", "The patient notices rubbing under the edge of a garment after sitting for a prolonged period. The area is easier to inspect with good light and a mirror placed securely on a stable surface. We discussed gentle washing, drying and the usual skin-care routine. The patient can describe what would represent a change from the familiar appearance. The plan emphasizes making the inspection practical enough to occur during an existing personal-care task."),
    ("Footwear", "The preferred shoes are easy to put on while seated, although one fastening requires a stronger pinch than the other. We observed how the patient adjusts the heel before standing. A shoe that slips during the first steps may change confidence even if it initially feels soft and comfortable. The patient will bring the usual pair to practice instead of relying on a description. The team can then consider comfort, fit and ease of fastening together."),
    ("Vision during tasks", "Small print is difficult when the page lies flat under uneven light. Raising the page and turning the lamp reduces glare without increasing the volume of information. The patient prefers a short line of instructions with adequate space around it. We checked whether the illustrated movement could be recognized from the usual sitting position. The patient will use the same viewing arrangement when completing the home practice so that the written prompt remains useful."),
    ("Listening conditions", "The patient follows a single speaker more easily when background conversation is reduced. During instruction we faced the patient, completed one point and then asked for a demonstration. Repeating an entire paragraph more loudly was less helpful than shortening the instruction and allowing time for a response. The patient asked that future telephone reviews begin by checking whether the room is quiet enough. This preference was added to the communication plan."),
    ("Attention and recall", "Recall is strongest for actions that are connected to an existing habit. The patient can describe the morning sequence in detail but has trouble placing an unfamiliar exercise within it. We linked practice to the period after a routine seated task and kept the materials nearby. The patient then explained the plan without reading. At the next contact the team will ask how this arrangement worked, including whether the reminder was visible when it was needed."),
    ("Emotional response", "The patient expresses frustration about needing more preparation for ordinary activities. This frustration is balanced by a clear wish to remain involved in decisions. We made time for the patient to describe what feels like useful help and what feels intrusive. The patient prefers another person to wait through the first attempt before stepping in. The team will use that preference when discussing support, while allowing the patient to request earlier assistance when fatigue is greater."),
    ("Shared household tasks", "Some household tasks can be divided without removing the patient's preferred role. The patient would like to arrange items and choose the sequence even when another person carries the heavier object. We discussed how an agreed division can reduce repeated negotiation during the activity. The patient identified one task that could be tried in this way before changing the rest of the routine. The result will be judged by comfort, confidence and whether the task still feels meaningful."),
    ("Caregiver observations", "The caregiver reports that the patient often appears less tired at the beginning of a visit than later during household activity. We asked for concrete observations such as longer pauses or a change in walking pace rather than broad judgments about effort. The patient agreed that these observations would be useful if discussed respectfully. A shared record can contain both accounts when they differ, so the next review can explore the setting rather than choosing one description prematurely."),
    ("Equipment fit", "The usual aid was checked with the patient wearing the shoes normally used indoors. Small changes in posture were more obvious during the turn than during a straight path. The patient could feel when the support was placed too far forward. We practiced stopping, repositioning and starting again. The patient will avoid adjusting several settings between reviews, because a single comfortable setup is easier to evaluate during a repeated household task."),
    ("Equipment maintenance", "The patient knows where the spare supplies are stored but had not checked whether the package could be opened comfortably. We practiced opening one package while seated with both forearms supported. The discarded wrapper was placed where it would not fall into the walking route. Maintenance will be attached to an existing weekly household check. The patient prefers this predictable reminder to a new list posted in a different room."),
    ("Managing appointments", "The patient uses a paper calendar for appointments and a separate notebook for questions. Bringing the two together before the visit helps avoid searching for a concern while the conversation is underway. We suggested writing the most important question at the top of the page. The patient can add details below it as they arise. This approach preserves the patient's choice of priorities while keeping the opening part of the visit focused."),
    ("Travel preparation", "Leaving the home requires gathering the aid, a light bag and any instructions that need discussion. The patient becomes tired when these are collected from different rooms after getting dressed. We discussed placing the items together earlier, while keeping the walking route clear. The patient would like enough time to sit before departure. The practical plan therefore begins before the final dressing step instead of adding another task at the door."),
    ("Telephone review", "During a telephone review the patient may be describing an activity that the clinician cannot see. We practiced naming the starting position, the object used and the point where the symptom appears. The patient found this structure more useful than being asked whether everything was better. Future calls can use the same sequence and allow a pause while the patient looks at the notebook. The patient prefers a short conversation with clear next steps."),
    ("Remote readings", "The home device is most useful when the reading is connected to how the patient feels and what has just happened. The patient sometimes repeats a measurement immediately when the first result is unexpected. We discussed checking position and following the device instructions before interpreting a change. The team will review the pattern with the patient rather than relying on one isolated value. Any difficulty operating the device should be recorded alongside the reading."),
    ("Review of outside advice", "The patient brought a public information sheet and asked how its recommendations fit with the current plan. We distinguished general suggestions from the specific sequence already agreed during the visit. The patient does not need to add every exercise on the sheet at once. We selected the portions that explain the purpose of pacing and left room to ask about the others later. The patient can keep the sheet with the shorter individualized instructions."),
    ("Teach-back", "At the end of the practical portion the patient explained the plan as if describing it to someone who had not attended. This revealed one unclear step: whether the seated pause should come before or after preparing the supplies. We clarified the sequence and asked for another demonstration. The second explanation was shorter and more precise. The patient said that rehearsing the order was more useful than receiving an additional page of written advice."),
    ("Recognizing change", "The patient can identify familiar symptoms that settle with the usual pause and also describe a change that would merit earlier contact. We used the patient's usual language rather than asking for technical terms. The discussion included whether a task remains possible, how long recovery feels and whether a new symptom accompanies the familiar one. The patient repeated the contact plan and asked that it be placed near the device instructions for easy reference."),
    ("Balancing practice and rest", "Practice should leave enough reserve for the activities the patient values later in the day. We reviewed a recent day when completing every exercise first made meal preparation more difficult. The patient prefers to split practice around a meal and a usual rest period. This arrangement will be tried consistently before the amount of practice changes. The patient will record the response to the whole day, including whether the final activities remain manageable."),
    ("Follow-through", "The written plan contains a small number of actions in the order they are likely to occur. The patient selected the place where it will be kept and checked that the print is readable there. The caregiver can prompt with the same words if help is wanted. The next review will begin by asking which parts were actually used, which were inconvenient and which felt unnecessary. The plan can then be adjusted around the observed routine."),
    ("Community participation", "The patient would like to resume a short social visit without spending the rest of the day recovering. We considered the whole sequence, including preparation, travel, sitting in a different chair and returning home. A shorter first visit would allow the patient to judge how the unfamiliar setting affects effort. The patient prefers to choose the activity and decide in advance where a comfortable pause could occur, rather than leaving only when symptoms become difficult."),
    ("Changes between good and difficult days", "The patient recognizes that the same task may require a different pace on different days. The concern is deciding how much to change without losing the routine altogether. We discussed keeping the beginning of the sequence familiar while shortening the active portion when reserve is lower. The patient can use the same written plan with an extra pause. This is easier to follow than maintaining two completely separate sets of instructions."),
    ("Review of the room arrangement", "The patient described where the chair, lamp and frequently used supplies are kept. A small side table provides a useful resting place but can narrow the approach to the chair if pulled too far forward. We rehearsed the route with the table slightly farther to the side. The patient could still reach the notebook and drink while seated. The preferred arrangement balances access to supplies with enough space to turn and sit comfortably."),
    ("Choosing a practical measure", "We asked how the patient would know that the plan was helping. The patient chose completing the meal-carrying task with fewer unplanned stops while still feeling able to talk afterward. This description captures both the activity and its cost. A faster time alone might encourage rushing. The patient will compare the same task in the same setting and mention any change in the objects carried, the footwear used or the help available."),
    ("Closing discussion", "The patient remained engaged through the review and requested time to consider one optional equipment change. No decision was required during the visit. We summarized the agreed actions and asked the patient to choose the first one to try. The patient chose the meal-carrying sequence because its effect should be easy to notice. The team will revisit that goal alongside symptom tolerance, confidence and the practical burden of maintaining the plan."),
)


def token_span_positions(text: str, spans: list[dict], encoding) -> list[dict]:
    tokens = encoding.encode(text, disallowed_special=())
    edges, total = [0], 0
    for token in tokens:
        total += len(encoding.decode_single_token_bytes(token))
        edges.append(total)
    byte_offsets = [0]
    for char in text:
        byte_offsets.append(byte_offsets[-1] + len(char.encode("utf-8")))
    import bisect
    result = []
    for span in spans:
        start, end = byte_offsets[span["start"]], byte_offsets[span["end"]]
        first = bisect.bisect_right(edges, start) - 1
        last = bisect.bisect_left(edges, end)
        ownership = [join for join in range(384, len(tokens), 256) if first < join < last]
        ordinary = [join for join in range(512, len(tokens), 512) if first < join < last]
        result.append({"label": span["label"], "start": span["start"], "end": span["end"],
                       "token_start": first, "token_end": last, "ownership_joins_crossed": ownership,
                       "ordinary_window_seams_crossed": ordinary})
    return result


def natural_long(case: Case, index: int, encoding) -> dict:
    heading, phenomena = patient_context(case, index)
    passages = list(LONG_PASSAGES)
    # Preserve the opening and closing clinical sequence; vary middle domains.
    middle = passages[3:-3]
    random.Random(SEED + index).shuffle(middle)
    passages = passages[:3] + middle + passages[-3:]
    marked = [heading, clinical_body(SPECIALTIES[index % 15], index, positive=True)]
    for number, (title, body) in enumerate(passages):
        paragraph = f"{title}\n{body}"
        if number == 9:
            paragraph += f" The home routine was last reviewed in {case.m('DATE')}; a published {2018 + case.n % 7} article was discussed during that review."
        if number == 21:
            paragraph += f" Dr. {mark('NAME', case.clinician)} discussed the practical sequence with the patient's sibling {mark('NAME', case.relative)}."
        if number == 35:
            paragraph += " " + administrative_context(case, index % 3)
        marked.append(paragraph)
    row = row_from_marked(f"v12-d-long-natural-{index:02}", "\n\n".join(marked), case,
                         pair_id=f"v12-d-long-natural-{index:02}", family="v12d_extended_rehabilitation_narrative",
                         phenomena=phenomena + ["natural_long_narrative", "multidomain_care_review", "ordinary_age", "public_dates"],
                         cohort="long_document", subcohort="natural_narrative")
    row["authored_paragraphs"] = len(passages)
    row["o200k_tokens"] = len(encoding.encode(row["text"], disallowed_special=()))
    row["span_token_positions"] = token_span_positions(row["text"], row["spans"], encoding)
    if row["o200k_tokens"] < 4200:
        raise ValueError(f"Natural long note below predeclared minimum: {row['o200k_tokens']}")
    return row


def seam_long(case: Case, index: int, encoding) -> dict:
    target_label = ENTITY_TYPES[index % len(ENTITY_TYPES)]
    target = case.value(target_label)
    # Deliberately repeated observations are limited to this explicit stress slice.
    prefix = f"Rehabilitation progress review\n{case.m('NAME')} reports a steadier routine over 12 days. "
    observation = "The patient pauses before standing, checks foot position, then reaches for the chair arm. Breathing settles during seated rest. "
    join = 384 + 256 * (index % 12)
    # Locate the marked value across an actual ownership join. Padding is clinical
    # chart shorthand, and its repetitive nature is disclosed in row metadata.
    plain_prefix = render(prefix)[0]
    cue = {
        "NAME": "Reviewed with ", "DATE": "Patient's procedure date: ", "ADDRESS": "Delivery address: ",
        "PHONE": "Call-back telephone: ", "EMAIL": "Contact email: ", "MRN": "Medical record: ",
        "SSN": "Benefits SSN: ", "ACCOUNT": "Patient account: ", "ID": "Patient credential: ",
        "URL": "Patient visit link: ", "IP": "Patient monitor IP: ", "AGE": "Patient age in years: ",
        "LOCATION": "Patient home locality: ", "FACILITY": "Transferring facility: ",
    }[target_label]
    while len(encoding.encode(plain_prefix + observation + cue, disallowed_special=())) < join - 18:
        prefix += observation
        plain_prefix += observation
    placed = None
    def place(value):
        for extra in ("", "Condition stable. "):
            for count in range(80):
                candidate = prefix + "Stable. " * count + extra + cue + mark(target_label, value) + ".\n"
                text, spans = render(candidate)
                positions = token_span_positions(text, spans, encoding)
                if join in positions[-1]["ownership_joins_crossed"]:
                    return candidate
        return None
    placed = place(target)
    if placed is None:
        # A one-token age cannot cross a boundary. Use its spelled-out value,
        # preserving the predeclared policy while retaining the same family.
        if target_label == "AGE":
            target = "ninety-seven"
            placed = place(target)
    if placed is None:
        raise ValueError(f"Unable to position {target_label} at ownership join {join}")
    marked = placed
    # Repeated progress is separated by original brief clinical responses; it is
    # still explicitly stress data, not a natural-narrative claim.
    count = 0
    while len(encoding.encode(render(marked)[0], disallowed_special=())) < 4300:
        marked += f"\nObservation {count + 1}: " + observation + OBSERVATIONS[(count + index) % len(OBSERVATIONS)] + "\n"
        if count % 9 == 0:
            marked += selected_contacts(case, count // 9) + "\n"
        count += 1
    row = row_from_marked(f"v12-d-long-seam-{index:02}", marked, case, pair_id=f"v12-d-long-seam-{index:02}",
                         family="v12d_explicit_repeated_progress_seam", phenomena=["repeated_progress_seam_stress", "ownership_join_crossing", "ordinary_window_seams"],
                         cohort="long_document", subcohort="repeated_progress_seam_stress")
    row["o200k_tokens"] = len(encoding.encode(row["text"], disallowed_special=()))
    row["span_token_positions"] = token_span_positions(row["text"], row["spans"], encoding)
    row["target_ownership_join"] = join
    row["target_ownership_label"] = target_label
    row["window_geometry"] = {"window_tokens": 512, "context_each_side": 128, "stride": 256, "first_join": 384}
    if not any(join in p["ownership_joins_crossed"] and p["label"] == target_label for p in row["span_token_positions"]):
        raise ValueError("Ownership crossing changed after completion")
    return row


def build_rows(encoding) -> list[dict]:
    pool = identity_pool()
    random.Random(SEED).shuffle(pool)
    rows = []

    def case_for(n):
        return Case(n, pool[n % 1000], pool[1000 + n % 100], pool[1100 + n % 100])

    for specialty_index, specialty in enumerate(SPECIALTIES):
        for within in range(60):
            n = specialty_index * 60 + within
            case = case_for(n)
            context, phenomena = patient_context(case, n)
            body = clinical_body(specialty, n, positive=True)
            contacts = selected_contacts(case, n)
            parts = ([context, body, contacts], [body, context, contacts], [context, contacts, body])[within % 3]
            pair = f"v12-d-pair-{n:04}"
            family = "v12d_" + specialty[0]
            rows.append(row_from_marked(f"v12-d-real-{n:04}", "\n\n".join(parts), case, pair_id=pair, family=family,
                        phenomena=phenomena + ["ordinary_age", "ordinary_pronoun", "duration", "specialty_prose", "reserved_identifiers"],
                        layout=("narrative", "prose_then_record", "mixed_form")[within % 3]))
            if within < (18 if specialty_index == 0 else 17):
                negative = clinical_body(specialty, n, positive=False)
                negative += "\n\n" + (
                    "The general medical ward uses a short handover sheet. Pharmacy and occupational therapy will compare the practical sequence during the next review.",
                    "Review item | Observation\nRest | Comfortable while seated\nActivity | Slower when carrying an object\nCare area | Rehabilitation gym\nPlan | Use the agreed pause before the next attempt",
                    "The patient may need time to consider the change. Pain rose briefly during practice, then settled. We will keep the first step simple and gently pat the skin dry after washing.",
                )[within % 3]
                rows.append(row_from_marked(f"v12-d-negative-{n:04}", negative, case, pair_id=pair, family=family,
                            phenomena=["paired_phi_free", "ordinary_age", "ordinary_pronoun", "duration", "public_dates", "ordinary_words", "care_units"],
                            layout=("narrative", "table", "narrative")[within % 3]))
    for index in range(32):
        word = UNUSUAL[index % len(UNUSUAL)]
        case = unusual_case(word, index, case_for(900 + index))
        ordinary = ORDINARY[word]
        body = clinical_body(SPECIALTIES[index % 15], 900 + index, positive=False)
        if index < 16:
            positive = f"Patient name: {mark('NAME', word)}. The registration card gives the full name as {case.m('NAME')}. {ordinary}\n\n{body}"
        else:
            positive = f"{mark('NAME', word.lower())} prefers a short explanation. The patient {case.m('NAME')} uses that lowercase personal alias in messages. {ordinary}\n\n{body}"
        pair = f"v12-d-unusual-pair-{index:02}"
        for kind, marked in (("positive", positive), ("negative", ordinary + "\n\n" + body)):
            rows.append(row_from_marked(f"v12-d-unusual-{kind}-{index:02}", marked, case, pair_id=pair,
                        family="v12d_unusual_explicit_identity", phenomena=["explicit_unusual_name" if kind == "positive" else "paired_phi_free", "same_spelling_different_context", "ordinary_age"],
                        cohort="unusual_name"))
    for index in range(40):
        rows.append(natural_long(case_for(932 + index), index, encoding))
    for index in range(20):
        rows.append(seam_long(case_for(972 + index), index, encoding))
    validate_rows(rows, encoding)
    return rows


def validate_rows(rows, encoding):
    if len(rows) != TOTAL_DOCUMENTS:
        raise ValueError("Predeclared document count changed")
    counts = Counter(span["label"] for row in rows for span in row["spans"])
    if any(counts[label] < 200 for label in ENTITY_TYPES):
        raise ValueError("Insufficient predeclared label support")
    if sum(not row["spans"] for row in rows) != 288:
        raise ValueError("Predeclared PHI-free support changed")
    if len({row["text"] for row in rows}) != len(rows):
        raise ValueError("Duplicate authored text")
    for row in rows:
        if annotation_errors(row["text"], row["spans"]):
            raise ValueError("Invalid annotation")
        if any(identity_split(group) != "validation" for group in row["identity_groups"]):
            raise ValueError("Wrong identity partition")
        row["o200k_tokens"] = len(encoding.encode(row["text"], disallowed_special=()))


def _exclusive_json(path: Path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def teaching_cue_review(rows) -> dict:
    """Explain a reserved-domain collision in the generic auditor cue regex."""
    pattern = re.compile(r"\b(?:teaching|textbook|educational|example)\b", re.I)
    result = {}
    for phi_bearing, key in ((True, "phi_bearing"), (False, "phi_free")):
        members = [row for row in rows if bool(row["spans"]) == phi_bearing]
        original, masked = 0, 0
        for row in members:
            original += bool(pattern.search(row["text"]))
            text = row["text"]
            for span in reversed(row["spans"]):
                if span["label"] in {"EMAIL", "URL"}:
                    text = text[:span["start"]] + "[contact]" + text[span["end"]:]
            masked += bool(pattern.search(text))
        result[key] = {"documents": len(members), "unmodified_cue_documents": original,
                       "email_url_masked_cue_documents": masked,
                       "unmodified_rate": original / len(members), "email_url_masked_rate": masked / len(members)}
    return {"disposition": "The reserved word example in EMAIL/URL spans drives the generic teaching-word cue gap. The original warning is retained; masking only those spans gives similar clinical cue rates. This diagnostic does not validate semantics.",
            "strata": result}


def write_bank(root: Path, cache_dir: Path):
    output = root / "data/v12_validation.jsonl"
    manifest_path = root / "data/v12_validation.jsonl.manifest.json"
    declaration_path = root / "data/v12_validation.author_declaration.json"
    report_path = root / "data/v12_validation.quality.json"
    targets = (output, manifest_path, declaration_path, report_path)
    if any(path.exists() or path.is_symlink() for path in targets):
        raise FileExistsError("Refusing to overwrite any existing development artifact")
    predeclaration_path = root / "data/v12_validation.predeclaration.json"
    predeclaration_bytes = predeclaration_path.read_bytes()
    predeclaration = json.loads(predeclaration_bytes)
    if predeclaration["total_documents"] != TOTAL_DOCUMENTS:
        raise ValueError("Predeclaration mismatch")
    encoding = local_tokenizer(cache_dir)
    rows = build_rows(encoding)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    generator_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    declaration = {
        "author": "v12_independent_development_author", "split": "validation", "suite": "development",
        "annotation_policy": POLICY_VERSION, "corpus_sha256": digest, "generator_sha256": generator_hash,
        "original_synthetic_authorship": True, "external_clinical_dataset_used": False,
        "existing_corpora_read": False, "sibling_training_content_read": False,
        "model_outputs_or_scores_read": False, "protected_material_read": False,
        "authorship_statement": "All clinical prose, identities, addresses, scenarios and layouts were independently invented for this development bank. No existing clinical corpus, generator, training-author text or detector output was consulted. A helper reviewed the annotation contract only and authored no clinical text.",
        "allowed_read_inputs": predeclaration["authorship_inputs"],
        "predeclaration_sha256": hashlib.sha256(predeclaration_bytes).hexdigest(),
        "long_narrative_composition": "40 natural long notes each use the same 48 original multi-domain rehabilitation topic paragraphs once, with varying internal order, specialty openings, identities and inserted coordination records. They are highly correlated authored composites, not 40 independent clinical contexts.",
        "long_stress_composition": "20 separate repeated-progress notes target every label at least once across actual ownership joins 384+256*k. Repetition and brief padding are deliberate stress construction, not a natural-note claim.",
        "synthetic_schemes": {
            "names": "Invented combinations with canonical NFC casefold SHA-256 partition, including all named clinicians and relatives.",
            "facilities_addresses_localities": "Invented authored place combinations; state abbreviations and numeric postal-like suffixes are formatting examples, not verified destinations.",
            "phones": "US area codes with 555-0100 through 555-0199 reserved fictional subscriber lines.",
            "email_url": "Only .test, .example, example.com, example.net and example.org reserved domains.",
            "ip": "Only RFC 5737 TEST-NET IPv4 blocks 192.0.2/24, 198.51.100/24 and 203.0.113/24.",
            "ssn": "Invalid unassignable 000 first group; remaining groups generated from development row index.",
            "mrn_account_id": "Invented VD/D/78, VDAC/V12/AC and VDS/DEV-ID/QX schemes derived only from development row index; no external records.",
        },
        "limitations": predeclaration["limitations_predeclared"],
    }
    with output.open("xb") as handle:
        handle.write(payload)
    _exclusive_json(declaration_path, declaration)
    manifest = {
        "split": "validation", "suite": "development", "annotation_policy": POLICY_VERSION,
        "sha256": digest, "documents": len(rows), "independent_final": False,
        "used_for_training": False, "used_for_checkpoint_selection": True,
        "generator": "src/clinical_deid/v12_development.py", "generator_sha256": generator_hash,
        "author_declaration": "data/v12_validation.author_declaration.json",
        "author_declaration_sha256": hashlib.sha256(declaration_path.read_bytes()).hexdigest(),
        "predeclaration_sha256": hashlib.sha256(predeclaration_bytes).hexdigest(),
        "original_synthetic_authorship": True, "tokenizer": "o200k_base", "tokenizer_cache_sha256": CACHE_SHA256,
        "cohorts": dict(Counter(row["cohort"] for row in rows)),
        "subcohorts": dict(Counter(row.get("subcohort", row["cohort"]) for row in rows)),
        "label_support": dict(Counter(span["label"] for row in rows for span in row["spans"])),
        "phi_free_documents": sum(not row["spans"] for row in rows),
        "long_token_range": [min(row["o200k_tokens"] for row in rows if row["cohort"] == "long_document"), max(row["o200k_tokens"] for row in rows if row["cohort"] == "long_document")],
        "limitations": predeclaration["limitations_predeclared"],
    }
    _exclusive_json(manifest_path, manifest)
    audit = audit_corpora({"validation": [output]}, strict_policy=True,
                         token_counter=lambda value: len(encoding.encode(value, disallowed_special=())), tokenizer_name="o200k_base")
    audit["author_warning_dispositions"] = {"teaching_wording": teaching_cue_review(rows)}
    _exclusive_json(report_path, audit)
    if not audit["passed"]:
        raise ValueError("Authored development corpus failed strict audit")
    return {"corpus_sha256": digest, "generator_sha256": generator_hash, "documents": len(rows),
            "labels": manifest["label_support"], "cohorts": manifest["cohorts"], "warnings": audit["warnings"],
            "long_token_range": manifest["long_token_range"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--tokenizer-cache", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_bank(args.root, args.tokenizer_cache), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
