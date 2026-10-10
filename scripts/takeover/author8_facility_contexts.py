"""Independently authored synthetic facility-boundary TRAINING corpus.

No corpus is an input. Run from this worktree with its virtual environment:
  PYTHONPATH=src python scripts/takeover/author8_facility_contexts.py --preview
  PYTHONPATH=src python scripts/takeover/author8_facility_contexts.py --write

--write publishes new files only and freezes them. --verify regenerates in memory
and checks the previously published bytes. No inference, normalization CLI,
external network, dataset discovery, or admission to a mixture occurs here.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import re
import statistics
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/tk8_train_facility_contexts.jsonl"
MANIFEST = CORPUS.with_suffix(".jsonl.manifest.json")
REPORT = ROOT / "reports/takeover/mix3_001/author8_facility_contexts/author_report.json"
CHECKPOINT = ROOT / "models/takeover-mbbase-mix3-001/step-003081"
LABELS = ("NAME", "DATE", "ADDRESS", "PHONE", "EMAIL", "MRN", "SSN", "ACCOUNT", "ID", "URL", "IP", "AGE", "LOCATION", "FACILITY")
LAYOUTS = ("narrative", "problem_list", "clinical_form", "handoff_table", "correspondence", "case_conference")

# Whole institution names are authored as values, never derived by merging at.
AT_ORGS = [
    "Hospital at Silver Trestle", "Medical Center at Foxglass Reach",
    "Rehabilitation Hospital at Lantern Cove", "Care Institute at Alder Compass",
    "Community Hospital at Cloudfern Crossing", "Medical Center at Cedar Loom",
    "Recovery Hospital at Quiet Saffron", "Hospital at Meadow Dial",
    "Health Center at Juniper Ribbon", "Regional Hospital at Brass Willow",
    "Surgical Hospital at Amber Thread", "Hospital at Morrow Lantern",
    "Rehabilitation Center at Thistle Arc", "Medical Institute at Violet Cairn",
    "Community Hospital at Birch Meridian", "Medical Center at Lark Compass",
    "Recovery Institute at Copper Orchard", "Hospital at Fern Mosaic",
    "Care Center at Hazel Tern", "Medical Institute at Dappled Quay",
    "Hospital at Sunward Flint", "Community Medical Center at Bracken Bell",
    "Rehabilitation Hospital at Velvet Spruce", "Health Institute at Otter Loom",
    "Hospital at Marigold Trestle", "Medical Center at Mossfinch Crossing",
    "Care Hospital at Plover Lantern", "Recovery Institute at Elm Lantern",
    "Community Hospital at Topaz Reed", "Medical Center at Wren Mosaic",
    "Hospital at Auburn Compass", "Rehabilitation Center at Sable Clover",
    "Hospital at Kestrel Thread", "Medical Institute at Willow Prism",
    "Recovery Center at Russet Tern", "Community Hospital at Dewberry Arc",
    "Medical Center at Hawthorn Dial", "Hospital at Petal Cairn",
    "Care Institute at Goldfinch Reach", "Rehabilitation Hospital at Mallow Quay",
]
OTHER_ORGS = [
    "Institute of Cedar Lantern Medicine", "Center for Foxglass Recovery",
    "Care in Balance Hospital", "Hospital on Silver Trestle Green",
    "Institute of Bracken Health", "Center for Juniper Mobility",
    "Care in Motion Medical Center", "Hospital on Lark Compass Green",
    "Institute of Meadow Renewal", "Center for Violet Cairn Health",
    "Partners in Recovery Hospital", "Medical Center on Fern Mosaic Green",
    "Institute of Alder Rehabilitation", "Center for Otter Loom Medicine",
    "Hope in Practice Hospital", "Hospital on Copper Orchard Green",
    "Institute of Thistle Wellness", "Center for Hazel Tern Recovery",
    "Care in Community Hospital", "Medical Center on Birch Meridian Green",
    "Institute of Willow Prism Health", "Center for Topaz Reed Medicine",
    "Trust in Recovery Hospital", "Hospital on Mossfinch Green",
    "Institute of Marigold Healing", "Center for Plover Rehabilitation",
    "Strength in Motion Hospital", "Medical Center on Sable Clover Green",
    "Institute of Dewberry Care", "Center for Auburn Compass Health",
    "Continuity in Care Hospital", "Hospital on Mallow Quay Green",
    "Institute of Russet Tern Medicine", "Center for Wren Mosaic Recovery",
    "Progress in Practice Hospital", "Medical Center on Petal Cairn Green",
    "Institute of Hawthorn Rehabilitation", "Center for Goldfinch Reach Care",
    "Partners in Wellness Hospital", "Hospital on Kestrel Thread Green",
]
PLAIN_ORGS = [
    "Larkspindle Medical Center", "Alderquill General Hospital", "Fernwhistle Health Institute",
    "Copperlilt Rehabilitation Hospital", "Marigoldglass Medical Center", "Thornharbor Hospital",
    "Brackenwheel Health Center", "Juniperskein Hospital", "Willowbadge Medical Institute",
    "Otterfield Recovery Hospital", "Birchglimmer Medical Center", "Mallowtorch Hospital",
    "Thistlelane Care Institute", "Cloudribbon Hospital", "Hawthornkite Medical Center",
    "Plovermantle Health Institute", "Cedardial Hospital", "Orchardfable Medical Center",
    "Meadowtwill Rehabilitation Hospital", "Hazeltrellis Hospital",
]
RESIDENTIAL_ORGS = [
    "Alder Lantern Residential Care Home", "Foxglass Nursing and Residential Center",
    "Silver Trestle Assisted Living Residence", "Bracken Compass Senior Care Home",
    "Mallow Orchard Residential Nursing Home", "Thistle Quay Assisted Living Center",
    "Copper Finch Residential Care Home", "Juniper Mosaic Nursing Residence",
    "Wren Lantern Assisted Living Home", "Petal Cairn Residential Care Center",
    "Willow Prism Nursing and Residential Home", "Fern Tern Senior Care Residence",
    "Hazel Bell Assisted Living Center", "Mossfinch Residential Nursing Home",
    "Goldfinch Thread Care Home", "Dewberry Compass Assisted Living Residence",
    "Auburn Reed Residential Care Center", "Kestrel Loom Nursing Home",
    "Plover Glass Assisted Living Center", "Hawthorn Ribbon Residential Nursing Home",
]
LOCALITIES = [
    "Briarfen, MT", "Larkmere, UT", "Alderwick, CO", "Foxmarsh, WY", "Mossquay, OR",
    "Thistlemere, ID", "Brackenford, NM", "Cedarwold, AZ", "Willowfen, NV", "Hazelwick, ME",
    "Plovermere, MN", "Orchardfen, WI", "Mallowford, IA", "Copperwold, NE", "Dewberry Quay, KS",
    "Birchwick, SD", "Wrenford, ND", "Fernmarsh, OK", "Topazwick, AR", "Kestrelmere, VT",
]
ZIP_BASE = dict(zip(("MT", "UT", "CO", "WY", "OR", "ID", "NM", "AZ", "NV", "ME", "MN", "WI", "IA", "NE", "KS", "SD", "ND", "OK", "AR", "VT"), (59000, 84000, 80000, 82000, 97000, 83000, 87000, 85000, 89000, 4000, 55000, 53000, 50000, 68000, 66000, 57000, 58000, 73000, 72000, 5000)))
STREETS = ["Lantern Spur", "Compass Lane", "Trestle Way", "Mosaic Road", "Quill Avenue", "Ribbon Court", "Cairn Drive", "Loom Street", "Tern Circle", "Prism Terrace"]
GIVEN = ["Alina", "Bastien", "Celia", "Dario", "Elise", "Farid", "Gita", "Hugo", "Inez", "Jalen", "Keira", "Lucian", "Marta", "Nolan", "Oriana", "Pavel", "Quiana", "Ravi", "Sabine", "Tomas", "Usha", "Vera", "Wesley", "Xenia", "Yasmin", "Zubin", "Amélie", "Béla", "Chloé", "Dávid", "Esmé", "Félix", "Güneş", "Héloïse", "Íñigo", "João", "Klara", "Léonie", "Maëlle", "Núria", "Óscar", "Päivi", "Róisín", "Søren", "Théo", "Úna", "Valérie", "Zélie"]
SURNAMES = ["Ardenby", "Bellwick", "Corven", "Darnell", "Eldmere", "Farrowell", "Glenford", "Hawkin", "Iverly", "Josswick", "Kerrigan", "Lindmar", "Merriton", "Norring", "Orling", "Pellworth", "Quenby", "Rowden", "Sayerly", "Tarnell", "Uldren", "Vennick", "Westerly", "Yorland", "Zellman", "O’Verran", "D’Arliss", "Mora-Celwyn", "Lund-Perrow", "de Valeira", "van Orwen", "al Naref"]
MIDDLES = ["Ada", "Bren", "Cleo", "Dane", "Eli", "Faye", "Glen", "Hope", "Ira", "Jules", "Kai", "Leigh"]

# These passages describe ordinary observations, with no institution named.
GENERAL = [
    ("Participation", "The patient asked for a pause while the discussion moved from symptoms to daily routines. After the pause, the patient could describe which task was most difficult and which part felt manageable. Staff allowed time for the answer without completing the sentence. The concern was entered in the plan so that the next conversation could begin with the same priority."),
    ("Environment", "The room was quiet during the assessment, although conversation in the corridor became audible near the end. The patient preferred to keep the door partly closed. Needed belongings were placed within comfortable reach, and the chair position was adjusted before the interview continued. The patient did not describe a new symptom related to the room or the brief interruption."),
    ("Daily routine", "The patient described mornings as easier to organize than evenings. Several small tasks tended to accumulate when fatigue increased, making it hard to distinguish the essential activities from those that could wait. The discussion focused on the sequence of the existing routine. Staff asked the patient to identify a realistic starting point rather than agree to an entire new schedule."),
    ("Understanding", "The clinician asked the patient to explain the current plan in ordinary language. The first explanation covered the main goal but left out when a change should be reported. The clinician clarified that part and invited another explanation. The patient then described the distinction between an expected variation and a concern that would need a call, with a written prompt available."),
    ("Comfort", "The patient changed position during the conversation and said that remaining in one position had become uncomfortable. A short pause allowed the assessment to continue without rushing. The patient was able to attend to the remaining questions. Staff recorded what the patient reported rather than assuming that the visible movement indicated either deterioration or complete relief."),
    ("Supports", "The patient described having help with some household tasks but preferred to complete personal activities independently when able. The availability of help varied across the day. The team discussed this as a practical planning issue and left space for the patient to correct the account. No commitment by a support person was assumed solely from the patient's expectation."),
    ("Reconciliation", "The patient brought a written list used to remember the current routine. Some entries were clear and others were abbreviated. The clinician compared the list with the information already discussed and marked the items that still required confirmation. The patient was encouraged to keep the list available for the next review so that uncertainty would not be mistaken for a confirmed change."),
    ("Handoff", "At the end of the review, the nurse repeated the patient's main concern and the task assigned to the receiving team. The patient agreed that this represented the discussion. A question remained about the timing of the next contact, so the record distinguished the proposed arrangement from a confirmed appointment. The handoff included the unresolved question for the next clinician."),
]

# Contexts convey institution identity through real administrative/clinical roles;
# quote use, position, and layout vary independently of target category.
ORG_CONTEXTS = [
    "The referral was accepted by {org}. Its admissions office requested the latest functional assessment before confirming the proposed care setting.",
    "{org} sent the prior discharge summary. The treating organization documented a gradual change in function, which the present team compared with the patient's account.",
    "The patient selected {org} as the provider for continuing care. Staff verified the organization's full name against the patient-held referral before discussing transport.",
    "Records requested from the hospital named {org} had arrived by the time of this review. The packet contained a medication list and an account of the prior assessment.",
    "A clinician employed by {org} joined the care discussion. The institution will retain responsibility for the outstanding result until the next team acknowledges receipt.",
    "The consent form names {org} as the organization authorized to receive the summary. The patient reviewed the purpose of the transfer before agreeing.",
    "The outside provider's letterhead reads {org}. Its summary describes the treatment already completed and identifies a remaining question for the receiving clinician.",
    "The patient described the previous admission to {org}, identifying it as the hospital that arranged the current follow-up. The account was consistent with the transfer document.",
    "The care coordinator called {org} and reached the institution's admissions desk. A bed had been discussed, but acceptance still depended on review of the current support needs.",
    "The receiving organization is listed in the referral as {org}. Its nurse asked for a description of the patient's usual daily routine, rather than only the most recent observation.",
    "A discharge planner from {org} confirmed that the hospital could send the missing therapy assessment. The present discussion continued while that document was being requested.",
    "The patient brought an envelope bearing the treating institution's name, {org}. Inside was an instruction sheet that the clinician reviewed alongside the current symptom history.",
    "{org} will provide the next assessment if the proposed referral is accepted. The coordinator confirmed that this is the organization's full name before entering the referral.",
    "The patient asked that the report be routed to {org}, the hospital responsible for the earlier work-up. Staff recorded the recipient and the reason for sharing the report.",
    "The originating institution, {org}, reported that the patient had participated in the discharge discussion. The next team will review whether the proposed support remains available.",
    "The transfer agreement identifies {org} as the receiving hospital. Staff discussed the expected clinical handoff and the remaining question about readiness for transfer.",
    "{org} returned the rehabilitation referral with a request for more detail about endurance. Its assessment team will reconsider the application after receiving the updated observations.",
    "The patient recognized {org} as the organization on the prior treatment plan. The clinician read the relevant instructions aloud and checked which parts were still being followed.",
    "The record from {org} includes both a discharge summary and a therapy note. The hospital's account of assistance needs was reviewed before the next plan was drafted.",
    "A records clerk at the organization registered as {org} located the requested visit summary. Clinical questions were sent to the treating team rather than resolved from the administrative entry.",
]
SPLIT_CONTEXTS = [
    "The referral names {org} as the provider in {loc}. The locality is entered separately as the municipality of the appointment, and the patient confirmed that travel to that town is feasible.",
    "Follow-up was proposed with {org} at {loc}, the town selected for the visit. The town appears in the destination field separately from the provider's name; the patient still needs to confirm transport.",
    "The outside report came from {org} in {loc}. The cover sheet lists the latter as the hospital's municipality, which matters because the proposed transport serves that town.",
    "The patient chose {org} at {loc} for the next assessment. The scheduling clerk confirmed the town using the postal locality field before discussing available visit times.",
    "The receiving provider is {org}; its appointment office is in {loc}. The patient described the municipality as familiar from previous travel and preferred to return there if the referral is accepted.",
    "The referral will go to {org} in {loc}, the municipality recorded on the routing form. The clinical summary and the travel question will be sent to different staff members for review.",
    "Transport was being arranged to {org} at {address}. The coordinator read the complete street address from the appointment notice and asked the patient to confirm the intended destination.",
    "The discharge packet names {org} in {loc}; the appointment address is {address}. Staff checked the street destination before discussing the clinical handoff.",
    "The patient showed a visit card from {org}, with {address} printed as its street address. The location was reviewed for transport planning while the clinician checked the visit instructions.",
    "The next review is at {org}. Its mailing address, {address}, was copied from the referral for the patient's transport arrangements; the patient has not yet confirmed the visit.",
]
RES_CONTEXTS = [
    "The patient lives at {org}, an independently operated residential-care organization with its own admissions office. The home's nurse described the assistance usually available during dressing and meals.",
    "A nurse employed by {org} sent a summary of the resident's routine. This named nursing home will review the proposed changes before the patient returns to its residential program.",
    "The discharge destination is {org}, the licensed residential-care organization chosen by the patient. Its admissions coordinator confirmed that the proposed support should be reviewed against the resident's usual needs.",
    "{org} accepted the residential-care referral. The institution's manager requested a description of mobility and communication needs so that staff could plan the transition with the patient.",
    "The patient asked to return to {org}, the residential nursing organization where the patient currently receives daily assistance. The home's clinician will compare the new plan with the established routine.",
]
CONTROL_CONTEXTS = [
    "The patient rested at the window while waiting for the next assessment. The clinic was quieter after the earlier appointments, and staff used that time to review the patient's priorities.",
    "The examination took place in the treatment room, with a chair positioned near the bed. The patient could hear the questions clearly and asked for a brief pause between tasks.",
    "The patient will attend a clinic for the next review once transport is arranged. A specific provider has not yet been chosen, so the conversation focused on the support needed during travel.",
    "The patient practiced the transfer at the bedside with the therapist nearby. Equipment was returned to its usual place after the task, and the patient described which part required the most effort.",
    "The team discussed options for care in a residential setting. The patient preferred to review the level of assistance before choosing a home, and no institution was selected during this conversation.",
    "The nurse reviewed the plan on the ward after the patient had rested. The patient asked for the main steps in writing and wanted time to compare them with the usual home routine.",
    "The discussion included access to a clinic on the ground floor. The patient said that walking from the entrance could be tiring, so staff recorded the need to review access before an appointment is arranged.",
    "The patient described a recent stay in hospital and recalled the care received. The team compared those instructions with the current symptom history, leaving the unresolved administrative details for a later review.",
    "A relative offered to help with transport to a center for rehabilitation. The patient wanted to understand the expected length of each visit before deciding whether that plan would be practical.",
    "The patient waited at reception and then moved to the consultation room. Staff checked comfort and hearing before beginning the review, allowing the patient to settle before describing the current concern.",
]


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def jbytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def marked(label, text):
    assert label in LABELS and "[[" not in text and "]]" not in text
    return f"[[{label}|{text}]]"


def unmark(value):
    parts, spans, offset, prev = [], [], 0, 0
    for m in re.finditer(r"\[\[([A-Z]+)\|([^\[\]]+)\]\]", value):
        literal = value[prev:m.start()]
        parts.extend((literal, m[2]))
        offset += len(literal)
        spans.append({"start": offset, "end": offset + len(m[2]), "label": m[1], "text": m[2]})
        offset += len(m[2])
        prev = m.end()
    parts.append(value[prev:])
    text = "".join(parts)
    assert "[[" not in text and "]]" not in text
    return text, spans


def full_identity(index):
    # Rejection sampling spans the same complete spelling pools for every draw.
    attempt = 0
    while True:
        n = index * 11 + attempt
        full = f"{GIVEN[n % len(GIVEN)]} {MIDDLES[(n // len(GIVEN)) % len(MIDDLES)]} {SURNAMES[(n * 7 + n // 19) % len(SURNAMES)]}"
        group = sha(unicodedata.normalize("NFC", full).casefold().encode())
        if int(group[:8], 16) % 10 < 8:
            return full, group, attempt
        attempt += 1


def context_for(fi, k):
    quote = (fi * 3 + k) % 4 == 0
    if k < 15:
        category, org = "internal_at", AT_ORGS[(fi + k * 3) % len(AT_ORGS)]
        context = ORG_CONTEXTS[(fi * 9 + k) % len(ORG_CONTEXTS)]
    elif k < 18:
        category, org = "internal_other_preposition", OTHER_ORGS[(fi * 3 + k - 15) % len(OTHER_ORGS)]
        context = ORG_CONTEXTS[(fi * 9 + k) % len(ORG_CONTEXTS)]
    elif k < 28:
        category, org = "separate_geography", PLAIN_ORGS[(fi * 3 + k) % len(PLAIN_ORGS)]
        context = SPLIT_CONTEXTS[k - 18]
    elif k < 33:
        category, org = "residential_organization", RESIDENTIAL_ORGS[(fi + 3 * k) % len(RESIDENTIAL_ORGS)]
        context = RES_CONTEXTS[(fi + k) % len(RES_CONTEXTS)]
    else:
        text = CONTROL_CONTEXTS[(fi + k) % len(CONTROL_CONTEXTS)]
        if quote:
            text += " The patient said, “I want to understand the next step,” and asked for time to discuss the plan."
        return "phi_free", text, False
    org_mark = marked("FACILITY", org)
    if quote:
        org_mark = "“" + org_mark + "”"
    loc = LOCALITIES[(fi * 7 + k) % len(LOCALITIES)]
    address = f"{210 + (fi * 29 + k * 7) % 700} {STREETS[(fi + k) % len(STREETS)]}, {loc} {ZIP_BASE[loc[-2:]] + (fi * 31 + k) % 799:05d}"
    return category, context.format(org=org_mark, loc=marked("LOCATION", loc), address=marked("ADDRESS", address)), quote


def optional_context(fi, k, full, category):
    if category == "phi_free":
        return []
    extras = []
    if (fi + 2 * k) % 5 == 0:
        extras.append(("Patient priorities", f"{marked('NAME', full)} asked that the review begin with the difficulty that affects daily routines most. The clinician invited corrections to the account before completing the care plan."))
    if (fi * 3 + k) % 9 == 0:
        months = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
        date = f"{months[(fi + k) % 12]} {1 + (fi * 3 + k) % 27}, {2022 + (fi % 4)}"
        extras.append(("Record review", f"The previous assessment dated {marked('DATE', date)} was available during this visit. The clinician compared the stated support needs with the patient's current description before deciding what required clarification."))
    if (fi + k * 3) % 23 == 0:
        phone = f"(202) 555-{100 + (fi * 7 + k) % 100:04d}"
        extras.append(("Contact preference", f"The patient confirmed {marked('PHONE', phone)} as the preferred callback number. Staff asked whether a brief message could be left and entered the patient's preference in the communication plan."))
    if (fi * 11 + k) % 37 == 0:
        email = f"visit{fi:02d}{k:02d}@patient.example"
        extras.append(("Written communication", f"The patient asked for the visit instructions to be sent to {marked('EMAIL', email)}. The nurse confirmed that written information was helpful and checked which questions should be addressed before the patient left."))
    return extras


def render(title, sections, layout, age):
    intro = f"{'An' if 80 <= age <= 89 else 'A'} {age}-year-old adult attended a review of current function and the care plan. The patient was given time to describe the main concern before the structured assessment began."
    if layout == "narrative":
        return title + "\n\n" + intro + "\n\n" + "\n\n".join(text for _, text in sections)
    if layout == "problem_list":
        return title + "\n" + intro + "\n\n" + "\n".join(f"{i + 1}. {label}: {text}" for i, (label, text) in enumerate(sections))
    if layout == "clinical_form":
        return "Clinical review\nReason: " + title + "\nVisit summary: " + intro + "\n\n" + "\n\n".join(f"{label}\n{text}" for label, text in sections)
    if layout == "handoff_table":
        return title + "\n" + intro + "\n\nTopic | Review detail\n----- | -------------\n" + "\n".join(f"{label} | {text}" for label, text in sections)
    if layout == "correspondence":
        return f"To the receiving clinician,\n\nRe: {title.lower()}\n\n{intro}\n\n" + "\n\n".join(text for _, text in sections) + "\n\nPlease include the unresolved questions in the next care review.\nThe care coordination team"
    return "Case conference: " + title + "\n" + intro + "\n\n" + "\n\n".join(f"{('Nursing', 'Therapy', 'Clinician', 'Coordinator')[i % 4]} — {label}: {text}" for i, (label, text) in enumerate(sections))


def runtime():
    # Load only tokenizer/config files, never model weights. Pull the actual
    # runtime window function via AST without importing the model implementation.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT / "hf"), local_files_only=True)
    source_path = ROOT / "src/outis/detection/hf_encoder.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"window_jobs", "special_layout"}]
    ns = {}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(source_path), 'exec'), ns)
    config = json.loads((CHECKPOINT / 'config.json').read_text())
    prefix, suffix = ns['special_layout'](tokenizer)
    content = config['max_length'] - len(prefix) - len(suffix)
    pins = {}
    for p in (CHECKPOINT / 'config.json', CHECKPOINT / 'hf/tokenizer.json', CHECKPOINT / 'hf/tokenizer_config.json', CHECKPOINT / 'hf/config.json', source_path):
        pins[str(p.relative_to(ROOT))] = sha(p.read_bytes())
    return tokenizer, ns['window_jobs'], content, pins, len(prefix) + len(suffix)


def get_positions(offsets, spans, jobs):
    positions = []
    for span in spans:
        covered = [i for i, (a, b) in enumerate(offsets) if a < span['end'] and b > span['start']]
        assert covered
        first, last = covered[0], covered[-1] + 1
        owners = [j for j, (_, _, left, right) in enumerate(jobs) if first < right and last > left]
        positions.append({'label': span['label'], 'char_start': span['start'], 'char_end': span['end'], 'token_start': first, 'token_end': last, 'owner_windows': owners, 'crosses_ownership_boundary': len(owners) > 1, 'document_third': min(2, (3 * first) // max(1, len(offsets)))})
    return positions


def validate(rows):
    assert len(rows) == 1720
    assert len({r['id'] for r in rows}) == len(rows)
    assert len({r['text'] for r in rows}) == len(rows)
    assert len({r['template_family'] for r in rows}) == 40
    assert len({r['layout'] for r in rows}) == 6
    expected = {'internal_at': 600, 'internal_other_preposition': 120, 'separate_geography': 400, 'residential_organization': 200, 'phi_free': 400}
    assert Counter(r['target_category'] for r in rows) == expected
    for category in expected:
        assert len({r['layout'] for r in rows if r['target_category'] == category}) == 6
    for row in rows:
        assert row['split'] == row['suite'] == 'train'
        assert row['annotation_policy'] == 'clinical-phi-v2'
        assert row['phenomena'] and len(row['phenomena']) == len(set(row['phenomena']))
        assert row['pair_id'] and row['identity_group']
        assert row['negative_only'] == (not row['spans'])
        previous = -1
        for s in row['spans']:
            assert s['label'] in LABELS
            assert previous <= s['start'] < s['end'] <= len(row['text'])
            assert row['text'][s['start']:s['end']] == s['text'] == s['text'].strip()
            assert not s['text'].endswith(('.', ';', ':'))
            previous = s['end']
        for name, group in zip(row['full_identities'], row['identity_groups']):
            assert sha(unicodedata.normalize('NFC', name).casefold().encode()) == group
            assert int(group[:8], 16) % 10 < 8
        if row['target_category'] == 'phi_free':
            assert not row['spans'] and not row['identity_groups'] and not row['full_identities']
        else:
            assert len(row['identity_groups']) == 1
        if row['cohort'] == 'long_document':
            assert row['runtime']['window_count'] >= 3
    assert sum(r['cohort'] == 'long_document' for r in rows) >= .15 * len(rows)


def build():
    tokenizer, window_jobs, content_window, pins, specials = runtime()
    rows, rejected = [], 0
    for fi, scene in enumerate(SCENARIOS):
        clinical_keys = ('summary', 'observation', 'assessment', 'plan', 'home', 'rehab', 'safety', 'followup')
        assert all(isinstance(scene[key], str) and len(scene[key].split()) >= 35 for key in clinical_keys)
        for k in range(43):
            idx = fi * 43 + k
            category, target, quoted = context_for(fi, k)
            is_long = ((k * 17 + fi * 11) % 43) < 9
            layout = LAYOUTS[(fi * 7 + k) % 6]
            full, group, tries = full_identity(idx)
            rejected += tries
            order = list(clinical_keys)
            # The presenting concern anchors every note. Other short sections
            # rotate, while a long review includes every clinical domain once.
            if is_long:
                keys = order
                sections = [(key.replace('_', ' ').capitalize(), scene[key]) for key in keys]
                general_start = (fi + k) % len(GENERAL)
                sections += [GENERAL[(general_start + j) % len(GENERAL)] for j in range(7)]
            else:
                start = (fi + k) % 7
                keys = ['summary'] + [order[1 + (start + j) % 7] for j in range(3)]
                sections = [(key.capitalize(), scene[key]) for key in keys]
                sections.append(GENERAL[(fi + k * 3) % len(GENERAL)])
            # Insertion positions vary independently of names/quotes/layouts.
            anchor = (fi * 13 + k * 7) % (len(sections) + 1)
            sections.insert(anchor, ('Care coordination', target))
            for extra_no, extra in enumerate(optional_context(fi, k, full, category)):
                position = (fi * 5 + k * 11 + extra_no * 3) % (len(sections) + 1)
                sections.insert(position, extra)
            text, spans = unmark(render(scene['title'], sections, layout, 34 + (fi * 3 + k) % 51))
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = encoded['offset_mapping']
            jobs = list(window_jobs(len(encoded['input_ids']), content_window))
            positions = get_positions(offsets, spans, jobs)
            row = {
                'id': f'tk8_facility_contexts_{fi:02d}_{k:02d}', 'text': text, 'spans': spans,
                'split': 'train', 'suite': 'train', 'role': 'training', 'annotation_policy': 'clinical-phi-v2',
                'identity_group': f'author8-no-person-{fi:02d}-{k:02d}' if category == 'phi_free' else group,
                'identity_groups': [] if category == 'phi_free' else [group],
                'full_identities': [] if category == 'phi_free' else [full],
                'pair_id': f'author8-unpaired-{fi:02d}-{k:02d}',
                'template_family': 'author8_' + scene['key'], 'family': 'author8_' + scene['key'],
                'phenomena': [category, 'clinical_context', 'ordinary_age', 'varied_document_layout', 'long_runtime_windows' if is_long else 'routine_length'],
                'cohort': 'long_document' if is_long else 'realistic',
                'layout': layout, 'target_category': category, 'quoted_organization': quoted,
                'negative_only': not spans,
                'runtime': {'token_count': len(encoded['input_ids']), 'window_count': len(jobs), 'windows': [list(job) for job in jobs], 'gold_positions': positions},
                'synthetic_provenance': {'author': 'author8_facility_contexts', 'source': 'independently_authored', 'identity_scheme': 'invented full names; NFC/casefold SHA256 first-8-hex modulo-10 rejection sampling to train'},
            }
            rows.append(row)
    validate(rows)
    support = Counter(s['label'] for r in rows for s in r['spans'])
    categories = Counter(r['target_category'] for r in rows)
    cohorts = Counter(r['cohort'] for r in rows)
    layouts = Counter(r['layout'] for r in rows)
    lengths = sorted(r['runtime']['token_count'] for r in rows)
    long_rows = [r for r in rows if r['cohort'] == 'long_document']
    facility_positions = Counter(p['document_third'] for r in long_rows for p in r['runtime']['gold_positions'] if p['label'] == 'FACILITY')
    ownership_crossings = Counter(p['label'] for r in rows for p in r['runtime']['gold_positions'] if p['crosses_ownership_boundary'])
    corpus_blob = b''.join((json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode() for r in rows)
    source_sha = sha(Path(__file__).read_bytes())
    manifest = {
        'schema_version': 1, 'corpus': str(CORPUS.relative_to(ROOT)), 'sha256': sha(corpus_blob),
        'split': 'train', 'suite': 'train', 'role': 'training', 'annotation_policy': 'clinical-phi-v2',
        'rows': len(rows), 'documents': len(rows), 'count': len(rows), 'spans': sum(support.values()),
        'label_support': {label: support[label] for label in LABELS},
        'entity_support': {label: support[label] for label in LABELS},
        'label_counts': {label: support[label] for label in LABELS},
        'phi_free_rows': categories['phi_free'], 'cohort_counts': dict(cohorts),
        'template_family_count': 40, 'layout_counts': dict(layouts), 'category_counts': dict(categories),
        'generator': str(Path(__file__).relative_to(ROOT)), 'generator_sha256': source_sha,
        'synthetic_provenance': 'New author-written scenarios and institution names, expanded deterministically. No dataset supplied or read. No real clinical record source.',
        'identity_assignment': 'int(SHA256(NFC(full_name).casefold().encode(utf8))[:8],16)%10<8; reject and draw again otherwise',
        'publication_purpose': 'training draft for a possible later round; not admitted to the frozen current mixture',
        'runtime': {'max_length': content_window + specials, 'special_tokens': specials, 'content_window': content_window, 'actual_overlap': min(128, content_window // 4), 'actual_stride': content_window - 2 * min(128, content_window // 4), 'source_pins': pins},
        'reserved_identifiers': 'Phones use US 202-555-0100 through 0199; email addresses use patient.example. Addresses and localities independently invented; non-deliverability not externally checked.',
        'outputs_frozen_mode': '0444',
    }
    manifest_blob = jbytes(manifest)
    report = {
        'schema_version': 1, 'status': 'training_draft', 'independent_qa_complete': False, 'split': 'train', 'suite': 'train',
        'corpus': {'path': str(CORPUS.relative_to(ROOT)), 'sha256': sha(corpus_blob), 'rows': len(rows)},
        'manifest': {'path': str(MANIFEST.relative_to(ROOT)), 'sha256': sha(manifest_blob)},
        'generator': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': source_sha},
        'category_counts': dict(categories), 'label_support': dict(support), 'cohort_counts': dict(cohorts),
        'families': [scene['key'] for scene in SCENARIOS], 'family_count': 40, 'layout_counts': dict(layouts),
        'category_layout_counts': {cat: dict(Counter(r['layout'] for r in rows if r['target_category'] == cat)) for cat in categories},
        'quoted_organizations': sum(r['quoted_organization'] for r in rows),
        'any_curly_quotation_by_category': {cat: sum('“' in r['text'] for r in rows if r['target_category'] == cat) for cat in categories},
        'quoted_by_category': {cat: sum(r['quoted_organization'] for r in rows if r['target_category'] == cat) for cat in categories},
        'internal_preposition_row_counts': dict(Counter(next(prep for prep in ('at', 'of', 'for', 'in', 'on') if f' {prep} ' in next(s['text'] for s in r['spans'] if s['label'] == 'FACILITY')) for r in rows if r['target_category'].startswith('internal_'))),
        'unique_facility_values': len({s['text'] for r in rows for s in r['spans'] if s['label'] == 'FACILITY'}),
        'span_surface_profile': {
            label: {
                'count': support[label],
                'non_ascii_count': sum(any(ord(c) > 127 for c in s['text']) for r in rows for s in r['spans'] if s['label'] == label),
                'apostrophe_count': sum(any(c in s['text'] for c in ("'", '’')) for r in rows for s in r['spans'] if s['label'] == label),
                'hyphen_count': sum('-' in s['text'] for r in rows for s in r['spans'] if s['label'] == label),
                'all_uppercase_count': sum(s['text'].isupper() for r in rows for s in r['spans'] if s['label'] == label),
                'all_lowercase_count': sum(s['text'].islower() for r in rows for s in r['spans'] if s['label'] == label),
                'min_characters': min(len(s['text']) for r in rows for s in r['spans'] if s['label'] == label),
                'max_characters': max(len(s['text']) for r in rows for s in r['spans'] if s['label'] == label),
            } for label in support
        },
        'script_coverage': 'Latin-script clinical prose and names, including accented Latin person names; Arabic-script and East Asian-script names were not authored.',
        'full_person_identity_metadata': {'rows': sum(bool(r['full_identities']) for r in rows), 'unique_identities': len({g for r in rows for g in r['identity_groups']}), 'rejected_draws': rejected, 'all_accepted_partition': 'train', 'name_annotation_rows': sum(any(s['label'] == 'NAME' for s in r['spans']) for r in rows), 'note': 'Synthetic encounter identities are recorded for all PHI-bearing notes, including notes without a written person name; no person identity is assigned to controls.'},
        'runtime_window_coverage': {'source_pins': pins, 'library_versions': {name: package_version(name) for name in ('transformers', 'tokenizers')}, 'max_length': content_window + specials, 'content_window': content_window, 'special_tokens': specials, 'actual_overlap': min(128, content_window // 4), 'actual_stride': content_window - 2 * min(128, content_window // 4),
            'window_count_histogram': dict(Counter(r['runtime']['window_count'] for r in rows)),
            'tokens': {'min': min(lengths), 'median': statistics.median(lengths), 'p90': lengths[int(.9 * len(lengths))], 'max': max(lengths)},
            'long_rows': len(long_rows), 'long_fraction': len(long_rows) / len(rows),
            'long_min_tokens': min(r['runtime']['token_count'] for r in long_rows), 'long_min_windows': min(r['runtime']['window_count'] for r in long_rows),
            'long_by_category': dict(Counter(r['target_category'] for r in long_rows)),
            'long_facility_position_thirds': {str(k): v for k, v in sorted(facility_positions.items())},
            'gold_crosses_ownership_boundary_by_label': dict(ownership_crossings),
            'placement_method': 'Arithmetic section insertion varies organization position; actual tokenizer offsets and actual runtime window ownership recorded on every row. No model inference.'},
        'structural_checks': {'unique_ids': True, 'unique_texts': True, 'valid_nonoverlapping_exact_character_spans': True, 'metadata_complete': True, 'identities_train_hash_partition': True, 'controls_have_zero_spans': True, 'minimum_three_actual_windows_for_all_long_notes': True, 'all_target_categories_in_all_layouts': True},
        'author_review': AUTHOR_REVIEW,
        'read_scope': ['docs/annotation-policy-v2.md', 'src/outis/data/annotation_policy.py', 'scripts/takeover/normalize_conventions.py', 'scripts/takeover/normalize_date_intervals.py', 'scripts/takeover/prepare_training_mixture.py', 'src/outis/detection/hf_encoder.py', 'models/takeover-mbbase-mix3-001/step-003081/config.json', 'models/takeover-mbbase-mix3-001/step-003081/hf/tokenizer.json', 'models/takeover-mbbase-mix3-001/step-003081/hf/tokenizer_config.json', 'models/takeover-mbbase-mix3-001/step-003081/hf/config.json'],
        'limitations': ['Synthetic prose uses 40 author-created clinical scenarios and reusable coordination contexts; row count is not independent clinical evidence.', 'Institution names are invented but coincidental real-world matches were not checked through external search.', 'Long notes are deliberately expanded clinical-domain reviews with shared neutral nursing paragraphs; they are authored stress records, not naturally sampled encounters.', 'No development or sealed comparison, model inference, quality auditor, normalization CLI, or training-mixture mutation was run.', 'An independent semantic review and separately authorized collision/admission checks remain necessary before any future training use.', 'No institutional subunits or ambiguous named wards were intentionally authored.'],
        'current_mixture_changed': False, 'outputs_frozen_mode': '0444',
    }
    return rows, corpus_blob, manifest_blob, jbytes(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preview', action='store_true')
    action.add_argument('--write', action='store_true')
    action.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    rows, corpus_blob, manifest_blob, report_blob = build()
    outputs = ((CORPUS, corpus_blob), (MANIFEST, manifest_blob), (REPORT, report_blob))
    if args.preview:
        summary = json.loads(report_blob)
        print(json.dumps({key: summary[key] for key in ('category_counts', 'label_support', 'runtime_window_coverage', 'category_layout_counts')}, indent=2))
        for fi, k in ((0, 0), (1, 15), (2, 16), (3, 17), (4, 19), (5, 24), (6, 25), (7, 28), (8, 31), (9, 33), (10, 37), (11, 42)):
            row = rows[fi * 43 + k]
            print(json.dumps({'id': row['id'], 'layout': row['layout'], 'text': row['text'], 'spans': row['spans'], 'runtime': row['runtime']}, ensure_ascii=False, indent=2))
    elif args.write:
        for path, _ in outputs:
            if path.exists() or path.is_symlink():
                raise FileExistsError(f'Refusing overwrite: {path}')
        for path, blob in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
                os.fchmod(handle.fileno(), 0o444)
        Path(__file__).chmod(0o444)
        print(json.dumps({'status': 'training_draft', 'independent_qa_complete': False, 'files': [{'path': str(path), 'sha256': sha(blob), 'bytes': len(blob)} for path, blob in outputs], 'generator_sha256': sha(Path(__file__).read_bytes())}, indent=2))
    else:
        for path, blob in outputs:
            assert path.read_bytes() == blob, f'regeneration differs: {path}'
            assert path.stat().st_mode & 0o777 == 0o444
        assert Path(__file__).stat().st_mode & 0o777 == 0o444
        print(json.dumps({'verified': True, 'rows': len(rows), 'corpus_sha256': sha(corpus_blob), 'manifest_sha256': sha(manifest_blob), 'report_sha256': sha(report_blob), 'generator_sha256': sha(Path(__file__).read_bytes())}))


AUTHOR_REVIEW = {
    'status': 'author_checked',
    'semantic_basis': 'Full institution names explicitly supported by treating/receiving organization roles; separate geography corroborated by locality, municipality, postal, or street-address language; residential entities are complete organizations.',
    'controls': 'Generic unnamed clinics, wards, reception, building positions, and services left unannotated; no blanket merging at prepositions.',
    'manual_review_sample_ids': ['tk8_facility_contexts_%02d_%02d' % (fi, k) for fi, k in ((0, 0), (1, 15), (2, 16), (3, 17), (4, 19), (5, 24), (6, 25), (7, 28), (8, 31), (9, 33), (10, 37), (11, 42), (1, 16), (1, 17), (2, 15))],
    'scope': 'Author read all 40 scenario summaries, all authored coordination-context definitions, and 15 assembled note/span samples spanning six layouts and every target category and internal preposition; this is not independent QA.',
    'author_repairs_before_publication': ['An article for ages beginning with eighty', 'Generic residential-care license wording', 'Ordinary patient quotation in one-quarter of controls', 'Removed an unnecessarily explicit unnamed-organization cue from a control'],
}

# Independent clinical scenario prose is embedded below, not read from a corpus.
SCENARIOS = [
  {
    "key": "wound_recovery",
    "title": "Recovery after a lower limb wound",
    "summary": "The patient is recovering from a shallow lower limb wound that has limited walking and made dressing changes uncomfortable. Drainage has decreased, but tenderness remains near one edge. The patient wants to resume ordinary household tasks and is uncertain how much activity is compatible with the current wound care plan.",
    "observation": "The dressing was intact on arrival, with a small area of dried drainage and no obvious strike through. Surrounding skin appeared mildly irritated where adhesive had been applied. The patient moved cautiously when standing and described discomfort as localized rather than spreading, although the wound itself was not fully assessed during this encounter.",
    "assessment": "Healing appears to be progressing by the patient’s account, but the available observations do not establish the depth or cause of the wound. Local irritation may contribute to discomfort alongside the wound itself. Reduced activity also appears to have lowered confidence with walking, creating a functional concern that merits attention during recovery.",
    "plan": "The clinician will review the documented wound care instructions and compare the next examination with prior descriptions. The patient’s questions about dressing comfort and household activity will be included in that review. Decisions about changing products or increasing activity will depend on direct assessment, symptom trends, and the patient’s ability to carry out the plan.",
    "home": "At home, the patient has arranged commonly used supplies within reach and uses a chair for tasks that previously required prolonged standing. A household member can assist with heavier chores but is not always available. The patient described wanting clear instructions that fit this arrangement without relying on help for every routine activity.",
    "rehab": "Rehabilitation discussion focused on preserving comfortable movement while respecting the existing wound care restrictions. The patient practiced a short transfer sequence and identified when discomfort begins to alter foot placement. Future sessions may address pacing and gradual return to usual tasks, with progression coordinated with the clinician responsible for assessing the wound.",
    "safety": "The patient can describe which changes have previously prompted contact with the care team, but is less certain about the significance of mild skin irritation. The clinician will clarify the individualized escalation instructions. Supplies are stored away from the walking path, and the patient has agreed to avoid improvising changes to the dressing regimen.",
    "followup": "Followup will include a direct wound review, comparison of drainage and surrounding skin, and discussion of how dressing care is working at home. The patient also wants to revisit walking tolerance. Persistent or changing discomfort will be assessed in context rather than assumed to represent either routine healing or a definite complication."
  },
  {
    "key": "gait_confidence",
    "title": "Gait confidence after a recent fall",
    "summary": "The patient reports reduced confidence with walking after a fall that occurred while turning toward a chair. There has been no further fall reported, but movement around the home has become slower and more deliberate. The patient is particularly concerned about changing direction while carrying an object or responding to another person.",
    "observation": "During observation, the patient took short steps and paused before turns. Standing balance appeared steadier when both hands were available for support. The patient looked toward the floor frequently and required extra time to approach a seat, although the transfer was completed without physical assistance during the observed sequence in this encounter.",
    "assessment": "The current limitation appears to involve both balance concerns and fear of another fall. The relative contribution of strength, sensation, vision, and medication effects has not been established. Avoiding activity may further restrict function, so the assessment will consider confidence and participation alongside physical findings rather than attributing the difficulty to a single cause.",
    "plan": "The plan is to complete the relevant mobility assessment and review the circumstances surrounding the fall in more detail. The clinician will also reconcile any recent changes that could affect steadiness. Agreed goals will emphasize reliable movement during ordinary tasks, with the patient involved in choosing priorities that feel useful and achievable.",
    "home": "The patient describes narrow passages created by movable household items and sometimes leaves objects on a chair before sitting. A household member is willing to help rearrange the busiest areas. The patient prefers small changes that preserve familiar routines and does not want essential belongings moved without discussion about where they will go.",
    "rehab": "Rehabilitation will focus on the observed challenges with approaching a chair and changing direction. Practice will be supervised and adjusted to the patient’s demonstrated stability rather than a fixed distance target. The therapist will also explore how attention, fatigue, and carrying objects affect performance during tasks that resemble the patient’s ordinary routine.",
    "safety": "The patient was encouraged to discuss any further loss of balance with the care team so that the pattern can be reviewed. The current support arrangement and equipment fit will be checked before increasing task difficulty. The patient understands that a successful practice attempt does not by itself establish safety in every home situation.",
    "followup": "At followup, the team will review falls, near falls, confidence, and the practical effects of any changes made at home. Performance will be compared across similar tasks rather than judged only by walking distance. The patient wants progress described in terms of activities regained and the amount of assistance still needed."
  },
  {
    "key": "migraine_interference",
    "title": "Recurrent headache affecting daily function",
    "summary": "The patient describes recurring headaches associated with light sensitivity and difficulty concentrating. Episodes vary in intensity and sometimes interrupt household responsibilities. The patient has noticed possible links with disrupted sleep and missed meals but is unsure whether those patterns are consistent, and would like a clearer record before drawing conclusions about triggers.",
    "observation": "The patient was comfortable enough to participate in conversation but preferred reduced ambient light. Speech was clear, and the patient followed the discussion without obvious difficulty. Symptoms described during the visit resembled the patient’s reported usual pattern, although this brief observation did not replace a complete headache or neurological assessment by the clinician.",
    "assessment": "The history suggests a recurring headache syndrome with meaningful effects on function, but the contribution of sleep, stress, and medication use remains uncertain. The patient’s possible trigger observations are preliminary. Assessment will distinguish the established pattern from any new features and consider whether current symptom management is meeting the patient’s practical goals.",
    "plan": "The clinician will review headache frequency, associated symptoms, functional impact, and the response to the existing treatment plan. The patient will be invited to keep a simple record that is manageable during symptomatic periods. Any treatment adjustment will follow review of the complete history, current medicines, and factors that may influence tolerability.",
    "home": "At home, the patient has identified a quieter area where interruptions can be reduced during an episode. Family responsibilities sometimes make rest difficult, and the patient worries about appearing unreliable. Discussion focused on communicating anticipated needs and arranging flexible help without assuming that every headache will require the same level of support.",
    "rehab": "Rehabilitation needs center on maintaining participation and avoiding a cycle of overactivity followed by prolonged withdrawal. The patient will explore pacing with the clinician and identify tasks that remain manageable during milder symptoms. Visual or concentration demands may be adjusted temporarily, with attention to recovery of ordinary function rather than symptom avoidance alone.",
    "safety": "The patient’s understanding of the existing instructions for a headache that differs from the usual pattern will be checked. Medication use will be reviewed without assuming that an available product is appropriate for every episode. The clinician will also assess whether symptoms interfere with tasks that require sustained attention or rapid responses.",
    "followup": "Followup will compare the symptom record with the patient’s account of daily functioning and review whether proposed triggers recur consistently. The patient would like to measure improvement by fewer interrupted activities as well as headache intensity. Uncertainty about the diagnosis or response to treatment will be documented and revisited as additional information becomes available."
  },
  {
    "key": "chronic_breathlessness",
    "title": "Chronic breathlessness during ordinary activity",
    "summary": "The patient reports chronic breathlessness that becomes more noticeable during dressing and household movement. Respiratory symptoms have been variable, with occasional cough and periods of greater fatigue. The patient is concerned about losing independence and wants help distinguishing ordinary variation from changes that should be discussed promptly with the treating clinical team.",
    "observation": "During conversation, the patient spoke in complete phrases and paused after walking a short distance. Shoulder tension increased during exertion and eased after seated rest. No conclusion about oxygenation or the cause of breathlessness was drawn from appearance alone, and the patient’s account will be considered alongside the planned respiratory assessment.",
    "assessment": "Several factors may contribute to the activity limitation, including the underlying respiratory condition, reduced conditioning, and apprehension about becoming breathless. Their relative importance is not yet clear. The patient’s tendency to rush through personal care may also make tasks harder, but that observation does not fully explain the reported variability in symptoms.",
    "plan": "The clinician will review the recent respiratory history, current treatment use, and any barriers to following the established plan. The patient’s goals will be translated into specific activities for reassessment. Changes to respiratory treatment or monitoring will be based on clinical evaluation rather than on a single observed episode of exertional breathlessness.",
    "home": "The patient has begun arranging toiletries and clothing before starting personal care, which reduces repeated trips across the room. Some household tasks still involve reaching and carrying at the same time. Available help is intermittent, so the patient prefers practical adjustments that can be used independently and do not require complicated equipment.",
    "rehab": "Rehabilitation will address pacing, efficient task setup, and activity tolerance within the treating clinician’s recommendations. The therapist will observe how the patient responds to short functional tasks and recovery periods. Progress will depend on symptoms and the wider assessment, with the patient encouraged to report what feels manageable rather than pushing toward an arbitrary target.",
    "safety": "The patient has an existing respiratory action plan but finds some of its wording difficult to apply to everyday symptoms. The care team will review it using the patient’s own examples. Equipment use and treatment technique will also be checked, with any uncertainty referred to the responsible clinician before independent changes are made.",
    "followup": "At followup, the patient’s experience with personal care and household movement will be reviewed alongside respiratory findings. The team will ask about symptom changes, treatment difficulties, and the usefulness of pacing strategies. Improvement will be considered in terms of consistent participation and confidence as well as the amount of activity completed."
  },
  {
    "key": "reflux_discomfort",
    "title": "Upper digestive discomfort with disrupted routines",
    "summary": "The patient describes intermittent burning discomfort after meals and an unpleasant taste when reclining. Symptoms have led to smaller evening meals and worry about eating before rest. The patient is unsure whether specific foods are responsible because symptoms also occur during periods of irregular meals, disrupted sleep, and increased personal stress.",
    "observation": "The patient participated comfortably in the discussion and described symptoms without visible distress during the encounter. There was no meal observation or direct assessment of swallowing. The account suggested variation across routines, but the available information was insufficient to determine whether the discomfort reflects one consistent process or more than one contributing problem.",
    "assessment": "The symptom pattern may be compatible with reflux, though the history requires fuller assessment before that explanation is accepted as complete. Changes in eating behavior could affect nutritional adequacy if they become more restrictive. The patient’s concerns also warrant attention because fear of symptoms is beginning to shape ordinary social and household choices.",
    "plan": "The clinician will review symptom timing, associated features, current medicines, and the impact of self directed dietary changes. The patient will be asked which parts of the existing plan are clear and which remain difficult. Further evaluation and any treatment changes will follow the clinical history and examination rather than assumptions about particular foods.",
    "home": "At home, meal timing varies with shared responsibilities, and the patient sometimes eats quickly after a long gap. The household can support a more predictable routine, but preferences differ among members. The patient would like a plan that preserves shared meals and avoids an increasingly narrow list of foods without clear evidence.",
    "rehab": "Rehabilitation discussion focused on restoring comfortable participation in ordinary tasks and social meals. The patient identified worry about bending and reclining as reasons for avoiding some activities. The team will coordinate functional advice with the medical assessment so that adjustments remain proportionate and do not reinforce unnecessary restrictions before the symptoms are better understood.",
    "safety": "The patient’s understanding of when digestive symptoms need additional assessment will be reviewed using the existing clinical instructions. Over the counter products and prescribed medicines will be reconciled to identify duplication or confusion. The patient will be supported in raising new concerns rather than assuming that all upper abdominal or chest discomfort has the same explanation.",
    "followup": "Followup will assess symptom frequency, the effect on eating and sleep, and whether the plan is practical within the household routine. The clinician will revisit any associated features that were uncertain at the first encounter. The patient wants reassurance grounded in an adequate assessment and a clear explanation of the remaining uncertainty."
  },
  {
    "key": "constipation_routine",
    "title": "Constipation during reduced activity",
    "summary": "The patient reports less frequent bowel movements and a sense of incomplete evacuation after a period of reduced activity. Appetite remains variable, and the patient has become hesitant to leave home when bowel habits feel unpredictable. Several self directed changes have been tried, but the patient cannot clearly identify which, if any, were helpful.",
    "observation": "The patient described abdominal fullness without appearing acutely distressed during conversation. The history of stool consistency and timing was incomplete because the patient had not kept a record. No abdominal examination was performed within this portion of the encounter, so the observations cannot establish the severity or underlying cause of the bowel change.",
    "assessment": "Reduced activity, changes in intake, and medication effects are possible contributors, but none has been confirmed as the primary cause. Anxiety about bowel predictability may be amplifying the effect on participation. The assessment should address the change from the patient’s usual pattern while avoiding assumptions that constipation is an inevitable consequence of recovery.",
    "plan": "The clinician will review the bowel history, associated symptoms, current medicines, and all remedies already used. A simple record may help clarify the pattern if the patient finds it manageable. The treatment plan will be individualized after assessment, with clear explanations of what is expected and how unresolved concerns should be communicated.",
    "home": "At home, access to the preferred bathroom is sometimes interrupted by shared household routines. The patient occasionally postpones responding to the urge to open the bowels while completing other responsibilities. Discussion identified privacy and predictable opportunities as practical issues, though the team will not assume that changing these alone will resolve the symptoms.",
    "rehab": "Rehabilitation will consider comfortable movement and participation within the patient’s broader recovery plan. The patient has been avoiding walks because of concern about bathroom access, which may further reduce daily activity. Functional goals will therefore account for access and confidence while the medical team evaluates the bowel symptoms and advises on appropriate management.",
    "safety": "The patient has used several products with overlapping purposes and is uncertain which were intended for regular versus occasional use. The clinician will reconcile these before recommending changes. The existing instructions about concerning symptom changes will be clarified, and persistent discomfort will not be dismissed solely because constipation has occurred in the past.",
    "followup": "Followup will review bowel pattern, discomfort, appetite, and the effect on leaving home or completing ordinary tasks. The team will check whether the agreed plan is understandable and whether any products remain confusing. If symptoms persist or change, the original assessment will be reconsidered rather than repeatedly extending an ineffective routine."
  },
  {
    "key": "vertigo_transfers",
    "title": "Positional dizziness affecting transfers",
    "summary": "The patient reports brief episodes of dizziness when changing position, particularly during bed transfers and when looking upward. The episodes have made the patient cautious with movement and less willing to complete tasks alone. The patient cannot consistently distinguish spinning from lightheadedness and would like help describing the symptoms more accurately during assessment.",
    "observation": "During the encounter, the patient moved slowly from sitting to standing and waited before walking. No sustained episode was observed, and there was no attempt to provoke symptoms outside the planned assessment. The patient described feeling steadier with a nearby support surface, although confidence and actual balance performance may not always change together.",
    "assessment": "The positional association is relevant, but the nature and cause of the dizziness remain uncertain. A vestibular problem, circulatory factors, medication effects, or other contributors may require consideration by the clinician. The functional impact is already clear because the patient has restricted movement and relies more heavily on others during routine transfers.",
    "plan": "The clinician will obtain a fuller symptom history and determine which examinations are appropriate. The patient will be asked to describe duration, associated sensations, and the circumstances of episodes in ordinary language. Recommendations will follow those findings, and any targeted movement procedures will be undertaken only within the responsible clinician’s assessment and treatment plan.",
    "home": "At home, the patient has begun placing frequently used items at a comfortable height and sits briefly before leaving bed. A household member sometimes stays nearby for transfers but cannot provide continuous supervision. The patient wants a realistic approach that supports independence while acknowledging that symptoms are intermittent and therefore difficult for others to anticipate.",
    "rehab": "Rehabilitation will begin with observation of the transfers and movements that matter most to the patient. The therapist will consider balance, confidence, and symptom behavior without presuming that all dizziness has the same mechanism. Activities will be adjusted to the clinical findings, and progress will include reduced reliance on reassurance as well as improved movement.",
    "safety": "The patient will receive clarification of the existing instructions for new, prolonged, or otherwise different symptoms. Activities in which an unexpected episode could lead to injury will be discussed individually. The team will also review whether support surfaces and current equipment are appropriate, rather than relying on improvised arrangements that have not been assessed.",
    "followup": "Followup will revisit the patient’s description of dizziness and compare it with findings from the targeted assessment. The patient will report which transfers remain difficult and whether episodes have changed in character. The care team will explain any remaining diagnostic uncertainty and revise the functional plan if the initial approach does not help."
  },
  {
    "key": "hand_function",
    "title": "Hand stiffness limiting routine tasks",
    "summary": "The patient describes hand stiffness and aching that interfere with opening containers, fastening clothing, and preparing meals. Symptoms fluctuate and are more noticeable after repeated gripping. The patient has begun using the other hand for several tasks but finds this awkward, and is concerned about gradually losing the ability to manage familiar routines independently.",
    "observation": "Observation showed slower finger movement and a tendency to use the whole hand when handling small objects. The patient paused during a fastening task because of discomfort and completed it with a modified grip. These findings demonstrate task difficulty but do not establish the source of the stiffness or the extent of any joint involvement.",
    "assessment": "The functional limitations may reflect pain, stiffness, reduced grip efficiency, or a combination of these factors. Compensating with the opposite hand could increase effort elsewhere, although this has not been formally assessed. The clinician will consider symptom distribution and changes over time before attributing the presentation to a specific musculoskeletal or neurological condition.",
    "plan": "The plan is to complete a focused assessment and identify the tasks the patient most wants to regain. Current symptom management and any devices already purchased will be reviewed. Recommendations will be tailored to examination findings and practical needs, with an emphasis on understanding which adjustments reduce effort without creating additional discomfort or dependence.",
    "home": "At home, the patient has moved heavier containers to accessible shelves and sometimes asks for help with tightly sealed packages. Assistance is available for some meals, but personal care remains a priority for independence. The patient prefers discreet adaptations that can be incorporated into existing routines without replacing every familiar household item.",
    "rehab": "Rehabilitation will examine grip patterns, joint movement, and task demands using ordinary activities as examples. The therapist will trial selected adaptations with the patient and assess both comfort and usefulness. Exercise or loading recommendations will depend on the clinical assessment, and improvement will be measured through meaningful tasks rather than grip strength alone.",
    "safety": "The patient occasionally uses excessive force when a container resists opening and recognizes that frustration contributes to this pattern. The team will discuss safer task alternatives after observing the relevant difficulties. Skin condition, sensation, and equipment fit will be considered where appropriate, particularly before recommending devices that apply pressure or require sustained gripping.",
    "followup": "At followup, the patient will review which tasks have become easier and which adaptations were abandoned or inconvenient. The clinician will reassess any changes in pain, swelling, sensation, or movement. The plan will remain open to revision if the initial explanation does not account for the full pattern of symptoms or functional difficulty."
  },
  {
    "key": "knee_recovery",
    "title": "Knee recovery after a procedure",
    "summary": "The patient is recovering from a knee procedure and reports a mixture of stiffness, soreness, and concern about movement. Walking has improved compared with the earliest recovery period, but stairs and getting into a low chair remain difficult. The patient wants to understand whether current progress fits the individualized expectations discussed by the treating team.",
    "observation": "The patient approached the chair carefully and used both arms during the transfer. The involved leg remained slightly forward when sitting, suggesting an attempt to limit discomfort. A short walk showed uneven step length, but the observation was too limited to establish endurance, movement range, or the status of the healing tissues.",
    "assessment": "The observed movement pattern may reflect expected protective behavior, persistent symptoms, or uncertainty about permitted activity. These possibilities cannot be separated without reviewing the procedure specific plan and examination findings. The patient’s concern about doing too much appears to influence participation and should be addressed alongside objective measures of recovery and function.",
    "plan": "The clinician will confirm current precautions and compare the patient’s progress with the documented recovery plan. Questions about stairs, seating, and symptom fluctuations will be reviewed directly. Any change in exercises, support equipment, or activity expectations will be coordinated with the treating team and based on the patient’s current presentation rather than a generic schedule.",
    "home": "At home, the patient uses a higher chair and has arranged essential items to reduce unnecessary trips. The bathroom setup remains awkward, and assistance is needed for some lower body tasks. The patient would like to become less dependent but recognizes that household convenience does not by itself determine whether an activity is appropriate.",
    "rehab": "Rehabilitation will focus on the transfers and walking patterns observed during assessment, within confirmed precautions. The therapist will monitor how symptoms respond and clarify the purpose of each activity. Progression will reflect movement quality, confidence, and functional tolerance, while concerns about the healing process will be referred back to the responsible clinical team.",
    "safety": "The patient is unsure whether all previously provided instructions remain current, particularly those concerning support equipment and stairs. The team will reconcile these instructions before recommending changes. Any new or unexpected symptom pattern will be considered clinically rather than automatically categorized as routine postoperative discomfort, and the patient’s escalation plan will be reviewed.",
    "followup": "Followup will compare the same functional tasks and reassess the patient’s understanding of the recovery plan. The patient wants to track progress through easier chair transfers and more consistent household walking. The team will document barriers, clarify unresolved restrictions, and reconsider the approach if symptoms or function diverge from the expected individual course."
  },
  {
    "key": "shoulder_recovery",
    "title": "Shoulder discomfort after an injury",
    "summary": "The patient reports shoulder discomfort after an injury and has difficulty reaching for clothing, managing hair care, and finding a comfortable sleeping position. The patient has reduced use of the arm because movement feels uncertain. There is concern that ordinary tasks might delay recovery, although the current activity instructions are not fully understood.",
    "observation": "The patient kept the affected arm close to the body and used the other hand to help position it during a clothing task. Movement appeared guarded, with a pause before reaching. No attempt was made to test the limits of movement during this observation because the relevant precautions and injury details required confirmation.",
    "assessment": "Pain and uncertainty may both contribute to the guarded movement, but structural injury and other causes need consideration within the clinical assessment. The current degree of avoidance may be greater than the patient intended. The team will distinguish necessary protection from unnecessary restriction once the treating clinician’s recommendations and examination findings are available.",
    "plan": "The clinician will review the injury history, current symptoms, and any previous imaging or specialist recommendations already available in the record. The patient’s questions about movement and personal care will guide discussion. Recommendations will be specific to the assessment, with clear wording about what is permitted and when the plan should be reconsidered.",
    "home": "At home, the patient has moved light items to reachable surfaces and accepts help with tasks that require overhead use. Dressing takes longer, especially when clothing has narrow openings. The patient would prefer a small number of reliable adjustments and wants household members to understand when assistance is useful and when independence remains possible.",
    "rehab": "Rehabilitation will address comfortable task performance within the confirmed restrictions. The therapist will observe dressing and reaching strategies before suggesting modifications or exercises. The patient’s response will guide progression, and any discrepancy between symptoms and the expected recovery pattern will be discussed with the clinician rather than managed by repeatedly increasing exercise demands.",
    "safety": "The patient has tried different resting positions but has not changed the prescribed support arrangement. The team will confirm equipment fit and explain any limitations relevant to ordinary tasks. The patient’s instructions for reporting unexpected changes will be reviewed, with particular attention to concerns that have been difficult to distinguish from familiar discomfort.",
    "followup": "Followup will assess changes in personal care, sleep disruption, and confidence using the arm. The clinician will revisit any unresolved questions about the injury and current precautions. The patient wants progress measured by practical activities regained, while recognizing that improved confidence and tissue recovery may not proceed at exactly the same pace."
  },
  {
    "key": "back_pain_tasks",
    "title": "Back discomfort during household tasks",
    "summary": "Back discomfort has made the patient reluctant to bend, lift light household objects, or remain seated through a meal. The symptoms fluctuate, and the patient cannot identify a single position that is always comfortable. Worry about causing further injury has become a substantial part of the difficulty with returning to ordinary responsibilities.",
    "observation": "During a simple object transfer, the patient held the trunk rigid and moved through several small steps. The task was completed, but effort appeared greater than the object’s size would suggest. The patient described discomfort afterward without a clear change in its location, and the observation was limited to a brief functional sequence.",
    "assessment": "The guarded strategy may reflect pain, fear, or an attempt to follow earlier advice whose meaning is now unclear. The cause of the back symptoms cannot be determined from movement behavior alone. Assessment will consider neurological and musculoskeletal findings, the course of symptoms, and the way beliefs about movement are affecting participation.",
    "plan": "The clinician will clarify the symptom history and review the advice the patient has received from different sources. A shared plan will identify meaningful activities for reassessment and explain how decisions about progression will be made. Further investigation or treatment changes will depend on the clinical findings and the course of the presentation.",
    "home": "The patient has delegated laundry and shopping tasks but continues to prepare simple meals with frequent position changes. Some help is welcome, though the patient feels frustrated when others take over without asking. Household planning will focus on matching assistance to current needs and maintaining a role in activities that remain manageable.",
    "rehab": "Rehabilitation will explore a range of task strategies rather than assuming there is one correct way to move. The therapist will observe comfort, confidence, and functional tolerance within the assessed clinical context. Goals will emphasize useful participation and reduced guarding, with the patient involved in identifying which changes feel practical and sustainable.",
    "safety": "The patient’s instructions for reporting new or substantially changed symptoms will be checked, and any unclear advice about lifting or rest will be reconciled. Current medication use and associated effects will be considered when discussing daily activities. The team will avoid treating the patient’s ability to complete one task as proof of unrestricted capacity.",
    "followup": "Followup will compare the patient’s experience across sitting, meal preparation, and other chosen activities. The clinician will assess whether symptoms are becoming more predictable and whether concern about movement has changed. A lack of progress will prompt reconsideration of the assessment and plan rather than an assumption that the patient has not tried hard enough."
  },
  {
    "key": "postviral_fatigue",
    "title": "Persistent fatigue after an illness",
    "summary": "The patient describes persistent fatigue after an illness that disrupted normal activity. Energy varies considerably, and the patient sometimes feels worse after completing a task that seemed manageable at the time. This unpredictability has made it difficult to plan household duties, maintain social contact, and judge whether apparent improvement will last.",
    "observation": "The patient followed the discussion but needed pauses when several topics were introduced together. Posture became more supported as the encounter continued, and the patient reported increasing mental effort. These observations are compatible with fatigue but are nonspecific, and no conclusion about physiological exercise tolerance was drawn from conversational endurance alone.",
    "assessment": "The fatigue may have several contributing factors, and the relationship between exertion and delayed symptoms needs clearer characterization. Sleep disruption, emotional strain, and the consequences of reduced activity may also be relevant without fully explaining the presentation. Assessment should preserve uncertainty and avoid interpreting variable performance as a lack of motivation or consistency.",
    "plan": "The clinician will review the illness course, associated symptoms, sleep, current medicines, and any investigations already completed. The patient will help identify a feasible way to describe activity and subsequent symptoms. Planning will prioritize an individualized understanding of tolerance, with any activity recommendations based on the wider clinical assessment rather than a standard progression.",
    "home": "At home, the patient alternates essential tasks with rest but finds it hard to stop when there is unfinished work. Household members sometimes interpret a productive period as evidence of full recovery. The patient would like help explaining the variability and arranging support that can adapt without requiring a new negotiation whenever symptoms fluctuate.",
    "rehab": "Rehabilitation will begin by understanding the patient’s pattern of activity and recovery, including any delayed worsening. The therapist will coordinate with the medical team before setting exertional goals. Practical work may focus on task simplification, prioritization, and maintaining valued roles, with progression reconsidered if the patient’s symptom response differs from what was anticipated.",
    "safety": "The patient reports occasionally attempting to complete several overdue tasks during a better period and then struggling afterward. The team will discuss this pattern within an individualized plan rather than offering a universal activity target. Instructions for reporting new symptoms or a clear change from the established pattern will also be reviewed.",
    "followup": "Followup will examine both immediate function and the patient’s experience after activity. The team will ask whether the plan has made daily demands more predictable and whether support arrangements are working. Continuing uncertainty about the cause or expected duration of fatigue will be acknowledged, with reassessment guided by changes in the clinical picture."
  },
  {
    "key": "diabetes_routines",
    "title": "Diabetes management amid changing routines",
    "summary": "The patient is managing diabetes while household routines have become less predictable. Meals, activity, and the timing of self care tasks vary, and the patient finds it difficult to connect recorded readings with what happened beforehand. The main concern is maintaining a workable routine without feeling that every fluctuation represents a personal failure.",
    "observation": "The patient brought a monitoring record with several gaps and explained that some entries were missed during busy periods. The device could be operated during the encounter, but the patient hesitated when interpreting the display. This suggested a need to review understanding and usability, without establishing whether the prescribed monitoring approach itself requires modification.",
    "assessment": "The current challenge appears to involve practical organization and interpretation of information as much as the measurements themselves. The available record is incomplete, so patterns should not be inferred with confidence. Assessment will consider treatment effects, access to supplies, meal variability, and the patient’s priorities before proposing changes to the established management plan.",
    "plan": "The clinician will review the existing diabetes plan, current medicines, monitoring instructions, and the circumstances surrounding concerning readings. Education will be linked to the patient’s actual routine and checked for understanding. Any adjustment to treatment or monitoring will be made by the responsible clinician using the full clinical picture and relevant available results.",
    "home": "At home, the patient shares meal preparation and sometimes depends on what others have available. Supplies are stored together, but reminders are inconsistent when the usual routine changes. The patient prefers a simple organizational approach that fits ordinary household life and does not require carrying several separate written records or devices between rooms.",
    "rehab": "Rehabilitation goals include maintaining comfortable participation in daily activity within the established clinical recommendations. The patient’s confidence, footwear needs, and any sensory or mobility concerns will be considered where relevant. Activity planning will be coordinated with the diabetes team so that functional goals and treatment instructions remain compatible and do not create unnecessary confusion.",
    "safety": "The patient’s understanding of the individualized response plan for problematic readings will be reviewed with examples from the existing record. Access to supplies and any barriers to contacting the clinical team will also be checked. The patient will be supported in reporting uncertainty rather than independently compensating with unreviewed changes to prescribed treatment.",
    "followup": "Followup will assess whether the agreed organizational changes reduced missed tasks and improved confidence interpreting the established plan. The clinician will review the quality of the available monitoring information before drawing conclusions about trends. The patient wants discussions to recognize real life constraints while still providing clear, individualized guidance about unresolved management concerns."
  },
  {
    "key": "foot_sensation",
    "title": "Altered foot sensation and walking confidence",
    "summary": "The patient reports altered sensation in both feet, describing a mixture of numbness and an unfamiliar feeling when placing weight through the soles. Walking on uneven surfaces has become less comfortable. The patient is unsure whether the symptoms have changed recently or whether closer attention has made a longstanding problem more noticeable.",
    "observation": "The patient watched foot placement during walking and used a hand support when turning. Footwear was easy to remove, but the patient had difficulty describing whether light contact felt the same across different areas. This informal observation was insufficient to characterize sensory loss and will be followed by the appropriate clinical examination.",
    "assessment": "Changes in sensation may contribute to the patient’s walking uncertainty, but strength, balance, vision, and footwear could also influence performance. The cause and extent of the sensory symptoms remain unconfirmed. The assessment will include their distribution and course while considering how reduced awareness of foot position affects everyday function and self care.",
    "plan": "The clinician will review the sensory history, relevant health conditions, current medicines, and the findings of a focused examination. The patient’s questions about walking and foot care will be addressed within that context. Referrals, equipment, or treatment changes will depend on the assessment rather than on the patient’s informal comparison of sensations alone.",
    "home": "At home, the patient usually wears familiar footwear but sometimes moves short distances without it. The patient can inspect some areas of the feet but has difficulty seeing the soles clearly. Available household support will be discussed without assuming that another person is always present or able to perform detailed care tasks.",
    "rehab": "Rehabilitation will assess walking on suitable surfaces and explore how the patient uses visual information and support. Footwear and any existing mobility equipment will be considered as part of the functional assessment. Practice will be individualized, and improvement will include confidence with ordinary routes as well as observed balance and movement quality.",
    "safety": "The team will clarify the patient’s existing instructions for skin checks and reporting changes that may be less noticeable because of altered sensation. Any proposed equipment must be assessed for fit and pressure. The patient’s concern about falling will also be addressed through a practical plan rather than reassurance based only on one observed walk.",
    "followup": "Followup will review symptom changes, foot condition, footwear tolerance, and the patient’s experience moving around the home. The clinician will compare formal findings where available and reconsider the differential if the pattern evolves. The patient would like a clearer explanation of what is known, what remains uncertain, and which functional concerns deserve priority."
  },
  {
    "key": "arm_weakness_recovery",
    "title": "Arm weakness during neurological recovery",
    "summary": "The patient is recovering from a neurological event and continues to have difficulty using one arm during dressing and meal preparation. There has been some return of movement, but coordination remains inconsistent. The patient feels encouraged by small improvements while also worrying that reliance on the stronger arm could become a permanent habit.",
    "observation": "During a tabletop task, the patient initiated movement with the weaker arm but needed extra time to position the hand. The shoulder elevated as effort increased, and the patient paused after several repetitions. The observation showed both emerging function and compensatory movement, though it did not establish the patient’s capacity across a full day.",
    "assessment": "Motor recovery appears to be occurring, but the interaction of weakness, coordination, sensation, and fatigue requires further assessment. The patient’s wish to use the arm more is a valuable goal, while frustration may lead to tasks that exceed current tolerance. Planning will balance participation with the specific findings and established neurological rehabilitation recommendations.",
    "plan": "The team will review the current rehabilitation plan and confirm any restrictions related to the arm or shoulder. Goals will be chosen around meaningful daily tasks and revisited as function changes. The patient will receive consistent explanations across disciplines so that advice about assistance, practice, and protection does not become contradictory or difficult to follow.",
    "home": "At home, the patient uses simplified clothing and has arranged frequently used objects where they can be reached comfortably. A household member helps with some two handed tasks but sometimes steps in before the patient has finished trying. The patient would like agreed cues that preserve the opportunity to participate without making assistance feel unavailable.",
    "rehab": "Rehabilitation will build from observed movement and the patient’s selected tasks. The therapist will consider positioning, coordination, fatigue, and the usefulness of the weaker arm during real activities. Practice will be adapted to quality and response, with attention to avoiding discomfort and maintaining motivation when progress is uneven or slower than the patient hoped.",
    "safety": "The patient’s support needs during transfers and personal care will be reviewed alongside arm function. Household members will be offered clear handling guidance through the treating team, particularly where the patient needs assistance. Any new symptoms or unexpected loss of function will be considered separately from the established recovery pattern and communicated appropriately.",
    "followup": "Followup will document specific activities the patient can now perform, the assistance still required, and any discomfort associated with practice. The team will ask whether household support is helping participation or unintentionally limiting it. Goals will be revised collaboratively, acknowledging that recovery may be uneven and that meaningful gains are not limited to isolated strength measurements."
  },
  {
    "key": "voice_fatigue",
    "title": "Voice fatigue affecting communication",
    "summary": "The patient describes a voice that becomes quieter and less comfortable after extended conversation. Telephone calls and speaking over background noise are particularly difficult. The patient has started declining some social contact because repeating statements feels tiring, and would like to understand whether the problem relates to voice use, recovery from illness, or another cause.",
    "observation": "The voice was audible in a quiet setting but became less consistent during a longer explanation. The patient occasionally cleared the throat and shortened responses after speaking for several minutes. These observations identify a communication difficulty but do not establish the condition of the vocal structures or the cause of the change in voice quality.",
    "assessment": "Several contributors may be relevant, including vocal effort, environmental demands, irritation, or an underlying laryngeal problem that requires assessment. The patient’s reduction in social participation is significant even though short conversations remain possible. The clinical review will avoid equating an audible voice in one setting with adequate communication across the patient’s usual activities.",
    "plan": "The clinician will review the onset, variability, associated symptoms, and any prior assessment of the voice. Referral or further examination will be considered according to the findings. Communication goals will be defined with the patient, and any voice exercises or treatment recommendations will follow an appropriate evaluation rather than informal attempts to force greater volume.",
    "home": "At home, background sound sometimes makes the patient repeat statements, especially when several people are speaking. Household members are willing to adjust the conversation environment but do not always notice when the voice is tiring. The patient prefers a simple signal for requesting a pause or a quieter exchange without drawing prolonged attention to the problem.",
    "rehab": "Rehabilitation will consider communication efficiency, conversational demands, and the patient’s response to different settings. A speech and language clinician may explore strategies within the assessed diagnosis and treatment plan. Goals will emphasize being understood with manageable effort and remaining socially involved, while changes in voice quality that require medical review will be referred appropriately.",
    "safety": "The patient has tried several informal remedies and is unsure whether any have helped. These will be reviewed as part of the clinical history rather than assumed to be harmless or effective. The team will clarify the existing instructions for reporting persistent or changing symptoms and ensure that communication difficulty does not obscure other clinical concerns.",
    "followup": "Followup will review the voice across ordinary conversations, telephone use, and noisier settings. The patient will be asked whether reduced effort or greater participation matters more than increased loudness alone. Findings from any further examination will guide the plan, and unresolved uncertainty about the source of the voice change will be discussed openly."
  },
  {
    "key": "swallowing_concern",
    "title": "Concern about swallowing during meals",
    "summary": "The patient reports occasional coughing during meals and a feeling that eating now requires more concentration. The pattern is inconsistent, and the patient cannot identify one food or drink that reliably causes difficulty. Concern about embarrassment has led to slower meals and fewer shared eating occasions, despite the patient’s wish to maintain familiar routines.",
    "observation": "The patient described the concern clearly and managed saliva during conversation without an obvious problem. No food or fluid trial was performed during this observation because an appropriate swallowing assessment was still required. The absence of visible difficulty while talking therefore does not resolve the patient’s report of symptoms during actual meals.",
    "assessment": "The history warrants a swallowing assessment, but the mechanism and severity of any impairment remain uncertain. Coughing may have several explanations, and the patient’s own food restrictions could affect intake or enjoyment. Assessment will consider the whole mealtime experience, relevant medical history, and the distinction between perceived effort and observed swallowing performance.",
    "plan": "The clinician will arrange or complete the indicated swallowing evaluation and review the patient’s current eating instructions. The patient will be asked about symptom timing, associated sensations, and changes in intake. Any recommendations about food texture, fluid consistency, posture, or exercises will be individualized after assessment and communicated consistently to those involved in meals.",
    "home": "At home, the patient tends to eat separately when worried about coughing, which reduces opportunities for shared conversation. A household member has suggested several food changes, but these have not been formally reviewed. The patient would like guidance that preserves enjoyment and social participation while giving everyone a clear understanding of the assessed needs.",
    "rehab": "Rehabilitation will follow the findings of the swallowing assessment and focus on the patient’s practical mealtime goals. The treating clinician will determine whether specific strategies or further investigation are appropriate. Participation may include discussing meal setup and communication of needs, without asking the patient to practice unassessed swallowing techniques or independently alter the prescribed diet.",
    "safety": "The team will reconcile existing instructions and clarify what the patient should do if swallowing symptoms change or a meal becomes difficult. Household members will be asked to avoid introducing competing recommendations. The patient’s ability to understand and follow any individualized plan will be checked, including practical barriers related to food preparation and available support.",
    "followup": "Followup will review assessment findings, the patient’s experience during meals, and whether any recommended changes are acceptable and feasible. Intake and social participation will be considered alongside symptom reports. If the initial plan does not explain or improve the difficulty, the team will revisit the assessment rather than assuming inconsistent symptoms indicate no meaningful problem."
  },
  {
    "key": "tremor_tasks",
    "title": "Tremor interfering with fine motor tasks",
    "summary": "The patient reports a hand tremor that becomes noticeable when carrying a cup, writing a short note, or using utensils. The severity varies, and concern about being observed sometimes makes tasks feel harder. The patient wants to maintain independence and understand whether practical adaptations could help while the medical assessment continues.",
    "observation": "A tremor was visible during a reaching task and less apparent while the hand rested on the table. The patient steadied an object with the other hand and completed the task slowly. The brief observation did not establish the tremor’s full pattern, its underlying cause, or how it changes across different activities and periods of fatigue.",
    "assessment": "The movement affects precision and confidence, but the contribution of tremor type, medication effects, fatigue, and situational stress remains uncertain. The patient’s concern about public tasks may amplify disability without explaining the movement itself. Clinical assessment will therefore address both the observed motor features and the practical consequences for eating, writing, and carrying objects.",
    "plan": "The clinician will review symptom onset, distribution, variability, associated features, and the current medication list. The patient’s most troublesome tasks will guide functional assessment. Treatment decisions and any specialist referral will follow the clinical findings, while selected adaptations can be evaluated for usefulness without presuming that one device or strategy will address every activity.",
    "home": "At home, the patient chooses containers that feel easier to hold and often sits for tasks requiring precision. Household members sometimes complete the task instead of allowing extra time. The patient appreciates help but wants the opportunity to decide when assistance is needed, particularly for meals and other activities closely tied to independence.",
    "rehab": "Rehabilitation will examine task setup, grip, support surfaces, and the effort required by different objects. The therapist will trial practical adaptations with the patient and judge them by actual performance rather than appearance. Goals will include reducing spills or repeated attempts and supporting participation, with recognition that tremor severity may vary between sessions.",
    "safety": "The patient reports occasional spills but no recent injury associated with the tremor. Tasks involving hot liquids, sharp objects, or fragile items will be considered individually in the functional assessment. The team will also clarify how to report a new pattern of movement or associated symptoms, rather than assuming every change belongs to the existing tremor.",
    "followup": "Followup will compare performance on the patient’s chosen tasks and ask which adaptations were useful outside the assessment setting. The clinician will review any changes in the movement pattern or associated symptoms. The patient wants a plan that remains practical on more difficult days and does not define progress solely as the disappearance of tremor."
  },
  {
    "key": "fluid_related_symptoms",
    "title": "Fluctuating swelling and activity tolerance",
    "summary": "The patient reports fluctuating lower limb swelling and fatigue within an established cardiac care plan. Shoes sometimes feel tighter, and ordinary walking can require more pauses than expected. The patient has been recording symptoms inconsistently and is uncertain how much daily variation is meaningful, particularly when household routines and meal patterns also change.",
    "observation": "Mild swelling was visible above the footwear, and the patient paused after walking into the room. The patient remained engaged in conversation and described feeling comfortable while seated. These observations provide context but cannot establish fluid status or cardiac stability without the relevant clinical examination, measurements, and review of recent trends.",
    "assessment": "The reported variability could relate to fluid balance, activity demands, treatment effects, or other contributors that need clinical review. An isolated observation does not establish whether the underlying condition has changed. The patient’s uncertainty about the action plan is itself important because it may delay reporting concerns or lead to unnecessary worry about ordinary variation.",
    "plan": "The clinician will review the current cardiac plan, recent symptoms, available measurements, and medication use. The patient will be asked to explain how the individualized instructions are interpreted at home. Any adjustments to medicines, intake guidance, or monitoring will come from the responsible clinician after assessment rather than from general advice about swelling.",
    "home": "At home, the patient manages most personal care but finds footwear and prolonged standing more difficult when swelling increases. Shared meals make the existing dietary instructions harder to apply consistently. The patient would like practical clarification that respects household preferences and avoids placing responsibility for every symptom fluctuation on a single meal or activity.",
    "rehab": "Rehabilitation will focus on maintaining useful activity within the cardiac team’s recommendations and the patient’s assessed tolerance. The therapist will observe functional tasks and recovery, and coordinate concerns about changing symptoms with the medical team. Goals will account for variability, with progress measured through reliable participation as well as distance or task duration.",
    "safety": "The patient’s individualized instructions for concerning changes will be reviewed, including whom to contact and what information is useful to report. Medication reconciliation will address any confusion about scheduled versus conditional directions. The team will avoid encouraging independent dose changes or interpreting a comfortable seated appearance as confirmation that exertional symptoms need no further review.",
    "followup": "Followup will compare symptom trends, available measurements, and daily function with the patient’s usual pattern. The team will assess whether the clarified plan is easier to use and whether recording demands are manageable. Persistent uncertainty or worsening function will prompt clinical reassessment rather than repeated reassurance based on incomplete information about the underlying condition."
  },
  {
    "key": "urinary_urgency",
    "title": "Urinary urgency affecting daily participation",
    "summary": "The patient reports urinary urgency that has made outings and household tasks feel less predictable. There have been occasional leakage episodes, and the patient now plans activities around bathroom access. Embarrassment has limited discussion with others, but the patient wants an assessment that considers practical function and dignity as well as the urinary symptoms.",
    "observation": "The patient was comfortable during the encounter and could describe the sequence from first noticing urgency to reaching the bathroom. Clothing fasteners and a slow chair transfer sometimes add difficulty. No conclusion about the cause of the urinary symptoms was drawn from this functional account, and the relevant medical history still needs fuller review.",
    "assessment": "The impact may reflect an interaction between bladder symptoms, mobility, clothing, and environmental access. The underlying urinary mechanism has not been established, and the patient’s coping strategies may not address all contributors. Assessment will consider associated symptoms and treatment history while recognizing that embarrassment can lead to underreporting of both frequency and functional disruption.",
    "plan": "The clinician will review the urinary history, current medicines, relevant examination findings, and any previous testing. The patient will be offered a manageable way to describe the pattern if additional information is needed. Treatment and any targeted exercises will follow the assessment, with preferences and practical barriers included in decisions about the next steps.",
    "home": "At home, the patient uses the nearest bathroom but sometimes encounters a closed door or items left in the passage. A household member can help improve access without needing details of every symptom. The patient prefers discreet adjustments and wants to remain involved in social plans rather than routinely staying home to avoid embarrassment.",
    "rehab": "Rehabilitation will examine transfer efficiency, clothing management, and access to the bathroom as functional components of the problem. The therapist will coordinate with the medical team rather than assume that mobility adjustments resolve the urinary cause. Goals will include greater confidence and reduced disruption, with any specialized pelvic health work guided by an appropriate assessment.",
    "safety": "The patient sometimes rushes when urgency occurs and recognizes that this makes balance less steady. The team will review the route and current support needs, while clarifying the clinical instructions for reporting new or changing urinary symptoms. Any changes to fluid intake or medicines will be discussed with the responsible clinician rather than improvised independently.",
    "followup": "Followup will review the urinary pattern, leakage concerns, confidence outside the home, and the usefulness of environmental changes. The patient will be asked whether the plan protects privacy and remains practical. If symptoms persist, the clinical assessment will be revisited rather than assuming that better access or greater effort should have solved the difficulty."
  },
  {
    "key": "stoma_adjustment",
    "title": "Adjustment to a new stoma routine",
    "summary": "The patient is learning to manage a stoma and describes alternating confidence and frustration with the care routine. Emptying the appliance feels more familiar, but changes remain slow and emotionally demanding. The patient worries about leakage during ordinary activities and wants support that addresses both practical skills and the effect on personal confidence.",
    "observation": "The patient could identify the main supplies and explain several steps in the established routine. Hesitation increased when discussing the fit of the appliance and the condition of surrounding skin. No appliance change or direct skin examination occurred during this conversation, so the patient’s reported concerns require assessment by the appropriate clinician.",
    "assessment": "The main difficulties may involve technique, equipment fit, skin comfort, and adjustment to a changed body routine. Their relative contribution is not yet clear. The patient’s fear of leakage is affecting participation even when no leakage occurs, making confidence and emotional adjustment relevant alongside the technical aspects of care and clinical review.",
    "plan": "The clinician will review the current stoma care instructions and arrange direct assessment of the concerns about fit and skin. Teaching will build on the steps the patient already understands. Product changes or additional treatments will be considered by the responsible team, with the patient involved in choosing a routine that is feasible at home.",
    "home": "The patient has organized supplies in one place but lacks a comfortable surface for longer care tasks. A household member can help gather items, although the patient prefers privacy during personal care. Planning will respect that preference while exploring how practical support can remain available without turning every appliance change into a shared task.",
    "rehab": "Rehabilitation will consider posture, reaching, dexterity, and the patient’s confidence completing the care sequence. The therapist and stoma clinician will coordinate recommendations so that functional adaptations support the established technique. Goals may include completing selected steps independently and returning to valued activities, with attention to fatigue and the emotional effort associated with practice.",
    "safety": "The team will clarify the patient’s individualized instructions for reporting changes in output, discomfort, leakage, or surrounding skin. Supply availability and the ability to obtain timely help will also be reviewed. The patient will be encouraged to seek clarification before making unreviewed product substitutions or treating unfamiliar symptoms as an inevitable part of adjustment.",
    "followup": "Followup will include direct review of the appliance concerns and discussion of how the routine works during ordinary household activity. The patient wants to feel prepared for minor practical problems without expecting perfect performance immediately. Progress will therefore include confidence, comfort, and participation as well as the number of care steps completed without assistance."
  },
  {
    "key": "itch_skin_routine",
    "title": "Persistent itching disrupting comfort and sleep",
    "summary": "The patient describes persistent itching that interferes with sleep and makes it difficult to concentrate on quiet activities. Symptoms vary across the day, and scratching sometimes occurs before the patient becomes aware of it. Several skin products have been tried, leaving the patient uncertain about which are part of the current clinical plan.",
    "observation": "Visible areas of skin appeared dry, with scattered marks consistent with the patient’s report of scratching. The patient rubbed one forearm repeatedly while speaking but stopped when attention shifted to a task. This limited observation does not establish the cause or extent of the skin condition and does not substitute for a full examination.",
    "assessment": "The itching may reflect the known skin condition, irritation from products, or another contributor that requires assessment. Sleep disruption could increase distress and reduce coping capacity without explaining the skin symptoms themselves. The clinician will consider the distribution and course of the problem while avoiding assumptions based solely on the appearance of the visible areas.",
    "plan": "The plan is to review the complete product and medication history and examine the affected skin. The patient’s experience of comfort, sleep, and treatment burden will inform decisions. Any changes to prescribed products or their use will follow clinical review, with clear instructions that distinguish the agreed regimen from items tried independently.",
    "home": "At home, the patient keeps several products beside the bed and sometimes applies whichever is easiest to reach. Clothing and bedding choices have also changed in response to discomfort. The patient wants a simpler routine and would welcome help organizing supplies, provided that useful items are not discarded before the clinician has reviewed them.",
    "rehab": "Rehabilitation needs relate mainly to preserving sleep routines, comfortable movement, and participation in tasks interrupted by itching. The team will consider whether skin discomfort affects clothing tolerance or equipment use. Practical adaptations will be coordinated with the skin care plan so that functional advice does not inadvertently introduce materials or products that worsen irritation.",
    "safety": "The patient is uncertain how to distinguish ordinary fluctuation from a skin change that warrants additional review. The clinician will clarify the relevant individualized instructions and examine any areas of concern. Product duplication, application difficulties, and the effect of symptoms on attention will be assessed before assuming the current regimen is being used as intended.",
    "followup": "Followup will review the skin findings, symptom burden, sleep, and the practical use of the agreed products. The patient will be asked whether the simpler routine is understandable and acceptable. If itching remains disproportionate to the visible findings or changes in character, the clinician will reconsider the assessment rather than simply repeating earlier advice."
  },
  {
    "key": "visual_recovery",
    "title": "Visual adjustment during recovery",
    "summary": "The patient is adjusting to altered vision during recovery from an eye procedure. Reading small print and judging edges feel less reliable than expected, especially when lighting changes. The patient is following an existing care plan but remains unsure which activities are suitable and wants clearer guidance about returning to familiar household routines.",
    "observation": "The patient held written material farther away and turned it toward a stronger light source. Movement through the room was cautious near changes in floor contrast. These behaviors suggest a functional effect of the visual symptoms, but they do not establish visual acuity, depth perception, or the status of healing after the procedure.",
    "assessment": "The functional difficulty could reflect expected visual adjustment, the underlying eye condition, or another issue that needs examination. The current information is insufficient to distinguish these possibilities. The patient’s uncertainty about activity restrictions also contributes to avoidance, so the team will review both the clinical course and the clarity of the postoperative instructions.",
    "plan": "The clinician will confirm the procedure specific plan and arrange review of the patient’s reported visual difficulties. Current eye treatments and their administration will be reconciled. Advice about reading, household tasks, and other activities will be individualized according to the examination and existing restrictions, with any unresolved questions directed to the responsible eye care team.",
    "home": "At home, the patient has increased lighting near reading materials and asks for help checking small labels. Several frequently used objects blend into the surfaces beneath them. Household adjustments will be discussed with the patient so that improved visibility does not come at the cost of moving familiar belongings to unexpected or difficult locations.",
    "rehab": "Rehabilitation will examine how the patient uses available visual information during practical tasks. The therapist may consider lighting, contrast, organization, and the effort required to read or locate items. Any adaptations will remain compatible with the eye care plan, and progress will be judged through reliable function rather than assumptions about healing based on confidence alone.",
    "safety": "The team will review the patient’s individualized instructions for new or changing visual symptoms and confirm that treatment labels can be read accurately. Tasks requiring reliable depth judgment or sustained visual attention will be assessed separately. The patient’s ability to navigate one familiar room will not be treated as evidence that every activity is equally manageable.",
    "followup": "Followup will compare the patient’s experience with reading, locating objects, and moving through familiar spaces. The eye care findings will guide decisions about restrictions and further adaptation. The patient wants a clear distinction between expected adjustment and symptoms requiring review, with remaining uncertainty explained rather than dismissed because some tasks have become easier."
  },
  {
    "key": "hearing_participation",
    "title": "Hearing difficulty during shared conversations",
    "summary": "The patient reports increasing difficulty following conversations when several people speak or when background sound is present. Individual exchanges in quiet settings remain easier. The patient sometimes agrees without fully understanding, then feels embarrassed asking for repetition, and would like support that improves participation without making every interaction revolve around the hearing difficulty.",
    "observation": "During conversation, the patient responded more consistently when the speaker faced them and spoke one thought at a time. Several requests for repetition occurred when the speaker looked away. This pattern suggests a communication access problem but does not establish the degree or type of hearing impairment or whether current hearing equipment is functioning optimally.",
    "assessment": "The patient’s difficulty appears sensitive to the communication environment, but hearing status, equipment performance, attention, and fatigue may all contribute. Misunderstandings can affect both social participation and the accuracy of clinical information exchange. Assessment will therefore consider the patient’s hearing needs alongside the way conversations are organized, rather than attributing every missed detail to inattention.",
    "plan": "The clinician will review the hearing history, current equipment, and any prior assessment available in the record. Communication preferences will be documented in practical terms and checked with the patient. Further hearing evaluation or equipment review will be arranged as indicated, while important clinical information will be confirmed using a format the patient can access.",
    "home": "At home, shared conversations often take place while other activities continue in the background. The patient has begun withdrawing from these exchanges because following several voices requires considerable effort. Household members are willing to help but need specific feedback about what makes a difference, especially when familiar conversational habits return without anyone noticing.",
    "rehab": "Rehabilitation will focus on communication strategies and participation in the situations the patient values most. The team will explore environmental changes, use of existing equipment, and ways to request clarification comfortably. Goals will include more reliable understanding and reduced listening effort, with the patient deciding which approaches feel respectful and sustainable in everyday relationships.",
    "safety": "The patient’s understanding of important care instructions will be checked without relying on a nod or brief agreement alone. The team will also consider whether alarms, calls for attention, or spoken warnings are accessible in the patient’s usual environment. Any equipment concerns will be reviewed by the appropriate professional rather than addressed through untested adjustments.",
    "followup": "Followup will ask whether conversations are easier in the settings that previously led to withdrawal or misunderstanding. The patient will be invited to describe both helpful changes and awkward experiences. Hearing assessment findings and equipment review will inform the next steps, while progress will include confidence asking for clarification as well as improved access to speech."
  },
  {
    "key": "sleep_continuity",
    "title": "Fragmented sleep with daytime consequences",
    "summary": "The patient describes fragmented sleep and increasing concern about whether enough rest will be obtained. Time in bed has expanded, but the patient does not feel more refreshed. Daytime concentration and patience are affected, and the patient has begun planning activities around anticipated tiredness rather than the demands or enjoyment of the activity itself.",
    "observation": "The patient remained attentive during the encounter but occasionally lost the thread while describing a complicated routine. There was no direct observation of sleep, and the patient’s estimates of waking periods were uncertain. The account demonstrates distress and functional impact, but it cannot by itself establish a specific sleep disorder or its causes.",
    "assessment": "The sleep difficulty may involve behavioral patterns, physical symptoms, emotional strain, medication effects, or another sleep related condition. Worry about sleep appears to increase the burden but should not be assumed to explain the entire presentation. Assessment will consider the broader history and distinguish perceived sleep quality from any available objective or collateral information.",
    "plan": "The clinician will review the sleep pattern, daytime effects, associated symptoms, and all medicines or products used for rest. The patient will be offered a manageable way to record relevant details if useful. Treatment options and any structured behavioral approach will follow assessment and be adapted to the patient’s circumstances and preferences.",
    "home": "At home, the patient sometimes moves between rooms during wakeful periods and feels pressure to avoid disturbing others. Household routines and background sound vary, which makes consistency difficult. The patient wants a plan that acknowledges shared living arrangements and does not require every other person to organize their routine around the patient’s sleep.",
    "rehab": "Rehabilitation will consider the effect of tiredness on pacing, attention, and participation in valued activities. The team will help identify realistic functional goals while the sleep concern is assessed. Activity planning will avoid assuming that fatigue always improves through greater effort, and progress will include steadier participation rather than a single idealized sleep duration.",
    "safety": "The patient’s use of sedating products and any associated daytime effects will be reviewed by the clinician. Tasks requiring sustained alertness will be discussed in relation to the patient’s actual symptoms. The team will clarify how to report significant changes and will avoid suggesting independent medication combinations or changes based solely on a difficult night.",
    "followup": "Followup will review sleep quality, daytime function, distress, and the feasibility of the agreed plan. The patient will be asked whether tracking is helpful or adds to preoccupation with sleep. If the initial approach has little effect, the assessment will be reconsidered, including possible contributing conditions and the burden of the treatment itself."
  },
  {
    "key": "procedure_anxiety",
    "title": "Anxiety around an upcoming clinical procedure",
    "summary": "The patient reports anxiety about a planned procedure and difficulty separating known information from imagined outcomes. Sleep and concentration have been affected, and repeated online reading has left the patient more confused. The patient wants clear explanations and a sense of involvement, while acknowledging uncertainty about which questions would be most useful to ask.",
    "observation": "The patient spoke rapidly when describing possible complications and became more settled when the discussion returned to the actual care plan. Several questions concerned details that had not yet been explained by the procedural team. The encounter showed distress and a need for clarification but did not establish the severity of any underlying anxiety condition.",
    "assessment": "The anxiety appears closely linked to uncertainty and the perceived loss of control surrounding the procedure. Prior experiences may also be relevant, though they have not been explored fully. The team will avoid interpreting repeated questions as resistance and will consider how information delivery, support, and the patient’s preferences can influence the experience.",
    "plan": "The clinician will help organize the patient’s questions and identify which require answers from the procedural team. Existing preparation instructions will be reviewed for clarity and consistency. Emotional support options will be discussed according to the patient’s wishes, with no assumption that reassurance alone will resolve concerns that depend on specific clinical information.",
    "home": "At home, the patient alternates between seeking reassurance from others and avoiding discussion altogether. Household members offer conflicting opinions, which sometimes increases uncertainty. The patient would like a more contained way to prepare and prefers that practical support be organized around the confirmed instructions rather than advice based on other people’s different experiences.",
    "rehab": "Rehabilitation considerations involve preserving ordinary routines and participation while preparation is underway. The patient identified manageable activities that provide structure without adding to physical or emotional strain. The team will support these within the clinical plan, while recognizing that maintaining activity is not a substitute for answering procedural questions or addressing substantial anxiety.",
    "safety": "The patient’s understanding of preparation instructions will be confirmed directly, especially where worry has made information hard to retain. Medication and fasting directions, if relevant, will be clarified only through the responsible team. The patient will have a clear route for reporting new concerns rather than making independent changes based on informal or conflicting advice.",
    "followup": "Followup will check whether the patient’s main questions were answered and whether the preparation plan feels understandable. Distress, sleep, and support needs will be revisited as the procedure approaches. The team will document unresolved uncertainty and avoid promising an outcome that cannot be guaranteed, while helping the patient participate meaningfully in the planned care."
  },
  {
    "key": "low_mood_routine",
    "title": "Low mood with reduced daily participation",
    "summary": "The patient describes low mood, reduced interest in familiar activities, and difficulty beginning ordinary tasks. Responsibilities have accumulated, which adds to feelings of discouragement. The patient wants to regain a more predictable routine but finds broad encouragement unhelpful, and would prefer a discussion that recognizes both emotional symptoms and the practical barriers to participation.",
    "observation": "The patient engaged throughout the conversation, although responses were sometimes brief and required additional time. Affect appeared subdued, and the patient described several unfinished tasks without identifying a clear starting point. These observations support the reported functional difficulty but do not establish the diagnosis, duration, or full severity of the mood symptoms.",
    "assessment": "The presentation may reflect a depressive condition, stress related difficulty, physical contributors, or a combination that requires further assessment. Reduced activity and accumulating demands may be reinforcing distress. The clinician will explore the broader symptom pattern and available supports while avoiding assumptions that the patient’s difficulty starting tasks represents unwillingness or a simple organizational problem.",
    "plan": "The plan is to complete a fuller mood assessment, review relevant health and treatment factors, and identify the patient’s immediate priorities. Options for psychological and other clinical support will be discussed collaboratively. The patient will help choose a manageable initial focus, with expectations that allow for variability and do not turn every difficult period into perceived failure.",
    "home": "At home, the patient has maintained some essential tasks but has withdrawn from activities that previously provided structure or pleasure. Support is available, though the patient worries about being judged for unfinished work. Discussion will consider how others can offer practical help without taking over decisions or adding pressure to appear better than the patient feels.",
    "rehab": "Rehabilitation will focus on meaningful participation and the obstacles that interfere with beginning or sustaining an activity. The team will help the patient identify tasks with personal relevance and assess the effort they require. Progress will include engagement and reduced burden, with the plan coordinated with mental health care rather than treated as a replacement for it.",
    "safety": "The clinician will assess safety and support needs directly as part of the broader mood review, without inferring risk from appearance alone. The patient’s ability to seek help when distress increases will be discussed in a respectful, individualized way. Medication effects and any difficulties carrying out essential self care will also be considered.",
    "followup": "Followup will review mood, daily function, support, and the usefulness of the initial plan. The patient will be invited to explain what felt achievable and what remained too demanding. Lack of improvement will prompt reassessment of the clinical picture and treatment approach rather than an assumption that the patient failed to follow simple advice."
  },
  {
    "key": "bereavement_adjustment",
    "title": "Daily routines during bereavement",
    "summary": "The patient is adjusting to bereavement and describes a household routine that feels unfamiliar without the person who previously shared many daily tasks. Appetite, sleep, and concentration vary. The patient remains interested in some activities but feels uncertain about participating, and wants space to discuss practical difficulties without having every emotional response treated as abnormal.",
    "observation": "The patient became tearful when describing changes in the home and was also able to recall moments of comfort and connection. Attention was sustained when the conversation moved to practical matters. This variability is meaningful context, but the encounter does not establish whether the patient’s experience requires a particular diagnosis or level of intervention.",
    "assessment": "The difficulties occur in the context of loss and changing responsibilities, with both emotional and practical consequences. The clinician will consider the course, severity, and functional impact rather than assuming a uniform pattern of grief. Existing health conditions and support needs may also influence adaptation and should not be overlooked because bereavement provides an understandable context.",
    "plan": "The plan is to review immediate concerns about sleep, intake, health care routines, and available support while allowing the patient to set the pace of emotional discussion. Relevant community or clinical resources may be explored according to preference. Any further assessment will be guided by the patient’s functioning, wishes, and the broader clinical picture.",
    "home": "At home, several tasks were previously shared, and the patient is still learning how to organize them independently. Some belongings provide comfort while others make routine activity difficult. The patient does not want others to make rapid changes to the household and would prefer practical assistance that respects personal decisions about space and possessions.",
    "rehab": "Rehabilitation will support reestablishing useful routines where physical or cognitive demands have become difficult. The patient identified selected household activities that feel important and others that can be deferred or shared. Goals will remain flexible and personally meaningful, without treating a return to a previous level of activity as the only acceptable sign of adaptation.",
    "safety": "The clinician will assess whether essential needs are being met and discuss any periods when distress makes self care difficult. Support and contact arrangements will be reviewed without presuming that being alone is itself unsafe. Any concerning changes in mood, function, or health will be assessed directly rather than automatically attributed to grief.",
    "followup": "Followup will revisit practical burdens, health routines, and the kind of support the patient finds helpful. Emotional experiences will be discussed without requiring a particular pace or sequence of recovery. The team will remain attentive to emerging clinical concerns while recognizing that periods of sadness and changes in participation can coexist with meaningful adaptation."
  },
  {
    "key": "memory_concerns",
    "title": "Memory concerns affecting everyday organization",
    "summary": "The patient reports forgetting the purpose of a trip to another room and occasionally losing track of a household task. The changes are worrying, although the patient continues to manage many familiar routines. Fatigue and interruptions seem relevant, but the patient is unsure whether they explain the pattern or simply make an underlying problem more noticeable.",
    "observation": "The patient recalled the main purpose of the encounter and followed a short sequence of questions. More difficulty occurred when several details were introduced together, and the patient requested a recap. This brief interaction does not establish cognitive status, and familiar conversation may not reveal the difficulties experienced during more demanding everyday tasks.",
    "assessment": "The reported lapses may involve attention, memory, sleep, emotional factors, medication effects, or another cognitive process that needs assessment. Preserved performance in some familiar activities does not exclude a meaningful problem elsewhere. The clinician will consider change from the patient’s usual function and seek appropriate supporting information with the patient’s knowledge and agreement.",
    "plan": "The clinician will review the course of the concerns, associated symptoms, relevant medicines, and examples from daily life. Formal assessment or further investigation will be considered as indicated. The patient’s goals will guide practical support, and explanations will distinguish observed difficulties from interpretations that remain uncertain rather than assigning a premature label.",
    "home": "At home, the patient uses written reminders but keeps them in several places, making some difficult to find when needed. Interruptions are common during meal preparation and other multistep tasks. The patient would like a simpler system that supports independence and prefers that household members offer reminders without testing memory or correcting every minor lapse.",
    "rehab": "Rehabilitation will examine specific tasks to identify where information is lost or sequencing becomes difficult. The therapist will trial organizational supports with the patient and assess whether they are actually used in daily life. Goals will focus on reliable task completion and reduced effort, with strategies adjusted if they become more burdensome than the original problem.",
    "safety": "The team will review any effect on medicines, cooking, appointments, and other tasks with meaningful consequences, using concrete examples rather than assumptions about overall ability. Support will be proportionate to the assessed need. The patient’s preferences and autonomy will remain central while uncertainties are investigated and any immediate practical difficulties are addressed.",
    "followup": "Followup will compare the patient’s experience with selected tasks and review whether the organizational supports improved reliability. The clinician will consider information from further assessment and any changes in the pattern of concern. The patient wants an honest explanation of what the findings show, what they do not establish, and how the next steps are chosen."
  },
  {
    "key": "concentration_recovery",
    "title": "Concentration difficulties after a head injury",
    "summary": "The patient is recovering from a head injury and reports headaches, mental fatigue, and difficulty sustaining attention during reading or screen use. Symptoms vary, and the patient is uncertain how to judge readiness for usual responsibilities. Concern about falling behind has led to alternating periods of intense effort and withdrawal from activity.",
    "observation": "During the encounter, the patient followed short explanations but requested repetition after a longer discussion. The patient described increasing head discomfort and preferred a pause before continuing. These observations are relevant to tolerance in the moment but do not establish the full extent of cognitive impairment or the expected course of recovery.",
    "assessment": "The symptoms may reflect the effects of the injury, sleep disruption, emotional strain, or other contributors that require clinical review. The relationship between task demands and symptom change needs clarification. The team will avoid using a single successful conversation or brief task as evidence that the patient can sustain usual demands throughout an entire day.",
    "plan": "The clinician will review the injury course, associated symptoms, current restrictions, and prior assessment findings. The patient’s usual responsibilities will help define meaningful recovery goals. Recommendations about cognitive and physical activity will follow the individualized clinical plan, with consistent wording about how progress is assessed and how new concerns should be reported.",
    "home": "At home, the patient has reduced background distractions but still receives frequent messages and requests for help. Others sometimes expect normal performance because the injury is not outwardly visible. The patient would like a practical explanation of current limits and a support arrangement that can change as tolerance improves without requiring constant justification.",
    "rehab": "Rehabilitation will assess selected cognitive and physical tasks within the established recovery recommendations. The therapist will consider accuracy, effort, symptom response, and recovery after the task rather than duration alone. Progression will be individualized, and the team will coordinate advice so the patient is not given conflicting expectations about rest, practice, and participation.",
    "safety": "The patient’s understanding of the existing instructions for new or worsening symptoms will be checked. Activities requiring rapid decisions, reliable balance, or sustained attention will be reviewed separately before return. The team will also consider whether fatigue or treatment effects interfere with judgment, without assuming that the patient’s desire to resume responsibilities establishes readiness.",
    "followup": "Followup will review symptoms, task tolerance, and the practical impact of changes made at home. The patient will be asked about both immediate and later effects of demanding activity. If progress stalls or the pattern changes, the clinical assessment and recovery plan will be revisited rather than applying an inflexible expectation about recovery speed."
  },
  {
    "key": "kidney_care_routine",
    "title": "Kidney care instructions within daily routines",
    "summary": "The patient is following a kidney care plan and finds it difficult to reconcile different instructions about food, fluids, medicines, and routine monitoring. Some advice was received at different stages of care and may no longer apply. The patient wants a clear current plan that can be understood without interpreting several separate handouts.",
    "observation": "The patient brought a list of questions and could describe the purpose of some medicines but was less certain about the dietary instructions. Several statements were remembered in absolute terms without the original context. No conclusion about kidney function or nutritional status can be drawn from this discussion, and relevant clinical information requires review.",
    "assessment": "The main concern is the usability and consistency of the care plan, rather than an established failure to follow it. Conflicting or outdated information may be contributing to unnecessary restriction. The clinician will consider current kidney status, other health needs, and the patient’s practical circumstances before confirming which instructions remain appropriate.",
    "plan": "The responsible team will reconcile the current medication, dietary, fluid, and monitoring instructions and provide a coordinated explanation. The patient’s questions will help identify where wording is ambiguous or assumptions differ between clinicians. Any treatment changes will depend on the relevant assessment and results, with the updated plan checked for understanding and practical feasibility.",
    "home": "At home, shared meals and limited preparation time make separate dietary routines difficult. The patient has removed several familiar foods because of uncertainty, which has reduced enjoyment and variety. Household support may help with planning, but the patient wants changes based on current individualized advice rather than an expanding set of restrictions from informal sources.",
    "rehab": "Rehabilitation will consider fatigue, mobility, and participation in daily tasks within the broader kidney care plan. The therapist will coordinate with the medical team when symptoms or treatment schedules affect tolerance. Functional goals will remain specific to the patient, and changes in activity capacity will not be attributed automatically to the kidney condition without assessment.",
    "safety": "The team will clarify the individualized instructions for reporting symptoms and handling uncertainty about treatment. All medicines and supplements will be included in reconciliation, without assuming that a product is suitable because it is readily available. The patient will receive a clear point of contact for questions that cannot be resolved from the written plan.",
    "followup": "Followup will review whether the coordinated instructions reduced confusion and whether food choices, treatment tasks, and monitoring are manageable. Relevant clinical results will guide further decisions. The patient’s understanding will be checked through discussion of ordinary situations, while unresolved inconsistencies will be addressed directly rather than left for the patient to reconcile independently."
  },
  {
    "key": "treatment_fatigue",
    "title": "Fatigue and appetite changes during treatment",
    "summary": "The patient is receiving a course of treatment and reports fluctuating fatigue, altered appetite, and reduced tolerance for ordinary chores. Better periods can be difficult to predict, making plans feel uncertain. The patient wants to retain a role in household life while also understanding which symptoms should be discussed with the treating team.",
    "observation": "The patient remained engaged but chose a supported sitting position and paused between longer responses. The patient described foods seeming less appealing without identifying a consistent taste or texture problem. These observations provide a limited view of symptom burden and do not establish hydration, nutritional status, or the cause of the reported fatigue.",
    "assessment": "The symptoms may relate to treatment, the underlying condition, disrupted routines, or other contributors requiring assessment. Their variability does not make the functional effect less important. The team will consider the timing and associated features while avoiding an assumption that every new symptom is an expected treatment effect that needs no further investigation.",
    "plan": "The clinician will review the current treatment plan, symptom course, intake concerns, and any existing supportive care instructions. The patient’s priorities will guide discussion of manageable household roles. Changes to medicines, nutritional support, or treatment scheduling will be considered by the responsible team using the full clinical picture and the patient’s preferences.",
    "home": "At home, meals are often prepared by others, and the patient feels uncomfortable declining food after effort has been spent preparing it. Household members also differ in how much activity they encourage. The patient would like a shared understanding that appetite and energy fluctuate, with practical support that can adapt without blame or pressure.",
    "rehab": "Rehabilitation will focus on preserving meaningful activity within the treating team’s recommendations and the patient’s assessed tolerance. Tasks may be simplified or shared where this helps sustain participation. The therapist will pay attention to symptom response and coordinate concerns with the medical team, rather than judging progress solely by increased exercise or household output.",
    "safety": "The patient’s treatment specific instructions for reporting symptoms will be reviewed, including uncertainty about changes in intake or functional ability. Medication and supplement use will be reconciled with the treating team. The patient will be supported in raising concerns early without assuming that a familiar symptom label explains a new severity or pattern.",
    "followup": "Followup will review fatigue, appetite, daily function, and the usefulness of available support. The patient will be asked which activities remain important and whether the current plan allows flexible participation. Persistent or changing symptoms will prompt clinical reassessment, while goals will be adapted to the treatment course and the patient’s own priorities."
  },
  {
    "key": "prosthetic_adjustment",
    "title": "Adjustment to prosthetic use",
    "summary": "The patient is adapting to a lower limb prosthesis and describes mixed confidence during standing and short walks. The device feels more familiar in some situations than others, and skin comfort varies after use. The patient wants to increase independence but is unsure how to distinguish ordinary adjustment from a fit issue requiring review.",
    "observation": "During a supervised transfer, the patient checked foot placement carefully and shifted weight toward the opposite side before standing. Walking was deliberate, with a pause before turning. The patient reported awareness of pressure but did not describe it consistently, and no conclusion about fit was drawn without the planned residual limb and prosthetic assessment.",
    "assessment": "The observed movement may reflect balance demands, strength, unfamiliarity, prosthetic alignment, or discomfort. These factors need assessment together because compensations can appear useful while increasing effort elsewhere. The patient’s motivation is clear, but confidence should not be treated as the sole criterion for progression or as evidence that the device is fitting appropriately.",
    "plan": "The team will review the prosthetic plan, current use instructions, skin concerns, and the patient’s functional goals. The prosthetic clinician and therapist will coordinate any recommendations. Adjustments to the device or wearing schedule will follow assessment, with a clear explanation of how comfort, skin findings, and movement quality influence decisions about progression.",
    "home": "At home, the patient has arranged a stable seating area for managing the prosthesis and keeps supplies nearby. Narrow turns and uneven transitions between surfaces remain challenging. Household members can help with some tasks, but the patient prefers to practice independence within agreed limits and wants assistance offered without unexpected pulling or hurried handling.",
    "rehab": "Rehabilitation will assess transfers, standing balance, and walking within the confirmed prosthetic plan. The therapist will observe weight transfer and the patient’s response to different tasks, allowing time to describe pressure or discomfort. Goals will emphasize reliable movement and manageable effort, with device concerns referred for assessment rather than addressed through compensatory movement alone.",
    "safety": "The patient’s individualized instructions for skin inspection, device concerns, and stopping or modifying use will be reviewed by the treating team. Equipment and support surfaces will be checked for suitability. The patient will be encouraged to report unfamiliar pressure or changing function rather than assuming that discomfort must be tolerated to demonstrate progress.",
    "followup": "Followup will compare skin findings, comfort, confidence, and performance during selected tasks. The patient will identify situations in which the prosthesis remains difficult to manage outside supervised practice. The team will revise goals and equipment recommendations collaboratively, recognizing that increased wearing time is not the only meaningful measure of successful adaptation."
  },
  {
    "key": "scar_recovery",
    "title": "Skin and movement during scar recovery",
    "summary": "The patient is recovering from a healed skin injury and reports tightness, sensitivity, and discomfort when stretching the affected area during daily tasks. The skin surface appears closed by the patient’s account, but confidence remains limited. The patient wants to resume familiar activities while understanding which sensations are expected and which need review.",
    "observation": "The patient moved cautiously when reaching across the affected area and adjusted clothing to reduce contact. Visible skin showed altered texture, though a complete examination was not performed during this observation. The patient’s response demonstrated sensitivity and movement limitation, but did not establish the depth of restriction or the condition of the underlying tissues.",
    "assessment": "The difficulty may involve scar sensitivity, restricted movement, protective behavior, or a combination of these factors. The fact that the surface has closed does not establish full recovery or explain every symptom. Assessment will consider skin integrity, movement, sensation, and the effect of clothing or equipment on the patient’s ordinary activities.",
    "plan": "The clinician will review the healing course and assess the areas associated with tightness or sensitivity. Existing skin care, movement, and garment instructions will be reconciled before changes are proposed. The patient’s practical goals will guide the plan, with any specialized treatment based on examination and coordinated across the professionals involved in recovery.",
    "home": "At home, the patient has selected softer clothing and avoids tasks that repeatedly rub or stretch the affected region. Some adaptations help, while others make the patient feel that recovery is standing still. Discussion will distinguish useful temporary adjustments from restrictions that may no longer be needed once the clinical assessment is complete.",
    "rehab": "Rehabilitation will examine the specific movements limited by tightness or sensitivity and relate them to the patient’s chosen activities. The therapist will coordinate with the skin care team before recommending targeted work. Progress will reflect comfort, movement quality, and participation, rather than requiring the patient to disregard symptoms in pursuit of a fixed range.",
    "safety": "The patient has received informal suggestions about massage, products, and pressure but has not been sure which apply to the current stage of recovery. The treating team will clarify the individualized plan. Skin changes and equipment tolerance will be reviewed, and new discomfort will not automatically be interpreted as a necessary consequence of scar improvement.",
    "followup": "Followup will assess the skin, movement restrictions, and the patient’s experience with clothing and ordinary tasks. The team will ask whether the agreed approach feels manageable and whether sensitivity is changing. If the pattern remains unclear or function declines, the assessment will be revisited rather than repeating the same intervention without reconsidering its purpose."
  },
  {
    "key": "jaw_function",
    "title": "Jaw discomfort affecting eating and conversation",
    "summary": "The patient reports jaw discomfort during prolonged chewing and notices occasional tension while concentrating. Meals take longer, and the patient has begun avoiding some foods because of anticipated discomfort. The patient is unsure whether the problem is related to clenching, a dental issue, or another cause, and wants a coordinated assessment before making further changes.",
    "observation": "The patient spoke comfortably for most of the encounter but paused after a longer explanation and described a sense of tightness. Mouth opening was not formally measured, and no dental examination was performed. The observation therefore provides limited functional context and cannot establish the source of discomfort or whether joint movement is restricted.",
    "assessment": "Possible contributors include muscle tension, joint dysfunction, dental factors, or other conditions that require assessment. The patient’s food avoidance may reduce meal enjoyment and variety even before the cause is clear. The team will consider symptoms and function together while avoiding assumptions that stress or clenching alone accounts for the entire presentation.",
    "plan": "The clinician will review the symptom course, associated features, dental history, and any prior advice or appliances. Appropriate examination and referral will be guided by those findings. Recommendations about food choices, exercises, or devices will be individualized after assessment, with the patient involved in deciding which functional concerns need the most immediate attention.",
    "home": "At home, the patient chooses meals that seem easier to manage and sometimes eats separately to avoid holding others up. Concentration tasks appear to coincide with jaw tension, although the association is uncertain. The patient would like practical support that preserves shared meals and does not require constant monitoring of every facial movement or sensation.",
    "rehab": "Rehabilitation will focus on assessed movement and task demands rather than prescribing generic jaw exercises. The clinician may explore how chewing, speaking, and sustained concentration relate to symptoms. Goals will include more comfortable meals and conversation, with coordination between relevant professionals so that advice about appliances, movement, and symptom management remains consistent.",
    "safety": "The patient has considered purchasing an appliance without assessment but is uncertain whether it would suit the problem. The team will review this question through the appropriate clinical evaluation. Instructions for reporting changing symptoms will be clarified, and discomfort will not automatically be attributed to a familiar jaw problem when its character or context differs.",
    "followup": "Followup will review examination findings, eating tolerance, communication comfort, and the effect of any agreed changes. The patient will be asked whether food restriction has broadened or become less necessary. If the initial explanation does not account for the pattern, the team will reconsider contributing factors and coordinate further review rather than relying on repeated reassurance."
  },
  {
    "key": "pelvic_pain_function",
    "title": "Pelvic discomfort affecting sitting and activity",
    "summary": "The patient describes pelvic discomfort that varies with sitting, household activity, and periods of increased stress. The symptoms have led to frequent changes of position and concern about planning longer outings. The patient wants a respectful assessment that considers physical contributors and daily function without assuming that variable symptoms are either insignificant or entirely emotional.",
    "observation": "The patient shifted position several times during conversation and preferred a chair that allowed easy changes in posture. The discomfort was described as difficult to localize consistently. No internal examination or provocative testing occurred during this encounter, and the observed behavior does not establish the anatomical source or mechanism of the pain.",
    "assessment": "The presentation may involve several interacting physical and contextual factors, and the current information is insufficient to identify a single cause. Variable symptoms do not exclude a clinically meaningful problem. Assessment will consider relevant medical history, associated symptoms, and functional effects while acknowledging the patient’s previous experience of feeling that the discomfort was difficult to explain.",
    "plan": "The clinician will review the symptom pattern and determine which examinations or referrals are appropriate with the patient’s informed agreement. The patient’s priorities will shape the discussion of treatment options. Any targeted exercises or other interventions will follow assessment, with clear explanations of their purpose and an opportunity to raise concerns about comfort or acceptability.",
    "home": "At home, the patient alternates between sitting and standing but sometimes abandons an activity because no position remains comfortable. Household responsibilities are difficult to share when symptoms are unpredictable. The patient would like support that can adapt to fluctuation and avoids requiring detailed justification whenever an activity needs to be shortened or postponed.",
    "rehab": "Rehabilitation will examine functional tasks and the relationship between position, movement, and symptoms within the assessed clinical context. The therapist will avoid assuming that strengthening or relaxation is appropriate before identifying relevant findings. Goals will focus on useful participation and manageable symptom impact, with progression guided by the patient’s response and coordinated medical input.",
    "safety": "The team will clarify the individualized instructions for new or changing symptoms and ensure the patient knows how to contact the responsible clinician. Any examination or intervention will be explained and consent confirmed. The patient’s ability to tolerate a brief task will not be used to dismiss difficulties with sustained activity or symptoms occurring elsewhere.",
    "followup": "Followup will review symptom behavior, daily participation, and whether the assessment and treatment approach feels understandable and respectful. The patient will identify activities that have become more manageable and any new concerns. If the initial plan does not help, the team will revisit the working explanation rather than simply increasing demands or attributing the result to anxiety."
  },
  {
    "key": "abdominal_recovery",
    "title": "Functional recovery after an abdominal procedure",
    "summary": "The patient is recovering from an abdominal procedure and reports fatigue, pulling discomfort, and uncertainty about returning to ordinary tasks. Personal care is manageable with extra time, but household responsibilities remain difficult. The patient wants clear guidance that reflects the specific recovery plan and helps distinguish necessary restrictions from activities that may already be appropriate.",
    "observation": "The patient used both arms to assist with rising and paused before standing fully upright. Walking across the room was deliberate, and the patient described a pulling sensation during the transfer. This brief observation does not establish the condition of the healing tissues or explain the discomfort without review of the procedure and clinical findings.",
    "assessment": "The functional limitations may reflect healing, fatigue, protective movement, or uncertainty about the existing instructions. Their relative contribution is not yet clear. The patient's ability to complete essential tasks does not establish tolerance for repeated or prolonged effort, so assessment will consider the cumulative demands of daily life as well as individual movements.",
    "plan": "The clinician will review the procedure specific recovery instructions, current symptoms, and any prior recommendations about activity. The patient's questions about transfers, personal care, and household tasks will guide discussion. Changes to exercise, support needs, or restrictions will follow assessment and coordination with the responsible team rather than a general expectation about recovery speed.",
    "home": "At home, the patient has arranged essential supplies within reach and accepts help with heavier tasks. Some routine activities still require repeated bending or movement between rooms. The patient would like modest changes that reduce unnecessary effort while preserving independence, with household assistance matched to the current plan rather than based on conflicting informal advice.",
    "rehab": "Rehabilitation will examine transfers, walking, and the practical demands of personal care within confirmed precautions. The therapist will observe symptom response and discuss alternatives that fit the patient's environment. Progress will be measured through comfort and sustainable participation, while targeted abdominal work or increased loading will follow the individualized clinical recommendations and relevant examination findings.",
    "safety": "The patient's understanding of the postoperative escalation instructions will be checked, including which symptoms should not be assumed to reflect ordinary fatigue or healing. The team will review assistance needs and the effects of any medicines on daily tasks. Advice will be coordinated so that changing recommendations remain clear and do not create competing expectations.",
    "followup": "Followup will review discomfort, fatigue, function, and the practical effect of household changes. The patient will be asked whether the plan is understandable and whether assistance remains appropriate. Ongoing or changing symptoms will prompt reassessment rather than being dismissed because the patient has managed essential tasks, and restrictions will be updated through the responsible clinical team."
  },
  {
    "key": "joint_fluctuation",
    "title": "Fluctuating joint symptoms and daily function",
    "summary": "The patient reports fluctuating joint pain and stiffness that make mornings and repetitive tasks more difficult. Several joints are involved, but their relative contribution changes. The patient is concerned about planning ahead because a manageable activity on one occasion may be difficult on another, and wants help distinguishing symptom variation from a possible change in disease activity.",
    "observation": "The patient took extra time to rise from a chair and described stiffness during the first movements. Hand use during a small object task was slower on one side. These observations demonstrate functional impact but do not establish active inflammation, structural change, or the cause of the difference between the patient’s more and less symptomatic periods.",
    "assessment": "The reported variability may involve inflammatory activity, mechanical demands, fatigue, treatment effects, or several factors together. Function should therefore be interpreted alongside the clinical examination and relevant results. The patient’s concern about unpredictability is a meaningful part of the burden, even when the observed task can be completed during a relatively manageable period.",
    "plan": "The clinician will review the symptom course, current treatment, associated features, and any changes from the established pattern. Examination and available results will guide decisions about disease activity and treatment. Functional goals will be discussed alongside medical management, with consistent advice about how to adapt tasks and communicate concerns when symptoms fluctuate.",
    "home": "At home, the patient prepares some items in advance but finds that extensive planning can itself become tiring. Household members are willing to help, though they sometimes assume improvement is permanent after a better period. The patient wants a flexible division of responsibilities that allows participation without promising the same capacity for every occasion.",
    "rehab": "Rehabilitation will assess joint movement, task demands, and the patient’s preferred ways of conserving effort. The therapist will coordinate recommendations with the medical plan and adapt practice to the current presentation. Goals will include maintaining useful activity and reducing avoidable strain, without assuming that either unrestricted activity or broad avoidance is appropriate for all joints.",
    "safety": "The team will review the patient’s individualized instructions for reporting a substantial change in symptoms and clarify any uncertainty about medicines or activity restrictions. Devices and supports already used will be checked for suitability. New functional loss will be considered clinically rather than automatically categorized as an ordinary fluctuation of the known joint condition.",
    "followup": "Followup will compare symptoms and function across the patient’s chosen activities, including how the plan works during more difficult periods. The clinician will review any evidence relevant to disease activity and treatment response. Progress will include greater predictability and effective adaptation, while persistent barriers will prompt reconsideration of both medical and rehabilitation approaches."
  },
  {
    "key": "widespread_pain",
    "title": "Widespread pain with variable activity tolerance",
    "summary": "The patient describes widespread pain, unrefreshing rest, and variable tolerance for ordinary activity. Several previous explanations have been offered, leaving the patient unsure how the symptoms fit together. The patient wants a coherent plan that acknowledges the pain and supports meaningful participation without suggesting that successful activity requires ignoring symptoms or proving their severity.",
    "observation": "The patient changed position during the encounter and described discomfort in more than one area. Movement remained possible, but the patient anticipated increased symptoms after sustained tasks. The observation captured only a brief period and does not establish the source of pain, the degree of later symptom change, or overall functional capacity.",
    "assessment": "The presentation may involve multiple interacting contributors, and the current information does not support assigning every symptom to a single mechanism. Sleep, emotional strain, movement patterns, and other health conditions may influence the experience without making it less real. The assessment will consider consistency, associated findings, prior evaluations, and the patient’s own priorities.",
    "plan": "The clinician will review the history and previous explanations, looking for unresolved questions and duplicated or conflicting recommendations. The patient will help identify a small set of meaningful goals for the coordinated plan. Treatment decisions will be individualized, and any uncertainty about diagnosis or expected response will be stated clearly rather than concealed by broad reassurance.",
    "home": "At home, the patient alternates between catching up on responsibilities and withdrawing when symptoms increase. Others sometimes interpret this pattern as inconsistency, which adds strain to relationships. The patient would like assistance organizing shared expectations and identifying forms of support that preserve autonomy without requiring a detailed account of pain before every adjustment.",
    "rehab": "Rehabilitation will explore the patient’s activity pattern and response to meaningful tasks within the broader clinical plan. The therapist will consider pacing, confidence, and movement options without assuming that one approach suits every symptom. Progress will include participation and reduced disruption, with goals revised when the patient’s response differs from the initial expectation.",
    "safety": "The team will reconcile prescribed and nonprescribed treatments and review any effects on alertness, balance, or daily function. New symptoms or a clear change in pattern will receive appropriate assessment rather than being attributed automatically to the existing pain condition. The patient’s individualized plan for seeking help will be checked for clarity and accessibility.",
    "followup": "Followup will review function, symptom burden, treatment effects, and the patient’s understanding of the coordinated plan. The team will ask what improved daily life and what added effort without benefit. Lack of progress will prompt reconsideration of the approach and relevant clinical questions, rather than an assumption that the patient has failed to engage."
  },
  {
    "key": "bowel_urgency",
    "title": "Bowel urgency and confidence outside the home",
    "summary": "The patient reports episodes of bowel urgency that have reduced confidence leaving home and attending shared activities. The pattern is inconsistent, and occasional leakage has been difficult to discuss. The patient has changed meal choices and timing in an effort to prevent episodes but is unsure whether those changes are helping or creating additional problems.",
    "observation": "The patient could describe several recent situations but had difficulty comparing stool consistency, urgency, and associated discomfort across them. There was no direct observation of the symptoms or abdominal examination during this part of the encounter. The account demonstrates substantial functional impact while leaving the underlying bowel pattern and cause insufficiently characterized.",
    "assessment": "The urgency may relate to gastrointestinal function, medication effects, access barriers, or other contributors requiring assessment. Self imposed dietary restrictions could affect nutrition and social participation without reliably addressing the cause. The clinician will consider the complete symptom history and avoid interpreting embarrassment, delayed reporting, or variable episodes as evidence that the problem is minor.",
    "plan": "The plan is to review the bowel history, associated symptoms, current medicines, and changes in eating or drinking. Further examination or testing will be determined by the clinical findings. The patient will be offered a manageable way to clarify the pattern, and any treatment or dietary changes will be individualized after the relevant assessment.",
    "home": "At home, the patient keeps supplies available but feels anxious when bathroom access is uncertain. Shared meals have become less enjoyable because attention remains focused on possible symptoms afterward. The patient would like practical support that protects privacy and permits participation, without requiring household members to know details that the patient does not wish to share.",
    "rehab": "Rehabilitation will consider access to the bathroom, clothing management, transfer speed, and confidence with ordinary outings. The therapist will coordinate with the medical team so that functional adjustments support rather than replace assessment of the bowel symptoms. Goals will include reduced disruption and restored participation, with any specialized pelvic rehabilitation based on appropriate findings.",
    "safety": "The patient sometimes rushes during urgency and has become less attentive to obstacles in the route. The team will review this practical risk alongside the individualized instructions for reporting changing bowel symptoms. Unreviewed medication or dietary changes will be discussed with the responsible clinician, especially where several self directed approaches have been tried together.",
    "followup": "Followup will review the bowel pattern, leakage concerns, eating changes, and participation outside the home. The patient will be asked whether the plan remains discreet, practical, and acceptable. If symptoms continue or change, the clinical assessment will be reconsidered rather than assuming that confidence or environmental adjustments alone should resolve the difficulty."
  }
]

if __name__ == '__main__':
    main()
