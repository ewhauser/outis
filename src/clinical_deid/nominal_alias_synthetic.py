"""Original nominal-alias contrasts: Apache-2.0 code, CC0-1.0 text/labels.

The anchored arm retains the same explicit full identity in both notes. The
unanchored arm contrasts short human names with wholly PHI-free clinical notes.
Reduced mentions are personal identifiers only under nominal human grammar;
ordinary grammar, public terms and pronouns stay unlabelled despite an anchor.
No retired sentences, identities, generator or qualification input is used.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import re
import unicodedata

import tiktoken

from . import generalization_synthetic as audit
from .types import ENTITY_TYPES

VERSION = "1.0"
SEED = 7493181
FUNCTION_NAMES = (
    "Have",
    "For",
    "An",
    "The",
    "No",
    "Na",
    "A",
    "To",
    "Do",
    "In",
    "Be",
    "Will",
    "May",
    "Not",
    "But",
    "As",
    "Or",
)
PRONOUN_NAMES = ("He", "We", "I", "You", "Me", "Us", "It")
COMMON_NAMES = FUNCTION_NAMES + PRONOUN_NAMES
_COMMON_KEYS = {word.casefold() for word in COMMON_NAMES}
_STARTS = {
    "train": ("Br", "Fl", "Gr", "Pr", "Tr", "Vr", "Bl", "Fr"),
    "validation": ("Ch", "Dh", "Jh", "Kh", "Sh", "Th", "Zh", "Xh"),
}
_VOWELS = ("a", "e", "i", "o", "u", "ae", "eo", "ia", "ui")
_LINKS = ("b", "d", "f", "g", "l", "m", "n", "p", "r", "s", "t", "v")
_ENDS = ("len", "mar", "rin", "ven", "ris", "lan", "mir", "nel")
_ACCENTS = ("Á", "É", "Í", "Ó", "Å", "Ö", "Ü", "Ê", "Ē")
LAYOUTS = ("prose", "sections", "bullets", "numbered", "handoff", "compact")
CAPTIONS = {
    "train": (
        "Identity verified with the {role}; full legal name: {v}.",
        "Full personal name recorded after the {role}'s identification check = {v}.",
        "The {role} read their complete name aloud: {v}.",
        "In the personal identity box, the {role} wrote {v}.",
        "The signed human identity field belongs to the {role}, {v}.",
        "For the identity record, {v} is the {role}'s full name.",
        "The {role}'s identity form states their complete personal name as {v}.",
        "Full name, personally confirmed with the {role} >> {v}.",
        "The individual receiving care confirmed the full name {v}; role: {role}.",
        "Personal naming statement of the {role}: [{v}].",
        "Identification interview with the {role}; the complete name was {v}.",
        "The full-name field signed by the {role} reads '{v}'.",
        "PERSONAL FULL NAME, {role_upper} IDENTITY CHECK: {v}.",
        "The {role} supplied {v} as the full name identifying them personally.",
    ),
    "validation": (
        "At identification, the {role} gave their complete name as {v}.",
        "The authenticated full-name entry for the {role} contains {v}.",
        "Personal identification completed; {role}'s full name = {v}.",
        "The {role}'s signed naming declaration was read as {v}.",
        "Complete human identity confirmed by the {role}: [{v}].",
        "During the identity interview, {v} was stated as the {role}'s full name.",
        "The personal full name attested by the {role} is {v}.",
        "Confirmed identification name for this {role} — {v}.",
        "The complete personal-name line supplied by the {role} reads '{v}'.",
        "Full identifying name on the {role}'s signed record: {v}.",
    ),
}
# No bare subject frame is used for a word that can be an ordinary pronoun.
NOMINAL_FRAMES = {
    "train": (
        "At the symptom interview, {v} answered the questions without prompting.",
        "Following the examination explanation, {v} recounted the morning routine.",
        "Before the mobility review ended, {v} offered details about daily support.",
        "During history taking, {v} clarified the sequence of treatment changes.",
        "After the findings were discussed, {v} confirmed the transport booking.",
        "{v} agreed to coordinate the lift needed for the next care visit.",
        "With the options explained, {v} volunteered to contact the care coordinator.",
        "{v} will organise the support needed for the planned review.",
        "The family of {v} explained how assistance is shared at home.",
        "Relatives of {v} described the home's accessible entrance.",
        "The history obtained from {v} included the usual rest pattern.",
        "Advice was reviewed with {v} before the appointment arrangements were made.",
        "The seated transfer was demonstrated by {v} after the instruction.",
        "The review questions were answered by {v} during the care discussion.",
        "{v}'s household arranged help with the regular medicine routine.",
        "{v}'s care partner described the changes noticed between visits.",
        "The clinician asked {v} to describe how symptoms affect usual activity.",
        "A copy of the care plan was handed to {v} for discussion with family.",
        "The patient {v} answered questions about the recent symptom course.",
        "The caregiver {v} described the practical support available at home.",
        "During the interview, {v}, the patient, supplied the relevant history.",
        "{v}, acting as caregiver, confirmed the arrangements for the next visit.",
        "The patient called {v} explained the medicine routine in their own words.",
        "The caregiver known personally as {v} confirmed the agreed transfer support.",
    ),
    "validation": (
        "When the history was requested, {v} described the course of recovery.",
        "After listening to the assessment, {v} added detail about usual activity.",
        "At the end of the interview, {v} supplied information about household help.",
        "In reviewing past symptoms, {v} explained when daily tasks became easier.",
        "Once the examination had finished, {v} checked the travel arrangements.",
        "{v} offered to book the transport needed for the clinical follow-up.",
        "After care options were outlined, {v} undertook to arrange the support visit.",
        "{v} will contact the care team to confirm the next appointment.",
        "Members of the family of {v} confirmed that help is available locally.",
        "The relatives of {v} outlined the adaptations already made at home.",
        "Information gathered from {v} clarified the regular treatment schedule.",
        "The examination plan was discussed with {v} before the interview closed.",
        "The supported movement was shown by {v} during the practical review.",
        "Questions about care needs were addressed by {v} at this meeting.",
        "{v}'s family made arrangements for assistance with everyday tasks.",
        "{v}'s support partner explained the usual monitoring routine.",
        "The interviewer invited {v} to describe the home-care arrangements.",
        "The monitoring instructions were given to {v} for review with relatives.",
        "The patient {v} recounted how the symptoms changed with rest.",
        "The caregiver {v} explained the assistance normally provided.",
        "At the review, {v}, the patient, gave the history directly.",
        "{v}, serving as caregiver, confirmed the planned household support.",
        "The patient introduced as {v} described the daily treatment use.",
        "The caregiver personally named {v} explained the observation schedule.",
    ),
}
# Direct role/name adjacency and literal naming supply nominal proof. Appositive
# "I, the patient" or "I, acting as caregiver" remain ordinary pronouns and
# cannot establish a personal-name mention, even beside a full identity anchor.
# Subject-case He/We/I can additionally be nominal after a preposition where an
# ordinary pronoun would require him/us/me. Other object-case pronouns cannot.
PRONOUN_ROLE_FRAMES = (18, 19, 22, 23)
SUBJECT_ONLY_PRONOUNS = {"he", "we", "i"}
OBJECT_NOMINAL_FRAMES = (8, 9, 10, 11, 12, 13, 16, 17)
ORDINARY_CLAUSES = {
    "have": "Staff noted that mild fatigue can {v} several contributing causes.",
    "for": "{v} routine observation, the team suggested a short symptom diary.",
    "an": "{v} extra seated pause allowed breathing effort to settle.",
    "the": "{v} current exercise schedule allows a rest between tasks.",
    "no": "{v} new swelling or persistent chest discomfort was found.",
    "na": "{v} {sodium} mmol/L was reviewed as an ordinary electrolyte result.",
    "a": "{v} gentle stretch was included after the supported transfer.",
    "to": "{v} limit fatigue, each activity was followed by a seated pause.",
    "do": "Staff instructed the group: '{v} stop the exercise if dizziness persists'.",
    "in": "{v} quiet breathing, respiratory effort remained comfortable.",
    "be": "The written advice was '{v} cautious when moving from lying to standing'.",
    "will": "The care checklist {v} help the team review treatment tolerance.",
    "may": "The planned rest break {v} make the next transfer more comfortable.",
    "not": "The examination findings are {v} suggestive of a new fever.",
    "but": "Walking was easier {v} the transfer still needed supervision.",
    "as": "{v} comfort improved, the supported task was repeated slowly.",
    "or": "The task can be completed seated {v} deferred until after a rest.",
    "he": "{v} answered the review questions after a brief seated rest.",
    "we": "{v} checked the treatment technique and discussed the findings together.",
    "i": "{v} explained the examination results during the team handoff.",
    "you": "The exercise instruction was '{v} may pause whenever the task feels tiring'.",
    "me": "The recipient asked the therapist to 'help {v} practise the seated reach'.",
    "us": "Relatives asked staff to 'show {v} how to provide transfer assistance'.",
    "it": "{v} was agreed that monitoring could continue with gradual activity.",
}
PUBLIC_WORD_FRAMES = (
    "The demonstration equipment sheet uses '{v}' as a public model heading.",
    "In the medicine catalogue, {v} is a public formulation label.",
    "The printed teaching card contains '{v}' as a vocabulary exercise.",
    "A public procedure worksheet labels its practice sequence {v}.",
    "The reference diagram is captioned '{v}' for staff education.",
    "Under nonpersonal catalogue terminology, the sample device label reads {v}.",
    "The clinical glossary gives '{v}' as the title of a demonstration test.",
    "The public exercise manual displays {v} as an instructional heading.",
)
CLINICAL_PROFILES = (
    (
        "airway technique",
        "Spacer use was checked and the inhaled budesonide dose is 180 mcg twice daily.",
    ),
    (
        "postural comfort",
        "A seated pause eased brief light-headedness; hydration and slow transitions were discussed.",
    ),
    (
        "oral examination",
        "The oral mucosa was moist, swallowing was comfortable and there was no new ulceration.",
    ),
    (
        "skin assessment",
        "Skin remained intact at pressure areas; gentle cleansing and pressure relief were reviewed.",
    ),
    (
        "activity tolerance",
        "Fatigue settled during recovery from the timed walk and breathing remained even.",
    ),
    (
        "sensory screening",
        "Monofilament sensation was checked, with balanced foot placement during the supported stand.",
    ),
    (
        "ear inspection",
        "The ear examination was tolerated and no new drainage or canal irritation was seen.",
    ),
    (
        "joint function",
        "Flexion improved after rest; paracetamol 500 mg as needed was reviewed for treatment tolerance.",
    ),
    (
        "glucose monitoring",
        "Food intake was steady and metformin 500 mg twice daily caused no new gastrointestinal symptoms.",
    ),
    (
        "circulation review",
        "Peripheral warmth and colour were satisfactory; ankle swelling had not increased.",
    ),
    (
        "balance observation",
        "Controlled turns were more comfortable and no persistent disequilibrium was observed.",
    ),
    (
        "respiratory recovery",
        "Air entry remained adequate during the seated recovery phase following the supported walk.",
    ),
    (
        "lower-limb therapy",
        "Supported stepping was possible, with symmetrical movement and a short rest before transfer.",
    ),
    (
        "wound prevention",
        "Pressure warning signs and the use of a bland emollient were explained during skin review.",
    ),
    (
        "medicine routine",
        "The regular treatment schedule was read back accurately and no extra dose was recommended.",
    ),
    (
        "upper-limb tasks",
        "Shoulder movement was comfortable enough for dressing practice; reach improved with a seated rest.",
    ),
    (
        "nutrition check",
        "Appetite was maintained and ordinary meal planning was discussed alongside fluid intake.",
    ),
    (
        "sleep and recovery",
        "Sleep interruption had eased, with activity pacing advised during the daytime routine.",
    ),
    (
        "head and neck review",
        "Neck movement was comfortable and there was no new facial weakness or swallowing change.",
    ),
    (
        "transfer practice",
        "The support technique was demonstrated slowly, with breathing and comfort checked between tasks.",
    ),
    (
        "neurological observations",
        "The screen included limb movement, sensation and controlled posture without a new focal change.",
    ),
    (
        "pelvic exercise teaching",
        "Pelvic movement and relaxed breathing were explained using a public demonstration diagram.",
    ),
    (
        "clinical ultrasound discussion",
        "The report described ordinary tissue appearance and the procedure precautions were reviewed.",
    ),
    (
        "teach-back and planning",
        "Teach-back confirmed the rest instructions; the care team checked when further advice would be needed.",
    ),
)
PUBLIC_REFERENCES = (
    "The public practice guideline carries publication year {year}, as a bibliographic reference.",
    "A public training manual was revised on {year}-04-19; its date refers to that manual.",
    "The manufacturer catalogue has an edition date of November {year}, independent of clinical events.",
    "The publicly released medicine guide has copyright year {year}.",
    "Public study NCT{digits} and article PMID {digits} identify reference material, not a person.",
    "The public device series DEMO-{code} and catalogue lot PUBLIC-{code} are product descriptions.",
    "Teaching examples use strength 5/5, pain 2/10 and screening 28/30 as clinical scales.",
    "The observation examples describe 104 days and one hundred and eight minutes, both durations.",
)


def canonical(value):
    return unicodedata.normalize("NFC", value).casefold()


def word(rng, split, *, short=False, accented=False):
    value = (
        rng.choice(_STARTS[split]) + rng.choice(_VOWELS)
        if short
        else rng.choice(_STARTS[split])
        + rng.choice(_VOWELS)
        + rng.choice(_LINKS)
        + rng.choice(_VOWELS)
        + rng.choice(_ENDS)
    )
    return rng.choice(_ACCENTS) + value.lower() if accented else value


def identity(rng, split, index, *, secondary=False, shared_with=None):
    mode = index % 12
    if not secondary and mode < 5:
        first = FUNCTION_NAMES[(index // 12 + mode * 5) % len(FUNCTION_NAMES)]
    elif not secondary and mode in {5, 6}:
        first = PRONOUN_NAMES[(index // 12 + mode) % len(PRONOUN_NAMES)]
    else:
        first = word(rng, split, short=mode == 7, accented=mode in {8, 9, 10})
    family = word(rng, split)
    if shared_with is not None:
        first = shared_with["first"]
    if mode == 10:
        family = rng.choice(("de", "van", "al", "von")) + " " + family
    initial = rng.choice(_ACCENTS) + "." if mode == 9 else ""
    suffix = rng.choice(("Jr.", "Sr.", "II", "III")) if mode == 11 else ""
    if mode == 8 or index % 13 == 0:
        first, family, initial, suffix = (
            part.lower() for part in (first, family, initial, suffix)
        )
    if mode == 9:
        first, initial = (
            unicodedata.normalize("NFD", part) for part in (first, initial)
        )
    if shared_with is not None:
        first = shared_with["first"]
    comma = mode in {6, 10} or index % 9 == 0
    full = (
        " ".join(part for part in (family, suffix) if part)
        + ", "
        + " ".join(part for part in (first, initial) if part)
        if comma
        else " ".join(part for part in (first, initial, family, suffix) if part)
    )
    return {
        "full": full,
        "first": first,
        "surname": family,
        "mode": mode,
        "common": canonical(first) in _COMMON_KEYS,
        "pronoun": canonical(first) in {v.casefold() for v in PRONOUN_NAMES},
        "comma_order": comma,
        "suffix": suffix,
    }


def clinical_body(rng, profile):
    topic, finding = CLINICAL_PROFILES[profile]
    return (
        f"Clinical reassessment / {topic}\n{finding} "
        f"BP {rng.randrange(108, 148)}/{rng.randrange(60, 87)} mmHg; pulse {rng.randrange(58, 94)}/min; "
        f"oxygen saturation {rng.randrange(95, 100)}%. Na {rng.randrange(136, 143)} mmol/L; "
        f"glucose {rng.randrange(86, 175)} mg/dL; creatinine {rng.randrange(7, 14) / 10:.1f} mg/dL. "
        f"The clinical teaching example concerns an adult aged {rng.randrange(34, 86)} years, "
        f"with {rng.randrange(3, 15)} days of symptom monitoring. "
        "S: comfort at rest. O: controlled movement. A/P: continue paced activity."
    )


def _frame_choices(person, split):
    if not person["pronoun"]:
        return tuple(range(len(NOMINAL_FRAMES[split])))
    extra = (
        OBJECT_NOMINAL_FRAMES
        if canonical(person["first"]) in SUBJECT_ONLY_PRONOUNS
        else ()
    )
    return PRONOUN_ROLE_FRAMES + extra


def _ordinary(person, rng):
    first = person["first"]
    if person["common"]:
        return ORDINARY_CLAUSES[canonical(first)].format(
            v=first, sodium=rng.randrange(136, 143)
        )
    return rng.choice(PUBLIC_WORD_FRAMES).format(v=first)


def generate_pair(index, *, split="train", seed=SEED, attempt=0):
    if (
        split not in CAPTIONS
        or not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
    ):
        raise ValueError(
            "A nonnegative index and declared train/validation split are required"
        )
    rng = random.Random(
        (seed << 32) + index * 890069 + attempt + (0 if split == "train" else 619871)
    )
    person = identity(rng, split, index)
    shared = index % 12 == 11
    other = (
        identity(
            rng,
            split,
            index + 271,
            secondary=True,
            shared_with=person if shared else None,
        )
        if shared or index % 4 == 0
        else None
    )
    profile, layout = index % len(CLINICAL_PROFILES), rng.choice(LAYOUTS)
    body = clinical_body(rng, profile)
    role = "caregiver" if index % 3 == 1 else "patient"
    caption = rng.randrange(len(CAPTIONS[split]))
    choices = tuple(range(18)) if shared else _frame_choices(person, split)
    preferred = (index + index // len(NOMINAL_FRAMES[split])) % len(
        NOMINAL_FRAMES[split]
    )
    frame = preferred if preferred in choices else rng.choice(choices)
    # Role-specific frames must agree with the identity's declared role.
    if frame in {18, 20, 22}:
        role = "patient"
    elif frame in {19, 21, 23}:
        role = "caregiver"
    separator = {
        "prose": "\n",
        "sections": "\nHistory: ",
        "bullets": "\n- ",
        "numbered": "\n2. ",
        "handoff": "\nClinical discussion / ",
        "compact": " | ",
    }[layout]
    episode = date(2014, 1, 1) + timedelta(days=rng.randrange(4100))
    public_year = rng.randrange(1997, min(2021, episode.year - 1) + 1)
    reference = PUBLIC_REFERENCES[index % len(PUBLIC_REFERENCES)].format(
        year=public_year,
        digits=rng.randrange(10000000, 99999999),
        code=rng.randrange(10000, 99999),
    )
    reference += " Public examination terminology includes Babinski testing, Baker cyst and Addison disease."
    ordinary = _ordinary(person, rng)
    rows = []
    for variant in ("nominal", "counterfactual"):
        note, mentions, exclusions = audit.Note(), [], []

        def name_clause(template, owner, kind, **arguments):
            note.clause(template, "NAME", owner[kind], **arguments)
            mentions.append(
                {"person": "primary" if owner is person else "other", "kind": kind}
            )

        def nonpersonal(value, interpretation):
            start = note.length
            note.plain(value)
            exclusions.append(
                {"start": start, "end": note.length, "interpretation": interpretation}
            )

        note.plain(body + separator)
        name_clause(
            CAPTIONS[split][caption], person, "full", role=role, role_upper=role.upper()
        )
        if other is not None:
            note.plain("\n")
            name_clause(
                "The consulting clinician's verified full personal name is {v}.",
                other,
                "full",
            )
        note.plain(separator)
        if variant == "nominal":
            name_clause(NOMINAL_FRAMES[split][frame], person, "first")
            if shared:
                mentions[-1]["person"] = "shared"
            note.plain(" ")
            # A surname carries no pronoun meaning and has a unique owner here.
            name_clause(
                "Care instructions were discussed with {v} before the history review closed.",
                person,
                "surname",
            )
        else:
            nonpersonal(
                ordinary,
                "ordinary_function_or_pronoun"
                if person["common"]
                else "public_product_term",
            )
            note.plain(" ")
            nonpersonal(
                PUBLIC_WORD_FRAMES[(index + 3) % len(PUBLIC_WORD_FRAMES)].format(
                    v=person["surname"]
                ),
                "public_surname_spelling",
            )
        # The same ordinary-pronoun/public occurrence appears even in positives.
        note.plain("\nClinical team observations: ")
        nonpersonal(
            ordinary,
            "ordinary_function_or_pronoun"
            if person["common"]
            else "public_product_term",
        )
        if other is not None:
            note.plain("\n")
            name_clause(
                "Dr. {v} reviewed the clinical precautions with the care team.",
                other,
                "surname",
            )
        if index % 6 == 0:
            note.plain("\n")
            note.clause(
                "The private assessment date is {v}.", "DATE", episode.isoformat()
            )
            note.plain(" ")
            note.clause(
                "The individual's follow-up date is {v}.",
                "DATE",
                (episode + timedelta(days=14)).isoformat(),
            )
        note.plain("\n")
        nonpersonal(reference, "public_reference_and_clinical_terms")
        row = {
            "id": f"nominal-alias-{split}-{seed}-{index:05d}-{variant}",
            "pair_id": f"nominal-alias-{split}-{seed}-{index:05d}",
            "text": note.text,
            "spans": note.spans,
            "split": split,
            "suite": f"authored_nominal_alias_{'training' if split == 'train' else 'development'}",
            "family": f"nominal_alias.{split}.{layout}.clinical-{profile}.alias-{frame}",
            "variant": variant,
            "negative": False,
            "person_identity": person,
            "other_identity": other,
            "shared_first_alias": shared,
            "name_mentions": mentions,
            "nonpersonal_clauses": exclusions,
            "caption_context": caption,
            "alias_context": frame,
            "clinical_profile": profile,
            "public_reference_context": index % len(PUBLIC_REFERENCES),
            "publication_year": public_year,
            "clinical_episode_year": episode.year,
        }
        validate_row(row)
        rows.append(row)
    return rows


def short_identity(rng, split, index):
    mode = index % 10
    if mode == 8:
        first = FUNCTION_NAMES[(index // 10) % len(FUNCTION_NAMES)]
    elif mode == 9:
        first = PRONOUN_NAMES[(index // 10) % len(PRONOUN_NAMES)]
    else:
        # Productive letters, rather than a private-name dictionary. Short
        # morphology gives meaningful singleton BIOES training exposure.
        consonants = "bdflmnprst" if split == "train" else "cghjkvwxz"
        first = rng.choice(consonants).upper() + rng.choice("aeiou")
        if mode in {2, 3}:
            first += rng.choice(consonants)
        if mode in {4, 5}:
            first = rng.choice(_ACCENTS) + first.lower()
        if mode == 5:
            first = unicodedata.normalize("NFD", first)
        if mode in {6, 7}:
            first = first.lower()
    # Accidental common spellings obey the same safety policy as chosen ones.
    return {
        "full": None,
        "first": first,
        "surname": None,
        "mode": mode,
        "common": canonical(first) in _COMMON_KEYS,
        "pronoun": canonical(first) in {word.casefold() for word in PRONOUN_NAMES},
        "comma_order": False,
        "suffix": "",
    }


def generate_unanchored_pair(index, *, split="train", seed=SEED, attempt=0):
    if (
        split not in CAPTIONS
        or not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
    ):
        raise ValueError(
            "A nonnegative index and declared train/validation split are required"
        )
    rng = random.Random(
        (seed << 32) + index * 790063 + attempt + (0 if split == "train" else 818471)
    )
    profile = index % len(CLINICAL_PROFILES)
    person = short_identity(rng, split, index)
    value = person["first"]
    body = clinical_body(rng, profile)
    ordinary = _ordinary(person, rng)
    choices = tuple(range(18)) if not person["common"] else PRONOUN_ROLE_FRAMES
    first_frame, second_frame = rng.sample(choices, 2)
    if person["common"]:
        # Both repetitions have independently clear human-role/naming proof.
        # Keep the chosen role consistent rather than mixing patient/caregiver.
        choices = (
            ((18, 22) if (index // 10) % 2 == 0 else (19, 23))
            if person["pronoun"]
            else ((18, 20, 22) if (index // 10) % 2 == 0 else (19, 21, 23))
        )
        first_frame = choices[(index // 10) % len(choices)]
        second_frame = rng.choice(
            tuple(frame for frame in choices if frame != first_frame)
        )
    episode_year = rng.randrange(2014, 2026)
    public_year = rng.randrange(1997, min(2021, episode_year - 1) + 1)
    reference = PUBLIC_REFERENCES[index % len(PUBLIC_REFERENCES)].format(
        year=public_year,
        digits=rng.randrange(10000000, 99999999),
        code=rng.randrange(10000, 99999),
    )
    rows = []
    for variant in ("unanchored", "phi_free"):
        note, mentions, exclusions = audit.Note(), [], []
        note.plain(body + "\n")
        if variant == "unanchored":
            for frame in (first_frame, second_frame):
                note.clause(NOMINAL_FRAMES[split][frame], "NAME", value)
                mentions.append({"person": "primary", "kind": "first"})
                note.plain("\n")
        else:
            if index % 2 == 0:
                note.plain(rng.choice(("Nurse", "Clinician", "Caregiver")) + "\n")
                start = note.length
                note.plain(ordinary)
                exclusions.append(
                    {
                        "start": start,
                        "end": note.length,
                        "interpretation": "split_heading_ordinary_grammar",
                    }
                )
                note.plain("\n")
            start = note.length
            note.plain(
                PUBLIC_WORD_FRAMES[(index + 5) % len(PUBLIC_WORD_FRAMES)].format(
                    v=value
                )
            )
            exclusions.append(
                {
                    "start": start,
                    "end": note.length,
                    "interpretation": "public_same_short_spelling",
                }
            )
            note.plain("\n")
        note.plain("Clinical team observations: ")
        for text, meaning in (
            (
                ordinary,
                "ordinary_function_or_pronoun"
                if person["common"]
                else "public_product_term",
            ),
            (reference, "public_reference"),
        ):
            start = note.length
            note.plain(text)
            exclusions.append(
                {"start": start, "end": note.length, "interpretation": meaning}
            )
            note.plain("\n")
        row = {
            "id": f"nominal-alias-{split}-{seed}-short-{index:05d}-{variant}",
            "pair_id": f"nominal-alias-{split}-{seed}-short-{index:05d}",
            "text": note.text,
            "spans": note.spans,
            "split": split,
            "suite": f"authored_nominal_alias_{'training' if split == 'train' else 'development'}",
            "family": f"nominal_alias.{split}.unanchored.clinical-{profile}",
            "variant": variant,
            "negative": variant == "phi_free",
            "person_identity": person,
            "other_identity": None,
            "shared_first_alias": False,
            "name_mentions": mentions,
            "nonpersonal_clauses": exclusions,
            "alias_context": first_frame,
            "second_alias_context": second_frame,
            "clinical_profile": profile,
            "public_reference_context": index % len(PUBLIC_REFERENCES),
            "publication_year": public_year,
            "clinical_episode_year": episode_year,
        }
        validate_row(row)
        rows.append(row)
    return rows


def validate_row(row):
    split = row.get("split")
    if (
        split not in CAPTIONS
        or row.get("suite")
        != f"authored_nominal_alias_{'training' if split == 'train' else 'development'}"
    ):
        raise ValueError(
            "Only declared authored training/development rows are permitted"
        )
    previous, names = 0, []
    for span in row["spans"]:
        start, end = span["start"], span["end"]
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= len(row["text"])
            or start < previous
            or span["label"] not in ENTITY_TYPES
        ):
            raise ValueError("Gold offsets must be ordered, disjoint and valid")
        if unicodedata.category(row["text"][start]).startswith("M") or (
            end < len(row["text"])
            and unicodedata.category(row["text"][end]).startswith("M")
        ):
            raise ValueError("Gold offsets may not cut a Unicode combining sequence")
        previous = end
        if span["label"] == "NAME":
            names.append(row["text"][start:end])
    if row["negative"] != (not row["spans"]) or len(names) != len(row["name_mentions"]):
        raise ValueError("Annotation role and mention ownership disagree")
    for value, mention in zip(names, row["name_mentions"]):
        if mention["person"] == "shared":
            if (
                not row["shared_first_alias"]
                or mention["kind"] != "first"
                or value != row["person_identity"]["first"]
                or value != row["other_identity"]["first"]
            ):
                raise ValueError(
                    "A shared nominal alias must name both declared owners"
                )
        else:
            owner = (
                row["person_identity"]
                if mention["person"] == "primary"
                else row["other_identity"]
            )
            if owner is None or value != owner[mention["kind"]]:
                raise ValueError("Every name must belong to its declared full identity")
    for exclusion in row["nonpersonal_clauses"]:
        start, end = exclusion["start"], exclusion["end"]
        if not 0 <= start < end <= len(row["text"]) or any(
            start < span["end"] and end > span["start"] for span in row["spans"]
        ):
            raise ValueError("Ordinary grammar/public clauses must not own PHI")
    if row["variant"] != "phi_free":
        if row["publication_year"] >= row["clinical_episode_year"]:
            raise ValueError(
                "Public reference dates must precede the private assessment"
            )
        if row["person_identity"]["pronoun"] and any(
            row[field] not in _frame_choices(row["person_identity"], split)
            for field in ("alias_context", "second_alias_context")
            if field in row
        ):
            raise ValueError(
                "An ordinary pronoun subject cannot become a personal alias"
            )


def _forbidden_declaration(manifest):
    declared_split = manifest.get("split")
    suite = str(manifest.get("suite", "")).lower()
    return (
        declared_split is not None
        and declared_split not in {"train", "validation"}
        or manifest.get("independent_final") is True
        or re.search(r"(?:^|[_ -])(?:final|test|qualification)(?:$|[_ -])", suite)
    )


def _comparison_records(paths):
    # Preflight every declaration before the shared auditor can read any payload.
    legacy_declarations = {}
    for name in paths:
        path = Path(name)
        sidecar = path.with_suffix(path.suffix + ".manifest.json")
        if not sidecar.exists():
            declared = False
            for filename in ("manifest.json", "augmentation_manifest.json"):
                aggregate = path.parent / filename
                if not aggregate.exists():
                    continue
                payload = aggregate.read_bytes()
                document = json.loads(payload)
                if _forbidden_declaration(document):
                    raise ValueError(
                        "Qualification or undeclared corpora cannot be opened"
                    )
                for split, entry in document.get("datasets", {}).items():
                    if entry.get("file") != path.name:
                        continue
                    if split not in {"train", "validation"} or _forbidden_declaration(
                        entry
                    ):
                        raise ValueError(
                            "Qualification or undeclared corpora cannot be opened"
                        )
                    declared = True
                    legacy_declarations[str(path.resolve())] = {
                        "legacy_split_declaration": str(aggregate.resolve()),
                        "legacy_split_declaration_sha256": hashlib.sha256(
                            payload
                        ).hexdigest(),
                    }
                    break
                if declared:
                    break
            if not declared:
                raise ValueError("Qualification or undeclared corpora cannot be opened")
            continue
        manifest = json.loads(sidecar.read_bytes())
        if _forbidden_declaration(manifest):
            raise ValueError("Qualification or undeclared corpora cannot be opened")
        declared_split = manifest.get("split")
        if declared_split is None:
            conversion = manifest.get("training_conversion", {})
            if (
                conversion.get("declared") is True
                and conversion.get("source_suite") == "development"
                and conversion.get("source_split") == "validation"
            ):
                declared_split = "train"
            else:
                # These original authored coverage banks predate split fields in
                # adjacent sidecars; their aggregate declares the exact hash.
                for filename in ("context_manifest.json", "coverage_manifest.json"):
                    aggregate = path.parent / filename
                    if not aggregate.exists():
                        continue
                    payload = aggregate.read_bytes()
                    document = json.loads(payload)
                    if _forbidden_declaration(document):
                        raise ValueError(
                            "Qualification or undeclared corpora cannot be opened"
                        )
                    for split, entry in document.get("datasets", {}).items():
                        if entry.get("file") == path.name and _forbidden_declaration(
                            entry
                        ):
                            raise ValueError(
                                "Qualification or undeclared corpora cannot be opened"
                            )
                        if (
                            split in {"train", "validation"}
                            and entry.get("file") == path.name
                            and entry.get("sha256") == manifest.get("sha256")
                            and entry.get("count") == manifest.get("rows")
                        ):
                            declared_split = split
                            legacy_declarations[str(path.resolve())] = {
                                "legacy_split_declaration": str(aggregate.resolve()),
                                "legacy_split_declaration_sha256": hashlib.sha256(
                                    payload
                                ).hexdigest(),
                            }
        if declared_split not in {"train", "validation"}:
            raise ValueError("Qualification or undeclared corpora cannot be opened")
    records = audit.comparison_hashes(paths)
    for record in records:
        legacy = legacy_declarations.get(record["path"])
        if legacy is not None:
            if (
                hashlib.sha256(
                    Path(legacy["legacy_split_declaration"]).read_bytes()
                ).hexdigest()
                != legacy["legacy_split_declaration_sha256"]
            ):
                raise ValueError("Legacy comparison declaration changed during audit")
            record.update(legacy)
    return records


def write_dataset(
    output="data",
    train_count=2000,
    val_count=500,
    seed=SEED,
    *,
    compare_corpora=(),
    max_tokens=512,
):
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 8 or count % 4
        for count in (train_count, val_count)
    ):
        raise ValueError(
            "Each split needs an integer note count divisible by four and at least eight"
        )
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens < 100
    ):
        raise ValueError("Token ceiling must be an integer at least 100")
    comparisons = _comparison_records(compare_corpora)
    forbidden_text = set().union(*(item["text_hashes"] for item in comparisons))
    forbidden_ids, prior_names = set(), set()
    for record in comparisons:
        payload = Path(record["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(
                "Approved comparison corpus changed during ownership audit"
            )
        for line in payload.splitlines():
            row = json.loads(line)
            forbidden_ids.add(row["id"])
            prior_names.update(
                canonical(row["text"][span["start"] : span["end"]])
                for span in row["spans"]
                if span["label"] == "NAME"
            )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        split: output
        / f"nominal_alias_{'train' if split == 'train' else 'validation'}.jsonl"
        for split in CAPTIONS
    }
    if any(
        path.exists() or path.with_suffix(".jsonl.manifest.json").exists()
        for path in paths.values()
    ):
        raise FileExistsError("Authored corpora are immutable; choose a new directory")
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(".cache/tiktoken").resolve()))
    encoding = tiktoken.get_encoding("o200k_base")
    seen_texts, seen_ids, train_full, split_aliases, manifests, drafts = (
        set(),
        set(),
        set(),
        {},
        {},
        [],
    )
    try:
        for split, count in (("train", train_count), ("validation", val_count)):
            path = paths[split]
            draft = path.with_suffix(".jsonl.tmp")
            drafts.append(draft)
            labels, tags, traits, context_counts, profile_counts, reference_counts = (
                Counter() for _ in range(6)
            )
            variant_tags = {}
            full_names, aliases, lengths, retries = set(), set(), [], 0
            pair_count = count // 4
            unanchored_pairs = count // 4
            with draft.open("x", encoding="utf-8") as handle:
                for unit in range(pair_count + unanchored_pairs):
                    paired = unit < pair_count
                    for attempt in range(80):
                        rows = (
                            generate_pair(unit, split=split, seed=seed, attempt=attempt)
                            if paired
                            else generate_unanchored_pair(
                                unit - pair_count,
                                split=split,
                                seed=seed,
                                attempt=attempt,
                            )
                        )
                        statistics = [audit.token_audit(row, encoding) for row in rows]
                        identities = [
                            rows[0]["person_identity"],
                            rows[0]["other_identity"],
                        ]
                        private_full = {
                            canonical(person["full"])
                            for person in identities
                            if person is not None and person["full"] is not None
                        }
                        if max(
                            length for length, _ in statistics
                        ) <= max_tokens and not private_full & (
                            prior_names
                            | full_names
                            | (train_full if split == "validation" else set())
                        ):
                            break
                    else:
                        raise ValueError(
                            "Could not author short, fresh, split-disjoint identities"
                        )
                    retries += attempt
                    full_names.update(private_full)
                    for row, (length, row_tags) in zip(rows, statistics):
                        validate_row(row)
                        text_hash = hashlib.sha256(row["text"].encode()).hexdigest()
                        if (
                            text_hash in seen_texts | forbidden_text
                            or row["id"] in seen_ids | forbidden_ids
                        ):
                            raise ValueError(
                                "Authored note text/ID overlaps a split or approved corpus"
                            )
                        seen_texts.add(text_hash)
                        seen_ids.add(row["id"])
                        lengths.append(length)
                        labels.update(span["label"] for span in row["spans"])
                        tags.update(row_tags)
                        variant_tags.setdefault(row["variant"], Counter()).update(
                            row_tags
                        )
                        traits[row["variant"] + "_notes"] += 1
                        profile_counts[str(row["clinical_profile"])] += 1
                        reference_counts[str(row["public_reference_context"])] += 1
                        if paired:
                            context_counts[f"caption-{row['caption_context']}"] += 1
                        if row["variant"] in {"nominal", "unanchored"}:
                            context_counts[f"alias-{row['alias_context']}"] += 1
                            if row["variant"] == "unanchored":
                                context_counts[
                                    f"unanchored-alias-{row['alias_context']}"
                                ] += 1
                                context_counts[
                                    f"unanchored-alias-{row['second_alias_context']}"
                                ] += 1
                                traits["unanchored_short_name_notes"] += 1
                            traits["bare_nominal_alias_notes"] += (
                                row["alias_context"] < 18
                            )
                            traits["common_nominal_alias_notes"] += row[
                                "person_identity"
                            ]["common"]
                            traits["bare_common_nominal_alias_notes"] += (
                                row["person_identity"]["common"]
                                and row["alias_context"] < 18
                            )
                            traits["pronoun_spelling_nominal_notes"] += row[
                                "person_identity"
                            ]["pronoun"]
                            traits["shared_owner_alias_notes"] += row[
                                "shared_first_alias"
                            ]
                        if row["spans"]:
                            name_spans = [
                                span for span in row["spans"] if span["label"] == "NAME"
                            ]
                            for span, mention in zip(name_spans, row["name_mentions"]):
                                value = row["text"][span["start"] : span["end"]]
                                traits["name_spans"] += 1
                                traits[mention["kind"] + "_name_spans"] += 1
                                traits["lowercase_name_spans"] += value.islower()
                                traits["non_ascii_name_spans"] += not value.isascii()
                                traits["nfd_name_spans"] += (
                                    value != unicodedata.normalize("NFC", value)
                                )
                                if mention["kind"] != "full":
                                    aliases.add(canonical(value))
                                    traits["common_reduced_name_spans"] += (
                                        canonical(value) in _COMMON_KEYS
                                    )
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if split == "train":
                train_full = full_names
            split_aliases[split] = aliases
            ordered = sorted(lengths)
            manifests[split] = {
                "schema_version": 1,
                "generator": "clinical_deid.nominal_alias_synthetic",
                "generator_version": VERSION,
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "shared_annotation_audit_helper_sha256": hashlib.sha256(
                    Path(audit.__file__).read_bytes()
                ).hexdigest(),
                "seed": seed,
                "rows": count,
                "sha256": hashlib.sha256(draft.read_bytes()).hexdigest(),
                "split": split,
                "suite": f"authored_nominal_alias_{'training' if split == 'train' else 'development'}",
                "license": "CC0-1.0",
                "code_license": "Apache-2.0",
                "external_datasets": [],
                "source": "Original authored nominal human-alias/ordinary-function contrasts and productive fresh identities; shared owned offset/token/approved-comparison auditor only. No copied retired sentences, names, generator or qualification input.",
                "used_for_training": split == "train",
                "used_for_checkpoint_selection": split == "validation",
                "independent_final": False,
                "development_scope": "New authored nominal-alias coverage, reported separately; not independent qualification.",
                "anchored_contrast_pairs": pair_count,
                "unanchored_positive_notes": unanchored_pairs,
                "phi_free_notes": unanchored_pairs,
                "label_support": {label: labels[label] for label in ENTITY_TYPES},
                "bioes_tag_support": dict(sorted(tags.items())),
                "bioes_tag_support_by_variant": {
                    variant: dict(sorted(counts.items()))
                    for variant, counts in sorted(variant_tags.items())
                },
                "name_and_contrast_traits": dict(sorted(traits.items())),
                "context_note_support": dict(sorted(context_counts.items())),
                "clinical_profile_note_support": dict(sorted(profile_counts.items())),
                "public_reference_note_support": dict(sorted(reference_counts.items())),
                "tokenizer": "o200k_base / OPF original Unicode character projection",
                "token_length": {
                    "min": min(lengths),
                    "mean": sum(lengths) / len(lengths),
                    "p95": ordered[int(0.95 * (len(ordered) - 1))],
                    "max": max(lengths),
                    "ceiling": max_tokens,
                },
                "annotation_audit": {
                    "rows_validated": count,
                    "token_aligned_rows": count,
                    "gold_overlap": 0,
                    "shared_phi_tokens": 0,
                    "duplicate_texts": 0,
                    "length_or_identity_retries": retries,
                },
                "private_value_policy": "Explicit full private identities are NFC/casefold-disjoint across splits and exclude previously annotated NAME values. Common English reduced aliases and newly productive short spellings may overlap earlier NAME values; no identity lookup is used. Unanchored singleton mentions make no cross-note owner claim. Clinical/public vocabulary may overlap.",
                "private_values": {
                    "distinct_full_identities": len(full_names),
                    "distinct_reduced_aliases": len(aliases),
                    "prior_annotated_name_full_identity_overlap": 0,
                },
                "cross_split_audit": {
                    "id_overlap": 0,
                    "exact_text_overlap": 0,
                    "full_identity_overlap": 0,
                },
                "clinical_coherence": {
                    "anchored_pair_full_identity_fields_preserved": True,
                    "ordinary_pronouns_and_public_occurrences_unlabelled_after_anchors": True,
                    "shared_alias_owners_declared_without_disambiguation": True,
                    "private_followup_is_14_days_after_assessment": True,
                    "public_reference_year_precedes_assessment_year": True,
                    "normal_ages_labs_doses_durations_scales_public_dates_unlabelled": True,
                },
                "comparison_audit": [
                    {
                        key: value
                        for key, value in record.items()
                        if key != "text_hashes"
                    }
                    | {"exact_text_overlap": 0, "exact_id_overlap": 0}
                    for record in comparisons
                ],
            }
        overlap = split_aliases["train"] & split_aliases["validation"]
        for manifest in manifests.values():
            manifest["cross_split_audit"]["shared_reduced_alias_values"] = len(overlap)
            manifest["cross_split_audit"]["shared_common_word_aliases"] = sorted(
                overlap & _COMMON_KEYS
            )
        for split, path in paths.items():
            draft = path.with_suffix(".jsonl.tmp")
            os.link(draft, path)
            draft.unlink()
            with path.with_suffix(".jsonl.manifest.json").open(
                "x", encoding="utf-8"
            ) as handle:
                json.dump(manifests[split], handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        return manifests
    finally:
        for draft in drafts:
            draft.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--compare-corpus", action="append", default=[])
    args = parser.parse_args()
    result = write_dataset(
        args.output,
        args.train_count,
        args.val_count,
        args.seed,
        compare_corpora=args.compare_corpus,
        max_tokens=args.max_tokens,
    )
    print(
        json.dumps(
            {
                split: {
                    key: value
                    for key, value in manifest.items()
                    if key
                    in {
                        "rows",
                        "sha256",
                        "anchored_contrast_pairs",
                        "phi_free_notes",
                        "token_length",
                        "name_and_contrast_traits",
                        "cross_split_audit",
                    }
                }
                for split, manifest in result.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
