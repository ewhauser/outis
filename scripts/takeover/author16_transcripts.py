#!/usr/bin/env python3
"""Deterministic generator for tk16: phone and telehealth transcripts.

Writes data/tk16_train_transcripts.jsonl and its manifest. Every value is
invented (reserved phone block, reserved domains, TEST-NET addresses). The
conventions are the transcript decisions T1 and T2 in
docs/annotation-policy-v2.md. Every row passes the blind-bank exclusion filter
(inserted values and every 1-8 word sequence of the text); only counts are
recorded, never the strings that matched.

Usage:
  PYTHONPATH=src python scripts/takeover/author16_transcripts.py [--rows 1500] [--seed 16]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import statistics
import string
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / ".cache" / "tiktoken"))

from outis.data.annotation_policy import (  # noqa: E402
    POLICY_VERSION,
    annotation_errors,
    canonical_identity_group,
    identity_split,
)
from outis.data.generators.places import (  # noqa: E402
    excluded_sequences,
    excluded_value,
)

BANK = "tk16_train_transcripts"
OUT_JSONL = ROOT / "data" / "tk16_train_transcripts.jsonl"
DEFAULT_ROWS = 1500
DEFAULT_SEED = 16
GENERATOR_VERSION = "tk16-v1"

FAMILY_SHARES = (("telehealth", 0.35), ("phone", 0.25), ("interpreter", 0.10),
                 ("coordination", 0.15), ("stt", 0.15))
PHI_FREE_SHARE = 0.20
CASES = ("normal", "lower", "upper")
FMTS = ("name_full", "name_first", "name_last", "title_last", "role_paren", "ts", "name_role")
LENGTH_CLASSES = (("short", 300, 600, 0.15), ("medium", 600, 1000, 0.30),
                  ("long", 1000, 1700, 0.35), ("xlong", 1700, 2400, 0.20))

MARK = re.compile(r"\[\[([A-Z]+)\|([^\[\]]+)\]\]")


def M(label: str, value: str) -> str:
    return f"[[{label}|{value}]]"


class Exhausted(Exception):
    """A value pool has no value left under the repetition cap."""


class Redraw(Exception):
    """The draft is rejected and drawn again."""


# --------------------------------------------------------------------------
# invented names

_STYLES = {
    "celtic": (
        "Tav Cor Wen Mar Bran Dun Gar Kel Lor Mor Nev Ros Tor Ven Wyn Ash Bryn Cael Dar Eir Fen Gael Haw Kes Lenn Mael Orin Pen Quill Rhys Cad Dov Elm Fal Gow Hal Ker Lis Nor Oss".split(),
        "a ine wen ric an ys ith en ell ia ey lyn ard".split(),
        "wick nock ford ley mere ton well by ham ock ough dale wood stone".split()),
    "romance": (
        "Al Bel Cam Cel Del Es Fab Gal Hel Is Jul Lor Mar Nad Or Pal Ren Sab Tel Val Vel Zan Ant Bru Cos Dom Eli Fio Gio Lu Mat Nic Pas Rom Sol Tib Ugo Vit Rav Ces".split(),
        "ina ia ela ora ana ito ero elo isa ena etta ian aro".split(),
        "etti ardi ini oni ano ero ez oza ales eda ucci elli ari aldo".split()),
    "slavic": (
        "Bog Dra Gor Ila Jan Kas Len Mir Nad Ol Pav Rad Sta Tom Vas Zor Bel Cher Dob Kal Lub Mal Nik Ost Pet Rok Sem Tur Vlad Yar Zel Hal Ivo Ras Sav".split(),
        "a ka mir ina ek an ya slav ush ik ana ena ko".split(),
        "ski ova enko ak ek ich chuk off ev ner vik owicz".split()),
    "southasian": (
        "Ar Bha Cha Dev Gau Har Ish Jay Kav Lak Man Nir Pra Rav San Tar Uma Vik Aru Bal Dip Gir Kir Mah Nan Pad Rag Shr Tej Vij Yas Anj Dha Moh Sur".split(),
        "a ika ini esh ita an ana ya vi ul ish ush ti".split(),
        "wal kar pati rajan sen nathan dhar mani rao jit lingam pal wani doss".split()),
    "eastasian": (
        "Ah Bai Chi Dan Fu Gao Han Jin Kai Lin Mei Nao Qin Ren Sai Tan Wei Xia Yun Zhi Hiro Kana Mina Nori Taka Yuki Haru Ryo Tomo Shin Dai Ume Aki Sora".split(),
        "ko mi ra ne to ki yo ru ma sa".split(),
        "moto shi gawa yama rin hara zuki mura nari".split()),
    "westafrican": (
        "Ade Bayo Chi Dayo Efe Fola Gbe Ife Kwa Lemi Mba Ndu Obi Sefa Tunde Uche Yaw Zola Amah Nana Oye Kemi Seun Tobi Wale Ayo Eme Ibe Kolo Moyo Olu Ruka Temi".split(),
        "a ola dele mi chi ike yemi ndu bola ke".split(),
        "yemi sola kunle wande ojo ebe mbe nwosu dele bisi".split()),
    "arabic": (
        "Am Bas Dal Far Ghal Had Ibt Jam Kar Lat Mas Nas Rah Sam Tar Wal Yas Zah Sal Nab Jab Khal Hab Mun Raz Suh Tam Ham".split(),
        "a im ira een ar an id ia ma".split(),
        "ouni adi ari awi ani oud ek idi aki oun".split()),
    "nordic": (
        "Ing Hel Sig Tor Bjor Eri Gun Hal Kar Mag Ols Ras Sten Tyr Vil Ulf Ake Dag Frey Aud Bri Eld Fre Gis Ket Lei Nils Rol Sve".split(),
        "a vid ric dis borg ild un ve gerd mar".split(),
        "strand gren holm lund quist berg dahl vik sen mark".split()),
}
_STYLE_KEYS = tuple(_STYLES)
_ACCENT = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú"}


def _accent(tok: str, rng: random.Random) -> str:
    idx = [i for i, ch in enumerate(tok) if ch in _ACCENT and i > 0]
    if not idx:
        return tok
    i = rng.choice(idx)
    return tok[:i] + _ACCENT[tok[i]] + tok[i + 1:]


def gen_given(rng: random.Random) -> str:
    on, g_end, _ = _STYLES[rng.choice(_STYLE_KEYS)]
    onset = rng.choice(on)
    ending = rng.choice(_STYLES[rng.choice(_STYLE_KEYS)][1] if rng.random() < 0.25 else g_end)
    name = onset + ending
    return _accent(name, rng) if rng.random() < 0.06 else name


def gen_surname(rng: random.Random) -> str:
    on, _, s_end = _STYLES[rng.choice(_STYLE_KEYS)]
    onset = rng.choice(on)
    name = onset + rng.choice(_STYLES[rng.choice(_STYLE_KEYS)][2] if rng.random() < 0.25 else s_end)
    r = rng.random()
    if r < 0.05:
        name = name + "-" + rng.choice(_STYLES[rng.choice(_STYLE_KEYS)][0]) + rng.choice(_STYLES[rng.choice(_STYLE_KEYS)][2])
    elif r < 0.08:
        name = rng.choice(["O'", "D'", "Mc"]) + name
    elif r < 0.13:
        name = _accent(name, rng)
    return name


def respell(tok: str, rng: random.Random) -> str:
    """A speech-to-text mishearing of a name; the occurrence is still a name."""
    for _ in range(4):
        r = rng.random()
        if r < 0.45:
            idx = [i for i, ch in enumerate(tok) if ch.lower() in "aeiou" and i > 0]
            if idx:
                i = rng.choice(idx)
                new = rng.choice([v for v in "aeiou" if v != tok[i].lower()])
                return tok[:i] + new + tok[i + 1:]
        elif r < 0.75:
            idx = [i for i, ch in enumerate(tok) if ch.lower() in "bcdfglmnprstvz" and i > 0]
            if idx:
                i = rng.choice(idx)
                return tok[:i] + tok[i] + tok[i:]
        elif len(tok) > 5:
            return tok[:-1]
    return tok


# --------------------------------------------------------------------------
# invented places, facilities, identifiers

_STATES = "UT NM OR MN WI NH ID MT NE KS IA VT ME WA CO AZ OK AR MO OH PA NC SC TN KY IN MI NY NJ MA".split()
_STEM_A = "Ash Bren Cal Dun Elm Fair Glen Hart Iver Kest Linden Marl North Oak Pine Quin Rook Sand Thorn Under Vale West Yew Wren Aln Bar Cor Dal Eske Frim Gor Hol Ing Jun Kel".split()
_STEM_B = "field mere ham wick brook haven ridge crest gate stead hollow bury ford moor vale wood combe shaw worth".split()
_ADJ = "Silver Golden Quiet Bright Stone Willow Cobalt Amber Juniper Heron Linnet Plover Sparrow Marsh Fallow Gentle Maple Lantern Harbor Copper Misty Rowan Sable Tamarack Ivory".split()
_NOUN = "Finch Lake Meadow Harbor Bridge Orchard Terrace Springs Landing Pointe Grove Fields Shore Hill Crossing Reach".split()
_SAINTS = "Odile Brannoc Ciaran Elowen Tiber Maren Lucan Isolde Aldric Verena Hesper Anselm Perpetua Corvin".split()
_PLACE_SUF = ["Falls", "Heights", "Junction", "Springs", "Park", "Crossing", "Landing", "Corners", "Station"]
_STREET = "Linden Sparrow Juniper Marlow Alder Quarry Heron Cobble Fairmont Sorrel Tamarack Wicklow Hollis Ferndale Bramble Meridian Orchard Saltmarsh Kestrel Aspen Birchwood Primrose Thistle Windmere Elder Foxglove Cardinal Plover Bluff Mill".split()
_STREET_T = "Way Lane Court Drive Road Street Avenue Terrace Place Circle Boulevard".split()
_FAC_TYPES = {
    "clinic": ["Family Medicine", "Health Center", "Medical Group", "Primary Care Associates", "Internal Medicine", "Clinic", "Family Practice"],
    "hosp": ["Medical Center", "Regional Hospital", "Memorial Hospital", "General Hospital", "Community Hospital", "University Hospital"],
    "pharm": ["Drug", "Pharmacy", "Apothecary", "Family Pharmacy", "Community Pharmacy"],
    "lab": ["Diagnostic Laboratory", "Laboratory Services", "Imaging Center", "Outpatient Lab"],
    "hh": ["VNA", "Home Health", "Visiting Nurse Association", "Home Health Services"],
    "hospice": ["Hospice", "Hospice and Palliative Care", "Hospice Services"],
    "dme": ["Medical Supply", "Home Medical Equipment", "Respiratory and Mobility Supply"],
    "adc": ["Adult Day Program", "Adult Day Center", "Adult Day Health"],
    "sl": ["Senior Living", "Assisted Living", "Retirement Community", "Gardens Senior Living"],
    "snf": ["Rehabilitation and Nursing Center", "Care Center", "Skilled Nursing and Rehab", "Health and Rehabilitation"],
    "mail": ["Mail Pharmacy", "Specialty Pharmacy", "Home Delivery Pharmacy"],
    "urgent": ["Urgent Care", "Walk-in Clinic", "Express Care"],
    "funeral": ["Funeral Home", "Funeral Chapel"],
}
_AREA = "202 212 213 303 305 312 314 404 415 502 503 505 512 602 612 615 617 702 713 801 808 816 919 971 406 218 701 605 307 207 802 603 509 360 541 435 575 316 402 515".split()
_MONTHS = ("January February March April May June July August September October November December").split()
_WEEKDAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def gen_stem(rng):
    return rng.choice(_STEM_A) + rng.choice(_STEM_B)


def gen_locality(rng, with_state=None):
    r = rng.random()
    if r < 0.5:
        base = gen_stem(rng)
    elif r < 0.8:
        base = gen_stem(rng) + " " + rng.choice(_PLACE_SUF)
    else:
        base = rng.choice(["North", "East", "West", "South", "Upper", "Lower"]) + " " + gen_stem(rng)
    if with_state is None:
        with_state = rng.random() < 0.55
    return f"{base}, {rng.choice(_STATES)}" if with_state else base


def gen_street_address(rng, unit=None):
    num = rng.randrange(100, 9900)
    if rng.random() < 0.55:
        street = f"{num} {rng.choice(_STREET)} {rng.choice(_STREET_T)}"
    else:
        street = f"{num} {gen_stem(rng)} {rng.choice(_STREET_T)}"
    if unit is None:
        unit = rng.random() < 0.3
    if unit:
        street += rng.choice([f" Apt {rng.randrange(1, 40)}", f" Unit {rng.randrange(1, 300)}", f" #{rng.randrange(1, 90)}{rng.choice('ABC')}"])
    city = gen_locality(rng, with_state=False)
    return f"{street}, {city}, {rng.choice(_STATES)} {rng.randrange(10000, 99999)}"


def gen_facility(rng, kind):
    typ = rng.choice(_FAC_TYPES[kind])
    r = rng.random()
    if r < 0.38:
        head = gen_stem(rng)
    elif r < 0.68:
        head = f"{rng.choice(_ADJ)} {rng.choice(_NOUN)}"
    elif r < 0.82:
        head = f"Saint {rng.choice(_SAINTS)}"
    else:
        head = gen_surname(rng).replace("'", "")
    if kind == "sl" and rng.random() < 0.3:
        return f"The {head} at {gen_stem(rng)} Ridge"
    return f"{head} {typ}"


def gen_phone(rng, ext=False):
    a, n = rng.choice(_AREA), f"555-01{rng.randrange(100):02d}"
    base = rng.choice(["({a}) {n}", "{a}-{n}", "{a}.{n}", "{a} {n}", "+1 {a} {n}", "+1 ({a}) {n}", "{a}-{n}", "({a}) {n}"]).format(a=a, n=n)
    if ext:
        e = rng.randrange(100, 9999)
        base += rng.choice([f", extension {e}", f" ext. {e}", f" x{e}", f" EXT {e}", f", ext {e}", f" extension {e}"])
    return base


def gen_email(rng, first, last):
    local = rng.choice([f"{first}.{last}", f"{first[0]}{last}", f"{first}{rng.randrange(10, 99)}", f"{last}.{first[0]}"])
    local = unicodedata.normalize("NFKD", local).encode("ascii", "ignore").decode().replace("'", "").lower()
    return local + "@" + rng.choice(["mail.example", "inbox.example.com", "post.example.org", "webmail.example.net", "home.example", "mail.test"])


def gen_mrn(rng):
    if rng.random() < 0.4:
        return "".join(rng.choice(string.ascii_uppercase) for _ in range(2)) + "-" + f"{rng.randrange(10**7):07d}"
    width = rng.choice([7, 8])
    return f"{rng.randrange(10**width):0{width}d}"


def gen_code(rng, kind):
    up = "ABCDEFGHJKMNPQRSTVWXYZ"
    if kind == "session":
        return rng.choice([f"{rng.choice(up)}{rng.choice(up)}-{rng.randrange(1000, 999999)}",
                           f"S{rng.randrange(10**7, 10**9)}", f"TH-{rng.randrange(2023, 2027)}-{rng.randrange(1000, 99999)}",
                           f"{rng.choice(up)}{rng.randrange(100, 999)}-{rng.choice(up)}{rng.choice(up)}{rng.randrange(10, 99)}"])
    if kind == "activation":
        al = up + "23456789"
        return "-".join("".join(rng.choice(al) for _ in range(4)) for _ in range(2))
    if kind == "member":
        return rng.choice([f"{rng.choice(up)}{rng.randrange(10**7, 10**9)}", f"{rng.choice(up)}{rng.choice(up)}{rng.randrange(10**8, 10**10)}"])
    if kind == "auth":
        return rng.choice([f"PA-{rng.randrange(2023, 2027)}-{rng.randrange(10000, 99999)}", f"AU{rng.randrange(10**7, 10**9)}"])
    if kind == "referral":
        return rng.choice([f"REF-{rng.randrange(10000, 999999)}", f"R{rng.randrange(10**6, 10**8)}"])
    if kind == "case":
        return rng.choice([f"CM-{rng.randrange(10000, 99999)}", f"{rng.choice(up)}{rng.choice(up)}{rng.randrange(1000, 999999)}"])
    if kind == "rx":
        return str(rng.randrange(10**6, 10**8))
    if kind == "account":
        return str(rng.randrange(10**7, 10**10))
    if kind == "interp":
        return str(rng.randrange(1000, 99999))
    raise KeyError(kind)


def gen_url(rng):
    host = rng.choice(["visit.example.test", "meet.clinic.test", "join.example.org", "video.example.com", "room.example.net", "tele.example.test"])
    tail = rng.choice([f"r/{rng.randrange(1000, 99999)}",
                       f"room/{''.join(rng.choice(string.ascii_lowercase) for _ in range(4))}-{rng.randrange(1000, 9999)}",
                       f"v/{''.join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(8))}", f"j/{rng.randrange(10**8, 10**10)}"])
    return f"{rng.choice(['https://', 'https://', 'http://'])}{host}/{tail}"


def gen_ip(rng):
    block = rng.choice(["192.0.2", "198.51.100", "203.0.113"])
    return f"{block}.{rng.randrange(1, 254)}"


def rand_date(rng, lo, hi):
    return lo + dt.timedelta(days=rng.randrange((hi - lo).days + 1))


def fmt_date(d: dt.date, style: str) -> str:
    mon, wd = _MONTHS[d.month - 1], _WEEKDAYS[d.weekday()]
    return {
        "mdy_ord": f"{mon} {ordinal(d.day)}, {d.year}",
        "mdy": f"{mon} {d.day}, {d.year}",
        "num": f"{d.month:02d}/{d.day:02d}/{d.year}",
        "dmy": f"{d.day} {mon} {d.year}",
        "iso": d.isoformat(),
        "rfc": f"{wd[:3]}, {d.day} {mon[:3]} {d.year}",
        "dow_ord": f"{wd} {mon} {ordinal(d.day)}",
        "dow": f"{wd}, {mon} {d.day}",
        "short": f"{mon} {ordinal(d.day)}",
        "num_year": f"{d.month}/{d.day}/{d.year}",
        "month_year": f"{mon} {d.year}",
    }[style]


def spell_age(n: int) -> str:
    if n >= 100:
        return "one hundred" if n == 100 else f"one hundred and {spell_age(n - 100)}"
    if n < 20:
        return _ONES[n]
    tens = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}[n // 10]
    return tens if n % 10 == 0 else f"{tens}-{_ONES[n % 10]}"


_DRUGS = ("metformin lisinopril atorvastatin amlodipine sertraline levothyroxine omeprazole gabapentin albuterol losartan metoprolol hydrochlorothiazide "
          "prednisone amoxicillin ibuprofen naproxen tamsulosin apixaban furosemide cetirizine montelukast escitalopram duloxetine cyclobenzaprine "
          "nitrofurantoin sumatriptan azithromycin clopidogrel pantoprazole warfarin acetaminophen fluticasone meloxicam trazodone").split()
_DOSES = ["5 milligrams", "10 milligrams", "20 milligrams", "25 milligrams", "40 milligrams", "50 milligrams", "100 milligrams", "250 milligrams", "500 milligrams"]
_SYMS = ["headaches", "cough", "chest tightness", "knee pain", "rash", "sore throat", "dizziness", "nausea", "back pain", "shortness of breath",
         "ankle swelling", "fatigue", "trouble sleeping", "heartburn", "stomach pain", "fever", "ear pain", "hand numbness", "palpitations", "wheezing",
         "burning with urination", "shoulder pain", "constipation", "eye irritation", "muscle aches", "congestion", "leg cramps", "lightheadedness"]
_DURS = ["two days", "a week", "three weeks", "a few months", "two months", "ten days", "five days", "a couple of weeks", "four days", "a month"]
_RELDATES = ["three days ago", "yesterday", "last week", "about two weeks ago", "on Monday", "this morning", "over the weekend", "a couple of days ago", "last night", "on Friday", "earlier this week"]
_LANGS = ["Spanish", "Vietnamese", "Mandarin", "Somali", "Arabic", "Russian", "Haitian Creole", "Tagalog", "Hmong", "Nepali", "Portuguese", "Korean", "Amharic", "Swahili", "Cantonese", "Burmese", "Farsi", "Ukrainian"]
_MEDIA = ["the Evening Ledger news hour", "the Sunday Wellness Hour on the radio", "a podcast called Quiet Hours", "the Morning Table cooking show",
          "a video channel about sleep", "the Harbor Lights weekly newsletter", "a documentary about gardening", "a talk radio program about retirement"]
_FAMILY_RELS = ["daughter", "son", "wife", "husband", "sister", "brother", "granddaughter", "grandson", "niece", "nephew", "mother", "daughter-in-law"]


# --------------------------------------------------------------------------
# dialogue pools. A variant is a block of "R: text" lines; variants are split by "---".
# Roles: C clinician, P patient, N nurse, I interpreter, F family member, G second
# family member or neighbor, S staff, M case manager or social worker, A agency
# representative, W social worker, H pharmacist, D on-call clinician.
# Slots in braces are filled per document or per exchange. Role-only text never
# names a person; only the slots below create name spans.

POOL_TEXT: dict[str, str] = {}

POOL_TEXT["T_TECH"] = """
C: Good afternoon, can you hear me okay?
P: I can hear you, but the picture keeps freezing.
C: Let me turn my camera off for a second and see if that helps the connection. There we go, is that better?
P: Yes, much better now, thank you.
---
C: I'm showing you in the waiting room on session {sess}. Give me one moment to admit you.
P: Okay, I'm here. Can you see me?
C: I can see you, and your audio is coming through clearly.
---
P: The link you texted me, {url}, opened right away on my phone.
C: Great. If it drops, just tap the same link again and it will put you back in the room.
---
C: Before we begin, the system shows app version {ver}. If the video lags, we can switch to a regular phone call.
P: Let's try the video first.
---
C: The connection log says you joined from {ip}, and it looks stable on our side.
P: Good. My router has been acting up, so I moved closer to it.
---
P: Sorry, I'm still looking for the button to turn on my microphone.
C: It's at the bottom of the screen, the little microphone. Tap it once.
P: There it is. Can you hear me now?
C: Loud and clear.
---
C: Can you hold the phone a little further back so I can see your face?
P: Is this better? I'm propping it against a stack of books.
C: Perfect, I can see you well.
---
C: We're on a secure link, session {sess}. If we get disconnected I'll call you right back at the number on file.
P: Okay, that sounds fine.
---
C: Is the camera working on your end? I only see a black square.
P: Hold on, there's a sticker over it. Let me peel it off.
C: There you are.
---
C: You're breaking up a little. Do you have headphones you can plug in?
P: I do, give me a second to find them.
C: Take your time.
---
P: It says to update the app before joining. Should I do that now?
C: It might take a minute. If it asks, accept the update to version {ver} and then use the same link, {url}, to come back in.
P: Okay, back in a minute.
"""

POOL_TEXT["T_INTRO"] = """
C: Hello, I'm {c_drfull}. I'll be seeing you today instead of your regular doctor.
P: Hi {c_dr}, nice to meet you.
---
C: Good morning, this is {c_dr}. Is this {pt_hon}?
P: Yes, speaking.
---
C: Hi {pt_first}, it's {c_first} from the clinic. Can you see and hear me?
P: I can, hi {c_first}.
---
N: Hello, I'm {n_nurse}, I'll get you checked in before {c_dr} joins.
P: Thanks, I'm ready.
---
C: Hi, I'm {c_dr}. I'm on the screen with {pt_full}, correct?
P: That's me.
---
C: Welcome, I'm the covering physician today. Thanks for your patience.
P: No problem at all.
---
C: Hello, I'm the clinician who will be doing your visit today. Is this a good time?
P: Yes, I cleared my afternoon.
---
C: Good afternoon, {pt_hon}. This is {c_dr} returning to our video room. Sorry for the wait.
P: That's okay, I know you're busy.
"""

POOL_TEXT["T_VERIFY"] = """
C: Before we go on, can you state your full name and date of birth for me?
P: {pt_full}, born {dob}.
C: Thank you. And the address we have on file is {addr}, is that still right?
P: Yes, that's correct.
---
C: For verification, please tell me your date of birth.
P: It's {dob}.
C: And the best phone number to reach you if we get cut off?
P: {phone}.
---
C: I want to be sure I have the right chart. Can you spell your last name?
P: It's {pt_last}, same as in the portal.
C: Thank you, {pt_first}. And your birthday?
P: {dob}.
---
C: Is your mailing address still {addr}?
P: No, we moved. It's {addr2} now.
C: Got it, I'll update that.
---
C: I'll confirm your member number for the visit. Is it {member}?
P: That matches my card.
---
C: Your record number on this screen reads {mrn}. Does that look right to you?
P: I don't know the number by heart, but the name is right.
C: That's fine.
---
C: Please read me the email on your portal account.
P: {email}.
C: Perfect, that's what I have.
---
C: I just need to confirm that the name and date of birth on the check-in form match your ID.
P: They do. Everything on the form is right.
C: Thank you.
---
C: Can you confirm you're physically in the state where you'll be receiving care today?
P: Yes, I'm at home.
C: Great.
---
C: And the phone number on file ends in {phone}, right?
P: Yes, the full number is {phone}, that's my cell.
"""

POOL_TEXT["T_CONSENT"] = """
C: This visit is billed like an in-person visit and we do not record the video. Do you consent to proceed by video?
P: Yes, I consent.
C: And where are you located right now, in case I need to send help?
P: At home in {loc}.
---
C: I need to confirm your location for safety. Are you at {addr}?
P: I'm at my {f_rel}'s place today, in {loc2}.
C: Thank you. Can you give me that street address in case of an emergency?
P: {addr2}.
---
C: Who should we call in an emergency?
P: My {f_rel}, {f_full}. The number is {phone2}.
C: Thank you, I've added that to your chart.
---
C: Do you agree to the telehealth consent form you signed in the portal?
P: Yes, I read it and signed it.
---
C: May a family member sit in on the visit today?
P: Yes, my {f_rel} {f_first} is here with me, off camera.
C: Welcome, {f_first}.
---
C: This session is not recorded, but the system keeps a log of the connection under session {sess}. Is that okay?
P: That's fine.
---
C: Is there anyone else in the room with you who can hear us?
P: Just my cat.
C: That's fine, as long as you're comfortable talking.
"""

POOL_TEXT["T_CC"] = """
C: What brings you in today?
P: I've had {sym} that started {reldate}, and it isn't getting any better.
---
C: Tell me what's been going on.
P: Mostly the {sym}. It started about {dur} ago and now I'm also noticing {sym2}.
---
C: So this is a follow-up for your {sym}, is that right?
P: Yes, and I wanted to ask about my {drug}, it's been upsetting my stomach.
---
P: I wasn't going to book a visit, but my family said the {sym} has gone on too long.
C: I'm glad you did. Let's go through it.
---
C: I see in the message you sent that you're worried about {sym}.
P: Yes, it comes and goes. Worse in the evening.
---
C: Is this a new problem or something we've talked about before?
P: A bit of both. The {sym} is familiar, but the {sym2} is new.
---
C: I have you down for a medication check and some questions about your {sym}.
P: That's right. I also have a question about my lab results.
"""


POOL_TEXT["T_HIST"] = """
C: Any surgeries in the past?
P: I had my gallbladder out in {year_hist}, and a knee scope in {year_hist2}.
---
C: When was your last colonoscopy?
P: {month_year}. They said it was normal and to come back in ten years.
---
C: And the last time you had bloodwork?
P: {date_past}, I think. The lab called the next morning.
---
C: How long have you had the diagnosis of high blood pressure?
P: Since {year_hist}. My doctor started me on {drug} right away.
---
C: Any hospitalizations?
P: One, at {fac_hosp}, back in {year_hist}. I stayed three nights.
---
C: When did you last see a specialist for this?
P: In {month_year}, at {fac_hosp}. They did an ultrasound.
---
C: When was your last flu shot?
P: {date_past}, at the pharmacy.
---
C: Family history of heart disease or diabetes?
P: My mother had diabetes, and my father had a heart attack at 61.
---
C: Do you have any allergies to medications?
P: Penicillin gives me hives, and nothing else that I know of.
---
C: Do you smoke or drink?
P: I quit smoking years ago and I have a glass of wine on the weekends.
---
C: Were you ever told you had a problem with your thyroid?
P: Yes, I was diagnosed in {year_hist}, and I've been on {drug} since.
---
C: When was your last mammogram or prostate screening, depending on what applies?
P: {month_year}. Everything was normal.
"""

POOL_TEXT["T_MEDS"] = """
C: Let's go over your medications. You're on {drug} {dose}, is that right?
P: Yes, once a day in the morning.
C: And {drug2} {dose2}?
P: That one is at bedtime.
---
C: Are you missing doses of the {drug}?
P: Maybe once or twice a week, I forget the evening one.
C: Try tying it to brushing your teeth.
---
C: Any side effects from the {drug}?
P: A dry cough sometimes.
C: That can happen. Let's talk about whether to switch.
---
C: Do you take any over the counter products or supplements?
P: Just {drug3} for pain and a multivitamin.
---
C: Is it okay if I send a new prescription for {drug} {dose}?
P: Yes, please.
---
P: I ran out of {drug} on {date_past} and haven't had it since.
C: We can refill it today.
---
C: What's the prescription number on your last bottle?
P: It says Rx {rx}.
C: Good, that's the one I see on my screen.
---
C: Are you taking the {drug} with food?
P: Most days. When I skip breakfast I get a little queasy.
---
C: Have you ever had a bad reaction to {drug3}?
P: Not that I remember.
---
C: How many pills are left in the bottle of {drug2}?
P: About ten, I think. Let me count. Eleven.
---
C: Do you use a pill organizer?
P: Yes, I fill it every Sunday night.
"""

POOL_TEXT["T_SOCIAL"] = """
C: Who's at home with you?
P: Just me and my {f_rel}, {f_first}.
C: Does {f_first} help with your medications?
P: Yes, {f_first} sets out the pills every Sunday.
---
C: Do you live alone?
P: Yes, but my {f_rel} lives in {loc2}, about twenty minutes away.
---
C: Are you still working?
P: Part time, at the front desk of a dental office.
---
C: Any stress at home or work?
P: My {f_rel} {f_first} moved in last month and things have been tense.
---
P: I walk every morning with a neighbor, {g_first}.
C: That's wonderful. Walking helps.
---
C: Have you been feeling down lately?
P: A bit. I haven't been sleeping and I don't feel like seeing anyone.
C: Thank you for telling me. We'll talk about that.
---
C: Do you have transportation to your appointments?
P: My {f_rel} {f_first} drives me.
"""

POOL_TEXT["T_EXAM"] = """
C: Can you hold the camera so I can see the area?
P: Like this?
C: A little closer. There is mild redness but no swelling that I can see.
---
C: Please take a deep breath in and out for me.
P: Okay.
C: I'm not hearing any wheeze through the microphone, but it's a limited exam.
---
C: Can you walk across the room for me?
P: Sure.
C: Your steady gait looks fine from here.
---
C: Press gently on your lower belly and tell me if it's tender.
P: No, not tender there. A little on the right side though.
---
C: Stick out your tongue and say ah.
P: Ahh.
C: Your throat looks a little red, no white patches.
---
C: Can you check your pulse at your wrist for me and count for fifteen seconds?
P: Okay. Eighteen.
---
C: Raise both arms over your head.
P: It pulls a little on the left.
---
C: Show me where it hurts.
P: Right here along the side.
"""

POOL_TEXT["T_PLAN"] = """
C: Here's what I'd suggest. Start {drug} {dose} daily, keep a symptom diary, and call if it isn't better in {dur}.
P: That sounds reasonable.
---
C: I'd like you to have labs drawn at {fac_lab} sometime in the next week.
P: Do I need to fast?
C: Yes, for eight hours.
---
C: I'm going to refer you to a specialist at {fac_hosp}, and their office will call you.
P: Will they call from a blocked number?
C: Sometimes, so please answer unknown numbers this week.
---
C: I'd like an X-ray, which you can get at {fac_lab} in {loc2}.
P: Do I need an appointment?
C: They take walk-ins until four.
---
C: I want you to stop the {drug} for now and see if the {sym} settles.
P: For how long?
C: Two weeks, then we'll check in.
---
C: Rest, fluids, and over the counter pain relievers.
P: Is {drug3} okay?
C: Yes, with food.
---
C: If you develop chest pain, trouble breathing, or fainting, go to the nearest emergency department.
P: Understood.
---
C: I'll put in an order for physical therapy. They'll schedule you for an evaluation within two weeks.
P: Good, I'd like to get moving again.
"""

POOL_TEXT["T_PHARM"] = """
C: Which pharmacy should I send it to?
P: {fac_pharm} in {loc2}.
C: The one on the corner?
P: Yes.
---
C: Is {fac_pharm} still your preferred pharmacy?
P: I switched to {fac_mail}, the mail order one.
C: Okay, I'll send a ninety day supply there.
---
C: I've sent the prescription to {fac_pharm}. It should be ready within the hour.
P: Great, I'll pick it up on my way home.
---
P: Can you send it to {fac_pharm2} instead? It's closer to {loc2}.
C: Sure, I'll change it.
---
C: The pharmacy may call you about a prior authorization, reference {authno}.
P: Okay, I'll watch for the call.
---
C: I've sent it to your pharmacy.
P: Thank you, I'll pick it up tonight.
"""

POOL_TEXT["T_FOLLOW"] = """
C: Let's schedule a follow-up. Are you free {date_future}?
P: That works.
C: I'll book it for {timeword}.
---
C: If anything changes, call the clinic at {ext_phone}.
P: Can you repeat that?
C: {ext_phone}.
---
C: You'll get a text with an activation code for the portal, {code}. It expires tomorrow.
P: I'll enter it tonight.
---
C: I'll have {n_nurse} call you on {date_future} with the lab results. Is {phone} still the best number?
P: Yes, leave a message if I don't pick up.
---
C: Our nurse line for after hours questions is {nl_phone}.
P: Got it, I'll put it in my phone.
---
C: Your next appointment is {date_future}. You'll get a reminder by email at {email}.
P: Perfect.
---
C: I'll send the after visit summary to {email}, and it will also appear in the portal.
P: Thank you.
---
C: The clinic fax number for forms is {fax}, if your employer needs paperwork.
P: Okay, I'll let them know.
---
C: I'll see you in about two weeks.
P: Okay, thank you.
"""

POOL_TEXT["T_CLOSE"] = """
C: Do you have any other questions?
P: No, that covers it. Thank you so much.
C: Take care, and have a good rest of your day.
---
C: Is there anything else I can help with today?
P: I don't think so. I appreciate your time.
C: Of course. Feel better.
---
C: I'll let you go. Please don't hesitate to message us in the portal.
P: I will. Goodbye.
---
P: Thanks, doctor, this was easier than driving in.
C: I'm glad it worked. Take care.
"""

POOL_TEXT["T_AGE"] = """
C: And how old are you now?
P: I'm {age_over}, believe it or not.
C: You're doing wonderfully for {age_over}.
---
P: I turned {age_over} this spring.
C: Congratulations. Do you still manage the stairs?
P: Slowly, with the rail.
---
C: How are you managing day to day, now that you're in your {age_decade}?
P: My family helps, but I still cook for myself.
---
C: Has anyone talked with you about fall prevention, at {age_over}?
P: The therapist did, and I have grab bars in the bathroom now.
"""

POOL_TEXT["T_NEG"] = """
P: I have faith in this plan, honestly. I just want to feel normal again.
C: That's a good attitude.
---
C: Will you be able to take it with food?
P: Yes, I will.
---
C: I hope that helps.
P: Me too. I have a lot of hope for this one.
---
P: I saw a segment about this on {media}, and now I'm worried.
C: Those are general. Let's focus on your numbers.
---
P: My sister sent me {media}. It said to cut back on caffeine.
C: That is reasonable advice.
---
C: Do you have a living will or a healthcare proxy?
P: Yes, my will and the proxy form are both done.
---
P: My uncle had Parkinson disease, and I worry about the tremor in my hand.
C: A tremor in one hand has many causes. Let's look at yours.
---
C: Your labs are in line with a mild thyroid problem, like Hashimoto thyroiditis, so I want to repeat them.
P: Is that serious?
C: It's common and treatable.
---
P: I have a meeting at 10:30 AM tomorrow, so can we finish by eleven?
C: Of course.
---
C: The software updated to version {ver} overnight. If anything looks odd on your screen, tell me.
P: It looks the same to me.
---
C: Mark any episodes in a diary, with the time and what you were doing.
P: I'll use my phone.
---
P: Rose water doesn't seem to help my eyes.
C: I wouldn't expect it to. Let's try artificial tears.
---
C: Grace period on the new prescription plan? That's a question for your pharmacy.
P: Okay, I'll ask them.
---
P: I told them I'm not paying a penny more than the copay.
C: That's reasonable. Billing can sort it out.
---
P: My grandmother had Raynaud syndrome, and my fingers go white too in the cold.
C: That is worth mentioning. How often?
"""


POOL_TEXT["L_OPEN"] = """
N: Thank you for calling the {fac_nl} nurse advice line, this is {n_first}. How can I help you?
P: Hi, I'm calling about myself. I've had {sym} that started {reldate}.
---
N: Nurse line, this is {n_nurse} speaking. What's going on today?
P: I'm not sure if I should come in. It's the {sym}.
---
N: You've reached the after hours nurse line for {fac_nl}. My name is {n_full}. What can I do for you?
P: I'm worried about {sym}, and my clinic is closed.
---
N: Good evening, triage nurse speaking. Are you calling for yourself or someone else?
P: For myself. I've had {sym} for {dur}.
---
N: Nurse advice line, how can I help?
P: I've been having {sym} and I can't tell if it's serious.
N: Let's go through some questions together.
"""

POOL_TEXT["L_CALLER"] = """
N: Can I get your full name and date of birth?
P: {pt_full}, {dob}.
N: Thank you, {pt_hon}. What's a good number to reach you?
P: {phone}.
---
N: First, may I have your name and the phone number you're calling from?
P: It's {pt_full}, and I'm on my cell, {phone}.
N: And your date of birth?
P: {dob}.
---
N: If we get disconnected, which number should I call?
P: Use {ext_phone}, that's my desk at work.
N: Got it, {ext_phone}. And your date of birth?
P: {dob}.
---
N: I'm pulling up your chart. Is your record number {mrn}?
P: I'd have to check.
N: That's all right, I have you by name and birthday.
---
N: Are you at home right now?
P: Yes, at {addr}.
N: Thanks, I'm noting the address in case I need to send help.
---
N: Can you verify your date of birth for me?
P: I'd rather not say it aloud, I'm in a public place.
N: Understood. We can verify with the number on the back of your insurance card instead.
P: Okay, one moment, I have it right here.
---
N: And the best phone number for a call back?
P: {phone}, and if I don't answer, try {phone2}.
"""

POOL_TEXT["L_RED"] = """
N: Are you having any chest pain right now?
P: No.
N: Any trouble speaking, or weakness on one side?
P: No, nothing like that.
---
N: Are you able to speak in full sentences without getting short of breath?
P: Yes, I'm fine talking.
---
N: Have you passed out, or felt like you were about to?
P: No, just lightheaded when I stand up.
---
N: Is there any bleeding that won't stop?
P: No bleeding.
---
N: Any confusion, or a stiff neck with the fever?
P: No, my neck is fine.
---
N: Have you had a fever over one hundred and three?
P: No, the highest was one hundred and one.
---
N: Is the pain getting worse, or is it staying the same?
P: Staying about the same.
---
N: Are you able to keep fluids down?
P: Yes, small sips.
---
N: When did you last urinate?
P: This morning.
---
N: Are you pregnant, or is there any chance you could be?
P: No chance.
"""

POOL_TEXT["L_DISP"] = """
N: Based on what you're telling me, I recommend you be seen within twenty-four hours. I can book you with {c_dr} on {date_future}.
P: That works.
---
N: I'd like you to go to the emergency department at {fac_hosp} now. Do you have someone who can drive you?
P: My {f_rel} {f_first} can.
---
N: The urgent care at {fac_urgent} in {loc2} is open until nine tonight.
P: Do I need to bring anything?
N: Your insurance card and your medication list.
---
N: This can be managed at home. Rest, fluids, and {drug3} as directed on the box. Call back if it gets worse.
P: Okay, I'll do that.
---
N: I'm going to page the on-call clinician, {c2_dr}, and have them call you back within thirty minutes at {ext_phone}.
P: Okay, I'll keep my phone nearby.
---
N: I'll send a message to your regular clinic, {fac_clinic}, so they know about this call.
P: Thank you.
---
N: Please call 911 right away if the pain spreads to your arm or jaw, or you can't catch your breath.
P: I will.
---
N: I'd like you to see someone today. I can offer a video visit this afternoon.
P: Yes, please.
"""

POOL_TEXT["L_ONCALL"] = """
D: This is {c2_dr}, the physician on call for {fac_clinic}. I'm returning your call about the {sym}.
P: Thank you for calling back so fast.
---
D: Hi, is this {pt_first}? It's {c2_drfull}, I'm covering tonight.
P: Yes, thank you for calling.
---
D: I reviewed the nurse's note. Let me ask you a few more questions about the {sym}.
P: Of course.
D: When did it begin, exactly?
P: {reldate}, in the afternoon.
---
D: I'm {c2_dr}, the covering doctor. I see you spoke with {n_first} earlier.
P: Yes, that was about an hour ago.
"""

POOL_TEXT["L_CALLBACK"] = """
N: Let me read the number back. {ext_phone}. Is that right?
P: Yes, that's it.
---
N: And a second number?
P: My {f_rel}'s cell, {phone2}.
N: Got it, I'll note both.
---
P: Please don't leave details on a message, just ask them to call {nl_phone}.
N: I'll note that on your chart.
---
N: Our callback window is between four and six. If you miss us, the number to call is {nl_phone}.
P: Okay.
---
N: If you don't hear from us within two hours, call back and mention this is a callback request.
P: I will.
"""

POOL_TEXT["L_TRANSFER"] = """
N: I'm going to transfer you to the pharmacy line, extension {ext_only}. If you get disconnected, dial it directly.
P: Extension {ext_only}, got it.
---
N: The billing office is on extension {ext_only}, and they're open until five.
P: Thank you.
---
S: Please hold, I'm connecting you to extension {ext_only}.
P: Okay.
"""

POOL_TEXT["L_CLOSE"] = """
N: Do you have any other questions?
P: No, that's everything. Thank you.
N: Take care, and call us back if anything changes.
---
N: You did the right thing by calling. Please follow the advice we discussed.
P: I will. Goodbye.
---
N: Is there anything else I can help you with tonight?
P: No, you've been very helpful.
"""

POOL_TEXT["LB_OPEN"] = """
N: Nurse line, {n_first} speaking. Who am I speaking with, and who is the patient?
F: This is {f_full}. I'm calling about my {f_rel}, {pt_full}.
N: Thank you. What is {pt_first}'s date of birth?
F: {dob}.
---
F: Hi, my {f_rel} isn't doing well. {pt_His} name is {pt_full}.
N: I'm sorry to hear that. Is {pt_he} with you now?
F: Yes, {pt_he} is right here in the kitchen.
---
N: Triage nurse speaking, how can I help?
F: I'm {f_first}, and I'm calling for {pt_hon}, who lives with me.
N: Can you give me {pt_his} date of birth, and a number to reach you?
F: {dob}, and you can reach me at {phone}.
---
F: I need to ask about my {f_rel}. We're at {addr}.
N: Okay, stay on the line with me. What is the patient's name?
F: {pt_full}.
"""

POOL_TEXT["LB_SYMP"] = """
N: How long has {pt_first} had the {sym}?
F: It started {reldate}.
---
N: Is {pt_he} awake and answering you normally?
F: Mostly, but {pt_he} is very tired.
---
N: Has {pt_he} taken any medication for it?
F: {pt_He} took {drug} about an hour ago.
---
N: Does {pt_he} have any allergies?
F: Penicillin.
---
N: Is {pt_he} having any trouble breathing?
F: A little, when {pt_he} climbs the stairs.
---
N: How old is {pt_he}?
F: {pt_He} is 84.
---
N: Has {pt_first} eaten or had anything to drink today?
F: A little tea and some toast.
---
N: Has {pt_he} fallen recently?
F: Not that I know of. {pt_He} says no.
---
N: Can you take {pt_his} temperature for me?
F: One moment. It's one hundred point four.
---
N: Is {pt_he} taking {drug} or {drug2}?
F: Both, in the morning.
"""

POOL_TEXT["LB_AGE"] = """
N: How old is {pt_first}?
F: {pt_He} is {age_over}.
N: Thank you, that helps me choose the right questions.
---
F: {pt_first} is {age_over}, so I worry more than I would otherwise.
N: That's understandable. Let's go through some questions.
"""

POOL_TEXT["I_OPEN"] = """
I: Good morning, this is interpreter number {intid}, and I'll be interpreting {lang} today. Who is on the line?
C: I'm {c_dr} at {fac_clinic}, and I have {pt_hon} and {pt_hon}'s {f_rel} with me.
---
C: I'm calling the language line to connect an interpreter for {lang}.
I: You're connected, my interpreter ID is {intid}. May I have the language you need?
C: Yes, {lang}. We need it for {pt_full}.
---
I: Hello, interpreter {intid}. I will interpret everything that is said, in the first person. Please speak in short sentences and pause.
C: Thank you. Let me introduce everyone first.
---
I: Hello, I'm your interpreter. I interpret everything that is said, and I keep what I hear confidential.
C: Thank you. Please let the patient know that I'm the doctor today.
---
C: Is this the {lang} interpreter?
I: Yes, this is interpreter {intid}. I'm ready when you are.
---
I: This is the interpreter. I'm going to turn on my video so everyone can see me.
C: Great, thank you. I can see you now.
"""

POOL_TEXT["I_INTRO"] = """
C: Please tell {pt_hon} that I am {c_dr}, and that I'll be asking some questions about the visit today.
I: [speaking {lang}] This is {c_dr}, the doctor, who would like to ask some questions about your health today.
P: [speaking {lang}]
I: The patient says that's fine, and thanks you for calling.
---
C: Please ask who is with the patient today.
I: [speaking {lang}] Who is with you today?
F: [speaking {lang}]
I: It's the {f_rel}, {f_first}. {f_first} says {f_first} will help with questions.
C: Welcome, {f_first}.
---
C: Please explain that you are the interpreter and that this is a confidential conversation.
I: [speaking {lang}] I am the interpreter and what is said here is confidential.
P: [speaking {lang}]
I: The patient understands.
---
C: Please tell {pt_hon} that I'm glad {pt_he} could come in.
I: [speaking {lang}] The doctor says hello and is glad you could be here.
P: [speaking {lang}]
I: {pt_He} says hello and thank you.
"""

POOL_TEXT["I_VERIFY"] = """
C: Please ask for the patient's full name and date of birth.
I: [speaking {lang}] Can you state your full name and date of birth?
P: [speaking {lang}]
I: {pt_full}, {dob}.
---
C: I have the address as {addr}. Can you confirm?
I: [speaking {lang}] Is your address {addr}?
P: [speaking {lang}]
I: Yes, that is correct.
---
C: And a phone number where we can reach the patient?
I: [speaking {lang}] What is the best number to reach you?
P: [speaking {lang}]
I: {phone}.
---
C: Please ask whether the patient prefers an interpreter for all visits.
I: [speaking {lang}] Do you want an interpreter at every visit?
P: [speaking {lang}]
I: {pt_He} says yes, for every visit.
---
C: Can you confirm the patient's name for me?
I: [speaking {lang}] Is your name {pt_full}?
P: [speaking {lang}]
I: {pt_He} says that is correct.
"""

POOL_TEXT["I_HPI"] = """
C: Please ask when the {sym} started.
I: [speaking {lang}] When did the {sym} start?
P: [speaking {lang}]
I: About {dur} ago.
---
C: Ask whether the {sym} is constant or comes and goes.
I: [speaking {lang}] Is it there all the time, or does it come and go?
P: [speaking {lang}]
I: It comes and goes, mostly at night.
---
C: Ask if there has been any fever.
I: [speaking {lang}] Have you had a fever?
P: [speaking {lang}]
I: No fever, but {pt_he} has felt hot at night.
---
C: Please ask how severe it is, from zero to ten.
I: [speaking {lang}] From zero to ten, how bad is it?
P: [speaking {lang}]
I: About a six.
---
C: Ask if anything makes it better or worse.
I: [speaking {lang}] Does anything help, or make it worse?
P: [speaking {lang}]
I: Resting helps. Walking makes it worse.
---
C: Ask whether {pt_he} has trouble breathing.
I: [speaking {lang}] Do you have any trouble breathing?
P: [speaking {lang}]
I: No, {pt_he} says breathing is fine.
---
C: Ask how {pt_he} has been sleeping.
I: [speaking {lang}] How have you been sleeping?
P: [speaking {lang}]
I: Not well. {pt_He} wakes up several times.
---
C: Ask if {pt_he} has tried anything at home.
I: [speaking {lang}] Did you try any medicine at home?
P: [speaking {lang}]
I: {pt_He} took {drug} for two days.
---
C: Ask about appetite.
I: [speaking {lang}] How is your appetite?
P: [speaking {lang}]
I: Smaller than usual, starting {reldate}.
"""

POOL_TEXT["I_FAM"] = """
F: [speaking {lang}]
I: {f_first}, the {f_rel}, says the patient has been dizzy since it started {reldate} and sleeping more.
C: Thank you, {f_first}. Does {pt_hon} ever fall?
I: [speaking {lang}] Does the patient ever fall?
F: [speaking {lang}]
I: Once, in the bathroom, last week.
---
C: Is {f_first} the one who manages the medication?
I: [speaking {lang}] Are you the one who gives the medicine?
F: [speaking {lang}]
I: Yes, {f_first} fills the pill box every Sunday.
---
C: {f_first}, can you give me your phone number in case we need to call you?
I: [speaking {lang}] Can you give us your phone number?
F: [speaking {lang}]
I: It's {phone}.
---
F: [speaking {lang}]
I: {f_first} says the family is worried because {pt_hon} is not eating as before.
C: I understand. Let's talk about what we can do.
---
C: Who else is at home to help?
I: [speaking {lang}] Who else is at home?
F: [speaking {lang}]
I: {f_first} says another relative, {g_first}, comes on weekends.
"""

POOL_TEXT["I_MEDS"] = """
C: Please ask the patient to read me the names of the medicines.
I: [speaking {lang}] Please tell us which medicines you take.
P: [speaking {lang}]
I: {drug} in the morning and {drug2} at night.
---
C: Ask whether the patient has missed any doses.
I: [speaking {lang}] Have you missed any doses?
P: [speaking {lang}]
I: A few, when {pt_he} ran out last week.
---
C: Please explain that {drug} should be taken with food.
I: [speaking {lang}] The doctor says to take {drug} with food.
P: [speaking {lang}]
I: {pt_He} understands.
---
C: Ask if there are any side effects from {drug}.
I: [speaking {lang}] Do you have any problems from the medicine?
P: [speaking {lang}]
I: Sometimes a stomach ache.
---
C: Does the patient use any herbal remedies?
I: [speaking {lang}] Do you take any herbs, teas or supplements?
P: [speaking {lang}]
I: A ginger tea in the evenings.
"""

POOL_TEXT["I_PLAN"] = """
C: I'd like to start {drug} {dose} once a day. Please explain that.
I: [speaking {lang}] The doctor wants you to start {drug}, {dose}, once a day.
P: [speaking {lang}]
I: {pt_He} asks if it is safe with {drug2}.
C: Yes, it is.
---
C: I want blood work done at {fac_lab}. Please explain that {pt_he} needs to fast.
I: [speaking {lang}] You need blood tests at {fac_lab}. Do not eat for eight hours before.
P: [speaking {lang}]
I: {pt_He} understands.
---
C: Tell the patient to call us, or go to the emergency department, if the {sym} gets worse.
I: [speaking {lang}] Call the clinic, or go to the emergency room, if it gets worse.
P: [speaking {lang}]
I: {pt_He} agrees.
---
C: Please explain that I'm sending the prescription to {fac_pharm}.
I: [speaking {lang}] The prescription will go to {fac_pharm}.
P: [speaking {lang}]
I: That's the right pharmacy.
"""

POOL_TEXT["I_FOLLOW"] = """
C: We'll schedule an interpreter for the next visit on {date_future}.
I: [speaking {lang}] The next visit is {date_future}. We will have an interpreter.
P: [speaking {lang}]
I: {pt_He} says that day is fine.
---
C: The clinic number is {ext_phone}. Please ask them to write it down.
I: [speaking {lang}] The clinic number is {ext_phone}.
F: [speaking {lang}]
I: {f_first} has it.
---
C: Please explain that a code for the patient portal, {code}, will be sent by text.
I: [speaking {lang}] You will get a text with a code for the portal.
P: [speaking {lang}]
I: {pt_He} says the family will enter it.
---
C: The language line can also be reached on {nl_phone} to book an interpreter.
I: [speaking {lang}] To request an interpreter, call {nl_phone}.
F: [speaking {lang}]
I: {f_first} will call.
"""

POOL_TEXT["I_CLOSE"] = """
C: Does the patient have any questions?
I: [speaking {lang}] Do you have any questions?
P: [speaking {lang}]
I: No, {pt_he} says thank you.
C: Thank you for your help today.
---
C: Please thank the patient and family for their time.
I: [speaking {lang}] The doctor thanks you for your time.
P: [speaking {lang}]
I: {pt_He} says thank you, doctor.
I: This is the interpreter, disconnecting now.
"""

POOL_TEXT["I_NEG"] = """
I: Interpreter note, the line is breaking up. May I ask you to repeat?
C: Of course, let me say that again more slowly.
---
C: Please explain that the visit software, version {ver}, may show the interpreter's picture in a small box.
I: [speaking {lang}] The screen may show my picture in a small box.
P: [speaking {lang}]
I: {pt_He} says that is fine.
---
I: Interpreter speaking. I'd like to clarify a term. Did the doctor mean a will, as in a legal document?
C: Yes, the living will form.
---
C: Please ask whether {pt_he} has hope that the treatment will help.
I: [speaking {lang}] Do you feel hopeful about this treatment?
P: [speaking {lang}]
I: {pt_He} says yes, and has faith in the plan.
"""

POOL_TEXT["C_OPEN_CG"] = """
M: Hello, this is {m_full} from the care coordination team at {fac_clinic}. Is this {f_first}?
F: Yes, it is. Is something wrong with {pt_first}?
M: No, nothing urgent. I'm following up after the discharge. Do you have a few minutes?
---
M: Hi {pt_hon}, it's {m_first} with social work. Is this a good time to talk?
P: Sure, I was expecting your call.
---
M: I'm calling about the referral for {pt_full}. Am I speaking with the caregiver?
F: I am. I'm {f_full}, {pt_first}'s {f_rel}.
---
M: Good morning, this is the care coordination office. I'm calling to check on how things are going at home.
F: It's been a hard week, but we're managing.
---
M: Hello, I'm a social worker with the hospital's discharge team. Is the patient's caregiver available?
F: This is the caregiver. How can I help?
---
M: This is {m_hon} from the clinic's social work team. I'm calling to follow up on the home safety questions.
F: Thank you for calling back. I'm {f_first}.
"""

POOL_TEXT["C_ID"] = """
M: Can you confirm {pt_first}'s date of birth for me?
F: {dob}.
M: And the address on the referral is {addr}?
F: Yes, that's right.
---
M: Let me verify a few details. The best phone number for {pt_first} is {phone}?
F: Yes, but call me instead, at {phone2}.
M: I'll note that.
---
M: I have the record number as {mrn}. Does that match what you were given at discharge?
F: I don't have the paperwork in front of me, but the name is correct.
---
M: What name should I use for the insurance paperwork, and is the member ID {member}?
F: {pt_full}, and yes, that ID is on the card.
---
M: Can you give me the date {pt_first} was discharged?
F: It was {date_past}.
M: Thank you.
---
M: Before I share any details, I need to confirm who I'm speaking with. Can you verify the patient's date of birth?
F: It's {dob}.
"""

POOL_TEXT["C_SIT"] = """
M: How are things at home?
F: Honestly, it's been hard. {pt_first} is up at night and I'm exhausted.
M: That sounds draining. Do you have other family nearby?
F: My brother lives in {loc2} but works long hours.
---
M: Is anyone else helping?
F: A neighbor, {g_first}, brings meals twice a week.
M: That's a real help. Does {g_first} have a way to reach you in an emergency?
F: Yes, we've exchanged numbers.
---
M: Has {pt_first} had any falls since coming home?
F: One, in the hallway, but no injury.
M: I'd like to arrange a home safety evaluation.
---
M: How is the medication routine going?
F: It's a lot. {pt_first} takes eleven pills a day and I use a weekly box.
M: Let's see if a pharmacist can simplify the schedule.
---
M: Are you getting any time for yourself?
F: Not really. I haven't left the house in a week.
M: There are respite options we can look at.
---
M: Does {pt_first} have enough food in the house?
F: We do, thanks to the church volunteers who drop off meals.
---
M: I want to check on transportation to appointments.
F: I drive, but I can't leave {pt_first} alone for long.
---
M: Is the home set up with a bed on the first floor?
F: Not yet. We've been using the couch.
M: I'll add that to the equipment request.
---
M: How has {pt_first}'s mood been?
F: Quiet. {pt_He} doesn't want visitors.
M: That's common after a hospital stay. Let's keep an eye on it.
"""

POOL_TEXT["C_SVC"] = """
M: I've referred {pt_first} to {fac_hh} for skilled nursing and physical therapy. They should call you within two days.
F: Do you have their number?
M: Yes, it's {ext_phone}.
---
M: The equipment, a walker and a shower chair, will come from {fac_dme}. They'll want to deliver to {addr}.
F: That's right.
---
M: There is an opening at {fac_adc} on weekdays, and they offer transportation.
F: How much does it cost?
M: I'll send you the fee sheet. Some of it may be covered.
---
M: If you'd like to tour {fac_sl}, I can set that up.
F: Yes, we'd like to see it before deciding.
---
M: The hospice team at {fac_hospice} can come out for an intake visit.
F: Does that mean we're giving up?
M: Not at all. It means more support at home.
---
M: {fac_snf} has a bed available for short-term rehab, starting {date_future}.
F: How long would {pt_first} stay?
M: Usually two to three weeks, depending on progress.
---
M: The mail order pharmacy, {fac_mail}, can deliver ninety day supplies.
F: That would save us trips.
---
F: We've already spoken to {fac_funeral} about arrangements.
M: I'm sorry. Thank you for telling me. I'll note that.
---
M: A home health aide from {fac_hh} can help with bathing three mornings a week.
F: That would be a relief.
---
M: There is a support group for caregivers that meets online, at {url}.
F: I'll look at it tonight.
"""

POOL_TEXT["C_AUTH"] = """
M: The authorization number is {authno}. Please give that to the agency.
F: Let me write that down. {authno}.
---
A: Can you give me the referral number?
M: Referral {refno}.
A: Thank you.
---
M: Your case number with our team is {caseno}.
F: Let me write it down.
---
A: What's the member ID?
M: {member}.
A: And the group number?
M: I'll have to send that by fax.
---
M: The billing account for the equipment is {acct}.
F: I'll mention that when they call.
---
M: Please reference case {caseno} whenever you call the office.
F: Okay.
"""

POOL_TEXT["C_AGENCY"] = """
A: Thank you for calling {fac_hh}, intake. This is {a_first}.
M: Hi {a_first}, this is {m_full} calling from {fac_hosp} about a referral for {pt_full}.
A: One moment, I'll pull it up. What's the date of birth?
M: {dob}.
---
A: Intake, this is {a_hon}. How can I help?
M: I'm {m_full}, a social worker. I'm calling about a new referral for home health.
A: Do you have the referral number?
M: Yes, {refno}.
---
A: Do you have the patient's address for the first visit?
M: {addr}.
A: And a phone number for scheduling?
M: {phone}.
---
A: We can start the first visit on {date_future}. Does that work for the family?
M: I'll confirm with {f_first} and call you back.
---
A: What services were ordered?
M: Skilled nursing twice a week, physical therapy, and a home safety evaluation.
A: I'll add them to the intake.
---
A: We'll need the orders faxed to {fax}.
M: I'll send them this afternoon.
---
A: Does the patient have any pets, or anything we should know about before a visit?
M: A small dog, friendly. And the family prefers visits in the morning.
"""

POOL_TEXT["C_DISCHARGE"] = """
M: I'm calling to coordinate a discharge to {fac_snf}. The patient is {pt_full}.
A: Let me check bed availability. When would {pt_he} be ready?
M: Tomorrow afternoon, from {fac_hosp}.
---
A: Do you have the therapy notes from the hospital stay?
M: Yes. I'll fax them to {fax}.
---
M: The family would like to tour first. Is {date_future} possible?
A: That works. Ask for me at the front desk.
---
A: What is the patient's level of assistance with transfers?
M: One person assist, with a walker.
A: And any wounds or lines?
M: No wounds, and no lines.
---
M: I'd like to confirm the authorization, {authno}, with your admissions office.
A: I have that. It covers the first twenty days.
---
M: The patient lives with {f_first} at {addr}. Please send the bill and paperwork there.
A: Noted.
"""

POOL_TEXT["C_SW_FAM"] = """
M: I wanted to talk with you about options for long term care. Where is the family at with that?
F: We're torn. {pt_first} wants to stay home, but we can't provide that much help.
M: Those are hard decisions. Let's look at the options together.
---
M: Has there been any conversation about who makes decisions if {pt_first} cannot?
F: {pt_He} named me. We signed the forms last year.
M: Please send us a copy so it's in the chart.
---
M: Some families find a short stay at {fac_sl} helpful as a trial.
F: Can {pt_first} come back home after?
M: Yes, if that's the plan.
---
M: I can connect you with {fac_adc}. They offer meals and activities.
F: That might work. {pt_first} gets lonely.
---
F: I just feel guilty. A relative, {g_first}, thinks we should keep {pt_first} here.
M: Guilt is very common. It sounds like you're doing a lot.
---
M: Have you talked to {pt_first} about what matters most?
F: {pt_He} says staying near the garden, and seeing the grandkids.
"""

POOL_TEXT["C_LOGISTICS"] = """
M: I can arrange transport for the appointment on {date_future}. Where will {pt_first} be picked up?
F: At {addr}.
---
A: When can we deliver the equipment?
M: Anytime after {date_future}. Call {f_first} at {phone} first.
---
F: Will someone call before they come?
A: Yes, the nurse will call about an hour ahead.
---
M: I'll call you back on {date_future} with an update.
F: Please use {phone}, I'm not always at home.
---
A: Our scheduling line is {ext_phone}, if you need to change anything.
F: I'll save it.
---
M: The paperwork will come by email to {email}, and I'll also send it by mail.
F: Email is fine.
---
M: Please keep the packet of discharge papers in one folder. The agency will want it at the first visit.
F: We have a folder on the fridge.
"""

POOL_TEXT["C_AGE"] = """
M: How old is {pt_first}, again, for the form?
F: {pt_He} is {age_over}.
M: Thank you.
---
F: {pt_first} is {age_over}, and the stairs are too much now.
M: That's an important detail for the home evaluation.
"""

POOL_TEXT["C_CLOSE"] = """
M: Thank you for your time. I'll follow up as promised.
F: Thank you for all of your help.
---
M: Is there anything else you need from our team today?
F: No, I think that covers it.
M: Okay. Please call if anything changes.
---
M: I'll send a summary of this call by email, and you can reach me during the week.
F: That would be great. Take care.
---
A: Thank you for calling. We'll be in touch once the paperwork is complete.
M: Thank you. Goodbye.
"""

POOL_TEXT["C_NEG"] = """
M: Home health referral sent. I'm also sending the hospice information packet, whenever you're ready.
F: Thank you. We aren't ready yet.
---
A: We don't have a bed today, but I'll call you as soon as one opens.
M: Thank you, I appreciate it.
---
F: I read about this on {media}. It made it sound simple.
M: It can be, but every plan is different.
---
M: I'll mark the date on the calendar. Mark it on yours as well.
F: I will.
---
F: I have faith we'll figure it out.
M: You will. I'll be here.
---
M: Will you be home on the morning of the visit?
F: Yes, I will.
"""

POOL_TEXT["C_MEET"] = """
M: Our care team meets online each Tuesday. The link is {url}.
F: Do I need to sign in?
M: No, just click the link.
---
M: The care conference is booked, and the call-in number is {phone2}.
F: What time?
M: {timeword}.
"""

POOL_TEXT["LB_DISP"] = """
N: I recommend {pt_first} be seen today. Can you bring {pt_him} to the urgent care at {fac_urgent} in {loc2}?
F: Yes, I can drive.
N: Bring {pt_his} insurance card and a list of medications.
---
N: I'd like {pt_him} evaluated at the emergency department at {fac_hosp}. Is there someone who can go with you?
F: My neighbor, {g_first}, can come.
---
N: This can wait for a visit. I can book {pt_first} with {c_dr} on {date_future}.
F: That would be fine.
---
N: I'll page the on-call clinician, {c2_dr}, who will call you at {phone}.
F: Thank you, I'll keep the phone close.
---
N: Please give {pt_him} fluids and watch {pt_him} closely. Call back, or call 911, if {pt_he} gets worse.
F: I will.
"""

POOL_TEXT["I_AGE"] = """
C: Please ask the patient's age.
I: [speaking {lang}] How old are you?
P: [speaking {lang}]
I: {age_over}.
---
C: The record says the patient is {age_over}. Please confirm.
I: [speaking {lang}] The doctor says you are {age_over}. Is that right?
P: [speaking {lang}]
I: {pt_He} says yes.
"""

POOL_TEXT["C_AM"] = """
A: Is the patient able to participate in therapy for an hour a day?
M: Probably in shorter sessions at first. {pt_He} tires easily.
A: We can start with two short sessions.
---
M: Does your program accept the patient's insurance?
A: We verify that on our end once we have the member ID.
M: I'll send the card images.
---
A: What dietary needs should we plan for?
M: Low salt, and soft foods for now.
---
A: Does the patient have any behavioral concerns we should know about?
M: Some evening agitation. The family can tell you more.
---
M: Please call me back when the intake review is finished.
A: I'll let you know by the end of the day.
---
A: We'll need the medication list and the most recent physician note.
M: Both are in the packet I'm sending.
---
M: How soon can a nurse do the first assessment?
A: Within forty-eight hours of the referral being accepted.
---
A: Who is the primary contact for the family?
M: The patient's {f_rel}, {f_full}, at {phone}.
"""

POOL_TEXT["L_AGE"] = """
N: And how old are you?
P: I'm {age_over}.
N: Thank you, that helps me choose the right questions.
---
N: Can I confirm your age for the triage form?
P: {age_over}, going on a year older in the fall.
"""

POOL_TEXT["C_AMAGE"] = """
A: How old is the patient?
M: {pt_He} is {age_over}.
A: Thank you, I'll note that on the intake.
---
M: The patient is {age_over}, so we'll want a fall risk review on the first visit.
A: We always do that.
"""


# --------------------------------------------------------------------------
# question and answer topics. A topic pairs several phrasings of a question with
# several first-person answers, so an exchange can be recombined many ways. All
# slots are non-identifying. Groups: hpi, med, life (clinician or nurse asks,
# patient answers) and coord, am (case manager asks; caregiver or agency answers).

TOPIC_TEXT = """
# onset | hpi
Q: How long has the {sym} been going on?
Q: When did the {sym} first start?
Q: Can you tell me when you first noticed the {sym}?
Q: How long have you had the {sym}?
Q: When did this all start?
A: About {dur}.
A: It started {reldate}.
A: Maybe {dur}, I'm not completely sure.
A: It began {reldate} and has been on and off since.
A: I first noticed it {reldate}.
---
# pattern | hpi
Q: Is it constant, or does it come and go?
Q: Does the {sym} stay all day or come in waves?
Q: Is the {sym} there all the time?
Q: Would you say it is constant or intermittent?
A: It comes and goes, mostly in the afternoon.
A: It's there all day, but worse at night.
A: It comes in waves, maybe three or four times a day.
A: Pretty constant, honestly.
A: Mostly in the evenings.
---
# severity | hpi
Q: On a scale of zero to ten, how bad is it at its worst?
Q: How would you rate it, zero to ten?
Q: How bad does it get on a ten point scale?
A: Probably a six, but most of the time it's a three.
A: About a seven when it's at its worst.
A: A four, I can live with it.
A: It was an eight yesterday, but a two today.
A: Maybe a five.
---
# relief | hpi
Q: Does anything make it better?
Q: Have you found anything that helps?
Q: Is there anything that eases the {sym}?
Q: What helps, if anything?
A: Resting helps a little.
A: A warm compress, when I can.
A: Not really, nothing helps for long.
A: Lying down in a quiet room helps.
A: Taking {drug} takes the edge off.
---
# aggravate | hpi
Q: Is there anything that makes it worse?
Q: What makes the {sym} worse?
Q: Does anything seem to bring it on?
A: Stress, and not sleeping.
A: Climbing the stairs.
A: Eating a big meal.
A: Being on my feet all day.
A: Lying flat at night.
---
# fever | hpi
Q: Have you had any fever or chills?
Q: Any fevers, chills or sweats?
Q: Have you taken your temperature?
A: No fever. I took my temperature this morning and it was normal.
A: A little low grade fever last night, one hundred point two.
A: I felt chilled yesterday, but I didn't take a reading.
A: No, nothing like that.
---
# appetite | hpi
Q: Any changes in your appetite or weight?
Q: Have you been eating normally?
Q: Has your weight changed recently?
A: I've lost a couple of pounds, I think, but I haven't been trying.
A: My appetite is smaller than usual.
A: No change that I've noticed.
A: I'm eating, but everything tastes flat.
---
# sleep | hpi
Q: Have you been able to sleep?
Q: How has your sleep been?
Q: Is the {sym} keeping you up at night?
A: Not well. I wake up around three most nights.
A: About six hours, broken up.
A: I fall asleep fine but wake up early.
A: Pretty good, actually.
---
# home_tx | hpi
Q: Did you try anything at home for it?
Q: Have you taken anything for the {sym}?
Q: Any home remedies or over the counter medicines so far?
A: I took {drug} twice a day and it helped a little.
A: Just rest and fluids so far.
A: I tried {drug}, but it didn't do much.
A: A heating pad, mostly.
---
# breathing | hpi
Q: Do you have any trouble breathing, or chest pain with the {sym}?
Q: Any shortness of breath?
Q: Does it ever get hard to catch your breath?
A: No, nothing like that.
A: A little, when I climb stairs.
A: Not at rest, but when I walk fast.
A: No chest pain, just a bit winded.
---
# swelling | hpi
Q: Have you noticed any swelling, redness, or warmth?
Q: Is there any swelling in your legs or feet?
Q: Any redness around the area?
A: A little redness, no warmth.
A: Some swelling in my ankles by evening.
A: No swelling that I can see.
A: My shoes feel tight by the end of the day.
---
# function | hpi
Q: Has this affected your work or daily activities?
Q: Is it getting in the way of your day?
Q: Are you able to do your usual activities?
A: Some days I can't get through a full shift.
A: I've had to slow down, yes.
A: Mostly I can manage.
A: I stopped my morning walks.
---
# exposure | hpi
Q: Any recent travel or sick contacts?
Q: Has anyone around you been sick?
Q: Have you been around anyone with a cough or fever?
A: Not that I know of. Nobody at home has been sick.
A: A coworker had the same thing last week.
A: We took a road trip two weeks ago, nothing else.
A: A grandchild had a cold.
---
# prior | hpi
Q: Have you had this before?
Q: Is this the first time this has happened?
Q: Has anything like this happened in the past?
A: Once, years ago, and it went away on its own.
A: No, never.
A: Yes, a few times, usually in the spring.
A: It's happened twice before.
---
# home_bp | hpi
Q: Have you been checking your blood pressure at home?
Q: What has your blood pressure been at home?
Q: Do you have a home cuff?
A: This morning it was {bp}.
A: It's been running around {bp}.
A: I check it every few days, usually {bp}.
A: I don't have a cuff, I use the one at the pharmacy.
---
# sugar | hpi
Q: Do you check your sugar?
Q: Are you checking your blood sugar at home?
Q: How have your sugars been?
A: Yes, fasting this morning was 118, and my last A1c was {a1c}.
A: A little high after dinner, around 180.
A: My last A1c was {a1c}.
A: I check twice a week, usually in the 120s.
---
# fluids | hpi
Q: Have you been drinking enough fluids?
Q: How much water do you drink in a day?
Q: Are you staying hydrated?
A: I try. About six glasses a day.
A: Mostly coffee, to be honest.
A: Plenty of water, a couple of liters.
A: Not enough, probably.
---
# night | hpi
Q: Does the {sym} wake you up at night?
Q: Do you wake up with the {sym}?
A: Twice this week. It's worse when I lie flat.
A: Not usually.
A: Most nights, around the same time.
A: No, it settles when I'm asleep.
---
# gi | hpi
Q: Any nausea, vomiting, or changes in your bowel movements?
Q: How are your bowels?
Q: Any stomach trouble along with it?
A: Some nausea in the mornings, but no vomiting.
A: Regular, no changes.
A: A bit constipated this week.
A: Loose stools for two days.
---
# urinary | hpi
Q: Any burning, urgency, or changes when you urinate?
Q: How is your urination?
Q: Have you noticed any changes in urination?
A: A little burning at the end.
A: No changes.
A: I'm up twice at night to go.
A: More frequent, but not painful.
---
# dizzy | hpi
Q: Do you feel dizzy or lightheaded when you stand up?
Q: Any fainting or near fainting?
Q: Have you felt unsteady on your feet?
A: Sometimes, if I stand up too fast.
A: No fainting, just a swimmy feeling.
A: No, I feel steady.
A: Once, in the kitchen, but I caught myself.
---
# location | hpi
Q: Where exactly do you feel the {sym}?
Q: Is the {sym} on one side or both?
Q: Can you point to where it bothers you most?
A: Mostly the left, but it moves around.
A: Right here along the side.
A: Both sides, though more on the right.
A: Low and in the middle.
---
# falls | hpi
Q: Have you had any falls in the last year?
Q: Any falls or near falls recently?
A: One, on the ice last winter, but I wasn't hurt.
A: No falls.
A: A couple of near misses on the stairs.
A: I tripped on the rug last month.
---
# adherence | med
Q: Are you missing any doses of your medicines?
Q: How often do you forget a dose?
Q: Are you taking the {drug} the way it was prescribed?
A: Maybe once or twice a week, I forget the evening one.
A: I take everything every day, I use a pill box.
A: I skip it when I feel fine, I know I shouldn't.
A: I ran short last week, so I stretched it out.
---
# side_effects | med
Q: Any side effects from the {drug}?
Q: Have you had any problems with the {drug}?
Q: Does the {drug} agree with you?
A: A dry cough sometimes.
A: It upsets my stomach a little.
A: Not that I've noticed.
A: I feel sleepy in the afternoon.
---
# allergy | med
Q: Do you have any allergies to medications?
Q: Any drug allergies I should know about?
A: Penicillin gives me hives, and nothing else that I know of.
A: No known allergies.
A: Sulfa drugs gave me a rash once.
A: Just seasonal allergies, no medicines.
---
# substances | life
Q: Do you smoke or drink?
Q: How much alcohol do you have in a typical week?
Q: Do you use any tobacco or other substances?
A: I quit smoking years ago and have a glass of wine on the weekends.
A: Maybe two drinks, usually on weekends.
A: Neither, I never started.
A: A beer or two with dinner.
---
# diet | life
Q: How would you describe your diet?
Q: What does a typical day of eating look like?
Q: Are you eating three meals a day?
A: Mostly home cooking, though I eat out on Fridays.
A: Two meals, usually. I skip lunch when I'm busy.
A: Lots of vegetables, not much meat.
A: Not great, a lot of takeout lately.
---
# exercise | life
Q: Do you get any exercise?
Q: How active have you been?
Q: Are you walking or exercising at all?
A: I walk about thirty minutes most days.
A: Not much since the {sym} started.
A: I swim twice a week.
A: Only chores around the house.
---
# caffeine | life
Q: How much caffeine do you have?
Q: Do you drink coffee or tea?
A: Two cups of coffee in the morning and nothing after noon.
A: Four or five cups a day, I know.
A: Just tea, one or two cups.
A: None, I switched to decaf.
---
# safety | life
Q: Do you feel safe at home?
Q: Is there anything at home that worries you for your safety?
A: Yes, completely.
A: Yes, though the stairs worry me.
A: Yes, I do.
A: I have grab bars now, so yes.
---
# mood | life
Q: Over the last two weeks, have you felt down or hopeless?
Q: How have your spirits been lately?
Q: Have you lost interest in things you usually enjoy?
A: Not really. Some days are slower than others.
A: A bit down, since the {sym} started.
A: I'm worried, mostly.
A: No, I feel like myself.
---
# stress | life
Q: How is your stress level, zero to ten?
Q: Is anything stressing you at the moment?
A: A five, mostly from work.
A: Higher than usual, maybe a seven.
A: I'm okay.
A: Money, mostly.
---
# vaccines | life
Q: Are you up to date on your vaccines?
Q: Have you had a flu shot this year?
A: I think so. I got the flu shot in the fall.
A: I'm not sure, I'd have to check.
A: Yes, I had everything at my last physical.
A: I skipped it this year.
---
# senses | life
Q: Do you have trouble with vision or hearing?
Q: Any problems with your eyes or ears?
A: My reading glasses are getting weak, that's all.
A: My hearing is not what it was.
A: No, nothing new.
A: A bit of ringing in my left ear.
---
# dental | life
Q: Any problems with your teeth or gums?
Q: When did you last see a dentist?
A: I'm overdue for a cleaning.
A: No problems.
A: One tooth has been sensitive.
A: Last spring, I think.
---
# snore | life
Q: Do you snore, or has anyone noticed you stop breathing at night?
Q: Do you wake up tired?
A: I snore. Nobody has said anything about stopping.
A: I wake up tired most mornings.
A: Not that I know of.
A: People tell me I snore loudly.
---
# driving | life
Q: Do you drive, or do you rely on others?
Q: How do you get to your appointments?
A: I still drive, but not at night.
A: A family member drives me.
A: I take the bus.
A: I drive myself, it's only ten minutes.
---
# c_home | coord
Q: How are things at home?
Q: How has the week gone at home?
Q: How is everyone holding up?
A: Honestly, it's been hard.
A: We're managing, but it's a lot.
A: Better than last week, thank you.
A: It's been up and down.
---
# c_strain | coord
Q: Are you getting any time for yourself?
Q: How are you holding up as the caregiver?
Q: Is there anything that would make the days easier for you?
A: Not really. I haven't left the house in a week.
A: I'm tired, but I'm okay.
A: A few hours of help a week would change everything.
A: I walk around the block when I can.
---
# c_nights | coord
Q: How have the nights been?
Q: Is anyone getting any sleep at your house?
A: Up two or three times a night.
A: Better since the new medicine.
A: Terrible, honestly.
A: We take turns.
---
# c_meals | coord
Q: Is there enough food in the house?
Q: Who is cooking and shopping right now?
Q: How are the meals going?
A: We're fine, volunteers from a church group drop off food.
A: I shop on Saturdays and cook most nights.
A: A neighbor brings something twice a week.
A: Mostly soup and toast, to be honest.
---
# c_mobility | coord
Q: How is the walking and getting around?
Q: Is a walker or a cane being used at home?
Q: Can the stairs be managed?
A: With the walker, slowly.
A: We've been staying on the first floor.
A: A cane, but I'd feel safer with a walker.
A: Not without help.
---
# c_meds | coord
Q: How is the medication routine going?
Q: Who manages the pills at home?
Q: Is the pill box working out?
A: I fill a weekly box every Sunday.
A: It's a lot of pills, but we have a system.
A: I lose track of the evening ones.
A: It's all in a folder on the counter.
---
# c_transport | coord
Q: How are you getting to appointments?
Q: Do you need transportation to the follow-up?
A: I drive, but I can't leave home for long.
A: A neighbor has been giving us rides.
A: We could use help with that.
A: Our daughter takes time off work.
---
# c_money | coord
Q: Are there any worries about costs or paperwork?
Q: Has anyone talked with you about the cost of services?
A: A little, the equipment bills worry me.
A: We got a letter and I don't understand it.
A: We're all right for now.
A: I haven't opened the mail in days.
---
# c_equipment | coord
Q: What equipment is already in the home?
Q: Is there a hospital bed, a commode, or a shower chair yet?
A: Just the walker so far.
A: A shower chair, and nothing else.
A: Nothing yet, we're using a regular bed.
A: We rented a commode last week.
---
# c_mood | coord
Q: How has the patient's mood been?
Q: Has there been any change in how the patient is feeling emotionally?
A: Quiet, mostly.
A: More irritable since the hospital stay.
A: Pretty good, really.
A: Weepy in the evenings.
---
# c_safety | coord
Q: Have there been any safety concerns at home?
Q: Are there any worries about wandering, the stove, or medication mix-ups?
A: Not so far.
A: The stove is a worry, we've started turning off the gas.
A: A door alarm would help.
A: A near miss with the pills last week.
---
# c_appts | coord
Q: Which appointments are coming up?
Q: Do you have a list of the follow-up visits?
A: Just the one with the clinic next week.
A: Therapy twice a week, and a specialist next month.
A: I'm not sure, I'd have to look at the folder.
A: Two, and I need help getting to both.
---
# am_function | am
Q: What is the patient's current level of function?
Q: How much help does the patient need with daily care?
A: One person assist for transfers, with a walker.
A: Independent in the room, but needs supervision on the stairs.
A: Needs help with bathing and dressing.
A: Mostly independent, with cueing for medications.
---
# am_diet | am
Q: Are there any diet restrictions?
Q: What is the patient eating?
A: Low salt, and soft foods for now.
A: Regular diet, with thickened liquids.
A: Diabetic diet, no concentrated sweets.
A: No restrictions, but small meals.
---
# am_orders | am
Q: What services were ordered at discharge?
Q: Which therapies are in the orders?
A: Skilled nursing twice a week, plus physical therapy.
A: Physical and occupational therapy, and a home safety evaluation.
A: Wound care three times a week.
A: Nursing for medication teaching, and a home health aide.
---
# am_timing | am
Q: When would you like to start?
Q: How soon is the patient ready?
A: As soon as possible, within two days.
A: The doctor wants it started by the end of the week.
A: After the patient is settled at home, early next week.
A: Tomorrow afternoon, if a bed is open.
---
# c_wishes | coord
Q: Has the patient said what matters most to them right now?
Q: What does the patient want for the next few months?
Q: Has there been a conversation about the patient's wishes?
A: To stay in the house, as long as it's safe.
A: To see the grandchildren more often.
A: They don't say much, but they keep asking about the garden.
A: We haven't really talked about it yet.
---
# c_decider | coord
Q: Who helps make decisions when the patient is not able to?
Q: Is there a healthcare proxy or power of attorney in place?
Q: Who should we call first for important decisions?
A: There is a form, but I'd need to find a copy.
A: The papers were signed last year, and the lawyer has them.
A: I'm the one, but my brother would want to be told.
A: I think so, but I'm not certain it's up to date.
---
# c_hospital | coord
Q: How did the hospital stay go, from your point of view?
Q: What was the hardest part of the hospital stay?
Q: Did anyone explain the discharge plan to you?
A: Rushed, honestly. We left with a stack of papers.
A: The nurses were kind, but nobody told us who to call.
A: It went fine until the last day.
A: They explained it, but I was too tired to take it in.
---
# c_teaching | coord
Q: Did the teaching about the new medicines make sense?
Q: Do you feel confident with the instructions you were given?
Q: Is there anything in the discharge papers that is unclear?
A: Mostly, but I'm not sure about the water pill.
A: I'd like to go over it again.
A: Yes, the nurse showed me everything twice.
A: The diet part is confusing.
---
# c_pain | coord
Q: How is the patient's pain at the moment?
Q: Is the pain controlled, or is it getting in the way?
A: It's better in the morning and worse by evening.
A: Controlled, as long as the medicine is on time.
A: Not great. It keeps them awake some nights.
A: They don't like to complain, so it's hard to tell.
---
# c_bathing | coord
Q: How is bathing and dressing going?
Q: Does the patient need help getting washed and dressed?
A: I help with the shower, and they manage the rest.
A: It's the hardest part of the day.
A: They can do it, but it takes a long time.
A: We use a sponge bath for now.
---
# c_toilet | coord
Q: Is toileting a problem during the day or at night?
Q: Are there any accidents, or trouble getting to the bathroom?
A: Mostly at night.
A: Not so far, but we're close to it.
A: We got a bedside commode and that helped.
A: There were a couple of accidents last week.
---
# c_memory | coord
Q: Have you noticed any changes in memory or confusion?
Q: Is the patient oriented, or is there some confusion in the evenings?
A: Some, especially at the end of the day.
A: Not really, sharp as ever.
A: A bit forgetful since the hospital.
A: Yes, and it frightens them.
---
# c_visitors | coord
Q: Are friends or family visiting?
Q: Does the patient have company during the week?
A: A few friends stop by on weekends.
A: Not as often as before.
A: The neighbors come by most days.
A: Mostly just me.
---
# c_house | coord
Q: Tell me about the layout of the home. Are there stairs to get in or to the bedroom?
Q: Is the bathroom on the same floor as the bedroom?
A: Three steps at the front, and the bedroom is upstairs.
A: It's a single floor, but the doorways are narrow.
A: The bathroom is downstairs, so we sleep down there too.
A: A ramp would help.
---
# c_forms | coord
Q: Have you been able to complete the forms we sent?
Q: Is there any paperwork we can help you with?
A: I filled out half of them.
A: Not yet, they're on the kitchen table.
A: We sent them back last week.
A: I'd appreciate help with the long one.
---
# c_emergency | coord
Q: Do you have a plan for what to do in an emergency?
Q: Who would you call if something happened in the middle of the night?
A: I'd call 911 first, then the rest of the family.
A: I have the numbers on the refrigerator.
A: I haven't really thought about it.
A: My neighbor has a key.
---
# c_family | coord
Q: Is there other family who can share the load?
Q: How involved is the rest of the family?
A: Not much, they live far away.
A: My sibling helps on weekends.
A: They call, but they don't visit.
A: We have a group chat to keep everyone updated.
---
# c_fatigue | coord
Q: How is your own health, with all of this on your plate?
Q: Have you been able to keep up with your own appointments?
A: I've let a few slide.
A: Fine, just tired.
A: My back is bothering me from the lifting.
A: I'm doing okay, I just need a break.
---
# am_cognition | am
Q: Is the patient oriented and able to follow directions?
Q: Are there any memory or cognitive concerns we should plan for?
A: Oriented most of the time, with some confusion in the evenings.
A: Fully oriented, and follows two step directions.
A: Some memory trouble, but no wandering.
A: Needs repeated reminders about the walker.
---
# am_continence | am
Q: Is the patient continent of bladder and bowel?
Q: Are there any continence needs?
A: Continent, with an occasional accident at night.
A: Needs a schedule for toileting.
A: Fully continent.
A: Uses an incontinence product at night.
---
# am_skin | am
Q: Are there any skin concerns or wounds?
Q: Does the patient have any pressure injuries?
A: A small red area on the heel, no open wound.
A: Skin is intact.
A: One healing wound on the left leg, being dressed daily.
A: No wounds, but fragile skin.
---
# am_aids | am
Q: What mobility aids does the patient use?
Q: Is the patient using a walker, a cane, or a wheelchair?
A: A rolling walker, mostly.
A: A cane indoors and a wheelchair for longer trips.
A: A walker, and a gait belt for transfers.
A: No aids, but needs standby assistance.
---
# am_oxygen | am
Q: Does the patient use oxygen or any other equipment?
Q: Are there any lines, tubes, or special equipment?
A: Oxygen at two liters at night only.
A: No oxygen, and no lines.
A: A nebulizer twice a day.
A: A hospital bed and an air mattress.
---
# am_behavior | am
Q: Are there any behaviors the team should be aware of?
Q: How has the patient been in the evenings?
A: Some agitation around dinner time.
A: Calm and cooperative.
A: A little anxious when family leaves.
A: No concerns.
---
# am_goals | am
Q: What are the family's goals for the stay?
Q: What does the family hope to get out of therapy?
A: To get back to walking on their own.
A: To go home with some help.
A: They want to regain strength and be safe on the stairs.
A: They'd like to talk about long term options as well.
---
# am_prior | am
Q: What was the patient's level of function before the hospital?
Q: How independent was the patient before this illness?
A: Independent, and still driving.
A: Walked with a cane and cooked for themselves.
A: Lived alone, with help from family for shopping.
A: Needed some help with stairs, otherwise independent.
---
# am_swallow | am
Q: Any problems with swallowing or choking?
Q: Is the patient on a modified diet?
A: No problems.
A: Thickened liquids since the hospital.
A: Soft foods, and they eat slowly.
A: Cuts food small, no coughing with meals.
---
# am_vitals | am
Q: How have the vital signs been the last day or two?
Q: Has the patient been stable?
A: Stable, afebrile, and eating well.
A: Blood pressure a little high this morning, otherwise stable.
A: Stable, with a slight cough.
A: Heart rate is steady, and oxygen is fine on room air.
---
"""


# --------------------------------------------------------------------------
# headers and footers (marked templates; same slot rules as the pools)

HEADER_TEXT = {
    "telehealth": [
        "TELEHEALTH VISIT TRANSCRIPT\nSession: {sess}\nStarted: {call_date} 14:02\nParticipants: {c_drfull} (clinician); {pt_full} (patient)",
        "Video visit transcript\nJoined via {url}\nCall date: {call_date}\nPlatform: telehealth app version {ver}",
        "Telehealth transcript, session {sess}\nDate: {call_date}",
        "VIDEO VISIT\nParticipants: {pt_full}, {c_dr}\nVisit started {call_date} at 10:43",
        "Telehealth visit transcript\nParticipants: clinician, patient\nPlatform: app version {ver}",
        "Visit transcript (auto generated)",
        "",
        "",
    ],
    "phone": [
        "NURSE LINE CALL TRANSCRIPT\nCall received {call_date} 21:15\nHandled by: {n_nurse}",
        "After hours call log\nDate: {call_date}\nCall type: triage",
        "Triage call transcript\nCall length: 11:42",
        "Nurse advice line transcript\nParticipants: nurse, caller",
        "Call recording transcript (auto generated)",
        "",
        "",
    ],
    "interpreter": [
        "INTERPRETER ASSISTED VISIT\nLanguage: {lang}\nInterpreter ID: {intid}\nDate: {call_date}",
        "Phone interpreter transcript\nLanguage: {lang}\nSession {sess}",
        "Interpreted visit transcript\nLanguage: {lang}\nParticipants: clinician, interpreter, patient, family",
        "Visit transcript with interpreter\nStarted {call_date} 09:30",
        "",
        "",
    ],
    "coordination": [
        "CARE COORDINATION CALL\nCase: {caseno}\nDate: {call_date}",
        "Social work call transcript\nCall date: {call_date}\nParticipants: {m_full} (social work)",
        "Care coordination transcript\nParticipants: case manager, caregiver",
        "Call transcript\nCall length: 18:20\nReference: {refno}",
        "",
        "",
    ],
}
FOOTER_TEXT = ["", "", "[end of transcript]", "End of transcript.", "Transcript generated automatically and may contain errors.", "Duration: 24:13"]

ROLE_KEY = {"C": "c", "P": "pt", "N": "n", "I": "i", "F": "f", "G": "g", "S": "s", "M": "m", "A": "a", "H": "h", "D": "c2"}
PERSON_FORMS = ("first", "last", "full", "hon", "dr", "drfull", "nurse")
PERSON_SLOT = re.compile(r"^(pt|c2|c|n|i|f|g|m|a|h|s)_(first|last|full|hon|dr|drfull|nurse)$")
SLOT_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")
LINE_RE = re.compile(r"^([A-Z]): ?(.*)$")

SAFE_SLOTS = {"drug", "drug2", "drug3", "dose", "dose2", "sym", "sym2", "dur", "reldate", "bp", "a1c", "ver", "lang", "timeword", "media", "ext_only",
              "f_rel", "pt_he", "pt_He", "pt_his", "pt_His", "pt_him"}
OVER89_SLOTS = {"age_over", "age_decade"}
PHI_SLOTS = {"dob", "addr", "addr2", "loc", "loc2", "phone", "phone2", "ext_phone", "nl_phone", "fax", "email", "mrn", "member", "acct", "sess", "code", "url",
             "ip", "authno", "refno", "caseno", "rx", "intid", "call_date", "date_future", "date_future2", "date_past", "month_year", "year_hist", "year_hist2",
             "fac_clinic", "fac_hosp", "fac_pharm", "fac_pharm2", "fac_lab", "fac_hh", "fac_hospice", "fac_dme", "fac_adc", "fac_sl", "fac_snf", "fac_mail",
             "fac_urgent", "fac_funeral", "fac_nl"} | OVER89_SLOTS
PERSIST = {"dob", "addr", "addr2", "loc", "phone", "phone2", "nl_phone", "email", "mrn", "member", "acct", "sess", "code", "url", "ip", "intid", "call_date",
           "age_over", "age_decade", "f_rel", "lang", "ver"} | {s for s in PHI_SLOTS if s.startswith("fac_")}
FAC_KIND = {"fac_clinic": "clinic", "fac_hosp": "hosp", "fac_pharm": "pharm", "fac_pharm2": "pharm", "fac_lab": "lab", "fac_hh": "hh", "fac_hospice": "hospice",
            "fac_dme": "dme", "fac_adc": "adc", "fac_sl": "sl", "fac_snf": "snf", "fac_mail": "mail", "fac_urgent": "urgent", "fac_funeral": "funeral", "fac_nl": "clinic"}


class Variant:
    __slots__ = ("lines", "slots", "phi", "needs_over89", "roles")

    def __init__(self, lines, slots):
        self.lines = lines
        self.slots = slots
        self.roles = {r for r, _ in lines}
        self.needs_over89 = bool(slots & OVER89_SLOTS)
        self.phi = any(s in PHI_SLOTS or PERSON_SLOT.match(s) for s in slots)


def _slot_known(name: str) -> bool:
    return name in SAFE_SLOTS or name in PHI_SLOTS or bool(PERSON_SLOT.match(name))


DROPPED = Counter()


def _strip_slots(text: str) -> str:
    return SLOT_RE.sub(" ", text)


def parse_pool(name: str, text: str) -> list[Variant]:
    out = []
    for block in text.strip().split("\n---\n"):
        lines = []
        for raw in block.strip().splitlines():
            m = LINE_RE.match(raw)
            if not m:
                raise ValueError(f"pool {name}: bad line {raw!r}")
            lines.append((m[1], m[2]))
        slots = set()
        for _, t in lines:
            slots.update(SLOT_RE.findall(t))
        for s in slots:
            if not _slot_known(s):
                raise ValueError(f"pool {name}: unknown slot {s}")
        # blind-bank exclusion pre-check on the fixed words only (counts, never strings)
        if excluded_sequences(" ".join(_strip_slots(t) for _, t in lines)) > 0:
            DROPPED["pool_variants_dropped"] += 1
            continue
        out.append(Variant(lines, frozenset(slots)))
    return out


def parse_headers() -> dict[str, list[Variant]]:
    out = {}
    for fam, items in HEADER_TEXT.items():
        vs = []
        for t in items:
            slots = frozenset(SLOT_RE.findall(t))
            for s in slots:
                if not _slot_known(s):
                    raise ValueError(f"header slot {s}")
            if t and excluded_sequences(_strip_slots(t)) > 0:
                DROPPED["header_variants_dropped"] += 1
                continue
            v = Variant([("H", t)], slots)
            vs.append(v)
        out[fam] = vs
    return out


POOLS: dict[str, list[Variant]] = {}
HEADERS: dict[str, list[Variant]] = {}
TOPICS: dict[str, list] = {}
VOCAB: set[str] = set()


def parse_topics():
    for block in TOPIC_TEXT.strip().split("\n---"):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        name, group = [x.strip() for x in lines[0].lstrip("# ").split("|")]
        qs = [ln[3:] for ln in lines[1:] if ln.startswith("Q: ")]
        ans = [ln[3:] for ln in lines[1:] if ln.startswith("A: ")]
        for t in qs + ans:
            for s_ in SLOT_RE.findall(t):
                if s_ not in SAFE_SLOTS:
                    raise ValueError(f"topic {name}: slot {s_} is not a safe slot")
        keep_q = [t for t in qs if excluded_sequences(_strip_slots(t)) == 0]
        keep_a = [t for t in ans if excluded_sequences(_strip_slots(t)) == 0]
        DROPPED["topic_lines_dropped"] += (len(qs) - len(keep_q)) + (len(ans) - len(keep_a))
        if keep_q and keep_a:
            TOPICS.setdefault(group, []).append((name, keep_q, keep_a))
        for t in qs + ans:
            VOCAB.update(w.lower() for w in re.findall(r"[A-Za-z']+", _strip_slots(t)))


def init_pools():
    if POOLS:
        return
    DROPPED.clear()
    for name, text in POOL_TEXT.items():
        POOLS[name] = parse_pool(name, text)
        for ln in text.splitlines():
            VOCAB.update(w.lower() for w in re.findall(r"[A-Za-z']+", _strip_slots(ln)))
    parse_topics()
    HEADERS.update(parse_headers())
    for items in HEADER_TEXT.values():
        for t in items:
            VOCAB.update(w.lower() for w in re.findall(r"[A-Za-z']+", _strip_slots(t)))
    VOCAB.update(w.lower() for lst in (_DRUGS, _SYMS, _DURS, _RELDATES, _FAMILY_RELS, _MONTHS, _WEEKDAYS, _STEM_A, _ADJ, _NOUN, _STREET) for item in lst for w in item.split())


# --------------------------------------------------------------------------
# registry, context, persons


class Reg:
    def __init__(self, cap: int):
        self.cap = cap
        self.usage: Counter = Counter()
        self.rej: Counter = Counter()

    def commit(self, spans):
        seen = set()
        for sp in spans:
            seen.add(sp["text"].casefold())
            if sp["label"] == "NAME":
                for tok in sp["text"].split():
                    seen.add("n:" + tok.casefold())
        for k in seen:
            self.usage[k] += 1


class Person:
    def __init__(self, key, first, last, gender, c):
        self.key, self.first, self.last, self.gender = key, first, last, gender
        base = "Mr" if gender == "m" else c.rng.choice(["Ms", "Ms", "Mrs"])
        self.hon_base = "Dr" if key in ("c", "c2") else ("Nurse" if key == "n" else base)
        self.dot = c.title_dot and self.hon_base != "Nurse"

    @property
    def title(self):
        return self.hon_base + ("." if self.dot else "")

    @property
    def full(self):
        return f"{self.first} {self.last}"


_FEMALE_RELS = {"wife", "daughter", "sister", "granddaughter", "niece", "mother", "daughter-in-law"}

ROLE_WORDS = {"C": ["Clinician", "Provider", "Doctor", "Physician"], "D": ["On-call clinician", "Provider"], "P": ["Patient"],
              "N": ["Nurse", "Triage nurse", "RN"], "I": ["Interpreter"], "F": ["Caregiver", "Family member"], "G": ["Neighbor", "Family member"],
              "S": ["Scheduler", "Staff"], "M": ["Case manager", "Care coordinator", "Social worker"], "A": ["Agency rep", "Intake coordinator", "Representative"],
              "H": ["Pharmacist"]}
NAMED_P = {"P": 0.92, "C": 0.92, "D": 0.9, "N": 0.8, "I": 0.3, "F": 0.88, "G": 0.8, "S": 0.7, "M": 0.92, "A": 0.8, "H": 0.7}
FILLERS = ["um", "uh", "so", "yeah", "okay so", "you know", "like", "well", "i mean", "right so"]
INTERJECT = ["mm hmm", "yeah", "right", "okay", "uh huh", "sure", "[crosstalk]", "mm"]
MISHEAR = {"metformin": "met form in", "lisinopril": "lysine april", "atorvastatin": "a tore va statin", "amlodipine": "am lodge a peen", "sertraline": "sir tra lean",
           "levothyroxine": "levo thigh rocks in", "omeprazole": "o mep ra zole", "gabapentin": "gabba pen tin", "albuterol": "all beauty role", "losartan": "low sar tan",
           "metoprolol": "metro pro lol", "prednisone": "pred ni zone", "ibuprofen": "i b profin", "naproxen": "na proxen", "furosemide": "fur o say mide",
           "hypertension": "high per tension", "pharmacy": "farmacy", "prescription": "prescrip shun", "appointment": "a point meant", "cough": "cuff",
           "dizziness": "dizzy ness", "acetaminophen": "a set a min o fin", "warfarin": "war far in", "apixaban": "a pix a ban"}
MISHEAR_RARE = {"there": "their", "your": "you're", "to": "too", "for": "four", "right": "write", "two": "to", "weeks": "weaks", "know": "no", "here": "hear"}


class Ctx:
    def __init__(self, spec, rng, reg):
        self.spec, self.rng, self.reg = spec, rng, reg
        self.phi = spec["phi"]
        self.stt = spec["stt"]
        self.case = spec["case"]
        self.over89 = spec["over89"] and self.phi
        self.rolemap: dict[str, str] = {}
        self.persist: dict[str, str] = {}
        self.persons: dict[str, Person] = {}
        self.name_tokens: set[str] = set()
        self.seen: dict[str, set] = {}
        self.tcount: Counter = Counter()
        self.tpairs: set = set()
        self.words = 0
        self.drift_turn = 0
        self.title_dot = rng.random() < 0.6
        self.gender = rng.choice("mf")
        self.age = rng.randrange(90, 104) if self.over89 else rng.randrange(24, 89)
        self.date_style = rng.choice(["iso", "rfc", "num", "mdy", "dmy"])
        self.dob_style = rng.choice(["mdy_ord", "mdy", "num", "dmy", "num_year"])
        self.fut_styles = ["dow_ord", "dow", "num", "short"]
        self.role_upper = rng.random() < 0.5
        self.target = rng.randrange(spec["words_lo"], spec["words_hi"] + 1)
        self.call_dt = self._pick_call_date()

    # -- drawing -----------------------------------------------------------
    def draw(self, make, tries=200):
        for _ in range(tries):
            v = make(self.rng)
            if excluded_value(v):
                self.reg.rej["value_exclusion"] += 1
                continue
            if self.reg.usage[v.casefold()] >= self.reg.cap:
                self.reg.rej["value_cap"] += 1
                continue
            return v
        raise Exhausted

    def _pick_call_date(self):
        lo, hi = dt.date(2023, 1, 9), dt.date(2026, 9, 28)
        for _ in range(500):
            d = rand_date(self.rng, lo, hi)
            s = fmt_date(d, self.date_style)
            if not excluded_value(s) and self.reg.usage[s.casefold()] < self.reg.cap:
                return d
        raise Exhausted

    # -- people ------------------------------------------------------------
    def person(self, key: str) -> Person:
        if key in self.persons:
            return self.persons[key]
        if not self.phi:
            raise RuntimeError("PHI-free draft must not create people")
        rng = self.rng
        if key == "f":
            rel = self.get("f_rel")
            gender = "f" if rel in _FEMALE_RELS else "m"
        elif key == "pt":
            gender = self.gender
        else:
            gender = rng.choice("mf")
        for _ in range(600):
            first, last = gen_given(rng), gen_surname(rng)
            low = [first.casefold(), last.casefold()]
            if min(len(first), len(last)) < 3 or low[0] == low[1] or low[0] in VOCAB or low[1] in VOCAB:
                self.reg.rej["name_vocabulary"] += 1
                continue
            if any(t in self.name_tokens for t in low):
                continue
            if any(excluded_value(x) for x in (first, last, f"{first} {last}")):
                self.reg.rej["value_exclusion"] += 1
                continue
            if any(self.reg.usage["n:" + t] >= self.reg.cap for t in low):
                self.reg.rej["value_cap"] += 1
                continue
            if identity_split(canonical_identity_group(f"{first} {last}")) != "train":
                self.reg.rej["identity_partition"] += 1
                continue
            break
        else:
            raise Exhausted
        self.name_tokens.update(low)
        p = Person(key, first, last, gender, self)
        self.persons[key] = p
        return p

    def name_mark(self, p: Person, form: str) -> str:
        first, last = p.first, p.last
        if self.stt and self.rng.random() < 0.12:
            first, last = respell(first, self.rng), respell(last, self.rng)
        if form == "first":
            return M("NAME", first)
        if form == "last":
            return M("NAME", last)
        if form == "full":
            return M("NAME", f"{first} {last}")
        if form == "hon":
            return f"{p.title} {M('NAME', last)}"
        if form == "dr":
            return f"Dr. {M('NAME', last)}"
        if form == "drfull":
            return f"Dr. {M('NAME', f'{first} {last}')}"
        if form == "nurse":
            return f"Nurse {M('NAME', first)}"
        raise KeyError(form)

    # -- slots -------------------------------------------------------------
    def get(self, name):
        return self.slot(name, self.persist)

    def slot(self, name, vol):
        if name in self.persist:
            return self.persist[name]
        if name in vol:
            return vol[name]
        value = self._make(name, vol)
        (self.persist if name in PERSIST else vol)[name] = value
        return value

    def _fut(self, lo, hi, styles=None, key=None):
        styles = styles or self.fut_styles
        return self.draw(lambda r: fmt_date(self.call_dt + dt.timedelta(days=r.randrange(lo, hi)), r.choice(styles)))

    def _make(self, name, vol):
        r = self.rng
        m = PERSON_SLOT.match(name)
        if m:
            return self.name_mark(self.person(m[1]), m[2])
        if name in ("pt_he", "pt_He", "pt_his", "pt_His", "pt_him"):
            f = self.gender == "f"
            return {"pt_he": "she" if f else "he", "pt_His": "Her" if f else "His", "pt_his": "her" if f else "his", "pt_him": "her" if f else "him", "pt_He": "She" if f else "He"}[name]
        if name in FAC_KIND:
            return M("FACILITY", self.draw(lambda rr: gen_facility(rr, FAC_KIND[name])))
        simple = {
            "drug": lambda: r.choice(_DRUGS), "dose": lambda: r.choice(_DOSES), "dose2": lambda: r.choice(_DOSES),
            "sym": lambda: r.choice(_SYMS), "dur": lambda: r.choice(_DURS), "reldate": lambda: r.choice(_RELDATES),
            "bp": lambda: f"{r.randrange(108, 168)} over {r.randrange(62, 100)}", "a1c": lambda: f"{r.randrange(52, 99) / 10:.1f}",
            "ver": lambda: f"{r.randrange(2, 9)}.{r.randrange(0, 30)}.{r.randrange(0, 12)}", "lang": lambda: r.choice(_LANGS),
            "timeword": lambda: f"{r.choice([8, 9, 10, 11, 1, 2, 3, 4])}:{r.choice(['00', '15', '30', '45'])} {'AM' if r.random() < 0.5 else 'PM'}",
            "media": lambda: r.choice(_MEDIA), "ext_only": lambda: str(r.randrange(1000, 9999)), "f_rel": lambda: r.choice(_FAMILY_RELS),
        }
        if name in simple:
            return simple[name]()
        if name == "drug2":
            return self._distinct(lambda: r.choice(_DRUGS), vol, ["drug"])
        if name == "drug3":
            return self._distinct(lambda: r.choice(["ibuprofen", "acetaminophen", "naproxen", "cetirizine", "loratadine", "famotidine"]), vol, ["drug", "drug2"])
        if name == "sym2":
            return self._distinct(lambda: r.choice(_SYMS), vol, ["sym"])
        if name == "dob":
            style = self.dob_style
            return M("DATE", self.draw(lambda rr: fmt_date(self.call_dt - dt.timedelta(days=int(self.age * 365.25) + rr.randrange(0, 365)), style)))
        if name == "age_over":
            n = self.age
            return M("AGE", self.draw(lambda rr: str(n) if rr.random() < 0.6 else spell_age(n)))
        if name == "age_decade":
            return M("AGE", self.draw(lambda rr: "nineties"))
        if name == "call_date":
            return M("DATE", fmt_date(self.call_dt, self.date_style))
        if name == "date_future":
            return M("DATE", self._fut(2, 45))
        if name == "date_future2":
            return M("DATE", self._fut(46, 120))
        if name == "date_past":
            return M("DATE", self.draw(lambda rr: fmt_date(self.call_dt - dt.timedelta(days=rr.randrange(3, 330)), rr.choice(["short", "mdy", "num"]))))
        if name == "month_year":
            return M("DATE", self.draw(lambda rr: fmt_date(self.call_dt - dt.timedelta(days=rr.randrange(200, 9000)), "month_year")))
        if name in ("year_hist", "year_hist2"):
            return M("DATE", self.draw(lambda rr: str(self.call_dt.year - rr.randrange(2, 58))))
        if name == "addr" or name == "addr2":
            return M("ADDRESS", self.draw(gen_street_address))
        if name == "loc":
            return M("LOCATION", self.draw(gen_locality))
        if name == "loc2":
            return M("LOCATION", self.draw(gen_locality))
        if name in ("phone", "phone2"):
            return M("PHONE", self.draw(lambda rr: gen_phone(rr)))
        if name in ("ext_phone", "nl_phone"):
            return M("PHONE", self.draw(lambda rr: gen_phone(rr, ext=True)))
        if name == "fax":
            return M("PHONE", self.draw(lambda rr: gen_phone(rr)))
        if name == "email":
            p = self.person("pt")
            return M("EMAIL", self.draw(lambda rr: gen_email(rr, p.first, p.last)))
        if name == "mrn":
            return M("MRN", self.draw(gen_mrn))
        if name == "member":
            return M("ID", self.draw(lambda rr: gen_code(rr, "member")))
        if name == "acct":
            return M("ACCOUNT", self.draw(lambda rr: gen_code(rr, "account")))
        if name == "sess":
            return M("ID", self.draw(lambda rr: gen_code(rr, "session")))
        if name == "code":
            return M("ID", self.draw(lambda rr: gen_code(rr, "activation")))
        if name == "url":
            return M("URL", self.draw(gen_url))
        if name == "ip":
            return M("IP", self.draw(gen_ip))
        if name == "authno":
            return M("ID", self.draw(lambda rr: gen_code(rr, "auth")))
        if name == "refno":
            return M("ID", self.draw(lambda rr: gen_code(rr, "referral")))
        if name == "caseno":
            return M("ID", self.draw(lambda rr: gen_code(rr, "case")))
        if name == "rx":
            return M("ID", self.draw(lambda rr: gen_code(rr, "rx")))
        if name == "intid":
            return M("ID", self.draw(lambda rr: gen_code(rr, "interp")))
        raise KeyError(name)

    def _distinct(self, make, vol, others):
        taken = {self.persist.get(o) or vol.get(o) for o in others}
        for _ in range(50):
            v = make()
            if v not in taken:
                return v
        return v

    # -- picking dialogue -------------------------------------------------
    def eligible(self, pool, roles=None):
        out = []
        for v in POOLS[pool]:
            if v.needs_over89 and not self.over89:
                continue
            if not self.phi and v.phi:
                continue
            if roles is not None and not v.roles <= roles:
                continue
            out.append(v)
        return out

    def pick(self, pool, k=1, roles=None, rolemap=None):
        elig = self.eligible(pool, roles)
        seen = self.seen.setdefault(pool, set())
        turns = []
        for _ in range(k):
            fresh = [v for v in elig if id(v) not in seen]
            if not fresh:
                break
            order = sorted(fresh, key=lambda v: -(self.rng.random() ** (1.0 / (3.0 if v.phi else 1.0))))
            for v in order:
                try:
                    block = self._expand(v, rolemap or {})
                except Exhausted:
                    continue
                seen.add(id(v))
                turns.extend(block)
                break
        return turns

    def topic(self, group, asker="C", answerer="P", wrap=False):
        rng = self.rng
        limit = 3 if group in ("coord", "am") else 2
        cands = [t for t in TOPICS[group] if self.tcount[t[0]] < limit]
        if not cands:
            return []
        fresh = [t for t in cands if self.tcount[t[0]] == 0] or cands
        name, qs, ans = rng.choice(fresh)
        for _ in range(8):
            qi, ai = rng.randrange(len(qs)), rng.randrange(len(ans))
            if (name, qi, ai) not in self.tpairs:
                break
        self.tpairs.add((name, qi, ai))
        self.tcount[name] += 1
        vol: dict[str, str] = {}
        q = SLOT_RE.sub(lambda m: self.slot(m[1], vol), qs[qi])
        a = SLOT_RE.sub(lambda m: self.slot(m[1], vol), ans[ai])
        if wrap:
            lang = self.get("lang")
            block = [(asker, q), ("I", f"[speaking {lang}] {q}"), (answerer, f"[speaking {lang}]"), ("I", a)]
        else:
            block = [(asker, q), (answerer, a)]
        for _, t in block:
            self.words += len(t.split())
        return block

    def _expand(self, v, rolemap):
        vol: dict[str, str] = {}
        block = []
        for role, tmpl in v.lines:
            text = SLOT_RE.sub(lambda m: self.slot(m[1], vol), tmpl)
            block.append((rolemap.get(role, role), text))
        for _, t in block:
            self.words += len(MARK.sub(lambda m: m[2], t).split())
        return block


# --------------------------------------------------------------------------
# plans


def _weighted(rng, items):
    total = sum(w for _, w in items)
    x = rng.random() * total
    for it, w in items:
        x -= w
        if x <= 0:
            return it
    return items[-1][0]


def _middle(c, weights, budget_words, roles=None, rolemap=None, asker="C", answerer="P", wrap=False):
    turns = []
    empty = 0
    while c.words < budget_words and empty < 60:
        pool = _weighted(c.rng, weights)
        if pool.startswith("T:"):
            block = c.topic(pool[2:], asker=asker, answerer=answerer, wrap=wrap)
            if rolemap:
                block = [(rolemap.get(r_, r_), t_) for r_, t_ in block]
        else:
            k = 1 if c.rng.random() < 0.55 else 2
            block = c.pick(pool, k, roles=roles, rolemap=rolemap)
        if block:
            empty = 0
            turns += block
        else:
            empty += 1
    return turns


def plan_telehealth(c):
    sub = c.spec["layout"]
    r = c.rng
    t = []
    t += c.pick("T_TECH", 1 if r.random() < 0.75 else 2)
    t += c.pick("T_INTRO")
    t += c.pick("T_VERIFY", 1 if c.target < 900 else 2)
    if c.over89:
        t += c.pick("T_AGE")
    t += c.pick("T_CONSENT")
    c.drift_turn = len(t)
    t += c.pick("T_CC")
    weights = {
        "acute": [("T:hpi", 9), ("T_EXAM", 3), ("T_HIST", 2), ("T_MEDS", 2), ("T:med", 2), ("T:life", 2), ("T_SOCIAL", 1), ("T_NEG", 1)],
        "chronic": [("T:med", 4), ("T_MEDS", 4), ("T:hpi", 5), ("T:life", 5), ("T_HIST", 2), ("T_SOCIAL", 2), ("T_NEG", 1), ("T_EXAM", 1)],
        "medreview": [("T_MEDS", 6), ("T:med", 5), ("T:life", 3), ("T_SOCIAL", 2), ("T:hpi", 3), ("T_NEG", 1), ("T_HIST", 2)],
        "postdischarge": [("T_HIST", 3), ("T_SOCIAL", 3), ("T:hpi", 5), ("T_MEDS", 3), ("T:med", 2), ("T:life", 3), ("T_NEG", 1), ("T_EXAM", 1)],
    }[sub]
    if r.random() < 0.6:
        t += c.pick("T_NEG")
    t += _middle(c, weights, c.target - 200)
    t += c.pick("T_PLAN", 1 if r.random() < 0.6 else 2)
    t += c.pick("T_PHARM")
    t += c.pick("T_FOLLOW", 1 if r.random() < 0.5 else 2)
    t += c.pick("T_CLOSE")
    return t


def plan_phone(c):
    sub = c.spec["layout"]
    r = c.rng
    t = []
    if sub == "caller_behalf":
        t += c.pick("LB_OPEN")
        if c.over89:
            t += c.pick("LB_AGE")
        c.drift_turn = len(t)
        weights = [("LB_SYMP", 6), ("L_RED", 3), ("T_NEG", 1)]
        rm = {"P": "F", "C": "N"}

        def mid():
            out = []
            empty = 0
            while c.words < c.target - 220 and empty < 30:
                pool = _weighted(r, weights)
                got = c.pick(pool, 2 if pool == "LB_SYMP" else 1, rolemap=rm if pool in ("L_RED", "T_NEG") else None)
                empty = 0 if got else empty + 1
                out += got
            return out
        t += mid()
        t += c.pick("LB_DISP")
        t += c.pick("L_CLOSE", rolemap={"P": "F"})
        return t
    t += c.pick("L_OPEN")
    t += c.pick("L_CALLER")
    if c.over89:
        t += c.pick("L_AGE")
    c.drift_turn = len(t)
    rm = {"C": "N"}
    weights = {"nurse_line": [("T:hpi", 9), ("L_RED", 4), ("T_MEDS", 2), ("T:med", 2), ("T_HIST", 2), ("T:life", 2), ("T_NEG", 1)],
               "after_hours": [("T:hpi", 8), ("L_RED", 5), ("T_MEDS", 1), ("T:med", 1), ("T_HIST", 2), ("T_NEG", 1)]}[sub]
    t += _middle(c, weights, c.target - 250, rolemap=rm, asker="N")
    t += c.pick("L_DISP", 1 if r.random() < 0.6 else 2)
    if sub == "after_hours" or r.random() < 0.4:
        t += c.pick("L_ONCALL")
    t += c.pick("L_CALLBACK")
    if r.random() < 0.35:
        t += c.pick("L_TRANSFER")
    t += c.pick("L_CLOSE")
    return t


def plan_interpreter(c):
    r = c.rng
    t = []
    t += c.pick("I_OPEN")
    t += c.pick("I_INTRO")
    t += c.pick("I_VERIFY", 1 if c.target < 900 else 2)
    if c.over89:
        t += c.pick("I_AGE")
    c.drift_turn = len(t)
    t += _middle(c, [("T:hpi", 9), ("I_HPI", 3), ("I_FAM", 3), ("T:med", 3), ("I_MEDS", 2), ("T:life", 2), ("I_NEG", 1)], c.target - 250, wrap=True)
    t += c.pick("I_PLAN", 1 if r.random() < 0.6 else 2)
    t += c.pick("I_FOLLOW")
    t += c.pick("I_CLOSE")
    return t


def plan_coordination(c):
    sub = c.spec["layout"]
    t = []
    am = {"M", "A"}
    cg = {"M", "F", "P", "G"}
    if sub == "cm_caregiver":
        t += c.pick("C_OPEN_CG", roles=cg)
        t += c.pick("C_ID", 1 if c.target < 900 else 2, roles=cg)
        if c.over89:
            t += c.pick("C_AGE", roles=cg)
        c.drift_turn = len(t)
        t += _middle(c, [("T:coord", 9), ("C_SIT", 3), ("C_SVC", 4), ("C_AUTH", 2), ("C_LOGISTICS", 2), ("C_NEG", 1), ("C_MEET", 1)], c.target - 200, roles=cg, asker="M", answerer="F")
        t += c.pick("C_CLOSE", roles=cg)
    elif sub == "sw_family":
        t += c.pick("C_OPEN_CG", roles=cg)
        t += c.pick("C_ID", roles=cg)
        if c.over89:
            t += c.pick("C_AGE", roles=cg)
        c.drift_turn = len(t)
        t += _middle(c, [("C_SW_FAM", 6), ("T:coord", 6), ("C_SIT", 3), ("C_SVC", 2), ("C_NEG", 1)], c.target - 200, roles=cg, asker="M", answerer="F")
        t += c.pick("C_CLOSE", roles=cg)
    else:
        first = "C_AGENCY" if sub == "cm_agency" else "C_DISCHARGE"
        t += c.pick(first, roles=am)
        t += c.pick("C_AGENCY", roles=am)
        if c.over89:
            t += c.pick("C_AMAGE")
        c.drift_turn = len(t)
        t += _middle(c, [("T:am", 5), ("C_AGENCY", 3), ("C_DISCHARGE", 3), ("C_AM", 3), ("C_AUTH", 2), ("C_LOGISTICS", 2), ("C_NEG", 1)], c.target - 200, roles=am, asker="A", answerer="M")
        t += c.pick("C_CLOSE", roles=am)
    return t


PLANS = {"telehealth": plan_telehealth, "phone": plan_phone, "interpreter": plan_interpreter, "coordination": plan_coordination}
LAYOUTS = {"telehealth": ["acute", "chronic", "medreview", "postdischarge"], "phone": ["nurse_line", "after_hours", "caller_behalf"],
           "interpreter": ["phone_interpreter", "video_interpreter"], "coordination": ["cm_caregiver", "cm_agency", "discharge_planning", "sw_family"]}
HEADER_FAMILY = {"telehealth": "telehealth", "phone": "phone", "interpreter": "interpreter", "coordination": "coordination"}


# --------------------------------------------------------------------------
# labels and rendering


def label_marked(kind, *, first="", last="", title=None, role=None, name_form="full", role_upper=False, ts_prefix=None, title_full=False):
    """Marked text of a speaker label, without the closing colon."""
    name = {"full": f"{first} {last}", "first": first, "last": last}[name_form]
    nm = M("NAME", name)
    if kind in ("name_full", "name_first", "name_last", "ts"):
        core = f"{title} {nm}" if (kind == "ts" and title) else nm
    elif kind == "title_last":
        core = f"{title} {nm}" if title else nm
    elif kind == "role_paren":
        word = role.upper() if role_upper else role
        core = f"{word} ({nm})"
    elif kind == "name_role":
        core = f"{nm} ({role.upper() if role_upper else role.lower()})"
    elif kind == "role_only":
        core = role.upper() if role_upper else role
    else:
        raise KeyError(kind)
    return f"{ts_prefix} {core}" if ts_prefix else core


class Out:
    def __init__(self, case, stt):
        self.parts: list[str] = []
        self.spans: list[dict] = []
        self.n = 0
        self.case, self.stt = case, stt

    def _lit(self, s: str, strip: bool):
        if self.case == "lower":
            s = s.lower()
        elif self.case == "upper":
            s = s.upper()
        if strip:
            s = re.sub(r"(?<!\d)\.|\.(?!\d)|[,?!;\"“”]|(?<!\d):|:(?!\d)", "", s)
        self.parts.append(s)
        self.n += len(s)

    def raw(self, s: str):
        self.parts.append(s)
        self.n += len(s)

    def mk(self, marked: str, strip: bool = False):
        prev = 0
        for m in MARK.finditer(marked):
            self._lit(marked[prev:m.start()], strip)
            v = m[2]
            if self.case == "lower":
                v = v.lower()
            elif self.case == "upper":
                v = v.upper()
            if strip and m[1] in ("ADDRESS", "DATE", "LOCATION", "PHONE"):
                v = v.replace(",", "")
            self.spans.append({"start": self.n, "end": self.n + len(v), "label": m[1], "text": v})
            self.parts.append(v)
            self.n += len(v)
            prev = m.end()
        self._lit(marked[prev:], strip)

    def text(self):
        return "".join(self.parts)


def stt_body(c, s: str) -> str:
    """Mishearings, fillers and repeats on the literal parts of a turn."""
    rng = c.rng
    pieces = re.split(r"(\[\[[A-Z]+\|[^\[\]]+\]\])", s)
    for i, p in enumerate(pieces):
        if p.startswith("[["):
            continue

        def sub(m):
            w = m.group(0)
            lw = w.lower()
            if lw in MISHEAR and rng.random() < 0.5:
                return MISHEAR[lw]
            if lw in MISHEAR_RARE and rng.random() < 0.05:
                return MISHEAR_RARE[lw]
            return w
        p = re.sub(r"[A-Za-z']+", sub, p)
        if rng.random() < 0.5:
            p = re.sub(r"(?<=[.?!]) (?=[A-Za-z])", lambda m: f" {rng.choice(FILLERS)} " if rng.random() < 0.18 else " ", p)
        if i == 0 and rng.random() < 0.25:
            p = f"{rng.choice(FILLERS)} " + p
        if rng.random() < 0.03:
            p = re.sub(r"\b(the|a|my|i|it)\b ", lambda m: f"{m.group(1)} {m.group(1)} " if rng.random() < 0.5 else m.group(0), p, count=1)
        pieces[i] = p
    return "".join(pieces)


def split_overlap(c, text):
    """Split a long literal stretch into two parts so another speaker can cut in."""
    pieces = re.split(r"(\[\[[A-Z]+\|[^\[\]]+\]\])", text)
    for i, p in enumerate(pieces):
        if not p.startswith("[[") and len(p.split()) >= 9:
            words = p.split(" ")
            cut = len(words) // 2
            a = " ".join(words[:cut])
            b = " ".join(words[cut:])
            return "".join(pieces[:i]) + a, b + "".join(pieces[i + 1:])
    return None


def assemble(c: Ctx, turns, header: str, footer: str):
    rng = c.rng
    spec = c.spec
    roles = []
    for r_, _ in turns:
        if r_ not in roles:
            roles.append(r_)
    if c.phi:
        for r_ in roles:
            key = ROLE_KEY[r_]
            if key not in c.persons and rng.random() < NAMED_P[r_]:
                c.person(key)
        if not any(ROLE_KEY[r_] in c.persons for r_ in roles):
            c.person(ROLE_KEY[roles[0]])
    # label scheme
    base = spec["fmt"]
    fmt_of = {}
    for r_ in roles:
        fmt_of[r_] = rng.choice(FMTS) if (spec["hetero"] and c.phi) else base
    switch_fmt = {r_: rng.choice([f for f in FMTS if f != fmt_of[r_]]) for r_ in roles}
    name_form = rng.choice(["full", "first", "last"])
    rw = {r_: rng.choice(ROLE_WORDS[r_]) for r_ in roles}
    if "F" in rw and c.phi and rng.random() < 0.5:
        rw["F"] = c.get("f_rel").capitalize()
    if spec["layout"] == "sw_family" and "M" in rw:
        rw["M"] = "Social worker"
    ts_title = {r_: rng.random() < 0.3 for r_ in roles}
    title_last_only = {r_: rng.random() < 0.85 for r_ in roles}
    ts_extra = c.phi and rng.random() < 0.08
    dated_ts = c.phi and rng.random() < 0.05
    ts_style = rng.choice(["[{h:02d}:{m:02d}:{s:02d}]", "[{h:02d}:{m:02d}:{s:02d}]", "({h:02d}:{m:02d}:{s:02d})", "{h:02d}:{m:02d}:{s:02d}", "[{mm:02d}:{s:02d}]"])
    clock0 = rng.randrange(8 * 3600, 17 * 3600)
    elapsed = rng.randrange(0, 90)
    speaker_no = {r_: i + 1 for i, r_ in enumerate(roles)}
    stt_style = _weighted(rng, [("colon", 5), ("nl", 3), ("nocolon", 2)]) if c.stt else "colon"
    sep = "\n\n" if (not c.stt and rng.random() < 0.3) else "\n"

    # speech-to-text overlap: interjections and split turns
    seq = list(turns)
    if c.stt:
        new = []
        for r_, text in seq:
            part = split_overlap(c, text) if (rng.random() < 0.07 and len(roles) > 1) else None
            if part:
                other = rng.choice([x for x in roles if x != r_])
                new += [(r_, part[0]), (other, rng.choice(INTERJECT)), (r_, part[1])]
            else:
                new.append((r_, text))
                if len(roles) > 1 and rng.random() < 0.07:
                    new.append((rng.choice([x for x in roles if x != r_]), rng.choice(INTERJECT)))
        seq = new

    out = Out(c.case, c.stt)
    if header:
        out.mk(header, strip=c.stt)
        out.raw("\n\n")
    for idx, (r_, text) in enumerate(seq):
        key = ROLE_KEY[r_]
        p = c.persons.get(key)
        named_now = p is not None and not (spec["drift"] == "intro" and idx < c.drift_turn)
        kind = fmt_of[r_]
        if spec["drift"] == "switch" and idx >= c.drift_turn:
            kind = switch_fmt[r_]
        elapsed += rng.randrange(4, 41)
        ts_prefix = None
        if (kind == "ts" or ts_extra) and c.phi:
            h, rem = divmod(elapsed, 3600)
            m_, s = divmod(rem, 60)
            if dated_ts:
                cl = clock0 + elapsed
                ts_prefix = "[" + M("DATE", fmt_date(c.call_dt, "iso")) + f" {cl // 3600 % 24:02d}:{cl // 60 % 60:02d}:{cl % 60:02d}]"
            else:
                ts_prefix = ts_style.format(h=h, m=m_, s=s, mm=elapsed // 60)
        elif kind == "ts" and not c.phi:
            h, rem = divmod(elapsed, 3600)
            m_, s = divmod(rem, 60)
            ts_prefix = ts_style.format(h=h, m=m_, s=s, mm=elapsed // 60)
        if named_now:
            if kind == "ts":
                title = p.title if ts_title[r_] else None
                label = label_marked("ts", first=p.first, last=p.last, title=title, name_form="last" if title else name_form, ts_prefix=ts_prefix)
            elif kind == "title_last":
                label = label_marked("title_last", first=p.first, last=p.last, title=p.title, name_form="last" if title_last_only[r_] else "full", ts_prefix=ts_prefix)
            elif kind == "role_paren":
                label = label_marked("role_paren", first=p.first, last=p.last, role=rw[r_], name_form=name_form, role_upper=c.role_upper, ts_prefix=ts_prefix)
            elif kind == "name_role":
                role = rw[r_]
                label = label_marked("name_role", first=p.first, last=p.last, role=role, name_form=name_form, role_upper=c.role_upper, ts_prefix=ts_prefix)
            else:
                nf = {"name_full": "full", "name_first": "first", "name_last": "last"}[kind]
                label = label_marked(kind, first=p.first, last=p.last, name_form=nf, ts_prefix=ts_prefix)
        else:
            if c.stt:
                label = f"{ts_prefix} speaker {speaker_no[r_]}" if ts_prefix else f"speaker {speaker_no[r_]}"
            else:
                label = label_marked("role_only", role=rw[r_], role_upper=c.role_upper, ts_prefix=ts_prefix)
        body = stt_body(c, text) if c.stt else text
        out.mk(label, strip=c.stt)
        out.raw("\n" if (c.stt and stt_style == "nl") else (" " if (c.stt and stt_style == "nocolon") else ": "))
        out.mk(body, strip=c.stt)
        out.raw(sep if idx < len(seq) - 1 else "")
    if footer:
        out.raw("\n\n")
        out.mk(footer, strip=c.stt)
    return out.text(), out.spans


# --------------------------------------------------------------------------
# documents, specs, build

STT_HEADERS = ["", "", "automatic transcript", "recording transcript machine generated", "audio transcript {call_date} 14:02", "auto captions"]
NAME_FOUND_MIN = 3


def pick_header(c: Ctx, family: str) -> str:
    if c.stt:
        pool = [Variant([("H", t)], frozenset(SLOT_RE.findall(t))) for t in STT_HEADERS]
    else:
        pool = HEADERS[HEADER_FAMILY[family]]
    elig = [v for v in pool if c.phi or not v.phi]
    v = c.rng.choice(elig)
    vol: dict[str, str] = {}
    return SLOT_RE.sub(lambda m: c.slot(m[1], vol), v.lines[0][1])


def gen_doc(spec, rng, reg):
    c = Ctx(spec, rng, reg)
    base = spec["base"]
    turns = PLANS[base](c)
    if base == "phone" and spec["layout"] == "caller_behalf":
        pass
    header = pick_header(c, base)
    footer = "" if c.stt else rng.choice(FOOTER_TEXT)
    text, spans = assemble(c, turns, header, footer)
    text_out = text.strip()
    if text_out != text:
        raise Redraw("whitespace")
    if "[[" in text or "]]" in text:
        raise Redraw("markup_residue")
    if c.phi and not spans:
        raise Redraw("no_spans")
    if not c.phi and spans:
        raise Redraw("phi_free_has_span")
    if c.phi and not any(s["label"] == "NAME" for s in spans):
        raise Redraw("no_name")
    if len(text.split()) < 0.45 * spec["words_lo"]:
        raise Redraw("short_text")
    # every occurrence of a person's name must be annotated
    all_iv = [(s["start"], s["end"]) for s in spans]
    low = text.lower()
    if len(low) == len(text):
        for p in c.persons.values():
            for tok in (p.first, p.last):
                for m in re.finditer(r"(?<![\w'-])" + re.escape(tok.lower()) + r"(?![\w-])", low):
                    if not any(a <= m.start() and m.end() <= b for a, b in all_iv):
                        reg.rej["unlabeled_name_occurrence"] += 1
                        raise Redraw("unlabeled_name")
    # blind-bank exclusion filter on the whole text (counts only)
    if excluded_sequences(text) > 0:
        reg.rej["row_sequence_exclusion"] += 1
        raise Redraw("sequence_exclusion")
    errs = annotation_errors(text, spans)
    if errs:
        raise AssertionError(f"annotation errors in {spec['key']}: {errs[:3]}")
    for s in spans:
        if text[s["start"]:s["end"]] != s["text"]:
            raise AssertionError("span text differs from the annotated substring")
    # identities
    joined = " ".join(s["text"].casefold() for s in spans if s["label"] == "NAME")
    groups, ordered = {}, []
    for key, p in c.persons.items():
        if c.stt or p.first.casefold() in joined or p.last.casefold() in joined:
            h = canonical_identity_group(p.full)
            groups[key] = h
            ordered.append(h)
    if c.phi:
        identity_group = groups.get("pt") or ordered[0]
        identity_groups = sorted(set(ordered))
    else:
        identity_group = f"phi_free_{spec['key']}"
        identity_groups = []
    return {"text": text, "spans": spans, "identity_group": identity_group, "identity_groups": identity_groups,
            "words": len(text.split()), "named_roles": sorted(c.persons), "over89": c.over89}


_ENC = {}


def count_tokens(text: str):
    if "enc" not in _ENC:
        try:
            import tiktoken
            _ENC["enc"] = tiktoken.get_encoding("o200k_base")
        except Exception:  # pragma: no cover - environment dependent
            _ENC["enc"] = None
    enc = _ENC["enc"]
    if enc is None:
        return int(len(text.split()) * 1.4), False
    return len(enc.encode(text, disallowed_special=())), True


def _family_counts(n):
    raw = [(f, n * s) for f, s in FAMILY_SHARES]
    counts = {f: int(v) for f, v in raw}
    rest = n - sum(counts.values())
    for f, _v in sorted(raw, key=lambda x: -(x[1] - int(x[1])))[:rest]:
        counts[f] += 1
    return counts


def make_specs(n, seed):
    rng = random.Random(f"tk16:{seed}:specs")
    specs = []
    for family, k in _family_counts(n).items():
        neg = round(k * PHI_FREE_SHARE)
        pos = k - neg
        cases = ("lower", "lower", "lower", "lower", "normal", "upper") if family == "stt" else ("normal", "normal", "lower", "upper")
        combos = []
        while len(combos) < pos:
            batch = [(f, cs) for f in FMTS for cs in cases]
            rng.shuffle(batch)
            combos += batch
        positives = []
        for j in range(pos):
            fmt, case = combos[j]
            base = family if family != "stt" else _weighted(rng, [("telehealth", 0.35), ("phone", 0.30), ("coordination", 0.20), ("interpreter", 0.15)])
            layout = rng.choice(LAYOUTS[base])
            lc = _weighted(rng, [((name, lo, hi), w) for name, lo, hi, w in LENGTH_CLASSES])
            if layout in ("cm_agency", "discharge_planning") and lc[2] > 1000:
                lc = ("medium", 600, 1000)
            if layout == "caller_behalf" and lc[2] > 600:
                lc = ("short", 300, 600)
            spec = {"family": family, "base": base, "layout": layout, "phi": True, "stt": family == "stt", "fmt": fmt, "case": case,
                    "hetero": rng.random() < 0.25, "drift": _weighted(rng, [("none", 0.70), ("intro", 0.18), ("switch", 0.12)]),
                    "length_class": lc[0], "words_lo": lc[1], "words_hi": lc[2], "over89": rng.random() < 0.08}
            positives.append(spec)
        rng.shuffle(positives)
        allspecs = []
        for j, sp in enumerate(positives):
            sp["pair"] = f"tk16-pair-{family}-{j:04d}" if j < neg else None
            allspecs.append(sp)
        negatives = []
        for j in range(neg):
            twin = dict(positives[j])
            twin.update(phi=False, drift="none", over89=False, hetero=False, pair=positives[j]["pair"])
            negatives.append(twin)
        allspecs += negatives
        rng.shuffle(allspecs)
        for k_, sp in enumerate(allspecs):
            sp["index"] = k_
            sp["key"] = f"{family}-{k_:04d}"
            sp["id"] = f"tk16-{family}-{k_:04d}"
            sp["pair_id"] = sp["pair"] or sp["id"]
            specs.append(sp)
    return specs


def build(n=DEFAULT_ROWS, seed=DEFAULT_SEED, cap=None, log=None):
    init_pools()
    cap = cap if cap is not None else max(1, int(0.005 * n))
    reg = Reg(cap)
    specs = make_specs(n, seed)
    order = list(range(len(specs)))
    random.Random(f"tk16:{seed}:order").shuffle(order)
    redraws = Counter()
    rows = {}
    for count, i in enumerate(order):
        spec = specs[i]
        for attempt in range(120):
            rng = random.Random(f"tk16:{seed}:{spec['key']}:{attempt}")
            try:
                doc = gen_doc(spec, rng, reg)
            except Redraw as e:
                redraws[str(e)] += 1
                continue
            except Exhausted:
                redraws["pool_exhausted"] += 1
                continue
            reg.commit(doc["spans"])
            tokens, exact = count_tokens(doc["text"])
            rows[spec["key"]] = make_row(spec, doc, tokens, attempt)
            break
        else:
            raise RuntimeError(f"could not draw {spec['key']}: {dict(redraws)} {dict(reg.rej)}")
        if log and count % 100 == 0:
            log(f"  {count}/{len(order)} drawn")
    ordered = [rows[s["key"]] for s in specs]
    return ordered, {"cap": cap, "rejected_draws": dict(reg.rej), "redrawn_documents": dict(redraws),
                     "template_variants_dropped_by_exclusion": dict(DROPPED), "tokenizer_exact": count_tokens("x")[1]}


def make_row(spec, doc, tokens, attempt):
    case, fmt = spec["case"], spec["fmt"]
    phen = ["transcript", "speaker_label", f"case_{case}", f"layout_{spec['layout']}"]
    if spec["phi"]:
        phen.append(f"label_format_{fmt}")
        if spec["drift"] != "none":
            phen.append(f"label_change_{spec['drift']}")
        if spec["hetero"]:
            phen.append("mixed_label_formats")
        if doc["over89"]:
            phen.append("age_over_89")
    else:
        phen += ["role_only_labels", "no_phi", "matched_negative"]
    if spec["stt"]:
        phen.append("speech_to_text_raw")
    labels = {s["label"] for s in doc["spans"]}
    for lab, name in (("ID", "session_or_access_ids"), ("URL", "meeting_links"), ("PHONE", "phone_numbers"), ("IP", "ip_addresses"), ("DATE", "dates")):
        if lab in labels:
            phen.append(name)
    return {
        "id": spec["id"], "text": doc["text"], "spans": doc["spans"], "split": "train", "suite": "tk16_transcripts",
        "annotation_policy": POLICY_VERSION, "cohort": "long_document" if tokens > 512 else "realistic",
        "family": f"tk16_transcript_{spec['family']}", "template_family": f"tk16_transcript_{spec['family']}",
        "phenomena": phen, "identity_group": doc["identity_group"], "identity_groups": doc["identity_groups"],
        "pair_id": spec["pair_id"], "layout": spec["layout"], "source_family": spec["base"], "case": case,
        "label_format": fmt if spec["phi"] else "role_only", "label_change": spec["drift"], "length_class": spec["length_class"],
        "phi_free": not spec["phi"], "words": doc["words"], "tokens": tokens, "draw_attempt": attempt,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bank(rows, stats, out: Path, seed, n):
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    out.write_text(payload, encoding="utf-8")
    support = Counter(s["label"] for r in rows for s in r["spans"])
    from outis.types import ENTITY_TYPES
    label_support = {lab: support[lab] for lab in ENTITY_TYPES}
    tokens = sorted(r["tokens"] for r in rows)
    pct = lambda q: tokens[min(len(tokens) - 1, int(q * len(tokens)))]  # noqa: E731
    usage = Counter()
    for r in rows:
        for k in {s["text"].casefold() for s in r["spans"]}:
            usage[k] += 1
        for k in {t.casefold() for s in r["spans"] if s["label"] == "NAME" for t in s["text"].split()}:
            usage["n:" + k] += 1
    manifest = {
        "bank": BANK, "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(), "rows": len(rows), "documents": len(rows), "split": "train",
        "annotation_policy": POLICY_VERSION, "independent_final": False, "role": "training bank (development evidence only)",
        "generator": "scripts/takeover/author16_transcripts.py", "generator_sha256": _sha(Path(__file__)), "generator_version": GENERATOR_VERSION,
        "seed": seed, "requested_rows": n, "label_support": label_support, "entity_support": label_support, "label_counts": label_support,
        "families": dict(Counter(r["family"] for r in rows)), "source_families": dict(Counter(r["source_family"] for r in rows)),
        "layouts": dict(Counter(r["layout"] for r in rows)), "cases": dict(Counter(r["case"] for r in rows)),
        "label_formats_by_case": {f"{f}|{cs}": sum(1 for r in rows if r["label_format"] == f and r["case"] == cs and not r["phi_free"]) for f in FMTS for cs in CASES},
        "label_change": dict(Counter(r["label_change"] for r in rows)), "phi_free_rows": sum(1 for r in rows if r["phi_free"]),
        "token_length": {"min": tokens[0], "p10": pct(0.10), "p25": pct(0.25), "median": pct(0.5), "p75": pct(0.75), "p90": pct(0.90), "max": tokens[-1],
                         "over_512": sum(1 for t in tokens if t > 512), "over_2048": sum(1 for t in tokens if t > 2048), "mean": round(statistics.mean(tokens), 1)},
        "value_repetition": {"cap_rows": stats["cap"], "max_rows_for_any_span_value": max(usage.values()) if usage else 0,
                             "max_rows_for_any_name_token": max((v for k, v in usage.items() if k.startswith("n:")), default=0)},
        "exclusion_filter": {"source": "data/places_exclusions.sha256.json via outis.data.generators.places", "counts_only": True,
                             "value_exclusion_rejects": stats["rejected_draws"].get("value_exclusion", 0),
                             "row_sequence_exclusion_rejects": stats["rejected_draws"].get("row_sequence_exclusion", 0),
                             "template_variants_dropped": stats["template_variants_dropped_by_exclusion"],
                             "rows_matching_after_filter": sum(1 for r in rows if excluded_sequences(r["text"]) > 0)},
        "rejected_draws": stats["rejected_draws"], "redrawn_documents": stats["redrawn_documents"], "tokenizer": "o200k_base" if stats["tokenizer_exact"] else "estimate",
        "reserved_schemes": "phones 555-0100..0199; domains .test/.example/example.com|net|org; IPs 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24",
    }
    mpath = out.with_suffix(out.suffix + ".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def render_marked(marked: str, case: str = "normal"):
    """Text and spans (offsets and labels) for a marked string, using the bank's renderer."""
    out = Out(case, False)
    out.mk(marked)
    return out.text(), [{"start": sp["start"], "end": sp["end"], "label": sp["label"]} for sp in out.spans]


def rebuild_transcript_fixtures() -> dict:
    """Rebuild the TRANSCRIPT_FIXTURES texts and gold spans with the generator's own primitives."""
    lines = [
        label_marked("title_last", first="Anya", last="Marwick", title="dr", name_form="last", ts_prefix="[00:08:22]") + ": any palpitations?",
        label_marked("role_only", role="Patient") + ": no.",
        label_marked("role_paren", first="Anya", last="Brannock", role="CLINICIAN", name_form="last") + ": Take it with food.",
        label_marked("role_only", role="Interpreter") + ": She agrees.",
    ]
    speaker = "\n".join(lines)
    stamps = (f"Call started {M('DATE', fmt_date(dt.date(2025, 8, 24), 'iso'))} 14:02. "
              f"Note signed {M('DATE', fmt_date(dt.date(2023, 12, 18), 'rfc'))} at 10:43.")
    codes = (f"Video session {M('ID', 'VX-4471')} joined from {M('URL', 'https://visit.example.test/r/8812')}; "
             f"portal activation code {M('ID', 'K7TQ-2RMB')}. App version 4.12.0.")
    return {"speaker_label_names": render_marked(speaker), "timestamp_with_date": render_marked(stamps), "session_and_access_codes": render_marked(codes)}


def excerpt(row, rng, head=900, window=700):
    text = row["text"]
    spans = row["spans"]
    pieces = []
    cuts = [(0, min(len(text), head))]
    if len(text) > head + window + 200 and spans:
        mid = [s_ for s_ in spans if s_["start"] > head + 100] or spans
        centre = rng.choice(mid)["start"]
        lo = max(head, centre - window // 2)
        cuts.append((lo, min(len(text), lo + window)))
    for a, b in cuts:
        seg, prev = [], a
        for sp in spans:
            if sp["start"] >= a and sp["end"] <= b:
                seg.append(text[prev:sp["start"]])
                seg.append(f"[[{sp['label']}|{text[sp['start']:sp['end']]}]]")
                prev = sp["end"]
        seg.append(text[prev:b])
        pieces.append("".join(seg))
    return "\n[...]\n".join(pieces), len(text) > cuts[-1][1]


def write_spot_check(rows, path: Path, seed: int, n: int = 100):
    rng = random.Random(f"tk16:{seed}:spot")
    by = {}
    for r in rows:
        by.setdefault((r["family"], r["phi_free"]), []).append(r)
    chosen, quota = [], {}
    for fam, share in FAMILY_SHARES:
        quota[(f"tk16_transcript_{fam}", False)] = round(n * share * (1 - PHI_FREE_SHARE))
        quota[(f"tk16_transcript_{fam}", True)] = round(n * share * PHI_FREE_SHARE)
    for key, q in quota.items():
        pool = by.get(key, [])
        rng.shuffle(pool)
        # spread formats and cases first
        seen, picked = set(), []
        for r in pool:
            tag = (r["label_format"], r["case"])
            if tag not in seen and len(picked) < q:
                seen.add(tag)
                picked.append(r)
        for r in pool:
            if len(picked) >= q:
                break
            if r not in picked:
                picked.append(r)
        chosen += picked
    chosen = chosen[:n]
    lines = ["# tk16 spot-check sample", "",
             f"{len(chosen)} transcripts or excerpts, drawn by `author16_transcripts.py --spot-check` (seed {seed}), stratified by family, "
             "PHI-free versus PHI-bearing, label format and case. Gold spans are marked `[[LABEL|text]]`. Long transcripts show the opening and one "
             "later window; `[...]` marks omitted text. Everything here is synthetic.", ""]
    for i, r in enumerate(chosen, 1):
        body, trunc = excerpt(r, rng)
        lines += [f"## {i}. {r['id']}", "",
                  f"family {r['family'].replace('tk16_transcript_', '')}, layout {r['layout']}, case {r['case']}, label format {r['label_format']}, "
                  f"label change {r['label_change']}, {r['words']} words, {r['tokens']} tokens, {'no PHI beyond role labels' if r['phi_free'] else str(len(r['spans'])) + ' spans'}",
                  "", "```text", body, "```", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=str(OUT_JSONL))
    ap.add_argument("--spot-check", default=None, help="write the 100-transcript spot-check sample to this markdown path")
    ap.add_argument("--cap", type=int, default=None, help="rows per value; default floor(0.5 percent of rows)")
    args = ap.parse_args(argv)
    rows, stats = build(args.rows, args.seed, cap=args.cap, log=print)
    manifest = write_bank(rows, stats, Path(args.out), args.seed, args.rows)
    if args.spot_check:
        write_spot_check(rows, Path(args.spot_check), args.seed)
    print(json.dumps({k: manifest[k] for k in ("rows", "label_support", "families", "token_length", "value_repetition", "exclusion_filter", "rejected_draws", "redrawn_documents")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
