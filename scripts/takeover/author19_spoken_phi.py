#!/usr/bin/env python
"""Author the tk19 spoken-and-spelled PHI training bank (synthetic, deterministic).

Brief: docs/spoken-phi-bank-brief.md. Policy: docs/annotation-policy-v2.md,
decisions S1 to S7, with fixtures in annotation_policy_spoken.py.

Every person, number, address, facility and domain is invented. Phones use the
fictional 555-0100 to 555-0199 block, SSNs the never-issued 900 to 999 area, email
and web hosts the reserved example/test names, IPs the TEST-NET blocks.

Rows are assembled from marked strings (``[[LABEL:s|text]]`` for a spoken or
spelled rendering, ``[[LABEL:w|text]]`` for an orthographic one) so offsets are
computed, never typed. Each row is rejected and re-drawn when any 1-8 word
sequence of its text, or any span value, is on the blind-bank exclusion list
(only counts are reported), or when a span value would exceed the 0.5% per-value
row cap.

Usage:
  PYTHONPATH=src python scripts/takeover/author19_spoken_phi.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import re
import sys
from collections import Counter
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
    exclusion_hashes,
    excluded_sequences,
    excluded_value,
)
from clinical_deid.types import ENTITY_TYPES  # noqa: E402

SEED = 190006
N_ROWS = 1500
VALUE_CAP = 7  # 0.5% of 1,500 rows is 7.5
OUT = ROOT / "data" / "tk19_train_spoken_phi.jsonl"
SUITE = "tk19_spoken_phi"

FAMILY_ROWS = (
    ("verify", 750),
    ("name_id_email", 225),
    ("sched_pharm_ins", 225),
    ("clinical_speech", 150),
    ("stt_artefacts", 150),
)
NEGATIVE_ROWS = {"verify": 250, "name_id_email": 75, "sched_pharm_ins": 75, "clinical_speech": 50, "stt_artefacts": 50}
CASE_QUOTA = {"normal": 300, "caps": 225, "lower": 825}  # the 150 STT rows are all lowercase

# --------------------------------------------------------------------------- markup

MARK = re.compile(r"\[\[([A-Z]+):([sw])\|(.*?)\]\]", re.S)


def S(label: str, text: str) -> str:
    assert "|" not in text and "]" not in text and text == text.strip(), text
    return f"[[{label}:s|{text}]]"


def W(label: str, text: str) -> str:
    assert "|" not in text and "]" not in text and text == text.strip(), text
    return f"[[{label}:w|{text}]]"


def mark_fixture(text: str, spans: list[dict], form: str = "s") -> str:
    """Turn a plain text plus spans into the marked form (used by the tests)."""
    out, previous = [], 0
    for span in sorted(spans, key=lambda s: s["start"]):
        out.append(text[previous:span["start"]])
        out.append(f"[[{span['label']}:{form}|{text[span['start']:span['end']]}]]")
        previous = span["end"]
    out.append(text[previous:])
    return "".join(out)


class Builder:
    """Accumulates final text and spans; applies case and punctuation transforms per piece."""

    def __init__(self, case: str, strip: bool):
        self.case, self.strip = case, strip
        self.parts: list[str] = []
        self.pos = 0
        self.spans: list[dict] = []
        self.forms: list[str] = []

    def tx(self, s: str) -> str:
        if self.strip:
            s = re.sub(r"[,;?!…]", "", s)
            s = re.sub(r"\.+(?=\s|$)", "", s)
            s = re.sub(r" {2,}", " ", s)
        out = s.upper() if self.case == "caps" else s.lower() if self.case == "lower" else s
        assert len(out) == len(s), (s, out)
        return out

    def lit(self, s: str):
        self.parts.append(s)
        self.pos += len(s)

    def mark(self, marked: str):
        previous = 0
        pieces = []
        for m in MARK.finditer(marked):
            pieces.append(("lit", marked[previous:m.start()]))
            pieces.append(("span", m))
            previous = m.end()
        pieces.append(("lit", marked[previous:]))
        for i, (kind, item) in enumerate(pieces):
            if kind == "lit":
                text = self.tx(item)
                if i == 0:
                    text = text.lstrip(" ")
                if i == len(pieces) - 1:
                    text = text.rstrip(" ")
                self.lit(text)
            else:
                value = self.tx(item[3]).strip()
                self.spans.append({"start": self.pos, "end": self.pos + len(value), "label": item[1], "text": value})
                self.forms.append(item[2])
                self.lit(value)

    def result(self):
        return "".join(self.parts), self.spans, self.forms


def build_marked(marked: str, case: str = "normal", strip: bool = False):
    b = Builder(case, strip)
    b.mark(marked)
    return b.result()


# --------------------------------------------------------------------------- number words

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth",
       10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
       16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth", 20: "twentieth",
       21: "twenty-first", 22: "twenty-second", 23: "twenty-third", 24: "twenty-fourth", 25: "twenty-fifth",
       26: "twenty-sixth", 27: "twenty-seventh", 28: "twenty-eighth", 29: "twenty-ninth", 30: "thirtieth",
       31: "thirty-first"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def nw(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n >= 100:
        return nw3(n)
    return TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else "")


def nw3(n: int) -> str:
    if n < 100:
        return nw(n)
    h, r = divmod(n, 100)
    return ONES[h] + " hundred" + (" " + nw(r) if r else "")


def ord_suffix(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def year_words(rng: random.Random, y: int) -> str:
    """Four-digit year as it is said."""
    if 1900 <= y < 2000:
        r = y % 100
        return "nineteen " + ("hundred" if r == 0 else "oh " + ONES[r] if r < 10 else nw(r))
    if y == 2000:
        return "two thousand"
    if 2000 < y < 2010:
        return rng.choice(["two thousand " + ONES[y % 10], "twenty oh " + ONES[y % 10]])
    r = y % 100
    return rng.choice(["two thousand " + nw(r), "twenty " + nw(r)])


def year_short(y: int) -> str:
    r = y % 100
    return "oh " + ONES[r] if r < 10 else nw(r)


def wchoice(rng, pairs):
    total = sum(w for _, w in pairs)
    x = rng.random() * total
    for item, w in pairs:
        x -= w
        if x <= 0:
            return item
    return pairs[-1][0]


# --------------------------------------------------------------------------- invented pools

FIRST_NAMES = """Ottoline Perrin Isolde Caspian Thessaly Wystan Marisol Leander Philippa Anselm Calliope Barnaby Odette
Rosalind Evander Linnea Tobias Marguerite Desmond Cordelia Lysander Winifred Barnabas Peregrine Seraphina Ignatius
Henrietta Cormac Delphine Thaddeus Beatrix Ambrose Octavia Leopold Prudence Fitzgerald Rhiannon Matthias Tamsin
Bartholomew Esmeralda Cedric Ludovica Percival Genevieve Ulysses Hortense Magnus Clementine Lazarus Ophelia
Jasper Marianne Casimir Ginevra Alistair Theodora Emeric Verity Horatio Griselda Lucian Annelise Rufus Saskia
Quentin Philomena Gideon Aurelia Bram Constance Dorian Eulalia Florian Hedwig Ivo Jocasta Kasimir Lenora Merrick
Nerys Orson Petronella Rafferty Sabine Tiernan Ursula Valentin Wilhelmina Xavier Yolanda Zebedee Alaric Briony
Corwin Dagny Elric Fenella Gareth Hilda Ingrid Jory Kendrick Lavinia Mordecai Nadia Oswin Pandora Radcliffe
Selwyn Tabitha Udo Vesper Wendeline Yves Zora Abelard Bronwen Cyprian Dahlia Edmund Fiona Gawain Harriet
Ishmael Juniper Kenelm Lorcan Mirabel Nicanor Odile Piers Quillon Roswitha Stellan Teodora Urban Viola
Wolfram Ximena Yeva Zenobia Adelmar Brigid Caius Dessa Emrys Fabrizio Galen Honor Irving Jessamine Kieran
Leocadia Maddox Neville Orla Pascoe Reuben Sorcha Torquil Una Vance Wynn Yarrow Zachariah""".split()

SURNAMES_HAND = """Halvorne Tessimer Quillfeather Ostrander Pennywhistle Ashgrove Brindlecombe Calloway Dunmore Eskridge
Fairweather Garrity Hollowell Ivesmith Jessamy Kettleburn Lindqvist Mortlake Nightingale Oakhaven Pellman Rushworth
Stanhope Thistlewood Underhill Vandermeer Wetherby Yarnell Zeitlin Abernathy Blackwater Cresswell Delacroix
Everhart Fenwick Goldthwaite Harrowgate Inglewood Jorgensen Kingsbury Lockhart Merriweather Northcott Ormsby
Prendergast Quarles Ravenscroft Sutcliffe Tregarthen Uttley Vickery Whitlock Yeoman Zimmerli Armitage Bellweather
Cathcart Drummond Elkington Farnsworth Gladwin Hargreave Ilsley Jarrold Kilbride Lansdowne Moxley Netherby
Oglethorpe Pettigrew Radnor Sallow Tolliver Upshaw Verrall Wainwright Yorke Ziegenfuss Aldercott Brackenridge
Coldwell Dewhurst Entwistle Fothergill Greaves Hepworth Isherwood Jessop Knightley Lovelace Mowbray Needham
Ormerod Plowright Quincey Redfern Shuttleworth Thackeray Unwin Venables Winterbourne Yardley Zouch""".split()
SURNAME_STEMS = """Ash Bar Cal Dun Elm Fen Gar Hal Ire Jes Kel Lin Mor Nor Ost Pel Quin Rav Sel Tes Ulv Vor Wen Yar Zel Ber
Cor Dar Esk Fal Gren Hav Ket Lar Mal Nev Ors Pen Rud Sal Tor Ven Wil Brin Crow Dray Fair Gil Harr Kirk Lund Marl
Norr Pask Rand Skel Tarn Wick Brom Cald Dorn Gall Holm Kern Lorr Mern Pell Rosk Starr Thorn Vell""".split()
SURNAME_ENDS = ["ton", "by", "wick", "worth", "dale", "mere", "ford", "stead", "ley", "hurst", "ham", "well", "croft",
                "wyn", "ard", "ell", "man", "son", "burn", "field"]

STREET_STEMS = """Harrowby Teasdale Quenby Ostler Marrick Sallowfield Peverell Linnet Crowhurst Birchmoor Fenwater Garrow
Holloway Alderney Brindle Cressida Dovecote Elderflower Fallowmere Gorse Heronsgate Ironbridge Juniper Kestrel
Larkfield Mallow Nettlebed Orchard Pimpernel Quarry Rushmead Sandpiper Thornfield Umber Valewood Willowmere
Yarrow Ashby Bracken Coppice Dunlin Eastcote Foxglove Greystone Hazelwood Kingfisher Lavender Meadowsweet
Nightjar Oakridge Plover Redwing Sorrel Tansy Wagtail Bluebell Cranmere Dapple Eglantine Fernbank""".split()
STREET_TYPES = ["Lane", "Road", "Street", "Avenue", "Court", "Drive", "Way", "Terrace", "Place", "Circle", "Crossing",
                "Row", "Trail"]
LOCALITY_STEMS = """Tarn Hollin Brack Corr Dunmoor Elder Fenn Garth Hesper Ing Jessup Kest Lorn Marl Nether Oak Pell Quarry
Rook Sedge Thorn Umber Vale Wick Yew Aldren Bryn Calder Dray Emsley Farrow Gorm Haver Isley Jarl Kirby Lyle
Mossop Norbury Orrin Pent Rimmer Selby Tolland Upton Varley Wendel""".split()
LOCALITY_ENDS = ["wick", "ford", "field", "haven", "gap", "mere", "brook", "hollow", "ridge", "port", "bury", "dale",
                 "ton", "stead"]
STATES = [("Alabama", "AL"), ("Arizona", "AZ"), ("Arkansas", "AR"), ("Colorado", "CO"), ("Connecticut", "CT"),
          ("Delaware", "DE"), ("Georgia", "GA"), ("Idaho", "ID"), ("Illinois", "IL"), ("Indiana", "IN"),
          ("Iowa", "IA"), ("Kansas", "KS"), ("Kentucky", "KY"), ("Maine", "ME"), ("Maryland", "MD"),
          ("Michigan", "MI"), ("Minnesota", "MN"), ("Missouri", "MO"), ("Montana", "MT"), ("Nebraska", "NE"),
          ("Nevada", "NV"), ("New Mexico", "NM"), ("Ohio", "OH"), ("Oklahoma", "OK"), ("Oregon", "OR"),
          ("Tennessee", "TN"), ("Utah", "UT"), ("Vermont", "VT"), ("Virginia", "VA"), ("Washington", "WA"),
          ("Wisconsin", "WI"), ("Wyoming", "WY")]
FAC_STEMS = """Thornmere Quillmoor Ashdown Brackley Corriden Dunhollow Eastvane Fennimore Gallowmere Hartsmoor
Ironvale Jessamine Kestrelmoor Larchmont Marlowe Nethervane Oakleigh Pennywell Ravenhollow Sedgemoor Tolliver
Upperwick Valemont Westerly Yarrowby Aldermoor Birchwell Cinderford Draymont Elmstead Fallowfield Gorsemoor
Heathermere Ivybridge Juniperhill Kingsmoor Lindenfold Mossgrove Northwell Orchardvale Pinehollow Rookwood""".split()
CLINIC_TYPES = ["Family Medicine", "Medical Center", "Clinic", "Health Center", "Internal Medicine", "Medical Group",
                "Hospital", "Urgent Care", "Pediatrics", "Cardiology"]
PHARMACY_TYPES = ["Pharmacy", "Drug", "Apothecary", "Drugs", "Compounding Pharmacy"]
ASIN = {"a": "apple", "b": "boy", "c": "charlie", "d": "david", "e": "edward", "f": "frank", "g": "george",
        "h": "henry", "i": "ida", "j": "john", "k": "king", "l": "lincoln", "m": "mary", "n": "nancy", "o": "ocean",
        "p": "peter", "q": "queen", "r": "robert", "s": "sam", "t": "tom", "u": "union", "v": "victor",
        "w": "william", "x": "xray", "y": "yellow", "z": "zebra"}
NATO = {"a": "alpha", "b": "bravo", "c": "charlie", "d": "delta", "e": "echo", "f": "foxtrot", "g": "golf",
        "h": "hotel", "i": "india", "j": "juliet", "k": "kilo", "l": "lima", "m": "mike", "n": "november",
        "o": "oscar", "p": "papa", "q": "quebec", "r": "romeo", "s": "sierra", "t": "tango", "u": "uniform",
        "v": "victor", "w": "whiskey", "x": "x-ray", "y": "yankee", "z": "zulu"}
MISHEAR = {"b": "p", "p": "b", "d": "t", "t": "d", "m": "n", "n": "m", "f": "s", "s": "f", "v": "b", "c": "k",
           "k": "c", "g": "j", "j": "g", "e": "a"}
DOMAINS = [("example", "com"), ("example", "org"), ("example", "net"), ("mail", "example"), ("inbox", "test"),
           ("clinic", "test"), ("portal", "test"), ("webmail", "example")]
DRUGS = ["metoprolol", "lisinopril", "amlodipine", "atorvastatin", "levothyroxine", "gabapentin", "omeprazole",
         "furosemide", "warfarin", "sertraline", "clopidogrel", "tamsulosin", "carvedilol", "losartan", "apixaban",
         "pantoprazole", "metformin", "hydrochlorothiazide", "prednisone", "montelukast", "escitalopram",
         "rosuvastatin", "spironolactone", "diltiazem", "glipizide", "allopurinol"]
CONDITIONS = ["hypertension", "osteoporosis", "neuropathy", "arrhythmia", "pneumonia", "diverticulitis",
              "hypothyroidism", "angina", "bronchitis", "gastroparesis", "claudication", "vertigo"]
PROCEDURES = ["gallbladder surgery", "hip replacement", "cataract surgery", "knee surgery", "bypass surgery",
              "a hysterectomy", "my appendix out", "the pacemaker", "cardiac stent", "a colonoscopy", "shoulder repair",
              "a mastectomy", "the hernia repair"]


# --------------------------------------------------------------------------- people and values


def make_surname(rng: random.Random) -> str:
    x = rng.random()
    if x < 0.40:
        return rng.choice(SURNAMES_HAND)
    if x < 0.82:
        return rng.choice(SURNAME_STEMS) + rng.choice(SURNAME_ENDS)
    if x < 0.92:
        return rng.choice(SURNAMES_HAND) + "-" + rng.choice(SURNAMES_HAND)
    if x < 0.97:
        return rng.choice(["O'", "D'"]) + rng.choice(SURNAMES_HAND)
    return rng.choice(["Mc", "Mac"]) + rng.choice(SURNAME_STEMS) + rng.choice(SURNAME_ENDS)


def partition_ok(first: str, last: str) -> bool:
    return identity_split(canonical_identity_group(f"{first} {last}")) == "train"


def draw_name(rng: random.Random) -> tuple[str, str]:
    for _ in range(200):
        first, last = rng.choice(FIRST_NAMES), make_surname(rng)
        if partition_ok(first, last):
            return first, last
    raise RuntimeError("no train-partition name drawn")


def draw_date(rng: random.Random, lo: int, hi: int) -> dt.date:
    start, end = dt.date(lo, 1, 1), dt.date(hi, 12, 28)
    return start + dt.timedelta(days=rng.randrange((end - start).days))


class Person:
    def __init__(self, rng: random.Random):
        self.first, self.last = draw_name(rng)
        self.full = f"{self.first} {self.last}"
        self.dob = draw_date(rng, 1924, 2009)
        self.area = f"{rng.choice('23456789')}{rng.choice('012345678')}{rng.randrange(10)}"
        while self.area[1:] == "11" or self.area == "555":
            self.area = f"{rng.choice('23456789')}{rng.choice('012345678')}{rng.randrange(10)}"
        self.line = f"01{rng.randrange(100):02d}"
        self.num = rng.choice([rng.randrange(1, 100), rng.randrange(100, 1000), rng.randrange(100, 10000)])
        self.street = f"{rng.choice(STREET_STEMS)} {rng.choice(STREET_TYPES)}"
        unit_kind = rng.choice(["apt", "unit", "suite", "number", None, None])
        self.unit = None if unit_kind is None else (unit_kind, rng.randrange(1, 40), rng.choice("abcd"))
        self.city = rng.choice(LOCALITY_STEMS) + rng.choice(LOCALITY_ENDS)
        self.state = rng.choice(STATES)
        self.zip = f"{rng.randrange(10000, 99999)}"
        self.zip4 = f"{rng.randrange(10000):04d}"


def person_iterable(ctx) -> list[Person]:
    return [ctx.patient, ctx.agent]


class Ctx:
    """Row-level random state, flags and the value renderers."""

    def __init__(self, rng: random.Random, family: str, neg: bool, stt: bool, case: str):
        self.rng, self.family, self.neg, self.stt, self.case = rng, family, neg, stt, case
        self.phen: list[str] = []
        self.patient = Person(rng)
        self.agent = Person(rng)
        while self.agent.last == self.patient.last:
            self.agent = Person(rng)
        self.agent_word = rng.choice(["Agent", "Representative", "Scheduler", "Nurse", "Receptionist", "Intake",
                                      "Coordinator", "Specialist"])
        self.pt_word = rng.choice(["Patient", "Patient", "Caller", "Member"])
        self.fac = f"{rng.choice(FAC_STEMS)} {rng.choice(CLINIC_TYPES)}"
        self.pharm = f"{rng.choice(FAC_STEMS)} {rng.choice(PHARMACY_TYPES)}"
        self.mentions: set[str] = set()  # persons whose names appear in the row

    # -- small choices
    def zero(self) -> str:
        return self.rng.choice(["oh", "oh", "zero"])

    def sep(self) -> str:
        r = self.rng
        if self.stt:
            return r.choice([" ", " ", " ", " uh ", " um "])
        return wchoice(r, [(", ", 6), (" ", 2), (", uh, ", 1), (", um, ", 1)])

    def mix(self) -> float:
        return 0.4 if self.stt and self.rng.random() < 0.6 else 0.0

    # -- digits
    def digits(self, ds: str, pair: float = 0.0, dbl: float = 0.0, zero: str | None = None) -> str:
        r = self.rng
        zero = zero or self.zero()
        mix = self.mix()
        out, i = [], 0

        def word(d):
            return zero if d == "0" else ONES[int(d)]

        while i < len(ds):
            d = ds[i]
            if dbl and i + 1 < len(ds) and ds[i + 1] == d and r.random() < dbl:
                if i + 2 < len(ds) and ds[i + 2] == d and r.random() < 0.5:
                    out.append("triple " + word(d))
                    i += 3
                else:
                    out.append("double " + word(d))
                    i += 2
                continue
            if pair and i + 1 < len(ds) and r.random() < pair:
                n = int(ds[i:i + 2])
                out.append(f"{zero} {word(ds[i + 1])}" if d == "0" else nw(n))
                i += 2
                continue
            out.append(d if mix and r.random() < mix else word(d))
            i += 1
        return " ".join(out)

    def groups(self, parts: list[str]) -> str:
        return self.sep().join(parts) if self.rng.random() < 0.8 else " ".join(parts)

    # -- phone
    def phone_parts(self, p: Person):
        r = self.rng
        zero = self.zero()
        pair = r.choice([0.0, 0.0, 0.0, 0.6])
        if r.random() < 0.2:
            area = self.digits(p.area[0], zero=zero) + " " + self.digits(p.area[1:], pair=1.0, zero=zero)
        else:
            area = self.digits(p.area, zero=zero)
        exch = wchoice(r, [("five five five", 6), ("triple five", 2), ("five fifty-five", 1)])
        if self.stt and r.random() < 0.3:
            exch = "5 5 5"
        line = self.digits(p.line, pair=pair, dbl=0.25, zero=zero)
        return area, exch, line

    def phone_spoken(self, p: Person, area: bool = True, ext: int | None = None, plus1: bool = False) -> tuple[str, str]:
        r = self.rng
        if self.stt and r.random() < 0.2:  # run-together numerals
            digits = (p.area if area else "") + "555" + p.line
            if ext is not None:
                digits += f" ext {ext}"
            return digits, "w"
        a, e, ln = self.phone_parts(p)
        parts = ([a] if area else []) + [e, ln]
        txt = self.groups(parts)
        if plus1 and area:
            txt = "plus one " + txt
        if ext is not None:
            txt += r.choice([", extension ", ", ext ", " extension ", ", extension "]) + self.digits(str(ext), pair=0.3)
        return txt, "s"

    def phone_written(self, p: Person, ext: int | None = None) -> str:
        r = self.rng
        fmt = r.choice(["({a}) 555-{l}", "{a}-555-{l}", "{a}.555.{l}", "+1 {a} 555 {l}", "+1 ({a}) 555-{l}",
                        "{a} 555 {l}", "{a}-555-{l}"])
        txt = fmt.format(a=p.area, l=p.line)
        if ext is not None:
            txt += r.choice([f" ext. {ext}", f" x{ext}", f", extension {ext}", f" EXT {ext}"])
        return txt

    def phone(self, p: Person, spoken: bool | None = None, area: bool = True, ext: int | None = None) -> str:
        r = self.rng
        if spoken is None:
            spoken = r.random() < 0.75
        if not spoken:
            return W("PHONE", self.phone_written(p, ext))
        txt, form = self.phone_spoken(p, area, ext, plus1=r.random() < 0.06)
        return (S if form == "s" else W)("PHONE", txt)

    # -- spelling
    def spell(self, word: str) -> str:
        r = self.rng
        toks = []
        for ch in word.lower():
            toks.append({"-": "dash", "'": "apostrophe", " ": "space"}.get(ch, ch))
        if self.stt and r.random() < 0.4:
            cand = [i for i, t in enumerate(toks) if t in MISHEAR]
            if cand:
                i = r.choice(cand)
                toks[i] = MISHEAR[toks[i]]
        letters = [t.upper() if len(t) == 1 else t for t in toks]
        style = wchoice(r, [("plain", 5), ("dash", 2), ("comma", 2), ("nato", 2), ("asin", 2), ("chunk", 1)])
        if style == "dash" and len(letters) == len(word):
            return "-".join(letters)
        if style == "comma":
            return ", ".join(letters)
        if style == "nato":
            return " ".join(NATO[t.lower()].capitalize() if len(t) == 1 else t for t in letters)
        if style == "asin":
            join = ", " if not self.stt else " "
            return join.join(f"{t} as in {ASIN[t.lower()].capitalize()}" if len(t) == 1 else t for t in letters)
        if style == "chunk" and len(letters) > 5:
            k = r.randrange(2, 4)
            chunks = [" ".join(letters[i:i + k]) for i in range(0, len(letters), k)]
            return ", ".join(chunks)
        return " ".join(letters)

    def name_spoken(self, p: Person, which: str) -> str:
        return {"first": p.first, "last": p.last, "full": p.full}[which]

    # -- dates
    def year4(self, y: int) -> str:
        return year_words(self.rng, y)

    def date(self, d: dt.date, kind: str | None = None, year: bool = True, weekday: bool = False,
             allow_written: bool = True) -> str:
        r = self.rng
        m, day, y = d.month, d.day, d.year
        wd = WEEKDAYS[d.weekday()]
        mon = MONTHS[m - 1]
        zero = self.zero()
        yy = year_short(y)
        if kind is None:
            if year:
                pool = [("oh_mdy", 18), ("digits_mdy", 8), ("month_ord_year", 18), ("the_of_year", 12),
                        ("month_card_year", 6), ("digit_by_digit", 6), ("mixed", 5)]
                if self.stt:
                    pool += [("mixed", 10), ("digit_by_digit", 6)]
                if allow_written:
                    pool += [("slash4", 5), ("slash2", 3), ("long", 6), ("iso", 2), ("dash", 2), ("ord_written", 3),
                             ("dmy", 2)]
            else:
                pool = [("month_ord", 12), ("the_of_month", 10), ("month_card", 4), ("digits_md", 5), ("oh_md", 5),
                        ("the_ord", 5)]
                if allow_written:
                    pool += [("slash_md", 3), ("long_md", 3)]
            kind = wchoice(r, pool)
        form = "s"
        if kind == "oh_mdy":
            mo = f"{zero} {ONES[m]}" if m < 10 else nw(m)
            da = f"{zero} {ONES[day]}" if day < 10 else nw(day)
            yr = yy if r.random() < 0.7 else year_words(r, y)
            txt = self.groups([mo, da, yr])
        elif kind == "digits_mdy":
            txt = self.groups([nw(m), nw(day), yy if r.random() < 0.7 else year_words(r, y)])
        elif kind == "month_ord_year":
            txt = f"{mon} {ORD[day]}, {year_words(r, y)}"
        elif kind == "the_of_year":
            txt = f"the {ORD[day]} of {mon}, {year_words(r, y)}"
        elif kind == "month_card_year":
            txt = f"{mon} {nw(day)}, {yy}"
        elif kind == "digit_by_digit":
            txt = self.digits(f"{m:02d}{day:02d}{y % 100:02d}", zero=zero)
        elif kind == "mixed":
            txt = self.groups([str(m), str(day), year_words(r, y)]) if r.random() < 0.5 else \
                f"{mon} {day}, {year_words(r, y)}"
        elif kind in ("slash4", "slash2", "long", "iso", "dash", "ord_written", "dmy"):
            form = "w"
            txt = {"slash4": f"{m:02d}/{day:02d}/{y}", "slash2": f"{m}/{day}/{y % 100:02d}",
                   "long": f"{mon} {day}, {y}", "iso": f"{y}-{m:02d}-{day:02d}", "dash": f"{m}-{day}-{y}",
                   "ord_written": f"{mon} {ord_suffix(day)}, {y}", "dmy": f"{day} {mon} {y}"}[kind]
        elif kind == "month_ord":
            txt = f"{mon} {ORD[day]}"
        elif kind == "the_of_month":
            txt = f"the {ORD[day]} of {mon}"
        elif kind == "month_card":
            txt = f"{mon} {nw(day)}"
        elif kind == "digits_md":
            txt = self.groups([nw(m), nw(day)])
        elif kind == "oh_md":
            txt = self.groups([f"{zero} {ONES[m]}" if m < 10 else nw(m), f"{zero} {ONES[day]}" if day < 10 else nw(day)])
        elif kind == "the_ord":
            txt = f"the {ORD[day]}"
        elif kind == "slash_md":
            form, txt = "w", f"{m}/{day}"
        elif kind == "long_md":
            form, txt = "w", f"{mon} {day}"
        else:
            raise ValueError(kind)
        if weekday:
            txt = f"{wd} {txt}" if self.rng.random() < 0.6 else f"{wd}, {txt}"
        return (S if form == "s" else W)("DATE", txt)

    # -- addresses
    def house_number(self, n: int) -> str:
        r = self.rng
        s = str(n)
        if n < 100:
            return nw(n) if r.random() < 0.7 else self.digits(s)
        style = wchoice(r, [("digits", 3), ("pairs", 5), ("hundreds", 2)])
        if style == "digits":
            return self.digits(s)
        if style == "hundreds" and n < 1000 and n % 100 != 0:
            return nw3(n)
        head, tail = divmod(n, 100)
        t = f"{self.zero()} {ONES[tail]}" if tail < 10 else nw(tail)
        if tail == 0:
            t = "hundred" if n < 1000 else "hundred"
        return f"{nw(head)} {t}"

    def unit_spoken(self, p: Person) -> str:
        kind, n, letter = p.unit
        word = {"apt": self.rng.choice(["apartment", "apt"]), "unit": "unit", "suite": "suite", "number": "number"}[kind]
        style = self.rng.randrange(3)
        if style == 0:
            return f"{word} {nw(n)} {letter}"
        if style == 1:
            return f"{word} {nw(n)}"
        return f"{word} {self.digits(str(n * 10 + 1), pair=0.3)}"

    def unit_written(self, p: Person) -> str:
        kind, n, letter = p.unit
        return {"apt": f"Apt {n}{letter.upper()}", "unit": f"Unit {n}", "suite": f"Suite {n * 10}",
                "number": f"#{n}"}[kind]

    def street_spoken(self, p: Person, with_number: bool = True) -> str:
        return (self.house_number(p.num) + " " if with_number else "") + p.street

    def zip_spoken(self, p: Person, plus4: bool = False) -> str:
        txt = self.digits(p.zip, pair=0.0)
        if self.rng.random() < 0.15:
            txt = self.digits(p.zip[:2]) + self.sep() + self.digits(p.zip[2:])
        if plus4:
            txt += self.sep() + self.digits(p.zip4)
        return txt

    def zip_marked(self, p: Person, plus4: bool = False) -> str:
        if self.rng.random() < 0.22:
            return W("LOCATION", p.zip + (f"-{p.zip4}" if plus4 else ""))
        return S("LOCATION", self.zip_spoken(p, plus4))

    def city_state_spoken(self, p: Person, abbr: bool = False) -> str:
        if abbr or self.rng.random() < 0.2:
            return f"{p.city}, {p.state[1]}"
        return f"{p.city}, {p.state[0]}"

    def address_full(self, p: Person, written: bool | None = None, with_zip: bool = True, with_city: bool = True) -> str:
        r = self.rng
        if written is None:
            written = r.random() < 0.3
        if written:
            parts = [f"{p.num} {p.street}"]
            if p.unit:
                parts.append(self.unit_written(p))
            tail = ""
            if with_city:
                parts.append(p.city)
                tail = f"{p.state[1]} {p.zip}" if with_zip else p.state[1]
                parts.append(tail)
            return W("ADDRESS", ", ".join(parts))
        parts = [self.street_spoken(p)]
        if p.unit:
            parts.append(self.unit_spoken(p))
        if with_city:
            parts.append(p.city)
            parts.append(p.state[0])
            if with_zip:
                parts.append(self.zip_spoken(p))
        return S("ADDRESS", ", ".join(parts) if r.random() < 0.8 else " ".join(parts))


# --------------------------------------------------------------------------- identifiers, email, web, age


def rdigits(rng: random.Random, n: int, lead_nonzero: bool = False) -> str:
    s = "".join(rng.choice("0123456789") for _ in range(n))
    if lead_nonzero and s[0] == "0":
        s = rng.choice("123456789") + s[1:]
    return s


def rletters(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(n))


def ident_say(ctx: Ctx, letters: str, digits: str, group: int = 3) -> str:
    """Spoken alphanumeric identifier: spelled letters, then grouped digits."""
    r = ctx.rng
    parts = []
    if letters:
        mode = wchoice(r, [("spaced", 6), ("nato", 2), ("asin", 1)])
        if mode == "nato":
            parts.append(" ".join(NATO[c.lower()].capitalize() for c in letters))
        elif mode == "asin":
            parts.append(", ".join(f"{c} as in {ASIN[c.lower()].capitalize()}" for c in letters))
        else:
            ls = list(letters)
            if ctx.stt and r.random() < 0.35:
                i = r.randrange(len(ls))
                ls[i] = MISHEAR.get(ls[i].lower(), ls[i].lower()).upper()
            parts.append(" ".join(ls))
    zero = ctx.zero()
    k = r.choice([3, 3, 4, 2]) if group else len(digits)
    for i in range(0, len(digits), k):
        parts.append(ctx.digits(digits[i:i + k], zero=zero, pair=r.choice([0, 0, 0.3])))
    return ctx.groups(parts)


def ident_written(letters: str, digits: str, dash: bool) -> str:
    return f"{letters}{'-' if dash and letters else ''}{digits}"


def ident(ctx: Ctx, label: str, letters: str, digits: str, spoken: bool | None = None, dash: bool | None = None) -> str:
    r = ctx.rng
    if spoken is None:
        spoken = r.random() < 0.72
    if dash is None:
        dash = r.random() < 0.6
    if spoken:
        return S(label, ident_say(ctx, letters, digits))
    return W(label, ident_written(letters, digits, dash))


def ssn_raw(rng: random.Random) -> tuple[str, str, str]:
    return str(rng.randrange(900, 1000)), f"{rng.randrange(1, 100):02d}", f"{rng.randrange(1, 10000):04d}"


def ssn_marked(ctx: Ctx, raw, spoken: bool | None = None) -> str:
    r = ctx.rng
    if spoken is None:
        spoken = r.random() < 0.75
    a, g, s = raw
    if not spoken:
        return W("SSN", r.choice([f"{a}-{g}-{s}", f"{a} {g} {s}", f"{a}{g}{s}"]))
    zero = ctx.zero()
    return S("SSN", ctx.groups([ctx.digits(a, zero=zero), ctx.digits(g, zero=zero), ctx.digits(s, zero=zero)]))


def email_local(ctx: Ctx, p: Person):
    r = ctx.rng
    f, ln = p.first.lower(), re.sub(r"[^a-z]", "", p.last.lower())
    pat = r.choice(["f.l", "fl", "f_l", "f-l", "i.l", "l", "il", "f.l.d", "l.d"])
    d = rdigits(r, r.choice([1, 2, 2, 3]))
    return {"f.l": [(f, "w"), ".", (ln, "w")], "fl": [(f + ln, "w")], "f_l": [(f, "w"), "_", (ln, "w")],
            "f-l": [(f, "w"), "-", (ln, "w")], "i.l": [(f[0], "w"), ".", (ln, "w")], "l": [(ln, "w"), (d, "d")],
            "il": [(f[0] + ln, "w")], "f.l.d": [(f, "w"), ".", (ln, "w"), (d, "d")],
            "l.d": [(ln, "w"), ".", (d, "d")]}[pat]


def email_raw(ctx: Ctx, p: Person):
    return email_local(ctx, p), ctx.rng.choice(DOMAINS)


def email_written(raw) -> str:
    local, host = raw
    return "".join(x if isinstance(x, str) else x[0] for x in local) + "@" + ".".join(host)


def email_spoken(ctx: Ctx, raw, spell: bool | None = None) -> str:
    r = ctx.rng
    local, host = raw
    if spell is None:
        spell = r.random() < 0.3
    out = []
    for item in local:
        if item == ".":
            out.append("dot")
        elif item == "_":
            out.append("underscore")
        elif item == "-":
            out.append("dash")
        else:
            text, kind = item
            if kind == "d":
                out.append(ctx.digits(text, zero="zero" if r.random() < 0.3 else "oh"))
            elif spell and len(text) > 1:
                out.append(" ".join(text))
            else:
                out.append(text)
    out.append("at")
    out.append(" dot ".join(host))
    return " ".join(out)


def email_marked(ctx: Ctx, raw, spoken: bool | None = None) -> str:
    if spoken is None:
        spoken = ctx.rng.random() < 0.75
    return S("EMAIL", email_spoken(ctx, raw)) if spoken else W("EMAIL", email_written(raw))


URL_HOSTS = ["visit.example.test", "portal.example.org", "meet.clinic.test", "connect.example.net",
             "telehealth.example.com", "my.portal.test", "join.example.net"]


def url_marked(ctx: Ctx, spoken: bool | None = None, scheme: bool | None = None) -> str:
    r = ctx.rng
    host = r.choice(URL_HOSTS)
    path = r.choice(["r", "visit", "join", "s", "room"])
    code = rdigits(r, r.choice([3, 4, 4, 5]), True)
    if spoken is None:
        spoken = r.random() < 0.6
    if scheme is None:
        scheme = r.random() < 0.4
    if not spoken:
        return W("URL", f"{'https://' if scheme else ''}{host}/{path}/{code}")
    txt = " dot ".join(host.split(".")) + f" slash {path} slash " + ctx.digits(code)
    if scheme:
        txt = "h t t p s colon slash slash " + txt
    return S("URL", txt)


def ip_marked(ctx: Ctx, spoken: bool | None = None) -> str:
    r = ctx.rng
    net = r.choice([(192, 0, 2), (198, 51, 100), (203, 0, 113)])
    quad = (*net, r.randrange(1, 254))
    if spoken is None:
        spoken = r.random() < 0.7
    if not spoken:
        return W("IP", ".".join(map(str, quad)))
    return S("IP", " dot ".join(r.choice([nw3(q) if q > 99 else nw(q), ctx.digits(str(q))]) for q in quad))


def age_marked(ctx: Ctx, age: int, spoken: bool | None = None) -> str:
    r = ctx.rng
    if spoken is None:
        spoken = r.random() < 0.8
    if not spoken:
        return W("AGE", str(age))
    txt = nw(age)
    if ctx.stt and r.random() < 0.5:
        txt = txt.replace("-", " ")
    return S("AGE", txt)


def confirm(r) -> str:
    return r.choice(["Yes.", "Yes, that's right.", "Correct.", "That's right.", "Yep.", "Mm-hm, yes.", "Yeah.",
                     "Yes, that's correct.", "Right.", "Yes, exactly.", "That is correct."])


def ack(r) -> str:
    return r.choice(["Okay.", "Thank you.", "Got it.", "Perfect.", "Alright.", "Great, thanks.", "Okay, thank you.",
                     "Thanks.", "Okay, got that.", "Alright, thank you."])


def decoy(ctx: Ctx) -> str:
    """A spoken number that is not PHI, said by the agent."""
    r = ctx.rng
    return r.choice([
        f"Take {nw(r.randrange(1, 4))} tablets in the morning and {nw(r.randrange(1, 3))} at night.",
        f"Your blood pressure was {nw3(r.choice([110, 118, 124, 132, 138, 142, 150]))} over {nw(r.choice([68, 72, 76, 80, 84, 88, 90]))}.",
        f"It's {nw(r.choice([5, 10, 15, 20, 25, 40, 50, 75, 100]))} milligrams once a day.",
        f"We open at {r.choice(['eight', 'eight thirty', 'nine', 'seven thirty'])} and close at {r.choice(['four', 'five', 'five thirty', 'six'])}.",
        f"The wait is about {nw(r.randrange(3, 25))} minutes.",
        f"Your last A1c was {nw(r.randrange(5, 9))} point {nw(r.randrange(0, 10))}.",
        f"Walk for {nw(r.choice([10, 15, 20, 30]))} minutes, {nw(r.randrange(3, 6))} times a week.",
        f"Press {nw(r.randrange(1, 9))} for billing or {nw(r.randrange(1, 9))} for the pharmacy.",
        f"The study from twenty twenty-{nw(r.randrange(1, 5))} changed the guidance.",
        f"Your copay is {nw(r.choice([10, 20, 25, 30, 40, 45]))} dollars.",
        f"Eat {nw(r.randrange(3, 6))} small meals a day and drink {nw(r.randrange(6, 10))} glasses of water.",
    ])


def decoy_p(ctx: Ctx) -> str:
    """A spoken number that is not PHI, said by the patient."""
    r = ctx.rng
    return r.choice([
        f"On a scale of one to ten, I'd say the pain is a {nw(r.randrange(2, 9))}.",
        f"It started about {nw(r.randrange(2, 9))} days ago.",
        f"I've had this cough for {nw(r.randrange(2, 7))} weeks.",
        f"I weigh about {nw3(r.choice([128, 141, 156, 163, 175, 188, 204]))} pounds.",
        f"I'm eighty-{nw(r.randrange(1, 10))} so I get tired.",
        f"I take {nw(r.randrange(1, 4))} pills a day and sleep about {nw(r.randrange(4, 9))} hours.",
        f"My sugar this morning was {nw3(r.randrange(90, 190))}.",
    ])


# --------------------------------------------------------------------------- scenarios: positives
# Each returns a list of (role, marked_text); roles: A agent/clinician, P patient, S system.

DOB_Q = ["Can you confirm your date of birth?", "What's your date of birth?", "And your birthday, month day and year?",
         "Date of birth for verification, please.", "I need your date of birth, please.",
         "Could you give me your date of birth?", "For security, what is your date of birth?",
         "Can you tell me your birth date?", "And the date of birth on the account?"]
DOB_PRE = ["", "", "It's ", "My date of birth is ", "Born ", "Uh, it's ", "That would be ", "My birthday is ", "Sure, it's "]
READBACK = ["So that's {x}, correct?", "Let me read that back, {x}.", "{x}, is that right?", "Okay, {x}. Thank you.",
            "I have {x}. Is that right?", "Just to confirm, {x}?", "Alright, {x}.", "{x}, did I get that right?"]
ADDR_Q = ["What's your home address?", "Can you give me your street address?", "And your address, please?",
          "Which address are we mailing to?", "What address do we have you at?", "Please confirm your home address."]
PHONE_Q = ["What's the best callback number?", "What number can we reach you at?", "Can I get a phone number?",
           "And a good number to reach you?", "What's your phone number?", "Can you give me a callback number?",
           "Which number should we put on file?"]
PRE_PHONE = ["", "It's ", "My number is ", "Sure, it's ", "You can reach me at ", "Uh, it's ", "It's ", "Call me at "]
CORR_CUES = ["sorry", "no wait", "I mean", "no, sorry", "actually", "let me start over", "no", "oops"]
NAME_Q = ["Can you spell your last name for me?", "How do you spell your last name?", "Could you spell that, please?",
          "And the spelling of your last name?", "Can you spell that for me?", "What's the spelling?"]
ID_NAMES = {"MRN": ["medical record number", "record number", "M R N", "chart number"],
            "ID": ["member ID", "policy number", "subscriber ID", "ID number on the card", "member number"],
            "ACCOUNT": ["account number", "billing account number", "patient account number", "guarantor account number"]}


def other_date(rng, d: dt.date) -> dt.date:
    """A different date, for the first attempt of a correction."""
    if rng.random() < 0.5:
        day = ((d.day + rng.randrange(1, 7) - 1) % 28) + 1
        return d.replace(day=day) if day != d.day else d.replace(day=(d.day % 28) + 1)
    m = (d.month + rng.randrange(1, 11)) % 12 + 1
    return d.replace(month=m, day=min(d.day, 28))


def fmt(template: str, **kw) -> str:
    return template.format(**kw)


def sc_v_dob_basic(ctx: Ctx):
    r, d = ctx.rng, ctx.patient.dob
    ctx.phen += ["spoken_date", "dob_verification"]
    first = ctx.date(d)
    turns = [("A", r.choice(DOB_Q)), ("P", f"{r.choice(DOB_PRE)}{first}.")]
    if r.random() < 0.65:
        turns += [("A", fmt(r.choice(READBACK), x=ctx.date(d))), ("P", confirm(r))]
        ctx.phen.append("read_back")
    else:
        turns.append(("A", ack(r)))
    return turns


def sc_v_dob_split(ctx: Ctx):
    r, d = ctx.rng, ctx.patient.dob
    ctx.phen += ["split_turns", "spoken_date", "read_back"]
    mon, day = MONTHS[d.month - 1], d.day
    zero = ctx.zero()
    style = wchoice(r, [("a", 5), ("b", 2), ("c", 3), ("d", 3)])
    if style == "a":
        p1, p2 = S("DATE", f"{mon} {ORD[day]}"), S("DATE", year_words(r, d.year))
        turns = [("A", r.choice(DOB_Q)), ("P", f"{p1}."), ("A", "And the year?"), ("P", f"{p2}.")]
    elif style == "b":
        p1 = S("DATE", mon)
        p2 = S("DATE", f"{ORD[day]}, {year_short(d.year)}")
        turns = [("A", r.choice(DOB_Q)), ("P", f"{p1}..."), ("A", f"{S('DATE', mon)}, go ahead."), ("P", f"{p2}.")]
    elif style == "c":
        mo = f"{zero} {ONES[d.month]}" if d.month < 10 else nw(d.month)
        da = f"{zero} {ONES[day]}" if day < 10 else nw(day)
        p1 = S("DATE", ctx.groups([mo, da]))
        p2 = S("DATE", year_short(d.year) if r.random() < 0.6 else year_words(r, d.year))
        turns = [("A", r.choice(DOB_Q)), ("P", f"{p1}."), ("A", "And the year of birth?"), ("P", f"{p2}.")]
    else:
        p1 = S("DATE", f"the {ORD[day]} of {mon}")
        p2 = S("DATE", year_words(r, d.year))
        turns = [("A", r.choice(DOB_Q)), ("P", f"{p1}, uh,"), ("A", "Which year?"), ("P", f"{p2}.")]
    turns += [("A", fmt(r.choice(READBACK), x=ctx.date(d))), ("P", confirm(r))]
    return turns


def sc_v_dob_year(ctx: Ctx):
    r, d = ctx.rng, ctx.patient.dob
    ctx.phen += ["spoken_year", "dob_verification"]
    yr = S("DATE", year_words(r, d.year) if r.random() < 0.55 else year_short(d.year))
    return [("A", r.choice(["And what year were you born?", "Which year were you born?", "Birth year?"])),
            ("P", r.choice([f"I was born in {yr}.", f"Born in {yr}.", f"It's {yr}.", f"{sentence_case_marked(yr)}."])),
            ("A", ack(r))]


def sentence_case_marked(s: str) -> str:
    return s


def sc_v_dob_correction(ctx: Ctx):
    r, d = ctx.rng, ctx.patient.dob
    ctx.phen += ["correction", "spoken_date"]
    wrong = ctx.date(other_date(r, d), allow_written=False)
    right = ctx.date(d, allow_written=False)
    return [("A", r.choice(DOB_Q)),
            ("P", f"{wrong}, {r.choice(CORR_CUES)}, {right}."),
            ("A", fmt(r.choice(READBACK), x=ctx.date(d))), ("P", confirm(r))]


def sc_v_dob_written(ctx: Ctx):
    r, d = ctx.rng, ctx.patient.dob
    ctx.phen += ["written_readback", "date_boundaries"]
    sysd = W("DATE", r.choice([f"{d.month:02d}/{d.day:02d}/{d.year}", f"{MONTHS[d.month - 1]} {d.day}, {d.year}",
                               f"{d.year}-{d.month:02d}-{d.day:02d}"]))
    turns = [("A", r.choice([f"I have a date of birth of {sysd} on file.", f"The date of birth on the account is {sysd}.",
                             f"Our records show {sysd}."]) + " " + r.choice(["Is that right?", "Can you confirm?"]))]
    if r.random() < 0.5:
        turns.append(("P", confirm(r)))
    else:
        ctx.phen.append("correction")
        turns += [("P", f"No, that's wrong. It's {ctx.date(d, allow_written=False)}."), ("A", ack(r))]
    return turns


def sc_v_addr_full(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["spoken_address", "address_boundary"]
    full = ctx.address_full(p, written=False, with_zip=r.random() < 0.6, with_city=r.random() < 0.7)
    pre = r.choice(["", "I'm at ", "It's ", "I live at ", "Sure, it's "])
    turns = [("A", r.choice(ADDR_Q)), ("P", f"{pre}{full}.")]
    if r.random() < 0.55:
        street = S("ADDRESS", ctx.street_spoken(p))
        turns += [("A", r.choice([f"So that's {street}, correct?", f"{street}, did I get that right?"])), ("P", confirm(r))]
        ctx.phen.append("read_back")
    else:
        turns.append(("A", ack(r)))
    return turns


def sc_v_addr_partial(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["partial_address", "partial_confirmation"]
    street = W("ADDRESS", p.street)
    num = S("ADDRESS", ctx.house_number(p.num))
    turns = []
    style = r.randrange(4)
    if style == 0:
        turns = [("A", r.choice([f"Are you still on {street}?", f"We have you on {street}, is that still right?",
                                 f"Do you still live on {street}?"]))]
        if p.unit:
            turns.append(("P", f"Yes, {num}, {S('ADDRESS', ctx.unit_spoken(p))}."))
        else:
            turns.append(("P", f"Yes, {num}."))
    elif style == 1:
        turns = [("A", "What's the house number?"), ("P", f"{num}."), ("A", "And the street?"), ("P", f"{street}.")]
        if p.unit:
            turns += [("A", "Is there a unit or apartment?"), ("P", f"{S('ADDRESS', ctx.unit_spoken(p))}.")]
    elif style == 2:
        turns = [("A", f"Still at {W('ADDRESS', f'{p.num} {p.street}')}?"), ("P", "Yes.")]
        if p.unit:
            turns += [("A", "And the apartment number?"), ("P", f"{S('ADDRESS', ctx.unit_spoken(p))}.")]
    else:
        turns = [("A", "Can you confirm the house number for me?"), ("P", f"{num}, yes."), ("A", f"On {street}, right?"),
                 ("P", confirm(r))]
    if r.random() < 0.5:
        turns += [("A", r.choice(["And the ZIP code?", "What's the ZIP?", "Zip code?"])),
                  ("P", f"{ctx.zip_marked(p)}.")]
        ctx.phen.append("spoken_zip")
    return turns


def sc_v_addr_zip(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["spoken_zip", "read_back"]
    z = ctx.zip_marked(p, plus4=r.random() < 0.2)
    pre = r.choice(["", "It's ", "Zip is ", "Sure, "])
    return [("A", r.choice(["And your ZIP code?", "What's the ZIP code?", "Zip code on that address?"])),
            ("P", f"{pre}{z}."),
            ("A", fmt(r.choice(READBACK), x=ctx.zip_marked(p))), ("P", confirm(r))]


def sc_v_addr_city(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["spoken_locality", "spoken_zip"]
    city = S("LOCATION", ctx.city_state_spoken(p)) if r.random() < 0.6 else W("LOCATION", f"{p.city}, {p.state[1]}")
    return [("A", r.choice(["What city do you live in?", "Which town is that in?", "And the city?"])),
            ("P", f"{city}."), ("A", "And the ZIP?"), ("P", f"{ctx.zip_marked(p)}.")]


def sc_v_addr_written(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["written_readback", "address_boundary"]
    full = ctx.address_full(p, written=True, with_zip=True, with_city=True)
    turns = [("A", r.choice([f"I have {full} on file. Is that correct?", f"The address on file is {full}. Still current?"]))]
    if r.random() < 0.5:
        turns.append(("P", confirm(r)))
    else:
        ctx.phen.append("correction")
        q = Person(r)
        q.num = p.num + r.randrange(1, 9) if r.random() < 0.5 else p.num
        q.street = p.street
        turns += [("P", f"Almost. It's {S('ADDRESS', ctx.street_spoken(q))}."), ("A", ack(r))]
    return turns


def sc_v_phone_basic(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["spoken_phone", "read_back"]
    turns = [("A", r.choice(PHONE_Q)), ("P", f"{r.choice(PRE_PHONE)}{ctx.phone(p, spoken=True)}.")]
    if r.random() < 0.7:
        turns += [("A", fmt(r.choice(READBACK), x=ctx.phone(p, spoken=r.random() < 0.6))), ("P", confirm(r))]
    else:
        turns.append(("A", ack(r)))
    return turns


def sc_v_phone_area_apart(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["spoken_phone", "area_code_apart"]
    a, e, ln = ctx.phone_parts(p)
    rest = S("PHONE", ctx.groups([e, ln]))
    area = S("PHONE", a)
    s = r.randrange(4)
    if s == 0:
        body = f"My callback is {rest}, area code {area}."
    elif s == 1:
        body = f"It's {rest}. The area code is {area}."
    elif s == 2:
        body = f"Area code {area}, and then {rest}."
    else:
        body = f"{sentence_cap(rest)}, area code {area}."
    return [("A", r.choice(PHONE_Q)), ("P", body), ("A", r.choice(["Got it.", "Thank you.", ack(r)]))]


def sentence_cap(s: str) -> str:
    return s


def sc_v_phone_split(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["split_turns", "spoken_phone", "read_back"]
    a, e, ln = ctx.phone_parts(p)
    rest = S("PHONE", ctx.groups([e, ln]))
    area = S("PHONE", a)
    turns = [("A", "What's your area code?"), ("P", f"{area}."), ("A", f"{S('PHONE', a)}, and the rest of the number?"),
             ("P", f"{rest}.")]
    turns += [("A", fmt(r.choice(READBACK), x=S("PHONE", ctx.groups([a, e, ln])))), ("P", confirm(r))]
    return turns


def sc_v_phone_correction(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["correction", "spoken_phone"]
    a, e, ln = ctx.phone_parts(p)
    zero = ctx.zero()
    wrong_exch = ctx.digits(f"555{r.randrange(10)}"[:3] if r.random() < 0.5 else "556", zero=zero)
    if r.random() < 0.5:
        wrong = S("PHONE", ctx.groups([a, wrong_exch]))
        right = S("PHONE", ctx.groups([a, e, ln]))
    else:
        wrong_line = ctx.digits(f"01{r.randrange(100):02d}", zero=zero)
        wrong = S("PHONE", ctx.groups([e, wrong_line]))
        right = S("PHONE", ctx.groups([e, ln]))
    return [("A", r.choice(PHONE_Q)), ("P", f"{r.choice(PRE_PHONE)}{wrong}, {r.choice(CORR_CUES)}, {right}."),
            ("A", fmt(r.choice(READBACK), x=ctx.phone(p, spoken=True))), ("P", confirm(r))]


def sc_v_phone_written(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["written_readback", "phone_boundaries"]
    sysp = ctx.phone(p, spoken=False)
    turns = [("A", r.choice([f"I have {sysp} on file.", f"The number on file is {sysp}.", f"We have {sysp}."]) +
              " " + r.choice(["Is that still good?", "Is that right?", "Still correct?"]))]
    if r.random() < 0.5:
        turns.append(("P", confirm(r)))
    else:
        ctx.phen += ["correction", "spoken_phone"]
        turns += [("P", f"No, use {ctx.phone(p, spoken=True)}."), ("A", ack(r))]
    return turns


def sc_v_combo(ctx: Ctx):
    r = ctx.rng
    ctx.phen.append("combined_verification")
    turns = []
    parts = [r.choice([sc_v_dob_basic, sc_v_dob_split, sc_v_dob_year]),
             r.choice([sc_v_addr_full, sc_v_addr_partial, sc_v_addr_zip]),
             r.choice([sc_v_phone_basic, sc_v_phone_area_apart, sc_v_phone_split])]
    r.shuffle(parts)
    for fn in parts:
        turns += fn(ctx)
    return turns


def sc_v_name_check(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["spelled_name", "name_verification"]
    turns = [("A", r.choice(["Can you confirm your full name?", "And who am I speaking with?", "Please state your full name."])),
             ("P", f"{r.choice(['', 'It is ', 'This is '])}{W('NAME', p.full)}."),
             ("A", r.choice(NAME_Q)), ("P", f"{S('NAME', ctx.spell(p.last))}.")]
    if r.random() < 0.5:
        turns += [("A", f"{W('NAME', p.first)} {S('NAME', ctx.spell(p.last))}, is that right?"), ("P", confirm(r))]
        ctx.phen.append("read_back")
    return turns


# ---- name, ID and email


def sc_n_name_spell(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["spelled_name", "letter_by_letter"]
    which = r.choice(["last", "last", "both"])
    if which == "last":
        a = f"{W('NAME', p.last)}. {S('NAME', ctx.spell(p.last))}."
        pre = r.choice(["", "It's ", "That's ", "Sure. "])
        turns = [("A", r.choice(NAME_Q)), ("P", f"{pre}{a}")]
    else:
        a = (f"{W('NAME', p.full)}. First name, {S('NAME', ctx.spell(p.first))}. "
             f"Last name, {S('NAME', ctx.spell(p.last))}.")
        turns = [("A", "Can I get your full name and the spelling?"), ("P", a)]
    if r.random() < 0.5:
        turns += [("A", r.choice([f"{S('NAME', ctx.spell(p.last))}, is that right?", f"So {W('NAME', p.last)}, "
                                  f"{S('NAME', ctx.spell(p.last))}?"])), ("P", confirm(r))]
        ctx.phen.append("read_back")
    else:
        turns.append(("A", ack(r)))
    return turns


def sc_n_name_correction(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["correction", "spelled_name"]
    base = p.last
    cut = base[:-1] if len(base) > 4 else base + "e"
    turns = [("A", "And your last name?"),
             ("P", f"{W('NAME', cut)}, {r.choice(CORR_CUES)}, {W('NAME', base)}."),
             ("A", r.choice(NAME_Q)), ("P", f"{S('NAME', ctx.spell(base))}.")]
    return turns


def sc_n_name_first_last(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["written_name", "spelled_name"]
    turns = [("A", r.choice(["Who am I speaking with?", "May I have your name?", "And your name?"])),
             ("P", r.choice([f"This is {W('NAME', p.full)}.", f"{W('NAME', p.full)}.", f"It's {W('NAME', p.full)}."]))]
    turns += [("A", r.choice([f"Thank you, {W('NAME', p.first)}.", f"Thanks, {W('NAME', p.first)}. Could you spell your "
                                                                  f"first name?"])),
              ("P", f"{S('NAME', ctx.spell(p.first))}.")]
    return turns


def _id_scenario(ctx: Ctx, label: str, letters_len: int, digits_len: int, phen: str, caption_pool=None):
    r = ctx.rng
    ctx.phen += [phen, "spoken_identifier", "read_back"]
    letters = rletters(r, letters_len) if letters_len else ""
    digits = rdigits(r, digits_len, True)
    cap = r.choice(caption_pool or ID_NAMES[label])
    val = ident(ctx, label, letters, digits)
    pre = r.choice(["", "It's ", "Sure, it's ", "Let me see, it's ", "Okay, it's "])
    turns = [("A", r.choice([f"What's the {cap}?", f"Can you read me the {cap}?", f"And your {cap}?",
                             f"Do you have the {cap} handy?"])), ("P", f"{pre}{val}.")]
    if r.random() < 0.7:
        turns += [("A", fmt(r.choice(READBACK), x=ident(ctx, label, letters, digits, spoken=r.random() < 0.5))),
                  ("P", confirm(r))]
    else:
        turns.append(("A", ack(r)))
    return turns


def sc_n_mrn(ctx: Ctx):
    return _id_scenario(ctx, "MRN", ctx.rng.choice([0, 2, 2]), ctx.rng.choice([6, 7, 7, 8]), "mrn")


def sc_n_member(ctx: Ctx):
    return _id_scenario(ctx, "ID", ctx.rng.choice([2, 3, 3]), ctx.rng.choice([6, 8, 9]), "member_id")


def sc_n_account(ctx: Ctx):
    return _id_scenario(ctx, "ACCOUNT", ctx.rng.choice([0, 0, 2]), ctx.rng.choice([8, 9, 10]), "account")


def sc_n_ident_correction(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["correction", "spoken_identifier"]
    label = r.choice(["MRN", "ID", "ACCOUNT"])
    letters = rletters(r, r.choice([0, 2])) if label != "ACCOUNT" else ""
    digits = rdigits(r, 7, True)
    wrong = rdigits(r, 7, True)[:r.choice([3, 4, 5])]
    cap = r.choice(ID_NAMES[label])
    first = S(label, ident_say(ctx, letters, wrong))
    second = ident(ctx, label, letters, digits, spoken=True)
    return [("A", f"What's the {cap}?"), ("P", f"{first}, {r.choice(CORR_CUES)}, {second}."),
            ("A", fmt(r.choice(READBACK), x=ident(ctx, label, letters, digits, spoken=False))), ("P", confirm(r))]


def sc_n_ssn(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["spoken_identifier", "ssn"]
    raw = ssn_raw(r)
    cap = r.choice(["social security number", "social", "SSN", "social for the application"])
    pre = r.choice(["", "It's ", "Sure, it's ", "My social is "])
    turns = [("A", r.choice([f"Can I get the {cap}?", f"And the {cap}?", f"I'll need the full {cap}, please."])),
             ("P", f"{pre}{ssn_marked(ctx, raw, spoken=True)}.")]
    if r.random() < 0.5:
        turns += [("A", fmt(r.choice(READBACK), x=ssn_marked(ctx, raw, spoken=r.random() < 0.5))), ("P", confirm(r))]
        ctx.phen.append("read_back")
    else:
        turns.append(("A", ack(r)))
    return turns


def sc_n_email(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["spoken_email"]
    raw = email_raw(ctx, p)
    spell = r.random() < 0.35
    val = S("EMAIL", email_spoken(ctx, raw, spell=spell))
    pre = r.choice(["", "It's ", "Sure, it's ", "The email is ", "Let me see, it's "])
    turns = [("A", r.choice(["What's the best email address?", "Can I get an email for the portal?",
                             "And your email address?", "What email should we send the link to?"])),
             ("P", f"{pre}{val}.")]
    if r.random() < 0.65:
        turns += [("A", fmt(r.choice(READBACK), x=email_marked(ctx, raw, spoken=r.random() < 0.5))), ("P", confirm(r))]
        ctx.phen.append("read_back")
    else:
        turns.append(("A", ack(r)))
    return turns


def sc_n_email_written(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["written_readback", "email_boundaries"]
    raw = email_raw(ctx, p)
    turns = [("A", r.choice([f"I have {email_marked(ctx, raw, spoken=False)} for you. Is that right?",
                             f"Your email on file is {email_marked(ctx, raw, spoken=False)}, correct?"]))]
    if r.random() < 0.5:
        turns.append(("P", confirm(r)))
    else:
        raw2 = email_raw(ctx, p)
        ctx.phen += ["correction", "spoken_email"]
        turns += [("P", f"No, it's {S('EMAIL', email_spoken(ctx, raw2))}."), ("A", ack(r))]
    return turns


def sc_n_url(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["spoken_url"]
    turns = [("A", r.choice(["I'm sending you the visit link. Can you type it in?", "Go to this address to join the video visit.",
                             "The link for the telehealth room is below."])),
             ("A", f"{url_marked(ctx)}.")]
    turns.append(("P", r.choice(["Okay, give me a second.", "Got it, typing it in.", "Okay, I'm there."])))
    if r.random() < 0.5:
        turns += [("A", f"Your session code is {ident(ctx, 'ID', rletters(r, 2), rdigits(r, 4, True))}."), ("P", ack(r))]
        ctx.phen.append("session_id")
    return turns


def sc_n_ip(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["spoken_ip"]
    return [("A", r.choice(["Your device isn't connecting. What's the IP address in network settings?",
                            "Can you read me the IP address on the settings screen?"])),
            ("P", f"{r.choice(['It says ', 'It shows ', ''])}{ip_marked(ctx)}."),
            ("A", ack(r))]


def sc_n_portal(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["spoken_email", "access_code"]
    code = ident(ctx, "ID", rletters(r, 4), rdigits(r, 4, True), spoken=True)
    raw = email_raw(ctx, p)
    return [("A", "I'm sending a portal activation code to your email. What email should I use?"),
            ("P", f"{email_marked(ctx, raw, spoken=True)}."),
            ("A", f"Okay. The activation code is {code}. It expires in {nw(r.choice([10, 15, 30]))} minutes."),
            ("P", ack(r))]


# ---- scheduling, pharmacy and insurance

TIMES = ["ten thirty", "nine", "two fifteen", "eleven forty-five", "one thirty", "three", "eight fifteen", "four"]


def sc_s_callback(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["spoken_phone", "callback"]
    s = r.randrange(4)
    if s == 0:
        a, e, ln = ctx.phone_parts(p)
        body = (f"Please call me back at {S('PHONE', ctx.groups([e, ln]))}, area code {S('PHONE', a)}, "
                f"after {r.choice(TIMES)}.")
    elif s == 1:
        ext = r.randrange(100, 9999)
        body = f"You can reach me at {ctx.phone(p, spoken=True, ext=ext)}, any time after {r.choice(TIMES)}."
        ctx.phen.append("extension")
    elif s == 2:
        body = f"Call {ctx.phone(p, spoken=True)} please. If I don't pick up, leave a message."
    else:
        body = f"My cell is {ctx.phone(p, spoken=r.random() < 0.7)}. It's the best way to reach me."
    turns = [("A", r.choice(["What number should the nurse call back?", "Where can the doctor's office reach you?",
                             "What's a good callback number for the care team?"])), ("P", body)]
    if r.random() < 0.5:
        turns += [("A", fmt(r.choice(READBACK), x=ctx.phone(p, spoken=r.random() < 0.5))), ("P", confirm(r))]
        ctx.phen.append("read_back")
    return turns


def sc_s_appt(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["appointment_date", "spoken_date"]
    d = draw_date(r, 2025, 2027)
    when = ctx.date(d, kind=r.choice(["month_ord", "the_of_month", "month_card", "digits_md", "oh_md", "the_ord",
                                      "slash_md", "long_md"]), year=False, weekday=r.random() < 0.3)
    t = r.choice(TIMES)
    ctx.mentions.add("agent")
    doc = W("NAME", ctx.agent.last)
    fac = W("FACILITY", ctx.fac)
    turns = [("A", r.choice([f"We have an opening on {when} at {t} with Dr. {doc} at {fac}.",
                             f"I can book you with Dr. {doc} on {when}, at {t}.",
                             f"The next available is {when} at {t}. That's at {fac}."])),
             ("P", r.choice(["That works.", "Perfect, book it.", f"Can we do {ctx.date(draw_date(r, 2025, 2027), kind='the_ord', year=False)} instead?", "Okay, that's fine."]))]
    if r.random() < 0.5:
        turns += [("A", f"Okay, you're set for {ctx.date(d, kind=r.choice(['month_ord', 'the_of_month']), year=False)}. "
                        f"Please arrive {nw(r.choice([10, 15, 20]))} minutes early."), ("P", ack(r))]
        ctx.phen.append("read_back")
    return turns


def sc_s_reschedule(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["spoken_date", "two_dates"]
    d1, d2 = draw_date(r, 2025, 2027), draw_date(r, 2025, 2027)
    a = ctx.date(d1, kind=r.choice(["month_ord", "the_ord", "the_of_month"]), year=False)
    b = ctx.date(d2, kind=r.choice(["month_ord", "the_ord", "the_of_month"]), year=False)
    return [("P", r.choice([f"I need to move my appointment from {a}.", f"I'm calling about my visit on {a}."])),
            ("A", r.choice(["What day works better?", "When would you like to come in instead?"])),
            ("P", f"{r.choice(['How about ', 'Could we do ', 'Make it '])}{b}?"),
            ("A", f"Done. I've moved it to {ctx.date(d2, kind=r.choice(['month_ord', 'the_of_month']), year=False)}.")]


def sc_s_pharmacy(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["rx_number", "pharmacy"]
    rx = S("ID", ident_say(ctx, "", rdigits(r, r.choice([7, 7, 8]), True))) if r.random() < 0.7 else \
        W("ID", r.choice([f"RX-{rdigits(r, 5)}", rdigits(r, 7, True)]))
    drug = r.choice(DRUGS)
    turns = [("P", r.choice([f"I'm calling to refill {drug}.", f"I need a refill on my {drug}."])),
             ("A", "What's the prescription number? It starts with Rx."),
             ("P", f"{r.choice(['', 'Rx ', 'Rx number '])}{rx}.")]
    s = r.randrange(3)
    if s == 0:
        turns += [("A", "Which pharmacy should we send it to?"), ("P", f"{W('FACILITY', ctx.pharm)}.")]
        if r.random() < 0.6:
            turns += [("A", "And the ZIP for that store?"), ("P", f"{ctx.zip_marked(p)}.")]
    elif s == 1:
        turns += [("A", "And the pharmacy's phone number, in case we need it?"), ("P", f"{ctx.phone(p, spoken=True)}.")]
    else:
        d = draw_date(r, 2025, 2026)
        turns += [("A", "When was it last filled?"), ("P", f"{r.choice(['I think ', 'It was ', ''])}{ctx.date(d, year=False)}.")]
    return turns


def sc_s_insurance(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["member_id", "spoken_identifier"]
    mem = ident(ctx, "ID", rletters(r, 3), rdigits(r, 8, True))
    grp = ident(ctx, "ID", "", rdigits(r, 6, True), spoken=r.random() < 0.7)
    turns = [("A", "I'll need the member ID from the card."), ("P", f"{r.choice(['', 'It says ', 'It is '])}{mem}."),
             ("A", "And the group number?"), ("P", f"{grp}.")]
    if r.random() < 0.5:
        turns += [("A", "Is the subscriber the same as the patient?"),
                  ("P", f"Yes, it's me. My date of birth is {ctx.date(ctx.patient.dob)}.")]
        ctx.phen.append("spoken_date")
    return turns


def sc_s_billing(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["account_number", "spoken_identifier"]
    acct = ident(ctx, "ACCOUNT", "", rdigits(r, r.choice([8, 9, 10]), True))
    turns = [("P", "I'm calling about a bill I got in the mail."), ("A", "What's the account number at the top?"),
             ("P", f"{acct}."), ("A", f"Your balance is {nw(r.choice([40, 85, 120, 215]))} dollars. Would you like to pay now?")]
    if r.random() < 0.5:
        turns += [("P", "Not today."), ("A", f"We'll send a reminder on {ctx.date(draw_date(r, 2025, 2026), kind='month_ord', year=False)}.")]
        ctx.phen.append("spoken_date")
    return turns


def sc_s_claim(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["claim_number", "encounter_number"]
    claim = ident(ctx, "ID", r.choice(["CL", "HX", "RK"]), rdigits(r, 8, True))
    enc = ident(ctx, "ID", "", rdigits(r, 9, True), spoken=r.random() < 0.7)
    return [("P", "I have a question about a denied claim."), ("A", "What's the claim number on the letter?"),
            ("P", f"{claim}."), ("A", "And the encounter or visit number, if it's listed?"), ("P", f"{enc}.")]


# ---- clinical history in speech


def sc_c_history(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["spoken_year", "history_year"]
    n = r.randrange(2, 4)
    turns = [("A", r.choice(["Tell me about any past surgeries or hospital stays.", "What's your surgical history?",
                             "Any major procedures or illnesses in the past?"]))]
    items = r.sample(PROCEDURES, n)
    for it in items:
        y = r.randrange(1985, 2024)
        yr = S("DATE", year_short(y) if r.random() < 0.5 else year_words(r, y))
        turns.append(("P", r.choice([f"I had {it} back in {yr}.", f"{sentence_cap(it).capitalize()} was in {yr}.",
                                     f"There was {it}, that was {yr}.", f"{it.capitalize()}, back in {yr}."])))
        if r.random() < 0.4:
            turns.append(("A", r.choice(["Okay.", "Any complications?", "And how did that go?"])))
    return turns


def sc_c_born(ctx: Ctx):
    r, p = ctx.rng, ctx.patient
    ctx.phen += ["spoken_year", "born_in"]
    y = p.dob.year
    yr = S("DATE", year_short(y) if r.random() < 0.6 else year_words(r, y))
    return [("A", r.choice(["How long have you lived in the area?", "Tell me a little about yourself.", "Where did you grow up?"])),
            ("P", r.choice([f"Well, I was born in {yr}, so I've seen a lot.", f"I was born in {yr} and grew up on a farm.",
                            f"Born in {yr}. I've been here most of my life."])),
            ("A", "Thank you for sharing that.")]


def sc_c_ordinal_day(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["spoken_date", "ordinal_day"]
    d = draw_date(r, 2024, 2026)
    ev = r.choice(["got the flu shot", "had the blood work done", "started the new pills", "went to urgent care",
                   "had the fall", "saw the eye doctor", "had the x-ray"])
    when = ctx.date(d, kind=r.choice(["the_ord", "the_ord", "month_ord", "the_of_month"]), year=False)
    return [("A", r.choice(["When did that happen?", "Remind me when that was.", "What day was that?"])),
            ("P", r.choice([f"I {ev} on {when}.", f"It was on {when}, I {ev}.", f"That was {when}. I {ev}."])),
            ("A", r.choice(["And you've felt worse since then?", "Okay, and how are you feeling now?"])),
            ("P", r.choice(["A bit worse since last week.", "About the same as three days ago.", "Better since yesterday."]))]


def sc_c_age(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["spoken_age", "age_over_89"]
    age = r.randrange(90, 104)
    s = r.randrange(4)
    a = age_marked(ctx, age)
    if s == 0:
        turns = [("A", "How old are you, if you don't mind my asking?"),
                 ("P", r.choice(["I am ", "I'm "]) + f"{a} years old.")]
    elif s == 1:
        turns = [("A", "And how old are you now?"), ("P", f"{a}.")]
    elif s == 2:
        dec = S("AGE", "nineties")
        turns = [("A", "Can you tell me roughly how old you are?"), ("P", f"I'm in my {dec}.")]
    else:
        turns = [("A", "I see you're up to date on screening. How old are you?"),
                 ("P", f"{r.choice(['I turned ', 'I am '])}{a} {r.choice(['last month', 'this year', 'now'])}.")]
    turns.append(("A", r.choice(["Thank you.", "Okay, that helps.", "You're doing great."])))
    return turns


def sc_c_mixed_history(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["spoken_year", "spoken_date", "spoken_age", "relative_date"]
    y = r.randrange(1990, 2022)
    yr = S("DATE", year_short(y))
    d = draw_date(r, 2024, 2026)
    turns = [("A", "Let's go over the timeline."),
             ("P", f"I was diagnosed back in {yr}."), ("A", "And the most recent flare?"),
             ("P", f"It was on {ctx.date(d, kind='the_ord', year=False)}, and it's been worse since last week."),
             ("A", "How old are you now?"),
             ("P", f"{age_marked(ctx, r.randrange(90, 100))}.")]
    return turns


def sc_c_diag_year(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["spoken_year", "history_year", "future_year"]
    y = r.randrange(1990, 2022)
    cond = r.choice(CONDITIONS)
    yr = S("DATE", year_short(y) if r.random() < 0.5 else year_words(r, y))
    nxt = S("DATE", year_words(r, r.randrange(2026, 2031)))
    return [("A", f"How long have you had {cond}?"), ("P", f"Since {yr}. That's when they told me."),
            ("A", f"Your next screening is due in {nxt}.")]


# ---- STT artefact helpers: noise turns inserted between content turns

NOISE = ["[inaudible]", "[crosstalk]", "um", "uh", "hold on", "sorry the line is bad", "can you repeat that", "hello",
         "[noise]", "you're breaking up"]


# --------------------------------------------------------------------------- scenarios: matched negatives (no PHI)


def ng_ivr(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["ivr_menu_numbers"]
    a, b, c = r.sample(range(1, 9), 3)
    return [("S", f"For scheduling, press {nw(a)}. For billing, press {nw(b)}. For the pharmacy, press {nw(c)}. "
                  f"To repeat this menu, press nine."),
            ("P", f"{nw(r.choice([a, b, c])).capitalize()}."),
            ("S", f"Your estimated wait time is {nw(r.randrange(2, 20))} minutes. You are number {nw(r.randrange(1, 9))} in line.")]


def ng_meds(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["dose_numbers", "vitals_numbers"]
    drug = r.choice(DRUGS)
    return [("A", f"Your blood pressure was {nw3(r.choice([112, 128, 134, 142, 150]))} over {nw(r.choice([70, 78, 82, 88, 92]))} today."),
            ("P", "Is that high?"),
            ("A", f"A little. Keep taking the {drug}, {nw(r.choice([5, 10, 20, 25, 40, 50]))} milligrams, "
                  f"{r.choice(['once', 'twice'])} a day."),
            ("P", f"And the pain? It's about a {nw(r.randrange(3, 8))} out of ten."),
            ("A", decoy(ctx))]


def ng_spell_drug(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["spelled_non_phi", "drug_spelling"]
    drug = r.choice(DRUGS)
    style = r.randrange(3)
    letters = " ".join(drug.upper()) if style < 2 else ", ".join(drug.upper())
    return [("A", r.choice(["The pharmacist asked how that medication is spelled.", "Can you spell the medication for me?",
                            "How do you spell the name of that drug?"])),
            ("P", f"{r.choice(['Sure, ', '', 'It is '])}{letters}."),
            ("A", f"Okay, {drug}. {r.choice(['Thank you.', 'Got it.', 'Perfect.'])}")]


def ng_spell_word(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["spelled_non_phi"]
    word = r.choice(CONDITIONS)
    return [("A", f"I'm putting in a note about {word}. How is that spelled?"),
            ("P", f"{' '.join(word.upper())}."), ("A", ack(r)),
            ("A", decoy(ctx))]


def ng_relative(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["relative_dates"]
    return [("A", r.choice(["When did the symptoms start?", "How long has this been going on?", "And when did you last eat?"])),
            ("P", r.choice([f"About {nw(r.randrange(2, 9))} days ago.", "Last week, I think.", "Since yesterday.",
                            f"Maybe {nw(r.randrange(2, 5))} weeks back.", "This morning.", "A couple of months ago."])),
            ("A", "Has it gotten better or worse since then?"),
            ("P", r.choice(["A little worse since last week.", "Better than three days ago.", "About the same as yesterday."]))]


def ng_public(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["public_years", "ordinary_age"]
    y = r.randrange(2016, 2025)
    age = r.randrange(55, 89)
    return [("A", f"The {y} guidelines changed the recommended screening interval."),
            ("P", f"I'm {nw(age)}, does that apply to me?"), ("A", f"Yes. Anyone over {nw(r.choice([50, 65, 75]))} should be screened every "
                                                                f"{nw(r.randrange(1, 6))} years."),
            ("P", ack(r))]


def ng_times(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["clock_times", "durations"]
    return [("A", f"The clinic is open from {r.choice(['eight', 'nine', 'seven thirty'])} to {r.choice(['four', 'five', 'six'])}."),
            ("P", f"Can I come at {r.choice(TIMES)}?"),
            ("A", f"That slot is {r.choice(['open', 'taken'])}. Your visit will take about {nw(r.choice([20, 30, 45]))} minutes."),
            ("P", ack(r))]


def ng_declined(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["declined_verification"]
    return [("A", r.choice(DOB_Q + ADDR_Q + PHONE_Q)),
            ("P", r.choice(["I'd rather not give that over the phone.", "I don't have that in front of me.",
                            "Can you call me back, I'm driving.", "I'm not comfortable saying that out loud."])),
            ("A", r.choice(["No problem, I can transfer you to someone who can verify you another way.",
                            "That's fine, we can use the portal instead.", "Understood, we'll mail you a form."])),
            ("P", ack(r))]


def ng_onfile(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["on_file_confirmation"]
    what = r.choice(["address", "phone number", "date of birth", "insurance information", "email"])
    return [("A", f"I have your {what} on file. Is everything still the same?"),
            ("P", r.choice(["Yes, nothing has changed.", "Yes, same as before.", "It's all the same."])),
            ("A", r.choice(["Great, that's all I needed.", "Perfect, thank you."])),
            ("P", ack(r))]


def ng_billing(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["money_numbers"]
    return [("P", "How much do I owe?"),
            ("A", f"Your balance is {nw(r.choice([35, 60, 90, 145, 250]))} dollars and {nw(r.choice([0, 15, 25, 50, 75])).replace('zero', 'no')} cents."),
            ("P", "Can I pay half now?"), ("A", f"Yes, that would be {nw(r.choice([20, 30, 45, 70, 125]))} dollars today.")]


def ng_instructions(ctx: Ctx):
    r = ctx.rng
    ctx.phen += ["instruction_numbers"]
    return [("A", decoy(ctx)), ("P", r.choice(["Okay, I'll write that down.", "Can you repeat that?", "Got it."])),
            ("A", decoy(ctx)), ("P", ack(r))]


NEGATIVES = {
    "verify": [ng_declined, ng_onfile, ng_ivr, ng_relative, ng_times, ng_billing],
    "name_id_email": [ng_spell_drug, ng_spell_word, ng_declined, ng_ivr, ng_onfile],
    "sched_pharm_ins": [ng_ivr, ng_times, ng_billing, ng_spell_drug, ng_relative, ng_instructions],
    "clinical_speech": [ng_meds, ng_relative, ng_public, ng_instructions, ng_times],
    "stt_artefacts": [ng_ivr, ng_meds, ng_spell_drug, ng_relative, ng_public, ng_times, ng_declined],
}

POSITIVES = {
    "verify": [(sc_v_dob_basic, 14), (sc_v_dob_split, 12), (sc_v_dob_year, 3), (sc_v_dob_correction, 7),
               (sc_v_dob_written, 4), (sc_v_addr_full, 10), (sc_v_addr_partial, 12), (sc_v_addr_zip, 5),
               (sc_v_addr_city, 3), (sc_v_addr_written, 5), (sc_v_phone_basic, 12), (sc_v_phone_area_apart, 8),
               (sc_v_phone_split, 7), (sc_v_phone_correction, 6), (sc_v_phone_written, 4), (sc_v_combo, 6)],
    "name_id_email": [(sc_n_name_spell, 20), (sc_n_name_correction, 7), (sc_n_name_first_last, 6), (sc_v_name_check, 9), (sc_n_mrn, 17),
                      (sc_n_member, 10), (sc_n_account, 9), (sc_n_ident_correction, 5), (sc_n_ssn, 12),
                      (sc_n_email, 14), (sc_n_email_written, 5), (sc_n_url, 10), (sc_n_ip, 4), (sc_n_portal, 5)],
    "sched_pharm_ins": [(sc_s_callback, 14), (sc_s_appt, 14), (sc_s_reschedule, 8), (sc_s_pharmacy, 14),
                        (sc_s_insurance, 10), (sc_s_billing, 6), (sc_s_claim, 5)],
    "clinical_speech": [(sc_c_history, 22), (sc_c_born, 10), (sc_c_ordinal_day, 14), (sc_c_age, 26),
                        (sc_c_mixed_history, 10), (sc_c_diag_year, 8)],
    "stt_artefacts": [(sc_v_dob_basic, 8), (sc_v_dob_split, 5), (sc_v_phone_basic, 8), (sc_v_phone_split, 5),
                      (sc_v_phone_area_apart, 6), (sc_v_addr_full, 6), (sc_v_addr_partial, 5), (sc_n_name_spell, 12), (sc_v_name_check, 4),
                      (sc_n_mrn, 5), (sc_n_email, 6), (sc_n_member, 4), (sc_n_ssn, 4), (sc_c_history, 4),
                      (sc_c_ordinal_day, 3), (sc_s_callback, 4), (sc_n_ident_correction, 3)],
}

GREET_PT = {
    "verify": ["Hi, I'm calling to check on my account.", "Hello, I need to update my information.",
               "Hi, I got a message to call back.", "Hi, I'm calling about my appointment."],
    "name_id_email": ["Hi, I'm trying to set up the patient portal.", "Hello, I need help with my insurance card.",
                      "Hi, I'm calling to register for a visit.", "Hello, I need to get my records sent."],
    "sched_pharm_ins": ["Hi, I need to schedule a visit.", "Hello, I'm calling about a prescription.",
                        "Hi, I have a billing question.", "Hello, I need to change an appointment."],
    "clinical_speech": ["Hi doctor, thanks for seeing me.", "Hello, I'm here for my follow-up.",
                        "Hi, I'm doing okay, a little tired."],
    "stt_artefacts": ["hi i'm calling about my appointment", "hello can you hear me", "hi i need some help"],
}
CLOSINGS = ["Is there anything else I can help you with?", "Thank you for calling, have a good day.",
            "We'll send a confirmation shortly.", "Take care, and call us if anything changes.",
            "Thanks for your patience today."]
PT_CLOSE = ["No, that's everything.", "That's all, thank you.", "No, thanks. Bye.", "Nope, that's it."]


def defill(text: str) -> str:
    """Agents read values back cleanly: no spoken fillers inside their turns."""
    text = re.sub(r",? (?:uh|um),", ",", text)
    return re.sub(r" (?:uh|um) ", " ", text)


def frame(ctx: Ctx, turns: list, positive: bool) -> list:
    r = ctx.rng
    out = [(role, defill(text)) if role == "A" and len(text) > 12 else (role, text) for role, text in turns]
    pre = []
    if r.random() < 0.35 and ctx.family == "clinical_speech":
        pre = [("A", r.choice(["Good morning, how have you been feeling?", "Come on in, what brings you in today?",
                               "Thanks for joining the visit. How are you doing?"])),
               ("P", r.choice(GREET_PT["clinical_speech"]))]
    elif r.random() < 0.35:
        if positive and r.random() < 0.6:
            greet = r.choice([f"Thank you for calling {W('FACILITY', ctx.fac)}, this is {W('NAME', ctx.agent.first)}, how can I help?",
                              f"{W('FACILITY', ctx.fac)}, {W('NAME', ctx.agent.first)} speaking. How can I help you today?"])
        else:
            greet = r.choice(["Thank you for calling, how can I help you?", "Good morning, how can I help?",
                              "Patient services, how may I help you?", "Thanks for holding. How can I help?"])
        pre = [("A", greet), ("P", r.choice(GREET_PT[ctx.family]))]
    post = []
    if r.random() < 0.3:
        post = [("A", r.choice(CLOSINGS)), ("P", r.choice(PT_CLOSE))]
    if r.random() < 0.22 and len(out) > 2:
        out.insert(r.choice([0, len(out)]) if pre else len(out), r.choice([("A", decoy(ctx)), ("P", decoy_p(ctx))]))
    if ctx.stt:
        for _ in range(r.randrange(1, 4)):
            i = r.randrange(0, len(out) + 1)
            who = r.choice(["A", "P", "P"])
            out.insert(i, (who, r.choice(NOISE)))
    return pre + out + post


# --------------------------------------------------------------------------- layouts

LAYOUTS = {
    "role_nl": dict(label="role", sep="\n"),
    "role_blank": dict(label="role", sep="\n\n"),
    "short_nl": dict(label="short", sep="\n"),
    "num_nl": dict(label="num", sep="\n"),
    "rep_caller": dict(label="rep", sep="\n"),
    "named_role": dict(label="namedrole", sep="\n", named=True),
    "dr_label": dict(label="dr", sep="\n", named=True),
    "chat_name": dict(label="chatname", sep="\n", named=True, ts="clock_br"),
    "bracket_tag": dict(label="brackets", sep="\n"),
    "qa": dict(label="qa", sep="\n"),
    "arrow": dict(label="arrow", sep="\n"),
    "dash": dict(label="dash", sep="\n"),
    "pipe_table": dict(label="pipe", sep="\n", ts="mmss_plain"),
    "ts_mmss": dict(label="role", sep="\n", ts="mmss"),
    "ts_hms": dict(label="role", sep="\n", ts="hms"),
    "ts_ownline": dict(label="role", sep="\n", ts="ownline"),
    "ts_clock": dict(label="short", sep="\n", ts="clock"),
    "inline_runon": dict(label="role", sep=" "),
    "inline_ts": dict(label="short", sep=" ", ts="mmss"),
    "header_call": dict(label="role", sep="\n", header="call", headered=True),
    "header_visit": dict(label="rep", sep="\n", header="visit", headered=True),
    "footer_end": dict(label="role", sep="\n", footer="end"),
    "footer_dur": dict(label="short", sep="\n", footer="dur"),
    "excerpt_ellipsis": dict(label="role", sep="\n", ellipsis=True),
    "numbered": dict(label="role", sep="\n", numbered=True),
    "speaker_id": dict(label="spk", sep="\n"),
    "said": dict(label="said", sep="\n"),
    "excerpt_blank_ts": dict(label="rep", sep="\n\n", ts="mmss", ellipsis=True),
}
LAYOUT_IDS = list(LAYOUTS)
NEG_LAYOUT_IDS = [k for k, v in LAYOUTS.items() if not v.get("named") and not v.get("headered")]


def label_marked(ctx: Ctx, style: str, role: str) -> str:
    aw, pw = ctx.agent_word, ctx.pt_word
    if role == "S":
        return {"short": "S", "num": "Speaker 3", "rep": "IVR", "qa": "S", "brackets": "[SYSTEM]", "spk": "Speaker C (System)",
                "said": "System said"}.get(style, "System")
    if style == "role":
        return aw if role == "A" else pw
    if style == "short":
        return "A" if role == "A" else "P"
    if style == "num":
        return "Speaker 1" if role == "A" else "Speaker 2"
    if style == "rep":
        return "Rep" if role == "A" else "Caller"
    if style == "namedrole":
        return f"{aw} ({W('NAME', ctx.agent.last)})" if role == "A" else f"{pw} ({W('NAME', ctx.patient.last)})"
    if style == "dr":
        return f"Dr {W('NAME', ctx.agent.last)}" if role == "A" else pw
    if style == "chatname":
        return f"{W('NAME', ctx.agent.first)} ({aw})" if role == "A" else f"{W('NAME', ctx.patient.first)} ({pw})"
    if style == "brackets":
        return f"[{aw.upper()}]" if role == "A" else f"[{pw.upper()}]"
    if style == "qa":
        return "Q" if role == "A" else "A"
    if style == "arrow":
        return aw if role == "A" else pw
    if style == "dash":
        return aw if role == "A" else pw
    if style == "pipe":
        return aw if role == "A" else pw
    if style == "spk":
        return f"Speaker A ({aw})" if role == "A" else f"Speaker B ({pw})"
    if style == "said":
        return f"{aw} said" if role == "A" else f"{pw} said"
    raise ValueError(style)


def header_marked(ctx: Ctx, kind: str) -> str:
    r = ctx.rng
    d = draw_date(r, 2025, 2026)
    dm = W("DATE", r.choice([f"{d.year}-{d.month:02d}-{d.day:02d}", f"{d.month:02d}/{d.day:02d}/{d.year}",
                             f"{MONTHS[d.month - 1]} {d.day}, {d.year}"]))
    clock = f"{r.randrange(7, 19):02d}:{r.randrange(60):02d}"
    if kind == "call":
        cid = W("ID", f"{r.choice(['CX', 'CL', 'TC'])}-{rdigits(r, 5, True)}")
        return r.choice([f"Call transcript\nCall started {dm} {clock} | Call ID {cid}\n",
                         f"Phone recording transcript\nDate {dm}\nCall ID {cid}\n"])
    cid = W("ID", f"{r.choice(['VX', 'TV'])}-{rdigits(r, 4, True)}")
    return r.choice([f"Telehealth visit transcript\nVisit session {cid} on {dm}\n",
                     f"Video visit transcript, session {cid}\nStarted {dm} at {clock}\n"])


def render_layout(ctx: Ctx, layout_id: str, turns: list, case: str, strip: bool):
    spec = LAYOUTS[layout_id]
    r = ctx.rng
    b = Builder(case, strip)
    sep = spec["sep"]
    if spec.get("ellipsis"):
        b.lit(r.choice(["[...] ", "... ", "[earlier part of call omitted]\n"]))
    if spec.get("header"):
        b.mark(header_marked(ctx, spec["header"]))
        b.lit("\n")
    secs = r.randrange(5, 600)
    base = r.randrange(7, 18) * 60 + r.randrange(60)
    for i, (role, text) in enumerate(turns):
        secs += r.randrange(4, 30)
        ts = spec.get("ts")
        lab = label_marked(ctx, spec["label"], role)
        mm, ss = divmod(secs, 60)
        hh, mm = divmod(mm, 60)
        stamp = {"mmss": f"[{mm:02d}:{ss:02d}] ", "hms": f"{hh:02d}:{mm:02d}:{ss:02d} ", "ownline": f"[{hh:02d}:{mm:02d}:{ss:02d}]\n",
                 "clock": f"{(base + secs // 60) // 60 % 24:02d}:{(base + secs // 60) % 60:02d} ", "mmss_plain": f"{mm:02d}:{ss:02d}",
                 "clock_br": f"[{(base + secs // 60) // 60 % 24:02d}:{(base + secs // 60) % 60:02d}] ", None: ""}[ts]
        style = spec["label"]
        own = ts == "ownline"
        inline = "" if own else stamp
        if style == "pipe":
            line = f"{stamp} | {lab} | {text}"
        elif style == "arrow":
            line = f"{inline}{lab} > {text}"
        elif style == "dash":
            line = f"{inline}{lab} - {text}"
        elif style == "brackets":
            line = f"{inline}{lab} {text}"
        elif spec.get("numbered"):
            line = f"{i + 1}. {lab}: {text}"
        else:
            line = f"{inline}{lab}: {text}"
        if own:
            b.lit(stamp)
        b.mark(line)
        if i < len(turns) - 1:
            b.lit(sep)
    if spec.get("footer") == "end":
        b.lit("\n")
        b.mark(r.choice(["[end of call]", "(call ended)", "--- end of recording ---"]))
    elif spec.get("footer") == "dur":
        b.lit("\n")
        b.mark(f"(call ended, duration {r.randrange(2, 9):02d}:{r.randrange(60):02d})")
    if spec.get("ellipsis"):
        b.lit(r.choice([" [...]", " ...", ""]))
    return b.result()


# --------------------------------------------------------------------------- fixture rebuilding


def spell_plain(word: str) -> str:
    """Letter-by-letter spelling, one letter at a time (the plain style of Ctx.spell)."""
    return " ".join(word.lower())


def fixture_markup() -> dict[str, str]:
    """Rebuild every SPOKEN_FIXTURES example from the generator's own primitives (marked form)."""
    c = Ctx(random.Random(0), "verify", False, False, "normal")

    def say(ds: str, zero: str = "oh") -> str:
        return c.digits(ds, zero=zero)

    def dob_oh() -> str:
        return ", ".join([f"oh {ONES[3]}", f"oh {ONES[3]}", year_short(1941)])

    email = ([("dally", "w"), ".", ("brannock", "w")], ("example", "com"))
    return {
        "spoken_phone_and_area_code": (
            f"patient: my callback is {S('PHONE', ', '.join([say('555'), 'uh', say('0142')]))}, "
            f"area code {S('PHONE', say('612'))}."),
        "spoken_ssn_mrn_zip": (
            f"caller: my social is {S('SSN', ', '.join([say('912', 'zero'), say('44', 'zero'), say('8810', 'zero')]))}. "
            f"record number {S('MRN', say('7720931', 'zero'))}. zip is {S('LOCATION', say('53728', 'zero'))}."),
        "spelled_name": (
            f"agent: last name? caller: {W('NAME', 'Brannock')}, {S('NAME', spell_plain('Brannock'))}."),
        "spoken_email": f"patient: the email is {S('EMAIL', email_spoken(c, email, spell=False))}.",
        "spoken_dates": (
            f"patient: date of birth {S('DATE', dob_oh())}. birthday's {S('DATE', f'{MONTHS[2].lower()} {ORD[3]}')}. "
            f"surgery was back in {S('DATE', year_short(2009))}. born in {S('DATE', year_short(1941))}. "
            f"got the shot on {S('DATE', f'the {ORD[12]}')}. felt worse last week."),
        "partial_address_verification": (
            f"agent: still on {W('ADDRESS', 'wenlow lane')}? patient: yes, {S('ADDRESS', f'{nw(4)} {nw(12)}')}, "
            f"{S('ADDRESS', f'apartment {nw(2)} b')}. agent: zip? patient: {S('LOCATION', say('53728', 'zero'))}."),
        "split_across_turns": (
            f"agent: date of birth {S('DATE', MONTHS[2].lower())}... patient: "
            f"{S('DATE', f'{ORD[3]}, {year_short(1941)}')}. agent: so that's "
            f"{S('PHONE', ', '.join([say('555'), say('0142')]))}? patient: correct."),
        "spoken_correction": (
            f"caller: it's {S('PHONE', say('556'))}, sorry, {S('PHONE', ', '.join([say('557'), say('0142')]))}."),
    }


# --------------------------------------------------------------------------- planning and row assembly


def build_plan() -> list[dict]:
    rng = random.Random(SEED)
    plan = []
    for fam, n in FAMILY_ROWS:
        negs = NEGATIVE_ROWS[fam]
        plan += [{"family": fam, "neg": i < negs} for i in range(n)]
    rng.shuffle(plan)
    cases = ["normal"] * CASE_QUOTA["normal"] + ["caps"] * CASE_QUOTA["caps"] + ["lower"] * CASE_QUOTA["lower"]
    rng.shuffle(cases)
    it = iter(cases)
    for spec in plan:
        spec["case"] = "lower" if spec["family"] == "stt_artefacts" else next(it)
        spec["strip"] = spec["family"] == "stt_artefacts" or (spec["case"] == "lower" and rng.random() < 0.45)

    def cycle(ids, count):
        seq = []
        while len(seq) < count:
            block = list(ids)
            rng.shuffle(block)
            seq += block
        return seq[:count]

    pos = [s for s in plan if not s["neg"]]
    neg = [s for s in plan if s["neg"]]
    for spec, lid in zip(pos, cycle(LAYOUT_IDS, len(pos))):
        spec["layout"] = lid
    for spec, lid in zip(neg, cycle(NEG_LAYOUT_IDS, len(neg))):
        spec["layout"] = lid
    for i, spec in enumerate(plan):
        spec["index"] = i
    return plan


def norm_value(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", text.lower())).strip()


TOPIC_PREFIX = (("sc_v_dob", "dob"), ("sc_v_addr", "addr"), ("sc_v_phone", "phone"), ("sc_v_name", "name"),
                ("sc_n_name", "name"), ("sc_n_email", "email"), ("sc_n_portal", "email"), ("sc_n_mrn", "id"),
                ("sc_n_member", "id"), ("sc_n_account", "id"), ("sc_n_ident", "id"), ("sc_n_url", "web"),
                ("sc_n_ip", "web"), ("sc_v_combo", "combo"))


def topic(fn) -> str:
    for prefix, name in TOPIC_PREFIX:
        if fn.__name__.startswith(prefix):
            return name
    return fn.__name__


class Rejected(Exception):
    pass


def make_row(spec: dict, attempt: int, value_rows: Counter, stats: Counter) -> dict:
    index = spec["index"]
    fam, neg = spec["family"], spec["neg"]
    rng = random.Random(f"{SEED}:{index}:{attempt}")
    ctx = Ctx(rng, fam, neg, fam == "stt_artefacts", spec["case"])
    if neg:
        fn = rng.choice(NEGATIVES[fam])
    else:
        fn = wchoice(rng, POSITIVES[fam])
    turns = fn(ctx)
    long_row = False
    if not neg and rng.random() < (0.12 if fam != "clinical_speech" else 0.3):
        long_row = True
        used = {topic(fn)}
        for _ in range(rng.randrange(1, 3)):
            for _try in range(12):
                extra = wchoice(rng, POSITIVES[fam])
                if topic(extra) not in used and extra is not sc_v_combo:
                    break
            else:
                continue
            used.add(topic(extra))
            turns += [("P", rng.choice(["One more thing.", "Also, I have another question.", "And one other thing."])),
                      ("A", rng.choice(["Sure, go ahead.", "Of course.", "Okay, what else?"]))]
            turns += extra(ctx)
        ctx.phen.append("long_visit")
    turns = frame(ctx, turns, not neg)
    text, spans, forms = render_layout(ctx, spec["layout"], turns, spec["case"], spec["strip"])
    if text != text.strip():
        raise Rejected("structure")
    errors = annotation_errors(text, spans)
    if errors or any("\n" in s["text"] for s in spans):
        raise Rejected("structure")
    if (not spans) != neg:
        raise Rejected("structure")
    if "[[" in text or "]]" in text or "|" in text and "[[" in text:
        raise Rejected("structure")
    if excluded_sequences(text) > 0 or any(excluded_value(s["text"]) for s in spans):
        raise Rejected("exclusion")
    values = {norm_value(s["text"]) for s in spans}
    if any(value_rows[v] >= VALUE_CAP for v in values if v):
        raise Rejected("value_cap")
    for v in values:
        value_rows[v] += 1
    scenario = fn.__name__
    template = f"{fam}/{scenario}"
    pat, ag = canonical_identity_group(ctx.patient.full), canonical_identity_group(ctx.agent.full)
    if neg:
        identity, identities = f"phi-free:{template}:{index:04d}", []
    else:
        assert identity_split(pat) == "train" and identity_split(ag) == "train"
        identity, identities = pat, [pat, ag]
    phen = list(dict.fromkeys(ctx.phen or ["spoken_values"]))
    row = {
        "id": f"tk19-{index:04d}",
        "text": text,
        "spans": spans,
        "split": "train",
        "suite": SUITE,
        "annotation_policy": POLICY_VERSION,
        "cohort": "realistic",
        "family": template,
        "template_family": template,
        "bank_family": fam,
        "scenario": scenario,
        "layout": spec["layout"],
        "case": spec["case"],
        "negative": neg,
        "long_visit": long_row,
        "phenomena": phen,
        "identity_group": identity,
        "identity_groups": identities,
        "pair_id": f"tk19-{index:04d}",
        "span_forms": forms,
    }
    return row


def generate(limit: int | None = None):
    """Deterministically build every row. Returns (rows, stats)."""
    exclusion_hashes()  # verifies the pinned list before any row is built
    plan = build_plan()
    if limit:
        plan = plan[:limit]
    value_rows: Counter = Counter()
    rejects: Counter = Counter()
    rows = []
    for spec in plan:
        attempt = 0
        while True:
            try:
                rows.append(make_row(spec, attempt, value_rows, rejects))
                break
            except Rejected as e:
                rejects[str(e)] += 1
                attempt += 1
                if attempt > 400:
                    raise RuntimeError(f"row {spec['index']} could not be drawn: {dict(rejects)}") from e
    return rows, {"rejected_draws": dict(rejects), "value_rows": value_rows}


def summarise(rows: list[dict], gen_stats: dict) -> dict:
    support = Counter()
    forms = {label: Counter() for label in ENTITY_TYPES}
    for row in rows:
        for s, f in zip(row["spans"], row["span_forms"]):
            support[s["label"]] += 1
            forms[s["label"]]["spoken" if f == "s" else "written"] += 1
    fam = Counter((r["bank_family"], r["negative"]) for r in rows)
    layouts = Counter(r["layout"] for r in rows)
    cases = Counter(r["case"] for r in rows)
    return {
        "rows": len(rows),
        "positives": sum(not r["negative"] for r in rows),
        "negatives": sum(r["negative"] for r in rows),
        "label_support": {label: support[label] for label in ENTITY_TYPES},
        "spoken_written": {label: dict(forms[label]) for label in ENTITY_TYPES},
        "rows_per_family": {f: {"positive": fam[(f, False)], "negative": fam[(f, True)]} for f, _ in FAMILY_ROWS},
        "layouts": dict(layouts),
        "layout_count": len(layouts),
        "max_layout_share": max(layouts.values()) / len(rows),
        "case": dict(cases),
        "case_share": {k: v / len(rows) for k, v in cases.items()},
        "rejected_draws": gen_stats["rejected_draws"],
        "max_value_rows": max(gen_stats["value_rows"].values()) if gen_stats["value_rows"] else 0,
        "max_value_share": (max(gen_stats["value_rows"].values()) / len(rows)) if gen_stats["value_rows"] else 0,
        "distinct_span_values": len(gen_stats["value_rows"]),
        "long_visits": sum(r["long_visit"] for r in rows),
    }


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def spot_check(rows: list[dict]) -> str:
    rng = random.Random(SEED + 7)
    sample = sorted(rng.sample(rows, min(100, len(rows))), key=lambda r: r["id"])
    out = ["# tk19 spoken PHI bank: spot-check sample", "",
           "100 rows drawn at random (seeded). Gold spans are shown as `[[LABEL|text]]`; a span of spoken "
           "form is marked `s`, an orthographic one `w` (`[[LABEL:s|text]]`). Layout line breaks are kept.", ""]
    for row in sample:
        marked, prev, text = [], 0, row["text"]
        for s, f in zip(row["spans"], row["span_forms"]):
            marked.append(text[prev:s["start"]])
            marked.append(f"[[{s['label']}:{f}|{text[s['start']:s['end']]}]]")
            prev = s["end"]
        marked.append(text[prev:])
        out += [f"### {row['id']}  {row['bank_family']} / {row['scenario']}  layout={row['layout']}  case={row['case']}"
                f"{'  NEGATIVE' if row['negative'] else ''}", "", "```", "".join(marked), "```", ""]
    return "\n".join(out)


def write_outputs(rows: list[dict], stats: dict, out: Path, report_dir: Path):
    payload = ("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=False) for r in rows) + "\n").encode("utf-8")
    out.write_bytes(payload)
    summary = summarise(rows, stats)
    source = Path(__file__).read_bytes()
    exclusions = ROOT / "data" / "places_exclusions.sha256.json"
    manifest = {
        "name": out.stem,
        "role": "training bank",
        "split": "train",
        "rows": len(rows),
        "documents": len(rows),
        "sha256": sha256_bytes(payload),
        "annotation_policy": POLICY_VERSION,
        "policy_decisions": "S1-S7 (annotation_policy_spoken.py), T1-T2, places P1-P7",
        "independent_final": False,
        "label_support": summary["label_support"],
        "entity_support": summary["label_support"],
        "label_counts": summary["label_support"],
        "spoken_written_by_type": summary["spoken_written"],
        "rows_per_family": summary["rows_per_family"],
        "matched_negative_rows": summary["negatives"],
        "layout_count": summary["layout_count"],
        "max_layout_share": summary["max_layout_share"],
        "case_share": summary["case_share"],
        "exclusion_filter": {"list": str(exclusions.relative_to(ROOT)), "list_sha256": sha256_bytes(exclusions.read_bytes()),
                             "rejected_draws": summary["rejected_draws"], "rows_matching_after_filter": 0},
        "value_cap": {"per_value_rows": VALUE_CAP, "max_value_rows": summary["max_value_rows"],
                      "max_value_share": summary["max_value_share"]},
        "generator": "scripts/takeover/author19_spoken_phi.py",
        "generator_sha256": sha256_bytes(source),
        "seed": SEED,
        "provenance": "synthetic; invented people, places and numbers under the reserved schemes (phones 555-0100 to "
                      "555-0199, SSN area 900-999, example and test domains, TEST-NET IPs); no external data",
    }
    (out.with_suffix(out.suffix + ".manifest.json")).write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "stats.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    (report_dir / "spot_check_sample.md").write_text(spot_check(rows))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--report-dir", default=str(ROOT / "reports" / "takeover" / "author19_spoken_phi"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    rows, stats = generate(args.limit)
    summary = write_outputs(rows, stats, Path(args.out), Path(args.report_dir))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
