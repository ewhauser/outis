#!/usr/bin/env python3
"""tk17: synthetic laboratory and pathology report training bank.

Deterministic generator for docs/lab-pathology-bank-brief.md. Everything here
is invented: names are composed from syllable pools, identifiers follow the
invented schemes documented in AUTHOR_REPORT.md, phones are 555-0100..0199,
domains are reserved and IPs are TEST-NET. Conventions follow
docs/annotation-policy-v2.md, decisions L1 to L4
(outis.data.annotation_policy_reports.REPORT_FIXTURES).

Usage (from the worktree root):
  PYTHONPATH=src python scripts/takeover/author17_lab_pathology.py
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / ".cache" / "tiktoken"))

from outis.data.annotation_policy import (  # noqa: E402
    POLICY_VERSION,
    canonical_identity_group,
    identity_split,
)
from outis.data.generators.places import excluded_sequences  # noqa: E402

SEED = 17017
SUITE = "tk17_lab_pathology"
CORPUS = ROOT / "data" / "tk17_train_lab_pathology.jsonl"
REPORT_DIR = ROOT / "reports" / "takeover" / "author17_lab_pathology"
VALUE_CAP = 9  # 0.5% of 1,800 rows
FAMILIES = (
    ("chem", "lab_chem_heme_coag", 540),
    ("micro", "lab_microbiology", 270),
    ("path", "lab_surgical_cytology", 360),
    ("poc", "lab_poc_sendout", 270),
    ("portal", "lab_portal_fax_copies", 360),
)
CASE_PATTERN = ["normal"] * 12 + ["caps"] * 5 + ["lower"] * 3

# --------------------------------------------------------------------------
# Value pools (all invented)
# --------------------------------------------------------------------------

FIRST_CURATED = """Aldith Amara Anselm Aurelio Beatrix Berengar Bronwen Caspian Cordelia Corvin Dagny Dariel
Edda Eldred Elowen Emeric Fenwick Fionnuala Galen Gisela Godric Halvard Hesper Idris Ione Isolde Jorund
Kestrel Kieran Leofric Linnea Lovisa Lysander Maelis Marisol Mathis Merrick Niamh Nils Oriel Osric Ottoline
Perrin Philippa Quillon Radmila Rafferty Rosalind Saoirse Sigrun Soren Sunniva Tamsin Tavish Thessaly
Torquil Ulrike Una Valdis Verity Vesper Wilhelmina Wystan Xanthe Yaroslav Ysolde Zdenek Zelda Anneliese
Bertil Calla Dorrit Eirik Fidelma Gudrun Hjalmar Ingrith Jessamy Kenric Lorcan Matilde Nerys Ottilie
Pelagia Renzo Sabine Teodor Ursel Viggo Winifred Aoife Brannoch Carys Desmond Elspeth Faolan Greta
Hollis Imelda Joaquim Katarzyna Lenka Mirela Nuala Orsolya Petra Quentin Rhiannon Sorcha Tomasz Yvaine
Zoë Renée Inés Tomás Åsa Søren Brígida Mélisande Thaïs Dagmar Eamon Fabrizio Ignatius Leocadia Hortense
Ambrose Blythe Clemency Dorothea Evander Frederika Gwendolen Hector Ivo Jocasta Kerensa Leander Morwenna""".split()
FIRST_STEMS = """Ald Bren Cael Dar Eld Fen Gar Hal Ise Jas Kel Lor Mar Nor Ors Per Quin Rob Sol Tam Ulr Var Wen
Yor Zan Adel Bri Cor Dev Eli Fran Gem Hen Ing Jul Kar Lin Mel Nic Ole Pen Ros Sig Tor Una Vic Wil Ana Bea Cle""".split()
FIRST_ENDS = """a e ia ina wyn ric mund bert dric lyn ette ard ald na ra ton sen tha vin lo ren dis mar ley""".split()

LAST_CURATED = """Quennell Brannock Ravelston Halloran Verhoeven Ostrander Pellham Marwick Ombrey Tindle Vossmere
Yarrowick Felsham Orlinde Quarrington Brackenwheel Fennick Ashgrove Larkspur Mossmere Ellery Kestrine
Tollund Marrowdale Gorsefield Corriston Vellmont Saint-Orme Underhaye Wetherell Thackeray Pennywhistle
Okonkwo-Dray Lindqvist Brännström Nuñez-Varga Haverkamp Szymanowicz Tranter
Vandermeulen Abernethy Blackwood-Hale Cavendish-Orr Dunmore Eastlake Fairweather Galbraith Hollingsworth
Iverson Jorgensen Kettering Lockhart Montclair Nightingale Oglethorpe Plimsoll Quigley Rutherglen Stanhope
Truscott Umbridge Venables Whitcombe Yardley Zeller Aldenhoven Bexley Crowhurst Dovedale Embleton Fitzwarren""".split()
LAST_PRE = """Ash Brack Cor Dun Ell Fen Gar Hol Ing Kes Lar Mor Nor Ors Pen Quen Rav Sel Tor Ulm Vos Wen Yar Zel
Ber Cal Dra Eld Fal Gren Hav Jen Kell Lind Mer Nas Ost Pol Rin Sar Tav Ves Wil Bar Cres Dov Ev Fair Gil
Hesk Isen Jor Kar Lom Mar Oak Pry Rus Sark Thorn Whit Tam Pell Quar Marl Haw Brin Cass""".split()
LAST_SUF = """ell ock ston ford well mere ham wick ley ner dale by son ing stead hurst croft man ow ard ick vane
sley gate field bury combe holt wyn thorpe worth shaw ridge stow""".split()
PARTICLES = ["de la ", "van ", "von ", "al-", "O'", "Mc", "Di ", "ten "]
CREDS = ["MD", "DO", "MD, PhD", "NP", "PA-C", "FACP", "DNP", "MBBS"]
PATH_CREDS = ["MD", "MD, PhD", "DO", "MBBS"]
STOP_INITIALS = {"ED", "RN", "MD", "PA", "NP", "OR", "ER", "IV", "BP", "HR", "DO", "AM", "PM", "ID", "PT", "MT",
                 "CT", "MR", "NA", "NO", "ON", "TO", "IN", "AT", "BY", "OF", "MS", "MX", "DR", "ICU", "CBC",
                 "BMP", "CMP", "PSA", "INR", "PTT", "TSH", "ESR", "CRP", "HIV", "MLS", "RRT", "LPN", "CNA",
                 "WBC", "RBC", "MCV", "UTI", "HCG", "ANA", "ASC", "LIS", "QC", "CO", "ME", "HE", "SHE", "HER",
                 "DNA", "RNA", "PHD", "DNP", "RPH", "CLS", "MLT", "NPI", "DOB", "MRN", "SEX", "LAB", "FAX", "TEL", "AGE", "PAP", "FNA", "BUN", "GFR", "HGB", "HCT", "MCH", "PTH", "FDA", "CDC", "NHS", "TSB", "CHF", "COPD", "UTC", "AFB", "CSF", "UA", "PCR", "ANC", "ABG", "TNT", "ESBL"}

PLACE_PRE = """Ostra Brack Cedar Dun Fenn Gor Hal Ink Jun Kest Lark Mar Nor Orrin Pell Quill Rav Sil Tarn Ulv
Vell Wex Yarr Zann Amber Birch Cor Dale Elm Frost Grove Heath Iron Jade Kiln Loam Moss Nettle Ox Pine
Quarry Reed Slate Thistle Umber Vale Willow Aspen Bram Clay Drift""".split()
PLACE_SUF = """vel mere holt wick haven field ridge wood stone brook ford gate mont ton bury crest dale fall port
vale ham mouth minster hollow""".split()
STREET_TYPES = ["Street", "Road", "Avenue", "Boulevard", "Lane", "Way", "Drive", "Parkway", "Court", "Place"]
STATES = """AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND
OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY""".split()
AREA_CODES = """202 207 218 303 304 312 406 415 469 503 505 512 573 602 608 612 617 646 651 701 702 713 719 720
801 816 901 913 919 971 314 404 614 828""".split()
LAB_TYPES = ["Reference Laboratory", "Clinical Laboratories", "Pathology Associates", "Laboratory Services",
             "Diagnostics", "Pathology Group", "Medical Laboratories", "Regional Laboratory", "Cytology Services"]
SITE_TYPES = ["General Hospital", "Medical Center", "Community Hospital", "Health Center", "Clinic",
              "Family Medicine", "Regional Medical Center", "Oncology Associates", "Internal Medicine Group",
              "Cancer Institute", "Memorial Hospital"]
GENERIC_LABS = ["Core Laboratory", "Main Laboratory", "Hospital Laboratory", "Central Lab", "Stat Laboratory",
                "Outpatient Laboratory"]
EMAIL_LOCALS = ["results", "labreports", "pathology", "lab.support", "records", "labdesk", "portal.help"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_FULL = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# analyte: key -> (display, unit, lo, hi, decimals, loinc)
AN = {
    "na": ("Sodium", "mmol/L", 135, 145, 0, "2951-2"), "k": ("Potassium", "mmol/L", 3.5, 5.1, 1, "2823-3"),
    "cl": ("Chloride", "mmol/L", 98, 107, 0, "2075-0"), "co2": ("CO2", "mmol/L", 22, 29, 0, "2028-9"),
    "bun": ("BUN", "mg/dL", 7, 20, 0, "3094-0"), "cr": ("Creatinine", "mg/dL", 0.6, 1.2, 2, "2160-0"),
    "glu": ("Glucose", "mg/dL", 70, 99, 0, "2345-7"), "ca": ("Calcium", "mg/dL", 8.5, 10.2, 1, "17861-6"),
    "tp": ("Total protein", "g/dL", 6.0, 8.3, 1, "2885-2"), "alb": ("Albumin", "g/dL", 3.5, 5.0, 1, "1751-7"),
    "tbili": ("Total bilirubin", "mg/dL", 0.2, 1.2, 1, "1975-2"), "alp": ("Alk phos", "U/L", 44, 147, 0, "6768-6"),
    "ast": ("AST", "U/L", 10, 40, 0, "1920-8"), "alt": ("ALT", "U/L", 7, 56, 0, "1742-6"),
    "mg": ("Magnesium", "mg/dL", 1.7, 2.2, 1, "19123-9"), "phos": ("Phosphorus", "mg/dL", 2.5, 4.5, 1, "2777-1"),
    "chol": ("Cholesterol, total", "mg/dL", None, 200, 0, "2093-3"), "hdl": ("HDL cholesterol", "mg/dL", 40, None, 0, "2085-9"),
    "ldl": ("LDL cholesterol", "mg/dL", None, 100, 0, "13457-7"), "trig": ("Triglycerides", "mg/dL", None, 150, 0, "2571-8"),
    "tsh": ("TSH", "mIU/L", 0.4, 4.0, 2, "3016-3"), "ft4": ("Free T4", "ng/dL", 0.8, 1.8, 2, "3024-7"),
    "a1c": ("Hemoglobin A1c", "%", 4.0, 5.6, 1, "4548-4"), "ferr": ("Ferritin", "ng/mL", 15, 200, 0, "2276-4"),
    "iron": ("Iron", "ug/dL", 60, 170, 0, "2498-4"), "tibc": ("TIBC", "ug/dL", 250, 450, 0, "2500-7"),
    "wbc": ("WBC", "x10^3/uL", 4.0, 11.0, 1, "6690-2"), "rbc": ("RBC", "x10^6/uL", 4.2, 5.9, 2, "789-8"),
    "hgb": ("Hemoglobin", "g/dL", 12.0, 16.0, 1, "718-7"), "hct": ("Hematocrit", "%", 36, 48, 1, "4544-3"),
    "mcv": ("MCV", "fL", 80, 100, 1, "787-2"), "mch": ("MCH", "pg", 27, 33, 1, "785-6"),
    "mchc": ("MCHC", "g/dL", 32, 36, 1, "786-4"), "rdw": ("RDW", "%", 11.5, 14.5, 1, "788-0"),
    "plt": ("Platelets", "x10^3/uL", 150, 400, 0, "777-3"), "neut": ("Neutrophils", "%", 40, 70, 0, "770-8"),
    "lym": ("Lymphocytes", "%", 20, 45, 0, "736-9"), "mono": ("Monocytes", "%", 2, 10, 0, "5905-5"),
    "eos": ("Eosinophils", "%", 0, 6, 0, "713-8"),
    "pt": ("PT", "sec", 11.0, 13.5, 1, "5902-2"), "inr": ("INR", "", 0.9, 1.1, 1, "6301-6"),
    "aptt": ("aPTT", "sec", 25, 35, 0, "3173-2"), "fib": ("Fibrinogen", "mg/dL", 200, 400, 0, "3255-7"),
    "ddim": ("D-dimer", "ug/mL FEU", None, 0.5, 2, "48066-5"),
}
PANELS = {
    "bmp": ["na", "k", "cl", "co2", "bun", "cr", "glu", "ca"],
    "cmp": ["na", "k", "cl", "co2", "bun", "cr", "glu", "ca", "tp", "alb", "tbili", "alp", "ast", "alt"],
    "cbc": ["wbc", "rbc", "hgb", "hct", "mcv", "mch", "mchc", "rdw", "plt"],
    "cbcd": ["wbc", "rbc", "hgb", "hct", "mcv", "rdw", "plt", "neut", "lym", "mono", "eos"],
    "lipid": ["chol", "hdl", "ldl", "trig"], "thyroid": ["tsh", "ft4"], "iron": ["iron", "tibc", "ferr"],
    "coag": ["pt", "inr", "aptt", "fib", "ddim"], "mix": ["a1c", "glu", "cr", "mg", "phos"],
}
PANEL_NAMES = {
    "bmp": ["BASIC METABOLIC PANEL", "Basic Metabolic Panel", "Chem 8"],
    "cmp": ["COMPREHENSIVE METABOLIC PANEL", "Comprehensive Metabolic Panel", "Chem 14"],
    "cbc": ["CBC", "Complete Blood Count", "Hematology Report"],
    "cbcd": ["CBC WITH DIFFERENTIAL", "Complete Blood Count with Differential"],
    "lipid": ["Lipid Panel", "LIPID PROFILE"], "thyroid": ["Thyroid Function", "TSH / Free T4"],
    "iron": ["Iron Studies"], "coag": ["Coagulation Panel", "PT/INR and aPTT", "COAGULATION STUDIES"],
    "mix": ["Chemistry Results", "Diabetes Monitoring Panel", "Renal and Glycemic Panel"],
}
CPT = {"bmp": "80048", "cmp": "80053", "cbc": "85027", "cbcd": "85025", "lipid": "80061", "thyroid": "84443",
       "iron": "83540", "coag": "85610", "mix": "83036"}
INSTRUMENTS = ["AU-680", "cobas c503", "Sysmex XN-9000", "Atellica CH 930", "ACL TOP 750", "Alinity ci", "DxC 700 AU",
               "Vitros 5600", "Stago STA-R Max"]
LOT_DECOYS = ["Lot 4471B", "Reagent lot 24A113", "Calibrator lot C-88120", "Reagent lot 555-0120", "QC lot L-7731",
              "Lot AB-4471-C", "Control lot 0155-9A"]
COUNT_DECOYS = ["Creatine kinase 4471 U/L", "NT-proBNP 10432 pg/mL", "Positive ratio 12/15", "Titer 1:640",
                "Cell count 555-0141 /uL (reportable range 20-900000)", "Lipase 91234 U/L (reported as >90000)",
                "Specific gravity 1.015", "Count 0100-0199 (instrument units)", "Panel ratio 12/15"]
EPONYM_NOTES = ["Direct Coombs test negative.", "Gram stain reviewed.", "Ziehl-Neelsen stain not indicated.",
                "Wright-Giemsa smear reviewed.", "Reed-Sternberg cells not identified.",
                "Reported per the Bethesda System.", "Hashimoto thyroiditis pattern not seen.",
                "Gleason score assigned per ISUP 2014 grading.", "Breslow thickness reported in mm.",
                "Epstein-Barr virus serology not requested.", "Philadelphia chromosome FISH not requested.",
                "Hurthle cell change absent.", "Warthin-Starry stain negative for organisms.",
                "West Nile virus IgM not requested.", "Lyme disease serology not requested.",
                "Nottingham grade assigned per CAP protocol.", "Milan System categories apply."]
ORGANISMS = ["Escherichia coli", "Klebsiella pneumoniae", "Staphylococcus aureus", "Proteus mirabilis",
             "Enterococcus faecalis", "Pseudomonas aeruginosa", "Streptococcus agalactiae", "Candida albicans",
             "Staphylococcus epidermidis", "Enterobacter cloacae", "Streptococcus pyogenes", "Serratia marcescens"]
GRAM_POS = ("Staphylococcus", "Enterococcus", "Streptococcus")
ABX = {"neg": ["Ampicillin", "Cefazolin", "Ceftriaxone", "Ciprofloxacin", "Gentamicin", "Nitrofurantoin",
               "Trimethoprim-sulfamethoxazole", "Piperacillin-tazobactam", "Meropenem"],
       "pos": ["Oxacillin", "Vancomycin", "Clindamycin", "Doxycycline", "Linezolid", "Erythromycin",
               "Trimethoprim-sulfamethoxazole", "Penicillin"]}
MIC = ["<=0.25", "<=0.5", "<=1", "<=2", "<=4", "0.5", "1", "2", "4", "8", ">=8", ">=16", ">=32", "<=8", "16"]


def invent_place(r):
    return r.choice(PLACE_PRE) + r.choice(PLACE_SUF)


def titled(word):
    return word[:1].upper() + word[1:]


def first_name(r):
    if r.random() < 0.4:
        return r.choice(FIRST_CURATED)
    return titled((r.choice(FIRST_STEMS) + r.choice(FIRST_ENDS)).lower())


def last_name(r):
    x = r.random()
    if x < 0.12:
        return r.choice(LAST_CURATED)
    base = titled((r.choice(LAST_PRE) + r.choice(LAST_SUF)).lower())
    if x < 0.3:
        p = r.choice(PARTICLES)
        return p + base if p.endswith((" ", "-")) else p + base
    if x < 0.37:
        return base + "-" + titled((r.choice(LAST_PRE) + r.choice(LAST_SUF)).lower())
    return base


def initials(r):
    """Two or three letters with no vowel in the middle, so they do not read as words."""
    n = r.choice([2, 3, 3])
    cons, vowels = "BCDFGHJKLMNPQRSTVWXZ", "AEIOU"
    while True:
        chars = [r.choice(cons) for _ in range(n)]
        if r.random() < 0.2:
            chars[r.choice([0, n - 1])] = r.choice(vowels)
        s = "".join(chars)
        if s not in STOP_INITIALS:
            return s


def digits(r, n):
    return "".join(r.choice("0123456789") for _ in range(n))


def letters2(r, n=2):
    return "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(n))


# --------------------------------------------------------------------------
# Document builder
# --------------------------------------------------------------------------

MARK = re.compile(r"\[\[([A-Z]+)\|([^\[\]]+)\]\]")


class Doc:
    """Ordered (text, label|None) pieces; renders text and spans for a case style."""

    def __init__(self):
        self.parts = []

    def t(self, s):
        if s:
            self.parts.append((s, None))

    def s(self, label, value):
        self.parts.append((value, label))

    @classmethod
    def from_marked(cls, marked):
        doc, prev = cls(), 0
        for m in MARK.finditer(marked):
            doc.t(marked[prev:m.start()])
            doc.s(m[1], m[2])
            prev = m.end()
        doc.t(marked[prev:])
        return doc

    def render(self, case="normal"):
        fn = {"normal": lambda x: x, "caps": str.upper, "lower": str.lower}[case]
        out, spans, pos = [], [], 0
        for piece, label in self.parts:
            conv = fn(piece)
            if len(conv) != len(piece):
                raise ValueError(f"case conversion changed length: {piece!r}")
            if label:
                spans.append({"start": pos, "end": pos + len(conv), "label": label, "text": conv})
            out.append(conv)
            pos += len(conv)
        text = "".join(out)
        text = text.rstrip()
        if len(text) != len(text.lstrip()):
            raise ValueError("leading whitespace")
        return text, [s for s in spans if s["end"] <= len(text)]



# --------------------------------------------------------------------------
# Case (shared by a positive row and its matched negative)
# --------------------------------------------------------------------------

class NS:
    pass


STATS = collections.Counter()  # redraw counters (counts only)


def fmt_date(d, st):
    m = d.month - 1
    if st == 0:
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    if st == 1:
        return f"{d.year}-{d.month:02d}-{d.day:02d}"
    if st == 2:
        return f"{MONTHS[m]} {d.day}, {d.year}"
    if st == 3:
        return f"{d.day:02d}-{MONTHS[m]}-{d.year}"
    if st == 4:
        return f"{MONTHS_FULL[m]} {d.day}, {d.year}"
    if st == 5:
        return f"{d.month}/{d.day}/{d.year % 100:02d}"
    if st == 6:
        return f"{d.day} {MONTHS[m]} {d.year}"
    return f"{WEEKDAYS[d.weekday()]}, {MONTHS_FULL[m]} {d.day}, {d.year}"


def fmt_time(h, mi, st):
    if st == 0:
        return f"{h:02d}:{mi:02d}"
    if st == 1:
        return f"{h:02d}{mi:02d}"
    hh = h % 12 or 12
    return f"{hh}:{mi:02d} {'AM' if h < 12 else 'PM'}"


def fmt_phone(r, style=None, ext=False):
    area, last = r.choice(AREA_CODES), r.randint(100, 199)
    st = r.randrange(5) if style is None else style
    core = {0: f"({area}) 555-0{last}", 1: f"{area}-555-0{last}", 2: f"{area}.555.0{last}",
            3: f"+1 {area} 555 0{last}", 4: f"1-{area}-555-0{last}"}[st]
    if ext:
        core += r.choice([" ext. ", " x", " EXT. ", ", extension "]) + digits(r, r.choice([3, 4]))
    return core


def person(r, kind="clin"):
    """A full person whose canonical identity falls in the training partition."""
    while True:
        p = NS()
        p.first, p.last = first_name(r), last_name(r)
        if identity_split(canonical_identity_group(f"{p.first} {p.last}")) == "train":
            break
        STATS["identity_redraws"] += 1
    p.cred = r.choice(PATH_CREDS if kind == "path" else CREDS)
    p.form = r.choices([0, 1, 2, 3], [4, 3, 2, 3])[0]
    return p


def street_address(r, suite=True):
    num = r.randint(11, 9899)
    street = invent_place(r) + " " + r.choice(STREET_TYPES) if r.random() < 0.8 else r.choice(
        ["N ", "S ", "E ", "W "]) + invent_place(r) + " " + r.choice(STREET_TYPES)
    unit = ""
    if suite and r.random() < 0.3:
        unit = r.choice([", Suite ", ", Ste. ", ", Unit ", ", Bldg "]) + str(r.randint(2, 640))
    return f"{num} {street}{unit}, {invent_place(r)}, {r.choice(STATES)} {digits(r, 5)}"


def make_case(r, family):
    k = NS()
    k.family = family
    d0 = dt.date(2023, 1, 1) + dt.timedelta(days=r.randrange(1370))
    k.dc = d0
    k.dr = d0 + dt.timedelta(days=r.choice([0, 0, 0, 1]))
    k.drep = k.dr + dt.timedelta(days=r.choice([0, 0, 1, 1, 2, 3]))
    k.dprint = k.drep + dt.timedelta(days=r.choice([0, 0, 1]))
    k.tc = (r.randint(5, 17), r.choice([0, 7, 12, 15, 21, 30, 42, 48, 55, 5]))
    k.tr = (min(23, k.tc[0] + r.randint(1, 5)), r.randint(0, 59))
    k.trep = (min(23, r.randint(8, 22)), r.randint(0, 59))
    k.tprint = (min(23, r.randint(8, 22)), r.randint(0, 59))
    k.dstyle = r.choices([0, 1, 2, 3, 4, 5, 6, 7], [30, 18, 16, 10, 12, 4, 6, 4])[0]
    k.dob_style = k.dstyle if r.random() < 0.8 else r.choice([0, 1, 2])
    k.tstyle = r.choices([0, 1, 2], [70, 10, 20])[0]
    # patient
    p = person(r)
    k.patient = p
    x = r.random()
    k.mi = letters2(r, 1)
    k.suffix = r.choice(["Jr.", "Sr.", "III", "II"])
    k.pform = 0 if x < 0.62 else (1 if x < 0.85 else (2 if x < 0.93 else 3))
    if k.pform == 0:
        k.pname = f"{p.first} {p.last}"
    elif k.pform == 1:
        k.pname = f"{p.first} {k.mi}. {p.last}" if r.random() < 0.7 else f"{p.first} {k.mi} {p.last}"
    elif k.pform == 2:
        k.pname = f"{p.first} {p.last} {k.suffix}"
    else:
        k.pname = f"{p.first} {p.last}"
    k.sex = r.choice(["F", "M"])
    k.age = r.randint(90, 107) if r.random() < 0.2 else (r.randint(1, 17) if (r.random() < 0.1 and family != "path") else r.randint(18, 89))
    k.dob = dt.date(d0.year - k.age, d0.month, min(d0.day, 28)) - dt.timedelta(days=r.randint(0, 340))
    k.agestyle = r.randrange(4)
    k.mrn = r.choice([lambda: digits(r, r.choice([6, 7, 8])), lambda: f"{letters2(r)}-{digits(r, 7)}",
                      lambda: f"{letters2(r, 3)}{digits(r, 6)}", lambda: f"M{digits(r, 8)}"])()
    k.acct = r.choice([lambda: digits(r, 8), lambda: f"ACCT-{digits(r, 7)}", lambda: f"A{digits(r, 9)}"])()
    k.enc = r.choice([lambda: digits(r, 9), lambda: f"CSN{digits(r, 8)}", lambda: f"V{digits(r, 8)}"])()
    k.member = r.choice([lambda: f"{letters2(r, 3)}{digits(r, 9)}", lambda: f"W{digits(r, 9)}",
                         lambda: f"{digits(r, 3)}-{digits(r, 6)}-{letters2(r)}"])()
    k.ssn = f"9{digits(r, 2)}-{digits(r, 2)}-{digits(r, 4)}"
    k.pphone = fmt_phone(r)
    k.paddr = street_address(r, suite=False)
    k.pcity = f"{invent_place(r)}, {r.choice(STATES)}"
    k.pzip = digits(r, 5) if r.random() < 0.7 else f"{digits(r, 5)}-{digits(r, 4)}"
    k.town = invent_place(r)
    k.county = invent_place(r) + " County" + (f", {r.choice(STATES)}" if r.random() < 0.3 else "")
    # ordering side
    k.ordp = person(r)
    k.npi = "99" + digits(r, 8)
    k.ordphone, k.ordfax = fmt_phone(r), fmt_phone(r)
    k.ordfac = invent_place(r) + " " + r.choice(SITE_TYPES)
    k.ordaddr = street_address(r)
    # performing lab
    k.lab = invent_place(r) + " " + r.choice(LAB_TYPES)
    k.laddr = street_address(r)
    k.lphone, k.lfax = fmt_phone(r, ext=r.random() < 0.15), fmt_phone(r)
    k.clia = f"{r.randint(91, 99)}D{digits(r, 7)}"
    k.lic = r.choice([lambda: f"CL-{digits(r, 7)}", lambda: f"LAB{digits(r, 6)}", lambda: f"{r.choice(STATES)}-LC-{digits(r, 6)}"])()
    k.director = person(r, "path")
    k.pathologist = person(r, "path")
    k.pathologist2 = person(r, "path")
    k.domain = invent_place(r).lower()
    # accession-type identifiers
    yy = f"{d0.year % 100:02d}"
    k.spec = r.choice([lambda: f"SP-{yy}-{digits(r, 6)}", lambda: f"L{yy}{digits(r, 6)}", lambda: digits(r, 10),
                       lambda: f"{letters2(r)}{digits(r, 7)}", lambda: f"{d0.year}-{digits(r, 7)}"])()
    k.acc = r.choice([lambda: f"RA-{digits(r, 7)}", lambda: f"ACC{digits(r, 8)}", lambda: f"{yy}{d0.month:02d}-{digits(r, 6)}",
                      lambda: f"{letters2(r, 3)}-{digits(r, 7)}"])()
    k.pathacc = r.choice([lambda: f"S{yy}-{digits(r, 5)}", lambda: f"SP{yy}-{digits(r, 5)}", lambda: f"{letters2(r)}-{yy}-{digits(r, 5)}",
                          lambda: f"SU-{d0.year}-{digits(r, 6)}"])()
    k.cytacc = r.choice([lambda: f"CY{yy}-{digits(r, 4)}", lambda: f"C{yy}-{digits(r, 5)}", lambda: f"GYN-{yy}-{digits(r, 5)}"])()
    k.microacc = r.choice([lambda: f"MB{yy}-{digits(r, 6)}", lambda: f"M{digits(r, 8)}", lambda: f"{d0.year}-{digits(r, 7)}-M"])()
    k.order = r.choice([lambda: f"{digits(r, 7)}-{letters2(r, 1)}", lambda: f"ORD{digits(r, 8)}", lambda: digits(r, 9)])()
    k.sendout = r.choice([lambda: f"SO-{digits(r, 6)}", lambda: f"REF{digits(r, 7)}", lambda: f"X{digits(r, 8)}"])()
    k.outacc = f"{letters2(r)}{(d0.year - r.randint(0, 3)) % 100:02d}-{digits(r, 5)}"
    k.outfac = invent_place(r) + " " + r.choice(SITE_TYPES + LAB_TYPES)
    k.outlab_phone = fmt_phone(r)
    k.outlab = invent_place(r) + " " + r.choice(LAB_TYPES)
    k.outlab_addr = street_address(r)
    k.code = f"{''.join(r.choice('BCDFGHJKLMNPQRSTVWXZ23456789') for _ in range(4))}-{''.join(r.choice('BCDFGHJKLMNPQRSTVWXZ23456789') for _ in range(4))}"
    k.url = r.choice([f"https://portal.{k.domain}.example/results/{digits(r, 6)}", f"https://results.{k.domain}.test/r/{digits(r, 4)}",
                      f"https://labs.{k.domain}.example.org/view/{digits(r, 5)}", f"https://my.{k.domain}.example.net/report/{digits(r, 7)}",
                      f"portal.{k.domain}.example/lab/{digits(r, 5)}"])
    k.email = f"{r.choice(EMAIL_LOCALS)}@{r.choice(['', 'mail.'])}{k.domain}.{r.choice(['example', 'test', 'example.org', 'example.com', 'example.net'])}"
    k.ip = r.choice(["192.0.2.", "198.51.100.", "203.0.113."]) + str(r.randint(2, 250))
    k.tech, k.verified = initials(r), initials(r)
    k.recvby = initials(r)
    k.callee = person(r)
    k.callrole = r.choice(["RN", "Dr.", "charge nurse", "NP", "unit clerk"])
    k.caller = initials(r)
    k.drawn_by = initials(r)
    return k


def build_case(seed, family):
    return make_case(random.Random(f"{seed}:case"), family)



# --------------------------------------------------------------------------
# Rendering context: positive mode emits PHI spans, negative mode withholds them
# --------------------------------------------------------------------------

NEG_OK = {"collected", "received", "reported", "printed", "agesex", "lab", "sex", "agephrase_ok"}
PHENOMENA = {
    "patient": "patient_name", "pfirst": "first_name_only", "phon": "honorific_surname", "dob": "dob_date",
    "mrn": "mrn_caption_outside", "acct": "account_number", "enc": "encounter_id", "member": "insurance_id", "ssn": "ssn",
    "ordprov": "clinician_name", "npi": "clinician_npi", "ordfac": "ordering_facility", "spec": "specimen_id",
    "acc": "accession_id", "pathacc": "accession_id", "cytacc": "accession_id", "microacc": "accession_id",
    "order": "order_id", "sendout": "sendout_id", "outacc": "outside_accession", "collected": "date_in_timestamp",
    "received": "date_in_timestamp", "reported": "date_in_timestamp", "printed": "date_in_timestamp",
    "cdate": "date_boundaries", "rdate": "date_boundaries", "repdate": "date_boundaries",
    "lab": "performing_lab", "labaddr": "lab_address", "labphone": "phone_numbers", "labfax": "fax_number",
    "clia": "lab_identifier", "lic": "lab_identifier", "director": "lab_director", "path": "pathologist_signout",
    "path2": "pathologist_signout", "tech": "initials_only_name", "verified": "initials_only_name",
    "recvby": "initials_only_name", "caller": "initials_only_name", "callee": "callback_staff",
    "pphone": "phone_numbers", "paddr": "patient_address", "pcity": "locality", "pzip": "zip_alone", "county": "county_name", "town": "p7_city_modifier",
    "email": "email", "url": "portal_url", "code": "portal_access_code", "ip": "ip_address", "outfac": "outside_material",
    "outlab": "send_out_lab", "outlab_addr": "lab_address", "outlab_phone": "phone_numbers", "ordphone": "phone_numbers",
    "ordfax": "fax_number", "ordaddr": "ordering_address", "agesex": "age_over_89_only", "agephrase": "age_over_89_only",
    "drawn": "initials_only_name",
}
HONORIFIC_CREDS = ("MD", "DO", "MD, PhD", "FACP", "MBBS")


class Ctx:
    def __init__(self, k, phi, seed):
        self.k, self.phi = k, phi
        self.r = random.Random(f"{seed}:style")
        self.cr = random.Random(f"{seed}:content")
        self.d = Doc()
        self.kvsep = ": "
        self.phen = []
        self.people = []

    # -- plumbing ----------------------------------------------------------
    def t(self, s):
        self.d.t(s)

    def nl(self, n=1):
        self.d.t("\n" * n)

    def tag(self, name):
        if name not in self.phen:
            self.phen.append(name)

    def has(self, kind):
        return self.phi or kind in NEG_OK

    def emit(self, kind):
        if kind in PHENOMENA:
            self.tag(PHENOMENA[kind])
        getattr(self, "e_" + kind)()

    def line(self, *items, sep="  ", kvsep=None, end="\n"):
        kvsep = self.kvsep if kvsep is None else kvsep
        live = []
        for it in items:
            label, kind = it[0], it[1]
            alt = it[2] if len(it) > 2 else None
            if self.has(kind):
                live.append((label, kind, None))
            elif alt:
                live.append((label, None, alt))
        if not live:
            return False
        for i, (label, kind, alt) in enumerate(live):
            if i:
                self.t(sep)
            if label:
                self.t(label + kvsep)
            if kind:
                self.emit(kind)
            else:
                self.t(alt)
        self.t(end)
        return True

    def sent(self, parts, alt=None, end=""):
        if all(isinstance(p, str) or self.has(p[0]) for p in parts):
            for p in parts:
                if isinstance(p, str):
                    self.t(p)
                else:
                    self.emit(p[0])
            self.t(end)
            return True
        if alt:
            self.t(alt + end)
        return False

    # -- value emitters ----------------------------------------------------
    def date(self, d, st=None):
        self.d.s("DATE", fmt_date(d, self.k.dstyle if st is None else st))

    def stamp(self, d, hm):
        tm = fmt_time(*hm, self.k.tstyle)
        if self.phi:
            self.date(d)
            self.t(" " + tm)
        else:
            self.t(tm)

    def person(self, p, hon=True):
        self.people.append(p)
        md = p.cred in HONORIFIC_CREDS
        if p.form == 0:
            self.t("Dr. " if md else "")
            self.d.s("NAME", f"{p.first} {p.last}")
        elif p.form == 1:
            self.t("Dr. " if md else p.cred.split(",")[0] + " ")
            self.d.s("NAME", p.last)
        elif p.form == 2:
            self.d.s("NAME", f"{p.first} {p.last}")
            self.t(", " + p.cred)
        else:
            self.t("Dr. " if md else "")
            self.d.s("NAME", f"{p.first} {p.last}")
            self.t(", " + p.cred)

    def e_patient(self):
        self.people.append(self.k.patient)
        self.d.s("NAME", self.k.pname)

    def e_pfirst(self):
        self.people.append(self.k.patient)
        self.d.s("NAME", self.k.patient.first)

    def e_phon(self):
        self.people.append(self.k.patient)
        self.t("Ms. " if self.k.sex == "F" else "Mr. ")
        self.d.s("NAME", self.k.patient.last)

    def e_dob(self):
        self.date(self.k.dob, self.k.dob_style)

    def e_sex(self):
        self.t(self.k.sex)

    def age_span(self):
        k = self.k
        if k.age > 89 and self.phi:
            self.d.s("AGE", str(k.age))
        else:
            self.t(str(k.age))

    def e_agesex(self):
        k = self.k
        if k.age > 89 and not self.phi:
            self.t(k.sex)
            return
        st = k.agestyle
        if st == 0:
            self.age_span()
            self.t("Y " + k.sex)
        elif st == 1:
            self.age_span()
            self.t(" y/o " + k.sex)
        elif st == 2:
            self.t(k.sex + ", ")
            self.age_span()
            self.t("y")
        else:
            self.age_span()
            self.t(" years / " + k.sex)

    def e_agephrase(self):
        k = self.k
        word = ("boy" if k.sex == "M" else "girl") if k.age < 18 else ("man" if k.sex == "M" else "woman")
        self.age_span()
        self.t(f"-year-old {word}")

    def simple(self, label, value):
        self.d.s(label, value)

    def e_mrn(self): self.simple("MRN", self.k.mrn)
    def e_acct(self): self.simple("ACCOUNT", self.k.acct)
    def e_enc(self): self.simple("ID", self.k.enc)
    def e_member(self): self.simple("ID", self.k.member)
    def e_ssn(self): self.simple("SSN", self.k.ssn)
    def e_ordprov(self): self.person(self.k.ordp)
    def e_npi(self): self.simple("ID", self.k.npi)
    def e_ordfac(self): self.simple("FACILITY", self.k.ordfac)
    def e_ordaddr(self): self.simple("ADDRESS", self.k.ordaddr)
    def e_ordphone(self): self.simple("PHONE", self.k.ordphone)
    def e_ordfax(self): self.simple("PHONE", self.k.ordfax)
    def e_spec(self): self.simple("ID", self.k.spec)
    def e_acc(self): self.simple("ID", self.k.acc)
    def e_pathacc(self): self.simple("ID", self.k.pathacc)
    def e_cytacc(self): self.simple("ID", self.k.cytacc)
    def e_microacc(self): self.simple("ID", self.k.microacc)
    def e_order(self): self.simple("ID", self.k.order)
    def e_sendout(self): self.simple("ID", self.k.sendout)
    def e_outacc(self): self.simple("ID", self.k.outacc)
    def e_collected(self): self.stamp(self.k.dc, self.k.tc)
    def e_received(self): self.stamp(self.k.dr, self.k.tr)
    def e_reported(self): self.stamp(self.k.drep, self.k.trep)
    def e_printed(self): self.stamp(self.k.dprint, self.k.tprint)
    def e_cdate(self): self.date(self.k.dc)
    def e_rdate(self): self.date(self.k.dr)
    def e_repdate(self): self.date(self.k.drep)

    def e_lab(self):
        if self.phi:
            self.simple("FACILITY", self.k.lab)
        else:
            self.t(GENERIC_LABS[self.k.dc.toordinal() % len(GENERIC_LABS)])

    def e_labaddr(self): self.simple("ADDRESS", self.k.laddr)
    def e_labphone(self): self.simple("PHONE", self.k.lphone)
    def e_labfax(self): self.simple("PHONE", self.k.lfax)
    def e_clia(self): self.simple("ID", self.k.clia)
    def e_lic(self): self.simple("ID", self.k.lic)
    def e_director(self): self.person(self.k.director)
    def e_path(self): self.person(self.k.pathologist)
    def e_path2(self): self.person(self.k.pathologist2)
    def e_tech(self): self.simple("NAME", self.k.tech)
    def e_verified(self): self.simple("NAME", self.k.verified)
    def e_recvby(self): self.simple("NAME", self.k.recvby)
    def e_caller(self): self.simple("NAME", self.k.caller)
    def e_drawn(self): self.simple("NAME", self.k.drawn_by)

    def e_callee(self):
        c = self.k.callee
        self.people.append(c)
        self.t(self.k.callrole + " ")
        self.d.s("NAME", f"{c.first} {c.last}" if self.k.callee.form != 1 else c.last)

    def e_pphone(self): self.simple("PHONE", self.k.pphone)
    def e_paddr(self): self.simple("ADDRESS", self.k.paddr)
    def e_pcity(self): self.simple("LOCATION", self.k.pcity)
    def e_pzip(self): self.simple("LOCATION", self.k.pzip)
    def e_county(self): self.simple("LOCATION", self.k.county)
    def e_town(self): self.simple("LOCATION", self.k.town)
    def e_email(self): self.simple("EMAIL", self.k.email)
    def e_url(self): self.simple("URL", self.k.url)
    def e_code(self): self.simple("ID", self.k.code)
    def e_ip(self): self.simple("IP", self.k.ip)
    def e_outfac(self): self.simple("FACILITY", self.k.outfac)
    def e_outlab(self): self.simple("FACILITY", self.k.outlab)
    def e_outlab_addr(self): self.simple("ADDRESS", self.k.outlab_addr)
    def e_outlab_phone(self): self.simple("PHONE", self.k.outlab_phone)

    # -- shared content helpers (content rng only, so pairs match) -----------
    def draw(self, key):
        r = self.cr
        disp, unit, lo, hi, dec, loinc = AN[key]
        x = r.random()
        if lo is not None and hi is not None:
            if x < 0.68:
                v = r.uniform(lo, hi)
            elif x < 0.84:
                v = hi * r.uniform(1.02, 1.25)
            elif x < 0.95:
                v = lo * r.uniform(0.8, 0.97)
            elif x < 0.975:
                v = hi * r.uniform(1.5, 2.4)
            else:
                v = lo * r.uniform(0.35, 0.65)
            flag = "" if lo <= v <= hi else ("H" if v > hi else "L")
            if v > hi * 1.45:
                flag = "HH"
            elif v < lo * 0.7:
                flag = "LL"
        elif hi is not None:
            v = hi * r.uniform(0.35, 0.98) if x < 0.75 else hi * r.uniform(1.05, 1.9)
            flag = "" if v <= hi else "H"
        else:
            v = lo * r.uniform(1.02, 1.9) if x < 0.75 else lo * r.uniform(0.6, 0.97)
            flag = "" if v >= lo else "L"
        ref = f"{lo}-{hi}" if lo is not None and hi is not None else (f"<{hi}" if hi is not None else f">{lo}")
        return {"key": key, "name": disp, "val": f"{v:.{dec}f}", "unit": unit, "ref": ref, "flag": flag, "loinc": loinc,
                "crit": flag in ("HH", "LL")}

    def panel(self, name):
        return [self.draw(key) for key in PANELS[name]]

    def table(self, res, style):
        if style == "cols":
            head = ["Test", "Result", "Flag", "Units", "Reference"]
            rows = [[x["name"], x["val"], x["flag"], x["unit"], x["ref"]] for x in res]
            w = [max(len(r[i]) for r in [head] + rows) for i in range(5)]
            for row in [head, ["-" * w[0], "-" * w[1], "-" * w[2], "-" * w[3], "-" * w[4]]] + rows:
                self.t("  ".join(c.ljust(w[i]) for i, c in enumerate(row)).rstrip() + "\n")
        elif style == "inline":
            for x in res:
                self.t(f"{x['name']} {x['val']} {x['unit']}".rstrip() + f" ({x['ref']})" + (f" {x['flag']}" if x["flag"] else "") + "\n")
        elif style == "dots":
            for x in res:
                self.t(f"{x['name']:<22}{x['val']:>7} {x['flag']:<2} {x['unit']:<10} [{x['ref']}]".rstrip() + "\n")
        elif style == "loinc":
            for x in res:
                self.t(f"{x['name']} (LOINC {x['loinc']}): {x['val']} {x['unit']}".rstrip() + f"  Ref {x['ref']}" + (f" {x['flag']}" if x["flag"] else "") + "\n")
        elif style == "pipe":
            for x in res:
                self.t(f"{x['name']}|{x['val']}|{x['unit']}|{x['ref']}|{x['flag']}\n")
        else:  # compact
            chunk = []
            for x in res:
                chunk.append(f"{x['name']} {x['val']}" + (f" {x['flag']}" if x["flag"] else ""))
            for i in range(0, len(chunk), 4):
                self.t(" | ".join(chunk[i:i + 4]) + "\n")

    def crit_note(self, res):
        crit = [x for x in res if x["crit"]]
        if not crit:
            return
        x = crit[0]
        self.tag("critical_callback")
        self.t(f"Critical value: {x['name']} {x['val']} {x['unit']}".rstrip() + " ")
        self.sent(["called to ", ("callee",), " by ", ("caller",), " at ", ("collected",), "; read-back verified.\n"],
                  alt="called to the unit and read back verified.\n")

    def decoy_line(self, n=1):
        r = self.cr
        pool = LOT_DECOYS + COUNT_DECOYS
        for _ in range(n):
            self.t(r.choice(pool) + "\n")
        self.tag("look_alike_numbers")

    def instrument_line(self, cpt=None):
        r = self.cr
        bits = [f"Analyzer: {r.choice(INSTRUMENTS)}", r.choice(LOT_DECOYS)]
        if cpt:
            bits.append(f"CPT {cpt}")
        self.t("  ".join(bits) + "\n")
        self.tag("non_phi_codes")

    def lab_block(self, style=0, director=True, license=True):
        if style == 0:
            self.line(("Performed at", "lab"), ("CLIA", "clia"), sep=", ")
            self.line((None, "labaddr"), ("Phone", "labphone"), sep=" | ")
            if director:
                self.line(("Laboratory Director", "director"))
        elif style == 1:
            self.line(("Performing laboratory", "lab"), sep="")
            self.line(("Address", "labaddr"))
            self.line(("CLIA #", "clia"), ("State licence", "lic") if license else ("Fax", "labfax"), sep="   ")
            if director:
                self.line(("Medical Director", "director"), ("Tel", "labphone"), sep="   ")
        else:
            self.line((None, "lab"), (None, "labaddr"), sep=" - ")
            self.line(("Tel", "labphone"), ("Fax", "labfax"), ("CLIA", "clia"), sep=" / ")
            if director:
                self.line(("Director", "director"))
        self.tag("lab_identifier")

    def footer(self):
        r = self.r
        opts = ["End of report.", "*** End of Report ***", "Results relate only to the specimen as received.",
                "Page 1 of 1", "Page 2 of 2", "Reference ranges are method specific.",
                "Report status: FINAL", "Interpretive comments are available on request."]
        self.t(r.choice(opts) + "\n")



# --------------------------------------------------------------------------
# Layouts. Style choices use c.r, content choices use c.cr, so a matched
# negative reproduces the same clinical content as its positive.
# --------------------------------------------------------------------------

LAYOUTS = collections.defaultdict(list)


def layout(family, name):
    def deco(fn):
        LAYOUTS[family].append((name, fn))
        return fn
    return deco


def demographics(c, style=0):
    """Patient header block in several surface styles."""
    if style == 0:
        c.line(("Patient", "patient"), ("DOB", "dob"), ("Age/Sex", "agesex"), sep="   ")
        c.line(("MRN", "mrn"), ("Encounter", "enc"), sep="   ")
    elif style == 1:
        c.line(("Name", "patient"))
        c.line(("DOB", "dob"), ("Age/Sex", "agesex"), ("MRN", "mrn"), sep=" | ")
    elif style == 2:
        c.line(("Patient name", "patient"))
        c.line(("Date of birth", "dob"))
        c.line(("Medical record no.", "mrn"), kvsep=" ")
        c.line(("Age", "agesex"))
    else:
        c.line(("PT", "patient"), ("MRN", "mrn"), ("DOB", "dob"), ("A/S", "agesex"), sep="  ", kvsep=" ")


def times_block(c, style=0):
    if style == 0:
        c.line(("Collected", "collected"), ("Received", "received"), sep="   ")
        c.line(("Reported", "reported"))
    elif style == 1:
        c.line(("Collection", "collected"))
        c.line(("Received", "received"))
        c.line(("Final report", "reported"))
    else:
        c.line(("Coll.", "collected"), ("Recd.", "received"), ("Rpt.", "reported"), sep=" | ", kvsep=" ")


def provider_block(c, style=0):
    if style == 0:
        c.line(("Ordering provider", "ordprov"), ("NPI", "npi"), sep="   ")
    elif style == 1:
        c.line(("Ordered by", "ordprov"))
        c.line(("NPI", "npi"), ("Phone", "ordphone"), sep="   ")
    else:
        c.line(("Requesting physician", "ordprov"), ("Facility", "ordfac"), sep=" | ")


@layout("chem", "chem_columns")
def chem_columns(c):
    pn = c.cr.choice(["cmp", "bmp", "cbcd", "lipid", "mix"])
    res = c.panel(pn)
    c.t(c.cr.choice(PANEL_NAMES[pn]) + "\n")
    demographics(c, c.r.choice([0, 1, 3]))
    provider_block(c, c.r.randrange(3))
    c.line(("Specimen ID", "spec"), ("Order", "order"), sep="   ")
    if c.r.random() < 0.3:
        c.sent(["Draw site: our ", ("town",), " clinic\n"], alt="")
    times_block(c, c.r.randrange(2))
    c.nl()
    c.table(res, "cols")
    c.nl()
    c.crit_note(res)
    c.instrument_line(CPT[pn])
    if c.r.random() < 0.5:
        c.decoy_line()
    c.lab_block(c.r.randrange(3))
    c.line(("Tech", "tech"), ("Verified by", "verified"), sep="   ")
    c.footer()


@layout("chem", "chem_pipe_header")
def chem_pipe_header(c):
    pn = c.cr.choice(["cbc", "cbcd", "bmp", "iron"])
    res = c.panel(pn)
    c.t(c.cr.choice(PANEL_NAMES[pn]) + " - FINAL\n")
    c.line(("Name", "patient"), ("DOB", "dob"), ("MRN", "mrn"), ("Spec", "spec"), sep=" | ", kvsep=" ")
    c.line(("Coll", "collected"), ("Rcvd", "received"), ("Rptd", "reported"), sep=" | ", kvsep=" ")
    c.line(("Ordering", "ordprov"), ("NPI", "npi"), ("Fax", "ordfax"), sep=" | ", kvsep=" ")
    c.nl()
    c.table(res, "inline")
    c.crit_note(res)
    c.t(c.cr.choice(EPONYM_NOTES) + "\n")
    c.instrument_line(CPT[pn])
    c.line(("Lab", "lab"), ("CLIA", "clia"), ("Tel", "labphone"), sep=" | ", kvsep=" ")
    c.line(("Tech", "tech"), ("Verified", "verified"), sep=" | ", kvsep=" ")


@layout("chem", "chem_coag_form")
def chem_coag_form(c):
    res = c.panel("coag")
    c.kvsep = " ..... "
    c.t(c.cr.choice(PANEL_NAMES["coag"]) + "\n")
    c.line(("Patient", "patient"))
    c.line(("Date of birth", "dob"))
    c.line(("MRN", "mrn"))
    c.line(("Specimen", "spec"))
    c.line(("Accession", "acc"))
    c.line(("Ordering MD", "ordprov"))
    c.line(("Collected", "collected"))
    c.line(("Reported", "reported"))
    c.kvsep = ": "
    c.t("Specimen type: citrated plasma, 3.2% sodium citrate\n")
    c.t("\n")
    c.table(res, "dots")
    c.t(f"Therapeutic INR range for warfarin: {c.cr.choice(['2.0-3.0', '2.5-3.5'])}\n")
    c.crit_note(res)
    c.decoy_line()
    c.lab_block(c.r.choice([0, 2]), director=False)


@layout("chem", "chem_loinc_prose")
def chem_loinc_prose(c):
    pn = c.cr.choice(["lipid", "thyroid", "mix", "iron"])
    res = c.panel(pn)
    c.sent(["Report for ", ("patient",), " (MRN ", ("mrn",), "), collected ", ("collected",), ", received ", ("received",), ".\n"],
           alt="Report for the specimen as received.\n")
    c.sent(["Ordered by ", ("ordprov",), ", ", ("ordfac",), ".\n"], alt="")
    c.t(c.cr.choice(["Fasting status: fasting for 12 hours.", "Fasting status: not fasting.", "Specimen: serum, gel tube."]) + "\n")
    if c.r.random() < 0.3:
        c.sent(["Collected at our ", ("town",), " clinic.\n"], alt="")
    c.nl()
    c.table(res, "loinc")
    c.nl()
    c.instrument_line(CPT[pn])
    c.sent(["Reviewed and released ", ("reported",), " by ", ("verified",), ".\n"], alt="Reviewed and released.\n")
    c.lab_block(1)


@layout("chem", "chem_lis_keyvalue")
def chem_lis_keyvalue(c):
    pn = c.cr.choice(["cmp", "cbc", "bmp", "coag"])
    res = c.panel(pn)
    c.line(("PT_NAME", "patient"), ("PT_DOB", "dob"), ("PT_MRN", "mrn"), sep=";", kvsep="=")
    c.line(("ACC", "acc"), ("SPEC", "spec"), ("ORD_PROV", "ordprov"), ("ORD_NPI", "npi"), sep=";", kvsep="=")
    c.line(("COLL", "collected"), ("RECV", "received"), ("RSLT", "reported"), sep=";", kvsep="=")
    c.line(("PERF_LAB", "lab"), ("CLIA", "clia"), sep=";", kvsep="=")
    c.t(f"PANEL={pn.upper()};CPT={CPT[pn]};STATUS=F\n")
    c.table(res, "pipe")
    c.crit_note(res)
    c.decoy_line()


@layout("chem", "chem_cumulative")
def chem_cumulative(c):
    pn = c.cr.choice(["bmp", "cbc", "mix"])
    keys = PANELS[pn]
    k = c.k
    days = [k.dc - dt.timedelta(days=c.cr.choice([4, 3, 6])), k.dc - dt.timedelta(days=c.cr.choice([1, 2])), k.dc]
    c.t("CUMULATIVE RESULTS\n")
    demographics(c, c.r.choice([0, 1, 2]))
    provider_block(c, 0)
    c.nl()
    c.t("Test".ljust(18))
    for i, d in enumerate(days):
        if c.phi:
            c.date(d)
            c.t("   ")
        else:
            c.t(f"Day {i + 1}".ljust(12))
    c.t("Reference\n")
    for key in keys:
        disp, unit, lo, hi, dec, loinc = AN[key]
        vals = [c.draw(key) for _ in range(3)]
        cells = "".join((v["val"] + (" " + v["flag"] if v["flag"] else "")).ljust(13) for v in vals)
        c.t(disp.ljust(18) + cells + vals[0]["ref"] + " " + unit + "\n")
    c.t("H = high, L = low, HH/LL = critical\n")
    c.line(("Performing laboratory", "lab"), ("CLIA", "clia"), sep=", ")


# ---- microbiology ---------------------------------------------------------

def susc_table(c, org):
    gram = "pos" if org.startswith(GRAM_POS) else "neg"
    drugs = c.cr.sample(ABX[gram], c.cr.randint(5, len(ABX[gram])))
    c.t("Antibiotic".ljust(32) + "MIC".ljust(10) + "Interpretation\n")
    for dname in drugs:
        interp = c.cr.choices(["S", "S", "S", "I", "R"], [4, 4, 4, 1, 2])[0]
        c.t(dname.ljust(32) + c.cr.choice(MIC).ljust(10) + interp + "\n")
    c.t("S = susceptible, I = intermediate, R = resistant. Breakpoints per CLSI M100.\n")


@layout("micro", "micro_urine_final")
def micro_urine_final(c):
    org = c.cr.choice(ORGANISMS)
    c.t("URINE CULTURE - FINAL REPORT\n")
    demographics(c, c.r.choice([0, 2]))
    provider_block(c, c.r.randrange(3))
    c.line(("Accession", "microacc"), sep="")
    times_block(c, 1)
    c.t("Source: " + c.cr.choice(["clean catch urine", "catheterized urine", "midstream urine"]) + "\n\n")
    c.t(f"Colony count: {c.cr.choice(['>100,000', '50,000-100,000', '>=10,000'])} CFU/mL\n")
    c.t(f"Organism: {org}\n")
    susc_table(c, org)
    c.t(c.cr.choice(EPONYM_NOTES) + "\n")
    c.lab_block(c.r.randrange(3))
    c.line(("Reviewed by", "verified"))


@layout("micro", "micro_blood_prelim")
def micro_blood_prelim(c):
    c.t("BLOOD CULTURE - PRELIMINARY\n")
    demographics(c, c.r.choice([1, 3]))
    c.line(("Accession", "microacc"), ("Ordering", "ordprov"), sep="   ")
    if c.r.random() < 0.5:
        c.line(("Residence", "pcity"))
    c.line(("Drawn", "collected"), ("Received", "received"), sep="   ")
    c.line(("Drawn by", "drawn"))
    c.nl()
    ttp = c.cr.choice(["12.6", "18.4", "21.0", "30.2"])
    c.t(f"Set 1 aerobic bottle: POSITIVE at {ttp} hours of incubation.\n")
    c.t("Set 1 anaerobic bottle: no growth to date.\n")
    c.t("Set 2 (2 bottles): no growth to date.\n")
    c.t("Gram stain: " + c.cr.choice(["Gram-positive cocci in clusters", "Gram-negative rods", "Gram-positive cocci in chains",
                                       "Yeast forms seen"]) + ".\n")
    c.tag("critical_callback")
    c.sent(["Critical value called to ", ("callee",), " by ", ("caller",), " on ", ("reported",), "; read-back verified.\n"],
           alt="Critical value called to the unit; read-back verified.\n")
    c.t("Identification and susceptibilities to follow. Final report pending.\n")
    c.decoy_line()
    c.lab_block(c.r.randrange(3), director=False)


@layout("micro", "micro_wound_susceptibility")
def micro_wound_susceptibility(c):
    o1, o2 = c.cr.sample(ORGANISMS, 2)
    c.t("WOUND CULTURE WITH SUSCEPTIBILITY\n")
    c.t("SPECIMEN INFORMATION\n")
    c.kvsep = ": "
    c.line(("Patient", "patient"), ("MRN", "mrn"), sep="   ")
    c.line(("Specimen ID", "spec"), ("Collected", "collected"), sep="   ")
    c.line(("Ordering provider", "ordprov"), ("Phone", "ordphone"), sep="   ")
    c.t("Source: " + c.cr.choice(["left foot ulcer swab", "surgical wound, abdomen", "right heel wound", "sacral wound"]) + "\n\n")
    c.t("Gram stain: moderate WBC, moderate Gram-negative rods and Gram-positive cocci.\n\n")
    for i, org in enumerate((o1, o2), 1):
        c.t(f"ORGANISM {i}: {org} ({c.cr.choice(['heavy', 'moderate', 'light'])} growth)\n")
        susc_table(c, org)
        c.nl()
    c.line(("Reported", "reported"), ("by", "verified"), sep=" ", kvsep=" ")
    c.lab_block(1)


@layout("micro", "micro_resp_pcr")
def micro_resp_pcr(c):
    c.t("RESPIRATORY PATHOGEN PCR AND AFB SMEAR\n")
    demographics(c, 0)
    provider_block(c, 1)
    c.line(("Specimen ID", "spec"), sep="")
    times_block(c, 2)
    c.nl()
    targets = ["Influenza A", "Influenza B", "SARS-CoV-2", "RSV", "Adenovirus", "Rhinovirus/Enterovirus", "Mycoplasma pneumoniae",
               "Bordetella pertussis", "Human metapneumovirus"]
    for tname in targets:
        c.t(f"{tname}: {c.cr.choices(['Not detected', 'DETECTED'], [8, 1])[0]}\n")
    c.t("AFB smear (Ziehl-Neelsen): " + c.cr.choice(["no acid-fast bacilli seen", "rare acid-fast bacilli seen"]) + "\n")
    c.t("M. tuberculosis complex PCR: not detected\n")
    c.t(f"Positive control cycle threshold {c.cr.randint(22, 31)}.{c.cr.randint(0, 9)}; internal control valid.\n")
    c.decoy_line()
    c.lab_block(0)


@layout("micro", "micro_stool_amended")
def micro_stool_amended(c):
    c.t("GI PATHOGEN PANEL / C. DIFFICILE - AMENDED REPORT\n")
    c.line(("Patient", "patient"), ("MRN", "mrn"), sep=" / ", kvsep=" ")
    c.line(("DOB", "dob"), ("Sex", "sex"), sep=" / ", kvsep=" ")
    c.line(("Accession", "microacc"))
    c.line(("County of residence", "county"))
    c.line(("Collected", "collected"), ("Reported", "reported"), sep="   ")
    c.line(("Ordering", "ordprov"), ("NPI", "npi"), sep="   ")
    c.nl()
    for tname in ["Campylobacter", "Salmonella", "Shigella/EIEC", "Shiga toxin 1/2", "Norovirus GI/GII", "Giardia lamblia",
                  "Cryptosporidium"]:
        c.t(f"{tname}: {c.cr.choices(['Not detected', 'Detected'], [9, 1])[0]}\n")
    c.t("C. difficile toxin B gene (PCR): " + c.cr.choice(["Negative", "Positive"]) + "\n")
    c.sent(["Amended ", ("repdate",), ": result corrected after specimen identity review; corrected by ", ("verified",), ". Prior report superseded.\n"],
           alt="Amended: result corrected after specimen identity review. Prior report superseded.\n")
    c.t(c.cr.choice(EPONYM_NOTES) + "\n")
    c.lab_block(2)


# ---- surgical pathology and cytology ----------------------------------------

SURG = [
    ("Skin, left forearm, shave biopsy", "Compound melanocytic nevus, benign. No atypia identified.",
     "Received in formalin is a tan skin shave measuring {a} x {b} x 0.2 cm. The specimen is bisected and entirely submitted in A1.",
     "Nests of nevus cells at the dermoepidermal junction and in the superficial dermis. No mitoses. Breslow depth not applicable."),
    ("Skin, upper back, punch biopsy", "Basal cell carcinoma, nodular type. Margins uninvolved.",
     "A {a} cm punch biopsy of pink-tan skin, bisected and submitted in A1.",
     "Nodules of basaloid cells with peripheral palisading and retraction clefts in the dermis."),
    ("Colon, hepatic flexure, polypectomy", "Tubular adenoma with low-grade dysplasia. No high-grade dysplasia or carcinoma.",
     "Received in formalin are multiple tan fragments aggregating {a} x {b} x 0.3 cm, entirely submitted in A1.",
     "Crowded elongated glands lined by pencillate hyperchromatic nuclei with low-grade dysplasia."),
    ("Stomach, antrum, biopsy", "Chronic gastritis, mild activity. Warthin-Starry stain negative for organisms.",
     "Two tan tissue fragments, {a} and {b} cm, submitted in A1.",
     "Lamina propria with a mild lymphoplasmacytic infiltrate and focal neutrophilic activity."),
    ("Prostate, left base, needle core biopsies", "Adenocarcinoma, Gleason score 3+4=7 (grade group 2), involving 12/15 cores.",
     "Received are {a} cores of tan tissue, each {b} cm or less, submitted in A1 to A3.",
     "Infiltrating small glands with perineural invasion. Tumor involves 12/15 cores."),
    ("Breast, right, core needle biopsy", "Invasive ductal carcinoma, Nottingham grade 2 (3+2+1=6). ER positive.",
     "Received are three cores of fibrofatty tissue up to {a} cm, entirely submitted in A1.",
     "Cords and nests of atypical cells infiltrating fibrous stroma; tubule formation present."),
    ("Endometrium, curettage", "Proliferative endometrium. No hyperplasia or malignancy.",
     "Received is {a} mL of tan blood-tinged tissue, filtered and submitted in A1.",
     "Proliferative-phase glands and stroma with scattered mitoses."),
    ("Appendix, appendectomy", "Acute appendicitis with periappendicitis. No neoplasm identified.",
     "An appendix {a} cm in length and {b} cm in diameter with a dull serosa; the lumen contains fecal material.",
     "Transmural neutrophilic infiltrate with serosal fibrinopurulent exudate."),
    ("Lymph node, left axilla, excision", "Reactive follicular hyperplasia. No Reed-Sternberg cells identified.",
     "A lymph node {a} x {b} x 0.8 cm, bisected and submitted in A1.",
     "Enlarged follicles with polarised germinal centers and preserved architecture."),
]
CYTO = [
    ("Thyroid, right lobe, fine needle aspiration", "Benign follicular nodule (Bethesda category II).",
     "Adequacy: satisfactory. Moderately cellular specimen with colloid and benign follicular cells."),
    ("Cervix, liquid-based Pap test", "Negative for intraepithelial lesion or malignancy (NILM), reported per the Bethesda System.",
     "Adequacy: satisfactory for evaluation; transformation zone component present."),
    ("Urine, voided, cytology", "Negative for high-grade urothelial carcinoma (Paris System).",
     "Adequacy: satisfactory. Few benign urothelial cells and squamous cells."),
    ("Pleural fluid, thoracentesis", "Reactive mesothelial cells and lymphocytes. Negative for malignancy.",
     "Cell block prepared; sufficient cellularity."),
]
CLIN_HIST = ["Rule out malignancy.", "Changing lesion.", "Screening.", "Abnormal imaging.", "Follow-up of prior biopsy.",
             "Dyspepsia.", "Elevated PSA.", "Palpable mass."]


def path_header(c, acc_kind, style):
    if style == 0:
        c.line(("Patient", "patient"), ("DOB", "dob"), ("Age/Sex", "agesex"), sep="   ")
        c.line(("MRN", "mrn"), (acc_kind[1], acc_kind[0]), sep="   ")
        c.line(("Ordering physician", "ordprov"), ("NPI", "npi"), sep="   ")
        c.line(("Collected", "collected"), ("Received", "received"), sep="   ")
    elif style == 1:
        c.line((acc_kind[1], acc_kind[0]))
        c.line(("Patient name", "patient"))
        c.line(("Date of birth", "dob"), ("MRN", "mrn"), sep=" | ")
        c.line(("Clinician", "ordprov"), ("Clinic", "ordfac"), sep=" | ")
        c.line(("Procedure date", "cdate"), ("Received", "rdate"), sep=" | ")
    else:
        c.line(("CASE", acc_kind[0]), ("PATIENT", "patient"), ("MRN", "mrn"), sep=" / ", kvsep=" ")
        c.line(("PHYSICIAN", "ordprov"), sep="", kvsep=" ")
        c.line(("COLLECTED", "collected"), ("REPORTED", "reported"), sep=" / ", kvsep=" ")


SITE_HIST = {"Skin": ["Changing pigmented lesion.", "Non-healing lesion.", "Rule out malignancy."],
             "Colon": ["Screening colonoscopy, polyp found.", "Surveillance, prior adenoma."],
             "Stomach": ["Dyspepsia.", "Epigastric pain, rule out H. pylori."],
             "Prostate": ["Elevated PSA.", "Abnormal digital rectal exam."],
             "Breast": ["Abnormal screening mammogram.", "Palpable mass."],
             "Endometrium": ["Postmenopausal bleeding.", "Abnormal uterine bleeding."],
             "Appendix": ["Right lower quadrant pain.", "Acute abdomen."],
             "Lymph": ["Enlarging lymph node.", "Lymphadenopathy, rule out lymphoma."],
             "Thyroid": ["Palpable nodule.", "Nodule on ultrasound."], "Cervix": ["Routine screening.", "Prior abnormal Pap."],
             "Urine": ["Hematuria.", "Surveillance, prior bladder tumor."], "Pleural": ["Recurrent effusion.", "Dyspnea, effusion on imaging."]}


def fits_sex(label, sex):
    head = label.split(",")[0]
    if head in ("Prostate",):
        return sex == "M"
    if head in ("Endometrium", "Cervix", "Breast"):
        return sex == "F"
    return True


def pick_site(c, pool):
    ok = [x for x in pool if fits_sex(x[0], c.k.sex)]
    return c.cr.choice(ok)


def history_line(c, site=None):
    pool = SITE_HIST.get((site or "").split(",")[0], CLIN_HIST)
    hist = c.cr.choice(pool)
    c.sent(["Clinical history: ", ("agephrase",), ". ", hist + "\n"], alt="Clinical history: " + hist + "\n")


def signout(c, style=0):
    if style == 0:
        c.sent(["Electronically signed out by ", ("path",), " on ", ("reported",), "\n"], alt="Electronically signed out.\n")
    elif style == 1:
        c.line(("Pathologist", "path"))
        c.line(("Report date", "repdate"))
    else:
        c.sent(["Interpreted by ", ("path",), ". Reviewed ", ("reported",), ".\n"], alt="Interpreted and reviewed.\n")


@layout("path", "path_surgical_classic")
def path_surgical_classic(c):
    site = pick_site(c, SURG)
    a, b = c.cr.choice(["0.8", "1.2", "0.5", "1.5", "2.1", "0.9"]), c.cr.choice(["0.4", "0.6", "1.0", "0.3", "1.4"])
    c.t("SURGICAL PATHOLOGY REPORT\n")
    path_header(c, ("pathacc", "Accession"), 0)
    c.nl()
    history_line(c, site[0])
    c.t(f"Specimen received: A. {site[0]}\n\n")
    c.t("FINAL DIAGNOSIS\n")
    c.t(f"A. {site[0]}:\n   {site[1]}\n\n")
    c.t("GROSS DESCRIPTION\n" + site[2].format(a=a, b=b) + "\n\n")
    c.t("MICROSCOPIC DESCRIPTION\n" + site[3] + "\n\n")
    c.t(c.cr.choice(EPONYM_NOTES) + "\n")
    signout(c, c.r.choice([0, 1]))
    c.lab_block(c.r.randrange(3), director=False)


@layout("path", "path_cytology")
def path_cytology(c):
    cy = pick_site(c, CYTO)
    c.t("CYTOLOGY REPORT\n")
    path_header(c, ("cytacc", "Cytology no."), c.r.choice([0, 1]))
    c.nl()
    c.t(f"Specimen: {cy[0]}\n")
    history_line(c, cy[0])
    c.t(f"ADEQUACY / DESCRIPTION\n{cy[2]}\n\n")
    c.t(f"INTERPRETATION\n{cy[1]}\n\n")
    c.line(("Screened by", "tech"), sep="")
    c.sent(["Pathologist review: ", ("path",), "\n"], alt="Pathologist review completed.\n")
    c.t(c.cr.choice(EPONYM_NOTES) + "\n")
    c.line(("Report date", "repdate"))
    c.lab_block(1)


@layout("path", "path_synoptic")
def path_synoptic(c):
    site = pick_site(c, [s for s in SURG if "carcinoma" in s[1].lower() or "adenoma" in s[1].lower()])
    c.t("SURGICAL PATHOLOGY - SYNOPTIC REPORT\n")
    path_header(c, ("pathacc", "Surgical no."), 2)
    c.nl()
    c.t(f"PROCEDURE: {site[0]}\n")
    c.t(f"TUMOR SIZE: {c.cr.choice(['1.2', '2.4', '0.9', '3.1'])} cm in greatest dimension\n")
    c.t("HISTOLOGIC TYPE: " + site[1].split(",")[0].split(".")[0] + "\n")
    c.t("MARGINS: " + c.cr.choice(["all margins uninvolved by tumor", "invasive tumor at inked margin", "closest margin 0.2 cm"]) + "\n")
    examined, involved = c.cr.randint(8, 18), c.cr.randint(0, 3)
    c.t(f"LYMPH NODES: examined {examined}; involved {involved}; ratio {involved}/{examined}\n")
    c.t(f"PATHOLOGIC STAGE: pT{c.cr.randint(1, 3)} pN{c.cr.randint(0, 1)}\n")
    c.t("Ki-67 index: " + str(c.cr.randint(4, 40)) + "%\n")
    c.t("Reported using the CAP cancer protocol checklist.\n")
    c.sent(["Pathologist: ", ("path",), "     Signed: ", ("reported",), "\n"], alt="Signed by the reporting pathologist.\n")
    c.line(("Laboratory", "lab"), ("CLIA", "clia"), sep=", ")


@layout("path", "path_consult_addendum")
def path_consult_addendum(c):
    site = pick_site(c, SURG)
    c.t("PATHOLOGY CONSULTATION REPORT\n")
    path_header(c, ("pathacc", "Our case"), 1)
    c.nl()
    c.sent(["Outside material: slides and block from ", ("outfac",), ", case ", ("outacc",), ".\n"],
           alt="Outside material: slides and block from the referring institution.\n")
    c.t(f"Outside diagnosis: {site[1].split('.')[0]}.\n")
    c.t(f"Material received: {c.cr.randint(2, 12)} H&E slides, {c.cr.randint(0, 8)} unstained slides, {c.cr.randint(1, 4)} block(s).\n")
    c.t(f"CONSULTATION DIAGNOSIS\n{site[0]}: {site[1]}\n\n")
    c.t("COMMENT\nSlides were reviewed; " + c.cr.choice(["findings are concordant with the outside interpretation.",
                                                          "the diagnosis is modified as above.",
                                                          "additional immunostains were performed and are concordant."]) + "\n")
    c.t(c.cr.choice(EPONYM_NOTES) + "\n")
    c.sent(["Case discussed in intradepartmental consultation with ", ("path2",), ".\n"], alt="Case discussed in intradepartmental consultation.\n")
    c.sent(["ADDENDUM ", ("repdate",), ": immunostains reviewed by ", ("path",), "; diagnosis unchanged.\n"],
           alt="ADDENDUM: immunostains reviewed; diagnosis unchanged.\n")
    c.lab_block(0, director=False)


@layout("path", "path_frozen_gross")
def path_frozen_gross(c):
    site = pick_site(c, SURG)
    c.t("INTRAOPERATIVE CONSULTATION AND GROSS ONLY\n")
    c.line(("Case", "pathacc"), ("Patient", "patient"), ("MRN", "mrn"), sep="   ", kvsep=" ")
    c.line(("Surgeon", "ordprov"), ("Procedure date", "cdate"), sep="   ")
    c.nl()
    c.t(f"FS1: {site[0]}\n")
    c.t(f"Frozen section diagnosis: {site[1].split('.')[0].split(',')[0]}; margin status to be confirmed on permanent sections.\n")
    c.sent(["Result called to the operating room, ", ("callee",), ", at ", ("collected",), ".\n"], alt="Result called to the operating room.\n")
    c.t("GROSS: " + site[2].format(a="1.1", b="0.7") + "\n")
    c.t(f"Frozen section turnaround {c.cr.randint(8, 25)} minutes; tissue size {c.cr.randint(5, 30) / 10:.1f} cm.\n")
    c.t(f"Cassettes: A1-A{c.cr.randint(2, 6)}; frozen section remnant retained.\n")
    c.t(c.cr.choice(LOT_DECOYS + EPONYM_NOTES) + "\n")
    c.line(("Gross dictated by", "tech"), ("Pathologist", "path"), sep="   ")
    c.lab_block(2, director=False)


# ---- point of care and send-outs ---------------------------------------------

@layout("poc", "poc_glucose_meter")
def poc_glucose_meter(c):
    c.t("POINT OF CARE TESTING RESULT\n")
    c.line(("Patient", "patient"), ("MRN", "mrn"), sep="   ")
    c.line(("DOB", "dob"), ("Age/Sex", "agesex"), ("Location", "ordfac"), sep="   ")
    c.line(("Date", "cdate"), sep="")
    c.t(f"Meter {c.cr.choice(INSTRUMENTS)}; strip {c.cr.choice(LOT_DECOYS)}\n")
    for _ in range(c.cr.randint(1, 3)):
        g = c.draw("glu")
        h, m = c.cr.randint(5, 21), c.cr.randint(0, 59)
        c.t(f"{h:02d}:{m:02d}  Glucose {g['val']} mg/dL {g['flag']}".rstrip() + "\n")
    c.t("Quality control: within acceptable range.\n")
    c.line(("Operator", "tech"), ("Reviewed by", "verified"), sep="   ")
    c.decoy_line()
    c.sent(["Results transmitted to the record of ", ("patient",), " (", ("mrn",), ").\n"], alt="Results transmitted to the record.\n")


@layout("poc", "poc_rapid_log")
def poc_rapid_log(c):
    c.t("POC RAPID TEST LOG ENTRY\n")
    c.line(("Patient", "patient"), ("DOB", "dob"), sep=" | ")
    c.line(("MRN", "mrn"), ("Visit", "enc"), ("Provider", "ordprov"), sep=" | ")
    c.line(("Collected", "collected"))
    tests = c.cr.sample([("Rapid strep A antigen", ["Negative", "Positive"]), ("Influenza A/B antigen", ["Negative", "Influenza A positive"]),
                         ("Urine hCG", ["Negative", "Positive"]), ("Urinalysis dipstick: specific gravity", ["1.010", "1.020", "1.030"]),
                         ("Hemoglobin A1c (POC)", ["5.4 %", "6.8 %", "8.1 %"]), ("INR (POC)", ["1.1", "2.4", "3.2"]),
                         ("Rapid COVID-19 antigen", ["Negative", "Positive"])], c.cr.randint(2, 4))
    for name, vals in tests:
        c.t(f"{name}: {c.cr.choice(vals)}\n")
    c.t("Internal control valid. Kit " + c.cr.choice(LOT_DECOYS).lower() + ".\n")
    c.line(("Performed by", "tech"))
    c.sent(["Provider notified: ", ("ordprov",), ".\n"], alt="Provider notified.\n")
    c.decoy_line()


@layout("poc", "poc_sendout_requisition")
def poc_sendout_requisition(c):
    c.t("REFERENCE LABORATORY REQUISITION\n")
    c.line(("Send-out ID", "sendout"), ("Order", "order"), sep="   ")
    c.line(("Ship to", "outlab"), sep="")
    c.line((None, "outlab_addr"), ("Tel", "outlab_phone"), sep=" | ")
    c.t("\nPATIENT\n")
    c.line(("Name", "patient"))
    c.line(("DOB", "dob"), ("MRN", "mrn"), sep="   ")
    c.line(("Age/Sex", "agesex"))
    if c.cr.random() < 0.6:
        c.line(("Address", "paddr"))
        c.line(("Phone", "pphone"))
    else:
        c.line(("City/State", "pcity"), ("ZIP", "pzip"), sep="   ")
    c.line(("Insurance member ID", "member"), ("Guarantor account", "acct"), sep="   ")
    if c.cr.random() < 0.35:
        c.line(("SSN", "ssn"))
    if c.phi:
        c.t("\nORDERING\n")
    c.line(("Provider", "ordprov"), ("NPI", "npi"), sep="   ")
    c.line(("Facility", "ordfac"))
    c.line(("Phone", "ordphone"), ("Fax", "ordfax"), sep="   ")
    c.t("\nSPECIMEN\n" if c.phi else "\n")
    c.t(f"Test: {c.cr.choice(['Methylmalonic acid', 'Heavy metals panel', 'Celiac disease panel', 'Cystic fibrosis genotyping', 'Vitamin D 25-OH', 'HLA-B27 typing'])}"
        f"   CPT {c.cr.choice(['83921', '83018', '86364', '81220', '82306', '81374'])}\n")
    c.t("Specimen type: " + c.cr.choice(["serum", "whole blood EDTA", "plasma", "random urine"]) + "\n")
    c.line(("Collected", "collected"))
    c.line(("Drawn by", "drawn"))
    c.decoy_line()


@layout("poc", "poc_sendout_receipt")
def poc_sendout_receipt(c):
    c.t("SEND-OUT TRACKING AND RECEIPT\n")
    c.line(("Send-out ID", "sendout"), ("Reference lab accession", "outacc"), sep="   ")
    c.line(("Patient", "patient"), ("DOB", "dob"), ("Patient ZIP", "pzip"), sep="   ")
    c.sent(["Specimen picked up by courier at ", ("collected",), " from ", ("ordfac",), ".\n"], alt="Specimen picked up by courier.\n")
    c.sent(["Received at ", ("outlab",), " ", ("received",), " by ", ("recvby",), ".\n"], alt="Received at the reference laboratory by courier hand-off.\n")
    c.t("Condition on receipt: " + c.cr.choice(["acceptable, ambient", "acceptable, refrigerated 4 C", "acceptable, frozen", "hemolyzed, recollect requested"]) + "\n")
    c.t(f"Expected turnaround: {c.cr.randint(2, 7)} business days; status: IN PROCESS\n")
    c.t(c.cr.choice(EPONYM_NOTES) + "\n")
    c.line(("Questions", "outlab_phone"), sep="   ", kvsep=" call ")
    c.decoy_line()


@layout("poc", "poc_referred_result")
def poc_referred_result(c):
    res = [c.draw(x) for x in c.cr.sample(["tsh", "ft4", "a1c", "ferr", "iron", "mg"], 3)]
    c.t("REFERRED TEST REPORT\n")
    c.sent(["Performed by ", ("outlab",), ", ", ("outlab_addr",), "\n"], alt="Performed by the reference laboratory.\n")
    c.line(("CLIA", "clia"), ("Original report", "outacc"), sep="   ")
    c.line(("Patient", "patient"), ("MRN", "mrn"), sep="   ")
    c.line(("Collected", "collected"), ("Reported", "reported"), sep="   ")
    c.nl()
    c.table(res, "dots")
    c.t("Method: " + c.cr.choice(["chemiluminescent immunoassay", "LC-MS/MS", "ICP-MS", "enzyme immunoassay"]) + "\n")
    c.sent(["Result forwarded by ", ("ordfac",), " on ", ("printed",), ".\n"], alt="Result forwarded to the ordering site.\n")
    c.decoy_line()


# ---- portal and fax copies ----------------------------------------------------

@layout("portal", "portal_release_message")
def portal_release_message(c):
    cr = c.cr
    c.t("Subject: " + cr.choice(["New lab results are available", "Your results have been released", "Lab results ready to view",
                                  "Results update from the laboratory"]) + "\n\n")
    greet = cr.choice(["Hello", "Hi", "Good morning", "Good afternoon"])
    c.sent([greet + " ", ("pfirst",), ",\n\n"], alt=greet + ",\n\n")
    what = cr.choice(["blood work", "lab results", "test results", "laboratory tests"])
    c.sent(["Your " + what + " from ", ("lab",), " collected on ", ("cdate",), " have been released to your portal.\n"],
           alt=f"Your {what} have been released to your portal.\n")
    tests = cr.sample(["Basic Metabolic Panel", "CBC with differential", "Lipid panel", "Hemoglobin A1c", "TSH", "Urinalysis",
                       "PT/INR", "Ferritin", "Vitamin D", "Comprehensive Metabolic Panel", "Urine culture"], cr.randint(1, 3))
    c.t("Included: " + ", ".join(tests) + ".\n")
    if cr.random() < 0.55:
        x = c.draw(cr.choice(["glu", "hgb", "a1c", "tsh", "ldl", "cr", "k", "plt"]))
        c.t(f"Highlight: {x['name']} {x['val']} {x['unit']} (ref {x['ref']})".replace("  ", " ") + (f" {x['flag']}" if x["flag"] else "") + "\n")
        c.tag("look_alike_numbers")
    c.sent(["You can view them at ", ("url",), "\n"], alt=cr.choice(["You can view them in the portal under Test Results.\n",
                                                                         "Open the Results tab in the patient portal to see them.\n"]))
    if c.r.random() < 0.3:
        c.sent(["Walk-in draws are available at our ", ("town",), " clinic.\n"], alt="")
    if cr.random() < 0.4:
        c.sent(["Your activation code is ", ("code",), " and expires in " + cr.choice(["24 hours", "48 hours", "7 days"]) + ".\n"], alt="")
    c.sent(["Your provider, ", ("ordprov",), ", will review the results and contact you if anything needs follow-up.\n"],
           alt=cr.choice(["Your provider will review the results and contact you if anything needs follow-up.\n",
                          "Your care team has been notified and will reach out if needed.\n"]))
    c.sent(["Questions? Call the lab at ", ("labphone",), " or write to ", ("email",), ".\n"],
           alt=cr.choice(["Questions? Call the lab during business hours.\n", "Questions? Reply through the portal messaging tool.\n"]))
    c.t("\n" + cr.choice(["This message was sent by the results release system.", "This is an automated notification.",
                           "Messages sent after hours are answered the next business day."]) + "\n")
    c.sent([("lab",), " Patient Portal\n"], alt=cr.choice(["Patient Portal\n", "Laboratory Patient Portal\n"]))


@layout("portal", "portal_fax_cover")
def portal_fax_cover(c):
    c.t("FAX TRANSMITTAL\n")
    c.line(("TO", "ordprov"), ("FAX", "ordfax"), sep="   ")
    c.line(("FROM", "lab"), ("FAX", "labfax"), sep="   ")
    c.line(("TIME SENT", "printed"), sep="")
    c.t(f"PAGES: {c.cr.randint(2, 9)} (including cover)   URGENT: {c.cr.choice(['NO', 'YES'])}\n")
    c.sent(["RE: ", ("patient",), "   DOB ", ("dob",), "   MRN ", ("mrn",), "\n"], alt="RE: laboratory results\n")
    c.line(("Specimen", "spec"), ("Collected", "collected"), sep="   ")
    c.t("\nResults attached.\n")
    if c.cr.random() < 0.6:
        c.decoy_line()
    c.t("CONFIDENTIALITY NOTICE: This facsimile contains confidential health information intended only for the addressee. "
        "If you received it in error, notify the sender by telephone and destroy all copies.\n")
    c.line(("Questions", "labphone"), sep="", kvsep="? Call ")


@layout("portal", "portal_legacy_print")
def portal_legacy_print(c):
    pn = c.cr.choice(["bmp", "cbc", "mix"])
    res = c.panel(pn)
    c.t("*** LABORATORY REPORT ***                    PAGE 1 OF 1\n")
    c.kvsep = "....."
    c.line(("PATIENT", "patient"))
    c.line(("MRN", "mrn"), ("DOB", "dob"), sep="  ", kvsep=".")
    c.line(("PHYSICIAN", "ordprov"))
    c.line(("SPECIMEN", "spec"))
    c.line(("COLLECTED", "collected"))
    c.line(("PRINTED", "printed"))
    c.kvsep = ": "
    c.t("\n")
    c.table(res, "dots")
    c.crit_note(res)
    c.sent(["PERFORMED AT ", ("lab",), ", ", ("labaddr",), "\n"], alt="PERFORMED AT THE HOSPITAL LABORATORY\n")
    c.line(("CLIA", "clia"), ("TECH", "tech"), sep="  ", kvsep=" ")
    c.t("*** END OF REPORT ***\n")


@layout("portal", "portal_email_notice")
def portal_email_notice(c):
    cr = c.cr
    c.line(("From", "email"))
    c.line(("Date", "cdate"))
    c.t("Subject: " + cr.choice(["Your test results are ready", "Test results now final", "Action: view your lab report"]) + "\n\n")
    c.sent(["Dear ", ("phon",), ",\n\n"], alt=cr.choice(["Dear patient,\n\n", "Dear member,\n\n"]))
    n = cr.randint(1, 4)
    c.sent(["The results of the tests collected on ", ("cdate",), " at ", ("ordfac",), " are now final.\n"],
           alt=f"The results of your {n} recent test{'s' if n > 1 else ''} are now final.\n")
    c.sent(["Sign in at ", ("url",), " and enter access code ", ("code",), ".\n"],
           alt=cr.choice(["Sign in to the portal to see them.\n", "Use your portal login to open the report.\n"]))
    c.sent(["For help call ", ("labphone",), ". Account number ", ("acct",), " is needed for billing questions.\n"],
           alt=cr.choice(["For help call the number on your bill.\n", "Billing questions go to the number on your statement.\n"]))
    if cr.random() < 0.6:
        c.sent(["Sent from ", ("ip",), "\n"], alt="")
    c.t(cr.choice(["Please do not reply to this message.", "Replies to this address are not monitored.",
                   "Delete this message after reading if you share this device."]) + "\n")
    if cr.random() < 0.5:
        pages = cr.randint(1, 6)
        c.t(f"Report contains {pages} page{'s' if pages > 1 else ''}; CPT {cr.choice(['80053', '85025', '84443', '83036'])}.\n")


@layout("portal", "portal_fax_stamp_summary")
def portal_fax_stamp_summary(c):
    pn = c.cr.choice(["cbc", "bmp", "lipid"])
    res = c.panel(pn)
    c.sent([("printed",), " FAX ", ("labfax",), " TO ", ("ordfax",), " P. 01/0" + str(c.cr.randint(2, 6)) + "\n"],
           alt="FAX P. 01/0" + str(c.cr.randint(2, 6)) + "\n")
    c.t("RESULT SUMMARY COPY\n")
    c.line(("Patient", "patient"), ("DOB", "dob"), ("Age/Sex", "agesex"), sep="   ")
    c.line(("MRN", "mrn"), ("Account", "acct"), sep="   ")
    c.line(("Specimen", "spec"), ("Collected", "collected"), sep="   ")
    c.line(("Attending", "ordprov"))
    c.nl()
    c.table(res, "compact")
    c.crit_note(res)
    c.decoy_line()
    c.sent(["Performed at ", ("lab",), ". Results reviewed by ", ("path",), ".\n"], alt="Results reviewed by the laboratory director.\n")



# --------------------------------------------------------------------------
# Corpus construction
# --------------------------------------------------------------------------

NAME_PARTICLES = {"de", "la", "van", "von", "al", "di", "ten"}
FAMILY_PHENOMENA = {
    "chem": ["lab_report_header", "result_table", "reference_ranges"],
    "micro": ["culture_report", "susceptibility_table"],
    "path": ["pathology_report", "pathologist_signout"],
    "poc": ["poc_or_sendout", "requisition_receipt"],
    "portal": ["portal_or_fax_copy", "patient_facing"],
}


def value_keys(spans):
    keys = set()
    for s in spans:
        keys.add(s["text"].casefold())
        if s["label"] == "NAME":
            for tok in s["text"].replace("-", " ").split():
                core = tok.strip(".,")
                if len(core) >= 3 and core.casefold() not in NAME_PARTICLES:
                    keys.add(core.casefold())
    return keys


def render_row(fn, case, phi, seed, case_style):
    ctx = Ctx(case, phi, seed)
    fn(ctx)
    text, spans = ctx.d.render(case_style)
    return ctx, text, spans


def build_rows(seed=SEED):
    STATS.clear()
    rows, usage, seen_texts = [], collections.Counter(), set()
    assign = random.Random(f"{seed}:assign")
    for fam, tfam, n in FAMILIES:
        negs = n // 3
        singles = n - 2 * negs
        fns = LAYOUTS[fam]
        for kind, count in (("pair", negs), ("single", singles)):
            cases = [CASE_PATTERN[i % len(CASE_PATTERN)] for i in range(count)]
            assign.shuffle(cases)
            offset = assign.randrange(len(fns))
            for i in range(count):
                layout_index = (i + offset) % len(fns)
                lname, fn = fns[layout_index]
                case_style = cases[i]
                ui = f"{kind[0]}{i:04d}"
                for attempt in range(400):
                    useed = f"{seed}:{fam}:{ui}:{attempt}"
                    case = build_case(useed, fam)
                    ctx, text, spans = render_row(fn, case, True, useed, case_style)
                    texts = [text]
                    neg_text = None
                    if kind == "pair":
                        nctx, neg_text, nspans = render_row(fn, case, False, useed, case_style)
                        if nspans:
                            raise AssertionError("negative rendering produced spans")
                        texts.append(neg_text)
                    if not spans:
                        STATS["positive_without_spans_rejects"] += 1
                        continue
                    if any(excluded_sequences(t) for t in texts):
                        STATS["exclusion_filter_rejects"] += 1
                        continue
                    if text in seen_texts or (neg_text is not None and neg_text in seen_texts):
                        STATS["duplicate_text_rejects"] += 1
                        continue
                    keys = value_keys(spans)
                    if any(usage[x] >= VALUE_CAP for x in keys):
                        STATS["value_cap_rejects"] += 1
                        continue
                    for x in keys:
                        usage[x] += 1
                    seen_texts.update(texts)
                    break
                else:
                    raise RuntimeError(f"could not build unit {fam} {ui}")
                patient_hash = canonical_identity_group(f"{case.patient.first} {case.patient.last}")
                groups = [patient_hash]
                for p in ctx.people:
                    h = canonical_identity_group(f"{p.first} {p.last}")
                    if h not in groups:
                        groups.append(h)
                pair_id = f"tk17-{kind}-{fam}-{ui}"
                base = {
                    "split": "train", "suite": SUITE, "annotation_policy": POLICY_VERSION, "cohort": "realistic",
                    "family": tfam, "template_family": tfam, "layout": lname, "case_style": case_style,
                    "identity_group": patient_hash, "identity_groups": groups, "pair_id": pair_id,
                }
                role = "paired_positive" if kind == "pair" else "unpaired_positive"
                phen = list(dict.fromkeys(FAMILY_PHENOMENA[fam] + ctx.phen + [role]))
                rows.append({"id": f"tk17-{fam}-{ui}-pos", "text": text, "spans": spans, "role": role,
                             "phenomena": phen, **base})
                if kind == "pair":
                    nph = list(dict.fromkeys(FAMILY_PHENOMENA[fam] + ["matched_negative", "no_phi_report", "look_alike_numbers"]))
                    rows.append({"id": f"tk17-{fam}-{ui}-neg", "text": neg_text, "spans": [], "role": "matched_negative",
                                 "phenomena": nph, **base})
    order = random.Random(f"{seed}:order")
    order.shuffle(rows)
    STATS["max_value_usage"] = max(usage.values()) if usage else 0
    STATS["distinct_annotated_values"] = len(usage)
    return rows, dict(STATS)


def to_marked(row):
    out, prev = [], 0
    for s in sorted(row["spans"], key=lambda s: s["start"]):
        out.append(row["text"][prev:s["start"]])
        out.append(f"[[{s['label']}|{row['text'][s['start']:s['end']]}]]")
        prev = s["end"]
    out.append(row["text"][prev:])
    return "".join(out)


def summarize(rows, rejects):
    fam = collections.Counter(r["family"] for r in rows)
    lay = collections.Counter(r["layout"] for r in rows)
    case = collections.Counter(r["case_style"] for r in rows)
    support = collections.Counter(s["label"] for r in rows for s in r["spans"])
    row_cov = collections.Counter(lab for r in rows for lab in {s["label"] for s in r["spans"]})
    negs = sum(1 for r in rows if not r["spans"])
    n = len(rows)
    decoy = re.compile(r"\blot\b|CPT \d|LOINC|Analyzer|12/15|NT-proBNP|Titer|555-01|0100-0199|Coombs|Gram|Ziehl|Wright|Reed-|Bethesda"
                       r"|Hashimoto|Gleason|Breslow|Epstein|Philadelphia|Hurthle|Warthin|West Nile|Lyme|Nottingham|Milan|Paris System"
                       r"|\(ref |Ref \d|\[[<>]?\d|\d-\d+ (?:mg|mmol|U/L|g/dL|%|fL|pg|x10)|Reference", re.I)
    decoy_rows = sum(1 for r in rows if decoy.search(r["text"]))
    usage = collections.Counter()
    for r in rows:
        usage.update(value_keys(r["spans"]))
    firsts = {s["text"] for r in rows for s in r["spans"] if s["label"] == "NAME" and " " not in s["text"]}
    return {
        "rows": n, "rows_per_family": dict(fam), "layouts": len(lay), "layout_counts": dict(sorted(lay.items())),
        "max_layout_share": max(lay.values()) / n, "case_mix": {k: v / n for k, v in case.items()},
        "case_counts": dict(case), "label_support": {lab: support.get(lab, 0) for lab in
                                                     ["NAME", "DATE", "ADDRESS", "PHONE", "EMAIL", "MRN", "SSN", "ACCOUNT", "ID",
                                                      "URL", "IP", "AGE", "LOCATION", "FACILITY"]},
        "rows_with_label": dict(row_cov), "phi_free_rows": negs, "phi_free_share": negs / n,
        "rejected_draws": rejects, "rows_with_decoy_marker": decoy_rows, "decoy_share": decoy_rows / n,
        "max_value_row_share": max(usage.values()) / n, "max_value_rows": max(usage.values()),
        "distinct_annotated_values": len(usage), "distinct_single_token_names": len(firsts),
        "rows_with_age_over_89": sum(1 for r in rows if any(s["label"] == "AGE" for s in r["spans"])),
    }


def write_outputs(rows, rejects, out_path, report_dir):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")
    out_path.write_bytes(payload)
    summary = summarize(rows, rejects)
    exclusion_file = ROOT / "data" / "places_exclusions.sha256.json"
    manifest = {
        "name": "tk17_train_lab_pathology", "split": "train", "role": "train", "annotation_policy": POLICY_VERSION,
        "independent_final": False, "rows": len(rows), "documents": len(rows), "sha256": hashlib.sha256(payload).hexdigest(),
        "label_support": summary["label_support"], "entity_support": summary["label_support"],
        "label_counts": summary["label_support"], "rows_per_family": summary["rows_per_family"],
        "layouts": summary["layout_counts"], "case_counts": summary["case_counts"], "case_mix": summary["case_mix"],
        "phi_free_rows": summary["phi_free_rows"], "rejected_draws": rejects, "seed": SEED, "value_cap_per_value": VALUE_CAP,
        "decisions": ["L1", "L2", "L3", "L4"], "generator": "scripts/takeover/author17_lab_pathology.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "exclusion_list_sha256": hashlib.sha256(exclusion_file.read_bytes()).hexdigest(),
        "note": "Synthetic development training bank; not an evaluation bank. Written without reading any evaluation, "
                "development, held-out, blind or sealed bank.",
    }
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "stats.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    pick = random.Random(f"{SEED}:spot").sample(rows, 100)
    lines = ["# Spot check sample", "",
             "100 rows drawn at random (seed fixed) from `data/tk17_train_lab_pathology.jsonl`. Gold spans are marked "
             "`[[LABEL|text]]`. Case and layout shown per row.", ""]
    for r in pick:
        lines += [f"## {r['id']}  ({r['family']}, layout `{r['layout']}`, {r['case_style']}, {r['role']})", "", "```", to_marked(r), "```", ""]
    (report_dir / "spot_check_sample.md").write_text("\n".join(lines))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CORPUS))
    ap.add_argument("--report-dir", default=str(REPORT_DIR))
    args = ap.parse_args()
    rows, rejects = build_rows()
    manifest = write_outputs(rows, rejects, Path(args.out), Path(args.report_dir))
    print(json.dumps({k: manifest[k] for k in ("rows", "sha256", "label_support", "rows_per_family", "case_counts", "rejected_draws")},
                     indent=1))


if __name__ == "__main__":
    main()
