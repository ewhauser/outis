"""Author the tk18 radiology training bank (about 1,500 synthetic reports).

Six report families (CT and MRI, plain films and ultrasound, mammography,
interventional notes, teleradiology and critical-result communication, DICOM
style headers and worklist exports), 24 layouts, matched PHI-free negatives
(about one third), decoys throughout, and a blind-bank exclusion filter on
every row. Everything is invented. Deterministic: the same seed and source
give the same bytes.

Conventions come from docs/annotation-policy-v2.md and the report fixtures in
src/clinical_deid/data/annotation_policy_reports.py. Briefly: accession numbers
are ID; MRN captions stay outside the span; a person known only by initials is
NAME; honorifics and credentials stay outside NAME; series and image numbers,
measurements, slice positions, device and scanner models, contrast doses and
lot numbers, BI-RADS and similar categories, CPT codes and anatomical eponyms
are not annotated; a date-time has the date as DATE and the clock time outside
(a weekday stays inside); comparison-study dates are DATE; facilities and named
units are FACILITY; ages only over 89 and only the number.

Usage:
  PYTHONPATH=src python scripts/takeover/author18_radiology.py            # write corpus
  PYTHONPATH=src python scripts/takeover/author18_radiology.py --check    # build, do not write
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
    EXCLUSIONS_SHA256,
    excluded_sequences,
    excluded_value,
)
from clinical_deid.types import ENTITY_TYPES  # noqa: E402

SEED = 18_2026
ROWS = 1500
OUT = ROOT / "data" / "tk18_train_radiology.jsonl"
VALUE_CAP = 7  # a value may appear in at most 7 rows (0.5% of 1500 = 7.5)

FAMILY_SHARES = (
    ("ct_mri", 0.25),
    ("plain_us", 0.20),
    ("mammo", 0.15),
    ("interventional", 0.10),
    ("tele_critical", 0.15),
    ("dicom_worklist", 0.15),
)
# normal / all-caps / lowercase weights per family; DICOM exports lean all-caps.
CASE_WEIGHTS = {
    "dicom_worklist": (0.30, 0.55, 0.15),
}
CASE_DEFAULT = (0.594, 0.256, 0.15)

MARK = re.compile(r"\[\[([A-Z]+)\|([^\[\]|]+)\]\]")


def M(label: str, value: str) -> str:
    return f"[[{label}|{value}]]"


# --------------------------------------------------------------------------
# value pools (all invented)
# --------------------------------------------------------------------------
FIRST_HAND = """Aldric Amara Anneke Anselm Ardith Aurelio Azra Bertil Briony Calloway Caspian Cordelia Corvin Dagny Daphne
Delphine Desmond Dorothea Eamon Edda Eldridge Elspeth Emeric Evander Fenella Fitzroy Florian Freya Gideon Gisela Godfrey
Greer Gunnar Halcyon Hartwell Hester Hollis Ignatius Imogen Ione Isidore Jessamy Jocasta Kaspar Kestrel Kieran Ladislav
Leopold Linnea Lorcan Lucasta Magnus Maren Marcellus Mirabel Modesto Morwenna Nerys Nikolai Octavia Odalys Orrin Ottoline
Peregrine Philippa Quentin Rafferty Reuben Rhiannon Rosalind Rufus Saoirse Sebastian Selwyn Sigrid Soledad Tamsin Tobias
Ulrike Valentin Vesper Wilhelmina Winslow Xenia Yannick Ysolde Zacharias Zelda Abelard Adaeze Akira Amadou Anouk Arvid
Beatrix Bronagh Cosima Cyprian Dmitri Ebele Elodie Emrys Esperanza Farid Fenwick Gwendal Haruka Ilse Jorund Kalani Kamran
Lazlo Linus Livia Lucien Mabyn Malika Matthias Meriel Nadezhda Naveen Niamh Oriel Osvaldo Padraig Pelagia Percival Petra
Radmila Renzo Roisin Ronan Sabine Salome Siegfried Sunniva Tadeo Tancred Teodor Thaddeus Thora Tiberius Ulises Una Vasco
Verity Viggo Wendeline Wolfram Yara Yusuf Zenobia Zoltan Aileen Bjorn Cormac Dalia Esme Gaspar Hanneke Idris Jovan Kasimir
Lowenna Mattias Neville Oona Pascale Quill Rowan Sorcha Torin Ursel Vidar Willem Yolanda Zephyr Ambrose Brigid Casimir Donal
Eilidh Fabrizio Gwenllian Hugo Isolde Jasper Katarina Leandro Marit Nuala Orsolya Pavel Rosalie Stellan Theodora Udo Vivian""".split()
FIRST_ACCENT = ["Zoë", "Séverin", "Renée", "Åsa", "Håkan", "Inés", "Mathéo", "Noémie", "Jörg", "Søren", "Dagmar", "Chloé"]
FIRST_ON = """Mar Tav Cor Isa Lev Nik Ora Pen Rho Sev Tam Ulr Vas Wil Yas Zan Ale Bri Cas Del Eli Fio Gwe Hen Ila Jos Kal Lio Mir
Nad Oli Pri Qui Rem Sam The Uma Ver Wen Xav""".split()
FIRST_END = "na len rin dor mia ton vie sa ric lyn wen an el ik ora ana as is ette omar".split()

LAST_ON = """Quen Mar Rav Ost Bren Tal Kes Vell Dor Fen Hal Ire Jor Lund Mor Nev Orl Pell Rin Sav Tor Ulm Wyn Yar Zel Ash Bex
Cor Dun Elm Fal Gor Hes Ing Jas Kor Lis Mel Nor Oak Pen Ros Sed Tam Vas Wex Alb Bar Cal Dar Ev Fro Gar Hol Ik Jen
Kal Lor Mak Nas Ol Pas Rad Sol Tre Ul Vik Wal Aber Bram Cad Drey Ede Fisk Garr Hay Isk Kes Lan Mord Nis Ott Pry Rask""".split()
LAST_END = """nell wick ston ford ley man ham by ner sen ova ski dottir berg mont cott ard well hurst ridge vik enko aris ini
oglu gaard hollow thorpe stead wyn lund holm dale combe worth""".split()
LAST_HAND = """Abernathy Achterberg Adeyemi Aldecott Ashgrave Balogun Bellweather Brightwater Calloway-Reyes Chakraborty
Dunmarrow Eastoft Fairweather Fitzgibbon Galbraith Hargreave Idowu Ingleby Jankowski Kettleborough Lindqvist Machado
Nwosu Oyelaran Pemberton Quillfeather Rosenkranz Szymanski Thistlewood Underhill Vandermeer Wojciechowski Yarborough
Zielinski Ballantyne Chevalier Delacroix Eriksdottir Fontaine Grimshaw Haverford Iversen Jablonski Kowalczyk Lombardi
Mbeki Nakagawa Okonkwo-Hale Papadakis Rinaldi Sandoval Tennyson Ustinov Valderrama Whitlock Xiong Yoshimura Zambrano""".split()
LAST_PREFIX = ["van der ", "de la ", "O’", "Mc", "von ", "al-", "Mac", "di "]

PLACE_A = """Alder Birchwell Calder Dunmore Eastlake Fairhollow Galloway Hartsfield Ironwood Juniper Kingsmoor Linden Marshwood
Northcliffe Oakhaven Pinecrest Quarry Redfern Stonebridge Thornbury Upland Valemont Westerleigh Yewdale Zephyr Amberly
Brookmere Copperfield Dovedale Elderbank Foxglove Glenhaven Heathrow Isleworth Jadewater Kestlow Lavender Millbrook
Nettlefield Orchard Primrose Ravenscar Sedgemoor Tamarack Umberfield Violet Willowmere Yarrow Ashcombe Bramblewood
Cinder Driftwood Emberly Fernhollow Greywater Harrowgate Inglenook Larchmont Mistral Nightingale Oriole Peppercorn
Quillon Rushmere Sandpiper Tidewater Vesper Whinmoor""".split()
PLACE_END = "ford ton field haven ville burg mont port dale wick stead ridge".split()
STATES = "AL AK AZ AR CA CO CT DE FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()
FAC_TYPES = [
    "Medical Center", "Regional Hospital", "Imaging Center", "Radiology Associates", "Community Hospital",
    "Memorial Hospital", "Diagnostic Imaging", "Breast Health Center", "Health Center", "Outpatient Imaging",
    "University Hospital", "Women's Imaging Center", "Specialty Hospital", "Medical Group",
]
UNIT_TYPES = ["Ward", "Pavilion", "Unit", "Tower", "Wing"]
STATION_TYPES = ["MRI Suite", "CT Suite", "Imaging Suite", "Mammography Suite", "Interventional Suite", "Ultrasound Suite"]
TELE_TYPES = ["Teleradiology Group", "Teleradiology Partners", "Remote Reading Services", "Virtual Radiology"]
STREET_A = """Aldous Brindle Cobbett Dellwood Everly Fallow Gantry Hobart Ivory Jessop Kimber Lowry Marden Norwood Orchard Pullman
Quarrel Rowntree Sorrel Tilden Ulster Verbena Wexley Yorke Ambrose Bishop Chantry Dunlin""".split()
STREET_T = ["Road", "Lane", "Avenue", "Boulevard", "Way", "Court", "Drive", "Street", "Terrace"]
AREA = "202 212 303 312 404 415 503 512 612 617 702 718 801 913 206 305 214 602 720 919".split()
EMAIL_ROLES = ["radiology", "imaging.records", "scheduling", "reading.room", "mammo.desk", "ir.desk", "pacs.support", "results"]
EMAIL_DOMAINS = ["example.org", "example.net", "example.com", "example", "test"]
SCANNERS = [
    "Aurelion 64", "Vantis 3T", "Corvane Revolve 128", "Helix Lumina 1.5T", "Meridia X7", "Torrance Edge 256",
    "Solace 1.5T Pro", "Quantis DR-300", "Veridian Flex 16", "Lumera Duo 3T", "Nimbus Portable DR", "Auric UltraSound E9",
    "Pelagos Tomo 4D", "Kestrion IR-8", "Falcon Sono 600",
]
CONTRAST = ["Iovexan 350", "Gadoprime 0.5 M", "Iohexalar 300", "Gadovita 1 M", "Visiprest 320"]
SOFTWARE = ["4.12.0", "VB20A", "syngo-like 7.3.1", "SW 2.8.14", "R5.9.2", "v11.4", "build 20631"]
CPT = ["71045", "71046", "71250", "71260", "72148", "72141", "74177", "74176", "70450", "70553", "73721", "73221",
       "74183", "72197", "76700", "76705", "76770", "76856", "76536", "93970", "77066", "77067", "76641", "76942",
       "36556", "47000", "32555", "50432", "37191", "62270", "73562", "73610", "73502", "72100"]
WORDLIKE = set("""AM PM AS IN ON OR TO IT IS IF AT BY DO GO HE ME MY NO OF SO UP US WE RN MD CT MR XR IR ER OB ID AND THE FOR ARE BUT NOT ALL ANY CAN HAD HER WAS ONE OUR OUT DAY GET HAS HIM HIS HOW MAN NEW NOW OLD SEE TWO WAY WHO BOY DID ITS LET PUT SAY SHE TOO USE MRI CPT DAP EBL ICU RAD TEC PET TAB HRL LAB WET DRY SIT HIT BIT FIT KIT PIT TIP TOP HOT""".split())
MD_CRED = ["MD", "DO", "MBBS", "MD, PhD", "MD MPH"]
RN_CRED = ["RN", "BSN RN", "RT(R)", "RDMS"]

EPONYMS = [
    "Morison pouch", "Kerley B lines", "Hampton hump", "Westermark sign", "Chilaiditi sign", "Hill-Sachs lesion",
    "Bankart lesion", "Schmorl node", "Tarlov cyst", "Baker cyst", "Rigler sign", "Murphy sign", "Bochdalek hernia",
    "Morgagni hernia", "Virchow-Robin spaces", "Rathke cleft cyst", "Paget disease", "Meckel diverticulum", "Nabothian cysts",
    "Chiari I", "Pouch of Douglas", "Waldeyer ring", "Riedel lobe", "Sylvian fissure", "circle of Willis",
    "Colles fracture", "Jones fracture", "Salter-Harris II", "Looser zones", "Cooper ligaments", "Glisson capsule",
]


def _cap(word: str) -> str:
    return word[:1].upper() + word[1:]


# --------------------------------------------------------------------------
# row context
# --------------------------------------------------------------------------
class Ctx:
    def __init__(self, rng: random.Random, use: Counter, phi: bool, family: str, layout: str):
        self.r = rng
        self.use = use
        self.phi = phi
        self.family = family
        self.layout = layout
        self.names_drawn: set[str] = set()
        self.identities: list[str] = []
        self.ph: set[str] = set()
        self.reject_reason = ""
        # one facility per row
        self.facility = self._facility()
        self.city, self.state = self._city(), rng.choice(STATES)
        self.address = self._address()
        self.phone_main = self.phone()
        self.fax = self.phone()
        self.datestyle = rng.choice(["slash", "slash", "iso", "long", "abbr", "dmy", "slash2"])
        self.timestyle = rng.choice(["24", "24", "ampm", "compact"])
        self.exam_date = self.rand_date()
        if phi:
            self.patient = self._patient()
        else:
            self.patient = None
        self.age = self._age()

    # ----- names -----
    def _pick_token(self, make) -> str:
        tok = make()
        for _ in range(25):
            key = tok.lower()
            if self.use[key] + (key in self.names_drawn) < VALUE_CAP - 1:
                break
            tok = make()
        self.names_drawn.add(tok.lower())
        return tok

    def first(self) -> str:
        r = self.r

        def make():
            x = r.random()
            if x < 0.45:
                return r.choice(FIRST_HAND)
            if x < 0.52:
                return r.choice(FIRST_ACCENT)
            return r.choice(FIRST_ON) + r.choice(FIRST_END)

        return self._pick_token(make)

    def last(self) -> str:
        r = self.r

        def make():
            x = r.random()
            if x < 0.15:
                return r.choice(LAST_HAND)
            if x < 0.19:
                p = r.choice(LAST_PREFIX)
                core = r.choice(LAST_ON) + r.choice(LAST_END)
                return p + (core if p.endswith(("’", "-")) or p in ("Mc", "Mac") else core)
            return r.choice(LAST_ON) + r.choice(LAST_END)

        name = self._pick_token(make)
        if r.random() < 0.06:  # hyphenated double surname
            name = name + "-" + self._pick_token(lambda: r.choice(LAST_ON) + r.choice(LAST_END))
        return name

    def full_name(self) -> str:
        """Invented full name whose identity hashes to the train partition."""
        r = self.r
        for _ in range(60):
            parts = [self.first()]
            if r.random() < 0.18:
                parts.append(r.choice("ABCDEFGHJKLMNPRSTW") + r.choice([".", ""]))
            parts.append(self.last())
            if r.random() < 0.04:
                parts.append(r.choice(["Jr.", "III"]))
            name = " ".join(parts)
            ident = canonical_identity_group(name)
            if identity_split(ident) == "train":
                self.identities.append(ident)
                return name
        raise RuntimeError("no train-partition name")

    def surname_only(self) -> str:
        return self.last()

    def initials(self) -> str:
        r = self.r
        n = r.choice([2, 3, 3, 3])
        while True:
            letters = "".join(r.choice("ABCDEFGHJKLMNPRSTVWY") for _ in range(n))
            if letters not in WORDLIKE and sum(ch in "AEIOUY" for ch in letters) <= 1:
                break
        style = r.choice(["plain", "plain", "dots"])
        self.ph.add("initials_only_name")
        return ".".join(letters) + "." if style == "dots" else letters

    def NI(self) -> str:
        return M("NAME", self.initials())

    def clin(self, prefix: str = "Dr. ", cred: bool | None = None) -> str:
        """A clinician: honorific and credential outside the span."""
        r = self.r
        name = self.full_name() if r.random() < 0.55 else self.surname_only()
        s = f"{prefix}{M('NAME', name)}" if prefix else M("NAME", name)
        if cred is None:
            cred = r.random() < 0.45
        if cred:
            s += ", " + r.choice(MD_CRED)
        return s

    def staff(self) -> str:
        """A non-physician staff member with credential, no honorific."""
        return f"{M('NAME', self.full_name())}, {self.r.choice(RN_CRED)}"

    def plain_name(self) -> str:
        return M("NAME", self.full_name())

    # ----- patient -----
    def _age(self) -> int:
        r = self.r
        if self.phi and r.random() < 0.065:
            return r.randint(90, 103)
        if self.family == "mammo":
            return r.randint(40, 88)
        return r.randint(19, 88)

    def _patient(self) -> dict:
        r = self.r
        name = self.full_name()
        self.patient_name = name
        sex = "F" if self.family == "mammo" else r.choice(["F", "M"])
        return {"name": name, "sex": sex, "mrn": self.mrn(), "acct": self.number(r.choice([8, 9, 10]))}

    def mrn(self) -> str:
        r = self.r
        style = r.randint(0, 4)
        if style == 0:
            return str(r.randint(1_000_000, 99_999_999))
        if style == 1:
            return r.choice(["QX", "MR", "HX", "PN", "RM"]) + "-" + self.number(7)
        if style == 2:
            return self.number(9)
        if style == 3:
            return r.choice("ABCDEFGHJKLMNPRSTUVWXYZ") + self.number(8)
        return self.number(6) + "-" + str(r.randint(0, 9))

    def number(self, n: int) -> str:
        return str(self.r.randint(10 ** (n - 1), 10**n - 1))

    def accession(self) -> str:
        r = self.r
        style = r.randint(0, 5)
        if style == 0:
            return "RA-" + self.number(7)
        if style == 1:
            return "ACC" + self.number(9)
        if style == 2:
            return r.choice(["CT", "MR", "US", "XR", "IR", "MG"]) + str(r.randint(24, 26)) + "-" + self.number(6)
        if style == 3:
            return "A-" + str(r.randint(24, 26)) + "-" + self.number(7)
        if style == 4:
            return r.choice(["RAD", "IMG", "EXM"]) + self.number(8)
        return self.number(10)

    def serial(self) -> str:
        r = self.r
        return r.choice(["SN", "SER", ""]) + ("-" if r.random() < 0.5 else "") + r.choice("KLMNPRTV") + self.number(7)

    def npi(self) -> str:
        return self.number(10)

    def lot(self) -> str:
        return self.number(4) + self.r.choice("ABCDEFGHJK")

    # ----- places and contact -----
    def _facility(self) -> str:
        r = self.r
        return f"{r.choice(PLACE_A)} {r.choice(FAC_TYPES)}"

    def other_facility(self) -> str:
        r = self.r
        return f"{r.choice(PLACE_A)} {r.choice(FAC_TYPES)}"

    def _city(self) -> str:
        r = self.r
        return r.choice(PLACE_A) + r.choice(PLACE_END)

    def city_state(self) -> str:
        return f"{self._city()}, {self.r.choice(STATES)}"

    def _address(self) -> str:
        r = self.r
        street = f"{r.randint(12, 9890)} {r.choice(STREET_A)} {r.choice(STREET_T)}"
        if r.random() < 0.3:
            street += f", Suite {r.randint(100, 640)}"
        return f"{street}, {self._city()}, {r.choice(STATES)} {r.randint(10000, 99950)}"

    def new_address(self) -> str:
        return self._address()

    def unit(self) -> str:
        return f"{self.r.choice(PLACE_A)} {self.r.choice(UNIT_TYPES)}"

    def station(self, modality: str | None = None) -> str:
        kind = {"CT": "CT Suite", "MR": "MRI Suite", "XR": "Imaging Suite", "US": "Ultrasound Suite", "MG": "Mammography Suite", "IR": "Interventional Suite"}.get(modality or "")
        return f"{self.r.choice(PLACE_A)} {kind or self.r.choice(STATION_TYPES)}"

    def tele_group(self) -> str:
        return f"{self.r.choice(PLACE_A)} {self.r.choice(TELE_TYPES)}"

    def phone(self) -> str:
        r = self.r
        a, line = r.choice(AREA), r.randint(100, 199)
        style = r.randint(0, 3)
        if style == 0:
            return f"({a}) 555-{line:04d}"
        if style == 1:
            return f"{a}-555-{line:04d}"
        if style == 2:
            return f"{a}.555.{line:04d}"
        return f"+1 ({a}) 555-{line:04d}"

    def phone_ext(self) -> str:
        r = self.r
        return self.phone() + r.choice([" ext. ", " x", ", extension "]) + str(r.randint(200, 9999))

    def email(self) -> str:
        r = self.r
        dom = r.choice(EMAIL_DOMAINS)
        slug = r.choice(PLACE_A).lower()
        host = f"{slug}.{dom}" if dom in ("example.org", "example.net", "example.com") else f"{slug}-imaging.{dom}"
        return f"{r.choice(EMAIL_ROLES)}@{host}"

    def url(self) -> str:
        r = self.r
        dom = r.choice(EMAIL_DOMAINS)
        slug = r.choice(PLACE_A).lower()
        host = f"{slug}.{dom}" if dom in ("example.org", "example.net", "example.com") else f"{slug}-pacs.{dom}"
        token = "".join(r.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(r.randint(6, 10)))
        path = r.choice(["/s/", "/viewer/", "/share/", "/r/", "/study?k="])
        scheme = r.choice(["https://", "https://", "https://"])
        return f"{scheme}{host}{path}{token}"

    def ip(self) -> str:
        r = self.r
        net = r.choice(["192.0.2.", "198.51.100.", "203.0.113."])
        return net + str(r.randint(2, 250))

    # ----- dates -----
    def rand_date(self, start: dt.date = dt.date(2019, 1, 1), end: dt.date = dt.date(2026, 9, 30)) -> dt.date:
        return start + dt.timedelta(days=self.r.randint(0, (end - start).days))

    def before(self, d: dt.date, lo: int = 20, hi: int = 900) -> dt.date:
        return d - dt.timedelta(days=self.r.randint(lo, hi))

    def after(self, d: dt.date, lo: int = 1, hi: int = 60) -> dt.date:
        return d + dt.timedelta(days=self.r.randint(lo, hi))

    def fmt(self, d: dt.date, style: str | None = None) -> str:
        style = style or self.datestyle
        if style == "slash":
            return d.strftime("%m/%d/%Y")
        if style == "slash2":
            return d.strftime("%m/%d/%y")
        if style == "iso":
            return d.isoformat()
        if style == "long":
            return f"{d.strftime('%B')} {d.day}, {d.year}"
        if style == "abbr":
            return f"{d.strftime('%b')} {d.day}, {d.year}"
        if style == "dmy":
            return f"{d.day} {d.strftime('%b')} {d.year}"
        if style == "weekday":
            return f"{d.strftime('%A')}, {d.strftime('%B')} {d.day}, {d.year}"
        if style == "weekday_abbr":
            return f"{d.strftime('%a')} {d.strftime('%m/%d/%Y')}"
        if style == "monthyear":
            return f"{d.strftime('%b')} {d.year}"
        if style == "monthyear_long":
            return f"{d.strftime('%B')} {d.year}"
        if style == "year":
            return str(d.year)
        raise ValueError(style)

    def D(self, d: dt.date, style: str | None = None) -> str:
        return M("DATE", self.fmt(d, style))

    def Dw(self, d: dt.date) -> str:
        """A date with the weekday inside the span."""
        self.ph.add("weekday_date")
        return M("DATE", self.fmt(d, self.r.choice(["weekday", "weekday_abbr"])))

    def time(self) -> str:
        r = self.r
        h, m = r.randint(0, 23), r.randint(0, 59)
        if self.timestyle == "ampm":
            hh = h % 12 or 12
            return f"{hh}:{m:02d} {'AM' if h < 12 else 'PM'}"
        if self.timestyle == "compact":
            return f"{h:02d}{m:02d}"
        return f"{h:02d}:{m:02d}"

    def dtm(self, d: dt.date, joiner: str | None = None) -> str:
        """Date as DATE, time left outside."""
        self.ph.add("date_in_timestamp")
        return f"{self.D(d)}{joiner if joiner is not None else self.r.choice([' ', ' ', ' at '])}{self.time()}"

    def dob(self) -> dt.date:
        d = self.exam_date
        return dt.date(d.year - self.age, self.r.randint(1, 12), self.r.randint(1, 28))

    def range_dates(self, d: dt.date) -> str:
        e = d + dt.timedelta(days=self.r.randint(2, 9))
        self.ph.add("date_range")
        return M("DATE", f"{self.fmt(d, 'slash')} to {self.fmt(e, 'slash')}")

    # ----- ages -----
    def agephrase(self, sex: str | None = None) -> str:
        r = self.r
        sex = sex or (self.patient["sex"] if self.patient else r.choice(["F", "M"]))
        word = {"F": "woman", "M": "man"}[sex]
        n = self.age
        if n < 18:
            word = {"F": "girl", "M": "boy"}[sex]
        if n >= 90:
            self.ph.add("age_over_89")
            style = r.randint(0, 3)
            if style == 0:
                return f"{M('AGE', str(n))}-year-old {word}"
            if style == 1:
                return f"{M('AGE', str(n))} y/o {sex}"
            if style == 2:
                return f"{M('AGE', str(n))}{sex}"
            return f"{M('AGE', str(n))} year old {word}"
        style = r.randint(0, 3)
        if style == 0:
            return f"{n}-year-old {word}"
        if style == 1:
            return f"{n} y/o {sex}"
        if style == 2:
            return f"{n}{sex}"
        return f"{n} year old {word}"

    def agefield(self) -> str:
        """Value for an 'Age:' field."""
        n = self.age
        if n >= 90:
            self.ph.add("age_over_89")
            return f"{M('AGE', str(n))}{self.r.choice(['', ' years', 'Y'])}"
        return f"{n}{self.r.choice(['', ' years', 'Y'])}"

    # ----- decoys -----
    def mm(self) -> str:
        r = self.r
        return r.choice([f"{r.randint(2, 19)} mm", f"{r.randint(11, 69)}.{r.randint(0, 9)} mm", f"{r.randint(1, 9)}.{r.randint(0, 9)} cm"])

    def cm2(self) -> str:
        r = self.r
        return f"{r.randint(1, 6)}.{r.randint(0, 9)} x {r.randint(1, 5)}.{r.randint(0, 9)} cm"

    def sliceref(self) -> str:
        r = self.r
        s, i = r.randint(2, 9), r.randint(14, 480)
        self.ph.add("series_image_numbers")
        st = r.randint(0, 6)
        if st == 0:
            return f"(series {s}, image {i})"
        if st == 1:
            return f"(Se {s} Im {i})"
        if st == 2:
            return f"(image {i} of {i + r.randint(5, 300)})"
        if st == 3:
            return f"(series {r.randint(100, 604)}, image {i})"
        if st == 4:
            return f"(coronal image {i})"
        if st == 5:
            return f"(slice position {r.choice(['-', ''])}{r.randint(2, 180)}.{r.randint(0, 9)} mm)"
        return f"(series {s}/image {i})"

    def codeline(self) -> str:
        self.ph.add("cpt_code")
        return f"CPT {self.r.choice(CPT)}"

    def scanner(self) -> str:
        self.ph.add("device_model")
        return self.r.choice(SCANNERS)

    def contrast_line(self) -> str:
        r = self.r
        self.ph.add("contrast_lot")
        return f"{r.choice(CONTRAST)} {r.randint(20, 120)} mL IV, lot {self.lot()}"

    def eponym(self) -> str:
        self.ph.add("eponym")
        return self.r.choice(EPONYMS)

    def honorific_surname(self) -> str:
        """Mr./Ms. plus the patient's surname; the honorific stays outside the NAME span."""
        parts = [t for t in self.patient["name"].split() if t not in ("Jr.", "III")]
        word = "Mr." if self.patient["sex"] == "M" else "Ms."
        return f"{word} {M('NAME', parts[-1])}"

    def fill(self, s: str) -> str:
        """Replace {mm}, {cm2}, {sl}, {n}, {hu}, {ml} placeholders with invented decoy values."""
        r = self.r
        makers = {
            "mm": self.mm, "cm2": self.cm2, "sl": self.sliceref,
            "n": lambda: str(r.randint(2, 9)), "hu": lambda: f"{r.randint(8, 80)} HU", "ml": lambda: f"{r.randint(5, 900)} mL",
        }
        return re.sub(r"\{(mm|cm2|sl|n|hu|ml)\}", lambda m: makers[m[1]](), s)


# --------------------------------------------------------------------------
# clinical content (invented prose; placeholders are filled by Ctx.fill)
# --------------------------------------------------------------------------
# (modality, exam name, cpt, region)
CTMR_EXAMS = [
    ("CT", "CT chest with IV contrast", "71260", "chest"),
    ("CT", "CT chest without contrast", "71250", "chest"),
    ("CT", "CT abdomen and pelvis with IV contrast", "74177", "abdomen"),
    ("CT", "CT abdomen and pelvis without contrast", "74176", "abdomen"),
    ("CT", "CT head without contrast", "70450", "head"),
    ("CT", "CT angiogram of the head and neck", "70496", "head"),
    ("CT", "CT cervical spine without contrast", "72125", "spine"),
    ("MR", "MRI brain with and without contrast", "70553", "brain"),
    ("MR", "MRI lumbar spine without contrast", "72148", "spine"),
    ("MR", "MRI cervical spine without contrast", "72141", "spine"),
    ("MR", "MRI right knee without contrast", "73721", "knee"),
    ("MR", "MRI left shoulder without contrast", "73221", "shoulder"),
    ("MR", "MRI abdomen with and without contrast", "74183", "abdomen_mr"),
    ("MR", "MRI pelvis with and without contrast", "72197", "pelvis"),
]
PLAIN_EXAMS = [
    ("XR", "portable chest, 1 view", "71045", "xr_chest"),
    ("XR", "chest, 2 views", "71046", "xr_chest"),
    ("XR", "abdomen, supine and upright", "74019", "xr_abd"),
    ("XR", "right knee, 3 views", "73562", "xr_knee"),
    ("XR", "left wrist, 3 views", "73110", "xr_wrist"),
    ("XR", "lumbar spine, 4 views", "72110", "xr_spine"),
    ("US", "ultrasound right upper quadrant", "76705", "us_ruq"),
    ("US", "ultrasound kidneys and bladder", "76770", "us_renal"),
    ("US", "ultrasound pelvis, transabdominal and transvaginal", "76856", "us_pelvis"),
    ("US", "ultrasound thyroid", "76536", "us_thyroid"),
    ("US", "venous duplex of the left lower extremity", "93970", "us_dvt"),
]

HIST = {
    "chest": ["cough and fever for {n} days", "follow-up of a pulmonary nodule", "shortness of breath, evaluate for embolism", "screening for lung cancer", "chest pain after a fall"],
    "abdomen": ["right lower quadrant pain", "abdominal pain with leukocytosis", "follow-up of a hepatic lesion", "weight loss and early satiety", "flank pain, evaluate for stone"],
    "abdomen_mr": ["cirrhosis surveillance", "characterize an indeterminate liver lesion", "elevated liver enzymes", "pancreatic cyst follow-up"],
    "head": ["headache after minor trauma", "sudden left-sided weakness", "dizziness and a fall", "confusion of {n} days"],
    "brain": ["new seizure", "known metastatic disease, restaging", "progressive memory loss", "demyelinating disease follow-up"],
    "spine": ["low back pain with radiculopathy", "neck pain after a motor vehicle collision", "numbness in both feet", "follow-up of a compression fracture"],
    "knee": ["twisting injury, locking sensation", "chronic medial knee pain", "effusion after a fall"],
    "shoulder": ["pain with overhead activity", "dislocation, reduced yesterday", "weakness in abduction"],
    "pelvis": ["pelvic mass on ultrasound", "staging of a cervical lesion", "elevated PSA, prostate evaluation"],
    "xr_chest": ["fever and cough", "line placement check", "post-procedure check", "dyspnea", "pre-operative evaluation"],
    "xr_abd": ["abdominal distension, evaluate for obstruction", "constipation and vomiting", "swallowed foreign body"],
    "xr_knee": ["fall onto the knee", "chronic pain and swelling", "post-operative follow-up"],
    "xr_wrist": ["fall onto an outstretched hand", "pain over the snuffbox"],
    "xr_spine": ["low back pain", "fall with back pain", "scoliosis follow-up"],
    "us_ruq": ["right upper quadrant pain after meals", "elevated bilirubin", "gallstones on prior imaging"],
    "us_renal": ["rising creatinine", "hematuria", "flank pain"],
    "us_pelvis": ["pelvic pain", "abnormal uterine bleeding", "follow-up of an ovarian cyst"],
    "us_thyroid": ["palpable neck nodule", "goiter follow-up", "abnormal thyroid function tests"],
    "us_dvt": ["calf swelling and pain", "post-operative leg swelling", "elevated D-dimer"],
}

FIND = {
    "chest": [
        "The lungs are clear apart from a {mm} nodule in the right upper lobe {sl}, unchanged in size.",
        "No pleural effusion or pneumothorax. No Kerley B lines.",
        "Heart size is normal. No pericardial effusion.",
        "Mild dependent atelectasis at both bases. No consolidation.",
        "Mediastinal and hilar lymph nodes are not enlarged; the largest subcarinal node measures {mm} in short axis.",
        "A {mm} ground-glass nodule in the left lower lobe {sl} is new.",
        "Emphysematous changes in the upper lobes. Thoracic aorta is normal in caliber.",
        "No filling defect in the main or lobar pulmonary arteries {sl}.",
        "Mild interstitial thickening at the bases without Kerley B lines or Hampton hump.",
    ],
    "abdomen": [
        "The liver is normal in size and attenuation. A {mm} hypodensity in segment 6 {sl} is too small to characterize and statistically benign.",
        "The gallbladder is distended without wall thickening or pericholecystic fluid.",
        "The appendix is dilated to {mm} with periappendiceal stranding {sl}.",
        "No hydronephrosis. A {mm} nonobstructing stone in the left lower pole.",
        "No free fluid, including the Morison pouch.",
        "The pancreas and spleen are unremarkable. Adrenal glands are normal.",
        "Small fat-containing umbilical hernia. No bowel obstruction.",
        "Sigmoid diverticulosis without diverticulitis. No Chilaiditi sign.",
        "A {cm2} simple-appearing cyst arises from the upper pole of the right kidney {sl}.",
    ],
    "abdomen_mr": [
        "A {cm2} arterially enhancing observation in segment 8 {sl} with washout, LI-RADS LR-4.",
        "No biliary ductal dilatation. The common duct measures {mm}.",
        "A {mm} T2 hyperintense lesion in segment 3 with peripheral nodular enhancement {sl}, consistent with a hemangioma.",
        "Cirrhotic morphology with splenomegaly. Patent portal vein.",
        "Pancreatic duct measures {mm}; a {mm} cystic lesion in the body is unchanged.",
        "No suspicious focus in the spleen or adrenal glands.",
    ],
    "head": [
        "No acute intracranial hemorrhage, mass effect or midline shift.",
        "Gray-white differentiation is preserved. No CT evidence of acute infarct.",
        "Mild periventricular hypodensity, nonspecific, likely chronic microvascular change.",
        "The ventricles and sulci are age-appropriate. Prominent Virchow-Robin spaces are noted.",
        "No acute fracture. The paranasal sinuses and mastoids are clear.",
        "The vessels of the circle of Willis opacify normally {sl}. No aneurysm greater than {mm}.",
        "A {mm} focus of calcification in the right basal ganglia {sl}, likely incidental.",
    ],
    "brain": [
        "No restricted diffusion. No abnormal enhancement {sl}.",
        "A {mm} enhancing lesion in the left frontal lobe {sl} with surrounding vasogenic edema.",
        "Scattered T2/FLAIR hyperintense foci in the periventricular white matter, unchanged.",
        "The pituitary gland is normal in size. A {mm} Rathke cleft cyst is incidental.",
        "Mild cerebellar tonsillar descent, {mm} below the foramen magnum.",
        "The ventricles are normal in size. No extra-axial collection.",
    ],
    "spine": [
        "At L4-L5 there is a broad-based disc bulge with moderate bilateral foraminal narrowing {sl}.",
        "A {mm} Tarlov cyst at the S2 level is incidental.",
        "Vertebral body heights are maintained. Mild chronic endplate change at L2 with a Schmorl node.",
        "No cord signal abnormality. The conus ends at L1.",
        "Moderate central canal stenosis at C5-C6, canal diameter {mm} {sl}.",
        "Facet arthropathy at L5-S1. No spondylolisthesis.",
        "Prevertebral soft tissues are normal. No acute fracture.",
    ],
    "knee": [
        "Complex tear of the posterior horn of the medial meniscus {sl}.",
        "The anterior and posterior cruciate ligaments are intact.",
        "Moderate joint effusion. A {cm2} Baker cyst extends into the popliteal fossa.",
        "Grade 3 chondral thinning in the medial femoral condyle.",
        "No fracture or bone contusion. Collateral ligaments are intact.",
    ],
    "shoulder": [
        "Full-thickness tear of the supraspinatus tendon with {mm} retraction {sl}.",
        "A Hill-Sachs lesion of the posterolateral humeral head.",
        "Moderate acromioclavicular joint arthrosis with inferior spurring.",
        "The labrum is intact. Small joint effusion.",
        "Biceps tendon is normally positioned within the groove.",
    ],
    "pelvis": [
        "A {cm2} T2 hypointense lesion in the peripheral zone {sl}, PI-RADS 4.",
        "The uterus is anteverted and measures {mm} in thickness. Several Nabothian cysts are present.",
        "Left ovarian cyst measuring {mm}, O-RADS MRI 2.",
        "No pelvic lymphadenopathy. No free fluid.",
        "The bladder wall is normal. Seminal vesicles are symmetric.",
    ],
    "xr_chest": [
        "The lungs are clear. No focal consolidation, effusion or pneumothorax.",
        "Endotracheal tube tip is {mm} above the carina. Enteric tube courses below the diaphragm.",
        "Right-sided central venous catheter terminates at the cavoatrial junction.",
        "Mild cardiomegaly with Kerley B lines and small bilateral effusions.",
        "Streaky left basilar opacity, atelectasis versus early pneumonia.",
        "Normal cardiomediastinal silhouette. Osseous structures are intact.",
    ],
    "xr_abd": [
        "Nonobstructive bowel gas pattern. No free air under the diaphragm; no Rigler sign.",
        "Mildly dilated loops of small bowel with air-fluid levels.",
        "Moderate stool burden in the colon. No radiopaque foreign body.",
        "A {mm} calcification projects over the right renal shadow.",
    ],
    "xr_knee": [
        "No acute fracture or dislocation. Small suprapatellar effusion.",
        "Mild medial compartment joint space narrowing with marginal osteophytes.",
        "Postsurgical changes of anterior cruciate ligament reconstruction; tunnels are in satisfactory position.",
        "Soft tissue swelling anterior to the patella.",
    ],
    "xr_wrist": [
        "Transverse fracture of the distal radius with mild dorsal angulation, a Colles fracture.",
        "No scaphoid fracture is identified. Carpal alignment is normal.",
        "Mild soft tissue swelling about the radial styloid.",
    ],
    "xr_spine": [
        "Mild dextroconvex curvature with Cobb angle of {n} degrees.",
        "Vertebral body heights are maintained. A Schmorl node at L1.",
        "Disc space narrowing at L4-L5 and L5-S1. No acute fracture.",
    ],
    "us_ruq": [
        "The liver is normal in echotexture and measures {mm}. No focal lesion.",
        "Multiple gallstones with a {mm} wall. Negative Murphy sign.",
        "The common bile duct measures {mm}. No intrahepatic ductal dilatation.",
        "The right kidney measures {n}.{n} cm without hydronephrosis.",
    ],
    "us_renal": [
        "Both kidneys are normal in size and echogenicity. No hydronephrosis.",
        "A {cm2} simple cyst in the left kidney. No solid mass.",
        "The bladder is partially distended; postvoid residual {ml}.",
        "A {mm} echogenic focus with shadowing in the right lower pole, consistent with a stone.",
    ],
    "us_pelvis": [
        "The uterus measures {mm} in length with a normal endometrial stripe of {mm}.",
        "A {cm2} right ovarian cyst with thin septation, O-RADS 2. Trace fluid in the pouch of Douglas.",
        "The left ovary is normal with arterial and venous flow.",
        "No free pelvic fluid.",
    ],
    "us_thyroid": [
        "A {cm2} solid hypoechoic nodule in the right lobe, ACR TI-RADS TR4, with a macrolobulated margin.",
        "The left lobe is normal. The isthmus measures {mm}.",
        "No suspicious cervical lymph nodes.",
        "A {mm} spongiform nodule in the left lobe, TR1.",
    ],
    "us_dvt": [
        "The common femoral, femoral and popliteal veins are compressible with normal flow.",
        "Noncompressible left popliteal vein with echogenic intraluminal thrombus.",
        "A {cm2} Baker cyst in the popliteal fossa.",
        "Calf veins are patent where visualized.",
    ],
}
IMPR = {
    "chest": ["No acute cardiopulmonary process.", "Stable {mm} right upper lobe nodule; Fleischner guidance applies, follow-up in 12 months.", "New {mm} ground-glass nodule, recommend follow-up CT in 3 months.", "Lung-RADS 2, continue annual screening.", "No pulmonary embolism."],
    "abdomen": ["Findings compatible with acute appendicitis.", "No acute abdominal or pelvic abnormality.", "Nonobstructing left renal calculus, no hydronephrosis.", "Uncomplicated sigmoid diverticulosis.", "Cholelithiasis without cholecystitis."],
    "abdomen_mr": ["LI-RADS LR-4 observation in segment 8; multidisciplinary review recommended.", "Benign hemangioma in segment 3.", "Stable pancreatic cystic lesion; recheck in 12 months.", "Cirrhosis with no suspicious observation."],
    "head": ["No acute intracranial abnormality.", "Chronic microvascular ischemic change.", "No large vessel occlusion or aneurysm.", "Incidental basal ganglia calcification."],
    "brain": ["Left frontal enhancing lesion consistent with metastasis; oncology review.", "No acute infarct or abnormal enhancement.", "Stable white matter disease.", "No change compared with the prior examination."],
    "spine": ["Moderate L4-L5 foraminal stenosis likely accounting for symptoms.", "Moderate canal stenosis at C5-C6 without cord signal change.", "No acute fracture.", "Incidental Tarlov cyst."],
    "knee": ["Complex medial meniscal tear.", "Intact cruciate ligaments.", "Moderate effusion with a Baker cyst."],
    "shoulder": ["Full-thickness supraspinatus tear with retraction.", "Hill-Sachs deformity without labral tear.", "Acromioclavicular arthrosis."],
    "pelvis": ["PI-RADS 4 lesion in the peripheral zone; targeted biopsy suggested.", "O-RADS MRI 2, benign-appearing left ovarian cyst.", "No pelvic lymphadenopathy."],
    "xr_chest": ["No acute cardiopulmonary disease.", "Lines and tubes in satisfactory position.", "Left basilar atelectasis versus pneumonia.", "Mild pulmonary edema with small effusions."],
    "xr_abd": ["Nonobstructive bowel gas pattern.", "Findings suggest small bowel obstruction.", "Moderate constipation.", "Possible right renal calculus."],
    "xr_knee": ["No acute osseous abnormality.", "Mild medial compartment osteoarthritis.", "Small joint effusion."],
    "xr_wrist": ["Acute distal radius fracture.", "No acute fracture; occult scaphoid fracture not excluded, repeat views in 10 days."],
    "xr_spine": ["Degenerative changes at L4-S1.", "Mild scoliosis, unchanged."],
    "us_ruq": ["Cholelithiasis without sonographic evidence of cholecystitis.", "Normal right upper quadrant ultrasound."],
    "us_renal": ["No hydronephrosis.", "Right renal calculus.", "Simple left renal cyst, Bosniak I."],
    "us_pelvis": ["Simple right ovarian cyst, O-RADS 2, no follow-up needed.", "Normal pelvic ultrasound."],
    "us_thyroid": ["TR4 right nodule; fine-needle aspiration recommended.", "TR1 left nodule, benign.", "Normal thyroid gland."],
    "us_dvt": ["Acute left popliteal deep vein thrombosis.", "No deep vein thrombosis.", "Baker cyst, no thrombosis."],
}
TECH_CT = [
    "Axial images were obtained from the {part} with multiplanar reformats. {contrast}.",
    "Helical acquisition, 2.5 mm slice thickness, {part}. {contrast}. CTDIvol @DOSE@ mGy.",
    "Low-dose protocol, {part}, kVp 120, DLP @DLP@ mGy-cm. {contrast}.",
]
TECH_MR = [
    "Multiplanar T1, T2 and STIR sequences of the {part} on a {tesla} scanner. {contrast}.",
    "Axial and sagittal T2, T1, DWI and post-contrast T1 fat-saturated sequences. {contrast}. Magnet: {tesla}.",
    "Standard protocol with 3 mm slices. {contrast}. SAR within limits.",
]
PARTS = {"chest": "chest", "abdomen": "abdomen and pelvis", "abdomen_mr": "abdomen", "head": "head", "brain": "brain", "spine": "spine", "knee": "right knee", "shoulder": "left shoulder", "pelvis": "pelvis"}

MAMMO_FIND = [
    "The breasts are heterogeneously dense, which may lower the sensitivity of mammography.",
    "There are scattered areas of fibroglandular density. No suspicious mass, calcification or architectural distortion.",
    "A {mm} oval circumscribed mass in the upper outer left breast at 2 o'clock, {mm} from the nipple, unchanged.",
    "New group of fine pleomorphic calcifications spanning {mm} in the right breast at 10 o'clock, middle depth.",
    "Benign-appearing vascular and dermal calcifications. No axillary adenopathy.",
    "Targeted ultrasound shows a {cm2} irregular hypoechoic mass at the left 2 o'clock position {mm} from the nipple.",
    "Stable post-biopsy marker clip in the right breast at 9 o'clock.",
    "Asymmetry in the left upper breast resolves on spot compression views.",
]
MAMMO_IMPR = [
    ("BI-RADS 1: negative. Routine annual screening is recommended.", False),
    ("BI-RADS 2: benign finding. Routine annual screening is recommended.", False),
    ("BI-RADS 0: incomplete, additional imaging evaluation needed. Recall for diagnostic views and ultrasound.", True),
    ("BI-RADS 3: probably benign. Short-interval follow-up in 6 months.", False),
    ("BI-RADS 4A: low suspicion for malignancy. Image-guided biopsy recommended.", True),
    ("BI-RADS 4B: moderate suspicion. Stereotactic biopsy recommended.", True),
    ("BI-RADS 5: highly suggestive of malignancy. Biopsy and surgical consultation.", True),
]
IR_PROCS = [
    ("ultrasound-guided right internal jugular tunneled catheter placement", "CPT 36558"),
    ("CT-guided liver biopsy", "CPT 47000"),
    ("fluoroscopic left PICC placement", "CPT 36569"),
    ("ultrasound-guided paracentesis", "CPT 49083"),
    ("ultrasound-guided right thoracentesis", "CPT 32555"),
    ("left percutaneous nephrostomy tube placement", "CPT 50432"),
    ("retrievable IVC filter placement", "CPT 37191"),
    ("fluoroscopically guided lumbar puncture", "CPT 62270"),
    ("subcutaneous port placement, right chest", "CPT 36561"),
    ("CT-guided lung nodule biopsy", "CPT 32408"),
    ("ultrasound-guided thyroid nodule fine-needle aspiration", "CPT 10005"),
    ("prostatic artery embolization", "CPT 37243"),
]
DEVICES = ["Corvane 5 Fr micropuncture set", "Tessera 14.5 Fr dual-lumen catheter", "Aquila 18 gauge core needle", "Marlin 8.5 Fr pigtail drain", "Pelion 6 Fr sheath", "Strata retrievable filter", "Hexa implantable port 8 Fr"]
COMPLICATIONS = ["No immediate complications.", "No complications. Estimated blood loss less than 5 mL.", "A small subcapsular hematoma was seen at the end of the procedure and was stable on repeat imaging.", "Transient vasovagal episode, resolved with positioning. No other complications.", "No pneumothorax on the post-procedure image."]
SEDATION = ["moderate sedation, fentanyl and midazolam titrated by the nurse", "local anesthesia with 1% lidocaine", "no sedation", "monitored anesthesia care by the anesthesia service"]
CRIT_BY_REGION = {
    "chest": ["acute pulmonary embolism in the right lower lobar artery", "large right pneumothorax"],
    "xr_chest": ["large pneumothorax", "malpositioned endotracheal tube in the right mainstem bronchus"],
    "abdomen": ["free intraperitoneal air", "acute appendicitis with perforation", "ruptured abdominal aortic aneurysm"],
    "abdomen_mr": ["acute cholangitis with ductal obstruction"],
    "xr_abd": ["free intraperitoneal air", "high-grade small bowel obstruction"],
    "head": ["intracranial hemorrhage with midline shift", "acute large vessel occlusion"],
    "brain": ["intracranial hemorrhage with midline shift", "acute infarct with mass effect"],
    "spine": ["acute cervical spine fracture with canal compromise", "epidural abscess with cord compression"],
    "us_dvt": ["acute left popliteal deep vein thrombosis"],
    "us_ruq": ["acute cholecystitis with gallbladder perforation"],
    "us_renal": ["obstructing stone with infected hydronephrosis"],
    "us_pelvis": ["ruptured ectopic pregnancy with free fluid"],
    "us_thyroid": ["expanding neck hematoma with airway compromise"],
    "knee": ["open fracture with vascular compromise"],
    "shoulder": ["posterior dislocation with neurovascular compromise"],
    "pelvis": ["pelvic abscess with free air"],
    "xr_knee": ["displaced fracture with vascular compromise"],
    "xr_wrist": ["displaced fracture with vascular compromise"],
    "xr_spine": ["unstable vertebral fracture"],
}
CRIT_FALLBACK = ["acute pulmonary embolism in the right lower lobar artery", "intracranial hemorrhage", "acute appendicitis with perforation"]
CRIT = [
    "acute pulmonary embolism in the right lower lobar artery", "large pneumothorax", "intracranial hemorrhage with midline shift",
    "free intraperitoneal air", "acute left popliteal deep vein thrombosis", "ruptured abdominal aortic aneurysm",
    "malpositioned endotracheal tube in the right mainstem bronchus", "acute appendicitis with perforation",
]


def sample(r: random.Random, items: list, k: int) -> list:
    return r.sample(items, min(k, len(items)))


def crit_for(c: Ctx, exam: tuple) -> str:
    return c.r.choice(CRIT_BY_REGION.get(exam[3], CRIT_FALLBACK))


def clin_history(c: Ctx, region: str) -> str:
    h = c.fill(c.r.choice(HIST[region]))
    return f"{c.agephrase()} with {h}."


def tech_text(c: Ctx, exam: tuple) -> str:
    r = c.r
    mod, name, _, region = exam
    if mod == "CT":
        con = c.contrast_line() if "with IV" in name or "angiogram" in name else "No contrast was administered"
        s = r.choice(TECH_CT)
    else:
        con = c.contrast_line() if "with and without" in name else "No contrast was administered"
        s = r.choice(TECH_MR)
    s = s.replace("{part}", PARTS.get(region, region)).replace("{contrast}", con).replace("{tesla}", c.scanner())
    s = s.replace("@DOSE@", f"{r.randint(8, 29)}.{r.randint(0, 9)}").replace("@DLP@", str(r.randint(220, 980)))
    return c.fill(s) + (f" Scanner: {c.scanner()}." if mod == "CT" and r.random() < 0.5 else "")


def comparison(c: Ctx, exam: tuple, n: int | None = None) -> str:
    """A comparison line. PHI rows carry study dates as DATE; negatives use relative wording."""
    r = c.r
    mod, name, _, region = exam
    short = {"CT": "CT", "MR": "MRI", "XR": "radiograph", "US": "ultrasound"}[mod]
    where = PARTS.get(region, region.replace("xr_", "").replace("us_", ""))
    if not c.phi:
        return r.choice([
            f"Comparison: prior {short} of the {where} {r.randint(2, 30)} months earlier.",
            "Comparison: none available.",
            f"Comparison: {short} {where}, 6 weeks prior.",
        ])
    n = n or r.choice([1, 1, 1, 2])
    parts = []
    for i in range(n):
        d = c.before(c.exam_date, 25 + 200 * i, 400 + 500 * i)
        if r.random() < 0.18:
            style = "year"
            c.ph.add("history_year")
        else:
            style = r.choice([None, None, "monthyear", "slash"])
        parts.append(f"{short} {where} {r.choice(['dated ', 'from ', 'of ', ''])}{c.D(d, style)}")
    c.ph.add("comparison_date")
    return "Comparison: " + r.choice(["; ", " and "]).join(parts) + "."


def find_sents(c: Ctx, region: str, k: int) -> list[str]:
    return [c.fill(s) for s in sample(c.r, FIND[region], k)]


def impr_sents(c: Ctx, region: str, k: int = 2) -> list[str]:
    return [c.fill(s) for s in sample(c.r, IMPR[region], k)]


# --------------------------------------------------------------------------
# shared layout pieces
# --------------------------------------------------------------------------
def j(*lines) -> str:
    return "\n".join(x for x in lines if x is not None)


def letterhead(c: Ctx, style: int = 0) -> list[str]:
    """Facility letterhead; PHI rows only (a facility name is annotated)."""
    if not c.phi:
        return ["RADIOLOGY REPORT" if style % 2 == 0 else "Department of Radiology"]
    lines = [M("FACILITY", c.facility)]
    if style % 3 == 0:
        lines.append("Department of Radiology")
    if style % 2 == 0:
        lines.append(M("ADDRESS", c.address))
    tel = f"Tel {M('PHONE', c.phone_main)}"
    if style % 3 != 2:
        tel += f"   Fax {M('PHONE', c.fax)}"
    lines.append(tel)
    return lines


def phdr(c: Ctx, style: int) -> list[str]:
    """Patient header lines. Empty for PHI-free rows."""
    if not c.phi:
        return []
    p = c.patient
    name, mrn = M("NAME", p["name"]), M("MRN", p["mrn"])
    dob = c.D(c.dob())
    if style == 0:
        return [f"Patient: {name}", f"MRN: {mrn}", f"DOB: {dob}   Age: {c.agefield()}   Sex: {p['sex']}", f"Accession: {M('ID', c.accession())}"]
    if style == 1:
        return [f"Name: {name}   MRN#: {mrn}   Acct: {M('ACCOUNT', p['acct'])}", f"DOB {dob}   {p['sex']}"]
    if style == 2:
        return [f"{name} ({p['sex']}, DOB {dob})", f"MRN {mrn}   Accession {M('ID', c.accession())}"]
    return [f"PT: {name}  MRN: {mrn}  DOB: {dob}", f"ENC: {M('ID', 'E' + c.number(9))}   ACC: {M('ID', c.accession())}"]


def ordering(c: Ctx, form: int = 0) -> str | None:
    if not c.phi:
        return "Ordering provider: on file" if form == 0 else None
    if form == 0:
        return f"Ordering provider: {c.clin()} (NPI {M('ID', c.npi())})"
    if form == 1:
        return f"Referred by {c.clin()}"
    return f"Ref. physician: {c.clin()}"


def exam_dt(c: Ctx, label: str = "Exam date") -> str | None:
    if not c.phi:
        return None
    if c.r.random() < 0.5:
        return f"{label}: {c.dtm(c.exam_date)}"
    return f"{label}: {c.D(c.exam_date)}"


def signoff(c: Ctx, style: int) -> list[str]:
    r = c.r
    if not c.phi:
        return [r.choice(["Final report.", "Electronically signed.", "Report status: final.", "Signed by the attending radiologist."])]
    d1 = c.after(c.exam_date, 0, 3)
    if style == 0:
        return [f"Electronically signed by: {c.clin()}   {c.dtm(d1)}"]
    if style == 1:
        d2 = c.after(d1, 0, 2)
        return [f"Dictated by: {c.clin()}", f"Transcribed by: {c.NI()}", f"D: {c.dtm(d1)}   T: {c.dtm(d2)}"]
    if style == 2:
        return [f"Read by {c.clin()}.   Technologist: {c.NI()}"]
    return [f"Final report authenticated by {c.clin(prefix='')} on {c.D(d1)} at {c.time()}"]


def numbered(items: list[str], r: random.Random) -> list[str]:
    style = r.choice(["1.", "1)", "-"])
    out = []
    for i, s in enumerate(items, 1):
        out.append(f"{i}. {s}" if style == "1." else f"{i}) {s}" if style == "1)" else f"- {s}")
    return out


def codes_line(c: Ctx, exam: tuple) -> str:
    c.ph.add("cpt_code")
    return f"CPT {exam[2]}"


# --------------------------------------------------------------------------
# CT and MRI layouts
# --------------------------------------------------------------------------
def L_ct_formal(c: Ctx) -> str:
    r = c.r
    exam = r.choice(CTMR_EXAMS)
    region = exam[3]
    head = letterhead(c, r.randint(0, 5))
    body = [
        f"EXAM: {exam[1]} ({codes_line(c, exam)})",
        exam_dt(c),
        ordering(c, 0),
        f"CLINICAL HISTORY: {clin_history(c, region)}",
        f"TECHNIQUE: {tech_text(c, exam)}",
        comparison(c, exam),
        "",
        "FINDINGS:",
        *find_sents(c, region, r.randint(3, 5)),
        "",
        "IMPRESSION:",
        *numbered(impr_sents(c, region, r.randint(1, 3)), r),
        "",
        *signoff(c, r.choice([0, 0, 3])),
    ]
    return j(*head, "", *phdr(c, 0), "", *body)


def L_ct_ris(c: Ctx) -> str:
    r = c.r
    exam = r.choice(CTMR_EXAMS)
    region = exam[3]
    ph = phdr(c, r.choice([1, 2, 3]))
    top = " | ".join(ph) if c.phi else f"Exam: {exam[1]}"
    lines = [
        top,
        f"Exam: {exam[1]}" if c.phi else None,
        exam_dt(c, "Performed"),
        f"Hx: {clin_history(c, region)}",
        comparison(c, exam),
        "Findings -",
        " ".join(find_sents(c, region, r.randint(3, 5))),
        "Impression -",
        *impr_sents(c, region, r.randint(1, 2)),
        f"Tech: {c.scanner()}, {c.contrast_line()}" if r.random() < 0.6 else None,
        *signoff(c, r.choice([1, 2])),
    ]
    return j(*lines)


def L_ct_narrative(c: Ctx) -> str:
    r = c.r
    exam = r.choice(CTMR_EXAMS)
    region = exam[3]
    sents = find_sents(c, region, r.randint(3, 4))
    if c.phi:
        p = c.patient
        intro = (f"{exam[1][0].upper()}{exam[1][1:]} of {M('NAME', p['name'])}, {c.agephrase()}, performed {c.D(c.exam_date)} at "
                 f"{M('FACILITY', c.facility)} in {M('LOCATION', c.city + ', ' + c.state)}. "
                 f"{ordering(c, 1)} for {c.fill(r.choice(HIST[region]))}. MRN {M('MRN', p['mrn'])}.")
        who = f"{c.clin(cred=False)}"
        out = f"Dictated by {who}, {r.choice(MD_CRED)}."
    else:
        intro = f"{exam[1][0].upper()}{exam[1][1:]} was performed for {c.fill(r.choice(HIST[region]))} in a {c.agephrase('F')}."
        out = "Dictated by the attending radiologist."
    cmp_ = comparison(c, exam)
    return j(intro, "", tech_text(c, exam), cmp_.replace("Comparison: ", "The study was compared with "), "", " ".join(sents), "",
             "In summary: " + " ".join(impr_sents(c, region, 2)) if r.random() < 0.5 else "Impression: " + " ".join(impr_sents(c, region, 2)), "", out)


def L_mri_seq(c: Ctx) -> str:
    r = c.r
    exam = r.choice([e for e in CTMR_EXAMS if e[0] == "MR"])
    region = exam[3]
    seqs = ["Sag T1", "Sag T2", "Axial T2", "Axial T1", "STIR", "DWI b=1000", "Axial FLAIR", "Coronal T2 FS", "Post-contrast T1 FS"]
    seq_lines = [f"  {s}: series {r.randint(2, 9)}, {r.randint(18, 60)} images, {r.choice(['3', '4', '5'])} mm" for s in sample(r, seqs, r.randint(3, 5))]
    c.ph.add("series_image_numbers")
    lines = []
    if c.phi:
        p = c.patient
        lines += [f"MR REPORT  {M('FACILITY', c.facility)}", f"Name: {M('NAME', p['name'])}    MRN: {M('MRN', p['mrn'])}", f"Study: {exam[1]}   Date: {c.D(c.exam_date)}   Accession: {M('ID', c.accession())}"]
    else:
        lines += ["MR REPORT", f"Study: {exam[1]}"]
    lines += [
        f"Indication: {clin_history(c, region)}",
        f"Magnet: {c.scanner()}; {c.contrast_line()}" if "with and without" in exam[1] else f"Magnet: {c.scanner()}",
        "Sequences:", *seq_lines,
        comparison(c, exam, 2),
        "Findings:", *["  " + s for s in find_sents(c, region, r.randint(3, 5))],
        "Impression:", *["  " + s for s in impr_sents(c, region, 2)],
    ]
    if c.phi:
        lines.append(f"Attending: {c.clin()}   Resident: {c.clin(cred=False)}   {c.dtm(c.after(c.exam_date, 0, 2))}")
    else:
        lines.append("Attending and resident review completed.")
    return j(*lines)


def L_ct_onc(c: Ctx) -> str:
    r = c.r
    exam = r.choice([e for e in CTMR_EXAMS if e[0] == "CT" and e[3] in ("chest", "abdomen")])
    lines = []
    if c.phi:
        lines += [f"ONCOLOGY RESTAGING  -  {M('FACILITY', c.facility)}", *phdr(c, r.choice([0, 2]))]
    else:
        lines += ["ONCOLOGY RESTAGING CT"]
    lines += [f"Exam: {exam[1]}", exam_dt(c)]
    if c.phi:
        priors = []
        d = c.exam_date
        for _ in range(r.randint(2, 3)):
            d = c.before(d, 40, 130)
            priors.append(f"  {exam[0]} {c.D(d)}")
        c.ph.add("comparison_date")
        lines += ["Prior studies:", *priors]
    else:
        lines += ["Prior studies: two earlier CT examinations at intervals of about 3 months."]
    organs = ["right hepatic lobe", "left upper lobe", "retroperitoneal node", "right adrenal gland", "pelvic sidewall node", "left lower lobe"]
    for i, org in enumerate(sample(r, organs, r.randint(2, 3)), 1):
        lines.append(f"Target {i}: {org}, {c.mm()} (prior {c.mm()}, baseline {c.mm()}) {c.sliceref()}")
    lines += [
        "Non-target lesions: stable. No new lesions.",
        c.fill("Incidental: " + c.r.choice(["Morison pouch free of fluid", "Bochdalek hernia", "Schmorl node", "Tarlov cyst", "Riedel lobe"]) + " unchanged."),
        "Impression: " + r.choice(["stable disease by RECIST 1.1.", "partial response.", "interval progression."]),
        f"Reviewed by {c.clin()}; tech {c.NI()}" if c.phi else "Reviewed by the oncologic imaging attending.",
    ]
    return j(*lines)


# --------------------------------------------------------------------------
# plain film and ultrasound layouts
# --------------------------------------------------------------------------
def pick_plain(c: Ctx, kind: str | None = None) -> tuple:
    pool = [e for e in PLAIN_EXAMS if kind is None or e[0] == kind]
    return c.r.choice(pool)


def L_xr_prelim(c: Ctx) -> str:
    r = c.r
    exam = pick_plain(c, "XR")
    region = exam[3]
    lines = []
    if c.phi:
        p = c.patient
        lines += [f"PRELIMINARY - {exam[0]} {exam[1].upper()}", f"{M('NAME', p['name'])}  MRN {M('MRN', p['mrn'])}  {c.agephrase()}",
                  f"Location: {M('FACILITY', c.unit() + ' at ' + c.facility)}" if r.random() < 0.5 else f"Location: {M('FACILITY', c.unit())}",
                  f"Performed {c.dtm(c.exam_date)}"]
    else:
        lines += [f"PRELIMINARY - {exam[0]} {exam[1].upper()}", clin_history(c, region)]
    lines += [f"Ind: {c.fill(r.choice(HIST[region]))}", f"Unit: {c.scanner()}; {c.sliceref()}", *find_sents(c, region, r.randint(1, 2)), "Impression: " + impr_sents(c, region, 1)[0]]
    if c.phi:
        lines.append(f"Prelim read by {c.clin(cred=False)}; final report to follow. Tech {c.NI()}.")
    else:
        lines.append("Preliminary read; final report to follow.")
    return j(*lines)


def L_xr_final(c: Ctx) -> str:
    r = c.r
    exam = pick_plain(c, "XR")
    region = exam[3]
    lines = [*letterhead(c, r.randint(0, 5)), ""]
    lines += phdr(c, r.choice([0, 1, 2, 3]))
    lines += [
        f"EXAM: XR {exam[1]}   {codes_line(c, exam)}",
        exam_dt(c),
        f"History: {clin_history(c, region)}",
        comparison(c, exam),
        f"Technique: {r.choice(['AP and lateral', 'PA and lateral', 'portable AP', 'standing views'])}; {c.scanner()}; {r.randint(2, 40)} mAs {r.randint(55, 125)} kVp.",
        "Findings:", *find_sents(c, region, r.randint(2, 3)),
        "Impression:", *numbered(impr_sents(c, region, 2), r),
        *signoff(c, r.choice([2, 0, 3])),
    ]
    return j(*lines)


MEAS_LINES = {
    "us_ruq": ["Liver: {cm}  Gallbladder wall: {mm}  CBD: {mm}", "Spleen: {cm}  Aorta: {mm}  Right kidney: {cm}"],
    "us_renal": ["Right kidney: {cm} x {cm}  Left kidney: {cm} x {cm}", "Bladder wall: {mm}  Prostate: {cm}"],
    "us_pelvis": ["Uterus: {cm} x {cm} x {cm}  Endometrium: {mm}", "Right ovary: {cm} x {cm}  Left ovary: {cm} x {cm}"],
    "us_thyroid": ["Right lobe: {cm}  Left lobe: {cm}  Isthmus: {mm}", "Dominant nodule: {cm} x {cm} x {cm}"],
    "us_dvt": ["Popliteal vein diameter: {mm}  Femoral vein diameter: {mm}", "Great saphenous vein: {mm}  Baker cyst: {cm} x {cm}"],
}


def meas_row(c: Ctx, region: str) -> str:
    r = c.r
    s = r.choice(MEAS_LINES[region])
    while "{cm}" in s:
        s = s.replace("{cm}", f"{r.randint(1, 15)}.{r.randint(0, 9)} cm", 1)
    while "{mm}" in s:
        s = s.replace("{mm}", f"{r.randint(2, 9)}.{r.randint(0, 9)} mm", 1)
    return s


def L_us_report(c: Ctx) -> str:
    r = c.r
    exam = pick_plain(c, "US")
    region = exam[3]
    lines = []
    if c.phi:
        lines += [f"{M('FACILITY', c.facility)} - Ultrasound", *phdr(c, r.choice([0, 1, 2])), exam_dt(c)]
    else:
        lines += ["ULTRASOUND REPORT"]
    lines += [
        f"Study: {exam[1]}   {codes_line(c, exam)}",
        f"Indication: {clin_history(c, region)}",
        f"Equipment: {c.scanner()}, 5-2 MHz curvilinear probe",
        "Measurements:", meas_row(c, region), meas_row(c, region),
        "Findings:", *find_sents(c, region, r.randint(2, 4)),
        "Impression:", *impr_sents(c, region, 1),
        f"Sonographer: {c.NI()}   Radiologist: {c.clin()}" if c.phi else "Sonographer and radiologist review on file.",
    ]
    return j(*lines)


def L_us_vasc(c: Ctx) -> str:
    r = c.r
    exam = r.choice([e for e in PLAIN_EXAMS if e[3] in ("us_dvt", "us_thyroid", "us_renal")])
    region = exam[3]
    lines = []
    if c.phi:
        p = c.patient
        lines += [f"Patient {M('NAME', p['name'])}, MRN {M('MRN', p['mrn'])}", f"Exam {exam[1]} on {c.D(c.exam_date)}", f"Referring: {c.clin()}"]
    else:
        lines += [f"Exam: {exam[1]}"]
    lines += [
        f"{c.agephrase()}; {c.fill(r.choice(HIST[region]))}.",
        *find_sents(c, region, r.randint(2, 3)),
        *impr_sents(c, region, 1),
    ]
    if c.phi and region == "us_dvt":
        lines.append(f"Positive result discussed with {c.clin()} at {c.time()} on {c.D(c.exam_date)}. Probe {c.scanner()}; {c.codeline()}.")
    else:
        lines.append(f"Wet read given to the sonographer. Probe {c.scanner()}; {c.codeline()}.")
    lines.append(f"Signed {c.clin(prefix='', cred=True)}" if c.phi else "Signed electronically.")
    return j(*lines)


def L_xr_short(c: Ctx) -> str:
    r = c.r
    exam = pick_plain(c, "XR")
    region = exam[3]
    if c.phi:
        p = c.patient
        head = f"pt {M('NAME', p['name'])} mrn {M('MRN', p['mrn'])} acc {M('ID', c.accession())} {exam[0].lower()} {exam[1]} {c.D(c.exam_date)}"
    else:
        head = f"{exam[0].lower()} {exam[1]}, {c.agephrase()}"
    lines = [head, f"hx {c.fill(r.choice(HIST[region]))}", *find_sents(c, region, r.randint(1, 2)), "imp: " + impr_sents(c, region, 1)[0], f"ref {c.sliceref()} {c.scanner()}"]
    if c.phi:
        lines.append(f"{c.clin(cred=False)} / {c.NI()}")
    return j(*lines)


# --------------------------------------------------------------------------
# mammography layouts
# --------------------------------------------------------------------------
def prior_mammos(c: Ctx) -> str:
    r = c.r
    if not c.phi:
        return r.choice(["Comparison: prior screening mammograms from the last 2 years.", "Comparison: none."])
    c.ph.add("comparison_date")
    ds = [c.before(c.exam_date, 300, 420), c.before(c.exam_date, 700, 800)][: r.choice([1, 2])]
    return "Comparison: " + r.choice([", ", "; "]).join(f"mammogram {c.D(d)}" for d in ds) + "."


def L_mg_screen(c: Ctx) -> str:
    r = c.r
    imp, recall = r.choice(MAMMO_IMPR[:4])
    lines = []
    if c.phi:
        lines += [*letterhead(c, r.randint(0, 5)), "", "SCREENING MAMMOGRAM, BILATERAL, WITH TOMOSYNTHESIS", *phdr(c, r.choice([0, 1, 2, 3])), exam_dt(c)]
    else:
        lines += ["SCREENING MAMMOGRAM, BILATERAL, WITH TOMOSYNTHESIS"]
    lines += [
        f"Technique: digital breast tomosynthesis, CC and MLO views, {c.scanner()}; CPT 77067.",
        f"History: {c.agephrase('F')}, screening, {r.choice(['no family history of breast cancer', 'mother with breast cancer', 'no new symptoms'])}.",
        prior_mammos(c),
        "Findings:", *[c.fill(s) for s in sample(r, MAMMO_FIND[:2] + MAMMO_FIND[4:], r.randint(2, 3))],
        f"Impression: {imp}",
    ]
    if c.phi:
        lines.append(f"Result letter sent to patient {c.D(c.after(c.exam_date, 1, 6))}. Copy to {c.clin()}.")
        lines += signoff(c, r.choice([0, 3]))
    else:
        lines += signoff(c, 0)
    return j(*lines)


def L_mg_diag(c: Ctx) -> str:
    r = c.r
    imp, recall = r.choice(MAMMO_IMPR[2:])
    lines = []
    if c.phi:
        p = c.patient
        lines += [f"{M('FACILITY', c.facility)} Breast Imaging", f"Patient: {M('NAME', p['name'])}   MRN: {M('MRN', p['mrn'])}   DOB: {c.D(c.dob())}",
                  f"Accession: {M('ID', c.accession())}   Diagnostic exam {c.D(c.exam_date)}"]
    else:
        lines += ["DIAGNOSTIC MAMMOGRAM AND TARGETED ULTRASOUND"]
    lines += [
        f"Reason: recall from screening for {r.choice(['an asymmetry', 'a new mass', 'calcifications'])}; {c.agephrase('F')}.",
        prior_mammos(c),
        "Findings:", *[c.fill(s) for s in sample(r, MAMMO_FIND[2:], r.randint(2, 4))],
        f"Impression: {imp}",
    ]
    if recall and c.phi:
        bd = c.after(c.exam_date, 3, 14)
        lines += [f"Biopsy scheduled for {c.Dw(bd)} at {c.time()} at {M('FACILITY', c.other_facility())}.",
                  f"To reschedule call {M('PHONE', c.phone_main)}. Patient to hold aspirin as directed.",
                  f"Result discussed with the patient by {c.clin(cred=False)}."]
    elif recall:
        lines.append("Biopsy scheduling to be arranged; patient counselled on the findings.")
    else:
        lines.append("Short-interval follow-up recommended.")
    lines += signoff(c, r.choice([0, 2]))
    return j(*lines)


def L_mg_letter(c: Ctx) -> str:
    r = c.r
    imp, recall = r.choice(MAMMO_IMPR)
    lines = []
    plain = {True: "We would like to take a second look at some pictures from your recent mammogram.", False: "Your mammogram did not show signs of cancer."}[recall and "BI-RADS 0" in imp or recall]
    if c.phi:
        p = c.patient
        last = p["name"].split()[-1] if not p["name"].endswith(("Jr.", "III")) else p["name"].split()[-2]
        lines += [M("FACILITY", c.facility), M("ADDRESS", c.address), "", c.D(c.after(c.exam_date, 2, 9)), "",
                  M("NAME", p["name"]), M("ADDRESS", c.new_address()), "",
                  f"Dear Ms. {M('NAME', last)},", "",
                  f"Thank you for having your mammogram with us on {c.D(c.exam_date)}. {plain}",
                  ]
        if recall:
            lines.append(f"Please call {M('PHONE', c.phone_ext())} to schedule a diagnostic visit within {r.choice(['two', 'three'])} weeks.")
        else:
            lines.append(f"Your next screening mammogram is due {c.D(c.after(c.exam_date, 340, 380), 'monthyear_long')}.")
        lines += [f"Reference: {M('ID', c.accession())}  (exam code CPT 77067; assessment {imp.split(':')[0]}).", "", "Sincerely,", f"Breast Imaging, {M('FACILITY', c.facility)}", f"Questions: {M('EMAIL', c.email())}"]
    else:
        opener = r.choice(["Dear patient,", "Dear patient:", "To our patient,", "Hello,"])
        thanks = r.choice(["Thank you for having your mammogram with us.", "Thank you for choosing us for your screening mammogram.",
                           "This letter reports the result of your recent breast imaging.", "We have reviewed your recent mammogram."])
        density = r.choice(["Your breast tissue is dense, which can make small findings harder to see.", "Your breast tissue is not dense.",
                            "Your breasts have scattered areas of fibroglandular tissue.", ""])
        contact = r.choice(["Please call the number on your appointment card if you have questions.", "Call the scheduling desk listed on your visit summary with any questions.",
                            "Your ordering clinician has a copy of this letter.", "Keep this letter with your health records."])
        lines += [opener, "", f"{thanks} {plain}", density or None, contact,
                  f"Assessment {imp.split(':')[0]}; exam code CPT 77067; {r.randint(2, 4)} views per breast; {c.scanner()}.", "", "Sincerely,", r.choice(["Breast Imaging", "The Breast Imaging team"])]
    return j(*lines)


# --------------------------------------------------------------------------
# interventional layouts
# --------------------------------------------------------------------------
def consent_line(c: Ctx) -> str:
    r = c.r
    if not c.phi:
        return r.choice(["Informed consent was obtained from the patient.", "Consent was obtained from the patient's representative."])
    if r.random() < 0.5:
        rel = r.choice(["daughter", "son", "wife", "husband", "sister"])
        return f"Informed consent was obtained from the patient's {rel}, {c.plain_name()}, and witnessed by {c.NI()}."
    return f"Informed consent was obtained from the patient and witnessed by {c.NI()}."


def device_lines(c: Ctx) -> list[str]:
    r = c.r
    out = []
    for dv in sample(r, DEVICES, r.randint(1, 2)):
        line = f"Device: {dv}, lot {c.lot()}"
        c.ph.add("contrast_lot")
        if c.phi and r.random() < 0.6:
            line += f", serial {M('ID', c.serial())}"
            c.ph.add("device_serial")
        out.append(line)
    return out


def L_ir_note(c: Ctx) -> str:
    r = c.r
    proc, cpt = r.choice(IR_PROCS)
    lines = [*letterhead(c, r.randint(0, 5)), "", "INTERVENTIONAL RADIOLOGY PROCEDURE NOTE"]
    lines += phdr(c, r.choice([0, 1, 3]))
    lines += [
        f"Date of procedure: {c.D(c.exam_date)}" if c.phi else None,
        f"Procedure: {proc} ({cpt})",
        f"Operator: {c.clin()}   Assistant: {c.clin(cred=False)}" if c.phi else "Operator: attending interventional radiologist",
        f"Referring service: {c.clin()}" if c.phi else None,
        consent_line(c),
        f"Sedation: {r.choice(SEDATION)}.",
        f"Pre-procedure diagnosis: {c.fill(r.choice(HIST['abdomen']))}.",
        f"Technique: time-out performed. Under {r.choice(['ultrasound', 'fluoroscopic', 'CT'])} guidance, access was obtained. Fluoroscopy time {r.randint(1, 18)}.{r.randint(0, 9)} min, air kerma {r.randint(8, 400)} mGy.",
        *device_lines(c),
        f"Specimen: {r.randint(2, 6)} passes sent to the laboratory." if r.random() < 0.4 else None,
        f"Complications: {r.choice(COMPLICATIONS)}",
        f"{c.honorific_surname()} tolerated the procedure well; {c.honorific_surname()}'s family was updated." if c.phi and r.random() < 0.35 else None,
        f"Disposition: returned to {M('FACILITY', c.unit()) if c.phi else 'the unit'} in stable condition.",
        *signoff(c, r.choice([0, 1])),
    ]
    return j(*lines)


def L_ir_log(c: Ctx) -> str:
    r = c.r
    proc, cpt = r.choice(IR_PROCS)
    rows = [
        ("PROCEDURE", f"{proc} [{cpt}]"),
        ("DATE", c.D(c.exam_date) if c.phi else None),
        ("PATIENT", f"{M('NAME', c.patient['name'])} (MRN {M('MRN', c.patient['mrn'])})" if c.phi else None),
        ("AGE/SEX", f"{c.agefield()} / {c.patient['sex']}" if c.phi else c.agephrase('F')),
        ("OPERATOR", c.clin() if c.phi else "attending"),
        ("NURSE", c.staff() if c.phi else None),
        ("TECH", c.NI() if c.phi else None),
        ("SEDATION", r.choice(SEDATION)),
        ("ACCESS", r.choice(["right common femoral", "right internal jugular", "left basilic", "percutaneous"])),
        ("DEVICE", "; ".join(device_lines(c))),
        ("FLUORO", f"{r.randint(1, 18)}.{r.randint(0, 9)} min, DAP {r.randint(5, 300)} Gy-cm2"),
        ("EBL", f"{r.randint(1, 50)} mL"),
        ("COMPLICATIONS", r.choice(COMPLICATIONS)),
    ]
    sep = r.choice([": ", " | ", "\t"])
    return j(*[f"{k}{sep}{v}" for k, v in rows if v is not None])


def L_ir_post(c: Ctx) -> str:
    r = c.r
    proc, cpt = r.choice(IR_PROCS)
    lines = []
    if c.phi:
        p = c.patient
        lines += [f"POST-PROCEDURE INSTRUCTIONS - {M('FACILITY', c.facility)} Interventional Radiology",
                  f"Patient: {M('NAME', p['name'])}   MRN {M('MRN', p['mrn'])}", f"Procedure: {proc}, {c.D(c.exam_date)}"]
    else:
        lines += ["POST-PROCEDURE INSTRUCTIONS - Interventional Radiology", f"Procedure: {proc}"]
    lines += [
        consent_line(c),
        r.choice(["Keep the dressing dry for 48 hours.", "Lie flat for 4 hours.", "No lifting more than 10 pounds for 5 days."]),
        f"Complications: {r.choice(COMPLICATIONS)}",
        "Call for fever above 101 F, bleeding, or increasing pain.",
    ]
    if c.phi:
        lines += [f"Questions: interventional desk {M('PHONE', c.phone_ext())}; after hours {M('PHONE', c.phone())}.",
                  f"Follow-up visit {c.Dw(c.after(c.exam_date, 7, 30))} at {c.time()} with {c.clin(cred=False)}.",
                  f"Discharge reviewed by {c.staff()}.", f"Procedure code {c.codeline()}."]
    else:
        lines += ["Follow-up visit in 2 weeks with the interventional radiology clinic.", "Discharge instructions reviewed.", f"Procedure code {c.codeline()}."]
    return j(*lines)


# --------------------------------------------------------------------------
# teleradiology, addenda and critical-result layouts
# --------------------------------------------------------------------------
def L_tele_prelim(c: Ctx) -> str:
    r = c.r
    exam = r.choice(CTMR_EXAMS if r.random() < 0.6 else PLAIN_EXAMS)
    region = exam[3]
    lines = []
    if c.phi:
        p = c.patient
        lines += [f"TELERADIOLOGY PRELIMINARY REPORT - {M('FACILITY', c.tele_group())}",
                  f"Acquired at {M('FACILITY', c.facility)}, {M('LOCATION', c.city_state())}",
                  f"Patient: {M('NAME', p['name'])}  MRN: {M('MRN', p['mrn'])}  Accession: {M('ID', c.accession())}",
                  f"Study: {exam[1]}   {c.dtm(c.exam_date)}",
                  f"Image link: {M('URL', c.url())}"]
    else:
        lines += ["TELERADIOLOGY PRELIMINARY REPORT", f"Study: {exam[1]}"]
    lines += [
        f"Indication: {clin_history(c, region)}",
        comparison(c, exam, 1),
        f"Received {r.randint(2, 9)} series {c.sliceref()} from {c.scanner()}.",
        "Preliminary findings:", *find_sents(c, region, r.randint(2, 3)),
        "Impression: " + impr_sents(c, region, 1)[0],
    ]
    if c.phi:
        lines += [f"Interpreting radiologist: {c.clin()}   Reader initials {c.NI()}",
                  f"After-hours read {c.dtm(c.exam_date)}. Reader desk: {M('PHONE', c.phone_ext())} or {M('EMAIL', c.email())}.",
                  "A final interpretation will follow from the primary site."]
    else:
        lines += ["Interpreting radiologist: on-call reader.", "A final interpretation will follow from the primary site."]
    return j(*lines)


def L_addendum(c: Ctx) -> str:
    r = c.r
    exam = r.choice(CTMR_EXAMS + PLAIN_EXAMS)
    crit = crit_for(c, exam)
    lines = []
    if c.phi:
        p = c.patient
        d2 = c.after(c.exam_date, 1, 3)
        lines += [f"ADDENDUM TO {exam[1].upper()}", f"{M('NAME', p['name'])}  MRN {M('MRN', p['mrn'])}  Acc {M('ID', c.accession())}",
                  f"Original report {c.dtm(c.exam_date)}; addendum {c.dtm(d2)}"]
        who = c.clin()
        lines += [
            f"On further review of the images {c.sliceref()}, there is {crit}. This was not stated in the original report.",
            f"The finding was discussed with {who} at {c.time()} on {c.D(d2)}; read-back confirmed.",
            f"{c.honorific_surname()} was notified of the change by phone." if r.random() < 0.4 else None,
            f"Addendum by {c.clin(prefix='')}. Tech {c.NI()}.",
        ]
    else:
        lines += [f"ADDENDUM TO {exam[1].upper()}", "Original report was finalized earlier today; this addendum was entered later the same day.",
                  f"On further review of the images {c.sliceref()}, there is {crit}. This was not stated in the original report.",
                  "The finding was discussed with the ordering clinician; read-back confirmed.", "Addendum signed electronically."]
    return j(*lines)


def L_critical_log(c: Ctx) -> str:
    r = c.r
    exam = r.choice(CTMR_EXAMS + PLAIN_EXAMS)
    crit = crit_for(c, exam)
    lines = []
    if c.phi:
        p = c.patient
        d = c.exam_date
        lines += [f"CRITICAL RESULT COMMUNICATION - {M('FACILITY', c.facility)}",
                  f"Patient: {M('NAME', p['name'])} | MRN: {M('MRN', p['mrn'])} | {c.agefield()} {p['sex']}",
                  f"Study: {exam[1]} ({M('ID', c.accession())})",
                  f"Critical finding: {crit}",
                  f"Communicated to: {c.staff()} on {M('FACILITY', c.unit())}" if r.random() < 0.5 else f"Communicated to: {c.clin()}",
                  f"Communicated by: {c.clin(cred=False)}",
                  f"Date and time: {c.dtm(d)}",
                  f"Method: telephone {M('PHONE', c.phone_ext())}; read-back {r.choice(['confirmed', 'completed'])}.",
                  f"Logged by {c.NI()}.", f"Image reference {c.sliceref()}; {c.codeline()}."]
    else:
        lines += ["CRITICAL RESULT COMMUNICATION", f"Study: {exam[1]}", f"Critical finding: {crit}",
                  "Communicated to: the covering nurse by telephone.", "Communicated by: the interpreting radiologist.",
                  f"Time of communication: {c.time()} (within {r.randint(5, 30)} minutes of the final image).", "Read-back confirmed.", f"Image reference {c.sliceref()}; {c.codeline()}."]
    return j(*lines)


def L_overread(c: Ctx) -> str:
    r = c.r
    exam = r.choice(CTMR_EXAMS)
    region = exam[3]
    lines = []
    if c.phi:
        p = c.patient
        lines += [f"OUTSIDE STUDY OVER-READ - {M('FACILITY', c.facility)}", f"Patient: {M('NAME', p['name'])}   MRN: {M('MRN', p['mrn'])}",
                  f"Outside study: {exam[1]} at {M('FACILITY', c.other_facility())} in {M('LOCATION', c.city_state())}, {c.D(c.before(c.exam_date, 2, 40))}",
                  f"Outside accession {M('ID', c.accession())}; images imported {c.dtm(c.exam_date)}.",
                  f"Requested by {c.clin()}."]
    else:
        lines += ["OUTSIDE STUDY OVER-READ", f"Outside study: {exam[1]}, images imported for review."]
    lines += [f"Clinical question: {c.fill(r.choice(HIST[region]))}.", "Findings:", *find_sents(c, region, r.randint(2, 3)),
              "Impression: " + impr_sents(c, region, 1)[0]]
    if c.phi:
        lines.append(f"Imported {r.randint(2, 9)} series {c.sliceref()}; {c.codeline()}.")
        lines.append(f"Over-read by {c.clin()}. Discussed with {c.clin(cred=False)} at {c.time()} on {c.D(c.exam_date)}.")
    else:
        lines.append(f"Imported {r.randint(2, 9)} series {c.sliceref()}; {c.codeline()}.")
        lines.append("Over-read complete; discussed with the requesting clinician.")
    return j(*lines)


# --------------------------------------------------------------------------
# DICOM-style headers and worklist exports
# --------------------------------------------------------------------------
def L_dcm_header(c: Ctx) -> str:
    r = c.r
    exam = r.choice(CTMR_EXAMS + PLAIN_EXAMS)
    mod = exam[0]
    c.ph.add("dicom_tags")
    rows = []
    if c.phi:
        p = c.patient
        rows += [("(0008,0080)", "InstitutionName", M("FACILITY", c.facility)), ("(0010,0010)", "PatientName", M("NAME", p["name"])),
                 ("(0010,0020)", "PatientID", M("MRN", p["mrn"])), ("(0010,0030)", "PatientBirthDate", c.D(c.dob(), "iso")),
                 ("(0010,1010)", "PatientAge", f"{c.agefield()}" if c.age >= 90 else f"0{c.age}Y" if c.age < 100 else str(c.age)),
                 ("(0008,0050)", "AccessionNumber", M("ID", c.accession())), ("(0008,0020)", "StudyDate", c.D(c.exam_date, "iso")),
                 ("(0008,0090)", "ReferringPhysicianName", c.clin(prefix="", cred=False)),
                 ("(0008,1050)", "PerformingPhysicianName", c.clin(prefix="", cred=False)),
                 ("(0008,1070)", "OperatorsName", c.NI()), ("(0008,1010)", "StationName", M("FACILITY", c.station(mod)))]
        if r.random() < 0.5:
            rows.append(("(0018,1000)", "DeviceSerialNumber", M("ID", c.serial())))
    else:
        rows += [("(0010,0010)", "PatientName", "(empty)"), ("(0010,0020)", "PatientID", "(empty)")]
    rows += [("(0008,0060)", "Modality", mod), ("(0008,1030)", "StudyDescription", exam[1]),
             ("(0008,1090)", "ManufacturerModelName", c.scanner()), ("(0018,1020)", "SoftwareVersions", r.choice(SOFTWARE)),
             ("(0020,0011)", "SeriesNumber", str(r.randint(1, 12))), ("(0020,0013)", "InstanceNumber", str(r.randint(1, 400))),
             ("(0018,0050)", "SliceThickness", f"{r.choice(['0.6', '1.25', '2.5', '3', '5'])}"), ("(0020,1041)", "SliceLocation", f"{r.choice(['-', ''])}{r.randint(1, 190)}.{r.randint(0, 9)}"),
             ("(0018,0060)", "KVP", str(r.choice([80, 100, 120, 140]))), ("(0018,1190)", "FocalSpots", "1.0")]
    if c.phi and r.random() < 0.6:
        rows.append(("(0008,0081)", "InstitutionAddress", M("ADDRESS", c.address)))
        rows.append(("(0008,0092)", "ReferringPhysicianTelephone", M("PHONE", c.phone())))
    r.shuffle(rows)
    must = [t for t in rows if t[1] in ("ManufacturerModelName", "SeriesNumber")]
    rows = [t for t in rows if t not in must]
    rows = rows[: r.randint(max(7, len(rows) - 5), len(rows))] + must
    # keep the sort stable per row: tag order
    rows.sort(key=lambda t: t[0])
    fmt = r.choice([lambda t: f"{t[0]} {t[1]:<26}{t[2]}", lambda t: f"{t[0]}  {t[1]} = [{t[2]}]", lambda t: f"{t[1]}: {t[2]}   {t[0]}"])
    return j("DICOM HEADER DUMP", *[fmt(t) for t in rows])


def L_worklist(c: Ctx) -> str:
    r = c.r
    n = r.randint(3, 5)
    c.ph.add("worklist_export")
    sep = r.choice(["|", "\t", " ; "])
    if c.phi:
        header = sep.join(["NAME", "MRN", "ACCESSION", "EXAM", "CODE", "SCHEDULED", "STATION", "LOC", "TECH"])
        lines = [f"RADIOLOGY WORKLIST  {M('FACILITY', c.facility)}", header]
        for _ in range(n):
            ex = r.choice(CTMR_EXAMS + PLAIN_EXAMS)
            when = c.dtm(c.rand_date(dt.date(2025, 1, 1), dt.date(2026, 9, 30)), " ")
            loc = M("FACILITY", c.unit()) if r.random() < 0.7 else (f"{M('FACILITY', c.unit())} at {M('FACILITY', c.other_facility())}")
            lines.append(sep.join([M("NAME", c.full_name()), M("MRN", c.mrn()), M("ID", c.accession()), ex[1], "CPT " + ex[2], when, M("FACILITY", c.station(ex[0])), loc, M("NAME", c.initials())]))
        lines.append(f"{n} exams. Exported by {c.NI()} from workstation {M('IP', c.ip())}.")
    else:
        header = sep.join(["EXAM", "STATUS", "DURATION", "SERIES", "IMAGES", "CPT"])
        lines = ["RADIOLOGY WORKLIST SUMMARY", header]
        for _ in range(n):
            ex = r.choice(CTMR_EXAMS + PLAIN_EXAMS)
            lines.append(sep.join([ex[1], r.choice(["COMPLETED", "IN PROGRESS", "SCHEDULED"]), f"{r.randint(5, 55)} min", str(r.randint(1, 9)), str(r.randint(12, 600)), "CPT " + ex[2]]))
        c.ph.add("cpt_code")
        lines.append(f"{n} exams. Software {r.choice(SOFTWARE)}.")
    return j(*lines)


def L_audit(c: Ctx) -> str:
    r = c.r
    exam = r.choice(CTMR_EXAMS + PLAIN_EXAMS)
    lines = ["REPORT AUDIT TRAIL"]
    if c.phi:
        p = c.patient
        d = c.exam_date
        lines += [f"Accession {M('ID', c.accession())}  MRN {M('MRN', p['mrn'])}  {M('NAME', p['name'])}", f"Study {exam[1]}   Scanner {c.scanner()}"]
        events = ["ORDERED", "SCHEDULED", "ARRIVED", "STARTED", "COMPLETED", "DICTATED", "TRANSCRIBED", "SIGNED"]
        for ev in events[: r.randint(5, 8)]:
            d = c.after(d, 0, 1)
            who = c.NI() if ev in ("STARTED", "COMPLETED", "TRANSCRIBED", "ARRIVED") else c.clin(prefix="", cred=False)
            lines.append(f"{ev:<12}{c.dtm(d)}   by {who}")
        lines.append(f"Printed from {M('URL', c.url())}")
    else:
        lines += [f"Study {exam[1]}   Scanner {c.scanner()}"]
        for ev in ["ORDERED", "SCHEDULED", "COMPLETED", "DICTATED", "SIGNED"][: r.randint(3, 5)]:
            lines.append(f"{ev:<12}{c.time()}   duration {r.randint(2, 90)} min")
    return j(*lines)


def L_pacs_card(c: Ctx) -> str:
    r = c.r
    exam = r.choice(CTMR_EXAMS + PLAIN_EXAMS)
    rows = []
    if c.phi:
        p = c.patient
        rows += [("Patient", M("NAME", p["name"])), ("MRN", M("MRN", p["mrn"])), ("Accession", M("ID", c.accession())),
                 ("Study date", c.dtm(c.exam_date)), ("Facility", M("FACILITY", c.facility)), ("Station", M("FACILITY", c.station(exam[0]))),
                 ("Technologist", c.NI()), ("Transcriptionist", c.NI()), ("Viewer", M("URL", c.url())),
                 ("Source IP", M("IP", c.ip())), ("Contact", M("PHONE", c.phone_ext())), ("Status contact", M("EMAIL", c.email()))]
    rows += [("Study", exam[1]), ("Series count", str(r.randint(2, 14))), ("Images", str(r.randint(20, 1200))), ("Modality", exam[0]),
             ("Scanner", c.scanner()), ("Software", r.choice(SOFTWARE)), ("CPT", exam[2]), ("Slice", f"{r.choice(['0.6', '1.25', '3', '5'])} mm")]
    c.ph.add("cpt_code")
    c.ph.add("series_image_numbers")
    if not c.phi:
        rows += [("Patient", "(not stored)")]
    r.shuffle(rows)
    sep = r.choice([": ", " = ", "\t"])
    return j("STUDY INFORMATION", *[f"{k}{sep}{v}" for k, v in rows])


# --------------------------------------------------------------------------
# registry, plan and assembly
# --------------------------------------------------------------------------
LAYOUTS = {
    "ct_mri": [("ct_formal", L_ct_formal), ("ct_ris_compact", L_ct_ris), ("ct_narrative", L_ct_narrative),
               ("mri_sequences", L_mri_seq), ("ct_oncology_followup", L_ct_onc)],
    "plain_us": [("xr_portable_prelim", L_xr_prelim), ("xr_final_multiview", L_xr_final), ("us_measurements", L_us_report),
                 ("us_vascular_callback", L_us_vasc), ("xr_short_text", L_xr_short)],
    "mammo": [("mg_screening", L_mg_screen), ("mg_diagnostic_biopsy", L_mg_diag), ("mg_patient_letter", L_mg_letter)],
    "interventional": [("ir_procedure_note", L_ir_note), ("ir_procedure_log", L_ir_log), ("ir_post_instructions", L_ir_post)],
    "tele_critical": [("tele_preliminary", L_tele_prelim), ("addendum_new_datetime", L_addendum),
                      ("critical_result_log", L_critical_log), ("outside_overread", L_overread)],
    "dicom_worklist": [("dicom_header_dump", L_dcm_header), ("worklist_export", L_worklist),
                       ("report_audit_trail", L_audit), ("pacs_study_card", L_pacs_card)],
}
FAMILY_PHENOMENA = {
    "ct_mri": "ct_mri_report", "plain_us": "plain_film_ultrasound", "mammo": "breast_imaging",
    "interventional": "interventional_note", "tele_critical": "teleradiology_critical_result",
    "dicom_worklist": "dicom_worklist_export",
}


def largest_remainder(total: int, weights: list[float]) -> list[int]:
    raw = [total * w for w in weights]
    out = [int(x) for x in raw]
    order = sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)
    for i in order[: total - sum(out)]:
        out[i] += 1
    return out


def make_plan(rows: int = ROWS) -> list[dict]:
    rng = random.Random(f"{SEED}:plan")
    counts = largest_remainder(rows, [w for _, w in FAMILY_SHARES])
    entries: list[dict] = []
    for (fam, _), n in zip(FAMILY_SHARES, counts):
        lays = LAYOUTS[fam]
        per = largest_remainder(n, [1 / len(lays)] * len(lays))
        fam_entries = []
        for (lay, _fn), nl in zip(lays, per):
            negs = round(nl / 3)
            flags = [False] * negs + [True] * (nl - negs)
            rng.shuffle(flags)
            fam_entries += [{"family": fam, "layout": lay, "phi": f} for f in flags]
        w = CASE_WEIGHTS.get(fam, CASE_DEFAULT)
        cc = largest_remainder(len(fam_entries), list(w))
        cases = ["normal"] * cc[0] + ["caps"] * cc[1] + ["lower"] * cc[2]
        rng.shuffle(cases)
        for e, cs in zip(fam_entries, cases):
            e["case"] = cs
        entries += fam_entries
    rng.shuffle(entries)
    for i, e in enumerate(entries):
        e["index"] = i
    return entries


def render(marked: str, case: str) -> tuple[str, list[dict]]:
    """Parse [[LABEL|value]] markup and apply the case style to every piece."""
    tr = {"normal": lambda s: s, "caps": str.upper, "lower": str.lower}[case]
    out, spans, pos, prev = [], [], 0, 0

    def lit(s: str) -> str:
        t = tr(s)
        if len(t) != len(s):
            raise ValueError("case change altered length")
        return t

    for m in MARK.finditer(marked):
        a = lit(marked[prev:m.start()])
        v = lit(m[2])
        out += [a, v]
        pos += len(a)
        spans.append({"start": pos, "end": pos + len(v), "label": m[1], "text": v})
        pos += len(v)
        prev = m.end()
    out.append(lit(marked[prev:]))
    text = "".join(out)
    return text, spans


def name_tokens(value: str) -> list[str]:
    return [t.casefold() for t in re.split(r"[\s\-]+", value) if len(t) > 1]


DECOY_PATTERNS = {
    "radiology_category": re.compile(r"\b(?:bi-rads|li-rads|lung-rads|ti-rads|pi-rads|o-rads|fleischner|bosniak|recist)\b", re.I),
    "cpt_code": re.compile(r"\bcpt \d{5}\b", re.I),
    "series_image_numbers": re.compile(r"\b(?:series|image|se \d+ im) ?\d", re.I),
    "measurement": re.compile(r"\b\d+(?:\.\d+)? (?:mm|cm)\b", re.I),
    "contrast_lot": re.compile(r"\blot \d", re.I),
    "device_model": re.compile("|".join(re.escape(m) for m in SCANNERS), re.I),
    "eponym": re.compile("|".join(re.escape(m) for m in EPONYMS + ["Hill-Sachs"]), re.I),
}


def decoy_flags(text: str) -> set[str]:
    """Decoy categories present in a note, found from its text (not from the markup)."""
    return {name for name, pat in DECOY_PATTERNS.items() if pat.search(text)}


def build_row(entry: dict, use: Counter, val_rows: Counter, seen_texts: set, rejects: Counter, max_attempts: int = 200) -> dict:
    fn = dict(LAYOUTS[entry["family"]])[entry["layout"]]
    for attempt in range(max_attempts):
        rng = random.Random(f"{SEED}:row:{entry['index']}:{attempt}")
        ctx = Ctx(rng, use, entry["phi"], entry["family"], entry["layout"])
        marked = fn(ctx)
        try:
            text, spans = render(marked, entry["case"])
        except ValueError:
            rejects["structural"] += 1
            continue
        if "[[" in text or "]]" in text or annotation_errors(text, spans):
            rejects["structural"] += 1
            continue
        if bool(spans) != entry["phi"]:
            rejects["structural"] += 1
            continue
        if excluded_sequences(text) > 0 or any(excluded_value(s["text"]) for s in spans):
            rejects["exclusion"] += 1
            continue
        if text in seen_texts:
            rejects["duplicate_text"] += 1
            continue
        keys = {s["text"].casefold() for s in spans}
        toks = {t for s in spans if s["label"] == "NAME" for t in name_tokens(s["text"])}
        if any(val_rows[k] >= VALUE_CAP for k in keys) or any(use[t] >= VALUE_CAP for t in toks):
            rejects["value_cap"] += 1
            continue
        for k in keys:
            val_rows[k] += 1
        for t in toks:
            use[t] += 1
        seen_texts.add(text)
        rid = f"tk18-rad-{entry['index']:05d}"
        tf = f"tk18_radiology_{entry['layout']}"
        if entry["phi"]:
            identity = ctx.identities[0]
            groups = list(dict.fromkeys(ctx.identities))
        else:
            identity, groups = f"phi-free:{rid}", []
        phen = [FAMILY_PHENOMENA[entry["family"]], "phi_bearing" if entry["phi"] else "phi_free", f"case_{entry['case']}"]
        phen += sorted(ctx.ph | decoy_flags(text))
        return {
            "id": rid, "text": text, "spans": spans, "split": "train", "suite": "tk18_radiology",
            "annotation_policy": POLICY_VERSION, "cohort": "realistic", "family": tf, "template_family": tf,
            "phenomena": phen, "identity_group": identity, "identity_groups": groups, "pair_id": f"tk18-rad-pair-{entry['index']:05d}",
            "layout": entry["layout"], "report_family": entry["family"], "case_style": entry["case"],
            "phi_free": not entry["phi"], "attempts": attempt + 1,
        }
    raise RuntimeError(f"row {entry['index']} ({entry['layout']}) rejected {max_attempts} times: {dict(rejects)}")


def build(rows: int = ROWS) -> tuple[list[dict], dict]:
    plan = make_plan(rows)
    use, val_rows, seen, rejects = Counter(), Counter(), set(), Counter()
    out = []
    for entry in plan:
        out.append(build_row(entry, use, val_rows, seen, rejects))
    out.sort(key=lambda r: r["id"])
    return out, dict(rejects)


def stats(rows: list[dict], rejects: dict) -> dict:
    support = Counter()
    per_family, per_layout, per_case, phi_free_by_family = Counter(), Counter(), Counter(), Counter()
    val_rows = Counter()
    phen = Counter()
    for r in rows:
        per_family[r["report_family"]] += 1
        per_layout[r["layout"]] += 1
        per_case[r["case_style"]] += 1
        phi_free_by_family[r["report_family"]] += r["phi_free"]
        for s in r["spans"]:
            support[s["label"]] += 1
        for k in {s["text"].casefold() for s in r["spans"]}:
            val_rows[k] += 1
        phen.update(r["phenomena"])
    n = len(rows)
    top_value, top_count = (val_rows.most_common(1) or [("", 0)])[0]
    return {
        "label_support": {label: support[label] for label in ENTITY_TYPES},
        "rows_per_family": dict(per_family),
        "phi_free_rows_per_family": dict(phi_free_by_family),
        "phi_free_rows": sum(phi_free_by_family.values()),
        "rows_per_layout": dict(sorted(per_layout.items())),
        "layouts": len(per_layout),
        "max_layout_share": round(max(per_layout.values()) / n, 4),
        "case_counts": dict(per_case),
        "case_shares": {k: round(v / n, 4) for k, v in per_case.items()},
        "max_value_rows": top_count,
        "max_value_share": round(top_count / n, 4),
        "rejected_draws": rejects,
        "phenomena": dict(sorted(phen.items())),
    }


def write(rows: list[dict], rejects: dict, out: Path = OUT) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n" for r in rows).encode("utf-8")
    out.write_bytes(payload)
    st = stats(rows, rejects)
    manifest = {
        "bank": "tk18_train_radiology", "role": "train", "split": "train", "independent_final": False,
        "annotation_policy": POLICY_VERSION, "report_decisions": "L1-L4 (annotation_policy_reports.py)",
        "generator": "scripts/takeover/author18_radiology.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": SEED, "rows": len(rows), "documents": len(rows), "sha256": hashlib.sha256(payload).hexdigest(),
        "label_support": st["label_support"], "entity_support": st["label_support"], "label_counts": st["label_support"],
        "value_cap_rows": VALUE_CAP, "exclusion_list_sha256": EXCLUSIONS_SHA256,
        "exclusion_filter": "excluded_sequences(text) == 0 and excluded_value(span) False on every row; counts only",
        "synthetic_provenance": "All people, facilities, localities, addresses, identifiers invented; phones 555-0100..0199; reserved domains and TEST-NET IPs.",
        "overlap_audits": "not run by the author; owner's agent runs the sealed-overlap and development-overlap audits",
        **{k: v for k, v in st.items() if k != "label_support"},
    }
    out.with_suffix(out.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="build and report, do not write")
    ap.add_argument("--rows", type=int, default=ROWS)
    args = ap.parse_args()
    rows, rejects = build(args.rows)
    st = stats(rows, rejects)
    if args.check:
        print(json.dumps({k: v for k, v in st.items() if k != "phenomena"}, indent=1))
        return 0
    manifest = write(rows, rejects)
    print(json.dumps({k: manifest[k] for k in ("rows", "sha256", "label_support", "case_counts", "rejected_draws")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
