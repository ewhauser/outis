#!/usr/bin/env python3
"""Author the tk20 FHIR training bank (synthetic, invented values only).

Writes data/tk20_train_fhir.jsonl and its manifest. Every resource template and
value pool is written by this project from the FHIR R4 specification; no
external generator or example set is used. Conventions are decisions F1 to F4
in docs/annotation-policy-v2.md and the fixtures in
src/outis/data/annotation_policy_fhir.py:

* every PHI value inside a JSON string is its own span; keys, quotes, JSON
  punctuation and XHTML tags stay outside;
* patient-linked logical ids are ID (also inside a reference); a fullUrl to
  the patient's record is URL; meta and coding or identifier system URIs are
  not annotated;
* a FHIR date-time has its date part as DATE, time and timezone outside.

Deterministic: the same seed gives byte-identical output.

Usage: PYTHONPATH=src python scripts/takeover/author20_fhir.py [--out PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from outis.data.annotation_policy import (  # noqa: E402
    POLICY_VERSION,
    canonical_identity_group,
    identity_split,
)
from outis.data.generators.places import (  # noqa: E402
    excluded_sequences,
    excluded_value,
)

SEED = 20261006
TARGET_ROWS = 1500
DRAW_CAP = 6        # a drawn value is refused once it is on this many accepted rows
FINAL_CAP = 7       # no span value on more than this many rows (0.5% of 1,500 is 7.5)
SUITE = "tk20_fhir"
DEFAULT_OUT = ROOT / "data" / "tk20_train_fhir.jsonl"

# --------------------------------------------------------------------------------------
# Value pools. Everything here is invented or generic; none is copied from a fixture.
# --------------------------------------------------------------------------------------

GIVEN_F = (
    "Aldith Brisa Calanthe Daphine Elowen Fenella Gudrun Halcyon Imara Jessamy Katriel Leocadia Marisol Nerys "
    "Oriel Perdita Quenby Rosalba Saoirse Thessaly Ulrika Vesna Winnifred Xiomara Yelena Zinnia Ambrosine Blythe "
    "Cressida Delphine Eulalia Fiammetta Giselle Henrietta Isaura Juniper Kalinda Liesel Merewyn Neve Octavia "
    "Philippa Radka Sunniva Tamsin Una Valeska Wilhelmina Ysolde Zdenka Anneliese Bronagh Caoimhe Dagmar Estella "
    "Freydis Greta Hesper Ilsabet Jorunn Kerensa Lowri Mirela Nuala Ottilie Prudence Renée Aurélie Ísold Ximena Yusra"
).split()
GIVEN_M = (
    "Absalom Benedek Corwin Dariusz Eamon Florian Garrick Hallam Ignatius Joaquim Kasimir Lysander Merrick Nikodem "
    "Osric Peregrine Quillan Radomir Silvan Thaddeus Ulrich Vasco Wendell Xavion Yannick Zephaniah Alaric Bertrand "
    "Cormac Desmond Emeric Fitzroy Gideon Hartwig Ivor Jocelyn Konrad Leopold Magnus Nathaniel Oswin Percival "
    "Rafferty Sigmund Tobias Ulysses Valentin Warrick Yorick Zbigniew Anselm Bartholomew Caspian Dunstan Evander "
    "Fabrizio Godfrey Heron Isidore Jasper Killian Lazarus Matthias Nestor Orrin Phineas Rupert Søren Tomás Matías"
).split()
GIVEN_N = (
    "Alder Briar Cassius Dell Ellery Fern Gale Harbor Indigo Jules Kestrel Lake Marlow Nico Onyx Pax Quill Rowan "
    "Sage Tarn Umber Vale Wren Yarrow Zephyr Ainsley Bellamy Carey Devon Emerson Finley Hollis Jordy Keegan"
).split()
GIVEN_ONSET = (
    "Al Bel Cal Dor El Fen Gar Hes Il Jor Kes Lun Mal Nor Or Pel Quin Ros Sel Tav Ur Vel Wyn Xan Yor Zel Ad Bri Cor Dar"
).split()
GIVEN_TAIL = "ena ric wyn anda ius elle ard isa mund ora ette bert lynn ison ina ald ith eric ana".split()

SURNAMES = (
    "Abernathy Achterberg Adeyemi Aldecott Amberley Anselmi Arkwright Balakrishnan Barrowclough Bellweather "
    "Bjornstad Blackmoor Bordeleau Brightwell Cadogan Calderwood Chowdhury Cordova Dalrymple Delacroix Dunleavy "
    "Eastergard Eberhardt Ellingham Fairweather Farquharson Fitzgibbon Galbraith Gethin Goldenbaum Halloran "
    "Hargreaves Hathaway Ingleby Iwasaki Jankowski Jorgensen Kavanagh Kowalczyk Lindqvist Llewellyn Macready "
    "Maldonado Montague Nakagawa Nightingale Oyelaran Pemberton Penhallow Quigley Rathbone Ravenscroft Sandoval "
    "Schoenfeld Thackeray Tremaine Underhill Valderrama Wetherby Whitcombe Yamashita Zielinski Okpara Mbeki "
    "Szymanski Papadakis Vasquez Karaman Haddad Nwosu Ferreira Lindgren Castellanos Boateng Rasmussen Delgado "
    "Fontaine Kuznetsov Hartigan Ostrowski Ahlgren Bhattacharya Tennant Strickland Redmayne Pettigrew Moorcroft "
    "Lockhart Isakson Havlicek Grosvenor Gallagher Fenwick Esposito Dvorak Crabtree Burrowes Aziz"
).split()
SURNAME_STEMS = (
    "Ald Bram Carr Dun Eld Fair Garr Hol Ing Jen Kes Lang Mor Nor Ox Pen Quill Rad Sed Tam Ush Vard Wex Yar Zann "
    "Ash Bex Cor Dray Elm Fox Gil Hax Irv Jur Kel Lor Mer Nan Ols Pru Rus Sut Tre Ulf Vin Wil Yat Zor Brad"
).split()
SURNAME_ENDS = (
    "ley wick worth ham ford son by croft stone wood well gate shaw ridge mont dorf ski ian ez ova ström berg ton "
    "field man"
).split()

CITY_STEMS = (
    "Amber Bexley Carlow Dovren Eskmere Falken Garnet Hallow Ironvale Jessup Kingsmoor Lathrop Marden Nettle "
    "Oakhurst Pellam Quarry Ravel Stonebrook Thistle Umbra Vesper Whinmoor Yewdale Zephyr Ainley Brindle Cobble "
    "Drumlin Eastwick Fernley Grimsby Harwell Ivybank Juniper Kelling Linden Merrow Northam Orchard Plover "
    "Queensmoor Rushmere Saltmarsh Tarrant Upton Verdan Wexley Yarrow Alnwick Brecon Calder Danby Elsing Fordham "
    "Glenmoor Haverill Isley Jarrow Kinsale Lymond Marsden Norwell Osgood Pickering Radley Selby Thornley"
).split()
CITY_ENDS = "ford ville burg mere ton field haven dale wick ridge port brook stead holm vale cross".split()
STREET_STEMS = (
    "Alder Birchwood Calloway Dunmore Elderberry Fernhollow Gannet Harrow Ivybridge Juniper Kestrel Lindenwood "
    "Mulberry Nightjar Oakridge Pinecrest Quail Rookery Sycamore Tamarack Upland Violet Willowbend Yarrow "
    "Applegate Briarwood Cinder Dovetail Evergreen Foxglove Goldfinch Heron Ironwood Jasmine Kingfisher Larkfield "
    "Magnolia Nutmeg Orchard Peppercorn Quarry Ridgeline Sandpiper Thornbury Underhill Vineyard Wagtail Yewtree "
    "Almond Bluebell Cobbler Driftwood Elmgrove Fieldstone Granary Hawthorn Inglenook Jonquil Kettle Lavender "
    "Millrace Nightingale Oriole Plumtree Redwing Starling Tansy Umber Vesper Wheatley Zinnia Arbor Beacon Cardinal"
).split()
STREET_TYPES = "Lane Road Street Avenue Drive Court Way Terrace Place Boulevard Circle Trail Parkway".split()
SAINTS = "Aldric Brendan Cuthbert Dunstan Emeric Fintan Gerasim Hilarion Isidor Jerome Kentigern Ludmila Medard Odran".split()
FACILITY_PATTERNS = (
    "{c} Regional Medical Center|{c} Community Hospital|{c} Family Medicine|{c} Health Center|"
    "{c} Pediatric Associates|{c} Imaging Center|{c} Laboratory Services|{c} Cardiology Group|"
    "{c} Orthopedic Clinic|{c} Women's Health|{c} Urgent Care|{c} Surgical Center|{c} Memorial Hospital|"
    "{c} Internal Medicine|{c} Pharmacy|{c} Dialysis Center|{c} Behavioral Health|{c} Eye Institute|"
    "{c} General Hospital|{c} Primary Care|St. {s} Medical Center|Saint {s} Hospital|{c} Rehabilitation Center"
).split("|")
UNIT_FORMS = ("Apt {n}{l}", "Suite {n}0{k}", "Unit {n}", "#{n}{l}", "Floor {k}", "Lot {n}", "Bldg {l} Unit {n}")
US_STATES = (
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR "
    "PA RI SC SD TN TX UT VT VA WA WV WI WY"
).split()
EMAIL_DOMAINS = (
    "example.com example.org example.net mail.example.test clinic.example.test patients.example.com "
    "portal.example.org post.example.net"
).split()
FHIR_HOSTS = (
    "fhir.example.test|ehr.example.com|api.example.org|fhir.hospital.example.net|records.example.test|"
    "data.clinic.example.com|r4.fhir.example.org|interop.example.test"
).split("|")
FHIR_BASES = ("", "r4/", "fhir/", "api/fhir/", "baseR4/", "R4/")

LOINC = (
    ("2345-7", "Glucose [Mass/volume] in Serum or Plasma", "mg/dL", (60, 240)),
    ("718-7", "Hemoglobin [Mass/volume] in Blood", "g/dL", (8, 17)),
    ("8480-6", "Systolic blood pressure", "mm[Hg]", (90, 190)),
    ("8462-4", "Diastolic blood pressure", "mm[Hg]", (50, 110)),
    ("8867-4", "Heart rate", "/min", (48, 130)),
    ("29463-7", "Body weight", "kg", (45, 140)),
    ("8302-2", "Body height", "cm", (140, 200)),
    ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma", "mg/dL", (0, 3)),
    ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "%", (4, 12)),
    ("2093-3", "Cholesterol [Mass/volume] in Serum or Plasma", "mg/dL", (120, 300)),
    ("6690-2", "Leukocytes [#/volume] in Blood", "10*3/uL", (3, 14)),
    ("777-3", "Platelets [#/volume] in Blood", "10*3/uL", (120, 450)),
    ("2951-2", "Sodium [Moles/volume] in Serum or Plasma", "mmol/L", (130, 148)),
    ("2823-3", "Potassium [Moles/volume] in Serum or Plasma", "mmol/L", (3, 6)),
    ("8310-5", "Body temperature", "Cel", (35, 40)),
    ("59408-5", "Oxygen saturation in Arterial blood by Pulse oximetry", "%", (88, 100)),
)
CONDITIONS = (
    ("38341003", "Hypertensive disorder, systemic arterial", "I10"),
    ("44054006", "Diabetes mellitus type 2", "E11.9"),
    ("195967001", "Asthma", "J45.909"),
    ("49436004", "Atrial fibrillation", "I48.91"),
    ("13645005", "Chronic obstructive lung disease", "J44.9"),
    ("40930008", "Hypothyroidism", "E03.9"),
    ("84114007", "Heart failure", "I50.9"),
    ("396275006", "Osteoarthritis", "M19.90"),
    ("35489007", "Depressive disorder", "F32.A"),
    ("709044004", "Chronic kidney disease", "N18.9"),
    ("56265001", "Heart disease", "I51.9"),
    ("37796009", "Migraine", "G43.909"),
)
MEDS = (
    ("197361", "Amlodipine 5 MG Oral Tablet"),
    ("314076", "Lisinopril 10 MG Oral Tablet"),
    ("860975", "Metformin hydrochloride 500 MG Oral Tablet"),
    ("617310", "Atorvastatin 20 MG Oral Tablet"),
    ("866924", "Metoprolol Tartrate 50 MG Oral Tablet"),
    ("198211", "Simvastatin 40 MG Oral Tablet"),
    ("310965", "Ibuprofen 200 MG Oral Tablet"),
    ("308056", "Amoxicillin 500 MG Oral Capsule"),
    ("198440", "Acetaminophen 500 MG Oral Tablet"),
    ("966222", "Levothyroxine Sodium 50 MCG Oral Tablet"),
)
VACCINES = (
    ("140", "Influenza, seasonal, injectable, preservative free"),
    ("208", "COVID-19, mRNA, LNP-S, PF, 30 mcg/0.3 mL dose"),
    ("115", "Tdap"),
    ("33", "pneumococcal polysaccharide vaccine, 23 valent"),
    ("187", "zoster vaccine recombinant"),
    ("133", "Pneumococcal conjugate PCV 13"),
)
ALLERGENS = (
    ("7980", "Penicillin G"), ("256349002", "Peanut"), ("91936005", "Sulfonamide"), ("226017009", "Shellfish"),
    ("782361003", "Latex"), ("1191", "Aspirin"), ("21212", "Codeine"),
)
DOC_TYPES = (
    ("34133-9", "Summarization of episode note"), ("18842-5", "Discharge summary"), ("11488-4", "Consult note"),
    ("11506-3", "Progress note"), ("34117-2", "History and physical note"),
)
ENC_CLASSES = (("AMB", "ambulatory"), ("IMP", "inpatient encounter"), ("EMER", "emergency"), ("HH", "home health"))
LANGS = (("en", "English"), ("es", "Spanish"), ("pl", "Polish"), ("vi", "Vietnamese"), ("so", "Somali"), ("ar", "Arabic"))
RELATIONSHIPS = (("MTH", "mother"), ("FTH", "father"), ("SPS", "spouse"), ("CHILD", "child"), ("SIB", "sibling"),
                 ("GRNDCHILD", "grandchild"), ("FRND", "unrelated friend"), ("C", "emergency contact"))
OID_ROOTS = ("2.16.840.1.113883.19", "2.16.840.1.113883.3", "1.3.6.1.4.1.54392", "2.16.840.1.113883.4",
             "1.2.840.113619.6", "2.16.840.1.114222.4", "1.3.6.1.4.1.32473")
SYSTEMS = (
    "http://loinc.org", "http://snomed.info/sct", "http://www.nlm.nih.gov/research/umls/rxnorm",
    "http://hl7.org/fhir/sid/icd-10-cm", "http://unitsofmeasure.org", "http://hl7.org/fhir/sid/cvx",
    "http://terminology.hl7.org/CodeSystem/v3-ActCode", "http://terminology.hl7.org/CodeSystem/v2-0203",
    "http://hl7.org/fhir/sid/ndc", "http://www.ama-assn.org/go/cpt",
)
V2_0203 = "http://terminology.hl7.org/CodeSystem/v2-0203"
NPI_SYSTEM = "http://hl7.org/fhir/sid/us-npi"
SSN_SYSTEM = "http://hl7.org/fhir/sid/us-ssn"
PHONE_AREAS = [str(n) for n in range(201, 990) if n % 100 != 11 and n not in (555, 800, 888, 877, 866)]
TZ = ("-05:00", "-06:00", "-07:00", "-08:00", "-04:00", "+00:00", "Z", "-05:00", "-06:00")
MONTHS = "January February March April May June July August September October November December".split()
MON3 = [m[:3] for m in MONTHS]
EXT_URLS = (
    "http://hl7.org/fhir/StructureDefinition/patient-birthPlace",
    "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
    "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity",
    "http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex",
    "http://hl7.org/fhir/StructureDefinition/patient-interpreterRequired",
    "http://hl7.org/fhir/StructureDefinition/patient-religion",
    "http://hl7.org/fhir/StructureDefinition/data-absent-reason",
    "http://hl7.org/fhir/StructureDefinition/patient-importance",
)


class Reject(Exception):
    """A drawn row failed a draw-level or row-level check; the row is re-drawn."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------------------
# Span-carrying JSON writer
# --------------------------------------------------------------------------------------

class P:
    """A PHI string value. Becomes its own span; quotes and keys stay outside."""

    __slots__ = ("label", "value")

    def __init__(self, label: str, value: str):
        self.label, self.value = label, value


class Rich(list):
    """A JSON or XHTML string made of literal pieces (str) and PHI pieces (P)."""


def esc(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)[1:-1]


class Out:
    def __init__(self):
        self.parts: list[str] = []
        self.pos = 0
        self.spans: list[dict] = []

    def lit(self, s: str) -> None:
        self.parts.append(s)
        self.pos += len(s)

    def phi(self, label: str, value: str) -> None:
        if not value or value != value.strip():
            raise Reject("bad span value")
        start = self.pos
        self.lit(value)
        self.spans.append({"start": start, "end": self.pos, "label": label, "text": value})

    @property
    def text(self) -> str:
        return "".join(self.parts)


class Style:
    def __init__(self, kind: str, indent: str = "  "):
        self.kind, self.indent = kind, indent
        self.item = {"pretty": ",", "min": ",", "flat": ", "}[kind]
        self.key = {"pretty": ": ", "min": ":", "flat": ": "}[kind]


def _raw_ok(value: str, xml: bool = False) -> None:
    if esc(value) != value or (xml and any(c in value for c in "&<>")):
        raise Reject("value needs escaping")


def write_json(out: Out, obj, st: Style, lvl: int = 0) -> None:
    if isinstance(obj, P):
        _raw_ok(obj.value)
        out.lit('"')
        out.phi(obj.label, obj.value)
        out.lit('"')
    elif isinstance(obj, Rich):
        out.lit('"')
        for part in obj:
            if isinstance(part, P):
                _raw_ok(part.value, xml=True)
                out.phi(part.label, part.value)
            else:
                out.lit(esc(part))
        out.lit('"')
    elif isinstance(obj, str):
        out.lit('"' + esc(obj) + '"')
    elif obj is None or isinstance(obj, (bool, int, float)):
        out.lit(json.dumps(obj))
    elif isinstance(obj, dict):
        if not obj:
            out.lit("{}")
            return
        out.lit("{")
        for i, (key, value) in enumerate(obj.items()):
            if i:
                out.lit(st.item)
            if st.kind == "pretty":
                out.lit("\n" + st.indent * (lvl + 1))
            out.lit('"' + esc(key) + '"' + st.key)
            write_json(out, value, st, lvl + 1)
        if st.kind == "pretty":
            out.lit("\n" + st.indent * lvl)
        out.lit("}")
    elif isinstance(obj, (list, tuple)):
        if not obj:
            out.lit("[]")
            return
        out.lit("[")
        for i, value in enumerate(obj):
            if i:
                out.lit(st.item)
            if st.kind == "pretty":
                out.lit("\n" + st.indent * (lvl + 1))
            write_json(out, value, st, lvl + 1)
        if st.kind == "pretty":
            out.lit("\n" + st.indent * lvl)
        out.lit("]")
    else:  # pragma: no cover
        raise TypeError(type(obj))


def render_json(obj, st: Style) -> Out:
    out = Out()
    write_json(out, obj, st)
    return out


def write_rich_raw(out: Out, parts) -> None:
    for part in parts:
        if isinstance(part, P):
            if any(c in part.value for c in "&<>"):
                raise Reject("value needs xml escaping")
            out.phi(part.label, part.value)
        else:
            out.lit(part)


def recase(parts, mode: str):
    """Re-case prose and PHI pieces; markup (pieces starting with '<') is untouched."""
    if mode == "normal":
        return parts
    fn = str.upper if mode == "upper" else str.lower
    result = []
    for part in parts:
        if isinstance(part, P):
            keep = part.label in ("URL", "EMAIL", "IP")
            result.append(part if keep else P(part.label, fn(part.value)))
        elif part.startswith("<"):
            result.append(part)
        else:
            result.append(fn(part))
    return Rich(result) if isinstance(parts, Rich) else result


# --------------------------------------------------------------------------------------
# Row context: seeded draws with the exclusion filter and the per-value cap
# --------------------------------------------------------------------------------------

class Ctx:
    def __init__(self, rng: random.Random, used: Counter, stats: Counter, phi: bool = True):
        self.rng, self.used, self.stats, self.phi = rng, used, stats, phi
        self.local: set[str] = set()
        self.persons: list = []
        self.idstyle = rng.choice(("prefix", "prefix", "uuid", "num", "hex", "alnum"))
        self.host = rng.choice(FHIR_HOSTS)
        self.base = rng.choice(FHIR_BASES)
        self.ref_year = rng.choice((2024, 2025, 2026))

    def ok(self, value: str) -> bool:
        if value in self.local:
            return True
        if self.used[value] >= DRAW_CAP:
            self.stats["draw_refused_value_cap"] += 1
            return False
        if excluded_value(value):
            self.stats["draw_refused_exclusion"] += 1
            return False
        return True

    def draw(self, fn) -> str:
        for _ in range(120):
            value = fn()
            if self.ok(value):
                self.local.add(value)
                return value
        raise Reject("pool exhausted")

    def keep(self, value: str) -> str:
        if not self.ok(value):
            raise Reject("derived value refused")
        self.local.add(value)
        return value

    def choice(self, seq):
        return self.rng.choice(seq)

    def chance(self, p: float) -> bool:
        return self.rng.random() < p


def ascii_slug(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", folded.lower())


def digits(rng, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def pick_surname(ctx: Ctx) -> str:
    def fn():
        if ctx.chance(0.5):
            return ctx.choice(SURNAMES)
        return ctx.choice(SURNAME_STEMS) + ctx.choice(SURNAME_ENDS)
    return ctx.draw(fn)


def pick_given(ctx: Ctx, gender: str) -> str:
    pool = {"female": GIVEN_F, "male": GIVEN_M}.get(gender, GIVEN_N)

    def fn():
        if ctx.chance(0.4):
            return ctx.choice(GIVEN_ONSET) + ctx.choice(GIVEN_TAIL)
        if ctx.chance(0.15):
            return ctx.choice(GIVEN_N)
        return ctx.choice(pool)
    return ctx.draw(fn)


class Person:
    def __init__(self, given, family, suffix, gender, initial_style):
        self.given, self.family, self.suffix, self.gender = given, family, suffix, gender
        self.initial_style = initial_style

    @property
    def display(self) -> str:
        """The name as a person would write it: given names, family name, generational suffix."""
        base = " ".join(self.given + [self.family])
        return base + (" " + self.suffix if self.suffix else "")

    @property
    def identity(self) -> str:
        return " ".join(g for g in self.given if len(g.rstrip(".")) > 1) + " " + self.family


def make_person(ctx: Ctx, role: str = "patient", gender: str | None = None, family: str | None = None) -> Person:
    rng = ctx.rng
    for _ in range(60):
        g = gender or rng.choices(("female", "male", "other", "unknown"), (48, 48, 2, 2))[0]
        count = rng.choices((1, 2, 3), (58, 36, 6))[0]
        given = [pick_given(ctx, g)]
        for _ in range(count - 1):
            if ctx.chance(0.2):
                letter = ctx.draw(lambda: ctx.choice("ABCDEFGHJKLMNPQRSTVWYZ") + ctx.choice(("", ".")))
                given.append(letter)
            else:
                given.append(pick_given(ctx, g))
        fam = family or pick_surname(ctx)
        suffix = None
        if role == "patient" and g == "male" and ctx.chance(0.05):
            suffix = ctx.choice(("Jr.", "Sr.", "III", "II"))
        person = Person(given, fam, suffix, g, "initial" if any(len(x.rstrip(".")) == 1 for x in given) else "full")
        if identity_split(canonical_identity_group(person.identity)) != "train":
            ctx.stats["draw_refused_identity_partition"] += 1
            continue
        ctx.persons.append(person)
        return person
    raise Reject("no identity in the training partition")


# ---- identifiers and contact values ---------------------------------------------------

def fmt_phone(ctx: Ctx) -> str:
    area, line = ctx.choice(PHONE_AREAS), f"01{ctx.rng.randint(0, 99):02d}"
    form = ctx.rng.choices(range(9), (30, 20, 12, 10, 8, 6, 5, 5, 4))[0]
    return {
        0: f"({area}) 555-{line}", 1: f"{area}-555-{line}", 2: f"{area}.555.{line}", 3: f"+1 {area} 555 {line}",
        4: f"+1-{area}-555-{line}", 5: f"{area} 555 {line}", 6: f"{area}555{line}",
        7: f"({area}) 555-{line} x{ctx.rng.randint(10, 9999)}", 8: f"+1 ({area}) 555-{line}",
    }[form]


def pick_phone(ctx: Ctx) -> str:
    return ctx.draw(lambda: fmt_phone(ctx))


def pick_email(ctx: Ctx, person: Person | None = None) -> str:
    def fn():
        rng = ctx.rng
        if person is None:
            local = ascii_slug(ctx.choice(SURNAME_STEMS) + ctx.choice(SURNAME_ENDS)) + digits(rng, 2)
        else:
            g, f = ascii_slug(person.given[0]) or "x", ascii_slug(person.family) or "y"
            local = ctx.choice((f"{g}.{f}", f"{g[0]}{f}", f"{f}{digits(rng, 2)}", f"{g}_{f}", f"{g}{f[0]}{digits(rng, 3)}"))
        return f"{local}@{ctx.choice(EMAIL_DOMAINS)}"
    return ctx.draw(fn)


def pick_ssn(ctx: Ctx) -> str:
    rng = ctx.rng

    def fn():
        a = f"{rng.randint(900, 999)}"
        b, c = f"{rng.randint(1, 99):02d}", f"{rng.randint(1, 9999):04d}"
        return ctx.choice((f"{a}-{b}-{c}", f"{a}-{b}-{c}", f"{a}{b}{c}", f"{a} {b} {c}"))
    return ctx.draw(fn)


def pick_mrn(ctx: Ctx) -> str:
    rng = ctx.rng
    prefix = ctx.choice(("", "", "MRN-", "MR", "QX-", "E", "HX-", "00", "M-"))
    letters = ctx.choice(("AB", "KT", "ZP", "NW", "RL", "VD"))

    def fn():
        n = rng.randint(6, 9)
        base = digits(rng, n)
        return ctx.choice((prefix + base, prefix + base, letters + "-" + digits(rng, 7)))
    return ctx.draw(fn)


def pick_account(ctx: Ctx) -> str:
    rng = ctx.rng
    return ctx.draw(lambda: ctx.choice(("", "ACCT-", "AC", "GA-")) + digits(rng, rng.randint(8, 12)))


def pick_member(ctx: Ctx) -> str:
    rng = ctx.rng
    return ctx.draw(lambda: ctx.choice(("W", "H", "XYZ", "MB", "")) + digits(rng, rng.randint(8, 11)))


def pick_npi(ctx: Ctx) -> str:
    rng = ctx.rng
    return ctx.draw(lambda: ctx.choice("12") + digits(rng, 9))


def pick_license(ctx: Ctx) -> str:
    rng = ctx.rng
    return ctx.draw(lambda: ctx.choice(("MD", "A", "G", "ME", "RN")) + digits(rng, rng.randint(5, 8)))


def pick_dl(ctx: Ctx) -> str:
    rng = ctx.rng
    return ctx.draw(lambda: ctx.choice("ABCDEFHKLMNPRSTW") + digits(rng, rng.randint(7, 12)))


def pick_misc_id(ctx: Ctx, prefix: str, lo: int = 6, hi: int = 9) -> str:
    rng = ctx.rng
    return ctx.draw(lambda: ctx.choice((prefix, prefix + "-", "")) + digits(rng, rng.randint(lo, hi)))


def pick_ip(ctx: Ctx) -> str:
    rng = ctx.rng
    return ctx.draw(lambda: ctx.choice(("192.0.2", "198.51.100", "203.0.113")) + f".{rng.randint(1, 254)}")


def pick_fid(ctx: Ctx, kind: str) -> str:
    rng = ctx.rng

    def fn():
        s = ctx.idstyle
        if s == "prefix":
            return f"{kind}-{digits(rng, rng.randint(3, 7))}"
        if s == "uuid":
            return str(uuid.UUID(int=rng.getrandbits(128), version=4))
        if s == "num":
            return digits(rng, rng.randint(4, 9))
        if s == "hex":
            return "".join(rng.choice("0123456789abcdef") for _ in range(rng.randint(8, 16)))
        return "".join(rng.choice("0123456789abcdefghjkmnpqrstvwxyz") for _ in range(rng.randint(6, 10)))
    return ctx.draw(fn)


def pick_url_host_path(ctx: Ctx, kind: str, ident: str) -> str:
    return f"https://{ctx.host}/{ctx.base}{kind}/{ident}"


# ---- places ----------------------------------------------------------------------------

def pick_city(ctx: Ctx) -> str:
    return ctx.draw(lambda: ctx.choice(CITY_STEMS) + ctx.choice(CITY_ENDS))


def pick_zip(ctx: Ctx) -> str:
    rng = ctx.rng
    return ctx.draw(lambda: f"{rng.randint(10000, 99899)}" + (f"-{rng.randint(1000, 9999)}" if ctx.chance(0.12) else ""))


def pick_street(ctx: Ctx) -> str:
    rng = ctx.rng
    return ctx.draw(lambda: f"{rng.randint(2, 9899)} {ctx.choice(STREET_STEMS)} {ctx.choice(STREET_TYPES)}"
                    if ctx.chance(0.94) else f"PO Box {rng.randint(10, 9899)}")


def pick_unit(ctx: Ctx) -> str:
    rng = ctx.rng
    form = ctx.choice(UNIT_FORMS)
    return ctx.draw(lambda: form.format(n=rng.randint(1, 48), l=ctx.choice("ABCDEFG"), k=rng.randint(1, 9)))


def pick_facility(ctx: Ctx) -> str:
    return ctx.draw(lambda: ctx.choice(FACILITY_PATTERNS).format(c=ctx.choice(CITY_STEMS) + (ctx.choice(CITY_ENDS) if ctx.chance(0.5) else ""), s=ctx.choice(SAINTS)))


class Address:
    def __init__(self, ctx: Ctx):
        self.street = pick_street(ctx)
        self.unit = pick_unit(ctx) if ctx.chance(0.35) and not self.street.startswith("PO Box") else None
        self.city = pick_city(ctx)
        self.state = ctx.choice(US_STATES)
        self.zip = pick_zip(ctx)
        self.district = (ctx.draw(lambda: ctx.choice(CITY_STEMS) + " County") if ctx.chance(0.12) else None)

    @property
    def full(self) -> str:
        return ", ".join([self.street] + ([self.unit] if self.unit else []) + [self.city]) + f", {self.state} {self.zip}"

    @property
    def lines(self) -> list[str]:
        return [self.street] + ([self.unit] if self.unit else [])


def address_json(ctx: Ctx, a: Address, use: str | None = None) -> dict:
    d: dict = {}
    if use:
        d["use"] = use
        if ctx.chance(0.3):
            d["type"] = ctx.choice(("postal", "physical", "both"))
    if ctx.chance(0.3):
        d["text"] = P("ADDRESS", a.full)
    d["line"] = [P("ADDRESS", x) for x in a.lines]
    d["city"] = P("LOCATION", a.city)
    if a.district:
        d["district"] = P("LOCATION", a.district)
    d["state"] = a.state
    d["postalCode"] = P("LOCATION", a.zip)
    d["country"] = ctx.choice(("US", "US", "USA"))
    return d


# ---- dates -----------------------------------------------------------------------------

def iso(d: date) -> str:
    return d.isoformat()


def pick_date(ctx: Ctx, lo: date, hi: date) -> date:
    span = (hi - lo).days
    value = ctx.draw(lambda: iso(lo + timedelta(days=ctx.rng.randint(0, span))))
    return date.fromisoformat(value)


def encounter_date(ctx: Ctx) -> date:
    return pick_date(ctx, date(2019, 1, 5), date(ctx.ref_year, 6, 28))


def clock(ctx: Ctx) -> str:
    return f"T{ctx.rng.randint(0, 23):02d}:{ctx.rng.randint(0, 59):02d}:{ctx.rng.randint(0, 59):02d}"


def fhir_date(ctx: Ctx, d: date, kind: str = "dt"):
    """A FHIR date or date-time value: the date part is DATE; time and zone stay outside."""
    if kind == "date":
        return P("DATE", iso(d))
    tz = ctx.choice(TZ)
    suffix = clock(ctx) + (".000" if kind == "instant" else "") + tz
    return Rich([P("DATE", iso(d)), suffix])


def shift(ctx: Ctx, d: date, days: int) -> date:
    new = d + timedelta(days=days)
    ctx.keep(iso(new))
    return new


def written_date(ctx: Ctx, d: date, fmt: int | None = None) -> str:
    fmt = ctx.rng.randrange(7) if fmt is None else fmt
    return (
        f"{MONTHS[d.month - 1]} {d.day}, {d.year}", f"{d.day} {MONTHS[d.month - 1]} {d.year}",
        f"{d.month:02d}/{d.day:02d}/{d.year}", iso(d), f"{MON3[d.month - 1]} {d.day}, {d.year}",
        f"{d.month}/{d.day}/{d.year}", f"{d.day:02d}-{MON3[d.month - 1]}-{d.year}",
    )[fmt]


def birth_date(ctx: Ctx, age: int | None = None) -> date:
    if age is None:
        age = ctx.rng.choices(range(100), [3] * 18 + [4] * 40 + [3] * 32 + [1] * 10)[0]
    ref = date(ctx.ref_year, 6, 28)
    lo = date(ref.year - age - 1, ref.month, ref.day) + timedelta(days=1)
    return pick_date(ctx, lo, date(ref.year - age, ref.month, ref.day))


# --------------------------------------------------------------------------------------
# Actors: patient, clinician, organisation
# --------------------------------------------------------------------------------------

class Pt:
    """One invented patient with lazily drawn identifiers and contact values."""

    def __init__(self, ctx: Ctx, age: int | None = None, gender: str | None = None):
        self.ctx = ctx
        self.person = make_person(ctx, "patient", gender)
        self.id = pick_fid(ctx, "pt")
        self.dob = birth_date(ctx, age)
        self.age = ctx.ref_year - self.dob.year - (1 if (self.dob.month, self.dob.day) > (6, 28) else 0)
        self._memo: dict = {}

    def memo(self, key: str, fn):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    def mrn(self):
        return self.memo("mrn", lambda: pick_mrn(self.ctx))
    def ssn(self):
        return self.memo("ssn", lambda: pick_ssn(self.ctx))
    def account(self):
        return self.memo("acct", lambda: pick_account(self.ctx))
    def member(self):
        return self.memo("member", lambda: pick_member(self.ctx))
    def dl(self):
        return self.memo("dl", lambda: pick_dl(self.ctx))
    def phone(self, n: int = 0):
        return self.memo(f"phone{n}", lambda: pick_phone(self.ctx))
    def email(self):
        return self.memo("email", lambda: pick_email(self.ctx, self.person))
    def addr(self):
        return self.memo("addr", lambda: Address(self.ctx))
    def portal(self):
        return self.memo("portal", lambda: f"https://{self.ctx.choice(('portal', 'my', 'patient'))}.{self.ctx.choice(EMAIL_DOMAINS)}/{self.ctx.choice(('u', 'p', 'member'))}/{digits(self.ctx.rng, 7)}")
    def fullurl(self):
        return pick_url_host_path(self.ctx, "Patient", self.id)

    def name_p(self) -> P:
        return P("NAME", self.person.display)


class Prac:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.person = make_person(ctx, "clinician")
        self.npi = pick_npi(ctx)
        self.creds = ctx.choice(("MD", "MD", "DO", "NP", "PA-C", "RN", "MD, PhD", "PharmD"))
        self._lic = None

    @property
    def lic(self):
        if self._lic is None:
            self._lic = pick_license(self.ctx)
        return self._lic

    @property
    def md(self) -> bool:
        return self.creds in ("MD", "DO", "MD, PhD")

    def display(self) -> Rich:
        name = P("NAME", self.person.display)
        return Rich(["Dr. ", name]) if self.md and self.ctx.chance(0.7) else Rich([name, f", {self.creds}"])


class Org:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.name = pick_facility(ctx)
        self.addr = Address(ctx)
        self.npi = pick_npi(ctx)
        self._memo: dict = {}

    def phone(self, n=0):
        if n not in self._memo:
            self._memo[n] = pick_phone(self.ctx)
        return self._memo[n]

    def email(self):
        if "e" not in self._memo:
            self._memo["e"] = pick_email(self.ctx)
        return self._memo["e"]


# --------------------------------------------------------------------------------------
# JSON fragments
# --------------------------------------------------------------------------------------

def coding(system: str, code: str, display: str | None = None) -> dict:
    d = {"system": system, "code": code}
    if display:
        d["display"] = display
    return d


def cc(system: str, code: str, display: str | None = None, text: str | None = None) -> dict:
    d: dict = {"coding": [coding(system, code, display)]}
    if text:
        d["text"] = text
    return d


def sys_oid(ctx: Ctx) -> str:
    return f"urn:oid:{ctx.choice(OID_ROOTS)}.{ctx.rng.randint(1, 999)}" + (f".{ctx.rng.randint(1, 99)}" if ctx.chance(0.6) else "")


def local_system(ctx: Ctx, kind: str) -> str:
    return ctx.choice((sys_oid(ctx), f"https://{ctx.host}/ids/{kind}", f"http://ehr.example.test/{kind}", sys_oid(ctx)))


def meta(ctx: Ctx) -> dict:
    m = {"versionId": str(ctx.rng.randint(1, 14)),
         "lastUpdated": f"{ctx.rng.randint(2019, 2026)}-{ctx.rng.randint(1, 12):02d}-{ctx.rng.randint(1, 28):02d}"
                        f"T{ctx.rng.randint(0, 23):02d}:{ctx.rng.randint(0, 59):02d}:{ctx.rng.randint(0, 59):02d}.000Z"}
    if ctx.chance(0.35):
        m["profile"] = [ctx.choice(("http://hl7.org/fhir/us/core/StructureDefinition/us-core-patient",
                                    "http://hl7.org/fhir/StructureDefinition/Patient",
                                    "http://hl7.org/fhir/us/core/StructureDefinition/us-core-encounter"))]
    return m


def identifier(ctx: Ctx, label: str, value: str, system: str, tcode: str | None = None, tdisp: str | None = None,
               use: str | None = None) -> dict:
    d: dict = {}
    if use:
        d["use"] = use
    if tcode:
        d["type"] = {"coding": [coding(V2_0203, tcode, tdisp)]}
    d["system"] = system
    d["value"] = P(label, value)
    return d


def hname(ctx: Ctx, person: Person, use: str | None = None, text: bool = False, prefix: str | None = None,
          creds: str | None = None) -> dict:
    d: dict = {}
    if use:
        d["use"] = use
    if text:
        parts: list = []
        if prefix:
            parts.append(prefix + " ")
        parts.append(P("NAME", person.display))
        if creds:
            parts.append(", " + creds)
        d["text"] = Rich(parts)
    d["family"] = P("NAME", person.family)
    d["given"] = [P("NAME", g) for g in person.given]
    if prefix:
        d["prefix"] = [prefix]
    suffix: list = []
    if person.suffix:
        suffix.append(P("NAME", person.suffix))
    if creds:
        suffix.append(creds)
    if suffix:
        d["suffix"] = suffix
    return d


def telecom(system: str, label: str, value: str, use: str | None = None, rank: int | None = None) -> dict:
    d: dict = {"system": system, "value": P(label, value)}
    if use:
        d["use"] = use
    if rank:
        d["rank"] = rank
    return d


def pref(ctx: Ctx, pt: Pt, display: bool | None = None) -> dict:
    d: dict = {"reference": Rich(["Patient/", P("ID", pt.id)])}
    if display if display is not None else ctx.chance(0.45):
        d["display"] = pt.name_p()
    return d


def prac_ref(ctx: Ctx, pr: Prac, with_id: bool = True) -> dict:
    d: dict = {}
    if with_id:
        d["identifier"] = {"system": NPI_SYSTEM, "value": P("ID", pr.npi)}
    d["display"] = pr.display()
    return d


def org_ref(org: Org, ident: bool = False) -> dict:
    d: dict = {}
    if ident:
        d["identifier"] = {"system": NPI_SYSTEM, "value": P("ID", org.npi)}
    d["display"] = P("FACILITY", org.name)
    return d


def phone_entries(ctx: Ctx, pt: Pt, n: int | None = None) -> list:
    entries = [telecom("phone", "PHONE", pt.phone(0), ctx.choice(("home", "mobile", "work")), 1)]
    if (n or ctx.rng.randint(1, 4)) > 1:
        entries.append(telecom("phone", "PHONE", pt.phone(1), ctx.choice(("mobile", "work", "temp"))))
    if ctx.chance(0.25):
        entries.append(telecom("fax", "PHONE", pt.phone(2), "home"))
    if ctx.chance(0.4):
        entries.append(telecom("email", "EMAIL", pt.email(), ctx.choice(("home", "work"))))
    if ctx.chance(0.12):
        entries.append(telecom("url", "URL", pt.portal(), "home"))
    return entries


MARITAL = (("M", "Married"), ("S", "Never Married"), ("D", "Divorced"), ("W", "Widowed"), ("U", "unmarried"))
V3_MARITAL = "http://terminology.hl7.org/CodeSystem/v3-MaritalStatus"


def lang_comm(ctx: Ctx) -> list:
    code, name = ctx.choice(LANGS)
    return [{"language": cc("urn:ietf:bcp:47", code, name), "preferred": ctx.chance(0.8)}]


def patient_identifiers(ctx: Ctx, pt: Pt, org: Org | None = None) -> list:
    ids = []
    mrn = identifier(ctx, "MRN", pt.mrn(), local_system(ctx, "mrn"), "MR", "Medical record number", "usual")
    if org is not None and ctx.chance(0.35):
        mrn["assigner"] = {"display": P("FACILITY", org.name)}
    ids.append(mrn)
    if ctx.chance(0.3):
        ids.append(identifier(ctx, "SSN", pt.ssn(), SSN_SYSTEM, "SS", "Social Security number"))
    if ctx.chance(0.16):
        ids.append(identifier(ctx, "ID", pt.dl(), f"https://dmv.example.test/license/{ctx.choice(US_STATES).lower()}",
                              "DL", "Driver's license number"))
    if ctx.chance(0.08):
        ids.append(identifier(ctx, "ID", pick_misc_id(ctx, "P", 8, 9), "https://passports.example.test/number", "PPN", "Passport number"))
    if ctx.chance(0.18):
        ids.append(identifier(ctx, "ACCOUNT", pt.account(), local_system(ctx, "account"), "AN", "Account number"))
    if ctx.chance(0.12):
        ids.append(identifier(ctx, "ID", pt.member(), f"https://payer.example.test/member/{ctx.rng.randint(10, 99)}", "MB", "Member Number"))
    ctx.rng.shuffle(ids)
    return ids


def relative(ctx: Ctx, pt: Pt, kin: bool = True) -> tuple[Person, str, str]:
    code, disp = ctx.choice(RELATIONSHIPS)
    fam = pt.person.family if ctx.chance(0.5) and code not in ("SPS", "FRND") else None
    return make_person(ctx, "related", family=fam), code, disp


def maybe(d: dict, key: str, value, ctx: Ctx, p: float) -> None:
    if ctx.chance(p):
        d[key] = value


# --------------------------------------------------------------------------------------
# Resources. Each takes (ctx, pt, ...) and returns a dict of JSON-ready values; when
# ctx.phi is False the builder returns the same shape with no identifying content.
# --------------------------------------------------------------------------------------

def r_patient(ctx: Ctx, pt: Pt | None, org: Org | None = None, pr: Prac | None = None, with_id: bool = True) -> dict:
    r: dict = {"resourceType": "Patient"}
    if not ctx.phi:
        if ctx.chance(0.5):
            r["meta"] = meta(ctx)
        r["active"] = ctx.chance(0.9)
        if ctx.chance(0.5):
            r["extension"] = [{"url": ctx.choice(EXT_URLS[1:4]), "valueCode": ctx.choice(("F", "M", "UNK", "2106-3"))}]
        r["gender"] = ctx.choice(("female", "male", "other", "unknown"))
        r["deceasedBoolean"] = ctx.chance(0.1)
        if ctx.chance(0.6):
            r["address"] = [{"use": "home", "state": ctx.choice(US_STATES), "country": "US"}]
        r["maritalStatus"] = cc(V3_MARITAL, *ctx.choice(MARITAL))
        r["multipleBirthBoolean"] = ctx.chance(0.1)
        if ctx.chance(0.7):
            r["communication"] = lang_comm(ctx)
        return r
    assert pt is not None
    person = pt.person
    if with_id:
        r["id"] = P("ID", pt.id)
    maybe(r, "meta", meta(ctx), ctx, 0.55)
    if ctx.chance(0.2):
        city = pick_city(ctx)
        r["extension"] = [{"url": EXT_URLS[0], "valueAddress": {"city": P("LOCATION", city), "state": ctx.choice(US_STATES), "country": "US"}}]
    r["identifier"] = patient_identifiers(ctx, pt, org)
    r["active"] = True
    names = [hname(ctx, person, "official", text=ctx.chance(0.25))]
    if ctx.chance(0.12):
        nick = ctx.draw(lambda: ctx.choice(GIVEN_N) + ctx.choice(("", "y", "ie")))
        names.append({"use": "nickname", "given": [P("NAME", nick)]})
    if person.gender == "female" and ctx.chance(0.1):
        names.append({"use": "maiden", "family": P("NAME", pick_surname(ctx))})
    r["name"] = names
    r["telecom"] = phone_entries(ctx, pt)
    r["gender"] = person.gender
    r["birthDate"] = birth_value(ctx, pt)
    if ctx.chance(0.07):
        r["deceasedDateTime"] = fhir_date(ctx, encounter_date(ctx))
    elif ctx.chance(0.4):
        r["deceasedBoolean"] = False
    a = [address_json(ctx, pt.addr(), "home")]
    if ctx.chance(0.15):
        a.append(address_json(ctx, Address(ctx), ctx.choice(("work", "temp", "old"))))
    r["address"] = a
    if ctx.chance(0.6):
        r["maritalStatus"] = cc(V3_MARITAL, *ctx.choice(MARITAL))
    if ctx.chance(0.4):
        rp, code, disp = relative(ctx, pt)
        contact: dict = {"relationship": [cc("http://terminology.hl7.org/CodeSystem/v2-0131", ctx.choice(("N", "E", "C", "BP")))],
                         "name": hname(ctx, rp), "telecom": [telecom("phone", "PHONE", pick_phone(ctx), ctx.choice(("home", "mobile")))]}
        if ctx.chance(0.4):
            contact["address"] = address_json(ctx, Address(ctx))
        maybe(contact, "gender", rp.gender, ctx, 0.5)
        r["contact"] = [contact]
    if ctx.chance(0.5):
        r["communication"] = lang_comm(ctx)
    if pr is not None and ctx.chance(0.5):
        r["generalPractitioner"] = [prac_ref(ctx, pr)]
    if org is not None and ctx.chance(0.5):
        r["managingOrganization"] = org_ref(org)
    return r


def birth_value(ctx: Ctx, pt: Pt):
    """birthDate is a full date, or a year-month or year for a small share."""
    roll = ctx.rng.random()
    if roll < 0.04:
        return P("DATE", f"{pt.dob.year}")
    if roll < 0.09:
        return P("DATE", f"{pt.dob.year}-{pt.dob.month:02d}")
    return P("DATE", iso(pt.dob))


def r_related_person(ctx: Ctx, pt: Pt | None, with_id: bool = True) -> dict:
    r: dict = {"resourceType": "RelatedPerson"}
    if not ctx.phi:
        maybe(r, "meta", meta(ctx), ctx, 0.5)
        r["active"] = True
        code, disp = ctx.choice(RELATIONSHIPS)
        r["relationship"] = [cc("http://terminology.hl7.org/CodeSystem/v3-RoleCode", code, disp)]
        r["gender"] = ctx.choice(("female", "male", "other"))
        r["communication"] = lang_comm(ctx)
        return r
    rp, code, disp = relative(ctx, pt)
    if with_id:
        r["id"] = P("ID", pick_fid(ctx, "rp"))
    maybe(r, "meta", meta(ctx), ctx, 0.4)
    if ctx.chance(0.2):
        r["identifier"] = [identifier(ctx, "SSN", pick_ssn(ctx), SSN_SYSTEM, "SS", "Social Security number")]
    r["active"] = True
    r["patient"] = pref(ctx, pt, display=None)
    r["relationship"] = [cc("http://terminology.hl7.org/CodeSystem/v3-RoleCode", code, disp)]
    r["name"] = [hname(ctx, rp, "official", text=ctx.chance(0.2))]
    tel = [telecom("phone", "PHONE", pick_phone(ctx), ctx.choice(("home", "mobile", "work")), 1)]
    if ctx.chance(0.5):
        tel.append(telecom("email", "EMAIL", pick_email(ctx, rp)))
    if ctx.chance(0.2):
        tel.append(telecom("fax", "PHONE", pick_phone(ctx), "work"))
    r["telecom"] = tel
    r["gender"] = rp.gender
    if ctx.chance(0.4):
        r["birthDate"] = P("DATE", iso(birth_date(ctx, ctx.rng.randint(18, 88))))
    if ctx.chance(0.7):
        r["address"] = [address_json(ctx, Address(ctx), "home")]
    if ctx.chance(0.2):
        r["period"] = {"start": fhir_date(ctx, encounter_date(ctx), "date")}
    return r


def r_practitioner(ctx: Ctx, pr: Prac | None) -> dict:
    r: dict = {"resourceType": "Practitioner"}
    if not ctx.phi:
        maybe(r, "meta", meta(ctx), ctx, 0.5)
        r["active"] = True
        r["gender"] = ctx.choice(("female", "male", "other"))
        r["qualification"] = [{"code": cc("http://terminology.hl7.org/CodeSystem/v2-0360", ctx.choice(("MD", "NP", "RN", "PA")))}]
        r["communication"] = [cc("urn:ietf:bcp:47", *ctx.choice(LANGS))]
        return r
    maybe(r, "meta", meta(ctx), ctx, 0.4)
    ids = [identifier(ctx, "ID", pr.npi, NPI_SYSTEM, "NPI", "National provider identifier")]
    if ctx.chance(0.6):
        ids.append(identifier(ctx, "ID", pr.lic, f"https://licensing.example.test/{ctx.choice(US_STATES).lower()}", "MD", "Medical license number"))
    r["identifier"] = ids
    r["active"] = True
    prefix = "Dr." if pr.md and ctx.chance(0.8) else None
    r["name"] = [hname(ctx, pr.person, "official", text=ctx.chance(0.3), prefix=prefix, creds=pr.creds if ctx.chance(0.7) else None)]
    tel = [telecom("phone", "PHONE", pick_phone(ctx), "work", 1)]
    if ctx.chance(0.5):
        tel.append(telecom("email", "EMAIL", pick_email(ctx, pr.person), "work"))
    if ctx.chance(0.3):
        tel.append(telecom("fax", "PHONE", pick_phone(ctx), "work"))
    if ctx.chance(0.15):
        tel.append(telecom("phone", "PHONE", pick_phone(ctx), "temp"))
    r["telecom"] = tel
    if ctx.chance(0.5):
        r["address"] = [address_json(ctx, Address(ctx), "work")]
    r["gender"] = pr.person.gender
    if ctx.chance(0.1):
        r["birthDate"] = P("DATE", iso(birth_date(ctx, ctx.rng.randint(30, 75))))
    if ctx.chance(0.5):
        r["qualification"] = [{"code": {"text": pr.creds.split(",")[0]}}]
    return r


def r_organization(ctx: Ctx, org: Org | None) -> dict:
    r: dict = {"resourceType": "Organization"}
    if not ctx.phi:
        maybe(r, "meta", meta(ctx), ctx, 0.5)
        r["active"] = ctx.chance(0.9)
        r["type"] = [cc("http://terminology.hl7.org/CodeSystem/organization-type", ctx.choice(("prov", "dept", "team", "pay", "other")))]
        return r
    maybe(r, "meta", meta(ctx), ctx, 0.4)
    ids = [identifier(ctx, "ID", org.npi, NPI_SYSTEM, "NPI", "National provider identifier")]
    r["identifier"] = ids
    r["active"] = True
    r["type"] = [cc("http://terminology.hl7.org/CodeSystem/organization-type", "prov", "Healthcare Provider")]
    r["name"] = P("FACILITY", org.name)
    if ctx.chance(0.3):
        r["alias"] = [P("FACILITY", ctx.draw(lambda: ctx.choice(CITY_STEMS) + " " + ctx.choice(("Clinic", "Health", "Hospital", "Medical Group"))))]
    tel = [telecom("phone", "PHONE", org.phone(0), "work")]
    if ctx.chance(0.5):
        tel.append(telecom("fax", "PHONE", org.phone(1), "work"))
    if ctx.chance(0.3):
        tel.append(telecom("email", "EMAIL", org.email(), "work"))
    r["telecom"] = tel
    r["address"] = [address_json(ctx, org.addr, "work")]
    if ctx.chance(0.2):
        r["partOf"] = {"display": P("FACILITY", pick_facility(ctx))}
    return r


def r_location(ctx: Ctx, org: Org | None) -> dict:
    r: dict = {"resourceType": "Location"}
    if not ctx.phi:
        maybe(r, "meta", meta(ctx), ctx, 0.5)
        r["status"] = ctx.choice(("active", "suspended", "inactive"))
        r["mode"] = ctx.choice(("instance", "kind"))
        r["type"] = [cc("http://terminology.hl7.org/CodeSystem/v3-RoleCode", ctx.choice(("HOSP", "PHARM", "OUTPHARM", "ER", "CHR")))]
        r["physicalType"] = cc("http://terminology.hl7.org/CodeSystem/location-physical-type", ctx.choice(("bu", "wa", "ro", "si")))
        return r
    maybe(r, "meta", meta(ctx), ctx, 0.4)
    r["status"] = "active"
    r["name"] = P("FACILITY", org.name)
    if ctx.chance(0.3):
        r["alias"] = [P("FACILITY", ctx.draw(lambda: ctx.choice(CITY_STEMS) + " " + ctx.choice(("Annex", "Campus", "Pavilion", "Outpatient Center"))))]
    r["mode"] = "instance"
    r["type"] = [cc("http://terminology.hl7.org/CodeSystem/v3-RoleCode", ctx.choice(("HOSP", "PHARM", "OUTPHARM", "ER")))]
    tel = [telecom("phone", "PHONE", org.phone(0), "work")]
    if ctx.chance(0.4):
        tel.append(telecom("fax", "PHONE", org.phone(1), "work"))
    r["telecom"] = tel
    r["address"] = address_json(ctx, org.addr)
    r["physicalType"] = cc("http://terminology.hl7.org/CodeSystem/location-physical-type", ctx.choice(("bu", "wa", "si")))
    if ctx.chance(0.5):
        r["managingOrganization"] = {"display": P("FACILITY", pick_facility(ctx))}
    return r


def r_encounter(ctx: Ctx, pt: Pt | None, org: Org | None, pr: Prac | None) -> dict:
    r: dict = {"resourceType": "Encounter"}
    cls = ctx.choice(ENC_CLASSES)
    if not ctx.phi:
        maybe(r, "meta", meta(ctx), ctx, 0.5)
        r["status"] = ctx.choice(("finished", "in-progress", "planned", "cancelled"))
        r["class"] = coding("http://terminology.hl7.org/CodeSystem/v3-ActCode", *cls)
        r["type"] = [cc("http://snomed.info/sct", "185349003", "Encounter for check up")]
        r["length"] = {"value": ctx.rng.randint(5, 300), "unit": "min", "system": "http://unitsofmeasure.org", "code": "min"}
        r["priority"] = cc("http://terminology.hl7.org/CodeSystem/v3-ActPriority", ctx.choice(("R", "UR", "EL")))
        return r
    r["id"] = P("ID", pick_fid(ctx, "enc"))
    maybe(r, "meta", meta(ctx), ctx, 0.5)
    r["identifier"] = [identifier(ctx, "ID", pick_misc_id(ctx, "V", 7, 10), local_system(ctx, "visit"), "VN", "Visit number", "usual")]
    r["status"] = ctx.choice(("finished", "finished", "in-progress", "planned"))
    r["class"] = coding("http://terminology.hl7.org/CodeSystem/v3-ActCode", *cls)
    r["type"] = [cc("http://snomed.info/sct", ctx.choice(("185349003", "11429006", "308335008")), ctx.choice(("Encounter for check up", "Consultation", "Patient encounter procedure")))]
    r["subject"] = pref(ctx, pt)
    if pr is not None:
        r["participant"] = [{"type": [cc("http://terminology.hl7.org/CodeSystem/v3-ParticipationType", ctx.choice(("ATND", "PPRF", "REF")))],
                             "individual": prac_ref(ctx, pr)}]
    start = encounter_date(ctx)
    period = {"start": fhir_date(ctx, start, ctx.choice(("dt", "dt", "instant")))}
    if ctx.chance(0.7):
        period["end"] = fhir_date(ctx, shift(ctx, start, ctx.rng.randint(0, 9)), "dt")
    r["period"] = period
    if ctx.chance(0.5):
        r["reasonCode"] = [{"text": ctx.choice(("follow-up visit", "annual wellness exam", "chest pain", "medication review", "post-operative check", "shortness of breath"))}]
    if org is not None:
        if ctx.chance(0.6):
            r["location"] = [{"location": {"display": P("FACILITY", org.name)}, "status": "completed"}]
        if ctx.chance(0.6):
            r["serviceProvider"] = org_ref(org)
    return r


def obs_value(ctx: Ctx):
    code, disp, unit, (lo, hi) = ctx.choice(LOINC)
    value = round(ctx.rng.uniform(lo, hi), 1) if hi - lo < 20 else ctx.rng.randint(lo, hi)
    return code, disp, unit, value, lo, hi


def r_observation(ctx: Ctx, pt: Pt | None, org: Org | None, pr: Prac | None, text_value: Rich | None = None,
                  note: list | None = None) -> dict:
    r: dict = {"resourceType": "Observation"}
    code, disp, unit, value, lo, hi = obs_value(ctx)
    if ctx.phi:
        r["id"] = P("ID", pick_fid(ctx, "obs"))
    maybe(r, "meta", meta(ctx), ctx, 0.45)
    r["status"] = ctx.choice(("final", "final", "preliminary", "amended"))
    r["category"] = [cc("http://terminology.hl7.org/CodeSystem/observation-category", ctx.choice(("laboratory", "vital-signs", "exam")))]
    r["code"] = cc("http://loinc.org", code, disp)
    if ctx.phi:
        r["subject"] = pref(ctx, pt)
        d = encounter_date(ctx)
        r["effectiveDateTime"] = fhir_date(ctx, d, "dt")
        if ctx.chance(0.5):
            r["issued"] = fhir_date(ctx, shift(ctx, d, ctx.rng.randint(0, 3)), "instant")
        if pr is not None and ctx.chance(0.5):
            r["performer"] = [prac_ref(ctx, pr)]
        elif org is not None and ctx.chance(0.5):
            r["performer"] = [org_ref(org)]
    if text_value is not None:
        r["valueString"] = text_value
    else:
        r["valueQuantity"] = {"value": value, "unit": unit, "system": "http://unitsofmeasure.org", "code": unit}
        if ctx.chance(0.4):
            r["interpretation"] = [cc("http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", ctx.choice(("N", "H", "L", "A")))]
        if ctx.chance(0.35):
            r["referenceRange"] = [{"low": {"value": lo, "unit": unit, "system": "http://unitsofmeasure.org", "code": unit},
                                    "high": {"value": hi, "unit": unit, "system": "http://unitsofmeasure.org", "code": unit}}]
    if note:
        r["note"] = note
    return r


def r_condition(ctx: Ctx, pt: Pt | None, pr: Prac | None, note: list | None = None) -> dict:
    r: dict = {"resourceType": "Condition"}
    snomed, disp, icd = ctx.choice(CONDITIONS)
    if ctx.phi:
        r["id"] = P("ID", pick_fid(ctx, "cond"))
    maybe(r, "meta", meta(ctx), ctx, 0.4)
    r["clinicalStatus"] = cc("http://terminology.hl7.org/CodeSystem/condition-clinical", ctx.choice(("active", "active", "resolved", "recurrence")))
    r["verificationStatus"] = cc("http://terminology.hl7.org/CodeSystem/condition-ver-status", ctx.choice(("confirmed", "provisional")))
    r["category"] = [cc("http://terminology.hl7.org/CodeSystem/condition-category", "problem-list-item")]
    if ctx.chance(0.4):
        r["severity"] = cc("http://snomed.info/sct", ctx.choice(("255604002", "6736007", "24484000")), ctx.choice(("Mild", "Moderate", "Severe")))
    r["code"] = {"coding": [coding("http://snomed.info/sct", snomed, disp), coding("http://hl7.org/fhir/sid/icd-10-cm", icd)]}
    if ctx.phi:
        r["subject"] = pref(ctx, pt)
        onset = encounter_date(ctx)
        r["onsetDateTime"] = fhir_date(ctx, onset, ctx.choice(("date", "dt")))
        if ctx.chance(0.5):
            r["recordedDate"] = fhir_date(ctx, shift(ctx, onset, ctx.rng.randint(0, 400)) if onset + timedelta(days=400) < date(2026, 8, 1) else onset, "date")
        if pr is not None and ctx.chance(0.5):
            r["asserter"] = prac_ref(ctx, pr)
    if note:
        r["note"] = note
    return r


def r_medication_request(ctx: Ctx, pt: Pt | None, org: Org | None, pr: Prac | None) -> dict:
    r: dict = {"resourceType": "MedicationRequest"}
    code, disp = ctx.choice(MEDS)
    if ctx.phi:
        r["id"] = P("ID", pick_fid(ctx, "mr"))
    maybe(r, "meta", meta(ctx), ctx, 0.4)
    if ctx.phi and ctx.chance(0.6):
        r["identifier"] = [identifier(ctx, "ID", pick_misc_id(ctx, "RX", 6, 9), local_system(ctx, "rx"), "PLAC", "Placer Identifier")]
    r["status"] = ctx.choice(("active", "completed", "stopped", "on-hold"))
    r["intent"] = ctx.choice(("order", "plan", "original-order"))
    r["medicationCodeableConcept"] = cc("http://www.nlm.nih.gov/research/umls/rxnorm", code, disp)
    if ctx.phi:
        r["subject"] = pref(ctx, pt)
        r["authoredOn"] = fhir_date(ctx, encounter_date(ctx), ctx.choice(("dt", "date")))
        if pr is not None:
            r["requester"] = prac_ref(ctx, pr)
    r["dosageInstruction"] = [{"text": ctx.choice(("1 tablet by mouth daily", "1 tablet by mouth twice daily", "take 2 tablets at bedtime", "1 capsule every 8 hours")),
                               "timing": {"repeat": {"frequency": ctx.rng.randint(1, 3), "period": 1, "periodUnit": "d"}}}]
    disp_req: dict = {"quantity": {"value": ctx.choice((30, 60, 90)), "unit": "tablet", "system": "http://unitsofmeasure.org", "code": "{tbl}"},
                      "numberOfRepeatsAllowed": ctx.rng.randint(0, 5)}
    if ctx.phi and org is not None and ctx.chance(0.6):
        disp_req["performer"] = {"display": P("FACILITY", org.name)}
    r["dispenseRequest"] = disp_req
    return r


def r_allergy(ctx: Ctx, pt: Pt | None, pr: Prac | None, note: list | None = None) -> dict:
    r: dict = {"resourceType": "AllergyIntolerance"}
    code, disp = ctx.choice(ALLERGENS)
    if ctx.phi:
        r["id"] = P("ID", pick_fid(ctx, "alg"))
    maybe(r, "meta", meta(ctx), ctx, 0.4)
    r["clinicalStatus"] = cc("http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical", ctx.choice(("active", "inactive", "resolved")))
    r["type"] = ctx.choice(("allergy", "intolerance"))
    r["category"] = [ctx.choice(("medication", "food", "environment"))]
    r["criticality"] = ctx.choice(("low", "high", "unable-to-assess"))
    r["code"] = cc("http://snomed.info/sct", code, disp)
    if ctx.phi:
        r["patient"] = pref(ctx, pt)
        r["recordedDate"] = fhir_date(ctx, encounter_date(ctx), ctx.choice(("dt", "date")))
        if pr is not None and ctx.chance(0.5):
            r["recorder"] = prac_ref(ctx, pr)
    r["reaction"] = [{"manifestation": [cc("http://snomed.info/sct", ctx.choice(("271807003", "39579001", "62315008")), ctx.choice(("Skin rash", "Anaphylaxis", "Diarrhea")))],
                      "severity": ctx.choice(("mild", "moderate", "severe"))}]
    if note:
        r["note"] = note
    return r


def r_immunization(ctx: Ctx, pt: Pt | None, org: Org | None, pr: Prac | None, note: list | None = None) -> dict:
    r: dict = {"resourceType": "Immunization"}
    code, disp = ctx.choice(VACCINES)
    if ctx.phi:
        r["id"] = P("ID", pick_fid(ctx, "imm"))
    maybe(r, "meta", meta(ctx), ctx, 0.4)
    r["status"] = ctx.choice(("completed", "completed", "entered-in-error", "not-done"))
    r["vaccineCode"] = cc("http://hl7.org/fhir/sid/cvx", code, disp)
    if ctx.phi:
        r["patient"] = pref(ctx, pt)
        r["occurrenceDateTime"] = fhir_date(ctx, encounter_date(ctx), ctx.choice(("dt", "date")))
    r["doseQuantity"] = {"value": ctx.choice((0.5, 0.3, 1)), "unit": "mL", "system": "http://unitsofmeasure.org", "code": "mL"}
    if ctx.phi:
        perf = []
        if pr is not None:
            perf.append({"function": cc("http://terminology.hl7.org/CodeSystem/v2-0443", "AP"), "actor": prac_ref(ctx, pr)})
        if perf:
            r["performer"] = perf
        if org is not None and ctx.chance(0.6):
            r["location"] = {"display": P("FACILITY", org.name)}
    if note:
        r["note"] = note
    return r


def doc_title(ctx: Ctx, pt: Pt | None, dtype: str) -> Rich | str:
    if pt is None or not ctx.phi:
        return dtype
    form = ctx.rng.randrange(4)
    when = encounter_date(ctx)
    if form == 0:
        return Rich([dtype + " for ", P("NAME", pt.person.display)])
    if form == 1:
        return Rich([dtype + " - ", P("NAME", pt.person.display), ", ", P("DATE", written_date(ctx, when, 0))])
    if form == 2:
        return Rich([dtype + " ", P("DATE", iso(when)), " (MRN ", P("MRN", pt.mrn()), ")"])
    return Rich([P("NAME", pt.person.display), " ", dtype.lower(), " ", P("DATE", written_date(ctx, when, ctx.rng.choice((1, 4))))])


def r_document_reference(ctx: Ctx, pt: Pt | None, org: Org | None, pr: Prac | None, title=None, desc=None) -> dict:
    r: dict = {"resourceType": "DocumentReference"}
    code, disp = ctx.choice(DOC_TYPES)
    if ctx.phi:
        r["id"] = P("ID", pick_fid(ctx, "doc"))
    maybe(r, "meta", meta(ctx), ctx, 0.4)
    r["status"] = "current"
    r["docStatus"] = ctx.choice(("final", "preliminary", "amended"))
    r["type"] = cc("http://loinc.org", code, disp)
    r["category"] = [cc("http://hl7.org/fhir/us/core/CodeSystem/us-core-documentreference-category", "clinical-note")]
    if ctx.phi:
        r["subject"] = pref(ctx, pt)
        r["date"] = fhir_date(ctx, encounter_date(ctx), "instant")
        if pr is not None:
            r["author"] = [prac_ref(ctx, pr)]
        if org is not None and ctx.chance(0.6):
            r["custodian"] = org_ref(org)
    if desc is not None:
        r["description"] = desc
    att: dict = {"contentType": ctx.choice(("text/plain", "application/pdf", "text/html")), "language": "en"}
    att["title"] = title if title is not None else doc_title(ctx, pt, disp)
    att["size"] = ctx.rng.randint(900, 90000)
    if ctx.phi and ctx.chance(0.3):
        att["creation"] = fhir_date(ctx, encounter_date(ctx), "date")
    r["content"] = [{"attachment": att}]
    return r


# ---- extra resource types used in bundles and exports -----------------------------------

def r_coverage(ctx: Ctx, pt: Pt | None, org: Org | None) -> dict:
    r: dict = {"resourceType": "Coverage"}
    if ctx.phi:
        r["id"] = P("ID", pick_fid(ctx, "cov"))
    r["status"] = ctx.choice(("active", "cancelled", "draft"))
    r["type"] = cc("http://terminology.hl7.org/CodeSystem/v3-ActCode", ctx.choice(("EHCPOL", "PUBLICPOL", "HIP")))
    if ctx.phi:
        r["subscriberId"] = P("ID", pt.member())
        r["beneficiary"] = pref(ctx, pt)
        r["identifier"] = [identifier(ctx, "ID", pick_misc_id(ctx, "GRP", 5, 8), f"https://payer.example.test/group/{ctx.rng.randint(10, 99)}", "FILL", "Filler Identifier")]
        if org is not None:
            r["payor"] = [{"display": P("FACILITY", org.name)}]
        r["period"] = {"start": fhir_date(ctx, encounter_date(ctx), "date")}
    else:
        r["relationship"] = cc("http://terminology.hl7.org/CodeSystem/subscriber-relationship", ctx.choice(("self", "spouse", "child")))
        r["order"] = ctx.rng.randint(1, 3)
    return r


def r_account(ctx: Ctx, pt: Pt | None, org: Org | None) -> dict:
    r: dict = {"resourceType": "Account"}
    if ctx.phi:
        r["id"] = P("ID", pick_fid(ctx, "acct"))
        r["identifier"] = [identifier(ctx, "ACCOUNT", pt.account(), local_system(ctx, "account"), "AN", "Account number")]
    r["status"] = ctx.choice(("active", "inactive"))
    r["type"] = cc("http://terminology.hl7.org/CodeSystem/v3-ActCode", ctx.choice(("PBILLACCT", "ACCTRECEIVABLE")))
    if ctx.phi:
        r["subject"] = [pref(ctx, pt)]
        if org is not None:
            r["owner"] = org_ref(org)
    return r


def r_audit_event(ctx: Ctx, pt: Pt | None, pr: Prac | None) -> dict:
    r: dict = {"resourceType": "AuditEvent"}
    r["type"] = coding("http://dicom.nema.org/resources/ontology/DCM", ctx.choice(("110110", "110112")), ctx.choice(("Patient Record", "Query")))
    r["action"] = ctx.choice(("R", "C", "U", "E"))
    if ctx.phi:
        r["recorded"] = fhir_date(ctx, encounter_date(ctx), "instant")
    r["outcome"] = ctx.choice(("0", "0", "4"))
    agent: dict = {"requestor": True}
    if ctx.phi:
        if pr is not None:
            agent["who"] = {"identifier": {"system": NPI_SYSTEM, "value": P("ID", pr.npi)}, "display": pr.display()}
            agent["name"] = P("NAME", pr.person.display)
        agent["network"] = {"address": P("IP", pick_ip(ctx)), "type": "2"}
    else:
        agent["network"] = {"type": "2"}
    r["agent"] = [agent]
    r["source"] = {"observer": {"display": ctx.choice(("record viewer", "export service", "portal gateway"))}}
    if ctx.phi:
        r["entity"] = [{"what": {"reference": Rich(["Patient/", P("ID", pt.id)])}, "type": coding("http://terminology.hl7.org/CodeSystem/audit-entity-type", "1", "Person")}]
    return r


def r_device(ctx: Ctx, pt: Pt | None) -> dict:
    r: dict = {"resourceType": "Device"}
    if ctx.phi:
        r["id"] = P("ID", pick_fid(ctx, "dev"))
    r["status"] = "active"
    r["deviceName"] = [{"name": ctx.choice(("glucose meter", "insulin pump", "pacemaker", "blood pressure cuff", "CPAP unit")), "type": "user-friendly-name"}]
    if ctx.phi:
        r["serialNumber"] = P("ID", pick_misc_id(ctx, "SN", 7, 10))
        r["patient"] = pref(ctx, pt)
    else:
        r["manufacturer"] = "generic"
        r["modelNumber"] = ctx.choice(("GM-100", "XT-45", "HC-7"))
    return r


def r_procedure(ctx: Ctx, pt: Pt | None, org: Org | None, pr: Prac | None) -> dict:
    r: dict = {"resourceType": "Procedure"}
    if ctx.phi:
        r["id"] = P("ID", pick_fid(ctx, "proc"))
    r["status"] = ctx.choice(("completed", "in-progress"))
    r["code"] = cc("http://snomed.info/sct", ctx.choice(("80146002", "73761001", "232717009")), ctx.choice(("Appendectomy", "Colonoscopy", "Coronary artery bypass graft")))
    if ctx.phi:
        r["subject"] = pref(ctx, pt)
        r["performedDateTime"] = fhir_date(ctx, encounter_date(ctx), "dt")
        if pr is not None:
            r["performer"] = [{"actor": prac_ref(ctx, pr)}]
        if org is not None and ctx.chance(0.6):
            r["location"] = {"display": P("FACILITY", org.name)}
    return r


def r_diagnostic_report(ctx: Ctx, pt: Pt | None, org: Org | None, pr: Prac | None) -> dict:
    r: dict = {"resourceType": "DiagnosticReport"}
    code, disp, *_ = ctx.choice(LOINC)
    if ctx.phi:
        r["id"] = P("ID", pick_fid(ctx, "dr"))
        r["identifier"] = [identifier(ctx, "ID", pick_misc_id(ctx, "AC", 7, 10), local_system(ctx, "accession"), "ACSN", "Accession ID")]
    r["status"] = ctx.choice(("final", "preliminary"))
    r["code"] = cc("http://loinc.org", code, disp)
    if ctx.phi:
        r["subject"] = pref(ctx, pt)
        d = encounter_date(ctx)
        r["effectiveDateTime"] = fhir_date(ctx, d, "dt")
        r["issued"] = fhir_date(ctx, shift(ctx, d, ctx.rng.randint(0, 2)), "instant")
        perf = []
        if org is not None:
            perf.append(org_ref(org))
        if pr is not None and ctx.chance(0.6):
            perf.append(prac_ref(ctx, pr))
        if perf:
            r["performer"] = perf
    return r


def r_composition(ctx: Ctx, pt: Pt, org: Org | None, pr: Prac | None, div: Rich | None) -> dict:
    code, disp = ctx.choice(DOC_TYPES)
    r: dict = {"resourceType": "Composition", "id": P("ID", pick_fid(ctx, "comp")), "status": "final",
               "type": cc("http://loinc.org", code, disp), "subject": pref(ctx, pt),
               "date": fhir_date(ctx, encounter_date(ctx), "dt")}
    if pr is not None:
        r["author"] = [prac_ref(ctx, pr)]
    r["title"] = disp
    if org is not None:
        r["custodian"] = org_ref(org)
    sec: dict = {"title": ctx.choice(("Summary", "History of present illness", "Plan")), "code": cc("http://loinc.org", ctx.choice(("11450-4", "10164-2", "18776-5")))}
    if div is not None:
        sec["text"] = {"status": "generated", "div": div}
    r["section"] = [sec]
    return r


# --------------------------------------------------------------------------------------
# Free text: prose with slots, narratives (XHTML), annotations
# --------------------------------------------------------------------------------------

AGE_VALUES = (90, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109)

PHI_SENTENCES = (
    "{name} was seen in clinic on {date} for a routine follow-up.",
    "Date of birth {dob}; record number {mrn}.",
    "Lives at {addr} and can be reached on {phone}.",
    "Referred by Dr. {doc} at {fac}.",
    "Emergency contact is {relname}, phone {relphone}.",
    "Prescriptions were sent to {pharm}.",
    "{given} reports that the symptoms began around {date}.",
    "Insurance member number {member}; billing account {acct}.",
    "Discharged on {date}; follow up with Dr. {doc} on {date2}.",
    "Please write to {email} or use the portal at {url}.",
    "Moved here from {cityst} last spring.",
    "{family} declined an interpreter and was alone at the visit.",
    "Labs drawn at {fac} were reviewed with {name}.",
    "Seen by Dr. {doc}; the next appointment is {date}.",
    "Identity confirmed with the social security number {ssn}.",
    "Transferred from {fac} on {date}.",
    "Spoke with {relname} by telephone on {date}; call-back number {relphone}.",
    "The family can also be reached at {phone2}.",
    "Mailing address on file: {addr}.",
)
AGE_SENTENCES = (
    "{name} is a {age}-year-old patient who lives at {addr}.",
    "A {age} year old, {given} was admitted from {fac} on {date}.",
    "Age {age}; lives alone in {city}.",
    "{family}, {age}yo, arrived by ambulance on {date}.",
    "Patient is {age} years old and was born {dob}.",
)
NEUTRAL_SENTENCES = (
    "Blood pressure 128/82 mmHg, heart rate 72 per minute.",
    "Continue metformin 500 mg twice daily with meals.",
    "Return in 3 months or sooner if symptoms worsen.",
    "Counselled on diet and regular walking.",
    "Reports a mild cough for 3 days; no fever.",
    "Hemoglobin 13.4 g/dL, within the reference range (12.0-16.0).",
    "Vaccinations are up to date.",
    "Weight stable at 72 kg over 6 months.",
    "Order set 4471 applied at the second visit.",
    "A 67-year-old presented with chest discomfort on exertion.",
    "The 74 y/o was counselled on falls prevention.",
    "She is 88 years old and walks daily.",
    "Oxygen saturation 97% on room air.",
    "The ratio was 12/15 on the second count.",
    "Platelets 53728 per microliter on the repeat.",
    "Protocol 555 applies to the standard dose.",
    "No known drug allergies were recorded in this section.",
    "Review of systems otherwise negative.",
    "The wound is clean, dry and intact.",
    "Chest film shows no acute process.",
    "She has had 2 admissions and 14 clinic visits in the last year.",
    "Sodium 139 mmol/L and potassium 4.1 mmol/L.",
    "Physical therapy twice weekly for 6 weeks.",
)


class Slots:
    """Lazily built PHI pieces for one note; the same slot gives the same value."""

    def __init__(self, ctx: Ctx, pt: Pt, org: Org, pr: Prac):
        self.ctx, self.pt, self.org, self.pr = ctx, pt, org, pr
        self.memo: dict = {}
        self.datefmt = ctx.rng.randrange(7)
        self.dobfmt = ctx.rng.randrange(7)
        self.rel = None

    def _rel(self) -> Person:
        if self.rel is None:
            self.rel = make_person(self.ctx, "related")
        return self.rel

    def get(self, key: str):
        if key not in self.memo:
            self.memo[key] = self._mk(key)
        return self.memo[key]

    def _mk(self, key: str):
        c, pt = self.ctx, self.pt
        if key == "name":
            return P("NAME", pt.person.display)
        if key == "given":
            return P("NAME", pt.person.given[0])
        if key == "family":
            return P("NAME", pt.person.family)
        if key == "dob":
            return P("DATE", written_date(c, pt.dob, self.dobfmt))
        if key == "mrn":
            return P("MRN", pt.mrn())
        if key == "ssn":
            return P("SSN", pt.ssn())
        if key == "phone":
            return P("PHONE", pt.phone(0))
        if key == "phone2":
            return P("PHONE", pt.phone(1))
        if key == "email":
            return P("EMAIL", pt.email())
        if key == "addr":
            return P("ADDRESS", pt.addr().full)
        if key == "street":
            return P("ADDRESS", pt.addr().street)
        if key == "city":
            return P("LOCATION", pt.addr().city)
        if key == "cityst":
            return P("LOCATION", f"{pt.addr().city}, {pt.addr().state}")
        if key == "zip":
            return P("LOCATION", pt.addr().zip)
        if key == "fac":
            return P("FACILITY", self.org.name)
        if key == "doc":
            return P("NAME", self.pr.person.display)
        if key == "pharm":
            return P("FACILITY", c.draw(lambda: c.choice(CITY_STEMS) + c.choice(CITY_ENDS) + " " + c.choice(("Pharmacy", "Drug", "Apothecary"))))
        if key == "relname":
            return P("NAME", self._rel().display)
        if key == "relphone":
            return P("PHONE", pick_phone(c))
        if key in ("date", "date2", "date3"):
            return P("DATE", written_date(c, encounter_date(c), self.datefmt))
        if key == "age":
            words = {90: "ninety", 95: "ninety-five", 96: "ninety-six", 97: "ninety-seven", 98: "ninety-eight", 99: "ninety-nine"}
            return P("AGE", words[pt.age] if pt.age in words and c.chance(0.3) else str(pt.age))
        if key == "member":
            return P("ID", pt.member())
        if key == "acct":
            return P("ACCOUNT", pt.account())
        if key == "url":
            return P("URL", pt.portal())
        raise KeyError(key)


def fill(template: str, S: Slots) -> list:
    parts: list = []
    for i, piece in enumerate(re.split(r"\{(\w+)\}", template)):
        if i % 2 == 0:
            if piece:
                parts.append(piece)
        else:
            parts.append(S.get(piece))
    return parts


def over89_age(ctx: Ctx) -> int:
    return int(ctx.draw(lambda: str(ctx.choice(AGE_VALUES))))


def sentences(ctx: Ctx, S: Slots | None, n: int, neutral: int = 1, age_ok: bool = False) -> list[list]:
    """n PHI sentences (or none when S is None) mixed with neutral ones, as lists of pieces."""
    out: list[list] = []
    if S is None:
        neutral = max(neutral, n)
    if S is not None:
        pool = list(PHI_SENTENCES) + (list(AGE_SENTENCES) * 2 if age_ok and S.pt.age > 89 else [])
        if age_ok and S.pt.age > 89:
            out.append(fill(ctx.choice(AGE_SENTENCES), S))
            n -= 1
        for t in ctx.rng.sample(pool, k=min(n, len(pool))):
            if "{age}" in t:
                continue
            out.append(fill(t, S))
    for t in ctx.rng.sample(NEUTRAL_SENTENCES, k=neutral):
        out.append([t])
    ctx.rng.shuffle(out)
    return out


def join_sentences(sents: list[list]) -> list:
    parts: list = []
    for i, s in enumerate(sents):
        if i:
            parts.append(" ")
        parts.extend(s)
    return parts


def merge(parts: list) -> Rich:
    """Merge adjacent literals; keep markup pieces (starting with '<') as separate literals."""
    result: list = []
    for part in parts:
        if isinstance(part, str) and result and isinstance(result[-1], str) and not part.startswith("<") and not result[-1].startswith("<"):
            result[-1] += part
        else:
            result.append(part)
    return Rich(result)


NS_DIV = '<div xmlns="http://www.w3.org/1999/xhtml">'
NS_HTML = '<html xmlns="http://www.w3.org/1999/xhtml">'


def table_rows(ctx: Ctx, S: Slots | None) -> list[tuple[str, object]]:
    if S is None:
        rows = [("Status", "final"), ("Heart rate", "72 /min"), ("Dose", "500 mg twice daily"), ("Batch size", "4471"),
                ("Reference range", "70-99 mg/dL"), ("Category", "laboratory"), ("Course", "6 weeks"),
                ("Count", "53728 per uL"), ("Ratio", "12/15"), ("Intent", "order"), ("Priority", "routine")]
        return ctx.rng.sample(rows, k=ctx.rng.randint(4, 7))
    rows = [("Name", S.get("name")), ("Date of birth", S.get("dob")), ("MRN", S.get("mrn")), ("Address", S.get("addr")),
            ("Phone", S.get("phone")), ("Email", S.get("email")), ("Facility", S.get("fac")), ("Admitted", S.get("date")),
            ("Member ID", S.get("member")), ("Account", S.get("acct")), ("City", S.get("city")), ("ZIP", S.get("zip"))]
    chosen = ctx.rng.sample(rows, k=ctx.rng.randint(4, 8))
    if ctx.chance(0.4):
        chosen.append(("Status", "final"))
    ctx.rng.shuffle(chosen)
    return chosen


def xhtml_body(ctx: Ctx, S: Slots | None, style: int, age_ok: bool = False) -> list:
    """Inner XHTML pieces for a narrative: paragraphs, a table or a list."""
    sep = ctx.choice(("", "\n", "\n  "))
    sents = sentences(ctx, S, ctx.rng.randint(2, 4), neutral=ctx.rng.randint(1, 2), age_ok=age_ok)
    parts: list = []
    if style == 0:
        for i, s in enumerate(sents):
            if i:
                parts.append(sep)
            parts.extend(["<p>", *s, "</p>"])
    elif style == 1:
        parts.append("<table>")
        if ctx.chance(0.4):
            parts.extend([sep, "<thead><tr><th>Field</th><th>Value</th></tr></thead>"])
        parts.extend([sep, "<tbody>"])
        for label, value in table_rows(ctx, S):
            parts.extend([sep, "<tr><td>", label, "</td><td>", value, "</td></tr>"])
        parts.extend([sep, "</tbody>", "</table>"])
        if sents and ctx.chance(0.5):
            parts.extend([sep, "<p>", *sents[0], "</p>"])
    else:
        parts.extend(["<h2>", ctx.choice(("Summary", "Visit note", "Care notes", "Overview")), "</h2>", sep, "<ul>"])
        for s in sents:
            parts.extend([sep, "<li>", *s, "</li>"])
        parts.extend([sep, "</ul>"])
    return parts


def xhtml_div(ctx: Ctx, S: Slots | None, style: int, age_ok: bool = False) -> Rich:
    return merge([NS_DIV, *xhtml_body(ctx, S, style, age_ok), "</div>"])


def annotations(ctx: Ctx, S: Slots | None, n: int = 1) -> list:
    notes = []
    for _ in range(n):
        note: dict = {}
        if S is not None:
            if ctx.chance(0.5):
                note["authorString"] = S.pr.display()
            else:
                note["authorReference"] = {"display": S.pr.display()}
            note["time"] = fhir_date(ctx, encounter_date(ctx), "dt")
        note["text"] = merge(join_sentences(sentences(ctx, S, ctx.rng.randint(1, 3), neutral=ctx.rng.randint(1, 2), age_ok=True)))
        notes.append(note)
    return notes


def insert_after(d: dict, after: str, key: str, value) -> dict:
    result: dict = {}
    placed = False
    for k, v in d.items():
        result[k] = v
        if k == after:
            result[key] = value
            placed = True
    if not placed:
        result[key] = value
    return result


def with_text_div(r: dict, div: Rich) -> dict:
    anchor = "meta" if "meta" in r else ("id" if "id" in r else "resourceType")
    return insert_after(r, anchor, "text", {"status": "generated", "div": div})


# --------------------------------------------------------------------------------------
# Key-plus-value rows
# --------------------------------------------------------------------------------------

def flatten(obj, path: str = ""):
    if isinstance(obj, Rich):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{path}[{i}]")
    elif obj is not None:
        yield path, obj


def value_parts(v) -> list:
    if isinstance(v, P):
        return [v]
    if isinstance(v, Rich):
        return list(v)
    if v is True:
        return ["true"]
    if v is False:
        return ["false"]
    return [str(v)]


def has_phi(v) -> bool:
    return isinstance(v, P) or (isinstance(v, Rich) and any(isinstance(p, P) for p in v))


def kv_recase(parts: list, mode: str) -> list:
    if mode == "normal":
        return parts
    fn = str.upper if mode == "upper" else str.lower
    out = []
    for p in parts:
        if isinstance(p, P):
            out.append(p if p.label in ("URL", "EMAIL", "IP") else P(p.label, fn(p.value)))
        elif "/" in p or ":" in p:
            out.append(p)
        else:
            out.append(fn(p))
    return out


def short_key(path: str) -> str:
    return re.sub(r"\[\d+\]", "", path.rsplit(".", 1)[-1])


def idx_style(path: str, style: str) -> str:
    return re.sub(r"\[(\d+)\]", (lambda m: f".{m.group(1)}") if style == "dot" else (lambda m: m.group(0)), path)


def yaml_write(out: Out, d: dict, level: int, mode: str) -> None:
    for k, v in d.items():
        pad = "  " * level
        if isinstance(v, dict):
            out.lit(f"{pad}{k}:\n")
            yaml_write(out, v, level + 1, mode)
        elif isinstance(v, (list, tuple)) and not isinstance(v, Rich):
            out.lit(f"{pad}{k}:\n")
            for item in v:
                if isinstance(item, dict):
                    first = True
                    for kk, vv in item.items():
                        lead = "  " * level + ("  - " if first else "    ")
                        first = False
                        if isinstance(vv, (dict, list, tuple)) and not isinstance(vv, Rich):
                            out.lit(f"{lead}{kk}:\n")
                            yaml_write_value(out, vv, level + 3, mode)
                        else:
                            out.lit(f"{lead}{kk}: ")
                            write_parts(out, kv_recase(value_parts(vv), mode))
                            out.lit("\n")
                else:
                    out.lit(f"{'  ' * level}  - ")
                    write_parts(out, kv_recase(value_parts(item), mode))
                    out.lit("\n")
        else:
            out.lit(f"{pad}{k}: ")
            write_parts(out, kv_recase(value_parts(v), mode))
            out.lit("\n")


def yaml_write_value(out: Out, v, level: int, mode: str) -> None:
    if isinstance(v, dict):
        yaml_write(out, v, level, mode)
    elif isinstance(v, (list, tuple)) and not isinstance(v, Rich):
        for item in v:
            if isinstance(item, dict):
                first = True
                for kk, vv in item.items():
                    lead = "  " * (level - 1) + ("  - " if first else "    ")
                    first = False
                    out.lit(f"{lead}{kk}: ")
                    write_parts(out, kv_recase(value_parts(vv), mode) if (isinstance(vv, Rich) or not isinstance(vv, (dict, list))) else ["..."])
                    out.lit("\n")
            else:
                out.lit(f"{'  ' * (level - 1)}  - ")
                write_parts(out, kv_recase(value_parts(item), mode))
                out.lit("\n")


def write_parts(out: Out, parts: list) -> None:
    for p in parts:
        if isinstance(p, P):
            out.phi(p.label, p.value)
        else:
            out.lit(p)


def kv_render(ctx: Ctx, resource: dict, style: str, mode: str, phi: bool) -> Out:
    out = Out()
    rtype = resource.get("resourceType", "Resource")
    if style == "yaml":
        keys = [k for k in resource if k not in ("resourceType",)]
        chosen = set(ctx.rng.sample(keys, k=min(len(keys), ctx.rng.randint(3, 8))))
        if phi:
            phi_keys = [k for k in keys if any(has_phi(v) for _, v in flatten(resource[k]))]
            chosen |= set(ctx.rng.sample(phi_keys, k=min(len(phi_keys), 2)))
        sub = {k: v for k, v in resource.items() if k in chosen}
        if ctx.chance(0.5):
            sub = {"resourceType": rtype, **sub}
        yaml_write(out, sub, 0, mode)
        if not out.parts:
            raise Reject("empty yaml")
        return out
    leaves = list(flatten(resource))
    phi_idx = [i for i, (_, v) in enumerate(leaves) if has_phi(v)]
    if style == "single":
        pool = phi_idx if phi else list(range(len(leaves)))
        chosen = {ctx.choice(pool)}
        lines = 1
    else:
        lines = ctx.rng.randint(4, 14)
        chosen = set(ctx.rng.sample(range(len(leaves)), k=min(lines, len(leaves))))
        if phi:
            by_label: dict[str, list[int]] = {}
            for i in phi_idx:
                first = next(p for p in value_parts(leaves[i][1]) if isinstance(p, P))
                by_label.setdefault(first.label, []).append(i)
            labels = sorted(by_label)
            ctx.rng.shuffle(labels)
            for label in labels[:ctx.rng.randint(2, 4)]:
                chosen.add(ctx.choice(by_label[label]))
    qprefix = ctx.choice((f"{rtype}.", "")) if style != "single" else ctx.choice((f"{rtype}.", ""))
    index_style = ctx.choice(("bracket", "dot"))
    sep_choice = ctx.choice((": ", ":", " = ", "="))
    for i in sorted(chosen):
        path, value = leaves[i]
        parts = kv_recase(value_parts(value), mode)
        if style == "colon":
            out.lit(f"{short_key(path)}: ")
        elif style == "path_eq":
            out.lit(f"{qprefix}{idx_style(path, index_style)}{ctx.choice(('=', '=')) if ctx.chance(0.7) else ' = '}")
        elif style == "tsv":
            out.lit(f"{qprefix}{idx_style(path, index_style)}\t")
        elif style == "quoted":
            out.lit(f"{qprefix}{idx_style(path, index_style)} = ")
            out.lit('"')
            write_parts(out, parts)
            out.lit('"\n')
            continue
        elif style == "single":
            out.lit(f"{short_key(path) if ctx.chance(0.5) else qprefix + idx_style(path, index_style)}{sep_choice}")
        write_parts(out, parts)
        out.lit("\n")
    return out


# --------------------------------------------------------------------------------------
# Decoy documents (no PHI at all)
# --------------------------------------------------------------------------------------

def oid(ctx: Ctx) -> str:
    return f"{ctx.choice(OID_ROOTS)}.{ctx.rng.randint(1, 999)}.{ctx.rng.randint(1, 99)}" + (f".{ctx.rng.randint(1, 9)}" if ctx.chance(0.5) else "")


def d_codings(ctx: Ctx) -> dict:
    code, disp, unit, _ = ctx.choice(LOINC)
    snomed, sdisp, icd = ctx.choice(CONDITIONS)
    r = {"resourceType": "Observation", "status": ctx.choice(("final", "amended")),
         "code": {"coding": [coding("http://loinc.org", code, disp), coding("http://snomed.info/sct", snomed, sdisp),
                             coding(f"urn:oid:{oid(ctx)}", digits(ctx.rng, ctx.rng.randint(4, 8)))], "text": disp},
         "valueCodeableConcept": {"coding": [coding("http://hl7.org/fhir/sid/icd-10-cm", icd), coding("http://snomed.info/sct", snomed)]},
         "interpretation": [cc("http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", ctx.choice(("N", "H", "L")))]}
    if ctx.chance(0.5):
        r["method"] = cc(f"urn:oid:{oid(ctx)}", digits(ctx.rng, 5))
    return r


def d_oids(ctx: Ctx) -> dict:
    uid = [{"type": "oid", "value": oid(ctx), "preferred": True}, {"type": "uri", "value": f"urn:oid:{oid(ctx)}"}]
    r: dict = {"resourceType": "NamingSystem", "name": ctx.choice(("Identifier_Namespace", "Local_Record_Numbers", "Billing_Accounts", "Visit_Numbers")),
               "status": "active", "kind": ctx.choice(("identifier", "codesystem")), "uniqueId": uid}
    if ctx.chance(0.6):
        r["identifierSystemExample"] = [{"system": f"urn:oid:{oid(ctx)}", "use": "usual"}, {"system": ctx.choice(SYSTEMS), "use": "secondary"}]
    if ctx.chance(0.4):
        r["version"] = f"{ctx.rng.randint(1, 9)}.{ctx.rng.randint(0, 20)}.{ctx.rng.randint(0, 9)}"
    return r


def d_quantities(ctx: Ctx) -> dict:
    comps = []
    for _ in range(ctx.rng.randint(3, 5)):
        code, disp, unit, (lo, hi) = ctx.choice(LOINC)
        value = ctx.choice((ctx.rng.randint(lo, hi), round(ctx.rng.uniform(lo, hi), 1), 53728, 4471, 5550142, 12, 15))
        comps.append({"code": cc("http://loinc.org", code, disp), "valueQuantity": {"value": value, "unit": unit, "system": "http://unitsofmeasure.org", "code": unit}})
    r = {"resourceType": "Observation", "status": "final", "code": {"text": "panel"}, "component": comps}
    if ctx.chance(0.5):
        r["valueRatio"] = {"numerator": {"value": 12, "unit": "count"}, "denominator": {"value": 15, "unit": "count"}}
    if ctx.chance(0.5):
        r["valueString"] = ctx.choice(("titer 1:64", "12/15", "pH 7.35", "4.5 x 10^3/uL", "grade 2", "grade 3"))
    return r


def d_enums(ctx: Ctx) -> dict:
    return {"resourceType": ctx.choice(("Patient", "Encounter", "MedicationRequest", "Practitioner", "Task")),
            "status": ctx.choice(("active", "finished", "completed", "draft")), "intent": ctx.choice(("order", "plan", "proposal")),
            "priority": ctx.choice(("routine", "urgent", "stat")), "gender": ctx.choice(("female", "male", "other", "unknown")),
            "active": ctx.chance(0.5), "multipleBirthBoolean": ctx.chance(0.3), "deceasedBoolean": False,
            "name": [{"use": ctx.choice(("official", "usual", "anonymous", "old"))}], "maritalStatus": {"coding": [coding(V3_MARITAL, ctx.choice(("M", "S", "D", "W", "U")))]},
            "address": [{"use": ctx.choice(("home", "work")), "type": ctx.choice(("postal", "physical", "both")), "state": ctx.choice(US_STATES), "country": "US"}]}


def d_meta(ctx: Ctx) -> dict:
    m = meta(ctx)
    m["tag"] = [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationValue", "code": ctx.choice(("SUBSETTED", "AGGREGATED", "REDACTED"))}]
    r = {"resourceType": ctx.choice(("Bundle", "ValueSet", "CodeSystem", "Parameters")), "meta": m}
    if r["resourceType"] == "Bundle":
        r["type"] = ctx.choice(("collection", "searchset", "batch-response"))
        r["total"] = ctx.rng.randint(0, 250)
    if r["resourceType"] in ("ValueSet", "CodeSystem"):
        r["status"] = "active"
        r["url"] = f"http://hl7.org/fhir/ValueSet/{ctx.choice(('administrative-gender', 'observation-status', 'encounter-status', 'condition-clinical'))}"
        r["version"] = f"{ctx.rng.randint(1, 5)}.{ctx.rng.randint(0, 9)}.{ctx.rng.randint(0, 9)}"
    return r


def d_extensions(ctx: Ctx) -> dict:
    exts = []
    for url in ctx.rng.sample(EXT_URLS, k=ctx.rng.randint(2, 4)):
        value = ctx.choice(({"valueCode": ctx.choice(("M", "F", "UNK", "ASKU"))}, {"valueBoolean": ctx.chance(0.5)},
                            {"valueString": ctx.choice(("not asked", "declined", "none recorded"))},
                            {"valueInteger": ctx.rng.randint(0, 9)}, {"valueCoding": coding("urn:oid:2.16.840.1.113883.6.238", ctx.choice(("2106-3", "2054-5", "2028-9")))}))
        exts.append({"url": url, **value})
    return {"resourceType": ctx.choice(("Patient", "Encounter", "Practitioner")), "extension": exts, "active": True}


# --------------------------------------------------------------------------------------
# Layout builders. Each returns (Out, format, resource types).
# --------------------------------------------------------------------------------------

def actors(ctx: Ctx, age: int | None = None):
    if not ctx.phi:
        return None, None, None
    return Pt(ctx, age=age), Org(ctx), Prac(ctx)


def make_resource(ctx: Ctx, rtype: str, pt, org, pr) -> dict:
    if rtype == "Patient":
        return r_patient(ctx, pt, org, pr)
    if rtype == "RelatedPerson":
        return r_related_person(ctx, pt)
    if rtype == "Practitioner":
        return r_practitioner(ctx, pr)
    if rtype == "Organization":
        return r_organization(ctx, org)
    if rtype == "Location":
        return r_location(ctx, org)
    if rtype == "Encounter":
        return r_encounter(ctx, pt, org, pr)
    if rtype == "Observation":
        return r_observation(ctx, pt, org, pr)
    if rtype == "Condition":
        return r_condition(ctx, pt, pr)
    if rtype == "MedicationRequest":
        return r_medication_request(ctx, pt, org, pr)
    if rtype == "AllergyIntolerance":
        return r_allergy(ctx, pt, pr)
    if rtype == "Immunization":
        return r_immunization(ctx, pt, org, pr)
    if rtype == "DocumentReference":
        return r_document_reference(ctx, pt, org, pr)
    if rtype == "Coverage":
        return r_coverage(ctx, pt, org)
    if rtype == "Account":
        return r_account(ctx, pt, org)
    if rtype == "AuditEvent":
        return r_audit_event(ctx, pt, pr)
    if rtype == "Device":
        return r_device(ctx, pt)
    if rtype == "Procedure":
        return r_procedure(ctx, pt, org, pr)
    if rtype == "DiagnosticReport":
        return r_diagnostic_report(ctx, pt, org, pr)
    raise KeyError(rtype)


SINGLE_TYPES = ("Patient", "RelatedPerson", "Practitioner", "Organization", "Location", "Encounter", "Observation",
                "Condition", "MedicationRequest", "AllergyIntolerance", "Immunization", "DocumentReference")
LINKED_TYPES = ("Encounter", "Observation", "Condition", "MedicationRequest", "AllergyIntolerance", "Immunization",
                "DocumentReference", "Coverage", "Procedure", "DiagnosticReport", "Device", "Account")


def pretty_style(ctx: Ctx) -> Style:
    return Style("pretty", ctx.choice(("  ", "  ", "  ", "    ", "\t")))


def build_single(rtype: str):
    def build(ctx: Ctx):
        pt, org, pr = actors(ctx)
        r = make_resource(ctx, rtype, pt, org, pr)
        return render_json(r, pretty_style(ctx)), "json", [rtype]
    return build


def entry(ctx: Ctx, rtype: str, r: dict, mode: str) -> dict:
    e: dict = {}
    rid = r.get("id")
    if isinstance(rid, P):
        e["fullUrl"] = P("URL", pick_url_host_path(ctx, rtype, rid.value))
    e["resource"] = r
    if mode == "tx":
        method = ctx.choice(("POST", "PUT")) if isinstance(rid, P) else "POST"
        if method == "PUT":
            e["request"] = {"method": "PUT", "url": Rich([rtype + "/", P("ID", rid.value)])}
        else:
            e["request"] = {"method": "POST", "url": rtype}
    elif mode == "search":
        e["search"] = {"mode": "match"}
    return e


def bundle_resources(ctx: Ctx, pt, org, pr, lo: int = 1, hi: int = 3, first: str | None = "Patient") -> list[tuple[str, dict]]:
    pool = list(LINKED_TYPES[:9]) + ["Procedure", "DiagnosticReport"]
    items: list[tuple[str, dict]] = []
    if first:
        items.append((first, make_resource(ctx, first, pt, org, pr)))
    for rtype in ctx.rng.sample(pool, k=ctx.rng.randint(lo, hi)):
        items.append((rtype, make_resource(ctx, rtype, pt, org, pr)))
    if ctx.phi and ctx.chance(0.35):
        items.append(("Practitioner", r_practitioner(ctx, pr)))
    if ctx.phi and ctx.chance(0.3):
        items.append(("Organization", r_organization(ctx, org)))
    return items


def build_bundle(kind: str, style_kind: str):
    def build(ctx: Ctx):
        pt, org, pr = actors(ctx)
        if kind == "document":
            if ctx.phi:
                div = xhtml_div(ctx, Slots(ctx, pt, org, pr), ctx.choice((0, 2)), age_ok=False) if ctx.chance(0.6) else None
                comp = r_composition(ctx, pt, org, pr, div)
                items = [("Composition", comp), ("Patient", r_patient(ctx, pt, org, pr))]
                for rtype in ctx.rng.sample(("Encounter", "Observation", "Condition", "MedicationRequest", "AllergyIntolerance"), k=ctx.rng.randint(1, 2)):
                    items.append((rtype, make_resource(ctx, rtype, pt, org, pr)))
                if ctx.chance(0.5):
                    items.append(("Practitioner", r_practitioner(ctx, pr)))
                if ctx.chance(0.4):
                    items.append(("Organization", r_organization(ctx, org)))
            else:
                items = [("Composition", {"resourceType": "Composition", "status": "final", "type": cc("http://loinc.org", *ctx.choice(DOC_TYPES)),
                                          "title": ctx.choice(("Summary", "Plan")), "section": [{"title": "Plan", "text": {"status": "generated", "div": NS_DIV + "<p>" + ctx.choice(NEUTRAL_SENTENCES) + "</p></div>"}}]}),
                         ("Patient", r_patient(ctx, None, None, None)), (ctx.choice(SINGLE_TYPES[5:]), None)]
                items[2] = (items[2][0], make_resource(ctx, items[2][0], None, None, None))
            mode, btype = "plain", "document"
        else:
            items = bundle_resources(ctx, pt, org, pr, 1, 3, first="Patient" if ctx.chance(0.8) else None)
            mode = "tx" if kind == "transaction" else "search"
            btype = "transaction" if kind == "transaction" else "searchset"
        b: dict = {"resourceType": "Bundle"}
        if ctx.chance(0.4):
            b["meta"] = {"lastUpdated": f"{ctx.rng.randint(2022, 2026)}-{ctx.rng.randint(1, 12):02d}-{ctx.rng.randint(1, 28):02d}T{ctx.rng.randint(0, 23):02d}:{ctx.rng.randint(0, 59):02d}:00Z"}
        b["type"] = btype
        if ctx.chance(0.5):
            b["timestamp"] = f"{ctx.rng.randint(2023, 2026)}-{ctx.rng.randint(1, 12):02d}-{ctx.rng.randint(1, 28):02d}T{ctx.rng.randint(0, 23):02d}:{ctx.rng.randint(0, 59):02d}:00Z"
        if kind == "searchset":
            b["total"] = len(items)
        b["entry"] = [entry(ctx, rtype, r, mode) for rtype, r in items]
        style = pretty_style(ctx) if style_kind == "pretty" else Style("min")
        return render_json(b, style), "json", ["Bundle"] + [t for t, _ in items]
    return build


def build_ndjson(kind: str):
    def build(ctx: Ctx):
        style = Style("min") if ctx.chance(0.7) else Style("flat")
        lines: list[tuple[str, dict]] = []
        pt, org, pr = actors(ctx)
        n = ctx.rng.randint(3, 5)
        if kind == "patients":
            for _ in range(n):
                if ctx.phi:
                    p2, o2, r2 = Pt(ctx), Org(ctx), Prac(ctx)
                else:
                    p2 = o2 = r2 = None
                lines.append(("Patient", r_patient(ctx, p2, o2, r2)))
        elif kind == "observations":
            for _ in range(n):
                lines.append(("Observation", r_observation(ctx, pt, org, pr)))
        elif kind == "mixed":
            lines.append(("Patient", r_patient(ctx, pt, org, pr)))
            for rtype in ctx.rng.sample(("Encounter", "Condition", "MedicationRequest", "AllergyIntolerance", "Immunization", "Observation", "Procedure"), k=n):
                lines.append((rtype, make_resource(ctx, rtype, pt, org, pr)))
        else:
            for _ in range(n):
                if ctx.phi and ctx.chance(0.5):
                    pt, pr = Pt(ctx), Prac(ctx)
                lines.append(("AuditEvent", r_audit_event(ctx, pt, pr)))
        out = Out()
        for i, (_, r) in enumerate(lines):
            if i:
                out.lit("\n")
            write_json(out, r, style)
        return out, "ndjson", sorted({t for t, _ in lines})
    return build


def make_slots(ctx: Ctx, want_age: bool = False):
    pt, org, pr = actors(ctx, age=over89_age(ctx) if (want_age and ctx.phi) else None)
    S = Slots(ctx, pt, org, pr) if ctx.phi else None
    return pt, org, pr, S


def build_narr_div(style: int):
    def build(ctx: Ctx):
        pt, org, pr, S = make_slots(ctx, want_age=ctx.chance(0.5))
        div = recase(xhtml_div(ctx, S, style, age_ok=True), ctx.case)
        out = Out()
        write_rich_raw(out, div)
        return out, "xhtml", ["XhtmlNarrative"]
    return build


def build_narr_resource_div(rtype: str):
    def build(ctx: Ctx):
        pt, org, pr, S = make_slots(ctx, want_age=ctx.chance(0.25))
        r = make_resource(ctx, rtype, pt, org, pr)
        div = recase(xhtml_div(ctx, S, ctx.choice((0, 0, 1, 2)), age_ok=True), ctx.case)
        return render_json(with_text_div(r, div), pretty_style(ctx)), "json", [rtype]
    return build


def build_narr_note(rtype: str):
    def build(ctx: Ctx):
        pt, org, pr, S = make_slots(ctx, want_age=ctx.chance(0.5))
        notes = annotations(ctx, S, ctx.rng.randint(1, 2))
        for n in notes:
            n["text"] = recase(n["text"], ctx.case)
        if rtype == "Condition":
            r = r_condition(ctx, pt, pr, note=notes)
        elif rtype == "Immunization":
            r = r_immunization(ctx, pt, org, pr, note=notes)
        elif rtype == "AllergyIntolerance":
            r = r_allergy(ctx, pt, pr, note=notes)
        else:
            r = r_observation(ctx, pt, org, pr, note=notes)
        return render_json(r, pretty_style(ctx)), "json", [rtype]
    return build


def build_narr_valuestring(ctx: Ctx):
    pt, org, pr, S = make_slots(ctx, want_age=ctx.chance(0.5))
    text = recase(merge(join_sentences(sentences(ctx, S, ctx.rng.randint(1, 3), neutral=ctx.rng.randint(1, 2), age_ok=True))), ctx.case)
    r = r_observation(ctx, pt, org, pr, text_value=text)
    return render_json(r, pretty_style(ctx)), "json", ["Observation"]


def build_narr_docref(ctx: Ctx):
    pt, org, pr, S = make_slots(ctx, want_age=ctx.chance(0.2))
    desc = recase(merge(join_sentences(sentences(ctx, S, ctx.rng.randint(1, 2), neutral=1, age_ok=True))), ctx.case)
    r = r_document_reference(ctx, pt, org, pr, desc=desc)
    if ctx.case != "normal" and isinstance(r["content"][0]["attachment"]["title"], Rich):
        r["content"][0]["attachment"]["title"] = recase(r["content"][0]["attachment"]["title"], ctx.case)
    return render_json(r, pretty_style(ctx)), "json", ["DocumentReference"]


def build_narr_decoded(ctx: Ctx):
    pt, org, pr, S = make_slots(ctx, want_age=ctx.chance(0.5))
    title = ctx.choice(("Discharge Summary", "Consult Note", "Progress Note", "History and Physical"))
    parts: list = [NS_HTML, "<head><title>", title, "</title></head>", ctx.choice(("", "\n")), "<body>", "<h1>", title, "</h1>"]
    parts.extend(xhtml_body(ctx, S, ctx.choice((0, 1, 2)), age_ok=True))
    if S is not None and ctx.chance(0.6):
        parts.extend(["<p>Signed by Dr. ", S.get("doc"), "</p>"])
    parts.extend(["</body>", "</html>"])
    out = Out()
    write_rich_raw(out, recase(merge(parts), ctx.case))
    return out, "xhtml", ["XhtmlDocument"]


def build_kv(style: str):
    def build(ctx: Ctx):
        pt, org, pr = actors(ctx)
        kinds = SINGLE_TYPES + ("Coverage", "Account", "Device")
        rtype = ctx.rng.choices(kinds, [6 if k == "Patient" else 1 for k in kinds])[0]
        r = make_resource(ctx, rtype, pt, org, pr)
        return kv_render(ctx, r, style, ctx.case, ctx.phi), "key_value", [rtype]
    return build


DECOYS = {
    "codings": d_codings, "oid_systems": d_oids, "quantities": d_quantities, "enums": d_enums,
    "meta_versions": d_meta, "extension_urls": d_extensions,
}


def build_decoy(kind: str):
    def build(ctx: Ctx):
        r = DECOYS[kind](ctx)
        if ctx.chance(0.3):
            style = Style("min")
        elif ctx.chance(0.2):
            style = Style("flat")
        else:
            style = pretty_style(ctx)
        return render_json(r, style), "json", [r["resourceType"]]
    return build


def build_decoy_kv(ctx: Ctx):
    r = ctx.choice((d_codings, d_oids, d_meta, d_extensions, d_enums, d_quantities))(ctx)
    return kv_render(ctx, r, ctx.choice(("colon", "path_eq", "tsv", "quoted", "yaml")), "normal", False), "key_value", [r["resourceType"]]


class Layout:
    def __init__(self, name, family, count, neg, build, cased=False):
        self.name, self.family, self.count, self.neg, self.build, self.cased = name, family, count, neg, build, cased


def make_layouts() -> list[Layout]:
    layouts: list[Layout] = []
    for i, rtype in enumerate(SINGLE_TYPES):
        layouts.append(Layout(f"single_{rtype.lower()}_pretty", "single", 39 if i < 6 else 38, 0.24, build_single(rtype)))
    for name, kind, style in (
        ("bundle_transaction_pretty", "transaction", "pretty"), ("bundle_transaction_minified", "transaction", "min"),
        ("bundle_searchset_pretty", "searchset", "pretty"), ("bundle_searchset_minified", "searchset", "min"),
        ("bundle_document_pretty", "document", "pretty"),
    ):
        layouts.append(Layout(name, "bundle", 41, 0.24, build_bundle(kind, style)))
    for kind in ("patients", "observations", "mixed", "audit"):
        layouts.append(Layout(f"ndjson_{kind}", "bundle", 41, 0.24, build_ndjson(kind)))
    for name, build in (
        ("narrative_div_paragraphs", build_narr_div(0)), ("narrative_div_table", build_narr_div(1)),
        ("narrative_div_list", build_narr_div(2)), ("narrative_patient_text_div", build_narr_resource_div("Patient")),
        ("narrative_encounter_text_div", build_narr_resource_div("Encounter")),
        ("narrative_condition_note", build_narr_note("Condition")), ("narrative_observation_valuestring", build_narr_valuestring),
        ("narrative_immunization_annotation", build_narr_note("Immunization")), ("narrative_documentreference_title", build_narr_docref),
        ("narrative_decoded_attachment", build_narr_decoded),
    ):
        layouts.append(Layout(name, "narrative", 30, 0.25, build, cased=True))
    for style in ("colon", "path_eq", "tsv", "quoted", "yaml", "single"):
        layouts.append(Layout(f"kv_{style}", "kv", 37, 0.33, build_kv(style), cased=True))
    for kind in DECOYS:
        layouts.append(Layout(f"decoy_{kind}", "decoy", 21, 1.0, build_decoy(kind)))
    layouts.append(Layout("decoy_kv_codes", "decoy", 21, 1.0, build_decoy_kv))
    return layouts


# --------------------------------------------------------------------------------------
# Rows: checks, metadata, assembly
# --------------------------------------------------------------------------------------

BANNED_METADATA = ("development", "validation", "test", "held-out", "heldout", "proxy")
KV_LINE = re.compile(r"^\s*(?:- )?[A-Za-z_][\w.\[\]]*\s*(?::|=|\t)|^\s*- \S")


def fixture_span_texts() -> frozenset[str]:
    import importlib

    texts: set[str] = set()
    for name in ("annotation_policy", "annotation_policy_places", "annotation_policy_care_sites", "annotation_policy_targeted",
                 "annotation_policy_transcripts", "annotation_policy_reports", "annotation_policy_spoken", "annotation_policy_fhir"):
        module = importlib.import_module(f"outis.data.{name}")
        for attr in dir(module):
            if attr.endswith("FIXTURES"):
                for fx in getattr(module, attr):
                    for s in fx["spans"]:
                        texts.add(fx["text"][s["start"]:s["end"]])
    return frozenset(texts)


FIXTURE_TEXTS = fixture_span_texts()
DEPUNCT = re.compile(r"[^\w\s'’.\-]")


def check_format(fmt: str, text: str) -> None:
    try:
        if fmt == "json":
            json.loads(text)
        elif fmt == "ndjson":
            lines = text.split("\n")
            if len(lines) < 2:
                raise ValueError("ndjson needs several lines")
            for line in lines:
                json.loads(line)
        elif fmt == "xhtml":
            ET.fromstring(text)
        elif fmt == "key_value":
            for line in text.split("\n"):
                if line.strip() and not KV_LINE.match(line):
                    raise ValueError("not a key-value line")
        else:  # pragma: no cover
            raise ValueError(fmt)
    except (ValueError, ET.ParseError):
        raise Reject("format invalid") from None


def phenomena_for(text: str, spans: list[dict], fmt: str, layout: Layout, case: str, phi: bool) -> list[str]:
    ph = [f"fhir_{fmt}", f"fhir_layout_{layout.name}"]
    labels = {s["label"] for s in spans}
    if not phi:
        ph.append("fhir_phi_free_match" if layout.family != "decoy" else "fhir_decoy_values")
    if "urn:oid:" in text or '"system"' in text or "system" in text:
        ph.append("fhir_system_uri")
    if any(s["label"] == "ID" and text[max(0, s["start"] - 8):s["start"]].endswith("Patient/") for s in spans):
        ph.append("fhir_reference")
    elif "ID" in labels:
        ph.append("fhir_logical_id")
    if any(s["label"] == "URL" for s in spans):
        ph.append("fhir_url")
    if any(s["label"] == "DATE" and re.match(r"T\d\d:\d\d", text[s["end"]:s["end"] + 6]) for s in spans):
        ph.append("fhir_datetime")
    if any(s["label"] == "NAME" for s in spans) and fmt in ("json", "ndjson"):
        ph.append("fhir_name_parts")
    if any(s["label"] in ("ADDRESS", "LOCATION") for s in spans):
        ph.append("fhir_address_parts")
    if "AGE" in labels:
        ph.append("fhir_age_over_89")
    if "IP" in labels:
        ph.append("fhir_network_address")
    if layout.family == "narrative":
        ph.append("fhir_narrative")
    if case != "normal":
        ph.append(f"fhir_case_{case}")
    return sorted(set(ph))


def row_identities(ctx: Ctx, spans: list[dict], phi: bool) -> list[str]:
    """Identity strings for the row: the primary person first, then any other person named in the text."""
    texts = " ".join(s["text"].lower() for s in spans)
    people = list(ctx.persons)
    if not people:
        if not phi:
            return []
        fac = [s["text"] for s in spans if s["label"] == "FACILITY"]
        if not fac:
            raise Reject("no identity")
        if identity_split(canonical_identity_group(fac[0])) != "train":
            ctx.stats["row_refused_identity_partition"] += 1
            raise Reject("identity partition")
        return [fac[0]]
    shown = [p.identity for p in people if p.family.lower() in texts]
    primary = people[0].identity
    return [primary] + [i for i in shown if i != primary]


def make_row(layout: Layout, ctx: Ctx, used: Counter, seen: set[str]) -> dict:
    out, fmt, rtypes = layout.build(ctx)
    text, spans = out.text, out.spans
    phi = ctx.phi
    if bool(spans) != phi:
        raise Reject("polarity")
    from outis.data.annotation_policy import annotation_errors

    if annotation_errors(text, spans):
        raise Reject("offsets")
    check_format(fmt, text)
    if text in seen:
        raise Reject("duplicate text")
    if any(s["text"] in FIXTURE_TEXTS for s in spans):
        raise Reject("fixture value")
    values = {s["text"] for s in spans}
    if any(used[v] >= FINAL_CAP for v in values):
        raise Reject("value cap")
    if excluded_sequences(text) or excluded_sequences(DEPUNCT.sub(" ", text)):
        raise Reject("exclusion")
    identities = row_identities(ctx, spans, phi)
    return {"text": text, "spans": spans, "fmt": fmt, "rtypes": rtypes, "identities": identities, "values": values}


def build_rows(seed: int = SEED, target: int = TARGET_ROWS):
    layouts = make_layouts()
    assert sum(lay.count for lay in layouts) == target
    used: Counter = Counter()
    stats: Counter = Counter()
    seen: set[str] = set()
    rows: list[dict] = []
    pair_no = 0
    for layout in layouts:
        n_neg = round(layout.count * layout.neg)
        n_pos = layout.count - n_neg
        positives: list[dict] = []
        for polarity, n in (("pos", n_pos), ("neg", n_neg)):
            for i in range(n):
                for attempt in range(400):
                    rng = random.Random(f"{seed}:{layout.name}:{polarity}:{i}:{attempt}")
                    ctx = Ctx(rng, used, stats, phi=(polarity == "pos"))
                    ctx.case = rng.choices(("normal", "upper", "lower"), (85, 8, 7))[0] if layout.cased else "normal"
                    try:
                        built = make_row(layout, ctx, used, seen)
                    except Reject as err:
                        stats[f"row_rejected_{err.reason.replace(' ', '_')}"] += 1
                        continue
                    break
                else:
                    raise RuntimeError(f"could not build {layout.name} {polarity} {i}")
                seen.add(built["text"])
                for v in built["values"]:
                    used[v] += 1
                row = {"layout": layout, "polarity": polarity, "case": ctx.case, **built}
                if polarity == "pos":
                    pair_no += 1
                    row["pair"] = pair_no
                    positives.append(row)
                else:
                    if i < len(positives):
                        partner = positives[i]
                        row["pair"] = partner["pair"]
                        row["identities"] = partner["identities"]
                    else:
                        pair_no += 1
                        row["pair"] = pair_no
                rows.append(row)
    random.Random(f"{seed}:order").shuffle(rows)
    final = []
    for n, r in enumerate(rows, 1):
        layout: Layout = r["layout"]
        rid = f"tk20-fhir-{n:05d}"
        identities = r["identities"]
        if identities:
            groups = []
            for ident in identities:
                g = canonical_identity_group(ident)
                if g not in groups:
                    groups.append(g)
            primary = groups[0]
        else:
            groups, primary = [], f"phi_free_decoy_{rid}"
        meta_row = {
            "id": rid, "text": r["text"],
            "spans": r["spans"], "split": "train", "suite": SUITE, "annotation_policy": POLICY_VERSION,
            "cohort": "realistic", "family": f"fhir_{layout.family}_{layout.name}", "template_family": f"fhir_{layout.family}_{layout.name}",
            "phenomena": phenomena_for(r["text"], r["spans"], r["fmt"], layout, r["case"], r["polarity"] == "pos"),
            "identity_group": primary, "identity_groups": groups, "pair_id": f"tk20-pair-{r['pair']:05d}" if layout.family != "decoy" else f"tk20-solo-{n:05d}",
            "layout": layout.name, "composition": layout.family, "format": r["fmt"], "resource_types": r["rtypes"],
            "case": r["case"], "has_phi": r["polarity"] == "pos",
        }
        final.append(meta_row)
    return final, stats, used


def label_support(rows) -> dict:
    from outis.types import ENTITY_TYPES

    c = Counter(s["label"] for r in rows for s in r["spans"])
    return {label: c[label] for label in ENTITY_TYPES}


def mark(text: str, spans: list[dict]) -> str:
    pieces, prev = [], 0
    for s in spans:
        pieces.append(text[prev:s["start"]])
        pieces.append(f"[[{s['label']}|{text[s['start']:s['end']]}]]")
        prev = s["end"]
    pieces.append(text[prev:])
    return "".join(pieces)


def write_spot_check(rows: list[dict], path: Path, seed: int, n: int = 100) -> None:
    sample = random.Random(f"{seed}:spot").sample(rows, n)
    lines = [f"# Spot check: {n} random rows of tk20_train_fhir", "",
             "Spans are marked `[[LABEL|text]]`; the marks are not part of the row text. "
             "Rows are drawn with a fixed seed from the whole bank, in no particular order.", ""]
    for r in sample:
        lines += [f"## {r['id']}  ({r['layout']}; case {r['case']}; {'positive' if r['has_phi'] else 'no PHI'})", "",
                  "```text", mark(r["text"], r["spans"]), "```", ""]
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--spot-check", default=None, help="also write a 100-row marked sample here")
    args = parser.parse_args()
    rows, stats, used = build_rows(args.seed)
    for r in rows:
        blob = json.dumps({k: v for k, v in r.items() if k not in ("text", "spans")}, ensure_ascii=False).lower()
        for word in BANNED_METADATA:
            if word in blob:
                raise SystemExit(f"banned word {word!r} in metadata of {r['id']}")
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    families = Counter(r["composition"] for r in rows)
    rtypes = Counter(t for r in rows for t in set(r["resource_types"]))
    support = label_support(rows)
    max_rows_per_value = max(used.values())
    manifest = {
        "name": "tk20_train_fhir",
        "role": "training bank",
        "split": "train",
        "annotation_policy": POLICY_VERSION,
        "policy_decisions": "FHIR decisions F1 to F4 in docs/annotation-policy-v2.md",
        "independent_final": False,
        "rows": len(rows),
        "documents": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "label_support": support,
        "entity_support": support,
        "label_counts": support,
        "phi_free_rows": sum(1 for r in rows if not r["spans"]),
        "families": dict(sorted(families.items())),
        "layouts": dict(sorted(Counter(r["layout"] for r in rows).items())),
        "resource_types": dict(sorted(rtypes.items())),
        "generator": "scripts/takeover/author20_fhir.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": args.seed,
        "authoring": "all templates and value pools written by this project; no external generator or example set",
        "max_rows_sharing_one_span_value": max_rows_per_value,
        "exclusion_filter": {
            "list": "places_exclusions.sha256.json (hashes only)",
            "applied_to": "every drawn value and every final row",
            "counts_only": {k: v for k, v in sorted(stats.items()) if "exclusion" in k},
            "matches_in_final_rows": 0,
        },
        "rejected_draw_counts": dict(sorted(stats.items())),
    }
    if args.spot_check:
        write_spot_check(rows, Path(args.spot_check), args.seed)
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "label_support": support, "stats": dict(stats), "max_rows_per_value": max_rows_per_value}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
