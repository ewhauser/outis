"""Author the tk13 care-site and locality training bank.

Deterministic for its seed. Follows docs/care-site-locality-bank-brief.md, the
policy (docs/annotation-policy-v2.md) and the places addendum (P5a, P7).

Real place and care-site names come only from the pinned public sources and
only from the ``train`` value pool. Everything else is invented.

Rows are authored as marked strings (``[[LABEL|text]]``). A row's case mode is
applied to every literal and span piece before offsets are computed, so spans
stay exact under upper and lower case.

Usage (from the repository root):
  PYTHONPATH=src python scripts/takeover/author13_care_sites.py [--reports]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / ".cache" / "tiktoken"))

from clinical_deid.data.annotation_policy import (  # noqa: E402
    POLICY_VERSION,
    annotation_errors,
    canonical_identity_group,
    identity_split,
)
from clinical_deid.data.annotation_policy_places import (  # noqa: E402
    ADDENDUM,
    place_value_pool,
)
from clinical_deid.types import ENTITY_TYPES  # noqa: E402

SEED = 13013
SEED_V2 = 14013
SUITE_V2 = "tk13_care_sites_localities_v2"
BANK_VERSION_V2 = "tk13-v2"
SUITE = "tk13_care_sites_localities"
OUT = ROOT / "data" / "tk13_train_care_sites_localities.jsonl"
OUT_V2 = ROOT / "data" / "tk13_train_care_sites_localities__v2.jsonl"
REPORT_DIR = ROOT / "reports" / "takeover" / "author13_care_sites"

SOURCE_PINS = {
    "2024_Gaz_place_national.zip": {
        "sha256": "cf262fc92b2326f7a8c62a89d156a60eb17d64d6d35f7a62310c43bb08972c06",
        "url": "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_place_national.zip",
    },
    "2024_Gaz_counties_national.zip": {
        "sha256": "3c337402b5c6e8d5aa26b4278ccf4edc8989f2683765b3ffbf22296cdb2df3a0",
        "url": "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_counties_national.zip",
        "used": False,
    },
    "Hospital_General_Information.csv": {
        "sha256": "f874f09fef895a1ccf5bb7392dcbb2be05c9860339261b093fe00f0c1b013480",
        "url": "https://data.cms.gov/provider-data/dataset/xubh-q36u",
    },
}
SOURCE_DIR_CANDIDATES = (
    ROOT / ".cache" / "place_sources",
    Path(".cache/place_sources"),
    Path(
        ".cache/place_sources"
    ),
)

FAMILY_ROWS_V2 = {
    "city_modifier": 450,
    "locality_short": 450,
    "service_org": 650,
    "practice_pharmacy": 450,
    "senior_living_at": 350,
    "town_narrative": 350,
}
FAMILY_ROWS = {
    "city_modifier": 400,
    "locality_short": 400,
    "service_org": 600,
    "practice_pharmacy": 400,
    "senior_living_at": 300,
    "town_narrative": 300,
}
CASE_TARGET = {"normal": 0.60, "upper": 0.20, "lower": 0.20}
LONG_FRACTION = 0.03
UNUSUAL_NAME_FRACTION = 0.03
MAX_VALUE_SHARE = 0.005
LONG_LAYOUT = "long_note"

# Strings from the brief and fixture modules that must not be reused verbatim.
BANNED_FRAGMENTS = (
    "vellmont",
    "corriston",
    "vossmere",
    "tindle brook",
    "saint odile",
    "yarrowick",
    "pellham",
    "marrowdale",
    "ashgrove",
    "quenby",
    "thrushcombe",
    "gallowmere",
    "marlowe",
    "fennick",
    "quillon",
    "ashby mill",
    "brackenwheel",
    "brackenford",
    "ellery falls",
    "lakeshore",
    "mossmere",
    "larkspur",
    "silver finch",
    "cedar haven",
    "testier",
    "mira wilson",
    "nia okonkwo",
    "o’rían",
    "ana maría",
    "imaginary lane",
    "mayo clinic",
    "ombrey",
    "gorsefield",
    "kestrine",
    "tollund",
    "hearth at",
    "saint brevin",
    "harlow vane",
)

STATES = (
    ("AL", "Alabama"),
    ("AK", "Alaska"),
    ("AZ", "Arizona"),
    ("AR", "Arkansas"),
    ("CA", "California"),
    ("CO", "Colorado"),
    ("CT", "Connecticut"),
    ("DE", "Delaware"),
    ("FL", "Florida"),
    ("GA", "Georgia"),
    ("HI", "Hawaii"),
    ("ID", "Idaho"),
    ("IL", "Illinois"),
    ("IN", "Indiana"),
    ("IA", "Iowa"),
    ("KS", "Kansas"),
    ("KY", "Kentucky"),
    ("LA", "Louisiana"),
    ("ME", "Maine"),
    ("MD", "Maryland"),
    ("MA", "Massachusetts"),
    ("MI", "Michigan"),
    ("MN", "Minnesota"),
    ("MS", "Mississippi"),
    ("MO", "Missouri"),
    ("MT", "Montana"),
    ("NE", "Nebraska"),
    ("NV", "Nevada"),
    ("NH", "New Hampshire"),
    ("NJ", "New Jersey"),
    ("NM", "New Mexico"),
    ("NY", "New York"),
    ("NC", "North Carolina"),
    ("ND", "North Dakota"),
    ("OH", "Ohio"),
    ("OK", "Oklahoma"),
    ("OR", "Oregon"),
    ("PA", "Pennsylvania"),
    ("RI", "Rhode Island"),
    ("SC", "South Carolina"),
    ("SD", "South Dakota"),
    ("TN", "Tennessee"),
    ("TX", "Texas"),
    ("UT", "Utah"),
    ("VT", "Vermont"),
    ("VA", "Virginia"),
    ("WA", "Washington"),
    ("WV", "West Virginia"),
    ("WI", "Wisconsin"),
    ("WY", "Wyoming"),
)
COUNTRIES = (
    "Mexico",
    "Canada",
    "Ireland",
    "Italy",
    "Japan",
    "Brazil",
    "Peru",
    "India",
    "Kenya",
    "Egypt",
    "Poland",
    "Germany",
    "Greece",
    "Spain",
    "Vietnam",
    "the Philippines",
    "Jamaica",
    "Portugal",
    "Norway",
    "Turkey",
    "Thailand",
    "Argentina",
    "Ghana",
    "Morocco",
    "Chile",
)

FIRST_NAMES = (
    "Adaeze Alden Amara Anselm Aurelio Beatrix Bryn Calista Cormac Dalia Declan Delphine Dmitri Eamon Elodie "
    "Esperanza Evander Farid Fenella Galen Greta Hadley Henrik Imani Ines Isidore Jasper Jorund Kalani "
    "Katarina Lachlan Leontine Lorcan Magdalena Marisol Matteo Nadia Nikolai Odalys Orla Osric Perrin "
    "Philippa Quentin Rafaela Reuben Rosalind Saoirse Selby Sunniva Tamsin Teodor Ulrika Vesper Wilhelmina "
    "Xavier Yasmin Zephyr Abelard Bronwen Casimir Dagny Emeric Fiora Gideon Hollis Ilse Jovan Keziah Lenore "
    "Mireille Nolan Octavia Pascal Ravi Sabine Tobias Una Valentin Winifred Yusuf Zora Ambrose Bettina "
    "Corwin Dorothea Ellis Fergus Gwendolyn Harlan Iris Joaquin Kieran Linnea Marguerite Neville Ottoline "
    "Phineas Rhiannon Stellan Thea Ugo Viola Wystan Yolanda Zane"
).split()
ACCENTED_FIRST = (
    "Renée",
    "Anaïs",
    "José",
    "Søren",
    "Tomás",
    "Chloé",
    "Amélie",
    "Matías",
    "Noémie",
    "Zoé",
)
SURNAME_A = (
    "Ald Bram Cal Dren Esk Fenn Gar Holm Ing Jor Kel Lund Mor Nor Ost Pen Quin Rav Sol Tar Ull Vand Wen Yor "
    "Zel Ash Brent Crow Dray Elk Fisk Grim Harp Iver Kemp Lamb Marsh Nash Orr Pike Rudd Stav Thorn Wick"
).split()
SURNAME_B = "ham son berg wood field ley ton stad man ford wright ner rick dell sen vik low mont gard".split()

STEM_A = (
    "Ash Bram Cal Dun Elm Fall Glen Hart Ing Jun Kes Lin Mar Nor Oak Pen Quill Rook Sut Thorn Ul Vail Wren "
    "Yar Zan Bar Cor Dal Eld Fen Gal Hal Isl Kir Lor Mor Nev Ors Pol Rav Sel Tal Und Ver Wex Ald Bex Cres "
    "Dov Ern Haw Kil Lun Mer Pyr Tarn Wyn"
).split()
STEM_B = (
    "mere wick ford dale brook haven moor stead field combe bury gate worth hurst well holm ridge vale wood "
    "cliff by ton ley shaw"
).split()
SAINTS = (
    "Agnes Brigid Clement Dunstan Edmund Fiacre Gilda Hadrian Ignatia Jerome Kilian Lucan Margit Nicasio "
    "Oswin Perpetua Rosalind Sabina Teodor Ursula Valerian Wenceslas"
).split()
STREETS = (
    "Alder Birchfield Copperleaf Dunmore Elderberry Foxglove Gannet Heronsgate Ironwood Juniper Kestrel "
    "Linden Mulberry Nightjar Orchard Plover Quarry Rowan Sandpiper Tamarack Upland Violet Willowbrook"
).split()
STREET_TYPES = ("Lane", "Road", "Street", "Avenue", "Court", "Drive", "Way")
RIDGE_WORDS = (
    "Ridge",
    "Mill",
    "Falls",
    "Landing",
    "Crossing",
    "Point",
    "Glen",
    "Run",
    "Pines",
    "Commons",
    "Green",
)

# Place words that have an ordinary-word or personal-name use, keyed to the ordinary phrase.
ORDINARY_WORDS = {
    "Reading": ["reading glasses", "reading lamp for the bedside"],
    "Mobile": ["mobile phone charger", "mobile phone alarm for doses"],
    "Bath": ["bath chair", "bath bench with a back"],
    "Orange": ["orange juice", "orange juice with breakfast"],
    "Normal": ["normal saline", "normal sinus rhythm"],
    "Buffalo": ["buffalo hump noted on exam"],
    "Plain": ["plain film of the chest", "plain film of the knee"],
    "Sandwich": ["turkey sandwich", "soft sandwich cut small"],
    "Rice": ["rice cereal", "white rice with meals"],
    "Cherry": ["cherry juice", "cherry flavored syrup"],
    "Mason": ["mason jar for the pills"],
    "Lemon": ["lemon water", "lemon tea"],
    "Independence": ["functional independence", "independence with dressing"],
}
PLACE_NAME_WORDS = (
    "Jackson",
    "Madison",
    "Lincoln",
    "Austin",
    "Chester",
    "Florence",
    "Sterling",
    "Hope",
    "Eden",
    "Bristol",
    "Dallas",
    "Orlando",
    "Phoenix",
    "Charlotte",
    "Savannah",
    "Warren",
    "Marion",
    "Clifton",
    "Camden",
)
# Known public named instruments and diseases: (full name, place word, kind).
INSTRUMENTS = (
    ("Chicago Classification", "Chicago"),
    ("Boston criteria", "Boston"),
    ("Seattle Heart Failure Model", "Seattle"),
    ("Ottawa ankle rules", "Ottawa"),
    ("San Francisco Syncope Rule", "San Francisco"),
    ("Framingham Risk Score", "Framingham"),
    ("Glasgow Coma Scale", "Glasgow"),
    ("Cincinnati Prehospital Stroke Scale", "Cincinnati"),
    ("Pittsburgh Sleep Quality Index", "Pittsburgh"),
    ("Denver Developmental Screening Test", "Denver"),
    ("Philadelphia criteria", "Philadelphia"),
    ("Rochester criteria", "Rochester"),
    ("Oxford knee score", "Oxford"),
    ("Berlin questionnaire", "Berlin"),
    ("Geneva score", "Geneva"),
    ("Kansas City Cardiomyopathy Questionnaire", "Kansas City"),
    ("Baltimore criteria", "Baltimore"),
    ("Los Angeles Prehospital Stroke Screen", "Los Angeles"),
    ("Cambridge score", "Cambridge"),
    ("Newcastle score", "Newcastle"),
    ("Dallas criteria", "Dallas"),
    ("Charlotte score", "Charlotte"),
)
DISEASES = (
    ("Lyme disease", "Lyme"),
    ("Norwalk virus", "Norwalk"),
    ("Coxsackie virus", "Coxsackie"),
    ("Pontiac fever", "Pontiac"),
    ("Jamestown Canyon virus", "Jamestown"),
    ("St. Louis encephalitis", "St. Louis"),
    ("Bourbon virus", "Bourbon"),
    ("Lassa fever", "Lassa"),
    ("Hendra virus", "Hendra"),
    ("Marburg virus", "Marburg"),
)
GENERIC_LOCATION_HEADS = (
    "clinic",
    "office",
    "site",
    "unit",
    "branch",
    "location",
    "center",
    "hospital",
    "campus",
    "neighborhood",
)

FLUFF = (
    "BP {sys}/{dia}, HR {hr}, afebrile.",
    "Takes metformin 500 mg twice daily and lisinopril 10 mg daily.",
    "Denies chest pain or shortness of breath.",
    "Weight stable at {wt} lb since the last weigh-in.",
    "Walker in use; one fall reported three days ago without injury.",
    "Appetite fair; eating about half of meals.",
    "Sleeping through most nights with the bedside lamp on.",
    "Skin intact except a stage 2 area over the sacrum that is improving.",
    "Oxygen saturation {spo2}% on room air.",
    "Pain {pain}/10 at rest, controlled with acetaminophen.",
    "Daughter helps with pills and keeps the pillbox filled each Sunday.",
    "Lungs clear to auscultation; mild bilateral ankle edema.",
    "Labs reviewed; creatinine {cr}, potassium {k}.",
    "Needs reminders to use the cane on stairs.",
    "Blood sugars running {gl} to {gl2} in the mornings per her log.",
    "Reports mild dizziness on standing; hydrate and rise slowly.",
    "Vaccines up to date per her own record.",
    "Hearing aids in place; speaks clearly and follows commands.",
    "Continue current inhaler technique review at each visit.",
    "No new rashes or swelling; wound edges pink and dry.",
    "Goals of care discussed; she wants to stay as independent as possible.",
    "Bowels regular; no urinary complaints today.",
    "Transport arranged for Tuesday; needs a wheelchair at pickup.",
    "Home safety review done; grab bars requested for the shower.",
    "Mood brighter this week; enjoys the garden when weather allows.",
    "Refill requested for the blood pressure tablets; two weeks left.",
    "Therapy twice a week, tolerating exercises well.",
    "Plan: recheck labs in {wk} weeks and call sooner for any change.",
    "Family meeting held; all questions answered.",
    "Follow up visit in {wk} weeks.",
)


# --------------------------------------------------------------------------- sources


def find_sources() -> Path:
    for directory in SOURCE_DIR_CANDIDATES:
        if all((directory / name).exists() for name in SOURCE_PINS):
            return directory
    raise FileNotFoundError("pinned place sources not found")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pins(directory: Path) -> None:
    for name, pin in SOURCE_PINS.items():
        if sha256_file(directory / name) != pin["sha256"]:
            raise ValueError(f"source pin mismatch for {name}")


PLACE_SUFFIXES = (" city", " town", " village", " borough", " CDP")
NAME_RE = re.compile(r"^[A-Z][A-Za-z.'\-]*(?: [A-Z][A-Za-z.'\-]*){0,2}$")
PLACE_BLOCKLIST = {
    "Center",
    "Home",
    "Clinic",
    "Hospital",
    "Medical",
    "Health",
    "Care",
    "Office",
    "Site",
    "Unit",
    "Park",
    "Mobile",
    "Reading",
    "Bath",
    "Orange",
    "Normal",
    "Buffalo",
    "Independence",
    "Hope",
    "Eden",
    "Sterling",
    "Warren",
    "Marion",
    "Jackson",
    "Madison",
    "Lincoln",
    "Austin",
    "Chester",
    "Florence",
    "Bristol",
    "Dallas",
    "Orlando",
    "Phoenix",
    "Charlotte",
    "Savannah",
    "Clifton",
    "Camden",
    "Lyme",
    "Norwalk",
    "Coxsackie",
    "Pontiac",
    "Jamestown",
    "Bourbon",
    "Lassa",
    "Hendra",
    "Marburg",
    "Chicago",
    "Boston",
    "Seattle",
    "Ottawa",
    "San Francisco",
    "Framingham",
    "Glasgow",
    "Cincinnati",
    "Pittsburgh",
    "Denver",
    "Philadelphia",
    "Rochester",
    "Oxford",
    "Berlin",
    "Geneva",
    "Kansas City",
    "Baltimore",
    "Los Angeles",
    "Cambridge",
    "Newcastle",
    "St. Louis",
    "Plain",
    "Sandwich",
    "Rice",
    "Cherry",
    "Mason",
    "Lemon",
}


def load_places(directory: Path):
    """Return (all_names, pools) from the Gazetteer place file.

    ``all_names`` maps bare name -> (state, kind) for every place (any pool),
    used to keep invented names distinct from real ones. ``pools`` holds
    train-pool picks by role.
    """
    with zipfile.ZipFile(directory / "2024_Gaz_place_national.zip") as archive:
        member = archive.namelist()[0]
        raw = archive.read(member).decode("latin-1")
    reader = csv.reader(io.StringIO(raw), delimiter="\t")
    header = [h.strip() for h in next(reader)]
    usps, name_idx = header.index("USPS"), header.index("NAME")
    all_names: dict[str, tuple[str, str]] = {}
    for row in reader:
        if len(row) <= name_idx:
            continue
        state, full = row[usps].strip(), row[name_idx].strip()
        for suffix in PLACE_SUFFIXES:
            if full.endswith(suffix):
                bare, kind = full[: -len(suffix)].strip(), suffix.strip().lower()
                if bare and bare not in all_names:
                    all_names[bare] = (state, kind)
                break
    return all_names


def load_counties(directory: Path):
    """Return sorted (value, state) for every county named `<Name> County`."""
    with zipfile.ZipFile(directory / "2024_Gaz_counties_national.zip") as archive:
        raw = archive.read(archive.namelist()[0]).decode("latin-1")
    reader = csv.reader(io.StringIO(raw), delimiter="\t")
    header = [h.strip() for h in next(reader)]
    usps, name_idx = header.index("USPS"), header.index("NAME")
    seen: dict[str, str] = {}
    for row in reader:
        if len(row) <= name_idx:
            continue
        full = row[name_idx].strip()
        if full.endswith(" County") and full not in seen:
            seen[full] = row[usps].strip()
    return sorted(seen.items())


def load_cms(directory: Path):
    names, cities = [], set()
    with (directory / "Hospital_General_Information.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        for row in csv.DictReader(handle):
            raw = row["Facility Name"].strip()
            cities.add(row["City/Town"].strip().title())
            names.append(raw)
    return names, cities


SMALL_WORDS = {"of", "and", "at", "for", "in", "the"}


def cms_title(raw: str) -> str | None:
    """Title-case a CMS facility name; None when the name is not clean enough."""
    if not re.fullmatch(r"[A-Z][A-Z' \-]+", raw):
        return None
    words = raw.split()
    if len(words) < 2 or words[0] == "THE" or len(words) > 6:
        return None
    out = []
    for index, word in enumerate(words):
        if (
            len(word) <= 4
            and not re.search(r"[AEIOUY]", word)
            and word not in {"ST", "MT"}
        ):
            return None  # acronym such as UPMC or UNC; title-casing would corrupt it
        low = word.lower()
        if word == "ST":
            out.append("St.")
        elif word == "MT":
            out.append("Mt.")
        elif index and low in SMALL_WORDS:
            out.append(low)
        else:
            out.append(
                "-".join(part.capitalize() for part in word.split("-")).replace(
                    "'S", "'s"
                )
            )
    return " ".join(out)


# --------------------------------------------------------------------------- markup and rows

MARK = re.compile(r"\[\[([A-Z]+)\|(.*?)\]\]")


def lab(label: str, text: str) -> str:
    assert "[[" not in text and "]]" not in text and text == text.strip(), text
    return f"[[{label}|{text}]]"


def apply_case(piece: str, case: str) -> str:
    if case == "upper":
        return piece.upper()
    if case == "lower":
        return piece.lower()
    return piece


def build_row(marked: str, case: str = "normal"):
    """Turn a marked string into (text, spans) with exact character offsets.

    The case mode is applied to each literal and span piece before offsets are
    computed, so the offsets stay valid whatever the case.
    """
    pieces, spans, offset, previous = [], [], 0, 0
    for match in MARK.finditer(marked):
        literal = apply_case(marked[previous : match.start()], case)
        value = apply_case(match[2], case)
        pieces.extend((literal, value))
        offset += len(literal)
        spans.append(
            {
                "start": offset,
                "end": offset + len(value),
                "label": match[1],
                "text": value,
            }
        )
        offset += len(value)
        previous = match.end()
    pieces.append(apply_case(marked[previous:], case))
    text = "".join(pieces)
    if "[[" in text or "]]" in text:
        raise ValueError("malformed markup")
    return text, spans


def marked_from_example(example: dict) -> str:
    """Rebuild marked text from a fixture's text and spans (used by the tests)."""
    text, out, previous = example["text"], [], 0
    for span in sorted(example["spans"], key=lambda s: s["start"]):
        out.append(text[previous : span["start"]])
        out.append(lab(span["label"], text[span["start"] : span["end"]]))
        previous = span["end"]
    out.append(text[previous:])
    return "".join(out)


NUM_WORDS = {
    90: "ninety",
    91: "ninety-one",
    92: "ninety-two",
    93: "ninety-three",
    94: "ninety-four",
    95: "ninety-five",
    96: "ninety-six",
    97: "ninety-seven",
    98: "ninety-eight",
    99: "ninety-nine",
}


# --------------------------------------------------------------------------- context


class Picker:
    """Cycle through a shuffled list so no value repeats until all are used."""

    def __init__(self, items, rng):
        self.items, self.rng, self.queue = list(items), rng, []

    def next(self):
        if not self.queue:
            self.queue = list(self.items)
            self.rng.shuffle(self.queue)
        return self.queue.pop()


class World:
    """Value pools shared by every row."""

    def __init__(self, rng: random.Random, version: str = "v1"):
        self.v2 = version == "v2"
        directory = find_sources()
        verify_pins(directory)
        self.rng = rng
        all_names = load_places(directory)
        cms_raw, cms_cities = load_cms(directory)
        self.real_names = set(all_names) | set(cms_cities)
        first_names = set(FIRST_NAMES)

        def usable(name: str) -> bool:
            return (
                NAME_RE.fullmatch(name) is not None
                and 4 <= len(name) <= 20
                and name not in PLACE_BLOCKLIST
                and name not in first_names
                and not any(b in name.lower() for b in BANNED_FRAGMENTS)
                and place_value_pool(name) == "train"
            )

        entries = sorted((n, s, k) for n, (s, k) in all_names.items() if usable(n))
        self.city_entries = [e for e in entries if e[2] in ("city", "town")]
        self.village_entries = [e for e in entries if e[2] in ("village", "borough")]
        self.cdp_entries = [e for e in entries if e[2] == "cdp"]
        self.city = Picker(self.city_entries, rng)
        self.town = Picker(self.city_entries + self.village_entries, rng)
        self.hood = Picker(self.cdp_entries + self.village_entries, rng)
        self.any_place = Picker(
            self.city_entries + self.village_entries + self.cdp_entries, rng
        )
        cms = []
        for raw in sorted(set(cms_raw)):
            title = cms_title(raw)
            if (
                title
                and place_value_pool(raw) == "train"
                and not any(b in title.lower() for b in BANNED_FRAGMENTS)
            ):
                cms.append((title, raw))
        self.cms = Picker(cms, rng)
        self.cms_count = len(cms)
        known = set(all_names)
        self.place_word_ok = lambda w: w in known and place_value_pool(w) == "train"
        self.instruments = [(n, w) for n, w in INSTRUMENTS if self.place_word_ok(w)]
        self.diseases = [(n, w) for n, w in DISEASES if self.place_word_ok(w)]
        self.ordinary = {
            w: p for w, p in ORDINARY_WORDS.items() if self.place_word_ok(w)
        }
        self.name_words = [w for w in PLACE_NAME_WORDS if self.place_word_ok(w)]
        self.ordinary_word = Picker(sorted(self.ordinary), rng)
        self.name_word = Picker(self.name_words, rng)
        self.instrument = Picker(self.instruments, rng)
        self.disease = Picker(self.diseases, rng)
        stems = []
        for a in STEM_A:
            for b in STEM_B:
                stem = a + b
                low = stem.lower()
                if stem in self.real_names or any(f in low for f in BANNED_FRAGMENTS):
                    continue
                stems.append(stem)
        self.stem = Picker(sorted(stems), rng)
        self.stem_count = len(stems)
        self.saint = Picker(SAINTS, rng)
        self.street = Picker(STREETS, rng)
        self.pins = {
            n: {"sha256": p["sha256"], "url": p["url"], "used": p.get("used", True)}
            for n, p in SOURCE_PINS.items()
        }
        self.source_dir = directory
        if self.v2:
            counties = []
            for value, state in load_counties(directory):
                base = value[: -len(" County")]
                if (
                    NAME_RE.fullmatch(base)
                    and len(base) >= 3
                    and base not in first_names
                    and not any(b in value.lower() for b in BANNED_FRAGMENTS)
                    and place_value_pool(value) == "train"
                ):
                    counties.append((value, state))
            self.county_picker = Picker(counties, rng)
            self.county_count = len(counties)
            self.pins["2024_Gaz_counties_national.zip"]["used"] = True


class Ctx:
    """Per-row state: random draws, real-value records and identities."""

    def __init__(self, world: World, rng: random.Random):
        self.w, self.rng = world, rng
        self.reals: list[dict] = []
        self.identities: list[str] = []
        self.names_seen: set[str] = set()
        self.flags: set[str] = set()

    # --- real values
    def record(self, value: str, source: str, role: str) -> None:
        pool = place_value_pool(value)
        assert pool == "train", (value, pool)
        self.reals.append(
            {"value": value, "source": source, "role": role, "pool": pool}
        )

    def city(self, role: str, picker=None):
        name, state, _ = (picker or self.w.city).next()
        self.record(name, "census_gazetteer_place_2024", role)
        return name, state

    def county(self):
        value, state = self.w.county_picker.next()
        self.record(value, "census_gazetteer_counties_2024", "county_name")
        return value, state

    def cms(self, role: str) -> str:
        title, raw = self.w.cms.next()
        self.record(raw, "cms_hospital_general_information", role)
        return title

    # --- invented values
    def stem(self) -> str:
        return self.w.stem.next()

    def saint(self) -> str:
        return self.w.saint.next()

    def person_name(self, first_pool=None):
        for _ in range(500):
            first = self.rng.choice(first_pool or FIRST_NAMES)
            last = self.rng.choice(SURNAME_A) + self.rng.choice(SURNAME_B)
            full = f"{first} {last}"
            if any(b in full.lower() for b in BANNED_FRAGMENTS):
                continue
            if identity_split(canonical_identity_group(full)) == "train":
                return first, last
        raise RuntimeError("no train-partition name found")

    def name(
        self, role: str = "other", first_pool=None, style: str | None = None
    ) -> str:
        first, last = self.person_name(first_pool)
        full = f"{first} {last}"
        self.identities.append(full)
        if style is None:
            style = (
                "last_first"
                if role == "patient" and self.rng.random() < 0.25
                else "plain"
            )
        text = f"{last}, {first}" if style == "last_first" else full
        return lab("NAME", text)

    def clinician(self) -> str:
        first, last = self.person_name()
        self.identities.append(f"{first} {last}")
        return f"Dr. {lab('NAME', first + ' ' + last)}"

    def nurse(self) -> str:
        first, last = self.person_name()
        self.identities.append(f"{first} {last}")
        return f"{lab('NAME', first + ' ' + last)}, {self.rng.choice(('RN', 'LPN', 'MSW', 'PT', 'NP'))}"

    def date(self, year_range=(2023, 2025), allow_range=True) -> str:
        r = self.rng
        start = dt.date(r.randint(*year_range), r.randint(1, 12), r.randint(1, 28))
        if allow_range and r.random() < 0.07:
            end = start + dt.timedelta(days=r.randint(3, 12))
            return lab(
                "DATE", f"{start.strftime('%m/%d/%Y')} to {end.strftime('%m/%d/%Y')}"
            )
        month = start.strftime("%B")
        forms = (
            start.strftime("%m/%d/%Y"),
            f"{month} {start.day}, {start.year}",
            f"{start.strftime('%b')} {start.day}, {start.year}",
            start.strftime("%Y-%m-%d"),
            f"{start.month}/{start.day}/{start.year}",
        )
        text = r.choice(forms)
        if r.random() < 0.08:
            text = f"{start.strftime('%A')}, {month} {start.day}, {start.year}"
        return lab("DATE", text)

    def birth_year(self) -> int:
        return (
            self.rng.randint(1925, 1935)
            if self.rng.random() < 0.14
            else self.rng.randint(1936, 1992)
        )

    def dob(self, year: int) -> str:
        d = dt.date(year, self.rng.randint(1, 12), self.rng.randint(1, 28))
        return lab(
            "DATE",
            self.rng.choice(
                (d.strftime("%m/%d/%Y"), f"{d.strftime('%B')} {d.day}, {d.year}")
            ),
        )

    def age_phrase(self, year: int) -> str:
        age = 2025 - year
        r = self.rng
        if age > 89:
            if r.random() < 0.2:
                return f"{lab('AGE', NUM_WORDS[min(age, 99)])} year old"
            number = lab("AGE", str(age))
            return r.choice(
                (
                    f"{number} y/o",
                    f"{number}-year-old",
                    f"{number}F",
                    f"{number} years old",
                    f"{number}yo",
                )
            )
        return r.choice(
            (f"{age} y/o", f"{age}-year-old", f"{age}M", f"{age} years old")
        )

    def phone(self) -> str:
        r = self.rng
        area, line = (
            r.choice([a for a in range(201, 990) if a != 555]),
            r.randint(100, 199),
        )
        base = r.choice(
            (f"({area}) 555-0{line}", f"{area}-555-0{line}", f"{area}.555.0{line}")
        )
        if r.random() < 0.1:
            base += f" ext. {r.randint(100, 999)}"
        if r.random() < 0.05:
            base = "+1 " + base
        return lab("PHONE", base)

    def mrn(self) -> str:
        r = self.rng
        if r.random() < 0.3:
            return lab(
                "MRN",
                f"{r.choice(('QX', 'TK', 'MV', 'RB', 'HL'))}-{r.randint(1000000, 9999999)}",
            )
        return lab("MRN", str(r.randint(1000000, 99999999)))

    def address(self) -> str:
        r = self.rng
        city, state = self.city("address_city")
        street = (
            f"{r.randint(12, 9890)} {self.w.street.next()} {r.choice(STREET_TYPES)}"
        )
        unit = f", Apt {r.randint(1, 40)}{r.choice('ABCD')}" if r.random() < 0.2 else ""
        return lab(
            "ADDRESS", f"{street}{unit}, {city}, {state} {r.randint(10000, 99999)}"
        )

    def email(self, first: str = "nursing") -> str:
        return lab(
            "EMAIL",
            f"{first.lower()}{self.rng.randint(1, 99)}@{self.rng.choice(('care.example', 'clinicmail.test', 'example.org', 'example.com'))}",
        )

    def url(self) -> str:
        return lab(
            "URL",
            f"https://{self.rng.choice(('portal', 'secure', 'msg'))}.{self.rng.choice(('clinic.test', 'example.org', 'care.example'))}/{self.rng.choice(('visit', 'thread', 'form'))}/{self.rng.randint(1000, 99999)}",
        )

    def idnum(self) -> str:
        return lab(
            "ID",
            f"{self.rng.choice(('XQ', 'MB', 'ZH', 'PL'))}{self.rng.randint(100000, 999999999)}",
        )

    def account(self) -> str:
        return lab("ACCOUNT", str(self.rng.randint(100000, 99999999)))

    def fluff(self, count: int) -> list[str]:
        r = self.rng
        lines = r.sample(FLUFF, count)
        return [
            line.format(
                sys=r.randint(104, 162),
                dia=r.randint(58, 94),
                hr=r.randint(54, 98),
                wt=r.randint(98, 214),
                spo2=r.randint(92, 99),
                pain=r.randint(0, 5),
                cr=f"{r.uniform(0.6, 1.8):.1f}",
                k=f"{r.uniform(3.6, 5.2):.1f}",
                gl=r.randint(92, 138),
                gl2=r.randint(140, 188),
                wk=r.randint(2, 8),
            )
            for line in lines
        ]


# --------------------------------------------------------------------------- family cores
# Each core returns (lines, phenomena, negative_type_or_None). Lines are marked strings.


def core_city_modifier(c: Ctx, positive: bool, case: str):
    r = c.rng
    if positive:
        picks = []
        templates = [
            "Seen at our {L} clinic last week.",
            "Follow-up booked at the {L} office.",
            "Labs drawn at the {L} site.",
            "Referred to the {L} dialysis unit for chair time.",
            "{L} dialysis unit called with a new chair time.",
            "Pt prefers the {L} location for infusions.",
            "Will see her again at our {L} branch.",
            "Our {L} clinic nurse will call with results.",
            "Transferred care to the {L} satellite clinic.",
            "Wound care continues at the {L} outpatient center.",
            "Bloodwork at the {L} lab draw site on Monday.",
            "Visit moved to the {L} office because of the snow.",
            "Pt asks to be seen at our {L} clinic instead.",
            "Infusion chair at the {L} cancer center is booked for Thursday.",
            "Imaging was done at the {L} imaging site.",
            "Staff at the {L} clinic confirmed the referral.",
        ]
        count = 1 if r.random() < 0.5 else 2
        phen = ["p7_city_modifier"]
        for t in r.sample(templates, count):
            city, _ = c.city("city_modifier_locality")
            picks.append(t.replace("{L}", lab("LOCATION", city)))
        if case == "normal" and r.random() < 0.18:
            city, _ = c.city("named_site_contrast_locality")
            head = r.choice(("Clinic", "Medical Group", "Health Center", "Urgent Care"))
            named = lab("FACILITY", f"{city} {head}")
            picks.append(
                r.choice(
                    ("Seen at {F} earlier this month.", "Records sent from {F}.")
                ).replace("{F}", named)
            )
            phen += ["facility_boundary", "named_site_contrast"]
            c.flags.add("normal_only")
        return picks, phen, None
    kind = r.choice(("n1", "n1", "n2", "n4", "n4", "n5", "generic"))
    if kind == "n1":
        name, word = c.w.instrument.next()
        c.record(word, "census_gazetteer_place_2024", "n1_named_instrument")
        return (
            [
                r.choice(
                    (
                        "Scored per the {X}.",
                        "{X} reviewed with the resident.",
                        "Criteria applied: {X}; result discussed.",
                        "Risk estimated using the {X}.",
                        "Assessed with the {X} today.",
                    )
                ).replace("{X}", name)
            ],
            ["n1_named_instrument"],
            "N1",
        )
    if kind == "n2":
        name, word = c.w.disease.next()
        c.record(word, "census_gazetteer_place_2024", "n2_place_disease")
        return (
            [
                r.choice(
                    (
                        "Tested for {X}; result pending.",
                        "No exposure risk for {X} identified.",
                        "Counselled about {X} prevention.",
                        "Ruled out {X} on serology.",
                    )
                ).replace("{X}", name)
            ],
            ["n2_place_disease"],
            "N2",
        )
    if kind == "n4":
        if r.random() < 0.5:
            state = r.choice(STATES)[1]
            return (
                [
                    r.choice(
                        (
                            "Valley fever is more common in {X}.",
                            "Family history is from {X}.",
                            "Tick exposure is common in {X} in summer.",
                            "Moved here from {X} years ago.",
                        )
                    ).replace("{X}", state)
                ],
                ["n4_state_or_country"],
                "N4",
            )
        country = r.choice(COUNTRIES)
        return (
            [
                r.choice(
                    (
                        "Plans travel to {X} next spring.",
                        "Travel to {X} two years ago; no illness.",
                        "Vaccines reviewed before the trip to {X}.",
                        "Born in {X}; English is a second language.",
                    )
                ).replace("{X}", country)
            ],
            ["n4_state_or_country"],
            "N4",
        )
    if kind == "n5":
        word = c.w.ordinary_word.next()
        c.record(word, "census_gazetteer_place_2024", "n5_ordinary_word")
        phrase = r.choice(c.w.ordinary[word])
        return (
            [
                r.choice(
                    (
                        "Needs {X}; order placed.",
                        "Supplies: {X}.",
                        "Reviewed {X} with the family.",
                        "Also asked about {X}.",
                    )
                ).replace("{X}", phrase)
            ],
            ["n5_place_word_other_role"],
            "N5",
        )
    return (
        [
            r.choice(
                (
                    "Seen at our clinic; labs drawn at the main office.",
                    "Follow-up booked at the clinic next door.",
                    "Dialysis unit called with a new chair time.",
                    "Labs drawn at the outpatient site on Monday.",
                    "Pt prefers the afternoon infusion chair.",
                    "Staff at the office confirmed the referral.",
                )
            )
        ],
        ["unnamed_site_no_locality"],
        "generic_unnamed",
    )


def core_locality_short(c: Ctx, positive: bool, case: str):
    r = c.rng
    if positive:
        templates = [
            ("Pt from {L}.", "city"),
            ("pt from {L} here for f/u.", "city"),
            ("Lives in {L} with her daughter.", "city"),
            ("Lives in {L}.", "hood"),
            ("Moved to {LS} last spring.", "citystate"),
            ("Daughter lives in {LS}.", "citystate"),
            ("Returned from {L} last week.", "city"),
            ("Visiting from {L} for two weeks.", "city"),
            ("Son lives in {L}; calls daily.", "hood"),
            ("Pt from {L}, here for follow up.", "town"),
            ("Recently moved to {L}.", "hood"),
            ("Came in from {L} by bus.", "city"),
            ("Relocated to {LS} in {D}.", "citystate"),
            ("Brother in {L} will drive her.", "city"),
            ("Household in {L} per intake.", "town"),
            ("Sister lives in {LS}; pt stays there in winter.", "citystate"),
        ]
        out = []
        for template, kind in r.sample(templates, 1 if r.random() < 0.55 else 2):
            picker = {"hood": c.w.hood, "town": c.w.town}.get(kind)
            name, state = c.city("patient_tied_locality", picker)
            line = template.replace(
                "{LS}", lab("LOCATION", f"{name}, {state}")
            ).replace("{L}", lab("LOCATION", name))
            if "{D}" in line:
                line = line.replace("{D}", c.date(allow_range=False))
            out.append(line)
        return out, ["p1_patient_locality"], None
    kind = r.choice(("n3", "n3", "n3b", "n4", "n4", "n4b"))
    if kind == "n3":
        name = c.w.city.next()[0]
        c.record(name, "census_gazetteer_place_2024", "n3_public_body")
        body = r.choice(("Health Department", "Public Health", "Department of Health"))
        return (
            [
                r.choice(
                    (
                        "Per {X} {B} guidance, hold the vaccine if febrile.",
                        "{X} {B} recommends masking during outbreaks.",
                        "Handout from {X} {B} given on fall prevention.",
                    )
                )
                .replace("{X}", name)
                .replace("{B}", body)
            ],
            ["n3_institution_source"],
            "N3",
        )
    if kind == "n3b":
        title = c.cms("n3_institution_source")
        return (
            [
                r.choice(
                    (
                        "Per {X} guidelines, continue the statin.",
                        "A {X} study found lower readmission with early follow up.",
                        "{X} recommends this screening interval.",
                        "Patient education sheet from {X} reviewed.",
                    )
                ).replace("{X}", title)
            ],
            ["n3_institution_source"],
            "N3",
        )
    if kind == "n4":
        state = r.choice(STATES)[1]
        return (
            [
                r.choice(
                    (
                        "Lives in {X}.",
                        "Moved to {X} last year.",
                        "Pt from {X}.",
                        "Sister lives in {X}.",
                        "Visiting from {X} for the holiday.",
                    )
                ).replace("{X}", state)
            ],
            ["n4_state_or_country"],
            "N4",
        )
    country = r.choice(COUNTRIES)
    return (
        [
            r.choice(
                (
                    "Travel to {X} planned for June.",
                    "Back from a trip to {X}; no fever.",
                    "Returned from {X} last week; asymptomatic.",
                    "Visiting family in {X} next month.",
                )
            ).replace("{X}", country)
        ],
        ["n4_state_or_country"],
        "N4",
    )


SERVICE_FORMS = (
    ("vna", "{S} VNA"),
    ("vna", "{S} Visiting Nurse Association"),
    ("vna", "VNA of {S}"),
    ("home_health", "{S} Home Health"),
    ("home_health", "{S} Home Health Agency"),
    ("hospice", "Hospice of {S}"),
    ("hospice", "{S} Hospice"),
    ("hospice", "{S} Hospice and Palliative Care"),
    ("dme", "{S} DME"),
    ("dme", "{S} Medical Equipment"),
    ("dme", "{S} Home Medical Supply"),
    ("dme", "{S} Durable Medical Equipment"),
    ("adult_day", "{S} Adult Day Program"),
    ("adult_day", "Saint {X} Adult Day Program"),
    ("adult_day", "{S} Adult Day Center"),
    ("adult_day", "{S} Adult Day Health"),
    ("rehab", "{S} Institute for Rehabilitation"),
    ("rehab", "{S} Rehab Center"),
    ("post_acute", "{S} Post-Acute"),
    ("post_acute", "{S} Post-Acute Care"),
    ("post_acute", "{S} Skilled Nursing"),
    ("lab", "{S} Reference Laboratory"),
    ("pharmacy", "{S} Specialty Pharmacy"),
    ("pharmacy", "{S} Mail-Order Pharmacy"),
    ("pharmacy", "{S} Home Infusion"),
)
SERVICE_TEMPLATES = {
    "vna": [
        "Referred to {O} for skilled nursing visits.",
        "{O} will start care on {D}.",
        "{O} nurse visits twice weekly for wound care.",
        "Home visits arranged through {O}.",
    ],
    "home_health": [
        "Referral sent to {O} for PT and OT.",
        "{O} start of care is {D}.",
        "Home health through {O}; nurse to call to schedule.",
        "Active with {O} since {D}.",
    ],
    "hospice": [
        "Admitted to {O} on {D}.",
        "Enrolled with {O}; the intake nurse visited today.",
        "Family chose {O} after the care conference.",
        "Comfort kit delivered by {O}.",
    ],
    "dme": [
        "DME from {O}.",
        "Hospital bed delivered by {O} on {D}.",
        "{O} dropped off the wheelchair and commode.",
        "Oxygen concentrator supplied by {O}; backup tanks to follow.",
    ],
    "adult_day": [
        "Attends {O} three days a week.",
        "Enrolled at {O}; transport picks her up at 8.",
        "{O} reports she ate well and joined group activities.",
        "Day program: {O}, Monday to Wednesday.",
    ],
    "rehab": [
        "Transferred to {O} for inpatient therapy.",
        "Discharged to {O} on {D}.",
        "Therapy at {O} five days a week.",
        "{O} will evaluate her on arrival.",
    ],
    "post_acute": [
        "Discharged to {O} for short-term rehab.",
        "Bed offered at {O}; transfer set for {D}.",
        "Stay at {O} expected to last two weeks.",
        "Family toured {O} yesterday.",
    ],
    "lab": [
        "Labs from {O}.",
        "Specimen sent to {O}; results in two days.",
        "Outside results from {O} reviewed.",
        "Standing orders go to {O}.",
    ],
    "pharmacy": [
        "Refills through {O}.",
        "Specialty drug shipped by {O} on {D}.",
        "Rx sent to {O}; delivery to the home.",
        "{O} will call about the prior authorization.",
    ],
}
SERVICE_NEGATIVES = (
    "Home health referral sent.",
    "DME delivered a hospital bed.",
    "The hospice team will visit twice weekly.",
    "Adult day program three days a week.",
    "Referred to home health for wound care.",
    "VNA to start Friday.",
    "Post-acute placement is pending.",
    "Family chose hospice after the care conference.",
    "Oxygen concentrator delivered by the equipment company.",
    "Mail-order pharmacy ships a 90 day supply.",
    "Rehab stay expected to last two weeks.",
    "Labs sent to the reference laboratory.",
    "Needs a home health aide twice a week.",
    "Specialty pharmacy will call about the prior authorization.",
    "Discussed hospice and palliative care with the family.",
    "Skilled nursing visits start next week.",
    "Day program staff reported good participation.",
    "Durable medical equipment ordered; awaiting delivery.",
    "Home infusion nurse to teach line care.",
    "Inpatient rehabilitation was recommended.",
    "A hospice nurse will call tomorrow to set up a visit.",
    "The home health agency has a start date next week.",
    "Waiting on the DME company to confirm delivery of the commode.",
    "Adult day program transport comes at 8 each morning.",
    "She is not ready for hospice at this time.",
    "VNA visits were stopped after the wound healed.",
    "Rehabilitation institute placement was declined by the family.",
    "Post-acute care options were explained to her son.",
    "Pharmacy delivery is set for Friday afternoon.",
    "Needs a hospital bed and a walker from the equipment supplier.",
    "The reference lab will call if the specimen is rejected.",
    "Referral to a specialty pharmacy is on the checklist.",
    "Hospice eligibility was reviewed with the family today.",
    "Home health physical therapy has been approved.",
    "Adult day health was discussed as an option for respite.",
    "DME order faxed; supplier has not responded.",
    "Skilled nursing facility stay was covered for twenty days.",
    "Home infusion pharmacy training is scheduled.",
    "A visiting nurse checks her blood pressure weekly.",
    "The post-acute unit has no beds this week.",
    "Hospice volunteers sit with her on Thursdays.",
    "Rehab center tour is planned for next month.",
)


def core_service_org(c: Ctx, positive: bool, case: str):
    r = c.rng
    if not positive:
        if r.random() < 0.15:
            return (
                [
                    f"Works at {c.stem()} {r.choice(('Logistics', 'Foods', 'Freight', 'Mills', 'Printing'))}; no exposures reported."
                ],
                ["employer_name"],
                "employer",
            )
        count = 1 if r.random() < 0.6 else 2
        return (
            r.sample(SERVICE_NEGATIVES, count),
            ["service_organisation_unnamed"],
            "unnamed_service",
        )
    lines, phen = [], ["service_organisation", "facility_boundary"]
    for _ in range(1 if r.random() < 0.6 else 2):
        kind, form = r.choice(SERVICE_FORMS)
        if r.random() < 0.2 and "{S}" in form and "Saint" not in form:
            stem, _ = c.city("service_org_name_place")
            phen.append("real_place_in_facility_name")
        else:
            stem = c.stem()
        org = form.replace("{S}", stem).replace("{X}", c.saint())
        line = r.choice(SERVICE_TEMPLATES[kind]).replace("{O}", lab("FACILITY", org))
        if "{D}" in line:
            line = line.replace("{D}", c.date(allow_range=False))
        lines.append(line)
    return lines, phen, None


PRACTICE_FORMS = (
    "{S} Health Partners",
    "{S} Family Health",
    "{S} Drug & Gift",
    "{S} Laboratory Services",
    "{S} Medical Associates",
    "{S} Pediatrics",
    "{S} Internal Medicine",
    "{S} Primary Care Group",
    "{S} Drug",
    "{S} Apothecary",
    "{S} Care Partners",
    "{S} Eye Associates",
    "{S} Orthopaedics",
    "{S} Women's Health",
    "{S} Cardiology Associates",
)
PRACTICE_TEMPLATES = (
    "Send it to {O}.",
    "Rx called in to {O}.",
    "PCP is Dr. {N} at {O}.",
    "Seen at {O} last month.",
    "The {O} pharmacist called about the dose.",
    "Her cardiologist's practice, {O}, sent records.",
    "Pharmacy: {O}; phone on file.",
    "Records requested from {O}.",
    "She was seen at the {O} office.",
    "Pick up at {O} after 4 pm.",
    "Outside labs drawn through {O}.",
    "Referred by {O} for further work up.",
    "Last visit at {O} was on {D}.",
    "Family doctor: {N} at {O}.",
    "Samples dropped off at {O}.",
)
PRACTICE_NEGATIVES = (
    "Family health history reviewed; mother with type 2 diabetes.",
    "Partners in care meeting held with the family.",
    "Care partner (spouse) present for the visit.",
    "Takes metformin 500 mg tablets; test strips and alcohol swabs refilled.",
    "Drug allergies: none known.",
    "Drug and supply list reviewed with the daughter.",
    "Family medicine follow up in six weeks.",
    "Laboratory services were not needed this visit.",
    "Health partners in the household include her sister.",
    "Primary care follow up arranged.",
    "Pharmacy will fill the prescription today.",
    "Gift shop volunteers visited the ward.",
    "Associates at her former workplace sent a card.",
    "Orthopaedics consult requested for the knee.",
    "Her pediatrician retired years ago.",
    "Eye drops refilled; apothecary jars were a family heirloom.",
    "Family health center visit was rescheduled for next week.",
    "Care partners meet monthly at the church hall.",
    "Drug interaction check was done at the visit.",
    "Medical associates from her old job sent flowers.",
    "Laboratory services closed early on the holiday.",
    "Internal medicine residents rounded this morning.",
    "Pharmacy hours were reviewed with the daughter.",
    "Cardiology associates were mentioned in the discharge plan.",
    "Health partners in the study program were not contacted.",
    "Primary care group visits were discussed at length.",
    "Family doctor appointment is in the afternoon.",
    "Women's health screening is due this spring.",
    "A drug and gift shop is on the first floor of the building.",
    "Her laboratory results were normal last year.",
    "Eye associates were not on the referral list.",
    "Pediatrics was her first job after nursing school.",
)


def core_practice_pharmacy(c: Ctx, positive: bool, case: str):
    r = c.rng
    if not positive:
        if r.random() < 0.2:
            return (
                [
                    f"Works for {c.stem()} {r.choice(('Logistics', 'Foods', 'Freight', 'Mills', 'Printing'))} part time."
                ],
                ["employer_name"],
                "employer",
            )
        return (
            r.sample(PRACTICE_NEGATIVES, 1 if r.random() < 0.6 else 2),
            ["practice_without_care_word_negative"],
            "ordinary_phrase",
        )
    lines = []
    for _ in range(1 if r.random() < 0.65 else 2):
        form = r.choice(PRACTICE_FORMS)
        org = form.replace("{S}", c.stem())
        template = r.choice(PRACTICE_TEMPLATES)
        line = template.replace("{O}", lab("FACILITY", org))
        if "{N}" in line:
            line = line.replace("{N}", c.name("other"))
        if "{D}" in line:
            line = line.replace("{D}", c.date(allow_range=False))
        lines.append(line)
    return (
        lines,
        ["practice_without_care_word", "facility_boundary", "the_article"],
        None,
    )


SENIOR_BRANDS = (
    "{S} Senior Living",
    "{S} Assisted Living",
    "{S} Memory Care",
    "{S} Retirement Community",
)


def _the_name(c: Ctx) -> str:
    r = c.rng
    form = r.choice(
        (
            "The {A} at {B} {W}",
            "The {A} at {B} {W}",
            "The {A}",
            "The {A} Commons",
            "The {A} Senior Residences",
        )
    )
    return (
        form.replace("{A}", c.stem())
        .replace("{B}", c.stem())
        .replace("{W}", r.choice(RIDGE_WORDS))
    )


def _loose_town(c: Ctx):
    """An unqualified town for an 'at <town>' continuation: real (train pool) or invented."""
    if c.rng.random() < 0.5:
        name, _ = c.city("at_continuation_town")
        return name
    return c.stem()


def core_senior_living(c: Ctx, positive: bool, case: str):
    r = c.rng
    if not positive:
        return (
            [
                r.choice(
                    (
                        "She was seen at the clinic, then the hospital.",
                        "Therapy at the house twice a week.",
                        "Lives at an assisted living facility near her son.",
                        "The community nurse visited on Tuesday.",
                        "Activities at the senior center include chair yoga.",
                        "Seen at home; family medicine follow up at the clinic.",
                        "Moved from the house at the end of the road.",
                        "The patient lives at home with her husband.",
                        "Rehab at the facility is going well.",
                        "The nursing home called about a bed.",
                        "Senior living options were reviewed with the family.",
                        "Her health center visit was rescheduled.",
                        "The unit at the hospital was quiet overnight.",
                        "Family medicine at the clinic is two floors down.",
                        "Discharged to the rehab unit at the hospital.",
                        "She stays at her daughter's house on weekends.",
                        "The assisted living facility called back about a tour.",
                        "Moved from the apartment on the second floor.",
                        "Staff at the nursing home report she eats well.",
                        "The health center is closed on holidays.",
                        "Follow up at the family practice in six weeks.",
                        "Seen at the bedside by the ward team.",
                        "The pharmacy at the hospital fills discharge prescriptions.",
                        "Lives at the end of a long gravel road.",
                        "Family medicine at the practice downtown was not available.",
                        "The residence hall nurse checked on her twice.",
                        "Transferred from the unit at the other end of the building.",
                        "Lab draws happen at the front desk of the unit.",
                        "The senior apartments have a waiting list.",
                        "Visiting nurse at the front door by noon.",
                        "Care at home is going well for now.",
                        "The home at the lake belongs to her sister.",
                        "Resident of the memory care wing at the facility.",
                    )
                )
            ],
            ["article_before_unnamed"],
            "article_unnamed",
        )
    kinds = [
        "the_name",
        "the_name",
        "senior",
        "at_continuation",
        "qualified",
        "unit_parent",
        "article_cms",
        "article_named",
    ]
    lines, phen = [], ["facility_boundary"]
    for kind in r.sample(kinds, 1 if r.random() < 0.6 else 2):
        if kind == "the_name":
            name = _the_name(c)
            line = r.choice(
                (
                    "Lives at {O}.",
                    "Moved from {O} to her daughter's home.",
                    "Resident of {O} since {D}.",
                    "{O} staff report she joins most meals.",
                    "Admitted from {O}.",
                )
            ).replace("{O}", lab("FACILITY", name))
            phen += ["the_in_name", "senior_living"]
            c.flags.add("normal_only")
        elif kind == "senior":
            name = r.choice(SENIOR_BRANDS).replace("{S}", c.stem())
            line = r.choice(
                (
                    "Lives at {O}.",
                    "Moved to {O} in {D}.",
                    "{O} staff manage her medications.",
                    "Family toured {O} last month.",
                )
            ).replace("{O}", lab("FACILITY", name))
            phen += ["senior_living", "service_organisation"]
        elif kind == "at_continuation":
            name = f"{c.stem()} {r.choice(('Family Medicine', 'Internal Medicine', 'Pediatrics', 'Health Associates'))} at {_loose_town(c)}"
            line = r.choice(
                (
                    "Seen at {O}.",
                    "PCP is at {O}.",
                    "Follow up at {O} in two weeks.",
                    "Records requested from {O}.",
                )
            ).replace("{O}", lab("FACILITY", name))
            phen += ["p5a_at_continuation"]
        elif kind == "qualified":
            town, state = (
                c.city("qualified_locality")
                if r.random() < 0.5
                else (c.stem(), r.choice(STATES)[0])
            )
            fac = f"{c.stem()} {r.choice(('Health Center', 'Medical Center', 'Clinic', 'Family Practice'))}"
            line = (
                r.choice(
                    (
                        "Seen at {F} at {L}.",
                        "Referred to {F} at {L}.",
                        "Labs at {F} at {L}.",
                    )
                )
                .replace("{F}", lab("FACILITY", fac))
                .replace("{L}", lab("LOCATION", f"{town}, {state}"))
            )
            phen += ["p5a_qualified_locality_split"]
        elif kind == "unit_parent":
            unit = f"{c.stem()} {r.choice(('Ward', 'Unit', 'Pavilion', 'Wing'))}"
            parent = f"{c.stem()} {r.choice(('Health Hub', 'Medical Campus', 'Health Center', 'Medical Center'))}"
            line = (
                r.choice(
                    (
                        "Transferred to {U} at {P}.",
                        "Moved to {U} at {P} on {D}.",
                        "Bed on {U} at {P}.",
                    )
                )
                .replace("{U}", lab("FACILITY", unit))
                .replace("{P}", lab("FACILITY", parent))
            )
            phen += ["p5a_unit_at_parent"]
        elif kind == "article_cms":
            title = c.cms("care_site_cms")
            line = r.choice(
                (
                    "Admitted from the {O} after a fall.",
                    "Seen in the ED at {O} on {D}.",
                    "Transferred from {O}.",
                )
            ).replace("{O}", lab("FACILITY", title))
            phen += ["p4_full_care_site", "the_article"]
        else:
            fac = f"{c.stem()} {r.choice(('Family Health', 'Medical Group', 'Health Center'))}"
            line = r.choice(
                (
                    "She was seen at the {O} office.",
                    "The {O} nurse called back.",
                    "Moved care to the {O} practice.",
                )
            ).replace("{O}", lab("FACILITY", fac))
            phen += ["the_article"]
        if "{D}" in line:
            line = line.replace("{D}", c.date(allow_range=False))
        lines.append(line)
    return lines, phen, None


def _town(c: Ctx, role: str = "narrative_town") -> str:
    if c.rng.random() < 0.5:
        return c.stem()
    name, _ = c.city(role)
    return name


def core_town_narrative(c: Ctx, positive: bool, case: str):
    r = c.rng
    if positive:
        out, phen = [], ["town_narrative"]
        templates = [
            "Splits time between {T} and {T2}.",
            "Grew up in {T}.",
            "Raised in {T}; moved here for work.",
            "Son drives in from {T} on weekends.",
            "Worked in {T} for thirty years.",
            "Winters in {T}.",
            "Retired to {T} after the mill closed.",
            "Hometown is {T}.",
            "Born in {T}.",
            "Lives in the {H} neighborhood of {T}.",
            "Spends summers in {T} with her sister.",
            "Her late husband was from {T}.",
            "Often visits friends in {T}.",
            "Family farm is outside {T}.",
        ]
        for template in r.sample(templates, 1 if r.random() < 0.55 else 2):
            line = template
            if "{H}" in line:
                if r.random() < 0.5:
                    hood = c.stem()
                else:
                    hood, _ = c.city("neighborhood_locality", c.w.hood)
                line = line.replace("{H}", lab("LOCATION", hood))
                phen.append("neighborhood_of_town")
            if "{T2}" in line:
                line = line.replace("{T2}", lab("LOCATION", _town(c)))
            line = line.replace("{T}", lab("LOCATION", _town(c)))
            out.append(line)
        return out, phen, None
    kind = r.choice(("surname", "given", "ordinary", "ordinary"))
    if kind == "surname":
        word = c.w.name_word.next()
        c.record(word, "census_gazetteer_place_2024", "n5_place_word_as_name")
        last = r.choice(SURNAME_A) + r.choice(SURNAME_B)
        full = f"{word} {last}"
        if identity_split(canonical_identity_group(full)) != "train":
            full = f"{word} {r.choice(SURNAME_A)}{r.choice(SURNAME_B)}"
            while identity_split(canonical_identity_group(full)) != "train":
                full = f"{word} {r.choice(SURNAME_A)}{r.choice(SURNAME_B)}"
        c.identities.append(full)
        return (
            [
                r.choice(
                    (
                        "Mr. {N} reports no new symptoms.",
                        "Spoke with {N}, her neighbor, about rides.",
                        "{N} will bring the forms tomorrow.",
                        "Her friend {N} attended the visit.",
                    )
                ).replace("{N}", lab("NAME", full))
            ],
            ["n5_place_word_other_role", "name_boundaries"],
            "N5",
        )
    if kind == "given":
        word = c.w.name_word.next()
        c.record(word, "census_gazetteer_place_2024", "n5_place_word_as_name")
        full = f"{word} {r.choice(SURNAME_A)}{r.choice(SURNAME_B)}"
        while identity_split(canonical_identity_group(full)) != "train":
            full = f"{word} {r.choice(SURNAME_A)}{r.choice(SURNAME_B)}"
        c.identities.append(full)
        return (
            [
                r.choice(
                    (
                        "Daughter {N} manages the pillbox.",
                        "Granddaughter {N} visits on Sundays.",
                        "Son {N} is the health care proxy.",
                    )
                ).replace("{N}", lab("NAME", full))
            ],
            ["n5_place_word_other_role", "name_boundaries"],
            "N5",
        )
    word = c.w.ordinary_word.next()
    c.record(word, "census_gazetteer_place_2024", "n5_ordinary_word")
    phrase = r.choice(c.w.ordinary[word])
    return (
        [
            r.choice(
                (
                    "Needs {X}; order placed.",
                    "Reviewed {X} with the family.",
                    "Also asked about {X}.",
                    "Supplies: {X}.",
                )
            ).replace("{X}", phrase)
        ],
        ["n5_place_word_other_role"],
        "N5",
    )


def core_unusual_name(c: Ctx, positive: bool, case: str):
    """Explicitly named patient with a place word as a name (unusual_name cohort)."""
    r = c.rng
    word = c.w.name_word.next()
    c.record(word, "census_gazetteer_place_2024", "n5_place_word_as_name")
    full = word
    while identity_split(canonical_identity_group(full)) != "train":
        word = c.w.name_word.next()
        full = word
    c.identities.append(full)
    name = lab("NAME", full)
    line = r.choice(
        (
            "Patient name: {N}. {N2} agreed to the plan.",
            "The patient named {N} asked for a printed schedule.",
            "Preferred name is {N}; she answered all questions.",
            "Per intake, the patient is named {N}.",
        )
    )
    return (
        [line.replace("{N}", name).replace("{N2}", "She")],
        ["n5_place_word_other_role", "explicit_naming"],
        "N5_unusual_name",
    )


CORES = {
    "city_modifier": core_city_modifier,
    "locality_short": core_locality_short,
    "service_org": core_service_org,
    "practice_pharmacy": core_practice_pharmacy,
    "senior_living_at": core_senior_living,
    "town_narrative": core_town_narrative,
}


# --------------------------------------------------------------------------- layouts


# --------------------------------------------------------------------------- v2 cores
# New cases from the follow-up decisions of 2026-10-05. Each returns a core tuple or None to
# fall back to the v1 core. Only used when the world is v2, so v1 output is unchanged.

SITE_HEADS = (
    "Clinic",
    "Medical Center",
    "Health Center",
    "Urgent Care",
    "Family Clinic",
    "Medical Group",
    "Dialysis Center",
)
CARE_TEAM_TEMPLATES = (
    "The {L} hospice team visited today.",
    "The {L} home health team will call to schedule.",
    "Our {L} palliative care team saw her this morning.",
    "The {L} wound care team changed the dressing.",
    "Spoke with the {L} care team about discharge.",
    "The {L} dialysis team confirmed the chair time.",
    "Our {L} infusion team will draw labs on Monday.",
    "The {L} nursing team called with an update.",
)
JOBS = ("nurse", "aide", "cook", "technician", "janitor", "clerk", "driver", "cashier")
EMPLOYER_TEMPLATES = (
    "Works as a {J} at {O}",
    "Employed by {O} as a {J}",
    "Her husband works at {O}",
    "Son is a {J} at {O}",
    "Retired from {O} as a {J}",
)
CARE_ROLE_TEMPLATES = (
    "admitted to {F} overnight",
    "seen in the ED at {F} last month",
    "had surgery at {F} in the spring",
    "followed at {F} for the thyroid",
)
THE_ORG_ARTICLE = (
    "The {O} nurse called back.",
    "The {O} driver arrived early.",
    "Spoke with the {O} intake coordinator.",
    "The {O} social worker visited on Tuesday.",
)
THE_ORG_IN_NAME = (
    "Enrolled with {O} hospice program.",
    "Attends {O} adult day program three days a week.",
    "Referred to {O} home health team.",
    "Care is provided by {O} palliative program.",
    "Started with {O} home health nursing.",
)
COUNTY_TEMPLATES = (
    "Lives in {C}.",
    "Pt from {C}.",
    "Moved from {CS}.",
    "Daughter lives in {C}.",
    "Drives in from {C} for visits.",
    "Moved to {CS} last spring.",
    "Son lives in {CS}.",
    "Family farm is in {C}.",
)
COUNTY_NEGATIVES = (
    "The county health department sent a notice about flu shots.",
    "Per county guidance, masks are optional in waiting rooms.",
    "She volunteers at the county fair each August.",
    "A county nurse called about the vaccine clinic.",
    "Transport from the county senior van was arranged.",
    "County records were not needed for the form.",
)
INVENTED_INSTRUMENT_WORDS = (
    "criteria",
    "Classification",
    "Score",
    "rule",
    "Risk Model",
    "Index",
)
INVENTED_DISEASE_WORDS = ("virus", "fever", "disease", "syndrome")


def _org_name(c: Ctx) -> str:
    """An employer or care organisation name: invented or a train-pool CMS name."""
    r = c.rng
    if r.random() < 0.5:
        tail = r.choice(
            (
                "General Hospital",
                "Medical Center",
                "Community Hospital",
                "Health System",
            )
        )
        return f"{c.stem()} {tail}"
    return c.cms("employer_care_org")


def _employer_line(c: Ctx, care: bool):
    r = c.rng
    org = _org_name(c)
    job = r.choice(JOBS)
    first = r.choice(EMPLOYER_TEMPLATES).replace("{O}", org).replace("{J}", job)
    if not care:
        return first + "."
    second = r.choice(CARE_ROLE_TEMPLATES).replace("{F}", lab("FACILITY", org))
    subject = "she was " if first.startswith(("Her", "Son")) else ""
    return f"{first}; {subject}{second}."


def new_city_modifier(c: Ctx, positive: bool, case: str):
    r = c.rng
    if positive:
        x = r.random()
        if x < 0.22:
            if r.random() < 0.5:
                place, _ = c.city("named_site_place")
            else:
                place = c.stem()
            site = lab("FACILITY", f"{place} {r.choice(SITE_HEADS)}")
            if case == "normal":
                line = r.choice(
                    (
                        "Seen at our {F}.",
                        "Labs drawn at our {F}.",
                        "Follow up at {F} next week.",
                        "Records sent from {F}.",
                        "Referred to the {F}.",
                    )
                )
                tag = "named_site_capitalised"
            else:
                line = r.choice(
                    (
                        "Seen at {F}.",
                        "Follow up at {F} next week.",
                        "Records sent from {F}.",
                        "Referred to {F}.",
                        "Labs drawn at {F}.",
                    )
                )
                tag = "named_site_case_hidden"
            lines = [line.replace("{F}", site)]
            phen = [tag, "facility_boundary"]
            if r.random() < 0.5:
                other, _ = c.city("city_modifier_locality")
                lines.append(
                    r.choice(
                        (
                            "Labs at our {L} clinic.",
                            "Our {L} site sent records.",
                            "Follow up at the {L} office.",
                        )
                    ).replace("{L}", lab("LOCATION", other))
                )
                phen.append("p7_city_modifier")
            return lines, phen, None
        if x < 0.40:
            city, _ = c.city("care_team_locality")
            line = r.choice(CARE_TEAM_TEMPLATES).replace("{L}", lab("LOCATION", city))
            return [line], ["p7_city_modifier", "p7_care_team"], None
        return None
    if r.random() < 0.35:
        stem = c.stem()
        if r.random() < 0.55:
            name = f"{stem} {r.choice(INVENTED_INSTRUMENT_WORDS)}"
            line = r.choice(
                (
                    "Scored per the {X}.",
                    "{X} reviewed with the resident.",
                    "Risk estimated using the {X}.",
                    "Assessed with the {X} today.",
                )
            )
            return (
                [line.replace("{X}", name)],
                ["n1_named_instrument", "invented_place_decoy"],
                "N1_invented",
            )
        name = f"{stem} {r.choice(INVENTED_DISEASE_WORDS)}"
        line = r.choice(
            (
                "Tested for {X}; result pending.",
                "No exposure risk for {X} identified.",
                "Counselled about {X} prevention.",
            )
        )
        return (
            [line.replace("{X}", name)],
            ["n2_place_disease", "invented_place_decoy"],
            "N2_invented",
        )
    return None


def new_locality_short(c: Ctx, positive: bool, case: str):
    r = c.rng
    if positive:
        if r.random() >= 0.30:
            return None
        if r.random() < 0.6:
            value, state = c.county()
        else:
            value, state = f"{c.stem()} County", r.choice(STATES)[0]
        template = r.choice(COUNTY_TEMPLATES)
        line = template.replace("{CS}", lab("LOCATION", f"{value}, {state}")).replace(
            "{C}", lab("LOCATION", value)
        )
        return [line], ["county_name", "p1_patient_locality"], None
    if r.random() < 0.12:
        return [r.choice(COUNTY_NEGATIVES)], ["county_word_generic"], "county_generic"
    return None


def new_service_org(c: Ctx, positive: bool, case: str):
    r = c.rng
    if positive:
        x = r.random()
        if x < 0.10:
            a, b = c.stem(), c.stem()
            if r.random() < 0.5:
                name = f"The {a} at {b} {r.choice(RIDGE_WORDS)}"
                line = r.choice(THE_ORG_IN_NAME).replace("{O}", lab("FACILITY", name))
            else:
                tail = r.choice(("Hospice", "Home Health", "Adult Day Program"))
                name = f"The {a} {tail}"
                line = r.choice(
                    (
                        "Enrolled with {O}.",
                        "Referred to {O} for care at home.",
                        "Care is provided by {O}.",
                    )
                ).replace("{O}", lab("FACILITY", name))
            return (
                [line],
                ["the_in_name", "the_service_org_in_name", "service_organisation"],
                None,
            )
        if x < 0.20:
            kind, form = r.choice([f for f in SERVICE_FORMS if "{X}" not in f[1]])
            org = form.replace("{S}", c.stem())
            line = r.choice(THE_ORG_ARTICLE).replace("{O}", lab("FACILITY", org))
            return (
                [line],
                ["the_article", "the_service_org_article", "service_organisation"],
                None,
            )
        if x < 0.28:
            city, _ = c.city("care_team_locality")
            line = r.choice(
                (
                    "The {L} hospice team visited on {D}.",
                    "The {L} home health team starts care next week.",
                    "Our {L} hospice nurse will call tomorrow.",
                )
            )
            line = line.replace("{L}", lab("LOCATION", city)).replace(
                "{D}", c.date(allow_range=False)
            )
            return (
                [line],
                ["p7_city_modifier", "p7_care_team", "service_organisation"],
                None,
            )
        if x < 0.38:
            return (
                [_employer_line(c, True)],
                ["employer_and_care_roles", "same_spelling_different_context"],
                None,
            )
        return None
    if r.random() < 0.12:
        return [_employer_line(c, False)], ["employer_name"], "employer_only"
    return None


def new_practice_pharmacy(c: Ctx, positive: bool, case: str):
    r = c.rng
    if positive:
        x = r.random()
        if x < 0.14:
            return (
                [_employer_line(c, True)],
                ["employer_and_care_roles", "same_spelling_different_context"],
                None,
            )
        if x < 0.34:
            form = r.choice(
                ("{S} Drug", "{S} Pharmacy", "{S} Drug & Gift", "{S} Apothecary")
            )
            org = lab("FACILITY", form.replace("{S}", c.stem()))
            line = r.choice(
                (
                    "Refills at {O}.",
                    "Send it to {O}.",
                    "Rx called in to {O}.",
                    "Pick up at {O} after 4 pm.",
                    "Pharmacy: {O}.",
                )
            ).replace("{O}", org)
            return (
                [line],
                ["named_pharmacy", "practice_without_care_word", "facility_boundary"],
                None,
            )
        return None
    if r.random() < 0.12:
        return [_employer_line(c, False)], ["employer_name"], "employer_only"
    return None


V2_NEW = {
    "city_modifier": new_city_modifier,
    "locality_short": new_locality_short,
    "service_org": new_service_org,
    "practice_pharmacy": new_practice_pharmacy,
}


def _wrap_v2(family):
    base, new = CORES[family], V2_NEW.get(family)

    def core(c, positive, case):
        if new is not None:
            result = new(c, positive, case)
            if result is not None:
                return result
        return base(c, positive, case)

    return core


V2_CORES = {family: _wrap_v2(family) for family in CORES}


def core_for(c: Ctx, family: str):
    return (V2_CORES if c.w.v2 else CORES)[family]


class Patient:
    def __init__(self, c: Ctx):
        self.c = c
        self.name = c.name("patient")
        self.year = c.birth_year()
        self.dob = c.dob(self.year)
        self.mrn = c.mrn()
        self.phone = c.phone()
        self.fax = c.phone()
        self.age = c.age_phrase(self.year)
        self.address = c.address()
        self.doc = c.clinician()
        self.doc2 = c.clinician()
        self.nurse = c.nurse()
        self.date = c.date(allow_range=False)
        self.date2 = c.date(allow_range=False)


def _body(core, fluff, joiner=" "):
    return joiner.join(core + fluff)


def L_fax_cover(p, core, fl):
    return f"FAX COVER SHEET\nTo: {p.doc}\nFax: {p.fax}\nFrom: {p.nurse}\nRe: {p.name}, DOB {p.dob}, MRN: {p.mrn}\nDate: {p.date}\n{_body(core, fl[:1])}\nPages: {p.c.rng.randint(2, 9)} including cover. Call {p.phone} with questions."


def L_referral_form(p, core, fl):
    return f"Referral request\nPatient: {p.name}\nDOB: {p.dob}\nPhone: {p.phone}\nMRN: {p.mrn}\nReason: follow up after recent visit.\nNotes: {_body(core, fl[:2])}\nReferring clinician: {p.doc}"


def L_home_health_soc(p, core, fl):
    return f"Start of care visit {p.date}\n{p.name}, {p.age}. MRN: {p.mrn}\nHome address: {p.address}\n{_body(core, fl[:2], chr(10))}\nVisit by {p.nurse}. Physician: {p.doc}."


def L_discharge_summary(p, core, fl):
    return f"DISCHARGE SUMMARY\nPatient: {p.name}   MRN: {p.mrn}\nAdmitted {p.date}; discharged {p.date2}.\nCourse: {fl[0]}\nDisposition: {_body(core, [])}\nFollow up with {p.doc} in two weeks."


def L_transfer_summary(p, core, fl):
    return f"Transfer summary for {p.name} ({p.age}).\nDOB {p.dob}. MRN: {p.mrn}.\n{_body(core, fl[:2])}\nHandoff given by {p.nurse} on {p.date}; callback {p.phone}."


def L_med_delivery(p, core, fl):
    return f"Medication delivery note\nRecipient: {p.name}\nDelivery address: {p.address}\nDate: {p.date}\n{_body(core, fl[:1])}\nQuestions: {p.phone}"


def L_equipment_delivery(p, core, fl):
    return f"Equipment delivery ticket {p.date}\nPatient {p.name}, phone {p.phone}\nOrder no. {p.c.idnum()}\n{_body(core, [], chr(10))}\nSigned for by family member."


def L_care_team(p, core, fl):
    return f"Care team for {p.name}\nPhysician: {p.doc} {p.phone}\nCase manager: {p.nurse}\nMRN: {p.mrn}\nNotes: {_body(core, fl[:1])}"


def L_portal_message(p, core, fl):
    return f"Portal message from {p.name}\nSent {p.date}\nHello, {_body(core, [])} Please let me know if you need anything else.\nThank you, {p.name}\nReply link: {p.c.url()}"


def L_telephone_note(p, core, fl):
    return f"Telephone encounter {p.date}\nCaller: {p.name} at {p.phone}\n{_body(core, fl[:1])}\nAdvised to call back if symptoms change. Message taken by {p.nurse}."


def L_intake_form(p, core, fl):
    return f"INTAKE FORM\nName: {p.name}\nDate of birth: {p.dob}\nAddress: {p.address}\nPhone: {p.phone}\nMRN: {p.mrn}\nComments: {_body(core, [])}"


def L_one_line(p, core, fl):
    return f"{p.name}: {_body(core, [])}"


def L_soap(p, core, fl):
    return f"S: {p.age} seen for follow up. {_body(core, fl[:1])}\nO: {fl[1]}\nA/P: continue plan; recheck in clinic.\n{p.name}, MRN: {p.mrn}. Seen {p.date} with {p.doc}."


def L_nursing_handoff(p, core, fl):
    return f"Nursing handoff {p.date}\n{p.name}, {p.age}\n{_body(core, fl[:2], chr(10))}\nReport given by {p.nurse}."


def L_social_work(p, core, fl):
    return f"Social work note\nClient: {p.name}, DOB {p.dob}\nPhone: {p.phone}\n{_body(core, fl[:1])}\nPlan reviewed with {p.nurse} on {p.date}."


def L_case_management(p, core, fl):
    return f"Case management review {p.date}\nMember: {p.name}  Member ID: {p.c.idnum()}\n{_body(core, fl[:1], chr(10))}\nCM: {p.nurse}"


def L_pharmacy_refill(p, core, fl):
    return f"Refill request\nPatient: {p.name}   DOB: {p.dob}\nPrescriber: {p.doc}\nPhone: {p.phone}\nComment: {_body(core, [])}"


def L_lab_requisition(p, core, fl):
    return f"Laboratory requisition\nPatient: {p.name}\nDOB: {p.dob}   MRN: {p.mrn}\nAccession: {p.c.idnum()}\nCollected {p.date}\nClinical note: {_body(core, [])}\nOrdered by {p.doc}."


def L_appointment(p, core, fl):
    return f"Appointment reminder for {p.name}\nYour visit is on {p.date}. Please call {p.phone} to confirm.\n{_body(core, [])}"


def L_care_plan(p, core, fl):
    return f"Care plan summary\nResident: {p.name}, {p.age}\nReviewed {p.date} with {p.doc}\n{_body(core, fl[:2], chr(10))}"


def L_hospice_admission(p, core, fl):
    return f"Admission assessment {p.date}\n{p.name}, {p.age}. MRN: {p.mrn}\n{_body(core, fl[:2])}\nSigned: {p.nurse}"


def L_therapy_eval(p, core, fl):
    return f"Physical therapy evaluation\nPatient: {p.name}   DOB: {p.dob}\nReferring physician: {p.doc}\nHistory: {_body(core, fl[:1])}\nPlan: therapy twice weekly. Evaluated {p.date}."


def L_email_forward(p, core, fl):
    return f"From: {p.c.email()}\nTo: {p.c.email('referrals')}\nSubject: {p.name}\nSent {p.date} from {lab('IP', '192.0.2.' + str(p.c.rng.randint(2, 250)))}\n\n{_body(core, [])}\nRegards, {p.nurse}"


def L_secure_chat(p, core, fl):
    return f"[{p.date}] {p.nurse}: re {p.name} - {_body(core, [])}\n[{p.date2}] {p.doc}: thanks, noted."


def L_consult_request(p, core, fl):
    return f"Consult request\nPatient: {p.name}, {p.age}\nRequested by: {p.doc}\nQuestion: {_body(core, fl[:1])}\nPlease call {p.phone}."


def L_prior_auth(p, core, fl):
    return f"Prior authorization request\nMember: {p.name}   Member ID: {p.c.idnum()}\nDOB: {p.dob}   Account: {p.c.account()}\nPrescriber: {p.doc}\nSupporting note: {_body(core, [])}"


def L_wound_care(p, core, fl):
    return f"Wound care visit {p.date}\n{p.name}, MRN: {p.mrn}\n{fl[0]} {_body(core, fl[1:2])}\nNext dressing change in three days. {p.nurse}."


def L_clinic_hpi(p, core, fl):
    return f"HPI: {p.name} is a {p.age} seen today. {_body(core, fl[:2])}\nPCP {p.doc}. Phone {p.phone}."


def L_med_rec(p, core, fl):
    return f"Medication reconciliation {p.date}\nPatient: {p.name}   MRN: {p.mrn}\n1. {fl[0]}\n2. {_body(core, [])}\nReviewed by {p.nurse}."


def L_visit_letter(p, core, fl):
    return f"Dear {p.name},\nThank you for coming in on {p.date}. {_body(core, fl[:1])}\nSincerely,\n{p.doc}"


LAYOUTS = {
    "fax_cover": L_fax_cover,
    "referral_form": L_referral_form,
    "home_health_soc": L_home_health_soc,
    "discharge_summary": L_discharge_summary,
    "transfer_summary": L_transfer_summary,
    "med_delivery": L_med_delivery,
    "equipment_delivery": L_equipment_delivery,
    "care_team_list": L_care_team,
    "portal_message": L_portal_message,
    "telephone_note": L_telephone_note,
    "intake_form": L_intake_form,
    "one_line_query": L_one_line,
    "soap_note": L_soap,
    "nursing_handoff": L_nursing_handoff,
    "social_work_note": L_social_work,
    "case_management": L_case_management,
    "pharmacy_refill": L_pharmacy_refill,
    "lab_requisition": L_lab_requisition,
    "appointment_reminder": L_appointment,
    "care_plan_summary": L_care_plan,
    "hospice_admission": L_hospice_admission,
    "therapy_eval": L_therapy_eval,
    "email_forward": L_email_forward,
    "secure_chat": L_secure_chat,
    "consult_request": L_consult_request,
    "prior_auth": L_prior_auth,
    "wound_care": L_wound_care,
    "clinic_hpi": L_clinic_hpi,
    "med_rec": L_med_rec,
    "visit_letter": L_visit_letter,
}


def long_note(p: Patient, c: Ctx, family: str, case: str):
    """A multi-window note built from several cores of one family plus filler."""
    sections, phen = [], set()
    heads = (
        "History",
        "Social history",
        "Home situation",
        "Medications",
        "Services",
        "Interval events",
        "Plan",
        "Family discussion",
    )
    flavours = [True, False, True, True, False, True, True, False]
    for head, positive in zip(heads, flavours):
        lines, ph, _ = core_for(c, family)(c, positive, case)
        phen.update(ph)
        paragraph = " ".join(lines + c.fluff(5))
        sections.append(f"{head}\n{paragraph}")
    text = (
        f"Comprehensive summary for {p.name}, {p.age}. MRN: {p.mrn}. DOB {p.dob}.\n"
        f"Attending: {p.doc}. Nurse: {p.nurse}. Address on file: {p.address}. Phone {p.phone}.\n\n"
        + "\n\n".join(sections)
        + f"\n\nNext review on {p.date} with {p.doc2}."
    )
    return text, sorted(phen)


# --------------------------------------------------------------------------- generation


def sha256_text(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def plan_units(rng: random.Random, family_rows=None):
    """Plan row units: pairs (positive + matched negative), single positives, long notes."""
    units = []
    for family, rows in (family_rows or FAMILY_ROWS).items():
        long_rows = round(rows * LONG_FRACTION)
        unusual_rows = (
            round(rows * UNUSUAL_NAME_FRACTION) if family == "town_narrative" else 0
        )
        neg_total = round(rows / 3)
        pair_count = neg_total
        singles = rows - 2 * pair_count - long_rows - unusual_rows
        if singles < 0:
            raise ValueError("row plan inconsistent")
        for _ in range(pair_count):
            units.append({"family": family, "kind": "pair", "n": 2})
        for _ in range(singles):
            units.append({"family": family, "kind": "single", "n": 1})
        for _ in range(long_rows):
            units.append({"family": family, "kind": "long", "n": 1})
        for _ in range(unusual_rows):
            units.append({"family": family, "kind": "unusual", "n": 1})
    # The matched negatives total about a third of rows; pairs carry them.
    rng.shuffle(units)
    names = sorted(LAYOUTS)
    cycle: list[str] = []
    for unit in units:
        if unit["kind"] == "long":
            unit["layout"] = LONG_LAYOUT
            continue
        if not cycle:
            cycle = list(names)
            rng.shuffle(cycle)
        unit["layout"] = cycle.pop()
    for index, unit in enumerate(units):
        unit["serial"] = index
    return units


def assign_cases(units, rng: random.Random):
    """Whole-unit case assignment hitting ~60/20/20 over rows; flagged units stay normal."""
    total = sum(u["n"] for u in units)
    quota = {
        "upper": round(total * CASE_TARGET["upper"]),
        "lower": round(total * CASE_TARGET["lower"]),
    }
    order = [u for u in units if not u.get("normal_only")]
    rng.shuffle(order)
    used = Counter()
    for unit in order:
        unit["case"] = "normal"
        for case in ("upper", "lower"):
            if used[case] + unit["n"] <= quota[case]:
                unit["case"] = case
                used[case] += unit["n"]
                break
    for unit in units:
        if unit.get("normal_only"):
            unit["case"] = "normal"
    return used


def generate(seed: int = SEED, version: str = "v1"):
    rng = random.Random(seed)
    world = World(rng, version)
    units = plan_units(rng, FAMILY_ROWS_V2 if world.v2 else None)
    suite = SUITE_V2 if world.v2 else SUITE
    for unit in units:
        unit["suite"] = suite
    texts_seen = set()
    if world.v2:
        # v2: the authored role decides in every case, so no row is forced to normal case and
        # the final case is fixed before authoring (named-site wording depends on it).
        assign_cases(units, random.Random(seed + 2))
        for unit in units:
            unit["content"] = author_unit(world, rng, unit, texts_seen)
    else:
        # Case is needed while authoring (named-site contrast), so draw a provisional case first.
        # Units whose gold depends on capitalisation are then forced to normal case.
        provisional = random.Random(seed + 1)
        for unit in units:
            unit["case"] = provisional.choices(
                ("normal", "upper", "lower"), (0.6, 0.2, 0.2)
            )[0]
        for unit in units:
            content = author_unit(world, rng, unit, texts_seen)
            unit["content"] = content
            if any("normal_only" in item["flags"] for item in content):
                unit["normal_only"] = True
        # Enforce the target mix over whole units (normal_only units stay normal).
        assign_cases(units, random.Random(seed + 2))
    out = []
    for unit in units:
        if unit.get("normal_only") and unit["case"] != "normal":
            raise AssertionError("normal-only unit assigned a non-normal case")
        for item in unit["content"]:
            text, spans = build_row(item["marked"], unit["case"])
            out.append(finish_row(item, unit, text, spans, len(out)))
    return out, world


def author_unit(world: World, rng: random.Random, unit: dict, seen: set) -> list[dict]:
    family, kind, layout = unit["family"], unit["kind"], unit["layout"]
    for _attempt in range(200):
        c = Ctx(world, rng)
        p = Patient(c)
        base = len(c.reals)
        items = []
        if kind == "long":
            marked, phen = long_note(p, c, family, unit["case"])
            items.append(
                make_item(
                    c,
                    p,
                    family,
                    marked,
                    phen,
                    "positive",
                    None,
                    layout,
                    "long_document",
                    c.reals,
                )
            )
        elif kind == "unusual":
            lines, phen, neg = core_unusual_name(c, False, unit["case"])
            marked = LAYOUTS[layout](p, lines, c.fluff(3))
            items.append(
                make_item(
                    c,
                    p,
                    family,
                    marked,
                    phen,
                    "negative",
                    neg,
                    layout,
                    "unusual_name",
                    c.reals,
                )
            )
        else:
            polarities = ["positive", "negative"] if kind == "pair" else ["positive"]
            for polarity in polarities:
                start = len(c.reals)
                lines, phen, neg = core_for(c, family)(
                    c, polarity == "positive", unit["case"]
                )
                marked = LAYOUTS[layout](p, lines, c.fluff(3))
                reals = c.reals[:base] + c.reals[start:]
                items.append(
                    make_item(
                        c,
                        p,
                        family,
                        marked,
                        phen,
                        polarity,
                        neg,
                        layout,
                        "realistic",
                        reals,
                    )
                )
        # Reject duplicate texts across the bank.
        keys = [re.sub(r"\[\[[A-Z]+\|(.*?)\]\]", r"\1", i["marked"]) for i in items]
        if any(k in seen for k in keys) or len(set(keys)) != len(keys):
            continue
        seen.update(keys)
        # A pair shares identity, pair id and layout; the negative inherits the same keys.
        pair_id = f"{'tk13v2' if world.v2 else 'tk13'}-{unit['serial']:05d}"
        for number, item in enumerate(items):
            item["pair_id"] = pair_id
            item["row_number"] = number
        return items
    raise RuntimeError("could not author a unique unit")


def make_item(
    c: Ctx, p: Patient, family, marked, phen, polarity, neg_type, layout, cohort, reals
):
    return {
        "marked": marked,
        "phenomena": list(phen),
        "polarity": polarity,
        "negative_type": neg_type,
        "layout": layout,
        "cohort": cohort,
        "family_key": family,
        "reals": [dict(x) for x in reals],
        "patient": p.name,
        "flags": sorted(c.flags),
    }


def plain_name(text: str) -> str:
    """Canonical `First Last` from a span written `Last, First`."""
    if ", " in text:
        last, first = text.split(", ", 1)
        return f"{first} {last}"
    return text


def finish_row(
    item: dict, unit: dict, text: str, spans: list[dict], index: int
) -> dict:
    errors = annotation_errors(text, spans)
    if errors:
        raise ValueError(f"{item['pair_id']}: {errors}\n{text}")
    plain_patient = plain_name(re.sub(r"\[\[NAME\|(.*?)\]\]", r"\1", item["patient"]))
    identity_group = canonical_identity_group(plain_patient)
    groups = sorted(
        {
            canonical_identity_group(plain_name(s["text"]))
            for s in spans
            if s["label"] == "NAME"
        }
        | {identity_group}
    )
    for g in groups:
        assert identity_split(g) == "train", "identity outside the training partition"
    family = f"tk13_{item['family_key']}"
    suite = unit.get("suite", SUITE)
    phen = set(item["phenomena"])
    if suite == SUITE_V2 and unit["case"] != "normal":
        if "the_in_name" in phen:
            phen.add("the_in_name_case_hidden")
        if "p7_city_modifier" in phen:
            phen.add("p7_case_hidden")
        if "named_pharmacy" in phen:
            phen.add("named_pharmacy_case_hidden")
        if "n5_place_word_other_role" in phen and not phen & {
            "name_boundaries",
            "explicit_naming",
        }:
            phen.add("ordinary_place_word_case_hidden")
    item = {**item, "phenomena": sorted(phen)}
    return {
        "id": f"{suite}-{index:05d}",
        "text": text,
        "spans": spans,
        "split": "train",
        "suite": suite,
        "annotation_policy": POLICY_VERSION,
        "cohort": item["cohort"],
        "family": family,
        "template_family": family,
        "layout": item["layout"],
        "phenomena": sorted(set(item["phenomena"])),
        "identity_group": identity_group,
        "identity_groups": groups,
        "pair_id": item["pair_id"],
        "polarity": item["polarity"],
        "negative_type": item["negative_type"],
        "case_mode": unit["case"],
        "pool": item["reals"],
    }


# --------------------------------------------------------------------------- output


def write_bank(rows, world, seed: int = SEED, version: str = "v1"):
    out = OUT_V2 if version == "v2" else OUT
    payload = (
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=False) for r in rows)
        + "\n"
    ).encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    support = Counter(s["label"] for r in rows for s in r["spans"])
    label_support = {label: support[label] for label in ENTITY_TYPES}
    manifest = {
        "annotation_policy": POLICY_VERSION,
        "places_addendum": ADDENDUM,
        "suite": SUITE_V2 if version == "v2" else SUITE,
        "version": BANK_VERSION_V2 if version == "v2" else "tk13-v1",
        "split": "train",
        "rows": len(rows),
        "documents": len(rows),
        "sha256": sha256_text(payload),
        "independent_final": False,
        "label_support": label_support,
        "entity_support": dict(label_support),
        "label_counts": dict(label_support),
        "cohort_counts": dict(Counter(r["cohort"] for r in rows)),
        "family_counts": dict(Counter(r["family"] for r in rows)),
        "layout_counts": dict(Counter(r["layout"] for r in rows)),
        "case_counts": dict(Counter(r["case_mode"] for r in rows)),
        "polarity_counts": dict(Counter(r["polarity"] for r in rows)),
        "author_provenance": {
            "generator": "scripts/takeover/author13_care_sites.py",
            "generator_sha256": sha256_file(Path(__file__)),
            "seed": seed,
            "brief": "docs/care-site-locality-bank-brief.md",
            "isolation": "authored from the policy, fixtures, validator and pinned public sources only",
            "overlap_audits": "not run by the author; the owner's agent runs the sealed-overlap audit and the development overlap check",
        },
        "source_pins": world.pins,
        "value_pools": {
            "rule": "int(sha256(nfc(casefold(value)))[:8], 16) % 10 < 8 is train",
            "used": "train only",
        },
    }
    Path(str(out) + ".manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def value_shares(rows):
    counter: dict[str, set] = defaultdict(set)
    for r in rows:
        for s in r["spans"]:
            if s["label"] in ("FACILITY", "LOCATION"):
                counter[s["text"].casefold()].add(r["id"])
        for real in r["pool"]:
            counter["real:" + real["value"].casefold()].add(r["id"])
    return {k: len(v) for k, v in counter.items()}


def check_bank(rows):
    shares = value_shares(rows)
    limit = MAX_VALUE_SHARE * len(rows)
    worst = max(shares.items(), key=lambda kv: kv[1])
    if worst[1] > limit:
        raise ValueError(
            f"value {worst[0]!r} appears in {worst[1]} rows (limit {limit})"
        )
    heldout = [
        x
        for r in rows
        for x in r["pool"]
        if place_value_pool(x["value"]) != "train" or x["pool"] != "train"
    ]
    if heldout:
        raise ValueError(f"heldout values present: {heldout[:3]}")
    low = " ".join(r["text"].casefold() for r in rows)
    bad = [b for b in BANNED_FRAGMENTS if b in low]
    if bad:
        raise ValueError(f"banned fragments in bank: {bad}")
    return shares


def write_reports(rows, manifest, world):
    from clinical_deid.data.annotation_policy import COHORTS  # noqa: F401

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    shares = value_shares(rows)
    n = len(rows)
    fam = Counter(r["family"] for r in rows)
    lay = Counter(r["layout"] for r in rows)
    case = Counter(r["case_mode"] for r in rows)
    pol = Counter(r["polarity"] for r in rows)
    negtypes = Counter(
        (r["family"], r["negative_type"]) for r in rows if r["polarity"] == "negative"
    )
    pools = Counter(x["pool"] for r in rows for x in r["pool"])
    sources = Counter(x["source"] for r in rows for x in r["pool"])
    distinct_reals = len({x["value"].casefold() for r in rows for x in r["pool"]})
    span_shares = {k: v for k, v in shares.items() if not k.startswith("real:")}
    real_shares = {k[5:]: v for k, v in shares.items() if k.startswith("real:")}
    top_span = sorted(span_shares.items(), key=lambda kv: -kv[1])[:5]
    top_real = sorted(real_shares.items(), key=lambda kv: -kv[1])[:5]
    phen = Counter(p for r in rows for p in r["phenomena"])
    lines = [
        "# Author report: tk13 care sites and localities",
        "",
        f"Bank: `data/tk13_train_care_sites_localities.jsonl` ({n} rows, sha256 `{manifest['sha256']}`).",
        f"Generator: `scripts/takeover/author13_care_sites.py`, seed {SEED}, sha256 `{manifest['author_provenance']['generator_sha256']}`.",
        "This is a synthetic training bank written from the policy, its fixtures and pinned public sources. "
        "Scores on it are not evidence of generalisation. The sealed-overlap audit and the development overlap "
        "check were not run by the author.",
        "",
        "## Rows per family",
        "",
        "| Family | Rows |",
        "|---|---:|",
        *[f"| {k} | {v} |" for k, v in sorted(fam.items())],
        "",
        f"Positive rows {pol['positive']}, matched negative rows {pol['negative']} "
        f"({pol['negative'] / n:.1%}). A negative has no LOCATION or FACILITY span; it still carries other PHI "
        "(names, dates, phones, MRNs, addresses) under the policy.",
        "",
        "Negative rows by family and type:",
        "",
        "| Family | Negative type | Rows |",
        "|---|---|---:|",
        *[
            f"| {f} | {t} | {v} |"
            for (f, t), v in sorted(
                negtypes.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))
            )
        ],
        "",
        "## Rows per layout",
        "",
        f"{len(lay)} layouts; the largest has {max(lay.values())} rows ({max(lay.values()) / n:.1%}; limit 8%).",
        "",
        "| Layout | Rows | Share |",
        "|---|---:|---:|",
        *[
            f"| {k} | {v} | {v / n:.1%} |"
            for k, v in sorted(lay.items(), key=lambda kv: -kv[1])
        ],
        "",
        "## Case mix (whole rows)",
        "",
        "| Case | Rows | Share |",
        "|---|---:|---:|",
        *[
            f"| {k} | {case[k]} | {case[k] / n:.1%} |"
            for k in ("normal", "upper", "lower")
        ],
        "",
        "Rows whose gold depends on capitalisation (a leading `The` that is part of a name, and a named site such "
        "as `<City> Clinic` contrasted with `our <City> clinic`) are authored in normal case only, because a lowercase "
        "or all-caps row cannot show the distinction.",
        "",
        "## Label support",
        "",
        "| Label | Spans |",
        "|---|---:|",
        *[f"| {k} | {v} |" for k, v in manifest["label_support"].items()],
        "",
        "## Cohorts",
        "",
        "| Cohort | Rows |",
        "|---|---:|",
        *[f"| {k} | {v} |" for k, v in sorted(manifest["cohort_counts"].items())],
        "",
        "`long_document` rows are multi-window notes built from several cores of one family plus filler; their "
        "token lengths are in `validate.json`. `unusual_name` rows name the patient with an ordinary place word "
        "(explicit naming, N5).",
        "",
        "## Real values and pools",
        "",
        f"Real value uses recorded in row `pool` metadata: {sum(pools.values())} "
        f"({distinct_reals} distinct values). Pool counts: train {pools['train']}, heldout {pools['heldout']}.",
        "",
        "| Source | Uses |",
        "|---|---:|",
        *[f"| {k} | {v} |" for k, v in sorted(sources.items())],
        "",
        f"Train-pool CMS care-site names available after cleaning: {world.cms_count}. The county file is pinned "
        "but unused (see unsettled cases).",
        "",
        "## Repetition",
        "",
        f"Limit: {MAX_VALUE_SHARE:.1%} of rows ({MAX_VALUE_SHARE * n:.0f} rows). Largest single FACILITY or "
        f"LOCATION span value: {top_span[0][1]} rows ({top_span[0][1] / n:.2%}). Largest single real value "
        f"(any use): {top_real[0][1]} rows ({top_real[0][1] / n:.2%}).",
        "",
        "Top five span values (row counts): "
        + ", ".join(f"`{k}` {v}" for k, v in top_span),
        "",
        "Top five real values (row counts): "
        + ", ".join(f"`{k}` {v}" for k, v in top_real),
        "",
        "## Phenomena",
        "",
        "| Phenomenon | Rows |",
        "|---|---:|",
        *[f"| {k} | {v} |" for k, v in sorted(phen.items())],
        "",
        "## Design notes",
        "",
        "- Spans are exact Python offsets built after the case mode is applied to each piece; none has edge whitespace "
        "or overlaps (checked by `annotation_errors`).",
        "- P7: LOCATION covers the city only; `clinic`, `office`, `site`, `unit`, `branch`, `location` stay outside.",
        "- D2: a service organisation is FACILITY including its type words; the same type words without a name are "
        "not annotated. An employer with no care role (`<Name> Logistics`) is not annotated.",
        "- D3: `The` is inside FACILITY only for authored names such as `The <Stem> at <Place> Ridge`; `the` as an "
        "article before a name stays outside.",
        "- P5a: `<Name> Family Medicine at <Town>` is one FACILITY; `<Name> Health Center` at `<Town>, <ST>` is a "
        "FACILITY and a LOCATION; a named unit at its parent is two FACILITY spans.",
        "- MRN captions (`MRN:`) are outside the span; names exclude honorifics and credentials; ages are annotated "
        "only above 89 and only the number or spelled number.",
        "- Invented facility and town stems are generated from syllables and exclude every Gazetteer place name and "
        "CMS city, so no invented value coincides with a real one.",
        "- N1 and N2 negatives use published instrument and disease names whose place word is in the Gazetteer and "
        "in the train pool; no instrument or disease name was invented.",
        "",
        "## Cases left out as unsettled",
        "",
        "The policy and brief do not settle these. None appears in the bank.",
        "",
        "1. **County names** (`lives in <Name> County`): the addendum permits counties as place values, but nothing "
        "says whether the word `County` is inside the LOCATION span. The county file is pinned and not used.",
        "2. **Capitalised generic head words** (`our Vellmont Clinic` versus `Vellmont Clinic` as a named site): in "
        "title case the P7 and P3 readings are indistinguishable without authoring intent. P7 rows use lowercase "
        "head words; `<City> Clinic` appears only in the named-site (FACILITY) reading and only in normal case.",
        "3. **`The` in lowercase and all-caps rows**: D3 turns on capital T as part of a brand. Whether `the marlowe`-style "
        "names keep `the` inside when the whole row is lowercased is undecided, so those rows are normal case only. "
        "The article reading (`the <Name> office`) appears in all three cases.",
        "4. **An article that is part of a generic phrase before a service organisation** (`the Vossmere VNA`): "
        "treated as an article (outside) by the leading-`the` rule; no row puts a capitalised `The` before an "
        "organisation to test the boundary.",
        "5. **City inside the name of a service organisation or practice** (`<City> VNA`): FACILITY on the whole "
        "name by D2; the city is not a nested LOCATION. Used in about a fifth of service-organisation rows. "
        "No row has a city next to a type word where the reading could be P7 (`<City> hospice team`).",
        "6. **A pharmacy or laboratory named only by place plus a type word, in lowercase** "
        "(`<city> pharmacy`): ambiguous between FACILITY and P7; not authored.",
        "7. **Place words as ordinary words in all-caps** (`READING GLASSES`): left in the bank as unannotated N5, "
        "which assumes the ordinary reading survives case folding.",
        "8. **Employer names inside a longer organisation phrase** (`works at <Name> Logistics`): authored only as "
        "plain unannotated employers per the employer rule; employers that also provide care are not authored.",
        "9. **N1 and N2 coverage**: only the published instruments and diseases in the train pool are used, "
        "so the negative set for these types is small and name-repeating (each at most the repetition limit).",
        "10. **Source location**: the pinned sources were not present under the main checkout's `.cache/place_sources`; "
        "they were read from the same-hash copy in a sibling worktree. Hashes match the pins.",
        "",
    ]
    (REPORT_DIR / "AUTHOR_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    sample_rng = random.Random(13)
    sample = sample_rng.sample(rows, 100)

    def mark(row):
        out, previous = [], 0
        for s in sorted(row["spans"], key=lambda s: s["start"]):
            out.append(row["text"][previous : s["start"]])
            out.append(f"[{s['label']}|{s['text']}]")
            previous = s["end"]
        out.append(row["text"][previous:])
        return "".join(out)

    parts = [
        "# Spot-check sample: tk13 care sites and localities",
        "",
        "100 rows drawn with `random.Random(13)`. Spans are shown as `[LABEL|text]`. Check against the policy, "
        "including P7, D2 and D3. Synthetic text; real place names are Census and CMS public values from the "
        "train pool.",
        "",
    ]
    for row in sample:
        parts += [
            f"### {row['id']}  ({row['family']}, {row['layout']}, {row['case_mode']}, {row['polarity']}"
            + (f", {row['negative_type']}" if row["negative_type"] else "")
            + ")",
            "",
            "```",
            mark(row),
            "```",
            "",
        ]
    (REPORT_DIR / "spot_check_sample.md").write_text("\n".join(parts), encoding="utf-8")


NEW_CASE_TAGS = (
    "county_name",
    "county_word_generic",
    "named_site_capitalised",
    "named_site_case_hidden",
    "p7_care_team",
    "p7_case_hidden",
    "the_in_name_case_hidden",
    "the_service_org_in_name",
    "the_service_org_article",
    "named_pharmacy",
    "named_pharmacy_case_hidden",
    "employer_and_care_roles",
    "employer_name",
    "invented_place_decoy",
    "ordinary_place_word_case_hidden",
)
V2_MARKER = "<!-- tk13-v2-section -->"


def _mark(row):
    out, previous = [], 0
    for s in sorted(row["spans"], key=lambda s: s["start"]):
        out.append(row["text"][previous : s["start"]])
        out.append(f"[{s['label']}|{s['text']}]")
        previous = s["end"]
    out.append(row["text"][previous:])
    return "".join(out)


def write_reports_v2(rows, manifest, world):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    fam = Counter(r["family"] for r in rows)
    lay = Counter(r["layout"] for r in rows)
    case = Counter(r["case_mode"] for r in rows)
    pol = Counter(r["polarity"] for r in rows)
    tags = Counter(t for r in rows for t in r["phenomena"] if t in NEW_CASE_TAGS)
    tag_case = Counter(
        (t, r["case_mode"]) for r in rows for t in r["phenomena"] if t in NEW_CASE_TAGS
    )
    pools = Counter(x["pool"] for r in rows for x in r["pool"])
    sources = Counter(x["source"] for r in rows for x in r["pool"])
    shares = value_shares(rows)
    span_shares = {k: v for k, v in shares.items() if not k.startswith("real:")}
    real_shares = {k[5:]: v for k, v in shares.items() if k.startswith("real:")}
    top_span = sorted(span_shares.items(), key=lambda kv: -kv[1])[:3]
    top_real = sorted(real_shares.items(), key=lambda kv: -kv[1])[:3]
    decoys = Counter()
    for r in rows:
        for x in r["pool"]:
            if x["role"] in ("n1_named_instrument", "n2_place_disease"):
                decoys[x["value"]] += 1
    worst_decoy = max(decoys.items(), key=lambda kv: kv[1]) if decoys else ("", 0)
    new_rows = sum(1 for r in rows if any(t in NEW_CASE_TAGS for t in r["phenomena"]))
    lines = [
        V2_MARKER,
        "",
        "# v2 bank: follow-up decisions of 2026-10-05",
        "",
        f"Bank: `data/tk13_train_care_sites_localities__v2.jsonl` ({n} rows, sha256 `{manifest['sha256']}`), "
        f"seed {SEED_V2}, version `{BANK_VERSION_V2}`, generator sha256 `{manifest['author_provenance']['generator_sha256']}`. "
        "The v1 bank and manifest are unchanged; v1 is reproduced by `generate(SEED, 'v1')` and by the v1 generator "
        "commit `5f47de4`. Scores on this synthetic bank are not evidence of generalisation. The overlap audits were "
        "not run by the author.",
        "",
        "## v2 rows per family",
        "",
        "| Family | Rows |",
        "|---|---:|",
        *[f"| {k} | {v} |" for k, v in sorted(fam.items())],
        "",
        f"Positive {pol['positive']}, matched negative {pol['negative']} ({pol['negative'] / n:.1%}). "
        f"{len(lay)} layouts; the largest has {max(lay.values())} rows ({max(lay.values()) / n:.1%}). "
        f"Case mix: normal {case['normal']} ({case['normal'] / n:.1%}), upper {case['upper']}, lower {case['lower']}. "
        f"Cohorts: {dict(manifest['cohort_counts'])}.",
        "",
        "## Rows per new case",
        "",
        f"{new_rows} rows carry at least one new-case tag.",
        "",
        "| New case (phenomenon tag) | Rows | normal | upper | lower |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {t} | {tags[t]} | {tag_case[(t, 'normal')]} | {tag_case[(t, 'upper')]} | {tag_case[(t, 'lower')]} |"
            for t in NEW_CASE_TAGS
        ],
        "",
        "Tag meanings: `county_name` LOCATION including `County` (with and without `, ST`; real train-pool counties "
        "and invented `<stem> County`); `county_word_generic` negative with `county` as an ordinary word; "
        "`named_site_capitalised` FACILITY on the whole of a capitalised site name in normal case (`our [X Clinic]`); "
        "`named_site_case_hidden` a named site in all-caps or lowercase rows (FACILITY by authored role, written "
        "without a determiner); `p7_care_team` LOCATION on the city in `the <city> hospice team`; `p7_case_hidden` "
        "P7 rows in all-caps or lowercase; `the_in_name_case_hidden` D3 `The`-inside names in all-caps or "
        "lowercase; `the_service_org_in_name` `The` inside a service organisation's name; `the_service_org_article` "
        "`The` as an article before a service organisation (outside); `named_pharmacy` and `named_pharmacy_case_hidden` "
        "named pharmacies (the latter in lowercase or all-caps); `employer_and_care_roles` the same organisation as an "
        "unannotated employer then FACILITY in a care role; `employer_name` an unannotated employer; "
        "`invented_place_decoy` decoys built on invented place-words (`<stem> criteria`, `<stem> virus`); "
        "`ordinary_place_word_case_hidden` ordinary place-words in all-caps or lowercase, unannotated.",
        "",
        "## v2 label support",
        "",
        "| Label | Spans |",
        "|---|---:|",
        *[f"| {k} | {v} |" for k, v in manifest["label_support"].items()],
        "",
        "## v2 pool check",
        "",
        f"Real value uses: {sum(pools.values())}; pools: train {pools['train']}, heldout {pools['heldout']}. "
        + ", ".join(f"{k} {v}" for k, v in sorted(sources.items()))
        + f". Train-pool counties available: {world.county_count}.",
        "",
        f"Repetition limit {MAX_VALUE_SHARE:.1%} of rows ({MAX_VALUE_SHARE * n:.1f}). Largest FACILITY or LOCATION "
        f"value: {top_span[0][1]} rows; largest real value: {top_real[0][1]} rows "
        f"(`{top_real[0][0]}`); largest real instrument or disease decoy: `{worst_decoy[0]}` in {worst_decoy[1]} rows "
        f"({worst_decoy[1] / n:.2%}).",
        "",
        "## v2 design notes",
        "",
        "- No row is forced to normal case. Case is assigned before authoring so the wording can respect the role.",
        "- In all-caps and lowercase rows a named site is written without a determiner (`seen at <name> clinic`) "
        "and the generic P7 form keeps `our` or `the`; in normal case both forms occur, with the capital site word "
        "marking the name (`our [Name Clinic]` is FACILITY only in normal case).",
        "- Decoys on real instrument and disease names are capped by the repetition limit; invented place-word "
        "decoys have the same shape.",
        "- Invented county, town, facility and site names exclude every Gazetteer place and CMS city.",
        "",
        "## v2 remaining unsettled cases",
        "",
        "1. **Capitalised site word with a determiner in all-caps or lowercase rows** (`OUR OMBREY CLINIC`): the "
        "surface is identical for the P7 and named-site roles, so the generator uses the generic role only. "
        "A named site in those cases is written without `our` or `the`. Left out rather than decided.",
        "2. **County with a generic word after it** (`the Kestrine County office`): not authored.",
        "3. **Invented `<stem> County` placed in a real state** is authored, which can resemble a real county; "
        "stems exclude Gazetteer names but not county names from other states.",
        "4. **A county written without the word `County`** (`lives in Kestrine`): not authored.",
        "5. **`The` before a service organisation whose name begins with `The` in lowercase text** "
        "(`enrolled with the hearth at ...`): authored with the role fixed by the author, as the policy says, "
        "but the surface is ambiguous between an article and part of the name.",
        "6. **Employer role when the employer is also the only care site named in a short note**: authored only with "
        "an explicit employer cue (`works as`, `employed by`, `husband works at`, `son is`, `retired from`).",
        "",
    ]
    report_path = REPORT_DIR / "AUTHOR_REPORT.md"
    existing = report_path.read_text(encoding="utf-8")
    if V2_MARKER in existing:
        existing = existing[: existing.index(V2_MARKER)].rstrip() + "\n\n"
    else:
        existing = existing.rstrip() + "\n\n"
    report_path.write_text(existing + "\n".join(lines), encoding="utf-8")

    sampler = random.Random(13)
    new = [r for r in rows if any(t in NEW_CASE_TAGS for t in r["phenomena"])]
    old = [r for r in rows if not any(t in NEW_CASE_TAGS for t in r["phenomena"])]
    chosen = sampler.sample(new, 55) + sampler.sample(old, 45)
    sampler.shuffle(chosen)
    parts = [
        "# Spot-check sample v2: tk13 care sites and localities",
        "",
        "100 rows drawn with `random.Random(13)`: 55 exercise a v2 new case, 45 do not. Spans are `[LABEL|text]`. "
        "Check against the policy and the follow-up decisions. Synthetic text; real names are Census and CMS train-pool values.",
        "",
    ]
    for row in chosen:
        new_tags = [t for t in row["phenomena"] if t in NEW_CASE_TAGS]
        parts += [
            f"### {row['id']}  ({row['family']}, {row['layout']}, {row['case_mode']}, {row['polarity']}"
            + (f", {row['negative_type']}" if row["negative_type"] else "")
            + ")"
            + (f"  new: {', '.join(new_tags)}" if new_tags else ""),
            "",
            "```",
            _mark(row),
            "```",
            "",
        ]
    (REPORT_DIR / "spot_check_sample_v2.md").write_text(
        "\n".join(parts), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports",
        action="store_true",
        help="also write the author report and spot-check sample",
    )
    parser.add_argument("--version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--force-v1", action="store_true", help="allow rewriting the immutable v1 bank"
    )
    args = parser.parse_args()
    if args.version == "v1" and not args.force_v1:
        raise SystemExit(
            "v1 is immutable; reproduce it with generate(SEED, 'v1') or pass --force-v1"
        )
    seed = (
        args.seed
        if args.seed is not None
        else (SEED_V2 if args.version == "v2" else SEED)
    )
    rows, world = generate(seed, args.version)
    check_bank(rows)
    manifest = write_bank(rows, world, seed, args.version)
    if args.reports:
        if args.version == "v2":
            write_reports_v2(rows, manifest, world)
        else:
            write_reports(rows, manifest, world)
    print(json.dumps({"rows": len(rows), "sha256": manifest["sha256"]}))


if __name__ == "__main__":
    main()
