"""Author the tk21 training bank: US care-site names and localities, boundaries.

Commissioned by docs/care-site-boundaries-bank-brief.md. Annotation contract:
docs/annotation-policy-v2.md with docs/annotation-policy-v2-places.md
(clinical-phi-v2, addendum clinical-phi-v2-places), including the all-ages
decisions of 2026-10-06 and 2026-10-07.

Real place and care-site names come only from the pinned public sources in
.cache/place_sources (Census Gazetteer places and counties 2024, CMS Hospital
General Information), only where place_value_pool returns "train", and never
when the blind-bank exclusion filter matches. Every other name, date, age and
identifier is invented under the reserved schemes recorded in the manifest.

Deterministic for its seed:

  PYTHONPATH=src python scripts/takeover/author21_care_site_boundaries.py [--seed N] [--no-write]
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from clinical_deid.data.annotation_policy import (  # noqa: E402
    POLICY_VERSION, annotation_errors, canonical_identity_group, identity_split,
)
from clinical_deid.data.annotation_policy_places import ADDENDUM, place_value_pool  # noqa: E402
from clinical_deid.data.generators import place_sources  # noqa: E402
from clinical_deid.data.generators.places import (  # noqa: E402
    EXCLUSIONS_SHA256, excluded_sequences, excluded_value,
)
from clinical_deid.types import ENTITY_TYPES  # noqa: E402

VERSION = "1.0"
SEED = 21007
GENERATOR = "scripts/takeover/author21_care_site_boundaries.py"
BRIEF = "docs/care-site-boundaries-bank-brief.md"
BANK = ROOT / "data" / "tk21_train_care_site_boundaries.jsonl"
REPORT_DIR = ROOT / "reports" / "takeover" / "author21_care_site_boundaries"
SUITE = "training"
FAMILY_ROWS = {
    "tk21_long_connector_names": 350,
    "tk21_care_type_tails": 400,
    "tk21_names_as_addresses": 300,
    "tk21_real_hospital_abbreviations": 300,
    "tk21_city_with_state": 400,
    "tk21_facility_and_locality": 250,
}
CASES = (("normal", 6), ("upper", 2), ("lower", 2))
REPEAT_FRACTION = 0.005
SCHEMES = {
    "NAME": "invented: given names from a fixed authored list; surnames from invented syllable pairs "
            "(rejected when they equal a Gazetteer place or county name) or a short list of common surnames; "
            "every identity in the train identity partition",
    "MRN": "M21- prefix and seven invented digits, or eight invented digits beginning 21",
    "ACCOUNT": "A21 prefix and eight invented digits",
    "ID": "PA21- (authorisation), MBR21 (member), AC21- (accession), ENC21 (encounter) prefixes with invented digits",
    "PHONE": "US 555-0100..0199 fictional lines with invented area codes",
    "EMAIL": "invented local parts at .example / .test / example.com, example.net, example.org",
    "DATE": "invented dates 2019-2026 for encounters; invented dates of birth consistent with the age",
    "AGE": "invented ages 19-103 (about 15% over 89); decades as words; every age annotated",
    "FACILITY": "invented care-site names from invented syllable stems (rejected when they equal a Gazetteer "
                "place, county or CMS name token), plus CMS Hospital General Information names (train pool)",
    "LOCATION": "Census Gazetteer 2024 places (train pool) and invented localities",
}

# --------------------------------------------------------------------------------------------
# Markup


MARK = re.compile(r"\[\[([A-Z]+)\|([^\[\]]+)\]\]")


def M(label: str, value: str) -> str:
    if label not in ENTITY_TYPES or "[" in value or "]" in value or not value.strip() or value != value.strip():
        raise ValueError(f"bad span value {label}:{value!r}")
    return f"[[{label}|{value}]]"


def parse(marked: str) -> tuple[str, list[dict]]:
    pieces, spans, offset, previous = [], [], 0, 0
    for match in MARK.finditer(marked):
        literal = marked[previous:match.start()]
        pieces.append(literal)
        offset += len(literal)
        spans.append({"start": offset, "end": offset + len(match[2]), "label": match[1]})
        pieces.append(match[2])
        offset += len(match[2])
        previous = match.end()
    pieces.append(marked[previous:])
    text = "".join(pieces)
    if "[[" in text or "]]" in text:
        raise ValueError("residual markup")
    return text, spans


# --------------------------------------------------------------------------------------------
# Convention renderers. Each returns marked text; the tests rebuild policy fixtures with them.


def site_in_locality(site: str, locality: str) -> str:
    """P5: a locality introduced by `in` is a separate LOCATION; the preposition stays outside."""
    return f"{M('FACILITY', site)} in {M('LOCATION', locality)}"


def site_comma_locality(site: str, locality: str) -> str:
    """P5: a locality after a comma is a separate LOCATION."""
    return f"{M('FACILITY', site)}, {M('LOCATION', locality)}"


def site_at_continuation(site: str, locality: str) -> str:
    """P5a: `at <name>` without a state or postal qualifier continues the FACILITY."""
    return M("FACILITY", f"{site} at {locality}")


def site_at_qualified(site: str, qualified_locality: str) -> str:
    """P5a: an `at` locality with a state qualifier is a separate LOCATION."""
    return f"{M('FACILITY', site)} at {M('LOCATION', qualified_locality)}"


def unit_at_parent(unit: str, parent: str) -> str:
    """v2 FACILITY rule: a named unit at its parent facility is two FACILITY spans."""
    return f"{M('FACILITY', unit)} at {M('FACILITY', parent)}"


def site_with_department(site: str, department: str) -> str:
    """Census: a department word next to an institution name stays outside the span."""
    return f"{M('FACILITY', site)} {department}"


def locality_with_state(city: str, state: str, separator: str = ", ") -> str:
    """Census: city with its state is one LOCATION span including the state."""
    return M("LOCATION", f"{city}{separator}{state}")


def article_site(article: str, site: str) -> str:
    """Census: a leading `the` used as an article stays outside the FACILITY span."""
    return f"{article} {M('FACILITY', site)}"


ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen").split()
TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}
DECADES = {2: "twenties", 3: "thirties", 4: "forties", 5: "fifties", 6: "sixties", 7: "seventies", 8: "eighties",
           9: "nineties"}


def number_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + (f"-{ONES[n % 10]}" if n % 10 else "")
    if n < 200:
        return "one hundred" + (f" {number_words(n - 100)}" if n > 100 else "")
    raise ValueError(n)


def an(n: int) -> str:
    return "an" if number_words(n)[0] in "aeiou" else "a"


def age_value(n: int, words: bool = False) -> str:
    """AGE covers the value only; units, hyphens, sex markers and suffixes stay outside."""
    return M("AGE", number_words(n) if words else str(n))


# --------------------------------------------------------------------------------------------
# Real-value pools


STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}
# Place inside a named instrument (N1) or disease (N2): (place word, sentence). Kept only when train and not excluded.
NAMED_INSTRUMENTS = (
    ("Chicago", "Manometry was interpreted per the Chicago Classification."),
    ("Boston", "Meets the Boston criteria for probable amyloid angiopathy."),
    ("Seattle", "Seattle Heart Failure Model estimate reviewed with cardiology."),
    ("Ottawa", "Ottawa ankle rules negative; no imaging needed."),
    ("Ottawa", "Ottawa knee rules negative."),
    ("San Francisco", "San Francisco Syncope Rule negative."),
    ("Framingham", "Framingham Risk Score 12%."),
    ("Pittsburgh", "Pittsburgh Sleep Quality Index score 11."),
    ("Kansas City", "Kansas City Cardiomyopathy Questionnaire completed."),
    ("Richmond", "Richmond Agitation-Sedation Scale score 0 overnight."),
    ("Oakland", "Oakland score 8 for the lower GI bleed."),
    ("Glasgow", "Glasgow Coma Scale 15 on arrival."),
    ("Boston", "Boston Bowel Preparation Scale 8."),
    ("Seattle", "Seattle Angina Questionnaire completed."),
    ("New York", "New York Heart Association class II symptoms."),
    ("Montreal", "Montreal Cognitive Assessment 24/30."),
    ("Pittsburgh", "Pittsburgh knee rules negative."),
    ("Denver", "Denver shunt considered for refractory ascites."),
)
NAMED_DISEASES = (
    ("Lyme", "Lyme disease serology negative."),
    ("West Nile", "West Nile virus IgM pending."),
    ("Rocky Mountain", "Rocky Mountain spotted fever treated with doxycycline."),
    ("Norwalk", "Norwalk virus suspected; contact precautions."),
    ("Coxsackie", "Coxsackie virus suspected for the rash."),
    ("Pontiac", "Pontiac fever considered and tested."),
    ("St. Louis", "St. Louis encephalitis panel negative."),
    ("La Crosse", "La Crosse encephalitis serology sent."),
    ("Powassan", "Powassan virus testing sent after the tick bite."),
    ("Jamestown Canyon", "Jamestown Canyon virus IgM negative."),
    ("Bourbon", "Bourbon virus considered after the tick bite."),
    ("Heartland", "Heartland virus testing sent per infectious disease."),
    ("Colorado", "Colorado tick fever considered."),
)
ACRONYMS = {"VA", "UPMC", "UCSF", "UCLA", "UC", "IU", "OSF", "CHI", "NYU", "LDS", "UNC", "UAB", "MUSC", "SSM",
            "HCA", "SMP", "UCI", "UCSD", "USC", "HSHS", "AHN", "UH", "UNM", "UVA", "OU", "KU", "UT", "UF", "LSU",
            "MD", "II", "III", "JFK", "AMG", "CHRISTUS"}
SMALL_WORDS = {"OF", "THE", "AND", "AT", "FOR", "IN", "ON"}
SAINTS = {"ST": "St.", "ST.": "St.", "MT": "Mt.", "MT.": "Mt.", "FT": "Ft.", "FT.": "Ft.", "STE": "Ste.", "STE.": "Ste."}
GENERIC_WORDS = {"university", "hospital", "hospitals", "medical", "center", "centre", "regional", "memorial",
                 "community", "general", "health", "healthcare", "system", "clinic", "county", "district", "the",
                 "of", "and", "saint", "st.", "mercy", "baptist", "methodist", "presbyterian", "lutheran",
                 "childrens", "children's", "women's", "va", "city", "valley", "behavioral", "rehabilitation",
                 "specialty", "surgical", "heart", "institute", "&", "care", "critical", "access", "area", "mental"}
ABBREVIATIONS = (  # (pattern on the title-cased full name, short form, label)
    ("Regional Medical Center", "Reg Med Ctr", "reg_med_ctr"),
    ("University Hospital", "Univ Hosp", "univ_hosp"),
    ("Memorial Hospital", "Mem Hosp", "mem_hosp"),
    ("Medical Center", "Med Ctr", "med_ctr"),
    ("Hospital", "Hosp", "hosp"),
)


def _case_token(token: str, first: bool) -> str:
    if token in SAINTS:
        return SAINTS[token]
    if token in SMALL_WORDS and not first:
        return token.lower()
    if token == "&":
        return token
    if token in ACRONYMS or (len(token) >= 2 and not re.search(r"[AEIOUY]", token) and token.isalpha()):
        return token
    if token.startswith("MC") and len(token) > 3 and token.isalpha():
        return "Mc" + token[2:].capitalize()
    if token.startswith("O'") and len(token) > 2:
        return "O'" + token[2:].capitalize()
    return token[:1] + token[1:].lower()


def title_case_cms(raw: str) -> str | None:
    """Title-case a CMS facility name, or None when the form cannot be cased reliably."""
    raw = re.sub(r"\s+", " ", raw.strip())
    if not raw or re.search(r"[^A-Z0-9 &.'\-]", raw) or re.search(r"\d", raw) or " - " in raw or raw.startswith("THE "):
        return None
    if " -" in raw or "- " in raw or raw.endswith("-"):
        return None
    words = []
    for index, token in enumerate(raw.split(" ")):
        parts = token.split("-")
        words.append("-".join(_case_token(part, index == 0 and n == 0) for n, part in enumerate(parts)))
    return " ".join(words)


@dataclass(frozen=True)
class City:
    name: str
    state: str

    @property
    def state_name(self) -> str:
        return STATES[self.state]


@dataclass(frozen=True)
class Hospital:
    name: str
    raw: str
    city: City | None
    short_forms: tuple  # ((label, short), ...)


class Pools:
    """Every real value used in the bank, filtered to the train pool and the exclusion filter."""

    def __init__(self):
        self.counts = {}
        places = place_sources.gazetteer_places()
        counties = place_sources.gazetteer_counties()
        cms = place_sources.cms_hospitals()
        self.gazetteer_names = {}
        for item in places:
            self.gazetteer_names.setdefault(item["name"].casefold(), item["name"])
        for item in counties:
            self.gazetteer_names.setdefault(item["name"].casefold(), item["name"])
            stem = re.sub(r" (?:County|Parish)$", "", item["name"])
            self.gazetteer_names.setdefault(stem.casefold(), stem)
        for name in STATES.values():
            self.gazetteer_names.setdefault(name.casefold(), name)

        # States: a state shares one pool with its abbreviation (the expansion's pool).
        state_counts = Counter()
        self.states = []
        for abbr, name in sorted(STATES.items()):
            pool = place_value_pool(abbr, expansion=name)
            state_counts[pool] += 1
            if pool != "train":
                continue
            if excluded_value(name) or excluded_value(abbr):
                state_counts["excluded"] += 1
                continue
            self.states.append(abbr)
        self.counts["states"] = {"candidates": len(STATES), **dict(state_counts), "usable": len(self.states)}
        train_states = set(self.states)

        # Gazetteer places.
        city_counts = Counter()
        seen, cities = set(), []
        state_name_words = {name.casefold() for name in STATES.values()}
        for item in places:
            name, state = item["name"], item["state"]
            if state not in STATES:
                continue
            key = (name.casefold(), state)
            if key in seen:
                continue
            seen.add(key)
            city_counts["candidates"] += 1
            if not re.fullmatch(r"[A-Z][A-Za-z.'\-]*(?: [A-Z][A-Za-z.'\-]*){0,2}", name) or len(name) < 4:
                city_counts["unsuitable_form"] += 1
                continue
            if any(word in name.casefold() for word in state_name_words) or name.casefold() in state_name_words:
                city_counts["contains_state_name"] += 1
                continue
            pool = place_value_pool(name)
            city_counts[pool] += 1
            if pool != "train":
                continue
            if state not in train_states:
                city_counts["state_not_train"] += 1
                continue
            if excluded_value(name) or excluded_value(f"{name}, {state}"):
                city_counts["excluded"] += 1
                continue
            cities.append(City(name, state))
        cities.sort(key=lambda c: (c.name, c.state))
        self.cities = cities
        city_counts["usable"] = len(cities)
        self.counts["gazetteer_places"] = dict(city_counts)
        self.city_lookup = {(c.name.casefold(), c.state): c for c in cities}

        # CMS hospitals.
        cms_counts = Counter()
        self.cms_tokens = set()
        hospitals, seen_names = [], set()
        for row in sorted(cms, key=lambda r: (r["Facility Name"], r["City/Town"], r["State"], r["Facility ID"])):
            raw = row["Facility Name"].strip()
            for token in re.split(r"[\s\-/]+", raw):
                if token:
                    self.cms_tokens.add(token.casefold())
            cms_counts["candidates"] += 1
            name = title_case_cms(raw)
            if name is None:
                cms_counts["unsuitable_form"] += 1
                continue
            if name.casefold() in seen_names:
                cms_counts["duplicate_name"] += 1
                continue
            seen_names.add(name.casefold())
            words = [w.casefold() for w in name.split()]
            if all(w in GENERIC_WORDS or len(w) <= 2 for w in words) or len(words) < 2:
                cms_counts["no_distinctive_word"] += 1
                continue
            pool = place_value_pool(name)
            if place_value_pool(raw) != pool:
                pool = "heldout"
            cms_counts[pool] += 1
            if pool != "train":
                continue
            if self.inner_heldout(name):
                cms_counts["inner_place_heldout"] += 1
                continue
            if excluded_value(name) or excluded_value(raw):
                cms_counts["excluded"] += 1
                continue
            short_forms = []
            for pattern, short, label in ABBREVIATIONS:
                if len(re.findall(rf"\b{re.escape(pattern)}\b", name)) == 1:
                    abbreviated = re.sub(rf"\b{re.escape(pattern)}\b", short, name)
                    if label == "hosp" and re.search(r"\b(?:Hospitals|Medical Center)\b", name):
                        continue
                    if place_value_pool(abbreviated, expansion=name) == "train" and not excluded_value(abbreviated):
                        short_forms.append((label, abbreviated))
                    break
            city_title = " ".join(_case_token(t, i == 0) for i, t in enumerate(row["City/Town"].split()))
            city = self.city_lookup.get((city_title.casefold(), row["State"]))
            hospitals.append(Hospital(name, raw, city, tuple(short_forms)))
        self.hospitals = hospitals
        cms_counts["usable"] = len(hospitals)
        cms_counts["usable_with_short_form"] = sum(1 for h in hospitals if h.short_forms)
        cms_counts["usable_with_train_city"] = sum(1 for h in hospitals if h.city)
        self.counts["cms_hospitals"] = dict(cms_counts)
        self.cms_names = {h.name.casefold() for h in hospitals} | {row["Facility Name"].casefold() for row in cms}

        # Instruments and diseases (N1, N2).
        named_counts = Counter()
        self.named_negatives = []
        for kind, items in (("n1_named_instrument", NAMED_INSTRUMENTS), ("n2_place_disease", NAMED_DISEASES)):
            for place, sentence in items:
                named_counts["candidates"] += 1
                pool = place_value_pool(place)
                named_counts[pool] += 1
                if pool != "train":
                    continue
                if excluded_value(place) or excluded_sequences(sentence):
                    named_counts["excluded"] += 1
                    continue
                self.named_negatives.append((kind, place, sentence))
        named_counts["usable"] = len(self.named_negatives)
        self.counts["named_instruments_and_diseases"] = dict(named_counts)

    def inner_heldout(self, value: str) -> bool:
        """True when any 1-3 word sequence inside ``value`` is a Gazetteer or state name in the heldout pool."""
        words = [w.strip(",.") for w in value.split()]
        for start in range(len(words)):
            for size in range(1, min(3, len(words) - start) + 1):
                sequence = " ".join(words[start:start + size])
                if sequence.casefold() in self.gazetteer_names and place_value_pool(sequence) != "train":
                    return True
        return False


# --------------------------------------------------------------------------------------------
# Invented vocabulary


PLACE_PREFIXES = ("Mar", "Fen", "Quen", "Tin", "Brann", "Od", "Vell", "Corr", "Gors", "Lark", "Moss", "Brack",
                  "Oll", "Pell", "Yarr", "Voss", "Tol", "Kest", "Omb", "Marl", "Quill", "Ferr", "Dun", "Wex",
                  "Thorn", "Cal", "Ros", "Wyn", "Hav", "Sar", "Tav", "Gar", "Pem", "Riv", "Sel", "Var", "Zel",
                  "Hob", "Kel", "Lud", "Mab", "Nev", "Orr", "Pry", "Quar", "Rud", "Sib", "Tre", "Vey", "Wim",
                  "Yel", "Brin", "Cor", "Dell", "Ess", "Fal", "Grim", "Hesk", "Ivel", "Jor", "Kin", "Lom", "Mird",
                  "Norr", "Ost", "Penn", "Rask", "Stav", "Tull", "Ulv", "Wend", "Ashm", "Bel", "Cran", "Drum")
PLACE_SUFFIXES = ("dale", "by", "ick", "ley", "more", "ton", "wick", "wood", "mere", "ford", "bury", "stead",
                  "combe", "leigh", "holt", "wyn", "ett", "ard", "en", "ham", "worth", "field", "crest", "brook",
                  "haven", "ow", "ing", "is", "ane", "ock", "ville", "hurst", "gate", "shaw")
GEO_WORDS = ("Ridge", "Mill", "Valley", "Hollow", "Crossing", "Pointe", "Grove", "Brook", "Meadows", "Heights",
             "Harbor", "Springs", "Glen", "Hill", "Creek", "Landing", "Bend", "Falls", "Prairie", "Bluff")
SURNAME_PREFIXES = ("Bran", "Hol", "Quist", "Mer", "Dal", "Tess", "Wen", "Kor", "Lav", "Pim", "Rask", "Sorr",
                    "Tull", "Vant", "Whit", "Yarb", "Zell", "Abern", "Coll", "Drey", "Ferr", "Gant", "Hask", "Ibb",
                    "Jess", "Kemm", "Lom", "Mott", "Nesb", "Ostr", "Pell", "Quin", "Rook", "Sted", "Trev", "Ulm",
                    "Vash", "Wick", "Bly", "Carr", "Dov", "Emb", "Fisk", "Gell", "Harr", "Imm", "Kav", "Lind")
SURNAME_SUFFIXES = ("ock", "ier", "man", "ley", "ett", "son", "ard", "ing", "ow", "er", "ane", "ick", "is", "el",
                    "ova", "ez", "ski", "berg", "stein", "ani", "ura", "ado", "ins", "ritt")
COMMON_SURNAMES = ("Nguyen", "Patel", "Garcia", "Okafor", "Kowalski", "Lindqvist", "Haddad", "Moreno", "Tanaka",
                   "Brennan", "Osei", "Ivanova", "Schultz", "Delgado", "Fitzgerald", "Abara", "Morales", "Kim",
                   "Novak", "Reyes", "Mensah", "Petrov", "Castillo", "Yamamoto", "Herrera", "Dubois", "Kaur",
                   "Achebe", "Sorensen", "Quispe", "Romano", "Baptiste", "Cohen", "Walsh", "Iyer", "Begay")
GIVEN_F = ("Ada", "Alma", "Anita", "Beatrice", "Bernadette", "Carla", "Celia", "Dolores", "Edith", "Elena",
           "Esther", "Fay", "Gloria", "Greta", "Hazel", "Ida", "Imelda", "Irene", "Joan", "Judith", "June", "Karen",
           "Lena", "Lillian", "Lorraine", "Mabel", "Marisol", "Maureen", "Nadia", "Nora", "Olive", "Opal", "Paula",
           "Priya", "Rosalind", "Ruth", "Selma", "Thea", "Ursula", "Vera", "Wanda", "Yolanda", "Zora", "Agnes",
           "Beryl", "Coral", "Dina", "Eunice", "Frieda", "Hilda", "Inez", "Kay", "Leona", "Mina", "Noreen",
           "Odette", "Petra", "Renata", "Sylvia", "Tamsin", "Wilhelmina", "Ngozi", "Mei", "Farah", "Lucia")
GIVEN_M = ("Abel", "Alvin", "Amos", "Arturo", "Bernard", "Boris", "Calvin", "Cyril", "Dale", "Desmond", "Edgar",
           "Elias", "Emil", "Felix", "Gordon", "Hector", "Ian", "Ivan", "Jasper", "Jerome", "Karl", "Lionel",
           "Malcolm", "Marcus", "Milo", "Oscar", "Otto", "Percy", "Quentin", "Ralph", "Rafael", "Rudy", "Silas",
           "Stanley", "Tobias", "Victor", "Walter", "Wendell", "Yusuf", "Anders", "Clement", "Dmitri", "Ernesto",
           "Gideon", "Horace", "Ignatius", "Jonas", "Leopold", "Mortimer", "Nestor", "Orrin", "Reuben",
           "Thaddeus", "Wilbur", "Kwame", "Hiroshi", "Tariq", "Mateo", "Bao")
SAINT_NAMES = ("Odile", "Brendan", "Agnes", "Ambrose", "Benedict", "Bridget", "Casimir", "Cecilia", "Dymphna",
               "Eligius", "Fiacre", "Gertrude", "Hedwig", "Isidore", "Jude", "Kilian", "Leocadia", "Malachy",
               "Odo", "Pancras", "Quiteria", "Remigius", "Scholastica", "Tarcisius", "Ulrich", "Wenceslas",
               "Walburga", "Zita", "Anselm", "Columba", "Finnian", "Hildegard", "Ignatia", "Lioba", "Petroc")


class Vocabulary:
    def __init__(self, pools: Pools):
        counts = Counter()
        reject = set(pools.gazetteer_names) | pools.cms_tokens

        def ok(word: str) -> bool:
            if word.casefold() in reject:
                counts["collides_with_real_name"] += 1
                return False
            if excluded_value(word):
                counts["excluded"] += 1
                return False
            return True

        words = sorted({p + s for p in PLACE_PREFIXES for s in PLACE_SUFFIXES})
        counts["place_word_candidates"] = len(words)
        self.place_words = [w for w in words if ok(w)]
        surnames = sorted({p + s for p in SURNAME_PREFIXES for s in SURNAME_SUFFIXES})
        counts["surname_candidates"] = len(surnames)
        self.surnames = [w for w in surnames if ok(w)]
        self.reject = reject
        self.counts = dict(counts)
        self.counts["place_words_usable"] = len(self.place_words)
        self.counts["surnames_usable"] = len(self.surnames)


@dataclass(frozen=True)
class Person:
    given: str
    surname: str
    sex: str

    @property
    def full(self) -> str:
        return f"{self.given} {self.surname}"

    @property
    def group(self) -> str:
        return canonical_identity_group(self.full)


class People:
    def __init__(self, rng: random.Random, vocab: Vocabulary):
        self.rng, self.vocab, self.used = rng, vocab, set()

    def new(self, sex: str) -> Person:
        while True:
            given = self.rng.choice(GIVEN_F if sex == "F" else GIVEN_M)
            surname = self.rng.choice(self.vocab.surnames) if self.rng.random() < 0.6 else self.rng.choice(COMMON_SURNAMES)
            person = Person(given, surname, sex)
            if person.full in self.used or identity_split(person.group) != "train":
                continue
            self.used.add(person.full)
            return person


class Inventor:
    """Invented care-site stems and localities."""

    def __init__(self, rng: random.Random, vocab: Vocabulary, pools: Pools):
        self.rng, self.vocab, self.pools = rng, vocab, pools

    def word(self) -> str:
        return self.rng.choice(self.vocab.place_words)

    def stem(self, two_word: float = 0.6) -> str:
        while True:
            value = f"{self.word()} {self.rng.choice(GEO_WORDS)}" if self.rng.random() < two_word else self.word()
            if value.casefold() not in self.vocab.reject:
                return value

    def site(self, name: str) -> str | None:
        if name.casefold() in self.pools.cms_names or excluded_value(name):
            return None
        return name


# --------------------------------------------------------------------------------------------
# Row context: invented people, dates and identifiers shared by a matched pair


DATE_STYLES = ("us", "us_short", "iso", "long", "abbr", "abbr_us")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
REL_ROLES = (("daughter", "F"), ("son", "M"), ("wife", "F"), ("husband", "M"), ("sister", "F"), ("brother", "M"),
             ("niece", "F"), ("nephew", "M"))
DOMAINS = ("mail.example", "example.com", "example.net", "example.org", "post.test", "inbox.example")
REASONS = ("higher level of care", "cardiology evaluation", "inpatient rehabilitation", "neurosurgical review",
           "dialysis access", "skilled nursing placement", "memory care placement", "orthopedic follow-up",
           "wound management", "stroke workup", "interventional radiology", "hip fracture repair",
           "behavioral health admission", "chemotherapy continuation", "ventilator weaning")
CLINICAL = (
    "Vitals stable overnight.", "No acute distress.", "Tolerating a regular diet.", "Pain controlled on oral regimen.",
    "Ambulating with a walker.", "Potassium 3.9, creatinine 1.1.", "Afebrile for 48 hours.",
    "Wound clean, dry and intact.", "Weaned to room air.", "Blood glucose 142-188 on basal insulin.",
    "INR 2.4 on warfarin.", "Plan reviewed with the patient and family.", "Telemetry without events.",
    "Hemoglobin stable at 10.2.", "Lisinopril held for soft pressures.", "Started on cefazolin.",
    "PT recommends supervision for stairs.", "Oriented to person and place.", "Foley removed; voiding.",
    "Diuresed 2 liters.", "Tube feeds at goal.", "Sacral skin intact.", "Fall precautions in place.",
    "Code status: full code.", "Allergies: penicillin (rash).", "Insulin sliding scale continued.",
    "Chest x-ray clear.", "Bowel regimen in place.", "Echo with EF 45%.", "Cultures no growth to date.",
)
DEPARTMENTS = ("Nurse Pool", "Scheduling", "Utilization Review", "Medical Records", "Health Information Management",
               "Case Management", "Patient Access", "Admissions", "Billing Office", "Central Intake",
               "Transfer Center", "Bed Management", "Infection Prevention", "Discharge Planning",
               "Referral Coordination", "Prior Authorization Team", "Lab Client Services", "Radiology Scheduling",
               "Social Work", "Nursing Administration", "Release of Information", "Patient Financial Services")


def fmt_date(d: date, style: str) -> str:
    if style == "us":
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    if style == "us_short":
        return f"{d.month}/{d.day}/{d.year % 100:02d}"
    if style == "iso":
        return d.isoformat()
    if style == "long":
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    if style == "abbr":
        return f"{d.day} {MONTHS[d.month - 1][:3]} {d.year}"
    return f"{MONTHS[d.month - 1][:3]} {d.day}, {d.year}"


def phone(rng: random.Random) -> str:
    area = rng.choice([a for a in range(201, 990) if a % 100 != 11 and str(a)[1] != "9"])
    line = f"555-01{rng.randint(0, 99):02d}"
    form = rng.randrange(4)
    if form == 0:
        return f"({area}) {line}"
    if form == 1:
        return f"{area}-{line}"
    if form == 2:
        return f"{area}.{line.replace('-', '.')}"
    return f"+1 {area}-{line}"


def digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randrange(10)) for _ in range(n))


class Ctx:
    def __init__(self, rng: random.Random, people: People):
        self.sex = rng.choice("FM")
        self.pron, self.poss, self.noun = ("she", "her", "woman") if self.sex == "F" else ("he", "his", "man")
        self.pt = people.new(self.sex)
        self.clin = people.new(rng.choice("FM"))
        self.clin2 = people.new(rng.choice("FM"))
        self.rel_role, rel_sex = rng.choice(REL_ROLES)
        self.rel = people.new(rel_sex)
        self.age = rng.randint(90, 103) if rng.random() < 0.15 else rng.randint(19, 89)
        self.date_style = rng.choice(DATE_STYLES)
        self.d1 = date(rng.randint(2019, 2026), rng.randint(1, 12), rng.randint(1, 28))
        if self.d1 > date(2026, 9, 30):
            self.d1 = date(2026, rng.randint(1, 9), self.d1.day)
        self.d2 = self.d1 + timedelta(days=rng.randint(1, 14))
        dob = date(self.d1.year - self.age, rng.randint(1, 12), rng.randint(1, 28))
        self.age = self.d1.year - dob.year - ((self.d1.month, self.d1.day) < (dob.month, dob.day))
        self.dob = dob
        offset = {"daughter": -rng.randint(22, 38), "son": -rng.randint(22, 38), "niece": -rng.randint(18, 40),
                  "nephew": -rng.randint(18, 40), "wife": rng.randint(-8, 8), "husband": rng.randint(-8, 8),
                  "sister": rng.randint(-12, 12), "brother": rng.randint(-12, 12)}[self.rel_role]
        self.rel_age = max(16, min(104, self.age + offset))
        self.year_moved = self.d1.year - rng.randint(1, 15)
        self.mrn = f"M21-{digits(rng, 7)}" if rng.random() < 0.5 else f"21{digits(rng, 6)}"
        self.acct = f"A21{digits(rng, 8)}"
        self.auth = f"PA21-{digits(rng, 6)}"
        self.member = f"MBR21{digits(rng, 7)}"
        self.accession = f"AC21-{digits(rng, 6)}"
        self.encounter = f"ENC21{digits(rng, 7)}"
        self.phone, self.fax, self.phone2 = phone(rng), phone(rng), phone(rng)
        self.domain = rng.choice(DOMAINS)
        self.reason = rng.choice(REASONS)
        self.fill = rng.sample(CLINICAL, 5)
        self.used: set = set()

    # Rendering helpers; ``r`` is the frame rng so a matched pair renders identically.
    def D(self, d: date) -> str:
        style = self.date_style
        if d == self.dob and style == "us_short":
            style = "us"  # a two-digit birth year would be ambiguous
        return M("DATE", fmt_date(d, style))

    def name(self, r: random.Random, person: Person, initial: float = 0.1) -> str:
        self.used.add(person)
        if r.random() < initial:
            return M("NAME", f"{person.given[0]}. {person.surname}")
        return M("NAME", person.full)

    def ptn(self, r):
        return self.name(r, self.pt)

    def drn(self, r, person=None):
        person = person or self.clin
        self.used.add(person)
        if r.random() < 0.4:
            return f"Dr. {M('NAME', person.surname)}"
        return f"Dr. {M('NAME', person.full)}"

    def initials(self, person):
        self.used.add(person)
        return M("NAME", person.given[0] + person.surname[0])

    def email(self, r, person):
        self.used.add(person)
        local = r.choice((f"{person.given}.{person.surname}", f"{person.given[0]}{person.surname}",
                          f"{person.surname}{r.randint(1, 99)}")).lower()
        return M("EMAIL", f"{local}@{self.domain}")

    def sexl(self):
        return self.sex

    def age_adj(self, r) -> str:
        n = self.age
        form = r.randrange(6)
        if form == 0:
            return f"{an(n)} {age_value(n)}-year-old {self.noun}"
        if form == 1:
            return f"{age_value(n)} y/o {self.sex}"
        if form == 2:
            return f"{age_value(n)}{self.sex}"
        if form == 3:
            return f"{age_value(n)} yo {self.noun}"
        if form == 4:
            return f"{an(n)} {age_value(n, words=True)}-year-old {self.noun}"
        return f"{an(n)} {self.noun} aged {age_value(n)}"

    def age_field(self, r) -> str:
        return r.choice((age_value(self.age), f"{age_value(self.age)} y", f"{age_value(self.age)} years"))

    def rel_line(self, r) -> str:
        if r.random() < 0.3:
            return f"{self.rel_role.capitalize()}, {self.name(r, self.rel)}, in {'her' if self.rel_role in ('daughter', 'wife', 'sister', 'niece') else 'his'} {M('AGE', DECADES[max(2, min(9, self.rel_age // 10))])}"
        return f"{self.rel_role.capitalize()} {self.name(r, self.rel)}, {age_value(self.rel_age)}"


# --------------------------------------------------------------------------------------------
# Layout frames. ``core`` is a list of marked lines in the layout's style.

LAYOUTS = {}


def layout(name, style):
    def register(fn):
        LAYOUTS[name] = (style, fn)
        return fn
    return register


def para(core, fill):
    return " ".join(list(core) + list(fill))


@layout("transfer_summary", "sentence")
def _transfer(r, c, core):
    lines = [r.choice(("TRANSFER SUMMARY", "Interfacility Transfer Summary", "Transfer Note")),
             f"Patient: {c.ptn(r)}    MRN: {M('MRN', c.mrn)}",
             f"DOB: {c.D(c.dob)}    Age: {c.age_field(r)}",
             f"Date of transfer: {c.D(c.d1)}",
             f"Reason for transfer: {c.reason}.",
             "Course: " + para(core, c.fill[:2]),
             f"Accepting physician: {c.drn(r)}"]
    return "\n".join(lines)


@layout("discharge_summary", "sentence")
def _discharge(r, c, core):
    lines = ["Discharge Summary",
             f"Name: {c.ptn(r)}  MRN {M('MRN', c.mrn)}",
             f"Admitted: {c.D(c.d1)}  Discharged: {c.D(c.d2)}",
             f"Hospital course: patient is {c.age_adj(r)} admitted for {c.reason}. "
             + para(core, c.fill[:3]),
             f"Follow-up: {c.drn(r)} in 1-2 weeks."]
    return "\n".join(lines)


@layout("referral_letter", "sentence")
def _referral(r, c, core):
    lines = [c.D(c.d1), "", f"Dear {c.drn(r, c.clin2)},", "",
             f"Thank you for seeing {c.ptn(r)} (DOB {c.D(c.dob)}), {c.age_adj(r)} with {c.reason} needs. " + para(core, c.fill[:1]),
             "", "Sincerely,", f"{c.drn(r)}", f"Phone {M('PHONE', c.phone)}"]
    return "\n".join(lines)


@layout("fax_cover", "field")
def _fax(r, c, core):
    lines = ["FAX COVER SHEET" if r.random() < 0.5 else "Fax Transmittal",
             f"Date: {c.D(c.d1)}",
             *core,
             f"Fax: {M('PHONE', c.fax)}",
             f"Re: {c.ptn(r)}, DOB {c.D(c.dob)}",
             f"Pages: {r.randint(2, 14)} including cover",
             "Confidential: intended only for the named recipient."]
    return "\n".join(lines)


@layout("secure_message", "sentence")
def _secure(r, c, core):
    lines = [f"Secure message {c.D(c.d1)} {r.randint(7, 18):02d}:{r.randint(0, 59):02d}",
             f"From: {c.drn(r)}", f"Regarding: {c.ptn(r)} (MRN {M('MRN', c.mrn)})", "",
             para(core, c.fill[:1]), "Please call back with questions."]
    return "\n".join(lines)


@layout("email_header", "field")
def _email(r, c, core):
    lines = [f"From: {c.email(r, c.clin)}", f"Sent: {c.D(c.d1)} {r.randint(7, 18)}:{r.randint(10, 59)}",
             *core,
             f"Subject: {r.choice(('records request', 'transfer packet', 'follow-up', 'care coordination'))} - {c.initials(c.pt)}",
             "", f"Attached are notes for {c.ptn(r)}, DOB {c.D(c.dob)}. {c.fill[0]}"]
    return "\n".join(lines)


@layout("care_team_list", "list")
def _care_team(r, c, core):
    lines = [f"Care team for {c.ptn(r)}:",
             f"- Attending: {c.drn(r)}",
             *core,
             f"- {c.rel_role.capitalize()}: {c.name(r, c.rel)} {M('PHONE', c.phone2)}",
             f"- Case manager: {c.name(r, c.clin2)}"]
    return "\n".join(lines)


@layout("prior_auth_letter", "sentence")
def _prior_auth(r, c, core):
    lines = ["Prior Authorization Request",
             f"Member: {c.ptn(r)}   Member ID: {M('ID', c.member)}",
             f"DOB: {c.D(c.dob)}   Auth #: {M('ID', c.auth)}",
             f"Requested service date: {c.D(c.d2)}",
             f"Clinical summary: {c.age_adj(r)} requiring {c.reason}. " + para(core, c.fill[:2]),
             f"Requesting provider: {c.drn(r)}, phone {M('PHONE', c.phone)}"]
    return "\n".join(lines)


@layout("lab_requisition", "field")
def _lab(r, c, core):
    lines = ["LAB REQUISITION", f"Patient: {c.ptn(r)}  DOB: {c.D(c.dob)}  Sex: {c.sex}",
             f"Accession: {M('ID', c.accession)}  Collected: {c.D(c.d1)}",
             *core,
             f"Ordering: {c.drn(r)}",
             f"Tests: {r.choice(('CBC, BMP', 'CMP, lipid panel', 'TSH, free T4', 'PT/INR', 'urinalysis, culture'))}"]
    return "\n".join(lines)


@layout("imaging_requisition", "field")
def _imaging(r, c, core):
    lines = ["Imaging Order",
             f"Name: {c.ptn(r)}   MRN: {M('MRN', c.mrn)}",
             f"Exam: {r.choice(('CT head without contrast', 'MRI lumbar spine', 'chest x-ray 2 views', 'renal ultrasound'))}",
             f"Requested: {c.D(c.d1)}",
             *core,
             f"Indication: {c.reason}"]
    return "\n".join(lines)


@layout("telephone_note", "sentence")
def _telephone(r, c, core):
    lines = [f"Telephone encounter {c.D(c.d1)}",
             f"Caller: {c.rel_role} {c.name(r, c.rel)} at {M('PHONE', c.phone2)}",
             f"Re: {c.ptn(r)}",
             para(core, c.fill[:1]) + " Advised to call back if worse."]
    return "\n".join(lines)


@layout("terse_query", "terse")
def _terse(r, c, core):
    return core[0]


@layout("ed_triage_note", "sentence")
def _triage(r, c, core):
    lines = [f"ED triage {c.D(c.d1)}",
             f"{c.age_adj(r)} presenting with {r.choice(('chest pain', 'shortness of breath', 'a fall', 'confusion', 'abdominal pain'))}.",
             para(core, c.fill[:1]),
             f"Triage RN: {c.name(r, c.clin2)}"]
    return "\n".join(lines)


@layout("case_management_note", "sentence")
def _case_mgmt(r, c, core):
    lines = [f"Case Management Note - {c.D(c.d1)}",
             f"Patient {c.ptn(r)}, {c.age_adj(r)}, MRN {M('MRN', c.mrn)}.",
             para(core, c.fill[:1]),
             f"Spoke with {c.rel_role} {c.name(r, c.rel)} ({M('PHONE', c.phone2)}).",
             f"CM: {c.name(r, c.clin2)}"]
    return "\n".join(lines)


@layout("social_work_note", "sentence")
def _social(r, c, core):
    lines = ["Social Work Assessment",
             f"Met with {c.ptn(r)} on {c.D(c.d1)}. {c.rel_line(r)}, is involved.",
             para(core, ()),
             "Coping appropriately; will continue to follow."]
    return "\n".join(lines)


@layout("nursing_handoff", "list")
def _handoff(r, c, core):
    lines = [f"SBAR handoff - {c.ptn(r)}, {c.age_adj(r)}",
             f"S: admitted {c.D(c.d1)} for {c.reason}",
             f"B: {c.fill[0]}",
             *core,
             f"A: {c.fill[1]}",
             f"R: {c.fill[2]}"]
    return "\n".join(lines)


@layout("consult_request", "sentence")
def _consult(r, c, core):
    lines = [f"Consult request to {r.choice(('cardiology', 'nephrology', 'geriatrics', 'palliative care', 'neurology'))}",
             f"Patient: {c.ptn(r)} ({M('MRN', c.mrn)})  Date: {c.D(c.d1)}",
             f"Question: {c.age_adj(r)} with {c.reason} needs. " + para(core, ()),
             f"Requested by {c.drn(r)}, pager {M('PHONE', c.phone)}"]
    return "\n".join(lines)


@layout("home_health_order", "field")
def _home_health(r, c, core):
    lines = ["Home Health Certification and Plan of Care",
             f"Patient: {c.ptn(r)}  DOB: {c.D(c.dob)}",
             f"Start of care: {c.D(c.d2)}",
             *core,
             f"Services: {r.choice(('skilled nursing, PT', 'PT, OT', 'skilled nursing, aide', 'wound care nursing'))}",
             f"Certifying physician: {c.drn(r)}"]
    return "\n".join(lines)


@layout("med_reconciliation", "list")
def _med_rec(r, c, core):
    lines = [f"Medication reconciliation - {c.ptn(r)} - {c.D(c.d1)}",
             "- metoprolol tartrate 25 mg twice daily",
             *core,
             "- apixaban 5 mg twice daily",
             f"- {r.choice(('furosemide 40 mg daily', 'sertraline 50 mg daily', 'atorvastatin 40 mg nightly'))}",
             f"Reviewed by {c.name(r, c.clin2)}"]
    return "\n".join(lines)


@layout("records_request", "field")
def _records(r, c, core):
    lines = ["Request for Medical Records",
             f"Date: {c.D(c.d1)}",
             *core,
             f"Patient: {c.ptn(r)}   DOB: {c.D(c.dob)}",
             f"Dates of service: {M('DATE', fmt_date(c.d1, c.date_style) + ' to ' + fmt_date(c.d2, c.date_style))}",
             f"Return by fax: {M('PHONE', c.fax)}"]
    return "\n".join(lines)


@layout("appointment_reminder", "sentence")
def _reminder(r, c, core):
    lines = [f"Hello {c.name(r, c.pt, initial=0.0)},",
             f"This is a reminder of your visit on {c.D(c.d2)} at {r.randint(8, 16)}:{r.choice(('00', '15', '30', '45'))}.",
             "Care team note: " + para(core, ()),
             f"Questions? Call {M('PHONE', c.phone)}."]
    return "\n".join(lines)


@layout("insurance_verification", "field")
def _insurance(r, c, core):
    lines = ["Insurance Verification",
             f"Subscriber: {c.ptn(r)}   DOB: {c.D(c.dob)}",
             f"Member ID: {M('ID', c.member)}   Account: {M('ACCOUNT', c.acct)}",
             *core,
             f"Verified {c.D(c.d1)} by {c.initials(c.clin2)}"]
    return "\n".join(lines)


@layout("ems_run_sheet", "field")
def _ems(r, c, core):
    lines = [f"EMS RUN REPORT  Incident date: {c.D(c.d1)}",
             f"Pt: {c.ptn(r)}  Age: {c.age_field(r)}  Sex: {c.sex}",
             *core,
             f"CC: {r.choice(('fall', 'syncope', 'dyspnea', 'altered mental status'))}",
             f"Medic: {c.initials(c.clin2)}"]
    return "\n".join(lines)


@layout("chart_header", "field")
def _chart(r, c, core):
    lines = [f"{c.ptn(r)} | MRN {M('MRN', c.mrn)} | DOB {c.D(c.dob)} | {c.age_field(r)}",
             *core,
             f"PCP: {c.drn(r)}",
             f"Encounter: {M('ID', c.encounter)}"]
    return "\n".join(lines)


@layout("portal_message", "sentence")
def _portal(r, c, core):
    lines = [f"Patient portal message from {c.ptn(r)} - {c.D(c.d1)}",
             f"Hi {c.drn(r)},", para(core, ()) + " Thank you.",
             f"Reply-to: {c.email(r, c.pt)}"]
    return "\n".join(lines)


@layout("progress_note", "sentence")
def _progress(r, c, core):
    lines = [f"Progress Note {c.D(c.d1)}",
             f"S: {c.ptn(r)} is {c.age_adj(r)}; reports feeling better.",
             f"O: {c.fill[0]} {c.fill[1]}",
             "A/P: " + para(core, c.fill[2:3]),
             f"{c.drn(r)}"]
    return "\n".join(lines)


@layout("chat_handoff", "terse")
def _chat(r, c, core):
    return f"[{r.randint(0, 23):02d}:{r.randint(0, 59):02d}] {c.initials(c.clin)}: {core[0]}"


@layout("admission_hp", "sentence")
def _hp(r, c, core):
    lines = ["History and Physical",
             f"Patient: {c.ptn(r)}   Date: {c.D(c.d1)}",
             f"HPI: {c.age_adj(r)} admitted for {c.reason}. " + para(core, c.fill[:1]),
             f"Social: {c.rel_line(r)}, lives nearby.",
             f"Admitting: {c.drn(r)}"]
    return "\n".join(lines)


@layout("operative_scheduling", "field")
def _or_sched(r, c, core):
    lines = ["Surgical Scheduling Request",
             f"Patient: {c.ptn(r)}   MRN: {M('MRN', c.mrn)}",
             f"Procedure: {r.choice(('total knee arthroplasty', 'laparoscopic cholecystectomy', 'carpal tunnel release'))}",
             f"Requested date: {c.D(c.d2)}",
             *core,
             f"Surgeon: {c.drn(r)}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Family cores. A plan is drawn once per unit; ``render(plan, role, style, c, r)`` gives marked lines.

SITE_SENTENCES = {
    "hospital": ("Transferred from {F} on {D}.", "Admitted to {F} on {D} for observation.",
                 "Outside records from {F} reviewed.", "{DR} from {F} called with results.",
                 "{P} was discharged from {F} on {D}.", "Prior admission at {F} in {D}."),
    "residence": ("{P} lives at {F}.", "Returns to {F} after discharge.", "Bed accepted at {F} for {D}.",
                  "Staff at {F} report a fall overnight.", "Family toured {F} on {D}."),
    "clinic": ("Referred to {F} on {D}.", "Followed at {F} for this problem.", "Seen at {F} on {D}.",
               "Appointment at {F} on {D}.", "{DR} at {F} will follow."),
    "service": ("Referred to {F} on {D}.", "Nurse from {F} to visit on {D}.", "Enrolled with {F} on {D}.",
                "{F} confirmed start of services."),
}
SITE_SENTENCES["rehab"] = SITE_SENTENCES["clinic"] + ("Accepted to {F} for therapy.",)
SITE_FIELDS = {
    "hospital": ("Sending facility: {F}", "Outside hospital: {F}", "Prior admission: {F}", "Transferring facility: {F}"),
    "residence": ("Residence: {F}", "Discharge to: {F}", "Living situation: {F}"),
    "clinic": ("Referred to: {F}", "Follow-up site: {F}", "Ordering location: {F}"),
    "service": ("Agency: {F}", "Provider: {F}", "Referred to: {F}"),
}
SITE_FIELDS["rehab"] = SITE_FIELDS["clinic"]
SITE_LISTS = ("- {F} ({D})", "- Prior site: {F}", "- Follow-up: {F}", "- Records: {F}, received {D}")
SITE_TERSE = ("records from {F}?", "anyone have a number for {F}", "bed at {F} confirmed for {D}",
              "pt goes back to {F} tomorrow", "is {F} still taking referrals?", "f/u with {F} next week")
NEG_SENTENCES = ("Discussed {N} with the family.", "{PC} asked about {N}.", "Information on {N} given.",
                 "Considering {N} after discharge.", "Family prefers {N} closer to home.")
NEG_FIELDS = ("Needs: {N}", "Services discussed: {N}", "Plan: {N}")
NEG_LISTS = ("- {N} (discussed)", "- Consider {N}")
NEG_TERSE = ("any info on {N}?", "family asking about {N}", "does she qualify for {N}?")


def fill_template(template: str, c: Ctx, r: random.Random, **values) -> str:
    out = template
    if "{D}" in out:
        out = out.replace("{D}", c.D(c.d1 if r.random() < 0.5 else c.d2))
    if "{DR}" in out:
        out = out.replace("{DR}", c.drn(r))
    out = out.replace("{P}", c.pron.capitalize()).replace("{PC}", c.pron.capitalize())
    for key, value in values.items():
        out = out.replace("{" + key + "}", value)
    return out


def site_phrase(kind: str, style: str, marked: str, c: Ctx, r: random.Random) -> str:
    if style == "sentence":
        return fill_template(r.choice(SITE_SENTENCES[kind]), c, r, F=marked)
    if style == "field":
        return fill_template(r.choice(SITE_FIELDS[kind]), c, r, F=marked)
    if style == "list":
        return fill_template(r.choice(SITE_LISTS), c, r, F=marked)
    return fill_template(r.choice(SITE_TERSE), c, r, F=marked)


def neg_phrase(style: str, np: str, c: Ctx, r: random.Random) -> str:
    table = {"sentence": NEG_SENTENCES, "field": NEG_FIELDS, "list": NEG_LISTS, "terse": NEG_TERSE}[style]
    out = fill_template(r.choice(table), c, r, N=np)
    if out.startswith(np) and style == "sentence":
        out = out[0].upper() + out[1:]
    return out


# Family 1: long names with inner connectors. (key, kind, builder, ordinary phrase)
CONNECTOR_TEMPLATES = (
    ("saints_home", "residence", lambda v: f"Saints {v.saint()} and {v.saint()} Home for the Aged", "a home for the aged"),
    ("sts_medical_center", "hospital", lambda v: f"Sts. {v.saint()} and {v.saint()} Medical Center", "a medical center near the saints' shrine"),
    ("saint_and_saint", "hospital", lambda v: f"Saint {v.saint()} and Saint {v.saint()} Hospital", "prayer cards of two saints"),
    ("our_lady_mc", "hospital", lambda v: f"Our Lady of {v.inv.word()} Medical Center", "a medal of Our Lady"),
    ("our_lady_hospital", "hospital", lambda v: f"Our Lady of {v.inv.word()} Hospital", "a statue of Our Lady"),
    ("our_lady_of_the", "hospital", lambda v: f"Our Lady of the {v.inv.word()} {v.rng.choice(('Valley', 'Hills', 'Lakes'))} Hospital", "a rosary of Our Lady"),
    ("women_infants", "hospital", lambda v: f"{v.inv.stem()} Center for Women and Infants", "a center for women and infants"),
    ("women_children", "hospital", lambda v: f"{v.inv.stem()} Hospital for Women and Children", "care for women and children"),
    ("rehab_recovery", "rehab", lambda v: f"{v.inv.stem()} Institute for Rehabilitation and Recovery", "rehabilitation and recovery goals"),
    ("children_families", "service", lambda v: f"{v.inv.stem()} Home for Children and Families", "support for children and families"),
    ("and_memorial", "hospital", lambda v: f"{v.inv.word()} and {v.inv.word()} Memorial Hospital", "a memorial service and a hospital visit"),
    ("eye_ear", "clinic", lambda v: f"{v.inv.stem()} Eye and Ear Infirmary", "an eye and ear exam"),
    ("heart_vascular", "clinic", lambda v: f"{v.inv.stem()} Center for Heart and Vascular Care", "heart and vascular care"),
    ("sisters_of", "hospital", lambda v: f"Sisters of St. {v.saint()} Hospital", "the sisters of the patient"),
    ("sisters_of_saint", "residence", lambda v: f"Sisters of Saint {v.saint()} Residence", "a residence near her sisters"),
    ("hospital_of_the", "hospital", lambda v: f"Hospital of the {v.inv.word()} Valley", "the hospital of her choice"),
    ("children_mc", "hospital", lambda v: f"{v.inv.stem()} Children's Hospital and Medical Center", "a children's hospital and medical center"),
    ("aged_infirm", "residence", lambda v: f"{v.inv.stem()} Home for the Aged and Infirm", "a home for the aged and infirm"),
    ("sleep_breathing", "clinic", lambda v: f"{v.inv.stem()} Center for Sleep and Breathing Disorders", "sleep and breathing disorders"),
    ("hospital_and_clinics", "hospital", lambda v: f"{v.inv.stem()} Hospital and Clinics", "hospital and clinic visits"),
    ("nursing_rehab", "residence", lambda v: f"{v.inv.stem()} Nursing and Rehabilitation Center", "nursing and rehabilitation needs"),
    ("brothers_of", "residence", lambda v: f"Brothers of St. {v.saint()} Care Home", "a care home run by brothers"),
    ("st_of_the", "hospital", lambda v: f"St. {v.saint()} of the {v.inv.word()} Hospital", "a hospital of the same order"),
    ("healthy_aging", "clinic", lambda v: f"{v.inv.stem()} Center for Healthy Aging and Memory", "healthy aging and memory"),
    ("mother_baby", "hospital", lambda v: f"{v.inv.stem()} Mother and Baby Hospital", "mother and baby care"),
    ("bone_joint", "clinic", lambda v: f"{v.inv.stem()} Institute for Bone and Joint Health", "bone and joint health"),
)
# Family 2: care-type tails. (tail, kind, ordinary phrase)
CARE_TAILS = (
    ("Memory Care Residences", "residence", "a memory care unit"),
    ("Memory Care Community", "residence", "memory care"),
    ("Behavioral Health Hospital", "hospital", "a behavioral health hospital"),
    ("Rehabilitation Hospital", "rehab", "an inpatient rehabilitation hospital"),
    ("Regional Medical Center - North Campus", "hospital", "the north campus"),
    ("Medical Center - South Campus", "hospital", "the south campus"),
    ("Regional Hospital - East Campus", "hospital", "the east campus"),
    ("Health Center - West Campus", "clinic", "the west campus"),
    ("Senior Living Community", "residence", "a senior living community"),
    ("Long-Term Acute Care Hospital", "hospital", "a long-term acute care hospital"),
    ("Ambulatory Surgery Center", "clinic", "an ambulatory surgery center"),
    ("Cancer Institute", "clinic", "a cancer institute"),
    ("Wound Care Clinic", "clinic", "a wound care clinic"),
    ("Dialysis Center", "clinic", "a dialysis center"),
    ("Hospice House", "residence", "an inpatient hospice house"),
    ("Assisted Living Residences", "residence", "assisted living"),
    ("Skilled Nursing Facility", "residence", "a skilled nursing facility"),
    ("Transitional Care Center", "rehab", "transitional care"),
    ("Infusion Center", "clinic", "the infusion center"),
    ("Urgent Care Clinic", "clinic", "an urgent care clinic"),
    ("Family Medicine Residency Clinic", "clinic", "the residency clinic"),
    ("Psychiatric Hospital", "hospital", "a psychiatric hospital"),
    ("Children's Specialty Clinic", "clinic", "a specialty clinic"),
    ("Heart Hospital", "hospital", "a heart hospital"),
    ("Orthopedic Specialty Hospital", "hospital", "an orthopedic specialty hospital"),
    ("Retirement Residences", "residence", "a retirement residence"),
    ("Endoscopy Center", "clinic", "the endoscopy center"),
    ("Imaging Center", "clinic", "an imaging center"),
    ("Sleep Disorders Center", "clinic", "a sleep center"),
    ("Home Health Services", "service", "home health services"),
    ("Adult Day Program", "service", "an adult day program"),
    ("Hospice and Palliative Care", "service", "hospice and palliative care"),
    ("Regional Medical Center", "hospital", "a regional medical center"),
    ("Community Hospital", "hospital", "a community hospital"),
    ("Medical Center Emergency Department", None, None),  # never used: department words stay outside
)
CARE_TAILS = tuple(t for t in CARE_TAILS if t[1] is not None)
CAMPUS_NEGATIVES = ("Labs drawn at {N}.", "Parking at {N} is limited for the family.", "Imaging done at {N}.")
ADDRESS_SITE_TYPES = ("Medical Center", "Hospital", "Regional Hospital", "Health", "Community Hospital", "Clinic",
                      "Health System", "Family Health Center", "Rehabilitation Hospital", "Regional Medical Center")
AT_SITE_TYPES = ("Family Medicine", "Health Center", "Orthopedics", "Pediatrics", "Women's Health", "Rehabilitation",
                 "Cardiology", "Primary Care", "Dialysis", "Urgent Care", "Internal Medicine", "Physical Therapy")


class Authoring:
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.pools = Pools()
        self.vocab = Vocabulary(self.pools)
        self.people = People(self.rng, self.vocab)
        self.inv = Inventor(self.rng, self.vocab, self.pools)
        self.value_rows = Counter()  # (label, casefold value) -> rows
        self.real_used = {}
        self.hospital_order = self.rng.sample(self.pools.hospitals, len(self.pools.hospitals))
        self.city_order = self.rng.sample(self.pools.cities, len(self.pools.cities))
        self.hospital_index = 0
        self.city_index = 0
        self.filter_counts = Counter()

    def saint(self) -> str:
        return self.rng.choice(SAINT_NAMES)

    # Real value draws, round-robin over shuffled pools so no value repeats often.
    def hospital(self, need=lambda h: True) -> Hospital:
        for _ in range(len(self.hospital_order)):
            h = self.hospital_order[self.hospital_index % len(self.hospital_order)]
            self.hospital_index += 1
            if need(h):
                return h
        raise RuntimeError("no hospital satisfies the request")

    def city(self) -> City:
        c = self.city_order[self.city_index % len(self.city_order)]
        self.city_index += 1
        return c

    def invented_site(self, builder) -> str:
        for _ in range(200):
            name = self.inv.site(builder())
            if name:
                return name
            self.filter_counts["invented_site_rejected"] += 1
        raise RuntimeError("cannot invent a site")

    def locality_surface(self, city: City, form: str) -> str | None:
        surface = {"comma_abbr": f"{city.name}, {city.state}", "space_abbr": f"{city.name} {city.state}",
                   "comma_full": f"{city.name}, {city.state_name}", "nospace_abbr": f"{city.name},{city.state}",
                   "bare": city.name}[form]
        if place_value_pool(surface) != "train" or excluded_value(surface):
            self.filter_counts["locality_surface_not_train_or_excluded"] += 1
            return None
        return surface

    def pick_locality(self, forms) -> tuple[City, str, str]:
        while True:
            city = self.city()
            form = self.rng.choice(forms)
            surface = self.locality_surface(city, form)
            if surface and self.value_rows[("LOCATION", surface.casefold())] < self.repeat_limit - 1:
                return city, form, surface

    # ---- plans -----------------------------------------------------------------------------
    def plan(self, family: str) -> dict:
        rng = self.rng
        if family == "tk21_long_connector_names":
            key, kind, builder, np = rng.choice(CONNECTOR_TEMPLATES)
            return {"key": key, "kind": kind, "site": self.invented_site(lambda: builder(self)), "np": np, "real": []}
        if family == "tk21_care_type_tails":
            tail, kind, np = rng.choice(CARE_TAILS)
            site = self.invented_site(lambda: f"{self.inv.stem(0.85)} {tail}")
            return {"key": tail, "kind": kind, "site": site, "np": np, "real": [], "campus": "Campus" in tail}
        if family == "tk21_names_as_addresses":
            dept = rng.choice(DEPARTMENTS)
            if rng.random() < 0.3:
                h = self.hospital()
                return {"key": "cms", "site": h.name, "dept": dept, "real": [(h.name, "cms_hospital_general_information")]}
            site = self.invented_site(lambda: f"{self.inv.stem(0.6)} {rng.choice(ADDRESS_SITE_TYPES)}")
            return {"key": "invented", "site": site, "dept": dept, "real": []}
        if family == "tk21_real_hospital_abbreviations":
            if rng.random() < 0.4:
                h = self.hospital()
                return {"key": "full", "site": h.name, "full": h.name, "real": [(h.name, "cms_hospital_general_information")]}
            label = rng.choice(["med_ctr", "med_ctr", "hosp", "hosp", "mem_hosp", "reg_med_ctr", "univ_hosp"])
            try:
                h = self.hospital(lambda h: any(l == label for l, _ in h.short_forms))
            except RuntimeError:
                h = self.hospital(lambda h: bool(h.short_forms))
            chosen = next((pair for pair in h.short_forms if pair[0] == label), h.short_forms[0])
            return {"key": chosen[0], "site": chosen[1], "full": h.name,
                    "real": [(chosen[1], "cms_hospital_general_information (short form of the train-pool name)")]}
        if family == "tk21_city_with_state":
            forms = ("comma_abbr", "comma_abbr", "space_abbr", "comma_full", "comma_full", "nospace_abbr")
            city, form, surface = self.pick_locality(forms)
            return {"key": form, "city": city, "loc": surface, "real": [(surface, "census_gazetteer_places_2024")],
                    "who": rng.choice(("patient", "patient", "relative")),
                    "neg": "state" if rng.random() < 0.6 else "named"}
        if family == "tk21_facility_and_locality":
            form = rng.choice(("in", "in", "comma", "at"))
            if form == "at":
                site = self.invented_site(lambda: f"{self.inv.stem(0.5)} {rng.choice(AT_SITE_TYPES)}")
                if rng.random() < 0.6:
                    city, _, bare = self.pick_locality(("bare",))
                    qualified = None
                    for qform in rng.sample(["comma_abbr", "space_abbr", "comma_full"], 3):
                        qualified = self.locality_surface(city, qform)
                        if qualified:
                            break
                    if not qualified:
                        return self.plan(family)
                    real = [(bare, "census_gazetteer_places_2024"), (qualified, "census_gazetteer_places_2024")]
                else:
                    bare = self.inv.stem(0.7)
                    state = rng.choice(self.pools.states)
                    qualified = rng.choice((f"{bare}, {state}", f"{bare} {state}", f"{bare}, {STATES[state]}"))
                    real = []
                return {"key": "at", "site": site, "bare": bare, "qualified": qualified, "real": real}
            if rng.random() < 0.4:
                h = self.hospital(lambda h: h.city is not None)
                city, site = h.city, h.name
                real = [(h.name, "cms_hospital_general_information")]
            else:
                site = self.invented_site(lambda: f"{self.inv.stem(0.6)} {rng.choice(ADDRESS_SITE_TYPES + AT_SITE_TYPES)}")
                city, real = self.city(), []
            if form == "in":
                options = ["comma_abbr", "comma_full", "space_abbr"]
            else:
                options = ["bare", "bare", "comma_abbr"]
            surface = None
            for lform in rng.sample(options, len(options)):
                surface = self.locality_surface(city, lform)
                if surface:
                    break
            if not surface:
                return self.plan(family)
            return {"key": form, "site": site, "loc": surface, "real": real + [(surface, "census_gazetteer_places_2024")]}
        raise KeyError(family)

    # ---- rendering -------------------------------------------------------------------------
    def render_core(self, family, plan, role, style, c, r) -> tuple[list[str], list[str]]:
        if family in ("tk21_long_connector_names", "tk21_care_type_tails"):
            tag = "long_name_inner_connector" if family == "tk21_long_connector_names" else "care_type_tail"
            if role == "positive":
                marked = M("FACILITY", plan["site"])
                if style == "sentence" and r.random() < 0.15:
                    marked = article_site("the", plan["site"]) if plan["kind"] != "service" else marked
                phen = [tag, "facility_boundary"] + (["campus_in_name"] if plan.get("campus") else [])
                return [site_phrase(plan["kind"], style, marked, c, r)], phen
            if plan.get("campus"):
                line = fill_template(r.choice(CAMPUS_NEGATIVES), c, r, N=plan["np"]) if style == "sentence" else neg_phrase(style, plan["np"], c, r)
            else:
                line = neg_phrase(style, plan["np"], c, r)
            neg_tag = "connector_phrase_ordinary" if family == "tk21_long_connector_names" else "care_type_words_alone"
            return [line], [neg_tag, "matched_negative"]

        if family == "tk21_names_as_addresses":
            dept = plan["dept"]
            target = site_with_department(plan["site"], dept) if role == "positive" else dept
            if style == "field":
                line = r.choice(("To: {T}", "From: {T}", "cc: {T}", "Fax to {T}", "Attn: {T}", "Route to: {T}",
                                 "Reply to: {T}")).replace("{T}", target)
                lines = [line]
            elif style == "sentence":
                lines = [r.choice(("Faxed records to {T} at {X}.", "Message routed to {T}.", "Copy sent to {T} on {D}.",
                                   "Spoke with {T} regarding the transfer.", "Left a voicemail for {T} at {X}.",
                                   "Forms returned to {T} on {D}."))
                         .replace("{T}", target).replace("{X}", M("PHONE", c.fax))]
                lines = [fill_template(lines[0], c, r)]
            elif style == "list":
                lines = [r.choice(("- cc: {T}", "- Fax to {T}", "- Notify {T}")).replace("{T}", target)]
            else:
                lines = [r.choice(("fax to {T}?", "who covers {T} tonight", "sent to {T}, waiting on reply",
                                   "can someone call {T}")).replace("{T}", target)]
            phen = ["site_as_address", "department_outside", "facility_boundary"] if role == "positive" else ["department_alone", "matched_negative"]
            return lines, phen

        if family == "tk21_real_hospital_abbreviations":
            if role == "positive":
                marked = M("FACILITY", plan["site"])
                if style == "sentence" and r.random() < 0.3:
                    line = fill_template(r.choice(("Transferred from {F} ED on {D}.", "Prior cath at {F} on {D}.",
                                                   "CT at {F} ED reviewed.", "Discharged from {F} on {D}.")), c, r, F=marked)
                elif style == "terse" and r.random() < 0.5:
                    line = r.choice(("osh = {F}, records?", "pt at {F} wants transfer", "{F} ED calling for a bed")).replace("{F}", marked)
                elif style == "field" and r.random() < 0.5:
                    line = fill_template(r.choice(("OSH: {F}", "Last admit: {F} {D}", "Outside imaging: {F}")), c, r, F=marked)
                else:
                    line = site_phrase("hospital", style, marked, c, r)
                phen = ["real_hospital_name"] + (["p4_full_care_site"] if plan["key"] == "full" else ["care_site_short_form"])
                if re.match(r"(?:St|Ste|Mt|Ft)\. ", plan["site"]):
                    phen.append("leading_abbreviation")
                return [line], phen
            np_by_key = {"med_ctr": ("per med ctr policy, visitors limited", "med ctr policy reviewed"),
                         "hosp": ("hosp f/u in 2 wks", "hosp course uncomplicated", "d/c from hosp yesterday"),
                         "mem_hosp": ("hosp f/u in 2 wks", "hosp course uncomplicated"),
                         "reg_med_ctr": ("transfer to a reg med ctr if worse", "reg med ctr transfer criteria reviewed"),
                         "univ_hosp": ("univ hosp referral if no improvement", "univ hosp second opinion offered"),
                         "full": ("hosp f/u in 2 wks", "hosp course uncomplicated", "med ctr policy reviewed")}
            if plan["key"] == "full" and "University" in plan["site"] and r.random() < 0.7:
                text = r.choice(("A {S} study supports early mobility.", "Per {S} published guidance, continue the statin."))
                line = text.replace("{S}", plan["site"])
                if style == "terse":
                    line = line[0].lower() + line[1:]
                return [line], ["n3_institution_source", "matched_negative"]
            phrase = r.choice(np_by_key[plan["key"]])
            if style == "sentence":
                line = phrase[0].upper() + phrase[1:] + "."
            elif style == "field":
                line = f"Plan: {phrase}"
            elif style == "list":
                line = f"- {phrase}"
            else:
                line = phrase
            return [line], ["abbreviation_without_name", "matched_negative"]

        if family == "tk21_city_with_state":
            if role == "positive":
                loc = M("LOCATION", plan["loc"])
                phen = ["p1_patient_locality", "city_with_state", "location_boundary", f"locality_form_{plan['key']}"]
            else:
                if plan["neg"] == "named":
                    kind, place, sentence = r.choice(self.pools.named_negatives)
                    self.render_real.append((place, f"{kind} place word", None))
                    line = sentence if style != "terse" else sentence[0].lower() + sentence[1:]
                    if style == "field":
                        line = f"Assessment: {sentence}"
                    elif style == "list":
                        line = f"- {sentence}"
                    return [line], [kind, "matched_negative"]
                loc = plan["city"].state_name if r.random() < 0.85 else plan["city"].state
                self.render_real.append((loc, "state name or USPS code (state pool)", plan["city"].state_name if loc != plan["city"].state_name else None))
                phen = ["n4_state_or_country", "matched_negative"]
            if style == "sentence":
                if plan["who"] == "relative":
                    line = r.choice((f"{c.rel_role.capitalize()} {c.name(r, c.rel)} lives in {loc}.",
                                     f"{c.rel_role.capitalize()} in {loc} is the main contact.",
                                     f"Plans to stay with {c.poss} {c.rel_role} in {loc} after discharge."))
                else:
                    line = r.choice((f"Lives in {loc} with {c.poss} {c.rel_role}.",
                                     f"Moved from {loc} in {M('DATE', str(c.year_moved))}.",
                                     f"Recently returned from {loc}.", f"Grew up in {loc}; retired machinist.",
                                     f"Visiting family from {loc}.", f"Plans to relocate to {loc} next month."))
            elif style == "field":
                line = r.choice((f"Home: {loc}", f"City/State: {loc}", f"Residence: {loc}", f"Hometown: {loc}",
                                 f"Emergency contact: {c.name(r, c.rel)} ({c.rel_role}), {loc}, {M('PHONE', c.phone2)}"))
                if role != "positive" and line.startswith(("City/State", "Hometown")):
                    line = "Home state: " + loc
            elif style == "list":
                line = r.choice((f"- {c.rel_role} {c.name(r, c.rel)} ({loc})", f"- Lives: {loc}",
                                 f"- Moving to {loc}"))
            else:
                line = r.choice((f"pt from {loc} needs local pcp", f"anyone know a cardiologist near {loc}?",
                                 f"moving to {loc} next month, transfer care?", f"family flying in from {loc} tonight"))
            return [line], phen

        if family == "tk21_facility_and_locality":
            if plan["key"] == "at":
                if role == "positive":
                    x = site_at_continuation(plan["site"], plan["bare"])
                    phen = ["p5a_at_continuation", "facility_boundary"]
                else:
                    x = site_at_qualified(plan["site"], plan["qualified"])
                    phen = ["p5a_qualified_locality_split", "facility_boundary", "location_boundary", "matched_contrast"]
            elif plan["key"] == "in":
                x = site_in_locality(plan["site"], plan["loc"])
                phen = ["p5_care_site_then_locality", "facility_boundary", "location_boundary"]
            else:
                x = site_comma_locality(plan["site"], plan["loc"])
                phen = ["p5_care_site_then_locality", "facility_boundary", "location_boundary"]
            if style == "sentence":
                line = fill_template(r.choice(("Transferred from {X} on {D}.", "Seen at {X} on {D}.",
                                               "Records from {X} reviewed.", "{DR} at {X} will follow.",
                                               "Recent admission to {X} for pneumonia.")), c, r, X=x)
            elif style == "field":
                line = r.choice(("Sending facility: {X}", "Prior care: {X}", "Referred from: {X}")).replace("{X}", x)
            elif style == "list":
                line = fill_template(r.choice(("- {X} ({D})", "- Prior site: {X}")), c, r, X=x)
            else:
                line = r.choice(("pt seen at {X}?", "records from {X} pending", "bed offered at {X}")).replace("{X}", x)
            return [line], phen
        raise KeyError(family)

    # ---- rows ------------------------------------------------------------------------------
    def make_row(self, family, layout_name, case, plan, role, ctx, frame_seed, pair_id):
        style, frame = LAYOUTS[layout_name]
        ctx.used = set()
        r = random.Random(frame_seed)
        self.render_real = []
        core, phen = self.render_core(family, plan, role, style, ctx, random.Random(frame_seed + 1))
        marked = frame(r, ctx, core)
        text, spans = parse(marked)
        if case != "normal":
            cased = text.upper() if case == "upper" else text.lower()
            if len(cased) == len(text):
                text = cased
            else:
                case = "normal"
        for span in spans:
            span["text"] = text[span["start"]:span["end"]]
        persons = sorted({ctx.pt, *ctx.used}, key=lambda p: p.full)
        groups = [ctx.pt.group] + sorted({p.group for p in persons} - {ctx.pt.group})
        candidates = [(v, s, plan.get("full") if plan.get("full") and v != plan.get("full") else None) for v, s in plan["real"]]
        real = [{"value": v, "source": s, "pool": place_value_pool(v, expansion=e)}
                for v, s, e in candidates + self.render_real if v.casefold() in text.casefold()]
        return {
            "id": None, "text": text, "spans": spans, "split": "train", "suite": SUITE,
            "annotation_policy": POLICY_VERSION, "cohort": "realistic", "family": family,
            "template_family": family, "layout": layout_name, "case": case, "role": role,
            "phenomena": sorted(set(phen) | {f"case_{case}"}), "identity_group": ctx.pt.group,
            "identity_groups": groups, "pair_id": pair_id, "real_place_values": real,
        }

    def generate(self) -> list[dict]:
        rng = self.rng
        total = sum(FAMILY_ROWS.values())
        self.repeat_limit = max(1, int(total * REPEAT_FRACTION))
        units = []
        for family, n in FAMILY_ROWS.items():
            pairs = round(n / 3)
            units += [(family, True)] * pairs + [(family, False)] * (n - 2 * pairs)
        rng.shuffle(units)
        names = sorted(LAYOUTS)
        # Balance layouts by rows: give each unit the least-used layout among a random shortlist.
        layout_rows = Counter({name: 0 for name in names})
        rows, texts = [], set()
        for number, (family, paired) in enumerate(units):
            shortlist = rng.sample(names, 6)
            layout_name = min(shortlist, key=lambda n: (layout_rows[n], n))
            case = rng.choices([c for c, _ in CASES], weights=[w for _, w in CASES])[0]
            for attempt in range(200):
                ctx = Ctx(rng, self.people)
                plan = self.plan(family)
                frame_seed = rng.randrange(2 ** 31)
                pair_id = f"tk21-pair-{number:05d}" if paired else f"tk21-unit-{number:05d}"
                built = [self.make_row(family, layout_name, case, plan, "positive", ctx, frame_seed, pair_id)]
                if paired:
                    role = "contrast" if family == "tk21_facility_and_locality" else "negative"
                    if family == "tk21_facility_and_locality" and plan["key"] != "at":
                        # P5a contrasts need an `at` plan; redraw until one comes up.
                        self.filter_counts["redraw_for_p5a_contrast"] += 1
                        continue
                    built.append(self.make_row(family, layout_name, case, plan, role, ctx, frame_seed, pair_id))
                elif family == "tk21_facility_and_locality" and plan["key"] == "at" and rng.random() < 0.5:
                    # Unpaired qualified splits so both P5a forms appear outside pairs too.
                    built = [self.make_row(family, layout_name, case, plan, "contrast", ctx, frame_seed, pair_id)]
                reason = self.reject_reason(built, texts)
                if reason:
                    self.filter_counts[reason] += 1
                    continue
                break
            else:
                raise RuntimeError(f"could not author unit {number} ({family})")
            for row in built:
                texts.add(row["text"])
                for key in {(s["label"], s["text"].casefold()) for s in row["spans"] if s["label"] in ("FACILITY", "LOCATION")}:
                    self.value_rows[key] += 1
                layout_rows[layout_name] += 1
                rows.append(row)
        order = list(range(len(rows)))
        rng.shuffle(order)
        rows = [rows[i] for i in order]
        for index, row in enumerate(rows):
            row["id"] = f"tk21-{index + 1:05d}"
        return rows

    def reject_reason(self, built, texts) -> str | None:
        for row in built:
            if row["text"] in texts or (len(built) == 2 and built[0]["text"] == built[1]["text"]):
                return "duplicate_text"
            if excluded_sequences(row["text"]):
                return "row_rejected_by_exclusion_filter"
            if annotation_errors(row["text"], row["spans"]):
                return "annotation_error"
            for key in {(s["label"], s["text"].casefold()) for s in row["spans"] if s["label"] in ("FACILITY", "LOCATION")}:
                if self.value_rows[key] + sum(1 for other in built for s in other["spans"] if (s["label"], s["text"].casefold()) == key) > self.repeat_limit:
                    return "repetition_limit"
            if any(item["pool"] != "train" for item in row["real_place_values"]):
                return "heldout_value"
        return None


# --------------------------------------------------------------------------------------------
# Artifacts


def support(rows):
    counts = Counter(s["label"] for row in rows for s in row["spans"])
    return {label: counts[label] for label in ENTITY_TYPES}


def payload(rows) -> bytes:
    return b"".join(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n" for row in rows)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generator_sha256() -> str:
    return sha256_bytes(Path(__file__).resolve().read_bytes())


def source_pins() -> dict:
    pins = {}
    for name, source in place_sources.SOURCES.items():
        pins[name] = dict(source)
        pins[name]["verified_sha256"] = sha256_bytes(place_sources.read_source(name, raw=True))
    return pins


def module_hashes() -> dict:
    out = {}
    for rel in ("src/clinical_deid/data/annotation_policy.py", "src/clinical_deid/data/annotation_policy_places.py",
                "src/clinical_deid/data/annotation_policy_care_sites.py", "src/clinical_deid/data/annotation_policy_ages.py",
                "src/clinical_deid/data/generators/places.py", "src/clinical_deid/data/generators/place_sources.py",
                BRIEF):
        out[rel] = sha256_bytes((ROOT / rel).read_bytes())
    return out


def build(seed: int = SEED) -> dict:
    authoring = Authoring(seed)
    rows = authoring.generate()
    data = payload(rows)
    label_support = support(rows)
    real_values = [item for row in rows for item in row["real_place_values"]]
    real_pools = Counter(item["pool"] for item in real_values)
    distinct_real = {(item["value"].casefold(), item["source"]) for item in real_values}
    value_rows = Counter()
    for row in rows:
        for key in {(s["label"], s["text"].casefold()) for s in row["spans"] if s["label"] in ("FACILITY", "LOCATION")}:
            value_rows[key] += 1
    stats = {
        "family_role_rows": {f: dict(Counter(r["role"] for r in rows if r["family"] == f)) for f in FAMILY_ROWS},
        "layout_rows": dict(sorted(Counter(r["layout"] for r in rows).items())),
        "case_rows": dict(Counter(r["case"] for r in rows)),
        "family_layout_rows": {f: dict(sorted(Counter(r["layout"] for r in rows if r["family"] == f).items())) for f in FAMILY_ROWS},
        "phenomena_rows": dict(sorted(Counter(p for r in rows for p in r["phenomena"]).items())),
        "rows_without_spans": sum(1 for r in rows if not r["spans"]),
        "rows_without_facility_or_location": sum(1 for r in rows if not any(s["label"] in ("FACILITY", "LOCATION") for s in r["spans"])),
        "max_rows_per_facility_or_location_value": max(value_rows.values()) if value_rows else 0,
        "repeat_limit_rows": authoring.repeat_limit,
        "distinct_facility_values": sum(1 for (label, _) in value_rows if label == "FACILITY"),
        "distinct_location_values": sum(1 for (label, _) in value_rows if label == "LOCATION"),
        "real_value_occurrences_by_pool": {"train": real_pools.get("train", 0), "heldout": real_pools.get("heldout", 0)},
        "distinct_real_values": len(distinct_real),
        "distinct_real_values_by_source": dict(Counter(source for _, source in distinct_real)),
        "pool_counts": authoring.pools.counts,
        "invented_vocabulary_counts": authoring.vocab.counts,
        "generation_filter_counts": dict(sorted(authoring.filter_counts.items())),
        "final_rows_with_excluded_sequences": sum(1 for r in rows if excluded_sequences(r["text"])),
        "ages_annotated": label_support["AGE"],
    }
    manifest = {
        "corpus": "data/tk21_train_care_site_boundaries.jsonl",
        "brief": BRIEF,
        "annotation_policy": POLICY_VERSION,
        "annotation_policy_addendum": ADDENDUM,
        "age_policy": "all ages annotated (2026-10-06), hypothetical ages excluded and decades as AGE (2026-10-07)",
        "split": "train",
        "suite": SUITE,
        "role": "train",
        "independent_final": False,
        "rows": len(rows),
        "documents": len(rows),
        "sha256": sha256_bytes(data),
        "label_support": label_support,
        "entity_support": label_support,
        "label_counts": label_support,
        "spans": sum(label_support.values()),
        "cohort_counts": dict(Counter(r["cohort"] for r in rows)),
        "family_counts": dict(Counter(r["family"] for r in rows)),
        "layout_counts": stats["layout_rows"],
        "case_counts": stats["case_rows"],
        "author_provenance": {
            "author": "isolated synthetic-data author (tk21)",
            "generator": GENERATOR,
            "generator_sha256": generator_sha256(),
            "generator_version": VERSION,
            "seed": seed,
            "input_sha256": module_hashes(),
            "inputs_read": [BRIEF, "docs/annotation-policy-v2.md", "docs/annotation-policy-v2-places.md",
                            "src/clinical_deid/types.py", "src/clinical_deid/data/annotation_policy*.py (fixture modules)",
                            "scripts/takeover/validate_bank.py", "AGENTS.md", "CLEANUP.md",
                            "helper functions exclusion_hash, excluded_sequences, excluded_value, place_value_pool, place_sources loaders",
                            ".cache/place_sources (pinned public sources)"],
            "synthetic_schemes": SCHEMES,
        },
        "source_pins": source_pins(),
        "exclusion_filter": {
            "list": "data/places_exclusions.sha256.json", "list_sha256": EXCLUSIONS_SHA256,
            "normalisation": "sha256(NFC(value.strip()).casefold()) over every 1-8 word sequence",
            "counts": {
                "states_excluded": authoring.pools.counts["states"].get("excluded", 0),
                "gazetteer_places_excluded": authoring.pools.counts["gazetteer_places"].get("excluded", 0),
                "cms_hospitals_excluded": authoring.pools.counts["cms_hospitals"].get("excluded", 0),
                "named_instruments_excluded": authoring.pools.counts["named_instruments_and_diseases"].get("excluded", 0),
                "invented_words_excluded": authoring.vocab.counts.get("excluded", 0),
                "rows_rejected_by_exclusion_filter": authoring.filter_counts.get("row_rejected_by_exclusion_filter", 0),
                "final_rows_with_excluded_sequences": stats["final_rows_with_excluded_sequences"],
            },
        },
        "stats": stats,
        "limitations": [
            "Synthetic training bank; not evaluation evidence. Layout frames and cores are authored templates.",
            "Sealed-overlap audit, development overlap and heldout_dev_v2 overlap counts are run by the owner's agent, not here.",
            "Family 6 pairs are P5a contrasts (the same site and locality written as an `at` continuation and as an "
            "`at` locality with a state qualifier); both members carry spans.",
        ],
    }
    return {"rows": rows, "payload": data, "manifest": manifest, "stats": stats}


def mark_row(row) -> str:
    text, out, previous = row["text"], [], 0
    for span in sorted(row["spans"], key=lambda s: s["start"]):
        out.append(text[previous:span["start"]])
        out.append(f"[{span['label']}: {text[span['start']:span['end']]}]")
        previous = span["end"]
    out.append(text[previous:])
    return "".join(out)


def spot_check(rows, seed) -> str:
    sample = random.Random(seed + 1).sample(rows, min(100, len(rows)))
    lines = ["# tk21 spot-check sample", "",
             f"100 rows drawn with `random.Random({seed + 1}).sample`. Gold is marked `[LABEL: text]`.", ""]
    for row in sorted(sample, key=lambda r: r["id"]):
        lines += [f"## {row['id']} - {row['family']} / {row['layout']} / {row['role']} / {row['case']}", "",
                  "```", mark_row(row), "```", ""]
    return "\n".join(lines)


def author_report(manifest) -> str:
    s = manifest["stats"]
    L = ["# tk21 author report: US care-site names and localities, boundaries", "",
         f"Bank `{manifest['corpus']}`, {manifest['rows']} rows, split `train`, policy `{manifest['annotation_policy']}` "
         f"with `{manifest['annotation_policy_addendum']}`; all ages annotated. Generator `{GENERATOR}` "
         f"(SHA-256 `{manifest['author_provenance']['generator_sha256']}`), seed {manifest['author_provenance']['seed']}. "
         f"Corpus SHA-256 `{manifest['sha256']}`.", "",
         "Counts only. This is a synthetic training bank and its contents are not evaluation evidence.", "",
         "## Rows per family and role", "", "| Family | Positive | Matched negative | P5a contrast | Total |", "|---|---:|---:|---:|---:|"]
    for family, roles in s["family_role_rows"].items():
        L.append(f"| {family} | {roles.get('positive', 0)} | {roles.get('negative', 0)} | {roles.get('contrast', 0)} | {sum(roles.values())} |")
    L += ["", f"Rows with no span: {s['rows_without_spans']}. Rows with no FACILITY or LOCATION span: "
               f"{s['rows_without_facility_or_location']}.", "",
          "## Rows per layout", "", f"{len(s['layout_rows'])} layouts; largest share "
          f"{max(s['layout_rows'].values()) / manifest['rows']:.1%}.", "", "| Layout | Rows | Share |", "|---|---:|---:|"]
    for name, n in s["layout_rows"].items():
        L.append(f"| {name} | {n} | {n / manifest['rows']:.1%} |")
    families = list(s["family_layout_rows"])
    L += ["", "## Rows per layout and family", "",
          "| Layout | " + " | ".join(f.replace("tk21_", "") for f in families) + " |",
          "|---|" + "---:|" * len(families)]
    for name in s["layout_rows"]:
        L.append(f"| {name} | " + " | ".join(str(s["family_layout_rows"][f].get(name, 0)) for f in families) + " |")
    L += ["", "## Case mix (whole rows)", "", "| Case | Rows | Share |", "|---|---:|---:|"]
    for name, n in sorted(s["case_rows"].items()):
        L.append(f"| {name} | {n} | {n / manifest['rows']:.1%} |")
    L += ["", "## Label support", "", "| Label | Spans |", "|---|---:|"]
    for label, n in manifest["label_support"].items():
        L.append(f"| {label} | {n} |")
    L += ["", "## Phenomena (rows)", "", "| Phenomenon | Rows |", "|---|---:|"]
    for name, n in s["phenomena_rows"].items():
        L.append(f"| {name} | {n} |")
    L += ["", "## Real-value pools", "",
          f"Real-value occurrences in the bank by pool: train {s['real_value_occurrences_by_pool']['train']}, "
          f"**heldout {s['real_value_occurrences_by_pool']['heldout']}**. Distinct real values: {s['distinct_real_values']} "
          f"({', '.join(f'{k}: {v}' for k, v in sorted(s['distinct_real_values_by_source'].items()))}).", "",
          "Source candidates before use (the heldout pool is never drawn from):", "",
          "| Source | Counts |", "|---|---|"]
    for name, counts in s["pool_counts"].items():
        L.append(f"| {name} | {', '.join(f'{k} {v}' for k, v in counts.items())} |")
    L += ["", f"Invented vocabulary: {', '.join(f'{k} {v}' for k, v in s['invented_vocabulary_counts'].items())}.", "",
          f"Repetition: the most frequent FACILITY or LOCATION value appears in {s['max_rows_per_facility_or_location_value']} rows "
          f"(limit {s['repeat_limit_rows']}, 0.5% of rows). Distinct FACILITY values {s['distinct_facility_values']}, "
          f"distinct LOCATION values {s['distinct_location_values']}.", "",
          "## Exclusion filter", "",
          f"List `{manifest['exclusion_filter']['list']}` (SHA-256 `{manifest['exclusion_filter']['list_sha256']}`), applied through "
          "`excluded_value` to every candidate real value, invented word and invented site, and through `excluded_sequences` to "
          "every authored row.", "", "| Filter | Count |", "|---|---:|"]
    for name, n in manifest["exclusion_filter"]["counts"].items():
        L.append(f"| {name} | {n} |")
    L += ["", "Generation retries by reason:", "", "| Reason | Count |", "|---|---:|"]
    for name, n in s["generation_filter_counts"].items():
        L.append(f"| {name} | {n} |")
    L += ["", "## Cases the policy did not settle (avoided, not decided)", ""]
    L += [f"- {item}" for item in UNSETTLED]
    L += ["", "## Not done here", "",
          "The sealed-overlap audit, the development overlap check and the heldout_dev_v2 overlap counts are for the owner's "
          "agent. No model was trained or evaluated. Nothing was committed.", ""]
    return "\n".join(L)


UNSETTLED = (
    "A campus or site suffix written after a comma (`<site> Medical Center, North Campus`). The brief settles the dash form "
    "authored as part of the name (inside the span); the comma form could also be read as a P5 split. Only the dash form was authored.",
    "A city, state and ZIP after a facility or `at` (`<site> at <city>, ST 12345`). Whether the ZIP joins the LOCATION span "
    "is not shown by any fixture; no ZIP was authored after a locality.",
    "A bare invented stem with no care-type word used as an address (`To: <stem> Nurse Pool`): the stem could be read as a "
    "site name (P3) or as a locality modifying a care team (P7). Every address-family site carries a care-type word.",
    "CMS names whose own text carries a trailing ` - <campus or city>` or `/` part, or that start with `THE`, were skipped "
    "because their surface form and the `The` convention for a real name could not be settled from the source alone.",
    "`Our Lady` as a religious reference in the matched negatives (`a statue of Our Lady`) is left unannotated as an ordinary "
    "phrase; the policy has no explicit fixture for religious titles that are not names of a care site.",
    "Names in `Surname, Given` order (an internal comma inside NAME) were not authored.",
)


def write(artifacts, seed) -> None:
    BANK.parent.mkdir(parents=True, exist_ok=True)
    BANK.write_bytes(artifacts["payload"])
    manifest_path = BANK.with_suffix(BANK.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(artifacts["manifest"], indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "AUTHOR_REPORT.md").write_text(author_report(artifacts["manifest"]))
    (REPORT_DIR / "spot_check_sample.md").write_text(spot_check(artifacts["rows"], seed))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    artifacts = build(args.seed)
    if not args.no_write:
        write(artifacts, args.seed)
    m = artifacts["manifest"]
    print(json.dumps({"rows": m["rows"], "sha256": m["sha256"], "label_support": m["label_support"],
                      "heldout": m["stats"]["real_value_occurrences_by_pool"]["heldout"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
