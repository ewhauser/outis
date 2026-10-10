"""Author tk24: named instruments, scores, criteria, signs, tests and eponyms.

Commissioned by docs/instrument-eponyms-bank-brief.md (policy decision E1,
2026-10-08, and places addendum N1). A person's or place's name inside a named
instrument, score, criterion, sign, test, manoeuvre, classification or device
is never annotated; a person who shares the surname is NAME. Every person,
date, age and identifier is invented. Instrument names are standard public
clinical terms listed from general knowledge; a real place name inside one is
used only when it is in the addendum's ``train`` value pool.

Rows are rejected when any inserted value, or any 1-8 word sequence of the
text, hashes onto either blind-bank exclusion list (hash-only files).

Usage:
  PYTHONPATH=src python scripts/takeover/author24_instrument_eponyms.py \
      [--seed 2424] [--rows 1500] [--output data/tk24_train_instrument_eponyms.jsonl]
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from outis.data.annotation_policy import (  # noqa: E402
    POLICY_VERSION, annotation_errors, canonical_identity_group, identity_split,
)
from outis.data.annotation_policy_places import ADDENDUM, place_value_pool  # noqa: E402
from outis.data.generators.places import (  # noqa: E402
    exclusion_hash, exclusion_hashes, word_sequences,
)
from outis.types import ENTITY_TYPES  # noqa: E402

SEED = 2424
ROWS = 1500
SUITE = "training"
BANK = "tk24_train_instrument_eponyms"
OUTPUT = ROOT / "data" / f"{BANK}.jsonl"
EXCLUSIONS_V2 = ROOT / "data" / "places_exclusions_v2.sha256.json"
EXCLUSIONS_V2_SHA256 = "ad952f5dffbd3f4eaca16159d43338838b3e6e0600b91557707e41164945d751"
MRN_SCHEME = "E24- prefix and seven invented random digits, or eight bare invented digits"
ID_SCHEME = "CSN 7 followed by nine invented digits; order numbers ORD- and six invented digits"
ACCOUNT_SCHEME = "24 followed by eight invented digits"
PHONE_SCHEME = "US 555-0100..0199 fictional lines with invented area codes"
CAP_SHARE = 0.02

FAMILY_PLAN = {
    "tk24_instruments_shorthand": 400,
    "tk24_instruments_fields": 300,
    "tk24_instruments_with_phi": 400,
    "tk24_same_surname": 200,
    "tk24_instruments_narrative": 200,
}
LAYOUT_PLAN = {
    "tk24_instruments_shorthand": {"sh_triage": 58, "sh_progress": 57, "sh_overnight": 57, "sh_slash": 57,
                                   "sh_bullets": 57, "sh_page": 57, "sh_semicolon": 57},
    "tk24_instruments_fields": {"fld_kv": 50, "fld_flowsheet": 50, "fld_pipe": 50, "fld_checklist": 50,
                                "fld_csv": 50, "fld_section": 50},
    "tk24_instruments_with_phi": {"phi_sh_header": 110, "phi_sh_note": 110, "phi_fld_header": 100,
                                  "phi_fld_order": 80},
    "tk24_same_surname": {"ss_shorthand": 80, "ss_fields": 50, "ss_narrative": 70},
    "tk24_instruments_narrative": {"nar_sentence": 50, "nar_query": 50, "nar_list": 50, "nar_plan": 50},
}
LAYOUT_STYLE = {
    **{k: "shorthand" for k in ("sh_triage", "sh_progress", "sh_overnight", "sh_slash", "sh_bullets", "sh_page",
                                "sh_semicolon", "phi_sh_header", "phi_sh_note", "ss_shorthand")},
    **{k: "structured" for k in ("fld_kv", "fld_flowsheet", "fld_pipe", "fld_checklist", "fld_csv",
                                 "fld_section", "phi_fld_header", "phi_fld_order", "ss_fields")},
    "ss_narrative": "narrative", "nar_sentence": "narrative", "nar_query": "query", "nar_list": "list",
    "nar_plan": "narrative",
}
CASE_SHARES = (("lower", 0.5), ("upper", 0.2), ("normal", 0.3))

# ---------------------------------------------------------------------------
# Instrument catalogue: standard public clinical terms, from general knowledge.
# (key, kind, origin, full name, shorthand surfaces, value spec, eponym word, population)
# value spec: "i:lo:hi[:step]" integers, "l:a;b;c" explicit list, or a named set below.
# eponym word: the surname for person origin (used for same-surname rows), the
# place for place origin (checked against the addendum value pool).
# ---------------------------------------------------------------------------
NAMED_VALUES = {
    "pn": ["neg", "pos", "negative", "positive", "-", "+"],
    "sign": ["neg", "pos", "negative", "positive", "absent", "present", "neg bilat", "pos R"],
    "test": ["neg", "pos", "negative", "positive", "neg bilat", "pos L", "pos R", "equivocal"],
    "crit": ["met", "not met", "neg", "pos", "negative", "positive"],
    "man": ["performed", "done", "successful", "repeated x2", "attempted"],
    "dev": ["in place", "placed", "removed", "d/c'd", "patent", "draining", "intact"],
}
CATALOGUE = [
    # ---- named after people: scores, scales, classifications ----
    ("wells", "score", "person", "Wells score", "Wells", "i:0:9", "Wells", None),
    ("bishop", "score", "person", "Bishop score", "Bishop", "i:2:13", "Bishop", "ob"),
    ("apgar", "score", "person", "Apgar score", "Apgar|Apgars", "l:8/9;7/9;9/9;6/8;8/8;5/7;9/10", "Apgar", "neonate"),
    ("braden", "scale", "person", "Braden Scale", "Braden", "i:9:23", "Braden", None),
    ("morse", "scale", "person", "Morse Fall Scale", "Morse|Morse fall score", "i:0:125:5", "Morse", None),
    ("norton", "scale", "person", "Norton Scale", "Norton", "i:6:20", "Norton", None),
    ("waterlow", "score", "person", "Waterlow score", "Waterlow", "i:4:28", "Waterlow", None),
    ("mallampati", "classification", "person", "Mallampati class", "Mallampati", "l:I;II;III;IV;1;2;3", "Mallampati", None),
    ("cormack", "classification", "person", "Cormack-Lehane grade", "Cormack-Lehane|CL grade", "l:1;2a;2b;3;I;II", "Lehane", None),
    ("childpugh", "classification", "person", "Child-Pugh class", "Child-Pugh|CTP", "l:A;B;C;A (5);B (8);B (7);C (11)", "Pugh", None),
    ("killip", "classification", "person", "Killip class", "Killip", "l:I;II;III;IV", "Killip", None),
    ("ranson", "criteria", "person", "Ranson criteria", "Ranson", "i:0:6", "Ranson", None),
    ("alvarado", "score", "person", "Alvarado score", "Alvarado", "i:1:10", "Alvarado", None),
    ("centor", "score", "person", "Centor score", "Centor", "i:0:4", "Centor", None),
    ("mcisaac", "score", "person", "McIsaac score", "McIsaac", "i:0:5", "McIsaac", None),
    ("caprini", "score", "person", "Caprini score", "Caprini", "i:0:12", "Caprini", None),
    ("khorana", "score", "person", "Khorana score", "Khorana", "i:0:4", "Khorana", None),
    ("charlson", "score", "person", "Charlson Comorbidity Index", "Charlson|CCI", "i:0:11", "Charlson", None),
    ("karnofsky", "scale", "person", "Karnofsky Performance Status", "Karnofsky|KPS", "i:30:100:10", "Karnofsky", None),
    ("lansky", "scale", "person", "Lansky Play-Performance Scale", "Lansky", "i:40:100:10", "Lansky", "peds"),
    ("barthel", "score", "person", "Barthel Index", "Barthel", "i:20:100:5", "Barthel", None),
    ("katz", "score", "person", "Katz ADL index", "Katz|Katz ADL", "i:0:6", "Katz", None),
    ("lawton", "scale", "person", "Lawton IADL scale", "Lawton|Lawton IADL", "i:0:8", "Lawton", None),
    ("berg", "scale", "person", "Berg Balance Scale", "Berg|BBS", "i:18:56", "Berg", None),
    ("tinetti", "score", "person", "Tinetti gait and balance score", "Tinetti", "i:10:28", "Tinetti", None),
    ("folstein", "test", "person", "Folstein Mini-Mental State Examination", "Folstein|Folstein MMSE", "l:24/30;27/30;29/30;21/30;18/30;30/30", "Folstein", None),
    ("beck", "scale", "person", "Beck Depression Inventory", "Beck|BDI-II|Beck inventory", "i:2:40", "Beck", None),
    ("hamilton", "scale", "person", "Hamilton Depression Rating Scale", "Hamilton|HAM-D", "i:3:30", "Hamilton", None),
    ("ramsay", "scale", "person", "Ramsay Sedation Scale", "Ramsay", "i:1:6", "Ramsay", None),
    ("ashworth", "scale", "person", "Modified Ashworth Scale", "Ashworth|MAS", "l:0;1;1+;2;3", "Ashworth", None),
    ("fitzpatrick", "classification", "person", "Fitzpatrick skin type", "Fitzpatrick", "l:I;II;III;IV;V;VI", "Fitzpatrick", None),
    ("gleason", "score", "person", "Gleason score", "Gleason", "l:3+3=6;3+4=7;4+3=7;4+4=8;4+5=9", "Gleason", None),
    ("breslow", "classification", "person", "Breslow thickness", "Breslow|Breslow depth", "l:0.4 mm;0.8 mm;1.2 mm;2.1 mm;3.5 mm", "Breslow", None),
    ("clark", "classification", "person", "Clark level", "Clark|Clark level", "l:II;III;IV", "Clark", None),
    ("hunthess", "classification", "person", "Hunt and Hess grade", "Hunt-Hess|H&H", "i:1:5", "Hess", None),
    ("fisher", "classification", "person", "Fisher grade", "Fisher|modified Fisher", "i:1:4", "Fisher", None),
    ("spetzler", "classification", "person", "Spetzler-Martin grade", "Spetzler-Martin|SM grade", "i:1:5", "Spetzler", None),
    ("rankin", "scale", "person", "modified Rankin Scale", "Rankin|mRS", "i:0:5", "Rankin", None),
    ("fuglmeyer", "score", "person", "Fugl-Meyer Assessment", "Fugl-Meyer|FMA-UE", "i:12:66", "Meyer", None),
    ("brunnstrom", "classification", "person", "Brunnstrom stage", "Brunnstrom", "i:1:6", "Brunnstrom", None),
    ("ballard", "score", "person", "New Ballard Score", "Ballard", "i:15:45:5", "Ballard", "neonate"),
    ("tanner", "classification", "person", "Tanner stage", "Tanner", "l:1;2;3;4;5;II;III;IV", "Tanner", "peds"),
    ("finnegan", "score", "person", "Finnegan score", "Finnegan|Finnegans", "i:2:14", "Finnegan", "neonate"),
    ("cobb", "classification", "person", "Cobb angle", "Cobb", "l:12 deg;18 deg;24 deg;31 deg;42 deg", "Cobb", None),
    ("beighton", "score", "person", "Beighton score", "Beighton", "i:0:9", "Beighton", None),
    ("risser", "classification", "person", "Risser grade", "Risser", "i:0:5", "Risser", "peds"),
    ("goldman", "classification", "person", "Goldman cardiac risk class", "Goldman", "l:I;II;III;IV", "Goldman", None),
    ("lee", "score", "person", "Lee Revised Cardiac Risk Index", "Lee index|Lee RCRI", "i:0:4", "Lee", None),
    ("rockall", "score", "person", "Rockall score", "Rockall", "i:0:8", "Rockall", None),
    ("maddrey", "score", "person", "Maddrey discriminant function", "Maddrey|Maddrey DF", "i:12:68", "Maddrey", None),
    ("truelove", "classification", "person", "Truelove and Witts severity", "Truelove-Witts", "l:mild;moderate;severe", "Witts", None),
    ("harvey", "score", "person", "Harvey-Bradshaw Index", "Harvey-Bradshaw|HBI", "i:1:14", "Bradshaw", None),
    ("hinchey", "classification", "person", "Hinchey classification", "Hinchey", "l:I;II;III;IV;Ia;Ib", "Hinchey", None),
    ("forrest", "classification", "person", "Forrest classification", "Forrest", "l:Ia;Ib;IIa;IIb;IIc;III", "Forrest", None),
    ("savary", "classification", "person", "Savary-Miller grade", "Savary-Miller", "l:I;II;III;IV", "Savary", None),
    ("salter", "classification", "person", "Salter-Harris classification", "Salter-Harris|SH", "l:I;II;III;IV;type II", "Salter", None),
    ("gustilo", "classification", "person", "Gustilo-Anderson classification", "Gustilo|Gustilo-Anderson", "l:I;II;IIIA;IIIB", "Gustilo", None),
    ("garden", "classification", "person", "Garden classification", "Garden", "l:I;II;III;IV", "Garden", None),
    ("schatzker", "classification", "person", "Schatzker classification", "Schatzker", "l:I;II;IV;VI", "Schatzker", None),
    ("gartland", "classification", "person", "Gartland classification", "Gartland", "l:I;II;III", "Gartland", "peds"),
    ("weberankle", "classification", "person", "Weber ankle fracture classification", "Weber", "l:A;B;C", "Weber", None),
    ("bosniak", "classification", "person", "Bosniak classification", "Bosniak", "l:I;II;IIF;III;IV", "Bosniak", None),
    ("rutherford", "classification", "person", "Rutherford classification", "Rutherford", "i:0:6", "Rutherford", None),
    ("fontaine", "classification", "person", "Fontaine stage", "Fontaine", "l:I;IIa;IIb;III;IV", "Fontaine", None),
    ("wongbaker", "scale", "person", "Wong-Baker FACES scale", "Wong-Baker|FACES", "i:0:10:2", "Baker", "peds"),
    ("sgarbossa", "criteria", "person", "Sgarbossa criteria", "Sgarbossa", "crit", "Sgarbossa", None),
    ("brugada", "criteria", "person", "Brugada criteria", "Brugada", "crit", "Brugada", None),
    ("light", "criteria", "person", "Light criteria", "Light's|Lights", "l:exudate;transudate;met;not met", "Light", None),
    ("dukecrit", "criteria", "person", "Duke criteria", "Duke|modified Duke", "l:definite;possible;rejected;not met", "Duke", None),
    ("jones", "criteria", "person", "Jones criteria", "Jones", "crit", "Jones", None),
    ("mcdonald", "criteria", "person", "McDonald criteria", "McDonald", "crit", "McDonald", None),
    ("amsel", "criteria", "person", "Amsel criteria", "Amsel", "l:3/4;4/4;2/4;1/4", "Amsel", None),
    ("silverman", "score", "person", "Silverman-Andersen score", "Silverman|Silverman-Andersen", "i:0:6", "Silverman", "neonate"),
    ("friedman", "classification", "person", "Friedman tongue position", "Friedman|FTP", "l:I;II;III;IV", "Friedman", None),
    ("brodsky", "classification", "person", "Brodsky tonsil grade", "Brodsky", "l:1+;2+;3+;4+", "Brodsky", None),
    ("snellen", "test", "person", "Snellen acuity", "Snellen", "l:20/20;20/25;20/40;20/70;20/200", "Snellen", None),
    ("jaeger", "test", "person", "Jaeger near acuity", "Jaeger", "l:J1;J2;J3;J5", "Jaeger", None),
    ("amsler", "test", "person", "Amsler grid", "Amsler", "l:normal;distortion OD;distortion OS;nl OU", "Amsler", None),
    ("ishihara", "test", "person", "Ishihara plates", "Ishihara", "l:14/14;12/14;8/14;nl", "Ishihara", None),
    ("duke_dasi", "score", "person", "Duke Activity Status Index", "DASI|Duke activity index", "i:9:58", "Duke", None),
    ("waddell", "sign", "person", "Waddell signs", "Waddell", "l:0/5;1/5;2/5;3/5", "Waddell", None),
    # ---- named after people: signs ----
    ("murphy", "sign", "person", "Murphy sign", "Murphy|Murphy's", "sign", "Murphy", None),
    ("rovsing", "sign", "person", "Rovsing sign", "Rovsing", "sign", "Rovsing", None),
    ("kernig", "sign", "person", "Kernig sign", "Kernig", "sign", "Kernig", None),
    ("brudzinski", "sign", "person", "Brudzinski sign", "Brudzinski", "sign", "Brudzinski", None),
    ("babinski", "sign", "person", "Babinski sign", "Babinski", "l:neg;downgoing;upgoing L;upgoing R;absent bilat", "Babinski", None),
    ("homans", "sign", "person", "Homans sign", "Homans", "sign", "Homans", None),
    ("trousseau", "sign", "person", "Trousseau sign", "Trousseau", "sign", "Trousseau", None),
    ("chvostek", "sign", "person", "Chvostek sign", "Chvostek", "sign", "Chvostek", None),
    ("kehr", "sign", "person", "Kehr sign", "Kehr", "sign", "Kehr", None),
    ("cullen", "sign", "person", "Cullen sign", "Cullen", "sign", "Cullen", None),
    ("greyturner", "sign", "person", "Grey Turner sign", "Grey Turner", "sign", "Turner", None),
    ("hoffmann", "sign", "person", "Hoffmann sign", "Hoffmann|Hoffman", "sign", "Hoffmann", None),
    ("lhermitte", "sign", "person", "Lhermitte sign", "Lhermitte", "sign", "Lhermitte", None),
    ("tinel", "sign", "person", "Tinel sign", "Tinel|Tinel's", "sign", "Tinel", None),
    ("nikolsky", "sign", "person", "Nikolsky sign", "Nikolsky", "sign", "Nikolsky", None),
    ("auspitz", "sign", "person", "Auspitz sign", "Auspitz", "sign", "Auspitz", None),
    ("darier", "sign", "person", "Darier sign", "Darier", "sign", "Darier", None),
    ("levine", "sign", "person", "Levine sign", "Levine", "sign", "Levine", None),
    ("battle", "sign", "person", "Battle sign", "Battle", "sign", "Battle", None),
    ("hoover", "sign", "person", "Hoover sign", "Hoover", "sign", "Hoover", None),
    ("romberg", "test", "person", "Romberg test", "Romberg", "test", "Romberg", None),
    ("trendelenburg_sign", "sign", "person", "Trendelenburg sign", "Trendelenburg", "sign", "Trendelenburg", None),
    # ---- named after people: tests ----
    ("phalen", "test", "person", "Phalen test", "Phalen|Phalen's", "test", "Phalen", None),
    ("finkelstein", "test", "person", "Finkelstein test", "Finkelstein", "test", "Finkelstein", None),
    ("lachman", "test", "person", "Lachman test", "Lachman", "test", "Lachman", None),
    ("mcmurray", "test", "person", "McMurray test", "McMurray", "test", "McMurray", None),
    ("thompson", "test", "person", "Thompson test", "Thompson|Thompson squeeze", "test", "Thompson", None),
    ("spurling", "test", "person", "Spurling test", "Spurling", "test", "Spurling", None),
    ("hawkins", "test", "person", "Hawkins-Kennedy test", "Hawkins|Hawkins-Kennedy", "test", "Hawkins", None),
    ("neer", "test", "person", "Neer impingement test", "Neer", "test", "Neer", None),
    ("ober", "test", "person", "Ober test", "Ober", "test", "Ober", None),
    ("thomas", "test", "person", "Thomas test", "Thomas", "test", "Thomas", None),
    ("patrick", "test", "person", "Patrick test", "Patrick|FABER", "test", "Patrick", None),
    ("allen", "test", "person", "Allen test", "Allen|modified Allen", "l:normal;abnormal;neg;pos;nl R", "Allen", None),
    ("webertest", "test", "person", "Weber tuning fork test", "Weber test", "l:midline;lateralizes L;lateralizes R", "Weber", None),
    ("rinne", "test", "person", "Rinne test", "Rinne", "l:AC>BC bilat;BC>AC L;AC>BC;pos bilat;neg R", "Rinne", None),
    ("dixhallpike", "test", "person", "Dix-Hallpike test", "Dix-Hallpike|Hallpike", "test", "Hallpike", None),
    ("coombs", "test", "person", "direct Coombs test", "Coombs|DAT Coombs", "pn", "Coombs", None),
    ("mantoux", "test", "person", "Mantoux tuberculin skin test", "Mantoux|Mantoux PPD", "l:0 mm;4 mm;8 mm;12 mm;neg;16 mm", "Mantoux", None),
    ("schirmer", "test", "person", "Schirmer test", "Schirmer", "l:5 mm;8 mm;12 mm;15 mm;3 mm OS", "Schirmer", None),
    ("seidel", "test", "person", "Seidel test", "Seidel", "pn", "Seidel", None),
    ("kleihauer", "test", "person", "Kleihauer-Betke test", "Kleihauer-Betke|KB", "l:neg;pos;0.1%;0.4%", "Betke", "ob"),
    ("tzanck", "test", "person", "Tzanck smear", "Tzanck", "pn", "Tzanck", None),
    ("apley", "test", "person", "Apley grind test", "Apley|Apley grind", "test", "Apley", None),
    ("yergason", "test", "person", "Yergason test", "Yergason", "test", "Yergason", None),
    ("speed", "test", "person", "Speed test", "Speed's", "test", "Speed", None),
    ("gaenslen", "test", "person", "Gaenslen test", "Gaenslen", "test", "Gaenslen", None),
    ("adson", "test", "person", "Adson test", "Adson", "test", "Adson", None),
    ("ortolani", "test", "person", "Ortolani test", "Ortolani", "pn", "Ortolani", "neonate"),
    ("barlow", "test", "person", "Barlow test", "Barlow", "pn", "Barlow", "neonate"),
    # ---- named after people: manoeuvres ----
    ("epley", "manoeuvre", "person", "Epley manoeuvre", "Epley|Epley maneuver", "man", "Epley", None),
    ("heimlich", "manoeuvre", "person", "Heimlich manoeuvre", "Heimlich", "man", "Heimlich", None),
    ("valsalva", "manoeuvre", "person", "Valsalva manoeuvre", "Valsalva", "l:performed;murmur louder w/;no change w/;done", "Valsalva", None),
    ("mcroberts", "manoeuvre", "person", "McRoberts manoeuvre", "McRoberts", "man", "McRoberts", "ob"),
    ("woods", "manoeuvre", "person", "Woods screw manoeuvre", "Woods screw", "man", "Woods", "ob"),
    ("leopold", "manoeuvre", "person", "Leopold manoeuvres", "Leopold|Leopolds", "l:vertex;cephalic;breech;transverse;vtx", "Leopold", "ob"),
    ("sellick", "manoeuvre", "person", "Sellick manoeuvre", "Sellick", "man", "Sellick", None),
    ("semont", "manoeuvre", "person", "Semont manoeuvre", "Semont", "man", "Semont", None),
    ("brandt", "manoeuvre", "person", "Brandt-Daroff exercises", "Brandt-Daroff", "l:taught;reviewed;done tid;demonstrated", "Daroff", None),
    ("kocher", "manoeuvre", "person", "Kocher reduction", "Kocher|Kocher method", "man", "Kocher", None),
    # ---- named after people: devices ----
    ("foley", "device", "person", "Foley catheter", "Foley|16Fr Foley", "dev", "Foley", None),
    ("swanganz", "device", "person", "Swan-Ganz catheter", "Swan-Ganz|Swan", "dev", "Ganz", None),
    ("hickman", "device", "person", "Hickman line", "Hickman", "dev", "Hickman", None),
    ("groshong", "device", "person", "Groshong catheter", "Groshong", "dev", "Groshong", None),
    ("jacksonpratt", "device", "person", "Jackson-Pratt drain", "Jackson-Pratt|JP drain", "l:in place;20 mL serosang;draining;removed;30 cc", "Pratt", None),
    ("blake", "device", "person", "Blake drain", "Blake", "dev", "Blake", None),
    ("penrose", "device", "person", "Penrose drain", "Penrose", "dev", "Penrose", None),
    ("sengstaken", "device", "person", "Sengstaken-Blakemore tube", "Sengstaken-Blakemore|Blakemore", "dev", "Blakemore", None),
    ("miller", "device", "person", "Miller blade", "Miller|Miller 2", "l:used;grade I view;first pass;2", "Miller", None),
    ("macintosh", "device", "person", "Macintosh blade", "Macintosh|Mac 3", "l:used;grade I view;first pass;3;4", "Macintosh", None),
    ("guedel", "device", "person", "Guedel airway", "Guedel", "dev", "Guedel", None),
    ("yankauer", "device", "person", "Yankauer suction", "Yankauer", "l:at bedside;used;in reach", "Yankauer", None),
    ("holter", "device", "person", "Holter monitor", "Holter|48h Holter", "l:placed;returned;sinus, no pauses;rare PVCs;ordered", "Holter", None),
    ("pavlik", "device", "person", "Pavlik harness", "Pavlik", "dev", "Pavlik", "neonate"),
    ("thomassplint", "device", "person", "Thomas splint", "Thomas splint", "dev", "Thomas", None),
    ("buck", "device", "person", "Buck traction", "Buck's|Bucks", "l:in place;5 lb;applied;removed", "Buck", None),
    ("ilizarov", "device", "person", "Ilizarov frame", "Ilizarov", "dev", "Ilizarov", None),
    ("kirschner", "device", "person", "Kirschner wire", "Kirschner|K-wire", "dev", "Kirschner", None),
    ("steinmann", "device", "person", "Steinmann pin", "Steinmann", "dev", "Steinmann", None),
    ("tenckhoff", "device", "person", "Tenckhoff catheter", "Tenckhoff", "dev", "Tenckhoff", None),
    ("quinton", "device", "person", "Quinton catheter", "Quinton", "dev", "Quinton", None),
    ("broviac", "device", "person", "Broviac catheter", "Broviac", "dev", "Broviac", None),
    ("malecot", "device", "person", "Malecot catheter", "Malecot", "dev", "Malecot", None),
    # ---- named after places: scores, criteria, rules, classifications, devices ----
    ("ottawa_ankle", "criteria", "place", "Ottawa ankle rules", "Ottawa ankle|Ottawa ank", "crit", "Ottawa", None),
    ("ottawa_knee", "criteria", "place", "Ottawa knee rule", "Ottawa knee", "crit", "Ottawa", None),
    ("ottawa_sah", "criteria", "place", "Ottawa SAH rule", "Ottawa SAH", "crit", "Ottawa", None),
    ("chicago", "classification", "place", "Chicago Classification", "Chicago class|CCv4", "l:normal;IEGJOO;achalasia type II;IEM;absent contractility", "Chicago", None),
    ("boston_bowel", "scale", "place", "Boston Bowel Preparation Scale", "Boston prep|BBPS", "l:9;8 (3/3/2);6;7;5", "Boston", None),
    ("boston_caa", "criteria", "place", "Boston criteria", "Boston crit", "l:probable CAA;possible CAA;not met", "Boston", None),
    ("seattle_hf", "score", "place", "Seattle Heart Failure Model", "Seattle HF|SHFM", "l:1y survival 92%;1y survival 84%;1y survival 71%", "Seattle", None),
    ("sf_syncope", "criteria", "place", "San Francisco Syncope Rule", "SF syncope|SFSR", "crit", "San Francisco", None),
    ("framingham", "score", "place", "Framingham Risk Score", "Framingham|Framingham risk", "l:6%;9%;12%;15%;22%", "Framingham", None),
    ("gcs", "scale", "place", "Glasgow Coma Scale", "Glasgow|GCS", "i:3:15", "Glasgow", None),
    ("glasgow_blatchford", "score", "place", "Glasgow-Blatchford score", "Glasgow-Blatchford|GBS", "i:0:15", "Glasgow", None),
    ("geneva", "score", "place", "revised Geneva score", "Geneva", "i:0:12", "Geneva", None),
    ("padua", "score", "place", "Padua Prediction Score", "Padua", "i:0:9", "Padua", None),
    ("rochester", "criteria", "place", "Rochester criteria", "Rochester", "l:low risk;met;not met", "Rochester", "neonate"),
    ("philadelphia", "criteria", "place", "Philadelphia criteria", "Philly criteria", "l:low risk;met;not met", "Philadelphia", "neonate"),
    ("milan", "criteria", "place", "Milan criteria", "Milan", "l:within;outside;met;not met", "Milan", None),
    ("lille", "score", "place", "Lille score", "Lille", "l:0.21;0.45;0.38;0.12", "Lille", None),
    ("oakland", "score", "place", "Oakland score", "Oakland", "i:6:24", "Oakland", None),
    ("edinburgh", "scale", "place", "Edinburgh Postnatal Depression Scale", "Edinburgh|EPDS", "i:2:22", "Edinburgh", "ob"),
    ("edmonton_esas", "scale", "place", "Edmonton Symptom Assessment System", "Edmonton ESAS|ESAS", "l:pain 4;fatigue 6;nausea 2;dyspnea 5", "Edmonton", None),
    ("edmonton_frail", "scale", "place", "Edmonton Frail Scale", "Edmonton frail|EFS", "i:2:14", "Edmonton", None),
    ("kccq", "score", "place", "Kansas City Cardiomyopathy Questionnaire", "KCCQ|Kansas City", "i:22:96", "Kansas City", None),
    ("pittsburgh_psqi", "scale", "place", "Pittsburgh Sleep Quality Index", "Pittsburgh|PSQI", "i:3:17", "Pittsburgh", None),
    ("pittsburgh_knee", "criteria", "place", "Pittsburgh knee rules", "Pittsburgh knee", "crit", "Pittsburgh", None),
    ("moca", "test", "place", "Montreal Cognitive Assessment", "Montreal|MoCA", "l:26/30;23/30;19/30;28/30;24/30", "Montreal", None),
    ("montreal_ibd", "classification", "place", "Montreal classification", "Montreal class", "l:A2 L3 B1;E2;E3;A3 L1 B2", "Montreal", None),
    ("slums", "test", "place", "Saint Louis University Mental Status exam", "SLUMS|St Louis mental status", "l:27/30;22/30;18/30;25/30", "Saint Louis", None),
    ("rass", "scale", "place", "Richmond Agitation-Sedation Scale", "Richmond|RASS", "l:0;-1;-2;+1;-3;+2", "Richmond", None),
    ("la_esoph", "classification", "place", "Los Angeles classification", "LA grade|LA class", "l:A;B;C;D", "Los Angeles", None),
    ("lams", "scale", "place", "Los Angeles Motor Scale", "LAMS|LA motor", "i:0:5", "Los Angeles", None),
    ("cincinnati", "scale", "place", "Cincinnati Prehospital Stroke Scale", "Cincinnati|CPSS", "l:neg;pos;0/3;1/3;2/3", "Cincinnati", None),
    ("bristol", "scale", "place", "Bristol Stool Scale", "Bristol|Bristol stool", "l:type 1;type 4;type 6;4;6;7", "Bristol", None),
    ("oswestry", "score", "place", "Oswestry Disability Index", "Oswestry|ODI", "l:18%;24%;36%;42%;58%", "Oswestry", None),
    ("rome", "criteria", "place", "Rome IV criteria", "Rome IV|Rome", "crit", "Rome", None),
    ("amsterdam", "criteria", "place", "Amsterdam II criteria", "Amsterdam", "crit", "Amsterdam", None),
    ("bethesda", "classification", "place", "Bethesda system", "Bethesda", "l:II;III;IV;V;VI;cat II", "Bethesda", None),
    ("prague", "classification", "place", "Prague C&M criteria", "Prague|Prague C&M", "l:C2M4;C0M1;C3M5;C1M3", "Prague", None),
    ("tokyo", "criteria", "place", "Tokyo Guidelines", "Tokyo|TG18", "l:grade I;grade II;grade III;met", "Tokyo", None),
    ("atlanta", "classification", "place", "revised Atlanta classification", "Atlanta", "l:mild;moderately severe;severe", "Atlanta", None),
    ("vienna", "classification", "place", "Vienna classification", "Vienna", "l:category 3;category 4;category 1", "Vienna", None),
    ("berlin_ards", "criteria", "place", "Berlin definition", "Berlin|Berlin ARDS", "l:mild;moderate;severe;met;not met", "Berlin", None),
    ("berlin_q", "test", "place", "Berlin Questionnaire", "Berlin Q", "l:high risk;low risk", "Berlin", None),
    ("paris", "classification", "place", "Paris classification", "Paris", "l:0-Is;0-Ip;0-IIa;0-IIb", "Paris", None),
    ("lugano", "classification", "place", "Lugano classification", "Lugano", "l:stage II;stage III;stage IV;stage I", "Lugano", None),
    ("annarbor", "classification", "place", "Ann Arbor staging", "Ann Arbor", "l:IIA;IIIB;IVB;IA", "Ann Arbor", None),
    ("dallas", "criteria", "place", "Dallas criteria", "Dallas", "l:met;not met;borderline", "Dallas", None),
    ("denver", "test", "place", "Denver Developmental Screening Test", "Denver II|Denver", "l:normal;suspect;untestable;nl", "Denver", "peds"),
    ("stanford", "classification", "place", "Stanford classification", "Stanford", "l:type A;type B;A;B", "Stanford", None),
    ("epworth", "scale", "place", "Epworth Sleepiness Scale", "Epworth|ESS", "i:2:20", "Epworth", None),
    ("manchester", "scale", "place", "Manchester Triage System", "Manchester|MTS", "l:yellow;orange;green;red", "Manchester", None),
    ("toronto", "test", "place", "Toronto Bedside Swallowing Screening Test", "TOR-BSST|Toronto swallow", "l:pass;fail", "Toronto", None),
    ("nottingham", "classification", "place", "Nottingham grade", "Nottingham", "l:1;2;3;grade 2", "Nottingham", None),
    ("nyha", "classification", "place", "New York Heart Association class", "NYHA", "l:I;II;III;IV;II-III", "New York", None),
    ("sydney", "classification", "place", "updated Sydney system", "Sydney", "l:chronic gastritis;mild;moderate", "Sydney", None),
    ("philly_collar", "device", "place", "Philadelphia collar", "Philly collar|Philadelphia", "dev", "Philadelphia", None),
    ("milwaukee", "device", "place", "Milwaukee brace", "Milwaukee", "dev", "Milwaukee", None),
    ("boston_brace", "device", "place", "Boston brace", "Boston brace", "dev", "Boston", "peds"),
    ("miami_j", "device", "place", "Miami J collar", "Miami J", "dev", "Miami", None),
    ("aspen", "device", "place", "Aspen collar", "Aspen", "dev", "Aspen", None),
    ("denver_shunt", "device", "place", "Denver shunt", "Denver shunt", "dev", "Denver", None),
    ("charleston", "device", "place", "Charleston bending brace", "Charleston brace", "dev", "Charleston", "peds"),
    ("kingston", "classification", "place", "Kingston classification", "Kingston", "l:A;B;C", "Kingston", None),
    ("oxford_knee", "score", "place", "Oxford Knee Score", "Oxford knee|OKS", "i:12:46", "Oxford", None),
    ("oxford_hip", "score", "place", "Oxford Hip Score", "Oxford hip|OHS", "i:10:45", "Oxford", None),
    ("oxford_iga", "classification", "place", "Oxford classification", "Oxford MEST-C|MEST-C", "l:M1 E0 S1 T0 C0;M0 E0 S0 T1 C0;M1 E1 S1 T1 C1", "Oxford", None),
    ("banff", "classification", "place", "Banff classification", "Banff", "l:borderline;IA;IB;no rejection", "Banff", None),
    ("lyon", "criteria", "place", "Lyon consensus", "Lyon", "l:conclusive;inconclusive;not met", "Lyon", None),
    ("chapel_hill", "classification", "place", "Chapel Hill consensus nomenclature", "Chapel Hill", "l:small vessel;medium vessel;large vessel", "Chapel Hill", None),
    ("sapporo", "criteria", "place", "Sapporo criteria", "Sapporo", "crit", "Sapporo", None),
    ("kyoto", "classification", "place", "Kyoto classification of gastritis", "Kyoto", "i:0:8", "Kyoto", None),
    ("kiel", "classification", "place", "Kiel classification", "Kiel", "l:low grade;high grade", "Kiel", None),
    ("leipzig", "score", "place", "Leipzig score", "Leipzig", "i:0:8", "Leipzig", None),
    ("bclc", "classification", "place", "Barcelona Clinic Liver Cancer staging", "BCLC|Barcelona", "l:0;A;B;C;D", "Barcelona", None),
    ("vancouver_scar", "scale", "place", "Vancouver Scar Scale", "Vancouver|VSS", "i:1:13", "Vancouver", None),
    ("seattle_angina", "score", "place", "Seattle Angina Questionnaire", "Seattle angina|SAQ-7", "i:30:100:5", "Seattle", None),
    ("cssrs", "scale", "place", "Columbia Suicide Severity Rating Scale", "Columbia|C-SSRS", "l:neg;low risk;moderate risk;screen neg", "Columbia", None),
    ("leicester", "score", "place", "Leicester Cough Questionnaire", "Leicester|LCQ", "l:9.8;12.4;15.1;17.6", "Leicester", None),
    ("nijmegen", "score", "place", "Nijmegen Questionnaire", "Nijmegen", "i:8:38", "Nijmegen", None),
    ("wisconsin", "test", "place", "Wisconsin Card Sorting Test", "Wisconsin|WCST", "l:impaired;wnl;6 categories;3 categories", "Wisconsin", None),
    ("mmpi", "test", "place", "Minnesota Multiphasic Personality Inventory", "MMPI-2|Minnesota", "l:valid profile;invalid profile;completed", "Minnesota", None),
    ("mnsi", "test", "place", "Michigan Neuropathy Screening Instrument", "Michigan|MNSI", "l:2.5;3;4.5;1", "Michigan", None),
    ("boston_naming", "test", "place", "Boston Naming Test", "Boston naming|BNT", "l:52/60;44/60;57/60;38/60", "Boston", None),
    ("rotterdam", "criteria", "place", "Rotterdam criteria", "Rotterdam", "l:2/3;3/3;not met;met", "Rotterdam", None),
    ("brighton", "criteria", "place", "Brighton criteria", "Brighton", "crit", "Brighton", None),
    ("ghent", "criteria", "place", "revised Ghent nosology", "Ghent", "l:met;not met;systemic score 5;systemic score 8", "Ghent", None),
    ("toronto_neuro", "score", "place", "Toronto Clinical Neuropathy Score", "Toronto|TCNS", "i:2:16", "Toronto", None),
    # ---- generic ----
    ("curb65", "score", "generic", "CURB-65", "CURB-65|curb65|CURB", "i:0:5", None, None),
    ("perc", "criteria", "generic", "PERC rule", "PERC", "crit", None, None),
    ("chadsvasc", "score", "generic", "CHA2DS2-VASc", "CHA2DS2-VASc|chadsvasc", "i:0:8", None, None),
    ("hasbled", "score", "generic", "HAS-BLED", "HAS-BLED|hasbled", "i:0:5", None, None),
    ("timi", "score", "generic", "TIMI risk score", "TIMI", "i:0:7", None, None),
    ("grace", "score", "generic", "GRACE score", "GRACE", "i:82:196", None, None),
    ("heart", "score", "generic", "HEART score", "HEART", "i:0:8", None, None),
    ("nihss", "scale", "generic", "NIH Stroke Scale", "NIHSS", "i:0:22", None, None),
    ("qsofa", "score", "generic", "qSOFA", "qSOFA|qsofa", "i:0:3", None, None),
    ("sofa", "score", "generic", "SOFA score", "SOFA", "i:0:14", None, None),
    ("apache", "score", "generic", "APACHE II", "APACHE II|APACHE", "i:6:32", None, None),
    ("meld", "score", "generic", "MELD-Na", "MELD|MELD-Na", "i:8:34", None, None),
    ("phq9", "scale", "generic", "PHQ-9", "PHQ-9|PHQ9", "i:0:24", None, None),
    ("gad7", "scale", "generic", "GAD-7", "GAD-7|GAD7", "i:0:21", None, None),
    ("ciwa", "scale", "generic", "CIWA-Ar", "CIWA|CIWA-Ar", "i:0:24", None, None),
    ("cows", "scale", "generic", "Clinical Opiate Withdrawal Scale", "COWS", "i:0:24", None, None),
    ("auditc", "test", "generic", "AUDIT-C", "AUDIT-C|AUDITC", "i:0:12", None, None),
    ("cage", "test", "generic", "CAGE questionnaire", "CAGE", "l:0/4;1/4;2/4;3/4", None, None),
    ("four", "scale", "generic", "FOUR score", "FOUR", "i:4:16", None, None),
    ("camicu", "test", "generic", "CAM-ICU", "CAM-ICU|CAM", "l:neg;pos;negative;positive;UTA", None, None),
    ("asa", "classification", "generic", "ASA physical status", "ASA", "l:I;II;III;IV;2;3;3E", None, None),
    ("ecog", "scale", "generic", "ECOG performance status", "ECOG", "i:0:4", None, None),
    ("tug", "test", "generic", "Timed Up and Go", "TUG", "l:9 s;12 s;15 s;22 s;31 s", None, None),
    ("stopbang", "test", "generic", "STOP-Bang", "STOP-BANG|stopbang", "i:0:8", None, None),
    ("pasi", "score", "generic", "PASI", "PASI", "i:2:28", None, None),
    ("minicog", "test", "generic", "Mini-Cog", "Mini-Cog|minicog", "i:0:5", None, None),
    ("abcd2", "score", "generic", "ABCD2 score", "ABCD2", "i:0:7", None, None),
    ("pesi", "score", "generic", "sPESI", "sPESI|PESI", "i:0:6", None, None),
    ("news2", "score", "generic", "NEWS2", "NEWS|NEWS2", "i:0:12", None, None),
    ("mews", "score", "generic", "MEWS", "MEWS", "i:0:9", None, None),
    ("pews", "score", "generic", "PEWS", "PEWS", "i:0:7", None, "peds"),
    ("bisap", "score", "generic", "BISAP", "BISAP", "i:0:5", None, None),
    ("fib4", "score", "generic", "FIB-4", "FIB-4|FIB4", "l:0.9;1.3;1.8;2.67;3.4", None, None),
    ("apri", "score", "generic", "APRI", "APRI", "l:0.4;0.7;1.1;1.6", None, None),
    ("flacc", "scale", "generic", "FLACC", "FLACC", "i:0:10", None, "peds"),
    ("nrs", "scale", "generic", "numeric pain rating", "NRS|pain", "l:3/10;5/10;7/10;0/10;8/10", None, None),
    ("vas", "scale", "generic", "visual analogue scale", "VAS", "l:30 mm;55 mm;72 mm;4/10", None, None),
    ("hendrich", "scale", "generic", "Hendrich II Fall Risk Model", "Hendrich II|HFRM", "i:0:12", None, None),
    ("mmse", "test", "generic", "MMSE", "MMSE", "l:24/30;27/30;21/30;29/30", None, None),
    ("nexus", "criteria", "generic", "NEXUS criteria", "NEXUS", "crit", None, None),
    ("pecarn", "criteria", "generic", "PECARN rule", "PECARN", "l:low risk;neg;met;not met", None, "peds"),
    ("mrc", "scale", "generic", "MRC strength grade", "MRC", "l:5/5;4/5;4+/5;3/5", None, None),
    ("ttm", "classification", "generic", "TNM stage", "TNM", "l:T2N0M0;T1cN0;T3N1M0;T2N1", None, None),
    ("birads", "classification", "generic", "BI-RADS", "BI-RADS|BIRADS", "l:1;2;3;4A;0", None, None),
    ("lirads", "classification", "generic", "LI-RADS", "LI-RADS|LIRADS", "l:LR-3;LR-4;LR-5;LR-2", None, None),
    ("tirads", "classification", "generic", "TI-RADS", "TI-RADS|TIRADS", "l:TR3;TR4;TR2;TR5", None, None),
    ("pirads", "classification", "generic", "PI-RADS", "PI-RADS|PIRADS", "l:2;3;4;5", None, None),
    ("gold_copd", "classification", "generic", "GOLD grade", "GOLD", "l:1;2;3;2B;3E", None, None),
    ("ctas", "scale", "generic", "CTAS level", "CTAS|ESI", "i:1:5", None, None),
    ("sirs", "criteria", "generic", "SIRS criteria", "SIRS", "l:0/4;1/4;2/4;3/4;met", None, None),
    ("mna", "score", "generic", "Mini Nutritional Assessment", "MNA|MNA-SF", "i:4:14", None, None),
    ("must", "score", "generic", "MUST score", "MUST", "i:0:4", None, None),
    ("painad", "scale", "generic", "PAINAD", "PAINAD", "i:0:10", None, None),
    ("cfs", "scale", "generic", "Clinical Frailty Scale", "CFS", "i:1:8", None, None),
    ("gds", "scale", "generic", "Geriatric Depression Scale", "GDS|GDS-15", "i:0:12", None, None),
    ("psi", "score", "generic", "Pneumonia Severity Index", "PSI|PORT", "l:class II;class III;class IV;72;94", None, None),
    ("rox", "score", "generic", "ROX index", "ROX", "l:3.9;4.88;5.6;7.2", None, None),
    ("pasero", "scale", "person", "Pasero Opioid-Induced Sedation Scale", "Pasero|POSS", "l:S;1;2;3", "Pasero", None),
    ("bps", "scale", "generic", "Behavioral Pain Scale", "BPS", "i:3:12", None, None),
]
KINDS = ("score", "scale", "criteria", "classification", "sign", "test", "manoeuvre", "device")
ORIGIN_WEIGHTS = (("person", 0.5), ("place", 0.25), ("generic", 0.25))


@dataclass(frozen=True)
class Instrument:
    key: str
    kind: str
    origin: str
    full: str
    shorts: tuple
    values: tuple
    eponym: str | None
    pop: str | None


def _values(spec: str) -> tuple:
    if spec in NAMED_VALUES:
        return tuple(NAMED_VALUES[spec])
    if spec.startswith("l:"):
        return tuple(spec[2:].split(";"))
    parts = spec.split(":")
    lo, hi = int(parts[1]), int(parts[2])
    step = int(parts[3]) if len(parts) > 3 else 1
    return tuple(str(v) for v in range(lo, hi + 1, step))


# ---------------------------------------------------------------------------
# Exclusion lists (hash only).
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def exclusion_hashes_v2() -> frozenset:
    payload = EXCLUSIONS_V2.read_bytes()
    if hashlib.sha256(payload).hexdigest() != EXCLUSIONS_V2_SHA256:
        raise ValueError(f"{EXCLUSIONS_V2} does not match its pinned SHA-256")
    hashes = json.loads(payload)["sha256"]
    if not all(re.fullmatch(r"[0-9a-f]{64}", item) for item in hashes):
        raise ValueError("Exclusion list must contain SHA-256 hashes only")
    return frozenset(hashes)


def exclusion_lists() -> dict:
    return {"v1": exclusion_hashes(), "v2": exclusion_hashes_v2()}


def exclusion_hits(values, text: str, lists: dict | None = None) -> set:
    """Names of the lists hit by any inserted value or any 1-8 word sequence of ``text``."""
    lists = lists if lists is not None else exclusion_lists()
    digests = {exclusion_hash(v) for v in values if v and v.strip()}
    for value in values:
        if value and value.strip():
            digests.update(exclusion_hash(s) for s in word_sequences(value) if s)
    digests.update(exclusion_hash(s) for s in word_sequences(text) if s)
    return {name for name, hashes in lists.items() if digests & hashes}


def build_catalogue(lists: dict | None = None):
    """Return the usable instruments and counts of those dropped by pool or exclusion."""
    lists = lists if lists is not None else exclusion_lists()
    usable, dropped = [], {"heldout_pool": [], "v1": 0, "v2": 0}
    for key, kind, origin, full, shorts, spec, eponym, pop in CATALOGUE:
        inst = Instrument(key, kind, origin, full, tuple(shorts.split("|")), _values(spec), eponym, pop)
        if origin == "place" and place_value_pool(eponym) != "train":
            dropped["heldout_pool"].append(key)
            continue
        surfaces = (inst.full,) + inst.shorts
        hits = exclusion_hits(surfaces, " ".join(surfaces), lists)
        if hits:
            for name in hits:
                dropped[name] += 1
            continue
        usable.append(inst)
    return usable, dropped


# ---------------------------------------------------------------------------
# Invented people and identifiers.
# ---------------------------------------------------------------------------
GIVEN = ("Corin Ana Mira Tavi Joelle Rasmus Imani Dariel Wren Soren Priya Ilse Teodor Nadia Kofi Amara Lucan "
         "Bettina Oren Sefa Halle Juniper Marek Esme Tomas Yara Idris Lena Rafael Keiko Anouk Desmond Fenna Gideon "
         "Hana Ines Jarrah Kamal Liesel Matthias Noor Olek Pilar Quentin Rhea Sabine Tobiah Ulla Vikram Willa Xavi "
         "Yusuf Zora Abel Brigid Cato Dalia Emeric Faye Galen Hollis Ivo Jessa Keziah Lorne Mabry Nell Odile Pascal "
         "Rowan Saoirse Thea Ugo Vera Wynn Ysolde Zeke Ada Bram Celine Dov Elio Fern Gus Hester Ike June Kit Lars "
         "Maeve Nico Opal Petra Reuben Signe Tate Una Viggo Wanda Arlo Bea Cyrus Della Ezra Flora Greer Huck Isla "
         "Jonah Kaia Leif Mona Ned Orla Percy Ruth Stellan Tova Vance Adaeze Bao Chiara Dmitri Eamon Farida Goran "
         "Hiroko Ilya Jovana Kwame Lucia Mehmet Nkechi Oskar Parisa Ravi Selin Tariq Uma Wiremu Ximena Yosef Zainab").split()
SUR_HEAD = ("Brann Mar Ravel Quill Hal Ost Feld Thorn Wex Corb Dun Ell Garr Hask Ivers Kell Lom Mott Nesb Orr "
            "Pell Rask Selw Tarr Vand Wick Yarb Ald Bex Crom").split()
SUR_TAIL = ("ock wick ston ford ley man by well den holt ard er lin rigg sey ton more combe ham thorpe").split()
SURNAMES = tuple(sorted({h + t for h in SUR_HEAD for t in SUR_TAIL}))
ROLES_CLIN = ("Dr.", "Dr", "dr")
SEX = ("M", "F")
FILLERS = ("ddimer neg", "afebrile", "vss", "no xr", "trop neg x2", "ekg nsr", "ambulating", "tol po", "cxr clear",
           "lytes ok", "wbc 11.2", "hgb 9.8", "on RA", "sat 96% RA", "no focal deficits", "abd soft", "lungs cta",
           "cr 1.1", "lactate 1.4", "ua neg", "bcx pending", "pain controlled", "a&ox3", "skin intact")
PLANS = ("dc home", "admit obs", "induce", "ct head", "no imaging", "repeat in 4h", "ortho f/u", "start ppx",
         "cards consult", "pt/ot eval", "fall precautions", "reassess q shift", "npo after mn", "gi consult",
         "us ruq", "mri if no improvement", "dc w/ pcp f/u", "trend", "recheck in am", "continue current plan")
SYMPTOMS = ("calf pain", "ruq pain", "ankle injury", "fall at home", "sob", "chest pain", "headache", "dizziness",
            "wrist numbness", "knee pain", "back pain", "melena", "confusion", "fever", "neck pain")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
FACILITY_TAIL = ("Medical Center", "Hospital", "Clinic", "Health Center", "Rehabilitation Hospital")


class Seg:
    __slots__ = ("text", "label", "instr")

    def __init__(self, text, label=None, instr=None):
        self.text, self.label, self.instr = text, label, instr


def P(text):  # plain text
    return Seg(text)


def build_from_segments(segments, case: str = "normal"):
    """Join segments into text with exact spans and instrument-mention offsets."""
    pieces, spans, mentions, offset = [], [], [], 0
    for seg in segments:
        text = seg.text
        if case == "lower":
            text = text.lower()
        elif case == "upper":
            text = text.upper()
        if len(text) != len(seg.text):
            raise ValueError("case change altered length")
        if seg.label:
            spans.append({"start": offset, "end": offset + len(text), "label": seg.label, "text": text})
        if seg.instr:
            mentions.append({"start": offset, "end": offset + len(text), "key": seg.instr})
        pieces.append(text)
        offset += len(text)
    return "".join(pieces), spans, mentions


class Author:
    def __init__(self, seed: int = SEED, rows: int = ROWS, lists: dict | None = None):
        self.rng = random.Random(seed)
        self.seed, self.rows = seed, rows
        self.lists = lists if lists is not None else exclusion_lists()
        self.instruments, self.catalogue_dropped = build_catalogue(self.lists)
        self.by_key = {i.key: i for i in self.instruments}
        self.use = Counter()
        self.cap = max(1, int(CAP_SHARE * rows))
        self.given = [g for g in GIVEN if not exclusion_hits([g], g, self.lists)]
        self.surnames = [s for s in SURNAMES if not exclusion_hits([s], s, self.lists)]
        self.name_pool_dropped = {"given": len(GIVEN) - len(self.given), "surnames": len(SURNAMES) - len(self.surnames)}
        self.rejected = Counter()
        self.seen_texts = set()

    # -- instruments --------------------------------------------------------
    def row_pops(self):
        """One clinical population per row, so neonatal tools do not sit beside adult ones."""
        return self.rng.choice(((None,),) * 5 + ((None, "ob"),) * 2 + (("neonate", "peds"),) + (("peds", None),) * 2)

    def pick(self, n, pending, origin=None, need_eponym=False, pops=(None,), exclude=()):
        rng, chosen = self.rng, []
        for _ in range(n):
            want = origin or rng.choices([o for o, _ in ORIGIN_WEIGHTS], [w for _, w in ORIGIN_WEIGHTS])[0]
            pool = [i for i in self.instruments
                    if i.origin == want and i.pop in pops and i.key not in exclude
                    and i not in chosen and self.use[i.key] + pending[i.key] < self.cap
                    and (not need_eponym or (i.eponym and i.origin == "person"))]
            if not pool:
                pool = [i for i in self.instruments if i.pop in pops and i not in chosen and i.key not in exclude
                        and self.use[i.key] + pending[i.key] < self.cap
                        and (not need_eponym or (i.eponym and i.origin == "person"))]
            if not pool:  # population exhausted under the cap: widen to general instruments
                pool = [i for i in self.instruments if i.pop in pops + (None,) and i not in chosen
                        and i.key not in exclude and self.use[i.key] + pending[i.key] < self.cap
                        and (not need_eponym or (i.eponym and i.origin == "person"))]
            low = min(self.use[i.key] + pending[i.key] for i in pool)
            tier = [i for i in pool if self.use[i.key] + pending[i.key] <= low + 2]
            inst = rng.choice(tier)
            chosen.append(inst)
            pending[inst.key] += 1
        return chosen

    def surface(self, inst, style, need=None):
        options = list(inst.shorts) + [inst.full] if style in ("shorthand",) else (
            [inst.full] * 2 + list(inst.shorts) if style in ("structured", "list", "query") else [inst.full] * 3 + list(inst.shorts[:1]))
        if need:
            options = [o for o in options if need.casefold() in o.casefold()] or [inst.full]
        return self.rng.choice(options)

    def mention(self, inst, style, need=None):
        rng = self.rng
        surf = self.surface(inst, style, need)
        value = rng.choice(inst.values)
        if style == "shorthand":
            sep = rng.choice((" ", " ", "=", ": ", " ")) if inst.kind not in ("device",) else " "
            if inst.kind == "device" and rng.random() < 0.4:
                return [P(value + " "), Seg(surf, instr=inst.key)] if value in ("placed", "removed", "d/c'd") else [Seg(surf, instr=inst.key), P(" " + value)]
            return [Seg(surf, instr=inst.key), P(sep + value)]
        if style in ("structured",):
            return [Seg(surf, instr=inst.key)], value
        # narrative forms
        long = {"neg": "negative", "pos": "positive", "-": "negative", "+": "positive", "d/c'd": "removed"}.get(value, value)
        if inst.kind in ("score", "scale"):
            form = rng.choice(("the {s} was {v}", "a {s} of {v}", "{s} {v}"))
        elif inst.kind in ("sign", "test"):
            form = rng.choice(("{s} was {v}", "the {s} was {v}", "{s} {v}"))
        elif inst.kind == "criteria":
            form = rng.choice(("{s} {v}", "the {s} were {v}" if inst.full.endswith(("criteria", "rules")) else "the {s} was {v}"))
        elif inst.kind == "classification":
            form = rng.choice(("{s} {v}", "{s} was {v}", "consistent with {s} {v}"))
        elif inst.kind == "manoeuvre":
            form = rng.choice(("{s} {v}", "the {s} was {v}"))
        else:
            form = rng.choice(("{s} {v}", "the {s} is {v}"))
        before, after = form.split("{s}")
        return [P(before), Seg(surf, instr=inst.key), P(after.replace("{v}", long))]

    # -- people and identifiers ---------------------------------------------
    def person(self, surname=None):
        for _ in range(200):
            given = self.rng.choice(self.given)
            last = surname or self.rng.choice(self.surnames)
            full = f"{given} {last}"
            if identity_split(canonical_identity_group(full)) == "train":
                return given, last, full
        raise RuntimeError("no train-partition identity")

    def age(self, pop=None):
        rng = self.rng
        if pop == "ob":
            return str(rng.randint(18, 42))
        if pop == "peds":
            return str(rng.randint(3, 16))
        return str(rng.choice((rng.randint(19, 89), rng.randint(40, 88), rng.randint(60, 99))))

    def date(self):
        rng = self.rng
        y, m, d = rng.randint(2019, 2026), rng.randint(1, 12), rng.randint(1, 28)
        return rng.choice((f"{m:02d}/{d:02d}/{y}", f"{m}/{d}/{y % 100:02d}", f"{y}-{m:02d}-{d:02d}",
                           f"{MONTHS[m - 1]} {d} {y}", f"{m}/{d}", f"{d} {MONTHS[m - 1]} {y}"))

    def dob(self, age):
        rng = self.rng
        y = 2026 - int(age) - rng.randint(0, 1)
        m, d = rng.randint(1, 12), rng.randint(1, 28)
        return rng.choice((f"{m:02d}/{d:02d}/{y}", f"{y}-{m:02d}-{d:02d}", f"{MONTHS[m - 1]} {d}, {y}"))

    def mrn(self):
        return self.rng.choice((f"E24-{self.rng.randint(0, 9999999):07d}", f"{self.rng.randint(10000000, 99999999)}"))

    def csn(self):
        return f"7{self.rng.randint(0, 999999999):09d}"

    def order_no(self):
        return f"ORD-{self.rng.randint(0, 999999):06d}"

    def account(self):
        return f"24{self.rng.randint(0, 99999999):08d}"

    def phone(self):
        return self.rng.choice(("({a}) 555-01{b:02d}", "{a}-555-01{b:02d}", "{a}.555.01{b:02d}")).format(
            a=self.rng.randint(201, 989), b=self.rng.randint(0, 99))

    def facility(self):
        return f"{self.rng.choice(self.surnames)} {self.rng.choice(FACILITY_TAIL)}"

    def age_seg(self, age, sex):
        form = self.rng.choice(("{a}{s}", "{a}{s}", "{a} y/o {s}", "{a}yo {s}", "{a} yo {sl}"))
        before, after = form.split("{a}")
        sl = "male" if sex == "M" else "female"
        return [P(before), Seg(age, "AGE"), P(after.format(s=sex, sl=sl))]

    def clinician_sig(self, people):
        given, last, full = self.person()
        people.append(full)
        form = self.rng.choice(("-- {n}, RN", "- dr {n}", "/{n} MD", "-{n}, NP", "signed: {n}, PA-C"))
        name = self.rng.choice((f"{given[0]}. {last}", last, f"{given} {last}"))
        before, after = form.split("{n}")
        return [P(before), Seg(name, "NAME"), P(after)]

    # -- layouts ------------------------------------------------------------
    def instruments_line(self, insts, style, joiner):
        segs = []
        for index, inst in enumerate(insts):
            if index:
                segs.append(P(joiner))
            segs.extend(self.mention(inst, style))
        return segs

    def fields_block(self, insts, layout):
        rng, segs = self.rng, []
        rows = [self.mention(i, "structured") for i in insts]
        if layout == "fld_kv":
            for k, (name, value) in enumerate(rows):
                segs += ([P("\n")] if k else []) + name + [P(rng.choice((": ", ": ", " - ", ":\t")) + value)]
        elif layout == "fld_flowsheet":
            for k, (name, value) in enumerate(rows):
                pad = "." * max(3, 34 - len(name[0].text))
                segs += ([P("\n")] if k else []) + name + [P(f" {pad} {value}")]
        elif layout == "fld_pipe":
            segs.append(P(rng.choice(("| Assessment | Result |\n|---|---|", "| Tool | Score |\n|---|---|"))))
            for name, value in rows:
                segs += [P("\n| ")] + name + [P(f" | {value} |")]
        elif layout == "fld_checklist":
            for k, (name, value) in enumerate(rows):
                segs += ([P("\n")] if k else []) + [P(rng.choice(("[x] ", "[X] ", "( ) ", "[ ] ")))] + name + [P(f" {value}")]
        elif layout == "fld_csv":
            segs.append(P(rng.choice(("assessment,result", "tool,value", "measure,score"))))
            for name, value in rows:
                segs += [P("\n")] + name + [P(f",{value}")]
        else:  # fld_section
            segs.append(P(rng.choice(("ASSESSMENT SCORES", "Screening tools", "NURSING ASSESSMENT", "Exam findings"))))
            for name, value in rows:
                segs += [P("\n")] + name + [P(f": {value}")]
            segs.append(P(rng.choice(("\nReassess: q shift", "\nNext due: 0800", "\nReviewed with charge RN", ""))))
        return segs

    def row_shorthand(self, layout, pending):
        rng = self.rng
        insts = self.pick(rng.randint(2, 4), pending, pops=self.row_pops())
        filler, plan = rng.choice(FILLERS), rng.choice(PLANS)
        if layout == "sh_triage":
            return [P(rng.choice(SYMPTOMS) + ". ")] + self.instruments_line(insts, "shorthand", ", ") + [P(f", {filler} -> {plan}")]
        if layout == "sh_progress":
            return [P(f"S: {rng.choice(SYMPTOMS)}\nO: {filler}, ")] + self.instruments_line(insts, "shorthand", ", ") + [P(f"\nA/P: {plan}")]
        if layout == "sh_overnight":
            return [P(rng.choice(("overnight: ", "o/n: ", "events: ")))] + self.instruments_line(insts, "shorthand", "; ") + [P(f"; {plan}")]
        if layout == "sh_slash":
            return self.instruments_line(insts, "shorthand", " / ") + [P(f" / {filler}")]
        if layout == "sh_bullets":
            return [P("- ")] + self.instruments_line(insts, "shorthand", "\n- ") + [P(f"\n- {filler}\n- {plan}")]
        if layout == "sh_page":
            return [P(rng.choice(("pls review: ", "fyi ", "page: ", "msg to team: ")))] + self.instruments_line(insts, "shorthand", ", ") + [P(f". {plan}?")]
        return self.instruments_line(insts, "shorthand", "; ") + [P(f"; {filler}; {plan}")]

    def row_fields(self, layout, pending):
        insts = self.pick(self.rng.randint(2, 5), pending, pops=self.row_pops())
        return self.fields_block(insts, layout)

    def header_person(self, people, pop=None):
        rng = self.rng
        given, last, full = self.person()
        people.append(full)
        sex = "F" if pop == "ob" else rng.choice(SEX)
        age = self.age(pop)
        name = rng.choice((f"{last}, {given}", f"{given} {last}", f"{last.upper()}, {given.upper()}"))
        return given, last, name, age, sex

    def row_with_phi(self, layout, pending, people):
        rng = self.rng
        pop = rng.choice((None, None, None, "ob"))
        insts = self.pick(rng.randint(2, 4), pending, pops=(None, pop))
        given, last, name, age, sex = self.header_person(people, pop)
        if layout == "phi_sh_header":
            segs = [Seg(name, "NAME"), P(" ")] + self.age_seg(age, sex)
            if rng.random() < 0.7:
                segs += [P(rng.choice((" MRN ", " mrn# ", " #"))), Seg(self.mrn(), "MRN")]
            segs += [P(rng.choice((" ", ". ", " - ")))] + self.instruments_line(insts, "shorthand", ", ")
            if rng.random() < 0.4:
                segs += [P(rng.choice((". mom w/ MI at ", ". father dx at ", ". sister, "))), Seg(str(rng.randint(41, 88)), "AGE")]
            segs.append(P(f". {rng.choice(PLANS)}"))
            return segs
        if layout == "phi_sh_note":
            segs = [Seg(self.date(), "DATE"), P(rng.choice((" ", " - ", ": "))), P(rng.choice(("pt ", "", "re: "))),
                    Seg(name, "NAME"), P(", ")] + self.age_seg(age, sex) + [P(f", {rng.choice(SYMPTOMS)}. ")]
            segs += self.instruments_line(insts, "shorthand", ", ") + [P(f", {rng.choice(FILLERS)}. {rng.choice(PLANS)} ")]
            segs += self.clinician_sig(people)
            if rng.random() < 0.3:
                segs += [P(rng.choice((" cb ", " call "))), Seg(self.phone(), "PHONE")]
            return segs
        if layout == "phi_fld_header":
            segs = [P(rng.choice(("Patient: ", "Name: ", "PATIENT NAME: "))), Seg(name, "NAME"),
                    P("\nDOB: "), Seg(self.dob(age), "DATE"), P("\nAge: "), Seg(age, "AGE"),
                    P(rng.choice(("\nMRN: ", "\nMRN#: "))), Seg(self.mrn(), "MRN")]
            if rng.random() < 0.4:
                segs += [P("\nAcct: "), Seg(self.account(), "ACCOUNT")]
            if rng.random() < 0.3:
                segs += [P("\nFacility: "), Seg(self.facility(), "FACILITY")]
            segs += [P("\nAssessed: "), Seg(self.date(), "DATE"), P("\n")]
            return segs + self.fields_block(insts, rng.choice(("fld_kv", "fld_flowsheet", "fld_section")))
        # phi_fld_order
        segs = [P("Order: "), Seg(self.order_no(), "ID"), P("\nEncounter CSN: "), Seg(self.csn(), "ID"),
                P("\nPatient: "), Seg(name, "NAME"), P(" ("), Seg(age, "AGE"), P(f"{sex})\nOrdered by: ")]
        segs += self.clinician_sig(people)[1:2]
        segs += [P("\nDate: "), Seg(self.date(), "DATE"), P("\nIndication: ")]
        segs += self.instruments_line(insts, "shorthand", ", ")
        return segs

    def row_same_surname(self, layout, pending, people):
        rng = self.rng
        inst = self.pick(1, pending, need_eponym=True, pops=(None, "ob"))[0]
        others = self.pick(rng.randint(0, 2), pending, pops=(None, inst.pop), exclude={inst.key})
        given, last, full = self.person(surname=inst.eponym)
        people.append(full)
        role = rng.choice(("clinician", "patient", "relative"))
        if layout == "ss_shorthand":
            mention = self.mention(inst, "shorthand", need=inst.eponym)
            rest = [P(", ")] + self.instruments_line(others, "shorthand", ", ") if others else []
            if role == "patient":
                pname = rng.choice((f"{last}, {given}", f"{given} {last}"))
                age, sex = self.age(inst.pop), "F" if inst.pop == "ob" else rng.choice(SEX)
                return [P(rng.choice(("pt ", "", "re "))), Seg(pname, "NAME"), P(" ")] + self.age_seg(age, sex) + [P(": ")] + mention + rest
            if role == "clinician":
                return mention + rest + [P(rng.choice(("; d/w dr ", ", per Dr. ", " - called dr ", "; seen by "))), Seg(rng.choice((last, f"{given[0]}. {last}", f"{given} {last}")), "NAME")]
            pname = self.person()
            people.append(pname[2])
            return ([Seg(f"{pname[1]}, {pname[0]}", "NAME"), P(": ")] + mention + rest
                    + [P(rng.choice(("; dtr ", "; son ", "; wife ", "; husband "))), Seg(f"{given} {last}", "NAME"),
                       P(rng.choice((" at bedside", " updated", " is hcp")))])
        if layout == "ss_fields":
            name_seg, value = self.mention(inst, "structured", need=inst.eponym)
            if role == "patient":
                segs = [P("Patient: "), Seg(rng.choice((f"{given} {last}", f"{last}, {given}")), "NAME"), P("\n")]
                tail = []
            else:
                pname = self.person()
                people.append(pname[2])
                segs = [P("Patient: "), Seg(f"{pname[1]}, {pname[0]}", "NAME"), P("\n")]
                label = "Scored by: " if role == "clinician" else rng.choice(("Contact: ", "Emergency contact: "))
                tail = [P("\n" + label), Seg(rng.choice((f"{given[0]}. {last}", f"{given} {last}")), "NAME")]
            segs += name_seg + [P(": " + value)]
            for other in others:
                o_name, o_val = self.mention(other, "structured")
                segs += [P("\n")] + o_name + [P(": " + o_val)]
            return segs + tail
        mention = self.mention(inst, "narrative", need=inst.eponym)
        if role == "clinician":
            return [P(rng.choice(("Dr. ", "Dr ", ""))), Seg(rng.choice((last, f"{given} {last}")), "NAME"),
                    P(rng.choice((" reviewed the case; ", " examined the patient and noted ", " documented that ")))] + mention + [P(".")]
        if role == "patient":
            return [Seg(f"{given} {last}", "NAME"), P(rng.choice((" was seen today. On exam, ", " returns for follow-up; ", " presented with "
                                                                 + rng.choice(SYMPTOMS) + ", and ")))] + mention + [P(".")]
        pname = self.person()
        people.append(pname[2])
        return ([P("Seen with "), Seg(f"{given} {last}", "NAME"), P(rng.choice((", her daughter. ", ", his son. ", ", the spouse. "))),
                 Seg(f"{pname[0]} {pname[1]}", "NAME"), P(" was assessed; ")] + mention + [P(".")])

    def row_narrative(self, layout, pending, people, with_phi):
        rng = self.rng
        insts = self.pick(rng.randint(1, 3), pending, pops=self.row_pops() if not with_phi else (None,))
        intro = []
        if with_phi:
            given, last, name, age, sex = self.header_person(people)
            article = "an" if age.startswith("8") or age in ("11", "18") else "a"
            intro = [Seg(f"{given} {last}", "NAME"), P(rng.choice((f" is {article} ", f", {article} "))), Seg(age, "AGE"),
                     P(rng.choice(("-year-old man", "-year-old woman", " year old patient", "-year-old"))),
                     P(rng.choice((" seen on ", " evaluated on "))), Seg(self.date(), "DATE"), P(". ")]
        if layout == "nar_sentence":
            body = []
            for k, inst in enumerate(insts):
                part = self.mention(inst, "narrative")
                part[0] = P(part[0].text.capitalize()) if part[0].text else part[0]
                body += ([P(" ")] if k else []) + part + [P(".")]
            return intro + body
        if layout == "nar_query":
            q = rng.choice(("can you calc {s} for this pt?", "is the {s} done?", "what was the {s}?",
                            "should we repeat the {s}?", "does {s} apply here?"))
            before, after = q.split("{s}")
            inst = insts[0]
            segs = intro + [P(before), Seg(self.surface(inst, "query"), instr=inst.key), P(after)]
            if len(insts) > 1:
                segs += [P(" also ")] + self.mention(insts[1], "shorthand") + [P(" earlier")]
            return segs
        if layout == "nar_list":
            segs = intro + [P(rng.choice(("Findings:\n", "Exam:\n", "Scores today:\n")))]
            bullet = rng.choice(("- ", "* ", "{n}. "))
            for k, inst in enumerate(insts):
                segs += ([P("\n")] if k else []) + [P(bullet.format(n=k + 1))] + self.mention(inst, "shorthand")
            return segs
        segs = intro + [P("Assessment: ")]
        for k, inst in enumerate(insts):
            segs += ([P("; ")] if k else []) + self.mention(inst, "narrative")
        tail = rng.choice((". Plan: {p}.", ". Plan: {p}; screen from age 50 per guideline.", ". Plan: {p}.",
                           ". {p} as discussed."))
        return segs + [P(tail.format(p=rng.choice(PLANS)))]

    # -- driver ---------------------------------------------------------------
    def schedule(self):
        rng, plan = self.rng, []
        scale = self.rows / ROWS
        for family, layouts in LAYOUT_PLAN.items():
            items = []
            for layout, count in layouts.items():
                items += [layout] * max(1, round(count * scale))
            n = len(items)
            cases = []
            for case, share in CASE_SHARES:
                cases += [case] * round(share * n)
            cases = (cases + ["normal"] * n)[:n]
            rng.shuffle(items)
            rng.shuffle(cases)
            for index, (layout, case) in enumerate(zip(items, cases)):
                with_phi = family == "tk24_instruments_narrative" and index % 2 == 0
                plan.append((family, layout, case, with_phi))
        return plan

    def generate(self):
        out = []
        for family, layout, case, with_phi in self.schedule():
            for _attempt in range(200):
                pending, people = Counter(), []
                if family == "tk24_instruments_shorthand":
                    segs = self.row_shorthand(layout, pending)
                elif family == "tk24_instruments_fields":
                    segs = self.row_fields(layout, pending)
                elif family == "tk24_instruments_with_phi":
                    segs = self.row_with_phi(layout, pending, people)
                elif family == "tk24_same_surname":
                    segs = self.row_same_surname(layout, pending, people)
                else:
                    segs = self.row_narrative(layout, pending, people, with_phi)
                text, spans, mentions = build_from_segments(segs, case)
                if text in self.seen_texts:
                    self.rejected["duplicate_text"] += 1
                    continue
                values = [s["text"] for s in spans] + [text[m["start"]:m["end"]] for m in mentions]
                hits = exclusion_hits(values, text, self.lists)
                if hits:
                    self.rejected["exclusion_" + "_and_".join(sorted(hits))] += 1
                    for name in hits:
                        self.rejected[f"rows_hitting_{name}"] += 1
                    continue
                if annotation_errors(text, spans):
                    raise ValueError(f"structural error: {annotation_errors(text, spans)}")
                break
            else:
                raise RuntimeError(f"could not author a row for {family}/{layout}")
            self.seen_texts.add(text)
            self.use.update(pending)
            out.append(self.make_row(len(out), family, layout, case, text, spans, mentions, people))
        return out

    def make_row(self, index, family, layout, case, text, spans, mentions, people):
        rid = f"tk24-{index:05d}"
        groups = list(dict.fromkeys(canonical_identity_group(p) for p in people))
        for group in groups:
            assert identity_split(group) == "train"
        origins = sorted({self.by_key[m["key"]].origin for m in mentions})
        kinds = sorted({self.by_key[m["key"]].kind for m in mentions})
        phenomena = ["eponym_instrument", f"layout_{LAYOUT_STYLE[layout]}", f"case_{case}"]
        phenomena += [f"{o}_named_instrument" if o != "generic" else "generic_instrument" for o in origins]
        if family == "tk24_same_surname":
            phenomena.append("same_surname_person")
        phenomena.append("phi_nearby" if spans else "phi_free")
        if any(s["label"] == "AGE" for s in spans):
            phenomena.append("all_ages")
        return {
            "id": rid, "text": text, "spans": spans, "split": "train", "suite": SUITE,
            "annotation_policy": POLICY_VERSION, "cohort": "realistic", "family": family,
            "template_family": family, "layout": layout, "layout_style": LAYOUT_STYLE[layout], "case": case,
            "phenomena": phenomena,
            "identity_group": groups[0] if groups else f"tk24:phi_free:{rid}",
            "identity_groups": groups, "pair_id": f"tk24:{rid}",
            "instrument_mentions": mentions, "instrument_kinds": kinds, "instrument_origins": origins,
        }


def manifest_for(rows, payload: bytes, author: Author, output: Path) -> dict:
    support = Counter(s["label"] for r in rows for s in r["spans"])
    label_support = {label: support[label] for label in ENTITY_TYPES}
    used = Counter(m["key"] for r in rows for m in r["instrument_mentions"])
    rows_per_inst = Counter(k for r in rows for k in {m["key"] for m in r["instrument_mentions"]})
    distinct = {author.by_key[k] for k in used}
    script = Path(__file__).resolve()
    return {
        "bank": BANK, "path": str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),
        "annotation_policy": POLICY_VERSION, "addenda": [ADDENDUM, "E1-2026-10-08"], "split": "train",
        "suite": SUITE, "independent_final": False, "rows": len(rows), "documents": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "label_support": label_support, "entity_support": label_support, "label_counts": label_support,
        "cohort_counts": dict(Counter(r["cohort"] for r in rows)),
        "family_counts": dict(Counter(r["family"] for r in rows)),
        "layout_counts": dict(Counter(r["layout"] for r in rows)),
        "layout_style_counts": dict(Counter(r["layout_style"] for r in rows)),
        "case_counts": dict(Counter(r["case"] for r in rows)),
        "phi_free_rows": sum(1 for r in rows if not r["spans"]),
        "instruments": {
            "distinct": len(distinct),
            "by_origin": dict(Counter(i.origin for i in distinct)),
            "by_kind": dict(Counter(i.kind for i in distinct)),
            "max_rows_single_instrument": max(rows_per_inst.values()),
            "max_row_share_single_instrument": round(max(rows_per_inst.values()) / len(rows), 4),
            "catalogue_size": len(CATALOGUE),
            "dropped_heldout_pool": len(author.catalogue_dropped["heldout_pool"]),
            "dropped_exclusion_v1": author.catalogue_dropped["v1"],
            "dropped_exclusion_v2": author.catalogue_dropped["v2"],
        },
        "exclusion_filter": {
            "lists": {"v1": {"path": "data/places_exclusions.sha256.json", "hashes": len(author.lists["v1"])},
                      "v2": {"path": "data/places_exclusions_v2.sha256.json", "sha256": EXCLUSIONS_V2_SHA256,
                             "hashes": len(author.lists["v2"])}},
            "rule": "reject a row if any inserted value or any 1-8 word sequence of its text hashes onto a list",
            "rejected_rows": dict(sorted(author.rejected.items())),
            "name_pool_dropped": author.name_pool_dropped,
        },
        "identifier_schemes": {"MRN": MRN_SCHEME, "ID": ID_SCHEME, "ACCOUNT": ACCOUNT_SCHEME, "PHONE": PHONE_SCHEME},
        "place_values": "place names only inside instruments, listed from general knowledge; train pool only (place_value_pool)",
        "author_provenance": {
            "generator": str(script.relative_to(ROOT)),
            "generator_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "seed": author.seed,
            "brief": "docs/instrument-eponyms-bank-brief.md",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    author = Author(args.seed, args.rows)
    rows = author.generate()
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    manifest = manifest_for(rows, payload, author, args.output.resolve())
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "label_support", "family_counts", "case_counts")}, indent=1))
    print(json.dumps(manifest["instruments"], indent=1))
    print(json.dumps(manifest["exclusion_filter"]["rejected_rows"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
