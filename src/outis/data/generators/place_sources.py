"""Pinned US federal public-domain sources of real place and care-site names.

Permitted by the places addendum (docs/annotation-policy-v2-places.md):
real public place names from the US Census Bureau Gazetteer Files and real
public care-site names from the CMS Hospital General Information file, plus a
curated, committed list of common care-site abbreviations and short names with
a recorded public source per entry. Neither federal file contains patient
information. Values are surface strings for synthetic training data only; the
detector contains no place dictionary.

Downloads go to a gitignored cache (``.cache/place_sources`` or
``$OUTIS_PLACE_CACHE``) and are verified against the pinned SHA-256
before any byte is parsed::

    .venv/bin/python -m outis.data.generators.place_sources fetch
    .venv/bin/python -m outis.data.generators.place_sources verify
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import urllib.request
import zipfile
from outis.paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
CURATED_CARE_SITES = ROOT / "data" / "care_site_names.json"
SOURCES = {
    "census_gazetteer_places_2024": {
        "url": "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_place_national.zip",
        "sha256": "cf262fc92b2326f7a8c62a89d156a60eb17d64d6d35f7a62310c43bb08972c06",
        "member": "2024_Gaz_place_national.txt",
        "publisher": "US Census Bureau",
        "landing_page": "https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html",
        "license": "US federal government work; public domain (17 U.S.C. 105)",
    },
    "census_gazetteer_counties_2024": {
        "url": "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_counties_national.zip",
        "sha256": "3c337402b5c6e8d5aa26b4278ccf4edc8989f2683765b3ffbf22296cdb2df3a0",
        "member": "2024_Gaz_counties_national.txt",
        "publisher": "US Census Bureau",
        "landing_page": "https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html",
        "license": "US federal government work; public domain (17 U.S.C. 105)",
    },
    "cms_hospital_general_information": {
        "url": "https://data.cms.gov/provider-data/sites/default/files/resources/893c372430d9d71a1c52737d01239d47_1785189955/Hospital_General_Information.csv",
        "sha256": "f874f09fef895a1ccf5bb7392dcbb2be05c9860339261b093fe00f0c1b013480",
        "member": None,
        "publisher": "Centers for Medicare & Medicaid Services",
        "landing_page": "https://data.cms.gov/provider-data/dataset/xubh-q36u",
        "license": "US federal government work; public domain (17 U.S.C. 105)",
        "retrieved": "2026-10-04",
    },
}
# Only these columns are read; CMS addresses and telephone numbers are ignored.
CMS_COLUMNS = ("Facility ID", "Facility Name", "City/Town", "State", "Hospital Type")
_LSAD = re.compile(
    r"\s+(?:city and borough|consolidated government|metropolitan government|unified government|"
    r"urban county|city|town|village|CDP|borough|municipality|plantation|township|corporation)$"
)


def cache_dir() -> Path:
    return Path(os.environ.get("OUTIS_PLACE_CACHE", ROOT / ".cache" / "place_sources"))


def _path(name: str) -> Path:
    return cache_dir() / SOURCES[name]["url"].rsplit("/", 1)[1]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(names=None) -> dict:
    """Download missing sources into the cache and verify every pin."""
    result = {}
    for name in names or SOURCES:
        path = _path(name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(SOURCES[name]["url"], timeout=120) as response:
                payload = response.read()
            if sha256(payload) != SOURCES[name]["sha256"]:
                raise ValueError(f"Downloaded {name} does not match its pinned SHA-256; refusing to cache it")
            partial = path.with_suffix(path.suffix + ".partial")
            partial.write_bytes(payload)
            partial.replace(path)
        result[name] = {"path": str(path), "sha256": sha256(read_source(name, raw=True))}
    return result


def read_source(name: str, *, raw: bool = False) -> bytes:
    """Return verified source bytes (the zip member for Gazetteer archives)."""
    path = _path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m outis.data.generators.place_sources fetch`")
    payload = path.read_bytes()
    if sha256(payload) != SOURCES[name]["sha256"]:
        raise ValueError(f"{path} does not match the pinned SHA-256 of {name}")
    if raw or SOURCES[name]["member"] is None:
        return payload
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return archive.read(SOURCES[name]["member"])


def sources_available() -> bool:
    return all(_path(name).exists() for name in SOURCES)


def _tsv(name: str) -> list[dict]:
    text = read_source(name).decode("utf-8")
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    header = [item.strip() for item in next(reader)]
    return [dict(zip(header, (item.strip() for item in row))) for row in reader if row]


def gazetteer_places() -> list[dict]:
    """Census places with the legal/statistical area suffix removed.

    Puerto Rico and composite names such as consolidated ``(balance)`` areas
    are skipped because their Gazetteer names are not ordinary surface forms.
    """
    places = []
    for row in _tsv("census_gazetteer_places_2024"):
        raw = row["NAME"]
        if row["USPS"] == "PR" or any(mark in raw for mark in "(/,") or any(ch.isdigit() for ch in raw):
            continue
        name = _LSAD.sub("", raw)
        if name == raw or not name or any(word[:1].islower() and word not in {"de", "la", "del", "los", "las", "du", "of", "on", "the", "and", "y"} for word in name.split()):
            continue
        places.append({"name": name, "state": row["USPS"], "geoid": row["GEOID"], "lsad": raw[len(name):].strip()})
    return places


def gazetteer_counties() -> list[dict]:
    return [{"name": row["NAME"], "state": row["USPS"], "geoid": row["GEOID"]}
            for row in _tsv("census_gazetteer_counties_2024")
            if row["USPS"] != "PR" and re.fullmatch(r"[A-Z][A-Za-z.' -]+ (?:County|Parish)", row["NAME"])]


def cms_hospitals() -> list[dict]:
    text = read_source("cms_hospital_general_information").decode("utf-8-sig")
    return [{key: row[key].strip() for key in CMS_COLUMNS} for row in csv.DictReader(io.StringIO(text))]


def curated_care_sites() -> dict:
    payload = CURATED_CARE_SITES.read_bytes()
    data = json.loads(payload)
    data["sha256"] = sha256(payload)
    return data


def pins() -> dict:
    """Manifest-ready description of every pinned input."""
    pinned = {name: {key: value for key, value in source.items()} for name, source in SOURCES.items()}
    curated = curated_care_sites()
    pinned["curated_care_site_names"] = {
        "path": str(CURATED_CARE_SITES.relative_to(ROOT)), "sha256": curated["sha256"],
        "entries": len(curated["entries"]), "license": curated["license"],
    }
    return pinned


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fetch", "verify"))
    args = parser.parse_args(argv)
    if args.command == "fetch":
        print(json.dumps(fetch(), indent=2))
    else:
        print(json.dumps({name: {"sha256": sha256(read_source(name, raw=True)), "pinned": True} for name in SOURCES}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
