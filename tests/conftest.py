"""Skip tests whose declared external fixtures are absent.

Training corpora, their manifests and the blind-bank exclusion lists live in
the project's private data repository, and the public place sources are
downloaded on demand (``python -m outis.data.generators.place_sources
fetch``). A test that needs one of those declares it with
``@pytest.mark.requires_data("name", ..., place_sources=True)``, where each
name is a path relative to ``data/``, and is skipped before it runs when a
declared fixture is missing. Any other missing file fails the test as usual.
"""

from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_data(*names, place_sources=False): skip unless these data/ "
        "files (and, if asked, the pinned place sources) are present",
    )


def pytest_runtest_setup(item):
    for marker in item.iter_markers("requires_data"):
        for name in marker.args:
            if not (DATA / name).exists():
                pytest.skip(f"needs data/{name} from the private data repository")
        if marker.kwargs.get("place_sources"):
            from outis.data.generators import place_sources

            if not place_sources.sources_available():
                pytest.skip("needs the pinned place sources (run place_sources fetch)")
