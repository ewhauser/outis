"""Skip tests that need files kept outside this repository.

Training corpora, their manifests and the blind-bank exclusion lists live in
the project's private data repository, and the public place sources are
downloaded on demand (``python -m outis.data.generators.place_sources
fetch``). A test that fails only because one of those is absent is reported as
skipped, with the reason, instead of as a failure.
"""

from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data"


def _missing_private_file(excinfo):
    if excinfo is None or not excinfo.errisinstance(FileNotFoundError):
        return None
    if "place sources" in str(excinfo.value):
        return "the pinned place sources (run place_sources fetch)"
    name = getattr(excinfo.value, "filename", None)
    if not name:
        return None
    path = Path(name)
    path = path if path.is_absolute() else Path.cwd() / path
    if not path.resolve().is_relative_to(DATA):
        return None
    return f"{name} from the private data repository"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    name = _missing_private_file(call.excinfo)
    if name and report.failed:
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0,
                           f"Skipped: needs {name}")
