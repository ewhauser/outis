"""The ``outis`` command line and the local server's persistent replacement key.

No model is loaded and no server is started: the detector is injected and
``uvicorn.run`` is replaced with a recorder.
"""

import importlib.util
import io
import json
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import outis.service
from outis import cli
from outis.service import Deidentifier
from outis.types import Span

SECRET = "test-only-secret-never-use-in-production-12345"
SCRIPT = Path(__file__).parents[1] / "scripts" / "serve_local.py"
ENVIRONMENT = (
    "OUTIS_SECRET",
    "OUTIS_MODEL_DIR",
    "OUTIS_DEVICE",
    "OUTIS_API_KEY",
    "OUTIS_SITE_DICTIONARIES",
)


class NameDetector:
    def detect_batch(self, texts):
        return [
            [Span(m.start(), m.end(), "NAME") for m in re.finditer("Zoë Núñez", t)]
            for t in texts
        ]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    # setenv first so monkeypatch restores the original state, including
    # absence, after the code under test writes os.environ directly.
    for name in ENVIRONMENT:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


@pytest.fixture
def constructed(monkeypatch):
    """Replace the CLI's Deidentifier and record how it was constructed."""
    calls = []

    def factory(model_dir, *, secret=None, device="auto"):
        calls.append({"model_dir": model_dir, "secret": secret, "device": device})
        return Deidentifier(detector=NameDetector(), secret=secret)

    monkeypatch.setattr(outis.service, "Deidentifier", factory)
    return calls


@pytest.fixture
def served(monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))
    return calls


def run_cli(monkeypatch, capsys, *args, stdin=""):
    monkeypatch.setattr(sys, "argv", ["outis", *args])
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    cli.main()
    return capsys.readouterr()


def test_text_from_stdin_redacts_and_prints_unescaped_json(
    monkeypatch, capsys, constructed
):
    out = run_cli(monkeypatch, capsys, "text", stdin="Seen: Zoë Núñez at café.\n")
    result = json.loads(out.out)
    assert result["text"] == "Seen: [NAME] at café.\n" and result["mode"] == "redact"
    assert "Zoë" not in out.out and "café" in out.out
    assert constructed == [{"model_dir": None, "secret": None, "device": "auto"}]


def test_text_loads_site_dictionaries_from_flag_or_environment(
    monkeypatch, capsys, tmp_path
):
    manifest = tmp_path / "site.json"
    manifest.write_text(
        json.dumps(
            {
                "dictionaries": [
                    {
                        "name": "facilities",
                        "label": "FACILITY",
                        "terms": ["Quillfeather Memorial Infirmary"],
                    }
                ]
            }
        )
    )
    calls = []

    def factory(model_dir, *, secret=None, device="auto", **options):
        calls.append(options)
        return Deidentifier(detector=NameDetector(), secret=secret, **options)

    monkeypatch.setattr(outis.service, "Deidentifier", factory)
    stdin = "Zoë Núñez left Quillfeather Memorial Infirmary.\n"
    flagged = run_cli(
        monkeypatch, capsys, "text", "--site-dictionaries", str(manifest), stdin=stdin
    )
    assert json.loads(flagged.out)["text"] == "[NAME] left [FACILITY].\n"
    monkeypatch.setenv("OUTIS_SITE_DICTIONARIES", str(manifest))
    from_env = run_cli(monkeypatch, capsys, "text", stdin=stdin)
    assert from_env.out == flagged.out
    assert calls == [{"site_dictionaries": str(manifest)}] * 2


def test_text_from_file_with_options(monkeypatch, capsys, constructed, tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("Zoë Núñez called.", encoding="utf-8")
    monkeypatch.setenv("OUTIS_SECRET", SECRET)
    args = ["text", "--input", str(note), "--mode", "synthetic"]
    args += ["--patient-id", "patient-7", "--model-dir", "/models/x", "--device", "cpu"]
    first = json.loads(run_cli(monkeypatch, capsys, *args).out)
    second = json.loads(run_cli(monkeypatch, capsys, *args).out)
    assert first == second and first["mode"] == "synthetic"
    assert "Zoë" not in first["text"] and first["text"].endswith(" called.")
    assert constructed[0] == {
        "model_dir": "/models/x",
        "secret": SECRET,
        "device": "cpu",
    }


def test_synthetic_mode_without_a_secret_fails_without_output(
    monkeypatch, capsys, constructed
):
    with pytest.raises(ValueError, match="at least 32 bytes"):
        run_cli(monkeypatch, capsys, "text", "--mode", "synthetic", stdin="Zoë Núñez")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("args", [[], ["text", "--mode", "unsafe"], ["nope"]])
def test_invalid_arguments_exit_with_usage(monkeypatch, capsys, args):
    with pytest.raises(SystemExit) as exit:
        run_cli(monkeypatch, capsys, *args)
    assert exit.value.code == 2


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_serve_on_loopback_needs_no_api_key(monkeypatch, capsys, served, host):
    run_cli(monkeypatch, capsys, "serve", "--host", host, "--port", "8123")
    assert served == [
        (("outis.api:app",), {"host": host, "port": 8123, "access_log": False})
    ]


def test_serve_on_a_public_interface_requires_an_api_key(monkeypatch, capsys, served):
    with pytest.raises(SystemExit) as exit:
        run_cli(monkeypatch, capsys, "serve", "--host", "0.0.0.0")
    assert exit.value.code == 2
    assert "OUTIS_API_KEY" in capsys.readouterr().err
    assert served == []
    monkeypatch.setenv("OUTIS_API_KEY", "a-configured-key")
    run_cli(monkeypatch, capsys, "serve", "--host", "0.0.0.0")
    assert served[0][1]["host"] == "0.0.0.0"


def test_console_entry_point_is_installed():
    completed = subprocess.run(
        [sys.executable, "-m", "outis.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0
    assert "serve" in completed.stdout and "text" in completed.stdout


# scripts/serve_local.py ----------------------------------------------------


@pytest.fixture
def checkout(tmp_path):
    """A copy of the script in a scratch checkout, so keys land under tmp_path."""
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(SCRIPT, tmp_path / "scripts" / "serve_local.py")
    return tmp_path


def serve_local(monkeypatch, checkout, *args):
    monkeypatch.setattr(sys, "argv", ["serve_local.py", *args])
    path = checkout / "scripts" / "serve_local.py"
    specification = importlib.util.spec_from_file_location("serve_local", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.main()


def key_file(checkout):
    return checkout / ".cache" / "local-api-secret"


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_local_key_is_created_private_and_reused(monkeypatch, checkout, served):
    import os

    serve_local(monkeypatch, checkout, "--port", "8001")
    key = key_file(checkout)
    created = key.read_text().strip()
    assert mode(key) == 0o600
    assert len(created.encode()) >= 32
    assert os.environ["OUTIS_SECRET"] == created
    assert os.environ["OUTIS_DEVICE"] == "auto"
    assert "OUTIS_MODEL_DIR" not in os.environ  # no local weights: download
    assert served == [
        (("outis.api:app",), {"host": "127.0.0.1", "port": 8001, "access_log": False})
    ]
    monkeypatch.delenv("OUTIS_SECRET")
    serve_local(monkeypatch, checkout)
    assert key.read_text().strip() == created
    assert os.environ["OUTIS_SECRET"] == created


def test_existing_key_permissions_are_tightened(monkeypatch, checkout, served):
    key = key_file(checkout)
    key.parent.mkdir()
    key.write_text("k" * 40 + "\n")
    key.chmod(0o644)
    serve_local(monkeypatch, checkout)
    assert mode(key) == 0o600
    assert key.read_text() == "k" * 40 + "\n"


def test_short_existing_key_is_refused_before_serving(monkeypatch, checkout, served):
    key = key_file(checkout)
    key.parent.mkdir()
    key.write_text("too-short\n")
    with pytest.raises(ValueError, match="at least 32 bytes"):
        serve_local(monkeypatch, checkout)
    assert served == []


def test_configured_secret_takes_precedence(monkeypatch, checkout, served):
    import os

    monkeypatch.setenv("OUTIS_SECRET", SECRET)
    serve_local(monkeypatch, checkout, "--device", "cpu")
    assert not key_file(checkout).exists()
    assert os.environ["OUTIS_SECRET"] == SECRET
    assert os.environ["OUTIS_DEVICE"] == "cpu"


def test_local_weights_are_preferred_when_present(monkeypatch, checkout, served):
    import os

    from outis.paths import DEFAULT_MODEL_DIR

    weights = checkout / DEFAULT_MODEL_DIR / "hf" / "model.safetensors"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    serve_local(monkeypatch, checkout)
    assert os.environ["OUTIS_MODEL_DIR"] == str(checkout / DEFAULT_MODEL_DIR)
