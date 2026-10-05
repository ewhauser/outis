"""Start the loopback API with a persistent, private local replacement key."""

import argparse
import os
import secrets
from pathlib import Path

import uvicorn

from clinical_deid.paths import DEFAULT_MODEL_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    os.environ.setdefault("DEID_MODEL_DIR", str(project / DEFAULT_MODEL_DIR))
    os.environ.setdefault("DEID_DEVICE", args.device)
    if not os.environ.get("DEID_SECRET"):
        key_file = project / ".cache/local-api-secret"
        key_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "w") as handle:
                handle.write(secrets.token_urlsafe(48) + "\n")
        key_file.chmod(0o600)
        secret = key_file.read_text().strip()
        if len(secret.encode("utf-8")) < 32:
            raise ValueError("Local replacement key must contain at least 32 bytes")
        os.environ["DEID_SECRET"] = secret
    uvicorn.run(
        "clinical_deid.api:app", host="127.0.0.1", port=args.port, access_log=False
    )


if __name__ == "__main__":
    main()
