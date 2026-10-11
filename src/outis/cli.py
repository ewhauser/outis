import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Outis: local clinical PHI de-identification"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Serve a local API with OpenAPI docs")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--host", default="127.0.0.1")
    redact = sub.add_parser("text", help="Read UTF-8 text from a file or stdin")
    redact.add_argument("--input", type=Path)
    redact.add_argument("--mode", choices=["redact", "synthetic"], default="redact")
    redact.add_argument("--patient-id")
    redact.add_argument(
        "--model-dir",
        help="Detector checkpoint directory (default: the published model, "
        "downloaded once into the Hugging Face cache)",
    )
    redact.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.command == "serve":
        if args.host not in {"127.0.0.1", "localhost", "::1"} and not os.environ.get(
            "OUTIS_API_KEY"
        ):
            parser.error("Set OUTIS_API_KEY before serving on a non-loopback interface")
        import uvicorn

        uvicorn.run("outis.api:app", host=args.host, port=args.port, access_log=False)
        return
    from outis.service import Deidentifier

    text = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
    deid = Deidentifier(
        args.model_dir, secret=os.environ.get("OUTIS_SECRET"), device=args.device
    )
    result = deid.deidentify(text, mode=args.mode, patient_id=args.patient_id)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
