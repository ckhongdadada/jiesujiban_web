from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.jsjb.core.config import load_runtime_config
from src.jsjb.core.model_manifest import (
    validate_model_manifest,
    write_model_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a runtime model artifact manifest for classifier and generator assets."
    )
    parser.add_argument(
        "--output",
        default="",
        help="Manifest output path. Defaults to RuntimeConfig.model_manifest_path.",
    )
    parser.add_argument(
        "--include-hashes",
        action="store_true",
        help="Hash artifact files with sha256. This is stricter but can be slow for large model weights.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the generated manifest against the current runtime config before exiting.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_runtime_config()
    output = Path(args.output or config.model_manifest_path)
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    manifest = write_model_manifest(
        output,
        config,
        include_hashes=args.include_hashes,
        created_by="scripts/ops/generate_model_manifest.py",
    )
    print(f"model manifest written: {output}")

    if args.check:
        status = validate_model_manifest(config, manifest)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0 if status["ok"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
