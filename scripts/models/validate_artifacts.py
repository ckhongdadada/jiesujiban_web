from __future__ import annotations

import argparse
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.jsjb.core.model_registry_file import validate_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate model artifact registry and active model files.")
    parser.add_argument("--registry", default="", help="Path to model_registry.json")
    parser.add_argument("--json", action="store_true", help="Print full JSON result")
    args = parser.parse_args()

    result = validate_registry(args.registry or None)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Model artifact validation: {'OK' if result['ok'] else 'FAILED'}")
        for error in result.get("errors", []):
            print(f"ERROR: {error}")
        for warning in result.get("warnings", []):
            print(f"WARNING: {warning}")
        for name, report in result.get("artifacts", {}).items():
            print(f"- {name}: {'OK' if report.get('ok') else 'FAILED'}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
