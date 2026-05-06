from __future__ import annotations

import argparse
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.jsjb.core.model_registry_file import promote_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a registered model artifact version to active.")
    parser.add_argument("--component", required=True, help="Component name, e.g. classifier/generator_lora/generator_base")
    parser.add_argument("--version", required=True, help="Registered artifact version to promote")
    parser.add_argument("--registry", default="", help="Path to model_registry.json")
    parser.add_argument("--copy-to-current", action="store_true", help="Copy artifact directory to a current directory")
    parser.add_argument("--current-path", default="", help="Target current directory when --copy-to-current is used")
    args = parser.parse_args()

    result = promote_artifact(
        component=args.component,
        version=args.version,
        registry_path=args.registry or None,
        copy_to_current=args.copy_to_current,
        current_path=args.current_path or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
