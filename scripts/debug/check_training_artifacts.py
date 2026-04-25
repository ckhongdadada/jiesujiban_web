from __future__ import annotations

import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enhancements.runtime_config import load_runtime_config
from enhancements.startup_checks import run_startup_checks, summarize_readiness


def main() -> None:
    config = load_runtime_config()
    checks = run_startup_checks(config)
    summary = summarize_readiness(checks)
    print(
        json.dumps(
            {
                "config": config.describe(),
                "readiness": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
