from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enhancements.data_paths import ensure_standard_data_layout, get_runtime_unit_catalog_path
from enhancements.unit_catalog import build_catalog_from_frames

MASTER_TABLE_CANDIDATES = [
    Path(r"C:\Users\28414\Documents\New project\raw_data_analysis\master_table_v1.csv"),
]
MAPPING_CANDIDATES = [
    Path(r"C:\Users\28414\Documents\New project\raw_data_analysis\unit_normalization_mapping_candidates.csv"),
]


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def main() -> None:
    ensure_standard_data_layout()
    master_path = _first_existing(MASTER_TABLE_CANDIDATES)
    mapping_path = _first_existing(MAPPING_CANDIDATES)

    master_df = None
    mapping_df = None

    if master_path:
        master_df = pd.read_csv(master_path, low_memory=False)
        mask = master_df["recommended_for_unit_cls"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        master_df = master_df[mask].copy()
        master_df = master_df[
            master_df["reply_unit_norm"].fillna("").astype(str).str.strip().ne("")
        ]

    if mapping_path:
        mapping_df = pd.read_csv(mapping_path, low_memory=False)

    catalog = build_catalog_from_frames(mapping_df, master_df)
    output_path = get_runtime_unit_catalog_path()
    output_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "output_path": str(output_path),
        "master_path": str(master_path) if master_path else "",
        "mapping_path": str(mapping_path) if mapping_path else "",
        "unit_count": len(catalog["unit_meta"]),
        "alias_count": len(catalog["alias_to_unit"]),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
