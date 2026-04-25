from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_data_root() -> Path:
    return get_project_root() / "data"


def get_runtime_dir() -> Path:
    return get_data_root() / "runtime"


def get_raw_dir() -> Path:
    return get_data_root() / "raw"


def get_processed_dir() -> Path:
    return get_data_root() / "processed"


def get_reports_dir() -> Path:
    return get_data_root() / "reports"


def get_universe_dir() -> Path:
    return get_data_root() / "universe"


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def ensure_standard_data_layout() -> None:
    dirs = [
        get_runtime_dir(),
        get_runtime_dir() / "place_taxonomy",
        get_raw_dir(),
        get_raw_dir() / "amap",
        get_raw_dir() / "places",
        get_processed_dir(),
        get_processed_dir() / "purified",
        get_reports_dir(),
        get_reports_dir() / "training",
        get_reports_dir() / "crawl",
        get_universe_dir(),
        get_universe_dir() / "v2",
    ]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)


def get_runtime_file(filename: str) -> Path:
    return _first_existing([get_runtime_dir() / filename, get_data_root() / filename])


def get_place_taxonomy_dir() -> Path:
    return _first_existing([get_runtime_dir() / "place_taxonomy", get_data_root() / "place_taxonomy"])


def get_policy_corpus_path() -> Path:
    return _first_existing([get_runtime_dir() / "policy_case_corpus.jsonl", get_data_root() / "policy_case_corpus.jsonl"])


def get_policy_corpus_sample_path() -> Path:
    return _first_existing(
        [get_runtime_dir() / "policy_case_corpus.sample.jsonl", get_data_root() / "policy_case_corpus.sample.jsonl"]
    )


def get_feedback_db_path() -> Path:
    return _first_existing([get_runtime_dir() / "feedback.db", get_data_root() / "feedback.db"])


def get_fact_kb_path() -> Path:
    return _first_existing([get_runtime_dir() / "fact_knowledge_base.json", get_data_root() / "fact_knowledge_base.json"])


def get_bulk_state_path() -> Path:
    return _first_existing([get_raw_dir() / "amap" / "amap_community_bulk_state.json", get_data_root() / "amap_community_bulk_state.json"])


def get_bulk_records_path() -> Path:
    return _first_existing(
        [get_raw_dir() / "amap" / "amap_community_bulk_records.jsonl", get_data_root() / "amap_community_bulk_records.jsonl"]
    )


def get_bulk_progress_path() -> Path:
    return _first_existing(
        [get_reports_dir() / "crawl" / "amap_community_bulk_progress.json", get_data_root() / "amap_community_bulk_progress.json"]
    )


def get_bulk_repair_report_path() -> Path:
    return _first_existing(
        [get_reports_dir() / "crawl" / "amap_community_bulk_repair_report.json", get_data_root() / "amap_community_bulk_repair_report.json"]
    )


def get_training_reports_dir() -> Path:
    return _first_existing([get_reports_dir() / "training", get_data_root() / "training_reports"])


def get_universe_v2_dir() -> Path:
    return _first_existing([get_universe_dir() / "v2", get_data_root() / "universe_v2"])


def get_purified_dir() -> Path:
    return _first_existing([get_processed_dir() / "purified", get_data_root() / "purified"])


def get_place_records_path(filename: str) -> Path:
    return _first_existing([get_raw_dir() / "places" / filename, get_data_root() / filename])


def get_runtime_catalog_path() -> Path:
    return get_runtime_file("place_alias_catalog.jsonl")


def get_runtime_district_file(filename: str) -> Path:
    return get_runtime_file(filename)


def get_runtime_unit_catalog_path() -> Path:
    return get_runtime_file("unit_catalog.json")
