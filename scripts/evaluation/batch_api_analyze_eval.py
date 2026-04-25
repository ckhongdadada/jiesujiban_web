from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app  # noqa: E402
from src.jsjb.unit_classifier.catalog import canonicalize_unit  # noqa: E402

MASTER_TABLE_PATH = Path(r"C:\Users\28414\Documents\New project\raw_data_analysis\master_table_v1.csv")
REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "evaluation"
BATCH_SIZE = 8
DISTRICT_NAMES = [
    "东城区", "西城区", "朝阳区", "丰台区", "石景山区", "海淀区", "门头沟区", "房山区",
    "通州区", "顺义区", "昌平区", "大兴区", "怀柔区", "平谷区", "密云区", "延庆区",
]


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _soft_unit_match(expected: str, predicted: str) -> bool:
    expected = canonicalize_unit(expected)
    predicted = canonicalize_unit(predicted)
    if not expected or not predicted:
        return False
    return expected == predicted or expected in predicted or predicted in expected


def _reply_mentions_other_district(reply: str, district: str | None) -> list[str]:
    reply = reply or ""
    hits = [name for name in DISTRICT_NAMES if name in reply]
    if not district:
        return hits
    return [name for name in hits if name != district]


def _pick_samples() -> pd.DataFrame:
    df = pd.read_csv(MASTER_TABLE_PATH, low_memory=False)
    df = df[
        df["message_title"].fillna("").astype(str).str.strip().ne("")
        & df["message_body"].fillna("").astype(str).str.strip().ne("")
        & df["reply_unit_norm"].fillna("").astype(str).str.strip().ne("")
        & df["district_from_file"].fillna("").astype(str).isin(DISTRICT_NAMES)
    ].copy()
    df = df[df["recommended_for_unit_cls"].apply(_as_bool)]
    df = df[df["recommended_for_reply_gen"].apply(_as_bool)]

    grouped = []
    for _, district_df in df.groupby("district_from_file"):
        grouped.append(district_df.sample(n=1, random_state=42))
    seed_df = pd.concat(grouped, ignore_index=True).sample(frac=1.0, random_state=42)

    if len(seed_df) >= BATCH_SIZE:
        return seed_df.head(BATCH_SIZE).reset_index(drop=True)

    remaining = df[~df["record_id"].isin(seed_df["record_id"])].sample(
        n=min(BATCH_SIZE - len(seed_df), len(df) - len(seed_df)),
        random_state=42,
    )
    return pd.concat([seed_df, remaining], ignore_index=True).head(BATCH_SIZE).reset_index(drop=True)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    app = create_app()
    client = app.test_client()

    samples = _pick_samples()
    rows = []
    started = time.time()

    for _, row in samples.iterrows():
        payload = {
            "tag": str(row.get("message_tag_level1", "") or ""),
            "title": str(row.get("message_title", "") or ""),
            "body": str(row.get("message_body", "") or ""),
        }
        request_started = time.time()
        response = client.post("/api/analyze", json=payload)
        elapsed = time.time() - request_started
        data = response.get_json() or {}

        location = data.get("location", {}) or {}
        units = data.get("units", []) or []
        retrieval = data.get("retrieval", []) or []
        reply = data.get("reply", "") or ""

        predicted_unit = units[0]["unit"] if units else ""
        top3_units = [item.get("unit", "") for item in units[:3]]
        expected_unit = canonicalize_unit(str(row.get("reply_unit_norm", "") or ""))
        expected_district = str(row.get("district_from_file", "") or "")
        predicted_district = location.get("district")

        rows.append(
            {
                "record_id": row.get("record_id"),
                "district_expected": expected_district,
                "district_predicted": predicted_district,
                "district_hit": predicted_district == expected_district,
                "unit_expected": expected_unit,
                "unit_top1": canonicalize_unit(predicted_unit),
                "unit_top1_soft_hit": _soft_unit_match(expected_unit, predicted_unit),
                "unit_top3_soft_hit": any(_soft_unit_match(expected_unit, unit) for unit in top3_units),
                "retrieval_top1_title": retrieval[0]["title"] if retrieval else "",
                "retrieval_top1_district": retrieval[0]["district"] if retrieval else "",
                "retrieval_top1_same_district": bool(retrieval and retrieval[0].get("district") == expected_district),
                "retrieval_any_same_district": any(hit.get("district") == expected_district for hit in retrieval),
                "retrieval_titles": " | ".join(hit.get("title", "") for hit in retrieval),
                "retrieval_matched_terms": " | ".join(",".join(hit.get("matched_terms", [])) for hit in retrieval),
                "reply_length": len(reply),
                "reply_other_districts": " | ".join(_reply_mentions_other_district(reply, predicted_district)),
                "reply_preview": reply[:180],
                "status_code": response.status_code,
                "elapsed_seconds": round(elapsed, 2),
                "title": payload["title"],
            }
        )

    report_df = pd.DataFrame(rows)
    total_elapsed = time.time() - started
    summary = {
        "sample_size": int(len(report_df)),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "avg_elapsed_seconds": round(float(report_df["elapsed_seconds"].mean()), 2) if len(report_df) else 0.0,
        "district_hit_rate": round(float(report_df["district_hit"].mean()), 4) if len(report_df) else 0.0,
        "unit_top1_soft_hit_rate": round(float(report_df["unit_top1_soft_hit"].mean()), 4) if len(report_df) else 0.0,
        "unit_top3_soft_hit_rate": round(float(report_df["unit_top3_soft_hit"].mean()), 4) if len(report_df) else 0.0,
        "retrieval_top1_same_district_rate": round(float(report_df["retrieval_top1_same_district"].mean()), 4) if len(report_df) else 0.0,
        "retrieval_any_same_district_rate": round(float(report_df["retrieval_any_same_district"].mean()), 4) if len(report_df) else 0.0,
        "reply_other_district_cases": int(report_df["reply_other_districts"].astype(str).str.len().gt(0).sum()),
    }

    csv_path = REPORT_DIR / f"batch_api_eval_{timestamp}.csv"
    json_path = REPORT_DIR / f"batch_api_eval_{timestamp}.json"
    latest_csv = REPORT_DIR / "batch_api_eval_latest.csv"
    latest_json = REPORT_DIR / "batch_api_eval_latest.json"

    report_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    report_df.to_csv(latest_csv, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV_REPORT={csv_path}")
    print(f"JSON_REPORT={json_path}")


if __name__ == "__main__":
    main()
