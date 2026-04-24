from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enhancements.unit_catalog import canonicalize_unit, infer_unit_district, score_unit_candidate


DISTRICT_NAMES = [
    "东城区",
    "西城区",
    "朝阳区",
    "丰台区",
    "石景山区",
    "海淀区",
    "门头沟区",
    "房山区",
    "通州区",
    "顺义区",
    "昌平区",
    "大兴区",
    "怀柔区",
    "平谷区",
    "密云区",
    "延庆区",
]

MULTI_UNIT_CUE_KEYWORDS = ["会同", "联合", "协同", "转交", "转办", "牵头", "配合", "相关部门", "属地", "街道"]
LINE_KEYWORDS = [
    "城管",
    "住建",
    "市场监管",
    "卫健",
    "公安",
    "水务",
    "交通",
    "教委",
    "教育",
    "民政",
    "生态环境",
    "应急",
    "发改",
    "人社",
    "医保",
    "文旅",
    "园林",
    "规划",
    "自然资源",
    "税务",
    "消防",
    "街道办",
    "镇政府",
    "社区",
]
CORE_STRIP_TOKENS = [
    "北京市",
    "北京",
    "人民政府办公室",
    "人民政府",
    "政府办",
    "政府",
    "管理委员会",
    "管理委",
    "委员会",
    "委",
    "管理局",
    "执法局",
    "分局",
    "支队",
    "大队",
    "中心",
    "局",
    "街道办事处",
    "街道办",
    "办事处",
    "镇政府",
    "镇",
    "乡",
    "街道",
    "社区",
]

MASTER_TABLE_CANDIDATES = [
    Path(r"C:\Users\28414\Documents\New project\raw_data_analysis\master_table_v1.csv"),
    PROJECT_ROOT / "raw_data_analysis" / "master_table_v1.csv",
]

MODEL_DIRS = [
    PROJECT_ROOT / "final_model_compare_cross_entropy",
    PROJECT_ROOT / "final_model_compare_focal",
    PROJECT_ROOT / "final_model_compare_rank_aware",
]

EVAL_REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "evaluation"
SAMPLE_SIZE = 600
BOOTSTRAP_ROUNDS = 2000
RANDOM_SEED = 42


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _soft_unit_match(expected: str, predicted: str) -> bool:
    expected_unit = canonicalize_unit(expected)
    predicted_unit = canonicalize_unit(predicted)
    if not expected_unit or not predicted_unit:
        return False
    return (
        expected_unit == predicted_unit
        or expected_unit in predicted_unit
        or predicted_unit in expected_unit
    )


def _unit_core(unit: str) -> str:
    text = canonicalize_unit(unit)
    for district in DISTRICT_NAMES:
        text = text.replace(district, "")
    for token in CORE_STRIP_TOKENS:
        text = text.replace(token, "")
    return re.sub(r"\s+", "", text)


def _same_line(unit_a: str, unit_b: str) -> bool:
    a = canonicalize_unit(unit_a)
    b = canonicalize_unit(unit_b)
    if not a or not b:
        return False
    keys_a = {key for key in LINE_KEYWORDS if key in a}
    keys_b = {key for key in LINE_KEYWORDS if key in b}
    return bool(keys_a & keys_b)


def _updown_related(unit_a: str, unit_b: str) -> bool:
    core_a = _unit_core(unit_a)
    core_b = _unit_core(unit_b)
    if len(core_a) < 2 or len(core_b) < 2:
        return False
    return core_a in core_b or core_b in core_a


def _acceptable_candidate(expected: str, candidate: str) -> bool:
    if _soft_unit_match(expected, candidate):
        return True
    expected_district = infer_unit_district(expected)
    candidate_district = infer_unit_district(candidate)
    same_district = bool(expected_district and candidate_district and expected_district == candidate_district)
    if same_district and _same_line(expected, candidate):
        return True
    if same_district and _updown_related(expected, candidate):
        return True
    if _same_line(expected, candidate) and _updown_related(expected, candidate):
        return True
    return False


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _load_eval_samples(sample_size: int) -> pd.DataFrame:
    master_path = _first_existing(MASTER_TABLE_CANDIDATES)
    if master_path is None:
        raise FileNotFoundError("master_table_v1.csv not found in configured candidates.")

    df = pd.read_csv(master_path, low_memory=False)
    required = ["message_title", "message_body", "reply_unit_norm", "district_from_file"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    mask = (
        df["message_title"].fillna("").astype(str).str.strip().ne("")
        & df["message_body"].fillna("").astype(str).str.strip().ne("")
        & df["reply_unit_norm"].fillna("").astype(str).str.strip().ne("")
        & df["district_from_file"].fillna("").astype(str).isin(DISTRICT_NAMES)
    )
    if "recommended_for_unit_cls" in df.columns:
        mask = mask & df["recommended_for_unit_cls"].apply(_as_bool)
    sampled_pool = df[mask].copy()
    if sampled_pool.empty:
        raise ValueError("No valid samples after filtering master table.")

    per_district = max(1, sample_size // len(DISTRICT_NAMES))
    grouped_parts = []
    for _, district_df in sampled_pool.groupby("district_from_file"):
        n = min(per_district, len(district_df))
        grouped_parts.append(district_df.sample(n=n, random_state=RANDOM_SEED))
    stratified = pd.concat(grouped_parts, ignore_index=True) if grouped_parts else sampled_pool.head(0)

    remaining_n = max(0, sample_size - len(stratified))
    if remaining_n > 0:
        remain = sampled_pool[~sampled_pool.index.isin(stratified.index)]
        if not remain.empty:
            fill = remain.sample(n=min(remaining_n, len(remain)), random_state=RANDOM_SEED)
            stratified = pd.concat([stratified, fill], ignore_index=True)

    result = stratified.head(sample_size).reset_index(drop=True)
    cue_pattern = re.compile("|".join(map(re.escape, MULTI_UNIT_CUE_KEYWORDS)))
    result["multi_unit_cue"] = result["message_body"].fillna("").astype(str).str.contains(cue_pattern)
    return result


def _bootstrap_delta_ci(top1: np.ndarray, top3: np.ndarray, rounds: int, seed: int) -> dict[str, float]:
    if len(top1) == 0:
        return {"delta_mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = np.random.default_rng(seed)
    n = len(top1)
    idx = rng.integers(0, n, size=(rounds, n))
    deltas = top3[idx].mean(axis=1) - top1[idx].mean(axis=1)
    return {
        "delta_mean": float(top3.mean() - top1.mean()),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
    }


def _evaluate_single_model(model_dir: Path, samples: pd.DataFrame, device: torch.device) -> tuple[dict, pd.DataFrame]:
    label_map_path = model_dir / "label_map.json"
    with open(label_map_path, "r", encoding="utf-8") as fp:
        id2label_raw = json.load(fp)
    id2label = {int(k): canonicalize_unit(v) for k, v in id2label_raw.items()}

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        num_labels=len(id2label),
        ignore_mismatched_sizes=True,
    ).to(device)
    model.eval()

    records = []
    with torch.no_grad():
        for _, row in samples.iterrows():
            tag = str(row.get("message_tag_level1", "") or "")
            title = str(row.get("message_title", "") or "")
            body = str(row.get("message_body", "") or "")
            district = str(row.get("district_from_file", "") or "")
            expected_unit = canonicalize_unit(str(row.get("reply_unit_norm", "") or ""))

            text = f"【{tag}】{title}。{body}"
            enc = tokenizer(
                text,
                add_special_tokens=True,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            logits = model(enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits
            probs = torch.softmax(logits, dim=-1)[0]

            top_k = 3
            candidate_k = min(max(top_k * 4, 12), len(id2label))
            values, indices = torch.topk(probs, k=candidate_k)

            reranked = []
            seen_units = set()
            for value, idx in zip(values.cpu().tolist(), indices.cpu().tolist()):
                unit = canonicalize_unit(id2label[int(idx)])
                if not unit or unit in seen_units:
                    continue
                seen_units.add(unit)
                reranked.append(
                    {
                        "unit": unit,
                        "prob": float(value),
                        "score": float(value) + score_unit_candidate(unit, district=district, tag=tag),
                    }
                )
            reranked.sort(key=lambda item: (-item["score"], -item["prob"], item["unit"]))
            top3_units = [item["unit"] for item in reranked[:3]]
            top1_unit = top3_units[0] if top3_units else ""

            top1_hit = _soft_unit_match(expected_unit, top1_unit)
            top3_hit = any(_soft_unit_match(expected_unit, unit) for unit in top3_units)
            top3_acceptable = any(_acceptable_candidate(expected_unit, unit) for unit in top3_units)

            records.append(
                {
                    "record_id": row.get("record_id"),
                    "district": district,
                    "expected_unit": expected_unit,
                    "top1_unit": top1_unit,
                    "top3_units": " | ".join(top3_units),
                    "top1_hit": bool(top1_hit),
                    "top3_hit": bool(top3_hit),
                    "top3_acceptable": bool(top3_acceptable),
                    "multi_unit_cue": bool(row.get("multi_unit_cue", False)),
                }
            )

    detail_df = pd.DataFrame(records)
    top1 = detail_df["top1_hit"].astype(float).to_numpy()
    top3 = detail_df["top3_hit"].astype(float).to_numpy()
    ci = _bootstrap_delta_ci(top1, top3, rounds=BOOTSTRAP_ROUNDS, seed=RANDOM_SEED)

    cue_df = detail_df[detail_df["multi_unit_cue"]]
    noncue_df = detail_df[~detail_df["multi_unit_cue"]]
    cue_delta = float(cue_df["top3_hit"].mean() - cue_df["top1_hit"].mean()) if len(cue_df) else 0.0
    noncue_delta = float(noncue_df["top3_hit"].mean() - noncue_df["top1_hit"].mean()) if len(noncue_df) else 0.0

    top1_wrong_df = detail_df[~detail_df["top1_hit"]]
    usable_ratio = float(top1_wrong_df["top3_acceptable"].mean()) if len(top1_wrong_df) else 0.0

    summary = {
        "model": model_dir.name,
        "sample_size": int(len(detail_df)),
        "top1_rate": float(detail_df["top1_hit"].mean()),
        "top3_rate": float(detail_df["top3_hit"].mean()),
        "delta_top3_minus_top1": float(detail_df["top3_hit"].mean() - detail_df["top1_hit"].mean()),
        "bootstrap_delta_ci95": ci,
        "stratified": {
            "cue_size": int(len(cue_df)),
            "noncue_size": int(len(noncue_df)),
            "cue_delta": cue_delta,
            "noncue_delta": noncue_delta,
            "cue_greater_than_noncue": bool(cue_delta > noncue_delta),
        },
        "business_usability": {
            "top1_wrong_size": int(len(top1_wrong_df)),
            "top1_wrong_top3_acceptable_ratio": usable_ratio,
        },
    }
    return summary, detail_df


def _parse_measured_table() -> list[dict]:
    table_path = PROJECT_ROOT / "model_efficiency_comparison.csv"
    if not table_path.exists():
        return []
    df = pd.read_csv(table_path, encoding="utf-8-sig")
    measured = []
    for _, row in df.iterrows():
        top1_text = str(row.iloc[6])
        top3_text = str(row.iloc[7])
        model_name = str(row.iloc[1])
        m1 = re.search(r"(\d+(?:\.\d+)?)%", top1_text)
        m3 = re.search(r"(\d+(?:\.\d+)?)%", top3_text)
        if not m1 or not m3:
            continue
        if "-" in top1_text:
            continue
        top1 = float(m1.group(1)) / 100.0
        top3 = float(m3.group(1)) / 100.0
        measured.append({"model": model_name, "top1": top1, "top3": top3, "delta": top3 - top1})
    return measured


def main() -> None:
    EVAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = _load_eval_samples(SAMPLE_SIZE)

    summaries = []
    detail_frames = []
    for model_dir in MODEL_DIRS:
        if not model_dir.exists():
            continue
        summary, detail_df = _evaluate_single_model(model_dir, samples, device=device)
        summaries.append(summary)
        detail_df["model"] = model_dir.name
        detail_frames.append(detail_df)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    measured_rows = _parse_measured_table()
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "sample_size": int(len(samples)),
        "cross_model_consistency": {
            "loss_models_all_delta_positive": bool(all(item["delta_top3_minus_top1"] > 0 for item in summaries))
            if summaries
            else False,
            "measured_models_all_delta_positive": bool(all(item["delta"] > 0 for item in measured_rows))
            if measured_rows
            else False,
            "measured_models": measured_rows,
        },
        "loss_model_summaries": summaries,
        "elapsed_seconds": round(time.time() - start, 2),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = EVAL_REPORT_DIR / f"topk_evidence_chain_{ts}.json"
    csv_path = EVAL_REPORT_DIR / f"topk_evidence_chain_details_{ts}.csv"
    latest_json = EVAL_REPORT_DIR / "topk_evidence_chain_latest.json"
    latest_csv = EVAL_REPORT_DIR / "topk_evidence_chain_details_latest.csv"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if detail_frames:
        detail_df = pd.concat(detail_frames, ignore_index=True)
        detail_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        detail_df.to_csv(latest_csv, index=False, encoding="utf-8-sig")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"JSON_REPORT={json_path}")
    print(f"CSV_REPORT={csv_path}")


if __name__ == "__main__":
    main()
