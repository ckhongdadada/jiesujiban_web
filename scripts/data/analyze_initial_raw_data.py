import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

DESKTOP = Path.home() / 'Desktop'
OUT_DIR = Path(__file__).resolve().parent / 'raw_data_analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_NAME_PATTERNS = [r"合并结果"]
EXCLUDE_PATH_PARTS = {"csv_data", "接诉即办处理后数据"}
TEXT_COL_CANDIDATES = {
    "title": ["留言标题", "标题", "主题"],
    "body": ["留言正文", "正文", "内容", "留言内容"],
    "reply_unit": ["官方回复单位", "回复单位", "承办单位", "单位"],
    "reply_text": ["官方回复正文", "官方回复", "回复正文", "答复内容", "回复内容"],
    "tag": ["留言标签", "标签", "问题类别", "类别"],
    "date": ["留言时间", "留言日期", "时间", "日期", "发表时间"],
    "status": ["办理状态", "状态"],
    "name": ["姓名", "网友", "留言人"],
    "location": ["发生地", "地点", "地址", "事发地"],
    "message_source": ["留言来源", "来源", "平台来源"],
}
LOW_QUALITY_PATTERNS = [
    r'正在.*?研究', r'正在.*?推进', r'请.*?耐心等待', r'已转.*?部门', r'已转办', r'请.*?关注.*?进展'
]
ONSITE_NEGATIVE_PATTERNS = [
    r'经.*?核查.*?未发现', r'经.*?核实.*?未发现', r'经.*?现场.*?未.*?发现', r'现场.*?查看.*?未.*?发现',
    r'经.*?核查.*?无.*?问题', r'经.*?核实.*?符合.*?标准', r'未见.*?异常', r'无明显.*?问题'
]
PLACE_HINT_PATTERNS = [
    r"[\u4e00-\u9fa5]{2,}(?:区|镇|乡|村|社区|小区|路|街|胡同|巷|大道|大街|桥|站|园|广场|大厦|学校|医院|公园|市场|地铁站)"
]
REGIONS = [
    "东城区", "西城区", "朝阳区", "海淀区", "丰台区", "石景山区", "门头沟区", "房山区", "通州区", "顺义区",
    "昌平区", "大兴区", "怀柔区", "平谷区", "密云区", "延庆区", "北京经济技术开发区", "北京市"
]


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_PATH_PARTS for part in path.parts) or any(re.search(pat, path.name) for pat in EXCLUDE_NAME_PATTERNS)


def find_first_path(base: Path, pattern: str, require_dir: bool = False, require_file: bool = False) -> Path | None:
    for path in sorted(base.glob(pattern)):
        if require_dir and not path.is_dir():
            continue
        if require_file and not path.is_file():
            continue
        return path
    return None


def resolve_inputs() -> tuple[Path, Path]:
    root_dir = find_first_path(DESKTOP, '*留言板*汇总*', require_dir=True)
    crawler_file = find_first_path(DESKTOP, '*留言板爬虫*.xlsx', require_file=True)
    if root_dir is None:
        raise FileNotFoundError(f'未找到根目录: {DESKTOP} 下 *留言板*汇总*')
    if crawler_file is None:
        raise FileNotFoundError(f'未找到爬虫文件: {DESKTOP} 下 *留言板爬虫*.xlsx')
    return root_dir, crawler_file


def detect_region(path: Path) -> str:
    path_str = str(path)
    for region in REGIONS:
        if region in path_str:
            return region
    return "未知"


def normalize_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def clean_training_data(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r'^[\s\S]*?(?:关于.*?回复信|关于.*?的函)\s*\n', '', text, flags=re.DOTALL)
    text = re.sub(r'您于[\d年月日]+.*?(?:现答复如下|回复如下)[：:。]?\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'您的留言.*?[，,。！!\n]\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'^(?:尊敬的.*?[，,！!\n]|您好\s*[！!，,：:\n]|你好\s*[！!，,：:\n])\s*', '', text, flags=re.DOTALL)
    for pat in [
        r'感谢您对.*?(?:理解[与和]?支持|关心[与和]?支持|关注[与和]?支持).*$',
        r'感谢您的.*?(?:理解|支持|关注|关心).*$', r'特此回复.*$', r'祝您.*?愉快.*$', r'请.*?谅解.*$',
        r'如有.*?疑问.*?联系.*$', r'欢迎.*?再次.*?留言.*$',
        r'\n\s*[\u4e00-\u9fa5]{2,20}(?:委员会|办公室|管理局|管理委|指挥中心|街道办|镇政府|局|处|科)\s*\n[\s\S]*$',
        r'\n\s*\d{4}年\d{1,2}月\d{1,2}日\s*$'
    ]:
        text = re.sub(pat, '', text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r'^[\s：:,，。！!\n]+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def is_low_quality(text: str) -> bool:
    text = str(text)
    return any(re.search(pat, text) for pat in LOW_QUALITY_PATTERNS) and len(text) < 120


def is_onsite_negative(text: str) -> bool:
    text = str(text)
    return any(re.search(pat, text) for pat in ONSITE_NEGATIVE_PATTERNS)


def find_col(columns, candidates):
    cols = list(columns)
    for candidate in candidates:
        if candidate in cols:
            return candidate
    for candidate in candidates:
        for col in cols:
            if candidate in str(col):
                return col
    return None


def standardize_df(df: pd.DataFrame, source_path: Path, sheet_name: str, crawler_file: Path):
    std = pd.DataFrame()
    mapping = {}
    for key, candidates in TEXT_COL_CANDIDATES.items():
        col = find_col(df.columns, candidates)
        mapping[key] = str(col) if col is not None else None
        std[key] = normalize_series(df[col]) if col is not None else ""
    std["source_file"] = str(source_path)
    std["source_name"] = source_path.name
    std["source_sheet"] = sheet_name
    std["source_region"] = detect_region(source_path)
    std["source_kind"] = "crawler_workbook" if source_path == crawler_file else "message_board_workbook"
    std["row_index"] = range(len(std))
    std["issue_text"] = (std["title"] + "。" + std["body"]).str.strip("。")
    std["issue_key"] = normalize_series(std["tag"] + "|" + std["title"] + "|" + std["body"])
    std["reply_clean"] = std["reply_text"].apply(clean_training_data)
    std["reply_unit_norm"] = normalize_series(std["reply_unit"])
    std["location_norm"] = normalize_series(std["location"])
    std["message_source_norm"] = normalize_series(std["message_source"])
    std["has_place_hint"] = std["issue_text"].apply(lambda x: any(re.search(pat, x) for pat in PLACE_HINT_PATTERNS))
    std["issue_len"] = std["issue_text"].str.len()
    std["reply_len"] = std["reply_text"].str.len()
    std["reply_clean_len"] = std["reply_clean"].str.len()
    std["reply_low_quality_flag"] = std["reply_clean"].apply(is_low_quality)
    std["reply_onsite_negative_flag"] = std["reply_clean"].apply(is_onsite_negative)
    return std, mapping


def load_workbook(path: Path, crawler_file: Path):
    sheets = pd.ExcelFile(path).sheet_names
    frames = []
    meta = []
    for sheet in sheets:
        try:
            df = pd.read_excel(path, sheet_name=sheet, engine='openpyxl')
        except Exception as exc:
            meta.append({"sheet": sheet, "error": str(exc)})
            continue
        std, mapping = standardize_df(df, path, sheet, crawler_file)
        meta.append({"sheet": sheet, "rows": int(len(df)), "columns": [str(c) for c in df.columns.tolist()], "mapping": mapping})
        frames.append(std)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return merged, meta


def summarize_counter(series: pd.Series, topn=20):
    counts = Counter(series.tolist())
    return [{"value": str(k), "count": int(v)} for k, v in counts.most_common(topn)]


def main():
    root_dir, crawler_file = resolve_inputs()
    inventory = []
    workbook_candidates = []
    for path in sorted(root_dir.rglob('*')):
        if not path.is_file():
            continue
        info = {
            "path": str(path), "name": path.name, "suffix": path.suffix.lower(), "size": path.stat().st_size,
            "excluded": is_excluded(path), "region": detect_region(path)
        }
        inventory.append(info)
        if path.suffix.lower() == '.xlsx' and not is_excluded(path):
            workbook_candidates.append(path)
    workbook_candidates.append(crawler_file)
    workbook_candidates = list(dict.fromkeys(workbook_candidates))

    all_frames = []
    workbook_reports = {}
    for path in workbook_candidates:
        merged, meta = load_workbook(path, crawler_file)
        workbook_reports[str(path)] = meta
        if not merged.empty:
            all_frames.append(merged)

    merged_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    if not merged_df.empty:
        merged_df["dedupe_issue_key"] = normalize_series(merged_df["tag"] + "|" + merged_df["title"] + "|" + merged_df["body"])
        merged_df["dedupe_gen_key"] = normalize_series(merged_df["tag"] + "|" + merged_df["title"] + "|" + merged_df["body"] + "|" + merged_df["reply_clean"])

    overall = {
        "inventory": {
            "all_files": len(inventory),
            "xlsx_candidates": len([i for i in inventory if i["suffix"] == '.xlsx' and not i["excluded"]]),
            "excluded_files": len([i for i in inventory if i["excluded"]]),
        },
        "root_dir": str(root_dir),
        "crawler_file": str(crawler_file),
        "merged": {
            "rows": int(len(merged_df)),
            "source_files": int(merged_df["source_file"].nunique()) if not merged_df.empty else 0,
            "regions": int(merged_df["source_region"].nunique()) if not merged_df.empty else 0,
            "sheets": int((merged_df["source_file"] + '|' + merged_df["source_sheet"]).nunique()) if not merged_df.empty else 0,
        },
        "task_views": {"ner": {}, "unit_classification": {}, "reply_generation": {}},
    }

    if not merged_df.empty:
        ner_df = merged_df[(merged_df["title"] != '') | (merged_df["body"] != '')].copy()
        cls_df = merged_df[(merged_df["reply_unit_norm"] != '') & ((merged_df["title"] != '') | (merged_df["body"] != ''))].copy()
        gen_df = merged_df[(merged_df["reply_text"] != '') & ((merged_df["title"] != '') | (merged_df["body"] != ''))].copy()

        unit_counts = cls_df["reply_unit_norm"].value_counts()
        overall["task_views"]["ner"] = {
            "candidate_rows": int(len(ner_df)),
            "with_location_field": int((ner_df["location_norm"] != "").sum()),
            "location_field_ratio": round(float((ner_df["location_norm"] != "").mean()), 4) if len(ner_df) else 0,
            "rows_with_place_hint": int(ner_df["has_place_hint"].sum()),
            "place_hint_ratio": round(float(ner_df["has_place_hint"].mean()), 4) if len(ner_df) else 0,
            "issue_len_mean": round(float(ner_df["issue_len"].mean()), 2) if len(ner_df) else 0,
            "issue_len_p95": round(float(ner_df["issue_len"].quantile(0.95)), 2) if len(ner_df) else 0,
            "top_regions": summarize_counter(ner_df["source_region"], 20),
            "top_tags": summarize_counter(ner_df["tag"], 20),
            "top_locations": summarize_counter(ner_df["location_norm"].replace("", pd.NA).dropna(), 25),
        }
        overall["task_views"]["unit_classification"] = {
            "candidate_rows": int(len(cls_df)),
            "unique_units": int(cls_df["reply_unit_norm"].nunique()),
            "units_lt_2": int((unit_counts < 2).sum()),
            "units_lt_5": int((unit_counts < 5).sum()),
            "duplicate_issue_keys": int(cls_df["dedupe_issue_key"].duplicated().sum()),
            "duplicate_issue_to_multi_units": int(cls_df.groupby("dedupe_issue_key")["reply_unit_norm"].nunique().gt(1).sum()),
            "top_units": summarize_counter(cls_df["reply_unit_norm"], 25),
            "top_tags": summarize_counter(cls_df["tag"], 20),
            "issue_len_mean": round(float(cls_df["issue_len"].mean()), 2) if len(cls_df) else 0,
            "issue_len_p95": round(float(cls_df["issue_len"].quantile(0.95)), 2) if len(cls_df) else 0,
        }
        overall["task_views"]["reply_generation"] = {
            "candidate_rows": int(len(gen_df)),
            "with_reply_unit": int((gen_df["reply_unit_norm"] != '').sum()),
            "unique_reply_units": int(gen_df["reply_unit_norm"].nunique()),
            "raw_duplicate_replies": int(gen_df["reply_text"].duplicated().sum()),
            "clean_duplicate_replies": int(gen_df["reply_clean"].duplicated().sum()),
            "duplicate_issue_reply_pairs": int(gen_df["dedupe_gen_key"].duplicated().sum()),
            "duplicate_issue_to_multi_replies": int(gen_df.groupby("dedupe_issue_key")["reply_clean"].nunique().gt(1).sum()),
            "low_quality_flag_rows": int(gen_df["reply_low_quality_flag"].sum()),
            "onsite_negative_flag_rows": int(gen_df["reply_onsite_negative_flag"].sum()),
            "reply_len_mean": round(float(gen_df["reply_len"].mean()), 2) if len(gen_df) else 0,
            "reply_len_p95": round(float(gen_df["reply_len"].quantile(0.95)), 2) if len(gen_df) else 0,
            "reply_clean_len_mean": round(float(gen_df["reply_clean_len"].mean()), 2) if len(gen_df) else 0,
            "reply_clean_len_p95": round(float(gen_df["reply_clean_len"].quantile(0.95)), 2) if len(gen_df) else 0,
            "top_units": summarize_counter(gen_df["reply_unit_norm"], 25),
            "top_tags": summarize_counter(gen_df["tag"], 20),
            "top_message_sources": summarize_counter(gen_df["message_source_norm"].replace("", pd.NA).dropna(), 20),
        }
        by_source = []
        for source, group in merged_df.groupby('source_file'):
            by_source.append({
                "source_file": source,
                "rows": int(len(group)),
                "region": group["source_region"].iloc[0],
                "with_issue": int(((group["title"] != '') | (group["body"] != '')).sum()),
                "with_unit": int((group["reply_unit_norm"] != '').sum()),
                "with_reply": int((group["reply_text"] != '').sum()),
                "unique_units": int(group["reply_unit_norm"].replace('', pd.NA).dropna().nunique()),
                "place_hint_ratio": round(float(group["has_place_hint"].mean()), 4) if len(group) else 0,
            }) 
        overall["by_source"] = sorted(by_source, key=lambda x: x["rows"], reverse=True)
        overall["cross_source"] = {
            "duplicate_issue_keys_total": int(merged_df["dedupe_issue_key"].duplicated().sum()),
            "duplicate_issue_reply_pairs_total": int(merged_df["dedupe_gen_key"].duplicated().sum()),
            "duplicate_issue_across_source_files": int(merged_df.groupby("dedupe_issue_key")["source_file"].nunique().gt(1).sum()),
        }

    (OUT_DIR / 'inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT_DIR / 'workbook_reports.json').write_text(json.dumps(workbook_reports, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT_DIR / 'merged_profile.json').write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding='utf-8')

    preview_cols = [
        'source_region', 'location_norm', 'message_source_norm', 'tag', 'title', 'body', 'reply_unit_norm', 'reply_text', 'reply_clean',
        'has_place_hint', 'issue_len', 'reply_len', 'reply_clean_len', 'source_file', 'source_sheet'
    ]
    if not merged_df.empty:
        merged_df[preview_cols].head(300).to_csv(OUT_DIR / 'merged_preview.csv', index=False, encoding='utf-8-sig')
    else:
        pd.DataFrame(columns=preview_cols).to_csv(OUT_DIR / 'merged_preview.csv', index=False, encoding='utf-8-sig')
    print(json.dumps({
        'out_dir': str(OUT_DIR),
        'inventory_files': len(inventory),
        'workbook_candidates': len(workbook_candidates),
        'merged_rows': int(len(merged_df)),
        'source_files_loaded': int(merged_df['source_file'].nunique()) if not merged_df.empty else 0,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False, indent=2))
