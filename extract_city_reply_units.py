from pathlib import Path
import json
import re
import pandas as pd

DESKTOP = Path.home() / 'Desktop'
OUT_DIR = Path(__file__).resolve().parent / 'raw_data_analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)
DISTRICTS = [
    '东城区', '西城区', '朝阳区', '海淀区', '丰台区', '石景山区', '门头沟区', '房山区', '通州区', '顺义区',
    '昌平区', '大兴区', '怀柔区', '平谷区', '密云区', '延庆区', '北京经济技术开发区'
]

EXPLICIT_PATTERNS = [
    ('explicit_reply', re.compile(r'([\u4e00-\u9fa5A-Za-z0-9（）()·、]{2,50}?(?:委员会|办公室|管理局|管理委|指挥中心|街道办|镇政府|人民政府|政府|税务局|公安局|交管局|教委|住建委|交通委|城管委|水务局|商务局|卫生健康委员会|卫生健康委|医保局|人力资源和社会保障局|规自分局|规自委|规划自然资源委|局|委|办|中心|镇|街道))回复[：:]')),
]
CONTEXT_PATTERNS = [
    ('verified_by', re.compile(r'经([\u4e00-\u9fa5A-Za-z0-9（）()·、]{2,50}?(?:委员会|办公室|管理局|管理委|指挥中心|街道办|镇政府|人民政府|政府|税务局|公安局|交管局|教委|住建委|交通委|城管委|水务局|商务局|卫生健康委员会|卫生健康委|医保局|人力资源和社会保障局|规自分局|规自委|规划自然资源委|局|委|办|中心|镇|街道))核实')),
    ('verified_generic', re.compile(r'经([区市][\u4e00-\u9fa5A-Za-z0-9（）()·、]{1,40}?(?:委员会|办公室|管理局|管理委|指挥中心|街道办|镇政府|人民政府|政府|税务局|公安局|交管局|教委|住建委|交通委|城管委|水务局|商务局|卫生健康委员会|卫生健康委|医保局|人力资源和社会保障局|规自分局|规自委|规划自然资源委|局|委|办|中心|镇|街道))核实')),
]
DISTRICT_REPLY_RE = re.compile(r'(' + '|'.join(map(re.escape, DISTRICTS)) + r')回复[：:]')
LEADING_REPLY_RE = re.compile(r'^您好，?关于您反映的问题，')


def find_first_path(base: Path, pattern: str, require_dir: bool = False) -> Path | None:
    for path in sorted(base.glob(pattern)):
        if require_dir and not path.is_dir():
            continue
        return path
    return None


def resolve_city_files() -> list[Path]:
    root_dir = find_first_path(DESKTOP, '*留言板*汇总*', require_dir=True)
    if root_dir is None:
        raise FileNotFoundError(f'未找到根目录: {DESKTOP} 下 *留言板*汇总*')
    preferred = [
        root_dir / '北京市委书记尹力留言板.xlsx',
        root_dir / '北京市市长殷勇留言板.xlsx',
    ]
    existing_preferred = [p for p in preferred if p.exists()]
    if existing_preferred:
        return existing_preferred
    fallback = sorted([p for p in root_dir.glob('北京市*.xlsx') if '合并结果' not in p.name and p.is_file()])
    if not fallback:
        raise FileNotFoundError(f'未找到北京市级文件: {root_dir}')
    return fallback


def norm_text(text: str) -> str:
    return str(text or '').replace('\r\n', '\n').replace('\r', '\n').strip()


def normalize_unit(unit: str, district_context: str | None = None) -> str:
    unit = str(unit or '').strip()
    unit = re.sub(r'^(经|由)', '', unit)
    unit = re.sub(r'[：:，,。；;\s]+$', '', unit)
    unit = unit.replace('北京市北京市', '北京市')
    if district_context:
        if unit.startswith('区'):
            unit = district_context + unit[1:]
        elif unit.startswith('开发区') and district_context == '北京经济技术开发区':
            unit = district_context + unit[3:]
    return unit.strip()


def infer_district_context(text: str) -> str | None:
    m = DISTRICT_REPLY_RE.search(text)
    return m.group(1) if m else None


def extract_one(text: str):
    district_context = infer_district_context(text)
    for rule_name, pattern in EXPLICIT_PATTERNS:
        m = pattern.search(text)
        if m:
            raw_unit = m.group(1).strip()
            return {
                'raw_unit': raw_unit,
                'normalized_unit': normalize_unit(raw_unit, district_context),
                'rule_name': rule_name,
                'confidence': 'high',
                'district_context': district_context,
            }
    for rule_name, pattern in CONTEXT_PATTERNS:
        m = pattern.search(text)
        if m:
            raw_unit = m.group(1).strip()
            normalized = normalize_unit(raw_unit, district_context)
            confidence = 'medium' if district_context or not raw_unit.startswith('区') else 'low'
            return {
                'raw_unit': raw_unit,
                'normalized_unit': normalized,
                'rule_name': rule_name,
                'confidence': confidence,
                'district_context': district_context,
            }
    return {
        'raw_unit': '',
        'normalized_unit': '',
        'rule_name': '',
        'confidence': 'none',
        'district_context': district_context,
    }


def main():
    city_files = resolve_city_files()
    rows = []
    summary = []
    for path in city_files:
        try:
            xls = pd.ExcelFile(path)
        except Exception:
            continue
        file_rows = []
        for sheet in xls.sheet_names:
            try:
                df = pd.read_excel(path, sheet_name=sheet, engine='openpyxl')
            except Exception:
                continue
            title_col = '留言标题' if '留言标题' in df.columns else None
            reply_col = '官方回复正文' if '官方回复正文' in df.columns else None
            id_col = '留言Id' if '留言Id' in df.columns else None
            date_col = '留言时间' if '留言时间' in df.columns else None
            tag_col = '留言标签' if '留言标签' in df.columns else None
            if reply_col is None:
                continue
            for idx, row in df.iterrows():
                reply = norm_text(row.get(reply_col, ''))
                extracted = extract_one(reply) if reply else {
                    'raw_unit': '', 'normalized_unit': '', 'rule_name': '', 'confidence': 'none', 'district_context': None
                }
                item = {
                    'source_file': str(path),
                    'source_name': path.name,
                    'source_sheet': sheet,
                    'row_index': int(idx),
                    'message_id': str(row.get(id_col, '')) if id_col else '',
                    'message_time': str(row.get(date_col, '')) if date_col else '',
                    'message_tag_raw': str(row.get(tag_col, '')) if tag_col else '',
                    'message_title': str(row.get(title_col, '')) if title_col else '',
                    'reply_text_raw': reply,
                    'reply_tail_preview': reply[-240:] if reply else '',
                    **extracted,
                }
                rows.append(item)
                file_rows.append(item)

        file_df = pd.DataFrame(file_rows)
        if file_df.empty:
            non_empty_reply = 0
            extracted_df = file_df
            high_rows = 0
            medium_rows = 0
            low_rows = 0
        else:
            non_empty_reply = int((file_df['reply_text_raw'] != '').sum())
            extracted_df = file_df[file_df['normalized_unit'] != '']
            high_rows = int((file_df['confidence'] == 'high').sum())
            medium_rows = int((file_df['confidence'] == 'medium').sum())
            low_rows = int((file_df['confidence'] == 'low').sum())
        summary.append({
            'source_file': str(path),
            'rows': int(len(file_df)),
            'non_empty_reply_rows': non_empty_reply,
            'extracted_rows': int(len(extracted_df)),
            'extracted_ratio': round(len(extracted_df) / non_empty_reply, 4) if non_empty_reply else 0,
            'high_confidence_rows': high_rows,
            'medium_confidence_rows': medium_rows,
            'low_confidence_rows': low_rows,
            'top_normalized_units': [
                {'unit': str(k), 'count': int(v)} for k, v in extracted_df['normalized_unit'].value_counts().head(25).items()
            ],
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_DIR / 'city_reply_unit_candidates.csv', index=False, encoding='utf-8-sig')
    result_df[result_df['normalized_unit'] != ''].to_csv(OUT_DIR / 'city_reply_unit_candidates_extracted_only.csv', index=False, encoding='utf-8-sig')

    extracted_examples = []
    if not result_df.empty:
        extracted_examples = result_df[result_df['normalized_unit'] != ''][[
            'source_name', 'source_sheet', 'row_index', 'message_title', 'raw_unit', 'normalized_unit', 'rule_name', 'confidence', 'district_context', 'reply_tail_preview'
        ]].head(40).to_dict(orient='records')

    if result_df.empty:
        extracted_rows = 0
        high_conf_rows = 0
        medium_conf_rows = 0
        low_conf_rows = 0
        unique_units = 0
    else:
        extracted_rows = int((result_df['normalized_unit'] != '').sum())
        high_conf_rows = int((result_df['confidence'] == 'high').sum())
        medium_conf_rows = int((result_df['confidence'] == 'medium').sum())
        low_conf_rows = int((result_df['confidence'] == 'low').sum())
        unique_units = int(result_df.loc[result_df['normalized_unit'] != '', 'normalized_unit'].nunique())

    report = {
        'city_files': summary,
        'overall': {
            'rows': int(len(result_df)),
            'extracted_rows': extracted_rows,
            'high_confidence_rows': high_conf_rows,
            'medium_confidence_rows': medium_conf_rows,
            'low_confidence_rows': low_conf_rows,
            'unique_normalized_units': unique_units,
        },
        'examples': extracted_examples,
    }
    (OUT_DIR / 'city_reply_unit_extraction_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False, indent=2))
