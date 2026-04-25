from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd

DESKTOP = Path.home() / 'Desktop'
OUT_DIR = Path(__file__).resolve().parent / 'raw_data_analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)
WEAK_UNIT_FILE = OUT_DIR / 'city_reply_unit_master_integration_ready.csv'

EXCLUDE_NAME_PATTERNS = [r'合并结果']
EXCLUDE_PATH_PARTS = {'csv_data', '接诉即办处理后数据'}
DISTRICTS = [
    '东城区', '西城区', '朝阳区', '海淀区', '丰台区', '石景山区', '门头沟区', '房山区', '通州区', '顺义区',
    '昌平区', '大兴区', '怀柔区', '平谷区', '密云区', '延庆区', '北京经济技术开发区'
]
TEXT_COL_CANDIDATES = {
    'name': ['留言人名称', '留言人', '姓名', '网友'],
    'message_id': ['留言Id', '留言ID', 'ID'],
    'message_time': ['留言时间', '留言日期', '时间'],
    'message_tag_raw': ['留言标签', '标签', '问题类别'],
    'message_location_raw': ['发生地', '地点', '地址', '事发地'],
    'message_title': ['留言标题', '标题', '主题'],
    'message_body': ['留言正文', '正文', '内容', '留言内容'],
    'reply_unit_raw': ['官方回复单位', '回复单位', '承办单位'],
    'reply_time': ['官方回复时间', '回复时间'],
    'reply_text_raw': ['官方回复正文', '回复正文', '官方回复', '回复内容'],
    'message_source': ['留言来源', '来源', '平台来源'],
}
LOW_QUALITY_PATTERNS = [
    r'正在.*?研究', r'正在.*?推进', r'请.*?耐心等待', r'已转.*?部门', r'已转办', r'请.*?关注.*?进展'
]
ONSITE_NEGATIVE_PATTERNS = [
    r'经.*?核查.*?未发现', r'经.*?核实.*?未发现', r'经.*?现场.*?未.*?发现', r'现场.*?查看.*?未.*?发现',
    r'经.*?核查.*?无.*?问题', r'经.*?核实.*?符合.*?标准', r'未见.*?异常', r'无明显.*?问题'
]
PLACE_CANDIDATE_RE = re.compile(r'[\u4e00-\u9fa5A-Za-z0-9]{2,}(?:区|镇|乡|村|社区|小区|路|街|胡同|巷|大道|大街|桥|站|园|广场|大厦|学校|医院|公园|市场|地铁站)')
PSEUDO_UNITS = {'@北京12345', '认领交办'}
UNIT_ALIASES = {
    '市规自委': '市规划自然资源委',
    '市规划自然资源委员会': '市规划自然资源委',
    '市住建委': '市住房城乡建设委',
    '市卫健委': '市卫生健康委员会',
    '市人社局': '市人力资源和社会保障局',
    '市医保局': '市医疗保障局',
    '市城管委': '市城市管理委员会',
    '区城管委': '区城市管理委员会',
}


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
    s = str(path)
    for district in DISTRICTS:
        if district in s:
            return district
    if '北京市' in s:
        return '北京市'
    return '未知'


def source_type_for(path: Path, crawler_file: Path) -> str:
    if path == crawler_file:
        return 'crawler_raw'
    region = detect_region(path)
    if region == '北京市':
        return 'city_raw'
    return 'district_raw'


def normalize_text(x) -> str:
    if pd.isna(x):
        return ''
    x = str(x)
    x = x.replace('\r\n', '\n').replace('\r', '\n')
    x = re.sub(r'\s+', ' ', x).strip()
    return x


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


def clean_reply_text(text: str) -> str:
    text = str(text or '').strip()
    text = re.sub(r'^[\s\S]*?(?:关于.*?回复信|关于.*?的函)\s*\n', '', text, flags=re.DOTALL)
    text = re.sub(r'您于[\d年月日]+.*?(?:现答复如下|回复如下)[：:。]?\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'您的留言.*?[，,。！!\n]\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'^(?:尊敬的.*?[，,！!\n]|您好\s*[！!，,：:\n]|你好\s*[！!，,：:\n])\s*', '', text, flags=re.DOTALL)
    tail_patterns = [
        r'感谢您对.*?(?:理解[与和]?支持|关心[与和]?支持|关注[与和]?支持).*$',
        r'感谢您的.*?(?:理解|支持|关注|关心).*$', r'特此回复.*$', r'祝您.*?愉快.*$', r'请.*?谅解.*$',
        r'如有.*?疑问.*?联系.*$', r'欢迎.*?再次.*?留言.*$',
        r'\n\s*[\u4e00-\u9fa5]{2,30}(?:委员会|办公室|管理局|管理委|指挥中心|街道办|镇政府|局|处|科)\s*\n[\s\S]*$',
        r'\n\s*\d{4}年\d{1,2}月\d{1,2}日\s*$'
    ]
    for pat in tail_patterns:
        text = re.sub(pat, '', text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r'^[\s：:,，。！!\n]+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def is_low_quality(text: str) -> bool:
    text = str(text or '')
    return any(re.search(pat, text) for pat in LOW_QUALITY_PATTERNS) and len(text) < 120


def is_onsite_negative(text: str) -> bool:
    text = str(text or '')
    return any(re.search(pat, text) for pat in ONSITE_NEGATIVE_PATTERNS)


def parse_tag(raw: str):
    raw = normalize_text(raw)
    if not raw:
        return '', '', ''
    if raw.startswith('(') and raw.endswith(')'):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (tuple, list)):
                parts = [normalize_text(x) for x in parsed]
                while len(parts) < 3:
                    parts.append('')
                return parts[0], parts[1], parts[2]
        except Exception:
            pass
    parts = [normalize_text(x) for x in re.split(r'[,|/]+', raw) if normalize_text(x)]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], ''
    return raw, '', ''


def normalize_location(raw: str, district: str) -> str:
    raw = normalize_text(raw)
    if raw in {'发生地', 'nan', 'None'}:
        return ''
    if raw.startswith('北京市北京市'):
        raw = raw.replace('北京市北京市', '北京市', 1)
    if district and district not in {'北京市', '未知'} and raw.startswith('区'):
        raw = district + raw[1:]
    return raw


def extract_place_candidates(text: str) -> list[str]:
    if not text:
        return []
    candidates = []
    seen = set()
    for match in PLACE_CANDIDATE_RE.findall(text):
        candidate = normalize_text(match)
        if len(candidate) < 2 or candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
        if len(candidates) >= 12:
            break
    return candidates


def normalize_unit(raw: str, district_context: str | None = None) -> str:
    raw = normalize_text(raw)
    if raw in {'nan', 'None'}:
        return ''
    raw = re.sub(r'^(经|由)', '', raw)
    raw = re.sub(r'[：:，,。；;\s]+$', '', raw)
    raw = raw.replace('北京市北京市', '北京市')
    raw = UNIT_ALIASES.get(raw, raw)
    if district_context:
        if raw.startswith('区'):
            raw = district_context + raw[1:]
        elif raw in {'住建委', '教委', '交通委', '城管委', '城市管理委员会', '房管局', '水务局', '卫生健康委'}:
            raw = district_context + raw
    return raw.strip()


def has_place_hint(text: str, location_norm: str) -> bool:
    return bool(location_norm) or bool(PLACE_CANDIDATE_RE.search(text or ''))


def build_keys(tag_raw: str, title: str, body: str, reply_clean: str):
    issue_key = normalize_text(f'{tag_raw}|{title}|{body}')
    gen_key = normalize_text(f'{tag_raw}|{title}|{body}|{reply_clean}')
    return issue_key, gen_key


def load_weak_units() -> dict[tuple[str, int], dict]:
    if not WEAK_UNIT_FILE.exists():
        return {}
    weak = pd.read_csv(WEAK_UNIT_FILE, encoding='utf-8-sig').fillna('')
    lookup = {}
    for _, row in weak.iterrows():
        try:
            row_index = int(row['row_index'])
        except Exception:
            continue
        key = (str(row['source_file']), row_index)
        lookup[key] = {
            'reply_unit_raw': normalize_text(row.get('raw_unit', '')),
            'reply_unit_norm': normalize_text(row.get('normalized_unit', '')),
            'reply_unit_source': 'weak_reply_extracted',
            'reply_unit_confidence': normalize_text(row.get('confidence', '')),
            'reply_unit_rule': normalize_text(row.get('rule_name', '')),
            'reply_unit_district_context': normalize_text(row.get('district_context', '')),
            'recommended_for_unit_cls': str(bool(row.get('recommended_for_cls_train', False))).lower(),
        }
    return lookup


def iter_workbooks(root_dir: Path, crawler_file: Path):
    workbook_paths = []
    for path in sorted(root_dir.rglob('*.xlsx')):
        if is_excluded(path):
            continue
        workbook_paths.append(path)
    workbook_paths.append(crawler_file)
    for path in dict.fromkeys(workbook_paths):
        yield path


def main():
    root_dir, crawler_file = resolve_inputs()
    weak_lookup = load_weak_units()
    records = []
    workbook_count = 0
    for path in iter_workbooks(root_dir, crawler_file):
        workbook_count += 1
        try:
            xls = pd.ExcelFile(path)
        except Exception:
            continue
        for sheet in xls.sheet_names:
            try:
                df = pd.read_excel(path, sheet_name=sheet, engine='openpyxl')
            except Exception:
                continue
            col_map = {key: find_col(df.columns, candidates) for key, candidates in TEXT_COL_CANDIDATES.items()}
            district = detect_region(path)
            source_type = source_type_for(path, crawler_file)
            for idx, row in df.iterrows():
                values = {key: normalize_text(row[col_map[key]]) if col_map[key] is not None else '' for key in TEXT_COL_CANDIDATES}
                title = values['message_title']
                body = values['message_body']
                reply_text_raw = values['reply_text_raw']
                tag_raw = values['message_tag_raw']
                tag_level1, tag_level2, message_status = parse_tag(tag_raw)
                location_raw = values['message_location_raw']
                location_norm = normalize_location(location_raw, district)
                message_text = normalize_text((title + '。' + body).strip('。'))
                place_candidates = extract_place_candidates(message_text)
                reply_clean = clean_reply_text(reply_text_raw)
                low_quality = is_low_quality(reply_clean)
                onsite_negative = is_onsite_negative(reply_clean)
                dedupe_issue_key, dedupe_generation_key = build_keys(tag_raw, title, body, reply_clean)

                explicit_raw = values['reply_unit_raw']
                explicit_norm = normalize_unit(explicit_raw, district if district not in {'北京市', '未知'} else None)
                reply_unit_raw = explicit_raw
                reply_unit_norm = explicit_norm
                reply_unit_source = 'explicit_field' if explicit_norm else ''
                reply_unit_confidence = 'high' if explicit_norm else ''
                reply_unit_rule = 'explicit_column' if explicit_norm else ''
                reply_unit_district_context = district if explicit_norm and district not in {'北京市', '未知'} else ''
                recommended_for_unit_cls = bool(explicit_norm and explicit_norm not in PSEUDO_UNITS and not explicit_norm.startswith('@'))

                weak_hit = weak_lookup.get((str(path), int(idx)))
                if not explicit_norm and weak_hit:
                    reply_unit_raw = weak_hit['reply_unit_raw']
                    reply_unit_norm = weak_hit['reply_unit_norm']
                    reply_unit_source = weak_hit['reply_unit_source']
                    reply_unit_confidence = weak_hit['reply_unit_confidence']
                    reply_unit_rule = weak_hit['reply_unit_rule']
                    reply_unit_district_context = weak_hit['reply_unit_district_context']
                    recommended_for_unit_cls = weak_hit['recommended_for_unit_cls'] == 'true'

                if reply_unit_norm in PSEUDO_UNITS or reply_unit_norm.startswith('@'):
                    recommended_for_unit_cls = False

                record_id = f'{path.name}::{sheet}::{idx}'
                records.append({
                    'record_id': record_id,
                    'source_file': str(path),
                    'source_name': path.name,
                    'source_sheet': sheet,
                    'source_type': source_type,
                    'district_from_file': district,
                    'message_source': values['message_source'],
                    'message_id': values['message_id'],
                    'message_time': values['message_time'],
                    'message_tag_raw': tag_raw,
                    'message_tag_level1': tag_level1,
                    'message_tag_level2': tag_level2,
                    'message_status': message_status,
                    'message_location_raw': location_raw,
                    'message_location_norm': location_norm,
                    'message_title': title,
                    'message_body': body,
                    'message_text': message_text,
                    'has_place_hint': has_place_hint(message_text, location_norm),
                    'place_mention_candidates': json.dumps(place_candidates, ensure_ascii=False),
                    'reply_unit_raw': reply_unit_raw,
                    'reply_unit_norm': reply_unit_norm,
                    'reply_unit_source': reply_unit_source,
                    'reply_unit_confidence': reply_unit_confidence,
                    'reply_unit_rule': reply_unit_rule,
                    'reply_unit_district_context': reply_unit_district_context,
                    'reply_time': values['reply_time'],
                    'reply_text_raw': reply_text_raw,
                    'reply_text_clean': reply_clean,
                    'reply_low_quality_flag': low_quality,
                    'reply_onsite_negative_flag': onsite_negative,
                    'dedupe_issue_key': dedupe_issue_key,
                    'dedupe_generation_key': dedupe_generation_key,
                    'recommended_for_ner': bool(message_text),
                    'recommended_for_unit_cls': bool(recommended_for_unit_cls),
                    'recommended_for_reply_gen': bool(message_text and reply_text_raw),
                    'notes': '',
                })

    master = pd.DataFrame(records)
    master.to_csv(OUT_DIR / 'master_table_v1.csv', index=False, encoding='utf-8-sig')

    summary = {
        'workbooks_scanned': workbook_count,
        'rows': int(len(master)),
        'source_type_counts': master['source_type'].value_counts().astype(int).to_dict() if not master.empty else {},
        'rows_with_message_text': int((master['message_text'] != '').sum()) if not master.empty else 0,
        'rows_with_reply_text': int((master['reply_text_raw'] != '').sum()) if not master.empty else 0,
        'rows_with_location_field': int((master['message_location_norm'] != '').sum()) if not master.empty else 0,
        'rows_with_place_hint': int(master['has_place_hint'].sum()) if not master.empty else 0,
        'rows_with_reply_unit': int((master['reply_unit_norm'] != '').sum()) if not master.empty else 0,
        'rows_with_explicit_unit': int((master['reply_unit_source'] == 'explicit_field').sum()) if not master.empty else 0,
        'rows_with_weak_unit': int((master['reply_unit_source'] == 'weak_reply_extracted').sum()) if not master.empty else 0,
        'recommended_for_ner': int(master['recommended_for_ner'].sum()) if not master.empty else 0,
        'recommended_for_unit_cls': int(master['recommended_for_unit_cls'].sum()) if not master.empty else 0,
        'recommended_for_reply_gen': int(master['recommended_for_reply_gen'].sum()) if not master.empty else 0,
        'top_level1_tags': [{ 'value': str(k), 'count': int(v)} for k, v in master['message_tag_level1'].value_counts().head(20).items()],
        'top_level2_tags': [{ 'value': str(k), 'count': int(v)} for k, v in master['message_tag_level2'].value_counts().head(20).items()],
        'top_reply_units': [{ 'value': str(k), 'count': int(v)} for k, v in master.loc[master['reply_unit_norm'] != '', 'reply_unit_norm'].value_counts().head(20).items()],
    }
    (OUT_DIR / 'master_table_v1_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False, indent=2))
