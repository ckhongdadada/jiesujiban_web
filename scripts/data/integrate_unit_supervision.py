from pathlib import Path
import json
import re
import pandas as pd

DESKTOP = Path.home() / 'Desktop'
OUT_DIR = Path(__file__).resolve().parent / 'raw_data_analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CITY_WEAK_FILE = OUT_DIR / 'city_reply_unit_candidates_extracted_only.csv'

EXCLUDE_NAME_PATTERNS = [r'合并结果']
EXCLUDE_PATH_PARTS = {'csv_data', '接诉即办处理后数据'}
DISTRICTS = [
    '东城区', '西城区', '朝阳区', '海淀区', '丰台区', '石景山区', '门头沟区', '房山区', '通州区', '顺义区',
    '昌平区', '大兴区', '怀柔区', '平谷区', '密云区', '延庆区', '北京经济技术开发区'
]

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
PSEUDO_UNITS = {'@北京12345', '认领交办', '留言人', '发生地', 'nan', 'None'}


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_PATH_PARTS for part in path.parts) or any(re.search(pat, path.name) for pat in EXCLUDE_NAME_PATTERNS)


def find_first_path(base: Path, pattern: str, require_dir: bool = False) -> Path | None:
    for path in sorted(base.glob(pattern)):
        if require_dir and not path.is_dir():
            continue
        return path
    return None


def resolve_root_dir() -> Path:
    root_dir = find_first_path(DESKTOP, '*留言板*汇总*', require_dir=True)
    if root_dir is None:
        raise FileNotFoundError(f'未找到根目录: {DESKTOP} 下 *留言板*汇总*')
    return root_dir


def detect_region(path: Path) -> str:
    s = str(path)
    for district in DISTRICTS + ['北京市']:
        if district in s:
            return district
    return '未知'


def normalize_text(x: str) -> str:
    if pd.isna(x):
        return ''
    x = str(x or '')
    x = re.sub(r'\s+', ' ', x).strip()
    return x


def normalize_unit(unit: str, district_context: str | None = None) -> str:
    unit = normalize_text(unit)
    unit = unit.replace('北京市北京市', '北京市')
    unit = re.sub(r'^[：:，,。；;\s]+', '', unit)
    unit = re.sub(r'[：:，,。；;\s]+$', '', unit)
    unit = re.sub(r'^(经|由)', '', unit)
    unit = UNIT_ALIASES.get(unit, unit)
    if district_context:
        if unit.startswith('区'):
            unit = district_context + unit[1:]
        elif unit in {'住建委', '教委', '交通委', '城管委', '城市管理委员会', '房管局', '水务局', '卫生健康委'}:
            unit = district_context + unit
    return unit.strip()


def classify_unit(unit: str) -> str:
    if not unit:
        return 'empty'
    if unit in PSEUDO_UNITS or unit.startswith('@'):
        return 'pseudo'
    if unit.endswith(('街道', '街道办', '镇')):
        return 'town_or_subdistrict'
    if '热线服务中心' in unit:
        return 'hotline_center'
    if unit.endswith(('委员会', '委', '局', '中心', '办公室', '管理局', '管理委', '指挥中心', '人民政府', '政府')):
        return 'department'
    return 'other'


def main():
    root_dir = resolve_root_dir()
    explicit_rows = []
    for path in sorted(root_dir.rglob('*.xlsx')):
        if is_excluded(path):
            continue
        try:
            xls = pd.ExcelFile(path)
        except Exception:
            continue
        region = detect_region(path)
        for sheet in xls.sheet_names:
            try:
                df = pd.read_excel(path, sheet_name=sheet, engine='openpyxl')
            except Exception:
                continue
            if '官方回复单位' not in df.columns:
                continue
            for idx, row in df.iterrows():
                raw_unit = normalize_text(row.get('官方回复单位', ''))
                if not raw_unit:
                    continue
                explicit_rows.append({
                    'source_type': 'explicit_field',
                    'source_file': str(path),
                    'source_name': path.name,
                    'source_sheet': sheet,
                    'district_context': region if region != '北京市' else '',
                    'row_index': int(idx),
                    'raw_unit': raw_unit,
                    'normalized_unit': normalize_unit(raw_unit, region if region != '北京市' else None),
                })

    explicit_df = pd.DataFrame(explicit_rows)
    if CITY_WEAK_FILE.exists():
        weak_df = pd.read_csv(CITY_WEAK_FILE, encoding='utf-8-sig').fillna('')
        weak_df['source_type'] = 'weak_reply_extracted'
        weak_df['district_context'] = weak_df['district_context'].fillna('')
        weak_df['source_sheet'] = weak_df.get('source_sheet', '')
        weak_df = weak_df[
            ['source_type', 'source_file', 'source_name', 'source_sheet', 'district_context', 'row_index', 'raw_unit', 'normalized_unit', 'confidence', 'rule_name']
        ]
    else:
        weak_df = pd.DataFrame(
            columns=['source_type', 'source_file', 'source_name', 'source_sheet', 'district_context', 'row_index', 'raw_unit', 'normalized_unit', 'confidence', 'rule_name']
        )

    if not explicit_df.empty:
        explicit_df['confidence'] = 'high'
        explicit_df['rule_name'] = 'explicit_column'

    all_units = pd.concat([explicit_df, weak_df], ignore_index=True)
    all_units['unit_category'] = all_units['normalized_unit'].apply(classify_unit)
    all_units['is_pseudo'] = all_units['unit_category'].eq('pseudo')

    all_units.to_csv(OUT_DIR / 'unit_supervision_candidates_all.csv', index=False, encoding='utf-8-sig')

    mapping = (
        all_units.groupby(['normalized_unit', 'unit_category', 'source_type', 'confidence'], dropna=False)
        .agg(
            sample_count=('normalized_unit', 'size'),
            source_files=('source_file', 'nunique'),
            raw_unit_examples=('raw_unit', lambda s: ' | '.join(pd.Series(s).drop_duplicates().head(5).tolist())),
            district_examples=('district_context', lambda s: ' | '.join([x for x in pd.Series(s).drop_duplicates().tolist() if x][:5])),
        )
        .reset_index()
        .sort_values(['sample_count', 'normalized_unit'], ascending=[False, True])
    )
    mapping.to_csv(OUT_DIR / 'unit_normalization_mapping_candidates.csv', index=False, encoding='utf-8-sig')

    master_ready = weak_df.copy()
    if not master_ready.empty:
        master_ready['supervision_type'] = master_ready['confidence'].map({'high': 'weak_supervision_high', 'medium': 'weak_supervision_medium'}).fillna('weak_supervision_other')
        master_ready['recommended_for_cls_train'] = master_ready['confidence'].eq('high')
        master_ready['recommended_for_manual_review'] = master_ready['confidence'].eq('medium')
        master_ready = master_ready[
            [
                'source_file',
                'source_name',
                'source_sheet',
                'row_index',
                'district_context',
                'raw_unit',
                'normalized_unit',
                'rule_name',
                'confidence',
                'supervision_type',
                'recommended_for_cls_train',
                'recommended_for_manual_review',
            ]
        ]
    master_ready.to_csv(OUT_DIR / 'city_reply_unit_master_integration_ready.csv', index=False, encoding='utf-8-sig')

    report = {
        'overall': {
            'explicit_rows': int(len(explicit_df)),
            'weak_rows': int(len(weak_df)),
            'all_rows': int(len(all_units)),
            'unique_normalized_units': int(all_units['normalized_unit'].replace('', pd.NA).dropna().nunique()) if not all_units.empty else 0,
            'pseudo_rows': int(all_units['is_pseudo'].sum()) if not all_units.empty else 0,
        },
        'by_source_type': all_units.groupby('source_type').size().astype(int).to_dict() if not all_units.empty else {},
        'top_units': [{'unit': str(k), 'count': int(v)} for k, v in all_units.loc[all_units['normalized_unit'] != '', 'normalized_unit'].value_counts().head(30).items()],
        'top_pseudo_like_units': [{'unit': str(k), 'count': int(v)} for k, v in all_units.loc[all_units['is_pseudo'], 'normalized_unit'].value_counts().head(20).items()],
        'category_counts': all_units['unit_category'].value_counts().astype(int).to_dict() if not all_units.empty else {},
    }
    (OUT_DIR / 'unit_supervision_integration_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False, indent=2))
