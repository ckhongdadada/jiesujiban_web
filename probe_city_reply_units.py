from pathlib import Path
import json
import re
import pandas as pd

DESKTOP = Path.home() / 'Desktop'
out_dir = Path(__file__).resolve().parent / 'raw_data_analysis'
out_dir.mkdir(parents=True, exist_ok=True)

suffix_patterns = [
    re.compile(r'\n\s*([\u4e00-\u9fa5A-Za-z0-9（）()·、]{2,40}(?:委员会|办公室|管理局|管理委|指挥中心|街道办|镇政府|人民政府|政府|局|委|办|中心|镇|街道))\s*\n\s*\d{4}年\d{1,2}月\d{1,2}日\s*$', re.S),
    re.compile(r'([\u4e00-\u9fa5A-Za-z0-9（）()·、]{2,40}(?:委员会|办公室|管理局|管理委|指挥中心|街道办|镇政府|人民政府|政府|局|委|办|中心|镇|街道))\s*\n\s*\d{4}年\d{1,2}月\d{1,2}日\s*$', re.S),
    re.compile(r'([\u4e00-\u9fa5A-Za-z0-9（）()·、]{2,40}(?:委员会|办公室|管理局|管理委|指挥中心|街道办|镇政府|人民政府|政府|局|委|办|中心|镇|街道))\s*$'),
]
inline_patterns = [
    re.compile(r'([\u4e00-\u9fa5A-Za-z0-9（）()·、]{2,40}(?:委员会|办公室|管理局|管理委|指挥中心|街道办|镇政府|人民政府|政府|局|委|办|中心|镇|街道))(?:回复|答复|办理|核实|表示)'),
    re.compile(r'由([\u4e00-\u9fa5A-Za-z0-9（）()·、]{2,40}(?:委员会|办公室|管理局|管理委|指挥中心|街道办|镇政府|人民政府|政府|局|委|办|中心|镇|街道))(?:核实|办理|答复)'),
]


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
    city_files = sorted([p for p in root_dir.glob('北京市*.xlsx') if '合并结果' not in p.name and p.is_file()])
    if not city_files:
        raise FileNotFoundError(f'未找到北京市级文件: {root_dir}')
    return city_files


def norm_text(x):
    return str(x or '').replace('\r\n', '\n').replace('\r', '\n').strip()


def main():
    city_files = resolve_city_files()
    results = []
    examples = []
    for path in city_files:
        try:
            xls = pd.ExcelFile(path)
        except Exception:
            continue

        total = 0
        non_empty = 0
        suffix_hit = 0
        inline_hit = 0
        any_hit = 0
        extracted = []

        for sheet in xls.sheet_names:
            try:
                df = pd.read_excel(path, sheet_name=sheet, engine='openpyxl')
            except Exception:
                continue
            reply_col = '官方回复正文' if '官方回复正文' in df.columns else None
            title_col = '留言标题' if '留言标题' in df.columns else None
            if reply_col is None:
                continue

            replies = df[reply_col].fillna('').astype(str)
            total += len(df)
            for idx, reply in enumerate(replies):
                text = norm_text(reply)
                if not text:
                    continue
                non_empty += 1
                unit = None
                hit_type = None
                for pat in suffix_patterns:
                    m = pat.search(text)
                    if m:
                        unit = m.group(1).strip()
                        hit_type = 'suffix'
                        suffix_hit += 1
                        break
                if unit is None:
                    for pat in inline_patterns:
                        m = pat.search(text)
                        if m:
                            unit = m.group(1).strip()
                            hit_type = 'inline'
                            inline_hit += 1
                            break
                if unit:
                    any_hit += 1
                    extracted.append(unit)
                    if len(examples) < 30:
                        examples.append({
                            'source_file': path.name,
                            'source_sheet': sheet,
                            'row_index': int(idx),
                            'title': str(df.iloc[idx][title_col]) if title_col else '',
                            'hit_type': hit_type,
                            'unit': unit,
                            'reply_tail': text[-220:],
                        })

        vc = pd.Series(extracted).value_counts() if extracted else pd.Series(dtype='int64')
        results.append({
            'source_file': str(path),
            'rows': total,
            'non_empty_reply_rows': non_empty,
            'suffix_hit_rows': suffix_hit,
            'inline_hit_rows': inline_hit,
            'any_hit_rows': any_hit,
            'any_hit_ratio': round(any_hit / non_empty, 4) if non_empty else 0,
            'top_extracted_units': [{'unit': str(k), 'count': int(v)} for k, v in vc.head(20).items()],
        })

    report = {'files': results, 'examples': examples}
    (out_dir / 'city_reply_unit_probe.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False, indent=2))
