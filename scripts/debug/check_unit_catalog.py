import json
from pathlib import Path

# 尝试多个可能的路径
paths = [
    'data/runtime/unit_catalog.json',
    'data/unit_catalog.json'
]

catalog = None
for p in paths:
    if Path(p).exists():
        print(f'找到文件: {p}')
        with open(p, 'r', encoding='utf-8') as f:
            catalog = json.load(f)
        break

if not catalog:
    print('未找到 unit_catalog.json 文件')
    print('尝试查找其他相关文件...')
    import os
    for root, dirs, files in os.walk('data'):
        for file in files:
            if 'catalog' in file.lower() or 'unit' in file.lower():
                print(f'  - {os.path.join(root, file)}')
else:
    alias_map = catalog.get('alias_to_unit', {})
    unit_meta = catalog.get('unit_meta', {})
    
    print(f'\n总单位数: {len(unit_meta)}')
    print(f'总别名数: {len(alias_map)}')
    
    print('\n=== 检查 政府 ===')
    gov_alias = alias_map.get('政府', '未找到')
    print(f'别名映射: {gov_alias}')
    if '政府' in unit_meta:
        meta = unit_meta['政府']
        print(f'样本数: {meta.get("sample_count", 0)}')
        print(f'类别: {meta.get("category", "")}')
        print(f'地区: {meta.get("districts", [])}')
    
    print('\n=== 检查 办事处 ===')
    office_alias = alias_map.get('办事处', '未找到')
    print(f'别名映射: {office_alias}')
    if '办事处' in unit_meta:
        meta = unit_meta['办事处']
        print(f'样本数: {meta.get("sample_count", 0)}')
        print(f'类别: {meta.get("category", "")}')
        print(f'地区: {meta.get("districts", [])}')
