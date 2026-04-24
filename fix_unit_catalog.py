"""
修复单位目录 - 从训练数据中添加缺失的单位（如政府、办事处）
"""
import json
import pandas as pd
from pathlib import Path
from collections import Counter

# 读取现有的单位目录
catalog_path = Path('data/runtime/unit_catalog.json')
with open(catalog_path, 'r', encoding='utf-8') as f:
    catalog = json.load(f)

alias_to_unit = catalog.get('alias_to_unit', {})
unit_meta = catalog.get('unit_meta', {})

# 读取训练数据
training_data_path = r'C:\Users\28414\Desktop\留言板合并数据.xlsx'
df = pd.read_excel(training_data_path)

# 统计所有单位
unit_counts = df['官方回复单位'].value_counts()

print(f'训练数据中共有 {len(unit_counts)} 个不同的单位')
print(f'单位目录中共有 {len(unit_meta)} 个单位')

# 找出缺失的单位
missing_units = []
for unit, count in unit_counts.items():
    unit_str = str(unit).strip()
    if unit_str and unit_str not in alias_to_unit and unit_str not in unit_meta:
        missing_units.append((unit_str, count))

print(f'\n发现 {len(missing_units)} 个缺失的单位:')
for unit, count in sorted(missing_units, key=lambda x: -x[1])[:20]:
    print(f'  {unit}: {count} 样本')

# 添加缺失的单位到目录
for unit_str, count in missing_units:
    # 添加到别名映射
    alias_to_unit[unit_str] = unit_str
    
    # 添加到单位元数据
    if unit_str not in unit_meta:
        unit_meta[unit_str] = {
            "districts": [],
            "top_tags": [],
            "category": "",
            "source_types": ["training_data"],
            "aliases": [unit_str],
            "sample_count": count
        }

# 保存更新后的目录
catalog['alias_to_unit'] = dict(sorted(alias_to_unit.items()))
catalog['unit_meta'] = dict(sorted(unit_meta.items()))

with open(catalog_path, 'w', encoding='utf-8') as f:
    json.dump(catalog, f, ensure_ascii=False, indent=2)

print(f'\n✅ 单位目录已更新')
print(f'   总单位数: {len(unit_meta)}')
print(f'   总别名数: {len(alias_to_unit)}')

# 验证政府和办事处是否已添加
print('\n验证关键单位:')
for key_unit in ['政府', '办事处']:
    if key_unit in alias_to_unit:
        print(f'  ✅ {key_unit}: 已添加 (样本数: {unit_meta[key_unit]["sample_count"]})')
    else:
        print(f'  ❌ {key_unit}: 仍然缺失')
