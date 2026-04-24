import pandas as pd
import re
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from enhancements.unit_catalog import (
    normalize_unit_text,
    canonicalize_unit,
    load_unit_catalog,
    GENERIC_BAD_UNITS,
    DISTRICT_NAMES,
)

def extract_reply_units_v2(text):
    if pd.isna(text) or not isinstance(text, str):
        return None
    
    patterns = [
        r'关于您反映的问题[，,]?\s*(.+?)回复[：:]',
        r'经\s*(.+?)\s*核实',
        r'(.+?)回复[：:]\s*经',
        r'回复单位[：:]\s*(.+?)[\s\n]',
        r'承办单位[：:]\s*(.+?)[\s\n]',
        r'办理单位[：:]\s*(.+?)[\s\n]',
    ]
    
    units = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                for m in match:
                    if m and len(m.strip()) > 1:
                        units.append(m.strip())
            else:
                if match and len(match.strip()) > 1:
                    units.append(match.strip())
    
    if units:
        for u in units:
            normalized = normalize_unit_text(u)
            if normalized and normalized not in GENERIC_BAD_UNITS:
                return normalized
    
    return None

def normalize_unit_with_catalog(unit, catalog):
    if pd.isna(unit) or not unit:
        return None
    
    unit_str = str(unit).strip()
    
    if '、' in unit_str or ',' in unit_str or '，' in unit_str:
        separators = ['、', ',', '，']
        parts = [unit_str]
        for sep in separators:
            new_parts = []
            for p in parts:
                new_parts.extend(p.split(sep))
            parts = new_parts
        parts = [p.strip() for p in parts if p.strip()]
        
        for part in parts:
            normalized = normalize_unit_text(part)
            if normalized and normalized not in GENERIC_BAD_UNITS:
                canonical = canonicalize_unit(normalized, catalog)
                if canonical and canonical not in GENERIC_BAD_UNITS:
                    return canonical
    
    normalized = normalize_unit_text(unit_str)
    if not normalized or normalized in GENERIC_BAD_UNITS:
        return None
    
    canonical = canonicalize_unit(normalized, catalog)
    if canonical and canonical not in GENERIC_BAD_UNITS:
        return canonical
    
    return None

DISTRICT_PREFIXES = [
    "东城区", "西城区", "朝阳区", "丰台区", "石景山区", "海淀区",
    "门头沟区", "房山区", "通州区", "顺义区", "昌平区", "大兴区",
    "怀柔区", "平谷区", "密云区", "延庆区",
    "东城", "西城", "朝阳", "丰台", "石景山", "海淀",
    "门头沟", "房山", "通州", "顺义", "昌平", "大兴",
    "怀柔", "平谷", "密云", "延庆",
]

CITY_PREFIXES = ["市", "北京市"]

KEEP_FULL_NAME_UNITS = {
    "北京经济技术开发区",
    "北京公共交通控股(集团)有限公司",
    "北京市地铁运营有限公司",
    "北京市基础设施投资有限公司",
    "北京市市民热线服务中心房山分中心",
}

MERGEABLE_UNIT_SUFFIXES = [
    "交通局", "交通支队", "教委", "住建委", "城管委", "城市管理委",
    "住房城市建设委", "市场监管局", "卫健委", "园林绿化局",
    "生态环境局", "水务局", "人社局", "民政局", "文旅局",
    "应急管理局", "审计局", "统计局", "发改委", "商务局",
    "农业农村局", "财政局", "司法局", "体育局", "信访办",
    "城管执法局", "房屋管理局", "房管局",
]

STREET_OFFICE_PATTERNS = [
    "街道办事处", "街道办", "地区办事处", "地区办",
]

def merge_government_and_street(unit):
    if pd.isna(unit) or not unit:
        return unit
    
    unit_str = str(unit).strip()
    
    if unit_str == "政府":
        return "政府/街道办"
    
    if unit_str == "办事处":
        return "政府/街道办"
    
    for pattern in STREET_OFFICE_PATTERNS:
        if unit_str.endswith(pattern):
            return "政府/街道办"
    
    if "街道" in unit_str and "办" in unit_str:
        return "政府/街道办"
    
    return unit_str

def remove_district_prefix(unit):
    if pd.isna(unit) or not unit:
        return unit
    
    unit_str = str(unit).strip()
    
    if unit_str in KEEP_FULL_NAME_UNITS:
        return unit_str
    
    for pattern in STREET_OFFICE_PATTERNS:
        if unit_str.endswith(pattern):
            return "政府/街道办"
    
    for prefix in DISTRICT_PREFIXES:
        if unit_str.startswith(prefix):
            remaining = unit_str[len(prefix):]
            if remaining and any(remaining.endswith(suffix) for suffix in MERGEABLE_UNIT_SUFFIXES):
                return remaining
            for pattern in STREET_OFFICE_PATTERNS:
                if remaining.endswith(pattern):
                    return "政府/街道办"
            break
    
    for prefix in CITY_PREFIXES:
        if unit_str.startswith(prefix):
            remaining = unit_str[len(prefix):]
            if remaining and any(remaining.endswith(suffix) for suffix in MERGEABLE_UNIT_SUFFIXES):
                return remaining
            break
    
    return unit_str

def infer_district_from_text(text, source_info=""):
    if pd.isna(text):
        text = ""
    if pd.isna(source_info):
        source_info = ""
    
    combined = str(text) + " " + str(source_info)
    
    for district in DISTRICT_NAMES:
        if district in combined:
            return district
    
    return None

def process_new_crawler_file(file_path, catalog):
    print(f"\n处理: {file_path}")
    df = pd.read_excel(file_path)
    print(f"  原始行数: {len(df)}")
    
    df['官方回复单位_规范化'] = df['官方回复单位'].apply(
        lambda x: normalize_unit_with_catalog(x, catalog)
    )
    
    null_count = df['官方回复单位_规范化'].isna().sum()
    print(f"  规范化后空值: {null_count}")
    
    invalid_mask = df['官方回复正文'].fillna('').str.contains('已转相关部门处理', na=False)
    df.loc[invalid_mask, '官方回复单位_规范化'] = None
    print(f"  过滤'已转相关部门处理'后空值: {df['官方回复单位_规范化'].isna().sum()}")
    
    return df

def process_extracted_file(file_path, catalog):
    print(f"\n处理: {file_path}")
    df = pd.read_excel(file_path)
    print(f"  原始行数: {len(df)}")
    
    if '官方回复单位' in df.columns:
        df['官方回复单位_规范化'] = df['官方回复单位'].apply(
            lambda x: normalize_unit_with_catalog(x, catalog)
        )
    else:
        df['官方回复单位_规范化'] = df['官方回复正文'].apply(extract_reply_units_v2)
        df['官方回复单位_规范化'] = df['官方回复单位_规范化'].apply(
            lambda x: canonicalize_unit(x, catalog) if x else None
        )
    
    invalid_mask = df['官方回复正文'].fillna('').str.contains('已转相关部门处理', na=False)
    df.loc[invalid_mask, '官方回复单位_规范化'] = None
    print(f"  规范化后空值: {df['官方回复单位_规范化'].isna().sum()}")
    
    return df

def process_current_training_file(file_path, catalog):
    print(f"\n处理: {file_path}")
    df = pd.read_excel(file_path)
    print(f"  原始行数: {len(df)}")
    
    df['官方回复单位_规范化'] = df['官方回复单位'].apply(
        lambda x: normalize_unit_with_catalog(x, catalog)
    )
    print(f"  规范化后空值: {df['官方回复单位_规范化'].isna().sum()}")
    
    return df

def main():
    catalog = load_unit_catalog()
    print(f"加载单位目录，别名数: {len(catalog.get('alias_to_unit', {}))}")
    
    all_data = []
    
    crawler_file = r'C:\Users\28414\Desktop\副本留言板爬虫20240101-20250423.xlsx'
    if os.path.exists(crawler_file):
        df1 = process_new_crawler_file(crawler_file, catalog)
        all_data.append(df1)
    
    extracted_file = r'C:\Users\28414\Desktop\北京各区留言板-汇总\留言板合并结果_提取单位.xlsx'
    if os.path.exists(extracted_file):
        df2 = process_extracted_file(extracted_file, catalog)
        all_data.append(df2)
    
    current_file = r'C:\Users\28414\Desktop\留言板合并数据.xlsx'
    if os.path.exists(current_file):
        df3 = process_current_training_file(current_file, catalog)
        all_data.append(df3)
    
    if not all_data:
        print("没有找到任何数据文件！")
        return
    
    print("\n" + "="*60)
    print("合并数据...")
    
    merged_df = pd.concat(all_data, ignore_index=True)
    print(f"合并后总行数: {len(merged_df)}")
    
    valid_df = merged_df[merged_df['官方回复单位_规范化'].notna()].copy()
    print(f"有效数据行数: {len(valid_df)}")
    
    valid_df['官方回复单位_规范化'] = valid_df['官方回复单位_规范化'].apply(remove_district_prefix)
    print("已去掉区名前缀，合并相似单位...")
    
    valid_df['官方回复单位_规范化'] = valid_df['官方回复单位_规范化'].apply(merge_government_and_street)
    print("已合并政府和街道办为'政府/街道办'...")
    
    unit_counts = valid_df['官方回复单位_规范化'].value_counts()
    print(f"唯一单位数: {len(unit_counts)}")
    
    print("\nTop 30 单位:")
    print(unit_counts.head(30))
    
    sample_threshold = 5
    rare_units = unit_counts[unit_counts < sample_threshold]
    print(f"\n样本数 < {sample_threshold} 的单位数: {len(rare_units)}")
    
    output_columns = []
    if '留言标签' in valid_df.columns:
        output_columns.append('留言标签')
    output_columns.extend(['留言标题', '留言正文', '官方回复单位_规范化'])
    
    output_df = valid_df[output_columns].copy()
    output_df.columns = output_columns[:-1] + ['官方回复单位']
    
    output_file = r'C:\Users\28414\Desktop\留言板合并数据_增强版.xlsx'
    output_df.to_excel(output_file, index=False)
    print(f"\n保存到: {output_file}")
    
    print("\n" + "="*60)
    print("数据统计:")
    print(f"  总样本数: {len(output_df)}")
    print(f"  唯一类别数: {len(unit_counts)}")
    print(f"  最大类样本数: {unit_counts.iloc[0]} ({unit_counts.index[0]})")
    print(f"  最小类样本数: {unit_counts.iloc[-1]} ({unit_counts.index[-1]})")
    
    class_counts = unit_counts.value_counts().sort_index()
    print("\n类别样本分布:")
    for count, num_classes in class_counts.items():
        if count <= 10 or count >= 100:
            print(f"  {count}个样本: {num_classes}个类别")

if __name__ == '__main__':
    main()
