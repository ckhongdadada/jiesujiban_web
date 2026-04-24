"""
分析训练数据中政府和办事处类别的样本
"""
import pandas as pd

# 读取训练数据
df = pd.read_excel(r'C:\Users\28414\Desktop\留言板合并数据.xlsx')

print("=" * 60)
print("政府类别样本分析")
print("=" * 60)

# 筛选政府类别
gov_samples = df[df['官方回复单位'] == '政府']
print(f"\n总样本数: {len(gov_samples)}")

print("\n【前10个样本的标题】")
for i, row in gov_samples.head(10).iterrows():
    print(f"{i+1}. [{row['留言标签']}] {row['留言标题'][:50]}")

print("\n【标签分布】")
tag_dist = gov_samples['留言标签'].value_counts()
print(tag_dist.head(10))

print("\n" + "=" * 60)
print("办事处类别样本分析")
print("=" * 60)

# 筛选办事处类别
office_samples = df[df['官方回复单位'] == '办事处']
print(f"\n总样本数: {len(office_samples)}")

print("\n【前10个样本的标题】")
for i, row in office_samples.head(10).iterrows():
    print(f"{i+1}. [{row['留言标签']}] {row['留言标题'][:50]}")

print("\n【标签分布】")
tag_dist = office_samples['留言标签'].value_counts()
print(tag_dist.head(10))

# 检查是否有更具体的政府/办事处单位
print("\n" + "=" * 60)
print("包含'政府'的所有单位")
print("=" * 60)
all_units = df['官方回复单位'].unique()
gov_units = [u for u in all_units if '政府' in str(u)]
print(f"找到 {len(gov_units)} 个包含'政府'的单位:")
for u in gov_units[:20]:
    count = len(df[df['官方回复单位'] == u])
    print(f"  {u}: {count} 样本")

print("\n" + "=" * 60)
print("包含'办事处'的所有单位")
print("=" * 60)
office_units = [u for u in all_units if '办事处' in str(u) or '街道办' in str(u)]
print(f"找到 {len(office_units)} 个包含'办事处'的单位:")
for u in office_units[:20]:
    count = len(df[df['官方回复单位'] == u])
    print(f"  {u}: {count} 样本")
