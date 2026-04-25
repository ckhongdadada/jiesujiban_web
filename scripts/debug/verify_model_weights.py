"""
验证模型权重是否正确加载
"""
import torch
import json

# 直接加载权重文件
weights = torch.load("final_model_fgm/pytorch_model.bin", map_location="cpu")

print("=" * 60)
print("权重文件内容检查")
print("=" * 60)

print(f"\n权重文件中的键数量: {len(weights.keys())}")
print(f"\n前10个键:")
for i, key in enumerate(list(weights.keys())[:10]):
    print(f"  {i+1}. {key}: {weights[key].shape}")

# 检查分类器层的权重
if "classifier.weight" in weights:
    classifier_weight = weights["classifier.weight"]
    print(f"\n分类器权重形状: {classifier_weight.shape}")
    print(f"  - 类别数: {classifier_weight.shape[0]}")
    print(f"  - 特征维度: {classifier_weight.shape[1]}")
    
    # 检查权重是否全为零（未训练的标志）
    is_zero = torch.all(classifier_weight == 0).item()
    print(f"  - 权重是否全为零: {is_zero}")
    
    # 检查权重的统计信息
    print(f"  - 权重均值: {classifier_weight.mean().item():.6f}")
    print(f"  - 权重标准差: {classifier_weight.std().item():.6f}")
    print(f"  - 权重最小值: {classifier_weight.min().item():.6f}")
    print(f"  - 权重最大值: {classifier_weight.max().item():.6f}")

# 加载 label_map 验证类别数
with open("final_model_fgm/label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

print(f"\nLabel Map 中的类别数: {len(label_map)}")

# 检查政府和办事处的索引
gov_idx = None
office_idx = None
for idx, label in label_map.items():
    if label == "政府":
        gov_idx = int(idx)
    elif label == "办事处":
        office_idx = int(idx)

if gov_idx is not None:
    print(f"\n政府类别索引: {gov_idx}")
    if "classifier.weight" in weights:
        gov_weights = weights["classifier.weight"][gov_idx]
        print(f"  - 权重均值: {gov_weights.mean().item():.6f}")
        print(f"  - 权重标准差: {gov_weights.std().item():.6f}")

if office_idx is not None:
    print(f"\n办事处类别索引: {office_idx}")
    if "classifier.weight" in weights:
        office_weights = weights["classifier.weight"][office_idx]
        print(f"  - 权重均值: {office_weights.mean().item():.6f}")
        print(f"  - 权重标准差: {office_weights.std().item():.6f}")
