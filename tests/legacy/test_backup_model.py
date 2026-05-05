"""
测试备份模型 final_model_fgm_master_smoke2 的预测效果

NOTE: 此文件为 legacy 测试，依赖本地模型文件，仅作手动运行参考。
      不会在 pytest 自动发现中执行（通过 pytestmark 跳过）。
"""

import os

import pytest

pytestmark = pytest.mark.skip(reason="legacy 测试，依赖本地模型文件，仅手动运行")

import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_dir = "final_model_fgm_master_smoke2"
base_model_dir = r"C:\python\接诉即办\.venv\local_roberta_model"

print("=" * 60)
print("测试备份模型: final_model_fgm_master_smoke2")
print("=" * 60)

if not os.path.exists(f"{model_dir}/pytorch_model.bin"):
    pytest.skip(f"{model_dir}/pytorch_model.bin 不存在，跳过")

if not os.path.exists(f"{model_dir}/label_map.json"):
    pytest.skip(f"{model_dir}/label_map.json 不存在，跳过")

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n设备: {device}")

# 加载 label_map
with open(f"{model_dir}/label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

print(f"类别数: {len(label_map)}")

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
model = AutoModelForSequenceClassification.from_pretrained(
    model_dir,
    num_labels=len(label_map),
    ignore_mismatched_sizes=True
).to(device)
model.eval()

print("\n模型加载成功！")

# 测试案例
test_cases = [
    {
        "tag": "政策咨询",
        "title": "咨询区政府关于人才引进政策",
        "body": "您好，我想咨询一下朝阳区政府关于高层次人才引进的相关政策和补贴标准。"
    },
    {
        "tag": "建议",
        "title": "建议街道办事处加强社区管理",
        "body": "希望街道办事处能够加强对社区环境的管理，定期组织清洁活动。"
    },
    {
        "tag": "投诉",
        "title": "反映政府部门办事效率问题",
        "body": "在区政府办理业务时，工作人员态度冷漠，办事效率低下，希望改进。"
    },
    {
        "tag": "咨询",
        "title": "咨询办事处管辖范围",
        "body": "请问我们小区属于哪个街道办事处管辖？需要办理相关手续应该去哪里？"
    }
]

print("\n" + "=" * 60)
print("预测结果测试")
print("=" * 60)

for i, case in enumerate(test_cases, 1):
    print(f"\n【测试案例 {i}】")
    print(f"标签: {case['tag']}")
    print(f"标题: {case['title']}")
    
    # 进行预测
    text = f"【{case['tag']}】{case['title']}。{case['body']}"
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
    
    probs = torch.softmax(logits, dim=-1)[0]
    
    # 获取 Top 10
    values, indices = torch.topk(probs, k=10)
    
    print(f"\n预测结果 (Top 10):")
    has_target = False
    for j, (value, index) in enumerate(zip(values.cpu().tolist(), indices.cpu().tolist()), 1):
        unit = label_map[str(index)]
        confidence = value * 100
        
        # 高亮显示政府和办事处
        if '政府' in unit or '办事处' in unit or '街道办' in unit:
            print(f"  {j}. ✅ {unit}: {confidence:.2f}%")
            has_target = True
        else:
            print(f"  {j}. {unit}: {confidence:.2f}%")
    
    if not has_target:
        print(f"\n  ⚠️  Top 10 中未包含政府/办事处类别")
        
        # 查找政府和办事处的排名
        all_predictions = []
        for idx, prob in enumerate(probs.cpu().tolist()):
            unit = label_map[str(idx)]
            all_predictions.append({"unit": unit, "confidence": prob * 100, "rank": 0})
        
        all_predictions.sort(key=lambda x: -x["confidence"])
        for rank, pred in enumerate(all_predictions, 1):
            pred["rank"] = rank
        
        print(f"\n  关键类别排名:")
        for target in ["政府", "办事处", "街道办"]:
            found = [p for p in all_predictions if target in p["unit"]]
            if found:
                for pred in found[:1]:
                    print(f"    {pred['unit']}: 排名 #{pred['rank']}, 置信度 {pred['confidence']:.2f}%")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
