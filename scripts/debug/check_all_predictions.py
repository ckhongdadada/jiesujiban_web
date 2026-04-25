"""
检查所有类别的预测概率 - 找出政府和办事处的排名
"""
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = "final_model_fgm"
base_model_dir = r"C:\python\接诉即办\.venv\local_roberta_model"

# 加载 label_map
with open(f"{model_dir}/label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
model = AutoModelForSequenceClassification.from_pretrained(
    model_dir,
    num_labels=len(label_map),
    ignore_mismatched_sizes=True
).to(device)
model.eval()

# 测试案例
test_case = {
    "tag": "政策咨询",
    "title": "咨询区政府关于人才引进政策",
    "body": "您好，我想咨询一下朝阳区政府关于高层次人才引进的相关政策和补贴标准。"
}

text = f"【{test_case['tag']}】{test_case['title']}。{test_case['body']}"
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

# 获取所有预测结果
all_predictions = []
for idx, prob in enumerate(probs.cpu().tolist()):
    unit = label_map[str(idx)]
    all_predictions.append({
        "unit": unit,
        "confidence": prob * 100,
        "rank": 0
    })

# 按置信度排序
all_predictions.sort(key=lambda x: -x["confidence"])
for i, pred in enumerate(all_predictions, 1):
    pred["rank"] = i

print("=" * 60)
print("完整预测结果分析")
print("=" * 60)
print(f"\n测试案例: {test_case['title']}")

print("\n【Top 10 预测结果】")
for pred in all_predictions[:10]:
    print(f"  {pred['rank']}. {pred['unit']}: {pred['confidence']:.2f}%")

print("\n【关键类别排名】")
target_units = ["政府", "办事处", "街道办"]
for target in target_units:
    found = [p for p in all_predictions if target in p["unit"]]
    if found:
        for pred in found[:3]:  # 显示前3个匹配
            print(f"  {pred['unit']}: 排名 #{pred['rank']}, 置信度 {pred['confidence']:.2f}%")
    else:
        print(f"  {target}: 未找到")

print("\n【Bottom 10 预测结果】")
for pred in all_predictions[-10:]:
    print(f"  {pred['rank']}. {pred['unit']}: {pred['confidence']:.4f}%")
