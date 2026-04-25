"""
测试预测功能 - 验证政府和办事处是否能正常出现在预测结果中
"""
import torch
from enhancements.classifier_runtime import ClassifierRuntime

# 初始化分类器
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = ClassifierRuntime(
    model_dir="final_model_fgm",
    base_model_dir=r"C:\python\接诉即办\.venv\local_roberta_model",
    device=device
)

# 测试用例 - 应该预测为政府的案例
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

print("=" * 60)
print("预测结果测试")
print("=" * 60)

for i, case in enumerate(test_cases, 1):
    print(f"\n【测试案例 {i}】")
    print(f"标签: {case['tag']}")
    print(f"标题: {case['title']}")
    print(f"正文: {case['body'][:50]}...")
    
    # 进行预测
    results = classifier.predict(
        tag=case['tag'],
        title=case['title'],
        body=case['body'],
        top_k=5
    )
    
    print(f"\n预测结果 (Top 5):")
    for j, result in enumerate(results, 1):
        unit = result['unit']
        confidence = result['confidence']
        # 高亮显示政府和办事处
        if '政府' in unit or '办事处' in unit:
            print(f"  {j}. ✅ {unit}: {confidence}%")
        else:
            print(f"  {j}. {unit}: {confidence}%")
    
    # 检查是否包含政府或办事处
    has_gov = any('政府' in r['unit'] for r in results)
    has_office = any('办事处' in r['unit'] for r in results)
    
    if has_gov or has_office:
        print(f"  ✅ 包含目标类别")
    else:
        print(f"  ⚠️  未包含政府/办事处类别")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
