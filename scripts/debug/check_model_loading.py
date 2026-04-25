"""
检查模型加载路径
"""
from enhancements.model_artifacts import inspect_classifier_artifacts

model_dir = "final_model_fgm"
base_model_dir = r"C:\python\接诉即办\.venv\local_roberta_model"

status = inspect_classifier_artifacts(model_dir, base_model_dir)

print("=" * 60)
print("模型加载路径检查")
print("=" * 60)

print(f"\nModel Dir: {model_dir}")
print(f"Base Model Dir: {base_model_dir}")

print(f"\n权重文件路径: {status['weights']['path']}")
print(f"权重文件存在: {status['weights']['exists']}")

print(f"\nConfig 路径: {status['config']['path']}")
print(f"Config 存在: {status['config']['exists']}")

print(f"\nTokenizer 来源: {status['tokenizer_source']}")

print(f"\n运行时就绪: {status['compatible_runtime_ready']}")

# 检查权重文件的实际大小和修改时间
import os
if status['weights']['exists']:
    weight_path = status['weights']['path']
    stat = os.stat(weight_path)
    print(f"\n权重文件大小: {stat.st_size / (1024**2):.2f} MB")
    import datetime
    print(f"最后修改时间: {datetime.datetime.fromtimestamp(stat.st_mtime)}")
