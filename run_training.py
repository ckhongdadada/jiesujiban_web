import subprocess
import sys
import pathlib

# 运行训练脚本并捕获输出
cmd = [
    sys.executable,
    "training/train_unit_classifier_fgm.py",
    "--use-cnn-attention",
    "--use-tfidf",
    "--batch-size", "8",
    "--grad-accum-steps", "2",
    "--disable-fgm",
    "--epochs", "1",
    "--save-dir", r"C:\Users\28414\PycharmProjects\接诉即办项目\data\models\classifier\bert_cnn_attn_test"
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

output = f"Return code: {result.returncode}\n\n"
output += f"STDOUT:\n{result.stdout}\n\n"
output += f"STDERR:\n{result.stderr}"

pathlib.Path("training_output.txt").write_text(output, encoding='utf-8')
print("Output saved to training_output.txt")
