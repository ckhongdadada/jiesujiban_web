import torch
with open("gpu_check_result.txt", "w") as f:
    f.write(f"CUDA可用: {torch.cuda.is_available()}\n")
    if torch.cuda.is_available():
        f.write(f"CUDA版本: {torch.version.cuda}\n")
        f.write(f"GPU数量: {torch.cuda.device_count()}\n")
        f.write(f"GPU名称: {torch.cuda.get_device_name(0)}\n")
    else:
        f.write("CUDA不可用，将使用CPU训练\n")
print("检查完成，结果已保存到 gpu_check_result.txt")
