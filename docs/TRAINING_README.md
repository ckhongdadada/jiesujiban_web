# 训练脚本说明

我已经把你贴过来的两段训练代码整理进项目，并补了训练后自动验收和自动写报告。

## 文件
- [train_unit_classifier_fgm.py](C:/Users/28414/PycharmProjects/接诉即办项目/training/train_unit_classifier_fgm.py)
- [train_qwen_reply_lora.py](C:/Users/28414/PycharmProjects/接诉即办项目/training/train_qwen_reply_lora.py)
- [post_training_acceptance.py](C:/Users/28414/PycharmProjects/接诉即办项目/tools/post_training_acceptance.py)

## 自动报告
训练完成后会自动把验收结果写到：
- [training_reports](C:/Users/28414/PycharmProjects/接诉即办项目/data/training_reports)

写出两类文件：
- `classifier_acceptance_latest.json` / `generator_acceptance_latest.json`
- 带时间戳的归档文件，例如 `classifier_acceptance_20260324_220000.json`

## 现在和原始代码的关键差别

### 1. 分类训练脚本会额外保存 `config.json`
这一步很重要，因为它让训练输出更接近完整 Hugging Face 目录。

当前脚本在保存最佳模型时会同时写出：
- `pytorch_model.bin`
- `label_map.json`
- tokenizer 文件
- `config.json`

### 2. 两个训练脚本结束后都会自动做一次接入验收并写报告
训练跑完以后，不只是“训练成功”，还会检查：
- 服务端需要的文件是不是都齐了
- 这些文件能不能真的被当前项目读取

## 常用命令

### 分类训练
```powershell
& C:\Users\28414\anaconda3\envs\qwen_env\python.exe C:\Users\28414\PycharmProjects\接诉即办项目\training\train_unit_classifier_fgm.py
```

### 生成训练
```powershell
& C:\Users\28414\anaconda3\envs\qwen_env\python.exe C:\Users\28414\PycharmProjects\接诉即办项目\training\train_qwen_reply_lora.py
```

### 单独验收
```powershell
& C:\Users\28414\anaconda3\envs\qwen_env\python.exe C:\Users\28414\PycharmProjects\接诉即办项目\tools\post_training_acceptance.py --target all
```
