# 训练产物接入说明

这份说明基于你当前的两段训练代码整理，目标是回答两件事：
1. 训练完成后，服务端到底应该读取哪些文件。
2. 怎样快速判断“训练成功”与“能被服务加载”是不是同一件事。

## 1. 官方回复单位分类模型

训练代码会把产物写到 [final_model_fgm](C:/Users/28414/PycharmProjects/接诉即办项目/final_model_fgm)。

按你当前脚本，稳定会产出的核心文件应该是：
- `pytorch_model.bin`
- `label_map.json`
- tokenizer 文件，如 `tokenizer_config.json`、`special_tokens_map.json`、`vocab.txt` 或 `tokenizer.json`

需要特别注意：
你当前分类训练脚本没有调用 `model.save_pretrained(save_dir)`，所以默认不会自动写出 `config.json`。

这会带来一个差别：
- 原始 [app.py](C:/Users/28414/PycharmProjects/接诉即办项目/app.py) 的严格加载方式，更偏向于把 `final_model_fgm` 当成完整 Hugging Face 模型目录，所以它更希望目录里有 `config.json`。
- 现在新增的增强版兼容加载器，会在 `config.json` 不存在时，改用 `CLASSIFIER_BASE_MODEL` 提供基础模型结构，再加载 `pytorch_model.bin`。

因此，分类模型有两种“成功”定义：
- 训练成功：`pytorch_model.bin + label_map.json + tokenizer 文件` 已经落盘。
- 原始服务严格可用：在上面的基础上，还要有 `config.json`。
- 增强版服务可用：在训练成功的基础上，再保证 `CLASSIFIER_BASE_MODEL` 可访问。

## 2. 回复生成模型

训练代码会把 LoRA 产物写到 `OUTPUT_DIR`，也就是默认的 `C:\Users\28414\Desktop\qwen_reply_model`。

服务端真正运行时需要：
- 基础模型目录可读：`GENERATOR_BASE_MODEL`
- LoRA 目录存在：`GENERATOR_LORA_DIR`
- `adapter_config.json`
- `adapter_model.safetensors` 或 `adapter_model.bin`

训练脚本里 `trainer.save_model(OUTPUT_DIR)` 成功执行后，通常就会把上面两类 LoRA 文件写到根目录。

如果你只看到：
- `checkpoint-200`
- `checkpoint-400`
- `checkpoint-600`

但根目录没有 `adapter_config.json`，通常说明：
- 训练中断了，或者
- 最终导出还没执行到，或者
- 输出目录不是服务当前读取的那个目录

## 3. 当前项目新增的检测方式

我已经新增了：
- [check_training_artifacts.py](C:/Users/28414/PycharmProjects/接诉即办项目/tools/check_training_artifacts.py)
- [model_artifacts.py](C:/Users/28414/PycharmProjects/接诉即办项目/enhancements/model_artifacts.py)
- [classifier_runtime.py](C:/Users/28414/PycharmProjects/接诉即办项目/enhancements/classifier_runtime.py)

你现在可以用：

```powershell
& C:\Users\28414\anaconda3\envs\qwen_env\python.exe C:\Users\28414\PycharmProjects\接诉即办项目\tools\check_training_artifacts.py
```

它会返回：
- 分类模型是否只是“训练产物已生成”
- 分类模型是否能被原始严格加载器使用
- 分类模型是否能被增强版兼容加载器使用
- 生成模型的 LoRA 根目录是否完整
- 是否只留下了 checkpoint 但没有最终导出

## 4. 现在增强版服务的变化

[app_enhanced.py](C:/Users/28414/PycharmProjects/接诉即办项目/app_enhanced.py) 已经支持：
- 当 `final_model_fgm` 没有 `config.json` 时，只要 `CLASSIFIER_BASE_MODEL` 可访问，也能兼容加载分类模型。

[app_enhanced_stable.py](C:/Users/28414/PycharmProjects/接诉即办项目/app_enhanced_stable.py) 新增了：
- `/api/health/live`
- `/api/health/ready`
- `/api/health/training`

其中 `/api/health/training` 会直接返回训练产物检查明细。

## 5. 你当前最值得盯的检查点

1. 分类模型训练完后，先看 `final_model_fgm` 里有没有：
   - `pytorch_model.bin`
   - `label_map.json`
   - tokenizer 文件
2. 如果你想兼容原始 `app.py`，最好额外补出 `config.json`。
3. 生成模型训练完后，先看 `qwen_reply_model` 根目录有没有：
   - `adapter_config.json`
   - `adapter_model.safetensors` 或 `adapter_model.bin`
4. 如果只有 `checkpoint-*`，说明还不能算真正接入成功。
