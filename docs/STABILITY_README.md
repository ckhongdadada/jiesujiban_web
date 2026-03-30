# 稳定版旁路入口

新增文件：
- [app_enhanced_stable.py](C:/Users/28414/PycharmProjects/接诉即办项目/app_enhanced_stable.py)
- [runtime_config.py](C:/Users/28414/PycharmProjects/接诉即办项目/enhancements/runtime_config.py)
- [startup_checks.py](C:/Users/28414/PycharmProjects/接诉即办项目/enhancements/startup_checks.py)

## 作用

在不修改原始 `app.py` 的前提下，新增：
- 配置外置化
- 启动自检
- 更清晰的 `live/ready` 健康检查

## 运行

```powershell
& C:\Users\28414\anaconda3\envs\qwen_env\python.exe C:\Users\28414\PycharmProjects\接诉即办项目\app_enhanced_stable.py
```

默认端口是 `5002`。

## 可覆盖的环境变量

- `CLASSIFIER_MODEL_DIR`
- `GENERATOR_BASE_MODEL`
- `GENERATOR_LORA_DIR`
- `ENHANCED_APP_PORT`
- `USE_CURATED_ALIASES`
