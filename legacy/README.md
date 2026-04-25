# Legacy 目录说明

本目录存放项目早期版本的代码，**仅供归档参考，不再维护**。

## 目录结构

```
legacy/
├── app_enhanced.py              早期增强版 Flask 应用入口 (已被 src/jsjb/web/app.py 取代)
├── enhancements/                早期功能增强模块
│   ├── async_processor.py       异步批处理器 (已迁移至 src/jsjb/core/async_processor.py)
│   ├── config_manager.py        配置管理器 (已迁移至 src/jsjb/core/config.py)
│   ├── model_manager_basic.py   基础模型管理器 (已迁移至 src/jsjb/core/model_registry.py)
│   ├── model_manager_with_cache.py  带缓存的模型管理器 (已迁移至 src/jsjb/core/model_registry.py)
│   ├── rag_retriever.py.backup  RAG 检索器备份 (已迁移至 src/jsjb/retrieval/)
│   └── update_scheduler.py      更新调度器 (已迁移至 src/jsjb/core/)
└── examples/                    早期使用示例
    ├── disambiguation_cache_example.py     消歧缓存示例
    ├── optimized_usage_example.py          优化使用示例
    └── use_knowledge_graph_and_active_learning.py  知识图谱与主动学习示例
```

## 迁移对照表

| Legacy 文件 | 当前位置 |
|---|---|
| `app_enhanced.py` | `src/jsjb/web/app.py` |
| `enhancements/config_manager.py` | `src/jsjb/core/config.py` |
| `enhancements/model_manager_basic.py` | `src/jsjb/core/model_registry.py` |
| `enhancements/model_manager_with_cache.py` | `src/jsjb/core/model_registry.py` |
| `enhancements/rag_retriever.py.backup` | `src/jsjb/retrieval/` |
| `enhancements/async_processor.py` | `src/jsjb/core/` |
| `enhancements/update_scheduler.py` | `src/jsjb/core/` |

## 注意事项

- 本目录下的代码**不应被任何当前模块 import**
- 如需参考早期实现逻辑，请对照迁移表查找当前版本
- 示例代码可能依赖已变更的 API，运行前需手动适配
