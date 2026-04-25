# 接诉即办智能服务系统

这是一个面向政务留言的智能分析与回复生成系统。系统围绕一条主流程工作：输入市民留言，识别发生地点，预测回复单位，检索相关政策/案例，最后由 Qwen + LoRA 生成政务回复，并记录用户反馈用于后续主动学习和知识更新。

## 当前主入口

```bash
python app.py
```

也可以使用：

```bash
python scripts/app/run_server.py
```

服务默认运行在：

```text
http://127.0.0.1:5000
```

根目录 `app.py` 只是兼容启动壳，真实 Web 应用入口在：

```text
src/jsjb/web/app.py
src/jsjb/web/app_factory.py
```

## 主要接口

- `GET /`：前端页面
- `GET /api/health`：综合健康检查
- `GET /api/health/live`：服务进程是否存活
- `GET /api/health/ready`：模型与依赖是否就绪
- `POST /api/analyze`：完整分析链路，包含地名识别、单位分类、RAG 检索、回复生成
- `POST /api/feedback`：保存用户对回复或 RAG 材料的反馈
- `GET /api/feedback/error-analysis/<id>`：查看某条反馈的生成错误分析
- `GET /api/knowledge-graph/review-candidates`：查看待审核知识候选
- `POST /api/knowledge-graph/review/<id>`：审核知识候选

## 重构后的目录结构

```text
src/jsjb/
  web/                 Web 层，包含 app factory 与路由
  location/            北京地名识别、区县映射、消歧
  retrieval/           RAG 检索，主线为 BGE/TF-IDF 兼容检索
  unit_classifier/     回复单位分类，包含线上推理、模型结构、训练逻辑、验收
  reply_generation/    Qwen + LoRA 回复生成、事实验证、错误分析
  feedback/            SQLite 用户反馈库
  active_learning/     样本收集、标注管理、增量训练入口
  knowledge/           知识图谱、结构化知识库、知识更新与融合
  core/                配置、路径、日志、校验、缓存、模型预加载等公共能力

scripts/               可执行脚本入口，核心逻辑应沉淀在 src/jsjb/
configs/               配置文件
checkpoints/           模型权重和 LoRA 适配器
outputs/               报告、比赛材料、运行产物
legacy/                已归档旧入口或旧实验实现
tests/                 测试代码
```

## 模型与核心组件

- 地名识别：规则词典 + 运行时地点库 + 行政区映射，主实现位于 `src/jsjb/location/`。
- 单位分类：当前演示主线为 `BERT/RoBERTa + CNN + Attention + TF-IDF`，推理入口位于 `src/jsjb/unit_classifier/runtime.py`。
- RAG 检索：主线位于 `src/jsjb/retrieval/bge_retriever.py`，支持 BGE 向量检索并在不可用时回退到 TF-IDF。
- 回复生成：`Qwen2.5-1.5B-Instruct + LoRA`，入口位于 `src/jsjb/reply_generation/qwen_lora.py`。
- 反馈与主动学习：反馈存入 SQLite，后续可进入错误分析、知识候选审核与主动学习流程。

## 配置与模型路径

- 应用配置：`configs/app/config.json`
- 环境变量示例：`configs/app/.env.example`
- 分类模型：`checkpoints/classifier/`
- 生成基础模型：`checkpoints/generator/base_models/`
- 生成 LoRA：`checkpoints/generator/lora/`

路径获取统一通过：

```text
src/jsjb/core/paths.py
src/jsjb/core/config.py
```

## 兼容说明

`enhancements/` 兼容层已经从主仓库结构中移除。当前正式代码统一从 `src/jsjb/*` 导入；旧版 `app_enhanced.py` 和早期实现已归档到 `legacy/`。

## 最小验证

```bash
python -m py_compile app.py src/jsjb/web/app.py
python app.py
```

启动后访问 `/api/health`，再用前端或 `/api/analyze` 跑一条真实留言样例。
