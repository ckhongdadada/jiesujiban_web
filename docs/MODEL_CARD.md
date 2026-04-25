# MODEL_CARD — final_model_hybrid_v3_32cls

> 当前项目主分类模型，由 `RuntimeConfig.classifier_model_dir` 自动指向。

## 基本信息

| 字段 | 值 |
|------|------|
| 模型架构 | BERT + TextCNN + Attention + TF-IDF 融合 |
| 基座模型 | RoBERTa-wwm-ext (chinese-roberta-wwm-ext) |
| 分类路线 | `hybrid_rankaware` |
| 训练脚本 | `src/jsjb/unit_classifier/training/hybrid.py` |
| 启动脚本 | `scripts/training/unit_classifier/run_hybrid_v3_32cls.py` |

## 训练数据

| 字段 | 值 |
|------|------|
| 数据版本 | v3 (留言板合并数据_最终训练版_v3.xlsx) |
| 样本数 | ~14,535 |
| 类别数 | 77 (当前)；目标 32 (待精简) |
| 标签体系 | 官方回复单位 (经规范化与合并处理) |
| 数据处理 | 政府/街道办合并、地区前缀去除、留言标签取中间值 |

## 损失函数

| 组件 | 说明 |
|------|------|
| Hard Loss | RankAwareLoss (Focal + Top-K 排名惩罚) |
| Soft Loss | EMA 教师模型 KL 蒸馏 (Top-K 掩码) |
| 权重比例 | hard_weight=0.82, distill_weight=0.18 |
| 蒸馏温度 | temperature=2.0 |
| 动态归一化 | distill_loss 自动缩放到与 hard_loss 同量级 |

## 训练配置

| 参数 | 值 |
|------|------|
| FGM 对抗训练 | 可选开关 |
| EMA decay | 0.999 |
| Soft Label Bank | 启用 |
| 学习率 | 见 best_params.json |
| Batch Size | 见 best_params.json |
| 最大序列长度 | 256 |
| TF-IDF 维度 | 3000 |
| TF-IDF 隐层 | 64 |

## 续训说明

- **允许续训**: 是，通过 `--init-from` 参数加载已有权重
- **断点恢复**: 支持，自动检测 `training_state.pt`
- **注意**: 续训时确保 `label_map.json` 与训练数据类别一致

## 产物文件

| 文件 | 说明 |
|------|------|
| `pytorch_model.bin` | 模型权重 |
| `label_map.json` | 标签 ID → 单位名称映射 |
| `model_meta.json` | 模型元信息 (架构、TF-IDF 配置等) |
| `tfidf_vectorizer.joblib` | TF-IDF 向量化器 |
| `training_state.pt` | 训练断点 (如存在) |
| `training_log.csv` | 训练日志 |

## 与其他模型的关系

| 模型目录 | 状态 | 说明 |
|----------|------|------|
| `final_model_hybrid_v3_32cls` | **当前主线** | 本文档描述的模型 |
| `final_model_hybrid` | 历史版本 | v2 数据训练，可归档 |
| `final_model_fgm` | 历史版本 | 纯 FGM 无蒸馏，可归档 |
| `final_model_distillation` | 历史版本 | 完整自蒸馏，可归档 |

## 更新记录

| 日期 | 变更 |
|------|------|
| 2026-04-25 | 创建 MODEL_CARD，记录当前主线模型信息 |
