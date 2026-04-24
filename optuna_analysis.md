# Optuna 超参数搜索脚本分析

## 文件位置
`tools/optuna_search_classifier.py`

## 功能概述
使用 Optuna 框架自动搜索分类器的最佳超参数组合。

---

## 合理性分析

### ✅ 优点

1. **搜索空间设计合理**
   - `loss_type`: 三种损失函数（cross_entropy, focal, rank_aware）
   - `learning_rate`: 1e-5 到 5e-5（对数尺度），适合 BERT 微调
   - `class_weight_power`: 0.3 到 0.8，用于处理类别不平衡
   - `focal_gamma`: 0.5 到 2.0（当使用 focal loss 时）
   - `rank_penalty`: 0.1 到 0.5（当使用 rank_aware loss 时）

2. **使用小样本快速验证**
   - `--sample-size`: 默认 4000 样本
   - `--epochs`: 默认 3 个 epoch
   - 大大加快搜索速度，适合快速迭代

3. **采样器和剪枝器配置合理**
   - `TPESampler`: Tree-structured Parzen Estimator，适合中小规模搜索
   - `MedianPruner`: 提前终止表现不佳的 trial，节省时间

4. **结果保存完整**
   - JSON 格式保存最佳参数和所有 trial 结果
   - CSV 格式保存 trial 历史，便于分析
   - 支持归档（带时间戳）

5. **资源管理**
   - 自动清理 trial 产物（除非指定 `--keep-trial-artifacts`）
   - 清理 CUDA 缓存

6. **灵活的评估指标**
   - 支持多种指标：macro_f1, weighted_f1, top1_acc, topk_acc
   - 可根据业务需求选择

---

### ⚠️ 潜在问题

1. **固定禁用了一些功能**
   ```python
   args.disable_fgm = False  # FGM 对抗训练始终启用
   args.use_cnn_attention = False  # CNN+Attention 始终禁用
   args.use_tfidf = False  # TF-IDF 特征始终禁用
   ```
   **建议**: 这些也可以作为超参数搜索空间的一部分

2. **样本量固定**
   - 默认只用 4000 样本（总共 13822 样本）
   - 可能导致搜索结果在全量数据上表现不同
   **建议**: 可以增加到 6000-8000 样本

3. **Epoch 数较少**
   - 默认只训练 3 个 epoch
   - 对于复杂模型可能不够充分
   **建议**: 可以增加到 5 个 epoch

4. **没有搜索 batch_size 和 grad_accum_steps**
   - 这两个参数对训练效果也有影响
   **建议**: 可以添加到搜索空间

5. **没有早停机制**
   - 如果验证集性能不再提升，可以提前停止
   **建议**: 添加 early stopping

---

## 使用建议

### 基础使用
```bash
# 默认配置（20 trials, 3 epochs, 4000 样本）
python tools/optuna_search_classifier.py

# 自定义配置
python tools/optuna_search_classifier.py \
    --n-trials 30 \
    --epochs 5 \
    --sample-size 6000 \
    --metric macro_f1
```

### 使用数据库持久化
```bash
python tools/optuna_search_classifier.py \
    --storage sqlite:///optuna_classifier.db \
    --study-name my_study
```

### 保留 trial 产物
```bash
python tools/optuna_search_classifier.py \
    --keep-trial-artifacts
```

---

## 改进建议

### 1. 扩展搜索空间
```python
# 添加更多超参数
batch_size = trial.suggest_categorical("batch_size", [16, 32])
grad_accum_steps = trial.suggest_categorical("grad_accum_steps", [1, 2, 4])
use_fgm = trial.suggest_categorical("use_fgm", [True, False])
use_cnn_attention = trial.suggest_categorical("use_cnn_attention", [True, False])
```

### 2. 增加样本量和 epoch
```python
parser.add_argument("--sample-size", type=int, default=6000)  # 从 4000 增加到 6000
parser.add_argument("--epochs", type=int, default=5)  # 从 3 增加到 5
```

### 3. 添加早停
```python
# 在训练循环中添加
if no_improvement_for_n_epochs >= patience:
    break
```

### 4. 使用更好的剪枝器
```python
# 对于深度学习任务，HyperbandPruner 可能更好
pruner = optuna.pruners.HyperbandPruner(
    min_resource=1,
    max_resource=args.epochs,
    reduction_factor=3
)
```

---

## 总体评价

**合理性评分**: ⭐⭐⭐⭐☆ (4/5)

这个脚本整体设计合理，适合快速搜索超参数。主要优点是：
- 搜索空间覆盖了关键参数
- 使用小样本加速搜索
- 结果保存完整

主要改进空间：
- 可以扩展搜索空间（batch_size, FGM, CNN+Attention 等）
- 增加样本量和 epoch 数以获得更可靠的结果
- 添加早停机制

**建议**: 在当前模型训练效果不理想的情况下，可以先运行这个脚本找到最佳超参数组合，然后用最佳参数重新训练完整模型。
