# 训练脚本使用说明

## 功能特性

该训练脚本现在支持多种配置选项，可以灵活选择模型架构和损失函数：

### 1. 模型架构选择

- **标准 BERT 分类器**（默认）：使用 transformers 的 AutoModelForSequenceClassification
- **BERT + CNN + Attention**：使用 `--use-cnn-attention` 参数启用增强架构

### 2. 损失函数选择

通过 `--loss-type` 参数选择：

- `rank_aware`（默认）：排名感知损失，对 Top-K 内的样本降低惩罚
- `focal`：Focal Loss，关注难分类样本
- `cross_entropy`：标准交叉熵损失

### 3. TF-IDF 特征

- 使用 `--use-tfidf` 启用 TF-IDF 特征（需配合 CNN+Attention 架构）
- 使用 `--tfidf-dim` 设置 TF-IDF 维度（默认 3000）

### 4. 数据增强

- 训练集自动启用数据增强（随机删除、随机交换）
- 验证集不使用数据增强

## 使用示例

### 基础训练（使用 RankAwareLoss）
```bash
python training/train_unit_classifier_fgm.py
```

### 使用 Focal Loss
```bash
python training/train_unit_classifier_fgm.py --loss-type focal
```

### 使用 CNN+Attention 架构 + TF-IDF + Focal Loss
```bash
python training/train_unit_classifier_fgm.py --use-cnn-attention --use-tfidf --loss-type focal
```

### 完整配置示例
```bash
python training/train_unit_classifier_fgm.py \
    --use-cnn-attention \
    --use-tfidf \
    --loss-type rank_aware \
    --epochs 8 \
    --batch-size 16 \
    --grad-accum-steps 2 \
    --learning-rate 2e-5
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-name` | 本地路径 | 预训练模型路径 |
| `--data-path` | 本地路径 | 训练数据文件路径 |
| `--save-dir` | final_model_fgm | 模型保存目录 |
| `--epochs` | 8 | 训练轮次 |
| `--batch-size` | 16 | 批次大小 |
| `--grad-accum-steps` | 2 | 梯度累积步数 |
| `--learning-rate` | 2e-5 | 学习率 |
| `--use-cnn-attention` | False | 启用 CNN+Attention 架构 |
| `--use-tfidf` | False | 启用 TF-IDF 特征 |
| `--loss-type` | rank_aware | 损失函数类型 |
| `--disable-fgm` | False | 禁用 FGM 对抗训练 |

## 关键改进

1. **保留了原有的 RankAwareLoss**：专门针对 Top-K 评估优化
2. **新增 FocalLoss**：可选的损失函数，关注难分类样本
3. **新增 CNN+Attention 架构**：可选的增强模型架构
4. **数据增强**：训练时自动应用随机删除和交换
5. **梯度累积**：支持更大的有效批次大小
6. **类别权重**：使用 sklearn 的 compute_class_weight 自动平衡类别
7. **灵活配置**：通过命令行参数轻松切换不同配置

## 注意事项

- TF-IDF 特征需要配合 `--use-cnn-attention` 使用
- 使用 CNN+Attention 架构时，模型保存方式略有不同（不保存 config.json）
- 梯度累积可以在显存有限时模拟更大的批次大小
