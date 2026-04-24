# 接诉即办智能回复系统

基于大语言模型的政务留言智能分析与回复生成系统。

## 项目简介

本项目是一个智能化的政务留言处理系统，能够自动分析市民留言内容，识别所属行政区域，预测回复单位，并生成专业的政务回复文本。

## 核心功能

- **地名识别**：自动识别留言中的北京市行政区划
- **单位分类**：预测留言应由哪个部门回复
- **智能回复**：基于RAG检索生成专业政务回复
- **Web界面**：提供友好的交互界面

## 技术架构

### 分类模型
- 当前演示权重：`final_model_fgm/`，运行时通过 `AutoModelForSequenceClassification` 加载 BERT/RoBERTa 兼容的序列分类模型，并通过 Softmax 输出回复单位 Top-K 概率。
- 训练与实验路径：`training/train_unit_classifier_fgm.py` 支持 `--use-cnn-attention` 与 `--use-tfidf`，可训练 **BERT + CNN + Attention + TF-IDF** 多组件融合架构。
- 说明：若要把演示运行时切换到 **BERT + CNN + Attention + TF-IDF**，需要使用同结构训练出的权重，并同时保存/加载 TF-IDF 向量器产物；不能直接用标准 `BertForSequenceClassification` 权重硬切。
- Focal Loss / RankAware Loss 处理类别不平衡与 Top-K 排名目标
- FGM 对抗训练提升鲁棒性

### 生成模型
- Qwen2.5-1.5B-Instruct 大语言模型
- LoRA 高效微调
- 4-bit 量化训练

### RAG检索
- 当前主流程使用 `BAAI/bge-small-zh-v1.5`，通过 `SentenceTransformer` 将政策/案例文档和用户留言转换为归一化稠密向量。
- 查询时先把留言编码为向量，再与已缓存的文档向量做点积相似度计算；由于向量已归一化，该点积等价于余弦相似度。
- 检索后会结合行政区、标签、回复单位、关键词命中等元数据做二次加权排序。
- 当 BGE 或向量依赖不可用时，系统会回退到 TF-IDF 向量化与余弦相似度检索。
- 政策案例知识库存放在 `data/runtime/` 下，反馈信息继续使用 SQLite 保存。

## 项目结构

```text
接诉即办项目/
├── app.py                  # 主应用入口
├── enhancements/           # 核心模块
├── training/               # 训练脚本
├── tools/                  # 工具与测试脚本
├── crawlers/               # 抓取脚本
├── templates/              # HTML模板
├── data/
│   ├── runtime/            # 运行时词典、知识库、反馈库
│   ├── raw/                # 原始抓取记录
│   ├── processed/          # 清洗后的中间产物
│   ├── reports/            # 训练/抓取报告
│   └── universe/           # 抓取规划
└── docs/                   # 项目文档（统一入口见 docs/INDEX.md）
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (推荐)
- 8GB+ 显存 (推荐)

## 安装

```bash
# 创建虚拟环境
conda create -n qwen_env python=3.10
conda activate qwen_env

# 安装依赖
pip install -r requirements.txt
```

## 使用

```bash
# 启动服务
python app.py

# 访问
http://localhost:5000
```

## 模型下载

由于模型文件较大，需要单独下载：
- Qwen2.5-1.5B-Instruct
- RoBERTa-base-chinese
- BGE-small-zh

## 许可证

MIT License

