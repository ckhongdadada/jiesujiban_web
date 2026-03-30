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
- BERT + CNN + Attention + TF-IDF 多组件融合架构
- Focal Loss 处理类别不平衡
- FGM 对抗训练提升鲁棒性

### 生成模型
- Qwen2.5-1.5B-Instruct 大语言模型
- LoRA 高效微调
- 4-bit 量化训练

### RAG检索
- BGE 向量检索
- 语义相似度匹配
- 政策案例知识库

## 项目结构

```
接诉即办项目/
├── app.py                  # 主应用入口
├── enhancements/           # 核心模块
│   ├── classifier_runtime.py    # 分类模型运行时
│   ├── enhanced_generation.py   # 回复生成模块
│   ├── location_ner.py          # 地名识别
│   ├── rag_retriever_bge.py     # RAG检索
│   └── ...
├── training/               # 训练脚本
├── docs/                   # 文档
├── static/                 # 静态资源
└── templates/              # HTML模板
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
