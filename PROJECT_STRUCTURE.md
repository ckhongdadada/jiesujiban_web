# 接诉即办项目结构说明

## 项目概述

接诉即办智能服务系统是一个基于深度学习的政务留言处理系统，集成了留言分类、智能回复生成、地名识别和政策检索等功能。

---

## 目录结构总览

```
ver1.3/
├── app.py                          # 主应用入口
├── requirements.txt                # 核心依赖
├── requirements_optional_rag_ner.txt # 可选依赖
├── templates/                      # 前端模板
├── enhancements/                   # 增强功能模块
├── training/                       # 模型训练脚本
├── tools/                          # 数据处理工具
├── crawlers/                       # 爬虫模块
├── data/                           # 数据存储
├── final_model_fgm/                # 分类模型
├── final_model_fgm_smoke/          # 轻量分类模型
└── *.md                            # 各类文档
```

---

## 核心文件说明

### 1. 应用入口
- **app.py** - Flask主应用，提供API服务和前端页面
  - 加载分类模型和生成模型
  - 提供 `/api/analyze` 接口处理留言分析
  - 提供 `/api/health` 接口检查服务状态
  - 支持预加载模型，避免首次请求延迟

### 2. 依赖管理
- **requirements.txt** - 核心Python依赖包
  - Flask、PyTorch、Transformers、PEFT等
- **requirements_optional_rag_ner.txt** - 可选依赖
  - RAG检索和NER识别相关依赖

---

## 目录详细说明

### 📁 templates/ - 前端模板
存放HTML模板文件，提供用户界面。

| 文件 | 说明 |
|------|------|
| index.html | 基础版前端页面，提供留言输入和回复展示 |
| index_enhanced.html | 增强版前端页面，集成地名识别和RAG检索展示 |

**功能特点**：
- 响应式设计，支持移动端
- 实时健康检查
- 支持切换候选单位重新生成回复

---

### 📁 enhancements/ - 增强功能模块
核心功能增强模块，提供高级NLP能力。

| 文件 | 功能说明 |
|------|----------|
| __init__.py | 模块初始化 |
| location_ner.py | **地名识别**：识别北京区级地名，支持词典匹配、LAC实体识别、高德API地理编码 |
| rag_retriever.py | **政策检索**：基于TF-IDF的政策和案例检索系统 |
| enhanced_generation.py | **增强生成**：结合地名和检索结果的智能回复生成 |
| classifier_runtime.py | 分类器运行时管理 |
| model_artifacts.py | 模型文件管理工具 |
| runtime_config.py | 运行时配置管理 |
| startup_checks.py | 启动前检查 |
| training_acceptance.py | 训练验收工具 |
| beijing_districts.json | 北京区级地名别名字典 |

**核心能力**：
1. **地名识别**：准确识别北京16个区的地名
2. **政策检索**：检索相关政策和案例
3. **智能生成**：生成专业、规范的政务回复

---

### 📁 training/ - 模型训练
模型训练脚本和配置。

| 文件 | 说明 |
|------|------|
| train_qwen_reply_lora.py | Qwen回复模型的LoRA微调脚本 |
| train_unit_classifier_fgm.py | 单位分类器的FGM对抗训练脚本 |

**训练流程**：
1. 准备训练数据
2. 配置模型参数
3. 运行训练脚本
4. 验证模型效果

---

### 📁 tools/ - 数据处理工具
数据处理和维护工具集。

| 文件 | 功能 |
|------|------|
| build_place_taxonomy.py | 构建地名分类体系 |
| curate_place_aliases.py | 整理地名别名 |
| merge_community_aliases.py | 合并社区别名数据 |
| data_purification.py | 数据清洗和净化 |
| prepare_rag_corpus.py | 准备RAG检索语料 |
| check_training_setup.py | 检查训练环境 |
| check_training_artifacts.py | 检查训练产物 |
| post_training_acceptance.py | 训练后验收 |
| bulk_amap_community_crawl.py | 批量爬取高德社区数据 |
| amap_community_grid_crawler.py | 高德社区网格爬虫 |
| amap_grid_regeo_crawler.py | 高德网格逆地理编码 |
| amap_road_regeo_crawler.py | 高德道路逆地理编码 |
| baidu_community_crawler.py | 百度社区爬虫 |
| baidu_api_test.py | 百度API测试 |
| run_legacy_classifier_service.ps1 | 旧版分类服务启动脚本 |

**工具分类**：
- **数据构建**：构建地名分类、别名整理
- **数据爬取**：高德、百度数据爬取
- **模型训练**：环境检查、训练验收
- **语料准备**：RAG语料准备

---

### 📁 crawlers/ - 爬虫模块
数据爬虫脚本。

| 文件 | 说明 |
|------|------|
| __init__.py | 模块初始化 |
| beijing_alias_crawler.py | 北京地名别名爬虫 |

**爬虫目标**：
- 安居客、链家等房产网站
- 获取社区、道路、地标等地名数据

---

### 📁 data/ - 数据存储
所有数据文件的存储目录。

#### 📂 place_taxonomy/ - 地名分类数据
各类地点的分类词典。

| 文件 | 内容 |
|------|------|
| bus_stop.json | 公交站点 |
| cinema.json | 电影院 |
| community.json | 社区/小区 |
| dictionary_core.json | 核心词典 |
| hospital.json | 医院 |
| hotel.json | 酒店 |
| mall.json | 商场 |
| park.json | 公园 |
| poi_misc.json | 其他POI |
| road.json | 道路 |
| school.json | 学校 |
| subway_station.json | 地铁站 |

#### 📂 purified/ - 净化数据
经过清洗和净化的数据。

| 文件 | 内容 |
|------|------|
| commercial.jsonl | 商业场所 |
| community_clean.jsonl | 清洗后的社区数据 |
| government.jsonl | 政府机构 |
| hospital.jsonl | 医疗机构 |
| school.jsonl | 教育机构 |
| transit.jsonl | 交通设施 |
| purification_report.json | 净化报告 |

#### 📂 training_reports/ - 训练报告
模型训练的验收报告。

| 文件 | 说明 |
|------|------|
| all_*.json | 综合报告 |
| classifier_acceptance_*.json | 分类器验收报告 |
| setup_check_*.json | 环境检查报告 |

#### 📄 其他数据文件
| 文件 | 说明 |
|------|------|
| beijing_districts_*.json | 北京区级数据（curated/extra/merged） |
| place_alias_catalog.jsonl | 地名别名目录 |
| policy_case_corpus.*.jsonl | 政策案例语料 |
| amap_community_bulk_*.json | 高德社区批量爬取数据 |
| beijing_place_records.jsonl | 北京地点记录 |
| community_merge_report.json | 社区合并报告 |
| crawl_report.json | 爬取报告 |
| curation_report.json | 整理报告 |

---

### 📁 final_model_fgm/ - 分类模型
单位分类模型文件。

| 文件 | 说明 |
|------|------|
| config.json | 模型配置 |
| label_map.json | 标签映射（单位列表） |
| pytorch_model.bin | 模型权重 |
| tokenizer.json | 分词器 |
| tokenizer_config.json | 分词器配置 |
| empty.gitkeep | Git占位文件 |

**模型信息**：
- 基于RoBERTa
- 使用FGM对抗训练
- 预测留言应转派的单位

---

### 📁 final_model_fgm_smoke/ - 轻量分类模型
轻量级分类模型，用于快速测试。

结构与 `final_model_fgm` 相同，但模型更小、推理更快。

---

## 文档说明

| 文档 | 内容 |
|------|------|
| PROJECT_REVIEW.md | 项目审视和问题分析 |
| TRAINING_README.md | 模型训练指南 |
| TRAINING_ARTIFACTS_README.md | 训练产物说明 |
| ENHANCEMENT_README.md | 增强功能说明 |
| RAG_ENGINEERING_README.md | RAG检索工程说明 |
| CRAWLER_README.md | 爬虫使用说明 |
| STABILITY_README.md | 系统稳定性说明 |

---

## 数据流向

```
用户留言
    ↓
[app.py] 接收请求
    ↓
[enhancements/classifier_runtime.py] 单位分类
    ↓
[enhancements/location_ner.py] 地名识别
    ↓
[enhancements/rag_retriever.py] 政策检索
    ↓
[enhancements/enhanced_generation.py] 回复生成
    ↓
返回结果给用户
```

---

## 模型依赖关系

```
分类模型 (final_model_fgm/)
    └── RoBERTa + FGM对抗训练
        └── 预测处理单位

生成模型 (qwen_models/)
    └── Qwen2.5-1.5B + LoRA微调
        └── 生成回复内容
```

---

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备模型
- 将分类模型放入 `final_model_fgm/` 目录
- 将生成模型放入 `qwen_models/` 目录
- 将LoRA权重放入 `qwen_reply_model/` 目录

### 3. 启动服务
```bash
python app.py
```

### 4. 访问服务
- 基础版：http://127.0.0.1:5000
- 增强版：http://127.0.0.1:5001（需运行 app_enhanced.py）

---

## 开发建议

### 数据维护
1. 定期更新地名别名（使用 `tools/curate_place_aliases.py`）
2. 补充政策案例语料（使用 `tools/prepare_rag_corpus.py`）
3. 清洗数据（使用 `tools/data_purification.py`）

### 模型优化
1. 收集新数据后重新训练模型
2. 使用 `training/` 中的脚本进行训练
3. 运行验收测试确保模型质量

### 功能扩展
1. 新增爬虫：在 `crawlers/` 中添加
2. 新增工具：在 `tools/` 中添加
3. 新增增强功能：在 `enhancements/` 中添加

---

## 注意事项

1. **模型路径**：确保模型文件路径正确，或通过环境变量配置
2. **内存要求**：建议至少8GB内存，推荐16GB
3. **GPU支持**：支持CUDA加速，无GPU也可运行（速度较慢）
4. **数据安全**：爬虫使用需遵守网站robots.txt和相关法律法规

---

## 版本信息

- **版本号**：ver1.3
- **更新日期**：2026-03-26
- **主要更新**：
  - 添加预加载模型功能
  - 优化健康检查逻辑
  - 修复_force_unit参数处理
  - 改用FP16精度加载模型
  - 添加限流机制

---

## 联系方式

如有问题或建议，请联系项目维护人员。
