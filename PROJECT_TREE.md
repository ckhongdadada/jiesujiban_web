# 接诉即办项目目录树

```
ver1.3/
│
├── 📄 app.py                          # 【核心】Flask主应用入口
├── 📄 requirements.txt                # 【核心】Python依赖包列表
├── 📄 requirements_optional_rag_ner.txt # 【可选】RAG和NER相关依赖
│
├── 📁 templates/                      # 【前端】HTML模板文件
│   ├── 📄 index.html                  # 基础版前端页面
│   └── 📄 index_enhanced.html         # 增强版前端页面（含地名识别、RAG展示）
│
├── 📁 enhancements/                   # 【增强】核心功能增强模块
│   ├── 📄 __init__.py                 # 模块初始化
│   ├── 📄 location_ner.py             # 地名识别（北京16区）
│   ├── 📄 rag_retriever.py            # 政策检索（TF-IDF）
│   ├── 📄 enhanced_generation.py      # 增强回复生成
│   ├── 📄 classifier_runtime.py       # 分类器运行时
│   ├── 📄 model_artifacts.py          # 模型文件管理
│   ├── 📄 runtime_config.py           # 运行时配置
│   ├── 📄 startup_checks.py           # 启动检查
│   ├── 📄 training_acceptance.py      # 训练验收
│   └── 📄 beijing_districts.json      # 北京区级地名别名
│
├── 📁 training/                       # 【训练】模型训练脚本
│   ├── 📄 train_qwen_reply_lora.py    # Qwen回复模型LoRA微调
│   └── 📄 train_unit_classifier_fgm.py # 单位分类器FGM训练
│
├── 📁 tools/                          # 【工具】数据处理工具集
│   ├── 📄 build_place_taxonomy.py     # 构建地名分类
│   ├── 📄 curate_place_aliases.py     # 整理地名别名
│   ├── 📄 merge_community_aliases.py  # 合并社区别名
│   ├── 📄 data_purification.py        # 数据清洗
│   ├── 📄 prepare_rag_corpus.py       # 准备RAG语料
│   ├── 📄 check_training_setup.py     # 检查训练环境
│   ├── 📄 check_training_artifacts.py # 检查训练产物
│   ├── 📄 post_training_acceptance.py # 训练后验收
│   ├── 📄 bulk_amap_community_crawl.py # 高德社区批量爬取
│   ├── 📄 amap_community_grid_crawler.py # 高德社区网格爬虫
│   ├── 📄 amap_grid_regeo_crawler.py  # 高德网格逆地理编码
│   ├── 📄 amap_road_regeo_crawler.py  # 高德道路逆地理编码
│   ├── 📄 baidu_community_crawler.py  # 百度社区爬虫
│   ├── 📄 baidu_api_test.py           # 百度API测试
│   ├── 📄 baidu_community_records.jsonl # 百度社区爬取记录
│   ├── 📄 road_township_dictionary.jsonl # 道路乡镇字典
│   └── 📄 run_legacy_classifier_service.ps1 # 旧版服务启动脚本
│
├── 📁 crawlers/                       # 【爬虫】数据爬虫模块
│   ├── 📄 __init__.py                 # 模块初始化
│   └── 📄 beijing_alias_crawler.py    # 北京地名别名爬虫
│
├── 📁 data/                           # 【数据】所有数据存储
│   │
│   ├── 📁 place_taxonomy/             # 地名分类数据
│   │   ├── 📄 bus_stop.json           # 公交站点
│   │   ├── 📄 cinema.json             # 电影院
│   │   ├── 📄 community.json          # 社区/小区
│   │   ├── 📄 dictionary_core.json    # 核心词典
│   │   ├── 📄 hospital.json           # 医院
│   │   ├── 📄 hotel.json              # 酒店
│   │   ├── 📄 mall.json               # 商场
│   │   ├── 📄 park.json               # 公园
│   │   ├── 📄 poi_misc.json           # 其他POI
│   │   ├── 📄 road.json               # 道路
│   │   ├── 📄 school.json             # 学校
│   │   └── 📄 subway_station.json     # 地铁站
│   │
│   ├── 📁 purified/                   # 净化数据
│   │   ├── 📄 commercial.jsonl        # 商业场所
│   │   ├── 📄 community_clean.jsonl   # 清洗后社区
│   │   ├── 📄 government.jsonl        # 政府机构
│   │   ├── 📄 hospital.jsonl          # 医疗机构
│   │   ├── 📄 school.jsonl            # 教育机构
│   │   ├── 📄 transit.jsonl           # 交通设施
│   │   └── 📄 purification_report.json # 净化报告
│   │
│   ├── 📁 training_reports/           # 训练报告
│   │   ├── 📄 all_*.json              # 综合报告
│   │   ├── 📄 classifier_acceptance_*.json # 分类器验收
│   │   └── 📄 setup_check_*.json      # 环境检查
│   │
│   ├── 📄 beijing_districts_curated.json    # 北京区级数据（精选）
│   ├── 📄 beijing_districts_extra.json      # 北京区级数据（扩展）
│   ├── 📄 beijing_districts_merged.json     # 北京区级数据（合并）
│   ├── 📄 beijing_districts_extra_communities.json # 扩展社区
│   ├── 📄 beijing_communities_curated.json  # 精选社区
│   ├── 📄 beijing_place_records.jsonl       # 北京地点记录
│   ├── 📄 beijing_place_records_communities.jsonl # 社区地点记录
│   ├── 📄 place_alias_catalog.jsonl         # 地名别名目录
│   ├── 📄 policy_case_corpus.sample.jsonl   # 政策案例样本
│   ├── 📄 policy_case_corpus.template.jsonl # 政策案例模板
│   ├── 📄 amap_community_bulk_progress.json # 高德批量爬取进度
│   ├── 📄 amap_community_bulk_records.jsonl # 高德批量爬取记录
│   ├── 📄 amap_community_bulk_state.json    # 高德批量爬取状态
│   ├── 📄 community_merge_report.json       # 社区合并报告
│   ├── 📄 crawl_report.json                 # 爬取报告
│   └── 📄 curation_report.json              # 整理报告
│
├── 📁 final_model_fgm/                # 【模型】分类模型
│   ├── 📄 config.json                 # 模型配置
│   ├── 📄 label_map.json              # 标签映射（单位列表）
│   ├── 📄 pytorch_model.bin           # 模型权重
│   ├── 📄 tokenizer.json              # 分词器
│   ├── 📄 tokenizer_config.json       # 分词器配置
│   └── 📄 empty.gitkeep               # Git占位文件
│
├── 📁 final_model_fgm_smoke/          # 【模型】轻量分类模型
│   ├── 📄 config.json
│   ├── 📄 label_map.json
│   ├── 📄 pytorch_model.bin
│   ├── 📄 tokenizer.json
│   └── 📄 tokenizer_config.json
│
└── 📄 *.md                            # 【文档】各类说明文档
    ├── PROJECT_REVIEW.md              # 项目审视
    ├── TRAINING_README.md             # 训练指南
    ├── TRAINING_ARTIFACTS_README.md   # 训练产物说明
    ├── ENHANCEMENT_README.md          # 增强功能说明
    ├── RAG_ENGINEERING_README.md      # RAG工程说明
    ├── CRAWLER_README.md              # 爬虫说明
    └── STABILITY_README.md            # 稳定性说明
```

---

## 图例说明

| 符号 | 含义 |
|------|------|
| 📄 | 文件 |
| 📁 | 目录 |
| 【核心】 | 核心文件，必须存在 |
| 【前端】 | 前端相关文件 |
| 【增强】 | 增强功能模块 |
| 【训练】 | 模型训练相关 |
| 【工具】 | 数据处理工具 |
| 【爬虫】 | 数据爬取工具 |
| 【数据】 | 数据存储 |
| 【模型】 | 模型文件 |
| 【文档】 | 说明文档 |

---

## 核心模块关系图

```
┌─────────────────────────────────────────────────────────┐
│                      用户请求                            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    app.py (主入口)                       │
│  - 接收HTTP请求                                          │
│  - 路由分发                                              │
│  - 返回JSON响应                                          │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ 分类模型      │ │ 生成模型      │ │ 增强模块      │
│ (RoBERTa)    │ │ (Qwen+LoRA)  │ │ (enhancements)│
└──────────────┘ └──────────────┘ └──────┬───────┘
                                         │
                               ┌─────────┼─────────┐
                               │         │         │
                               ▼         ▼         ▼
                         ┌──────────┐ ┌──────────┐ ┌──────────┐
                         │ 地名识别  │ │ 政策检索  │ │ 增强生成  │
                         │ (NER)    │ │ (RAG)    │ │          │
                         └──────────┘ └──────────┘ └──────────┘
```

---

## 数据流向图

```
┌─────────────┐
│ 用户留言    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 1. 单位分类 (final_model_fgm/)      │
│    预测应转派的单位                  │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 2. 地名识别 (enhancements/)         │
│    识别北京区级地名                  │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 3. 政策检索 (data/policy_case_*)    │
│    检索相关政策和案例                │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 4. 回复生成 (qwen_models/)          │
│    生成专业政务回复                  │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────┐
│ 返回结果    │
└─────────────┘
```

---

## 模型文件说明

### 分类模型 (final_model_fgm/)
- **基础模型**：RoBERTa
- **训练方法**：FGM对抗训练
- **功能**：预测留言应转派的单位
- **输入**：留言标签、标题、正文
- **输出**：候选单位列表（含置信度）

### 生成模型 (qwen_models/)
- **基础模型**：Qwen2.5-1.5B
- **训练方法**：LoRA微调
- **功能**：生成政务回复内容
- **输入**：留言信息 + 单位 + 地名 + 政策
- **输出**：专业政务回复文本

---

## 快速定位

### 我想修改前端界面
→ `templates/index.html` 或 `templates/index_enhanced.html`

### 我想调整分类逻辑
→ `enhancements/classifier_runtime.py` 或 `final_model_fgm/`

### 我想优化回复生成
→ `enhancements/enhanced_generation.py` 或重新训练模型

### 我想添加新的地名数据
→ `data/place_taxonomy/` 或 `tools/curate_place_aliases.py`

### 我想更新政策语料
→ `data/policy_case_corpus.*.jsonl` 或 `tools/prepare_rag_corpus.py`

### 我想重新训练模型
→ `training/` 目录

### 我想爬取新数据
→ `crawlers/` 或 `tools/` 中的爬虫脚本

---

## 文件大小参考

| 目录/文件 | 大小范围 | 说明 |
|-----------|----------|------|
| app.py | ~10KB | 主应用代码 |
| templates/ | ~50KB | 前端模板 |
| enhancements/ | ~100KB | 增强模块代码 |
| training/ | ~20KB | 训练脚本 |
| tools/ | ~200KB | 工具脚本 |
| crawlers/ | ~10KB | 爬虫脚本 |
| data/ | 10MB-100MB | 数据文件 |
| final_model_fgm/ | ~400MB | 分类模型 |
| final_model_fgm_smoke/ | ~100MB | 轻量模型 |
| qwen_models/ | ~3GB | 生成模型（需单独下载） |

---

## 总结

本项目采用模块化设计，各功能模块职责清晰：

1. **核心服务**：app.py 提供API服务
2. **增强功能**：enhancements/ 提供高级NLP能力
3. **数据处理**：tools/ 和 crawlers/ 负责数据维护
4. **模型训练**：training/ 负责模型优化
5. **数据存储**：data/ 存储所有数据
6. **模型文件**：final_model_*/ 存储训练好的模型

这种结构便于维护、扩展和部署。
