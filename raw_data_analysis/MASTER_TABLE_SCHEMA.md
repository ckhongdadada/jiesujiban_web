# 统一母表模板

## 文件

- `master_table_template.csv`：空白母表模板
- 本文档：字段说明

## 字段说明

| 字段名 | 含义 | 用途 |
| --- | --- | --- |
| `record_id` | 全局唯一记录 ID | 三任务共用 |
| `source_file` | 原始来源文件绝对路径 | 回溯原始数据 |
| `source_name` | 原始文件名 | 回溯原始数据 |
| `source_sheet` | 工作表名 | 回溯原始数据 |
| `source_type` | 来源类型，如 `district_raw` `city_raw` `crawler_raw` | 数据分层 |
| `district_from_file` | 从文件路径推断出的区县/市级来源 | NER、分类、生成 |
| `message_source` | 留言来源/平台路径 | 来源分析、生成 |
| `message_id` | 原始留言 ID | 去重、回溯 |
| `message_time` | 留言时间 | 时序分析 |
| `message_tag_raw` | 原始留言标签字符串 | 回溯、结构化前保留 |
| `message_tag_level1` | 一级标签，如 `投诉/求助` `咨询` `建言` | 分类、生成 |
| `message_tag_level2` | 二级标签，如 `城建` `交通` `教育` | NER、分类、生成 |
| `message_status` | 办理状态，如 `已办理` | 过滤规则 |
| `message_location_raw` | 原始 `发生地` 字段 | NER 高精度种子 |
| `message_location_norm` | 归一化后的地点字段 | NER、词典构建 |
| `message_title` | 留言标题 | 三任务共用 |
| `message_body` | 留言正文 | 三任务共用 |
| `message_text` | 标题+正文拼接文本 | 三任务共用 |
| `has_place_hint` | 是否含明显地点提示 | NER 候选筛选 |
| `place_mention_candidates` | 从正文抽出的地点候选列表/JSON | NER |
| `reply_unit_raw` | 原始回复单位，或正文抽取得到的原始单位 | 单位分类 |
| `reply_unit_norm` | 归一化回复单位 | 单位分类、生成 |
| `reply_unit_source` | 单位来源，如 `explicit_field` `weak_reply_extracted` | 单位监督分层 |
| `reply_unit_confidence` | 单位置信度，如 `high` `medium` | 单位监督分层 |
| `reply_unit_rule` | 单位抽取/映射规则名 | 审计、回溯 |
| `reply_unit_district_context` | 抽取单位时使用的区县上下文 | 单位归一化 |
| `reply_time` | 回复时间 | 时序分析 |
| `reply_text_raw` | 原始回复正文 | 生成 |
| `reply_text_clean` | 清洗后的回复正文 | 生成 |
| `reply_low_quality_flag` | 是否命中低质量回复规则 | 生成过滤 |
| `reply_onsite_negative_flag` | 是否命中现场核查否认类规则 | 生成过滤 |
| `dedupe_issue_key` | 问题去重键 | 去重 |
| `dedupe_generation_key` | 生成样本去重键 | 去重 |
| `recommended_for_ner` | 是否推荐进入 NER 训练候选 | NER |
| `recommended_for_unit_cls` | 是否推荐进入单位分类候选 | 分类 |
| `recommended_for_reply_gen` | 是否推荐进入回复生成候选 | 生成 |
| `notes` | 备注/人工复核意见 | 审计 |

## 使用建议

1. 这张母表只做“标准化汇总”，不替代原始数据。
2. 所有后续训练集都从这张母表切分，而不是直接从原始 Excel 再切。
3. `reply_unit_source` 和 `reply_unit_confidence` 必须保留，不然显式单位和弱监督单位会混在一起。
4. `message_location_raw` 和 `message_location_norm` 都要保留，便于后续地点词典与行政区映射。
