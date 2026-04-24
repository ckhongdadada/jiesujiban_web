# 市级弱监督单位接入说明

## 新增产物

- `C:\Users\28414\Documents\New project\raw_data_analysis\city_reply_unit_master_integration_ready.csv`
- `C:\Users\28414\Documents\New project\raw_data_analysis\unit_supervision_candidates_all.csv`
- `C:\Users\28414\Documents\New project\raw_data_analysis\unit_normalization_mapping_candidates.csv`
- `C:\Users\28414\Documents\New project\raw_data_analysis\unit_supervision_integration_report.json`

## 结果摘要

### 1. 市级弱监督单位样本

来自两份市级留言板的正文抽取结果：

- 总样本：`2575`
- 高置信度：`1102`
- 中置信度：`1473`

建议使用方式：

- `high`：可直接并入单位分类训练候选集
- `medium`：优先人工抽查或继续归一化后再并入

### 2. 全量单位监督候选

把区级显式 `官方回复单位` 与市级正文抽取单位合并后：

- 显式单位样本：`3282`
- 弱监督单位样本：`2575`
- 总监督候选：`5857`
- 唯一归一化单位：`882`

### 3. 当前主要伪标签

- `认领交办`

这类值建议从单位分类标签空间中移除或单独标记为流程状态，而不是承办单位。

## 建议并入统一母表的字段

对于市级弱监督单位，建议新增这些字段：

- `reply_unit_raw`
- `reply_unit_norm`
- `reply_unit_source = weak_reply_extracted`
- `reply_unit_confidence = high / medium`
- `reply_unit_rule = explicit_reply / verified_by / verified_generic`
- `reply_unit_district_context`
- `reply_unit_recommended_for_cls_train`

对于原始显式单位字段，建议：

- `reply_unit_source = explicit_field`
- `reply_unit_confidence = high`
- `reply_unit_rule = explicit_column`

## 面向三类任务的当前可用分层

### A. 地名识别

继续使用全部 `message_text`，并优先利用：

- `发生地`
- `district_from_file`
- `留言来源`

### B. 官方回复单位预测

推荐训练层级：

1. 强监督核心集
- 来源：显式 `官方回复单位`
- 过滤：去掉 `认领交办` 等伪标签

2. 弱监督扩充集
- 来源：市级正文抽取的 `high` 置信度样本
- 建议：先低权重并入或用于第二阶段微调

3. 待复核集
- 来源：市级正文抽取的 `medium` 置信度样本
- 建议：人工抽样审查后再决定是否纳入

### C. 官方回复生成

可以继续使用：

- 显式单位样本
- 无显式单位但能抽出单位的市级样本
- 无单位但回复正文质量高的样本

推荐在 prompt 中优先使用：

- `message_tag_level1`
- `message_tag_level2`
- `district_from_file`
- `reply_unit_norm`（若存在）

## 下一步最自然的工作

如果开始真正清洗，建议顺序：

1. 生成统一母表
2. 加入 `reply_unit_source / confidence / rule`
3. 用 `unit_normalization_mapping_candidates.csv` 做第一轮单位归一化
4. 切分三套训练子集
