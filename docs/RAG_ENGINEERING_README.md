# RAG 工程化说明

新增内容：
- [rag_retriever.py](C:/Users/28414/PycharmProjects/接诉即办项目/enhancements/rag_retriever.py)
- [prepare_rag_corpus.py](C:/Users/28414/PycharmProjects/接诉即办项目/tools/prepare_rag_corpus.py)

## 这次增强了什么

1. 检索结果去重  
按 `title + source` 去重，避免同一材料重复入榜。

2. 元数据重排  
支持按以下信息做加权：
- `district`
- `tag`
- `unit`
- `doc_type`
- `issue_type`

3. 问题关键词提取  
会从留言里提取类似：
- `垃圾清运`
- `异味扰民`
- `施工扰民`
- `停车秩序`
- `道路积水`

4. 知识库字段模板  
新增模板字段，便于后续导入真实政策/案例：
- `issue_type`
- `unit`
- `applicable_tags`

## 推荐知识库结构

每条 JSONL 建议包含：

```json
{
  "id": "case-001",
  "title": "夜间施工扰民处置案例",
  "doc_type": "案例",
  "district": "海淀区",
  "source": "案例库",
  "tags": ["噪声", "施工扰民"],
  "issue_type": "施工扰民",
  "unit": "城市管理部门",
  "applicable_tags": ["投诉/求助"],
  "content": "正文内容"
}
```

## 当前状态

当前仍然是 `TF-IDF` 检索后端，但已经不是简单相似度排序，而是“相似度 + 元数据加权 + 去重”的组合排序。
