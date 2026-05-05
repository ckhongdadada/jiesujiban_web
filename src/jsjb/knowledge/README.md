# 知识图谱模块

## 主用途定位

本模块在"接诉即办"系统中的主用途是**事实结构化 + 审核验证 + RAG 增强**：

| 用途 | 定位 | 优先级 |
|------|------|--------|
| 事实结构化 | 从留言和回复中提取结构化事实（项目、单位、状态、地点） | ★★★ 主用途 |
| 审核验证 | 校验提取的事实与图谱已有知识是否一致，发现冲突 | ★★★ 主用途 |
| RAG 增强 | 通过 GraphAugmentor 为检索结果注入图谱事实 | ★★★ 已实现 |
| 知识补全 | 为生成回复提供结构化上下文（责任链、历史案例、政策依据） | ★★ 辅助用途 |

## 模块架构

```
knowledge/
├── graph.py              # 图谱管理器（内存图/Neo4j双模式）
├── query.py              # 图查询引擎（高级查询和推理）
├── entity_extractor.py   # 实体提取器（正则 + NER双模式）
├── fusion.py             # 信息融合引擎（多源融合、冲突检测）
├── structured_kb.py      # 结构化知识库（项目状态、公共资源、单位映射）
├── updater.py            # 知识更新器（从反馈自动更新知识库）
├── builder.py            # 图谱构建脚本（从语料库构建图谱）
└── demo_pipeline.py      # Demo：完整业务流程演示
```

## 核心数据模型

### 节点类型

| 节点类型 | 说明 | 关键属性 |
|---------|------|---------|
| Project | 项目/工程 | name, type, status, description |
| Location | 地点 | name, district, street |
| Organization | 单位 | name, type, level, responsibilities |
| Policy | 政策法规 | title, doc_number, content, source |
| Case | 案例 | title, issue_type, resolution, district |
| Resource | 公共资源 | name, type, capacity, location |

### 关系类型

| 关系 | 说明 | 示例 |
|------|------|------|
| LOCATED_IN | 位于 | 项目 → 地点 |
| RESPONSIBLE_FOR | 负责 | 单位 → 项目 |
| REPORTS_TO | 上级 | 街道办 → 区政府 |
| BASED_ON | 依据政策 | 项目 → 政策 |
| SIMILAR_TO | 相似案例 | 案例A → 案例B |
| HAS_RESOURCE | 包含资源 | 地点 → 资源 |

## 完整业务流程

```
市民留言输入
    ↓
[1] 实体提取 (entity_extractor.py)
    → 提取：项目名、地点、单位、问题类型
    ↓
[2] 图谱查询 (query.py)
    → 查询：责任单位、上级单位、相关政策、相似案例
    ↓
[3] 审核验证 (fusion.py)
    → 校验：提取结果与图谱已有知识是否冲突
    → 计算：置信度得分
    ↓
[4] 知识补全 (structured_kb.py)
    → 补全：项目最新状态、责任单位联系方式、办理时限
    ↓
[5] 知识更新 (updater.py)
    → 审核通过后：将新事实写入图谱
    → 反馈循环：用户反馈驱动知识库持续更新
```

## 运行 Demo

```bash
python -m src.jsjb.knowledge.demo_pipeline
```

## RAG 集成

图谱模块通过 `GraphAugmentor` 与 RAG 检索深度集成：

```python
from src.jsjb.retrieval.bge_retriever import GraphAugmentor

augmentor = GraphAugmentor(graph_path="data/runtime/knowledge_graph.json")

# 查询相关事实（返回格式与 RAG 文档兼容）
facts = augmentor.query_related_facts(
    query="小区垃圾清运问题",
    district="朝阳区",
    limit=3
)
```

**注入流程**：

```text
用户查询
  -> BGE 混合检索
  -> GraphAugmentor.query_related_facts()
  -> 合并图谱命中项（带 [图谱] 前缀）
  -> Reranker 重排序
  -> 最终结果
```

**配置**（`configs/app/config.json`）：

```json
{
  "rag_enable_graph_augment": true
}
```
