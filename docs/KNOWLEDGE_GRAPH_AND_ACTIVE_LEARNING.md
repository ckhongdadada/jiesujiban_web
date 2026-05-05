# 知识图谱与主动学习

## 1. 知识图谱模块概述

知识图谱模块在"接诉即办"系统中承担以下职责：

| 用途 | 定位 | 优先级 |
|------|------|--------|
| 事实结构化 | 从留言和回复中提取结构化事实 | ★★★ 主用途 |
| 审核验证 | 校验提取的事实与图谱已有知识是否一致 | ★★★ 主用途 |
| RAG 增强 | 为检索结果注入图谱事实 | ★★★ 已实现 |
| 知识补全 | 为生成回复提供结构化上下文 | ★★ 辅助用途 |

## 2. 当前实现状态

### 2.1 图谱管理器 (`graph.py`)

支持双模式运行：

```python
from src.jsjb.knowledge.graph import KnowledgeGraphManager

# 内存模式（默认）
kg = KnowledgeGraphManager(backend="memory")

# Neo4j 模式
kg = KnowledgeGraphManager(
    backend="neo4j",
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)
```

### 2.2 图查询引擎 (`query.py`)

提供高级查询接口：

```python
from src.jsjb.knowledge.query import GraphQueryEngine

engine = GraphQueryEngine(kg)

# 查询单位职责
result = engine.query_unit_responsibilities("朝阳区城管委")

# 查询上级单位
parent = engine.query_parent_unit("望京街道办")

# 查询相似案例
cases = engine.query_similar_cases("垃圾清运", district="朝阳区")
```

### 2.3 RAG 集成 (`GraphAugmentor`)

**已实现**：图谱事实注入 RAG 检索结果

```python
from src.jsjb.retrieval.bge_retriever import GraphAugmentor

augmentor = GraphAugmentor(graph_path="data/runtime/knowledge_graph.json")

# 查询相关事实
facts = augmentor.query_related_facts(
    query="小区垃圾清运问题",
    district="朝阳区",
    limit=3
)

# facts 返回格式与 RAG 文档兼容，可直接注入检索结果
```

**注入流程**：

```text
用户查询
  -> BGE 混合检索
  -> GraphAugmentor.query_related_facts()
  -> 合并图谱命中项
  -> Reranker 重排序
  -> 最终结果
```

**配置方式**：

```json
// configs/app/config.json
{
  "rag_enable_graph_augment": true
}
```

## 3. 数据模型

### 3.1 节点类型

| 节点类型 | 说明 | 关键属性 |
|---------|------|---------|
| Project | 项目/工程 | name, type, status, description |
| Location | 地点 | name, district, street |
| Organization | 单位 | name, type, level, responsibilities |
| Policy | 政策法规 | title, doc_number, content, source |
| Case | 案例 | title, issue_type, resolution, district |
| Resource | 公共资源 | name, type, capacity, location |

### 3.2 关系类型

| 关系 | 说明 | 示例 |
|------|------|------|
| LOCATED_IN | 位于 | 项目 → 地点 |
| RESPONSIBLE_FOR | 负责 | 单位 → 项目 |
| REPORTS_TO | 上级 | 街道办 → 区政府 |
| BASED_ON | 依据政策 | 项目 → 政策 |
| SIMILAR_TO | 相似案例 | 案例A → 案例B |
| HAS_RESOURCE | 包含资源 | 地点 → 资源 |

## 4. 知识更新流程

### 4.1 结构化知识库 (`structured_kb.py`)

```python
from src.jsjb.knowledge.structured_kb import StructuredKnowledgeBase

kb = StructuredKnowledgeBase()

# 添加项目状态
kb.add_project({
    "name": "XX道路改造工程",
    "status": "进行中",
    "responsible_unit": "朝阳区城管委",
    "district": "朝阳区"
})

# 导出为 RAG 语料
kb.export_to_rag_corpus("data/rag/policy_corpus.jsonl")
```

### 4.2 知识更新器 (`updater.py`)

从反馈中自动更新知识库：

```python
from src.jsjb.knowledge.updater import KnowledgeUpdater

updater = KnowledgeUpdater()

# 处理反馈并提取事实
updater.process_feedback(feedback_id=42)

# 获取待审核事实
pending = updater.get_pending_facts()

# 审核通过后导入图谱
updater.approve_fact(fact_id=1)
```

## 5. 主动学习规划

### 5.1 目标

基于模型不确定性自动选择样本进行人工标注，形成闭环：

```text
模型预测
  -> 低置信度样本
  -> 人工标注
  -> 加入训练集
  -> 增量训练
```

### 5.2 不确定性度量

可选方法：

| 方法 | 说明 |
|------|------|
| 熵最大 | 预测分布熵值最高的样本 |
| 边缘采样 | Top-1 和 Top-2 概率差最小的样本 |
| 集成分歧 | 多模型预测不一致的样本 |

### 5.3 实现计划

```python
# 伪代码
def active_learning_loop(model, unlabeled_pool, budget=100):
    labeled = []
    for _ in range(budget):
        # 1. 模型预测
        predictions = model.predict_proba(unlabeled_pool)

        # 2. 计算不确定性
        uncertainties = entropy(predictions)

        # 3. 选择最不确定的样本
        top_idx = np.argsort(uncertainties)[-1]

        # 4. 人工标注
        label = human_label(unlabeled_pool[top_idx])
        labeled.append((unlabeled_pool[top_idx], label))

        # 5. 从池中移除
        unlabeled_pool.pop(top_idx)

        # 6. 增量训练
        model.partial_fit([unlabeled_pool[top_idx]], [label])

    return labeled
```

## 6. API 接口

### 6.1 获取待审核事实

```
GET /api/knowledge-graph/review?status=pending&limit=20
```

### 6.2 审核并导入图谱

```
POST /api/knowledge-graph/review/{candidate_id}

{
  "action": "approve",
  "reviewer": "admin",
  "review_notes": "信息准确"
}
```

## 7. 文件结构

```
src/jsjb/knowledge/
├── graph.py              # 图谱管理器（内存图/Neo4j双模式）
├── query.py              # 图查询引擎（高级查询和推理）
├── entity_extractor.py   # 实体提取器（正则 + NER双模式）
├── fusion.py             # 信息融合引擎（多源融合、冲突检测）
├── structured_kb.py      # 结构化知识库（项目状态、公共资源、单位映射）
├── updater.py            # 知识更新器（从反馈自动更新知识库）
├── builder.py            # 图谱构建脚本（从语料库构建图谱）
└── demo_pipeline.py      # Demo：完整业务流程演示
```

## 8. 运行 Demo

```bash
python -m src.jsjb.knowledge.demo_pipeline
```
