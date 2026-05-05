# 接诉即办系统 API 文档

## 目录

1. [概述](#概述)
2. [认证](#认证)
3. [基础接口](#基础接口)
4. [分类预测接口](#分类预测接口)
5. [回复生成接口](#回复生成接口)
6. [地名识别接口](#地名识别接口)
7. [RAG检索接口](#rag检索接口)
8. [反馈接口](#反馈接口)
9. [RAG管理接口](#rag管理接口)
10. [知识图谱审核接口](#知识图谱审核接口)
11. [管理接口](#管理接口)
12. [错误码](#错误码)

---

## 概述

### 基础URL

```
http://localhost:5000
```

### 请求格式

- Content-Type: `application/json`
- 字符编码: `UTF-8`

### 响应格式

所有响应均为JSON格式，包含以下字段：

```json
{
  "success": true,
  "data": {},
  "error": null,
  "timestamp": "2024-01-01T00:00:00"
}
```

---

## 认证

### API密钥认证

在请求头中添加API密钥：

```
Authorization: Bearer YOUR_API_KEY
```

### 获取访问令牌

**请求**

```
POST /api/auth/token
```

**参数**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| username | string | 是 | 用户名 |
| password | string | 是 | 密码 |

**响应**

```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "Bearer",
    "expires_in": 3600
  }
}
```

---

## 基础接口

### 健康检查

**请求**

```
GET /api/health
```

**响应**

```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "models_loaded": true,
    "database_connected": true
  }
}
```

### 系统状态

**请求**

```
GET /api/status
```

**响应**

```json
{
  "success": true,
  "data": {
    "classifier": {
      "loaded": true,
      "model_type": "bert-cnn-attention"
    },
    "generator": {
      "loaded": true,
      "model_type": "qwen-lora"
    },
    "rag": {
      "loaded": true,
      "document_count": 1000
    }
  }
}
```

---

## 分类预测接口

### 预测回复单位

**请求**

```
POST /api/predict
```

**参数**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| tag | string | 是 | 留言标签 |
| title | string | 是 | 留言标题 |
| body | string | 是 | 留言正文 |
| district | string | 否 | 行政区 |
| top_k | integer | 否 | 返回Top-K结果，默认3 |

**请求示例**

```json
{
  "tag": "投诉",
  "title": "小区垃圾没人清理",
  "body": "我们小区垃圾堆积严重，已经一周没人清理了，味道很大。",
  "district": "朝阳区",
  "top_k": 3
}
```

**响应**

```json
{
  "success": true,
  "data": {
    "predictions": [
      {
        "unit": "朝阳区城管委",
        "confidence": 85.5
      },
      {
        "unit": "朝阳区环卫中心",
        "confidence": 12.3
      },
      {
        "unit": "朝阳区住建委",
        "confidence": 2.2
      }
    ],
    "processing_time": 0.25
  }
}
```

---

## 回复生成接口

### 生成回复

**请求**

```
POST /api/generate
```

**参数**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| tag | string | 是 | 留言标签 |
| title | string | 是 | 留言标题 |
| body | string | 是 | 留言正文 |
| unit | string | 是 | 回复单位 |
| district | string | 否 | 行政区 |
| enable_verification | boolean | 否 | 是否启用事实验证，默认true |

**请求示例**

```json
{
  "tag": "投诉",
  "title": "小区垃圾没人清理",
  "body": "我们小区垃圾堆积严重，已经一周没人清理了。",
  "unit": "朝阳区城管委",
  "district": "朝阳区",
  "enable_verification": true
}
```

**响应**

```json
{
  "success": true,
  "data": {
    "reply": "经核实，您反映的小区垃圾堆积问题已转请朝阳区城管委结合现场情况进一步核查处理。后续将参考相关政策和类似案例中的办理口径，重点核实问题成因、责任主体和整改安排，并督促相关单位及时反馈办理进展。",
    "verification": {
      "is_valid": true,
      "warnings": [],
      "high_risk": false,
      "needs_review": false
    },
    "processing_time": 1.5
  }
}
```

---

## 地名识别接口

### 识别地名

**请求**

```
POST /api/location
```

**参数**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| text | string | 是 | 待识别文本 |
| district | string | 否 | 指定行政区 |

**请求示例**

```json
{
  "text": "朝阳区望京街道阜通东大街6号院",
  "district": "朝阳区"
}
```

**响应**

```json
{
  "success": true,
  "data": {
    "district": "朝阳区",
    "places": [
      {
        "matched_text": "望京街道",
        "district": "朝阳区",
        "source": "dictionary",
        "category": "administrative"
      },
      {
        "matched_text": "阜通东大街",
        "district": "朝阳区",
        "source": "dictionary",
        "category": "road"
      }
    ]
  }
}
```

---

## RAG检索接口

### 检索相关文档

**请求**

```
POST /api/search
```

**参数**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| query | string | 是 | 查询文本 |
| top_k | integer | 否 | 返回数量，默认5 |
| district | string | 否 | 行政区过滤 |
| doc_type | string | 否 | 文档类型过滤 |

**请求示例**

```json
{
  "query": "垃圾清运问题处理流程",
  "top_k": 3,
  "district": "朝阳区"
}
```

**响应**

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "title": "朝阳区垃圾清运管理办法",
        "doc_type": "policy",
        "district": "朝阳区",
        "snippet": "垃圾清运实行属地管理原则...",
        "score": 0.95,
        "full_content": "垃圾清运实行属地管理原则，各区城管委负责...",
        "retrieval_backend": "hybrid",
        "matched_terms": ["垃圾清运", "属地管理"],
        "dense_score": 0.92,
        "sparse_score": 0.88,
        "rerank_score": 0.95,
        "feedback_boost": 0.05
      }
    ],
    "total": 1
  }
}
```

> **字段说明**：
> - `full_content`: 文档完整内容（分块模式下可用于回溯到原文）
> - `retrieval_backend`: 检索后端类型，取值为 `hybrid`、`dense`、`sparse`、`empty`、`knowledge_graph`
> - `matched_terms`: 匹配的关键词列表
> - `dense_score` / `sparse_score`: 稠密/稀疏检索子分数
> - `rerank_score`: Cross-Encoder 重排序分数
> - `feedback_boost`: 用户反馈加权值（正为有用反馈，负为无用反馈）

---

## 反馈接口

### 提交反馈

**请求**

```
POST /api/feedback
```

**参数**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| tag | string | 是 | 留言标签 |
| title | string | 是 | 留言标题 |
| body | string | 是 | 留言正文 |
| reply | string | 是 | 生成的回复 |
| unit | string | 是 | 回复单位 |
| is_helpful | boolean | 是 | 是否有帮助 |
| comments | string | 否 | 评论 |

**请求示例**

```json
{
  "tag": "投诉",
  "title": "小区垃圾没人清理",
  "body": "我们小区垃圾堆积严重",
  "reply": "经核实，您反映的问题已转请...",
  "unit": "朝阳区城管委",
  "is_helpful": true,
  "comments": "回复很准确"
}
```

**响应**

```json
{
  "success": true,
  "data": {
    "feedback_id": 123,
    "message": "反馈提交成功"
  }
}
```

### 获取反馈统计

**请求**

```
GET /api/feedback/stats
```

**响应**

```json
{
  "success": true,
  "data": {
    "total_count": 1000,
    "helpful_count": 850,
    "unhelpful_count": 150,
    "helpful_rate": 85.0,
    "tag_distribution": [
      {"tag": "投诉", "count": 500},
      {"tag": "咨询", "count": 300}
    ]
  }
}
```

### 提交文档级反馈

用于 RAG 检索质量的反馈闭环，记录某条检索文档是否有帮助。

**请求**

```
POST /api/feedback/doc
```

**参数**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| doc_id | string | 是 | 文档ID |
| is_helpful | boolean | 是 | 是否有帮助 |
| query | string | 否 | 查询原文 |

**请求示例**

```json
{
  "doc_id": "policy-001",
  "is_helpful": true,
  "query": "小区垃圾清运问题"
}
```

**响应**

```json
{
  "success": true,
  "data": {
    "message": "文档反馈已记录"
  }
}
```

---

## RAG管理接口

### RAG热更新

无需重启服务，重新加载语料库并重建检索索引。

**请求**

```
POST /api/rag/reload
```

**响应**

```json
{
  "status": "ok",
  "old_document_count": 570,
  "new_document_count": 575,
  "old_chunk_count": 1820,
  "new_chunk_count": 1835,
  "active_backend": "hybrid"
}
```

> **使用场景**：向语料库追加新文档后，调用此接口即可热更新索引，无需重启服务。

---

## 知识图谱审核接口

### 获取待审核事实队列

**请求**

```
GET /api/knowledge-graph/review?status=pending&limit=20
```

**响应**

```json
{
  "candidates": [
    {
      "id": 1,
      "feedback_id": 42,
      "fact_type": "project_status",
      "fact_content": {"project": "XX工程", "status": "已完成"},
      "source_reply_type": "reference",
      "review_status": "pending"
    }
  ],
  "total": 1
}
```

### 审核并导入图谱

**请求**

```
POST /api/knowledge-graph/review/{candidate_id}
```

**参数**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| action | string | 是 | `approve` 或 `reject` |
| reviewer | string | 否 | 审核人 |
| review_notes | string | 否 | 审核备注 |

**响应**

```json
{
  "status": "ok",
  "candidate_id": 1,
  "action": "approved",
  "import_result": {"created_nodes": [], "created_relations": []}
}
```

---

## 管理接口

### 知识库管理

**获取知识库统计**

```
GET /api/admin/kb/stats
```

**更新知识库**

```
POST /api/admin/kb/update
```

**参数**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| fact_type | string | 是 | 事实类型 |
| fact_data | object | 是 | 事实数据 |

### 模型管理

**重新加载模型**

```
POST /api/admin/models/reload
```

**参数**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| model_type | string | 是 | 模型类型：classifier/generator |

---

## 错误码

| 错误码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 401 | 未授权 |
| 403 | 禁止访问 |
| 404 | 资源不存在 |
| 429 | 请求过于频繁 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用 |

**错误响应示例**

```json
{
  "success": false,
  "error": {
    "code": 400,
    "message": "缺少必填参数: tag",
    "details": "请求参数验证失败"
  }
}
```

---

## 速率限制

- 默认限制: 100次/分钟
- 认证用户: 1000次/分钟

超过限制将返回 `429` 错误。

---

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0.0 | 2024-01-01 | 初始版本 |
| 1.1.0 | 2024-02-01 | 添加事实验证接口 |
| 1.2.0 | 2024-03-01 | 添加知识库管理接口 |
| 1.3.0 | 2026-05-02 | 新增 `/api/rag/reload`、`/api/feedback/doc`、知识图谱审核接口；RAG检索响应新增 `full_content`、`retrieval_backend`、子分数字段 |
