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
9. [管理接口](#管理接口)
10. [错误码](#错误码)

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
        "score": 0.95
      }
    ],
    "total": 1
  }
}
```

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
