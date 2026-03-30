# 接诉即办增强方案说明

## 我做了什么

本次只新增文件，没有删除或修改以下已有文件：
- `app.py`
- `requirements.txt`
- `templates/index.html`

新增了一个可单独运行的增强版入口：`app_enhanced.py`。
它在原有“分类 + 回复生成”之外，增加了两层前置能力：
1. 北京地名识别与 16 区归类
2. RAG 检索增强生成

## 可行性结论

### 1. 地名识别与归类（NER）
你的三个思路都可行，但建议做成分层策略，而不是只押一个方案。

推荐顺序：
1. 词典映射作为主干
2. LAC 作为可选增强
3. 高德地理编码作为兜底

原因：
- 词典最稳，尤其适合“北京 16 区归类”这个明确目标，维护成本可控，解释性强。
- LAC 能补充地名切分，但它不直接解决“归到 16 区”的问题，仍然需要词典或映射规则。
- 高德 API 对模糊地址有帮助，但依赖外网、密钥、配额，也会引入时延和失败兜底逻辑，因此更适合做 fallback，而不是主路径。

本次实现：
- 已新增 `enhancements/location_ner.py`
- 已新增 `enhancements/beijing_districts.json`
- 当前默认走“直接区名 + 词典别名”
- 如果设置 `ENABLE_LAC=true`，会尝试加载 LAC
- 如果设置 `AMAP_API_KEY`，在本地命中不足时会尝试调用高德地理编码

### 2. RAG 增强生成
方向完全可行，但我建议分两步走：

第一步：
- 先把“知识库结构、检索接口、生成提示词拼装”跑通
- 先用本地 TF-IDF 检索，保证无需额外依赖就能验证流程

第二步：
- 再升级为真正的向量库，例如 Chroma + SentenceTransformer
- 替换底层检索器，不改上层接口

原因：
- 你当前项目依赖里没有向量库和 embedding 依赖
- 直接上向量库会带来模型选择、持久化、重建索引、中文 embedding 质量等一系列新问题
- 先把流程跑通，成功率最高，也方便后续逐步替换

本次实现：
- 已新增 `enhancements/rag_retriever.py`
- 已新增示例知识库 `data/policy_case_corpus.sample.jsonl`
- 当前默认后端是 `tfidf`
- 已预留 `requirements_optional_rag_ner.txt`，便于后续升级向量库

## 运行方式

在项目目录下运行：

```powershell
python app_enhanced.py
```

默认端口：`5001`

打开：
- `http://127.0.0.1:5001`

## 后续建议

### NER 侧
- 把你手头已有的街道、社区、小区、地标词表继续填进 `enhancements/beijing_districts.json`
- 如果留言里经常出现“路口、地铁站、园区、医院、学校”，建议继续扩充这些别名
- 如果后续真实数据里地址写法很散，可以再补一层模糊匹配

### RAG 侧
- 把真实政策、回复案例整理成统一字段：`title/content/doc_type/district/tags/source`
- 先积累 200 到 1000 条高质量材料，再切向量库效果会更明显
- 如果后续你要，我可以继续在不动原文件的前提下，给你再新增一个 Chroma 版本索引脚本
