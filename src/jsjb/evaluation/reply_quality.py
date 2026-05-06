"""Standard generated-reply quality evaluation."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from src.jsjb.reply_generation.error_analysis import ReplyErrorExtractor


DEFAULT_WEIGHTS = {
    "semantic_similarity": 0.25,
    "keyword_coverage": 0.20,
    "actionability": 0.20,
    "grounding": 0.15,
    "format": 0.10,
    "length": 0.10,
}

ACTION_KEYWORDS = [
    "核实",
    "查看",
    "现场",
    "协调",
    "处理",
    "整改",
    "清理",
    "维修",
    "督促",
    "责成",
    "反馈",
    "下一步",
]

FORMAT_BAD_PATTERNS = [
    r"您好|你好|尊敬的",
    r"感谢您|欢迎您|祝您|特此回复",
    r"\d{4}年\d{1,2}月\d{1,2}日\s*$",
]


@dataclass
class ReplyQualityConfig:
    """Weights and options for reply-quality scoring."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    min_length: int = 40
    max_length: int = 260
    use_bge_similarity: bool = False
    bge_model_name: str = "BAAI/bge-small-zh-v1.5"


class ReplyQualityEvaluator:
    """Evaluate generated government replies against references and evidence."""

    def __init__(self, config: ReplyQualityConfig | None = None) -> None:
        self.config = config or ReplyQualityConfig()
        self.error_extractor = ReplyErrorExtractor()
        self._embedding_model = None

    def evaluate_rows(self, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
        cases = [self.evaluate_one(row, index=index) for index, row in enumerate(rows)]
        return {
            "metrics": self.build_summary(cases),
            "cases": cases,
        }

    def evaluate_one(self, row: dict[str, Any], index: int = 0) -> dict[str, Any]:
        generated = self._pick(row, "generated_reply", "reply", "模型回复", "生成回复")
        reference = self._pick(row, "reference_reply", "official_reply", "官方回复正文", "标准回复")
        title = self._pick(row, "title", "留言标题")
        body = self._pick(row, "body", "留言正文")
        tag = self._pick(row, "tag", "留言标签")
        unit = self._pick(row, "unit", "官方回复单位", "expected_unit")
        district = self._pick(row, "district", "区县", "expected_district")
        retrieval_hits = self._parse_jsonish(row.get("rag_docs") or row.get("retrieval") or row.get("retrieval_hits") or [])

        rouge = compute_rouge(generated, reference)
        bleu4 = compute_bleu(generated, reference, max_n=4)
        semantic = self.semantic_similarity(generated, reference)
        keyword = keyword_coverage(generated, reference)
        actionability = actionability_score(generated)
        format_score = format_compliance_score(generated)
        length_score = length_compliance_score(generated, self.config.min_length, self.config.max_length)
        grounding = grounding_score(generated, retrieval_hits)
        quality = weighted_quality_score(
            {
                "semantic_similarity": semantic,
                "keyword_coverage": keyword,
                "actionability": actionability,
                "grounding": grounding,
                "format": format_score,
                "length": length_score,
            },
            self.config.weights,
        )
        attribution = self.error_extractor.analyze(
            generated_reply=generated,
            reference_reply=reference,
            retrieval_hits=retrieval_hits,
            location_result={"district": district},
            expected_unit=unit,
            feedback_type="offline_reply_quality_eval",
            comments="",
        )

        return {
            "index": index,
            "id": row.get("id", row.get("case_id", index)),
            "tag": tag,
            "title": title,
            "body": body,
            "district": district,
            "unit": unit,
            "generated_reply": generated,
            "reference_reply": reference,
            "metrics": {
                **rouge,
                "bleu_4": bleu4,
                "semantic_similarity": semantic,
                "keyword_coverage": keyword,
                "actionability_score": actionability,
                "grounding_score": grounding,
                "format_score": format_score,
                "length_score": length_score,
                "quality_score": quality,
            },
            "error_tags": attribution.get("error_types", []),
            "severity": attribution.get("severity", "low"),
            "diagnosis": attribution.get("summary", ""),
            "routing_recommendations": attribution.get("routing_recommendations", []),
            "quality_attribution": attribution,
        }

    def semantic_similarity(self, generated: str, reference: str) -> float:
        if not generated or not reference:
            return 0.0
        if self.config.use_bge_similarity:
            try:
                model = self._load_embedding_model()
                vectors = model.encode([generated, reference], normalize_embeddings=True, show_progress_bar=False)
                return round(float(vectors[0] @ vectors[1]), 4)
            except Exception:
                pass
        return round(SequenceMatcher(None, generated, reference).ratio(), 4)

    def _load_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.config.bge_model_name)
        return self._embedding_model

    @staticmethod
    def _pick(row: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    @staticmethod
    def _parse_jsonish(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [item for item in parsed if isinstance(item, dict)]
            except json.JSONDecodeError:
                return []
        return []

    @staticmethod
    def build_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
        ok_cases = [item for item in cases if item.get("generated_reply")]
        metric_keys = [
            "rouge_1",
            "rouge_2",
            "rouge_l",
            "bleu_4",
            "semantic_similarity",
            "keyword_coverage",
            "actionability_score",
            "grounding_score",
            "format_score",
            "length_score",
            "quality_score",
        ]
        averages = {}
        for key in metric_keys:
            values = [float(item.get("metrics", {}).get(key, 0.0) or 0.0) for item in ok_cases]
            averages[key] = round(sum(values) / len(values), 4) if values else 0.0

        error_counts: Counter[str] = Counter()
        severity_counts: Counter[str] = Counter()
        for item in ok_cases:
            severity_counts[str(item.get("severity") or "unknown")] += 1
            for tag in item.get("error_tags", []) or []:
                error_counts[str(tag)] += 1

        worst_cases = sorted(
            [
                {
                    "id": item.get("id"),
                    "title": item.get("title", ""),
                    "quality_score": item.get("metrics", {}).get("quality_score", 0),
                    "severity": item.get("severity", ""),
                    "error_tags": item.get("error_tags", []),
                    "diagnosis": item.get("diagnosis", ""),
                }
                for item in ok_cases
            ],
            key=lambda item: item["quality_score"],
        )[:20]

        return {
            "sample_count": len(cases),
            "evaluated_count": len(ok_cases),
            "metric_averages": averages,
            "error_distribution": dict(error_counts.most_common()),
            "severity_distribution": dict(severity_counts.most_common()),
            "format_violation_rate": round(
                sum(1 for item in ok_cases if item.get("metrics", {}).get("format_score", 1.0) < 1.0)
                / max(len(ok_cases), 1),
                4,
            ),
            "needs_review_rate": round(
                sum(1 for item in ok_cases if item.get("severity") in {"medium", "high"})
                / max(len(ok_cases), 1),
                4,
            ),
            "worst_cases": worst_cases,
            "recommendation": build_release_recommendation(averages, error_counts),
        }


def tokenize_text(text: str) -> list[str]:
    return [char for char in str(text or "") if not char.isspace()]


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_rouge(generated: str, reference: str) -> dict[str, float]:
    gen = tokenize_text(generated)
    ref = tokenize_text(reference)
    if not gen or not ref:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
    return {
        "rouge_1": round(overlap_recall(gen, ref, 1), 4),
        "rouge_2": round(overlap_recall(gen, ref, 2), 4),
        "rouge_l": round(lcs_f1(gen, ref), 4),
    }


def overlap_recall(generated_tokens: list[str], reference_tokens: list[str], n: int) -> float:
    gen_ngrams = ngrams(generated_tokens, n)
    ref_ngrams = ngrams(reference_tokens, n)
    if not ref_ngrams:
        return 0.0
    overlap = sum(min(count, gen_ngrams.get(gram, 0)) for gram, count in ref_ngrams.items())
    return overlap / max(sum(ref_ngrams.values()), 1)


def lcs_f1(generated_tokens: list[str], reference_tokens: list[str]) -> float:
    m, n = len(generated_tokens), len(reference_tokens)
    if not m or not n:
        return 0.0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if generated_tokens[i - 1] == reference_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[n]
    precision = lcs / m
    recall = lcs / n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_bleu(generated: str, reference: str, max_n: int = 4) -> float:
    gen = tokenize_text(generated)
    ref = tokenize_text(reference)
    if not gen or not ref:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        gen_ngrams = ngrams(gen, n)
        ref_ngrams = ngrams(ref, n)
        overlap = sum(min(count, ref_ngrams.get(gram, 0)) for gram, count in gen_ngrams.items())
        total = max(sum(gen_ngrams.values()), 1)
        precisions.append((overlap + 1) / (total + 1))
    brevity = 1.0 if len(gen) > len(ref) else math.exp(1 - len(ref) / max(len(gen), 1))
    return round(brevity * math.exp(sum(math.log(p) for p in precisions) / max_n), 4)


def keyword_coverage(generated: str, reference: str) -> float:
    keywords = extract_keywords(reference)
    if not keywords:
        return 1.0 if generated else 0.0
    hit = sum(1 for keyword in keywords if keyword in generated)
    return round(hit / len(keywords), 4)


def extract_keywords(text: str) -> list[str]:
    candidates = set()
    for pattern in [
        r"[\u4e00-\u9fa5]{2,12}(?:区|街道|镇|乡|社区|小区|村)",
        r"[\u4e00-\u9fa5]{2,20}(?:委|局|办|办事处|政府|中心|公司)",
        r"[\u4e00-\u9fa5]{2,12}(?:问题|诉求|维修|整改|清理|处理|协调)",
    ]:
        for match in re.findall(pattern, text or ""):
            candidates.add(match)
    return sorted(candidates)


def actionability_score(text: str) -> float:
    if not text:
        return 0.0
    hits = sum(1 for keyword in ACTION_KEYWORDS if keyword in text)
    return round(min(hits / 3, 1.0), 4)


def format_compliance_score(text: str) -> float:
    if not text:
        return 0.0
    violations = sum(1 for pattern in FORMAT_BAD_PATTERNS if re.search(pattern, text))
    return round(max(1.0 - violations * 0.35, 0.0), 4)


def length_compliance_score(text: str, min_length: int, max_length: int) -> float:
    length = len(text or "")
    if length == 0:
        return 0.0
    if min_length <= length <= max_length:
        return 1.0
    if length < min_length:
        return round(max(length / max(min_length, 1), 0.0), 4)
    return round(max(1.0 - (length - max_length) / max(max_length, 1), 0.0), 4)


def grounding_score(generated: str, retrieval_hits: list[dict[str, Any]]) -> float:
    if not generated:
        return 0.0
    if not retrieval_hits:
        return 0.5
    evidence_text = "\n".join(
        str(hit.get("content") or hit.get("snippet") or hit.get("title") or "")
        for hit in retrieval_hits[:5]
    )
    if not evidence_text.strip():
        return 0.5
    evidence_keywords = extract_keywords(evidence_text)
    if not evidence_keywords:
        return 0.65
    hit = sum(1 for keyword in evidence_keywords if keyword in generated)
    return round(min(0.4 + 0.6 * hit / len(evidence_keywords), 1.0), 4)


def weighted_quality_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(weights.values()) or 1.0
    score = sum(float(metrics.get(key, 0.0) or 0.0) * weight for key, weight in weights.items()) / total_weight
    return round(score, 4)


def build_release_recommendation(averages: dict[str, float], error_counts: Counter[str]) -> str:
    quality = averages.get("quality_score", 0.0)
    if quality >= 0.78 and error_counts.get("fact_conflict", 0) == 0:
        return "建议进入人工抽检或小流量演示。"
    if quality >= 0.65:
        return "建议保留当前版本，但优先复核低分样本和事实冲突样本。"
    return "不建议直接上线，优先优化 RAG 依据、生成 Prompt 或 LoRA 数据。"


def read_quality_rows(path: str) -> list[dict[str, Any]]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with input_path.open("r", encoding="utf-8-sig") as fp:
            for line in fp:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, list) else payload.get("cases", [])
    if suffix == ".csv":
        import pandas as pd

        return pd.read_csv(input_path).fillna("").to_dict(orient="records")
    if suffix in {".xlsx", ".xls"}:
        import pandas as pd

        return pd.read_excel(input_path).fillna("").to_dict(orient="records")
    raise ValueError(f"Unsupported input file type: {input_path.suffix}")


def load_reply_quality_config_file(path: str | None = None) -> ReplyQualityConfig:
    if path is None:
        from src.jsjb.core.config import load_reply_quality_config

        payload = load_reply_quality_config()
    else:
        payload_path = Path(path)
        payload = json.loads(payload_path.read_text(encoding="utf-8-sig")) if payload_path.exists() else {}
    weights = payload.get("quality_score_weights") or payload.get("weights") or DEFAULT_WEIGHTS
    return ReplyQualityConfig(
        weights=weights,
        min_length=int(payload.get("min_length", 40)),
        max_length=int(payload.get("max_length", 260)),
        use_bge_similarity=bool(payload.get("use_bge_similarity", False)),
        bge_model_name=payload.get("bge_model_name", "BAAI/bge-small-zh-v1.5"),
    )


def write_quality_outputs(result: dict[str, Any], output_dir: str) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    cases_path = output / "cases.jsonl"
    report_path = output / "report.md"
    xlsx_path = output / "report.xlsx"

    metrics_path.write_text(json.dumps(result.get("metrics", {}), ensure_ascii=False, indent=2), encoding="utf-8")
    with cases_path.open("w", encoding="utf-8") as fp:
        for case in result.get("cases", []):
            fp.write(json.dumps(case, ensure_ascii=False) + "\n")
    report_path.write_text(render_markdown_report(result), encoding="utf-8")

    written = {
        "metrics_json": str(metrics_path),
        "cases_jsonl": str(cases_path),
        "report_md": str(report_path),
    }
    try:
        import pandas as pd

        rows = []
        for case in result.get("cases", []):
            row = {
                "id": case.get("id"),
                "title": case.get("title"),
                "district": case.get("district"),
                "unit": case.get("unit"),
                "severity": case.get("severity"),
                "error_tags": ",".join(case.get("error_tags", []) or []),
                "diagnosis": case.get("diagnosis", ""),
            }
            row.update(case.get("metrics", {}))
            rows.append(row)
        pd.DataFrame(rows).to_excel(xlsx_path, index=False)
        written["report_xlsx"] = str(xlsx_path)
    except Exception as exc:
        written["xlsx_error"] = str(exc)
    return written


def render_markdown_report(result: dict[str, Any]) -> str:
    metrics = result.get("metrics", {})
    averages = metrics.get("metric_averages", {})
    lines = [
        "# 生成回复质量评测报告",
        "",
        "## 1. 总览",
        "",
        f"- 样本数：`{metrics.get('sample_count', 0)}`",
        f"- 有效评测数：`{metrics.get('evaluated_count', 0)}`",
        f"- 需复核比例：`{metrics.get('needs_review_rate', 0)}`",
        f"- 格式违规比例：`{metrics.get('format_violation_rate', 0)}`",
        f"- 上线建议：{metrics.get('recommendation', '')}",
        "",
        "## 2. 平均指标",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
    ]
    for key, value in averages.items():
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend(["", "## 3. 错误分布", "", "| 错误类型 | 次数 |", "|---|---:|"])
    for key, value in (metrics.get("error_distribution") or {}).items():
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend(["", "## 4. 最差样本 Top20", "", "| ID | 标题 | 综合分 | 严重度 | 错误标签 |", "|---|---|---:|---|---|"])
    for item in metrics.get("worst_cases", []) or []:
        tags = ",".join(item.get("error_tags", []) or [])
        title = str(item.get("title", "")).replace("|", " ")
        lines.append(f"| `{item.get('id')}` | {title[:60]} | `{item.get('quality_score')}` | `{item.get('severity')}` | `{tags}` |")

    lines.extend([
        "",
        "## 5. 指标说明",
        "",
        "- `ROUGE-1/2/L`：衡量生成回复与标准回复的字词覆盖和结构相似度。",
        "- `BLEU-4`：衡量生成回复 n-gram 精确匹配程度，通常更严格。",
        "- `semantic_similarity`：默认使用文本序列相似度，可选启用 BGE 向量余弦相似度。",
        "- `keyword_coverage`：衡量标准回复中的关键单位、地点、动作是否被覆盖。",
        "- `actionability_score`：衡量回复是否包含核实、处理、整改、协调等办理动作。",
        "- `grounding_score`：衡量回复与 RAG 参考材料的依据一致性。",
    ])
    return "\n".join(lines) + "\n"
