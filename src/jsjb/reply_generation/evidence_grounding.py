"""Evidence planning and sentence-level grounding for generated replies.

The goal of this module is deliberately conservative: before generation it
turns RAG hits into an evidence plan that tells the model what it may rely on;
after generation it checks every factual-looking sentence against those same
evidence items so unsupported claims can be reviewed instead of silently used.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


STOP_TERMS = {
    "关于",
    "问题",
    "情况",
    "相关",
    "进行",
    "处理",
    "核实",
    "办理",
    "回复",
    "反映",
    "留言",
    "已经",
    "目前",
    "后续",
    "进一步",
    "依法",
    "依规",
}


@dataclass
class AtomicFact:
    fact_id: str
    fact_type: str
    subject: str
    predicate: str
    value: str
    source_doc_id: str
    source_evidence_id: str
    source_text: str
    confidence: float = 0.8


@dataclass
class EvidenceItem:
    evidence_id: str
    doc_id: str
    title: str
    doc_type: str
    district: str
    unit: str
    score: float
    source: str
    evidence_text: str
    atomic_facts: list[AtomicFact] = field(default_factory=list)


@dataclass
class SentenceEvidenceBinding:
    sentence_index: int
    sentence: str
    is_factual: bool
    supported: bool
    support_score: float
    evidence_ids: list[str]
    matched_facts: list[str]
    reason: str


def _clip_text(text: str, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _hit_text(hit: dict[str, Any]) -> str:
    parts = [
        hit.get("child_snippet", ""),
        hit.get("snippet", ""),
        hit.get("parent_context", ""),
        hit.get("content", ""),
    ]
    return _clip_text(" ".join(str(part) for part in parts if part), limit=700)


def _term_set(text: str) -> set[str]:
    text = str(text or "")
    terms: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9_.-]+|[\u4e00-\u9fa5]{2,}", text):
        token = token.strip()
        if not token or token in STOP_TERMS:
            continue
        if re.fullmatch(r"[\u4e00-\u9fa5]{2,}", token):
            if 2 <= len(token) <= 8:
                terms.add(token)
            for n in (2, 3, 4):
                if len(token) >= n:
                    for i in range(0, len(token) - n + 1):
                        gram = token[i : i + n]
                        if gram not in STOP_TERMS:
                            terms.add(gram)
        else:
            terms.add(token.lower())
    return terms


def _source_text_for_fact(hit: dict[str, Any], evidence_text: str, field: str) -> str:
    value = str(hit.get(field, "") or "")
    if value:
        return value
    return evidence_text


def _add_fact(
    facts: list[AtomicFact],
    fact_type: str,
    subject: str,
    predicate: str,
    value: Any,
    evidence: EvidenceItem,
    source_text: str,
    confidence: float = 0.85,
) -> None:
    value_text = str(value or "").strip()
    if not value_text:
        return
    fact_id = f"{evidence.evidence_id}-F{len(facts) + 1}"
    facts.append(
        AtomicFact(
            fact_id=fact_id,
            fact_type=fact_type,
            subject=subject or evidence.title,
            predicate=predicate,
            value=value_text,
            source_doc_id=evidence.doc_id,
            source_evidence_id=evidence.evidence_id,
            source_text=_clip_text(source_text, 180),
            confidence=confidence,
        )
    )


def _extract_status_facts(evidence: EvidenceItem, hit: dict[str, Any], evidence_text: str) -> list[AtomicFact]:
    facts: list[AtomicFact] = []
    subject = hit.get("title") or evidence.title or "参考材料"
    _add_fact(facts, "district", subject, "belongs_to_district", hit.get("district"), evidence, evidence.district)
    _add_fact(facts, "unit", subject, "handled_by", hit.get("unit") or hit.get("responsible_unit"), evidence, evidence.unit)
    _add_fact(
        facts,
        "project_status",
        subject,
        "project_status",
        hit.get("project_status"),
        evidence,
        _source_text_for_fact(hit, evidence_text, "project_status"),
        confidence=0.9,
    )
    _add_fact(
        facts,
        "demolition_status",
        subject,
        "demolition_status",
        hit.get("demolition_status"),
        evidence,
        _source_text_for_fact(hit, evidence_text, "demolition_status"),
        confidence=0.9,
    )
    _add_fact(
        facts,
        "resource_status",
        subject,
        "resource_status",
        hit.get("resource_status"),
        evidence,
        _source_text_for_fact(hit, evidence_text, "resource_status"),
        confidence=0.9,
    )

    status_patterns = [
        (r"已(?:完成|完工|竣工|交付|建成|开放|运营|通车)", "status_phrase", "已完成/已运营"),
        (r"正在(?:施工|建设|推进|办理|整改|核查)", "status_phrase", "进行中"),
        (r"暂未(?:开始|启动|建设|开放)", "status_phrase", "暂未开始"),
        (r"计划.{0,12}(?:完成|开放|通车|启用)", "plan_phrase", "计划安排"),
    ]
    for pattern, fact_type, predicate in status_patterns:
        for match in re.finditer(pattern, evidence_text):
            _add_fact(
                facts,
                fact_type,
                subject,
                predicate,
                match.group(0),
                evidence,
                match.group(0),
                confidence=0.72,
            )
    return facts


def build_evidence_plan(
    retrieval_hits: list[dict[str, Any]],
    unit: str = "",
    location_result: dict[str, Any] | None = None,
    max_evidence: int = 3,
) -> dict[str, Any]:
    """Build a compact evidence plan from top RAG hits."""
    location_result = location_result or {}
    evidence_items: list[EvidenceItem] = []
    allowed_claims: list[str] = []

    for idx, hit in enumerate((retrieval_hits or [])[:max_evidence], start=1):
        evidence_id = f"E{idx}"
        evidence_text = _hit_text(hit)
        item = EvidenceItem(
            evidence_id=evidence_id,
            doc_id=str(hit.get("doc_id") or hit.get("id") or evidence_id),
            title=str(hit.get("title") or "未命名材料"),
            doc_type=str(hit.get("doc_type") or "参考材料"),
            district=str(hit.get("district") or ""),
            unit=str(hit.get("responsible_unit") or hit.get("unit") or unit or ""),
            score=float(hit.get("score") or 0.0),
            source=str(hit.get("source") or "本地知识库"),
            evidence_text=evidence_text,
        )
        item.atomic_facts = _extract_status_facts(item, hit, evidence_text)
        evidence_items.append(item)

    district = str(location_result.get("district") or "")
    if district:
        allowed_claims.append(f"输入地名识别结果限定行政区为「{district}」。")
    if unit:
        allowed_claims.append(f"分类/用户选择的办理单位为「{unit}」。")

    for item in evidence_items:
        for fact in item.atomic_facts:
            allowed_claims.append(
                f"{fact.source_evidence_id}: {fact.subject} / {fact.predicate} = {fact.value}"
            )

    return {
        "evidence_items": [asdict(item) for item in evidence_items],
        "allowed_claims": allowed_claims,
        "required_constraints": [
            "只能输出输入字段或证据计划中出现的具体行政区、单位、状态、时间、资源信息。",
            "没有证据支持的事实只能写成待核实、将督促、将反馈等保守办理表述。",
            "不得编造现场核查结论、处理时限、责任细节、项目进展或资源状态。",
        ],
        "unsupported_policy": "生成后若事实性句子无法绑定证据，应标记人工复核。",
        "coverage": "none" if not evidence_items else ("structured" if any(item.atomic_facts for item in evidence_items) else "text"),
    }


def format_evidence_plan_for_prompt(plan: dict[str, Any]) -> str:
    """Render the evidence plan into a prompt-friendly block."""
    if not plan.get("evidence_items"):
        return "无证据计划：当前没有可引用的检索材料，只能输出保守办理方向。"

    lines = ["证据计划（仅供生成时内部依据，不要在回复正文中输出证据编号）："]
    for item in plan.get("evidence_items", []):
        lines.append(
            f"- {item['evidence_id']} | {item.get('title', '未命名材料')} | "
            f"区域:{item.get('district') or '全市'} | 单位:{item.get('unit') or '未给出'} | "
            f"摘要:{_clip_text(item.get('evidence_text', ''), 260)}"
        )
        for fact in item.get("atomic_facts", [])[:6]:
            lines.append(
                f"  * {fact['fact_id']}: {fact['predicate']} = {fact['value']} "
                f"（来源:{fact['source_evidence_id']}）"
            )

    if plan.get("allowed_claims"):
        lines.append("允许使用的具体事实：")
        for claim in plan["allowed_claims"][:12]:
            lines.append(f"- {claim}")

    lines.append("事实约束：")
    for constraint in plan.get("required_constraints", []):
        lines.append(f"- {constraint}")
    return "\n".join(lines)


def split_reply_sentences(reply: str) -> list[str]:
    sentences = []
    for sentence in re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", str(reply or "")):
        sentence = sentence.strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def _is_factual_sentence(sentence: str) -> bool:
    factual_patterns = [
        r"经.*?核实",
        r"已(?:完成|完工|建成|开放|运营|转办|整改|清理|处理)",
        r"正在(?:施工|建设|推进|办理|整改|核查)",
        r"暂未|尚未|不涉及|未发现|不存在|无此",
        r"\d{4}年|\d{1,2}月|\d+日|\d+个工作日",
        r"(?:区|街道|镇|乡|局|委|办|公司|集团)",
        r"(?:道路|工程|项目|图书馆|学校|医院|公园|小区|物业)",
    ]
    return any(re.search(pattern, sentence) for pattern in factual_patterns)


def _score_sentence_against_evidence(sentence: str, item: dict[str, Any]) -> tuple[float, list[str]]:
    sentence_terms = _term_set(sentence)
    evidence_terms = _term_set(
        " ".join(
            [
                item.get("title", ""),
                item.get("district", ""),
                item.get("unit", ""),
                item.get("evidence_text", ""),
            ]
        )
    )
    matched_terms = sentence_terms & evidence_terms
    score = 0.0
    if sentence_terms:
        score += min(0.55, len(matched_terms) / max(len(sentence_terms), 1))

    matched_facts = []
    for fact in item.get("atomic_facts", []):
        value = str(fact.get("value", "") or "")
        source_text = str(fact.get("source_text", "") or "")
        predicate = str(fact.get("predicate", "") or "")
        if value and value in sentence:
            score += 0.35
            matched_facts.append(fact.get("fact_id", ""))
        elif source_text and source_text in sentence:
            score += 0.25
            matched_facts.append(fact.get("fact_id", ""))
        elif predicate and value and (predicate in sentence or value in sentence):
            score += 0.15
            matched_facts.append(fact.get("fact_id", ""))

    return min(score, 1.0), [fact_id for fact_id in matched_facts if fact_id]


def bind_reply_to_evidence(
    reply: str,
    evidence_plan: dict[str, Any],
    min_support_score: float = 0.18,
) -> dict[str, Any]:
    """Bind every generated sentence to supporting evidence where possible."""
    bindings: list[SentenceEvidenceBinding] = []
    unsupported: list[dict[str, Any]] = []
    evidence_items = evidence_plan.get("evidence_items", []) or []

    for idx, sentence in enumerate(split_reply_sentences(reply), start=1):
        factual = _is_factual_sentence(sentence)
        best_ids: list[str] = []
        best_facts: list[str] = []
        best_score = 0.0

        for item in evidence_items:
            score, facts = _score_sentence_against_evidence(sentence, item)
            if score > best_score:
                best_score = score
                best_ids = [item.get("evidence_id", "")]
                best_facts = facts

        supported = (not factual) or best_score >= min_support_score
        reason = "非事实性/流程性表述" if not factual else ("已绑定证据" if supported else "事实性句子未找到足够证据支持")
        binding = SentenceEvidenceBinding(
            sentence_index=idx,
            sentence=sentence,
            is_factual=factual,
            supported=supported,
            support_score=round(best_score, 4),
            evidence_ids=[e for e in best_ids if e],
            matched_facts=best_facts,
            reason=reason,
        )
        bindings.append(binding)
        if factual and not supported:
            unsupported.append(asdict(binding))

    factual_count = sum(1 for item in bindings if item.is_factual)
    supported_count = sum(1 for item in bindings if item.is_factual and item.supported)
    return {
        "bindings": [asdict(item) for item in bindings],
        "unsupported_sentences": unsupported,
        "factual_sentence_count": factual_count,
        "supported_factual_sentence_count": supported_count,
        "coverage_rate": round(supported_count / factual_count, 4) if factual_count else 1.0,
        "needs_review": bool(unsupported),
        "summary": "全部事实性句子均已绑定证据" if not unsupported else f"发现{len(unsupported)}句事实性表述缺少证据绑定",
    }
