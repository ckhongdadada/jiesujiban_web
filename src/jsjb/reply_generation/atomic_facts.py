"""Atomic fact decomposition for generated government replies.

This module splits a reply into sentence-level and claim-level facts so later
quality attribution can answer: what exactly did the model claim, where did it
come from, and which claim needs review.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from src.jsjb.reply_generation.evidence_grounding import split_reply_sentences


BEIJING_DISTRICTS = [
    "东城区",
    "西城区",
    "朝阳区",
    "海淀区",
    "丰台区",
    "石景山区",
    "门头沟区",
    "房山区",
    "通州区",
    "顺义区",
    "昌平区",
    "大兴区",
    "怀柔区",
    "平谷区",
    "密云区",
    "延庆区",
]


@dataclass
class ReplyAtomicFact:
    fact_id: str
    sentence_index: int
    sentence: str
    fact_type: str
    subject: str
    predicate: str
    value: str
    polarity: str
    confidence: float
    evidence_ids: list[str]
    source: str = "generated_reply"


class ReplyAtomicFactExtractor:
    """Rule-based atomic fact extractor for concise generated replies."""

    STATUS_PATTERNS = [
        (r"已(?:完成|完工|竣工|交付|建成|开放|运营|通车|整改|清理|处理|转办|安排|责成|督促)", "status", "affirmed", 0.86),
        (r"正在(?:施工|建设|推进|办理|整改|核查|协调|研究)", "status", "affirmed", 0.78),
        (r"将(?:继续)?(?:督促|协调|推进|核实|处理|反馈|整改)", "future_action", "planned", 0.76),
        (r"暂未(?:开始|启动|建设|开放|发现)|尚未(?:建设|开放|发现)|未发现|不存在|不涉及", "status", "negated", 0.84),
    ]
    ACTION_PATTERNS = [
        (r"已(?:安排|责成|督促|协调|处理|整改|清理|维修|转办)", "completed_action", "affirmed", 0.82),
        (r"现场(?:核查|查看)|经(?:核实|核查)", "verification_action", "affirmed", 0.80),
        (r"将(?:安排|责成|督促|协调|处理|整改|清理|维修|反馈|核实)", "planned_action", "planned", 0.76),
    ]
    TIME_PATTERN = re.compile(r"(?:\d{4}年)?\d{1,2}月\d{1,2}日|预计.{0,12}(?:完成|开放|通车|启用)|\d+个工作日")
    UNIT_PATTERN = re.compile(r"([\u4e00-\u9fa5]{2,24}(?:委员会|办公室|管理局|管理委|办事处|镇政府|街道|政府|中心|公司|集团|局|委|办))")
    PLACE_PATTERN = re.compile(r"([\u4e00-\u9fa5]{2,24}(?:小区|道路|路|街|桥|公园|学校|医院|图书馆|广场|工程|项目))")
    RESOURCE_PATTERN = re.compile(r"([\u4e00-\u9fa5]{2,24}(?:图书馆|图书室|学校|幼儿园|医院|卫生院|公园|广场|停车场))")

    def extract(self, reply: str, evidence_plan: dict[str, Any] | None = None) -> list[ReplyAtomicFact]:
        facts: list[ReplyAtomicFact] = []
        evidence_plan = evidence_plan or {}
        for sentence_index, sentence in enumerate(split_reply_sentences(reply), start=1):
            subjects = self._subjects(sentence)
            subject = subjects[0] if subjects else "本诉求"
            facts.extend(self._status_facts(sentence_index, sentence, subject))
            facts.extend(self._action_facts(sentence_index, sentence, subject))
            facts.extend(self._entity_facts(sentence_index, sentence))
            facts.extend(self._time_facts(sentence_index, sentence, subject))

        evidence_ids = [item.get("evidence_id", "") for item in evidence_plan.get("evidence_items", []) or []]
        for idx, fact in enumerate(facts, start=1):
            fact.fact_id = f"A{idx}"
            if evidence_ids and not fact.evidence_ids:
                fact.evidence_ids = evidence_ids[:3]
        return facts

    def to_dicts(self, reply: str, evidence_plan: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.extract(reply, evidence_plan=evidence_plan)]

    def _subjects(self, sentence: str) -> list[str]:
        subjects = []
        subjects.extend(DISTRICT for DISTRICT in BEIJING_DISTRICTS if DISTRICT in sentence)
        subjects.extend(match.group(1) for match in self.PLACE_PATTERN.finditer(sentence))
        subjects.extend(match.group(1) for match in self.RESOURCE_PATTERN.finditer(sentence))
        return list(dict.fromkeys(subjects))

    def _status_facts(self, sentence_index: int, sentence: str, subject: str) -> list[ReplyAtomicFact]:
        facts = []
        for pattern, predicate, polarity, confidence in self.STATUS_PATTERNS:
            for match in re.finditer(pattern, sentence):
                facts.append(
                    ReplyAtomicFact(
                        fact_id="",
                        sentence_index=sentence_index,
                        sentence=sentence,
                        fact_type="status",
                        subject=subject,
                        predicate=predicate,
                        value=match.group(0),
                        polarity=polarity,
                        confidence=confidence,
                        evidence_ids=[],
                    )
                )
        return facts

    def _action_facts(self, sentence_index: int, sentence: str, subject: str) -> list[ReplyAtomicFact]:
        facts = []
        for pattern, predicate, polarity, confidence in self.ACTION_PATTERNS:
            for match in re.finditer(pattern, sentence):
                facts.append(
                    ReplyAtomicFact(
                        fact_id="",
                        sentence_index=sentence_index,
                        sentence=sentence,
                        fact_type="action",
                        subject=subject,
                        predicate=predicate,
                        value=match.group(0),
                        polarity=polarity,
                        confidence=confidence,
                        evidence_ids=[],
                    )
                )
        return facts

    def _entity_facts(self, sentence_index: int, sentence: str) -> list[ReplyAtomicFact]:
        facts = []
        for district in BEIJING_DISTRICTS:
            if district in sentence:
                facts.append(
                    ReplyAtomicFact("", sentence_index, sentence, "district", "本诉求", "located_in", district, "affirmed", 0.92, [])
                )
        for match in self.UNIT_PATTERN.finditer(sentence):
            facts.append(
                ReplyAtomicFact("", sentence_index, sentence, "unit", "本诉求", "handled_by", match.group(1), "affirmed", 0.82, [])
            )
        for match in self.RESOURCE_PATTERN.finditer(sentence):
            facts.append(
                ReplyAtomicFact("", sentence_index, sentence, "resource", match.group(1), "mentioned", match.group(1), "affirmed", 0.74, [])
            )
        return facts

    def _time_facts(self, sentence_index: int, sentence: str, subject: str) -> list[ReplyAtomicFact]:
        return [
            ReplyAtomicFact("", sentence_index, sentence, "time", subject, "time_commitment", match.group(0), "affirmed", 0.80, [])
            for match in self.TIME_PATTERN.finditer(sentence)
        ]


def split_reply_to_atomic_facts(
    reply: str,
    evidence_plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper returning JSON-serializable atomic facts."""
    return ReplyAtomicFactExtractor().to_dicts(reply, evidence_plan=evidence_plan)
