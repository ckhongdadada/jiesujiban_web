from __future__ import annotations

import json
import os
from typing import Any


def normalize_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": row.get("id") or f"doc-{index:05d}",
        "title": row.get("title", "未命名材料"),
        "doc_type": row.get("doc_type", "参考材料"),
        "district": row.get("district", "全市"),
        "source": row.get("source", "导入知识库"),
        "tags": row.get("tags", []),
        "issue_type": row.get("issue_type", ""),
        "unit": row.get("unit", ""),
        "applicable_tags": row.get("applicable_tags", []),
        "content": row.get("content", ""),
    }


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(__file__))
    template_path = os.path.join(project_root, "data", "policy_case_corpus.template.jsonl")
    with open(template_path, "w", encoding="utf-8") as f:
        samples = [
            {
                "id": "policy-template-001",
                "title": "道路积水处置模板",
                "doc_type": "政策",
                "district": "全市",
                "source": "模板示例",
                "tags": ["道路积水", "排水设施"],
                "issue_type": "道路积水",
                "unit": "水务部门",
                "applicable_tags": ["投诉/求助", "建议"],
                "content": "请填写政策正文、办理原则、时限要求和处置要点。",
            },
            {
                "id": "case-template-001",
                "title": "物业卫生问题案例模板",
                "doc_type": "案例",
                "district": "朝阳区",
                "source": "模板示例",
                "tags": ["物业服务", "环境卫生", "垃圾清运"],
                "issue_type": "垃圾清运",
                "unit": "属地街道/社区",
                "applicable_tags": ["投诉/求助"],
                "content": "请填写问题背景、核查情况、整改动作和回访结果。",
            },
        ]
        for idx, row in enumerate(samples, start=1):
            f.write(json.dumps(normalize_row(row, idx), ensure_ascii=False) + "\n")
    print(template_path)


if __name__ == "__main__":
    main()
