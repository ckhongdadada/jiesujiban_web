#!/usr/bin/env python3
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.jsjb.retrieval.bge_retriever import PolicyRetriever


def load_eval_set():
    path = os.path.join(PROJECT_ROOT, "data", "runtime", "retrieval_eval_set.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(top_k=5):
    eval_set = load_eval_set()
    retriever = PolicyRetriever()

    results = []
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    district_match_top1 = 0
    issue_type_match_top1 = 0

    for item in eval_set:
        query = item["query"]
        district = item.get("district", "")
        expected_keywords = item.get("expected_keywords", [])
        expected_issue_type = item.get("expected_issue_type", "")
        acceptable_doc_types = item.get("acceptable_doc_types", [])

        hits = retriever.search(query, district=district or None, top_k=top_k)

        hit_details = []
        for rank, hit in enumerate(hits, 1):
            keyword_match = any(kw in hit.get("title", "") or kw in hit.get("snippet", "") for kw in expected_keywords)
            district_ok = not district or hit.get("district", "") in (district, "全市")
            issue_type_ok = not expected_issue_type or hit.get("issue_type", "") == expected_issue_type
            doc_type_ok = not acceptable_doc_types or hit.get("doc_type", "") in acceptable_doc_types
            hit_details.append({
                "rank": rank,
                "title": hit.get("title", ""),
                "district": hit.get("district", ""),
                "score": hit.get("score", 0),
                "keyword_match": keyword_match,
                "district_ok": district_ok,
                "issue_type_ok": issue_type_ok,
                "doc_type_ok": doc_type_ok,
            })

        def is_acceptable(detail):
            return detail["keyword_match"] and detail["district_ok"]

        top1_ok = any(is_acceptable(d) for d in hit_details[:1])
        top3_ok = any(is_acceptable(d) for d in hit_details[:3])
        top5_ok = any(is_acceptable(d) for d in hit_details[:5])

        if top1_ok:
            top1_hits += 1
        if top3_ok:
            top3_hits += 1
        if top5_ok:
            top5_hits += 1

        if hit_details and hit_details[0]["district_ok"]:
            district_match_top1 += 1
        if hit_details and hit_details[0]["issue_type_ok"]:
            issue_type_match_top1 += 1

        results.append({
            "query_id": item["query_id"],
            "query": query,
            "district": district,
            "expected_issue_type": expected_issue_type,
            "top1_ok": top1_ok,
            "top3_ok": top3_ok,
            "top5_ok": top5_ok,
            "hits": hit_details,
        })

    total = len(eval_set)
    print("=" * 60)
    print("检索评测报告")
    print("=" * 60)
    print(f"评测集大小: {total}")
    print(f"Top1 命中率: {top1_hits}/{total} = {top1_hits/total:.1%}")
    print(f"Top3 命中率: {top3_hits}/{total} = {top3_hits/total:.1%}")
    print(f"Top5 命中率: {top5_hits}/{total} = {top5_hits/total:.1%}")
    print(f"Top1 区匹配率: {district_match_top1}/{total} = {district_match_top1/total:.1%}")
    print(f"Top1 事项类型匹配率: {issue_type_match_top1}/{total} = {issue_type_match_top1/total:.1%}")
    print()

    print("-" * 60)
    print("逐条详情")
    print("-" * 60)
    for r in results:
        status = "✓" if r["top3_ok"] else "✗"
        print(f'{status} [{r["query_id"]}] {r["query"][:30]} (区={r["district"] or "不限"}, 类型={r["expected_issue_type"]})')
        if not r["top3_ok"]:
            for h in r["hits"][:3]:
                print(f'  Top{h["rank"]}: {h["title"][:40]} 区={h["district"]} 关键词={h["keyword_match"]} 区匹配={h["district_ok"]}')

    report_path = os.path.join(PROJECT_ROOT, "data", "reports", "evaluation", "retrieval_eval_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "top1_hit_rate": top1_hits / total,
            "top3_hit_rate": top3_hits / total,
            "top5_hit_rate": top5_hits / total,
            "district_match_top1_rate": district_match_top1 / total,
            "issue_type_match_top1_rate": issue_type_match_top1 / total,
            "details": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n评测报告已保存至: {report_path}")


if __name__ == "__main__":
    evaluate_retrieval()
