#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比BGE和TF-IDF检索效果
"""

import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from enhancements.rag_retriever_bge import PolicyRetriever as BGERetriever


def test_retrieval_effect():
    print("=" * 80)
    print("BGE vs TF-IDF 检索效果对比测试")
    print("=" * 80)
    
    # 测试查询
    test_queries = [
        ("小区垃圾没人清理", "垃圾清运"),
        ("楼下施工太吵了", "噪声扰民"),
        ("小区停车秩序混乱", "停车秩序"),
        ("家里暖气不热", "供暖问题"),
        ("路灯坏了没人修", "照明设施"),
        ("道路积水严重", "道路积水"),
        ("物业费不合理", "物业服务"),
        ("消防通道被占用", "消防通道"),
        ("占道经营严重", "占道经营"),
        ("排水设施堵塞", "排水设施"),
    ]
    
    # 测试BGE检索
    print("\n[1] 测试 BGE 向量检索")
    print("-" * 40)
    
    bge_retriever = BGERetriever(backend="bge")
    print("BGE 系统状态:", bge_retriever.describe())
    
    bge_results = {}
    bge_times = []
    
    for query, expected in test_queries:
        start_time = time.time()
        results = bge_retriever.search(query, top_k=3)
        end_time = time.time()
        
        bge_times.append(end_time - start_time)
        bge_results[query] = {
            "results": results,
            "time": end_time - start_time,
            "expected": expected
        }
        
        print(f"\n查询: {query}")
        print(f"预期主题: {expected}")
        print(f"耗时: {end_time - start_time:.3f}秒")
        print("检索结果:")
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['title']} (评分: {result['score']})")
                print(f"     匹配词: {', '.join(result['matched_terms'])}")
        else:
            print("  无相关结果")
    
    # 测试TF-IDF检索
    print("\n[2] 测试 TF-IDF 检索")
    print("-" * 40)
    
    tfidf_retriever = BGERetriever(backend="tfidf")
    print("TF-IDF 系统状态:", tfidf_retriever.describe())
    
    tfidf_results = {}
    tfidf_times = []
    
    for query, expected in test_queries:
        start_time = time.time()
        results = tfidf_retriever.search(query, top_k=3)
        end_time = time.time()
        
        tfidf_times.append(end_time - start_time)
        tfidf_results[query] = {
            "results": results,
            "time": end_time - start_time,
            "expected": expected
        }
        
        print(f"\n查询: {query}")
        print(f"预期主题: {expected}")
        print(f"耗时: {end_time - start_time:.3f}秒")
        print("检索结果:")
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['title']} (评分: {result['score']})")
                print(f"     匹配词: {', '.join(result['matched_terms'])}")
        else:
            print("  无相关结果")
    
    # 对比分析
    print("\n[3] 性能对比分析")
    print("-" * 40)
    
    avg_bge_time = sum(bge_times) / len(bge_times)
    avg_tfidf_time = sum(tfidf_times) / len(tfidf_times)
    
    print(f"平均检索时间:")
    print(f"  BGE: {avg_bge_time:.3f}秒")
    print(f"  TF-IDF: {avg_tfidf_time:.3f}秒")
    print(f"  速度比: {avg_tfidf_time/avg_bge_time:.2f}x")
    
    # 质量对比
    print("\n[4] 检索质量对比")
    print("-" * 40)
    
    for query, expected in test_queries:
        bge_hits = bge_results[query]["results"]
        tfidf_hits = tfidf_results[query]["results"]
        
        print(f"\n查询: {query}")
        print(f"预期主题: {expected}")
        
        bge_top_score = bge_hits[0]["score"] if bge_hits else 0
        tfidf_top_score = tfidf_hits[0]["score"] if tfidf_hits else 0
        
        print(f"BGE 最高分: {bge_top_score:.4f}")
        print(f"TF-IDF 最高分: {tfidf_top_score:.4f}")
        
        # 检查是否包含预期主题
        bge_has_expected = any(expected in result.get('tags', []) or 
                              expected in result.get('matched_terms', []) 
                              for result in bge_hits)
        tfidf_has_expected = any(expected in result.get('tags', []) or 
                                expected in result.get('matched_terms', []) 
                                for result in tfidf_hits)
        
        print(f"BGE 包含预期主题: {'是' if bge_has_expected else '否'}")
        print(f"TF-IDF 包含预期主题: {'是' if tfidf_has_expected else '否'}")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_retrieval_effect()
