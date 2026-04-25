
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试BGE向量检索功能
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from enhancements.rag_retriever import PolicyRetriever


def test_retrieval():
    print("=" * 60)
    print("RAG检索功能测试")
    print("=" * 60)
    
    retriever = PolicyRetriever()
    
    print("\n[系统状态]")
    print(retriever.describe())
    
    test_queries = [
        "小区垃圾没人清理",
        "楼下施工太吵了",
        "小区停车秩序混乱",
        "家里暖气不热",
        "路灯坏了没人修",
    ]
    
    print("\n" + "=" * 60)
    print("测试查询")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print("\n[{0}] 查询: {1}".format(i, query))
        results = retriever.search(query, top_k=3)
        
        if not results:
            print("    未找到相关结果")
            continue
        
        for j, result in enumerate(results, 1):
            print("    [{0}] {1}".format(j, result['title']))
            print("        类型: {0}".format(result['doc_type']))
            print("        地区: {0}".format(result['district']))
            print("        评分: {0}".format(result['score']))
            print("        匹配词: {0}".format(', '.join(result['matched_terms'])))
            print("        摘要: {0}...".format(result['snippet'][:80]))
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_retrieval()

