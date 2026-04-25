#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试BGE模块是否正常工作
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.jsjb.retrieval.bge_retriever import PolicyRetriever


def simple_test():
    print("=" * 60)
    print("BGE向量检索模块测试")
    print("=" * 60)
    
    try:
        # 测试BGE检索
        print("\n[1] 初始化BGE检索器...")
        retriever = PolicyRetriever(backend="bge")
        
        print("系统状态:")
        status = retriever.describe()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # 简单测试查询
        print("\n[2] 测试简单查询...")
        test_query = "小区垃圾没人清理"
        results = retriever.search(test_query, top_k=2)
        
        print(f"查询: {test_query}")
        print(f"找到 {len(results)} 个结果")
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n结果 {i}:")
                print(f"  标题: {result['title']}")
                print(f"  类型: {result['doc_type']}")
                print(f"  地区: {result['district']}")
                print(f"  评分: {result['score']}")
                print(f"  匹配词: {', '.join(result['matched_terms'])}")
        
        # 测试TF-IDF检索
        print("\n[3] 初始化TF-IDF检索器...")
        tfidf_retriever = PolicyRetriever(backend="tfidf")
        
        print("系统状态:")
        status = tfidf_retriever.describe()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # 简单测试查询
        print("\n[4] 测试TF-IDF查询...")
        results = tfidf_retriever.search(test_query, top_k=2)
        
        print(f"查询: {test_query}")
        print(f"找到 {len(results)} 个结果")
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n结果 {i}:")
                print(f"  标题: {result['title']}")
                print(f"  类型: {result['doc_type']}")
                print(f"  地区: {result['district']}")
                print(f"  评分: {result['score']}")
                print(f"  匹配词: {', '.join(result['matched_terms'])}")
        
        print("\n" + "=" * 60)
        print("测试完成！BGE模块集成成功！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    simple_test()