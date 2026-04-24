"""
优化启动脚本
预加载模型和缓存，提升首次请求速度
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

def preload_all_models():
    """预加载所有模型"""
    print("=" * 50)
    print("开始预加载模型...")
    print("=" * 50)
    
    # 1. 预加载分类器
    print("\n[1/4] 预加载分类器模型...")
    try:
        from enhancements.classifier_runtime import ClassifierRuntime
        classifier = ClassifierRuntime()
        classifier.predict("测试", "测试标题", "测试内容")
        print("  ✓ 分类器模型预加载成功")
    except Exception as e:
        print(f"  ✗ 分类器模型预加载失败: {e}")
    
    # 2. 预加载生成器
    print("\n[2/4] 预加载生成器模型...")
    try:
        from enhancements.enhanced_generation import load_generator
        load_generator()
        print("  ✓ 生成器模型预加载成功")
    except Exception as e:
        print(f"  ✗ 生成器模型预加载失败: {e}")
    
    # 3. 预加载地名识别
    print("\n[3/4] 预加载地名识别库...")
    try:
        from enhancements.location_ner import LocationNER
        ner = LocationNER()
        ner.extract("测试地址")
        print("  ✓ 地名识别库预加载成功")
    except Exception as e:
        print(f"  ✗ 地名识别库预加载失败: {e}")
    
    # 4. 预加载RAG检索
    print("\n[4/4] 预加载RAG检索索引...")
    try:
        from enhancements.rag_retriever_bge import RAGRetriever
        rag = RAGRetriever()
        rag.search("测试查询", top_k=1)
        print("  ✓ RAG检索索引预加载成功")
    except Exception as e:
        print(f"  ✗ RAG检索索引预加载失败: {e}")
    
    print("\n" + "=" * 50)
    print("模型预加载完成！")
    print("=" * 50)

if __name__ == "__main__":
    preload_all_models()
