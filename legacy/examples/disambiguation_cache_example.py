"""
地名消歧和Redis缓存使用示例

本示例展示如何使用新实现的地名消歧功能和Redis缓存系统
"""

import os
from enhancements.location_disambiguation import (
    LocationDisambiguator,
    LocationCandidate,
    disambiguate_location
)
from enhancements.cache_manager import (
    MultiLevelCache,
    init_cache,
    get_global_cache,
    cache_result
)
from enhancements.model_manager_advanced import (
    ModelManagerAdvanced,
    init_model_manager,
    get_model_manager,
    monitor_performance
)


def example_location_disambiguation():
    """地名消歧示例"""
    print("\n" + "="*60)
    print("地名消歧示例")
    print("="*60)
    
    disambiguator = LocationDisambiguator(
        amap_api_key=os.getenv("AMAP_API_KEY"),
        enable_context_disambiguation=True,
        enable_geo_validation=True
    )
    
    test_cases = [
        {
            "text": "朝阳区望京街道发生垃圾堆积问题，居民反映强烈",
            "location": "望京",
            "candidates": [
                LocationCandidate(
                    matched_text="望京",
                    district="朝阳区",
                    source="alias_match",
                    confidence=0.85
                ),
                LocationCandidate(
                    matched_text="望京",
                    district="海淀区",
                    source="lac_alias",
                    confidence=0.75
                )
            ]
        },
        {
            "text": "中关村软件园附近的交通拥堵问题严重",
            "location": "中关村",
            "candidates": [
                LocationCandidate(
                    matched_text="中关村",
                    district="海淀区",
                    source="direct_match",
                    confidence=0.95
                ),
                LocationCandidate(
                    matched_text="中关村",
                    district="朝阳区",
                    source="alias_match",
                    confidence=0.70
                )
            ]
        },
        {
            "text": "三里屯酒吧街噪声扰民问题",
            "location": "三里屯",
            "candidates": [
                LocationCandidate(
                    matched_text="三里屯",
                    district="朝阳区",
                    source="alias_match",
                    confidence=0.90
                )
            ]
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n案例 {i}:")
        print(f"  文本: {case['text']}")
        print(f"  地名: {case['location']}")
        print(f"  候选数: {len(case['candidates'])}")
        
        result = disambiguator.disambiguate(
            text=case['text'],
            location=case['location'],
            candidates=case['candidates']
        )
        
        if result:
            print(f"  结果:")
            print(f"    - 行政区: {result.district}")
            print(f"    - 置信度: {result.confidence:.3f}")
            print(f"    - 消歧方法: {result.disambiguation_method}")
        else:
            print("  结果: 无法消歧")
    
    stats = disambiguator.get_stats()
    print(f"\n消歧统计:")
    print(f"  - 总消歧次数: {stats['total_disambiguations']}")
    print(f"  - 上下文消歧: {stats['context_disambiguations']}")
    print(f"  - 地理编码验证: {stats['geo_disambiguations']}")
    print(f"  - 多源融合: {stats['multi_source_disambiguations']}")


def example_redis_cache():
    """Redis缓存示例"""
    print("\n" + "="*60)
    print("Redis缓存示例")
    print("="*60)
    
    cache = MultiLevelCache(
        local_cache_size=1000,
        enable_redis=True,
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        redis_db=int(os.getenv("REDIS_DB", "0")),
        redis_password=os.getenv("REDIS_PASSWORD"),
        default_ttl=3600,
        prefix="example:"
    )
    
    print("\n1. 基本操作")
    
    cache.set("user:1001", {"name": "张三", "age": 30}, ttl=600)
    print("  设置缓存: user:1001")
    
    user = cache.get("user:1001")
    print(f"  获取缓存: {user}")
    
    print("\n2. 缓存装饰器")
    
    @cache_result(key_prefix="expensive_func", ttl=1800, cache_instance=cache)
    def expensive_computation(n: int) -> int:
        import time
        time.sleep(0.1)
        return n * n
    
    print("  第一次调用（计算）:")
    result1 = expensive_computation(10)
    print(f"    结果: {result1}")
    
    print("  第二次调用（缓存）:")
    result2 = expensive_computation(10)
    print(f"    结果: {result2}")
    
    print("\n3. get_or_set 方法")
    
    def fetch_from_db(key: str) -> dict:
        print(f"    从数据库获取: {key}")
        return {"data": f"value_for_{key}"}
    
    print("  第一次获取（从数据库）:")
    data1 = cache.get_or_set("db:key1", lambda: fetch_from_db("key1"), ttl=300)
    print(f"    结果: {data1}")
    
    print("  第二次获取（从缓存）:")
    data2 = cache.get_or_set("db:key1", lambda: fetch_from_db("key1"), ttl=300)
    print(f"    结果: {data2}")
    
    print("\n4. 缓存统计")
    stats = cache.get_stats()
    print(f"  总请求数: {stats['total_requests']}")
    print(f"  L1命中: {stats['l1_hits']}")
    print(f"  L2命中: {stats['l2_hits']}")
    print(f"  未命中: {stats['misses']}")
    print(f"  总命中率: {stats['total_hit_rate']:.2%}")
    
    if 'redis_cache' in stats:
        redis_stats = stats['redis_cache']
        print(f"\n  Redis状态:")
        print(f"    连接状态: {'已连接' if redis_stats['connected'] else '未连接'}")
        print(f"    命中: {redis_stats['hits']}")
        print(f"    未命中: {redis_stats['misses']}")


def example_model_manager():
    """高级模型管理器示例"""
    print("\n" + "="*60)
    print("高级模型管理器示例")
    print("="*60)
    
    manager = init_model_manager(
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        enable_redis=True
    )
    
    print("\n1. 加载模型")
    manager.load_location_ner(enable_lac=True, enable_disambiguation=True)
    manager.load_rag_retriever(enable_bm25=True, enable_chunking=True)
    
    print("\n2. 地名识别（带消歧）")
    text = "朝阳区望京街道某小区垃圾堆积问题"
    location_result = manager.resolve_location(text)
    print(f"  文本: {text}")
    print(f"  识别结果: {location_result['district']}")
    print(f"  置信度: {location_result['confidence']:.3f}")
    
    print("\n3. RAG检索（带缓存）")
    query = "垃圾清运不及时"
    results = manager.search_rag(query, top_k=3)
    print(f"  查询: {query}")
    print(f"  结果数: {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"    {i}. {result['title']} (分数: {result['score']:.3f})")
    
    print("\n4. 缓存统计")
    stats = manager.cache.get_stats()
    print(f"  总命中率: {stats['total_hit_rate']:.2%}")
    print(f"  L1命中率: {stats['l1_hit_rate']:.2%}")
    print(f"  L2命中率: {stats['l2_hit_rate']:.2%}")
    
    print("\n5. 缓存预热")
    hot_queries = [
        "垃圾清运",
        "噪声扰民",
        "停车秩序",
        "道路积水",
        "物业服务"
    ]
    manager.warmup_cache(hot_queries)
    
    print("\n6. 模型状态")
    status = manager.get_status()
    for model_name, model_status in status.items():
        if model_name == 'cache_stats':
            continue
        if isinstance(model_status, dict):
            print(f"  {model_name}: loaded={model_status.get('loaded', False)}")


def example_performance_monitoring():
    """性能监控示例"""
    print("\n" + "="*60)
    print("性能监控示例")
    print("="*60)
    
    @monitor_performance
    def slow_function():
        import time
        time.sleep(0.1)
        return "完成"
    
    @monitor_performance
    def fast_function():
        return "完成"
    
    print("\n执行慢函数:")
    result1 = slow_function()
    
    print("\n执行快函数:")
    result2 = fast_function()


def example_batch_processing():
    """批量处理示例"""
    print("\n" + "="*60)
    print("批量处理示例")
    print("="*60)
    
    manager = get_model_manager()
    
    messages = [
        {
            "tag": "市容环卫",
            "title": "垃圾堆积问题",
            "body": "朝阳区望京街道某小区垃圾堆积严重"
        },
        {
            "tag": "噪声扰民",
            "title": "施工噪声扰民",
            "body": "海淀区中关村街道某工地夜间施工"
        },
        {
            "tag": "停车秩序",
            "title": "违停问题",
            "body": "丰台区方庄街道某路段车辆违停"
        }
    ]
    
    print(f"\n批量处理 {len(messages)} 条留言:")
    
    for i, msg in enumerate(messages, 1):
        print(f"\n{i}. {msg['title']}")
        
        location_result = manager.resolve_location(msg['body'])
        print(f"   行政区: {location_result['district']}")
        
        retrieval_hits = manager.search_rag(
            f"{msg['title']} {msg['body']}",
            top_k=2,
            district=location_result['district']
        )
        print(f"   检索结果: {len(retrieval_hits)} 条")
    
    stats = manager.cache.get_stats()
    print(f"\n缓存统计:")
    print(f"  总命中率: {stats['total_hit_rate']:.2%}")


def main():
    """主函数"""
    print("="*60)
    print("地名消歧和Redis缓存使用示例")
    print("="*60)
    
    try:
        example_location_disambiguation()
    except Exception as e:
        print(f"\n地名消歧示例失败: {e}")
    
    try:
        example_redis_cache()
    except Exception as e:
        print(f"\nRedis缓存示例失败: {e}")
    
    try:
        example_model_manager()
    except Exception as e:
        print(f"\n模型管理器示例失败: {e}")
    
    try:
        example_performance_monitoring()
    except Exception as e:
        print(f"\n性能监控示例失败: {e}")
    
    try:
        example_batch_processing()
    except Exception as e:
        print(f"\n批量处理示例失败: {e}")
    
    print("\n" + "="*60)
    print("示例运行完成")
    print("="*60)


if __name__ == "__main__":
    main()
