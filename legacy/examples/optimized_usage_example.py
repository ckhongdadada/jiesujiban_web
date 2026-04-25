"""
优化后的模型使用示例

本示例展示如何使用优化后的RAG检索、生成模型和地名识别模型
"""

from enhancements.model_manager import ModelManager, monitor_performance, cache_result
from enhancements.rag_retriever_bge_optimized import PolicyRetriever
from enhancements.enhanced_generation_optimized import (
    generate_reply_with_context,
    batch_generate,
    generate_stream,
    get_generation_stats
)
from enhancements.location_ner_optimized import BeijingDistrictResolverOptimized


@monitor_performance
def process_message_optimized(
    tag: str,
    title: str,
    body: str,
    manager: ModelManager
) -> dict:
    """处理单条留言（优化版）"""
    
    location_result = manager.location_ner.resolve(body)
    district = location_result.get("district", "未识别")
    
    retrieval_hits = manager.rag_retriever.search(
        query=f"{title} {body}",
        top_k=3,
        district=district,
        tag=tag
    )
    
    from enhancements.classifier_runtime import predict_units
    import torch
    
    unit_predictions = predict_units(
        tag=tag,
        title=title,
        body=body,
        model_dir=None,
        base_model_dir=None,
        label_map_path=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        top_k=3,
        district=district
    )
    
    if unit_predictions:
        unit = unit_predictions[0]["unit"]
    else:
        unit = "相关部门"
    
    reply_result = generate_reply_with_context(
        tag=tag,
        title=title,
        body=body,
        unit=unit,
        location_result=location_result,
        retrieval_hits=retrieval_hits,
        enable_verification=True
    )
    
    return {
        "unit_predictions": unit_predictions,
        "location_result": location_result,
        "retrieval_hits": retrieval_hits,
        "reply": reply_result["reply"],
        "verification": reply_result.get("verification"),
        "fallback": reply_result.get("fallback", False)
    }


def batch_process_messages(messages: list[dict], batch_size: int = 8) -> list[dict]:
    """批量处理留言"""
    
    manager = ModelManager()
    
    manager.load_location_ner(enable_lac=True)
    manager.load_rag_retriever(backend="bge")
    
    texts = [f"{msg['title']} {msg['body']}" for msg in messages]
    location_results = manager.location_ner.batch_resolve(texts)
    
    districts = [loc.get("district", "未识别") for loc in location_results]
    tags = [msg.get("tag", "") for msg in messages]
    
    retrieval_results = manager.rag_retriever.batch_search_optimized(
        queries=texts,
        top_k=3,
        districts=districts,
        tags=tags
    )
    
    generation_requests = []
    for i, msg in enumerate(messages):
        generation_requests.append({
            "tag": msg.get("tag", ""),
            "title": msg.get("title", ""),
            "body": msg.get("body", ""),
            "unit": msg.get("unit", "相关部门"),
            "location_result": location_results[i],
            "retrieval_hits": retrieval_results[i]
        })
    
    reply_results = batch_generate(
        requests=generation_requests,
        batch_size=batch_size
    )
    
    results = []
    for i, msg in enumerate(messages):
        results.append({
            "message_id": msg.get("id", i),
            "unit_predictions": msg.get("unit_predictions", []),
            "location_result": location_results[i],
            "retrieval_hits": retrieval_results[i],
            "reply": reply_results[i]["reply"],
            "fallback": reply_results[i].get("fallback", False)
        })
    
    return results


def stream_generate_example(tag: str, title: str, body: str, unit: str):
    """流式生成示例"""
    
    print(f"留言: {title}")
    print(f"回复: ", end="", flush=True)
    
    for chunk in generate_stream(
        tag=tag,
        title=title,
        body=body,
        unit=unit,
        location_result={"district": "朝阳区"},
        retrieval_hits=[]
    ):
        print(chunk, end="", flush=True)
    
    print()


def print_cache_stats(manager: ModelManager):
    """打印缓存统计"""
    
    print("\n=== 缓存统计 ===")
    
    print("\n统一缓存:")
    cache_stats = manager.unified_cache.get_stats()
    for key, value in cache_stats.items():
        print(f"  {key}: {value}")
    
    if manager.rag_retriever:
        print("\nRAG检索缓存:")
        rag_stats = manager.rag_retriever.get_cache_stats()
        for key, value in rag_stats.items():
            print(f"  {key}: {value}")
    
    if manager.location_ner:
        print("\n地名识别缓存:")
        ner_stats = manager.location_ner.get_stats()
        if "cache_stats" in ner_stats:
            for key, value in ner_stats["cache_stats"].items():
                print(f"  {key}: {value}")
    
    print("\n生成模型缓存:")
    gen_stats = get_generation_stats()
    print(f"  KV Cache统计:")
    for key, value in gen_stats.get("kv_cache_stats", {}).items():
        print(f"    {key}: {value}")


def main():
    """主函数"""
    
    print("=== 初始化模型管理器 ===")
    manager = ModelManager()
    
    print("\n=== 加载模型 ===")
    
    manager.load_location_ner(enable_lac=True)
    manager.load_rag_retriever(backend="bge")
    manager.load_generator()
    
    print("\n=== 模型状态 ===")
    status = manager.get_status()
    for model_name, model_status in status.items():
        if model_name == "cache_stats":
            continue
        print(f"{model_name}: loaded={model_status['loaded']}, status={model_status['status']}")
    
    print("\n=== 处理示例留言 ===")
    
    sample_message = {
        "tag": "市容环卫",
        "title": "朝阳区某小区垃圾清运不及时",
        "body": "我是朝阳区望京街道某小区的居民，近期发现小区内垃圾桶清运不及时，垃圾堆积严重，影响居民生活。希望相关部门尽快处理。"
    }
    
    result = process_message_optimized(
        tag=sample_message["tag"],
        title=sample_message["title"],
        body=sample_message["body"],
        manager=manager
    )
    
    print(f"\n识别行政区: {result['location_result']['district']}")
    print(f"预测承办单位: {result['unit_predictions'][0]['unit'] if result['unit_predictions'] else '未识别'}")
    print(f"检索结果数: {len(result['retrieval_hits'])}")
    print(f"\n生成的回复:\n{result['reply']}")
    
    print_cache_stats(manager)
    
    print("\n=== 批量处理示例 ===")
    
    batch_messages = [
        {
            "id": "msg_001",
            "tag": "市容环卫",
            "title": "垃圾堆积问题",
            "body": "海淀区中关村街道某小区垃圾堆积严重"
        },
        {
            "id": "msg_002",
            "tag": "噪声扰民",
            "title": "施工噪声扰民",
            "body": "朝阳区望京街道某工地夜间施工噪声扰民"
        }
    ]
    
    batch_results = batch_process_messages(batch_messages, batch_size=2)
    
    for res in batch_results:
        print(f"\n留言ID: {res['message_id']}")
        print(f"行政区: {res['location_result']['district']}")
        print(f"回复: {res['reply'][:100]}...")
    
    print("\n=== 清理缓存 ===")
    manager.clear_all_caches()
    print("缓存已清空")


if __name__ == "__main__":
    main()
