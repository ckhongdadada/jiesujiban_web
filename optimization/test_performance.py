"""
性能测试脚本
测试各环节的耗时
"""

import time
import json
from optimization.performance import get_performance_optimizer, profile

def test_performance():
    """测试性能"""
    print("=" * 50)
    print("性能测试")
    print("=" * 50)
    
    optimizer = get_performance_optimizer()
    
    test_cases = [
        {
            "tag": "投诉",
            "title": "小区垃圾没人清理",
            "body": "我们小区垃圾堆积严重，已经一周没人清理了，味道很大。",
            "unit": "朝阳区城管委"
        },
        {
            "tag": "咨询",
            "title": "鲁疃西路南延工程进展",
            "body": "请问鲁疃西路南延工程目前进展如何？预计何时通车？",
            "unit": "昌平区城管委"
        }
    ]
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {case['title']}")
        print("-" * 50)
        
        start_time = time.time()
        result = optimizer.process_request(
            tag=case["tag"],
            title=case["title"],
            body=case["body"],
            unit=case["unit"],
            enable_cache=False
        )
        total_time = (time.time() - start_time) * 1000
        
        print(f"总耗时: {total_time:.2f}ms")
        print(f"各环节耗时:")
        for stage, timing in result.get("timings", {}).items():
            if stage != "total":
                print(f"  - {stage}: {timing:.2f}ms")
        
        results.append({
            "case": i,
            "total_time": total_time,
            "timings": result.get("timings", {})
        })
    
    print("\n" + "=" * 50)
    print("性能统计")
    print("=" * 50)
    
    avg_time = sum(r["total_time"] for r in results) / len(results)
    print(f"平均总耗时: {avg_time:.2f}ms")
    
    stats = optimizer.profiler.get_stats()
    print("\n各操作统计:")
    for op, op_stats in stats.items():
        print(f"  {op}:")
        print(f"    - 调用次数: {op_stats['count']}")
        print(f"    - 平均耗时: {op_stats['avg_time_ms']:.2f}ms")
        print(f"    - 最小耗时: {op_stats['min_time_ms']:.2f}ms")
        print(f"    - 最大耗时: {op_stats['max_time_ms']:.2f}ms")

if __name__ == "__main__":
    test_performance()
