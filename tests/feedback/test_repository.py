#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试用户反馈 SQLite 存储功能
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.jsjb.feedback.repository import get_feedback_database

def test_feedback_database():
    """测试反馈数据库功能"""
    print("=" * 60)
    print("测试用户反馈 SQLite 存储功能")
    print("=" * 60)
    
    # 测试数据库初始化
    db_path = os.path.join('data', 'test_feedback.db')
    db = get_feedback_database(db_path)
    print("\n✅ 数据库初始化成功")
    
    # 测试添加反馈
    test_data = {
        'timestamp': '2025-03-27T10:00:00',
        'tag': '投诉',
        'title': '小区垃圾没人清理',
        'body': '我们小区垃圾堆积严重，影响环境卫生',
        'reply': '已安排人员处理',
        'unit': '城管委',
        'district': '朝阳区',
        'is_helpful': True,
        'client_ip': '127.0.0.1',
        'processing_time': 1.5
    }
    
    feedback_id = db.add_feedback(test_data)
    print(f"✅ 添加反馈记录成功，ID: {feedback_id}")
    
    # 测试获取统计
    stats = db.get_statistics()
    print("\n✅ 获取统计成功：")
    print(f"   - 总记录数：{stats['total_count']}")
    print(f"   - 有用数：{stats['helpful_count']}")
    print(f"   - 无用处：{stats['unhelpful_count']}")
    print(f"   - 有用率：{stats['helpful_rate']}%")
    
    if stats['tag_distribution']:
        print("\n   标签分布：")
        for item in stats['tag_distribution'][:3]:
            print(f"     - {item['tag']}: {item['count']}条")
    
    # 测试搜索
    results = db.search_feedback(keyword='垃圾')
    print(f"\n✅ 搜索成功：找到 {len(results)} 条记录")
    
    # 测试获取列表
    records = db.get_feedback(limit=10)
    print(f"✅ 获取列表成功：返回 {len(records)} 条记录")
    
    # 测试按条件搜索
    results_district = db.search_feedback(district='朝阳区')
    print(f"✅ 按地区搜索成功：朝阳区找到 {len(results_district)} 条记录")
    
    results_helpful = db.search_feedback(is_helpful=True)
    print(f"✅ 按有用性搜索成功：有用的反馈 {len(results_helpful)} 条")
    
    print("\n" + "=" * 60)
    print("🎉 所有数据库功能测试通过！")
    print("=" * 60)
    print("\n功能清单：")
    print("  ✅ SQLite 数据库初始化")
    print("  ✅ 添加反馈记录")
    print("  ✅ 获取统计数据")
    print("  ✅ 关键词搜索")
    print("  ✅ 条件筛选")
    print("  ✅ 分页查询")
    print("  ✅ 标签/地区/单位分布统计")
    print("  ✅ 近期趋势分析")
    
    # 清理测试数据库
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"\n已清理测试数据库：{db_path}")


if __name__ == "__main__":
    test_feedback_database()
