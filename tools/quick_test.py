#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速功能测试
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("🚀 快速功能测试开始")
print("=" * 50)

# 测试1: 导入模块
try:
    from enhancements.model_cache import ModelCache
    from enhancements.structured_logger import StructuredLogger
    from enhancements.input_validation import InputValidator
    print("✅ 模块导入成功")
except Exception as e:
    print(f"❌ 模块导入失败: {e}")

# 测试2: 创建实例
try:
    cache = ModelCache()
    logger = StructuredLogger()
    validator = InputValidator()
    print("✅ 实例创建成功")
except Exception as e:
    print(f"❌ 实例创建失败: {e}")

# 测试3: 测试缓存功能
try:
    cache.set("test", {"data": "test"})
    result = cache.get("test")
    if result:
        print("✅ 缓存功能正常")
    else:
        print("❌ 缓存功能异常")
except Exception as e:
    print(f"❌ 缓存测试失败: {e}")

# 测试4: 测试日志功能
try:
    logger.info("测试日志信息")
    print("✅ 日志功能正常")
except Exception as e:
    print(f"❌ 日志测试失败: {e}")

# 测试5: 测试输入验证
try:
    test_data = {"tag": "测试", "title": "标题", "body": "内容"}
    cleaned, errors = validator.validate_request_data(test_data)
    if not errors:
        print("✅ 输入验证正常")
    else:
        print(f"❌ 输入验证异常: {errors}")
except Exception as e:
    print(f"❌ 输入验证测试失败: {e}")

# 测试6: 测试逻辑修复
try:
    from enhancements.enhanced_generation import generate_simple_reply
    import inspect
    sig = inspect.signature(generate_simple_reply)
    params = list(sig.parameters.keys())
    if 'location_result' in params:
        print("✅ 逻辑漏洞已修复")
    else:
        print("❌ 逻辑漏洞未修复")
except Exception as e:
    print(f"❌ 逻辑修复测试失败: {e}")

# 测试7: 测试主应用
try:
    import app
    flask_app = app.create_app()
    print("✅ 主应用集成正常")
except Exception as e:
    print(f"❌ 主应用测试失败: {e}")

print("\n" + "=" * 50)
print("🎉 快速测试完成")
print("=" * 50)