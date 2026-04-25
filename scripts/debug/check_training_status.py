#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练进度监控脚本
用法: python check_training.py
"""

import os
import json
from pathlib import Path

def check_training_progress():
    """检查训练进度"""
    
    # 检查训练状态文件
    state_file = Path("final_model_fgm/training_state.pt")
    if state_file.exists():
        print("✅ 找到训练状态文件")
        print(f"   文件大小: {state_file.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"   最后修改: {state_file.stat().st_mtime}")
    else:
        print("⏳ 训练状态文件尚未生成（第一轮训练中）")
    
    # 检查模型文件
    model_file = Path("final_model_fgm/pytorch_model.bin")
    if model_file.exists():
        print("\n✅ 找到模型文件")
        print(f"   文件大小: {model_file.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print("\n⏳ 模型文件尚未生成")
    
    # 检查标签映射
    label_file = Path("final_model_fgm/label_map.json")
    if label_file.exists():
        print("\n✅ 找到标签映射文件")
        with open(label_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        print(f"   类别数量: {len(labels)}")
        print(f"   前5个类别: {list(labels.values())[:5]}")
    else:
        print("\n❌ 标签映射文件未找到")
    
    # 检查日志文件
    log_file = Path("logs/complaint_system.log")
    if log_file.exists():
        print(f"\n📋 日志文件: {log_file}")
        print(f"   文件大小: {log_file.stat().st_size / 1024:.2f} KB")
    
    print("\n" + "="*60)
    print("提示：训练需要较长时间，请耐心等待")
    print("可以使用以下命令查看实时输出：")
    print("  在 Kiro 中使用 getProcessOutput 工具")
    print("="*60)

if __name__ == "__main__":
    check_training_progress()
