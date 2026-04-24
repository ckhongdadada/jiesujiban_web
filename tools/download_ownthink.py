"""
OwnThink知识图谱下载脚本
========================
下载完整的OwnThink中文知识图谱数据（1.4亿三元组）

数据来源: https://github.com/ownthink/KnowledgeGraphData

下载方式:
  方式1: 百度网盘
    链接: https://pan.baidu.com/s/1LZjs9Dsta0yD9NH-1y0sAw
    提取码: 3hpp
    解压密码: https://www.ownthink.com/

  方式2: Kaggle
    https://www.kaggle.com/datasets/ownthink/knowledge-graph

使用方法:
  1. 手动从上述链接下载数据
  2. 解压后将 ownthink_v2.csv 放到 data/raw/ownthink/ 目录
  3. 运行: python tools/download_ownthink.py --verify

数据规模:
  - 文件: ownthink_v2.csv
  - 行数: 140,919,781 (约1.4亿条)
  - 格式: CSV (实体, 属性, 值)
  - 大小: 约 4-5 GB (压缩后约 1-2 GB)
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path


def verify_ownthink_data(filepath: str = None) -> dict:
    if filepath is None:
        project_root = Path(__file__).parent.parent
        filepath = project_root / "data" / "raw" / "ownthink" / "ownthink_v2.csv"
    
    if not os.path.exists(filepath):
        return {
            "status": "not_found",
            "message": f"数据文件不存在: {filepath}",
            "instructions": [
                "请从以下链接下载 OwnThink 知识图谱数据:",
                "",
                "方式1 - 百度网盘:",
                "  链接: https://pan.baidu.com/s/1LZjs9Dsta0yD9NH-1y0sAw",
                "  提取码: 3hpp",
                "  解压密码: https://www.ownthink.com/",
                "",
                "方式2 - Kaggle:",
                "  https://www.kaggle.com/datasets/ownthink/knowledge-graph",
                "",
                f"下载后将文件放到: {filepath}",
            ]
        }
    
    print(f"正在验证数据文件: {filepath}")
    
    line_count = 0
    sample_data = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            line_count += 1
            if i < 10:
                sample_data.append(row)
            if line_count % 10000000 == 0:
                print(f"  已读取: {line_count:,} 行")
    
    return {
        "status": "ok",
        "message": "数据文件验证成功",
        "filepath": str(filepath),
        "total_lines": line_count,
        "expected_lines": 140919781,
        "samples": sample_data[:5],
    }


def preview_data(filepath: str = None, num_lines: int = 20):
    if filepath is None:
        project_root = Path(__file__).parent.parent
        filepath = project_root / "data" / "raw" / "ownthink" / "ownthink_v2.csv"
    
    if not os.path.exists(filepath):
        print(f"数据文件不存在: {filepath}")
        return
    
    print(f"\n数据预览 ({num_lines} 行):")
    print("-" * 80)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= num_lines:
                break
            if len(row) >= 3:
                print(f"  实体: {row[0]}")
                print(f"  属性: {row[1]}")
                print(f"  值: {row[2][:100]}{'...' if len(row[2]) > 100 else ''}")
                print("-" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OwnThink知识图谱下载工具")
    parser.add_argument("--verify", action="store_true", help="验证数据文件")
    parser.add_argument("--preview", type=int, default=0, help="预览前N行数据")
    args = parser.parse_args()
    
    print("=" * 60)
    print("OwnThink 知识图谱下载工具")
    print("=" * 60)
    
    if args.verify:
        result = verify_ownthink_data()
        print(f"\n状态: {result['status']}")
        print(f"消息: {result['message']}")
        
        if result['status'] == 'not_found':
            print("\n下载说明:")
            for line in result['instructions']:
                print(f"  {line}")
        else:
            print(f"\n文件路径: {result['filepath']}")
            print(f"总行数: {result['total_lines']:,}")
            print(f"预期行数: {result['expected_lines']:,}")
            
            if result['samples']:
                print("\n样本数据:")
                for i, sample in enumerate(result['samples'], 1):
                    print(f"  {i}. {sample}")
    
    if args.preview > 0:
        preview_data(num_lines=args.preview)


if __name__ == "__main__":
    main()
