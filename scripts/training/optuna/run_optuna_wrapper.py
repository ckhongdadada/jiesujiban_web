"""
Optuna运行包装脚本
确保输出完整且实时显示
"""

import os
import sys

# 设置环境变量确保输出不被缓冲
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 禁用输出缓冲
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

# 直接导入并运行
if __name__ == "__main__":
    # 修改sys.argv来传递参数
    sys.argv = [
        "tools/optuna_search_classifier.py",
        "--n-trials", "5",
        "--search-epochs", "2",
        "--sample-size", "3000",
        "--metric", "macro_f1"
    ]
    
    # 导入并运行主函数
    import tools.optuna_search_classifier as optuna_script
    optuna_script.main()
