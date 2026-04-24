"""
Optuna参数搜索运行脚本
解决输出截断问题
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

def run_optuna_search(
    n_trials=20,
    search_epochs=3,
    sample_size=5000,
    metric="macro_f1"
):
    """
    运行Optuna参数搜索
    
    Args:
        n_trials: 搜索次数
        search_epochs: 每次trial的训练轮数
        sample_size: 样本数量
        metric: 优化指标
    """
    
    # 创建输出目录
    output_dir = Path("data/reports/training/optuna")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"unit_classifier_optuna_{timestamp}.log"
    err_file = output_dir / f"unit_classifier_optuna_{timestamp}.err.log"
    
    # 构建命令
    cmd = [
        sys.executable,  # 使用当前Python解释器
        "tools/optuna_search_classifier.py",
        "--n-trials", str(n_trials),
        "--search-epochs", str(search_epochs),
        "--sample-size", str(sample_size),
        "--metric", metric,
        "--output-dir", str(output_dir)
    ]
    
    print("=" * 70)
    print("Optuna 参数搜索")
    print("=" * 70)
    print(f"搜索次数: {n_trials}")
    print(f"每次训练轮数: {search_epochs}")
    print(f"样本数量: {sample_size}")
    print(f"优化指标: {metric}")
    print(f"输出目录: {output_dir}")
    print(f"日志文件: {log_file}")
    print(f"错误日志: {err_file}")
    print("=" * 70)
    print("\n开始搜索...\n")
    
    # 运行命令，实时输出
    try:
        with open(log_file, 'w', encoding='utf-8') as log_f, \
             open(err_file, 'w', encoding='utf-8') as err_f:
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                bufsize=1,  # 行缓冲
                universal_newlines=True
            )
            
            # 实时读取并输出
            while True:
                # 读取标准输出
                output = process.stdout.readline()
                if output:
                    print(output, end='', flush=True)
                    log_f.write(output)
                    log_f.flush()
                
                # 读取错误输出
                error = process.stderr.readline()
                if error:
                    print(error, end='', file=sys.stderr, flush=True)
                    err_f.write(error)
                    err_f.flush()
                
                # 检查进程是否结束
                if output == '' and error == '' and process.poll() is not None:
                    break
            
            # 获取返回码
            return_code = process.wait()
            
            print("\n" + "=" * 70)
            if return_code == 0:
                print("✓ 搜索完成!")
                print(f"✓ 日志已保存: {log_file}")
                print(f"✓ 结果文件: {output_dir / 'optuna_best_params_latest.json'}")
                print(f"✓ CSV文件: {output_dir / 'optuna_trials_latest.csv'}")
            else:
                print(f"✗ 搜索失败，返回码: {return_code}")
                print(f"✗ 请查看错误日志: {err_file}")
            print("=" * 70)
            
            return return_code
            
    except KeyboardInterrupt:
        print("\n\n用户中断 (Ctrl+C)")
        process.terminate()
        return 1
    
    except Exception as e:
        print(f"\n\n运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="运行Optuna参数搜索")
    parser.add_argument("--n-trials", type=int, default=20, help="搜索次数")
    parser.add_argument("--search-epochs", type=int, default=3, help="每次训练轮数")
    parser.add_argument("--sample-size", type=int, default=5000, help="样本数量")
    parser.add_argument("--metric", default="macro_f1", 
                       choices=["macro_f1", "weighted_f1", "top1_acc", "topk_acc"],
                       help="优化指标")
    
    args = parser.parse_args()
    
    return_code = run_optuna_search(
        n_trials=args.n_trials,
        search_epochs=args.search_epochs,
        sample_size=args.sample_size,
        metric=args.metric
    )
    
    sys.exit(return_code)


if __name__ == "__main__":
    main()
