"""
Optuna运行监控脚本
实时显示训练进度和结果
"""

import time
import json
from pathlib import Path
from datetime import datetime

def monitor_optuna_progress():
    """监控Optuna搜索进度"""
    
    optuna_dir = Path("data/reports/training/optuna")
    latest_json = optuna_dir / "optuna_best_params_latest.json"
    latest_csv = optuna_dir / "optuna_trials_latest.csv"
    
    print("=" * 70)
    print("Optuna 搜索监控")
    print("=" * 70)
    print(f"监控目录: {optuna_dir}")
    print(f"按 Ctrl+C 退出监控\n")
    
    last_trial_count = 0
    last_update_time = None
    
    try:
        while True:
            if latest_json.exists():
                # 读取最新结果
                with open(latest_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                stats = data.get("statistics", {})
                total = stats.get("total", 0)
                completed = stats.get("completed", 0)
                failed = stats.get("failed", 0)
                pruned = stats.get("pruned", 0)
                
                # 检查是否有新的trial
                if total != last_trial_count:
                    last_trial_count = total
                    last_update_time = datetime.now()
                    
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 进度更新:")
                    print(f"  总数: {total} | 完成: {completed} | 失败: {failed} | 剪枝: {pruned}")
                    
                    if completed > 0:
                        best_value = data.get("best_value")
                        best_params = data.get("best_params", {})
                        print(f"  当前最优: {best_value:.6f}")
                        print(f"  最优参数: {json.dumps(best_params, ensure_ascii=False)}")
                    
                    # 显示最近的trial
                    trials = data.get("trials", [])
                    if trials:
                        latest_trial = trials[-1]
                        print(f"\n  最新Trial #{latest_trial['number']}:")
                        print(f"    状态: {latest_trial['state']}")
                        if latest_trial['value'] is not None:
                            print(f"    得分: {latest_trial['value']:.6f}")
                        print(f"    参数: {json.dumps(latest_trial['params'], ensure_ascii=False)}")
                        
                        # 如果失败，显示错误
                        if latest_trial['state'] == 'TrialState.FAIL':
                            user_attrs = latest_trial.get('user_attrs', {})
                            error_type = user_attrs.get('error_type', '未知')
                            error_msg = user_attrs.get('error_message', '未知')
                            print(f"    ✗ 错误: {error_type}: {error_msg}")
            
            else:
                if last_update_time is None:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 等待搜索开始...")
            
            # 每5秒检查一次
            time.sleep(5)
    
    except KeyboardInterrupt:
        print("\n\n监控已停止")
        
        # 显示最终结果
        if latest_json.exists():
            with open(latest_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print("\n" + "=" * 70)
            print("最终结果")
            print("=" * 70)
            
            stats = data.get("statistics", {})
            print(f"总数: {stats.get('total', 0)}")
            print(f"完成: {stats.get('completed', 0)}")
            print(f"失败: {stats.get('failed', 0)}")
            print(f"剪枝: {stats.get('pruned', 0)}")
            
            if data.get("best_value"):
                print(f"\n最优得分: {data['best_value']:.6f}")
                print(f"最优参数:")
                for key, value in data.get("best_params", {}).items():
                    print(f"  {key}: {value}")
            
            print(f"\n详细结果: {latest_json}")
            print(f"CSV文件: {latest_csv}")
            print("=" * 70)


def show_latest_logs():
    """显示最新的日志文件"""
    optuna_dir = Path("data/reports/training/optuna")
    
    # 查找最新的日志文件
    log_files = sorted(optuna_dir.glob("optuna_search_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if log_files:
        latest_log = log_files[0]
        print(f"\n最新日志文件: {latest_log}")
        print("=" * 70)
        
        # 显示最后50行
        with open(latest_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[-50:]:
                print(line, end='')
    else:
        print("未找到日志文件")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--logs":
        show_latest_logs()
    else:
        monitor_optuna_progress()
