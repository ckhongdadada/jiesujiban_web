"""
快速查看Optuna搜索状态
"""

import json
from pathlib import Path
from datetime import datetime

def check_status():
    """检查Optuna搜索状态"""
    
    optuna_dir = Path("data/reports/training/optuna")
    latest_json = optuna_dir / "optuna_best_params_latest.json"
    
    print("=" * 70)
    print(f"Optuna 搜索状态 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    if not latest_json.exists():
        print("❌ 未找到搜索结果文件")
        print("   搜索可能尚未开始或文件被删除")
        return
    
    try:
        with open(latest_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 基本信息
        print(f"\n📋 搜索配置:")
        config = data.get("search_config", {})
        print(f"   目标trials: {config.get('n_trials', 'N/A')}")
        print(f"   每次训练轮数: {config.get('search_epochs', 'N/A')}")
        print(f"   样本数量: {config.get('sample_size', 'N/A')}")
        print(f"   优化指标: {data.get('metric', 'N/A')}")
        
        # 进度统计
        stats = data.get("statistics", {})
        total = stats.get("total", 0)
        completed = stats.get("completed", 0)
        failed = stats.get("failed", 0)
        pruned = stats.get("pruned", 0)
        
        print(f"\n📊 进度统计:")
        print(f"   总数: {total}")
        print(f"   ✓ 完成: {completed}")
        print(f"   ✗ 失败: {failed}")
        print(f"   ⊘ 剪枝: {pruned}")
        
        if total > 0:
            progress = (completed / config.get('n_trials', total)) * 100
            print(f"   进度: {progress:.1f}%")
        
        # 最优结果
        if completed > 0:
            print(f"\n🏆 当前最优:")
            print(f"   得分: {data.get('best_value', 'N/A'):.6f}")
            print(f"   参数:")
            for key, value in data.get("best_params", {}).items():
                if isinstance(value, float):
                    print(f"      {key}: {value:.6e}" if value < 0.001 else f"      {key}: {value:.4f}")
                else:
                    print(f"      {key}: {value}")
        
        # 最新trial
        trials = data.get("trials", [])
        if trials:
            latest = trials[-1]
            print(f"\n🔄 最新Trial #{latest['number']}:")
            print(f"   状态: {latest['state']}")
            if latest['value'] is not None:
                print(f"   得分: {latest['value']:.6f}")
            
            if latest['state'] == 'TrialState.FAIL':
                user_attrs = latest.get('user_attrs', {})
                print(f"   ❌ 错误: {user_attrs.get('error_type', '未知')}")
                print(f"      {user_attrs.get('error_message', '未知')[:100]}")
        
        # 文件位置
        print(f"\n📁 输出文件:")
        print(f"   结果: {latest_json}")
        print(f"   CSV: {optuna_dir / 'optuna_trials_latest.csv'}")
        
        # 查找最新日志
        log_files = sorted(optuna_dir.glob("optuna_search_*.log"), 
                          key=lambda x: x.stat().st_mtime, reverse=True)
        if log_files:
            print(f"   日志: {log_files[0]}")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ 读取结果文件失败: {e}")

if __name__ == "__main__":
    check_status()
