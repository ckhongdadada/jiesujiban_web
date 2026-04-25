import sys
import os
import time
import json
from datetime import datetime

sys.path.insert(0, ".")

start_time = time.time()

log_file = open("optuna_50_trials_log.txt", "w", encoding="utf-8")

def log(msg):
    elapsed = time.time() - start_time
    full_msg = f"[{elapsed:.1f}s] {msg}"
    print(full_msg)
    log_file.write(full_msg + "\n")
    log_file.flush()

log("=" * 60)
log("Optuna 参数搜索 - 50轮次")
log("=" * 60)

try:
    log("导入模块...")
    from tools.optuna_search_classifier import parse_args, objective
    from src.jsjb.core.paths import get_training_reports_dir
    import optuna
    from pathlib import Path
    log("模块导入完成")
    
    log("配置参数...")
    args = parse_args()
    args.n_trials = 50
    args.epochs = 1
    args.sample_size = 1000
    args.study_name = "unit_classifier_50_trials"
    log(f"搜索轮次: {args.n_trials}")
    log(f"训练轮数: {args.epochs}")
    log(f"样本大小: {args.sample_size}")
    log(f"研究名称: {args.study_name}")
    
    log("创建目录...")
    reports_dir = get_training_reports_dir() / "optuna"
    reports_dir.mkdir(parents=True, exist_ok=True)
    Path(args.save_root).mkdir(parents=True, exist_ok=True)
    
    log("创建Optuna研究...")
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=None,
        load_if_exists=False,
    )
    log(f"研究创建成功: {study.study_name}")
    
    log("开始优化...")
    log("预计每个trial需要约5-10分钟（CPU训练）")
    log("总预计时间: 约4-8小时")
    log("")
    
    trial_count = [0]
    
    def logging_objective(trial):
        trial_num = trial.number
        trial_count[0] = trial_num + 1
        log(f"{'='*50}")
        log(f"Trial {trial_num + 1}/{args.n_trials} 开始...")
        trial_start = time.time()
        try:
            result = objective(args)(trial)
            trial_time = time.time() - trial_start
            log(f"Trial {trial_num + 1} 完成: F1={result:.6f}, 耗时={trial_time/60:.1f}分钟")
            
            completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
            if completed > 0:
                current_best = max(t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value)
                log(f"当前最佳F1: {current_best:.6f}")
            log("")
            return result
        except Exception as e:
            import traceback
            trial_time = time.time() - trial_start
            log(f"Trial {trial_num + 1} 失败: {e}, 耗时={trial_time/60:.1f}分钟")
            log(traceback.format_exc())
            log("")
            raise
    
    study.optimize(
        logging_objective,
        n_trials=args.n_trials,
        show_progress_bar=False
    )
    
    total_time = time.time() - start_time
    log("=" * 60)
    log("优化完成!")
    log("=" * 60)
    log(f"总耗时: {total_time/3600:.2f}小时")
    log(f"最佳F1值: {study.best_value:.6f}")
    log(f"最佳参数:")
    for key, value in study.best_params.items():
        log(f"  {key}: {value}")
    
    best = {
        "study_name": args.study_name,
        "metric": args.metric,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": dict(study.best_trial.user_attrs) if study.best_trial else {},
        "total_time_hours": total_time / 3600,
        "trials": [
            {
                "number": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                "params": trial.params,
            }
            for trial in study.trials
        ],
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path = reports_dir / f"{args.study_name}_latest.json"
    archived_path = reports_dir / f"{args.study_name}_{timestamp}.json"
    latest_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    archived_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n结果已保存到: {latest_path}")
    
except Exception as e:
    import traceback
    log(f"错误: {e}")
    log(traceback.format_exc())

finally:
    log_file.close()
