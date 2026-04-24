import sys
import os
import time

sys.path.insert(0, ".")

start_time = time.time()

log_file = open("optuna_quick_log.txt", "w", encoding="utf-8")

def log(msg):
    elapsed = time.time() - start_time
    full_msg = f"[{elapsed:.1f}s] {msg}"
    print(full_msg)
    log_file.write(full_msg + "\n")
    log_file.flush()

log("开始快速Optuna参数搜索测试...")

try:
    log("导入模块...")
    from tools.optuna_search_classifier import parse_args, objective
    from enhancements.data_paths import get_training_reports_dir
    import optuna
    from pathlib import Path
    from datetime import datetime
    import json
    log("模块导入完成")
    
    log("解析参数...")
    args = parse_args()
    args.n_trials = 3
    args.epochs = 1
    args.sample_size = 500
    args.study_name = "quick_test"
    log(f"参数: trials={args.n_trials}, epochs={args.epochs}, sample={args.sample_size}")
    
    log("创建目录...")
    reports_dir = get_training_reports_dir() / "optuna"
    reports_dir.mkdir(parents=True, exist_ok=True)
    Path(args.save_root).mkdir(parents=True, exist_ok=True)
    
    log("创建Optuna研究...")
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0)
    
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
    
    def logging_objective(trial):
        log(f"Trial {trial.number} 开始...")
        try:
            result = objective(args)(trial)
            log(f"Trial {trial.number} 完成: {result:.6f}")
            return result
        except Exception as e:
            import traceback
            log(f"Trial {trial.number} 失败: {e}")
            log(traceback.format_exc())
            raise
    
    study.optimize(
        logging_objective,
        n_trials=args.n_trials,
        show_progress_bar=False
    )
    
    log("\n优化完成!")
    log(f"最佳值: {study.best_value:.6f}")
    log(f"最佳参数: {json.dumps(study.best_params, indent=2)}")
    
    best = {
        "study_name": args.study_name,
        "metric": args.metric,
        "best_value": study.best_value,
        "best_params": study.best_params,
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
    latest_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"结果已保存到: {latest_path}")
    
except Exception as e:
    import traceback
    log(f"错误: {e}")
    log(traceback.format_exc())

finally:
    log_file.close()
