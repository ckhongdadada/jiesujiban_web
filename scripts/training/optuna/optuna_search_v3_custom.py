import sys
import os

sys.path.insert(0, ".")

from tools.optuna_search_classifier import parse_args, objective, build_training_args
from enhancements.data_paths import get_training_reports_dir
import optuna
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

print("=" * 60)
print("Optuna 参数搜索")
print("=" * 60)

args = parse_args()
args.n_trials = 50
args.epochs = 2
args.sample_size = 2000
args.study_name = "unit_classifier_v3"

print(f"搜索轮次: {args.n_trials}")
print(f"训练轮数: {args.epochs}")
print(f"样本大小: {args.sample_size}")
print(f"研究名称: {args.study_name}")

reports_dir = get_training_reports_dir() / "optuna"
reports_dir.mkdir(parents=True, exist_ok=True)
Path(args.save_root).mkdir(parents=True, exist_ok=True)

print(f"报告目录: {reports_dir}")

sampler = optuna.samplers.TPESampler(seed=args.seed)
pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)

study = optuna.create_study(
    study_name=args.study_name,
    direction="maximize",
    sampler=sampler,
    pruner=pruner,
    storage=None,
    load_if_exists=False,
)

print(f"研究创建成功: {study.study_name}")
print("开始优化...")

study.optimize(
    objective(args),
    n_trials=args.n_trials,
    timeout=args.timeout if args.timeout > 0 else None,
    show_progress_bar=True
)

print("\n" + "=" * 60)
print("优化完成!")
print("=" * 60)
print(f"最佳值: {study.best_value:.6f}")
print(f"最佳参数: {json.dumps(study.best_params, indent=2)}")

best = {
    "study_name": args.study_name,
    "metric": args.metric,
    "best_value": study.best_value,
    "best_params": study.best_params,
    "best_user_attrs": study.best_trial.user_attrs,
    "trials": [
        {
            "number": trial.number,
            "state": str(trial.state),
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        for trial in study.trials
    ],
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
latest_path = reports_dir / f"{args.study_name}_latest.json"
archived_path = reports_dir / f"{args.study_name}_{timestamp}.json"
latest_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
archived_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"结果已保存到: {latest_path}")
