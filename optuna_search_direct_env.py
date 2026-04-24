import sys
import os

sys.path.insert(0, ".")

try:
    from tools.optuna_search_classifier import parse_args, objective
    from enhancements.data_paths import get_training_reports_dir
    import optuna
    from pathlib import Path
    from datetime import datetime
    import json
    import pandas as pd
    
    print("导入成功")
    
    args = parse_args()
    print(f"参数: n_trials={args.n_trials}, epochs={args.epochs}, sample_size={args.sample_size}")
    print(f"研究名称: {args.study_name}")
    
    reports_dir = get_training_reports_dir() / "optuna"
    reports_dir.mkdir(parents=True, exist_ok=True)
    Path(args.save_root).mkdir(parents=True, exist_ok=True)
    
    print(f"报告目录: {reports_dir}")
    print(f"保存目录: {args.save_root}")
    
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage or None,
        load_if_exists=bool(args.storage),
    )
    
    print(f"研究创建成功: {study.study_name}")
    
    study.optimize(
        objective(args),
        n_trials=args.n_trials,
        timeout=args.timeout if args.timeout > 0 else None,
    )
    
    print(f"\n优化完成!")
    print(f"最佳值: {study.best_value}")
    print(f"最佳参数: {study.best_params}")
    
except Exception as e:
    import traceback
    print(f"错误: {e}")
    traceback.print_exc()
