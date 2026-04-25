"""
Optuna 超参数搜索脚本 —— 单位分类模型 (BERT + CNN + Attention + TF-IDF)

用法:
    python tools/optuna_search_classifier.py --n-trials 50

搜索的超参数:
    - learning_rate: 学习率
    - batch_size: 批大小
    - loss_type: 损失函数类型 (cross_entropy / focal / rank_aware)
    - focal_gamma: Focal/RankAware 的 gamma
    - rank_penalty: RankAware 的惩罚系数
    - class_weight_power: 类别权重平滑指数
    - tfidf_dim: TF-IDF 特征维度
    - fgm_epsilon: FGM 对抗训练扰动系数

搜索使用小样本快速验证，找到最优参数后写入 best_params.json，
供正式训练脚本读取。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enhancements.data_paths import get_training_reports_dir
from training.train_unit_classifier_fgm import (
    BertClassifier,
    Config,
    DataProcessor,
    TextDataset,
    build_arg_parser,
    get_default_classifier_data_path,
)

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    raise SystemExit(
        "未安装 optuna，请先执行: pip install optuna"
    )


# ============================================================
# 日志配置
# ============================================================

def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"optuna_search_{ts}.log"

    logger = logging.getLogger("optuna_search")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    ))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger._log_file = str(log_file)
    
    # 确保输出不被缓冲
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None
    
    return logger


# ============================================================
# 命令行参数
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna 搜索单位分类模型超参数 (BERT+CNN+Attention+TF-IDF)")
    p.add_argument("--study-name", default="unit_cls_bert_cnn_attn")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--timeout", type=int, default=0, help="秒，0=不限时")
    p.add_argument("--search-epochs", type=int, default=3, help="每次 trial 训练轮数")
    p.add_argument("--sample-size", type=int, default=5000, help="小样本数据量")
    p.add_argument("--metric", default="macro_f1",
                   choices=["macro_f1", "weighted_f1", "top1_acc", "topk_acc"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--storage", default="", help="如 sqlite:///optuna_cls.db")
    p.add_argument("--keep-artifacts", action="store_true")
    p.add_argument("--output-dir",
                   default=str(get_training_reports_dir() / "optuna"))
    return p.parse_args()


# ============================================================
# 构建 trial 训练参数
# ============================================================

def build_trial_args(search_args: argparse.Namespace, trial: optuna.Trial) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args([])

    # ---- 搜索空间 ----
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    loss_type = trial.suggest_categorical("loss_type", ["cross_entropy", "focal", "rank_aware"])
    class_weight_power = trial.suggest_float("class_weight_power", 0.3, 0.8)
    tfidf_dim = trial.suggest_categorical("tfidf_dim", [2000, 3000, 5000])
    fgm_epsilon = trial.suggest_float("fgm_epsilon", 0.3, 1.5, step=0.1)

    if loss_type == "focal":
        focal_gamma = trial.suggest_float("focal_gamma", 0.5, 3.0)
        rank_penalty = 0.2
    elif loss_type == "rank_aware":
        focal_gamma = trial.suggest_float("focal_gamma", 0.5, 3.0)
        rank_penalty = trial.suggest_float("rank_penalty", 0.05, 0.5)
    else:
        focal_gamma = 1.0
        rank_penalty = 0.2

    # ---- 固定参数 ----
    save_dir = Path(search_args.output_dir) / "trials" / f"trial_{trial.number:03d}"

    args.data_path = get_default_classifier_data_path()
    args.save_dir = str(save_dir)
    args.report_dir = str(get_training_reports_dir())
    args.epochs = search_args.search_epochs
    args.seed = search_args.seed
    args.use_small_sample = True
    args.sample_size = search_args.sample_size
    args.train_from_scratch = True
    args.no_resume = True

    # BERT+CNN+Attention+TF-IDF 全部开启
    args.use_cnn_attention = True
    args.use_tfidf = True
    args.disable_fgm = False

    # ---- 搜索参数赋值 ----
    args.learning_rate = learning_rate
    args.batch_size = batch_size
    args.loss_type = loss_type
    args.focal_gamma = focal_gamma
    args.rank_penalty = rank_penalty
    args.class_weight_power = class_weight_power
    args.tfidf_dim = tfidf_dim

    # 把 fgm_epsilon 存到 args 上，后面传给 FGM
    args.fgm_epsilon = fgm_epsilon

    return args


# ============================================================
# 目标函数
# ============================================================

def create_objective(search_args: argparse.Namespace, logger: logging.Logger):
    def objective(trial: optuna.Trial) -> float:
        tag = f"[Trial {trial.number}]"
        args = build_trial_args(search_args, trial)
        save_dir = Path(args.save_dir)

        logger.info(f"{tag} ===== 开始 =====")
        logger.info(f"{tag} 参数: {json.dumps(trial.params, ensure_ascii=False)}")

        try:
            config = Config(args)
            config.set_seed()

            logger.debug(f"{tag} 初始化数据处理器...")
            processor = DataProcessor(config)
            train_df, val_df, train_tfidf, val_tfidf = processor.load_and_process()
            logger.info(f"{tag} 数据加载完成 - 训练: {len(train_df)}, 验证: {len(val_df)}")

            logger.debug(f"{tag} 初始化模型 (FGM epsilon={config.fgm_epsilon})...")
            classifier = BertClassifier(config)

            logger.debug(f"{tag} 创建 Dataset & DataLoader...")
            train_dataset = TextDataset(
                train_df, classifier.tokenizer, config.max_len,
                tfidf_data=train_tfidf, augment=True,
            )
            val_dataset = TextDataset(
                val_df, classifier.tokenizer, config.max_len,
                tfidf_data=val_tfidf, augment=False,
            )
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

            logger.info(f"{tag} 开始训练 ({config.epochs} epochs, batch={config.batch_size})...")
            classifier.train(train_loader, val_loader)

            logger.info(f"{tag} 开始评估...")
            metrics = classifier.evaluate(val_loader)

            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("save_dir", str(save_dir))

            value = float(metrics[search_args.metric])
            logger.info(f"{tag} 完成! {search_args.metric}={value:.6f} | 全部指标: {metrics}")
            return value

        except Exception as e:
            err_type = type(e).__name__
            err_msg = str(e)
            err_trace = traceback.format_exc()

            logger.error(f"{tag} 失败! {err_type}: {err_msg}")
            logger.error(f"{tag} 参数: {trial.params}")
            logger.error(f"{tag} 堆栈:\n{err_trace}")

            trial.set_user_attr("error_type", err_type)
            trial.set_user_attr("error_message", err_msg)
            trial.set_user_attr("error_traceback", err_trace)

            print(f"\n{'='*60}", file=sys.stderr)
            print(f"TRIAL {trial.number} FAILED: {err_type}: {err_msg}", file=sys.stderr)
            print(f"参数: {trial.params}", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)

            raise

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not search_args.keep_artifacts and save_dir.exists():
                shutil.rmtree(save_dir, ignore_errors=True)
                logger.debug(f"{tag} 已清理 {save_dir}")

    return objective


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir)

    logger.info("=" * 60)
    logger.info("Optuna 超参数搜索 — 单位分类模型 (BERT+CNN+Attention+TF-IDF)")
    logger.info(f"  study_name  = {args.study_name}")
    logger.info(f"  n_trials    = {args.n_trials}")
    logger.info(f"  search_epochs = {args.search_epochs}")
    logger.info(f"  sample_size = {args.sample_size}")
    logger.info(f"  metric      = {args.metric}")
    logger.info(f"  seed        = {args.seed}")
    logger.info(f"  输出目录    = {output_dir}")
    logger.info("=" * 60)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    try:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=args.storage or None,
            load_if_exists=bool(args.storage),
        )
    except Exception as e:
        logger.error(f"创建 Study 失败: {e}")
        logger.error(traceback.format_exc())
        raise

    logger.info("开始搜索...")
    try:
        study.optimize(
            create_objective(args, logger),
            n_trials=args.n_trials,
            timeout=args.timeout if args.timeout > 0 else None,
        )
    except KeyboardInterrupt:
        logger.warning("用户中断 (Ctrl+C)，保存当前结果...")
    except Exception as e:
        logger.error(f"搜索过程出错: {e}")
        logger.error(traceback.format_exc())

    # ---- 统计 ----
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    failed = [t for t in study.trials if t.state == TrialState.FAIL]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]

    logger.info("=" * 60)
    logger.info("搜索完成统计:")
    logger.info(f"  总数: {len(study.trials)} | 成功: {len(completed)} | "
                f"失败: {len(failed)} | 剪枝: {len(pruned)}")

    if failed:
        logger.warning(f"  失败的 trial:")
        for t in failed:
            et = t.user_attrs.get("error_type", "?")
            em = t.user_attrs.get("error_message", "?")[:120]
            logger.warning(f"    Trial {t.number}: {et} - {em}")

    # ---- 保存结果 ----
    result = {
        "study_name": args.study_name,
        "metric": args.metric,
        "search_config": {
            "n_trials": args.n_trials,
            "search_epochs": args.search_epochs,
            "sample_size": args.sample_size,
            "seed": args.seed,
        },
        "statistics": {
            "total": len(study.trials),
            "completed": len(completed),
            "failed": len(failed),
            "pruned": len(pruned),
        },
        "best_value": study.best_value if completed else None,
        "best_params": study.best_params if completed else None,
        "best_trial_user_attrs": study.best_trial.user_attrs if completed else None,
        "trials": [
            {
                "number": t.number,
                "state": str(t.state),
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
            }
            for t in study.trials
        ],
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_json = output_dir / "optuna_best_params_latest.json"
    archived_json = output_dir / f"optuna_best_params_{ts}.json"
    latest_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    archived_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"结果已保存: {latest_json}")

    # ---- CSV ----
    import pandas as pd
    rows = []
    for t in study.trials:
        row = {"number": t.number, "state": str(t.state), "value": t.value}
        row.update(t.params)
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        output_dir / "optuna_trials_latest.csv", index=False, encoding="utf-8-sig"
    )

    # ---- 写入 best_params.json 供训练脚本读取 ----
    best_params_path = PROJECT_ROOT / "training" / "best_params.json"
    if completed:
        best_params_path.write_text(
            json.dumps(study.best_params, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"最优参数已写入: {best_params_path}")
        logger.info(f"最优 {args.metric}: {study.best_value:.6f}")
        logger.info(f"最优参数: {json.dumps(study.best_params, ensure_ascii=False)}")
        print(f"\n最优 {args.metric}: {study.best_value:.6f}")
        print(f"最优参数: {json.dumps(study.best_params, ensure_ascii=False)}")
    else:
        logger.warning("没有成功完成的 trial!")
        print("警告: 没有成功完成的 trial!")

    print(f"详细报告: {latest_json}")
    print(f"日志文件: {logger._log_file}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
