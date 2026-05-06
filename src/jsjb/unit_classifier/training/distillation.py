from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import random
import tempfile
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.jsjb.unit_classifier.acceptance import build_acceptance_report, format_acceptance_report, write_acceptance_report
from src.jsjb.core.paths import get_training_reports_dir
from src.jsjb.unit_classifier.catalog import GENERIC_BAD_UNITS, canonicalize_unit, load_unit_catalog, normalize_unit_text
from src.jsjb.unit_classifier.tokenization import chinese_tokenizer
from src.jsjb.unit_classifier.model import BertCNNAttention
from src.jsjb.unit_classifier.training.data import TextDataset


DEFAULT_CLASSIFIER_DATA_CANDIDATES = [
    Path(r"C:\Users\28414\Desktop\留言板合并数据 - 副本(3).xlsx"),
    Path(r"C:\Users\28414\Desktop\留言板合并数据.xlsx"),
    Path(r"C:\Users\28414\Documents\New project\raw_data_analysis\master_table_v1.csv"),
    Path(r"C:\Users\28414\Desktop\接诉即办_留言板合并数据.xlsx"),
]


def _atomic_torch_save(obj, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path_obj.parent, delete=False, suffix=".tmp") as tmp_fp:
        tmp_path = tmp_fp.name
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _atomic_json_save(payload: dict, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path_obj.parent, delete=False, suffix=".tmp", mode="w", encoding="utf-8") as tmp_fp:
        json.dump(payload, tmp_fp, ensure_ascii=False, indent=2)
        tmp_path = tmp_fp.name
    try:
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _build_label_signature(id2label: dict[str, str]) -> str:
    canonical = json.dumps(id2label, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_default_classifier_data_path() -> str:
    for path in DEFAULT_CLASSIFIER_DATA_CANDIDATES:
        if path.exists():
            return str(path)
    return str(DEFAULT_CLASSIFIER_DATA_CANDIDATES[0])


_chinese_tokenizer = chinese_tokenizer

from src.jsjb.unit_classifier.training.losses import (
    EMAModel,
    FGM,
    FocalLoss,
    RankAwareLoss,
    SelfDistillationLoss,
    SoftLabelBank,
    build_unit_prior_matrix,
)




class Config:
    def __init__(self, args: argparse.Namespace):
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.save_dir = args.save_dir
        self.report_dir = args.report_dir
        self.train_from_scratch = args.train_from_scratch
        self.no_resume = args.no_resume
        self.init_from = args.init_from
        os.makedirs(self.save_dir, exist_ok=True)

        self.model_save_path = os.path.join(self.save_dir, "pytorch_model.bin")
        self.map_save_path = os.path.join(self.save_dir, "label_map.json")
        self.training_state_path = os.path.join(self.save_dir, "training_state.pt")
        self.model_meta_path = os.path.join(self.save_dir, "model_meta.json")
        self.tfidf_vectorizer_path = os.path.join(self.save_dir, "tfidf_vectorizer.joblib")
        self.soft_label_bank_path = os.path.join(self.save_dir, "soft_label_bank.pt")
        self.unit_prior_matrix_path = os.path.join(self.save_dir, "unit_prior_matrix.pt")
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.grad_accum_steps = args.grad_accum_steps
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = args.seed
        self.top_k_eval = args.top_k_eval
        self.use_fgm = not args.disable_fgm
        self.use_small_sample = args.use_small_sample
        self.sample_size = args.sample_size
        self.use_cnn_attention = args.use_cnn_attention
        self.use_tfidf = args.use_tfidf
        self.tfidf_dim = args.tfidf_dim
        self.tfidf_hidden = 64
        self.loss_type = args.loss_type
        self.focal_gamma = args.focal_gamma
        self.rank_penalty = args.rank_penalty
        self.class_weight_power = args.class_weight_power
        self.class_weight_min = args.class_weight_min
        self.class_weight_max = args.class_weight_max
        self.focal_warmup_epochs = args.focal_warmup_epochs
        self.best_metric = args.best_metric
        
        # 自蒸馏参数
        self.temperature = args.temperature
        self.temperature_start = getattr(args, "temperature_start", args.temperature)
        self.temperature_end = getattr(args, "temperature_end", 1.5)
        self.distill_weight = args.distill_weight
        self.hard_weight = args.hard_weight
        self.ema_decay = args.ema_decay
        self.distill_strategy = getattr(args, "distill_strategy", "ema_bank")
        self.distill_top_k = getattr(args, "distill_top_k", 5)
        self.ema_bank_momentum = getattr(args, "ema_bank_momentum", 0.9)
        self.prior_alpha = getattr(args, "prior_alpha", 0.15)
        self.prior_top_k = getattr(args, "prior_top_k", 8)
        self.prior_warmup_epochs = getattr(args, "prior_warmup_epochs", 1)
        
        self.num_classes = None
        self.id2label = None
        self.class_weights = None
        self.label_signature = ""

    def set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


class DataProcessor:
    REQUIRED_COLUMNS = ["留言标签", "留言标题", "留言正文", "官方回复单位"]
    MASTER_TABLE_COLUMNS = ["message_title", "message_body", "reply_unit_norm"]

    def __init__(self, config: Config):
        self.config = config
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=config.tfidf_dim,
            tokenizer=_chinese_tokenizer,
            token_pattern=None
        ) if config.use_tfidf else None

    def load_and_process(self):
        print(f"正在加载数据: {self.config.data_path} ...")
        if str(self.config.data_path).lower().endswith(".csv"):
            df = pd.read_csv(self.config.data_path, low_memory=False)
        else:
            df = pd.read_excel(self.config.data_path, engine="openpyxl")

        using_master_table = all(column in df.columns for column in self.MASTER_TABLE_COLUMNS)
        if using_master_table:
            print("检测到统一母表字段，分类训练将使用规范化单位标签。")
            if "recommended_for_unit_cls" in df.columns:
                mask = df["recommended_for_unit_cls"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
                df = df[mask].copy()

            if "message_tag_level1" not in df.columns:
                df["message_tag_level1"] = ""
            df["message_tag_level1"] = df["message_tag_level1"].fillna("").astype(str)
            df["message_title"] = df["message_title"].fillna("").astype(str)
            df["message_body"] = df["message_body"].fillna("").astype(str)
            df["target_unit"] = df["reply_unit_norm"].apply(canonicalize_unit)
            df = df[
                df["target_unit"].fillna("").astype(str).str.strip().ne("")
                & ~df["target_unit"].isin(GENERIC_BAD_UNITS)
                & df["message_title"].str.strip().ne("")
                & df["message_body"].str.strip().ne("")
            ].copy()
            tag_col = "message_tag_level1"
            title_col = "message_title"
            body_col = "message_body"
            target_col = "target_unit"
        else:
            missing = [column for column in self.REQUIRED_COLUMNS if column not in df.columns]
            if missing:
                raise ValueError(f"分类数据缺少必要列: {missing}")
            target_col = "官方回复单位"
            tag_col = "留言标签"
            title_col = "留言标题"
            body_col = "留言正文"
            df.dropna(subset=[target_col, tag_col, title_col, body_col], inplace=True)
            df[target_col] = df[target_col].apply(canonicalize_unit)
            df = df[
                df[target_col].fillna("").astype(str).str.strip().ne("")
                & ~df[target_col].isin(GENERIC_BAD_UNITS)
            ].copy()

        if self.config.use_small_sample and len(df) > self.config.sample_size:
            df = df.sample(n=self.config.sample_size, random_state=self.config.seed)
            print(f"警告：正在使用小样本模式 ({self.config.sample_size}条)")
        else:
            print(f"全量模式：共 {len(df)} 条数据")

        df[title_col] = df[title_col].fillna("")
        df[body_col] = df[body_col].fillna("")
        df[tag_col] = df[tag_col].fillna("").astype(str)

        df["full_text"] = (
            "【" + df[tag_col].astype(str)
            + "】"
            + df[title_col].astype(str)
            + "。"
            + df[body_col].astype(str)
        )

        unique_units = sorted(df[target_col].astype(str).map(normalize_unit_text).unique())
        self.config.num_classes = len(unique_units)
        self.config.id2label = {str(idx): str(label) for idx, label in enumerate(unique_units)}
        self.config.label_signature = _build_label_signature(self.config.id2label)
        label2id = {label: idx for idx, label in enumerate(unique_units)}
        print(f"标签签名: {self.config.label_signature[:12]}...（将在最佳权重保存时同步落盘）")

        df["label_id"] = df[target_col].astype(str).map(normalize_unit_text).map(label2id)
        print(f"分类数量: {self.config.num_classes}")

        try:
            indices = np.arange(len(df))
            train_idx, val_idx = train_test_split(
                indices,
                test_size=0.2,
                random_state=self.config.seed,
                stratify=df["label_id"],
            )
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)
        except ValueError:
            print("警告：无法分层抽样，转为随机抽样")
            indices = np.arange(len(df))
            train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=self.config.seed)
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)

        print("正在基于训练集计算类别权重...")
        train_labels = train_df["label_id"].values
        observed_classes = np.unique(train_labels)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=observed_classes,
            y=train_labels,
        )
        class_weights = np.power(class_weights, self.config.class_weight_power)
        class_weights = np.clip(class_weights, self.config.class_weight_min, self.config.class_weight_max)

        full_weights = torch.ones(self.config.num_classes)
        for cls, weight in zip(observed_classes, class_weights):
            full_weights[int(cls)] = float(weight)

        self.config.class_weights = full_weights.float()
        print(f"类别权重已生成 (Top 5 权重: {self.config.class_weights[:5]})")

        train_tfidf = None
        val_tfidf = None
        if self.config.use_tfidf and self.tfidf_vectorizer is not None:
            print("正在仅用训练集拟合 TF-IDF 向量器 (jieba分词)...")
            train_texts = train_df["full_text"].tolist()
            val_texts = val_df["full_text"].tolist()
            self.tfidf_vectorizer.fit(train_texts)
            train_tfidf = self.tfidf_vectorizer.transform(train_texts).toarray()
            val_tfidf = self.tfidf_vectorizer.transform(val_texts).toarray()
            self.config.tfidf_dim = int(train_tfidf.shape[1])
            print(f"TF-IDF 特征维度: {self.config.tfidf_dim}")

        return train_df, val_df, train_tfidf, val_tfidf










class BertClassifier:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        if config.use_cnn_attention:
            print(">>> 初始化 BERT + CNN + Attention + TF-IDF 模型...")
            self.model = BertCNNAttention(config).to(config.device)
        else:
            print(">>> 初始化标准 BERT 分类模型...")
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=config.num_classes,
                ignore_mismatched_sizes=True,
            ).to(config.device)
        
        self.class_weights = config.class_weights.to(config.device) if config.class_weights is not None else None
        self.criterion = None
        self._set_epoch_loss(0)
        self.fgm = FGM(self.model) if config.use_fgm else None
        if self.fgm is not None:
            print(">>> [System] FGM 对抗训练已启用")
        
        if config.init_from and os.path.exists(config.init_from):
            state_dict = torch.load(config.init_from, map_location=config.device)
            self.model.load_state_dict(state_dict)
            print(f">>> [System] 已从 {config.init_from} 加载预训练权重")
        
        self.ema_model = EMAModel(self.model, decay=config.ema_decay)
        print(f">>> [System] EMA教师模型已启用 (decay={config.ema_decay})")

        self.soft_label_bank = None
        self.unit_prior_matrix = None

    def _ensure_soft_label_bank(self, num_samples: int) -> None:
        if self.config.loss_type != "distillation" or self.config.distill_strategy not in {"ema_bank", "ema_bank_prior"}:
            return
        if self.soft_label_bank is None:
            self.soft_label_bank = SoftLabelBank(
                num_samples=num_samples,
                num_classes=int(self.config.num_classes or 0),
                momentum=self.config.ema_bank_momentum,
                top_k=self.config.distill_top_k,
            )
            print(
                f">>> [System] EMA Soft Label Bank enabled "
                f"(samples={num_samples}, classes={self.config.num_classes}, "
                f"momentum={self.config.ema_bank_momentum}, top_k={self.config.distill_top_k})"
            )

    def _save_soft_label_bank(self) -> None:
        if self.soft_label_bank is not None:
            _atomic_torch_save(self.soft_label_bank.state_dict(), self.config.soft_label_bank_path)

    def _ensure_unit_prior_matrix(self) -> None:
        if self.config.loss_type != "distillation" or self.config.distill_strategy not in {"prior", "ema_bank_prior"}:
            return
        if self.unit_prior_matrix is None:
            matrix = build_unit_prior_matrix(
                self.config.id2label,
                prior_top_k=self.config.prior_top_k,
            )
            self.unit_prior_matrix = matrix.to(self.config.device)
            _atomic_torch_save(matrix.cpu(), self.config.unit_prior_matrix_path)
            print(
                f">>> [System] Unit prior matrix enabled "
                f"(classes={self.config.num_classes}, prior_alpha={self.config.prior_alpha}, "
                f"top_k={self.config.prior_top_k})"
            )

    def _mix_with_unit_prior(
        self,
        teacher_probs: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        if (
            self.unit_prior_matrix is None
            or self.config.prior_alpha <= 0
            or epoch < self.config.prior_warmup_epochs
        ):
            return teacher_probs
        prior_rows = self.unit_prior_matrix[labels].to(device=teacher_probs.device, dtype=teacher_probs.dtype)
        mixed = (1.0 - self.config.prior_alpha) * teacher_probs + self.config.prior_alpha * prior_rows
        return mixed / (mixed.sum(dim=1, keepdim=True) + 1e-8)

    def _build_loss(self, loss_name: str):
        if loss_name == "focal":
            return FocalLoss(alpha=self.class_weights, gamma=self.config.focal_gamma)
        if loss_name == "rank_aware":
            return RankAwareLoss(
                alpha=self.class_weights,
                gamma=self.config.focal_gamma,
                top_k=self.config.top_k_eval,
                rank_penalty=self.config.rank_penalty,
            )
        if loss_name == "distillation":
            return SelfDistillationLoss(
                alpha=self.class_weights,
                gamma=self.config.focal_gamma,
                rank_top_k=self.config.top_k_eval,
                rank_penalty=self.config.rank_penalty,
                temperature=self.config.temperature,
                top_k=self.config.distill_top_k,
                hard_weight=self.config.hard_weight,
                distill_weight=self.config.distill_weight,
            )
        return nn.CrossEntropyLoss(weight=self.class_weights)

    def _set_epoch_loss(self, epoch: int) -> None:
        active_loss = self.config.loss_type
        if self.config.loss_type == "focal" and epoch < self.config.focal_warmup_epochs:
            active_loss = "cross_entropy"
        if active_loss == "distillation":
            progress = epoch / max(self.config.epochs - 1, 1)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            self.config.temperature = float(
                self.config.temperature_end
                + (self.config.temperature_start - self.config.temperature_end) * cosine
            )
        self.criterion = self._build_loss(active_loss)
        if active_loss == "focal":
            print(f">>> [System] 当前损失: FocalLoss(gamma={self.config.focal_gamma})")
        elif active_loss == "rank_aware":
            print(
                f">>> [System] 当前损失: RankAwareLoss(gamma={self.config.focal_gamma}, "
                f"rank_penalty={self.config.rank_penalty})"
            )
        elif active_loss == "distillation":
            print(
                f">>> [System] 当前损失: SelfDistillationLoss(temperature={self.config.temperature}, "
                f"top_k={self.config.distill_top_k}, strategy={self.config.distill_strategy}, hard_weight={self.config.hard_weight}, "
                f"distill_weight={self.config.distill_weight})"
            )
        else:
            print(">>> [System] 当前损失: CrossEntropyLoss")

    def _save_best_artifacts(self, best_score: float):
        _atomic_torch_save(self.model.state_dict(), self.config.model_save_path)
        if self.config.use_cnn_attention and hasattr(self.model, "bert"):
            self.model.bert.config.save_pretrained(self.config.save_dir)
        elif not self.config.use_cnn_attention:
            self.model.config.save_pretrained(self.config.save_dir)
        self.tokenizer.save_pretrained(self.config.save_dir)
        if self.config.use_tfidf and hasattr(self.config, "tfidf_vectorizer"):
            joblib.dump(self.config.tfidf_vectorizer, self.config.tfidf_vectorizer_path)
        self._save_soft_label_bank()
        _atomic_json_save(self.config.id2label, self.config.map_save_path)
        _atomic_json_save(
            {
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "best_metric_name": self.config.best_metric,
                "best_score": float(best_score),
                "num_classes": int(self.config.num_classes or 0),
                "architecture": "bert_cnn_attention_tfidf" if self.config.use_cnn_attention else "legacy_bert_sequence",
                "classifier_route": (
                    "hybrid_prior_distill"
                    if self.config.use_cnn_attention and self.config.distill_strategy == "ema_bank_prior"
                    else (
                    "hybrid_prior_teacher"
                    if self.config.use_cnn_attention and self.config.distill_strategy == "prior"
                    else (
                    "hybrid_ema_soft_label_bank"
                    if self.config.use_cnn_attention and self.config.distill_strategy == "ema_bank"
                    else ("hybrid_self_distill" if self.config.use_cnn_attention else "legacy_bert")
                    ))
                ),
                "base_model": self.config.model_name,
                "use_cnn_attention": bool(self.config.use_cnn_attention),
                "use_tfidf": bool(self.config.use_tfidf),
                "tfidf_dim": int(self.config.tfidf_dim),
                "tfidf_hidden": int(self.config.tfidf_hidden),
                "tfidf_vectorizer": "tfidf_vectorizer.joblib" if self.config.use_tfidf else "",
                "label_signature": self.config.label_signature,
                "loss_type": self.config.loss_type,
                "temperature": self.config.temperature,
                "temperature_start": self.config.temperature_start,
                "temperature_end": self.config.temperature_end,
                "distill_weight": self.config.distill_weight,
                "hard_weight": self.config.hard_weight,
                "distill_strategy": self.config.distill_strategy,
                "distill_top_k": self.config.distill_top_k,
                "ema_bank_momentum": self.config.ema_bank_momentum,
                "prior_alpha": self.config.prior_alpha,
                "prior_top_k": self.config.prior_top_k,
                "prior_warmup_epochs": self.config.prior_warmup_epochs,
                "soft_label_bank": "soft_label_bank.pt" if self.soft_label_bank is not None else "",
                "unit_prior_matrix": "unit_prior_matrix.pt" if self.unit_prior_matrix is not None else "",
            },
            self.config.model_meta_path,
        )

    def _save_training_state(self, epoch: float, best_score: float, optimizer, scheduler):
        _atomic_torch_save(
            {
                "epoch": epoch,
                "best_score": best_score,
                "best_acc": best_score,
                "best_metric_name": self.config.best_metric,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "num_classes": int(self.config.num_classes or 0),
                "use_cnn_attention": bool(self.config.use_cnn_attention),
                "use_tfidf": bool(self.config.use_tfidf),
                "label_signature": self.config.label_signature,
                "ema_shadow": self.ema_model.shadow,
                "soft_label_bank_state": self.soft_label_bank.state_dict() if self.soft_label_bank is not None else None,
            },
            self.config.training_state_path,
        )

    def _maybe_resume(self, optimizer, scheduler) -> tuple[int, float]:
        if self.config.train_from_scratch or self.config.no_resume:
            return 0, float("-inf")
        if not os.path.exists(self.config.training_state_path):
            return 0, float("-inf")

        checkpoint = torch.load(self.config.training_state_path, map_location=self.config.device)
        checkpoint_signature = checkpoint.get("label_signature", "")
        checkpoint_num_classes = int(checkpoint.get("num_classes", self.config.num_classes or 0))
        checkpoint_cnn = bool(checkpoint.get("use_cnn_attention", self.config.use_cnn_attention))
        checkpoint_tfidf = bool(checkpoint.get("use_tfidf", self.config.use_tfidf))

        incompatible_reason = None
        if checkpoint_signature and checkpoint_signature != self.config.label_signature:
            incompatible_reason = (
                "标签签名不一致（疑似数据/标签体系已变化），将忽略断点并从头训练，避免旧权重+新标签错配。"
            )
        elif checkpoint_num_classes and checkpoint_num_classes != int(self.config.num_classes or 0):
            incompatible_reason = "类别数不一致，将忽略断点并从头训练。"
        elif checkpoint_cnn != self.config.use_cnn_attention or checkpoint_tfidf != self.config.use_tfidf:
            incompatible_reason = "模型结构参数不一致（CNN/TF-IDF开关变化），将忽略断点并从头训练。"

        if incompatible_reason:
            print(f"警告：{incompatible_reason}")
            return 0, float("-inf")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # 恢复EMA参数
        if "ema_shadow" in checkpoint:
            self.ema_model.shadow = checkpoint["ema_shadow"]
            if self.soft_label_bank is not None and checkpoint.get("soft_label_bank_state") is not None:
                self.soft_label_bank.load_state_dict(checkpoint["soft_label_bank_state"])
                print(">>> [System] Soft Label Bank restored from training_state.pt")
        
        epoch = float(checkpoint.get("epoch", 0.0))
        start_epoch = int(epoch)
        best_score = float(checkpoint.get("best_score", checkpoint.get("best_acc", 0.0)))
        
        if epoch != start_epoch:
            print(f"检测到分类训练断点，将从 Epoch {epoch:.2f}/{self.config.epochs} 继续训练")
        else:
            print(f"检测到分类训练断点，将从 epoch {start_epoch + 1}/{self.config.epochs} 继续训练")
        
        return start_epoch, best_score

    def _metric_value(self, metrics: dict[str, float]) -> float:
        return float(metrics.get(self.config.best_metric, 0.0))

    def _get_logits(self, input_ids, attention_mask, tfidf_vec=None):
        """获取模型logits"""
        if self.config.use_cnn_attention:
            return self.model(input_ids, attention_mask, tfidf_vec)
        else:
            outputs = self.model(input_ids, attention_mask=attention_mask)
            return outputs.logits

    def train(self, train_loader, val_loader):
        self._ensure_soft_label_bank(len(train_loader.dataset))
        self._ensure_unit_prior_matrix()
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        num_update_steps = len(train_loader) // self.config.grad_accum_steps * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_update_steps * 0.1),
            num_training_steps=num_update_steps,
        )

        start_epoch, best_score = self._maybe_resume(optimizer, scheduler)
        best_artifacts_saved = os.path.exists(self.config.model_save_path) and os.path.exists(self.config.map_save_path)
        print(f"\n开始训练 (Epochs: {self.config.epochs}, Grad Accum: {self.config.grad_accum_steps})...")
        print(f"最佳模型判定指标: {self.config.best_metric}")
        
        steps_per_epoch = len(train_loader)
        save_interval_steps = max(1, steps_per_epoch // 4)
        print(f"断点保存策略: 每 {save_interval_steps} steps (约0.25 epoch) 保存一次")
        
        for epoch in range(start_epoch, self.config.epochs):
            self._set_epoch_loss(epoch)
            self.model.train()
            total_train_loss = 0.0
            loop = tqdm(train_loader, total=len(train_loader), leave=True)
            loop.set_description(f"Epoch [{epoch + 1}/{self.config.epochs}]")

            for step, batch in enumerate(loop):
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)
                sample_idx = batch.get("sample_idx")
                tfidf_vec = batch.get("tfidf_vec")
                if tfidf_vec is not None:
                    tfidf_vec = tfidf_vec.to(self.config.device)
                
                # 学生模型前向传播
                student_logits = self._get_logits(input_ids, attention_mask, tfidf_vec)
                
                # 计算损失
                if self.config.loss_type == "distillation":
                    # 使用EMA模型作为教师模型
                    self.ema_model.apply_shadow()
                    with torch.no_grad():
                        teacher_logits = self._get_logits(input_ids, attention_mask, tfidf_vec)
                    self.ema_model.restore()

                    teacher_probs = None
                    if self.config.distill_strategy in {"ema_bank", "ema_bank_prior"} and self.soft_label_bank is not None:
                        with torch.no_grad():
                            current_teacher_probs = self.criterion.build_soft_targets(teacher_logits)
                            bank_probs_cpu = self.soft_label_bank.update(sample_idx, current_teacher_probs)
                        teacher_probs = bank_probs_cpu.to(
                            device=self.config.device,
                            dtype=student_logits.dtype,
                        )
                    elif self.config.distill_strategy == "prior":
                        teacher_probs = self.criterion.build_soft_targets(teacher_logits)

                    if teacher_probs is not None:
                        teacher_probs = self._mix_with_unit_prior(teacher_probs, labels, epoch)

                    loss, loss_dict = self.criterion(
                        student_logits,
                        teacher_logits,
                        labels,
                        teacher_probs=teacher_probs,
                    )
                    loss_display = loss_dict['total_loss']
                else:
                    loss = self.criterion(student_logits, labels)
                    loss_display = loss.item()
                
                loss = loss / self.config.grad_accum_steps
                loss.backward()

                # FGM对抗训练
                if self.fgm is not None:
                    self.fgm.attack()
                    student_logits_adv = self._get_logits(input_ids, attention_mask, tfidf_vec)
                    
                    if self.config.loss_type == "distillation":
                        self.ema_model.apply_shadow()
                        with torch.no_grad():
                            teacher_logits_adv = self._get_logits(input_ids, attention_mask, tfidf_vec)
                        self.ema_model.restore()
                        if self.config.distill_strategy in {"ema_bank", "ema_bank_prior"} and self.soft_label_bank is not None:
                            teacher_probs_adv = self.soft_label_bank.get(
                                sample_idx,
                                device=self.config.device,
                                dtype=student_logits_adv.dtype,
                            )
                        elif self.config.distill_strategy == "prior":
                            teacher_probs_adv = self.criterion.build_soft_targets(teacher_logits_adv)
                        else:
                            teacher_probs_adv = None
                        if teacher_probs_adv is not None:
                            teacher_probs_adv = self._mix_with_unit_prior(teacher_probs_adv, labels, epoch)
                        loss_adv, _ = self.criterion(
                            student_logits_adv,
                            teacher_logits_adv,
                            labels,
                            teacher_probs=teacher_probs_adv,
                        )
                    else:
                        loss_adv = self.criterion(student_logits_adv, labels)
                    
                    loss_adv = loss_adv / self.config.grad_accum_steps
                    loss_adv.backward()
                    self.fgm.restore()

                if (step + 1) % self.config.grad_accum_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # 更新EMA模型
                    self.ema_model.update()

                current_epoch_fraction = (step + 1) / steps_per_epoch
                if (step + 1) % save_interval_steps == 0:
                    current_epoch = epoch + current_epoch_fraction
                    self._save_training_state(current_epoch, best_score, optimizer, scheduler)
                    print(f"  断点已保存 (Epoch {current_epoch:.2f})")

                total_train_loss += loss.item() * self.config.grad_accum_steps
                loop.set_postfix(loss=loss_display)

            avg_train_loss = total_train_loss / len(train_loader)
            metrics = self.evaluate(val_loader)
            current_score = self._metric_value(metrics)
            print(
                f"Epoch {epoch + 1} | Avg Loss: {avg_train_loss:.4f} | "
                f"Val Top-1 Acc: {metrics['top1_acc']:.4f} | "
                f"Val Top-{self.config.top_k_eval} Acc: {metrics['topk_acc']:.4f} | "
                f"Val Macro-F1: {metrics['macro_f1']:.4f} | "
                f"Val Weighted-F1: {metrics['weighted_f1']:.4f}"
            )

            if current_score > best_score:
                best_score = current_score
                self._save_best_artifacts(best_score)
                best_artifacts_saved = True
                print(f"  --> 保存最佳模型 ({self.config.best_metric}: {best_score:.4f})")

            self._save_training_state(epoch, best_score, optimizer, scheduler)

        if not best_artifacts_saved:
            print("警告：本次训练未触发最佳模型保存，正在强制保存最终权重与标签映射以确保产物一致。")
            self._save_best_artifacts(best_score)
        
        torch.cuda.empty_cache()

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        k = min(self.config.top_k_eval, self.config.num_classes)
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="评估中", leave=False):
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)
                sample_idx = batch.get("sample_idx")
                tfidf_vec = batch.get("tfidf_vec")
                if tfidf_vec is not None:
                    tfidf_vec = tfidf_vec.to(self.config.device)
                
                logits = self._get_logits(input_ids, attention_mask, tfidf_vec)

                if self.config.loss_type == "distillation":
                    self.ema_model.apply_shadow()
                    teacher_logits = self._get_logits(input_ids, attention_mask, tfidf_vec)
                    self.ema_model.restore()
                    loss, _ = self.criterion(logits, teacher_logits, labels)
                else:
                    loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
                _, topk_indices = torch.topk(logits, k=k, dim=1)
                labels_expanded = labels.view(-1, 1)
                total_correct += torch.sum(topk_indices == labels_expanded).item()
                total_samples += labels.size(0)

        topk_acc = total_correct / total_samples if total_samples > 0 else 0.0
        top1_acc = (
            sum(int(pred == label) for pred, label in zip(all_preds, all_labels)) / total_samples
            if total_samples > 0 else 0.0
        )
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0.0
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0) if all_labels else 0.0
        avg_loss = total_loss / max(len(data_loader), 1)
        return {
            "top1_acc": float(top1_acc),
            "topk_acc": float(topk_acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "avg_loss": float(avg_loss),
        }

    def full_report(self, data_loader):
        print("\n正在生成全量测试报告及错误分析...")
        self.model.load_state_dict(torch.load(self.config.model_save_path, map_location=self.config.device))
        self.model.eval()

        all_preds = []
        all_labels = []
        error_list = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Testing"):
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)
                sample_idx = batch.get("sample_idx")
                tfidf_vec = batch.get("tfidf_vec")
                if tfidf_vec is not None:
                    tfidf_vec = tfidf_vec.to(self.config.device)
                
                logits = self._get_logits(input_ids, attention_mask, tfidf_vec)
                preds = torch.argmax(logits, dim=1)

                batch_preds = preds.cpu().numpy()
                batch_labels = labels.cpu().numpy()
                all_preds.extend(batch_preds)
                all_labels.extend(batch_labels)

                for i in range(len(batch_labels)):
                    if batch_labels[i] != batch_preds[i]:
                        true_name = self.config.id2label[str(batch_labels[i])]
                        pred_name = self.config.id2label[str(batch_preds[i])]
                        error_list.append((true_name, pred_name))

        present_labels = sorted(list(set(all_labels) | set(all_preds)))
        target_names = [self.config.id2label[str(i)] for i in present_labels]
        report = classification_report(
            all_labels,
            all_preds,
            labels=present_labels,
            target_names=target_names,
            zero_division=0,
        )
        top1_acc = sum(int(pred == label) for pred, label in zip(all_preds, all_labels)) / max(len(all_labels), 1)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0.0
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0) if all_labels else 0.0

        print("\n" + "=" * 50)
        print("最终模型评估报告")
        print("=" * 50)
        print(f"Top-1 Acc: {top1_acc:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")
        print(f"Weighted-F1: {weighted_f1:.4f}")
        print(report)

        print("\n" + "=" * 50)
        print(f"错误分析 共发现 {len(error_list)} 个错误")
        print("=" * 50)
        if error_list:
            error_counts = Counter(error_list)
            print(f"{'真实单位':<20} -> {'被错判为':<20} | 次数")
            print("-" * 60)
            for (true_u, pred_u), count in error_counts.most_common(15):
                print(f"{true_u:<20} -> {pred_u:<20} | {count}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练官方回复单位分类模型 - 自蒸馏软标签版本")
    parser.add_argument("--model-name", default=r"C:\python\接诉即办\.venv\local_roberta_model")
    parser.add_argument("--data-path", default=get_default_classifier_data_path())
    parser.add_argument("--save-dir", default=r"C:\Users\28414\PycharmProjects\接诉即办项目\final_model_distillation")
    parser.add_argument("--report-dir", default=str(get_training_reports_dir()))
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-eval", type=int, default=3)
    parser.add_argument("--disable-fgm", action="store_true", default=True, help="默认关闭FGM对抗训练")
    parser.add_argument("--enable-fgm", dest="disable_fgm", action="store_false", help="启用FGM对抗训练")
    parser.add_argument("--use-small-sample", action="store_true")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--train-from-scratch", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--init-from", default=r"C:\Users\28414\PycharmProjects\接诉即办项目\final_model_fgm\pytorch_model.bin", help="从指定路径加载预训练权重")
    parser.add_argument("--legacy-bert", action="store_true", help="使用纯 BERT/RoBERTa 架构（关闭 CNN/Attention/TF-IDF）")
    parser.add_argument("--use-cnn-attention", action="store_true", default=True, help="启用 BERT+CNN+Attention 架构")
    parser.add_argument("--no-cnn-attention", dest="use_cnn_attention", action="store_false", help="关闭 CNN/Attention 分支")
    parser.add_argument("--use-tfidf", action="store_true", default=True, help="启用 TF-IDF 特征分支")
    parser.add_argument("--no-tfidf", dest="use_tfidf", action="store_false", help="关闭 TF-IDF 特征分支")
    parser.add_argument("--tfidf-dim", type=int, default=3000, help="TF-IDF 特征维度")
    parser.add_argument("--loss-type", type=str, default="distillation", 
                        choices=['focal', 'rank_aware', 'cross_entropy', 'distillation'],
                        help="损失函数类型")
    parser.add_argument("--focal-gamma", type=float, default=1.5, help="Focal/RankAware 的 gamma")
    parser.add_argument("--rank-penalty", type=float, default=0.15, help="RankAware Top-K 内样本惩罚系数")
    parser.add_argument("--class-weight-power", type=float, default=0.6, help="类别权重平滑指数")
    parser.add_argument("--class-weight-min", type=float, default=0.5, help="类别权重下限")
    parser.add_argument("--class-weight-max", type=float, default=5.0, help="类别权重上限")
    parser.add_argument("--focal-warmup-epochs", type=int, default=2, help="Focal 前先用 CE 预热的 epoch 数")
    parser.add_argument(
        "--best-metric",
        type=str,
        default="macro_f1",
        choices=["macro_f1", "weighted_f1", "top1_acc", "topk_acc"],
        help="保存最佳模型时使用的验证指标",
    )
    
    # 自蒸馏参数
    parser.add_argument("--temperature", type=float, default=4.0, 
                        help="蒸馏温度（越高分布越平滑）")
    parser.add_argument("--temperature-start", type=float, default=4.0, help="蒸馏温度起始值")
    parser.add_argument("--temperature-end", type=float, default=1.5, help="蒸馏温度结束值")
    parser.add_argument(
        "--distill-strategy",
        type=str,
        default="ema_bank_prior",
        choices=["ema_teacher", "ema_bank", "prior", "ema_bank_prior"],
        help="自蒸馏策略：ema_teacher / ema_bank / prior / ema_bank_prior。",
    )
    parser.add_argument("--distill-top-k", type=int, default=5, help="软标签分布保留的 Top-K 类别数")
    parser.add_argument("--ema-bank-momentum", type=float, default=0.9, help="EMA Soft Label Bank 更新动量")
    parser.add_argument("--prior-alpha", type=float, default=0.12, help="业务先验软标签融合权重")
    parser.add_argument("--prior-top-k", type=int, default=8, help="每个单位先验分布保留的相关单位数量")
    parser.add_argument("--prior-warmup-epochs", type=int, default=1, help="前几个 epoch 不启用业务先验融合")
    parser.add_argument("--distill-weight", type=float, default=0.18, 
                        help="蒸馏损失权重")
    parser.add_argument("--hard-weight", type=float, default=0.82, 
                        help="硬标签损失权重")
    parser.add_argument("--ema-decay", type=float, default=0.999, 
                        help="EMA模型衰减率")
    
    return parser


def parse_args() -> argparse.Namespace:
    args = build_arg_parser().parse_args()
    if args.legacy_bert:
        args.use_cnn_attention = False
        args.use_tfidf = False
        print(">>> 已启用 legacy_bert 路线，将关闭 CNN/Attention 与 TF-IDF 分支")
    return args


def run_training(args: argparse.Namespace) -> dict:
    config = Config(args)
    config.set_seed()

    if not os.path.exists(config.model_name):
        raise FileNotFoundError(f"未找到基础分类模型目录: {config.model_name}")
    if not os.path.exists(config.data_path):
        raise FileNotFoundError(f"未找到分类训练数据文件: {config.data_path}")

    processor = DataProcessor(config)
    train_df, val_df, train_tfidf, val_tfidf = processor.load_and_process()
    config.tfidf_vectorizer = processor.tfidf_vectorizer
    classifier = BertClassifier(config)

    train_dataset = TextDataset(train_df, classifier.tokenizer, config.max_len, 
                                 tfidf_data=train_tfidf, augment=True)
    val_dataset = TextDataset(val_df, classifier.tokenizer, config.max_len, 
                               tfidf_data=val_tfidf, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    classifier.train(train_loader, val_loader)
    classifier.full_report(val_loader)

    report = build_acceptance_report(
        target="classifier",
        classifier_model_dir=config.save_dir,
        classifier_base_model=config.model_name,
    )
    paths = write_acceptance_report(report, report_dir=config.report_dir, filename_prefix="classifier_distillation_acceptance")
    print("\n=== 训练产物验收 ===")
    print(format_acceptance_report(report))
    print(f"验收报告: {paths['latest']}")
    print(f"归档报告: {paths['archived']}")
    return {
        "config": config,
        "acceptance_report": report,
        "acceptance_paths": paths,
    }


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
