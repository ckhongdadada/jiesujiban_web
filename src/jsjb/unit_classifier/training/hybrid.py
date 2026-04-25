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

import jieba
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.jsjb.unit_classifier.acceptance import build_acceptance_report, format_acceptance_report, write_acceptance_report
from src.jsjb.core.paths import get_training_reports_dir
from src.jsjb.unit_classifier.catalog import GENERIC_BAD_UNITS, canonicalize_unit, normalize_unit_text
from src.jsjb.unit_classifier.tokenization import chinese_tokenizer


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


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name="embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class EMAModel:
    """指数移动平均模型（教师模型）"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """应用EMA参数（用于推理）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


class RankAwareDistillationLoss(nn.Module):
    """
    Ranking-aware self-distillation loss.

    Hard part keeps the FGM/RankAware Top-K objective; soft part uses EMA teacher
    probabilities. The KL term is dynamically scaled to the same magnitude as
    the hard loss so the soft labels do not dominate training.
    """

    def __init__(
        self,
        alpha=None,
        gamma=1.5,
        top_k=3,
        rank_penalty=0.15,
        temperature=2.0,
        distill_weight=0.18,
        hard_weight=0.82,
        normalize_distill=True,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.top_k = top_k
        self.rank_penalty = rank_penalty
        self.temperature = temperature
        self.distill_weight = distill_weight
        self.hard_weight = hard_weight
        self.normalize_distill = normalize_distill
        self.rank_aware = RankAwareLoss(
            alpha=alpha,
            gamma=gamma,
            top_k=top_k,
            rank_penalty=rank_penalty,
        )

    def forward(self, student_logits, teacher_logits, targets):
        rank_aware_loss = self.rank_aware(student_logits, targets)

        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        with torch.no_grad():
            k = min(self.top_k, teacher_soft.size(1))
            _, topk_indices = torch.topk(teacher_soft, k=k, dim=1)
            mask = torch.zeros_like(teacher_soft)
            mask.scatter_(1, topk_indices, 1.0)
            teacher_soft_topk = teacher_soft * mask
            teacher_soft_topk = teacher_soft_topk / (teacher_soft_topk.sum(dim=1, keepdim=True) + 1e-8)

        distill_loss = F.kl_div(student_soft, teacher_soft_topk, reduction='batchmean')
        distill_loss = distill_loss * (self.temperature ** 2)
        raw_distill_loss = distill_loss
        if self.normalize_distill:
            with torch.no_grad():
                scale = rank_aware_loss.detach() / (distill_loss.detach() + 1e-8)
                scale = torch.clamp(scale, min=0.05, max=5.0)
            distill_loss = distill_loss * scale

        total_loss = self.hard_weight * rank_aware_loss + self.distill_weight * distill_loss
        return total_loss, {
            'rank_aware_loss': rank_aware_loss.item(),
            'distill_loss': distill_loss.item(),
            'raw_distill_loss': raw_distill_loss.item(),
            'total_loss': total_loss.item(),
        }


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class RankAwareLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, top_k=3, rank_penalty=0.2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.top_k = top_k
        self.rank_penalty = rank_penalty

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        with torch.no_grad():
            k = min(self.top_k, inputs.size(1))
            _, topk_indices = torch.topk(inputs, k=k, dim=1)
            targets_expanded = targets.unsqueeze(1)

            in_topk = (topk_indices == targets_expanded).any(dim=1)
            top1_correct = topk_indices[:, 0].eq(targets)
            in_topk_but_not_top1 = in_topk & (~top1_correct)

            weight = torch.where(
                in_topk_but_not_top1,
                torch.full_like(focal_loss, self.rank_penalty),
                torch.ones_like(focal_loss),
            )

        weighted_loss = focal_loss * weight
        return weighted_loss.mean()


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

        # 融合损失参数
        self.temperature = args.temperature
        self.distill_weight = args.distill_weight
        self.hard_weight = args.hard_weight
        self.ema_decay = args.ema_decay
        self.normalize_distill = args.normalize_distill

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
            df = pd.read_excel(self.config.data_path)

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
        except ValueError:
            print("警告：无法分层抽样，转为随机抽样")
            indices = np.arange(len(df))
            train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=self.config.seed)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # 只用训练集计算类别权重，避免验证集信息泄露。
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
        print(f"类别权重已生成 (前5项: {self.config.class_weights[:5]})")

        train_tfidf = None
        val_tfidf = None
        if self.config.use_tfidf and self.tfidf_vectorizer is not None:
            print("正在仅用训练集拟合 TF-IDF 向量器 (jieba分词)...")
            train_texts = train_df["full_text"].tolist()
            val_texts = val_df["full_text"].tolist()
            self.tfidf_vectorizer.fit(train_texts)
            train_tfidf = self.tfidf_vectorizer.transform(train_texts).toarray()
            val_tfidf = self.tfidf_vectorizer.transform(val_texts).toarray()
            actual_dim = int(train_tfidf.shape[1])
            self.config.tfidf_dim = actual_dim
            print(f"TF-IDF 特征维度: {self.config.tfidf_dim}，实际输出维度: {actual_dim}")

        return train_df, val_df, train_tfidf, val_tfidf


class DataAugmenter:
    def __init__(self, delete_prob=0.15):
        self.delete_prob = delete_prob

    def random_delete(self, words):
        if len(words) < 2:
            return words
        new_words = [w for w in words if random.random() > self.delete_prob]
        return new_words if len(new_words) > 0 else words

    def random_swap(self, words, n=1):
        if len(words) < 2:
            return words
        new_words = words.copy()
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        return new_words

    def augment(self, text):
        words = jieba.lcut(text)
        words = self.random_swap(words)
        words = self.random_delete(words)
        return "".join(words)


class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, tfidf_data=None, augment=False):
        self.texts = dataframe["full_text"].tolist()
        self.labels = dataframe["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tfidf_data = tfidf_data
        self.augment = augment
        self.augmenter = DataAugmenter(delete_prob=0.15) if augment else None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        if self.augment and self.augmenter is not None:
            text = self.augmenter.augment(text)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        result = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

        if self.tfidf_data is not None:
            result["tfidf_vec"] = torch.tensor(self.tfidf_data[idx], dtype=torch.float32)

        return result


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.w = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states, attention_mask):
        energy = torch.tanh(self.w(hidden_states))
        scores = self.v(energy).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1)
        context = torch.sum(hidden_states * attn_weights, dim=1)
        return context


class BertCNNAttention(nn.Module):
    def __init__(self, config):
        super(BertCNNAttention, self).__init__()
        self.bert = AutoModel.from_pretrained(config.model_name)

        hidden_size = self.bert.config.hidden_size

        self.filter_sizes = [2, 3, 4]
        self.num_filters = 256
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size,
                      out_channels=self.num_filters,
                      kernel_size=k)
            for k in self.filter_sizes
        ])

        self.attention = Attention(hidden_size)

        self.use_tfidf = config.use_tfidf
        if self.use_tfidf:
            self.tfidf_net = nn.Sequential(
                nn.Linear(config.tfidf_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, config.tfidf_hidden),
                nn.LayerNorm(config.tfidf_hidden),
                nn.ReLU()
            )

        self.dropout = nn.Dropout(0.3)

        fusion_dim = (self.num_filters * len(self.filter_sizes)) + hidden_size
        if self.use_tfidf:
            fusion_dim += config.tfidf_hidden
        self.fc = nn.Linear(fusion_dim, config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids, attention_mask, tfidf_vec=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        encoder_out = outputs.last_hidden_state

        cnn_input = encoder_out.permute(0, 2, 1)
        cnn_out = torch.cat([self.conv_and_pool(cnn_input, conv) for conv in self.convs], 1)

        attn_out = self.attention(encoder_out, attention_mask)

        if self.use_tfidf and tfidf_vec is not None:
            tfidf_out = self.tfidf_net(tfidf_vec)
            combined = torch.cat([cnn_out, attn_out, tfidf_out], dim=1)
        else:
            combined = torch.cat([cnn_out, attn_out], dim=1)

        out = self.dropout(combined)
        logits = self.fc(out)

        return logits


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
        if loss_name == "hybrid":
            return RankAwareDistillationLoss(
                alpha=self.class_weights,
                gamma=self.config.focal_gamma,
                top_k=self.config.top_k_eval,
                rank_penalty=self.config.rank_penalty,
                temperature=self.config.temperature,
                distill_weight=self.config.distill_weight,
                hard_weight=self.config.hard_weight,
                normalize_distill=self.config.normalize_distill,
            )
        return nn.CrossEntropyLoss(weight=self.class_weights)

    def _set_epoch_loss(self, epoch: int) -> None:
        active_loss = self.config.loss_type

        if self.config.loss_type == "focal" and epoch < self.config.focal_warmup_epochs:
            active_loss = "cross_entropy"

        if self.config.loss_type == "hybrid" and epoch < self.config.focal_warmup_epochs:
            active_loss = "rank_aware"

        self.active_loss_name = active_loss
        self.criterion = self._build_loss(active_loss)
        if active_loss == "focal":
            print(f">>> [System] 当前损失: FocalLoss(gamma={self.config.focal_gamma})")
        elif active_loss == "rank_aware":
            warmup_note = "，hybrid预热阶段" if self.config.loss_type == "hybrid" else ""
            print(
                f">>> [System] 当前损失: RankAwareLoss(gamma={self.config.focal_gamma}, "
                f"rank_penalty={self.config.rank_penalty}{warmup_note})"
            )
        elif active_loss == "hybrid":
            print(
                f">>> [System] 当前损失: RankAwareDistillationLoss("
                f"gamma={self.config.focal_gamma}, rank_penalty={self.config.rank_penalty}, "
                f"temperature={self.config.temperature}, top_k={self.config.top_k_eval}, "
                f"hard_weight={self.config.hard_weight}, distill_weight={self.config.distill_weight}, "
                f"normalize_distill={self.config.normalize_distill})"
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
        _atomic_json_save(self.config.id2label, self.config.map_save_path)
        _atomic_json_save(
            {
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "best_metric_name": self.config.best_metric,
                "best_score": float(best_score),
                "num_classes": int(self.config.num_classes or 0),
                "architecture": "bert_cnn_attention_tfidf" if self.config.use_cnn_attention else "legacy_bert_sequence",
                "classifier_route": "hybrid_rankaware_distill" if self.config.use_cnn_attention else "legacy_bert",
                "use_cnn_attention": bool(self.config.use_cnn_attention),
                "use_tfidf": bool(self.config.use_tfidf),
                "tfidf_dim": int(self.config.tfidf_dim),
                "tfidf_hidden": int(self.config.tfidf_hidden),
                "tfidf_vectorizer": "tfidf_vectorizer.joblib" if self.config.use_tfidf else "",
                "label_signature": self.config.label_signature,
                "loss_type": self.config.loss_type,
                "focal_gamma": float(self.config.focal_gamma),
                "temperature": float(self.config.temperature),
                "distill_weight": float(self.config.distill_weight),
                "hard_weight": float(self.config.hard_weight),
                "normalize_distill": bool(self.config.normalize_distill),
                "rank_penalty": float(self.config.rank_penalty),
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
                tfidf_vec = batch.get("tfidf_vec")
                if tfidf_vec is not None:
                    tfidf_vec = tfidf_vec.to(self.config.device)

                # 学生模型前向传播
                student_logits = self._get_logits(input_ids, attention_mask, tfidf_vec)

                # 计算损失
                if self.active_loss_name == "hybrid":
                    # 使用EMA模型作为教师模型
                    self.ema_model.apply_shadow()
                    with torch.no_grad():
                        teacher_logits = self._get_logits(input_ids, attention_mask, tfidf_vec)
                    self.ema_model.restore()

                    loss, loss_dict = self.criterion(student_logits, teacher_logits, labels)
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

                    if self.active_loss_name == "hybrid":
                        self.ema_model.apply_shadow()
                        with torch.no_grad():
                            teacher_logits_adv = self._get_logits(input_ids, attention_mask, tfidf_vec)
                        self.ema_model.restore()
                        loss_adv, _ = self.criterion(student_logits_adv, teacher_logits_adv, labels)
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

        eval_loss_name = "rank_aware" if self.config.loss_type == "hybrid" else self.config.loss_type
        eval_criterion = self._build_loss(eval_loss_name)

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="评估中", leave=False):
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)
                tfidf_vec = batch.get("tfidf_vec")
                if tfidf_vec is not None:
                    tfidf_vec = tfidf_vec.to(self.config.device)

                logits = self._get_logits(input_ids, attention_mask, tfidf_vec)
                loss = eval_criterion(logits, labels)

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
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        return {
            "loss": avg_loss,
            "avg_loss": avg_loss,
            "top1_acc": top1_acc,
            "topk_acc": topk_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
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
    parser = argparse.ArgumentParser(description="训练官方回复单位分类模型 - 融合版本(RankAware + 自蒸馏)")
    parser.add_argument("--model-name", default=r"C:\python\接诉即办\.venv\local_roberta_model")
    parser.add_argument("--data-path", default=get_default_classifier_data_path())
    parser.add_argument("--save-dir", default=r"C:\Users\28414\PycharmProjects\接诉即办项目\final_model_hybrid")
    parser.add_argument("--report-dir", default=str(get_training_reports_dir()))
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-eval", type=int, default=3)
    parser.add_argument("--disable-fgm", action="store_true", default=True, help="禁用FGM对抗训练（默认禁用）")
    parser.add_argument("--enable-fgm", dest="disable_fgm", action="store_false", help="启用FGM对抗训练")
    parser.add_argument("--use-small-sample", action="store_true")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--train-from-scratch", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--init-from", default=r"C:\Users\28414\PycharmProjects\接诉即办项目\final_model_fgm\pytorch_model.bin", help="从指定路径加载预训练权重")
    parser.add_argument("--use-cnn-attention", action="store_true", default=True, help="使用 BERT+CNN+Attention 架构")
    parser.add_argument("--use-tfidf", action="store_true", default=True, help="启用 TF-IDF 特征分支")
    parser.add_argument("--no-tfidf", dest="use_tfidf", action="store_false", help="关闭 TF-IDF 特征分支")
    parser.add_argument("--tfidf-dim", type=int, default=3000, help="TF-IDF 特征维度")
    parser.add_argument("--loss-type", type=str, default="hybrid",
                        choices=['focal', 'rank_aware', 'cross_entropy', 'hybrid'],
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

    # 融合损失参数
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="蒸馏温度（越高分布越平滑）")
    parser.add_argument("--distill-weight", type=float, default=0.25,
                        help="蒸馏损失权重")
    parser.add_argument("--hard-weight", type=float, default=0.75,
                        help="硬标签损失权重")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA模型衰减率")
    parser.add_argument(
        "--no-normalize-distill",
        dest="normalize_distill",
        action="store_false",
        default=True,
        help="关闭KL量纲归一化（用于消融实验）",
    )

    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


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
    paths = write_acceptance_report(report, report_dir=config.report_dir, filename_prefix="classifier_hybrid_acceptance")
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
