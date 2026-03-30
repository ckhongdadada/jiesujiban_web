from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enhancements.training_acceptance import build_acceptance_report, format_acceptance_report, write_acceptance_report


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class Config:
    def __init__(self, args: argparse.Namespace):
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.save_dir = args.save_dir
        self.report_dir = args.report_dir
        self.train_from_scratch = args.train_from_scratch
        self.no_resume = args.no_resume
        os.makedirs(self.save_dir, exist_ok=True)

        self.model_save_path = os.path.join(self.save_dir, "pytorch_model.bin")
        self.map_save_path = os.path.join(self.save_dir, "label_map.json")
        self.training_state_path = os.path.join(self.save_dir, "training_state.pt")
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = args.seed
        self.top_k_eval = args.top_k_eval
        self.use_fgm = not args.disable_fgm
        self.use_small_sample = args.use_small_sample
        self.sample_size = args.sample_size
        self.num_classes = None
        self.id2label = None

    def set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


class DataProcessor:
    REQUIRED_COLUMNS = ["留言标签", "留言标题", "留言正文", "官方回复单位"]

    def __init__(self, config: Config):
        self.config = config

    def load_and_process(self):
        print(f"正在加载数据: {self.config.data_path} ...")
        df = pd.read_excel(self.config.data_path)
        missing = [column for column in self.REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"分类数据缺少必要列: {missing}")

        target_col = "官方回复单位"
        input_label_col = "留言标签"
        df.dropna(subset=[target_col, input_label_col, "留言标题", "留言正文"], inplace=True)

        if self.config.use_small_sample and len(df) > self.config.sample_size:
            df = df.sample(n=self.config.sample_size, random_state=self.config.seed)
            print(f"警告：正在使用小样本模式 ({self.config.sample_size}条)")
        else:
            print(f"全量模式：共 {len(df)} 条数据")

        df["留言标题"] = df["留言标题"].fillna("")
        df["留言正文"] = df["留言正文"].fillna("")
        df["full_text"] = (
            "【" + df[input_label_col].astype(str) + "】"
            + df["留言标题"].astype(str)
            + "。"
            + df["留言正文"].astype(str)
        )

        unique_units = sorted(df[target_col].unique())
        self.config.num_classes = len(unique_units)
        self.config.id2label = {str(idx): str(label) for idx, label in enumerate(unique_units)}
        label2id = {label: idx for idx, label in enumerate(unique_units)}

        with open(self.config.map_save_path, "w", encoding="utf-8") as fp:
            json.dump(self.config.id2label, fp, ensure_ascii=False, indent=2)
        print(f"标签字典已保存至: {self.config.map_save_path}")

        df["label_id"] = df[target_col].map(label2id)
        print(f"分类数量: {self.config.num_classes}")

        try:
            train_df, val_df = train_test_split(
                df,
                test_size=0.2,
                random_state=self.config.seed,
                stratify=df["label_id"],
            )
        except ValueError:
            print("警告：无法分层抽样，转为随机抽样")
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=self.config.seed)
        return train_df, val_df


class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.texts = dataframe["full_text"].tolist()
        self.labels = dataframe["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BertClassifier:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_classes,
            ignore_mismatched_sizes=True,
        ).to(config.device)
        self.criterion = nn.CrossEntropyLoss()
        self.fgm = FGM(self.model) if config.use_fgm else None
        if self.fgm is not None:
            print(">>> [System] FGM 对抗训练已启用")

    def _save_best_artifacts(self):
        torch.save(self.model.state_dict(), self.config.model_save_path)
        self.model.config.save_pretrained(self.config.save_dir)
        self.tokenizer.save_pretrained(self.config.save_dir)

    def _save_training_state(self, epoch: int, best_acc: float, optimizer, scheduler):
        torch.save(
            {
                "epoch": epoch,
                "best_acc": best_acc,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            self.config.training_state_path,
        )

    def _maybe_resume(self, optimizer, scheduler) -> tuple[int, float]:
        if self.config.train_from_scratch or self.config.no_resume:
            return 0, 0.0
        if not os.path.exists(self.config.training_state_path):
            return 0, 0.0

        checkpoint = torch.load(self.config.training_state_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_acc = float(checkpoint.get("best_acc", 0.0))
        print(f"检测到分类训练断点，将从 epoch {start_epoch + 1}/{self.config.epochs} 继续训练")
        return start_epoch, best_acc

    def train(self, train_loader, val_loader):
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        total_steps = len(train_loader) * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        start_epoch, best_acc = self._maybe_resume(optimizer, scheduler)
        print("\n开始训练...")
        for epoch in range(start_epoch, self.config.epochs):
            self.model.train()
            total_train_loss = 0.0
            loop = tqdm(train_loader, total=len(train_loader), leave=True)
            loop.set_description(f"Epoch [{epoch + 1}/{self.config.epochs}]")

            for batch in loop:
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs.logits, labels)
                loss.backward()

                if self.fgm is not None:
                    self.fgm.attack()
                    outputs_adv = self.model(input_ids, attention_mask=attention_mask)
                    loss_adv = self.criterion(outputs_adv.logits, labels)
                    loss_adv.backward()
                    self.fgm.restore()

                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_train_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_train_loss = total_train_loss / len(train_loader)
            val_acc = self.evaluate(val_loader)
            print(f"Epoch {epoch + 1} | Avg Loss: {avg_train_loss:.4f} | Val Top-{self.config.top_k_eval} Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                self._save_best_artifacts()
                print(f"  --> 保存最佳模型 (Acc: {best_acc:.4f})")

            self._save_training_state(epoch, best_acc, optimizer, scheduler)

    def evaluate(self, data_loader):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        k = min(self.config.top_k_eval, self.config.num_classes)

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="评估中", leave=False):
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)
                logits = self.model(input_ids, attention_mask=attention_mask).logits
                _, topk_indices = torch.topk(logits, k=k, dim=1)
                labels_expanded = labels.view(-1, 1)
                total_correct += torch.sum(topk_indices == labels_expanded).item()
                total_samples += labels.size(0)

        return total_correct / total_samples if total_samples > 0 else 0

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
                preds = torch.argmax(self.model(input_ids, attention_mask=attention_mask).logits, dim=1)

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

        print("\n" + "=" * 50)
        print("最终模型评估报告")
        print("=" * 50)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练官方回复单位分类模型")
    parser.add_argument("--model-name", default=r"C:\python\接诉即办\.venv\local_roberta_model")
    parser.add_argument("--data-path", default=r"C:\Users\28414\Desktop\留言板合并数据.xlsx")
    parser.add_argument("--save-dir", default=r"C:\Users\28414\PycharmProjects\接诉即办项目\final_model_fgm")
    parser.add_argument("--report-dir", default=r"C:\Users\28414\PycharmProjects\接诉即办项目\data\training_reports")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-eval", type=int, default=3)
    parser.add_argument("--disable-fgm", action="store_true")
    parser.add_argument("--use-small-sample", action="store_true")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--train-from-scratch", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(args)
    config.set_seed()

    if not os.path.exists(config.model_name):
        raise FileNotFoundError(f"未找到基础分类模型目录: {config.model_name}")
    if not os.path.exists(config.data_path):
        raise FileNotFoundError(f"未找到分类训练数据文件: {config.data_path}")

    processor = DataProcessor(config)
    train_df, val_df = processor.load_and_process()
    classifier = BertClassifier(config)

    train_dataset = TextDataset(train_df, classifier.tokenizer, config.max_len)
    val_dataset = TextDataset(val_df, classifier.tokenizer, config.max_len)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    classifier.train(train_loader, val_loader)
    classifier.full_report(val_loader)

    report = build_acceptance_report(
        target="classifier",
        classifier_model_dir=config.save_dir,
        classifier_base_model=config.model_name,
    )
    paths = write_acceptance_report(report, report_dir=config.report_dir, filename_prefix="classifier_acceptance")
    print("\n=== 训练产物验收 ===")
    print(format_acceptance_report(report))
    print(f"验收报告: {paths['latest']}")
    print(f"归档报告: {paths['archived']}")


if __name__ == "__main__":
    main()
