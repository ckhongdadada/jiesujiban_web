from __future__ import annotations

import json
import threading
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from enhancements.model_artifacts import inspect_classifier_artifacts

_cls_tokenizer = None
_cls_model = None
_id2label = None
_cls_lock = threading.Lock()


def _normalize_label(value: str) -> str:
    return " ".join(str(value).replace("\u3000", " ").split())


def classifier_status(model_dir: str, base_model_dir: str = "") -> dict[str, Any]:
    return inspect_classifier_artifacts(model_dir, base_model_dir)


def load_classifier_runtime(model_dir: str, base_model_dir: str, label_map_path: str, device: torch.device) -> bool:
    global _cls_tokenizer, _cls_model, _id2label

    with _cls_lock:
        if _cls_model is not None:
            return True

        status = inspect_classifier_artifacts(model_dir, base_model_dir)
        if not status["compatible_runtime_ready"]:
            return False

        with open(label_map_path, "r", encoding="utf-8") as fp:
            raw_labels = json.load(fp)
            _id2label = {key: _normalize_label(value) for key, value in raw_labels.items()}
        num_classes = len(_id2label)

        tokenizer_source = status["tokenizer_source"] or model_dir
        model_source = model_dir if status["config"]["exists"] else base_model_dir

        _cls_tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        _cls_model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        ).to(device)
        _cls_model.load_state_dict(torch.load(status["weights"]["path"], map_location=device))
        _cls_model.eval()
        return True


def predict_units(
    tag: str,
    title: str,
    body: str,
    model_dir: str,
    base_model_dir: str,
    label_map_path: str,
    device: torch.device,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    if not load_classifier_runtime(model_dir, base_model_dir, label_map_path, device):
        return []

    text = f"【{tag}】{title}。{body}"
    encoding = _cls_tokenizer(
        text,
        add_special_tokens=True,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = _cls_model(input_ids, attention_mask=attention_mask).logits

    probs = torch.softmax(logits, dim=-1)[0]
    k = min(top_k, len(_id2label))
    values, indices = torch.topk(probs, k=k)
    results = []
    for value, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
        results.append(
            {
                "unit": _id2label[str(index)],
                "confidence": round(value * 100, 1),
            }
        )
    return results


class ClassifierRuntime:
    def __init__(self, model_dir: str, base_model_dir: str = "", device: str = "cpu"):
        self.model_dir = model_dir
        self.base_model_dir = base_model_dir or model_dir
        self.device = torch.device(device)
        self.label_map_path = f"{model_dir}/label_map.json"
        self._model = None

    @property
    def model(self):
        return _cls_model

    def predict(self, tag: str, title: str, body: str, top_k: int = 3) -> list[dict[str, Any]]:
        return predict_units(
            tag, title, body,
            self.model_dir,
            self.base_model_dir,
            self.label_map_path,
            self.device,
            top_k
        )
