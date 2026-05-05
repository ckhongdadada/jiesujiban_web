from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.jsjb.unit_classifier.model import BertCNNAttention, HybridClassifierConfig
from src.jsjb.core.artifacts import inspect_classifier_artifacts
from src.jsjb.unit_classifier.catalog import canonicalize_unit, load_unit_catalog, normalize_unit_text
from src.jsjb.unit_classifier.tokenization import _chinese_tokenizer

_cls_tokenizer = None
_cls_model = None
_id2label = None
_tfidf_vectorizer = None
_runtime_architecture = "legacy_bert_sequence"
_loaded_key = None
_cls_lock = threading.Lock()


def _normalize_label(value: str) -> str:
    return normalize_unit_text(value)


def _load_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_existing_path(*candidates: str) -> str:
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else ""


def _resolve_tfidf_path(model_dir: str, meta: dict[str, Any]) -> str:
    raw_path = str(meta.get("tfidf_vectorizer") or "tfidf_vectorizer.joblib")
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.join(model_dir, raw_path)


def _load_tfidf_vectorizer(path: str):
    # Older vectorizers were saved by scripts executed as __main__; expose the
    # expected tokenizer name before unpickling to keep those artifacts usable.
    setattr(sys.modules["__main__"], "_chinese_tokenizer", _chinese_tokenizer)
    return joblib.load(path)


def _detect_architecture(meta: dict[str, Any]) -> str:
    architecture = str(meta.get("architecture") or "").strip()
    if architecture:
        return architecture
    if meta.get("use_cnn_attention"):
        return "bert_cnn_attention_tfidf"
    return "legacy_bert_sequence"


def _build_classifier_text(tag: str, title: str, body: str) -> str:
    return f"\u3010{tag}\u3011{title}\u3002{body}"


def classifier_status(model_dir: str, base_model_dir: str = "") -> dict[str, Any]:
    status = inspect_classifier_artifacts(model_dir, base_model_dir)
    meta = _load_json(os.path.join(model_dir, "model_meta.json"))
    architecture = _detect_architecture(meta)
    status["model_meta"] = {
        "path": os.path.join(model_dir, "model_meta.json"),
        "exists": bool(meta),
        "architecture": architecture,
        "classifier_route": meta.get("classifier_route", "legacy_bert" if architecture.startswith("legacy") else "hybrid"),
        "use_tfidf": bool(meta.get("use_tfidf", False)),
    }
    if status["model_meta"]["use_tfidf"]:
        tfidf_path = _resolve_tfidf_path(model_dir, meta)
        status["tfidf_vectorizer"] = {
            "path": tfidf_path,
            "exists": os.path.exists(tfidf_path),
        }
    return status


def load_classifier_runtime(model_dir: str, base_model_dir: str, label_map_path: str, device: torch.device) -> bool:
    global _cls_tokenizer, _cls_model, _id2label, _tfidf_vectorizer, _runtime_architecture, _loaded_key

    normalized_key = (os.path.abspath(model_dir), os.path.abspath(base_model_dir or ""), str(device))
    with _cls_lock:
        if _cls_model is not None and _loaded_key == normalized_key:
            return True

        status = inspect_classifier_artifacts(model_dir, base_model_dir)
        if not status["compatible_runtime_ready"]:
            return False

        meta = _load_json(os.path.join(model_dir, "model_meta.json"))
        architecture = _detect_architecture(meta)
        use_hybrid = architecture in {"bert_cnn_attention", "bert_cnn_attention_tfidf"}
        use_tfidf = bool(meta.get("use_tfidf", use_hybrid and architecture.endswith("_tfidf")))

        catalog = load_unit_catalog()
        with open(label_map_path, "r", encoding="utf-8") as fp:
            raw_labels = json.load(fp)
            _id2label = {key: canonicalize_unit(_normalize_label(value), catalog) for key, value in raw_labels.items()}
        num_classes = len(_id2label)

        tokenizer_source = status["tokenizer_source"] or model_dir
        _cls_tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        weights_path = status["weights"]["path"]
        if use_hybrid:
            model_source = _resolve_existing_path(
                base_model_dir,
                str(meta.get("base_model") or ""),
                model_dir,
            )
            if not model_source:
                return False
            hybrid_config = HybridClassifierConfig(
                model_name=model_source,
                num_classes=num_classes,
                use_tfidf=use_tfidf,
                tfidf_dim=int(meta.get("tfidf_dim", 3000)),
                tfidf_hidden=int(meta.get("tfidf_hidden", 64)),
            )
            _cls_model = BertCNNAttention(hybrid_config).to(device)
            if use_tfidf:
                tfidf_path = _resolve_tfidf_path(model_dir, meta)
                if not os.path.exists(tfidf_path):
                    return False
                _tfidf_vectorizer = _load_tfidf_vectorizer(tfidf_path)
            else:
                _tfidf_vectorizer = None
        else:
            model_source = model_dir if status["config"]["exists"] else base_model_dir
            _cls_model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            ).to(device)
            _tfidf_vectorizer = None

        _cls_model.load_state_dict(torch.load(weights_path, map_location=device))
        _cls_model.eval()
        _runtime_architecture = architecture
        _loaded_key = normalized_key
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
    district: str = "",
) -> list[dict[str, Any]]:
    if not load_classifier_runtime(model_dir, base_model_dir, label_map_path, device):
        return []

    text = _build_classifier_text(tag, title, body)
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
        if _runtime_architecture in {"bert_cnn_attention", "bert_cnn_attention_tfidf"}:
            tfidf_vec = None
            if _tfidf_vectorizer is not None:
                tfidf_arr = _tfidf_vectorizer.transform([text]).toarray()
                tfidf_vec = torch.tensor(tfidf_arr, dtype=torch.float32, device=device)
            logits = _cls_model(input_ids, attention_mask, tfidf_vec)
        else:
            logits = _cls_model(input_ids, attention_mask=attention_mask).logits

    probs = torch.softmax(logits, dim=-1)[0]
    candidate_k = min(max(top_k * 4, 12), len(_id2label))
    values, indices = torch.topk(probs, k=candidate_k)
    catalog = load_unit_catalog()
    reranked = []
    seen = set()

    for value, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
        unit = canonicalize_unit(_id2label[str(index)], catalog)
        if not unit or unit in seen:
            continue
        seen.add(unit)
        reranked.append(
            {
                "unit": unit,
                "confidence": round(value * 100, 1),
            }
        )

    reranked.sort(key=lambda item: (-item["confidence"], item["unit"]))
    return reranked[:top_k]


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

    def predict(self, tag: str, title: str, body: str, top_k: int = 3, district: str = "") -> list[dict[str, Any]]:
        return predict_units(
            tag,
            title,
            body,
            self.model_dir,
            self.base_model_dir,
            self.label_map_path,
            self.device,
            top_k,
            district,
        )
