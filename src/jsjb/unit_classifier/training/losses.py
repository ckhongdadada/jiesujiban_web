from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.jsjb.unit_classifier.catalog import canonicalize_unit, load_unit_catalog, normalize_unit_text

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

class SoftLabelBank:
    """EMA historical soft-label cache for each training sample."""

    def __init__(self, num_samples: int, num_classes: int, momentum: float = 0.9, top_k: int = 5):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.momentum = momentum
        self.top_k = top_k
        self.probs = torch.zeros(num_samples, num_classes, dtype=torch.float16)
        self.initialized = torch.zeros(num_samples, dtype=torch.bool)

    def _truncate_topk(self, probs: torch.Tensor) -> torch.Tensor:
        k = min(self.top_k, probs.size(1))
        _, topk_indices = torch.topk(probs, k=k, dim=1)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, topk_indices, 1.0)
        probs = probs * mask
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
        return probs

    def update(self, indices: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
        indices = indices.detach().cpu().long()
        teacher_probs = self._truncate_topk(teacher_probs.detach().cpu().float())
        old_probs = self.probs[indices].float()
        initialized = self.initialized[indices].unsqueeze(1)
        blended = torch.where(
            initialized,
            self.momentum * old_probs + (1.0 - self.momentum) * teacher_probs,
            teacher_probs,
        )
        blended = blended / (blended.sum(dim=1, keepdim=True) + 1e-8)
        self.probs[indices] = blended.to(dtype=torch.float16)
        self.initialized[indices] = True
        return blended

    def get(self, indices: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        indices = indices.detach().cpu().long()
        return self.probs[indices].to(device=device, dtype=dtype)

    def state_dict(self) -> dict:
        return {
            "num_samples": self.num_samples,
            "num_classes": self.num_classes,
            "momentum": self.momentum,
            "top_k": self.top_k,
            "probs": self.probs,
            "initialized": self.initialized,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("num_samples", self.num_samples)) != self.num_samples:
            raise ValueError("soft label bank sample count mismatch")
        if int(state.get("num_classes", self.num_classes)) != self.num_classes:
            raise ValueError("soft label bank class count mismatch")
        self.momentum = float(state.get("momentum", self.momentum))
        self.top_k = int(state.get("top_k", self.top_k))
        self.probs = state["probs"].cpu().to(dtype=torch.float16)
        self.initialized = state["initialized"].cpu().bool()


def _char_bigram_jaccard(text_a: str, text_b: str) -> float:
    def bigrams(text: str) -> set[str]:
        text = normalize_unit_text(text)
        if not text:
            return set()
        if len(text) == 1:
            return {text}
        return {text[i : i + 2] for i in range(len(text) - 1)}

    a = bigrams(text_a)
    b = bigrams(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def build_unit_prior_matrix(
    id2label: dict[str, str],
    prior_top_k: int = 8,
    self_score: float = 1.0,
) -> torch.Tensor:
    """Build a row-normalized unit similarity prior from unit_catalog metadata."""

    catalog = load_unit_catalog()
    unit_meta = catalog.get("unit_meta", {}) if isinstance(catalog, dict) else {}
    labels = [normalize_unit_text(id2label[str(i)]) for i in range(len(id2label))]
    num_classes = len(labels)
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.float32)

    for i, unit_i in enumerate(labels):
        meta_i = unit_meta.get(canonicalize_unit(unit_i, catalog), {}) or {}
        districts_i = set(meta_i.get("districts", []) or [])
        tags_i = set(meta_i.get("top_tags", []) or [])
        category_i = str(meta_i.get("category", "") or "")

        scores: list[tuple[float, int]] = []
        for j, unit_j in enumerate(labels):
            if i == j:
                scores.append((self_score, j))
                continue

            meta_j = unit_meta.get(canonicalize_unit(unit_j, catalog), {}) or {}
            districts_j = set(meta_j.get("districts", []) or [])
            tags_j = set(meta_j.get("top_tags", []) or [])
            category_j = str(meta_j.get("category", "") or "")

            score = 0.0
            if districts_i and districts_j and districts_i & districts_j:
                score += 0.35
            if category_i and category_i == category_j:
                score += 0.25
            if tags_i and tags_j:
                score += min(0.25, 0.08 * len(tags_i & tags_j))
            name_sim = _char_bigram_jaccard(unit_i, unit_j)
            if name_sim >= 0.35:
                score += 0.35 * name_sim
            if unit_i and unit_j and (unit_i in unit_j or unit_j in unit_i):
                score += 0.12

            if score > 0:
                scores.append((score, j))

        scores = sorted(scores, key=lambda item: (-item[0], item[1]))[: max(prior_top_k, 1)]
        total = sum(score for score, _ in scores)
        if total <= 0:
            matrix[i, i] = 1.0
        else:
            for score, j in scores:
                matrix[i, j] = float(score / total)

    return matrix


class SelfDistillationLoss(nn.Module):
    """Self-distillation loss with RankAware hard loss and optional EMA soft-label bank targets."""

    def __init__(
        self,
        alpha=None,
        gamma=1.5,
        rank_top_k=3,
        rank_penalty=0.15,
        temperature=3.0,
        top_k=5,
        hard_weight=0.82,
        distill_weight=0.18,
        normalize_distill=True,
    ):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.top_k = top_k
        self.hard_weight = hard_weight
        self.distill_weight = distill_weight
        self.normalize_distill = normalize_distill
        self.rank_aware = RankAwareLoss(
            alpha=alpha,
            gamma=gamma,
            top_k=rank_top_k,
            rank_penalty=rank_penalty,
        )

    def build_soft_targets(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        with torch.no_grad():
            k = min(self.top_k, teacher_soft.size(1))
            _, topk_indices = torch.topk(teacher_soft, k=k, dim=1)
            mask = torch.zeros_like(teacher_soft)
            mask.scatter_(1, topk_indices, 1.0)
            teacher_soft_topk = teacher_soft * mask
            teacher_soft_topk = teacher_soft_topk / (teacher_soft_topk.sum(dim=1, keepdim=True) + 1e-8)
        return teacher_soft_topk

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        targets: torch.Tensor,
        teacher_probs: torch.Tensor | None = None,
    ):
        # Keep the Top-K objective from the FGM/RankAware route so distillation does not sacrifice Top-3 accuracy.
        hard_loss = self.rank_aware(student_logits, targets)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)

        if teacher_probs is None:
            if teacher_logits is None:
                raise ValueError("teacher_logits or teacher_probs is required for distillation")
            teacher_soft_topk = self.build_soft_targets(teacher_logits)
        else:
            teacher_soft_topk = teacher_probs

        distill_loss = F.kl_div(student_soft, teacher_soft_topk, reduction="batchmean")
        distill_loss = distill_loss * (self.temperature ** 2)
        raw_distill_loss = distill_loss
        if self.normalize_distill:
            with torch.no_grad():
                scale = hard_loss.detach() / (distill_loss.detach() + 1e-8)
                scale = torch.clamp(scale, min=0.05, max=5.0)
            distill_loss = distill_loss * scale

        total_loss = self.hard_weight * hard_loss + self.distill_weight * distill_loss
        return total_loss, {
            "hard_loss": hard_loss.item(),
            "distill_loss": distill_loss.item(),
            "raw_distill_loss": raw_distill_loss.item(),
            "total_loss": total_loss.item(),
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
