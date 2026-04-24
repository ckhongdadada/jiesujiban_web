from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


@dataclass
class HybridClassifierConfig:
    model_name: str
    num_classes: int
    use_tfidf: bool = True
    tfidf_dim: int = 3000
    tfidf_hidden: int = 64


class Attention(nn.Module):
    """Token-level attention over BERT/RoBERTa contextual states."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.w = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        energy = torch.tanh(self.w(hidden_states))
        scores = self.v(energy).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(hidden_states * attn_weights, dim=1)


class BertCNNAttention(nn.Module):
    """BERT/RoBERTa encoder + TextCNN + Attention + optional TF-IDF fusion classifier."""

    def __init__(self, config: HybridClassifierConfig):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.model_name)
        hidden_size = self.bert.config.hidden_size

        self.filter_sizes = [2, 3, 4]
        self.num_filters = 256
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=self.num_filters,
                    kernel_size=kernel_size,
                )
                for kernel_size in self.filter_sizes
            ]
        )
        self.attention = Attention(hidden_size)

        self.use_tfidf = config.use_tfidf
        if self.use_tfidf:
            self.tfidf_net = nn.Sequential(
                nn.Linear(config.tfidf_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, config.tfidf_hidden),
                nn.LayerNorm(config.tfidf_hidden),
                nn.ReLU(),
            )

        self.dropout = nn.Dropout(0.3)
        fusion_dim = (self.num_filters * len(self.filter_sizes)) + hidden_size
        if self.use_tfidf:
            fusion_dim += config.tfidf_hidden
        self.fc = nn.Linear(fusion_dim, config.num_classes)

    @staticmethod
    def conv_and_pool(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        x = F.relu(conv(x))
        return F.max_pool1d(x, x.size(2)).squeeze(2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tfidf_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        encoder_out = outputs.last_hidden_state

        cnn_input = encoder_out.permute(0, 2, 1)
        cnn_out = torch.cat([self.conv_and_pool(cnn_input, conv) for conv in self.convs], dim=1)
        attn_out = self.attention(encoder_out, attention_mask)

        if self.use_tfidf and tfidf_vec is not None:
            tfidf_out = self.tfidf_net(tfidf_vec)
            combined = torch.cat([cnn_out, attn_out, tfidf_out], dim=1)
        else:
            combined = torch.cat([cnn_out, attn_out], dim=1)

        return self.fc(self.dropout(combined))
