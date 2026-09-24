"""The Tacet network: an mmBERT encoder, a two-layer transformer head and an option scorer.

The head and scorer have the layout of Laya's DecisionModel
(https://huggingface.co/convaiinnovations/laya, Apache 2.0, see NOTICE), so their weights
load by the same parameter names. Laya's single question-type embedding is replaced by a
per-token segment embedding, because a packed sequence holds several questions of
different types and the state.
"""

import torch
from torch import nn

from .packing import SEGMENT_COUNT

# Padded option slots get this logit so they take no probability after the softmax.
MASKED_LOGIT = -1e4


def build_head(hidden_size, layer_count):
    head_count = max(1, hidden_size // 64)
    layer = nn.TransformerEncoderLayer(hidden_size, head_count, 4 * hidden_size, dropout=0.1,
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layer_count, enable_nested_tensor=False)


def build_scorer(hidden_size):
    return nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.GELU(),
                         nn.Linear(hidden_size, 1))


class TacetNetwork(nn.Module):
    def __init__(self, encoder, head_layers=2):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.head = build_head(hidden_size, head_layers) if head_layers > 0 else None
        self.scorer = build_scorer(hidden_size)
        self.segment_embedding = nn.Embedding(SEGMENT_COUNT, hidden_size)

    def forward(self, input_ids, attention_mask, segment_ids, marker_positions, marker_mask):
        """Logits shaped [batch, questions, options], masked where there is no option."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden + self.segment_embedding(segment_ids)
        if self.head is not None:
            padding = ~attention_mask.bool()
            for layer in self.head.layers:
                hidden = layer(hidden, src_key_padding_mask=padding)

        batch_size, question_count, option_count = marker_positions.shape
        flat_positions = marker_positions.clamp(min=0).reshape(batch_size, question_count * option_count)
        gather_index = flat_positions[:, :, None].expand(-1, -1, hidden.size(-1))
        at_markers = torch.gather(hidden, 1, gather_index)
        logits = self.scorer(at_markers).squeeze(-1).float()
        logits = logits.reshape(batch_size, question_count, option_count)
        return logits.masked_fill(~marker_mask, MASKED_LOGIT)
