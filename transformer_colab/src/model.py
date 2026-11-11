"""Custom PyTorch encoder-decoder Transformer and decoding helpers."""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_length: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        positions = torch.arange(max_length).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * divisor)
        encoding[:, 1::2] = torch.cos(positions * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.size(1) > self.encoding.size(1):
            raise ValueError(
                "Sequence length {} exceeds positional limit {}".format(
                    values.size(1), self.encoding.size(1)
                )
            )
        return self.dropout(values + self.encoding[:, : values.size(1)])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.d_model = d_model
        self.heads = heads
        self.head_dimension = d_model // heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, _ = values.shape
        return (
            values.view(batch, length, self.heads, self.head_dimension)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries = self._split_heads(self.query_projection(query))
        keys = self._split_heads(self.key_projection(key))
        values = self._split_heads(self.value_projection(value))
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dimension)

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min
            )
        if attention_mask is not None:
            scores = scores.masked_fill(
                attention_mask[None, None, :, :], torch.finfo(scores.dtype).min
            )

        attention = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attention, values)
        batch, _, length, _ = context.shape
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, length, self.d_model)
        )
        return self.output_projection(context)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, d_model),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, hidden_size: int, dropout: float):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, heads, dropout)
        self.feed_forward = FeedForward(d_model, hidden_size, dropout)
        self.norm_attention = nn.LayerNorm(d_model)
        self.norm_feed_forward = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, values: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        attended = self.self_attention(
            values, values, values, key_padding_mask=padding_mask
        )
        values = self.norm_attention(values + self.dropout(attended))
        transformed = self.feed_forward(values)
        return self.norm_feed_forward(values + self.dropout(transformed))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, hidden_size: int, dropout: float):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, heads, dropout)
        self.source_attention = MultiHeadAttention(d_model, heads, dropout)
        self.feed_forward = FeedForward(d_model, hidden_size, dropout)
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_source = nn.LayerNorm(d_model)
        self.norm_feed_forward = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        memory: torch.Tensor,
        target_padding: torch.Tensor,
        source_padding: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended = self.self_attention(
            values,
            values,
            values,
            key_padding_mask=target_padding,
            attention_mask=causal_mask,
        )
        values = self.norm_self(values + self.dropout(attended))
        attended_source = self.source_attention(
            values, memory, memory, key_padding_mask=source_padding
        )
        values = self.norm_source(values + self.dropout(attended_source))
        transformed = self.feed_forward(values)
        return self.norm_feed_forward(values + self.dropout(transformed))


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        source_pad_id: int,
        target_pad_id: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_position_embeddings: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.source_pad_id = source_pad_id
        self.target_pad_id = target_pad_id
        self.source_embedding = nn.Embedding(
            source_vocab_size, d_model, padding_idx=source_pad_id
        )
        self.target_embedding = nn.Embedding(
            target_vocab_size, d_model, padding_idx=target_pad_id
        )
        self.source_position = PositionalEncoding(
            d_model, dropout, max_position_embeddings
        )
        self.target_position = PositionalEncoding(
            d_model, dropout, max_position_embeddings
        )
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, target_vocab_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def encode(self, source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        source_padding = source.eq(self.source_pad_id)
        embedded = self.source_position(
            self.source_embedding(source) * math.sqrt(self.d_model)
        )
        for layer in self.encoder_layers:
            embedded = layer(embedded, source_padding)
        return self.encoder_norm(embedded), source_padding

    def decode(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        source_padding: torch.Tensor,
    ) -> torch.Tensor:
        target_padding = target.eq(self.target_pad_id)
        causal_mask = torch.triu(
            torch.ones(
                target.size(1),
                target.size(1),
                dtype=torch.bool,
                device=target.device,
            ),
            diagonal=1,
        )
        embedded = self.target_position(
            self.target_embedding(target) * math.sqrt(self.d_model)
        )
        for layer in self.decoder_layers:
            embedded = layer(
                embedded,
                memory,
                target_padding,
                source_padding,
                causal_mask,
            )
        return self.output_projection(self.decoder_norm(embedded))

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        memory, source_padding = self.encode(source)
        return self.decode(target, memory, source_padding)


@torch.inference_mode()
def beam_search_decode(
    model: Seq2SeqTransformer,
    source: torch.Tensor,
    bos_id: int,
    eos_id: int,
    beam_width: int,
    max_length: int,
    length_penalty: float,
) -> List[int]:
    if source.dim() == 1:
        source = source.unsqueeze(0)
    if source.size(0) != 1:
        raise ValueError("beam_search_decode currently accepts one source sentence")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")

    model.eval()
    memory, source_padding = model.encode(source)
    beams: List[Tuple[List[int], float, bool]] = [([bos_id], 0.0, False)]

    for _ in range(max_length):
        candidates: List[Tuple[List[int], float, bool]] = []
        for tokens, score, finished in beams:
            if finished:
                candidates.append((tokens, score, True))
                continue
            target = torch.tensor(tokens, dtype=torch.long, device=source.device).unsqueeze(0)
            logits = model.decode(target, memory, source_padding)[:, -1, :]
            log_probabilities = F.log_softmax(logits, dim=-1)
            values, indices = torch.topk(log_probabilities, beam_width, dim=-1)
            for value, index in zip(values[0].tolist(), indices[0].tolist()):
                next_tokens = tokens + [int(index)]
                candidates.append(
                    (next_tokens, score + float(value), int(index) == eos_id)
                )

        def normalized(candidate: Tuple[List[int], float, bool]) -> float:
            tokens, score, _ = candidate
            generated_length = max(1, len(tokens) - 1)
            penalty = ((5.0 + generated_length) / 6.0) ** length_penalty
            return score / penalty

        candidates.sort(key=normalized, reverse=True)
        beams = candidates[:beam_width]
        if all(finished for _, _, finished in beams):
            break

    best = max(
        beams,
        key=lambda candidate: candidate[1]
        / (((5.0 + max(1, len(candidate[0]) - 1)) / 6.0) ** length_penalty),
    )
    return best[0]

