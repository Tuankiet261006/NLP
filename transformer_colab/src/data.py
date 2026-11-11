"""Deterministic word-level data pipeline without legacy torchtext/spaCy APIs."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Sampler


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)


def tokenize(text: str, lowercase: bool = False) -> List[str]:
    text = text.strip()
    if lowercase:
        text = text.lower()
    return text.split() if text else []


def read_lines(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("Invalid UTF-8: {}".format(path)) from exc


class Vocab:
    def __init__(self, itos: Sequence[str]):
        if tuple(itos[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError("Vocabulary must begin with {}".format(SPECIAL_TOKENS))
        self.itos = list(itos)
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    @classmethod
    def build(
        cls,
        lines: Iterable[str],
        min_frequency: int = 2,
        lowercase: bool = False,
    ) -> "Vocab":
        counts: Counter[str] = Counter()
        for line in lines:
            counts.update(tokenize(line, lowercase=lowercase))
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        tokens = list(SPECIAL_TOKENS)
        tokens.extend(
            token
            for token, frequency in ordered
            if frequency >= min_frequency and token not in SPECIAL_TOKENS
        )
        return cls(tokens)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(
        self,
        text: str,
        lowercase: bool = False,
        add_bos: bool = True,
        add_eos: bool = True,
        max_tokens: Optional[int] = None,
    ) -> List[int]:
        tokens = tokenize(text, lowercase=lowercase)
        if max_tokens is not None and max_tokens > 0:
            tokens = tokens[:max_tokens]
        ids = [self.stoi.get(token, self.unk_id) for token in tokens]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        tokens: List[str] = []
        for value in ids:
            index = int(value)
            if index == self.eos_id:
                break
            if index in (self.pad_id, self.bos_id):
                continue
            tokens.append(self.itos[index] if 0 <= index < len(self.itos) else UNK_TOKEN)
        return " ".join(tokens)

    def to_dict(self) -> Dict[str, object]:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Vocab":
        values = payload.get("itos")
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError("Invalid vocabulary payload")
        return cls(values)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class ParallelDataset(Dataset):
    def __init__(
        self,
        source_path: Path,
        target_path: Path,
        lowercase: bool,
        max_length: int,
        filter_long: bool,
        max_samples: Optional[int] = None,
    ):
        source_lines = read_lines(source_path)
        target_lines = read_lines(target_path)
        if len(source_lines) != len(target_lines):
            raise ValueError(
                "Parallel line mismatch: {}={} {}={}".format(
                    source_path, len(source_lines), target_path, len(target_lines)
                )
            )

        self.examples: List[Tuple[str, str]] = []
        self.lengths: List[int] = []
        self.dropped = 0
        self.truncated = 0
        for source, target in zip(source_lines, target_lines):
            source_tokens = tokenize(source, lowercase)
            target_tokens = tokenize(target, lowercase)
            is_long = len(source_tokens) > max_length or len(target_tokens) > max_length
            if filter_long and is_long:
                self.dropped += 1
                continue
            if is_long:
                self.truncated += 1
                source = " ".join(source_tokens[:max_length])
                target = " ".join(target_tokens[:max_length])
                source_tokens = source_tokens[:max_length]
                target_tokens = target_tokens[:max_length]
            self.examples.append((source, target))
            self.lengths.append(max(len(source_tokens), len(target_tokens)) + 2)
            if max_samples is not None and len(self.examples) >= max_samples:
                break
        if not self.examples:
            raise ValueError("Dataset has no usable examples: {}".format(source_path))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        return self.examples[index]


class TokenBatchSampler(Sampler[List[int]]):
    """Groups similarly sized examples under an approximate token budget."""

    def __init__(
        self,
        lengths: Sequence[int],
        max_tokens: int,
        shuffle: bool,
        seed: int,
        pool_size: int = 1000,
    ):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.lengths = list(lengths)
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.pool_size = pool_size
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.lengths)))
        randomizer = random.Random(self.seed + self.epoch)
        self.epoch += 1
        if self.shuffle:
            randomizer.shuffle(indices)

        pooled: List[int] = []
        for start in range(0, len(indices), self.pool_size):
            pool = indices[start : start + self.pool_size]
            pool.sort(key=lambda index: self.lengths[index])
            if self.shuffle and randomizer.random() < 0.5:
                pool.reverse()
            pooled.extend(pool)

        batches: List[List[int]] = []
        batch: List[int] = []
        batch_max = 0
        for index in pooled:
            proposed_max = max(batch_max, self.lengths[index])
            if batch and proposed_max * (len(batch) + 1) > self.max_tokens:
                batches.append(batch)
                batch = []
                batch_max = 0
            batch.append(index)
            batch_max = max(batch_max, self.lengths[index])
        if batch:
            batches.append(batch)
        if self.shuffle:
            randomizer.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        total = sum(max(1, length) for length in self.lengths)
        return max(1, math.ceil(total / self.max_tokens))


def pad_sequences(sequences: Sequence[Sequence[int]], pad_id: int) -> torch.Tensor:
    max_length = max(len(sequence) for sequence in sequences)
    output = torch.full(
        (len(sequences), max_length), fill_value=pad_id, dtype=torch.long
    )
    for row, sequence in enumerate(sequences):
        output[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return output


def make_parallel_collate(
    source_vocab: Vocab,
    target_vocab: Vocab,
    lowercase: bool,
    max_length: int,
):
    def collate(batch: Sequence[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        source_ids = [
            source_vocab.encode(
                source, lowercase=lowercase, max_tokens=max_length
            )
            for source, _ in batch
        ]
        target_ids = [
            target_vocab.encode(
                target, lowercase=lowercase, max_tokens=max_length
            )
            for _, target in batch
        ]
        return (
            pad_sequences(source_ids, source_vocab.pad_id),
            pad_sequences(target_ids, target_vocab.pad_id),
        )

    return collate

