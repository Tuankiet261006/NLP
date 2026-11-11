"""Shared data and runtime helpers for the TensorFlow 2 GPU NMT pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("seed", "data", "model", "training", "decoding", "artifacts")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("Config is missing keys: {}".format(", ".join(missing)))
    return payload


def atomic_json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def atomic_write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            for line in lines:
                stream.write(line + "\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_nmt_commit() -> Optional[str]:
    path = PROJECT_ROOT / "tensorflow_nmt/vendor/NMT_COMMIT.txt"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("commit="):
            return line.split("=", 1)[1].strip()
    return None


def read_lines(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8-sig").splitlines()


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as stream:
        return sum(1 for _ in stream)


class Vocabulary:
    """Vocabulary shared with the legacy NMT files plus an internal PAD id."""

    def __init__(self, tokens: List[str]):
        if tokens[:3] != [UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]:
            raise ValueError(
                "Vocabulary must start with {}, {}, {}".format(
                    UNK_TOKEN, BOS_TOKEN, EOS_TOKEN
                )
            )
        if PAD_TOKEN in tokens:
            raise ValueError("Shared vocabulary unexpectedly contains {}".format(PAD_TOKEN))
        self.tokens = list(tokens) + [PAD_TOKEN]
        self.stoi = {token: index for index, token in enumerate(self.tokens)}
        keys = tf.constant(self.tokens[:-1], dtype=tf.string)
        values = tf.range(len(self.tokens) - 1, dtype=tf.int32)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=tf.constant(self.unk_id, dtype=tf.int32),
        )

    @classmethod
    def from_file(cls, path: Path) -> "Vocabulary":
        return cls(read_lines(path))

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def unk_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def pad_id(self) -> int:
        return len(self.tokens) - 1

    def encode_source(self, text: str, max_length: int) -> List[int]:
        ids = [self.stoi.get(token, self.unk_id) for token in text.split()[:max_length]]
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        output: List[str] = []
        for raw_value in ids:
            value = int(raw_value)
            if value == self.eos_id:
                break
            if value in (self.bos_id, self.pad_id):
                continue
            output.append(self.tokens[value] if 0 <= value < len(self.tokens) else UNK_TOKEN)
        return " ".join(output)


def configure_runtime(seed: int, require_gpu: bool, mixed_precision: bool) -> str:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    if require_gpu and not gpus:
        raise RuntimeError("GPU mode requested but TensorFlow detected no GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    policy = "mixed_float16" if mixed_precision and require_gpu else "float32"
    tf.keras.mixed_precision.set_global_policy(policy)
    return policy


def runtime_metadata() -> Dict[str, Any]:
    build = tf.sysconfig.get_build_info()
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "python": sys.version,
        "framework": "TensorFlow {}".format(tf.__version__),
        "device": gpus[0].name if gpus else "CPU",
        "cuda_version": build.get("cuda_version"),
        "cudnn_version": build.get("cudnn_version"),
        "mixed_precision_policy": tf.keras.mixed_precision.global_policy().name,
    }


def make_parallel_dataset(
    source_path: Path,
    target_path: Path,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    shuffle_buffer: int,
    max_samples: Optional[int] = None,
) -> tf.data.Dataset:
    source_count = count_lines(source_path)
    target_count = count_lines(target_path)
    if source_count != target_count:
        raise ValueError(
            "Parallel line mismatch: {}={} {}={}".format(
                source_path, source_count, target_path, target_count
            )
        )

    dataset = tf.data.Dataset.zip(
        (
            tf.data.TextLineDataset(str(source_path)),
            tf.data.TextLineDataset(str(target_path)),
        )
    )
    if max_samples is not None:
        dataset = dataset.take(max_samples)

    eos_source = tf.constant([source_vocab.eos_id], dtype=tf.int32)
    bos_target = tf.constant([target_vocab.bos_id], dtype=tf.int32)
    eos_target = tf.constant([target_vocab.eos_id], dtype=tf.int32)

    def encode_pair(source: tf.Tensor, target: tf.Tensor) -> Tuple[tf.Tensor, ...]:
        source_tokens = tf.strings.split(source)[:max_length]
        target_tokens = tf.strings.split(target)[:max_length]
        source_ids = tf.concat([source_vocab.table.lookup(source_tokens), eos_source], axis=0)
        target_ids = target_vocab.table.lookup(target_tokens)
        target_input = tf.concat([bos_target, target_ids], axis=0)
        target_output = tf.concat([target_ids, eos_target], axis=0)
        return source_ids, target_input, target_output

    dataset = dataset.map(encode_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        effective_buffer = min(shuffle_buffer, max_samples or source_count)
        dataset = dataset.shuffle(
            max(1, effective_buffer), seed=seed, reshuffle_each_iteration=True
        )
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([None], [None], [None]),
        padding_values=(
            tf.constant(source_vocab.pad_id, tf.int32),
            tf.constant(target_vocab.pad_id, tf.int32),
            tf.constant(target_vocab.pad_id, tf.int32),
        ),
        drop_remainder=False,
    )
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 4
    dataset = dataset.with_options(options)
    return dataset.prefetch(tf.data.AUTOTUNE)


def load_vocabularies(config: Dict[str, Any]) -> Tuple[Vocabulary, Vocabulary]:
    data = config["data"]
    return (
        Vocabulary.from_file(resolve_project_path(data["src_vocab"])),
        Vocabulary.from_file(resolve_project_path(data["tgt_vocab"])),
    )


def artifact_paths(config: Dict[str, Any], smoke: bool = False) -> Tuple[Path, Path, Path]:
    artifacts = config["artifacts"]
    paths = [
        resolve_project_path(artifacts["checkpoint_dir"]),
        resolve_project_path(artifacts["log_dir"]),
        resolve_project_path(artifacts["output_dir"]),
    ]
    if smoke:
        paths = [path / "smoke" for path in paths]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths[0], paths[1], paths[2]
