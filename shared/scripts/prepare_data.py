#!/usr/bin/env python
"""Prepare one canonical IWSLT 2015 English-Vietnamese dataset.

The script keeps raw files untouched, normalizes parallel text deterministically,
builds vocabularies from the processed training split only, and writes a
checksum manifest shared by both model implementations.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SPLITS = ("train", "tst2012", "tst2013")
LANGUAGES = ("en", "vi")
SPECIAL_TOKENS = ("<unk>", "<s>", "</s>")
WHITESPACE_RE = re.compile(r"\s+")


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/../manifest.json.",
    )
    parser.add_argument(
        "--unicode-normalization",
        choices=("NFC", "NFKC", "NFD", "NFKD"),
        default="NFC",
    )
    parser.add_argument(
        "--max-train-tokens",
        type=int,
        default=160,
        help="Drop a train pair when either side exceeds this length; 0 disables.",
    )
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=0,
        help="Maximum vocabulary size including special tokens; 0 means unlimited.",
    )
    parser.add_argument(
        "--source-url",
        default="https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_utf8_lines(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("File is not valid UTF-8: {}".format(path)) from exc


def normalize_line(text: str, normalization: str) -> str:
    text = text.lstrip("\ufeff")
    text = unicodedata.normalize(normalization, text)
    return WHITESPACE_RE.sub(" ", text).strip()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def write_lines(path: Path, lines: Sequence[str]) -> None:
    payload = "\n".join(lines)
    if lines:
        payload += "\n"
    atomic_write_text(path, payload)


def prepare_split(
    split: str,
    input_dir: Path,
    output_dir: Path,
    normalization: str,
    max_train_tokens: int,
) -> Dict[str, int]:
    source_path = input_dir / "{}.en".format(split)
    target_path = input_dir / "{}.vi".format(split)
    if not source_path.is_file() or not target_path.is_file():
        missing = [str(p) for p in (source_path, target_path) if not p.is_file()]
        raise FileNotFoundError("Missing parallel file(s): {}".format(", ".join(missing)))

    source_lines = read_utf8_lines(source_path)
    target_lines = read_utf8_lines(target_path)
    if len(source_lines) != len(target_lines):
        raise ValueError(
            "{} line mismatch: en={} vi={}".format(
                split, len(source_lines), len(target_lines)
            )
        )

    output_source: List[str] = []
    output_target: List[str] = []
    dropped_empty = 0
    dropped_too_long = 0

    for source, target in zip(source_lines, target_lines):
        source = normalize_line(source, normalization)
        target = normalize_line(target, normalization)

        if split == "train" and (not source or not target):
            dropped_empty += 1
            continue

        if (
            split == "train"
            and max_train_tokens > 0
            and (
                len(source.split()) > max_train_tokens
                or len(target.split()) > max_train_tokens
            )
        ):
            dropped_too_long += 1
            continue

        output_source.append(source)
        output_target.append(target)

    write_lines(output_dir / "{}.en".format(split), output_source)
    write_lines(output_dir / "{}.vi".format(split), output_target)

    return {
        "input_pairs": len(source_lines),
        "output_pairs": len(output_source),
        "dropped_empty": dropped_empty,
        "dropped_too_long": dropped_too_long,
    }


def build_vocab(
    train_path: Path, max_vocab_size: int
) -> Tuple[List[str], Dict[str, int]]:
    counts: collections.Counter[str] = collections.Counter()
    for line in read_utf8_lines(train_path):
        counts.update(line.split())

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    tokens = list(SPECIAL_TOKENS)
    for token, _ in ordered:
        if token not in SPECIAL_TOKENS:
            tokens.append(token)
        if max_vocab_size > 0 and len(tokens) >= max_vocab_size:
            break

    statistics = {
        "size": len(tokens),
        "unique_train_tokens": len(counts),
        "total_train_tokens": sum(counts.values()),
    }
    return tokens, statistics


def remove_training_leakage(output_dir: Path) -> int:
    train_source = read_utf8_lines(output_dir / "train.en")
    train_target = read_utf8_lines(output_dir / "train.vi")
    evaluation_pairs = set()
    for split in ("tst2012", "tst2013"):
        sources = read_utf8_lines(output_dir / "{}.en".format(split))
        targets = read_utf8_lines(output_dir / "{}.vi".format(split))
        evaluation_pairs.update(zip(sources, targets))

    kept_source: List[str] = []
    kept_target: List[str] = []
    dropped = 0
    for source, target in zip(train_source, train_target):
        if (source, target) in evaluation_pairs:
            dropped += 1
            continue
        kept_source.append(source)
        kept_target.append(target)
    write_lines(output_dir / "train.en", kept_source)
    write_lines(output_dir / "train.vi", kept_target)
    return dropped


def file_record(path: Path, root: Path) -> Dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "lines": len(read_utf8_lines(path)),
    }


def main() -> int:
    configure_console()
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest
        else (output_dir.parent / "manifest.json").resolve()
    )

    if args.max_train_tokens < 0:
        raise ValueError("--max-train-tokens must be >= 0")
    if args.max_vocab_size and args.max_vocab_size < len(SPECIAL_TOKENS):
        raise ValueError("--max-vocab-size must be 0 or >= 3")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_statistics: Dict[str, Dict[str, int]] = {}
    for split in SPLITS:
        split_statistics[split] = prepare_split(
            split=split,
            input_dir=input_dir,
            output_dir=output_dir,
            normalization=args.unicode_normalization,
            max_train_tokens=args.max_train_tokens,
        )
    dropped_leakage = remove_training_leakage(output_dir)
    split_statistics["train"]["dropped_cross_split"] = dropped_leakage
    split_statistics["train"]["output_pairs"] -= dropped_leakage

    vocab_statistics: Dict[str, Dict[str, int]] = {}
    for language in LANGUAGES:
        vocab, stats = build_vocab(
            output_dir / "train.{}".format(language), args.max_vocab_size
        )
        write_lines(output_dir / "vocab.{}".format(language), vocab)
        vocab_statistics[language] = stats

    raw_records = {}
    processed_records = {}
    for split in SPLITS:
        for language in LANGUAGES:
            name = "{}.{}".format(split, language)
            raw_records[name] = file_record(input_dir / name, input_dir)
            processed_records[name] = file_record(output_dir / name, output_dir)
    for language in LANGUAGES:
        name = "vocab.{}".format(language)
        processed_records[name] = file_record(output_dir / name, output_dir)

    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "dataset": "IWSLT 2015 English-Vietnamese",
        "direction": "en-vi",
        "source_url": args.source_url,
        "normalization": args.unicode_normalization,
        "max_train_tokens": args.max_train_tokens,
        "max_vocab_size": args.max_vocab_size,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "splits": split_statistics,
        "vocabularies": vocab_statistics,
        "raw_files": raw_records,
        "processed_files": processed_records,
    }
    atomic_write_text(
        manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )

    print("Prepared IWSLT 2015 data in {}".format(output_dir))
    print("Manifest: {}".format(manifest_path))
    for split in SPLITS:
        stats = split_statistics[split]
        print(
            "{}: {} -> {} pairs (empty={}, too_long={}, leakage={})".format(
                split,
                stats["input_pairs"],
                stats["output_pairs"],
                stats["dropped_empty"],
                stats["dropped_too_long"],
                stats.get("dropped_cross_split", 0),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

