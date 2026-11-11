#!/usr/bin/env python
"""Combine two evaluation JSON files into CSV and a Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--tensorflow-nmt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--transformer-hypothesis", type=Path)
    parser.add_argument("--tensorflow-hypothesis", type=Path)
    parser.add_argument("--examples", type=int, default=10)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(mapping: Dict[str, object], *keys: str) -> object:
    value: object = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def atomic_write(path: Path, text: str) -> None:
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


def metric_row(result: Dict[str, object]) -> Dict[str, object]:
    metadata = result.get("run_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "method": result.get("method"),
        "sentences": result.get("sentence_count"),
        "bleu": nested(result, "metrics", "bleu", "score"),
        "chrf": nested(result, "metrics", "chrf", "score"),
        "training_seconds": metadata.get("training_seconds")
        or metadata.get("duration_seconds"),
        "inference_seconds": metadata.get("inference_seconds"),
        "sentences_per_second": metadata.get("sentences_per_second"),
        "trainable_parameters": metadata.get("trainable_parameters"),
        "device": metadata.get("device"),
        "framework": metadata.get("framework"),
        "seed": metadata.get("seed"),
    }


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return ("{:.%df}" % digits).format(value)
    return str(value)


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def load_optional_lines(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    return path.read_text(encoding="utf-8").splitlines()


def choose_examples(sources: Sequence[str], count: int) -> List[Tuple[int, str]]:
    if not sources or count <= 0:
        return []
    lengths = [len(line.split()) for line in sources]
    ordered = sorted(range(len(sources)), key=lambda index: (lengths[index], index))
    selected: List[Tuple[int, str]] = []
    short_count = min(3, count)
    long_count = min(3, max(0, count - short_count))
    for index in ordered[:short_count]:
        selected.append((index, "Câu ngắn"))
    for index in reversed(ordered[-long_count:]):
        if index not in {item[0] for item in selected}:
            selected.append((index, "Câu dài"))
    remaining = count - len(selected)
    if remaining > 0:
        stride = max(1, len(sources) // remaining)
        for index in range(0, len(sources), stride):
            if index not in {item[0] for item in selected}:
                selected.append((index, "Mẫu phân bố"))
            if len(selected) >= count:
                break
    return selected[:count]


def qualitative_section(args: argparse.Namespace) -> str:
    paths = (
        args.source,
        args.reference,
        args.transformer_hypothesis,
        args.tensorflow_hypothesis,
    )
    if any(path is None for path in paths):
        return (
            "## Phân tích định tính\n\n"
            "Chưa tạo bảng ví dụ vì chưa cung cấp đủ source, reference và hai "
            "file hypothesis cho compare.py.\n"
        )

    sources = load_optional_lines(args.source)
    references = load_optional_lines(args.reference)
    transformer = load_optional_lines(args.transformer_hypothesis)
    tensorflow_nmt = load_optional_lines(args.tensorflow_hypothesis)
    assert sources is not None
    assert references is not None
    assert transformer is not None
    assert tensorflow_nmt is not None
    counts = {len(sources), len(references), len(transformer), len(tensorflow_nmt)}
    if len(counts) != 1:
        raise ValueError("Qualitative input files have different line counts")

    lines = [
        "## Phân tích định tính",
        "",
        "| # | Nhóm | English | Tham chiếu | Transformer | TensorFlow NMT |",
        "|---:|---|---|---|---|---|",
    ]
    for index, category in choose_examples(sources, args.examples):
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                index + 1,
                category,
                markdown_escape(sources[index]),
                markdown_escape(references[index]),
                markdown_escape(transformer[index]),
                markdown_escape(tensorflow_nmt[index]),
            )
        )
    lines.extend(
        [
            "",
            "Cần bổ sung nhận xét thủ công về sai nghĩa, từ hiếm, trật tự từ và "
            "mức tự nhiên của từng ví dụ trước khi chốt báo cáo.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    configure_console()
    args = parse_args()
    transformer = load_json(args.transformer)
    tensorflow_nmt = load_json(args.tensorflow_nmt)
    rows = [metric_row(transformer), metric_row(tensorflow_nmt)]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metrics.csv"
    fieldnames = list(rows[0].keys())
    handle, temp_name = tempfile.mkstemp(
        prefix=csv_path.name + ".", suffix=".tmp", dir=str(output_dir)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, csv_path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

    report_lines = [
        "# Báo cáo so sánh dịch máy Anh - Việt",
        "",
        "## Kết quả định lượng",
        "",
        "| Phương pháp | BLEU | chrF | Câu | Thời gian train (s) | "
        "Thời gian dịch (s) | Câu/s | Tham số | Thiết bị |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        report_lines.append(
            "| {method} | {bleu} | {chrf} | {sentences} | {training} | "
            "{inference} | {speed} | {parameters} | {device} |".format(
                method=fmt(row["method"]),
                bleu=fmt(row["bleu"]),
                chrf=fmt(row["chrf"]),
                sentences=fmt(row["sentences"], 0),
                training=fmt(row["training_seconds"]),
                inference=fmt(row["inference_seconds"]),
                speed=fmt(row["sentences_per_second"]),
                parameters=fmt(row["trainable_parameters"], 0),
                device=fmt(row["device"]),
            )
        )

    report_lines.extend(
        [
            "",
            "Các số thời gian chỉ được so sánh trực tiếp khi hai lần chạy dùng "
            "phần cứng tương đương.",
            "",
            "## Chữ ký metric",
            "",
            "- Transformer BLEU: {}".format(
                nested(transformer, "metrics", "bleu", "signature")
            ),
            "- Transformer chrF: {}".format(
                nested(transformer, "metrics", "chrf", "signature")
            ),
            "- TensorFlow NMT BLEU: {}".format(
                nested(tensorflow_nmt, "metrics", "bleu", "signature")
            ),
            "- TensorFlow NMT chrF: {}".format(
                nested(tensorflow_nmt, "metrics", "chrf", "signature")
            ),
            "",
            qualitative_section(args),
            "## Kết luận",
            "",
            "Điền kết luận sau khi kiểm tra metric, điều kiện phần cứng và các "
            "ví dụ dịch định tính.",
            "",
        ]
    )
    report_path = output_dir / "comparison.md"
    atomic_write(report_path, "\n".join(report_lines))
    print("Saved {}".format(csv_path))
    print("Saved {}".format(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

