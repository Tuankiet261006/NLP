#!/usr/bin/env python
"""Validate prepared parallel data and its checksum manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple


SPLITS = ("train", "tst2012", "tst2013")
LANGUAGES = ("en", "vi")


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--allow-cross-split-overlap",
        action="store_true",
        help="Report but do not fail when identical pairs occur across splits.",
    )
    return parser.parse_args()


def read_lines(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("Invalid UTF-8: {}".format(path)) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_parallel_split(
    data_dir: Path, split: str
) -> Tuple[Set[Tuple[str, str]], List[str]]:
    errors: List[str] = []
    source_path = data_dir / "{}.en".format(split)
    target_path = data_dir / "{}.vi".format(split)
    for path in (source_path, target_path):
        if not path.is_file():
            errors.append("Missing file: {}".format(path))
    if errors:
        return set(), errors

    source_lines = read_lines(source_path)
    target_lines = read_lines(target_path)
    if len(source_lines) != len(target_lines):
        errors.append(
            "{} has different line counts: en={} vi={}".format(
                split, len(source_lines), len(target_lines)
            )
        )
        return set(), errors

    pairs: Set[Tuple[str, str]] = set()
    for index, (source, target) in enumerate(
        zip(source_lines, target_lines), start=1
    ):
        if not source.strip() or not target.strip():
            errors.append("{} line {} contains an empty side".format(split, index))
        pairs.add((source, target))
    return pairs, errors


def validate_manifest(
    data_dir: Path, manifest: Dict[str, object]
) -> List[str]:
    errors: List[str] = []
    records = manifest.get("processed_files")
    if not isinstance(records, dict):
        return ["Manifest has no processed_files object"]

    required = [
        "{}.{}".format(split, language)
        for split in SPLITS
        for language in LANGUAGES
    ] + ["vocab.en", "vocab.vi"]

    for name in required:
        record = records.get(name)
        path = data_dir / name
        if not isinstance(record, dict):
            errors.append("Manifest is missing record for {}".format(name))
            continue
        if not path.is_file():
            errors.append("Missing file referenced by manifest: {}".format(path))
            continue
        expected = record.get("sha256")
        actual = sha256_file(path)
        if expected != actual:
            errors.append(
                "SHA-256 mismatch for {}: expected={} actual={}".format(
                    name, expected, actual
                )
            )
        expected_lines = record.get("lines")
        actual_lines = len(read_lines(path))
        if expected_lines != actual_lines:
            errors.append(
                "Line-count mismatch for {}: expected={} actual={}".format(
                    name, expected_lines, actual_lines
                )
            )
    return errors


def main() -> int:
    configure_console()
    args = parse_args()
    data_dir = args.data_dir.resolve()
    manifest_path = args.manifest.resolve()
    errors: List[str] = []

    if not manifest_path.is_file():
        raise FileNotFoundError("Manifest not found: {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    split_pairs: Dict[str, Set[Tuple[str, str]]] = {}
    for split in SPLITS:
        pairs, split_errors = validate_parallel_split(data_dir, split)
        split_pairs[split] = pairs
        errors.extend(split_errors)

    overlaps = {
        "train/tst2012": split_pairs["train"] & split_pairs["tst2012"],
        "train/tst2013": split_pairs["train"] & split_pairs["tst2013"],
        "tst2012/tst2013": split_pairs["tst2012"] & split_pairs["tst2013"],
    }
    for label, pairs in overlaps.items():
        if pairs:
            message = "{} contains {} duplicate pair(s)".format(label, len(pairs))
            if label == "tst2012/tst2013":
                print("WARNING: official evaluation splits: " + message)
            elif args.allow_cross_split_overlap:
                print("WARNING: " + message)
            else:
                errors.append(message)

    errors.extend(validate_manifest(data_dir, manifest))

    if errors:
        print("DATA VALIDATION FAILED")
        for error in errors:
            print("- " + error)
        return 1

    print("DATA VALIDATION PASSED")
    for split in SPLITS:
        print("{}: {} unique pair(s)".format(split, len(split_pairs[split])))
    print("Manifest: {}".format(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

