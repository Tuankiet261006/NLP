#!/usr/bin/env python
"""Evaluate one Vietnamese hypothesis with reproducible SacreBLEU metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


SPACE_RE = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?%])")
SPACE_AFTER_OPEN_RE = re.compile(r"([\(\[\{])\s+")


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--hypothesis", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--tokenize", default="13a")
    parser.add_argument(
        "--no-detokenize",
        action="store_true",
        help="Do not apply the shared basic punctuation detokenizer.",
    )
    return parser.parse_args()


def read_lines(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
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


def normalize_line(line: str, detokenize: bool) -> str:
    line = unicodedata.normalize("NFC", line)
    line = SPACE_RE.sub(" ", line).strip()
    if detokenize:
        line = SPACE_BEFORE_PUNCT_RE.sub(r"\1", line)
        line = SPACE_AFTER_OPEN_RE.sub(r"\1", line)
    return line


def atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
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


def main() -> int:
    configure_console()
    args = parse_args()
    try:
        from sacrebleu.metrics import BLEU, CHRF
        import sacrebleu
    except ImportError as exc:
        raise RuntimeError(
            "SacreBLEU is missing. Install shared/requirements-eval.txt."
        ) from exc

    references = read_lines(args.reference)
    hypotheses = read_lines(args.hypothesis)
    if len(references) != len(hypotheses):
        raise ValueError(
            "Line-count mismatch: reference={} hypothesis={}".format(
                len(references), len(hypotheses)
            )
        )
    if not references:
        raise ValueError("Cannot evaluate an empty corpus")

    detokenize = not args.no_detokenize
    references = [normalize_line(line, detokenize) for line in references]
    hypotheses = [normalize_line(line, detokenize) for line in hypotheses]

    bleu_metric = BLEU(tokenize=args.tokenize)
    chrf_metric = CHRF()
    bleu = bleu_metric.corpus_score(hypotheses, [references])
    chrf = chrf_metric.corpus_score(hypotheses, [references])

    run_metadata = None
    if args.metadata:
        run_metadata = json.loads(args.metadata.read_text(encoding="utf-8"))

    payload: Dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "sentence_count": len(references),
        "reference": {
            "path": str(args.reference.resolve()),
            "sha256": sha256_file(args.reference),
        },
        "hypothesis": {
            "path": str(args.hypothesis.resolve()),
            "sha256": sha256_file(args.hypothesis),
        },
        "preprocessing": {
            "unicode_normalization": "NFC",
            "basic_detokenization": detokenize,
            "bleu_tokenizer": args.tokenize,
        },
        "sacrebleu_version": sacrebleu.__version__,
        "metrics": {
            "bleu": {
                "score": bleu.score,
                "signature": str(bleu_metric.get_signature()),
                "verbose": str(bleu),
            },
            "chrf": {
                "score": chrf.score,
                "signature": str(chrf_metric.get_signature()),
                "verbose": str(chrf),
            },
        },
        "run_metadata": run_metadata,
    }
    atomic_write_json(args.output, payload)
    print(
        "{}: BLEU={:.4f} chrF={:.4f} sentences={}".format(
            args.method, bleu.score, chrf.score, len(references)
        )
    )
    print("Saved {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

