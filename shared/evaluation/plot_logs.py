#!/usr/bin/env python
"""Plot Transformer JSONL epoch logs into reports/figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        action="append",
        required=True,
        help="JSONL training log. Repeat --log to merge resumed runs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    configure_console()
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is missing. Install shared/requirements-eval.txt."
        ) from exc

    records_by_epoch: Dict[int, Dict[str, object]] = {}
    for log_path in args.log:
        for raw_line in log_path.read_text(encoding="utf-8-sig").splitlines():
            if raw_line.strip():
                record = json.loads(raw_line)
                if record.get("event") == "epoch":
                    records_by_epoch[int(record["epoch"])] = record
    records: List[Dict[str, object]] = [
        records_by_epoch[epoch] for epoch in sorted(records_by_epoch)
    ]
    if not records:
        raise ValueError("No epoch records found in {}".format(args.log))

    epochs = [int(record["epoch"]) for record in records]
    train_loss = [float(record["train_loss"]) for record in records]
    dev_loss = [float(record["dev_loss"]) for record in records]
    dev_bleu = [float(record["dev_bleu"]) for record in records]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="train loss")
    axes[0].plot(epochs, dev_loss, label="dev loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_xticks(epochs)
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, dev_bleu, label="dev BLEU", color="tab:green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BLEU")
    axes[1].set_xticks(epochs)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print("Saved {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

