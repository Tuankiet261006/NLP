"""Run sentence or file inference from a Transformer checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from .common import (
    PROJECT_ROOT,
    atomic_json_dump,
    configure_console,
    count_trainable_parameters,
    resolve_project_path,
    runtime_metadata,
    set_seed,
    utc_now,
)
from .data import Vocab, read_lines
from .model import Seq2SeqTransformer, beam_search_decode
from .train import build_model, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "transformer_colab/checkpoints/best.pt",
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sentence")
    parser.add_argument(
        "--beam-width",
        type=int,
        default=3,
        help="Beam width for decoding (default: 3).",
    )
    parser.add_argument("--max-length", type=int)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args()


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


def translate(
    sentence: str,
    model: Seq2SeqTransformer,
    source_vocab: Vocab,
    target_vocab: Vocab,
    config: Dict[str, Any],
    device: torch.device,
    beam_width: int,
    max_output_length: int,
) -> str:
    data_config = config["data"]
    decoding_config = config["decoding"]
    if not sentence.strip():
        return ""
    source_ids = source_vocab.encode(
        sentence,
        lowercase=bool(data_config["lowercase"]),
        max_tokens=int(data_config["max_length"]),
    )
    source = torch.tensor(source_ids, dtype=torch.long, device=device)
    prediction = beam_search_decode(
        model,
        source,
        bos_id=target_vocab.bos_id,
        eos_id=target_vocab.eos_id,
        beam_width=beam_width,
        max_length=max_output_length,
        length_penalty=float(decoding_config["length_penalty"]),
    )
    return target_vocab.decode(prediction)


def main() -> int:
    configure_console()
    args = parse_args()
    if bool(args.input) == bool(args.sentence):
        raise ValueError("Provide exactly one of --input or --sentence")
    if args.input and not args.output:
        raise ValueError("--output is required with --input")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    source_vocab = Vocab.from_dict(checkpoint["source_vocab"])
    target_vocab = Vocab.from_dict(checkpoint["target_vocab"])
    set_seed(int(checkpoint.get("seed", config["seed"])))
    model = build_model(config, source_vocab, target_vocab).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    decoding_config = config["decoding"]
    beam_width = int(args.beam_width)
    max_output_length = args.max_length or int(decoding_config["max_length"])

    if args.sentence is not None:
        print(
            translate(
                args.sentence,
                model,
                source_vocab,
                target_vocab,
                config,
                device,
                beam_width,
                max_output_length,
            )
        )
        return 0

    assert args.input is not None and args.output is not None
    source_lines = read_lines(args.input)
    started_at = utc_now()
    started = time.perf_counter()
    hypotheses = [
        translate(
            sentence,
            model,
            source_vocab,
            target_vocab,
            config,
            device,
            beam_width,
            max_output_length,
        )
        for sentence in source_lines
    ]
    duration = time.perf_counter() - started
    atomic_write_lines(args.output, hypotheses)

    training_metadata_path = resolve_project_path(
        config["artifacts"]["output_dir"]
    ) / "run_metadata.json"
    training_metadata = (
        json.loads(training_metadata_path.read_text(encoding="utf-8"))
        if training_metadata_path.is_file()
        else {}
    )
    metadata = {
        "schema_version": 1,
        "run_id": time.strftime("%Y%m%dT%H%M%S"),
        "method": "transformer",
        "status": "completed",
        "started_at": started_at,
        "finished_at": utc_now(),
        "seed": int(checkpoint.get("seed", config["seed"])),
        "checkpoint": str(args.checkpoint.resolve()),
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "sentence_count": len(source_lines),
        "inference_seconds": duration,
        "sentences_per_second": len(source_lines) / duration if duration else None,
        "beam_width": beam_width,
        "max_length": max_output_length,
        "trainable_parameters": count_trainable_parameters(model),
        "training_seconds": training_metadata.get("training_seconds"),
        "training_run_id": training_metadata.get("run_id"),
        **runtime_metadata(device),
    }
    atomic_json_dump(args.output.parent / "run_metadata_infer.json", metadata)
    print(
        "Translated {} sentence(s) in {:.2f}s ({:.2f} sentences/s)".format(
            len(source_lines),
            duration,
            metadata["sentences_per_second"] or 0.0,
        )
    )
    print("Saved {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

