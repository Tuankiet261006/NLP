"""Beam-search inference for the TensorFlow 2 GPU Luong NMT model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import tensorflow as tf

from .tf2_common import (
    PROJECT_ROOT,
    artifact_paths,
    atomic_json_dump,
    atomic_write_lines,
    configure_console,
    configure_runtime,
    load_config,
    load_vocabularies,
    read_lines,
    resolve_project_path,
    runtime_metadata,
    utc_now,
)
from .tf2_model import beam_search_decode, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "tensorflow_nmt/configs/tf2_gpu.json",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sentence")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-sentences", type=int)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    configure_console()
    args = parse_args()
    if args.input is not None and args.sentence is not None:
        raise ValueError("Use either --input or --sentence, not both")
    config = load_config(args.config.resolve())
    configure_runtime(
        int(config["seed"]),
        require_gpu=args.device == "gpu",
        mixed_precision=bool(config["training"]["mixed_precision"]),
    )
    source_vocab, target_vocab = load_vocabularies(config)
    model = build_model(
        config,
        len(source_vocab),
        len(target_vocab),
        source_vocab.pad_id,
        target_vocab.pad_id,
    )
    checkpoint_dir, _, output_dir = artifact_paths(config, smoke=args.smoke)
    if args.checkpoint:
        checkpoint_path = str(args.checkpoint.resolve())
    else:
        manager = tf.train.CheckpointManager(
            tf.train.Checkpoint(model=model), str(checkpoint_dir / "best"), max_to_keep=1
        )
        checkpoint_path = manager.latest_checkpoint or ""
    if not checkpoint_path:
        raise FileNotFoundError("No best checkpoint found in {}".format(checkpoint_dir / "best"))
    tf.train.Checkpoint(model=model).restore(checkpoint_path).expect_partial()
    print("Loaded {}".format(checkpoint_path), flush=True)
    started_at = utc_now()

    decoding = config["decoding"]
    beam_width = int(args.beam_width)
    max_output_length = args.max_length or int(decoding["max_length"])
    max_input_length = int(config["data"]["max_length"])

    def translate(text: str) -> str:
        if not text.strip():
            return ""
        source_ids = source_vocab.encode_source(text, max_input_length)
        prediction = beam_search_decode(
            model,
            source_ids,
            target_vocab.bos_id,
            target_vocab.eos_id,
            beam_width,
            max_output_length,
            float(decoding["length_penalty"]),
        )
        return target_vocab.decode(prediction)

    if args.sentence is not None:
        print(translate(args.sentence))
        return 0

    input_path = (
        args.input.resolve()
        if args.input
        else resolve_project_path(config["data"]["test_src"])
    )
    destination = (
        args.output.resolve() if args.output else output_dir / "test.hyp.vi"
    )
    source_lines = read_lines(input_path)
    if args.max_sentences is not None:
        if args.max_sentences <= 0:
            raise ValueError("--max-sentences must be positive")
        source_lines = source_lines[: args.max_sentences]

    started = time.perf_counter()
    hypotheses: List[str] = []
    for index, source in enumerate(source_lines, start=1):
        hypotheses.append(translate(source))
        if index % 50 == 0 or index == len(source_lines):
            print("translated {}/{}".format(index, len(source_lines)), flush=True)
    duration = time.perf_counter() - started
    atomic_write_lines(destination, hypotheses)
    if len(hypotheses) != len(source_lines):
        raise RuntimeError("Inference output line count mismatch")

    metadata: Dict[str, Any] = {
        "schema_version": 1,
        "method": "tensorflow_nmt_tf2",
        "mode": "inference",
        "status": "completed",
        "started_at": started_at,
        "finished_at": utc_now(),
        "checkpoint": checkpoint_path,
        "input": str(input_path),
        "output": str(destination),
        "sentence_count": len(hypotheses),
        "beam_width": beam_width,
        "max_length": max_output_length,
        "inference_seconds": duration,
        "sentences_per_second": len(hypotheses) / duration if duration else None,
        "trainable_parameters": int(model.count_params()),
        **runtime_metadata(),
    }
    atomic_json_dump(output_dir / "run_metadata_infer.json", metadata)
    print("Saved {} ({:.2f}s)".format(destination, duration))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
