"""Train the English-Vietnamese Transformer with reproducible artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .common import (
    PROJECT_ROOT,
    append_jsonl,
    atomic_json_dump,
    configure_console,
    count_trainable_parameters,
    load_config,
    resolve_project_path,
    runtime_metadata,
    set_seed,
    sha256_file,
    utc_now,
)
from .data import (
    ParallelDataset,
    TokenBatchSampler,
    Vocab,
    make_parallel_collate,
)
from .model import Seq2SeqTransformer, beam_search_decode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "transformer_colab/configs/baseline.json",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use at most 500 train examples and one epoch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration/data and instantiate the model only.",
    )
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def build_model(
    config: Dict[str, Any], source_vocab: Vocab, target_vocab: Vocab
) -> Seq2SeqTransformer:
    model_config = config["model"]
    return Seq2SeqTransformer(
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        source_pad_id=source_vocab.pad_id,
        target_pad_id=target_vocab.pad_id,
        d_model=int(model_config["d_model"]),
        nhead=int(model_config["nhead"]),
        num_encoder_layers=int(model_config["num_encoder_layers"]),
        num_decoder_layers=int(model_config["num_decoder_layers"]),
        dim_feedforward=int(model_config["dim_feedforward"]),
        dropout=float(model_config["dropout"]),
        max_position_embeddings=int(model_config["max_position_embeddings"]),
    )


class NoamScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        scale: float,
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.scale = scale
        self.step_number = 0

    def _rate(self, step: int) -> float:
        step = max(1, step)
        return (
            self.scale
            * (self.d_model ** -0.5)
            * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        )

    def step(self) -> float:
        self.step_number += 1
        rate = self._rate(self.step_number)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        return rate

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "scale": self.scale,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.step_number = int(state["step_number"])


def atomic_torch_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(handle)
    try:
        torch.save(payload, temp_name)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def make_loader(
    dataset: ParallelDataset,
    source_vocab: Vocab,
    target_vocab: Vocab,
    lowercase: bool,
    max_length: int,
    max_tokens: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    sampler = TokenBatchSampler(
        dataset.lengths,
        max_tokens=max_tokens,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=make_parallel_collate(
            source_vocab, target_vocab, lowercase, max_length
        ),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def loss_for_batch(
    model: Seq2SeqTransformer,
    source: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
) -> Tuple[torch.Tensor, int]:
    target_input = target[:, :-1]
    target_labels = target[:, 1:]
    logits = model(source, target_input)
    loss = criterion(
        logits.reshape(-1, logits.size(-1)), target_labels.reshape(-1)
    )
    token_count = int(target_labels.ne(model.target_pad_id).sum().item())
    return loss, token_count


def train_epoch(
    model: Seq2SeqTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamScheduler,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip_norm: float,
    scaler: torch.amp.GradScaler,
    mixed_precision: bool,
) -> Tuple[float, float]:
    model.train()
    weighted_loss = 0.0
    total_tokens = 0
    last_rate = 0.0
    for source, target in loader:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=mixed_precision,
        ):
            loss, token_count = loss_for_batch(model, source, target, criterion)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss became non-finite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        last_rate = scheduler.step()
        scaler.step(optimizer)
        scaler.update()
        weighted_loss += float(loss.item()) * token_count
        total_tokens += token_count
    return weighted_loss / max(1, total_tokens), last_rate


@torch.inference_mode()
def evaluate_loss(
    model: Seq2SeqTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mixed_precision: bool,
) -> float:
    model.eval()
    weighted_loss = 0.0
    total_tokens = 0
    for source, target in loader:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=mixed_precision,
        ):
            loss, token_count = loss_for_batch(model, source, target, criterion)
        weighted_loss += float(loss.item()) * token_count
        total_tokens += token_count
    return weighted_loss / max(1, total_tokens)


@torch.inference_mode()
def evaluate_bleu(
    model: Seq2SeqTransformer,
    dataset: ParallelDataset,
    source_vocab: Vocab,
    target_vocab: Vocab,
    lowercase: bool,
    max_source_length: int,
    beam_width: int,
    max_output_length: int,
    length_penalty: float,
    device: torch.device,
    limit: int,
) -> Tuple[float, List[str]]:
    try:
        import sacrebleu
    except ImportError as exc:
        raise RuntimeError(
            "SacreBLEU is required. Install transformer_colab/requirements-colab.txt."
        ) from exc

    references: List[str] = []
    hypotheses: List[str] = []
    count = min(len(dataset), limit) if limit > 0 else len(dataset)
    for index in range(count):
        source_text, target_text = dataset[index]
        source_ids = source_vocab.encode(
            source_text,
            lowercase=lowercase,
            max_tokens=max_source_length,
        )
        source = torch.tensor(source_ids, dtype=torch.long, device=device)
        prediction = beam_search_decode(
            model,
            source,
            bos_id=target_vocab.bos_id,
            eos_id=target_vocab.eos_id,
            beam_width=beam_width,
            max_length=max_output_length,
            length_penalty=length_penalty,
        )
        hypotheses.append(target_vocab.decode(prediction))
        references.append(target_text)
    score = sacrebleu.corpus_bleu(hypotheses, [references]).score
    return float(score), hypotheses


def save_text_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8", newline="\n")


def main() -> int:
    configure_console()
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    seed = int(config["seed"])
    set_seed(seed)
    device = select_device(args.device)

    data_config = config["data"]
    training_config = config["training"]
    decoding_config = config["decoding"]
    artifact_config = config["artifacts"]
    max_samples = 500 if args.smoke else None
    epochs = 1 if args.smoke else int(training_config["epochs"])
    lowercase = bool(data_config["lowercase"])
    max_length = int(data_config["max_length"])

    train_dataset = ParallelDataset(
        resolve_project_path(data_config["train_src"]),
        resolve_project_path(data_config["train_tgt"]),
        lowercase=lowercase,
        max_length=max_length,
        filter_long=True,
        max_samples=max_samples,
    )
    dev_dataset = ParallelDataset(
        resolve_project_path(data_config["dev_src"]),
        resolve_project_path(data_config["dev_tgt"]),
        lowercase=lowercase,
        max_length=max_length,
        filter_long=False,
        max_samples=100 if args.smoke else None,
    )
    source_vocab = Vocab.build(
        (source for source, _ in train_dataset.examples),
        min_frequency=int(data_config["min_frequency"]),
        lowercase=lowercase,
    )
    target_vocab = Vocab.build(
        (target for _, target in train_dataset.examples),
        min_frequency=int(data_config["min_frequency"]),
        lowercase=lowercase,
    )

    checkpoint_dir = resolve_project_path(artifact_config["checkpoint_dir"])
    log_dir = resolve_project_path(artifact_config["log_dir"])
    output_dir = resolve_project_path(artifact_config["output_dir"])
    for directory in (checkpoint_dir, log_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)
    source_vocab.save(checkpoint_dir / "source_vocab.json")
    target_vocab.save(checkpoint_dir / "target_vocab.json")

    model = build_model(config, source_vocab, target_vocab).to(device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    parameter_count = count_trainable_parameters(model)
    print("Device: {}".format(device), flush=True)
    print("Train examples: {} (dropped={})".format(len(train_dataset), train_dataset.dropped), flush=True)
    print(
        "Dev examples: {} (truncated={})".format(
            len(dev_dataset), dev_dataset.truncated
        )
    )
    print("Vocabulary: en={} vi={}".format(len(source_vocab), len(target_vocab)), flush=True)
    print("Trainable parameters: {:,}".format(parameter_count), flush=True)

    if args.dry_run:
        print("Dry run passed: config, data, vocabularies, and model are valid.")
        return 0

    train_loader = make_loader(
        train_dataset,
        source_vocab,
        target_vocab,
        lowercase,
        max_length,
        int(training_config["batch_tokens"]),
        True,
        seed,
        int(training_config["num_workers"]),
    )
    dev_loader = make_loader(
        dev_dataset,
        source_vocab,
        target_vocab,
        lowercase,
        max_length,
        int(training_config["batch_tokens"]),
        False,
        seed,
        int(training_config["num_workers"]),
    )

    beta1, beta2 = training_config["adam_betas"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0,
        betas=(float(beta1), float(beta2)),
        eps=float(training_config["adam_epsilon"]),
    )
    scheduler = NoamScheduler(
        optimizer,
        d_model=int(config["model"]["d_model"]),
        warmup_steps=int(training_config["warmup_steps"]),
        scale=float(training_config["learning_rate_scale"]),
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=target_vocab.pad_id,
        label_smoothing=float(training_config["label_smoothing"]),
    )
    mixed_precision = bool(training_config.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    start_epoch = 1
    best_metric: Optional[float] = None
    bad_epochs = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = checkpoint.get("best_metric")
        bad_epochs = int(checkpoint.get("bad_epochs", 0))
        print("Resumed from {} at epoch {}".format(args.resume, start_epoch))

    run_id = time.strftime("%Y%m%dT%H%M%S")
    log_path = log_dir / "train_{}.jsonl".format(run_id)
    metadata_path = output_dir / "run_metadata.json"
    manifest_path = PROJECT_ROOT / "shared/data/manifest.json"
    metadata: Dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "method": "transformer",
        "status": "running",
        "started_at": utc_now(),
        "seed": seed,
        "config_path": str(config_path),
        "config": config,
        "trainable_parameters": parameter_count,
        "dataset_manifest_sha256": sha256_file(manifest_path)
        if manifest_path.is_file()
        else None,
        "train_examples": len(train_dataset),
        "dev_examples": len(dev_dataset),
        "train_dropped": train_dataset.dropped,
        "dev_truncated": dev_dataset.truncated,
        **runtime_metadata(device),
    }
    atomic_json_dump(metadata_path, metadata)
    append_jsonl(log_path, {"event": "run_start", **metadata})

    checkpoint_metric = str(training_config["checkpoint_metric"])
    if checkpoint_metric not in ("dev_loss", "dev_bleu"):
        raise ValueError("checkpoint_metric must be dev_loss or dev_bleu")
    started = time.perf_counter()
    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_started = time.perf_counter()
            train_loss, learning_rate = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                device,
                float(training_config["gradient_clip_norm"]),
                scaler,
                mixed_precision,
            )
            dev_loss = evaluate_loss(
                model, dev_loader, criterion, device, mixed_precision
            )
            dev_bleu, hypotheses = evaluate_bleu(
                model,
                dev_dataset,
                source_vocab,
                target_vocab,
                lowercase,
                max_length,
                int(training_config.get("dev_bleu_beam_width", 1)),
                int(decoding_config["max_length"]),
                float(decoding_config["length_penalty"]),
                device,
                10 if args.smoke else int(training_config["dev_bleu_max_sentences"]),
            )
            save_text_lines(output_dir / "dev.hyp.vi", hypotheses)

            current_metric = dev_loss if checkpoint_metric == "dev_loss" else dev_bleu
            improved = (
                best_metric is None
                or (checkpoint_metric == "dev_loss" and current_metric < best_metric)
                or (checkpoint_metric == "dev_bleu" and current_metric > best_metric)
            )
            if improved:
                best_metric = current_metric
                bad_epochs = 0
            else:
                bad_epochs += 1

            record = {
                "event": "epoch",
                "timestamp": utc_now(),
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "dev_bleu": dev_bleu,
                "learning_rate": learning_rate,
                "duration_seconds": time.perf_counter() - epoch_started,
                "best_metric": best_metric,
                "improved": improved,
            }
            append_jsonl(log_path, record)
            print(
                "epoch={:03d} train_loss={:.4f} dev_loss={:.4f} "
                "dev_bleu={:.2f} lr={:.6g} time={:.1f}s".format(
                    epoch,
                    train_loss,
                    dev_loss,
                    dev_bleu,
                    learning_rate,
                    record["duration_seconds"],
                ),
                flush=True,
            )

            checkpoint_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_metric": best_metric,
                "bad_epochs": bad_epochs,
                "config": config,
                "source_vocab": source_vocab.to_dict(),
                "target_vocab": target_vocab.to_dict(),
                "seed": seed,
            }
            atomic_torch_save(checkpoint_dir / "latest.pt", checkpoint_payload)
            if improved:
                atomic_torch_save(checkpoint_dir / "best.pt", checkpoint_payload)

            if bad_epochs >= int(training_config["early_stopping_patience"]):
                print("Early stopping after {} non-improving epoch(s).".format(bad_epochs))
                break

        metadata.update(
            {
                "status": "completed",
                "finished_at": utc_now(),
                "training_seconds": time.perf_counter() - started,
                "best_metric_name": checkpoint_metric,
                "best_metric": best_metric,
                "best_checkpoint": str(checkpoint_dir / "best.pt"),
                "log_path": str(log_path),
            }
        )
        atomic_json_dump(metadata_path, metadata)
        append_jsonl(log_path, {"event": "run_end", **metadata})
    except BaseException as exc:
        metadata.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "training_seconds": time.perf_counter() - started,
                "error": "{}: {}".format(type(exc).__name__, exc),
            }
        )
        atomic_json_dump(metadata_path, metadata)
        append_jsonl(log_path, {"event": "run_end", **metadata})
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

