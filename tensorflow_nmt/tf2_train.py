"""Train the TensorFlow 2 English-Vietnamese Luong NMT model on GPU."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sacrebleu
import tensorflow as tf

from .tf2_common import (
    PROJECT_ROOT,
    append_jsonl,
    artifact_paths,
    atomic_json_dump,
    atomic_write_lines,
    configure_console,
    configure_runtime,
    count_lines,
    load_config,
    load_vocabularies,
    make_parallel_dataset,
    read_lines,
    reference_nmt_commit,
    resolve_project_path,
    runtime_metadata,
    sha256_file,
    utc_now,
)
from .tf2_model import LuongSeq2Seq, beam_search_decode, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "tensorflow_nmt/configs/tf2_gpu.json",
    )
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def masked_loss(
    target: tf.Tensor, logits: tf.Tensor, pad_id: int
) -> tf.Tensor:
    per_token = tf.keras.losses.sparse_categorical_crossentropy(
        target, logits, from_logits=True
    )
    mask = tf.cast(tf.not_equal(target, pad_id), per_token.dtype)
    return tf.reduce_sum(per_token * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def learning_rate_for_epoch(epoch: int, training: Dict[str, Any]) -> float:
    base = float(training["learning_rate"])
    start = int(training["decay_start_epoch"])
    factor = float(training["decay_factor"])
    decay_count = max(0, epoch - start)
    return base * (factor ** decay_count)


def evaluate_dev_loss(
    model: LuongSeq2Seq,
    dataset: tf.data.Dataset,
    pad_id: int,
) -> float:
    metric = tf.keras.metrics.Mean()

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        )
    )
    def evaluate_step(
        source: tf.Tensor, target_input: tf.Tensor, target_output: tf.Tensor
    ) -> tf.Tensor:
        logits = model((source, target_input), training=False)
        return masked_loss(target_output, logits, pad_id)

    for source, target_input, target_output in dataset:
        metric.update_state(evaluate_step(source, target_input, target_output))
    return float(metric.result().numpy())


def evaluate_dev_bleu(
    model: LuongSeq2Seq,
    source_lines: List[str],
    reference_lines: List[str],
    source_vocab: Any,
    target_vocab: Any,
    config: Dict[str, Any],
    maximum: int,
) -> Tuple[float, List[str]]:
    decoding = config["decoding"]
    max_input_length = int(config["data"]["max_length"])
    hypotheses: List[str] = []
    references = reference_lines[:maximum]
    for source in source_lines[:maximum]:
        source_ids = source_vocab.encode_source(source, max_input_length)
        prediction = beam_search_decode(
            model,
            source_ids,
            target_vocab.bos_id,
            target_vocab.eos_id,
            int(decoding["beam_width"]),
            int(decoding["max_length"]),
            float(decoding["length_penalty"]),
        )
        hypotheses.append(target_vocab.decode(prediction))
    score = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="13a").score
    return float(score), hypotheses


def main() -> int:
    configure_console()
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    seed = int(config["seed"])
    training = config["training"]
    require_gpu = args.device == "gpu"
    policy = configure_runtime(
        seed,
        require_gpu=require_gpu,
        mixed_precision=bool(training["mixed_precision"]),
    )

    source_vocab, target_vocab = load_vocabularies(config)
    data = config["data"]
    train_source_path = resolve_project_path(data["train_src"])
    train_target_path = resolve_project_path(data["train_tgt"])
    dev_source_path = resolve_project_path(data["dev_src"])
    dev_target_path = resolve_project_path(data["dev_tgt"])
    max_length = int(data["max_length"])
    batch_size = int(training["batch_size"])
    max_train_samples = 256 if args.smoke else None
    max_dev_samples = 32 if args.smoke else None

    train_dataset = make_parallel_dataset(
        train_source_path,
        train_target_path,
        source_vocab,
        target_vocab,
        max_length,
        batch_size,
        shuffle=True,
        seed=seed,
        shuffle_buffer=int(training["shuffle_buffer"]),
        max_samples=max_train_samples,
    )
    dev_dataset = make_parallel_dataset(
        dev_source_path,
        dev_target_path,
        source_vocab,
        target_vocab,
        max_length,
        batch_size,
        shuffle=False,
        seed=seed,
        shuffle_buffer=1,
        max_samples=max_dev_samples,
    )

    model = build_model(
        config,
        len(source_vocab),
        len(target_vocab),
        source_vocab.pad_id,
        target_vocab.pad_id,
    )
    parameter_count = int(model.count_params())
    base_optimizer = tf.keras.optimizers.SGD(learning_rate=float(training["learning_rate"]))
    if policy == "mixed_float16":
        optimizer: Any = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
    else:
        optimizer = base_optimizer

    checkpoint_dir, log_dir, output_dir = artifact_paths(config, smoke=args.smoke)
    epoch_variable = tf.Variable(0, trainable=False, dtype=tf.int64, name="epoch")
    best_dev_loss = tf.Variable(math.inf, trainable=False, dtype=tf.float32, name="best_dev_loss")
    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch_variable,
        best_dev_loss=best_dev_loss,
    )
    latest_manager = tf.train.CheckpointManager(
        checkpoint, str(checkpoint_dir / "latest"), max_to_keep=3
    )
    best_manager = tf.train.CheckpointManager(
        checkpoint, str(checkpoint_dir / "best"), max_to_keep=1
    )
    if args.resume:
        if not latest_manager.latest_checkpoint:
            raise FileNotFoundError("No latest checkpoint in {}".format(checkpoint_dir / "latest"))
        checkpoint.restore(latest_manager.latest_checkpoint).expect_partial()
        print(
            "Resumed {} at epoch {}".format(
                latest_manager.latest_checkpoint, int(epoch_variable.numpy()) + 1
            ),
            flush=True,
        )

    estimated_train_examples = max_train_samples or count_lines(train_source_path)
    estimated_batches = math.ceil(estimated_train_examples / batch_size)
    print(
        "TensorFlow={} device={} policy={} parameters={:,} train_examples={} batches/epoch~{}".format(
            tf.__version__,
            runtime_metadata()["device"],
            policy,
            parameter_count,
            estimated_train_examples,
            estimated_batches,
        ),
        flush=True,
    )
    if args.dry_run:
        print("Dry run passed.")
        return 0

    clip_norm = float(training["gradient_clip_norm"])
    steps_per_stats = int(training["steps_per_stats"])

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        )
    )
    def train_step(
        source: tf.Tensor, target_input: tf.Tensor, target_output: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            logits = model((source, target_input), training=True)
            loss = masked_loss(target_output, logits, target_vocab.pad_id)
            scaled_loss = (
                optimizer.get_scaled_loss(loss)
                if policy == "mixed_float16"
                else loss
            )
        gradients = tape.gradient(scaled_loss, model.trainable_variables)
        if policy == "mixed_float16":
            gradients = optimizer.get_unscaled_gradients(gradients)
        gradient_variable_pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, model.trainable_variables)
            if gradient is not None
        ]
        clipped, global_norm = tf.clip_by_global_norm(
            [pair[0] for pair in gradient_variable_pairs], clip_norm
        )
        optimizer.apply_gradients(
            zip(clipped, [pair[1] for pair in gradient_variable_pairs])
        )
        return loss, global_norm

    run_id = time.strftime("%Y%m%dT%H%M%S")
    log_path = log_dir / "train_{}.jsonl".format(run_id)
    metadata_path = output_dir / "run_metadata.json"
    manifest_path = PROJECT_ROOT / "shared/data/manifest.json"
    metadata: Dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "method": "tensorflow_nmt_tf2",
        "mode": "smoke" if args.smoke else "train",
        "status": "running",
        "started_at": utc_now(),
        "seed": seed,
        "config_path": str(config_path),
        "config": config,
        "reference_nmt_commit": reference_nmt_commit(),
        "batch_size": batch_size,
        "trainable_parameters": parameter_count,
        "train_examples": estimated_train_examples,
        "dev_examples": max_dev_samples or count_lines(dev_source_path),
        "dataset_manifest_sha256": sha256_file(manifest_path)
        if manifest_path.is_file()
        else None,
        **runtime_metadata(),
    }
    atomic_json_dump(metadata_path, metadata)
    append_jsonl(log_path, {"event": "run_start", **metadata})
    dev_sources = read_lines(dev_source_path)
    dev_references = read_lines(dev_target_path)
    epochs = 1 if args.smoke else int(training["epochs"])
    start_epoch = int(epoch_variable.numpy()) + 1
    started = time.perf_counter()

    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_started = time.perf_counter()
            learning_rate = learning_rate_for_epoch(epoch, training)
            base_optimizer.learning_rate.assign(learning_rate)
            train_metric = tf.keras.metrics.Mean()
            norm_metric = tf.keras.metrics.Mean()
            for step, (source, target_input, target_output) in enumerate(train_dataset, start=1):
                loss, gradient_norm = train_step(source, target_input, target_output)
                if not bool(tf.math.is_finite(loss).numpy()):
                    raise FloatingPointError("Non-finite training loss at epoch {} step {}".format(epoch, step))
                train_metric.update_state(loss)
                norm_metric.update_state(gradient_norm)
                if step % steps_per_stats == 0:
                    print(
                        "epoch={} step={} loss={:.4f} grad_norm={:.3f}".format(
                            epoch,
                            step,
                            float(train_metric.result().numpy()),
                            float(norm_metric.result().numpy()),
                        ),
                        flush=True,
                    )

            dev_loss = evaluate_dev_loss(model, dev_dataset, target_vocab.pad_id)
            bleu_maximum = 5 if args.smoke else int(training["dev_bleu_max_sentences"])
            dev_bleu, hypotheses = evaluate_dev_bleu(
                model,
                dev_sources,
                dev_references,
                source_vocab,
                target_vocab,
                config,
                bleu_maximum,
            )
            atomic_write_lines(output_dir / "dev.hyp.vi", hypotheses)
            improved = dev_loss < float(best_dev_loss.numpy())
            epoch_variable.assign(epoch)
            if improved:
                best_dev_loss.assign(dev_loss)
            latest_path = latest_manager.save(checkpoint_number=epoch)
            best_path = best_manager.save(checkpoint_number=epoch) if improved else None
            record = {
                "event": "epoch",
                "timestamp": utc_now(),
                "epoch": epoch,
                "train_loss": float(train_metric.result().numpy()),
                "dev_loss": dev_loss,
                "dev_bleu": dev_bleu,
                "learning_rate": learning_rate,
                "gradient_norm": float(norm_metric.result().numpy()),
                "duration_seconds": time.perf_counter() - epoch_started,
                "improved": improved,
                "latest_checkpoint": latest_path,
                "best_checkpoint": best_path,
            }
            append_jsonl(log_path, record)
            print(
                "epoch={:03d} train_loss={:.4f} dev_loss={:.4f} dev_bleu={:.2f} "
                "lr={:.6g} time={:.1f}s".format(
                    epoch,
                    record["train_loss"],
                    dev_loss,
                    dev_bleu,
                    learning_rate,
                    record["duration_seconds"],
                ),
                flush=True,
            )

        metadata.update(
            {
                "status": "completed",
                "finished_at": utc_now(),
                "training_seconds": time.perf_counter() - started,
                "last_epoch": int(epoch_variable.numpy()),
                "best_dev_loss": float(best_dev_loss.numpy()),
                "best_checkpoint": best_manager.latest_checkpoint,
                "latest_checkpoint": latest_manager.latest_checkpoint,
                "log_path": str(log_path),
            }
        )
        atomic_json_dump(metadata_path, metadata)
        append_jsonl(log_path, {"event": "run_end", **metadata})
        print("Training completed. Best checkpoint: {}".format(best_manager.latest_checkpoint))
        return 0
    except BaseException as exc:
        metadata.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "training_seconds": time.perf_counter() - started,
                "error": "{}: {}".format(type(exc).__name__, exc),
                "last_epoch": int(epoch_variable.numpy()),
            }
        )
        atomic_json_dump(metadata_path, metadata)
        append_jsonl(log_path, {"event": "run_end", **metadata})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
