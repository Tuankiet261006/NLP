#!/usr/bin/env python
"""Run the vendored TensorFlow NMT code with fixed, recorded settings.

This wrapper is intentionally Python 3.7 compatible because TensorFlow 1.15
does not provide wheels for current Python releases.
"""

from __future__ import print_function

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = PROJECT_ROOT / "tensorflow_nmt"
VENDOR_ROOT = METHOD_ROOT / "vendor" / "nmt"
DATA_ROOT = PROJECT_ROOT / "shared" / "data" / "processed"
MANIFEST_PATH = PROJECT_ROOT / "shared" / "data" / "manifest.json"
TRAIN_EPOCHS = 10
TRAIN_STEPS_PER_EPOCH = 1000
NUM_TRAIN_STEPS = TRAIN_EPOCHS * TRAIN_STEPS_PER_EPOCH
DEFAULT_BEAM_WIDTH = 3


def configure_console():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "infer"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and print the subprocess command without running it.",
    )
    parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    return parser.parse_args()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temp_name, str(path))
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def require_files(paths):
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required file(s): {}".format(", ".join(missing)))


def git_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(VENDOR_ROOT), "rev-parse", "HEAD"],
                universal_newlines=True,
            )
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit_file = METHOD_ROOT / "vendor" / "NMT_COMMIT.txt"
        if commit_file.is_file():
            for line in commit_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("commit="):
                    return line.split("=", 1)[1]
        return None


def tensorflow_runtime():
    try:
        import tensorflow as tf

        return {
            "framework": "TensorFlow {}".format(tf.__version__),
            "device": "GPU" if tf.test.is_gpu_available(cuda_only=True) else "CPU",
        }
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Activate tf-nmt-legacy first."
        ) from exc


def common_training_args(out_dir, seed, data_root):
    prefix = str(data_root)
    return [
        "--src=en",
        "--tgt=vi",
        "--vocab_prefix={}".format(os.path.join(prefix, "vocab")),
        "--train_prefix={}".format(os.path.join(prefix, "train")),
        "--dev_prefix={}".format(os.path.join(prefix, "tst2012")),
        "--test_prefix={}".format(os.path.join(prefix, "tst2013")),
        "--out_dir={}".format(str(out_dir)),
        "--attention=scaled_luong",
        "--encoder_type=bi",
        "--num_layers=2",
        "--num_units=512",
        "--dropout=0.2",
        "--batch_size=128",
        "--optimizer=sgd",
        "--learning_rate=1.0",
        "--decay_scheme=luong234",
        "--num_train_steps={}".format(NUM_TRAIN_STEPS),
        "--steps_per_stats=100",
        "--src_max_len=160",
        "--tgt_max_len=160",
        "--src_max_len_infer=160",
        "--tgt_max_len_infer=160",
        "--infer_mode=beam_search",
        "--beam_width={}".format(DEFAULT_BEAM_WIDTH),
        "--metrics=bleu",
        "--random_seed={}".format(seed),
    ]


def build_command(mode, seed, beam_width, data_root):
    command = [sys.executable, "-m", "nmt.nmt"]
    if mode == "smoke":
        out_dir = METHOD_ROOT / "checkpoints" / "smoke"
        arguments = common_training_args(out_dir, seed, data_root)
        replacements = {
            "--num_units=512": "--num_units=128",
            "--num_train_steps={}".format(NUM_TRAIN_STEPS): "--num_train_steps=20",
            "--steps_per_stats=100": "--steps_per_stats=10",
        }
        arguments = [replacements.get(value, value) for value in arguments]
        arguments.append("--max_train=500")
        return command + arguments, out_dir
    if mode == "train":
        out_dir = METHOD_ROOT / "checkpoints" / "baseline"
        return command + common_training_args(out_dir, seed, data_root), out_dir

    out_dir = METHOD_ROOT / "checkpoints" / "baseline"
    output_file = METHOD_ROOT / "outputs" / "test.hyp.vi"
    arguments = [
        "--out_dir={}".format(str(out_dir)),
        "--inference_input_file={}".format(str(data_root / "tst2013.en")),
        "--inference_output_file={}".format(str(output_file)),
        "--infer_mode=beam_search",
        "--beam_width={}".format(beam_width),
    ]
    return command + arguments, out_dir


def preflight(mode, out_dir, data_root, manifest_path):
    require_files(
        [
            VENDOR_ROOT / "nmt" / "nmt.py",
            data_root / "train.en",
            data_root / "train.vi",
            data_root / "tst2012.en",
            data_root / "tst2012.vi",
            data_root / "tst2013.en",
            data_root / "tst2013.vi",
            data_root / "vocab.en",
            data_root / "vocab.vi",
            manifest_path,
        ]
    )
    if mode == "infer" and not out_dir.is_dir():
        raise FileNotFoundError("Checkpoint directory not found: {}".format(out_dir))


def count_logged_parameters(log_path):
    if not log_path.is_file():
        return None
    total = 0
    found = False
    shape_pattern = re.compile(r",\s*\(([0-9,\s]+)\),")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = shape_pattern.search(line)
        if not match:
            continue
        dimensions = [
            int(value.strip())
            for value in match.group(1).split(",")
            if value.strip()
        ]
        if dimensions:
            count = 1
            for dimension in dimensions:
                count *= dimension
            total += count
            found = True
    return total if found else None


def run_and_tee(command, cwd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_stream.write(line)
        return process.wait()


def main():
    configure_console()
    args = parse_args()
    data_root = args.data_dir.resolve()
    manifest_path = args.manifest.resolve()
    command, out_dir = build_command(
        args.mode, args.seed, args.beam_width, data_root
    )
    preflight(args.mode, out_dir, data_root, manifest_path)
    print("Working directory: {}".format(VENDOR_ROOT))
    print("Command:")
    print(" ".join('"{}"'.format(part) if " " in part else part for part in command))
    if args.dry_run:
        print("Dry run passed.")
        return 0

    runtime = tensorflow_runtime()
    run_id = time.strftime("%Y%m%dT%H%M%S")
    log_path = METHOD_ROOT / "logs" / "{}_{}.log".format(args.mode, run_id)
    metadata_path = METHOD_ROOT / "outputs" / "run_metadata_{}.json".format(
        args.mode
    )
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "method": "tensorflow_nmt",
        "mode": args.mode,
        "status": "running",
        "started_at": utc_now(),
        "seed": args.seed,
        "framework": runtime["framework"],
        "device": runtime["device"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_commit": git_commit(),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "command": command,
        "output_directory": str(out_dir),
        "log_path": str(log_path),
    }
    atomic_json_dump(metadata_path, metadata)

    if args.mode in ("smoke", "train"):
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        (METHOD_ROOT / "outputs").mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    return_code = run_and_tee(command, VENDOR_ROOT, log_path)
    duration = time.perf_counter() - started
    metadata.update(
        {
            "finished_at": utc_now(),
            "duration_seconds": duration,
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "trainable_parameters": count_logged_parameters(log_path),
        }
    )

    if args.mode == "train":
        metadata["training_seconds"] = duration
    if args.mode == "infer":
        training_metadata_path = METHOD_ROOT / "outputs" / "run_metadata_train.json"
        training_metadata = {}
        if training_metadata_path.is_file():
            training_metadata = json.loads(
                training_metadata_path.read_text(encoding="utf-8")
            )
        source_count = len((data_root / "tst2013.en").read_text(encoding="utf-8").splitlines())
        output_path = METHOD_ROOT / "outputs" / "test.hyp.vi"
        output_count = (
            len(output_path.read_text(encoding="utf-8").splitlines())
            if output_path.is_file()
            else 0
        )
        metadata.update(
            {
                "inference_seconds": duration,
                "sentence_count": output_count,
                "source_sentence_count": source_count,
                "sentences_per_second": output_count / duration if duration else None,
                "beam_width": args.beam_width,
                "training_seconds": training_metadata.get("training_seconds"),
                "training_run_id": training_metadata.get("run_id"),
            }
        )
        if metadata.get("trainable_parameters") is None:
            metadata["trainable_parameters"] = training_metadata.get(
                "trainable_parameters"
            )
        if return_code == 0 and output_count != source_count:
            metadata["status"] = "failed"
            metadata["error"] = "Inference output line count does not match source"
            return_code = 1

    atomic_json_dump(metadata_path, metadata)
    print("Metadata: {}".format(metadata_path))
    return return_code


if __name__ == "__main__":
    sys.exit(main())

