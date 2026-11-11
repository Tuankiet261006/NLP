"""Shared helpers for the Transformer training and inference entry points."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> Dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ("seed", "data", "model", "training", "decoding", "artifacts")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("Config is missing keys: {}".format(", ".join(missing)))
    return config


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(path: Path, payload: Dict[str, Any]) -> None:
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


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def device_description(device: torch.device) -> str:
    if device.type == "cuda":
        return "{} ({})".format(device, torch.cuda.get_device_name(device))
    return str(device)


def runtime_metadata(device: torch.device) -> Dict[str, Any]:
    return {
        "python": os.sys.version,
        "framework": "PyTorch {}".format(torch.__version__),
        "device": device_description(device),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
    }


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

