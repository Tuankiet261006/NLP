#!/usr/bin/env python
"""Download IWSLT 2015 en-vi with verified Stanford/GitHub sources."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
import hashlib
from typing import Iterable


STANFORD_BASE = "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi"
RAW_FILES = (
    "train.en",
    "train.vi",
    "tst2012.en",
    "tst2012.vi",
    "tst2013.en",
    "tst2013.vi",
)
GITHUB_BASE = (
    "https://raw.githubusercontent.com/KCDichDaNgu/"
    "KC4.0_MultilingualNMT/master/data/iwslt_en_vi"
)
EXPECTED_SHA256 = {
    "train.en": "c26dfeed74b6bf3752f5ca552f2412456f0de153f7c804df8717931fb3a5c78a",
    "train.vi": "707206edf2dc0280273952c7b70544ea8a1363aa69aaeb9d70514b888dc3067d",
    "tst2012.en": "b8fbcd0d4199d41276421d1f7228298df67dd9e2c5c134c2ee8cdf3ec27801e2",
    "tst2012.vi": "01004078b3f36b0c46e81b65fc6851230b3c823405c76d1d82ef3b786b512ff0",
    "tst2013.en": "3860ce60bc5b85d96cc2d3d580cf1bc9d62440e939c1b964676d3607c4e8f1df",
    "tst2013.vi": "29af16842808da2812b8fd4d47fc05d082ba870e60ab7755090ad04b380994cd",
}


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source",
        choices=("auto", "stanford", "github"),
        default="auto",
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
                stream.write(line.replace("\r", " ").replace("\n", " ").strip())
                stream.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_files(output_dir: Path, base_url: str, source_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        destination = output_dir / name
        temporary = destination.with_suffix(destination.suffix + ".part")
        request = urllib.request.Request(
            base_url + "/" + name,
            headers={"User-Agent": "Mozilla/5.0 IWSLT-downloader/1.0"},
        )
        print("Downloading {} {}...".format(source_name, name))
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                with temporary.open("wb") as stream:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
            if temporary.stat().st_size == 0:
                raise RuntimeError("Downloaded file is empty: {}".format(name))
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()


def verify_files(output_dir: Path) -> None:
    for name, expected in EXPECTED_SHA256.items():
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                "{} checksum mismatch: expected={} actual={}".format(
                    name, expected, actual
                )
            )
    for language in ("en", "vi"):
        dev = (output_dir / "tst2012.{}".format(language)).read_bytes()
        test = (output_dir / "tst2013.{}".format(language)).read_bytes()
        if dev == test:
            raise ValueError("tst2012.{} and tst2013.{} are identical".format(language, language))


def main() -> int:
    configure_console()
    args = parse_args()
    output_dir = args.output_dir.resolve()
    source_used = args.source
    if args.source in ("auto", "stanford"):
        try:
            download_files(output_dir, STANFORD_BASE, "Stanford")
            source_used = "stanford"
        except Exception as exc:
            if args.source == "stanford":
                raise
            print(
                "WARNING: Stanford download failed: {}: {}".format(
                    type(exc).__name__, exc
                )
            )
            print("Falling back to the checksum-matched GitHub mirror.")
            download_files(output_dir, GITHUB_BASE, "GitHub")
            source_used = "github"
    else:
        download_files(output_dir, GITHUB_BASE, "GitHub")

    verify_files(output_dir)
    print("IWSLT 2015 ready in {}".format(output_dir))
    print("Source used: {}".format(source_used))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

