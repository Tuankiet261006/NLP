from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DataPipelineTest(unittest.TestCase):
    def test_prepare_and_validate_small_parallel_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            raw = root / "raw"
            processed = root / "processed"
            manifest = root / "manifest.json"
            raw.mkdir()
            samples = {
                "train.en": ["hello world", "how are you"],
                "train.vi": ["xin chào thế giới", "bạn khỏe không"],
                "tst2012.en": ["good morning"],
                "tst2012.vi": ["chào buổi sáng"],
                "tst2013.en": ["see you"],
                "tst2013.vi": ["hẹn gặp lại"],
            }
            for name, lines in samples.items():
                (raw / name).write_text(
                    "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
                )

            prepare = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "shared/scripts/prepare_data.py"),
                    "--input-dir",
                    str(raw),
                    "--output-dir",
                    str(processed),
                    "--manifest",
                    str(manifest),
                    "--max-train-tokens",
                    "32",
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(prepare.returncode, 0, prepare.stderr)

            validate = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "shared/scripts/validate_data.py"),
                    "--data-dir",
                    str(processed),
                    "--manifest",
                    str(manifest),
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(validate.returncode, 0, validate.stdout + validate.stderr)
            self.assertIn("DATA VALIDATION PASSED", validate.stdout)

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["direction"], "en-vi")
            self.assertEqual(payload["splits"]["train"]["output_pairs"], 2)
            self.assertTrue((processed / "vocab.en").is_file())
            self.assertTrue((processed / "vocab.vi").is_file())


if __name__ == "__main__":
    unittest.main()

