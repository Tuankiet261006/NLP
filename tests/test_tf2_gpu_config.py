from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TensorFlowGpuConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        config_path = PROJECT_ROOT / "tensorflow_nmt/configs/tf2_gpu.json"
        self.config = json.loads(config_path.read_text(encoding="utf-8"))

    def test_requested_epoch_and_beam_settings(self) -> None:
        self.assertEqual(self.config["training"]["epochs"], 10)
        self.assertEqual(self.config["decoding"]["beam_width"], 3)
        self.assertTrue(self.config["training"]["mixed_precision"])

    def test_shared_splits_and_gpu_entry_points_exist(self) -> None:
        data = self.config["data"]
        self.assertIn("tst2012", data["dev_src"])
        self.assertIn("tst2013", data["test_src"])
        for relative in (
            "tensorflow_nmt/tf2_train.py",
            "tensorflow_nmt/tf2_infer.py",
            "tensorflow_nmt/train_gpu.ps1",
            "tensorflow_nmt/infer_gpu.ps1",
        ):
            self.assertTrue((PROJECT_ROOT / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
