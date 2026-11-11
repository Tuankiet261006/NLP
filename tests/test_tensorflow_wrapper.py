from __future__ import annotations

import unittest

from tensorflow_nmt.run_nmt import DATA_ROOT, build_command


class TensorFlowWrapperTest(unittest.TestCase):
    def test_train_command_has_fixed_direction_and_split(self) -> None:
        command, _ = build_command("train", 42, 3, DATA_ROOT)
        joined = " ".join(command)
        self.assertIn("--src=en", joined)
        self.assertIn("--tgt=vi", joined)
        self.assertIn("--dev_prefix=", joined)
        self.assertIn("tst2012", joined)
        self.assertIn("--test_prefix=", joined)
        self.assertIn("tst2013", joined)
        self.assertIn("--attention=scaled_luong", joined)
        self.assertIn("--random_seed=42", joined)
        self.assertIn("--num_train_steps=10000", joined)
        self.assertIn("--beam_width=3", joined)

    def test_inference_command_uses_beam_three(self) -> None:
        command, _ = build_command("infer", 42, 3, DATA_ROOT)
        joined = " ".join(command)
        self.assertIn("--infer_mode=beam_search", joined)
        self.assertIn("--beam_width=3", joined)


if __name__ == "__main__":
    unittest.main()

