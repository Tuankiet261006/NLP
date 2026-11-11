from __future__ import annotations

import unittest

import torch

from transformer_colab.src.data import Vocab, pad_sequences
from transformer_colab.src.model import Seq2SeqTransformer, beam_search_decode


class TransformerComponentsTest(unittest.TestCase):
    def test_model_forward_and_beam_search(self) -> None:
        source_vocab = Vocab.build(["hello world"], min_frequency=1)
        target_vocab = Vocab.build(["xin chào"], min_frequency=1)
        model = Seq2SeqTransformer(
            source_vocab_size=len(source_vocab),
            target_vocab_size=len(target_vocab),
            source_pad_id=source_vocab.pad_id,
            target_pad_id=target_vocab.pad_id,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            max_position_embeddings=32,
        )
        source = pad_sequences(
            [source_vocab.encode("hello world")], source_vocab.pad_id
        )
        target = pad_sequences(
            [target_vocab.encode("xin chào")], target_vocab.pad_id
        )
        logits = model(source, target[:, :-1])
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], target.shape[1] - 1)
        self.assertEqual(logits.shape[2], len(target_vocab))

        prediction = beam_search_decode(
            model,
            source[0],
            bos_id=target_vocab.bos_id,
            eos_id=target_vocab.eos_id,
            beam_width=2,
            max_length=5,
            length_penalty=0.6,
        )
        self.assertEqual(prediction[0], target_vocab.bos_id)
        self.assertLessEqual(len(prediction), 6)


if __name__ == "__main__":
    unittest.main()

