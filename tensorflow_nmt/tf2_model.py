"""TensorFlow 2 implementation of bidirectional LSTM NMT with Luong attention."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import tensorflow as tf


DecoderStates = Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]


class ScaledLuongAttention(tf.keras.layers.Layer):
    def __init__(self, units: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.units = units
        self.key_projection = tf.keras.layers.Dense(units, use_bias=False, name="memory_projection")

    def build(self, input_shape: Any) -> None:
        self.scale = self.add_weight(
            name="attention_scale",
            shape=(),
            initializer="ones",
            trainable=True,
            dtype=tf.float32,
        )
        super().build(input_shape)

    def project_memory(self, memory: tf.Tensor) -> tf.Tensor:
        return self.key_projection(memory)

    def call(
        self,
        query: tf.Tensor,
        memory: tf.Tensor,
        memory_mask: tf.Tensor,
        projected_memory: tf.Tensor | None = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        keys = projected_memory if projected_memory is not None else self.project_memory(memory)
        scores = tf.matmul(query, keys, transpose_b=True)
        scores = tf.cast(scores, tf.float32) * tf.cast(self.scale, tf.float32)
        valid = tf.expand_dims(memory_mask, axis=1)
        scores = tf.where(valid, scores, tf.constant(-1.0e9, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(tf.cast(weights, memory.dtype), memory)
        return context, weights


class LuongSeq2Seq(tf.keras.Model):
    """One bidirectional encoder layer feeding a two-layer LSTM decoder.

    The legacy tensorflow/nmt implementation interprets num_layers=2 with a
    bidirectional encoder as one forward and one backward encoder layer. Their
    states initialize the two decoder layers respectively.
    """

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        source_pad_id: int,
        target_pad_id: int,
        config: Dict[str, Any],
    ):
        super().__init__(name="tf2_luong_nmt")
        model_config = config["model"]
        units = int(model_config["units"])
        embedding_dim = int(model_config["embedding_dim"])
        dropout = float(model_config["dropout"])
        if int(model_config["decoder_layers"]) != 2:
            raise ValueError("This baseline requires exactly two decoder layers")
        if model_config["encoder_type"] != "bidirectional":
            raise ValueError("This baseline requires a bidirectional encoder")
        if model_config["attention"] != "scaled_luong":
            raise ValueError("This baseline requires scaled_luong attention")

        self.units = units
        self.source_pad_id = source_pad_id
        self.target_pad_id = target_pad_id
        self.source_embedding = tf.keras.layers.Embedding(
            source_vocab_size, embedding_dim, name="source_embedding"
        )
        self.target_embedding = tf.keras.layers.Embedding(
            target_vocab_size, embedding_dim, name="target_embedding"
        )
        self.embedding_dropout = tf.keras.layers.Dropout(dropout)
        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units,
                return_sequences=True,
                return_state=True,
                recurrent_dropout=0.0,
                name="encoder_lstm",
            ),
            merge_mode="concat",
            name="bidirectional_encoder",
        )
        self.decoder_layer_1 = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_dropout=0.0,
            name="decoder_lstm_1",
        )
        self.decoder_layer_2 = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_dropout=0.0,
            name="decoder_lstm_2",
        )
        self.decoder_dropout = tf.keras.layers.Dropout(dropout)
        self.attention = ScaledLuongAttention(units, name="scaled_luong_attention")
        self.attention_output = tf.keras.layers.Dense(
            units, activation="tanh", use_bias=False, name="attention_output"
        )
        self.output_projection = tf.keras.layers.Dense(
            target_vocab_size,
            name="output_projection",
            dtype="float32",
        )

    def encode(
        self, source: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, DecoderStates]:
        source_mask = tf.not_equal(source, self.source_pad_id)
        embedded = self.source_embedding(source)
        embedded = self.embedding_dropout(embedded, training=training)
        encoded = self.encoder(embedded, mask=source_mask, training=training)
        memory, forward_h, forward_c, backward_h, backward_c = encoded
        projected_memory = self.attention.project_memory(memory)
        states: DecoderStates = (
            (forward_h, forward_c),
            (backward_h, backward_c),
        )
        return memory, projected_memory, source_mask, states

    def attend(
        self,
        query: tf.Tensor,
        memory: tf.Tensor,
        projected_memory: tf.Tensor,
        memory_mask: tf.Tensor,
    ) -> tf.Tensor:
        context, _ = self.attention(
            query,
            memory,
            memory_mask,
            projected_memory=projected_memory,
        )
        combined = tf.concat([query, context], axis=-1)
        return self.attention_output(combined)

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        source, target_input = inputs
        memory, projected_memory, source_mask, states = self.encode(source, training=training)
        target_mask = tf.not_equal(target_input, self.target_pad_id)
        embedded = self.target_embedding(target_input)
        embedded = self.embedding_dropout(embedded, training=training)
        decoder_1, _, _ = self.decoder_layer_1(
            embedded,
            mask=target_mask,
            initial_state=list(states[0]),
            training=training,
        )
        decoder_1 = self.decoder_dropout(decoder_1, training=training)
        decoder_2, _, _ = self.decoder_layer_2(
            decoder_1,
            mask=target_mask,
            initial_state=list(states[1]),
            training=training,
        )
        decoder_2 = self.decoder_dropout(decoder_2, training=training)
        attended = self.attend(decoder_2, memory, projected_memory, source_mask)
        return self.output_projection(attended)

    def decode_step(
        self,
        token: tf.Tensor,
        memory: tf.Tensor,
        projected_memory: tf.Tensor,
        memory_mask: tf.Tensor,
        states: DecoderStates,
    ) -> Tuple[tf.Tensor, DecoderStates]:
        embedded = self.target_embedding(tf.expand_dims(token, axis=1))
        decoder_1, state_1_h, state_1_c = self.decoder_layer_1(
            embedded, initial_state=list(states[0]), training=False
        )
        decoder_2, state_2_h, state_2_c = self.decoder_layer_2(
            decoder_1, initial_state=list(states[1]), training=False
        )
        attended = self.attend(decoder_2, memory, projected_memory, memory_mask)
        logits = self.output_projection(attended)
        next_states: DecoderStates = (
            (state_1_h, state_1_c),
            (state_2_h, state_2_c),
        )
        return logits[:, 0, :], next_states


def build_model(
    config: Dict[str, Any],
    source_vocab_size: int,
    target_vocab_size: int,
    source_pad_id: int,
    target_pad_id: int,
) -> LuongSeq2Seq:
    model = LuongSeq2Seq(
        source_vocab_size,
        target_vocab_size,
        source_pad_id,
        target_pad_id,
        config,
    )
    source = tf.constant([[2]], dtype=tf.int32)
    target = tf.constant([[1]], dtype=tf.int32)
    model((source, target), training=False)
    return model


def beam_search_decode(
    model: LuongSeq2Seq,
    source_ids: Sequence[int],
    bos_id: int,
    eos_id: int,
    beam_width: int,
    max_length: int,
    length_penalty: float,
) -> List[int]:
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    source = tf.constant([list(source_ids)], dtype=tf.int32)
    memory, projected_memory, memory_mask, initial_states = model.encode(source, training=False)
    beams: List[Tuple[List[int], float, DecoderStates]] = [
        ([bos_id], 0.0, initial_states)
    ]
    completed: List[Tuple[List[int], float, DecoderStates]] = []

    def normalized_score(item: Tuple[List[int], float, DecoderStates]) -> float:
        tokens, score, _ = item
        generated_length = max(1, len(tokens) - 1)
        penalty = ((5.0 + generated_length) / 6.0) ** length_penalty
        return score / penalty

    for _ in range(max_length):
        candidates: List[Tuple[List[int], float, DecoderStates]] = []
        for tokens, score, states in beams:
            if tokens[-1] == eos_id:
                completed.append((tokens, score, states))
                continue
            token = tf.constant([tokens[-1]], dtype=tf.int32)
            logits, next_states = model.decode_step(
                token, memory, projected_memory, memory_mask, states
            )
            log_probabilities = tf.nn.log_softmax(logits[0], axis=-1)
            values, indices = tf.math.top_k(log_probabilities, k=beam_width)
            for value, index in zip(values.numpy().tolist(), indices.numpy().tolist()):
                candidates.append((tokens + [int(index)], score + float(value), next_states))
        if not candidates:
            break
        candidates.sort(key=normalized_score, reverse=True)
        beams = candidates[:beam_width]
        if all(tokens[-1] == eos_id for tokens, _, _ in beams):
            completed.extend(beams)
            break

    choices = completed or beams
    choices.sort(key=normalized_score, reverse=True)
    return choices[0][0][1:]
