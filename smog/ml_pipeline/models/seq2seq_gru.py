"""Shared seq2seq GRU utilities with attention and MC dropout."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    GRU,
    Input,
    Layer,
    RepeatVector,
    Reshape,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

@tf.keras.utils.register_keras_serializable(package="AtmosCare")
class MCDropout(Dropout):
    """Dropout layer that can be toggled on for inference-time uncertainty."""

    def __init__(self, rate: float, mc_dropout: bool = False, **kwargs):
        super().__init__(rate, **kwargs)
        self.mc_dropout = bool(mc_dropout)

    def call(self, inputs, training=None):
        training = self.mc_dropout if training is None else bool(training or self.mc_dropout)
        return super().call(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"mc_dropout": self.mc_dropout})
        return config


@tf.keras.utils.register_keras_serializable(package="AtmosCare")
class BahdanauAttention(Layer):
    """Additive attention for decoder-to-encoder alignment."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.W1 = Dense(self.units, use_bias=False)
        self.W2 = Dense(self.units, use_bias=False)
        self.V = Dense(1, use_bias=False)

    def build(self, input_shape):
        query_shape, values_shape = input_shape
        self.W1.build(values_shape)
        self.W2.build(query_shape)
        self.V.build(tf.TensorShape((*query_shape[:-1], self.units)))
        super().build(input_shape)

    def call(self, inputs):
        query, values = inputs
        query_time = tf.expand_dims(query, axis=2)
        values_time = tf.expand_dims(values, axis=1)
        score = self.V(tf.nn.tanh(self.W1(values_time) + self.W2(query_time)))
        attention_weights = tf.nn.softmax(score, axis=2)
        context_vector = tf.reduce_sum(attention_weights * values_time, axis=2)
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def _as_list(values: Optional[Sequence[int]], fallback: Sequence[int]) -> list[int]:
    if values is None:
        return list(fallback)
    return [int(v) for v in values]


def toggle_mc_dropout(model: tf.keras.Model, enabled: bool) -> None:
    """Enable or disable MC dropout recursively without touching BatchNorm."""
    for layer in model.layers:
        if isinstance(layer, MCDropout):
            layer.mc_dropout = bool(enabled)
        if hasattr(layer, "layers") and layer.layers:
            toggle_mc_dropout(layer, enabled)


def build_seq2seq_attention_gru(
    *,
    sequence_length: int,
    forecast_horizon: int,
    n_features: int,
    gru_units: Sequence[int] = (128, 64),
    dense_units: Sequence[int] = (64, 32),
    dropout_rate: float = 0.15,
    learning_rate: float = 0.001,
    loss: str = "huber",
    l2_rate: float = 1e-4,
    name: str = "Seq2Seq_GRU_Attention",
) -> tf.keras.Model:
    """Build an encoder-decoder GRU with Bahdanau attention."""
    gru_units = _as_list(gru_units, (128, 64))
    dense_units = _as_list(dense_units, (64, 32))
    encoder_units = int(gru_units[0])
    decoder_units = int(gru_units[-1])
    attention_units = max(decoder_units, int(gru_units[0]))

    inputs = Input(shape=(sequence_length, n_features), name="encoder_input")

    encoder_seq = GRU(
        encoder_units,
        return_sequences=True,
        return_state=True,
        kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
        name="encoder_gru_1",
    )(inputs)
    encoder_outputs, encoder_state = encoder_seq
    encoder_outputs = BatchNormalization(name="encoder_bn_1")(encoder_outputs)
    encoder_outputs = MCDropout(dropout_rate, name="encoder_dropout_1")(encoder_outputs)

    encoder_seq_2 = GRU(
        decoder_units,
        return_sequences=True,
        return_state=True,
        kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
        name="encoder_gru_2",
    )(encoder_outputs)
    encoder_outputs_2, encoder_state_2 = encoder_seq_2
    encoder_outputs_2 = BatchNormalization(name="encoder_bn_2")(encoder_outputs_2)
    encoder_outputs_2 = MCDropout(dropout_rate, name="encoder_dropout_2")(encoder_outputs_2)

    decoder_inputs = RepeatVector(forecast_horizon, name="decoder_repeat")(encoder_state_2)
    decoder_seq = GRU(
        decoder_units,
        return_sequences=True,
        kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
        name="decoder_gru",
    )(decoder_inputs, initial_state=encoder_state_2)
    decoder_seq = BatchNormalization(name="decoder_bn")(decoder_seq)
    decoder_seq = MCDropout(dropout_rate, name="decoder_dropout")(decoder_seq)

    context = BahdanauAttention(attention_units, name="bahdanau_attention")(
        [decoder_seq, encoder_outputs_2]
    )
    fused = Concatenate(name="attention_fusion")([decoder_seq, context])

    x = Dense(
        dense_units[0],
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
        name="dense_1",
    )(fused)
    x = BatchNormalization(name="dense_bn_1")(x)
    x = MCDropout(dropout_rate, name="dense_dropout_1")(x)

    if len(dense_units) > 1:
        x = Dense(
            dense_units[1],
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
            name="dense_2",
        )(x)
        x = BatchNormalization(name="dense_bn_2")(x)
        x = MCDropout(dropout_rate, name="dense_dropout_2")(x)

    outputs = Dense(1, name="forecast_token")(x)
    outputs = Reshape((forecast_horizon,), name="forecast_output")(outputs)

    model = Model(inputs=inputs, outputs=outputs, name=name)

    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True,
    )
    loss_obj = Huber(delta=1.0) if str(loss).lower() == "huber" else loss
    model.compile(
        optimizer=optimizer,
        loss=loss_obj,
        metrics=["mae", "mse", tf.keras.metrics.RootMeanSquaredError()],
    )
    return model


def mc_dropout_predict(
    model: tf.keras.Model,
    inputs: np.ndarray,
    n_samples: int = 50,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run stochastic forward passes with dropout enabled only on MCDropout layers."""
    inputs = np.asarray(inputs, dtype=np.float32)
    samples = []
    toggle_mc_dropout(model, True)
    try:
        for _ in range(int(n_samples)):
            preds = model(inputs, training=False)
            samples.append(np.asarray(preds, dtype=float))
    finally:
        toggle_mc_dropout(model, False)

    stacked = np.stack(samples, axis=0)
    mean = stacked.mean(axis=0)
    lower = np.percentile(stacked, 2.5, axis=0)
    upper = np.percentile(stacked, 97.5, axis=0)
    return mean, lower, upper, stacked
