"""
CNN Branch for Local Temporal Motif Extraction.

Implements the 1-D convolutional branch of the physics-informed ensemble
framework described in:
  Abubakkar et al., "Interpretable Physics-Informed Load Forecasting for
  U.S. Grid Resilience", IEEE, 2026.

Architecture:
  Input (T=24, F=12) → Conv1D(k=3, c=64) → BN → ReLU → MaxPool(s=2)
                      → Conv1D(k=3, c=64) → BN → ReLU → MaxPool(s=2)
                      → GlobalAvgPool → Dense(64) [embedding z_CNN]
                      → Dense(1) [branch forecast ŷ_CNN]
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_cnn_branch(
    seq_len: int = 24,
    n_features: int = 12,
    filters: int = 64,
    kernel_size: int = 3,
    pool_stride: int = 2,
    embed_dim: int = 64,
    dropout_rate: float = 0.2,
) -> Model:
    """
    Build the CNN branch for local feature extraction.

    Parameters
    ----------
    seq_len : int
        Length of the input sequence window (default: 24 hours).
    n_features : int
        Number of input features per time step (default: 12).
    filters : int
        Number of convolutional filters per block (default: 64).
    kernel_size : int
        Convolutional kernel size (default: 3).
    pool_stride : int
        MaxPooling stride (default: 2).
    embed_dim : int
        Dimensionality of the branch embedding z_CNN (default: 64).
    dropout_rate : float
        Dropout probability applied after each block (default: 0.2).

    Returns
    -------
    tf.keras.Model
        Compiled CNN branch model with outputs (embedding, forecast).
    """
    inputs = layers.Input(shape=(seq_len, n_features), name="cnn_input")

    # ── Block 1: Conv1D → BN → ReLU → MaxPool ──────────────────────────────
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        name="conv1d_block1",
    )(inputs)
    x = layers.BatchNormalization(name="bn_block1")(x)
    x = layers.ReLU(name="relu_block1")(x)
    x = layers.MaxPooling1D(pool_size=pool_stride, name="maxpool_block1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_block1")(x)

    # ── Block 2: Conv1D → BN → ReLU → MaxPool ──────────────────────────────
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        name="conv1d_block2",
    )(x)
    x = layers.BatchNormalization(name="bn_block2")(x)
    x = layers.ReLU(name="relu_block2")(x)
    x = layers.MaxPooling1D(pool_size=pool_stride, name="maxpool_block2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_block2")(x)

    # ── Global Average Pooling → Dense embedding ────────────────────────────
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    embedding = layers.Dense(embed_dim, activation="relu", name="z_cnn")(x)

    # ── Regression head ─────────────────────────────────────────────────────
    forecast = layers.Dense(1, name="y_hat_cnn")(embedding)

    model = Model(inputs=inputs, outputs=[embedding, forecast], name="CNN_Branch")
    return model


if __name__ == "__main__":
    model = build_cnn_branch()
    model.summary()
