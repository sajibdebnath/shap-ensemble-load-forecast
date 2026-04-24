"""
Transformer Branch for Long-Range Temporal Dependency Modeling.

Implements the Transformer encoder branch of the physics-informed ensemble
framework described in:
  Abubakkar et al., "Interpretable Physics-Informed Load Forecasting for
  U.S. Grid Resilience", IEEE, 2026.

Architecture:
  Input (T=24, F=12) → Linear Projection (d=64) → Sinusoidal Positional Encoding
                      → Encoder Block ×2 [MultiHead Self-Attention (h=4, d_k=16)
                         + LayerNorm + FeedForward(d_ff=256) + LayerNorm]
                      → GlobalAvgPool → Dense(64) [embedding z_T]
                      → Dense(1) [branch forecast ŷ_T]
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


# ── Sinusoidal Positional Encoding ──────────────────────────────────────────

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    """
    Compute the sinusoidal positional encoding matrix.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Parameters
    ----------
    seq_len : int
        Sequence length.
    d_model : int
        Model dimensionality.

    Returns
    -------
    tf.Tensor
        Shape (1, seq_len, d_model), dtype float32.
    """
    positions = np.arange(seq_len)[:, np.newaxis]            # (T, 1)
    dims = np.arange(d_model)[np.newaxis, :]                 # (1, d_model)
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)

    # Apply sin to even indices, cos to odd indices
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)  # (1, T, d_model)


# ── Transformer Encoder Block ────────────────────────────────────────────────

class TransformerEncoderBlock(layers.Layer):
    """
    A single Transformer encoder block with pre-LN style.

    Components
    ----------
    - Multi-head self-attention  (h heads, key dim d_k)
    - Add & LayerNorm
    - Position-wise feed-forward network (hidden dim d_ff)
    - Add & LayerNorm
    - Dropout after each sub-layer
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 256,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,   # d_k = 16 for d_model=64, h=4
            dropout=dropout_rate,
        )
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(d_ff, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(d_model),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training: bool = False):
        # Multi-head self-attention sub-layer
        attn_output = self.attention(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward sub-layer
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "d_ff": self.d_ff,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


# ── Full Transformer Branch ──────────────────────────────────────────────────

def build_transformer_branch(
    seq_len: int = 24,
    n_features: int = 12,
    d_model: int = 64,
    num_heads: int = 4,
    d_ff: int = 256,
    num_encoder_blocks: int = 2,
    embed_dim: int = 64,
    dropout_rate: float = 0.2,
) -> Model:
    """
    Build the Transformer branch for long-range dependency capture.

    Parameters
    ----------
    seq_len : int
        Input sequence length (default: 24 hours).
    n_features : int
        Number of input features (default: 12).
    d_model : int
        Transformer model dimensionality (default: 64).
    num_heads : int
        Number of attention heads (default: 4).
    d_ff : int
        Feed-forward hidden dimension (default: 256).
    num_encoder_blocks : int
        Number of stacked encoder blocks (default: 2).
    embed_dim : int
        Dimensionality of branch embedding z_T (default: 64).
    dropout_rate : float
        Dropout rate (default: 0.2).

    Returns
    -------
    tf.keras.Model
        Transformer branch model with outputs (embedding, forecast).
    """
    inputs = layers.Input(shape=(seq_len, n_features), name="transformer_input")

    # ── Linear projection to d_model dimensions ─────────────────────────────
    x = layers.Dense(d_model, name="linear_projection")(inputs)

    # ── Sinusoidal positional encoding (non-trainable) ───────────────────────
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    x = x + pe  # broadcast over batch dimension

    x = layers.Dropout(dropout_rate, name="input_dropout")(x)

    # ── Stacked Transformer encoder blocks ───────────────────────────────────
    for i in range(num_encoder_blocks):
        x = TransformerEncoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            name=f"encoder_block_{i + 1}",
        )(x)

    # ── Global Average Pooling → Dense embedding ─────────────────────────────
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    embedding = layers.Dense(embed_dim, activation="relu", name="z_transformer")(x)

    # ── Regression head ──────────────────────────────────────────────────────
    forecast = layers.Dense(1, name="y_hat_transformer")(embedding)

    model = Model(
        inputs=inputs,
        outputs=[embedding, forecast],
        name="Transformer_Branch",
    )
    return model


if __name__ == "__main__":
    model = build_transformer_branch()
    model.summary()
