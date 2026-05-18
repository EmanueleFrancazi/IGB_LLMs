"""Configuration objects for the local LLaMA-style model.

The fields mirror the educational LLaMA implementation selected for the first
baseline: model width, number of layers, attention heads, optional grouped
query attention, RMSNorm epsilon, SwiGLU feed-forward sizing, and cache limits.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LlamaConfig:
    """Configuration for a small LLaMA-style causal language model.

    The defaults describe a tiny model suitable for sanity checks. Larger
    experiments should override these values from YAML config files.

    Attributes:
        vocab_size: Number of discrete token IDs the model can predict.
        dim: Hidden size of the transformer.
        n_layers: Number of decoder blocks.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads. If ``None``, this defaults to
            ``n_heads`` and becomes standard multi-head attention. If smaller
            than ``n_heads``, the model uses grouped-query attention.
        multiple_of: Feed-forward hidden dimension is rounded up to a multiple
            of this value, following the LLaMA-family convention.
        ffn_dim_multiplier: Optional multiplier for the feed-forward hidden
            dimension.
        norm_eps: Epsilon used in RMSNorm.
        max_batch_size: Maximum batch size supported by the optional KV cache.
        max_seq_len: Maximum sequence length supported by the model.
        rope_theta: Rotary-position embedding base.
    """

    vocab_size: int
    dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int | None = None
    multiple_of: int = 64
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    max_batch_size: int = 8
    max_seq_len: int = 64
    rope_theta: float = 10_000.0

    def validate(self) -> None:
        """Validate the configuration before constructing model modules."""

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.dim <= 0:
            raise ValueError("dim must be positive.")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive.")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads.")

        head_dim = self.dim // self.n_heads
        if head_dim % 2 != 0:
            raise ValueError("Each attention head dimension must be even for RoPE.")

        n_kv_heads = self.n_heads if self.n_kv_heads is None else self.n_kv_heads
        if n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive when provided.")
        if self.n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads.")

        if self.multiple_of <= 0:
            raise ValueError("multiple_of must be positive.")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive.")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive.")
