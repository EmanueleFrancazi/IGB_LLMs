"""Educational LLaMA-style decoder-only transformer.

This module adapts the selected educational LLaMA implementation into the
Phase 1 project interfaces. The architectural components are intentionally
explicit: RMSNorm, rotary position embeddings, grouped-query self-attention,
SwiGLU feed-forward blocks, residual connections, logits, and optional
cross-entropy loss.

Local integration notes:
- The original educational implementation focuses on one-token-at-a-time
  inference with a KV cache. This module preserves that path through
  ``use_cache=True``.
- For future training phases, the project-level ``forward`` method also
  supports full-sequence causal language modeling without cache. This is a
  compatibility addition; it does not change the block mathematics.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from llm_behavior_lab.models.base import BaseLanguageModel, ModelOutput
from llm_behavior_lab.models.llama.config import LlamaConfig


class LlamaRMSNorm(nn.Module):
    """Root-mean-square normalization used in LLaMA-family models."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the normalization in fp32 for stability, then return to the
        # incoming dtype. This mirrors common LLaMA implementations.
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_rotary_frequencies(
    head_dim: int,
    seq_len: int,
    *,
    theta: float = 10_000.0,
) -> torch.Tensor:
    """Precompute complex rotary frequencies for RoPE.

    Args:
        head_dim: Dimension of one attention head. Must be even.
        seq_len: Number of positions to precompute.
        theta: RoPE base.

    Returns:
        Complex tensor with shape ``[seq_len, head_dim // 2]``.
    """

    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for rotary embeddings.")

    theta_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (theta_indices / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_embeddings(
    x: torch.Tensor,
    freqs_complex: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to query or key states.

    Args:
        x: Tensor with shape ``[batch, seq_len, n_heads, head_dim]``.
        freqs_complex: Complex frequencies with shape
            ``[seq_len, head_dim // 2]``.

    Returns:
        Tensor with the same shape and dtype as ``x``.
    """

    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    return x_out.type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for grouped-query attention.

    Args:
        x: Tensor with shape ``[batch, seq_len, n_kv_heads, head_dim]``.
        n_rep: Number of query-head groups sharing each key/value head.

    Returns:
        Tensor with shape ``[batch, seq_len, n_kv_heads * n_rep, head_dim]``.
    """

    if n_rep == 1:
        return x

    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class LlamaSelfAttention(nn.Module):
    """Causal self-attention with optional grouped-query attention and KV cache."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.n_heads_q = config.n_heads
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.max_batch_size = config.max_batch_size
        self.max_seq_len = config.max_seq_len

        self.wq = nn.Linear(config.dim, self.n_heads_q * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads_q * self.head_dim, config.dim, bias=False)

        self.register_buffer(
            "cache_k",
            torch.zeros(
                config.max_batch_size,
                config.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            ),
            persistent=False,
        )
        self.register_buffer(
            "cache_v",
            torch.zeros(
                config.max_batch_size,
                config.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        start_pos: int,
        freqs_complex: torch.Tensor,
        use_cache: bool,
        causal_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run attention over a sequence or one cached inference step."""

        batch_size, seq_len, _ = x.shape

        xq = self.wq(x).view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_embeddings(xq, freqs_complex)
        xk = apply_rotary_embeddings(xk, freqs_complex)

        if use_cache:
            if batch_size > self.max_batch_size:
                raise ValueError(
                    f"batch_size={batch_size} exceeds max_batch_size={self.max_batch_size}."
                )
            if start_pos + seq_len > self.max_seq_len:
                raise ValueError(
                    f"start_pos + seq_len={start_pos + seq_len} exceeds "
                    f"max_seq_len={self.max_seq_len}."
                )

            self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
            self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv
            keys = self.cache_k[:batch_size, : start_pos + seq_len]
            values = self.cache_v[:batch_size, : start_pos + seq_len]
        else:
            keys = xk
            values = xv

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            scores = scores + causal_mask

        attention_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(attention_weights, values)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)


class LlamaFeedForward(nn.Module):
    """SwiGLU feed-forward network used by LLaMA-style decoder blocks."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()

        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.multiple_of * (
            (hidden_dim + config.multiple_of - 1) // config.multiple_of
        )

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LlamaDecoderBlock(nn.Module):
    """Pre-norm transformer decoder block."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.attention = LlamaSelfAttention(config)
        self.feed_forward = LlamaFeedForward(config)
        self.attention_norm = LlamaRMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = LlamaRMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        *,
        start_pos: int,
        freqs_complex: torch.Tensor,
        use_cache: bool,
        causal_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = x + self.attention(
            self.attention_norm(x),
            start_pos=start_pos,
            freqs_complex=freqs_complex,
            use_cache=use_cache,
            causal_mask=causal_mask,
        )
        return h + self.feed_forward(self.ffn_norm(h))


class LlamaForCausalLM(BaseLanguageModel):
    """LLaMA-style causal language model compatible with project interfaces."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([LlamaDecoderBlock(config) for _ in range(config.n_layers)])
        self.norm = LlamaRMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        freqs_complex = precompute_rotary_frequencies(
            config.dim // config.n_heads,
            config.max_seq_len * 2,
            theta=config.rope_theta,
        )
        self.register_buffer("freqs_complex", freqs_complex, persistent=False)

    def _build_causal_mask(
        self,
        sequence_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Create an additive causal mask for full-sequence training/inference."""

        if sequence_length <= 1:
            return None

        mask = torch.full(
            (sequence_length, sequence_length),
            fill_value=torch.finfo(dtype).min,
            device=device,
            dtype=dtype,
        )
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        start_pos: int = 0,
        use_cache: bool = False,
        **_: Any,
    ) -> ModelOutput:
        """Run a causal language-model forward pass.

        Args:
            input_ids: Token IDs with shape ``[batch, sequence]``.
            targets: Optional next-token targets with the same shape.
            start_pos: Starting position for rotary embeddings and KV cache.
            use_cache: If true, update and read the KV cache. This preserves
                the original educational inference pathway. For Phase 2 sanity
                checks and future training, leave this false.

        Returns:
            ``ModelOutput`` containing logits and optional cross-entropy loss.
        """

        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}."
            )
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError(
                "targets must have the same shape as input_ids, "
                f"got targets={tuple(targets.shape)} and input_ids={tuple(input_ids.shape)}."
            )

        batch_size, sequence_length = input_ids.shape
        if sequence_length == 0:
            raise ValueError("sequence_length must be positive.")
        if start_pos < 0:
            raise ValueError("start_pos must be non-negative.")
        if start_pos + sequence_length > self.config.max_seq_len:
            raise ValueError(
                f"start_pos + sequence_length={start_pos + sequence_length} exceeds "
                f"max_seq_len={self.config.max_seq_len}."
            )
        if use_cache and batch_size > self.config.max_batch_size:
            raise ValueError(
                f"batch_size={batch_size} exceeds max_batch_size={self.config.max_batch_size}."
            )

        hidden_states = self.tok_embeddings(input_ids)
        freqs_complex = self.freqs_complex[start_pos : start_pos + sequence_length].to(
            device=hidden_states.device
        )

        causal_mask = None
        if not use_cache:
            causal_mask = self._build_causal_mask(
                sequence_length,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                start_pos=start_pos,
                freqs_complex=freqs_complex,
                use_cache=use_cache,
                causal_mask=causal_mask,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states).float()

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )

        return ModelOutput(logits=logits, loss=loss)


def build_llama_model(**kwargs: Any) -> LlamaForCausalLM:
    """Build a LLaMA-style model from registry keyword arguments."""

    config = LlamaConfig(**kwargs)
    return LlamaForCausalLM(config)
