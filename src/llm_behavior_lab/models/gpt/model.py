"""Explicit GPT-2-style decoder-only transformer.

This module adapts the nanoGPT reference implementation into the project
interfaces. It exists as a deliberately *different* architecture from
:mod:`llm_behavior_lab.models.llama`, so that a result reproduced on both is a
property of randomly initialized transformer language models rather than of one
recipe. The differences are the point and are kept faithful rather than
normalized toward LLaMA:

.. code-block:: text

    LLaMA family                       GPT-2 family (this module)
    ------------                       --------------------------
    RMSNorm, learnable gain, no bias    LayerNorm, gain and bias
    rotary positions, no parameters     learned absolute positions, wpe table
    grouped-query or multi-head         multi-head only
    SwiGLU feed-forward, 8/3 * dim      GELU feed-forward, 4 * dim
    no bias parameters anywhere         biases on every Linear and LayerNorm
    untied embedding and head           tied embedding and head
    PyTorch default initialization      explicit N(0, 0.02), scaled c_proj

Because they all move together, a difference measured between the two families
localizes to that **bundle**, not to any one of its members.

Local integration notes:

- ``inputs_embeds`` is accepted as an alternative to ``input_ids``. It bypasses
  the token lookup and nothing else -- learned positional embeddings are still
  added, every block is the same object on the same weights. It exists so a
  synthetic-input control can be compared against real tokens through an
  identical network.
- There is no KV cache and no ``start_pos``. nanoGPT recomputes the full
  sequence, and nothing in this project's measurement path uses cached
  inference for this family.
- Attention is written out explicitly rather than dispatched to a fused kernel,
  so the mask, the scale and the softmax stay readable.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from llm_behavior_lab.models.base import BaseLanguageModel, ModelOutput
from llm_behavior_lab.models.gpt.config import GPTConfig

#: Epsilon used by ``torch.nn.LayerNorm`` and therefore by GPT-2.
LAYER_NORM_EPS = 1e-5


class GPTLayerNorm(nn.Module):
    """LayerNorm with an optional bias.

    ``torch.nn.LayerNorm`` cannot drop its bias while keeping its gain, which is
    the configuration nanoGPT supports, so the module is written out. The gain
    starts at one and the bias at zero; both are deterministic and neither is
    touched by the initialization-scale intervention.
    """

    def __init__(self, dim: int, *, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, LAYER_NORM_EPS)


class GPTSelfAttention(nn.Module):
    """Causal multi-head self-attention with fused QKV projection.

    The three projections share one ``c_attn`` weight, as in GPT-2, and are
    split after the matmul. That is a parameter-layout choice rather than a
    mathematical one, but it is part of what the tensor names and the CountSketch
    parameter ordering describe, so it is preserved.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.dim, 3 * config.dim, bias=config.bias)
        self.c_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular admissibility, not an additive mask: stored as bool
        # so the full max_seq_len x max_seq_len table costs one byte per entry.
        # Non-persistent, so it never enters a state dict or a checkpoint.
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)
            ).view(1, 1, config.max_seq_len, config.max_seq_len),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape

        query, key, value = self.c_attn(x).split(dim, dim=2)
        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~self.causal_mask[:, :, :seq_len, :seq_len], float("-inf")
        )
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        output = attention_weights @ value
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.resid_dropout(self.c_proj(output))


class GPTMLP(nn.Module):
    """GELU feed-forward network at four times the residual width."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.dim, 4 * config.dim, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.dim, config.dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class GPTBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = GPTLayerNorm(config.dim, bias=config.bias)
        self.attn = GPTSelfAttention(config)
        self.ln_2 = GPTLayerNorm(config.dim, bias=config.bias)
        self.mlp = GPTMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPTForCausalLM(BaseLanguageModel):
    """GPT-2-style causal language model compatible with project interfaces."""

    #: Trainable tensors this family initializes deterministically, consumed by
    #: :func:`llm_behavior_lab.models.initialization_scale.classify_parameters`.
    #: Every LayerNorm gain starts at exactly one and every LayerNorm and Linear
    #: bias at exactly zero, so none of them is a zero-centred random draw and
    #: none may be multiplied by ``alpha``: scaling a gain would change the
    #: normalization itself, which is a different intervention.
    #:
    #: The suffixes are listed individually rather than excluding every
    #: ``.bias``. A family that drew a bias from a random distribution would
    #: need it scaled, and a blanket rule would silently stop scaling it.
    #: ``c_proj.bias`` covers both ``attn.c_proj.bias`` and ``mlp.c_proj.bias``.
    DETERMINISTIC_PARAMETER_SUFFIXES = (
        "ln_1.weight",
        "ln_1.bias",
        "ln_2.weight",
        "ln_2.bias",
        "ln_f.weight",
        "ln_f.bias",
        "c_attn.bias",
        "c_proj.bias",
        "c_fc.bias",
    )

    #: How this family initializes itself, persisted with every run that uses it.
    #: Read through
    #: :func:`llm_behavior_lab.models.initialization_scale.initialization_note`,
    #: which is the single source both the scale report and the experiment's
    #: metadata consume. Without it a GPT record would inherit the LLaMA-family
    #: default and claim kaiming-uniform linears and an architecture with no
    #: biases -- neither true here, and initialization is one of the variables a
    #: cross-architecture comparison is deliberately changing.
    INITIALIZATION_NOTE = (
        "This family states its own initialization rather than inheriting "
        "PyTorch's per-class defaults. Token embeddings, learned positional "
        "embeddings, and every attention and MLP weight are drawn from "
        "normal_(0, 0.02); the residual c_proj weights are then redrawn with "
        "standard deviation 0.02/sqrt(2 * n_layers). LayerNorm gains are "
        "initialized to one and LayerNorm biases to zero, and every linear bias "
        "is zero. The token embedding and the output head share one parameter. "
        "alpha multiplies the randomly initialized tensors only: LayerNorm gains "
        "and biases and the zeroed linear biases are left unscaled, and the tied "
        "vocabulary tensor is scaled exactly once. There is no single "
        "architecture-wide sigma_w, because the residual projections carry a "
        "depth-dependent scale."
    )

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.dim),
                "wpe": nn.Embedding(config.max_seq_len, config.dim),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([GPTBlock(config) for _ in range(config.n_layers)]),
                "ln_f": GPTLayerNorm(config.dim, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying, as GPT-2 and nanoGPT do it. One storage reachable under
        # two names: ``named_parameters()`` de-duplicates and yields it once,
        # under ``transformer.wte.weight``, because the transformer is
        # registered before the head. Everything downstream that walks
        # parameters -- the exact gradient norm, the CountSketch tensor order,
        # the initialization-scale audit -- therefore sees it exactly once.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # The GPT-2 paper scales the initialization of residual projections by
        # 1/sqrt(2 * n_layers), so the variance accumulated along the residual
        # stream does not grow with depth. Applied after the uniform pass, which
        # is what overwrites the ordinary draw for these tensors.
        scale = 0.02 / math.sqrt(2 * config.n_layers)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Explicit GPT-2 initialization.

        Unlike the LLaMA family, which inherits PyTorch's per-class defaults,
        this family states its own: every Linear and Embedding weight is drawn
        from ``N(0, 0.02)`` and every bias is zeroed. :class:`GPTLayerNorm`
        already sets its gain to one and its bias to zero in its constructor and
        is deliberately not revisited here.

        The tied vocabulary tensor is visited twice -- once as the embedding and
        once as the head -- and therefore redrawn. Both draws use the same
        distribution, so the outcome is distributionally identical to a single
        draw; the extra draw is nanoGPT's own behaviour and is kept rather than
        optimized away, since it is deterministic under a fixed seed and this
        family's initialization is part of what the arm varies.
        """

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- shared structural surface -------------------------------------------
    #
    # Read-only views onto the canonical nanoGPT modules, under the names the
    # rest of the project already uses for the LLaMA family. They are class
    # attributes, so ``nn.Module.__setattr__`` never registers them: there is one
    # state-dict path per tensor, one ``named_parameters()`` entry, one
    # CountSketch table, and no alias that could duplicate a projection.

    @property
    def tok_embeddings(self) -> nn.Embedding:
        """The token embedding table, shared with the output head."""

        return self.transformer.wte

    @property
    def layers(self) -> nn.ModuleList:
        """The transformer block stack."""

        return self.transformer.h

    @property
    def norm(self) -> GPTLayerNorm:
        """The final normalization applied before the output projection."""

        return self.transformer.ln_f

    @property
    def output(self) -> nn.Linear:
        """The output projection, sharing its weight with the embedding."""

        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> ModelOutput:
        """Run a causal language-model forward pass.

        Args:
            input_ids: Token IDs with shape ``[batch, sequence]``.
            targets: Optional next-token targets shaped like the input positions.
            inputs_embeds: Optional pre-computed input vectors with shape
                ``[batch, sequence, dim]``, supplied *instead of* ``input_ids``.
                This bypasses the token lookup and nothing else: the learned
                positional embedding is still added, and every block, the final
                norm and the output projection are the same objects on the same
                weights.

        Returns:
            ``ModelOutput`` containing logits and optional cross-entropy loss.

        Raises:
            ValueError: If neither or both input forms are given, or if a shape
                is wrong, or if the sequence exceeds ``max_seq_len``.
        """

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "Provide exactly one of input_ids or inputs_embeds; "
                f"got input_ids={'set' if input_ids is not None else 'None'} and "
                f"inputs_embeds={'set' if inputs_embeds is not None else 'None'}."
            )

        if input_ids is not None:
            if input_ids.ndim != 2:
                raise ValueError(
                    f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}."
                )
            position_shape = input_ids.shape
        else:
            if inputs_embeds.ndim != 3:
                raise ValueError(
                    "inputs_embeds must have shape [batch, sequence, dim], got "
                    f"{tuple(inputs_embeds.shape)}."
                )
            if inputs_embeds.shape[-1] != self.config.dim:
                raise ValueError(
                    f"inputs_embeds last dimension {inputs_embeds.shape[-1]} does not match "
                    f"the model dimension {self.config.dim}."
                )
            position_shape = inputs_embeds.shape[:2]

        if targets is not None and tuple(targets.shape) != tuple(position_shape):
            raise ValueError(
                "targets must have the same shape as the input positions, "
                f"got targets={tuple(targets.shape)} and inputs={tuple(position_shape)}."
            )

        _, sequence_length = position_shape
        if sequence_length == 0:
            raise ValueError("sequence_length must be positive.")
        if sequence_length > self.config.max_seq_len:
            raise ValueError(
                f"sequence_length={sequence_length} exceeds max_seq_len="
                f"{self.config.max_seq_len}. The learned positional table has no "
                "row for the trailing positions."
            )

        # The one and only divergence between token input and synthetic input:
        # after this line the two conditions share every parameter and every op,
        # the positional embedding below included.
        token_embeddings = (
            self.transformer.wte(input_ids)
            if input_ids is not None
            else inputs_embeds.to(dtype=self.transformer.wte.weight.dtype)
        )
        positions = torch.arange(
            sequence_length, dtype=torch.long, device=token_embeddings.device
        )
        hidden_states = self.transformer.drop(
            token_embeddings + self.transformer.wpe(positions)
        )

        for block in self.transformer.h:
            hidden_states = block(hidden_states)

        hidden_states = self.transformer.ln_f(hidden_states)
        logits = self.lm_head(hidden_states).float()

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )

        return ModelOutput(logits=logits, loss=loss)


def build_gpt_model(**kwargs: Any) -> GPTForCausalLM:
    """Build a GPT-2-style model from registry keyword arguments."""

    config = GPTConfig(**kwargs)
    return GPTForCausalLM(config)
