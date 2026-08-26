"""Configuration object for the local GPT-2-style model.

The fields describe a GPT-2 decoder as implemented by nanoGPT: model width,
number of layers, attention heads, learned positional capacity, whether the
linear and normalization layers carry biases, and the dropout probability.

Field names are the project-facing ones, shared with :class:`LlamaConfig` where
the quantity is the same, so the experiment runner reads one vocabulary,
one width and one length field regardless of family. The upstream nanoGPT
spellings are:

.. code-block:: text

    this project        nanoGPT             meaning
    ------------        -------             -------
    vocab_size          vocab_size          output vocabulary
    dim                 n_embd              residual-stream width
    n_layers            n_layer             number of transformer blocks
    n_heads             n_head              number of attention heads
    max_seq_len         block_size          learned positional capacity
    bias                bias                biases on Linear and LayerNorm
    dropout             dropout             dropout probability

``max_seq_len`` is the one worth care. It is a hard limit in either family -- a
longer sequence is rejected -- but in a GPT-2 model it is *not merely* a runtime
bound: it is also the number of rows in the learned positional embedding table,
so it is part of the parameterization as well. In the LLaMA family rotary
embeddings have no parameters and the corresponding field only sizes buffers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    """Configuration for a GPT-2-style causal language model.

    The defaults describe a tiny model suitable for sanity checks. Experiment
    configurations should override them from YAML.

    Attributes:
        vocab_size: Number of discrete token IDs the model can predict.
        dim: Width of the residual stream (nanoGPT ``n_embd``).
        n_layers: Number of transformer blocks (nanoGPT ``n_layer``).
        n_heads: Number of attention heads (nanoGPT ``n_head``). GPT-2 uses
            standard multi-head attention, so there is no separate key/value
            head count.
        max_seq_len: Rows in the learned positional embedding table, and the
            longest sequence the model accepts (nanoGPT ``block_size``).
        bias: Whether ``nn.Linear`` and the normalization layers carry biases.
            GPT-2 uses biases throughout; ``False`` gives the leaner variant
            nanoGPT also supports.
        dropout: Dropout probability applied after the embeddings, inside
            attention, and on each residual projection.
    """

    vocab_size: int
    dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    max_seq_len: int = 64
    bias: bool = True
    dropout: float = 0.0

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
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if not isinstance(self.bias, bool):
            raise TypeError("bias must be a boolean.")
        # p = 1 would discard every activation, which is never the intent and
        # would make a forward pass meaningless rather than merely regularized.
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")
