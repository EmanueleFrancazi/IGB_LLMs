# LLM Behavior Lab

This repository is an incremental research codebase for studying how language-model behavior evolves from random initialization through pre-training and fine-tuning.

The project prioritizes:

- transparent model implementations
- modular data, training, inference, evaluation, and logging components
- reproducible experiments
- easy extension to additional model families

## Current phase

The repository now includes **Phase 2: LLaMA-style model integration**.

Phase 1 created the package skeleton, model interface, registry, utilities, a debug model, and smoke tests.

Phase 2 adds a small but realistic LLaMA-style decoder-only model with explicit components:

- token embeddings
- RMSNorm
- rotary position embeddings
- grouped-query self-attention
- optional KV-cache path for incremental inference
- SwiGLU feed-forward blocks
- residual decoder blocks
- final normalization
- language-modeling logits
- optional cross-entropy loss when targets are provided

The implementation is adapted for the project structure from the selected educational LLaMA-style source while keeping the architecture explicit and easy to modify.

## Folder structure

```text
llm_behavior_lab/
  configs/
    model/
      tiny_debug.yaml
      tiny_llama.yaml

  scripts/
    smoke_test_model.py
    smoke_test_llama.py

  src/
    llm_behavior_lab/
      __init__.py
      models/
        __init__.py
        base.py
        debug.py
        registry.py
        llama/
          __init__.py
          config.py
          model.py
      utils/
        __init__.py
        device.py
        params.py
        seed.py

  tests/
    test_imports.py
    test_registry.py
    test_llama_shapes.py

  pyproject.toml
  README.md
```

## Install

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

## Run the Phase 1 debug smoke test

```bash
python scripts/smoke_test_model.py --config configs/model/tiny_debug.yaml
```

## Run the Phase 2 LLaMA sanity check

```bash
python scripts/smoke_test_llama.py --config configs/model/tiny_llama.yaml
```

Expected output includes:

- registered models, including `llama` and `llama_tiny`
- selected device
- parameter count
- input and target shapes
- logits shape
- scalar loss value

For the default config and script arguments, the logits shape should be:

```text
(2, 16, 256)
```

because the default synthetic batch size is 2, the sequence length is 16, and the tiny config vocabulary size is 256.

## Instantiate the LLaMA-style model from Python

```python
from llm_behavior_lab.models import build_model

model = build_model(
    "llama_tiny",
    vocab_size=256,
    dim=128,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
    max_batch_size=8,
    max_seq_len=64,
)
```

Or from a YAML-style dictionary:

```python
from llm_behavior_lab.models import build_model_from_config

config = {
    "model": {
        "name": "llama_tiny",
        "params": {
            "vocab_size": 256,
            "dim": 128,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "max_batch_size": 8,
            "max_seq_len": 64,
        },
    }
}

model = build_model_from_config(config)
```

## Run tests

```bash
pytest
```

## What is intentionally not included yet

Phase 2 does not add:

- tokenizer integration
- real datasets
- data loaders
- training loops
- validation loss evaluation
- checkpointing
- generation utilities
- bias or output-distribution analysis
- experiment logging beyond simple script output

These belong to later phases so that each component can be tested and understood incrementally.

## Next phase

Phase 3 will add the data pipeline:

- dataset loading
- tokenization strategy
- train/validation splits
- causal language-model batches
- shape validation for inputs and shifted targets
