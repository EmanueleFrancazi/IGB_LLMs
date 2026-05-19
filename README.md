# LLM Behavior Lab

LLM Behavior Lab is an incremental research codebase for studying how language-model behavior evolves from random initialization through inference, pre-training, and fine-tuning.

The long-term goal is to analyze how architectural and training choices affect:

- output behavior at initialization
- learning dynamics
- training efficiency
- predictive bias
- token, word, and subgroup preference
- convergence behavior
- perplexity and standard language-modeling performance

The project avoids treating models as black-box imports. Model components are implemented explicitly so they can be inspected, modified, and instrumented during experiments.

## Current phase

The repository now includes **Phase 3: Data Pipeline Integration and Repository Cleanup**.

Previous phases added:

- a modular package structure
- a shared model interface
- a model registry
- utility functions for seeding, device selection, and parameter counting
- a LLaMA-style decoder-only model scaffold
- a LLaMA model sanity-check script

Phase 3 adds:

- a small local text corpus
- a deterministic character-level tokenizer
- train/validation token splitting
- causal language-model batch creation
- a data-to-model compatibility script
- tests for the tokenizer and batcher

The Phase 1 toy/debug model files have been removed. The active baseline is now the LLaMA-style model.

## Repository structure

```text
llm_behavior_lab/
  configs/
    data/
      tiny_text.yaml
    model/
      tiny_llama.yaml

  data/
    raw/
      tiny_corpus.txt

  scripts/
    smoke_test_llama.py
    check_data_pipeline.py

  src/
    llm_behavior_lab/
      __init__.py

      data/
        __init__.py
        dataloader.py
        text_dataset.py
        tokenizer.py

      models/
        __init__.py
        base.py
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
    test_data_pipeline.py
    test_imports.py
    test_llama_shapes.py

  pyproject.toml
  README.md
```

### Key folders

- `configs/`: YAML configuration files for models and data.
- `data/raw/`: tiny local text data used for early pipeline checks.
- `scripts/`: runnable checks for model integration and data/model compatibility.
- `src/llm_behavior_lab/models/`: model interface, registry, and LLaMA-style implementation.
- `src/llm_behavior_lab/data/`: tokenizer, text loading, splitting, and causal LM batching utilities.
- `src/llm_behavior_lab/utils/`: reproducibility, device, and parameter-count helpers.
- `tests/`: lightweight sanity tests for the current implementation.

## Current implemented features

### LLaMA-style model integration

The model includes explicit implementations of:

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

### Data pipeline

The Phase 3 data pipeline includes:

- local text loading
- character-level tokenizer construction
- text-to-token encoding
- deterministic train/validation splitting
- random causal language-model batch sampling
- shifted input/target pair creation

For a token window of length `block_size + 1`, the first `block_size` tokens become `input_ids`, and the next `block_size` tokens become `targets`.

### Data/model compatibility check

The Phase 3 script verifies that one training batch can be passed through the LLaMA-style model and that logits have shape:

```text
(batch_size, block_size, model_vocab_size)
```

For the default configs, this is:

```text
(4, 16, 256)
```

## Removed legacy files

The Phase 1 toy/debug model artifacts were removed because the LLaMA-style model is now the active baseline.

Removed files:

- `configs/model/tiny_debug.yaml`
- `scripts/smoke_test_model.py`
- `src/llm_behavior_lab/models/debug.py`
- `tests/test_registry.py`

The shared base model interface, model registry, and utility modules remain because they are still used by the LLaMA-style model and future phases.

## Install

From the repository root:

```bash
python3 -m pip install -e ".[dev]"
```

## Run the Phase 2 model sanity check

```bash
python3 scripts/smoke_test_llama.py --config configs/model/tiny_llama.yaml
```

Expected output includes:

- registered models including `llama` and `llama_tiny`
- selected device
- parameter count
- input and target shapes
- logits shape
- scalar loss value

For the default model sanity check, the logits shape should be:

```text
(2, 16, 256)
```

## Run the Phase 3 data-pipeline check

```bash
python3 scripts/check_data_pipeline.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

Expected output includes:

- dataset path
- raw text character count
- tokenizer vocabulary size
- train and validation token counts
- train batch shapes
- validation batch shapes
- model parameter count
- logits shape
- scalar loss value
- decoded previews showing the input/target shift

For the default configs, the key shapes should be:

```text
Train input shape: (4, 16)
Train target shape: (4, 16)
Validation input shape: (4, 16)
Validation target shape: (4, 16)
Logits shape: (4, 16, 256)
Loss shape: ()
```

## Run tests

```bash
python3 -m pytest
```

## What is intentionally not included yet

Phase 3 does not add:

- full training loops
- optimizer or scheduler setup
- checkpointing
- generation utilities
- tokenizer persistence
- large dataset support
- validation loss evaluation over a full split
- output-distribution or bias metrics
- experiment logging infrastructure

Those components will be added in later phases.

## Next phases

Planned next steps:

1. **Phase 4 — Inference utilities**: prompt encoding, logits extraction, text generation, greedy/sampling decoding, and model-output inspection.
2. **Phase 5 — Baseline untrained-model analysis**: entropy, top-k statistics, token probability summaries, and initialization behavior checks.
3. **Phase 6 — Logging and checkpoint infrastructure**: experiment folders, JSON/CSV logs, metadata, and reproducible run records.
4. **Phase 7 — Pre-training loop**: optimizer, learning-rate schedule, loss logging, validation checks, and checkpoint evaluation.
5. **Phase 8+ — Training dynamics, fine-tuning, and multi-model extensions**.
