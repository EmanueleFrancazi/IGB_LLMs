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

The repository now includes **Phase 4: Inference Utilities**.

Previous phases added:

- a modular package structure
- a shared model interface
- a model registry
- utility functions for seeding, device selection, and parameter counting
- a LLaMA-style decoder-only model scaffold
- a LLaMA model sanity-check script
- a small local text corpus
- a deterministic character-level tokenizer
- train/validation token splitting
- causal language-model batch creation
- a data-to-model compatibility script

Phase 4 adds:

- prompt-to-token conversion
- model-ready prompt tensors
- full-logits extraction
- next-token logits extraction
- next-token probability extraction with softmax
- top-k next-token inspection
- greedy decoding
- sampling decoding with temperature and optional top-k filtering
- short text generation
- a runnable inference script
- lightweight inference tests

The model is still untrained. Generated text is expected to be random or meaningless at this stage. The goal is to validate the inference infrastructure before adding analysis, logging, checkpointing, and training.

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
    run_inference.py

  src/
    llm_behavior_lab/
      __init__.py

      data/
        __init__.py
        dataloader.py
        text_dataset.py
        tokenizer.py

      inference/
        __init__.py
        generation.py

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
    test_inference.py
    test_llama_shapes.py

  pyproject.toml
  README.md
```

### Key folders

- `configs/`: YAML configuration files for models and data.
- `data/raw/`: tiny local text data used for early pipeline checks.
- `scripts/`: runnable checks for model integration, data/model compatibility, and inference.
- `src/llm_behavior_lab/models/`: model interface, registry, and LLaMA-style implementation.
- `src/llm_behavior_lab/data/`: tokenizer, text loading, splitting, and causal LM batching utilities.
- `src/llm_behavior_lab/inference/`: prompt preparation, logits inspection, decoding, and short generation utilities.
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

The data pipeline includes:

- local text loading
- character-level tokenizer construction
- text-to-token encoding
- deterministic train/validation splitting
- random causal language-model batch sampling
- shifted input/target pair creation

For a token window of length `block_size + 1`, the first `block_size` tokens become `input_ids`, and the next `block_size` tokens become `targets`.

### Inference utilities

The inference utilities include:

- `prepare_prompt_tensor`: encode text prompts and create `[1, sequence]` input tensors
- `extract_logits`: run the model in `eval()` mode with gradients disabled
- `extract_next_token_logits`: extract final-position logits, optionally restricted to tokenizer-valid IDs
- `next_token_probabilities`: convert logits to probabilities with temperature scaling
- `top_k_predictions`: inspect likely next tokens and decode them into readable strings
- `select_next_token`: choose the next token by greedy or sampling decoding
- `generate_text`: generate a short continuation and decode it back into text

The current tiny LLaMA config uses a model vocabulary of 256, while the character tokenizer has fewer valid token IDs. During Phase 4 generation, next-token choices are restricted to the tokenizer vocabulary so generated IDs can be decoded.

## Removed legacy files

The Phase 1 toy/debug model artifacts were removed in Phase 3 because the LLaMA-style model became the active baseline.

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

## Run the Phase 4 inference check

```bash
python3 scripts/run_inference.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

Expected output includes:

- prompt text
- encoded prompt token IDs
- input tensor shape
- full logits shape
- next-token logits shape after tokenizer-vocabulary restriction
- top-k next-token predictions with probabilities
- generated token IDs
- decoded generated text
- a note that the model is untrained

For the default prompt, the key shapes should look like:

```text
Input tensor shape: (1, 4)
Full logits shape: (1, 4, 256)
Next-token logits shape after tokenizer-vocab restriction: (1, <tokenizer_vocab_size>)
```

The default run generates only 4 new tokens to keep the CPU check lightweight. The generated text may be random or repetitive because the model has not been trained yet.

You can also try sampling:

```bash
python3 scripts/run_inference.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml \
  --strategy sample \
  --temperature 0.8 \
  --top-k 10
```

## Run tests

```bash
python3 -m pytest
```

## What is intentionally not included yet

Phase 4 does not add:

- full training loops
- optimizer or scheduler setup
- checkpointing
- tokenizer persistence
- large dataset support
- validation loss evaluation over a full split
- output-distribution or bias metrics
- experiment logging infrastructure

Those components will be added in later phases.

## Next phases

Planned next steps:

1. **Phase 5 — Baseline untrained-model analysis**: entropy, top-k statistics, token probability summaries, and initialization behavior checks.
2. **Phase 6 — Logging and checkpoint infrastructure**: experiment folders, JSON/CSV logs, metadata, and reproducible run records.
3. **Phase 7 — Pre-training loop**: optimizer, learning-rate schedule, loss logging, validation checks, and checkpoint evaluation.
4. **Phase 8+ — Training dynamics, fine-tuning, and multi-model extensions**.
