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

The repository now includes **Phase 5: Baseline Untrained-Model Analysis**.

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
- inference utilities for prompt encoding, logits inspection, top-k predictions, and short generation

Phase 5 adds:

- output-distribution statistics for the untrained model
- entropy and probability-concentration summaries
- top-k prediction examples at selected positions
- empirical token-frequency computation from the tiny corpus
- comparison between average model probabilities and empirical token frequencies
- largest positive and negative probability-frequency gaps
- top-1 predicted-token frequency summaries
- simple KL and Jensen-Shannon divergence summaries
- a runnable untrained-model analysis script
- lightweight tests for the evaluation utilities

The model is still untrained. Phase 5 does not measure model quality; it establishes a reproducible baseline for how the randomly initialized model behaves before any optimization.

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
    analyze_untrained_model.py

  src/
    llm_behavior_lab/
      __init__.py

      data/
        __init__.py
        dataloader.py
        text_dataset.py
        tokenizer.py

      evaluation/
        __init__.py
        output_stats.py
        token_frequency.py
        untrained_analysis.py

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
    test_evaluation.py
    test_imports.py
    test_inference.py
    test_llama_shapes.py

  pyproject.toml
  README.md
```

## File/function map

### Model files

- `src/llm_behavior_lab/models/base.py`: shared `BaseLanguageModel` interface and `ModelOutput` container.
- `src/llm_behavior_lab/models/registry.py`: model registry and config-based model construction.
- `src/llm_behavior_lab/models/llama/config.py`: `LlamaConfig` dataclass and validation.
- `src/llm_behavior_lab/models/llama/model.py`: explicit LLaMA-style decoder-only model implementation.

The LLaMA-style model includes token embeddings, RMSNorm, RoPE, grouped-query self-attention, an optional KV-cache path, SwiGLU feed-forward blocks, residual decoder blocks, final normalization, logits, and optional cross-entropy loss.

### Data files

- `data/raw/tiny_corpus.txt`: tiny local text corpus used for early pipeline checks.
- `src/llm_behavior_lab/data/tokenizer.py`: deterministic character-level tokenizer.
- `src/llm_behavior_lab/data/text_dataset.py`: local text loading and train/validation splitting.
- `src/llm_behavior_lab/data/dataloader.py`: causal language-model batch construction.

### Inference files

- `src/llm_behavior_lab/inference/generation.py`: prompt preparation, logits extraction, probability extraction, top-k inspection, greedy decoding, sampling decoding, and short generation.
- `scripts/run_inference.py`: runnable inference check for the untrained model.

### Evaluation/analysis files

- `src/llm_behavior_lab/evaluation/output_stats.py`: entropy, top-k mass, top-1 probability, and output-distribution summaries.
- `src/llm_behavior_lab/evaluation/token_frequency.py`: empirical token counts/frequencies, average predicted probabilities, probability-frequency gaps, KL divergence, and JS divergence.
- `src/llm_behavior_lab/evaluation/untrained_analysis.py`: high-level baseline analysis that combines output statistics, top-k examples, top-1 prediction summaries, and token-frequency comparisons.
- `scripts/analyze_untrained_model.py`: runnable Phase 5 script for baseline analysis at initialization.

### Scripts

- `scripts/smoke_test_llama.py`: verifies model construction, dummy-token forward pass, logits shape, loss shape, and parameter count.
- `scripts/check_data_pipeline.py`: verifies tokenizer/data batching and model compatibility.
- `scripts/run_inference.py`: verifies prompt-based inference and short generation.
- `scripts/analyze_untrained_model.py`: analyzes output behavior of the randomly initialized model.

## Phase 5 execution flow

The untrained-model analysis script follows this flow:

1. Load data and model configs.
2. Load the tiny local text corpus.
3. Build the character-level tokenizer.
4. Tokenize the corpus and create train/validation splits.
5. Sample causal LM windows from the requested split.
6. Instantiate the randomly initialized LLaMA-style model from config.
7. Run a no-grad forward pass.
8. Convert logits to probabilities over tokenizer-valid tokens.
9. Compute output entropy and probability concentration.
10. Compute empirical token frequencies from the corpus.
11. Compare average predicted probabilities with empirical frequencies.
12. Print top-k examples, top-1 prediction summaries, probability-frequency gaps, and divergence summaries.

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

## Run the Phase 5 untrained-model analysis

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

Expected output includes:

- selected device
- dataset path
- tokenizer vocabulary size
- number of analyzed batches/windows/positions
- logits shape
- probability tensor shape
- mean, minimum, and maximum output entropy
- mean top-1 probability
- mean top-k probability mass
- top-1 assignment concentration
- KL and JS divergence summaries
- top-k predictions for a few example positions
- most frequent empirical dataset tokens
- most frequent top-1 predicted model tokens
- largest positive probability-frequency gaps
- largest negative probability-frequency gaps

For the default configs, the key shapes should look like:

```text
Input tensor shape: (16, 16)
Logits shape: (16, 16, 256)
Probability tensor shape: (16, 16, <tokenizer_vocab_size>)
```

The exact metric values will depend on random initialization and device, but the script should end with:

```text
Phase 5 untrained-model analysis completed successfully.
```

## Run tests

```bash
python3 -m pytest
```

## Current limitations

The repository still does not include:

- full training loops
- optimizer or scheduler setup
- checkpointing
- persistent experiment logging
- full validation-set evaluation
- fine-tuning
- multi-model comparison
- advanced bias or group-based evaluation metrics

Those components will be added in later phases.

## Next phases

Planned next steps:

1. **Phase 6 — Logging and checkpoint infrastructure**: experiment folders, JSON/CSV logs, metadata, config snapshots, and reproducible run records.
2. **Phase 7 — Pre-training loop**: optimizer, learning-rate schedule, loss logging, validation checks, and checkpoint evaluation.
3. **Phase 8 — Training-dynamics analysis**: compare Phase 5 initialization metrics against metrics collected during training.
4. **Phase 9 — Fine-tuning pipeline**: supervised fine-tuning data handling, fine-tuning loop, and checkpoint-based analysis.
5. **Phase 10 — Model extension phase**: add additional model implementations and compare behavior across architectures.
