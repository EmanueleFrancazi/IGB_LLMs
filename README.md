# LLM Behavior Lab

LLM Behavior Lab is an incremental research codebase for studying how language-model behavior evolves from random initialization through inference, pre-training, and fine-tuning.

The long-term goal is to analyze how architectural, data, and training choices affect:

- output behavior at initialization
- gradient stability
- learning dynamics
- training efficiency
- predictive bias
- token, word, and subgroup preference
- convergence behavior
- perplexity and standard language-modeling performance
- output-distribution changes over training checkpoints

The project avoids treating models as black-box imports. Model components are implemented explicitly so they can be inspected, modified, and instrumented during experiments.

## Current phase

The repository currently includes the Phase 5 analysis stack plus a dataset-extension refinement before Phase 6.

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
- prompt/inference utilities
- untrained-model output analysis
- optional per-layer gradient-norm diagnostics at initialization

This dataset-extension step adds:

- config-driven dataset source selection
- preserved local text loading as the default lightweight path
- optional Hugging Face dataset loading
- WikiText-2 raw config for a standard small language-modeling dataset
- TinyStories streaming config for optional small-model experiments
- a dataset-options check script
- tests for the dataset-source abstraction without requiring network downloads

The codebase still does not implement full training, checkpointing, or full experiment logging. Those are planned for later phases.

## Repository structure

```text
llm_behavior_lab/
  configs/
    data/
      tiny_text.yaml
      wikitext2.yaml
      tinystories_streaming.yaml
    model/
      tiny_llama.yaml

  data/
    raw/
      tiny_corpus.txt

  scripts/
    smoke_test_llama.py
    check_data_pipeline.py
    check_dataset_options.py
    run_inference.py
    analyze_untrained_model.py

  src/
    llm_behavior_lab/
      __init__.py

      data/
        __init__.py
        dataloader.py
        sources.py
        text_dataset.py
        tokenizer.py

      evaluation/
        __init__.py
        gradient_norms.py
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
    test_dataset_sources.py
    test_evaluation.py
    test_gradient_norms.py
    test_imports.py
    test_inference.py
    test_llama_shapes.py

  pyproject.toml
  README.md
```

## File and function map

### Model files

- `src/llm_behavior_lab/models/base.py`: shared `BaseLanguageModel` interface and `ModelOutput` container.
- `src/llm_behavior_lab/models/registry.py`: model registration and config-based construction.
- `src/llm_behavior_lab/models/llama/config.py`: LLaMA-style model config dataclass.
- `src/llm_behavior_lab/models/llama/model.py`: explicit LLaMA-style decoder-only implementation with RMSNorm, RoPE, grouped-query attention, SwiGLU blocks, residual connections, logits, and optional loss.

### Data files

- `src/llm_behavior_lab/data/sources.py`: config-driven dataset-source loading for local text and optional Hugging Face datasets.
- `src/llm_behavior_lab/data/text_dataset.py`: UTF-8 text loading and deterministic token splitting.
- `src/llm_behavior_lab/data/tokenizer.py`: deterministic character-level tokenizer.
- `src/llm_behavior_lab/data/dataloader.py`: causal language-model batch construction.
- `configs/data/tiny_text.yaml`: default local toy corpus config for smoke tests.
- `configs/data/wikitext2.yaml`: standard WikiText-2 raw config through Hugging Face `datasets`.
- `configs/data/tinystories_streaming.yaml`: optional TinyStories streaming config with small limits.

### Inference files

- `src/llm_behavior_lab/inference/generation.py`: prompt preparation, logits extraction, probability conversion, greedy/sampling decoding, and short generation.
- `scripts/run_inference.py`: lightweight inference check on the untrained model.

### Evaluation and analysis files

- `src/llm_behavior_lab/evaluation/output_stats.py`: entropy, top-k mass, top-1 probabilities, and output summaries.
- `src/llm_behavior_lab/evaluation/token_frequency.py`: empirical token frequencies, probability-frequency gaps, KL divergence, and JS divergence.
- `src/llm_behavior_lab/evaluation/untrained_analysis.py`: combines untrained output behavior diagnostics.
- `src/llm_behavior_lab/evaluation/gradient_norms.py`: optional per-layer squared L2 gradient-norm diagnostic and log-linear trend fit.
- `scripts/analyze_untrained_model.py`: runnable Phase 5 untrained-model analysis script.

### Scripts

- `scripts/smoke_test_llama.py`: model-only sanity check.
- `scripts/check_data_pipeline.py`: default data/model compatibility check.
- `scripts/check_dataset_options.py`: checks local or Hugging Face dataset configs through the unified data-source path.
- `scripts/run_inference.py`: prompt-to-generation inference check.
- `scripts/analyze_untrained_model.py`: initialization-time output and optional gradient analysis.

## Dataset support

### Local tiny corpus

The default config remains:

```yaml
dataset:
  source_type: local_text
  path: data/raw/tiny_corpus.txt
```

Use this for fast tests, offline development, and CPU smoke checks. It does not require internet access or optional dependencies.

### Hugging Face datasets

External dataset configs use:

```yaml
dataset:
  source_type: huggingface
  name: Salesforce/wikitext
  subset: wikitext-2-raw-v1
  split: train[:1000]
  text_field: text
  max_examples: 1000
  max_characters: 200000
```

The Hugging Face path is optional. Install it with:

```bash
python3 -m pip install -e ".[hf]"
```

If you already have a mixed system/user Python environment, refresh the optional dependencies with:

```bash
python3 -m pip install --upgrade --force-reinstall -e ".[dev,hf]"
```

This matters because Hugging Face `datasets` may import optional SciPy modules while constructing dataset builders. Older SciPy builds compiled against NumPy 1.x can fail when NumPy 2.x is installed, with errors such as `_ARRAY_API not found` or `numpy.core.multiarray failed to import`. The project pins `numpy<2.0` and includes `scipy>=1.11.4` in the optional `hf` dependency group to avoid this common mismatch.

External datasets may require internet access on first use. Hugging Face caches downloaded data in the normal datasets cache location, usually under `~/.cache/huggingface/datasets` unless environment variables override it.

### Recommended dataset choices right now

- Use `configs/data/tiny_text.yaml` for all fast local checks.
- Use `configs/data/wikitext2.yaml` when you want a small standard language-modeling dataset.
- Use `configs/data/tinystories_streaming.yaml` only when you want to experiment with synthetic short-story text and have internet access. It uses streaming plus small caps to avoid downloading the full dataset.

Large datasets such as C4, FineWeb, FineWeb-Edu, and The Pile are intentionally not default configs yet. They are useful future options, but they require more careful storage, streaming, sharding, and logging support.

## Install

From the repository root:

```bash
python3 -m pip install -e ".[dev]"
```

For Hugging Face dataset configs:

```bash
python3 -m pip install -e ".[dev,hf]"
```

If your environment already has incompatible NumPy/SciPy packages, force-refresh the optional stack:

```bash
python3 -m pip install --upgrade --force-reinstall -e ".[dev,hf]"
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

## Run the local data-pipeline check

```bash
python3 scripts/check_data_pipeline.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

Expected key shapes:

```text
Train input shape: (4, 16)
Train target shape: (4, 16)
Validation input shape: (4, 16)
Validation target shape: (4, 16)
Logits shape: (4, 16, 256)
Loss shape: ()
```

## Run the dataset-options check

Local dataset:

```bash
python3 scripts/check_dataset_options.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

WikiText-2 dataset:

```bash
python3 scripts/check_dataset_options.py \
  --data-config configs/data/wikitext2.yaml \
  --model-config configs/model/tiny_llama.yaml
```

If `datasets` is not installed, the WikiText-2 command will tell you to install the optional dependency. If your environment has a NumPy/SciPy binary mismatch, the command now raises a project-level `DatasetSourceError` with the recommended reinstall command instead of exposing a long SciPy traceback.

Expected output includes:

- dataset source type
- dataset source name
- source metadata
- number of raw examples used
- raw text character count
- tokenizer vocabulary size
- train/validation token counts
- batch shapes
- optional logits shape and loss after one model forward pass

## Run inference

```bash
python3 scripts/run_inference.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

Expected output includes prompt token IDs, input tensor shape, full logits shape, tokenizer-restricted next-token logits, top-k next-token predictions, generated token IDs, and decoded text.

## Run Phase 5 untrained-model analysis

Without gradient norms:

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

With optional gradient norms:

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml \
  --compute-grad-norms
```

The gradient diagnostic computes per-layer squared L2 norms of gradients with respect to decoder block outputs. It fits:

```text
log(g_l + eps) = intercept + slope * layer_index
```

Interpretation:

- slope near zero: gradients are roughly stable across depth
- negative slope: gradients decay with depth
- positive slope: gradients grow with depth
- large absolute slope: stronger vanishing or exploding trend

The JSON and CSV outputs are saved by default under:

```text
outputs/phase5_gradient_norms/
```

## Run tests

```bash
python3 -m pytest
```

The test suite does not download external datasets. Hugging Face dataset loading is tested with a mocked loader so local development remains fast and offline-friendly.

## Current limitations

This repository still does not include:

- full training loops
- optimizer or scheduler setup
- checkpointing
- persistent experiment logging
- full validation evaluation over large datasets
- tokenizer persistence
- streaming/sharded training loaders
- fine-tuning
- multi-model comparison

The current external dataset path concatenates a controlled number of text examples into one text stream, then reuses the existing character tokenizer and causal LM batcher. This is enough for pre-training pipeline preparation, but later phases should add persistent tokenized datasets, streaming/sharding, and experiment metadata.

## Next phases

1. **Phase 6 — Logging and checkpoint infrastructure**: experiment directories, config snapshots, metadata saving, JSONL/CSV metric writers, and artifact tracking.
2. **Phase 7 — Pre-training loop**: optimizer, learning-rate schedule, training steps, validation checks, and checkpoint saves.
3. **Phase 8 — Training-dynamics analysis**: loss/perplexity curves, output-distribution changes, gradient diagnostics over checkpoints, and bias metrics over training.
4. **Phase 9 — Fine-tuning pipeline**: supervised fine-tuning dataset support and checkpoint-based comparison.
5. **Phase 10 — Model extension phase**: add Gemma or other architectures and compare behavior across model families.
