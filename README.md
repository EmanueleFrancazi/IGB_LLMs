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

## Script-to-module map

| Script | Purpose | Main modules used |
|---|---|---|
| `scripts/smoke_test_llama.py` | Check that the LLaMA-style model builds and runs on dummy token IDs. | `models.registry` (`build_model_from_config`), `models.llama.model` (`LlamaForCausalLM`), `utils.device` (`get_device`), `utils.seed` (`seed_everything`) |
| `scripts/check_data_pipeline.py` | Check that local text data becomes causal LM batches compatible with the model. | `data.text_dataset` (`load_text_file`, `split_token_ids`), `data.tokenizer` (`CharTokenizer`), `data.dataloader` (`CausalLMBatcher`), `models.registry` (`build_model_from_config`) |
| `scripts/run_inference.py` | Check prompt encoding, logits extraction, top-k next-token inspection, and short generation. | `data.tokenizer` (`CharTokenizer`), `inference.generation` (`prepare_prompt_tensor`, `extract_logits`, `top_k_predictions`, `generate_text`), `models.registry` (`build_model_from_config`) |
| `scripts/analyze_untrained_model.py` | Analyze initialization-time output behavior of the untrained model. | `data.*`, `models.registry`, `evaluation.output_stats`, `evaluation.token_frequency`, `evaluation.untrained_analysis` |

## Phase 5 execution flow

The untrained-model analysis script follows this flow:

1. Load data and model configs.
   - Code path: `scripts/analyze_untrained_model.py` (`parse_args`, `main`, `load_yaml_config`) → `configs/data/tiny_text.yaml` + `configs/model/tiny_llama.yaml`.
2. Load the tiny local text corpus.
   - Code path: `scripts/analyze_untrained_model.py` (`main`, `resolve_repo_path`) → `src/llm_behavior_lab/data/text_dataset.py` (`load_text_file`) → `data/raw/tiny_corpus.txt`.
3. Build the character-level tokenizer.
   - Code path: `scripts/analyze_untrained_model.py` (`main`) → `src/llm_behavior_lab/data/tokenizer.py` (`CharTokenizer.from_text`).
4. Tokenize the corpus and create train/validation splits.
   - Code path: `scripts/analyze_untrained_model.py` (`main`) → `src/llm_behavior_lab/data/tokenizer.py` (`CharTokenizer.encode`) → `src/llm_behavior_lab/data/text_dataset.py` (`split_token_ids`).
5. Sample causal LM windows from the requested split.
   - Code path: `scripts/analyze_untrained_model.py` (`main`) → `src/llm_behavior_lab/data/dataloader.py` (`CausalLMBatcher`, `get_batch`) → `torch.cat`.
6. Instantiate the randomly initialized LLaMA-style model from config.
   - Code path: `scripts/analyze_untrained_model.py` (`main`) → `src/llm_behavior_lab/models/registry.py` (`build_model_from_config`) → `src/llm_behavior_lab/models/llama/model.py` (`build_llama_model`, `LlamaForCausalLM`).
7. Run a no-grad forward pass.
   - Code path: `scripts/analyze_untrained_model.py` (`main`) → `torch.no_grad` → `src/llm_behavior_lab/models/llama/model.py` (`LlamaForCausalLM.forward`).
8. Convert logits to probabilities over tokenizer-valid tokens.
   - Code path: `scripts/analyze_untrained_model.py` (`main`) → `src/llm_behavior_lab/evaluation/untrained_analysis.py` (`analyze_untrained_outputs`) → `src/llm_behavior_lab/evaluation/output_stats.py` (`logits_to_probabilities`).
9. Compute output entropy and probability concentration.
   - Code path: `src/llm_behavior_lab/evaluation/untrained_analysis.py` (`analyze_untrained_outputs`) → `src/llm_behavior_lab/evaluation/output_stats.py` (`summarize_output_distribution`, `entropy_from_probabilities`, `top1_probability_values`, `topk_probability_mass`).
10. Compute empirical token frequencies from the corpus.
    - Code path: `src/llm_behavior_lab/evaluation/untrained_analysis.py` (`analyze_untrained_outputs`) → `src/llm_behavior_lab/evaluation/token_frequency.py` (`empirical_token_counts`, `empirical_token_frequencies`, `top_token_frequencies`).
11. Compare average predicted probabilities with empirical frequencies.
    - Code path: `src/llm_behavior_lab/evaluation/untrained_analysis.py` (`analyze_untrained_outputs`) → `src/llm_behavior_lab/evaluation/token_frequency.py` (`average_predicted_probabilities`, `top_probability_gaps`, `kl_divergence`, `js_divergence`).
12. Print top-k examples, top-1 prediction summaries, probability-frequency gaps, and divergence summaries.
    - Code path: `scripts/analyze_untrained_model.py` (`main`, `_format_token`) → `src/llm_behavior_lab/evaluation/untrained_analysis.py` (`collect_topk_examples`, `summarize_top1_predictions`) → `src/llm_behavior_lab/inference/generation.py` (`top_k_predictions`).

## Phase 5 flow diagram

```text
configs/*.yaml
   |
   v
scripts/analyze_untrained_model.py
   |
   +--> data/raw/tiny_corpus.txt
   |       |
   |       v
   |   load_text_file
   |       |
   |       v
   |   CharTokenizer.from_text / encode
   |       |
   |       v
   |   split_token_ids
   |       |
   |       v
   |   CausalLMBatcher.get_batch
   |
   +--> build_model_from_config
   |       |
   |       v
   |   LlamaForCausalLM.forward
   |       |
   |       v
   |   logits
   |
   +--> analyze_untrained_outputs
           |
           +--> logits_to_probabilities
           +--> summarize_output_distribution
           +--> empirical_token_frequencies
           +--> average_predicted_probabilities
           +--> top_probability_gaps
           +--> collect_topk_examples
           |
           v
        printed Phase 5 report
```

## Why probabilities are restricted to tokenizer-valid tokens

The tiny LLaMA config currently uses a model vocabulary size of `256`, while the character-level tokenizer built from `data/raw/tiny_corpus.txt` has fewer valid token IDs. During Phase 4 inference and Phase 5 analysis, logits are restricted to the tokenizer vocabulary before softmax when the output needs to be decoded or compared with empirical token frequencies.

This keeps the analysis aligned with the active tokenizer:

- generated token IDs can be decoded by `CharTokenizer.decode`
- predicted probabilities can be compared against empirical frequencies from the corpus
- probability-frequency gaps use the same token index space on both sides

The relevant utilities are:

- `src/llm_behavior_lab/inference/generation.py` (`extract_next_token_logits`)
- `src/llm_behavior_lab/evaluation/output_stats.py` (`logits_to_probabilities`)
- `src/llm_behavior_lab/evaluation/untrained_analysis.py` (`analyze_untrained_outputs`)

## Common debugging paths

| Symptom | Where to look |
|---|---|
| Config file not found or malformed | `scripts/analyze_untrained_model.py` (`load_yaml_config`), `configs/data/tiny_text.yaml`, `configs/model/tiny_llama.yaml` |
| Dataset path error | `scripts/analyze_untrained_model.py` (`resolve_repo_path`), `src/llm_behavior_lab/data/text_dataset.py` (`load_text_file`) |
| Tokenizer cannot decode a token ID | `src/llm_behavior_lab/data/tokenizer.py` (`decode`), check whether model logits were restricted to `tokenizer.vocab_size` |
| Train/validation split too small | `src/llm_behavior_lab/data/text_dataset.py` (`split_token_ids`), `configs/data/tiny_text.yaml` (`batching.block_size`, `dataset.val_fraction`) |
| Batch shape mismatch | `src/llm_behavior_lab/data/dataloader.py` (`CausalLMBatcher.get_batch`), `scripts/check_data_pipeline.py`, `scripts/analyze_untrained_model.py` |
| Model construction failure | `src/llm_behavior_lab/models/registry.py` (`build_model_from_config`), `src/llm_behavior_lab/models/llama/config.py` (`LlamaConfig.validate`) |
| Forward-pass shape error | `src/llm_behavior_lab/models/llama/model.py` (`LlamaForCausalLM.forward`), check `max_seq_len` and `block_size` |
| Analysis metric shape error | `src/llm_behavior_lab/evaluation/output_stats.py`, `src/llm_behavior_lab/evaluation/token_frequency.py`, `src/llm_behavior_lab/evaluation/untrained_analysis.py` |

## Install

From the repository root:

```bash
python3 -m pip install -e ".[dev]"
```

## Run the Phase 2 model sanity check

```bash
python3 scripts/smoke_test_llama.py --config configs/model/tiny_llama.yaml
```

Successful output should include:

```text
Smoke test completed successfully.
Available registered models: ['llama', 'llama_tiny']
Selected model: llama_tiny
Device: cpu
Parameter count: 459392 (459.39K)
Input shape: (2, 16)
Target shape: (2, 16)
Logits shape: (2, 16, 256)
Loss shape: ()
Loss value: ...
```

## Run the Phase 3 data-pipeline check

```bash
python3 scripts/check_data_pipeline.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

Successful output should include:

```text
Phase 3 data pipeline check completed successfully.
Tokenizer type: char
Tokenizer vocab size: ...
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

Successful output should include:

```text
Phase 4 inference check completed successfully.
Prompt: 'The '
Encoded prompt token IDs: [...]
Input tensor shape: (1, 4)
Full logits shape: (1, 4, 256)
Next-token logits shape after tokenizer-vocab restriction: (1, <tokenizer_vocab_size>)
Top-k next-token predictions:
  1. token_id=...
Generated token IDs: [...]
Decoded generated text: ...
Note: the model is untrained, so generated text is expected to be random or meaningless.
```

## Run the Phase 5 untrained-model analysis

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

Successful output should include:

```text
Phase 5 untrained-model analysis completed successfully.
Note: the model is randomly initialized. These numbers describe baseline behavior, not quality.
Tokenizer vocab size: ...
Analyzed batches: 4
Analyzed windows: 16
Analyzed positions: 256
Input tensor shape: (16, 16)
Logits shape: (16, 16, 256)
Probability tensor shape: (16, 16, <tokenizer_vocab_size>)
Mean output entropy: ...
Mean top-1 probability: ...
Mean top-5 probability mass: ...
Top-1 assignment concentration: ...
KL(predicted || empirical): ...
JS(predicted, empirical): ...
```

The script also prints:

```text
Example top-k next-token predictions:
  batch=..., position=..., input_token_id=..., input_token=...
    1. token_id=... token=... probability=...

Most frequent empirical dataset tokens:
  token_id=... token=... count=... frequency=...

Most frequent top-1 predicted tokens:
  token_id=... token=... count=... frequency=...

Largest positive probability-frequency gaps:
  token_id=... token=... predicted=... empirical=... gap=...

Largest negative probability-frequency gaps:
  token_id=... token=... predicted=... empirical=... gap=...
```

The exact metric values depend on random initialization and device.

## Run tests

```bash
python3 -m pytest
```

Successful output should show all tests passing.

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
