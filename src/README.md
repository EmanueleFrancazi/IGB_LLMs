# Source package

This folder contains the reusable Python source code for the LLM Behavior Lab project.

The project uses a `src/` layout: the importable package lives under `src/llm_behavior_lab/`, while runnable scripts live separately under `scripts/`. This keeps reusable implementation code separate from command-line orchestration.

At the current project stage, the source package includes:

- data utilities for text loading, tokenization, splitting, and batching
- an explicit LLaMA-style decoder-only model implementation
- model configuration and model registry utilities
- inference utilities for prompt preparation, logits extraction, probability inspection, and short generation
- evaluation utilities for untrained-model output analysis and gradient-norm diagnostics
- shared utilities for device selection, parameter counting, and seeding

---

## Purpose of the `src/` folder

The `src/` folder is the main implementation area of the repository.

It is different from other top-level folders:

```text
src/
  reusable Python package code

scripts/
  command-line entry points that call into src/

configs/
  YAML configuration files consumed by scripts and source utilities

data/
  local raw or downloaded datasets

tests/
  automated tests for package behavior

README.md
  project-level roadmap and usage guide
```

Reusable logic should live under `src/llm_behavior_lab/`.

Scripts should stay thin: they parse arguments, load configs, call package functions/classes, and print or save outputs. This makes the code easier to test, reuse, and extend.

---

## Package overview

The main package is:

```text
src/llm_behavior_lab/
```

It can be imported as:

```python
import llm_behavior_lab
```

The package currently contains:

```text
src/llm_behavior_lab/
  __init__.py

  data/
    __init__.py
    dataloader.py
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
```

The package is imported by:

- scripts under `scripts/`
- tests under `tests/`
- future notebooks or experiment runners
- future training and checkpoint-analysis code

---

## Current source-code structure

## `llm_behavior_lab.data`

The `data` package contains utilities for early text-only language-modeling experiments.

Current files:

```text
src/llm_behavior_lab/data/
  __init__.py
  dataloader.py
  text_dataset.py
  tokenizer.py
```

### Main responsibilities

- load local UTF-8 text files
- split token IDs into train and validation portions
- build a deterministic character-level tokenizer
- encode and decode text
- create causal language-modeling batches

### Important objects and functions

```text
CharTokenizer
TokenSplits
load_text_file
split_token_ids
CausalLMBatch
CausalLMBatcher
```

### Where to look

| Task | File |
|---|---|
| Change local text loading behavior | `data/text_dataset.py` |
| Change train/validation split behavior | `data/text_dataset.py` |
| Modify the character tokenizer | `data/tokenizer.py` |
| Change batch sampling or input/target shifting | `data/dataloader.py` |
| Export new data utilities | `data/__init__.py` |

### Scripts using this package

- `scripts/check_data_pipeline.py`
- `scripts/run_inference.py`
- `scripts/analyze_untrained_model.py`

---

## `llm_behavior_lab.models`

The `models` package contains the model interface, model registry, and explicit LLaMA-style model implementation.

Current files:

```text
src/llm_behavior_lab/models/
  __init__.py
  base.py
  registry.py
  llama/
    __init__.py
    config.py
    model.py
```

### Main responsibilities

- define a shared language-model interface
- standardize model outputs
- register model constructors by name
- construct models from config dictionaries
- implement the current LLaMA-style decoder-only model explicitly

### Important objects and functions

```text
BaseLanguageModel
ModelOutput
register_model
get_model_builder
build_model
build_model_from_config
list_models
LlamaConfig
LlamaForCausalLM
LlamaDecoderBlock
LlamaSelfAttention
LlamaFeedForward
LlamaRMSNorm
```

### Current LLaMA-style components

The current model implementation includes:

- token embeddings
- RMSNorm
- rotary position embeddings
- grouped-query self-attention
- optional KV-cache path
- SwiGLU feed-forward blocks
- residual decoder blocks
- final normalization
- language-modeling logits
- optional cross-entropy loss when targets are provided

### Where to look

| Task | File |
|---|---|
| Change shared model output format | `models/base.py` |
| Add or inspect model registration | `models/registry.py` |
| Change LLaMA model hyperparameter validation | `models/llama/config.py` |
| Modify LLaMA architecture internals | `models/llama/model.py` |
| Register a new model family | `models/__init__.py` and/or a new subpackage |

### Scripts using this package

- `scripts/smoke_test_llama.py`
- `scripts/check_data_pipeline.py`
- `scripts/run_inference.py`
- `scripts/analyze_untrained_model.py`

---

## `llm_behavior_lab.inference`

The `inference` package contains utilities for running the model in inference mode.

Current files:

```text
src/llm_behavior_lab/inference/
  __init__.py
  generation.py
```

### Main responsibilities

- encode text prompts into token IDs
- create model-ready prompt tensors
- run model forward passes without gradients
- extract full logits
- extract next-token logits
- convert logits to probabilities
- inspect top-k next-token predictions
- select next tokens with greedy or sampling decoding
- generate short text continuations

### Important objects and functions

```text
PromptBatch
TopKPrediction
GenerationResult
prepare_prompt_tensor
extract_logits
extract_next_token_logits
next_token_probabilities
top_k_predictions
select_next_token
generate_text
```

### Where to look

| Task | File |
|---|---|
| Modify prompt tensor preparation | `inference/generation.py` |
| Change logits/probability extraction | `inference/generation.py` |
| Add decoding strategies | `inference/generation.py` |
| Change short generation behavior | `inference/generation.py` |
| Export new inference utilities | `inference/__init__.py` |

### Scripts using this package

- `scripts/run_inference.py`
- `scripts/analyze_untrained_model.py` indirectly uses top-k prediction utilities through evaluation modules

---

## `llm_behavior_lab.evaluation`

The `evaluation` package contains analysis utilities for model-output behavior.

Current files:

```text
src/llm_behavior_lab/evaluation/
  __init__.py
  gradient_norms.py
  output_stats.py
  token_frequency.py
  untrained_analysis.py
```

### Main responsibilities

- convert logits to probabilities
- compute entropy and probability concentration
- inspect top-1 and top-k output behavior
- compute empirical token frequencies
- compare predicted probability mass with dataset frequencies
- compute KL and Jensen-Shannon divergence summaries
- assemble Phase 5 untrained-model analysis results
- compute per-layer squared L2 gradient norms
- fit a log-linear trend to gradient norms across depth

### Important objects and functions

```text
OutputDistributionSummary
logits_to_probabilities
entropy_from_probabilities
topk_probability_mass
top1_token_ids
top1_probability_values
summarize_output_distribution

TokenFrequencySummary
TokenProbabilityGap
empirical_token_counts
empirical_token_frequencies
average_predicted_probabilities
top_token_frequencies
top_probability_gaps
kl_divergence
js_divergence

TopKPositionExample
UntrainedAnalysisResult
summarize_top1_predictions
collect_topk_examples
analyze_untrained_outputs

LayerGradientNorm
GradientTrendFit
GradientNormResult
fit_log_gradient_trend
compute_per_layer_gradient_norms
gradient_norm_result_to_dict
save_gradient_norm_result
```

### Where to look

| Task | File |
|---|---|
| Add entropy or concentration metrics | `evaluation/output_stats.py` |
| Add token-frequency comparison metrics | `evaluation/token_frequency.py` |
| Add a new untrained-model summary section | `evaluation/untrained_analysis.py` |
| Modify gradient-norm diagnostics | `evaluation/gradient_norms.py` |
| Export new evaluation utilities | `evaluation/__init__.py` |

### Scripts using this package

- `scripts/analyze_untrained_model.py`

---

## `llm_behavior_lab.utils`

The `utils` package contains small helpers shared across scripts and modules.

Current files:

```text
src/llm_behavior_lab/utils/
  __init__.py
  device.py
  params.py
  seed.py
```

### Main responsibilities

- select a PyTorch device from a user-friendly string
- count model parameters
- format parameter counts
- seed random number generators

### Important functions

```text
get_device
count_parameters
format_parameter_count
seed_everything
```

### Where to look

| Task | File |
|---|---|
| Change device selection policy | `utils/device.py` |
| Change parameter-count formatting | `utils/params.py` |
| Change seeding behavior | `utils/seed.py` |
| Export new helpers | `utils/__init__.py` |

### Scripts using this package

All current scripts use at least one utility from this package.

---

## Execution flow through `src/`

Most commands follow this pattern:

```text
script in scripts/
   |
   v
parse command-line arguments
   |
   v
load YAML configs from configs/
   |
   v
seed and select device using llm_behavior_lab.utils
   |
   v
prepare data using llm_behavior_lab.data
   |
   v
build model using llm_behavior_lab.models
   |
   v
run inference or analysis using llm_behavior_lab.inference / llm_behavior_lab.evaluation
   |
   v
print diagnostics or save lightweight outputs
```

Examples:

### Model sanity check

```text
scripts/smoke_test_llama.py
   |
   v
models.build_model_from_config
   |
   v
LlamaForCausalLM forward pass on dummy token IDs
```

### Data/model compatibility check

```text
scripts/check_data_pipeline.py
   |
   v
data.load_text_file
   |
   v
data.CharTokenizer
   |
   v
data.split_token_ids
   |
   v
data.CausalLMBatcher
   |
   v
models.build_model_from_config
   |
   v
model forward pass with input_ids and targets
```

### Inference check

```text
scripts/run_inference.py
   |
   v
data.CharTokenizer
   |
   v
models.build_model_from_config
   |
   v
inference.prepare_prompt_tensor
   |
   v
inference.extract_logits
   |
   v
inference.top_k_predictions
   |
   v
inference.generate_text
```

### Untrained-model analysis

```text
scripts/analyze_untrained_model.py
   |
   v
data pipeline
   |
   v
model forward pass
   |
   v
evaluation.analyze_untrained_outputs
   |
   v
optional evaluation.compute_per_layer_gradient_norms
```

---

## Relationship with other folders

### `scripts/`

Scripts are executable entry points. They call reusable code from `src/`.

If you are adding a new command-line workflow, add the orchestration script under `scripts/`, but keep reusable functions/classes under `src/llm_behavior_lab/`.

### `configs/`

Configs define data and model settings.

The source package consumes config dictionaries but does not own the YAML files. Config files should stay under `configs/`.

### `data/`

The `data/` folder stores local raw or downloaded datasets.

The source code that loads and processes those files lives under `src/llm_behavior_lab/data/`.

### `tests/`

Tests verify source package behavior.

If you add or modify reusable code under `src/`, add or update tests under `tests/`.

### Root `README.md`

The root README is the project-wide guide.

It should explain the whole repository, not every implementation detail of each source module.

### Local README files

Folder-level README files provide local guidance.

Current local documentation includes:

```text
data/README.md
scripts/README.md
src/README.md
```

---

## Extension guide

Use this section when deciding where to add new functionality.

### Add a new model architecture

Likely location:

```text
src/llm_behavior_lab/models/
```

Suggested structure:

```text
src/llm_behavior_lab/models/new_model/
  __init__.py
  config.py
  model.py
```

Then register it through the model registry so scripts can build it from config.

### Modify the current LLaMA-style architecture

Likely location:

```text
src/llm_behavior_lab/models/llama/model.py
src/llm_behavior_lab/models/llama/config.py
```

Use this for attention changes, normalization changes, MLP changes, RoPE changes, or layer-structure experiments.

### Add a new dataset source

Likely location:

```text
src/llm_behavior_lab/data/
```

The current attached source tree contains local text loading, tokenization, splitting, and batching utilities. A future dataset-source abstraction should also live under this package so scripts can keep using a shared data path.

### Add a new tokenizer

Likely location:

```text
src/llm_behavior_lab/data/tokenizer.py
```

If tokenizer support grows beyond one file, create a dedicated tokenizer submodule under `data/`.

### Add a new inference decoding method

Likely location:

```text
src/llm_behavior_lab/inference/generation.py
```

Examples:

- nucleus sampling
- repetition penalty
- beam search
- constrained decoding
- KV-cache optimized generation

### Add a new evaluation metric

Likely location:

```text
src/llm_behavior_lab/evaluation/
```

Choose the file based on metric type:

```text
output_stats.py        output-distribution metrics
token_frequency.py     frequency/probability comparisons
untrained_analysis.py  combined untrained-model summaries
gradient_norms.py      gradient-stability metrics
```

Create a new file if the metric category becomes large enough.

### Add structured logging

Likely future location:

```text
src/llm_behavior_lab/logging/
```

This should include reusable experiment directory creation, metric writers, metadata saving, and config snapshots. The scripts should call into this package rather than implementing logging directly.

### Add training loops

Likely future location:

```text
src/llm_behavior_lab/training/
```

This should contain reusable training-step logic, optimizer/scheduler construction, validation loops, checkpoint saving, and training-state handling.

### Add fine-tuning utilities

Likely future location:

```text
src/llm_behavior_lab/finetuning/
```

or, if kept smaller at first:

```text
src/llm_behavior_lab/training/
```

The exact split can be decided when supervised fine-tuning is implemented.

### Add checkpoint loading/evaluation

Likely future locations:

```text
src/llm_behavior_lab/checkpointing/
src/llm_behavior_lab/evaluation/
```

Checkpoint I/O should be separate from metric computation so evaluation functions remain reusable.

---

## Design principles for `src/`

The source package should follow these principles:

- keep reusable logic inside `src/`
- keep scripts thin and orchestration-focused
- keep model implementations explicit and inspectable
- prefer simple, testable functions
- avoid unnecessary abstraction
- expose stable dataclasses for structured results
- reuse utilities instead of duplicating code
- keep configs explicit
- keep data/model/inference/evaluation concerns separated
- preserve compatibility with future training and checkpoint phases
- add tests whenever source behavior changes

---

## Common navigation questions

### Where do I change the model architecture?

Use:

```text
src/llm_behavior_lab/models/llama/model.py
```

For config validation or hyperparameters, use:

```text
src/llm_behavior_lab/models/llama/config.py
```

### Where do I add a new model family?

Use:

```text
src/llm_behavior_lab/models/
```

Create a new subpackage and register a builder through:

```text
src/llm_behavior_lab/models/registry.py
```

### Where do I change data loading or batching?

Use:

```text
src/llm_behavior_lab/data/text_dataset.py
src/llm_behavior_lab/data/tokenizer.py
src/llm_behavior_lab/data/dataloader.py
```

### Where do I add a new dataset source later?

Add the reusable loader under:

```text
src/llm_behavior_lab/data/
```

Then call it from scripts rather than implementing loading logic directly in a script.

### Where do I modify inference behavior?

Use:

```text
src/llm_behavior_lab/inference/generation.py
```

### Where do I add a new output-analysis metric?

Use:

```text
src/llm_behavior_lab/evaluation/
```

For output distribution metrics, start with:

```text
src/llm_behavior_lab/evaluation/output_stats.py
```

For token-frequency comparisons, start with:

```text
src/llm_behavior_lab/evaluation/token_frequency.py
```

### Where do I modify gradient-norm diagnostics?

Use:

```text
src/llm_behavior_lab/evaluation/gradient_norms.py
```

### Where should training code go later?

Training code should not be added directly to existing scripts.

Add reusable training code under a future package such as:

```text
src/llm_behavior_lab/training/
```

Then add a thin script under:

```text
scripts/
```

### Where are helper utilities defined?

Use:

```text
src/llm_behavior_lab/utils/
```

Current helpers cover device selection, parameter counting, and seeding.

---

## Testing source changes

After changing code under `src/`, run:

```bash
python3 -m pytest
```

For script-level manual checks, run from the repository root:

```bash
python3 scripts/smoke_test_llama.py --config configs/model/tiny_llama.yaml

python3 scripts/check_data_pipeline.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml

python3 scripts/run_inference.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml

python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

If a source module changes, prefer adding a direct unit test under `tests/` in addition to checking a script manually.
