# LLM Behavior Lab

LLM Behavior Lab is an incremental research codebase for studying how language-model behavior evolves from random initialization through inference, pre-training, and fine-tuning.

The long-term goal is to analyze how architectural and training choices affect:

- output behavior at initialization
- gradient stability
- learning dynamics
- training efficiency
- predictive bias
- token, word, and subgroup preference
- convergence behavior
- perplexity and standard language-modeling performance

The project avoids treating models as black-box imports. Model components are implemented explicitly so they can be inspected, modified, and instrumented during experiments.

## Current phase

The repository is currently in **Phase 5: Baseline Untrained-Model Analysis**, refined with an optional **per-layer gradient-stability diagnostic**.

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
- inference utilities for prompt encoding, logits inspection, top-k predictions, greedy decoding, sampling, and short generation

Phase 5 adds:

- output-distribution statistics for the untrained model
- entropy and probability-concentration summaries
- top-k prediction examples at selected positions
- empirical token-frequency computation from the tiny corpus
- comparison between average model probabilities and empirical token frequencies
- largest positive and negative probability-frequency gaps
- top-1 predicted-token frequency summaries
- simple KL and Jensen-Shannon divergence summaries
- an optional per-layer squared L2 gradient-norm diagnostic
- a log-linear gradient trend fit across model depth
- lightweight JSON/CSV saving for gradient-norm results
- runnable analysis scripts and tests

The model is still untrained. Phase 5 does **not** measure language quality. It establishes reproducible baseline signals for how the randomly initialized model behaves before any optimization.

## Repository structure

```text
IGB_LLMs/
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
    test_evaluation.py
    test_gradient_norms.py
    test_imports.py
    test_inference.py
    test_llama_shapes.py

  pyproject.toml
  README.md
```

## Directory guide

| Path | Purpose |
|---|---|
| `configs/` | YAML files controlling model size, data source, batching, seed, and device. |
| `data/` | Local raw/downloaded datasets for smoke tests, inference checks, and initialization analysis.. See [`data/README.md`](data/README.md) for dataset-source details, config usage, and data-pipeline commands. |
| `scripts/` | Runnable entry points for sanity checks, inference, and Phase 5 analysis. |
| `src/llm_behavior_lab/models/` | Model interface, registry, and explicit LLaMA-style implementation. |
| `src/llm_behavior_lab/data/` | Text loading, tokenization, splitting, and causal LM batching utilities. |
| `src/llm_behavior_lab/inference/` | Prompt preparation, logits extraction, probability extraction, decoding, and generation. |
| `src/llm_behavior_lab/evaluation/` | Output-distribution, token-frequency, untrained-analysis, and gradient-norm diagnostics. |
| `src/llm_behavior_lab/utils/` | Shared helpers for seeds, device selection, and parameter counting. |
| `tests/` | Lightweight tests covering imports, model shapes, data pipeline, inference, evaluation, and gradient norms. |


## Git hygiene for datasets and experiments

The repository includes a `.gitignore` file that ignores the full `data/` directory:

```text
data/
```

This is intentional now that the project can use larger local or downloaded language-modeling datasets. Dataset **configs** remain tracked under `configs/data/`, while raw corpora, downloaded files, and generated dataset artifacts should live under `data/` and stay out of Git.

Important Git detail: `.gitignore` only prevents new untracked files from being added. If `data/raw/tiny_corpus.txt` or any other file under `data/` is already tracked in your repository history, Git will keep tracking it until you remove it from the index with:

```bash
git rm -r --cached data
git add .gitignore configs/data README.md
git commit -m "Ignore local dataset directory"
```

After that, keep or recreate local datasets on disk as needed, but do not commit them.

## File/function map

### Model files

- `src/llm_behavior_lab/models/base.py`
  - Defines `ModelOutput`, the standard model-return object.
  - Defines `BaseLanguageModel`, the shared interface expected by scripts and evaluation utilities.
  - Provides helpers such as `device` and `count_parameters`.

- `src/llm_behavior_lab/models/registry.py`
  - Defines the model registry.
  - Provides `register_model`, `build_model`, and `build_model_from_config`.
  - Allows scripts to instantiate models from YAML configs without hard-coding model classes.

- `src/llm_behavior_lab/models/llama/config.py`
  - Defines `LlamaConfig`.
  - Validates key architecture settings such as `dim`, `n_layers`, `n_heads`, `n_kv_heads`, `max_seq_len`, and `vocab_size`.

- `src/llm_behavior_lab/models/llama/model.py`
  - Implements the explicit LLaMA-style decoder-only model.
  - Includes token embeddings, RMSNorm, RoPE, grouped-query self-attention, optional KV-cache buffers, SwiGLU feed-forward blocks, residual decoder blocks, final normalization, logits, and optional cross-entropy loss.

### Data files

- `data/raw/tiny_corpus.txt`
  - Tiny local corpus used to test the data pipeline and produce deterministic tokenizer/data behavior.

- `src/llm_behavior_lab/data/tokenizer.py`
  - Defines `CharTokenizer`.
  - Builds a deterministic character-level vocabulary from text.
  - Provides `encode` and `decode`.

- `src/llm_behavior_lab/data/text_dataset.py`
  - Provides `load_text_file`.
  - Provides `split_token_ids` for deterministic train/validation token splits.

- `src/llm_behavior_lab/data/dataloader.py`
  - Defines `CausalLMBatch`.
  - Defines `CausalLMBatcher`.
  - Converts token streams into shifted causal language-modeling batches:

```text
input_ids = tokens[t : t + block_size]
targets   = tokens[t + 1 : t + block_size + 1]
```

### Inference files

- `src/llm_behavior_lab/inference/generation.py`
  - `prepare_prompt_tensor`: encode a prompt and create a `[1, sequence]` tensor.
  - `extract_logits`: run the model in eval/no-grad mode and return full logits.
  - `extract_next_token_logits`: extract final-position logits, optionally restricted to tokenizer-valid IDs.
  - `next_token_probabilities`: apply softmax with temperature.
  - `top_k_predictions`: inspect likely next tokens and decode them.
  - `select_next_token`: greedy or sampling-based next-token selection.
  - `generate_text`: short generation from an untrained model.

### Evaluation/analysis files

- `src/llm_behavior_lab/evaluation/output_stats.py`
  - `logits_to_probabilities`
  - `entropy_from_probabilities`
  - `topk_probability_mass`
  - `top1_token_ids`
  - `top1_probability_values`
  - `summarize_output_distribution`

- `src/llm_behavior_lab/evaluation/token_frequency.py`
  - `empirical_token_counts`
  - `empirical_token_frequencies`
  - `average_predicted_probabilities`
  - `top_token_frequencies`
  - `top_probability_gaps`
  - `kl_divergence`
  - `js_divergence`

- `src/llm_behavior_lab/evaluation/untrained_analysis.py`
  - Combines output-distribution and token-frequency metrics.
  - Produces a structured `UntrainedAnalysisResult`.
  - Collects top-k examples, top-1 prediction summaries, probability-frequency gaps, and divergence values.

- `src/llm_behavior_lab/evaluation/gradient_norms.py`
  - Computes per-layer squared L2 gradient norms.
  - Fits a log-linear gradient trend across layer index.
  - Saves gradient results to JSON and CSV.
  - Uses forward hooks on decoder block outputs, so the model implementation does not need to be modified.

### Utility files

- `src/llm_behavior_lab/utils/seed.py`
  - `seed_everything` for reproducibility.

- `src/llm_behavior_lab/utils/device.py`
  - `get_device` for `cpu`, `cuda`, `mps`, or auto-selection.

- `src/llm_behavior_lab/utils/params.py`
  - Parameter counting and formatting helpers.

## Script-to-module map

| Script | Purpose | Main modules used |
|---|---|---|
| `scripts/smoke_test_llama.py` | Check that the LLaMA-style model builds and runs on dummy token IDs. | `models.registry`, `models.llama`, `utils.device`, `utils.seed`, `utils.params` |
| `scripts/check_data_pipeline.py` | Check that local text becomes causal LM batches compatible with the model. | `data.text_dataset`, `data.tokenizer`, `data.dataloader`, `models.registry` |
| `scripts/run_inference.py` | Check prompt encoding, logits extraction, top-k next-token inspection, and short generation. | `data.tokenizer`, `data.text_dataset`, `inference.generation`, `models.registry` |
| `scripts/analyze_untrained_model.py` | Analyze initialization-time output behavior and optionally gradient stability. | `data.*`, `models.registry`, `evaluation.output_stats`, `evaluation.token_frequency`, `evaluation.untrained_analysis`, `evaluation.gradient_norms` |

## Current analysis capabilities

### Output-distribution analysis

The untrained model is evaluated on sampled causal LM windows. The analysis computes:

- logits shape
- probability tensor shape
- entropy per position
- mean/min/max entropy
- mean top-1 probability
- mean top-k probability mass
- top-k examples for selected positions

These metrics describe how concentrated or diffuse the untrained output distribution is.

### Empirical token-frequency comparison

The tokenizer is built from the tiny corpus, and empirical token frequencies are computed from the full tokenized text. The model's average predicted probabilities are then compared to these empirical frequencies.

This produces:

- most frequent empirical dataset tokens
- average predicted probability per token
- largest positive probability-frequency gaps
- largest negative probability-frequency gaps
- KL divergence between predicted and empirical distributions
- Jensen-Shannon divergence between predicted and empirical distributions

This is an early baseline for later bias/preference analysis.

### Top-1 prediction behavior

The analysis also measures how often each token is the model's top-1 prediction across analyzed positions.

This produces:

- top-1 predicted-token counts
- top-1 predicted-token frequencies
- top-1 assignment concentration

A high top-1 concentration can indicate that the random model repeatedly prefers a small subset of tokens.

### Per-layer gradient-stability diagnostic

The optional gradient diagnostic measures whether gradients are roughly stable through transformer depth at random initialization.

For a causal language-modeling batch, the script computes the loss, backpropagates through the untrained model, and measures the squared L2 norm of the gradient with respect to each decoder block output:

```text
g_l = || dL / dh_l ||_2^2
```

where:

- `L` is the causal LM loss
- `h_l` is the output residual-stream tensor of decoder block `l`
- `g_l` is the per-layer squared L2 gradient norm

The implementation uses forward hooks on `model.layers`, so it does not alter the model code.

To summarize whether gradients decay or grow with depth, the diagnostic fits:

```text
log(g_l + eps) = intercept + slope * layer_index
```

Interpretation:

| Slope | Meaning |
|---:|---|
| approximately `0` | Gradients are roughly stable across depth. |
| negative | Gradients decay with depth. |
| positive | Gradients grow with depth. |
| large absolute value | Stronger vanishing or exploding trend. |

The diagnostic reports:

- raw per-layer squared L2 gradient norms
- log gradient norms
- fitted slope
- fitted intercept
- slope standard error when statistically available
- R² when defined
- mean loss used for the gradient diagnostic

The diagnostic is disabled by default because it requires a backward pass.

## Phase 5 execution flow

The default untrained-model analysis follows this path:

1. Load data and model configs.
   - Code path: `scripts/analyze_untrained_model.py` → `load_yaml_config`
   - Files: `configs/data/tiny_text.yaml`, `configs/model/tiny_llama.yaml`

2. Load the tiny local text corpus.
   - Code path: `resolve_repo_path` → `load_text_file`
   - Files: `data/raw/tiny_corpus.txt`, `src/llm_behavior_lab/data/text_dataset.py`

3. Build the character-level tokenizer.
   - Code path: `CharTokenizer.from_text`
   - File: `src/llm_behavior_lab/data/tokenizer.py`

4. Tokenize the corpus and create train/validation splits.
   - Code path: `CharTokenizer.encode` → `split_token_ids`
   - Files: `tokenizer.py`, `text_dataset.py`

5. Sample causal LM windows from the requested split.
   - Code path: `CausalLMBatcher.get_batch`
   - File: `src/llm_behavior_lab/data/dataloader.py`

6. Instantiate the randomly initialized LLaMA-style model.
   - Code path: `build_model_from_config` → `build_llama_model` → `LlamaForCausalLM`
   - Files: `models/registry.py`, `models/llama/model.py`

7. Run a no-grad forward pass for output analysis.
   - Code path: `torch.no_grad()` → `LlamaForCausalLM.forward`

8. Convert logits to tokenizer-valid probabilities.
   - Code path: `analyze_untrained_outputs` → `logits_to_probabilities`
   - Files: `evaluation/untrained_analysis.py`, `evaluation/output_stats.py`

9. Compute output-distribution metrics.
   - Code path: `summarize_output_distribution`
   - File: `evaluation/output_stats.py`

10. Compute empirical token-frequency comparison.
    - Code path: `empirical_token_frequencies`, `average_predicted_probabilities`, `top_probability_gaps`
    - File: `evaluation/token_frequency.py`

11. Print the Phase 5 report.
    - Code path: `scripts/analyze_untrained_model.py`

12. Optionally compute gradient norms.
    - Code path: `compute_per_layer_gradient_norms`
    - File: `evaluation/gradient_norms.py`
    - Trigger: `--compute-grad-norms`

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
   |       |
   |       +--> logits_to_probabilities
   |       +--> summarize_output_distribution
   |       +--> empirical_token_frequencies
   |       +--> average_predicted_probabilities
   |       +--> top_probability_gaps
   |       +--> collect_topk_examples
   |       |
   |       v
   |    printed Phase 5 report
   |
   +--> optional: compute_per_layer_gradient_norms
           |
           +--> forward pass with targets
           +--> retain decoder-block output gradients
           +--> backward pass
           +--> squared L2 norms per layer
           +--> log-linear trend fit
           |
           v
        gradient_norms.json + gradient_norms.csv
```

## Gradient-norm diagnostic flow diagram

```text
CausalLMBatch(input_ids, targets)
   |
   v
LlamaForCausalLM.forward(input_ids, targets)
   |
   +--> decoder block 0 output h_0 -- retain grad
   +--> decoder block 1 output h_1 -- retain grad
   +--> ...
   |
   v
causal LM loss
   |
   v
loss.backward()
   |
   v
for each layer l:
    g_l = sum((dL / dh_l)^2)
   |
   v
log(g_l + eps) = intercept + slope * layer_index
   |
   v
print summary + save JSON/CSV
```

## Why probabilities are restricted to tokenizer-valid tokens

The tiny LLaMA config currently uses a model vocabulary size of `256`, while the character-level tokenizer built from `data/raw/tiny_corpus.txt` has fewer valid token IDs. During inference and Phase 5 analysis, logits are restricted to the tokenizer vocabulary before softmax when the output needs to be decoded or compared with empirical token frequencies.

This keeps the analysis aligned with the active tokenizer:

- generated token IDs can be decoded by `CharTokenizer.decode`
- predicted probabilities can be compared against empirical frequencies from the corpus
- probability-frequency gaps use the same token index space on both sides

The relevant utilities are:

- `src/llm_behavior_lab/inference/generation.py` (`extract_next_token_logits`)
- `src/llm_behavior_lab/evaluation/output_stats.py` (`logits_to_probabilities`)
- `src/llm_behavior_lab/evaluation/untrained_analysis.py` (`analyze_untrained_outputs`)

## Output files produced by gradient analysis

When `--compute-grad-norms` is enabled, the script saves results under:

```text
outputs/phase5_gradient_norms/
  gradient_norms.json
  gradient_norms.csv
```

The JSON file contains:

```text
definition
mean_loss
num_batches
layer_indices
squared_l2_norms
log_squared_l2_norms
trend_fit
per_layer
metadata
```

The CSV file contains one row per layer:

```text
layer_index,squared_l2_norm,log_squared_l2_norm
```

This is intentionally lightweight. Full experiment-directory management and structured logging are deferred to Phase 6.

## Common debugging paths

| Symptom | Where to look |
|---|---|
| Config file not found or malformed | `scripts/analyze_untrained_model.py` (`load_yaml_config`), `configs/data/tiny_text.yaml`, `configs/model/tiny_llama.yaml` |
| Dataset path error | `resolve_repo_path`, `data/text_dataset.py` (`load_text_file`) |
| Tokenizer cannot decode a token ID | `data/tokenizer.py` (`decode`), check whether logits were restricted to `tokenizer.vocab_size` |
| Train/validation split too small | `data/text_dataset.py` (`split_token_ids`), `configs/data/tiny_text.yaml` |
| Batch shape mismatch | `data/dataloader.py` (`CausalLMBatcher.get_batch`) |
| Model construction failure | `models/registry.py`, `models/llama/config.py` |
| Forward-pass shape error | `models/llama/model.py`, check `max_seq_len` and `block_size` |
| Analysis metric shape error | `evaluation/output_stats.py`, `evaluation/token_frequency.py`, `evaluation/untrained_analysis.py` |
| Gradient hooks fail | `evaluation/gradient_norms.py`, check that the model exposes `model.layers` |
| Gradient output files are missing | Confirm `--compute-grad-norms` was passed and inspect `--grad-norm-output-dir` |
| Slope standard error is `not available` | This is expected for a 2-layer model because the linear fit has zero residual degrees of freedom |

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

Without gradient norms:

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
Most frequent empirical dataset tokens:
Most frequent top-1 predicted tokens:
Largest positive probability-frequency gaps:
Largest negative probability-frequency gaps:
```

## Run the Phase 5 analysis with gradient norms

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml \
  --compute-grad-norms
```

Successful additional output should include:

```text
Computing per-layer gradient norms...

Per-layer gradient norm diagnostic:
  Definition: squared_l2_norm_of_gradient_wrt_decoder_block_output
  Gradient batches: 1
  Mean gradient-diagnostic loss: ...
  Squared L2 gradient norms: [...]
  Log squared L2 gradient norms: [...]
  Log-linear slope: ...
  Log-linear intercept: ...
  Slope standard error: ...
  R^2: ...
  Saved gradient JSON: outputs/phase5_gradient_norms/gradient_norms.json
  Saved gradient CSV: outputs/phase5_gradient_norms/gradient_norms.csv
```

Optional controls:

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml \
  --compute-grad-norms \
  --grad-norm-num-batches 2 \
  --grad-norm-eps 1e-12 \
  --grad-norm-output-dir outputs/phase5_gradient_norms
```

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
- gradient tracking over training checkpoints
- full validation-set evaluation
- fine-tuning
- multi-model comparison
- advanced bias or group-based evaluation metrics

The gradient diagnostic is currently an initialization-time check only. Phase 6 will add structured logging/checkpoint infrastructure, and later phases will reuse this diagnostic across training checkpoints.

## Next phases

Planned next steps:

1. **Phase 6 — Logging and checkpoint infrastructure**
   - experiment folders
   - JSON/CSV logs
   - metadata storage
   - config snapshots
   - reproducible run records
   - lightweight artifact persistence for Phase 5 outputs

2. **Phase 7 — Pre-training loop**
   - optimizer
   - learning-rate schedule
   - loss logging
   - validation checks
   - checkpoint evaluation

3. **Phase 8 — Training-dynamics analysis**
   - loss/perplexity curves
   - output-distribution changes
   - gradient diagnostics over checkpoints
   - bias metrics over time

4. **Phase 9 — Fine-tuning pipeline**
   - supervised fine-tuning
   - checkpoint-based evaluation
   - comparison between pre-training and fine-tuning behavior

5. **Phase 10 — Model extension phase**
   - additional architectures
   - Gemma-oriented variants
   - cross-model behavior comparison
