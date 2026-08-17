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

**Phase 6 (Experiment Persistence and Checkpointing) is complete.**

Current work is a **pre-Phase-7 research extension**: an initialization-distribution
experiment measuring how a randomly initialized model's token guesses compare with the
corpus token distribution. **Phase 7 (training) has not started.**

Earlier phases established the package structure, a shared model interface and registry,
an explicit LLaMA-style decoder-only model, a character tokenizer with train/validation
splitting and causal LM batching, inference utilities, initialization-time output and
gradient diagnostics, a config-driven multi-dataset layer with local-first resolution, and
structured experiment runs with metrics, array artifacts, and checkpoints.

The model remains **untrained** throughout. None of this measures language quality.

### The initialization-distribution experiment

At random initialization, how do the distributions of the model's **selected token
guesses** compare with the empirical token distribution of the corpus — in overall
concentration and token by token — and how stable is that across independent
initializations?

Everything except the model-initialization seed is held fixed: corpus, tokenizer,
vocabulary, analysis split, and the evaluation positions themselves. Two policies read the
same logits — greedy `argmax`, and temperature/top-p nucleus sampling with its own
sampling seed and replicates averaged within an initialization.

[`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) is the authoritative scientific
document: every definition, every measure, and the validated baseline and pilot history.

## Repository structure

```text
IGB_LLMs/
  configs/
    data/
      tiny_text.yaml
      wikitext2.yaml
      tinystories.yaml
      wikitext2_subword.yaml
    model/
      tiny_llama.yaml
      tiny_llama_32k.yaml
    experiment/
      phase6_smoke.yaml
      untrained_baseline.yaml
      initialization_distribution.yaml

  data/
    raw/
      tiny_corpus.txt

  scripts/
    smoke_test_llama.py
    check_data_pipeline.py
    run_inference.py
    analyze_untrained_model.py
    check_experiment_tracking.py
    prepare_dataset.py
    run_initialization_distribution_experiment.py
    run_initialization_scale_experiment.py
    benchmark_position_gradients.py
    render_record_figures.py

  docs/
    EXPERIMENT_LOG.md

  notebooks/
    initialization_distribution.ipynb

  src/
    llm_behavior_lab/
      __init__.py

      analysis/
        __init__.py
        aggregation.py
        figures.py
        gradients.py
        nulls.py
        predictive.py
        records.py
        scale_comparison.py
        transition.py

      data/
        __init__.py
        cli.py
        config.py
        dataloader.py
        errors.py
        huggingface.py
        prepared.py
        pretrained_tokenizer.py
        resolver.py
        text_dataset.py
        tokenizer.py

      evaluation/
        __init__.py
        gradient_norms.py
        guessing.py
        init_distribution.py
        input_conditions.py
        output_stats.py
        position_gradients.py
        token_frequency.py
        untrained_analysis.py

      experiment/
        __init__.py
        arrays.py
        checkpoints.py
        config.py
        metrics.py
        naming.py
        run.py
        serialization.py

      inference/
        __init__.py
        generation.py

      models/
        __init__.py
        base.py
        initialization_scale.py
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
    test_dataset_cli.py
    test_dataset_config.py
    test_dataset_huggingface.py
    test_dataset_prepared.py
    test_dataset_resolver.py
    test_dataset_tracking.py
    test_evaluation.py
    test_experiment_tracking.py
    test_gradient_norms.py
    test_imports.py
    test_inference.py
    test_llama_shapes.py
    test_migrated_workflows.py
    test_persisted_analysis_run.py
    test_guessing.py
    test_init_distribution.py
    test_tokenizers.py
    test_analysis_records.py
    test_analysis_aggregation.py
    test_analysis_figures.py
    test_experiment_naming.py
    test_input_conditions.py
    test_uniform_null.py
    test_temperature_sweep.py

  pyproject.toml
  README.md
```

## Directory guide

| Path | Purpose |
|---|---|
| `configs/` | YAML files controlling model size, data source, batching, seed, device, experiment persistence, and experiment protocols. |
| `data/` | Local raw/downloaded datasets for smoke tests, inference checks, and initialization analysis.. See [`data/README.md`](data/README.md) for dataset-source details, config usage, and data-pipeline commands. |
| `scripts/` | Runnable entry points for sanity checks, inference, initialization analysis, and experiment-tracking checks. See [`scripts/README.md`](scripts/README.md) for the script-by-script guide, options, and expected outputs. |
| `docs/` | The scientific experiment log. See [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md). |
| `notebooks/` | Readable scientific logs that load a persisted experiment record and explain it. Computation lives in the package, not in cells. See [`notebooks/README.md`](notebooks/README.md). |
| `src/` | Reusable Python package code. See [`src/README.md`](src/README.md) for source-module navigation and extension guidance. |
| `tests/` | Lightweight tests covering imports, model shapes, data pipeline, inference, evaluation, gradient norms, experiment persistence, and the persisted-analysis workflow. See [`tests/README.md`](tests/README.md) for detailed test-suite guidance. |


## Git hygiene for datasets and experiments

The repository ignores dataset contents under `data/` while keeping Markdown documentation tracked:

```text
data/**
!data/
!data/**/
!data/**/*.md
```

The `!data/` and `!data/**/` lines re-include directories so Git can still see allowed files inside them, and `!data/**/*.md` is why [`data/README.md`](data/README.md) remains tracked.

Two files under `data/` are tracked on purpose:

| Path | Why it is tracked |
|---|---|
| `data/README.md` | The data-subsystem guide, kept by the Markdown re-include rule. |
| `data/raw/tiny_corpus.txt` | The small offline fixture used by the default configs, the script examples, and the test suite. It is deliberately committed so the repository stays runnable and testable without any download. |

Anything else you place under `data/` is ignored by default, so larger local or downloaded corpora stay out of Git without further configuration. Dataset **configs** remain tracked under `configs/data/`.

Do not bulk-untrack the directory with `git rm -r --cached data`: that would remove both the tiny corpus and this documentation from the repository, and the ignore rules would then prevent the corpus from being added back.

Generated experiment output is handled separately. The `.gitignore` file also ignores `outputs/`, `experiments/`, `runs/`, and `wandb/`, so experiment runs created by the commands below never enter Git.

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
  - Tiny local corpus used to test the data pipeline and produce deterministic tokenizer/data behavior. Tracked deliberately so a fresh clone runs with no downloads.

- `src/llm_behavior_lab/data/config.py`
  - Defines `DatasetConfig`, the dataset identity read from a data config.
  - Defines `ResolutionPolicy`, what one invocation may do to obtain it.
  - Provides `resolve_data_root` for selecting the external data location.

- `src/llm_behavior_lab/data/resolver.py`
  - Defines `resolve_dataset`, the single entry point through which the project obtains a dataset, and `ResolvedDataset`.

- `src/llm_behavior_lab/data/prepared.py`
  - Prepared dataset directories and their manifests, including staleness and completeness checks.

- `src/llm_behavior_lab/data/huggingface.py`
  - Optional Hugging Face acquisition. The only module that can reach the network.

- `src/llm_behavior_lab/data/errors.py`
  - The dataset error hierarchy under `DatasetError`.

- `src/llm_behavior_lab/data/cli.py`
  - The dataset options shared by every data-consuming script.

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

### Experiment persistence files

The `experiment` package implements Phase 6 local persistence. See [`src/README.md`](src/README.md) for the runtime directory, metric, and checkpoint contracts, the full list of public symbols, and extension guidance.

- `src/llm_behavior_lab/experiment/config.py`
  - Validates the experiment, logging, and checkpoint settings loaded from a YAML config.

- `src/llm_behavior_lab/experiment/run.py`
  - Creates a run directory, writes immutable metadata, saves config snapshots, and stores structured analysis JSON.

- `src/llm_behavior_lab/experiment/metrics.py`
  - Appends and reads step-indexed scalar metric records in JSON Lines format.

- `src/llm_behavior_lab/experiment/arrays.py`
  - Saves and reloads array-valued diagnostics as compressed `.npz` artifacts.

- `src/llm_behavior_lab/experiment/checkpoints.py`
  - Saves, discovers, validates, and restores model checkpoints, including the latest-checkpoint pointer.

- `src/llm_behavior_lab/experiment/serialization.py`
  - Shared atomic JSON, YAML, and text writers plus value normalization.

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
| `scripts/analyze_untrained_model.py` | Analyze initialization-time output behavior, optionally gradient stability, and optionally persist a structured run. | `data.*`, `models.registry`, `evaluation.output_stats`, `evaluation.token_frequency`, `evaluation.untrained_analysis`, `evaluation.gradient_norms`, `experiment.run`, `experiment.config` |
| `scripts/check_experiment_tracking.py` | Create a run, log metrics and arrays, save a checkpoint, rediscover it, and restore it into a second model. | `experiment.run`, `experiment.config`, `experiment.metrics`, `experiment.arrays`, `experiment.checkpoints`, `models.registry` |
| `scripts/prepare_dataset.py` | Stage a dataset ahead of time, or report what is missing without obtaining it. | `data.config`, `data.resolver`, `data.cli` |
| `scripts/run_initialization_distribution_experiment.py` | Measure greedy and nucleus token guesses across several random initializations on fixed evaluation positions, optionally with per-position parameter-gradient norms. | `data.*`, `models.registry`, `evaluation.guessing`, `evaluation.init_distribution`, `evaluation.position_gradients`, `analysis.records`, `analysis.aggregation`, `analysis.gradients`, `analysis.figures`, `experiment.run` |
| `scripts/benchmark_position_gradients.py` | Time the exact per-position parameter-gradient measurement at several window counts. Performance only: writes no record. | `data.*`, `models.registry`, `evaluation.init_distribution`, `evaluation.position_gradients` |

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

This standalone path is intentionally lightweight and remains the default. When `--persist-run` is also supplied, the standalone files are not written; the same gradient results are stored inside a structured experiment run instead, as described in [Experiment run outputs](#experiment-run-outputs).

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

Externally hosted datasets need one more optional group. Everything else,
including the whole test suite, works without it:

```bash
python3 -m pip install -e ".[hf]"
```

## Datasets

Datasets are selected through configuration rather than code. `dataset.source`
picks how a dataset is obtained; tokenization, splitting, and batching are
identical either way.

| Source | Meaning |
|---|---|
| `local_text` | A text file already on disk. The default, needing no optional dependency. |
| `huggingface` | Obtained from the Hugging Face Hub and prepared to disk. Needs `pip install -e ".[hf]"`. |

Shipped configs: `configs/data/tiny_text.yaml` (the tracked fixture),
`configs/data/wikitext2.yaml`, and `configs/data/tinystories.yaml`.

### Where dataset contents live

Dataset code, configuration, and the tiny fixture are tracked. Downloaded and
prepared corpora are not. They live under an external data root chosen in this
order:

1. `--data-root` on the command line
2. the `LLM_BEHAVIOR_LAB_DATA_ROOT` environment variable
3. `$XDG_CACHE_HOME/llm-behavior-lab`
4. `~/.cache/llm-behavior-lab`

The default is always outside the repository, so a large download cannot land in
a tracked path. On a cluster, point it at scratch:

```bash
export LLM_BEHAVIOR_LAB_DATA_ROOT=/scratch/$USER/llm-behavior-lab
```

### How a dataset is found

Every workflow obtains its corpus through one resolver, which tries the
configured path, a prepared copy in the repository, a prepared copy under the
data root, an existing source cache, and finally acquisition. If none succeeds,
the error names every location tried and the command that would fix it.

Scripts acquire a missing dataset by default and announce it before any network
activity. These options control that:

| Option | Effect |
|---|---|
| `--data-root PATH` | Where prepared and cached data live. |
| `--no-download` | Reuse what is available; never obtain anything missing. |
| `--offline` | Forbid all network access. Implies `--no-download`. |
| `--force-refresh` | Re-acquire, replacing an existing prepared copy. |

A fresh clone runs immediately on the tracked fixture with no setup. The tiny
corpus never consults the data root or the network.

Prepare a dataset ahead of time, which is useful before a batch job:

```bash
python3 scripts/prepare_dataset.py --data-config configs/data/wikitext2.yaml
```

Or run a workflow directly on an external dataset; it is prepared on first use
and reused afterwards:

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/wikitext2.yaml \
  --model-config configs/model/tiny_llama.yaml
```

The character tokenizer derives its vocabulary from the prepared text, so model
compatibility depends on the corpus, its limits, and its upstream revision —
not on the dataset name. Both shipped external configs have been live-tested
end to end against `configs/model/tiny_llama.yaml` at their current limits
(WikiText-2 gave a tokenizer vocabulary of 147, TinyStories 70, both under
`vocab_size: 256`). Those are observations, not guarantees: raising a limit or
changing the revision may change them. The runtime check remains the authority —
if a run reports that the tokenizer vocab size exceeds the model vocab size,
raise `model.params.vocab_size`.

See [`data/README.md`](data/README.md) for the resolution order, prepared-data
manifests, staleness detection, and the limitations that come with them.

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

## Run the Phase 6 persisted initialization analysis

The Phase 5 analysis can record a complete experiment run instead of printing only. Add `--persist-run` and select an experiment config:

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml \
  --experiment-config configs/experiment/untrained_baseline.yaml \
  --persist-run
```

Gradient diagnostics can be persisted in the same run:

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml \
  --experiment-config configs/experiment/untrained_baseline.yaml \
  --persist-run \
  --compute-grad-norms
```

The script prints the full Phase 5 report first, then the run directory, the metadata path, the evaluation-metrics path, and the path of the initialized checkpoint.

Additional persistence options:

| Option | Meaning |
|---|---|
| `--experiment-config` | Experiment persistence config. Default: `configs/experiment/untrained_baseline.yaml`. |
| `--run-id` | Explicit run identifier. An existing run is never overwritten; a collision raises an error. |
| `--output-dir` | Output root override. Default comes from `experiment.output_dir` in the experiment config. |
| `--notes` | Free-text note stored in the run metadata. |

To keep a disposable run out of the default `outputs/` tree, pass an explicit output root:

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml \
  --experiment-config configs/experiment/untrained_baseline.yaml \
  --persist-run \
  --output-dir /tmp/llm_behavior_lab_runs
```

See [`scripts/README.md`](scripts/README.md) for the complete option reference and common failure modes.

## Run the Phase 6 experiment-tracking smoke test

This script exercises the persistence layer end to end without any analysis: it creates a run, snapshots configs, logs a scalar record and an array artifact, saves a checkpoint, rediscovers it, restores it into a second freshly built model, and verifies that every restored parameter matches.

```bash
python3 scripts/check_experiment_tracking.py \
  --model-config configs/model/tiny_llama.yaml \
  --data-config configs/data/tiny_text.yaml \
  --experiment-config configs/experiment/phase6_smoke.yaml
```

The script prints the run directory, the config snapshot paths, the metric-file paths, the array artifact, the saved and rediscovered checkpoint paths, the restored global step, and a confirmation that the checkpoint parameter round trip succeeded. It raises an error instead of printing that confirmation if any check fails.

Use `--output-dir` and `--run-id` to place a disposable run outside the default output root:

```bash
python3 scripts/check_experiment_tracking.py \
  --output-dir /tmp/llm_behavior_lab_runs \
  --run-id smoke_run
```

## Run the initialization-distribution experiment

Measures how a randomly initialized model's **selected token guesses** compare with the
corpus token distribution, and how much that comparison moves when only the initialization
changes.

> The model is untrained. Nothing this experiment reports is a statement about model
> quality.

Smoke check on the tracked fixture — fast, fully offline, and **not** scientifically
meaningful (595 characters, 39 tokens):

```bash
python3 scripts/run_initialization_distribution_experiment.py \
  --data-config configs/data/tiny_text.yaml \
  --num-initializations 3 --num-windows 16 --block-size 16 --num-replicates 2
```

First meaningful experiment, once WikiText-2 has been prepared:

```bash
python3 scripts/run_initialization_distribution_experiment.py \
  --data-config configs/data/wikitext2.yaml --offline
```

Figures need the optional extra:

```bash
python3 -m pip install -e ".[analysis]"
```

### What is held fixed

Everything except the model-initialization seed: corpus, tokenizer, vocabulary, split, and
the evaluation positions themselves. Positions are chosen deterministically *before* any
model exists, so no difference between initializations can come from looking at different
text. The protocol lives in `configs/experiment/initialization_distribution.yaml` and every
setting is recorded with the results.

### Two guessing policies, one forward pass

| Policy | Rule | Randomness |
|---|---|---|
| greedy | `argmax(logits)` | none beyond the initialization |
| nucleus | temperature `0.6`, top-p `0.9`, following the reference LLaMA rule | its own seed, replicated and averaged within an initialization |

Both read the **same** logits, so they describe one model state rather than two draws.

### Tokenizers

| | Character | Pretrained subword |
|---|---|---|
| vocabulary | derived from the corpus (39–147 observed) | Hugging Face, 32,000 for Mistral |
| dependency | none | `transformers`, optional `[tokenizers]` extra |
| artifacts | none | cached in `<data_root>/tokenizers/`, outside the repository |

Both sit behind one small interface, selected by `tokenizer.type` in the data config.
Character is the default, so existing configs are unaffected.

Tokenizer loading tries the local cache first, so a cached tokenizer works fully offline;
acquisition requires explicit permission and is announced before any network use. **Only
tokenizer files are ever fetched** — no model weights, enforced by a test.

### Support nomenclature

Four quantities that a 32k vocabulary makes genuinely different:

| Quantity | Meaning |
|---|---|
| `V` full | tokens the tokenizer defines |
| `V` eligible | tokens a model may be scored on, after excluding structural IDs |
| `V` corpus-observed | tokens that actually occur in the analysis split |
| effective support `exp(H)` | entropy-equivalent number of equally likely active tokens |

Only the first three are counts. Structural tokens are excluded from the predictive
support, never renumbered.

### Two empirical references

| Distribution | Role |
|---|---|
| whole analysis split | the primary reference the guesses are compared against |
| selected positions only | a **sampling-adequacy diagnostic** — are the analyzed positions representative of the split at all? |

### Figures

| File | Shows |
|---|---|
| `figure0_sampling_adequacy` | ranked split frequencies vs. ranked selected-position targets, with TV and JS |
| `figure1_ranked_frequency_profiles` | ranked concentration of corpus vs. both policies, with initialization SEM, plus the uniform-null reference |
| `figure2_token_wise_mismatch` | same-token `\|q - p\|` ranked after differencing, typical vs. persistent |
| `figure3_token_identity_scatter` | corpus fraction vs. mean guess fraction, per token, with the identity line |
| `figure4_input_structure_profiles` | ranked guess concentration under real, shuffled, and Gaussian input, one panel per policy |
| `figure5_temperature_ranked_profiles` | ranked guess profiles from the greedy anchor through each temperature to the null |
| `figure6_temperature_ranked_distances` | ranked-profile distance to greedy and to the null, against temperature |
| `figure7_temperature_support_and_greedy_agreement` | effective support relative to the null per input condition, plus agreement with greedy |

### Null comparisons

Two reference families sit alongside the model measurements:

- a **uniform categorical output null** — the same `D` draws over the same eligible
  support `K`, simulated as a joint multinomial, answering how much ranked structure
  differs from what finite sampling alone produces. Its envelope is a **Monte Carlo
  interval**, never a SEM across initializations;
- an **input-structure comparison** — one initialized model evaluated on real corpus
  windows, the same token multiset with its ordering destroyed, and Gaussian vectors at the
  embedding boundary.

The current protocol — selected explicitly in
`configs/experiment/initialization_distribution.yaml`, not inherited from a library
default — takes **one nucleus draw per initialization and position** (`R = 1`), so greedy
and nucleus summarise the same `D` assignments. Within-initialization stochastic
variance is therefore not estimable and is reported as such rather than as zero.

An optional **multi-temperature nucleus sweep** samples the *same* logits at several
temperatures, mapping the crossover from the greedy regime toward the uniform null. Greedy
is the `T = 0` anchor and is always computed by `argmax`, never by a zero-temperature
softmax, so only positive temperatures are swept. `top_p` stays fixed, which means the
sweep measures the transition **under a fixed nucleus threshold** rather than pure softmax
temperature — raising `T` flattens the profile and so widens the 0.9 nucleus itself.

```bash
python3 scripts/run_initialization_distribution_experiment.py \
  --data-config configs/data/wikitext2_subword.yaml \
  --model-config configs/model/tiny_llama_32k.yaml --offline \
  --temperatures 0.12 0.24 0.36 0.48 0.60 1.20
```

The sweep is additional analysis: figures 0–4 continue to use the canonical
`sampling.temperature`, and a config without a `temperature_sweep` block runs exactly as
before.

An optional **per-position gradient analysis** measures, for one initialization, the exact
L2 norm of the gradient of each position's own next-token cross-entropy with respect to
**every trainable parameter**, and aggregates it per token as `G_i`. Alongside it the
record keeps the greedy prediction at each of those same positions, so the guess fractions
it is compared against describe the same positions as the norms.

```bash
python3 scripts/run_initialization_distribution_experiment.py \
  --data-config configs/data/wikitext2_subword.yaml \
  --model-config configs/model/tiny_llama_32k.yaml --offline \
  --gradient-analysis
```

It is **off by default**: it costs one backward pass per evaluation position, where
everything else costs one forward pass per window. `--gradient-windows N` restricts it to a
deterministic evenly spaced subset of windows, which reduces the scientific position set
and should be chosen deliberately. Measure the cost before committing to a full run:

```bash
python3 scripts/benchmark_position_gradients.py \
  --data-config configs/data/wikitext2_subword.yaml \
  --model-config configs/model/tiny_llama_32k.yaml --offline \
  --window-counts 1 2 4 8
```

The benchmark writes no record and draws no figure; it times several window counts so the
scaling can be checked, and prints a projected full-run cost explicitly labelled as an
extrapolation.

The same probabilities are also evaluated at a grid of diagnostic temperatures,
`T = 0.12 … 1.20` with `T = 1` as the canonical reference, giving
`figure11_temperature_ranked_predictive_probabilities.svg`,
`figure12_temperature_max_predictive_probability.svg` and
`figure13_greedy_confidence_vs_temperature.svg`. Nothing is sampled and nothing is
truncated there: softmax is strictly increasing, so `argmax softmax(z/T) = argmax z`
and every temperature describes the **same greedy decisions** with different
confidence attached. That is what makes it a different experiment from the nucleus
sweep, which samples from the transformed distribution and therefore does change what
gets selected.

`figure14_ranked_mean_token_probabilities.svg` complements figure 11 by reversing the
order of two operations: it averages probabilities **at fixed token identity** across
positions and ranks afterwards, where figure 11 ranks within each position first. Figure
11 asks how concentrated a typical single prediction is; figure 14 asks whether the *same*
tokens are persistently favoured. Steep in 11 with a flat 14 means concentrated
predictions on input-dependent tokens; steep in both is a persistent identity bias.

### Initialization scale

`scripts/run_initialization_scale_experiment.py` repeats the whole pipeline at three
initialization scales, `alpha in {1.0, 0.5, 0.25}`, writing one self-contained run per
scale under `scale_1/`, `scale_0p5/` and `scale_0p25/` plus a parent manifest:

```bash
python3 scripts/run_initialization_scale_experiment.py \
  --data-config configs/data/wikitext2_subword.yaml \
  --model-config configs/model/tiny_llama_32k.yaml --offline --gradient-analysis
```

`alpha` multiplies every audited zero-centred random weight; the deterministic RMSNorm
gains are left alone and the architecture has no bias parameters. All three conditions
share one draw, so signs and directions are identical and only magnitude differs, and
`alpha = 1` is a literal no-op. **There is no single architecture-wide `sigma_w`** — the
embedding is `normal_(0,1)` while every linear is `kaiming_uniform_(a=sqrt(5))` with a
fan-in dependent scale — so standard deviations are reported per parameter group.

This is not the per-layer diagnostic in `evaluation/gradient_norms.py`, which
differentiates the *window-averaged* loss with respect to block activations. See
[`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) §5e for the exact definition.

A run carrying gradient data also gets `figure10_temperature_gradient_vs_initial_guess_bias.svg`,
six panels relating the gradient to the guessing bias across loss temperatures. There
temperature is inside the loss, `ell_T = -log softmax(z/T)[y]`, so the gradient really
changes — while `q_i`, `n_i` and `p_i` are identical in every panel because
`argmax softmax(z/T) = argmax z`. The single-temperature version remains as
`supplementary_t1_gradient_vs_initial_guess_bias.svg`:
one marker per token with `n_i > 0`, at `x = G_i` and `y = q_i`, coloured by corpus
frequency. Runs without gradient data produce exactly the figures they did before.

Figures and statistics can be regenerated from a finished record alone — no model,
no GPU, and nothing recomputed:

Every run also records the **raw predictive distribution** the model produces before
any sampling policy touches it: logits restricted to the eligible support, softmax at
`T = 1`, no top-p truncation, no greedy or nucleus decision. Two figures come from it.

`figure8_ranked_predictive_probabilities.svg` ranks the probabilities **within each
position** and only then averages at equal rank, against a uniform `1/K` reference. This
is not figure 1, which ranks guess frequencies accumulated *across* positions: a model
can be flat at every individual position and still produce a peaked aggregate.

`figure9_max_predictive_probability_distribution.svg` shows the distribution of the
probability carried by the greedy-selected token, as an ECDF with one curve per
initialization over the pooled curve. Greedy always takes the top-ranked token; whether
that token carries much mass is a separate question, and this is the figure that answers
it.

Only sufficient statistics are stored — a ranked `[I, K]` profile and three `[I, D]`
per-position vectors, about 12 MB — never the full `[I, D, K]` probability tensor, which
would be roughly 50 GiB.

```bash
python3 scripts/render_record_figures.py outputs/<run>/analyses --only figure8
python3 scripts/render_record_figures.py outputs/<run>/analyses --only figure9
python3 scripts/render_record_figures.py outputs/<run>/analyses --only figure10
python3 scripts/render_record_figures.py outputs/<run>/analyses --stats-only
```

The second form prints the distributions of `G_i`, `q_i`, `p_i` and `n_i` and the
three Spearman correlations without drawing anything. Neither form imports PyTorch,
which a test asserts.

See [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) for the definitions and what each
contrast can and cannot identify.

One **SVG** per figure — no companion PNG. Dense curves are rasterized inside the SVG, so
a 32k-token figure stays small while text and axes remain vector.

`outputs/` is ignored by Git, so results never enter the repository.

### Run directories

Run directories carry their own identity, so a listing is readable without opening
`metadata.json`:

```text
<timestamp>__<dataset>__<tokenizer>__<model>__N<positions>-I<inits>-R<replicates>__<code>
```

```text
20260812-140112__tiny-local-text__char-39__llama-tiny-256__N256-I3-R2__ef0cd1b9
20260812-140112__wikitext2-raw-train1k__mistral-7b-v0.1-32k__llama-tiny-32k__N8192-I4-R4__9bb2a015
```

Every component is derived from resolved runtime metadata. Only the axes worth scanning
for are included; temperature, top-p, seeds, block size, and revisions stay in
`metadata.json` and the config snapshots. The trailing code keeps two runs in the same
second distinct.

Each run holds:

```text
<run>/
  analyses/   complete per-token record (.npz) + scalar summary (.json)
  figures/    four SVGs
  config/     verbatim model/data/experiment snapshots
  metrics/    JSONL scalar metrics
  metadata.json
```

### Realistic subword tokenizer

The character baseline uses a vocabulary of 39–147 tokens. To ask the same
question at a realistic scale, the experiment can read the corpus through the
pretrained `mistralai/Mistral-7B-v0.1` tokenizer (~32k tokens) while the model
stays randomly initialized:

```bash
python3 -m pip install -e ".[tokenizers]"

python3 scripts/run_initialization_distribution_experiment.py \
  --data-config configs/data/wikitext2_subword.yaml \
  --model-config configs/model/tiny_llama_32k.yaml \
  --forward-batch-size 4
```

**Only the tokenizer is pretrained.** No model weights are downloaded, and a
test asserts the tokenizer backend never references a pretrained model loader.
Read the results accordingly: the corpus token distribution reflects the
tokenizer's own training, and only the *guess* distribution reflects the random
model.

Tokenizer files are cached under `<data_root>/tokenizers/`, outside the
repository. Once cached, `--offline` works with no network.

| Setting | Character | Subword |
|---|---|---|
| data config | `configs/data/wikitext2.yaml` | `configs/data/wikitext2_subword.yaml` |
| model config | `configs/model/tiny_llama.yaml` | `configs/model/tiny_llama_32k.yaml` |
| vocabulary | 147 observed | 32000 canonical |
| `--forward-batch-size` | 32 | 4–8 |

Sampling adequacy must be re-established after the switch: a subword tokenizer
changes the token count, the support, and the frequency structure at once, so a
position count that was adequate for characters says nothing about the new
regime. Read figure 0 first.

Read the results with `notebooks/initialization_distribution.ipynb`, and see
[`src/README.md`](src/README.md) for the precise definition of every distribution and
measure.

---

## Experiment run outputs

A persisted run is a self-contained directory under `<output_dir>/<experiment-name>/<run-id>/`:

```text
outputs/
  untrained_baseline/
    <timestamp>_<suffix>/
      metadata.json
      config/
        model_config.yaml
        data_config.yaml
        experiment_config.yaml
      metrics/
        training_metrics.jsonl
        evaluation_metrics.jsonl
        array_metrics/
          *.npz
      checkpoints/
        checkpoint_step_000000.pt
        latest.json
      analyses/
        untrained_analysis.json
        gradient_norm_analysis.json
      logs/
```

What each part holds:

| Path | Contents |
|---|---|
| `metadata.json` | Immutable run record: experiment name, run ID, creation time, phase, seed, tags, Git commit when available, and environment information. |
| `config/` | YAML snapshots of the model, data, and experiment configs actually used by the run. |
| `metrics/*.jsonl` | Append-only scalar metric records, one JSON object per line, each carrying step, stage, split, checkpoint reference, and artifact references. |
| `metrics/array_metrics/` | Compressed NumPy archives for vector-valued diagnostics such as token distributions and per-layer gradient norms. |
| `checkpoints/` | Model checkpoints plus `latest.json`, a small pointer used for latest-checkpoint discovery. |
| `analyses/` | Structured JSON summaries of the analysis results. |
| `logs/` | Reserved for run logs. |

Run identifiers are generated from a UTC timestamp plus a random suffix, so they sort chronologically and do not collide. Run directories are created exclusively: an existing run is never silently resumed or overwritten. Checkpoints are loaded with CPU device mapping by default, so a run saved on one machine can be inspected on another.

Output roots such as `outputs/` are ignored by Git. For the metric schema, checkpoint payload fields, and the complete persistence API, see [`src/README.md`](src/README.md).

## Run tests

```bash
python3 -m pytest
```

Successful output should show all tests passing.

The Phase 6 persistence layer is covered by two test files:

| File | Scope |
|---|---|
| `tests/test_experiment_tracking.py` | Component-level tests for run creation, collision safety, JSONL metrics, array round trips, and checkpoint save, discovery, and restoration. |
| `tests/test_persisted_analysis_run.py` | An end-to-end test that drives `scripts/analyze_untrained_model.py --persist-run` and checks the resulting run directory, metadata, config snapshots, metric records, array artifacts, and checkpoint restoration. |

See [`tests/README.md`](tests/README.md) for the full test-suite guide, categories, and guidance on adding tests.

## Current limitations

The repository still does not include:

- full training loops
- optimizer or scheduler setup
- learning-rate scheduling
- gradient tracking over training checkpoints
- full validation-set evaluation
- fine-tuning
- multi-model comparison
- advanced bias or group-based evaluation metrics
- persistent tokenized dataset caches
- streaming or sharded loading for large-scale training

Persistence is in place but not yet exercised by training. Checkpoints currently capture an initialized model at step zero; the optimizer and scheduler fields in the checkpoint format exist but stay empty until a training loop fills them.

The gradient diagnostic is still an initialization-time check only. Phase 7 will add the training loop that reuses this persistence layer, and later phases will repeat the diagnostic across training checkpoints.

## Next phases

Planned next steps:

1. **Phase 7 — Pre-training loop**
   - optimizer
   - learning-rate schedule
   - loss logging
   - validation checks
   - periodic checkpoint saving and training resumption
   - reuse of the Phase 6 persistence interfaces

2. **Phase 8 — Training-dynamics analysis**
   - loss/perplexity curves
   - output-distribution changes
   - gradient diagnostics over checkpoints
   - bias metrics over time

3. **Phase 9 — Fine-tuning pipeline**
   - supervised fine-tuning
   - checkpoint-based evaluation
   - comparison between pre-training and fine-tuning behavior

4. **Phase 10 — Model extension phase**
   - additional architectures
   - Gemma-oriented variants
   - cross-model behavior comparison
