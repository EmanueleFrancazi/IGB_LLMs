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

Quick subfolder overview:

| Path | Role |
|---|---|
| `src/llm_behavior_lab/models/` | Model interface, registry, and explicit LLaMA-style implementation. |
| `src/llm_behavior_lab/data/` | Text loading, tokenization, splitting, and causal LM batching utilities. |
| `src/llm_behavior_lab/inference/` | Prompt preparation, logits extraction, probability extraction, decoding, and generation. |
| `src/llm_behavior_lab/evaluation/` | Output-distribution, token-frequency, untrained-analysis, and gradient-norm diagnostics. |
| `src/llm_behavior_lab/utils/` | Shared helpers for seeds, device selection, parameter counting, and config loading. |

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

  analysis/
    __init__.py
    aggregation.py
    figures.py
    records.py

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
    output_stats.py
    token_frequency.py
    untrained_analysis.py

  experiment/
    __init__.py
    arrays.py
    checkpoints.py
    config.py
    metrics.py
    run.py
    serialization.py

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
    config.py
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

The `data` package selects, locates, loads, and batches text datasets.

Current files:

```text
src/llm_behavior_lab/data/
  __init__.py
  config.py        dataset identity and runtime resolution policy
  errors.py        dataset error hierarchy
  resolver.py      the single entry point that locates a dataset
  prepared.py      prepared directories and their manifests
  huggingface.py   acquisition; the only module that can reach the network
  cli.py           the dataset options shared by scripts
  text_dataset.py  local text loading and train/validation splitting
  tokenizer.py     deterministic character-level tokenizer
  dataloader.py    causal language-modeling batches
```

### Main responsibilities

- describe which dataset an experiment uses, separately from what one
  invocation is allowed to do to obtain it
- resolve a dataset local-first, acquiring only when explicitly permitted
- keep dataset contents outside the repository
- detect stale or incomplete prepared data
- load local UTF-8 text files and split token IDs
- build a deterministic character-level tokenizer
- create causal language-modeling batches

### Important objects and functions

```text
DatasetConfig
ResolutionPolicy
ResolvedDataset
resolve_dataset
resolve_data_root
resolve_repo_path
add_dataset_arguments
resolution_policy_from_args
write_prepared
load_prepared
CharTokenizer
TokenSplits
load_text_file
split_token_ids
CausalLMBatch
CausalLMBatcher
```

### Dataset resolution contract

`resolve_dataset(dataset_config, policy, repo_root=...)` is the only way the
project obtains a dataset. It tries, in order: the configured path, a prepared
copy in the repository, a prepared copy under the data root, a copy already in
the source cache, and finally acquisition when the policy permits it. A
`local_text` dataset resolves at the first step and never consults the data root
or the network.

Identity and policy stay separate. `DatasetConfig` comes from tracked YAML and
is part of reproducibility; `ResolutionPolicy` comes from flags and the
environment and is machine-specific, which is why the data root is not a config
field.

`ResolutionPolicy()` built without arguments cannot reach the network:
`allow_download` defaults to false, and `offline` forbids the network outright
regardless of it. Scripts opt in explicitly.

### Prepared-data contract

A prepared dataset directory holds `text.txt` and `manifest.json`. The manifest
is written last inside a staging directory moved into place atomically, so its
presence means the preparation completed. It records the requested identity, a
digest, counts, and a `resolved` block describing what the source returned. It
contains no filesystem paths.

The guarantee is that an incomplete preparation is never mistaken for a complete
one. Replacement is not transactional: an existing copy is removed before the
new one is renamed in, so a crash in that window leaves neither. Concurrent
preparation of one dataset is unsupported and fails with a filesystem error
rather than corrupting anything.

### Offline contract

Cache reads run inside a context manager that sets the offline environment
variables and the matching library constants, because `huggingface_hub` and
`datasets` read those variables once at import and the loader is imported
before the call.

Only the flags the installed release actually defines are touched, so the same
guarantee holds across the supported `datasets` range: older releases use
`HF_DATASETS_OFFLINE`, newer ones also carry the Hub-style name, and nothing is
created on a version that lacks it. Every original value is restored on exit
and on exceptions.

Those flags are process-global, so loader calls are serialized with a
module-level lock: a cache-only call cannot force a concurrent acquisition
offline, and overlapping calls cannot leave the process permanently offline.
Preparation across separate processes remains unsupported.

Enforcement is still delegated to those libraries; this package adds no
independent network barrier.

### Where to look

| Task | File |
|---|---|
| Add or change a dataset configuration field | `data/config.py` |
| Change where prepared and cached data live | `data/config.py` |
| Change the order locations are tried in | `data/resolver.py` |
| Change prepared-data layout or staleness rules | `data/prepared.py` |
| Change acquisition or add a new external source | `data/huggingface.py` |
| Change the shared command-line options | `data/cli.py` |
| Change local text loading or splitting | `data/text_dataset.py` |
| Modify the character tokenizer | `data/tokenizer.py` |
| Change batch sampling or input/target shifting | `data/dataloader.py` |
| Export new data utilities | `data/__init__.py` |

### Tokenizers

One interface, two implementations, no registry:

| Implementation | Vocabulary | External dependency |
|---|---|---|
| `CharTokenizer` | derived from the corpus, tens of tokens | none |
| `PretrainedTokenizer` | a Hugging Face subword vocabulary, ~32k | `transformers`, optional `[tokenizers]` extra |

`Tokenizer` is a `Protocol`, so `CharTokenizer` stays the plain frozen dataclass
it always was. Beyond encode/decode it requires what a large vocabulary forces
the analysis to know: `special_token_ids`, `eligible_token_ids`, `token_repr`,
and `describe()`.

`build_tokenizer` dispatches on the data config's `tokenizer.type`. Character is
the default, so every existing config keeps working untouched.

#### Tokenizer artifacts and offline use

`pretrained_tokenizer.py` loads **tokenizer files only**. It calls
`AutoTokenizer.from_pretrained` and nothing else — no `AutoModel`, no
`AutoModelForCausalLM`, no pipeline. A test asserts this against the parsed
source, because the interpretation of every result depends on the model staying
randomly initialized.

Loading tries the local cache first, so a cached tokenizer works with no network
at all. Acquisition happens only when explicitly permitted and is announced
first. Artifacts live in `<data_root>/tokenizers/`, outside the repository,
alongside the dataset caches.

#### Special tokens and the predictive support

A pretrained vocabulary contains structural IDs — BOS, EOS, padding — that are
not ordinary corpus tokens. The policy is:

1. keep the canonical vocabulary and canonical IDs;
2. encode corpus text with **no** automatic BOS/EOS;
3. identify structural IDs explicitly;
4. exclude them from the predictive support used for the empirical
   distributions, greedy selection, and nucleus sampling alike;
5. never renumber what remains.

Exclusion is implemented as a `-inf` mask applied once per batch. Softmax then
gives those tokens exactly zero, argmax cannot choose them, and nucleus
truncation sorts them to the tail — one masking step covers all three consumers,
which is what keeps them on the same support.

### Adding a dataset source

Sources are dispatched explicitly rather than through a registry, because there
are two of them. Add a module beside `huggingface.py`, add a branch in
`resolver.py`, and add the source name to `SUPPORTED_SOURCES`. Introduce a
registry only if a third source shows that the branch has become the problem.

### Scripts using this package

- `scripts/prepare_dataset.py`
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

## `llm_behavior_lab.experiment`

The `experiment` package implements Phase 6 local persistence. It is intentionally independent of training loops so initialization analysis, future pre-training, and future fine-tuning can share one run format.

Current files:

```text
src/llm_behavior_lab/experiment/
  __init__.py
  arrays.py
  checkpoints.py
  config.py
  metrics.py
  run.py
  serialization.py
```

### Main responsibilities

- validate experiment/logging/checkpoint YAML settings
- create collision-safe experiment runs
- write immutable metadata and config snapshots
- append scalar JSONL records
- save compressed array artifacts
- save/load/discover local PyTorch checkpoints
- capture and optionally restore RNG state
- expose predictable paths to analyses and logs

### Important objects and functions

```text
ExperimentSettings
LoggingSettings
CheckpointSettings
experiment_settings_from_config
ExperimentRun
RunPaths
MetricLogger
ArrayMetricStore
ArrayArtifact
CheckpointManager
CheckpointInfo
generate_run_id
collect_environment_info
```

### File-level responsibilities

| File | Responsibility |
|---|---|
| `experiment/config.py` | Validated settings and future logging/checkpoint cadence. |
| `experiment/run.py` | Run creation, metadata, config snapshots, analysis JSON. |
| `experiment/metrics.py` | Append-only scalar JSONL records. |
| `experiment/arrays.py` | Compressed `.npz` array artifacts and reload. |
| `experiment/checkpoints.py` | Atomic checkpoint save, latest discovery, load/restore. |
| `experiment/serialization.py` | Atomic JSON/YAML/text writes and value normalization. |
| `experiment/__init__.py` | Public Phase 6 API. |

### Runtime directory contract

```text
outputs/<experiment-name>/<run-id>/
  metadata.json
  config/
  metrics/
    training_metrics.jsonl
    evaluation_metrics.jsonl
    array_metrics/
  checkpoints/
    checkpoint_step_000000.pt
    latest.json
  analyses/
  logs/
```

`ExperimentRun.create` uses exclusive directory creation. Existing runs are not silently resumed or overwritten.

### Metric contract

`MetricLogger.log(...)` requires:

- non-negative `step`
- non-empty `stage`
- a non-empty mapping of scalar metrics

Array/list/dictionary values are rejected from the scalar metric dictionary. Save them through `ArrayMetricStore` and place the returned relative path in `artifacts`.

### Checkpoint contract

A checkpoint contains:

```text
format_version
created_at
global_step
model_state_dict
model_config
optimizer_state_dict
scheduler_state_dict
rng_state
additional_state
```

Only `format_version`, `global_step`, and `model_state_dict` are required for validation. Optimizer/scheduler fields are optional so Phase 6 can save an initialized model while Phase 7 can save resumable training state.

`CheckpointManager.load_payload(...)` and `CheckpointManager.restore(...)` accept an explicit `map_location` and default to `"cpu"`, so a checkpoint saved on an accelerator can be reloaded on a CPU-only machine without changing the call site. `tests/test_experiment_tracking.py::test_checkpoint_load_uses_map_location_cpu` verifies that a payload loaded this way places every tensor on CPU.

### Scripts using this package

- `scripts/check_experiment_tracking.py`
- `scripts/analyze_untrained_model.py` when `--persist-run` is enabled

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
  guessing.py
  init_distribution.py
  output_stats.py
  token_frequency.py
  untrained_analysis.py
```

### Guessing policies and the initialization experiment

`guessing.py` turns logits into the token a policy actually **selects**: `greedy_guess_ids`
takes the argmax, and `nucleus_guess_ids` applies temperature scaling followed by top-p
truncation, following the reference LLaMA rule (a token is dropped when the cumulative mass
*strictly before* it exceeds `top_p`, so the most probable token always survives).

`init_distribution.py` runs that comparison across initializations. It chooses evaluation
positions deterministically with `deterministic_window_starts`, so the same positions are
reused for every seed and no observed difference can come from looking at different text;
computes the logits once per initialization; and derives both policies from those same
logits, replicating the stochastic one so sampling noise can be averaged out within an
initialization.

Two quantities that are easy to confuse are kept under separate names everywhere:

| Quantity | Meaning |
|---|---|
| `mean_predicted_probabilities` | mean predictive mass per token, averaged over positions |
| guess counts / fractions | how often a policy actually **selected** each token |

They differ sharply at initialization and are never interchangeable.

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

## `llm_behavior_lab.analysis`

The `analysis` package is the **read side** of the project: experiments write records, and
everything here consumes them.

Current files:

```text
src/llm_behavior_lab/analysis/
  __init__.py
  aggregation.py
  figures.py
  records.py
```

### NumPy-only by design

`records` and `aggregation` depend on NumPy alone, so a finished experiment can be
re-analyzed without PyTorch and without a GPU. Analysis is revisited far more often than it
is produced, and the people revisiting it should not need the training stack installed.

`figures` additionally needs `matplotlib` and is therefore **not** imported by
`analysis/__init__.py`. Install it with `python3 -m pip install -e ".[analysis]"` and import
the module explicitly.

### Separation of concerns

| Module | Owns | Never does |
|---|---|---|
| `records.py` | the persisted format and its validation | compute statistics |
| `aggregation.py` | every statistic | read or write files |
| `figures.py` | drawing and file naming | compute a statistic of its own |

A figure can only draw what `aggregation` already returns, so a plot can never disagree
with the numbers printed beside it. Add a new measure to `aggregation` with a test, then
use it from a figure.

### The initialization-distribution record

One experiment writes one directory holding `initialization_distribution.npz` (all
per-token vectors) and `initialization_distribution.json` (the protocol). With `S`
initializations, `R` sampling replicates, and `V` valid tokens:

| Array | Shape | Meaning |
|---|---|---|
| `token_ids` | `[V]` | canonical alignment key; position in every array **is** the token ID |
| `corpus_counts` | `[V]` | token counts over the whole analysis split |
| `selected_target_counts` | `[V]` | next-token targets at the analyzed positions only |
| `greedy_counts` | `[S, V]` | argmax guesses per initialization |
| `nucleus_counts` | `[S, R, V]` | sampled guesses, per replicate |
| `mean_predicted_probabilities` | `[S, V]` | mean predictive mass — *not* guesses |
| `model_seeds` | `[S]` | initialization seed for each row |

Complete vectors are stored rather than the top-N summaries the console analysis prints,
because a truncated summary cannot answer a question nobody asked yet.

### Distributions the experiment defines

| Symbol | Name | Definition |
|---|---|---|
| `p` | whole-split empirical | token counts over the **entire analysis split**, normalized. Validation tokens are excluded when the split is `train`. |
| `p_selected` | selected-position empirical | next-token targets at exactly the analyzed positions, normalized. A **sampling-adequacy diagnostic**, not a model reference. |
| `q_greedy[s]` | greedy guess fractions | how often each token was the argmax, for initialization `s` |
| `q_nucleus[s]` | nucleus guess fractions | as above for the stochastic policy, averaged over replicates within `s` |

### Measures

#### Four vocabulary sizes, four questions

| Quantity | Question |
|---|---|
| `V` full | how many tokens the tokenizer defines |
| `V` eligible | how many a model may be scored on, after removing structural IDs |
| `V` corpus-observed | how many actually occur in the analysis split |
| `N_eff = exp(H)` | how broadly the mass is spread |

Only the first three are counts. Effective support is the size of a uniform
distribution with the same entropy, so it sits far below the observed support
whenever a few tokens dominate — which at initialization they do.

#### Zero frequency and effective support answer different questions

Both are retained; neither replaces the other.

*Zero-frequency count/fraction* asks **how much of the eligible vocabulary was
never selected in exactly `N` guesses**. It depends strongly on `N`: with fewer
draws than vocabulary entries, most tokens are unreachable regardless of the
model.

*Effective support* asks **how broadly the guess mass is distributed**. Two
policies can touch the same number of distinct tokens while one concentrates
almost all its mass on a handful; entropy separates them, a count cannot.

#### The zero-frequency comparison must use equal draw counts

Greedy makes exactly one selection per position, so it has `N` draws. A single
nucleus replicate also has `N`, but the `R` replicates *pooled* have `R*N`, and
more draws mean more chances to reach a rare token.

The headline nucleus figure is therefore **per replicate**: each replicate is
scored on its own `N` draws and the counts are averaged within the
initialization, then compared across initializations. Both policies are measured
at the same sample size.

```text
greedy   Z_s      = #{i in V_eligible : greedy count = 0}          over N draws
nucleus  Z_s      = (1/R) * sum_r #{i : replicate r count = 0}     over N draws each
pooled   (diagnostic only)                                         over R*N draws
```

The pooled figure is kept under the separate name
`pooled_zero_frequency_count_mean`. It answers a real question — what the policy
can reach given more attempts — but it is not comparable with greedy and is
never displayed beside it.

| Measure | Definition | Reads as |
|---|---|---|
| ranked profile | a distribution sorted descending | concentration, token identity discarded |
| token-wise gap | `\|q[s,i] - p[i]\|`, differenced **before** ranking | mismatch, token identity preserved |
| persistent gap | `\|E_s[q[s,i]] - p[i]\|` | mismatch that survives averaging over initializations |
| total variation | `0.5 * sum_i \|p_i - q_i\|`, in `[0, 1]` | overall disagreement |
| Jensen–Shannon | bounded by `ln 2` | overall disagreement, divergence-flavoured |
| entropy / effective support | `H(p)` in nats, `exp(H)` in tokens | how many tokens are really in play |
| zero-frequency count | eligible tokens never selected in `N` draws | breadth, **strongly** dependent on `N` |
| zero-frequency fraction | that count over the eligible support | the same, comparable across vocabulary sizes |
| corpus-observed zero-guess | corpus tokens never guessed | a secondary diagnostic, narrower than the above |
| top-1/top-2 gap | `p_(1) - p_(2)` | how far the leading token leads |

Ranking before differencing would answer a much weaker question — a distribution with the
corpus frequencies assigned to the *wrong* tokens has an identical ranked profile — so the
two orderings live in separate functions rather than behind a flag.

### Which variability is which

Three kinds are deliberately not mixed:

1. **Selected-position representativeness** — whether the analyzed positions stand in for
   the split. Reported by `sampling_adequacy` and figure 0. More initializations cannot
   repair a bad position set.
2. **Model-initialization variability** — the SEM shown on every band, computed across
   independent initializations with `s / sqrt(N)`.
3. **Stochastic-sampling variability** — replicate-to-replicate noise inside one
   initialization. Averaged out before initializations are compared, and reported on its own
   by `within_initialization_sampling_spread`.

The fixed whole-split empirical distribution never carries an initialization SEM: it is one
distribution, not a sample.

### Memory: nothing is indexed by evaluation position

`measure_initialization` runs the model batch by batch, folds each transient
`[batch, block, vocab]` tensor into `[vocab]`-sized counters, and discards it. No
accumulator is indexed by position, so memory is flat in the position count.

That is not an optimization. Holding every position's logits at a realistic
vocabulary would cost roughly 3.9 GiB for 32768 positions over 32000 tokens, so
the experiment simply could not run.

Sampling randomness is drawn up front and indexed by position, so a streamed
measurement equals an all-at-once one **exactly** rather than approximately.
`measure_from_logits` is the all-at-once reference the streaming path is tested
against; `compute_evaluation_logits` materializes everything and is appropriate
only for small vocabularies or few positions.

Size `forward_batch_size` against the vocabulary: the transient logits are
`batch x block x vocab`, and nucleus truncation sorts them, which costs several
multiples again.

### What the experiment does not measure

Model quality. The model is randomly initialized and has learned nothing. Results describe
the interaction between an untrained architecture, its initialization scheme, and the
corpus it is compared against.

With a pretrained tokenizer there is a second, sharper boundary. The tokenizer
carries linguistic structure learned from **its own** training corpus: the
segmentation, the frequency profile of the pieces, and which strings are single
tokens at all. The experiment therefore measures

> the distributional behavior of a randomly initialized tiny LLaMA **over a
> realistic pretrained subword vocabulary**,

not a completely unlearned text-processing system. Structure visible in the
corpus token distribution belongs to the tokenizer and the text; only the guess
distribution belongs to the random model. Do not attribute the former to the
latter.

### Scripts using this package

- `scripts/run_initialization_distribution_experiment.py`
- `notebooks/initialization_distribution.ipynb`

---

## `llm_behavior_lab.utils`

The `utils` package contains small helpers shared across scripts and modules.

Current files:

```text
src/llm_behavior_lab/utils/
  __init__.py
  config.py
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
| Change config-loading behavior | `utils/config.py` |
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

### Structured Phase 5 persistence

```text
analyze_untrained_model.py --persist-run
  -> load experiment YAML
  -> experiment_settings_from_config(...)
  -> ExperimentRun.create(...)
  -> snapshot model/data/experiment configs
  -> compute existing Phase 5 metrics
  -> ArrayMetricStore.save(...)
  -> MetricLogger.log(...)
  -> CheckpointManager.save(step=0)
  -> save analysis JSON
```

### Phase 6 smoke-test round trip

```text
check_experiment_tracking.py
  -> create run
  -> append training/evaluation JSONL
  -> save/load .npz array artifact
  -> save initialized checkpoint
  -> discover latest checkpoint
  -> restore into compatible CPU model
  -> compare every state-dict tensor
```

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

### Extend experiment logging

Current location:

```text
src/llm_behavior_lab/experiment/
```

Add new scalar channels through `MetricLogger`, new vectors through `ArrayMetricStore`, and new structured summaries under `ExperimentRun.paths.analyses_dir`. Keep training loops outside this package; Phase 7 should call these interfaces rather than replace them.

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

### Extend checkpoint loading/evaluation

Checkpoint I/O now lives in:

```text
src/llm_behavior_lab/experiment/checkpoints.py
```

Evaluation code should continue to receive already-built models and tensors. Scripts or future training code should use `CheckpointManager` to restore state before calling evaluation functions.

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

Current helpers cover config loading, device selection, parameter counting, and seeding.

---

### Where do I change experiment directories, metrics, or checkpoints?

- run layout and metadata: `experiment/run.py`
- scalar JSONL records: `experiment/metrics.py`
- array artifacts: `experiment/arrays.py`
- checkpoint format and restore behavior: `experiment/checkpoints.py`
- YAML settings: `experiment/config.py`

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
