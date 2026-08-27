# Scripts

This folder contains runnable command-line entry points for the LLM Behavior Lab project.

The Python package code lives under `src/llm_behavior_lab/`. The scripts in this folder are thin, practical entry points that connect configuration files, data utilities, model construction, inference helpers, and evaluation routines.

The scripts are intentionally kept outside the library modules so they can be run directly from the repository root during development.

---

## Purpose of the `scripts/` folder

The `scripts/` folder provides executable checks and analysis routines.

At the current project stage, scripts are used to:

- verify that the LLaMA-style model builds and runs
- verify that local text data can be tokenized and batched
- verify that batches are compatible with the model forward pass
- run prompt-based inference on the untrained model
- inspect output behavior at random initialization
- optionally compute per-layer gradient-norm diagnostics
- optionally persist Phase 5 analyses as structured experiment runs
- validate Phase 6 metric, array, and checkpoint round trips

The scripts do not define the core model, data, inference, or evaluation logic. Instead, they call reusable modules from:

```text
src/llm_behavior_lab/models/
src/llm_behavior_lab/data/
src/llm_behavior_lab/inference/
src/llm_behavior_lab/evaluation/
src/llm_behavior_lab/experiment/
src/llm_behavior_lab/utils/
```

This keeps the project modular: scripts provide user-facing execution paths, while library modules remain reusable by tests, notebooks, and future training loops.

---

## Current scripts overview

Current scripts:

```text
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
  write_nucleus_clustering_artifact.py
  write_cross_partition_artifact.py
```

| Script | Type | Phase / feature | Purpose |
|---|---|---|---|
| `smoke_test_llama.py` | Smoke test | Phase 2 model integration | Builds the tiny LLaMA-style model and runs a dummy forward pass. |
| `check_data_pipeline.py` | Sanity check | Phase 3 data pipeline | Loads the tiny local corpus, tokenizes it, creates train/validation batches, and checks model compatibility. |
| `run_inference.py` | Inference check | Phase 4 inference utilities | Encodes a prompt, extracts logits, prints top-k next-token predictions, and generates a short continuation. |
| `analyze_untrained_model.py` | Analysis script | Phase 5 + Phase 6 integration | Computes initialization metrics and optionally persists a structured run. |
| `check_experiment_tracking.py` | Smoke test | Phase 6 persistence | Creates a run, logs metrics/arrays, saves a checkpoint, restores it, and verifies parameter equality. |
| `prepare_dataset.py` | Utility | Dataset resolution | Stages a dataset ahead of time, or reports what is missing without obtaining it. |
| `run_initialization_distribution_experiment.py` | Experiment | Initialization distributions | Measures greedy and nucleus token guesses across several random initializations on fixed evaluation positions, then writes a record and figures. Supports both the character tokenizer and a pretrained subword tokenizer. |
| `run_initialization_scale_experiment.py` | Experiment | Initialization scale | Repeats the whole pipeline at three initialization scales, writing one self-contained run per scale plus a parent manifest. |
| `benchmark_position_gradients.py` | Benchmark | Per-position gradient cost | Times the exact per-position parameter-gradient measurement at several window counts. **Performance only: writes no record and draws no figure.** See the section below. |
| `render_record_figures.py` | Analysis | Figure rendering | Redraws figures, and prints gradient statistics, from a persisted record. Imports no PyTorch. |
| `write_nucleus_clustering_artifact.py` | Analysis | Nucleus-grouped gradients | Recovers a run's nucleus sample labels, gated on exact histogram equality, and clusters its gradients by them. Writes an artifact beside the record; never modifies it. |
| `write_cross_partition_artifact.py` | Analysis | Cross-partition geometry | Derives the target-versus-greedy cross-partition statistics from a record by NumPy alone, including the permutation null. Writes an artifact beside the record; never modifies it. |

---

## Recommended order for new users

When first checking the repository, run the scripts in this order:

1. **Model sanity check**
   ```bash
   python3 scripts/smoke_test_llama.py --config configs/model/tiny_llama.yaml
   ```

2. **Data/model compatibility check**
   ```bash
   python3 scripts/check_data_pipeline.py \
     --data-config configs/data/tiny_text.yaml \
     --model-config configs/model/tiny_llama.yaml
   ```

3. **Inference check**
   ```bash
   python3 scripts/run_inference.py \
     --data-config configs/data/tiny_text.yaml \
     --model-config configs/model/tiny_llama.yaml
   ```

4. **Untrained-model output analysis**
   ```bash
   python3 scripts/analyze_untrained_model.py \
     --data-config configs/data/tiny_text.yaml \
     --model-config configs/model/tiny_llama.yaml
   ```

5. **Optional gradient-norm diagnostic**
   ```bash
   python3 scripts/analyze_untrained_model.py \
     --data-config configs/data/tiny_text.yaml \
     --model-config configs/model/tiny_llama.yaml \
     --compute-grad-norms
   ```


6. **Phase 6 experiment-persistence smoke test**
   ```bash
   python3 scripts/check_experiment_tracking.py \
     --model-config configs/model/tiny_llama.yaml \
     --data-config configs/data/tiny_text.yaml \
     --experiment-config configs/experiment/phase6_smoke.yaml
   ```

This order follows the project development sequence: model first, data second, inference third, analysis fourth, and persistence after the reusable metrics are available.

---

## Common execution pattern

Most scripts follow the same high-level pattern:

1. Parse command-line arguments with `argparse`.
2. Load YAML configs from `configs/`.
3. Set seed and device using utilities from `src/llm_behavior_lab/utils/`.
4. Build data objects and/or the model.
5. Run a forward pass, inference call, or analysis routine.
6. Print diagnostics to the terminal.
7. Save lightweight artifacts only when explicitly requested by the script options.

Scripts can be run from the repository root without installing the package first because they add `src/` to `sys.path` at runtime. Editable installation is still recommended for normal development:

```bash
python3 -m pip install -e ".[dev]"
```

---

## Script-by-script guide

## `smoke_test_llama.py`

### Purpose

`smoke_test_llama.py` verifies that the LLaMA-style model can be built from a YAML config and can run a forward pass on synthetic token IDs.

This script does not use the tokenizer or real dataset. It is a model-only sanity check.

### Project phase

Phase 2: LLaMA-style model integration.

### Type

Smoke test.

### Main modules used

```text
llm_behavior_lab.models
llm_behavior_lab.utils.device
llm_behavior_lab.utils.seed
llm_behavior_lab.utils.params
```

### Inputs

Default config:

```text
configs/model/tiny_llama.yaml
```

Optional arguments:

| Argument | Meaning |
|---|---|
| `--config` | Model config path. |
| `--batch-size` | Synthetic batch size. Default: `2`. |
| `--sequence-length` | Synthetic sequence length. Default: `16`. Must be no larger than model `max_seq_len`. |

### Command

```bash
python3 scripts/smoke_test_llama.py --config configs/model/tiny_llama.yaml
```

Optional custom shape:

```bash
python3 scripts/smoke_test_llama.py \
  --config configs/model/tiny_llama.yaml \
  --batch-size 2 \
  --sequence-length 16
```


Persist the analysis as a structured run:

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml \
  --experiment-config configs/experiment/untrained_baseline.yaml \
  --persist-run
```

Persist output and gradient diagnostics together:

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml \
  --experiment-config configs/experiment/untrained_baseline.yaml \
  --persist-run \
  --compute-grad-norms
```

### Expected successful output

Successful output should include:

```text
Available registered models: ...
Using device: ...
Model name: llama_tiny
Parameter count: ...
Input shape: ...
Target shape: ...
Logits shape: ...
Loss shape: ()
```

For the default tiny config, logits should have shape:

```text
(2, 16, 256)
```

### Common failure modes

| Symptom | Likely cause | What to check |
|---|---|---|
| Config file not found | Wrong path or not running from repo root | Check `configs/model/tiny_llama.yaml`. |
| Sequence length error | Requested sequence exceeds model context length | Lower `--sequence-length` or increase `max_seq_len` in config. |
| CUDA warning | PyTorch detects an unusable CUDA driver | Use `device: cpu` in config or fix CUDA/driver installation. |
| Shape mismatch | Config inconsistency | Check `dim`, `n_heads`, `n_kv_heads`, and `vocab_size`. |

---

## `check_data_pipeline.py`

### Purpose

`check_data_pipeline.py` verifies that the local text data path works end to end:

1. load local text
2. build the character tokenizer
3. encode text into token IDs
4. create train/validation splits
5. sample causal LM batches
6. pass one batch through the model

This script checks that the Phase 3 data pipeline is compatible with the Phase 2 LLaMA-style model.

### Project phase

Phase 3: data pipeline integration.

### Type

Data/model compatibility sanity check.

### Main modules used

```text
llm_behavior_lab.data.text_dataset
llm_behavior_lab.data.tokenizer
llm_behavior_lab.data.dataloader
llm_behavior_lab.models
llm_behavior_lab.utils
```

### Inputs

Default data config:

```text
configs/data/tiny_text.yaml
```

Default model config:

```text
configs/model/tiny_llama.yaml
```

The data config points to the tiny local corpus:

```text
data/raw/tiny_corpus.txt
```

### Command

```bash
python3 scripts/check_data_pipeline.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

### Expected successful output

Successful output should include:

```text
Dataset path: ...
Raw text characters: ...
Tokenizer vocab size: ...
Total token count: ...
Train token count: ...
Validation token count: ...
Train input shape: (4, 16)
Train target shape: (4, 16)
Validation input shape: (4, 16)
Validation target shape: (4, 16)
Logits shape: (4, 16, 256)
Loss shape: ()
```

The exact token counts depend on the contents of `data/raw/tiny_corpus.txt`.

### Common failure modes

| Symptom | Likely cause | What to check |
|---|---|---|
| Dataset file not found | `data/raw/tiny_corpus.txt` missing | Restore local corpus or update `configs/data/tiny_text.yaml`. |
| Not enough tokens for split | Corpus too small for `block_size` and validation split | Lower `block_size`, lower `val_fraction`, or use more text. |
| Tokenizer vocab exceeds model vocab | Corpus has more unique characters than `model.params.vocab_size` | Increase model vocab size or change tokenizer/data. |
| Block size exceeds model context | Data `block_size` larger than model `max_seq_len` | Lower `block_size` or increase model `max_seq_len`. |

---

## `run_inference.py`

### Purpose

`run_inference.py` runs a short prompt-based inference check with the untrained model.

It verifies:

- prompt encoding
- model-ready prompt tensors
- full-logits extraction
- next-token logits extraction
- next-token probabilities
- top-k next-token inspection
- greedy or sampling decoding
- short text generation

The generated text is not expected to be meaningful because the model is randomly initialized.

### Project phase

Phase 4: inference utilities.

### Type

Inference sanity check.

### Main modules used

```text
llm_behavior_lab.data
llm_behavior_lab.inference.generation
llm_behavior_lab.models
llm_behavior_lab.utils
```

### Inputs

Default data config:

```text
configs/data/tiny_text.yaml
```

Default model config:

```text
configs/model/tiny_llama.yaml
```

Optional arguments:

| Argument | Meaning |
|---|---|
| `--prompt` | Prompt text. Default: `"The "`. |
| `--max-new-tokens` | Number of tokens to generate. Default: `4`. |
| `--strategy` | Decoding strategy: `greedy` or `sample`. |
| `--temperature` | Sampling temperature. Must be positive. |
| `--top-k` | Optional top-k sampling filter. |
| `--num-top-predictions` | Number of next-token predictions to print. |

### Command

```bash
python3 scripts/run_inference.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

Sampling example:

```bash
python3 scripts/run_inference.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml \
  --strategy sample \
  --temperature 0.8 \
  --top-k 10
```

### Expected successful output

Successful output should include:

```text
Phase 4 inference check completed successfully.
Tokenizer vocab size: ...
Model vocab size: ...
Prompt: ...
Encoded prompt token IDs: ...
Input tensor shape: ...
Full logits shape: ...
Next-token logits shape after tokenizer-vocab restriction: ...
Top-k next-token predictions:
Generated token IDs: ...
New token IDs: ...
Decoded generated text: ...
```

For the default prompt, input tensor shape should usually be:

```text
(1, 4)
```

Full logits shape should be:

```text
(1, 4, 256)
```

### Common failure modes

| Symptom | Likely cause | What to check |
|---|---|---|
| Prompt contains unknown character | Character not present in tokenizer vocabulary | Use a prompt with characters from the tiny corpus or expand the corpus/tokenizer. |
| Generated text is meaningless | Model is untrained | This is expected at the current phase. |
| Top-k error | Invalid `--top-k` or incompatible sampling settings | Use positive `--top-k` and positive `--temperature`. |
| CUDA warning | PyTorch/CUDA environment issue | Use CPU config or fix driver/server environment. |

---

## `analyze_untrained_model.py`

### Purpose

`analyze_untrained_model.py` runs the main Phase 5 baseline analysis of the randomly initialized model.

It measures output behavior before training, including:

- logits shape
- probability tensor shape
- entropy
- top-k examples
- top-1 prediction behavior
- empirical token-frequency comparison
- probability-frequency gaps
- KL divergence
- Jensen-Shannon divergence

It can also optionally compute per-layer gradient-stability diagnostics.

### Project phase

Phase 5: baseline untrained-model analysis.

### Type

Analysis script.

### Main modules used

```text
llm_behavior_lab.data
llm_behavior_lab.models
llm_behavior_lab.evaluation.output_stats
llm_behavior_lab.evaluation.token_frequency
llm_behavior_lab.evaluation.untrained_analysis
llm_behavior_lab.evaluation.gradient_norms
llm_behavior_lab.utils
```

### Inputs

Default data config:

```text
configs/data/tiny_text.yaml
```

Default model config:

```text
configs/model/tiny_llama.yaml
```

Optional arguments:

| Argument | Meaning |
|---|---|
| `--split` | Token split to analyze: `train` or `val`. Default: `train`. |
| `--num-batches` | Number of random batches/windows to analyze. Default: `4`. |
| `--top-k` | Number of top tokens/gaps to display. Default: `5`. |
| `--max-examples` | Number of example positions for detailed top-k predictions. Default: `3`. |
| `--compute-grad-norms` | Enables per-layer squared L2 gradient-norm diagnostic. |
| `--grad-norm-num-batches` | Number of batches for gradient-norm computation. Default: `1`. |
| `--grad-norm-eps` | Stabilizer for `log(g_l + eps)`. Default: `1e-12`. |
| `--grad-norm-output-dir` | Legacy standalone gradient JSON/CSV directory when `--persist-run` is not used. |
| `--persist-run` | Creates a structured Phase 6 experiment run. |
| `--experiment-config` | Experiment config; defaults to `configs/experiment/untrained_baseline.yaml`. |
| `--run-id` | Optional explicit run ID; collisions fail instead of overwriting. |
| `--output-dir` | Optional output-root override. |
| `--notes` | Optional run notes override. |

### Command

Default output analysis:

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

Use validation split:

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml \
  --split val
```

Compute gradient norms:

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml \
  --compute-grad-norms
```

Compute gradient norms with custom output directory:

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml \
  --compute-grad-norms \
  --grad-norm-num-batches 2 \
  --grad-norm-output-dir outputs/phase5_gradient_norms
```

### Expected successful output

Successful output should include:

```text
Phase 5 untrained-model analysis completed successfully.
Tokenizer vocab size: ...
Analyzed batches: ...
Analyzed windows: ...
Analyzed positions: ...
Input tensor shape: ...
Logits shape: ...
Probability tensor shape: ...
Mean output entropy: ...
Mean top-1 probability: ...
Top-1 assignment concentration: ...
KL(predicted || empirical): ...
JS(predicted, empirical): ...
```

It should then print:

```text
Example top-k next-token predictions:
Most frequent empirical dataset tokens:
Most frequent top-1 predicted tokens:
Largest positive probability-frequency gaps:
Largest negative probability-frequency gaps:
```

When `--compute-grad-norms` is enabled, output should also include:

```text
Computing per-layer gradient norms...

Per-layer gradient norm diagnostic:
  Definition: squared_l2_norm_of_gradient_wrt_decoder_block_output
  Gradient batches: ...
  Mean gradient-diagnostic loss: ...
  Squared L2 gradient norms: [...]
  Log squared L2 gradient norms: [...]
  Log-linear slope: ...
  Log-linear intercept: ...
  Slope standard error: ...
  R^2: ...
  Saved gradient JSON: ...
  Saved gradient CSV: ...
```

### Output files

By default, gradient outputs are saved to:

```text
outputs/phase5_gradient_norms/
  gradient_norms.json
  gradient_norms.csv
```

Without `--persist-run`, these files remain lightweight standalone artifacts. With `--persist-run`, Phase 6 stores scalar summaries, arrays, analysis JSON, configs, metadata, and an initialized checkpoint inside one run directory.

### Common failure modes

| Symptom | Likely cause | What to check |
|---|---|---|
| Dataset path missing | `data/raw/tiny_corpus.txt` missing or ignored locally | Recreate local corpus or update data config. |
| Tokenizer/model vocab mismatch | More tokenizer symbols than model vocab size | Increase `vocab_size` in model config. |
| Block size mismatch | Data block size exceeds model max sequence length | Adjust `configs/data/tiny_text.yaml` or `configs/model/tiny_llama.yaml`. |
| Gradient diagnostic is slower | Backward pass is enabled | This is expected with `--compute-grad-norms`. |
| CUDA warning during gradient norms | PyTorch detects CUDA but driver is unavailable or too old | Treat as useful environment information; run on CPU or fix GPU environment. |

---

## `check_experiment_tracking.py`

### Purpose

`check_experiment_tracking.py` validates the Phase 6 persistence lifecycle without performing training.

It demonstrates:

1. experiment settings loading
2. collision-safe run-directory creation
3. metadata and config snapshots
4. training/evaluation JSONL appends
5. compressed `.npz` array saving and loading
6. initialized model checkpoint saving
7. `latest.json` discovery
8. CPU checkpoint restoration into a new model
9. exact parameter equality after restoration

### Project phase

Phase 6: experiment logging and checkpoint infrastructure.

### Main modules used

```text
llm_behavior_lab.experiment.ExperimentRun
llm_behavior_lab.experiment.MetricLogger
llm_behavior_lab.experiment.ArrayMetricStore
llm_behavior_lab.experiment.CheckpointManager
llm_behavior_lab.models.build_model_from_config
```

### Command

```bash
python3 scripts/check_experiment_tracking.py \
  --model-config configs/model/tiny_llama.yaml \
  --data-config configs/data/tiny_text.yaml \
  --experiment-config configs/experiment/phase6_smoke.yaml
```

Use a disposable output root during local testing:

```bash
python3 scripts/check_experiment_tracking.py \
  --output-dir /tmp/igb_phase6_outputs \
  --run-id smoke_run
```

### Expected successful output

```text
Phase 6 experiment tracking check completed successfully.
Run directory: ...
Metadata path: .../metadata.json
Training metrics: .../training_metrics.jsonl
Evaluation metrics: .../evaluation_metrics.jsonl
Array artifact: .../.npz
Checkpoint: .../checkpoint_step_000000.pt
Latest checkpoint: .../checkpoint_step_000000.pt
Restored global step: 0
Checkpoint parameter round-trip: verified
```

### Side effects

The script writes a complete run under `outputs/<experiment-name>/<run-id>/`. `outputs/` is ignored by Git. Reusing an explicit `--run-id` raises `FileExistsError`; delete the test directory or choose a new ID.

### Common failure modes

| Symptom | Cause | Resolution |
|---|---|---|
| Run already exists | Explicit run ID collision | Choose a new `--run-id` or remove the disposable run. |
| Checkpoint missing | Save failed or directory changed | Inspect `checkpoints/` and `latest.json`. |
| State-dict mismatch | Model config differs from checkpoint | Restore into a model built from the snapshotted model config. |
| Permission error | Output root is not writable | Use `--output-dir` with a writable path. |

---

## `benchmark_position_gradients.py`

### Purpose

Times the exact per-position parameter-gradient measurement — one backward pass per
evaluation position per loss temperature — so the cost of a gradient run can be measured
before one is committed to.

**It is performance-only.** It writes no experiment record, produces no figure, and creates
no run directory. Nothing it prints is a scientific result, and no number from it belongs in
a findings table.

### Type

Benchmark. Not a sanity check and not an experiment.

### What it reports

- wall time per window count, so scaling can be **checked** rather than assumed from a
  single point — pass several values to `--window-counts`;
- measured positions/s and backwards/s;
- on CUDA, peak **allocated** and peak **reserved** device memory per row, from
  `torch.cuda.max_memory_allocated` and `max_memory_reserved`;
- process peak RSS, which is a high-water mark **over the whole benchmark** and therefore
  cannot be attributed to any single row;
- a confirmation that the model parameters were not modified;
- a projected full-run cost, printed under an explicit label saying it was **not** measured.

### Count-sketch options

The sketch is off by default, which keeps older invocations comparable. The map tables are
device-resident, so **the memory columns only describe a campaign configuration when
`--gradient-sketch` is on**; without it the benchmark says so rather than letting the
columns be misread.

| Option | Meaning |
|---|---|
| `--gradient-sketch` | Also project every gradient through the production count sketch, as a scientific run does. |
| `--sketch-dimension K` | Sketch width. Resolved exactly as the runner resolves it: this flag, then the experiment config's `gradient_analysis.sketch_dimension`, then the production default of 512. |
| `--sketch-maps M` | Independent map replicas. **Only `M = 1` can be measured**, because the production estimator builds a single map; a larger value is refused rather than silently timed, and stays refused until multi-map support lands in the estimator itself. |
| `--sketch-seed` | Seed for map construction. Benchmark-only — the runner has no such option and always uses the production default. Timing does not depend on it. |

Device-resident map tables cost **5 bytes per parameter per map** (an `int32` bucket and an
`int8` sign), and the benchmark derives that figure from the implementation's own dtypes
rather than restating it, so the two cannot drift apart.

### The narrow-vocabulary guard

A tokenizer narrower than the model's output head is **refused by default**. Logits are
truncated to the tokenizer vocabulary before the loss, so the output-layer gradient — the
dominant cost — would be measured over only a fraction of the head, and the result would not
be a production cost estimate. `--allow-narrow-vocabulary` permits it anyway, which the
legacy tiny-character pairing needs; passing it is an acknowledgement of what is being
measured.

### Command

```bash
python3 scripts/benchmark_position_gradients.py \
  --data-config configs/data/wikitext2_subword.yaml \
  --model-config configs/model/tiny_llama_32k.yaml --offline \
  --window-counts 1 2 4 8
```

With the sketch enabled, which is what a campaign configuration actually costs:

```bash
python3 scripts/benchmark_position_gradients.py \
  --data-config configs/data/wikitext2_subword.yaml \
  --model-config configs/model/tiny_llama_32k.yaml --offline \
  --window-counts 1 2 4 \
  --gradient-sketch --sketch-dimension 512 --sketch-maps 1
```

Pass `--temperatures 1.0` alone to time the canonical baseline for a like-for-like
comparison against older measurements.

---

## Dataset options shared by data-consuming scripts

`check_data_pipeline.py`, `run_inference.py`, `analyze_untrained_model.py`,
`prepare_dataset.py`, and `run_initialization_distribution_experiment.py` all accept the
same dataset-resolution options:

| Option | Meaning |
|---|---|
| `--data-root` | Root for prepared and cached datasets. Defaults to `$LLM_BEHAVIOR_LAB_DATA_ROOT`, then a platform cache directory outside the repository. |
| `--no-download` | Reuse datasets already available locally; never obtain a missing one. |
| `--offline` | Forbid all network access. Implies `--no-download`. |
| `--force-refresh` | Re-acquire an external dataset, replacing any prepared copy. |

Scripts acquire a missing dataset by default and announce it before any network
activity. The tracked tiny corpus never needs any of this: it resolves from the
repository and never consults the data root or the network.

`check_data_pipeline.py` additionally accepts `--skip-model-check` to verify
dataset loading, tokenization, and batching without a model forward pass.

See [`data/README.md`](../data/README.md) for the resolution order, the data
root, and prepared-data manifests.

---

## Interaction with configs

Scripts use configs from:

```text
configs/
  data/
    tiny_text.yaml
  model/
    tiny_llama.yaml
```

### Model config

Used by:

```text
smoke_test_llama.py
check_data_pipeline.py
run_inference.py
analyze_untrained_model.py
```

Changing the model config can affect:

- parameter count
- model vocabulary size
- context length
- number of layers
- attention heads
- hidden dimension
- logits shape
- memory use
- runtime

### Data config

Used by:

```text
check_data_pipeline.py
run_inference.py
analyze_untrained_model.py
```

Changing the data config can affect:

- dataset path
- validation split
- tokenizer vocabulary
- batch size
- block size
- runtime device
- model/data compatibility

### Runtime device

Most scripts use:

```yaml
runtime:
  device: auto
```

The device is resolved by:

```text
src/llm_behavior_lab/utils/device.py
```

`auto` will prefer available accelerators where supported. On machines with a CUDA-enabled PyTorch build but an old or unavailable NVIDIA driver, PyTorch may emit CUDA warnings. These warnings are useful because they reveal environment issues that may matter later when running on GPU servers.

---

## Interaction with source modules

The scripts are entry points, not implementation containers.

The reusable implementation lives in:

| Package area | Role |
|---|---|
| `src/llm_behavior_lab/models/` | Model interface, registry, and two explicit decoder implementations: LLaMA-style and GPT-2-style. |
| `src/llm_behavior_lab/data/` | Local text loading, tokenization, splitting, and causal LM batching. |
| `src/llm_behavior_lab/inference/` | Prompt preparation, logits extraction, probabilities, decoding, and generation. |
| `src/llm_behavior_lab/evaluation/` | Output statistics, token-frequency comparison, untrained analysis, and gradient norms. |
| `src/llm_behavior_lab/experiment/` | Run creation, metadata, JSONL metrics, array artifacts, and checkpoints. |
| `src/llm_behavior_lab/utils/` | Seeding, device selection, and parameter formatting. |

This structure allows later phases to add training and checkpointed evaluation without duplicating script logic.

---

## Output files and side effects

Most scripts only print diagnostics to the terminal.

| Script | Prints to terminal | Writes files | Notes |
|---|---:|---:|---|
| `smoke_test_llama.py` | Yes | No | Synthetic model smoke test. |
| `check_data_pipeline.py` | Yes | No | Reads local text file. |
| `run_inference.py` | Yes | No | Generated text is printed only. |
| `analyze_untrained_model.py` | Yes | Optional | Standalone gradient files or a full structured run with `--persist-run`. |
| `check_experiment_tracking.py` | Yes | Yes | Creates a Phase 6 smoke-test run and checkpoint. |

Generated outputs currently go under:

```text
outputs/
```

The `.gitignore` should keep generated outputs and large local datasets out of Git.

---

## Testing scripts

The test suite covers the reusable library code that the scripts call, plus one script workflow end to end.

Run:

```bash
python3 -m pytest
```

The tests cover:

- imports
- LLaMA shape checks
- tokenizer/data pipeline
- inference utilities
- output-analysis utilities
- gradient-norm diagnostics
- experiment-run creation and collision safety
- JSONL metrics and array artifact round trips
- checkpoint save/load, latest discovery, and CPU portability
- the persisted-run workflow of `analyze_untrained_model.py`, driven through its entry point in `tests/test_persisted_analysis_run.py`
- dataset configuration, resolution policy, prepared data, and manifests
- Hugging Face acquisition, exercised through a fake loader so the suite stays offline
- the migrated script entry points, driven through their real command lines

Apart from that persisted-run coverage, the scripts are intended as human-readable execution checks. If a script fails, first check:

1. that you are running from the repository root
2. that configs exist at the paths passed on the command line
3. that local data exists under `data/`
4. that package imports work
5. that the selected runtime device is available

---

## Future scripts

Likely future scripts include:

- `train_pretrain.py` for small-scale pre-training
- `evaluate_checkpoint.py` for checkpoint-based evaluation
- `analyze_training_dynamics.py` for comparing metrics over checkpoints
- `finetune.py` for supervised fine-tuning
- `compare_models.py` for model-family comparisons
- `compute_dataset_stats.py` for dataset/tokenizer statistics

These should remain thin command-line entry points that call reusable code under `src/llm_behavior_lab/`.
