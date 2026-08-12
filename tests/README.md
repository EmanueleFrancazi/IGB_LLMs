# Tests

This folder contains the automated test suite for the LLM Behavior Lab project.

The tests protect the incremental development process. Each project phase adds a small number of components, and the tests verify that those components keep working as new functionality is added.

At the current stage, the test suite covers:

- package imports
- LLaMA-style model construction and output shapes
- local text tokenization and causal language-modeling batches
- inference utilities
- Phase 5 output-analysis utilities
- Phase 5 gradient-norm diagnostics
- Phase 6 run creation, JSONL metrics, array artifacts, and checkpoint round trips

The tests are intentionally lightweight. They use small synthetic tensors and tiny text snippets rather than large datasets or expensive training runs.

---

## Purpose of the `tests/` folder

The `tests/` folder is separate from both source code and runnable scripts.

```text
src/
  reusable implementation code

scripts/
  command-line entry points

tests/
  automated checks for source behavior and integration contracts
```

This separation helps keep the project stable as new phases are added.

Tests are especially important in this repository because the codebase is being built incrementally. Each phase depends on earlier pieces continuing to work:

1. model construction must remain stable
2. data batches must keep the expected shape
3. inference helpers must remain compatible with the model and tokenizer
4. evaluation utilities must accept the expected logits/probability tensors
5. gradient diagnostics must remain compatible with the model architecture

When modifying source code under `src/`, add or update tests under `tests/`.

---

## Current test structure

Current files:

```text
tests/
  test_data_pipeline.py
  test_evaluation.py
  test_experiment_tracking.py
  test_gradient_norms.py
  test_imports.py
  test_inference.py
  test_llama_shapes.py
  test_persisted_analysis_run.py
  conftest.py
  test_dataset_config.py
  test_dataset_resolver.py
  test_dataset_prepared.py
  test_dataset_huggingface.py
  test_dataset_cli.py
  test_dataset_tracking.py
  test_migrated_workflows.py
```

| File | Main focus | Type | Related project area |
|---|---|---|---|
| `test_imports.py` | Core package imports and public exports | Import/smoke test | Package-level API |
| `test_llama_shapes.py` | LLaMA model registry, construction, forward shapes, cache path | Unit/integration tests | `models/` |
| `test_data_pipeline.py` | Character tokenizer, train/validation split, causal LM batcher | Unit tests | `data/` |
| `test_inference.py` | Prompt preparation, logits/probabilities, top-k predictions, decoding, generation | Unit/integration tests | `inference/` |
| `test_evaluation.py` | Output statistics, empirical frequencies, probability gaps, untrained analysis | Unit tests | `evaluation/` |
| `test_experiment_tracking.py` | Run creation, JSONL metrics, arrays, checkpoints, latest discovery | Filesystem/unit/integration tests | `experiment/` |
| `test_gradient_norms.py` | Per-layer gradient trend fit, gradient-norm computation, JSON/CSV saving | Unit/integration tests | `evaluation/gradient_norms.py` |
| `test_persisted_analysis_run.py` | Persisted initialization-analysis workflow: run layout, snapshots, metrics, arrays, checkpoint restore | End-to-end integration tests | `scripts/analyze_untrained_model.py --persist-run` |
| `test_dataset_config.py` | Dataset identity, resolution policy, data-root selection | Unit tests | `data/config.py`, `data/errors.py` |
| `test_dataset_resolver.py` | Local-first resolution, routes, unavailable-data messages | Unit/integration tests | `data/resolver.py` |
| `test_dataset_prepared.py` | Prepared directories, manifests, staleness, atomicity | Filesystem tests | `data/prepared.py` |
| `test_dataset_huggingface.py` | Acquisition, cache reuse, offline enforcement, refresh, resolved provenance, announcements | Mocked integration tests | `data/huggingface.py` |
| `test_dataset_cli.py` | Shared dataset options and policy construction | Unit tests | `data/cli.py` |
| `test_dataset_tracking.py` | Dataset contents cannot enter Git | Repository tests | `.gitignore` |
| `test_migrated_workflows.py` | Migrated scripts still behave as before | End-to-end integration tests | `scripts/*.py` |
| `test_guessing.py` | Greedy argmax, nucleus truncation rule, sampling reproducibility, support restriction | Unit tests | `evaluation/guessing.py` |
| `test_init_distribution.py` | Deterministic evaluation positions, seed separation, per-initialization measurement | Unit tests | `evaluation/init_distribution.py` |
| `test_analysis_records.py` | Record round trip, token alignment, validation | Filesystem/unit tests | `analysis/records.py` |
| `test_analysis_aggregation.py` | Ranked profiles, same-token gaps, SEM, scalar measures | Unit tests | `analysis/aggregation.py` |
| `test_analysis_figures.py` | Headless figure generation and deterministic filenames | Filesystem tests | `analysis/figures.py` |

---

## Test categories

## Import and public API tests

File:

```text
tests/test_imports.py
```

Purpose:

- verifies that the package imports correctly
- checks that important public functions/classes are exposed
- catches broken `__init__.py` exports early

Related source modules:

```text
src/llm_behavior_lab/__init__.py
src/llm_behavior_lab/data/__init__.py
src/llm_behavior_lab/models/__init__.py
src/llm_behavior_lab/inference/__init__.py
src/llm_behavior_lab/evaluation/__init__.py
src/llm_behavior_lab/utils/__init__.py
```

This is a lightweight smoke test for the package API.

---

## Model tests

File:

```text
tests/test_llama_shapes.py
```

Purpose:

- checks that LLaMA builders are registered
- verifies that `build_model` returns a `BaseLanguageModel`
- verifies logits shape from a forward pass
- verifies scalar loss when targets are provided
- verifies config-based model construction
- verifies the one-token KV-cache inference path

Related source modules:

```text
src/llm_behavior_lab/models/base.py
src/llm_behavior_lab/models/registry.py
src/llm_behavior_lab/models/llama/config.py
src/llm_behavior_lab/models/llama/model.py
```

Important shape contract:

```text
input_ids: [batch_size, sequence_length]
targets:   [batch_size, sequence_length]
logits:    [batch_size, sequence_length, vocab_size]
loss:      scalar when targets are provided
```

---

## Data-pipeline tests

File:

```text
tests/test_data_pipeline.py
```

Purpose:

- checks character tokenizer round-trip behavior
- checks deterministic train/validation split lengths
- checks causal LM input/target batch shapes
- verifies the one-token shift between inputs and targets

Related source modules:

```text
src/llm_behavior_lab/data/tokenizer.py
src/llm_behavior_lab/data/text_dataset.py
src/llm_behavior_lab/data/dataloader.py
```

Important batch contract:

```text
batch.input_ids.shape == (batch_size, block_size)
batch.targets.shape   == (batch_size, block_size)
batch.targets[:, :-1] == batch.input_ids[:, 1:]
```

These tests protect the core data shape assumptions that later training loops will depend on.

---

## Inference tests

File:

```text
tests/test_inference.py
```

Purpose:

- checks prompt encoding and left truncation
- checks logits extraction
- checks next-token logits and probabilities
- checks top-k prediction decoding
- checks greedy next-token selection
- checks short generation output structure

Related source modules:

```text
src/llm_behavior_lab/inference/generation.py
src/llm_behavior_lab/data/tokenizer.py
src/llm_behavior_lab/models/
```

These tests use a very small LLaMA-style model and tiny character tokenizer.

They do not check language quality. The current model is untrained, so generation quality is not meaningful yet. The tests only check interface and shape correctness.

---

## Evaluation tests

File:

```text
tests/test_evaluation.py
```

Purpose:

- checks logits-to-probabilities conversion
- checks entropy computation
- checks output-distribution summaries
- checks top-1 IDs and top-k probability mass
- checks empirical token counts and frequencies
- checks probability-frequency gaps
- checks KL and Jensen-Shannon divergence helpers
- checks the combined untrained-output analysis result structure

Related source modules:

```text
src/llm_behavior_lab/evaluation/output_stats.py
src/llm_behavior_lab/evaluation/token_frequency.py
src/llm_behavior_lab/evaluation/untrained_analysis.py
```

These tests protect the Phase 5 initialization-analysis utilities.

---

## Gradient-norm diagnostic tests

File:

```text
tests/test_gradient_norms.py
```

Purpose:

- checks that a known exponential gradient-norm sequence produces the expected log-linear slope
- builds a tiny multi-layer LLaMA-style model
- creates a synthetic causal LM batch
- runs a backward pass
- verifies one gradient norm per model layer
- verifies JSON and CSV saving through a temporary test directory

Related source modules:

```text
src/llm_behavior_lab/evaluation/gradient_norms.py
src/llm_behavior_lab/models/
src/llm_behavior_lab/data/dataloader.py
```

This test intentionally exercises `loss.backward()`. On machines with a CUDA-enabled PyTorch build but an old or unavailable NVIDIA driver, PyTorch may emit a CUDA initialization warning even if the test still passes. That warning is useful environment information and does not necessarily mean the test failed.

---

## Experiment-persistence tests

File:

```text
tests/test_experiment_tracking.py
```

Purpose:

- creates run directories under pytest temporary directories
- verifies explicit run-ID collisions fail instead of overwriting
- reads immutable metadata and YAML snapshots
- appends and reloads scalar JSONL records
- rejects arrays embedded directly in scalar logs
- saves and reloads `.npz` diagnostics
- saves model, optimizer, scheduler, RNG, and additional state
- discovers the latest checkpoint
- restores checkpoints onto CPU through `map_location`
- verifies restored parameters match exactly
- checks malformed and missing checkpoints produce informative errors

Related source modules:

```text
src/llm_behavior_lab/experiment/config.py
src/llm_behavior_lab/experiment/run.py
src/llm_behavior_lab/experiment/metrics.py
src/llm_behavior_lab/experiment/arrays.py
src/llm_behavior_lab/experiment/checkpoints.py
src/llm_behavior_lab/experiment/serialization.py
```

All filesystem writes use pytest `tmp_path`. No persistent artifacts are written into the repository, no internet access is used, and no GPU is required.

---

## Persisted analysis-run tests

File:

```text
tests/test_persisted_analysis_run.py
```

Purpose:

- drives `scripts/analyze_untrained_model.py --persist-run` through its entry point
- verifies the run-directory layout, including config, metrics, checkpoints, analyses, and logs
- checks that metadata records the experiment, device, model, dataset, and analysis settings
- confirms configuration snapshots reproduce the exact inputs of the run
- validates that both evaluation records are scalar-only and reference existing artifacts
- checks `.npz` distribution keys, shapes, normalization, and gap consistency
- checks per-layer gradient arrays against the model's decoder-block count
- verifies the structured analysis JSON files, including nested top-k predictions
- restores the step-zero checkpoint into a freshly built compatible model and verifies parameter-key compatibility and exact parameter equality
- confirms that reusing an explicit run ID fails instead of overwriting saved results
- confirms that no experiment output directory is created in the repository

Related source modules:

```text
scripts/analyze_untrained_model.py
src/llm_behavior_lab/experiment/run.py
src/llm_behavior_lab/experiment/metrics.py
src/llm_behavior_lab/experiment/arrays.py
src/llm_behavior_lab/experiment/checkpoints.py
src/llm_behavior_lab/experiment/config.py
src/llm_behavior_lab/evaluation/untrained_analysis.py
src/llm_behavior_lab/evaluation/gradient_norms.py
```

This test complements `tests/test_experiment_tracking.py`. The unit tests there
cover each persistence interface in isolation; this file covers the workflow that
combines them.

The tracked data config is copied into the temporary directory with
`runtime.device` pinned to `cpu`, so the run never selects an accelerator. All
output goes to a pytest temporary directory, no dataset is downloaded, no
network access is used, and no optional dependency is required.

---

## How to run tests

Run the full test suite from the repository root:

```bash
python3 -m pytest
```

The project configures pytest in:

```text
pyproject.toml
```

Current pytest settings include:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-q"
```

This means pytest automatically:

- looks under `tests/`
- adds `src/` to the Python import path
- runs in quiet mode by default

---

## Run individual test files

Run only Phase 6 tests:

```bash
python3 -m pytest tests/test_experiment_tracking.py
```

Run only the persisted analysis-run integration test:

```bash
python3 -m pytest tests/test_persisted_analysis_run.py -v
```

Run import tests:

```bash
python3 -m pytest tests/test_imports.py
```

Run model tests:

```bash
python3 -m pytest tests/test_llama_shapes.py
```

Run data-pipeline tests:

```bash
python3 -m pytest tests/test_data_pipeline.py
```

Run inference tests:

```bash
python3 -m pytest tests/test_inference.py
```

Run evaluation tests:

```bash
python3 -m pytest tests/test_evaluation.py
```

Run gradient-norm tests:

```bash
python3 -m pytest tests/test_gradient_norms.py
```

Run one specific test:

```bash
python3 -m pytest tests/test_llama_shapes.py::test_llama_forward_shape_and_loss
```

Run with verbose output:

```bash
python3 -m pytest -v
```

---

## Expected successful output

A successful full run should look like:

```text
.............................                                            [100%]
29 passed in ...
```

The exact number of dots and runtime may change as tests are added.

Warnings may appear depending on the local environment. For example, a CUDA initialization warning may appear on a machine with a CUDA-enabled PyTorch install but an incompatible NVIDIA driver. If all tests pass, such warnings are diagnostic environment information rather than test failures.

A failing run will show:

- the test file and test function that failed
- the assertion or exception
- the stack trace
- captured output, if any

When a test fails, start by identifying whether the failure is:

1. an import/export problem
2. a shape/interface mismatch
3. a config or path mismatch
4. a numerical expectation problem
5. an environment problem

---

## Relationship with the rest of the repository

## `src/`

Tests primarily verify reusable package code under:

```text
src/llm_behavior_lab/
```

If a source module changes, its tests should usually change with it.

Examples:

| Source area | Test file |
|---|---|
| `src/llm_behavior_lab/models/` | `tests/test_llama_shapes.py` |
| `src/llm_behavior_lab/data/` | `tests/test_data_pipeline.py` |
| `src/llm_behavior_lab/inference/` | `tests/test_inference.py` |
| `src/llm_behavior_lab/evaluation/` | `tests/test_evaluation.py`, `tests/test_gradient_norms.py` |
| `src/llm_behavior_lab/experiment/` | `tests/test_experiment_tracking.py` |
| `scripts/analyze_untrained_model.py` with `--persist-run` | `tests/test_persisted_analysis_run.py` |
| package exports | `tests/test_imports.py` |

## `scripts/`

Scripts are runnable entry points. Most tests exercise the reusable functions and classes that scripts call rather than the scripts themselves. The exception is `tests/test_persisted_analysis_run.py`, which imports `scripts/analyze_untrained_model.py` and calls its entry point directly. No test runs a script as a subprocess yet.

Manual script checks are still useful after source changes:

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

Future phases may add explicit script-level subprocess tests.

## `configs/`

Most tests use small dictionaries or synthetic data rather than reading YAML configs, which keeps them lightweight and independent of filesystem state. The persisted-run integration test is the exception: it loads the tracked model and experiment configs directly, and a copy of the tracked data config with the runtime device pinned to CPU.

Scripts use configs under:

```text
configs/data/
configs/model/
```

Future tests for logging, training, or experiment runners may load config files directly.

## `data/`

Tests avoid large datasets and external downloads.

The data-pipeline tests use small in-memory token lists and text snippets. This keeps the test suite fast and offline-friendly.

If future tests need local data fixtures, prefer tiny files or temporary directories created with pytest's `tmp_path` fixture.

## Documentation

The root README explains how to run the test suite at a high level.

This file documents the local test structure and conventions.

Related folder-level documentation:

```text
data/README.md
scripts/README.md
src/README.md
tests/README.md
```

---

## Testing philosophy

The test suite should remain:

- lightweight
- fast
- deterministic where practical
- independent of large downloads
- independent of expensive training runs
- focused on shapes, interfaces, and expected behavior
- useful during incremental development

The project is not yet at the stage where tests should run long training jobs. Instead, tests should protect the contracts needed for later training:

- token IDs have the expected shape
- model forward passes return expected outputs
- inference helpers operate on the correct tensors
- evaluation utilities return structured summaries
- gradient diagnostics remain connected to the model architecture

---

## Adding new tests

### Naming

Use the pattern:

```text
tests/test_<feature>.py
```

Examples:

```text
tests/test_logging.py
tests/test_training_loop.py
tests/test_checkpointing.py
```

### Structure

A simple test should:

1. create minimal inputs
2. call one function or class behavior
3. assert the expected shape, value, type, or side effect
4. avoid depending on external data unless explicitly marked or mocked

Example style:

```python
def test_some_behavior() -> None:
    result = function_under_test(...)
    assert result.shape == expected_shape
```

### Use small models and small data

When testing model behavior, use tiny model dimensions.

Avoid:

- large vocabularies
- large sequence lengths
- real training loops
- large downloaded datasets
- long-running GPU-only tests

### Use temporary paths for output tests

For file-writing tests, use pytest's `tmp_path` fixture:

```python
def test_saves_file(tmp_path) -> None:
    output_path = tmp_path / "result.json"
    ...
    assert output_path.exists()
```

This prevents test artifacts from polluting the repository.

### Avoid duplicated setup

If setup code becomes repeated across many test files, consider adding shared fixtures in a future `tests/conftest.py`.

At the current scale, the test files are simple enough without shared fixtures.

---

## Future testing extensions

Likely future tests include:

- logging directory creation tests
- JSON/CSV metric writer tests
- checkpoint save/load tests
- training-step smoke tests
- tiny pre-training mini-run tests
- validation-loss computation tests
- scheduler/optimizer construction tests
- checkpointed evaluation tests
- fine-tuning data-pipeline tests
- model-comparison compatibility tests
- dataset-source tests for external loaders with mocks
- output-metric regression tests
- CLI subprocess smoke tests for scripts

These should be added gradually as the corresponding project phases are implemented.

---

## Checklist before committing source changes

Before committing changes to `src/`, `scripts/`, or configs, run:

```bash
python3 -m pytest
```

For changes affecting scripts, also run the relevant manual command.

For example, after modifying model code:

```bash
python3 scripts/smoke_test_llama.py --config configs/model/tiny_llama.yaml
```

After modifying data code:

```bash
python3 scripts/check_data_pipeline.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

After modifying inference code:

```bash
python3 scripts/run_inference.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

After modifying evaluation code:

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

This keeps automated tests and user-facing scripts aligned.

---

## Dataset tests and the offline guarantee

The dataset layer can reach the network, so the default suite is kept offline
structurally rather than by convention.

`tests/conftest.py` applies an autouse fixture to every test that:

- points `LLM_BEHAVIOR_LAB_DATA_ROOT` at a pytest temporary directory, so no
  test can read or write the developer's real cache
- replaces `llm_behavior_lab.data.huggingface._import_load_dataset` with a
  function that raises, so no test can obtain the real loader

A test that needs to exercise acquisition supplies its own fake by patching that
same function. Because the block is on the import seam rather than on
`sys.modules`, the guarantee holds whether or not the optional `datasets`
package happens to be installed.

Nothing changes about how the suite is run:

```bash
python3 -m pytest
```

There is no network marker and no special invocation. Live acquisition against
the real Hugging Face Hub is a manual step, documented in
[`data/README.md`](../data/README.md), not part of the automated suite.
