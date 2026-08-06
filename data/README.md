# Data subsystem

This folder contains local dataset material used by the LLM Behavior Lab project.

The project separates data handling from models, inference, and evaluation so that datasets can be changed without modifying model code. This separation is important for later phases, where the same model and analysis utilities will be reused across different dataset scales, dataset sources, training runs, checkpoints, and fine-tuning datasets.

At the current stage, the data subsystem supports:

- a small local text corpus for smoke tests and lightweight analysis
- config-driven dataset selection
- optional Hugging Face dataset loading for standard language-modeling datasets
- deterministic character-level tokenization
- train/validation token splitting
- causal language-modeling batch creation

The data pipeline is intentionally simple for now. It is designed to be easy to inspect, easy to test, and easy to extend before larger-scale training infrastructure is added.

---

## Folder role

The `data/` folder is the place for local raw or downloaded datasets.

The source code that loads and processes data lives under:

```text
src/llm_behavior_lab/data/
```

The dataset configuration files live under:

```text
configs/data/
```

This split is intentional:

```text
data/
  raw or downloaded data files

configs/data/
  YAML files describing which dataset source to use

src/llm_behavior_lab/data/
  Python utilities for loading, tokenizing, splitting, and batching text
```

This keeps raw data separate from reusable data-loading logic.

---

## Git tracking policy

Dataset contents under `data/` are ignored by Git, while Markdown documentation inside the folder stays tracked:

```gitignore
data/**
!data/
!data/**/
!data/**/*.md
```

The `!data/` and `!data/**/` lines re-include directories so Git can still see allowed files inside them. The `!data/**/*.md` line is why this guide remains tracked.

This matters because local or downloaded corpora may be large. Anything you place under `data/` is ignored by default, with no further configuration needed.

Dataset configuration files remain tracked under:

```text
configs/data/
```

Two files under `data/` are tracked on purpose:

| Path | Why it is tracked |
|---|---|
| `data/README.md` | This guide, kept by the Markdown re-include rule. |
| `data/raw/tiny_corpus.txt` | The small offline fixture used by the default configs, the documented script commands, and the test suite. It is committed deliberately so the repository stays runnable and testable without any download. |

Everything else — downloaded datasets, generated dataset artifacts, and cached local data — should live under `data/` and stay out of Git.

Do not bulk-untrack the folder with `git rm -r --cached data`. That would remove both the tiny corpus and this guide from the repository, and the ignore rules would then prevent the corpus from being added back. If a specific large file was committed by mistake, untrack that one path deliberately instead.

---

## Current datasets and source types

### 1. Local tiny text corpus

The default lightweight dataset is a small local text corpus.

Expected local path:

```text
data/raw/tiny_corpus.txt
```

Default config:

```text
configs/data/tiny_text.yaml
```

This dataset is used for:

- fast smoke tests
- tokenizer checks
- data/model compatibility checks
- inference checks
- Phase 5 untrained-model analysis
- gradient-norm diagnostic tests on small batches

It is not intended for meaningful language-model training. Its purpose is to keep early development fast and reproducible.

Example config pattern:

```yaml
dataset:
  source_type: local_text
  name: tiny_local_text
  path: data/raw/tiny_corpus.txt
  val_fraction: 0.2

tokenizer:
  type: char

batching:
  batch_size: 4
  block_size: 16

runtime:
  seed: 1234
  device: auto
```

The key field is:

```yaml
source_type: local_text
```

This tells the data loader to read text from a local file path.

---

### 2. Hugging Face dataset source

The codebase also supports external datasets through Hugging Face `datasets`.

This is optional. It requires the optional dependency group:

```bash
python3 -m pip install -e ".[dev,hf]"
```

Example config:

```text
configs/data/wikitext2.yaml
```

Example config pattern:

```yaml
dataset:
  source_type: huggingface
  name: Salesforce/wikitext
  subset: wikitext-2-raw-v1
  split: train[:1000]
  text_field: text
  val_fraction: 0.1
  max_examples: 1000
  max_characters: 200000
  document_separator: "\n\n"
  streaming: false

tokenizer:
  type: char

batching:
  batch_size: 4
  block_size: 16

runtime:
  seed: 1234
  device: auto
```

The key field is:

```yaml
source_type: huggingface
```

This tells the data loader to call Hugging Face `load_dataset`.

Current external dataset configs may include:

```text
configs/data/wikitext2.yaml
configs/data/tinystories_streaming.yaml
```

These configs are intended for controlled experiments, not automatic test-suite downloads.

---

## Data configuration files

Data configs live in:

```text
configs/data/
```

Current examples:

```text
configs/data/tiny_text.yaml
configs/data/wikitext2.yaml
configs/data/tinystories_streaming.yaml
```

### Common config sections

Most data configs have four sections:

```yaml
dataset:
  ...

tokenizer:
  ...

batching:
  ...

runtime:
  ...
```

### `dataset`

Controls where the raw text comes from.

For a local file:

```yaml
dataset:
  source_type: local_text
  path: data/raw/tiny_corpus.txt
  val_fraction: 0.2
```

For a Hugging Face dataset:

```yaml
dataset:
  source_type: huggingface
  name: Salesforce/wikitext
  subset: wikitext-2-raw-v1
  split: train[:1000]
  text_field: text
  val_fraction: 0.1
  max_examples: 1000
  max_characters: 200000
  document_separator: "\n\n"
  streaming: false
```

Important fields:

| Field | Meaning |
|---|---|
| `source_type` | Dataset source kind. Currently `local_text` or `huggingface`. |
| `path` | Local text path for `local_text` sources. |
| `name` | Hugging Face dataset name. |
| `subset` | Hugging Face dataset subset/config, if needed. |
| `split` | Hugging Face split or split slice. |
| `text_field` | Field containing text in dataset rows. |
| `val_fraction` | Fraction of token IDs assigned to validation. |
| `max_examples` | Maximum external dataset rows to use. |
| `max_characters` | Maximum concatenated characters to use. |
| `streaming` | Whether to request streaming from Hugging Face. |

### `tokenizer`

Currently the project uses a deterministic character-level tokenizer:

```yaml
tokenizer:
  type: char
```

The tokenizer is implemented in:

```text
src/llm_behavior_lab/data/tokenizer.py
```

The current tokenizer builds its vocabulary from the loaded text. Later phases may add BPE, SentencePiece, Hugging Face tokenizers, or saved tokenizer artifacts.

### `batching`

Controls batch creation:

```yaml
batching:
  batch_size: 4
  block_size: 16
```

Important fields:

| Field | Meaning |
|---|---|
| `batch_size` | Number of sampled windows per batch. |
| `block_size` | Number of input tokens per sequence. |

Each causal LM example uses a token window of length:

```text
block_size + 1
```

The batcher creates:

```text
input_ids = tokens[t : t + block_size]
targets   = tokens[t + 1 : t + block_size + 1]
```

### `runtime`

Controls reproducibility and device selection:

```yaml
runtime:
  seed: 1234
  device: auto
```

The runtime device is interpreted by:

```text
src/llm_behavior_lab/utils/device.py
```

---

## Data-loading code

The main data utilities live in:

```text
src/llm_behavior_lab/data/
```

Current files:

```text
src/llm_behavior_lab/data/
  __init__.py
  dataloader.py
  sources.py
  text_dataset.py
  tokenizer.py
```

### `sources.py`

This file provides config-driven dataset loading.

Important functions/classes:

```text
LoadedTextDataset
DatasetSourceError
load_text_dataset_from_config
load_local_text_from_config
load_huggingface_text_from_config
concatenate_text_examples
```

The main entry point is:

```python
load_text_dataset_from_config(data_config, repo_root=REPO_ROOT)
```

It returns a `LoadedTextDataset` containing:

```text
source_type
source_name
text
num_examples
metadata
```

Downstream code uses the returned `.text` field, regardless of whether the source was local or external.

### `text_dataset.py`

This file provides lower-level text-file and token-split utilities.

Important functions/classes:

```text
load_text_file
split_token_ids
TokenSplits
```

`split_token_ids` creates deterministic contiguous train/validation splits.

### `tokenizer.py`

This file defines:

```text
CharTokenizer
```

The tokenizer:

- builds a vocabulary from the loaded text
- maps characters to integer token IDs
- maps token IDs back to characters
- is deterministic for a given corpus because the vocabulary is sorted

### `dataloader.py`

This file defines:

```text
CausalLMBatch
CausalLMBatcher
```

The batcher samples fixed-length causal LM windows from token IDs and returns tensors:

```text
input_ids: [batch_size, block_size]
targets:   [batch_size, block_size]
```

These tensors are compatible with the current LLaMA-style model forward pass.

---

## Data-loading flow

The current data path is:

```text
data config
   |
   v
load_text_dataset_from_config
   |
   v
raw text
   |
   v
CharTokenizer.from_text
   |
   v
token IDs
   |
   v
split_token_ids
   |
   v
train_ids / val_ids
   |
   v
CausalLMBatcher
   |
   v
CausalLMBatch(input_ids, targets)
   |
   v
LLaMA-style model
```

More explicitly:

1. A YAML config is loaded from `configs/data/`.
2. `dataset.source_type` determines the source loader.
3. The loader returns raw text.
4. `CharTokenizer.from_text` builds a tokenizer from that text.
5. The text is encoded into token IDs.
6. The token IDs are split into train and validation token streams.
7. `CausalLMBatcher` samples shifted causal LM batches.
8. The model receives `input_ids`.
9. If `targets` are provided, the model computes causal LM loss.

Expected model-facing shapes:

```text
input_ids: [batch_size, block_size]
targets:   [batch_size, block_size]
logits:    [batch_size, block_size, model_vocab_size]
```

---

## Interaction with the rest of the codebase

### Model sanity checks

The model-only smoke test does not need real data:

```text
scripts/smoke_test_llama.py
```

It creates dummy token IDs.

### Data/model compatibility checks

The data pipeline is tested with:

```text
scripts/check_data_pipeline.py
scripts/check_dataset_options.py
```

These scripts verify that text can be loaded, tokenized, batched, and passed into the model.

### Inference

Inference uses the tokenizer to encode prompts:

```text
scripts/run_inference.py
src/llm_behavior_lab/inference/generation.py
```

The current tokenizer is built from the selected dataset text. This means prompts should only contain characters present in the tokenizer vocabulary.

### Phase 5 untrained-model analysis

The untrained analysis script uses the same data path:

```text
scripts/analyze_untrained_model.py
```

It samples causal LM windows from the configured dataset and computes:

- output entropy
- top-k examples
- top-1 prediction behavior
- empirical token-frequency comparisons
- optional per-layer gradient norms

### Future training loops

Later training code should reuse:

```text
load_text_dataset_from_config
CharTokenizer
split_token_ids
CausalLMBatcher
```

This will keep training, evaluation, and analysis aligned around the same data representation.

---

## Scripts using the data pipeline

### `scripts/check_data_pipeline.py`

Purpose:

- checks the default text-to-batch path
- verifies data/model compatibility
- runs one model forward pass

Typical command:

```bash
python3 scripts/check_data_pipeline.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

Successful output should confirm:

```text
Train input shape: (4, 16)
Train target shape: (4, 16)
Validation input shape: (4, 16)
Validation target shape: (4, 16)
Logits shape: (4, 16, 256)
Loss shape: ()
```

### `scripts/check_dataset_options.py`

Purpose:

- checks config-driven dataset selection
- supports local and Hugging Face dataset configs
- verifies tokenization and batching
- optionally verifies model compatibility

Local dataset command:

```bash
python3 scripts/check_dataset_options.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

WikiText-2 command:

```bash
python3 scripts/check_dataset_options.py \
  --data-config configs/data/wikitext2.yaml \
  --model-config configs/model/tiny_llama.yaml
```

If using Hugging Face datasets, install:

```bash
python3 -m pip install -e ".[dev,hf]"
```

Successful output should confirm:

```text
Dataset source type: local_text
```

or:

```text
Dataset source type: huggingface
```

and should also print token counts, batch shapes, and optional model logits shape.

### `scripts/run_inference.py`

Purpose:

- builds the tokenizer from the selected data config
- encodes a prompt
- runs the model
- prints top-k next-token predictions
- generates a short continuation

Typical command:

```bash
python3 scripts/run_inference.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

Successful output should confirm:

```text
Input tensor shape: ...
Full logits shape: ...
Top-k next-token predictions:
Decoded generated text: ...
```

### `scripts/analyze_untrained_model.py`

Purpose:

- runs Phase 5 untrained-model analysis
- uses data batches from the configured dataset
- compares model output probabilities with empirical token frequencies
- optionally computes per-layer gradient norms

Typical command:

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

With gradient norms:

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml \
  --compute-grad-norms
```

Successful output should confirm:

```text
Phase 5 untrained-model analysis completed successfully.
Mean output entropy: ...
Top-1 assignment concentration: ...
KL(predicted || empirical): ...
JS(predicted, empirical): ...
```

With gradient norms enabled, it should also save:

```text
outputs/phase5_gradient_norms/gradient_norms.json
outputs/phase5_gradient_norms/gradient_norms.csv
```

---

## Practical usage

### Check the local data pipeline

```bash
python3 scripts/check_data_pipeline.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

### Check the local dataset through the dataset-options script

```bash
python3 scripts/check_dataset_options.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

### Check WikiText-2 support

Install optional dependencies first:

```bash
python3 -m pip install -e ".[dev,hf]"
```

Then run:

```bash
python3 scripts/check_dataset_options.py \
  --data-config configs/data/wikitext2.yaml \
  --model-config configs/model/tiny_llama.yaml
```

If your Python environment has an old SciPy build with NumPy 2.x, refresh the optional stack:

```bash
python3 -m pip install --upgrade --force-reinstall -e ".[dev,hf]"
```

### Skip the model forward pass

For dataset-only checks:

```bash
python3 scripts/check_dataset_options.py \
  --data-config configs/data/wikitext2.yaml \
  --model-config configs/model/tiny_llama.yaml \
  --skip-model-check
```

This is useful when you only want to verify dataset loading, tokenization, and batching.

---

## Practical limitations

The current data subsystem is intentionally simple.

Current limitations:

- the character tokenizer is rebuilt from the loaded text each run
- tokenizers are not yet saved or reused
- external datasets are concatenated into one text stream
- there is no persistent tokenized cache yet
- streaming/sharded training loaders are not implemented yet
- there is no train/validation/test split metadata file yet
- large-scale training is not implemented yet
- dataset statistics are not yet saved as structured artifacts

These limitations are acceptable for the current phase because the project is still building stable foundations before training infrastructure.

---

## Future data extensions

Planned data-side extensions include:

- larger language-modeling datasets
- persistent tokenizer artifacts
- persistent tokenized datasets
- streaming dataset support for training
- dataset sharding
- train/validation/test split improvements
- dataset metadata tracking
- dataset statistics and cached summaries
- token grouping for bias and preference analysis
- subgroup-aware or metadata-aware datasets
- supervised fine-tuning datasets
- instruction-tuning datasets
- checkpoint-compatible dataset state for reproducible training runs

These should be added gradually so the data path remains easy to test and debug.
